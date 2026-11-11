//! Inference mode: a trained [`Sequential`] compiled into a model that answers without the
//! training graph.
//!
//! [`InferenceModel::compile`] copies each layer's parameters into the layout the kernels
//! read (a `Linear` weight transposed to `[inputs, outputs]`), turns each neuron mask into a
//! step of its own, and fuses a `Linear` followed by a ReLU into one kernel call. One input
//! runs through the matvec kernel on the calling thread; a batch runs through the GEMM.

use std::fmt;
use std::sync::Arc;

use cavitation_core::kernels::gemm::gemm_f32_with;
use cavitation_core::kernels::matvec::matvec_f32_with;
use cavitation_core::kernels::{self, Level, MatMut, MatRef};
use cavitation_core::{Error, Gate, ParamStore, TensorView, log_softmax_row, softmax_row};

use crate::{Activation, ActivationKind, Sequential};

/// What one layer computes at inference time, with its parameters copied out of the store:
/// the form a [`Layer`](crate::Layer) reports through
/// [`inference`](crate::Layer::inference). Every built-in layer has one.
///
/// ```
/// use cavitation_core::ParamStore;
/// use cavitation_nn::{ActivationKind, InferenceLayer, Layer, Tanh};
/// let params = ParamStore::new();
/// let form = Layer::<f32>::inference(&Tanh, &params.values);
/// assert!(matches!(form, Some(InferenceLayer::Activation(ActivationKind::Tanh))));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum InferenceLayer {
    /// A fully connected layer, `y = x · w_t + bias`.
    Linear {
        /// The weight, transposed: `inputs` rows of `outputs` values, row-major.
        w_t: Vec<f32>,
        /// The bias, `outputs` values.
        bias: Vec<f32>,
        /// The number of input columns.
        inputs: usize,
        /// The number of output columns.
        outputs: usize,
    },
    /// A built-in element-wise activation.
    Activation(ActivationKind),
    /// PReLU with these slopes: one shared by every value, or one per input column.
    Prelu(Vec<f32>),
    /// Softmax over each row.
    Softmax,
    /// Log-softmax over each row.
    LogSoftmax,
    /// A gated linear unit with this gate, halving the width.
    Glu(Gate),
    /// A neuron mask: one value per column, 0 to set the column to exactly 0 and anything
    /// else to keep it.
    Mask(Vec<f32>),
}

/// One step of a compiled model: a layer's form, with the widths it reads and writes.
#[derive(Debug)]
struct Step {
    /// What the step computes.
    form: InferenceLayer,
    /// For a `Linear` step, whether a ReLU that followed it is fused into it.
    relu: bool,
    /// The width of each row the step reads.
    inputs: usize,
    /// The width of each row the step writes.
    outputs: usize,
}

/// The read-only part of a compiled model, shared by every clone.
#[derive(Debug)]
struct Compiled {
    /// The steps, in order; never empty.
    steps: Vec<Step>,
    /// The width of one input.
    input_width: usize,
    /// The width of one output.
    output_width: usize,
    /// The widest row any step reads or writes.
    max_width: usize,
}

/// A [`Sequential`] compiled for inference: it answers one input or a batch without
/// recording a training graph, through the matvec kernel for one input and the GEMM for a
/// batch, and allocates nothing once its buffers have grown to the largest batch it has
/// seen.
///
/// It gives the same answers as the network's forward pass on the tape, masks included,
/// within the rounding of the kernels: one input differs from it by at most the matvec
/// tolerance of each `Linear` layer, carried through the layers after it.
///
/// The compiled weights are copied out of the store, so training the network further does
/// not change them; compile again to pick up new weights. A clone shares the weights (they
/// sit behind an [`Arc`]) and has buffers of its own, so give each thread its own clone.
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::{InferenceModel, Linear, Relu, Sequential};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 4, 8, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 8, 3, &mut rng));
/// let mut model = InferenceModel::compile(&net, &params).unwrap();
/// let scores = model.infer(&[0.5, -1.0, 0.25, 2.0]);
/// assert_eq!(scores.len(), 3);
/// ```
#[derive(Clone)]
pub struct InferenceModel {
    /// The compiled steps and widths, shared by clones.
    shared: Arc<Compiled>,
    /// Two buffers the steps write to in turn, each at least `max_width` values per row of
    /// the largest batch seen so far.
    bufs: [Vec<f32>; 2],
}

impl fmt::Debug for InferenceModel {
    /// Shows the widths and the number of steps, not the weights.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceModel")
            .field("input_width", &self.shared.input_width)
            .field("output_width", &self.shared.output_width)
            .field("steps", &self.shared.steps.len())
            .finish()
    }
}

impl InferenceModel {
    /// Compiles `model` with the parameter values in `params` (the store its layers were
    /// created with). The values are copied: later changes to `params` do not reach the
    /// compiled model. A neuron mask on a layer becomes a step after it that sets the off
    /// neurons to exactly 0, as in training; a `Linear` followed directly by a ReLU runs as
    /// one kernel call.
    ///
    /// The input width is the one the first layer that names one declares (a `Linear`'s
    /// inputs, or a per-channel `Prelu`'s channels), provided no layer before it changes the
    /// width.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] if the model has no layers, if a layer has no
    /// inference form (the message names its index and name, for example
    /// `layer 1 (cube) has no inference form`), or if the model's input width cannot be
    /// known.
    ///
    /// # Panics
    ///
    /// Panics, naming the layer and both widths, if a layer's required input width differs
    /// from the width before it, if a layer's output width differs from the one it reports,
    /// or if a mask's width differs from its layer's output; and as a layer's
    /// [`inference`](crate::Layer::inference) does if its parameters have the wrong shape.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{InferenceModel, Linear, Sequential, Tanh};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    ///     .push(Tanh);
    /// let model = InferenceModel::compile(&net, &params).unwrap();
    /// assert_eq!((model.input_width(), model.output_width()), (2, 3));
    /// ```
    pub fn compile(
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
    ) -> Result<InferenceModel, Error> {
        if model.is_empty() {
            return Err(Error::Unsupported(
                "inference of a model with no layers".to_string(),
            ));
        }
        let mut forms = Vec::with_capacity(model.len());
        for i in 0..model.len() {
            let layer = model.layer(i);
            let form = layer.inference(&params.values).ok_or_else(|| {
                Error::Unsupported(format!(
                    "inference: layer {i} ({}) has no inference form",
                    layer.name()
                ))
            })?;
            forms.push(form);
        }
        let input_width = input_width(&forms, model)?;
        let mut steps: Vec<Step> = Vec::with_capacity(2 * forms.len());
        let (mut width, mut max_width) = (input_width, input_width);
        for (i, form) in forms.into_iter().enumerate() {
            let layer = model.layer(i);
            let name = layer.name();
            if let Some(expected) = layer.input_width().or(declared_width(&form)) {
                assert!(
                    expected == width,
                    "InferenceModel::compile: layer {i} ({name}) expects input width \
                     {expected}, but its input has width {width}"
                );
            }
            let produced = match &form {
                InferenceLayer::Linear {
                    w_t,
                    bias,
                    inputs,
                    outputs,
                } => {
                    assert!(
                        w_t.len() == inputs * outputs && bias.len() == *outputs,
                        "InferenceModel::compile: layer {i} ({name}) is {inputs} -> {outputs} \
                         but has {} weights and {} biases",
                        w_t.len(),
                        bias.len()
                    );
                    *outputs
                }
                InferenceLayer::Prelu(slopes) => {
                    assert!(
                        slopes.len() == 1 || slopes.len() == width,
                        "InferenceModel::compile: layer {i} ({name}) has {} slopes for width \
                         {width}",
                        slopes.len()
                    );
                    width
                }
                InferenceLayer::Glu(_) => {
                    assert!(
                        width % 2 == 0,
                        "InferenceModel::compile: layer {i} ({name}) needs an even width, got \
                         {width}"
                    );
                    width / 2
                }
                _ => width,
            };
            let reported = layer.output_width(width);
            assert!(
                produced == reported,
                "InferenceModel::compile: layer {i} ({name}) produces output width {produced}, \
                 but reports {reported} for input width {width}"
            );
            let fuses = form == InferenceLayer::Activation(ActivationKind::Relu)
                && steps
                    .last()
                    .is_some_and(|s| matches!(s.form, InferenceLayer::Linear { .. }) && !s.relu);
            match steps.last_mut() {
                Some(last) if fuses => last.relu = true,
                _ => steps.push(Step {
                    form,
                    relu: false,
                    inputs: width,
                    outputs: produced,
                }),
            }
            if let Some(mask) = model.mask(i) {
                assert!(
                    mask.width() == produced,
                    "InferenceModel::compile: the mask on layer {i} ({name}) has width {}, but \
                     the layer outputs width {produced}",
                    mask.width()
                );
                let keep = (0..produced)
                    .map(|j| if mask.is_on(j) { 1.0 } else { 0.0 })
                    .collect();
                steps.push(Step {
                    form: InferenceLayer::Mask(keep),
                    relu: false,
                    inputs: produced,
                    outputs: produced,
                });
            }
            width = produced;
            max_width = max_width.max(width);
        }
        Ok(InferenceModel {
            shared: Arc::new(Compiled {
                steps,
                input_width,
                output_width: width,
                max_width,
            }),
            bufs: [vec![0.0; max_width], vec![0.0; max_width]],
        })
    }

    /// The width of one input.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{InferenceModel, Linear, Sequential};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 5, 2, &mut Rng::seed(0)));
    /// assert_eq!(InferenceModel::compile(&net, &params).unwrap().input_width(), 5);
    /// ```
    pub fn input_width(&self) -> usize {
        self.shared.input_width
    }

    /// The width of one output.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{InferenceModel, Linear, Sequential};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 5, 2, &mut Rng::seed(0)));
    /// assert_eq!(InferenceModel::compile(&net, &params).unwrap().output_width(), 2);
    /// ```
    pub fn output_width(&self) -> usize {
        self.shared.output_width
    }

    /// Runs one input through the model on the calling thread, with the matvec kernel for
    /// the current [`level`](cavitation_core::kernels::level), and returns the output (in
    /// the model's own buffer, valid until the next call). Allocates nothing.
    ///
    /// # Panics
    ///
    /// Panics if `input.len()` differs from [`input_width`](Self::input_width); the message
    /// names both widths.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{InferenceModel, Linear, Sequential};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 2, 1, &mut Rng::seed(0));
    /// params.value_mut(layer.weight()).data_mut().copy_from_slice(&[2.0, -1.0]);
    /// params.value_mut(layer.bias()).data_mut()[0] = 0.5;
    /// let net = Sequential::new().push(layer);
    /// let mut model = InferenceModel::compile(&net, &params).unwrap();
    /// assert_eq!(model.infer(&[3.0, 1.0]), &[5.5]);
    /// ```
    pub fn infer(&mut self, input: &[f32]) -> &[f32] {
        let shared = &*self.shared;
        assert!(
            input.len() == shared.input_width,
            "infer: expected an input of width {}, got {}",
            shared.input_width,
            input.len()
        );
        let run = Run {
            level: kernels::level(),
            threads: 1,
            rows: 1,
            batched: false,
        };
        let at = run.forward(shared, input, &mut self.bufs);
        &self.bufs[at][..shared.output_width]
    }

    /// Runs a batch of inputs, shape `[batch, input_width]`, through the model with the GEMM
    /// for the current [`level`](cavitation_core::kernels::level) on
    /// [`threads`](cavitation_core::kernels::threads) threads, and returns the outputs,
    /// shape `[batch, output_width]` (in the model's own buffer, valid until the next call).
    /// Allocates only when the batch is larger than any before it.
    ///
    /// # Panics
    ///
    /// Panics if `inputs` is not rank 2 with [`input_width`](Self::input_width) columns; the
    /// message names the shape.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{InferenceModel, Linear, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 3, 4, &mut Rng::seed(0)))
    ///     .push(Relu);
    /// let mut model = InferenceModel::compile(&net, &params).unwrap();
    /// let x = Tensor::full(&[5, 3], 0.5f32);
    /// let y = model.infer_batch(x.view());
    /// assert_eq!(y.shape().dims(), &[5, 4]);
    /// assert!(y.data().iter().all(|&v| v >= 0.0));
    /// ```
    pub fn infer_batch(&mut self, inputs: TensorView<f32>) -> TensorView<'_, f32> {
        let shared = &*self.shared;
        let dims = inputs.shape().dims();
        assert!(
            dims.len() == 2 && dims[1] == shared.input_width,
            "infer_batch: expected inputs of shape [batch, {}], got {dims:?}",
            shared.input_width
        );
        let rows = dims[0];
        let need = rows * shared.max_width;
        for buf in &mut self.bufs {
            if buf.len() < need {
                buf.resize(need, 0.0);
            }
        }
        let run = Run {
            level: kernels::level(),
            threads: kernels::threads(),
            rows,
            batched: true,
        };
        let at = run.forward(shared, inputs.data(), &mut self.bufs);
        let out = &self.bufs[at][..rows * shared.output_width];
        TensorView::from_slice(out, &[rows, shared.output_width])
            .expect("the output buffer holds batch × output_width values")
    }
}

/// The width a layer's form itself requires of its input: a `Linear`'s inputs, the slopes
/// of a per-channel `Prelu`, a mask's width.
fn declared_width(form: &InferenceLayer) -> Option<usize> {
    match form {
        InferenceLayer::Linear { inputs, .. } => Some(*inputs),
        InferenceLayer::Prelu(slopes) if slopes.len() > 1 => Some(slopes.len()),
        InferenceLayer::Mask(keep) => Some(keep.len()),
        _ => None,
    }
}

/// The model's input width: the width the first layer that names one requires (from the
/// layer, or from its form), provided every layer before it keeps the width unchanged.
///
/// # Errors
///
/// Returns [`Error::Unsupported`] if no layer names a width before the first one that
/// changes it.
fn input_width(forms: &[InferenceLayer], model: &Sequential<f32>) -> Result<usize, Error> {
    for (i, form) in forms.iter().enumerate() {
        if let Some(w) = model.layer(i).input_width().or(declared_width(form)) {
            return Ok(w);
        }
        if matches!(form, InferenceLayer::Glu(_)) {
            break;
        }
    }
    Err(Error::Unsupported(
        "inference: the model's input width is unknown; no layer declares one before the \
         first layer that changes the width"
            .to_string(),
    ))
}

/// The settings of one call: which kernels to run and how many rows.
struct Run {
    /// The kernel level, read once per call.
    level: Level,
    /// The thread count for the GEMM.
    threads: usize,
    /// The number of input rows.
    rows: usize,
    /// Whether `Linear` steps use the GEMM (a batch) rather than the matvec (one input).
    batched: bool,
}

impl Run {
    /// Runs every step of `model` on `input` (`rows` rows of the input width), writing into
    /// `bufs` in turn, and returns the index of the buffer that holds the output.
    ///
    /// Each buffer must hold at least `rows · max_width` values.
    fn forward(&self, model: &Compiled, input: &[f32], bufs: &mut [Vec<f32>; 2]) -> usize {
        let rows = self.rows;
        // The buffer holding the current rows, or `None` while they are still `input`.
        let mut cur: Option<usize> = None;
        for step in &model.steps {
            let (n_in, n_out) = (rows * step.inputs, rows * step.outputs);
            match &step.form {
                InferenceLayer::Linear {
                    w_t,
                    bias,
                    inputs,
                    outputs,
                } => {
                    let to = cur.map_or(0, |c| 1 - c);
                    let [b0, b1] = bufs;
                    let (src, dst): (&[f32], &mut [f32]) = match cur {
                        None => (input, if to == 0 { b0 } else { b1 }),
                        Some(0) => (b0, b1),
                        Some(_) => (b1, b0),
                    };
                    self.linear(
                        (w_t, bias, *inputs, *outputs, step.relu),
                        &src[..n_in],
                        &mut dst[..n_out],
                    );
                    cur = Some(to);
                }
                InferenceLayer::Softmax | InferenceLayer::LogSoftmax => {
                    let to = cur.map_or(0, |c| 1 - c);
                    let [b0, b1] = bufs;
                    let (src, dst): (&[f32], &mut [f32]) = match cur {
                        None => (input, if to == 0 { b0 } else { b1 }),
                        Some(0) => (b0, b1),
                        Some(_) => (b1, b0),
                    };
                    let row = if matches!(step.form, InferenceLayer::Softmax) {
                        softmax_row::<f32>
                    } else {
                        log_softmax_row::<f32>
                    };
                    let width = step.inputs;
                    if width > 0 {
                        for (x, y) in src[..n_in]
                            .chunks_exact(width)
                            .zip(dst[..n_out].chunks_exact_mut(width))
                        {
                            row(x, y);
                        }
                    }
                    cur = Some(to);
                }
                InferenceLayer::Glu(gate) => {
                    let (width, half) = (step.inputs, step.outputs);
                    let c = cur.unwrap_or(0);
                    if cur.is_none() {
                        bufs[c][..n_in].copy_from_slice(&input[..n_in]);
                    }
                    // In place: output (r, k) is written after its two inputs (r, k) and
                    // (r, half + k) are read, and every later read is further along.
                    let buf = &mut bufs[c];
                    for r in 0..rows {
                        for k in 0..half {
                            let a = buf[r * width + k];
                            let b = buf[r * width + half + k];
                            buf[r * half + k] = a * gate_value(*gate, b);
                        }
                    }
                    cur = Some(c);
                }
                form => {
                    let c = cur.unwrap_or(0);
                    if cur.is_none() {
                        bufs[c][..n_in].copy_from_slice(&input[..n_in]);
                    }
                    element_wise(form, &mut bufs[c][..n_in], step.inputs);
                    cur = Some(c);
                }
            }
        }
        cur.expect("a compiled model has at least one step")
    }

    /// One `Linear` step, `(w_t, bias, inputs, outputs, relu)`, from `src` (`rows` rows of
    /// `inputs`) into `dst` (`rows` rows of `outputs`): the matvec for one input, else the
    /// GEMM followed by the bias and the fused ReLU, row by row.
    fn linear(
        &self,
        (w_t, bias, inputs, outputs, relu): (&[f32], &[f32], usize, usize, bool),
        src: &[f32],
        dst: &mut [f32],
    ) {
        let w = MatRef::new(w_t, inputs, outputs, outputs, 1)
            .expect("compile checked that w_t holds inputs × outputs values");
        if !self.batched {
            matvec_f32_with(self.level, w, src, Some(bias), relu, dst);
            return;
        }
        let rows = self.rows;
        let x = MatRef::new(src, rows, inputs, inputs, 1).expect("rows × inputs values");
        let y = MatMut::new(dst, rows, outputs, outputs, 1).expect("rows × outputs values");
        gemm_f32_with(self.level, self.threads, 1.0, x, w, 0.0, y);
        if outputs == 0 {
            return;
        }
        for row in dst.chunks_exact_mut(outputs) {
            if relu {
                for (o, &b) in row.iter_mut().zip(bias) {
                    *o = relu_value(*o + b);
                }
            } else {
                for (o, &b) in row.iter_mut().zip(bias) {
                    *o += b;
                }
            }
        }
    }
}

/// ReLU as the ReLU layer computes it: 0 at or below 0, `v` above, NaN stays NaN.
fn relu_value(v: f32) -> f32 {
    if v <= 0.0 { 0.0 } else { v }
}

/// The value of a gated linear unit's gate at `x`, as the tape computes it.
fn gate_value(gate: Gate, x: f32) -> f32 {
    let kind = match gate {
        Gate::Sigmoid => ActivationKind::Sigmoid,
        Gate::Gelu(approx) => ActivationKind::Gelu(approx),
        Gate::Relu => ActivationKind::Relu,
        Gate::Silu => ActivationKind::Silu,
    };
    Activation::<f32>::f(&kind, x)
}

/// Applies a step that keeps the width to `data` in place, `width` values per row.
fn element_wise(form: &InferenceLayer, data: &mut [f32], width: usize) {
    match form {
        InferenceLayer::Activation(ActivationKind::Relu) => {
            for v in data.iter_mut() {
                *v = relu_value(*v);
            }
        }
        InferenceLayer::Activation(kind) => {
            for v in data.iter_mut() {
                *v = Activation::<f32>::f(kind, *v);
            }
        }
        InferenceLayer::Prelu(slopes) => {
            let shared = slopes.len() == 1;
            for (i, v) in data.iter_mut().enumerate() {
                if *v <= 0.0 || v.is_nan() {
                    *v *= slopes[if shared { 0 } else { i % width }];
                }
            }
        }
        InferenceLayer::Mask(keep) => {
            for row in data.chunks_exact_mut(width) {
                for (v, &k) in row.iter_mut().zip(keep) {
                    if k == 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
        InferenceLayer::Linear { .. }
        | InferenceLayer::Glu(_)
        | InferenceLayer::Softmax
        | InferenceLayer::LogSoftmax => {
            unreachable!("Linear, Glu, Softmax and LogSoftmax steps run on their own")
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for inference mode: agreement with the tape's forward pass for one input and for
    //! batches, masks, every built-in layer, width checks, sharing between threads and
    //! layers without an inference form.

    use std::sync::Arc;

    use super::*;
    use crate::{
        Activation, ActivationKind, ActivationLayer, Celu, Elu, Gelu, Glu, Hardsigmoid, Hardswish,
        Hardtanh, Init, Layer, LeakyRelu, Linear, LogSigmoid, LogSoftmax, Mish, NeuronMask, Prelu,
        Relu, Relu6, Selu, Sequential, Sigmoid, Silu, Softmax, Softplus, Softsign, Tanh, Trace,
    };
    use cavitation_core::{Error, Gate, GeluApprox, Graph, ParamStore, Rng, Tensor, TensorView};

    /// A 64 → 256 → 128 → 10 ReLU network with He uniform weights and biases uniform in
    /// `[-0.1, 0.1)`, all from `Rng::seed(seed)`.
    fn network(params: &mut ParamStore, seed: u64) -> Sequential {
        let mut rng = Rng::seed(seed);
        let mut layer = |inputs, outputs| {
            let l = Linear::with_init(
                params,
                inputs,
                outputs,
                Init::HeUniform,
                Init::Zeros,
                &mut rng,
            );
            for b in params.value_mut(l.bias()).data_mut() {
                *b = rng.range(-0.1, 0.1) as f32;
            }
            l
        };
        Sequential::new()
            .push(layer(64, 256))
            .push(Relu)
            .push(layer(256, 128))
            .push(Relu)
            .push(layer(128, 10))
    }

    /// `rows` rows of `width` values uniform in `[-1, 1)` from `Rng::seed(seed)`.
    fn inputs(rows: usize, width: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::seed(seed);
        (0..rows * width)
            .map(|_| rng.range(-1.0, 1.0) as f32)
            .collect()
    }

    /// The tape's forward pass over `rows` rows of `x` at once, and for each output value the
    /// bound inference must stay within: the matvec tolerance of each `Linear` layer,
    /// `2·(k + 2)·1.19e-7·(Σᵢ |hᵢ|·|wᵢⱼ| + |bⱼ|)` with `h` the layer's input on the tape,
    /// plus the bound carried in from the layer before, `Σᵢ boundᵢ·|wᵢⱼ|`. Every other layer
    /// must be element-wise with slope at most 1 (ReLU), so it carries the bound through
    /// unchanged; a mask sets it to 0 for the neurons it turns off.
    fn tape_with_bounds(
        net: &Sequential,
        params: &ParamStore,
        x: &[f32],
        rows: usize,
    ) -> (Vec<f32>, Vec<f64>) {
        let width = x.len() / rows;
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let input = Tensor::from_vec(x.to_vec(), &[rows, width]).unwrap();
        let xv = tape.input(input.view());
        let mut trace = Trace::new();
        let y = net.forward_traced(&mut tape, xv, &mut trace);
        let mut bound = vec![0.0f64; x.len()];
        for i in 0..net.len() {
            let layer = net.layer(i);
            if layer.name() == "linear" {
                let ids = layer.parameters();
                let (w, b) = (params.value(ids[0].1), params.value(ids[1].1));
                let (outputs, k) = (w.shape().dims()[0], w.shape().dims()[1]);
                let (w, b) = (w.data(), b.data());
                let h = tape.value(trace.input(i));
                let h = h.data();
                let mut next = vec![0.0f64; rows * outputs];
                for r in 0..rows {
                    for j in 0..outputs {
                        let (mut s, mut carried) = (0.0f64, 0.0f64);
                        for p in 0..k {
                            let wv = f64::from(w[j * k + p]).abs();
                            s += f64::from(h[r * k + p]).abs() * wv;
                            carried += bound[r * k + p] * wv;
                        }
                        let own = 2.0 * (k as f64 + 2.0) * 1.19e-7 * (s + f64::from(b[j]).abs());
                        next[r * outputs + j] = own + carried + 1e-30;
                    }
                }
                bound = next;
            }
            if let Some(mask) = net.mask(i) {
                let w = mask.width();
                for (x, b) in bound.iter_mut().enumerate() {
                    if !mask.is_on(x % w) {
                        *b = 0.0;
                    }
                }
            }
        }
        (tape.value(y).data().to_vec(), bound)
    }

    /// Asserts `got` is within `bound` of `want` element by element (NaN only where `want` is
    /// NaN), naming `what` and the first element outside.
    fn assert_within(got: &[f32], want: &[f32], bound: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: lengths differ");
        for (i, ((&g, &w), &b)) in got.iter().zip(want).zip(bound).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "{what}: element {i} is {g}, the tape gives NaN");
                continue;
            }
            let diff = (f64::from(g) - f64::from(w)).abs();
            assert!(
                diff <= b,
                "{what}: element {i} is {g}, the tape gives {w}, difference {diff:e} above {b:e}"
            );
        }
    }

    /// Asserts `got` and `want` hold the same bits, or are both NaN, naming `what`.
    fn assert_same(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: lengths differ");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                "{what}: element {i} is {g}, the tape gives {w}"
            );
        }
    }

    /// One input at a time: 360 seeded inputs through the 64 → 256 → 128 → 10 network give
    /// the tape's outputs within the matvec bound, layer by layer.
    #[test]
    fn inference_matches_tape() {
        let mut params = ParamStore::new();
        let net = network(&mut params, 3);
        let x = inputs(360, 64, 4);
        let (want, bound) = tape_with_bounds(&net, &params, &x, 360);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        assert_eq!((model.input_width(), model.output_width()), (64, 10));
        for (r, row) in x.as_chunks::<64>().0.iter().enumerate() {
            let got = model.infer(row);
            let at = r * 10..(r + 1) * 10;
            assert_within(got, &want[at.clone()], &bound[at], &format!("input {r}"));
        }
    }

    /// Batches of 1, 7 and 128 rows (the last batch of each run shorter) give the tape's
    /// outputs within the same bound.
    #[test]
    fn infer_batch_matches_tape() {
        let mut params = ParamStore::new();
        let net = network(&mut params, 5);
        let x = inputs(360, 64, 6);
        let (want, bound) = tape_with_bounds(&net, &params, &x, 360);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        for batch in [1usize, 7, 128] {
            for start in (0..360).step_by(batch) {
                let end = (start + batch).min(360);
                let view =
                    TensorView::from_slice(&x[start * 64..end * 64], &[end - start, 64]).unwrap();
                let got = model.infer_batch(view);
                assert_eq!(got.shape().dims(), &[end - start, 10]);
                let at = start * 10..end * 10;
                let what = format!("batch {batch}, rows {start}..{end}");
                assert_within(got.data(), &want[at.clone()], &bound[at], &what);
            }
        }
    }

    /// Masks on a `Linear` and on a ReLU set the off neurons to exactly 0, as the tape does,
    /// even where the layer gives infinity (an overflowing weight) or NaN (a NaN input); a
    /// mask on the last `Linear` of the 64 → 256 → 128 → 10 network gives the tape's outputs
    /// within the bound, for one input and for a batch.
    #[test]
    fn inference_respects_masks() {
        let mut params = ParamStore::new();
        let first = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
        params
            .value_mut(first.weight())
            .data_mut()
            .copy_from_slice(&[1e30, 1e30, 0.5, -0.25, 1.0, 1.0]);
        let mut net = Sequential::new().push(first).push(Relu);
        let mut off0 = NeuronMask::all_on(3);
        off0.set(0, false);
        let mut off2 = NeuronMask::all_on(3);
        off2.set(2, false);
        net.set_mask(0, off0);
        net.set_mask(1, off2);
        let x = [1e10f32, 1e10, f32::NAN, 1.0, 2.0, -3.0];
        let (want, _) = tape_with_bounds(&net, &params, &x, 3);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        for (r, row) in x.as_chunks::<2>().0.iter().enumerate() {
            let got = model.infer(row).to_vec();
            assert_eq!(got[0].to_bits(), 0.0f32.to_bits(), "input {r}");
            assert_eq!(got[2].to_bits(), 0.0f32.to_bits(), "input {r}");
            assert_same(&got, &want[r * 3..(r + 1) * 3], &format!("input {r}"));
        }
        let batch = model.infer_batch(TensorView::from_slice(&x, &[3, 2]).unwrap());
        assert_same(batch.data(), &want, "batch");

        let mut params = ParamStore::new();
        let mut net = network(&mut params, 7);
        let mut mask = NeuronMask::all_on(10);
        mask.set(3, false);
        mask.set(9, false);
        net.set_mask(4, mask);
        let mut hidden = NeuronMask::all_on(256);
        hidden.set(17, false);
        net.set_mask(1, hidden);
        let x = inputs(20, 64, 8);
        let (want, bound) = tape_with_bounds(&net, &params, &x, 20);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        for (r, row) in x.as_chunks::<64>().0.iter().enumerate() {
            let got = model.infer(row);
            assert_eq!((got[3], got[9]), (0.0, 0.0));
            let at = r * 10..(r + 1) * 10;
            assert_within(got, &want[at.clone()], &bound[at], &format!("input {r}"));
        }
        let got = model.infer_batch(TensorView::from_slice(&x, &[20, 64]).unwrap());
        assert_within(got.data(), &want, &bound, "masked batch");
    }

    /// Values around every kink and threshold of the built-in layers: 4 rows of 8.
    const PROBES: [f32; 32] = [
        -100.0,
        -25.0,
        -6.5,
        -3.0,
        -2.5,
        -1.0,
        -0.5,
        -0.0, //
        0.0,
        0.25,
        0.5,
        1.0,
        2.5,
        3.0,
        6.0,
        6.5, //
        19.5,
        20.5,
        25.0,
        100.0,
        -1e-3,
        1e-3,
        -7.0,
        7.0, //
        f32::NAN,
        1.5,
        -1.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.75,
        -0.75,
        4.0,
    ];

    /// An 8 → 8 `Linear` whose weight is the identity and whose bias is zero, so its output
    /// is its input (with -0 turned into +0 and the infinities into NaN, on the tape and in
    /// inference alike).
    fn identity(params: &mut ParamStore) -> Linear {
        let l = Linear::new(params, 8, 8, &mut Rng::seed(0));
        let w = params.value_mut(l.weight()).data_mut();
        for (i, v) in w.iter_mut().enumerate() {
            *v = if i % 9 == 0 { 1.0 } else { 0.0 };
        }
        l
    }

    /// Every built-in layer gives the tape's bits on the probe values, behind an identity
    /// `Linear` (which inference fuses with a following ReLU), for one input and for a
    /// batch. A `Prelu` with one slope per channel as the first layer reads the input
    /// directly.
    #[test]
    fn every_layer_matches_tape() {
        let gelu_tanh = Gelu {
            approx: GeluApprox::Tanh,
        };
        let layers: Vec<MakeLayer> = vec![
            Box::new(|_| Box::new(Relu)),
            Box::new(|_| Box::new(LeakyRelu { slope: 0.1 })),
            Box::new(|_| Box::new(Tanh)),
            Box::new(|_| Box::new(Sigmoid)),
            Box::new(|_| {
                Box::new(Gelu {
                    approx: GeluApprox::Exact,
                })
            }),
            Box::new(move |_| Box::new(gelu_tanh)),
            Box::new(|_| Box::new(Elu { alpha: 0.7 })),
            Box::new(|_| Box::new(Selu)),
            Box::new(|_| Box::new(Celu { alpha: 1.3 })),
            Box::new(|_| Box::new(Silu)),
            Box::new(|_| Box::new(Mish)),
            Box::new(|_| {
                Box::new(Softplus {
                    beta: 1.0,
                    threshold: 20.0,
                })
            }),
            Box::new(|_| Box::new(Softsign)),
            Box::new(|_| {
                Box::new(Hardtanh {
                    min: -2.0,
                    max: 1.5,
                })
            }),
            Box::new(|_| Box::new(Relu6)),
            Box::new(|_| Box::new(Hardsigmoid)),
            Box::new(|_| Box::new(Hardswish)),
            Box::new(|_| Box::new(LogSigmoid)),
            Box::new(|_| Box::new(ActivationLayer::new(ActivationKind::Silu))),
            Box::new(|p| Box::new(Prelu::new(p, 1))),
            Box::new(|p| {
                let l = Prelu::new(p, 8);
                let slopes = p.value_mut(l.weight()).data_mut();
                for (i, s) in slopes.iter_mut().enumerate() {
                    *s = i as f32 * 0.1 - 0.3;
                }
                Box::new(l)
            }),
            Box::new(|_| Box::new(Softmax)),
            Box::new(|_| Box::new(LogSoftmax)),
            Box::new(|_| {
                Box::new(Glu {
                    gate: Gate::Sigmoid,
                })
            }),
            Box::new(|_| {
                Box::new(Glu {
                    gate: Gate::Gelu(GeluApprox::Exact),
                })
            }),
            Box::new(|_| Box::new(Glu { gate: Gate::Relu })),
            Box::new(|_| Box::new(Glu { gate: Gate::Silu })),
        ];
        for (index, make) in layers.iter().enumerate() {
            for first in [false, true] {
                let mut params = ParamStore::new();
                let lin = identity(&mut params);
                let layer = make(&mut params);
                let name = layer.name().to_string();
                if first && layer.input_width() != Some(8) {
                    continue;
                }
                let mut net = Sequential::new();
                if !first {
                    net = net.push(lin);
                }
                let net = push_boxed(net, layer);
                let (want, _) = tape_with_bounds(&net, &params, &PROBES, 4);
                let mut model = InferenceModel::compile(&net, &params).unwrap();
                let out = model.output_width();
                let what = format!("layer {index} ({name}), first {first}");
                for (r, row) in PROBES.as_chunks::<8>().0.iter().enumerate() {
                    let got = model.infer(row).to_vec();
                    assert_same(&got, &want[r * out..(r + 1) * out], &what);
                }
                let got = model.infer_batch(TensorView::from_slice(&PROBES, &[4, 8]).unwrap());
                assert_same(got.data(), &want, &format!("{what}, batch"));
            }
        }
    }

    /// Builds one layer, adding any parameters it has to the store.
    type MakeLayer = Box<dyn Fn(&mut ParamStore) -> Box<dyn Layer<f32>>>;

    /// A boxed layer that forwards every method to the layer inside, so a list of boxed
    /// layers can be pushed onto a `Sequential`.
    struct Boxed(Box<dyn Layer<f32>>);

    impl Layer<f32> for Boxed {
        /// The inner layer's name.
        fn name(&self) -> &str {
            self.0.name()
        }
        /// The inner layer's output width.
        fn output_width(&self, input_width: usize) -> usize {
            self.0.output_width(input_width)
        }
        /// The inner layer's forward pass.
        fn forward(
            &self,
            tape: &mut cavitation_core::Tape<'_, f32>,
            x: cavitation_core::Var,
        ) -> cavitation_core::Var {
            self.0.forward(tape, x)
        }
        /// The inner layer's parameters.
        fn parameters(&self) -> Vec<(&'static str, cavitation_core::ParamId)> {
            self.0.parameters()
        }
        /// The inner layer's input width.
        fn input_width(&self) -> Option<usize> {
            self.0.input_width()
        }
        /// The inner layer's inference form.
        fn inference(&self, params: &cavitation_core::ParamValues<f32>) -> Option<InferenceLayer> {
            self.0.inference(params)
        }
    }

    /// `net` with `layer` pushed at the end.
    fn push_boxed(net: Sequential, layer: Box<dyn Layer<f32>>) -> Sequential {
        net.push(Boxed(layer))
    }

    /// An input of the wrong width panics in `infer`, naming both widths.
    #[test]
    #[should_panic(expected = "infer: expected an input of width 64, got 63")]
    fn wrong_input_width_panics() {
        let mut params = ParamStore::new();
        let net = network(&mut params, 1);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        model.infer(&[0.0; 63]);
    }

    /// A batch of the wrong width panics in `infer_batch`, naming the shape.
    #[test]
    #[should_panic(expected = "infer_batch: expected inputs of shape [batch, 64], got [2, 65]")]
    fn wrong_batch_width_panics() {
        let mut params = ParamStore::new();
        let net = network(&mut params, 1);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        let x = [0.0f32; 130];
        model.infer_batch(TensorView::from_slice(&x, &[2, 65]).unwrap());
    }

    /// A clone shares the compiled weights; two clones on two threads give identical
    /// outputs for the same 50 inputs.
    #[test]
    fn clones_share_weights() {
        let mut params = ParamStore::new();
        let net = network(&mut params, 9);
        let model = InferenceModel::compile(&net, &params).unwrap();
        let other = model.clone();
        assert!(Arc::ptr_eq(&model.shared, &other.shared));
        let x = Arc::new(inputs(50, 64, 10));
        let run = |mut m: InferenceModel, x: Arc<Vec<f32>>| {
            std::thread::spawn(move || {
                x.as_chunks::<64>()
                    .0
                    .iter()
                    .flat_map(|row| m.infer(row).to_vec())
                    .map(f32::to_bits)
                    .collect::<Vec<u32>>()
            })
        };
        let a = run(model, Arc::clone(&x));
        let b = run(other, Arc::clone(&x));
        let (a, b) = (a.join().unwrap(), b.join().unwrap());
        assert_eq!(a.len(), 500);
        assert!(a == b, "the two clones gave different outputs");
    }

    /// `x³`, an activation with no built-in kind, so no inference form.
    struct Cube;

    impl Activation<f32> for Cube {
        /// Returns `"cube"`.
        fn name(&self) -> &str {
            "cube"
        }
        /// `x³`.
        fn f(&self, x: f32) -> f32 {
            x * x * x
        }
        /// `3x²`.
        fn df(&self, x: f32, _y: f32) -> f32 {
            3.0 * x * x
        }
    }

    /// A layer with no inference form is an `Unsupported` error naming its index and name;
    /// so is a network whose input width no layer declares.
    #[test]
    fn compile_rejects_what_it_cannot_run() {
        let mut params = ParamStore::new();
        let net = Sequential::new()
            .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
            .push(ActivationLayer::new(Cube));
        match InferenceModel::compile(&net, &params) {
            Err(Error::Unsupported(msg)) => assert!(msg.contains("layer 1 (cube)"), "{msg}"),
            other => panic!("expected Unsupported, got {:?}", other.map(|_| ())),
        }
        let only_relu = Sequential::new().push(Relu);
        match InferenceModel::compile(&only_relu, &params) {
            Err(Error::Unsupported(msg)) => assert!(msg.contains("input width"), "{msg}"),
            other => panic!("expected Unsupported, got {:?}", other.map(|_| ())),
        }
        match InferenceModel::compile(&Sequential::new(), &params) {
            Err(Error::Unsupported(msg)) => assert!(msg.contains("no layers"), "{msg}"),
            other => panic!("expected Unsupported, got {:?}", other.map(|_| ())),
        }
    }

    /// `Linear` and `Relu` report their inference forms: the weight transposed to
    /// `[inputs, outputs]`, and the activation kind.
    #[test]
    fn layers_report_inference_forms() {
        let mut params = ParamStore::new();
        let lin = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
        params
            .value_mut(lin.weight())
            .data_mut()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        params
            .value_mut(lin.bias())
            .data_mut()
            .copy_from_slice(&[0.5, 0.0, -0.5]);
        match Layer::<f32>::inference(&lin, &params.values) {
            Some(InferenceLayer::Linear {
                w_t,
                bias,
                inputs,
                outputs,
            }) => {
                assert_eq!(w_t, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
                assert_eq!(bias, [0.5, 0.0, -0.5]);
                assert_eq!((inputs, outputs), (2, 3));
            }
            other => panic!("expected a Linear form, got {other:?}"),
        }
        assert!(matches!(
            Layer::<f32>::inference(&Relu, &params.values),
            Some(InferenceLayer::Activation(ActivationKind::Relu))
        ));
        assert!(Layer::<f32>::inference(&ActivationLayer::new(Cube), &params.values).is_none());
    }
}
