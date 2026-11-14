//! Inference mode: a trained [`Sequential`] compiled into a model that answers without the
//! training graph.
//!
//! [`InferenceModel::compile`] copies each layer's parameters into the layout the kernels
//! read (a `Linear` weight transposed to `[inputs, outputs]`), turns each neuron mask into a
//! step of its own, and fuses a `Linear` followed by a ReLU into one kernel call. One input
//! runs through the matvec kernel on the calling thread; a batch of two or more rows runs through the GEMM.
//!
//! The weights are shared by every clone of a compiled model; the masks' keep values are
//! each instance's own, so [`InferenceModel::set_neuron_mask`] (by `Linear`) and
//! [`InferenceModel::set_mask`] (by layer) change one instance in place.
//! [`InferenceModel::infer_batch_observed`] reports each layer's output as it is computed.

use std::fmt;
use std::sync::Arc;

use cavitation_core::kernels::activation::{sigmoid_f32_with, tanh_f32_with};
use cavitation_core::kernels::gemm::gemm_f32_with;
use cavitation_core::kernels::matvec::matvec_f32_with;
use cavitation_core::kernels::{self, Level, MatMut, MatRef};
use cavitation_core::{Error, Gate, ParamStore, TensorView, log_softmax_row, softmax_row};

use crate::{Activation, ActivationKind, NeuronMask, Sequential};

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
    /// A fully connected layer, `y = x · w_t + bias`, or `y = x · w_t` without a bias.
    Linear {
        /// The weight, transposed: `inputs` rows of `outputs` values, row-major.
        w_t: Vec<f32>,
        /// The bias, `outputs` values, or `None` for a layer without one.
        bias: Option<Vec<f32>>,
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

/// The closure [`InferenceModel::infer_batch_observed`] calls with each layer's index and
/// output.
type Observer<'o> = &'o mut dyn FnMut(usize, TensorView<'_, f32>);

/// What one compiled step computes.
#[derive(Debug)]
enum Op {
    /// A layer's inference form, with its values in the shared compiled model.
    Layer(InferenceLayer),
    /// A neuron mask whose keep values are the instance's own mask number `slot` (see
    /// [`InferenceModel::masks`]), so that each instance can change them.
    Mask {
        /// The index of the keep values in [`InferenceModel::masks`].
        slot: usize,
    },
}

/// One step of a compiled model: what it computes, with the widths it reads and writes and
/// the layer of the [`Sequential`] whose output it completes.
#[derive(Debug)]
struct Step {
    /// What the step computes.
    op: Op,
    /// The index of the [`Sequential`] layer the step comes from: the layer whose inference
    /// form it holds, or the layer whose output it masks.
    layer: usize,
    /// For a `Linear` step, whether a ReLU that followed it is fused into it.
    relu: bool,
    /// The width of each row the step reads.
    inputs: usize,
    /// The width of each row the step writes.
    outputs: usize,
    /// The index of the [`Sequential`] layer whose output, as the next layer receives it,
    /// this step's output is: the layer's own step when it has no mask, its mask step when
    /// it has one, and for a `Linear` with a fused ReLU the ReLU's layer (the `Linear`'s own
    /// output never exists on its own). `None` for a step whose layer a later step
    /// completes.
    completes: Option<usize>,
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
    /// For each mask slot, the [`Sequential`] layer whose output the mask applies to, in
    /// increasing order.
    mask_layers: Vec<usize>,
    /// The layers given a mask step whether or not the model has a mask there: the `layers`
    /// of [`InferenceModel::compile_with_masks`].
    extra: Vec<usize>,
    /// Every `Linear` of the model, in order, with the layer that holds its neurons
    /// ([`Sequential::neuron_layer`]): `(linear, layer)`.
    neurons: Vec<(usize, usize)>,
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
/// sit behind an [`Arc`]) and has buffers and neuron masks of its own, so give each thread
/// its own clone, and change one clone's masks with
/// [`set_neuron_mask`](Self::set_neuron_mask) (by `Linear`) or [`set_mask`](Self::set_mask)
/// (by layer index) without touching the others'.
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
    /// This instance's neuron masks, one per mask step (in the order of
    /// [`Compiled::mask_layers`]): 0 for each neuron set to exactly 0, 1 for each kept.
    masks: Vec<Vec<f32>>,
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
        InferenceModel::compile_with_masks(model, params, &[])
    }

    /// Like [`compile`](Self::compile), and gives each layer listed in `layers` that has no
    /// neuron mask a mask step with every neuron on, so that [`set_mask`](Self::set_mask)
    /// can turn its neurons off later. A layer that already has a mask keeps it.
    ///
    /// `layers` are layer indices, not `Linear`s. To switch the neurons of a `Linear`, use
    /// [`compile_with_neuron_masks`](Self::compile_with_neuron_masks) and
    /// [`set_neuron_mask`](Self::set_neuron_mask), which put the mask step after the
    /// activations that follow it ([`Sequential::neuron_layer`]). Listing the `Linear`'s own
    /// index here masks its pre-activation instead: on a sigmoid an off neuron then sends
    /// `sigmoid(0) = 0.5` to the next layer, not 0. A mask step on a `Linear` itself also
    /// stops it from running fused with a ReLU that follows, so the ReLU then runs as a step
    /// of its own and is observed on its own.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if a listed layer is not below the model's length
    /// (the message names the index and the length), and the errors
    /// [`compile`](Self::compile) returns.
    ///
    /// # Panics
    ///
    /// Panics as [`compile`](Self::compile) does.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{InferenceModel, Linear, NeuronMask, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 1, 2, &mut Rng::seed(0));
    /// params.value_mut(layer.weight()).data_mut().copy_from_slice(&[1.0, 2.0]);
    /// let net = Sequential::new().push(layer).push(Relu);
    /// let mut model = InferenceModel::compile_with_masks(&net, &params, &[1]).unwrap();
    /// assert_eq!(model.infer(&[1.0]), &[1.0, 2.0]);
    /// model.set_mask(1, &NeuronMask::from_off(2, &[0])).unwrap();
    /// assert_eq!(model.infer(&[1.0]), &[0.0, 2.0]);
    /// ```
    pub fn compile_with_masks(
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
        layers: &[usize],
    ) -> Result<InferenceModel, Error> {
        if let Some(&layer) = layers.iter().find(|&&l| l >= model.len()) {
            return Err(Error::InvalidData(format!(
                "inference: layer {layer} is out of range for a model of {} layers",
                model.len()
            )));
        }
        let (compiled, masks) = Compiled::build(model, params, layers)?;
        let max_width = compiled.max_width;
        Ok(InferenceModel {
            shared: Arc::new(compiled),
            masks,
            bufs: [vec![0.0; max_width], vec![0.0; max_width]],
        })
    }

    /// Like [`compile`](Self::compile), and gives the neurons of each `Linear` listed in
    /// `linears` (by layer index) a mask step with every neuron on, at
    /// [`Sequential::neuron_layer`] of it, so that [`set_neuron_mask`](Self::set_neuron_mask)
    /// can turn them off later. A `Linear` whose neurons already have a mask keeps it. This
    /// is [`compile_with_masks`](Self::compile_with_masks) with each `Linear` mapped to the
    /// layer that holds its neurons.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if an entry of `linears` is not the index of a
    /// `Linear` layer (the message names it), and the errors [`compile`](Self::compile)
    /// returns.
    ///
    /// # Panics
    ///
    /// Panics as [`compile`](Self::compile) does.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{InferenceModel, Linear, NeuronMask, Sequential, Sigmoid};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    ///     .push(Sigmoid);
    /// let mut model = InferenceModel::compile_with_neuron_masks(&net, &params, &[0]).unwrap();
    /// model.set_neuron_mask(0, &NeuronMask::from_off(3, &[1])).unwrap();
    /// // The mask sits after the sigmoid, so the off neuron gives exactly 0.
    /// assert_eq!(model.infer(&[1.0, -1.0])[1], 0.0);
    /// ```
    pub fn compile_with_neuron_masks(
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
        linears: &[usize],
    ) -> Result<InferenceModel, Error> {
        let mut layers = Vec::with_capacity(linears.len());
        for &linear in linears {
            if model.linear(linear).is_none() {
                return Err(Error::InvalidData(format!(
                    "inference: layer {linear} is not a Linear layer"
                )));
            }
            layers.push(model.neuron_layer(linear));
        }
        InferenceModel::compile_with_masks(model, params, &layers)
    }

    /// Picks up new weights and masks from `model` and `params`, as
    /// [`compile_with_masks`](Self::compile_with_masks) would with the layers this model was
    /// compiled with (those still below `model.len()`), so a mask step added there is kept.
    ///
    /// The weights: when no clone shares this model's compiled steps and the new steps have
    /// the same layout (the same steps with the same widths, fused ReLUs, parameter counts
    /// and mask positions), the new values replace the old ones in place and the buffers are
    /// kept; returns `true`. Otherwise this model alone gets a freshly compiled copy,
    /// leaving any clone with the old weights; returns `false`.
    ///
    /// The masks are this instance's own, and a clone's never change. When the new mask
    /// steps sit on the same layers with the same widths, each one on a layer where `model`
    /// has a mask takes that mask's values, copied in place whether or not the weights were;
    /// each one `model` has no mask for (a step added by `compile_with_masks` or
    /// `compile_with_neuron_masks`) keeps the values [`set_mask`](Self::set_mask) or
    /// [`set_neuron_mask`](Self::set_neuron_mask) gave it. Otherwise the masks are compiled
    /// afresh, the added steps with every neuron on.
    ///
    /// # Errors
    ///
    /// Returns the errors [`compile`](Self::compile) returns; the model is then unchanged.
    ///
    /// # Panics
    ///
    /// Panics as [`compile`](Self::compile) does.
    pub(crate) fn refresh(
        &mut self,
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
    ) -> Result<bool, Error> {
        if self.refresh_in_place(model, params) {
            return Ok(true);
        }
        let extra: Vec<usize> = (self.shared.extra.iter())
            .copied()
            .filter(|&l| l < model.len())
            .collect();
        let (fresh, masks) = Compiled::build(model, params, &extra)?;
        let same_masks = self.shared.mask_layers == fresh.mask_layers
            && self
                .masks
                .iter()
                .zip(&masks)
                .all(|(a, b)| a.len() == b.len());
        if same_masks {
            for ((old, new), &layer) in self.masks.iter_mut().zip(&masks).zip(&fresh.mask_layers) {
                if model.mask(layer).is_some() {
                    old.copy_from_slice(new);
                }
            }
        } else {
            self.masks = masks;
        }
        if let Some(current) = Arc::get_mut(&mut self.shared)
            && current.same_layout(&fresh)
        {
            for (old, new) in current.steps.iter_mut().zip(fresh.steps) {
                old.op = new.op;
            }
            return Ok(true);
        }
        for buf in &mut self.bufs {
            if buf.len() < fresh.max_width {
                buf.resize(fresh.max_width, 0.0);
            }
        }
        self.shared = Arc::new(fresh);
        Ok(false)
    }

    /// Copies the weights of `model` and `params` into this compiled model in place, and the
    /// values of each mask `model` has into its mask step, without allocating: what
    /// [`refresh`](Self::refresh) does when the layout stays the same, for a model whose
    /// layers have not changed since it was compiled (only their parameter values and mask
    /// values). A mask step added by [`compile_with_masks`](Self::compile_with_masks) keeps
    /// its values, as in `refresh`.
    ///
    /// Returns `false`, changing nothing, when it cannot do that: a clone shares the
    /// weights, a mask was added or removed or changed width, a layer is not the kind or
    /// width it was, or a layer with parameters is neither a [`Linear`](crate::Linear) nor a
    /// [`Prelu`](crate::Prelu) (a layer of your own, whose values only its
    /// [`inference`](crate::Layer::inference) can give). [`refresh`](Self::refresh) then
    /// compiles afresh.
    ///
    /// Each parameter-free layer is checked by comparing its inference form with the
    /// compiled one; the built-in forms of those layers hold no heap data, so the check
    /// allocates nothing.
    pub(crate) fn refresh_in_place(
        &mut self,
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
    ) -> bool {
        let Some(compiled) = Arc::get_mut(&mut self.shared) else {
            return false;
        };
        if !compiled.fits_in_place(model, params, &self.masks) {
            return false;
        }
        let values = &params.values;
        for step in &mut compiled.steps {
            match &mut step.op {
                Op::Layer(InferenceLayer::Linear {
                    w_t,
                    bias,
                    inputs,
                    outputs,
                }) => {
                    let linear = model.linear(step.layer).expect("checked by fits_in_place");
                    let w = values.get(linear.weight()).data();
                    for (i, row) in w_t.chunks_exact_mut(*outputs).enumerate() {
                        for (j, v) in row.iter_mut().enumerate() {
                            *v = w[j * *inputs + i];
                        }
                    }
                    if let (Some(bias), Some(id)) = (bias.as_mut(), linear.bias()) {
                        bias.copy_from_slice(values.get(id).data());
                    }
                }
                Op::Layer(InferenceLayer::Prelu(slopes)) => {
                    let prelu = model
                        .layer(step.layer)
                        .as_prelu()
                        .expect("checked by fits_in_place");
                    slopes.copy_from_slice(values.get(prelu.weight()).data());
                }
                Op::Mask { slot } => {
                    if let Some(mask) = model.mask(step.layer) {
                        for (j, k) in self.masks[*slot].iter_mut().enumerate() {
                            *k = if mask.is_on(j) { 1.0 } else { 0.0 };
                        }
                    }
                }
                Op::Layer(_) => {}
            }
        }
        true
    }

    /// Whether this model and `other` share their compiled steps, the weights (one is a
    /// clone of the other and neither has been recompiled since). Their masks are never
    /// shared.
    #[cfg(test)]
    pub(crate) fn shares_steps(&self, other: &InferenceModel) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }

    /// Replaces the neuron mask on the output of layer `layer` with `mask`: each neuron
    /// `mask` turns off is set to exactly 0 from the next call on. Only this instance
    /// changes; its clones keep their own masks. Allocates nothing.
    ///
    /// `layer` is a layer index, not a `Linear`. To switch the neurons of a `Linear`, use
    /// [`set_neuron_mask`](Self::set_neuron_mask), which finds the layer that holds them
    /// ([`Sequential::neuron_layer`]).
    ///
    /// The layer must have a mask step: a mask on it when the model was compiled, or a
    /// place in the `layers` given to [`compile_with_masks`](Self::compile_with_masks).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] if layer `layer` has no mask step, and
    /// [`Error::InvalidData`] if `mask`'s width differs from the layer's output width (each
    /// message names the layer); the model is then unchanged.
    ///
    /// ```
    /// use cavitation_core::{Error, ParamStore, Rng};
    /// use cavitation_nn::{InferenceModel, Linear, NeuronMask, Sequential};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
    /// let mut model = InferenceModel::compile_with_masks(&net, &params, &[0]).unwrap();
    /// model.set_mask(0, &NeuronMask::all_off(3)).unwrap();
    /// assert_eq!(model.infer(&[1.0, -1.0]), &[0.0, 0.0, 0.0]);
    /// assert!(matches!(model.set_mask(0, &NeuronMask::all_on(2)), Err(Error::InvalidData(_))));
    /// let mut plain = InferenceModel::compile(&net, &params).unwrap();
    /// assert!(matches!(plain.set_mask(0, &NeuronMask::all_on(3)), Err(Error::Unsupported(_))));
    /// ```
    pub fn set_mask(&mut self, layer: usize, mask: &NeuronMask) -> Result<(), Error> {
        let Some(slot) = self.shared.mask_layers.iter().position(|&l| l == layer) else {
            return Err(Error::Unsupported(format!(
                "inference: layer {layer} has no mask step; compile with a mask on it, or \
                 list it in compile_with_masks"
            )));
        };
        self.copy_mask(slot, mask, &format_args!("layer {layer}"))
    }

    /// Replaces the mask on the neurons of the `Linear` at layer `linear` with `mask`: each
    /// neuron `mask` turns off outputs exactly 0 from the next call on, after the
    /// activations that follow the `Linear` (at [`Sequential::neuron_layer`]), so an off
    /// neuron sends 0 to the next layer whatever the activations give at 0. Only this
    /// instance changes; its clones keep their own masks. Allocates nothing.
    ///
    /// The neurons must have a mask step: a mask on them when the model was compiled (set
    /// with [`Sequential::set_neuron_mask`], say), or a place in the `linears` given to
    /// [`compile_with_neuron_masks`](Self::compile_with_neuron_masks).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if layer `linear` is not a `Linear` or `mask`'s width
    /// differs from its output width, and [`Error::Unsupported`] if its neurons have no mask
    /// step (each message names the `Linear`); the model is then unchanged.
    ///
    /// ```
    /// use cavitation_core::{Error, ParamStore, Rng};
    /// use cavitation_nn::{InferenceModel, Linear, NeuronMask, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 1, 2, &mut Rng::seed(0));
    /// params.value_mut(layer.weight()).data_mut().copy_from_slice(&[1.0, 2.0]);
    /// let net = Sequential::new().push(layer).push(Relu);
    /// let mut model = InferenceModel::compile_with_neuron_masks(&net, &params, &[0]).unwrap();
    /// model.set_neuron_mask(0, &NeuronMask::from_off(2, &[1])).unwrap();
    /// assert_eq!(model.infer(&[1.0]), &[1.0, 0.0]);
    /// assert!(matches!(
    ///     model.set_neuron_mask(1, &NeuronMask::all_on(2)),
    ///     Err(Error::InvalidData(_))
    /// ));
    /// ```
    pub fn set_neuron_mask(&mut self, linear: usize, mask: &NeuronMask) -> Result<(), Error> {
        let Some(&(_, layer)) = self.shared.neurons.iter().find(|(l, _)| *l == linear) else {
            return Err(Error::InvalidData(format!(
                "inference: layer {linear} is not a Linear layer"
            )));
        };
        let Some(slot) = self.shared.mask_layers.iter().position(|&l| l == layer) else {
            return Err(Error::Unsupported(format!(
                "inference: the neurons of the linear layer {linear} (at layer {layer}) have \
                 no mask step; compile with a neuron mask on it, or list it in \
                 compile_with_neuron_masks"
            )));
        };
        self.copy_mask(slot, mask, &format_args!("the linear layer {linear}"))
    }

    /// Copies `mask` into the keep values of mask step `slot`, or returns
    /// [`Error::InvalidData`] naming `what` if the widths differ. Allocates nothing unless
    /// it fails.
    fn copy_mask(
        &mut self,
        slot: usize,
        mask: &NeuronMask,
        what: &dyn fmt::Display,
    ) -> Result<(), Error> {
        let keep = &mut self.masks[slot];
        if mask.width() != keep.len() {
            return Err(Error::InvalidData(format!(
                "inference: a mask of width {} for {what}, which outputs width {}",
                mask.width(),
                keep.len()
            )));
        }
        for (j, k) in keep.iter_mut().enumerate() {
            *k = if mask.is_on(j) { 1.0 } else { 0.0 };
        }
        Ok(())
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
    /// params.value_mut(layer.bias().unwrap()).data_mut()[0] = 0.5;
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
            steps: 0..shared.steps.len(),
        };
        let at = run.forward(shared, &self.masks, input, &mut self.bufs, None);
        &self.bufs[at][..shared.output_width]
    }

    /// Runs a batch of inputs, shape `[batch, input_width]`, through the model with the GEMM
    /// for the current [`level`](cavitation_core::kernels::level) on
    /// [`threads`](cavitation_core::kernels::threads) threads, and returns the outputs,
    /// shape `[batch, output_width]` (in the model's own buffer, valid until the next call).
    /// A batch of one row runs [`infer`](Self::infer)'s matvec path instead, on the calling
    /// thread, and gives `infer`'s output bit for bit. Allocates only when the batch is
    /// larger than any before it. [`infer_rows`](Self::infer_rows) takes the inputs as a
    /// plain slice.
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
        self.batch("infer_batch", inputs, None)
    }

    /// [`infer_batch`](Self::infer_batch) on `rows` inputs laid out one after another in
    /// `inputs` (`rows · input_width` values), without building a tensor: returns the
    /// `rows · output_width` outputs, row after row, in the model's own buffer, valid until
    /// the next call. The outputs are `infer_batch`'s, bit for bit; one row runs
    /// [`infer`](Self::infer)'s path. Allocates only when `rows` is larger than any batch
    /// before it.
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len()` is not `rows · input_width`; the message names both.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{InferenceModel, Linear, Sequential};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 2, 1, &mut Rng::seed(0));
    /// params.value_mut(layer.weight()).data_mut().copy_from_slice(&[2.0, -1.0]);
    /// params.value_mut(layer.bias().unwrap()).data_mut()[0] = 0.5;
    /// let net = Sequential::new().push(layer);
    /// let mut model = InferenceModel::compile(&net, &params).unwrap();
    /// assert_eq!(model.infer_rows(&[3.0, 1.0, 0.0, 1.0], 2), &[5.5, -0.5]);
    /// ```
    pub fn infer_rows(&mut self, inputs: &[f32], rows: usize) -> &[f32] {
        let width = self.shared.input_width;
        assert!(
            Some(inputs.len()) == rows.checked_mul(width),
            "infer_rows: expected {rows} rows of width {width}, got {} values",
            inputs.len()
        );
        self.run_rows(inputs, rows, None)
    }

    /// Like [`infer_batch`](Self::infer_batch), and calls `observe(layer, output)` with the
    /// output of each layer of the [`Sequential`] as the next layer receives it (masked, if
    /// the layer has a mask), shape `[batch, width]`, as soon as it is computed: once per
    /// layer, in increasing layer order. A `Linear` followed by a ReLU runs as one step, so
    /// its own output is never computed and only the ReLU's layer is reported. Allocates
    /// only when the batch is larger than any before it.
    ///
    /// # Panics
    ///
    /// Panics as [`infer_batch`](Self::infer_batch) does.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{InferenceModel, Linear, Relu, Sequential, Tanh};
    /// let mut params = ParamStore::new();
    /// let mut rng = Rng::seed(0);
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 3, 4, &mut rng))
    ///     .push(Relu)
    ///     .push(Linear::new(&mut params, 4, 2, &mut rng))
    ///     .push(Tanh);
    /// let mut model = InferenceModel::compile(&net, &params).unwrap();
    /// let mut seen = Vec::new();
    /// let x = Tensor::full(&[5, 3], 0.5f32);
    /// model.infer_batch_observed(x.view(), &mut |layer, out| {
    ///     seen.push((layer, out.shape().dims().to_vec()));
    /// });
    /// // Layer 0 runs fused with the ReLU at layer 1, so only layer 1 is reported.
    /// assert_eq!(seen, [(1, vec![5, 4]), (2, vec![5, 2]), (3, vec![5, 2])]);
    /// ```
    pub fn infer_batch_observed(
        &mut self,
        inputs: TensorView<f32>,
        observe: &mut dyn FnMut(usize, TensorView<'_, f32>),
    ) -> TensorView<'_, f32> {
        self.batch("infer_batch_observed", inputs, Some(observe))
    }

    /// The work of [`infer_batch`](Self::infer_batch) and
    /// [`infer_batch_observed`](Self::infer_batch_observed); `what` names the caller in the
    /// panic message.
    fn batch(
        &mut self,
        what: &str,
        inputs: TensorView<f32>,
        observe: Option<Observer<'_>>,
    ) -> TensorView<'_, f32> {
        let shared = &*self.shared;
        let dims = inputs.shape().dims();
        assert!(
            dims.len() == 2 && dims[1] == shared.input_width,
            "{what}: expected inputs of shape [batch, {}], got {dims:?}",
            shared.input_width
        );
        let (rows, width) = (dims[0], shared.output_width);
        let out = self.run_rows(inputs.data(), rows, observe);
        TensorView::from_slice(out, &[rows, width])
            .expect("the output buffer holds batch × output_width values")
    }

    /// Runs `rows` rows of `inputs` (checked by the caller to hold `rows · input_width`
    /// values) through the model and returns the `rows · output_width` outputs: one row on
    /// the calling thread with the matvec, as [`infer`](Self::infer) runs it (the GEMM's
    /// set-up costs more than a single row's work), more with the GEMM on
    /// [`threads`](cavitation_core::kernels::threads) threads.
    fn run_rows(&mut self, inputs: &[f32], rows: usize, observe: Option<Observer<'_>>) -> &[f32] {
        let steps = 0..self.shared.steps.len();
        self.run_steps(steps, inputs, rows, observe)
    }

    /// [`run_rows`](Self::run_rows) for the steps `steps` only (a non-empty range): `inputs`
    /// holds `rows` rows of the first step's input width, and the result `rows` rows of the
    /// last step's output width. Running a model's steps in two ranges, the second on the
    /// first's output, gives the bits of running them all at once.
    fn run_steps(
        &mut self,
        steps: std::ops::Range<usize>,
        inputs: &[f32],
        rows: usize,
        observe: Option<Observer<'_>>,
    ) -> &[f32] {
        let shared = &*self.shared;
        let width = shared.steps[steps.end - 1].outputs;
        let need = rows
            .checked_mul(shared.max_width)
            .expect("the batch's buffers fit in memory");
        for buf in &mut self.bufs {
            if buf.len() < need {
                buf.resize(need, 0.0);
            }
        }
        let one = rows == 1;
        let run = Run {
            level: kernels::level(),
            threads: if one { 1 } else { kernels::threads() },
            rows,
            batched: !one,
            steps,
        };
        let at = run.forward(shared, &self.masks, inputs, &mut self.bufs, observe);
        &self.bufs[at][..rows * width]
    }

    /// The index of the step where layer `layer` begins, when the layer is a `Linear` that
    /// a step of its own computes: the steps before it compute layers `0..layer` (their
    /// mask steps included), so their output is the `Linear`'s input. `None` for any other
    /// layer, or one past the last.
    pub(crate) fn linear_step(&self, layer: usize) -> Option<usize> {
        let steps = &self.shared.steps;
        let k = steps.iter().position(|s| s.layer >= layer)?;
        let begins = steps[k].layer == layer
            && matches!(steps[k].op, Op::Layer(InferenceLayer::Linear { .. }));
        begins.then_some(k)
    }

    /// The number of steps.
    pub(crate) fn steps(&self) -> usize {
        self.shared.steps.len()
    }

    /// The width of each row step `k` reads.
    pub(crate) fn step_inputs(&self, k: usize) -> usize {
        self.shared.steps[k].inputs
    }

    /// Runs `rows` rows of `inputs` through the steps `steps` (a non-empty range; `inputs`
    /// holds `rows` rows of the first step's input width), as [`infer_batch`] runs them,
    /// and returns the last step's output: `infer_batch`'s output, bit for bit, when the
    /// range ends with the last step and `inputs` is what the steps before the range
    /// give.
    ///
    /// [`infer_batch`]: Self::infer_batch
    pub(crate) fn infer_steps(
        &mut self,
        steps: std::ops::Range<usize>,
        inputs: &[f32],
        rows: usize,
    ) -> &[f32] {
        assert!(
            !steps.is_empty() && steps.end <= self.steps(),
            "infer_steps: steps {steps:?} of {}",
            self.steps()
        );
        let width = self.step_inputs(steps.start);
        assert!(
            Some(inputs.len()) == rows.checked_mul(width),
            "infer_steps: expected {rows} rows of width {width}, got {} values",
            inputs.len()
        );
        self.run_steps(steps, inputs, rows, None)
    }
}

impl Compiled {
    /// Compiles `model` with the values in `params`, with a mask step on each layer that
    /// has a mask or is listed in `extra` (every listed index below `model.len()`): the
    /// work of [`InferenceModel::compile_with_masks`], without the buffers. Returns the
    /// shared part and the keep values of each mask step, in order.
    ///
    /// # Errors
    ///
    /// Returns the errors [`InferenceModel::compile`] returns.
    ///
    /// # Panics
    ///
    /// Panics as [`InferenceModel::compile`] does.
    fn build(
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
        extra: &[usize],
    ) -> Result<(Compiled, Vec<Vec<f32>>), Error> {
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
        let (mut masks, mut mask_layers) = (Vec::new(), Vec::new());
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
                    let biases = bias.as_ref().map_or(*outputs, Vec::len);
                    assert!(
                        w_t.len() == inputs * outputs && biases == *outputs,
                        "InferenceModel::compile: layer {i} ({name}) is {inputs} -> {outputs} \
                         but has {} weights and {biases} biases",
                        w_t.len(),
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
            let mask = model.mask(i);
            let masked = mask.is_some() || extra.contains(&i);
            // The layer's own step completes it unless a mask step follows.
            let completes = if masked { None } else { Some(i) };
            let fuses = form == InferenceLayer::Activation(ActivationKind::Relu)
                && steps.last().is_some_and(|s| {
                    matches!(s.op, Op::Layer(InferenceLayer::Linear { .. })) && !s.relu
                });
            match steps.last_mut() {
                Some(last) if fuses => {
                    last.relu = true;
                    last.completes = completes;
                }
                _ => steps.push(Step {
                    op: Op::Layer(form),
                    layer: i,
                    relu: false,
                    inputs: width,
                    outputs: produced,
                    completes,
                }),
            }
            if masked {
                let keep = match mask {
                    Some(mask) => {
                        assert!(
                            mask.width() == produced,
                            "InferenceModel::compile: the mask on layer {i} ({name}) has width \
                             {}, but the layer outputs width {produced}",
                            mask.width()
                        );
                        (0..produced)
                            .map(|j| if mask.is_on(j) { 1.0 } else { 0.0 })
                            .collect()
                    }
                    None => vec![1.0; produced],
                };
                steps.push(Step {
                    op: Op::Mask { slot: masks.len() },
                    layer: i,
                    relu: false,
                    inputs: produced,
                    outputs: produced,
                    completes: Some(i),
                });
                masks.push(keep);
                mask_layers.push(i);
            }
            width = produced;
            max_width = max_width.max(width);
        }
        let neurons = model
            .linears()
            .map(|(linear, _)| (linear, model.neuron_layer(linear)))
            .collect();
        let compiled = Compiled {
            steps,
            input_width,
            output_width: width,
            max_width,
            mask_layers,
            extra: extra.to_vec(),
            neurons,
        };
        Ok((compiled, masks))
    }

    /// Whether [`InferenceModel::refresh_in_place`] can copy the values of `model` and
    /// `params` into these steps: the steps cover `model`'s layers one for one (a fused
    /// ReLU covering its layer), each `Linear` and `Prelu` step's layer is still one with the
    /// same widths and parameters of the compiled sizes, every other layer still has the
    /// compiled inference form (a form with values of its own, from a layer of your own,
    /// does not fit), the mask steps sit on the layers `model` masks plus the added ones,
    /// and each of `model`'s masks has the width of its step's keep values in `masks`.
    /// Allocates nothing for built-in layers.
    fn fits_in_place(
        &self,
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
        masks: &[Vec<f32>],
    ) -> bool {
        let values = &params.values;
        let mut covered = 0;
        for step in &self.steps {
            let Op::Layer(form) = &step.op else {
                continue;
            };
            if step.layer != covered || step.layer >= model.len() {
                return false;
            }
            let layer = model.layer(step.layer);
            let fits = match form {
                InferenceLayer::Linear {
                    inputs,
                    outputs,
                    bias,
                    ..
                } => model.linear(step.layer).is_some_and(|linear| {
                    linear.inputs() == *inputs
                        && linear.outputs() == *outputs
                        && values.get(linear.weight()).shape().dims() == [*outputs, *inputs]
                        // A bias must be present in both or in neither, with the same width.
                        && match (bias, linear.bias()) {
                            (Some(_), Some(id)) => values.get(id).shape().dims() == [*outputs],
                            (None, None) => true,
                            _ => false,
                        }
                }),
                InferenceLayer::Prelu(slopes) => layer
                    .as_prelu()
                    .is_some_and(|prelu| values.get(prelu.weight()).data().len() == slopes.len()),
                InferenceLayer::Mask(_) => false,
                other => {
                    layer.as_linear().is_none()
                        && layer.as_prelu().is_none()
                        && layer.inference(values).as_ref() == Some(other)
                }
            };
            if !fits {
                return false;
            }
            covered = step.layer + 1;
            if step.relu {
                let relu = InferenceLayer::Activation(ActivationKind::Relu);
                if covered >= model.len() || model.layer(covered).inference(values) != Some(relu) {
                    return false;
                }
                covered += 1;
            }
        }
        if covered != model.len() {
            return false;
        }
        let mut slots = self.mask_layers.iter().zip(masks);
        for i in 0..model.len() {
            let mask = model.mask(i);
            if mask.is_none() && !self.extra.contains(&i) {
                continue;
            }
            match slots.next() {
                Some((&layer, keep))
                    if layer == i && mask.is_none_or(|m| m.width() == keep.len()) => {}
                _ => return false,
            }
        }
        slots.next().is_none()
    }

    /// Whether `other` has the same steps as `self`, in the same order, with the same
    /// widths, fused ReLUs, parameter counts, completed layers and mask positions, so that
    /// its values can replace these in place.
    fn same_layout(&self, other: &Compiled) -> bool {
        let params = |op: &Op| match op {
            // A bias-free layer counts 0 biases, which no layer with a bias has (its
            // outputs are positive), so one never takes the place of the other.
            Op::Layer(InferenceLayer::Linear { w_t, bias, .. }) => {
                (w_t.len(), bias.as_ref().map_or(0, Vec::len))
            }
            Op::Layer(InferenceLayer::Prelu(v) | InferenceLayer::Mask(v)) => (v.len(), 0),
            Op::Mask { slot } => (*slot, 0),
            Op::Layer(_) => (0, 0),
        };
        let kind = |op: &Op| match op {
            Op::Layer(form) => Some(std::mem::discriminant(form)),
            Op::Mask { .. } => None,
        };
        self.input_width == other.input_width
            && self.output_width == other.output_width
            && self.max_width == other.max_width
            && self.mask_layers == other.mask_layers
            && self.neurons == other.neurons
            && self.steps.len() == other.steps.len()
            && self.steps.iter().zip(&other.steps).all(|(a, b)| {
                kind(&a.op) == kind(&b.op)
                    && a.layer == b.layer
                    && a.relu == b.relu
                    && a.inputs == b.inputs
                    && a.outputs == b.outputs
                    && a.completes == b.completes
                    && params(&a.op) == params(&b.op)
            })
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
    /// The steps to run, a non-empty range of the model's steps.
    steps: std::ops::Range<usize>,
}

impl Run {
    /// Runs the steps `self.steps` of `model` on `input` (`rows` rows of the first step's
    /// input width), with the
    /// keep values of each mask step in `masks`, writing into `bufs` in turn, and returns
    /// the index of the buffer that holds the output. After each step that completes a
    /// layer, calls `observe` (if given) with the layer's index and its output.
    ///
    /// Each buffer must hold at least `rows · max_width` values.
    fn forward(
        &self,
        model: &Compiled,
        masks: &[Vec<f32>],
        input: &[f32],
        bufs: &mut [Vec<f32>; 2],
        mut observe: Option<Observer<'_>>,
    ) -> usize {
        let rows = self.rows;
        // The buffer holding the current rows, or `None` while they are still `input`.
        let mut cur: Option<usize> = None;
        for step in &model.steps[self.steps.clone()] {
            let (n_in, n_out) = (rows * step.inputs, rows * step.outputs);
            let form = match &step.op {
                Op::Layer(form) => form,
                Op::Mask { slot } => {
                    let c = cur.unwrap_or(0);
                    if cur.is_none() {
                        bufs[c][..n_in].copy_from_slice(&input[..n_in]);
                    }
                    apply_mask(&masks[*slot], &mut bufs[c][..n_in], step.inputs);
                    cur = Some(c);
                    observe_step(step, &mut observe, &bufs[c][..n_out], rows);
                    continue;
                }
            };
            match form {
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
                        (w_t, bias.as_deref(), *inputs, *outputs, step.relu),
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
                    let row = if matches!(form, InferenceLayer::Softmax) {
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
                    element_wise(self.level, form, &mut bufs[c][..n_in], step.inputs);
                    cur = Some(c);
                }
            }
            let c = cur.expect("every step writes into a buffer");
            observe_step(step, &mut observe, &bufs[c][..n_out], rows);
        }
        cur.expect("the range holds at least one step")
    }

    /// One `Linear` step, `(w_t, bias, inputs, outputs, relu)`, from `src` (`rows` rows of
    /// `inputs`) into `dst` (`rows` rows of `outputs`): the matvec for one input, else the
    /// GEMM followed by the bias (none for a layer without one) and the fused ReLU, row by
    /// row.
    fn linear(
        &self,
        (w_t, bias, inputs, outputs, relu): (&[f32], Option<&[f32]>, usize, usize, bool),
        src: &[f32],
        dst: &mut [f32],
    ) {
        let w = MatRef::new(w_t, inputs, outputs, outputs, 1)
            .expect("compile checked that w_t holds inputs × outputs values");
        if !self.batched {
            matvec_f32_with(self.level, w, src, bias, relu, dst);
            return;
        }
        let rows = self.rows;
        let x = MatRef::new(src, rows, inputs, inputs, 1).expect("rows × inputs values");
        let y = MatMut::new(dst, rows, outputs, outputs, 1).expect("rows × outputs values");
        gemm_f32_with(self.level, self.threads, 1.0, x, w, 0.0, y);
        if outputs == 0 {
            return;
        }
        match (bias, relu) {
            (Some(bias), true) => {
                for row in dst.chunks_exact_mut(outputs) {
                    for (o, &b) in row.iter_mut().zip(bias) {
                        *o = relu_value(*o + b);
                    }
                }
            }
            (Some(bias), false) => {
                for row in dst.chunks_exact_mut(outputs) {
                    for (o, &b) in row.iter_mut().zip(bias) {
                        *o += b;
                    }
                }
            }
            (None, true) => {
                for o in dst.iter_mut() {
                    *o = relu_value(*o);
                }
            }
            (None, false) => {}
        }
    }
}

/// Calls `observe`, if given, with the layer `step` completes and its output `out` (`rows`
/// rows of the step's output width); does nothing for a step that completes no layer.
fn observe_step(step: &Step, observe: &mut Option<Observer<'_>>, out: &[f32], rows: usize) {
    if let (Some(layer), Some(f)) = (step.completes, observe.as_mut()) {
        let view = TensorView::from_slice(out, &[rows, step.outputs])
            .expect("the step's output holds rows × outputs values");
        f(layer, view);
    }
}

/// Sets each value of `data` (rows of `width`) whose column's `keep` value is 0 to exactly
/// 0, whatever it was, NaN and infinity included.
fn apply_mask(keep: &[f32], data: &mut [f32], width: usize) {
    if width == 0 {
        return;
    }
    for row in data.chunks_exact_mut(width) {
        for (v, &k) in row.iter_mut().zip(keep) {
            if k == 0.0 {
                *v = 0.0;
            }
        }
    }
}

/// ReLU as the ReLU layer computes it: 0 at or below 0, `v` above, NaN stays NaN.
fn relu_value(v: f32) -> f32 {
    if v <= 0.0 { 0.0 } else { v }
}

/// The value of a gated linear unit's gate at `x`, as the tape computes it: its sigmoid gate
/// is the scalar sigmoid, not the SIMD kernel of the `Sigmoid` layer.
fn gate_value(gate: Gate, x: f32) -> f32 {
    let kind = match gate {
        Gate::Sigmoid => return crate::activation::sigmoid(x),
        Gate::Gelu(approx) => ActivationKind::Gelu(approx),
        Gate::Relu => ActivationKind::Relu,
        Gate::Silu => ActivationKind::Silu,
    };
    Activation::<f32>::f(&kind, x)
}

/// Applies a step that keeps the width to `data` in place, `width` values per row; `tanh`
/// and the sigmoid run the SIMD kernels of `level`.
fn element_wise(level: Level, form: &InferenceLayer, data: &mut [f32], width: usize) {
    match form {
        InferenceLayer::Activation(ActivationKind::Relu) => {
            for v in data.iter_mut() {
                *v = relu_value(*v);
            }
        }
        InferenceLayer::Activation(ActivationKind::Tanh) => tanh_f32_with(level, data),
        InferenceLayer::Activation(ActivationKind::Sigmoid) => sigmoid_f32_with(level, data),
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
        InferenceLayer::Mask(keep) => apply_mask(keep, data, width),
        InferenceLayer::Linear { .. }
        | InferenceLayer::Glu(_)
        | InferenceLayer::Softmax
        | InferenceLayer::LogSoftmax => {
            unreachable!("Linear, Glu, Softmax and LogSoftmax steps run on their own")
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! Tests for inference mode: agreement with the tape's forward pass for one input and for
    //! batches, masks, every built-in layer, width checks, sharing between threads, layers
    //! without an inference form, observed layer outputs and masks changed per instance.

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
            for b in params.value_mut(l.bias().unwrap()).data_mut() {
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

    /// The tape's forward pass over `rows` rows of `x` at once: for each layer, its output
    /// (as the next layer receives it, masked) and for each value of it the bound inference
    /// must stay within: the matvec tolerance of each `Linear` layer,
    /// `2·(k + 2)·1.19e-7·(Σᵢ |hᵢ|·|wᵢⱼ| + |bⱼ|)` with `h` the layer's input on the tape,
    /// plus the bound carried in from the layer before, `Σᵢ boundᵢ·|wᵢⱼ|`. Every other layer
    /// must be element-wise with slope at most 1 (ReLU), so it carries the bound through
    /// unchanged; a mask sets it to 0 for the neurons it turns off. The GEMM keeps to the
    /// same bound.
    pub(crate) fn layer_bounds(
        net: &Sequential,
        params: &ParamStore,
        x: &[f32],
        rows: usize,
    ) -> Vec<(Vec<f32>, Vec<f64>)> {
        let width = x.len() / rows;
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let input = Tensor::from_vec(x.to_vec(), &[rows, width]).unwrap();
        let xv = tape.input(input.view());
        let mut trace = Trace::new();
        net.forward_traced(&mut tape, xv, &mut trace);
        let mut bound = vec![0.0f64; x.len()];
        let mut layers = Vec::with_capacity(net.len());
        for i in 0..net.len() {
            let layer = net.layer(i);
            if layer.name() == "linear" {
                let ids = layer.parameters();
                let w = params.value(ids[0].1);
                let (outputs, k) = (w.shape().dims()[0], w.shape().dims()[1]);
                // A layer without a bias adds nothing to the sum: its `|bⱼ|` is 0.
                let zeros = vec![0.0f32; outputs];
                let b = ids
                    .get(1)
                    .map_or(&zeros[..], |&(_, id)| params.value(id).data());
                let w = w.data();
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
            let out = tape.value(trace.output(i)).data().to_vec();
            layers.push((out, bound.clone()));
        }
        layers
    }

    /// The tape's output over `rows` rows of `x` at once and the bound inference must stay
    /// within, from [`layer_bounds`]'s last layer.
    fn tape_with_bounds(
        net: &Sequential,
        params: &ParamStore,
        x: &[f32],
        rows: usize,
    ) -> (Vec<f32>, Vec<f64>) {
        layer_bounds(net, params, x, rows)
            .pop()
            .expect("the network has layers")
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

    /// A network whose `Linear`s have no bias (a 64 → 256 one before a fused ReLU and the
    /// 128 → 10 output) beside one that has (256 → 128) gives the tape's outputs within the
    /// bound, one input at a time and in batches of 1, 7 and 128 rows: the compiled steps
    /// skip the bias add, with and without the fused ReLU.
    #[test]
    fn layers_without_a_bias_match_tape() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(21);
        let mut bare = |params: &mut ParamStore, inputs, outputs| {
            Linear::without_bias(params, inputs, outputs, Init::HeUniform, &mut rng)
        };
        let first = bare(&mut params, 64, 256);
        let middle = Linear::new(&mut params, 256, 128, &mut Rng::seed(22));
        for b in params.value_mut(middle.bias().unwrap()).data_mut() {
            *b = 0.05;
        }
        let last = bare(&mut params, 128, 10);
        let net = Sequential::new()
            .push(first)
            .push(Relu)
            .push(middle)
            .push(Relu)
            .push(last);
        let x = inputs(360, 64, 23);
        let (want, bound) = tape_with_bounds(&net, &params, &x, 360);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        for (r, row) in x.as_chunks::<64>().0.iter().enumerate() {
            let at = r * 10..(r + 1) * 10;
            let got = model.infer(row);
            assert_within(got, &want[at.clone()], &bound[at], &format!("input {r}"));
        }
        for batch in [1usize, 7, 128] {
            for start in (0..360).step_by(batch) {
                let end = (start + batch).min(360);
                let view =
                    TensorView::from_slice(&x[start * 64..end * 64], &[end - start, 64]).unwrap();
                let got = model.infer_batch(view);
                let at = start * 10..end * 10;
                let what = format!("batch {batch}, rows {start}..{end}");
                assert_within(got.data(), &want[at.clone()], &bound[at], &what);
            }
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

    /// A one-row `infer_batch` (and `infer_batch_observed`, whose last observed output is the
    /// returned one) gives `infer`'s output bit for bit, and `infer_rows` gives
    /// `infer_batch`'s outputs bit for bit for 1, 2, 7 and 128 rows.
    #[test]
    fn one_row_batches_and_plain_rows_match() {
        let mut params = ParamStore::new();
        let net = network(&mut params, 5);
        let x = inputs(128, 64, 6);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        for r in 0..20 {
            let row = &x[r * 64..(r + 1) * 64];
            let want = model.infer(row).to_vec();
            let view = TensorView::from_slice(row, &[1, 64]).unwrap();
            assert_same(
                model.infer_batch(view).data(),
                &want,
                &format!("row {r}, batch"),
            );
            let mut last = Vec::new();
            let got = model
                .infer_batch_observed(view, &mut |_, out| last = out.data().to_vec())
                .data()
                .to_vec();
            assert_same(&got, &want, &format!("row {r}, observed"));
            assert_same(&last, &want, &format!("row {r}, last observed"));
        }
        for rows in [1usize, 2, 7, 128] {
            let slice = &x[..rows * 64];
            let view = TensorView::from_slice(slice, &[rows, 64]).unwrap();
            let want = model.infer_batch(view).data().to_vec();
            assert_same(
                model.infer_rows(slice, rows),
                &want,
                &format!("{rows} rows"),
            );
        }
    }

    /// `infer_rows` refuses inputs whose length is not rows × the input width.
    #[test]
    #[should_panic(expected = "infer_rows: expected 2 rows of width 64, got 127 values")]
    fn infer_rows_checks_the_length() {
        let mut params = ParamStore::new();
        let net = network(&mut params, 5);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        model.infer_rows(&[0.0; 127], 2);
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
            .value_mut(lin.bias().unwrap())
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
                assert_eq!(bias, Some(vec![0.5, 0.0, -0.5]));
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

    /// The digits-shaped network, 64 → 32 → ReLU → 10, with He uniform weights and biases
    /// uniform in `[-0.1, 0.1)`, all from `Rng::seed(seed)`.
    pub(crate) fn digits(params: &mut ParamStore, seed: u64) -> Sequential {
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
            for b in params.value_mut(l.bias().unwrap()).data_mut() {
                *b = rng.range(-0.1, 0.1) as f32;
            }
            l
        };
        Sequential::new()
            .push(layer(64, 32))
            .push(Relu)
            .push(layer(32, 10))
    }

    /// One observed layer: its index, the shape of its output and a copy of the output.
    type Observed = (usize, Vec<usize>, Vec<f32>);

    /// Runs `x` (`rows` rows) through `model.infer_batch_observed` and returns each observed
    /// layer with a copy of its output, in the order reported, and the returned output.
    fn observe_all(
        model: &mut InferenceModel,
        x: &[f32],
        rows: usize,
    ) -> (Vec<Observed>, Vec<f32>) {
        let view = TensorView::from_slice(x, &[rows, x.len() / rows]).unwrap();
        let mut seen = Vec::new();
        let out = model
            .infer_batch_observed(view, &mut |layer, out| {
                seen.push((layer, out.shape().dims().to_vec(), out.data().to_vec()));
            })
            .data()
            .to_vec();
        (seen, out)
    }

    /// The digits-shaped network, with a mask on its last layer, over 37 rows: every
    /// observed output equals the tape's output of that layer (from the trace) within the
    /// GEMM bound, the layers come in increasing order (the first `Linear`, fused with its
    /// ReLU, not on its own), and the last one observed is the returned output, bit for bit.
    #[test]
    fn observed_outputs_match_trace() {
        let mut params = ParamStore::new();
        let mut net = digits(&mut params, 11);
        net.set_mask(2, NeuronMask::from_off(10, &[4]));
        let x = inputs(37, 64, 12);
        let want = layer_bounds(&net, &params, &x, 37);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        let (seen, out) = observe_all(&mut model, &x, 37);
        let layers: Vec<usize> = seen.iter().map(|(l, _, _)| *l).collect();
        assert_eq!(layers, [1, 2]);
        for (layer, dims, got) in &seen {
            let (tape, bound) = &want[*layer];
            assert_eq!(dims, &[37, got.len() / 37]);
            assert_within(got, tape, bound, &format!("layer {layer}"));
        }
        assert_same(&seen[1].2, &out, "the last observed output");
        assert!(seen[1].2.chunks(10).all(|row| row[4].to_bits() == 0));
    }

    /// A `Linear` fused with the ReLU after it reports the ReLU's layer only, with the
    /// ReLU's output (and its mask, when the ReLU has one); a `Linear` with a mask of its
    /// own is not fused, so both layers are reported.
    #[test]
    fn fused_linear_reports_activation_layer() {
        let mut params = ParamStore::new();
        let x = inputs(5, 4, 13);
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 4, 6, &mut Rng::seed(14)))
            .push(Relu);
        let want = layer_bounds(&net, &params, &x, 5);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        let (seen, _) = observe_all(&mut model, &x, 5);
        assert_eq!(seen.len(), 1);
        assert_eq!((seen[0].0, seen[0].1.as_slice()), (1, &[5, 6][..]));
        assert_within(&seen[0].2, &want[1].0, &want[1].1, "fused");
        assert!(seen[0].2.iter().all(|&v| v >= 0.0));

        net.set_mask(1, NeuronMask::from_off(6, &[0]));
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        let (seen, _) = observe_all(&mut model, &x, 5);
        assert_eq!(seen.iter().map(|s| s.0).collect::<Vec<_>>(), [1]);
        assert!(seen[0].2.chunks(6).all(|row| row[0].to_bits() == 0));

        net.clear_mask(1);
        net.set_mask(0, NeuronMask::from_off(6, &[2]));
        let want = layer_bounds(&net, &params, &x, 5);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        let (seen, _) = observe_all(&mut model, &x, 5);
        assert_eq!(seen.iter().map(|s| s.0).collect::<Vec<_>>(), [0, 1]);
        for (layer, _, got) in &seen {
            assert_within(
                got,
                &want[*layer].0,
                &want[*layer].1,
                &format!("layer {layer}"),
            );
        }
        assert!(
            seen[0].2.iter().any(|&v| v < 0.0),
            "the Linear's own output"
        );
    }

    /// Asserts two models give the same bits for every one of `x`'s rows through `infer`
    /// and for all of them through `infer_batch`, naming `what`.
    fn assert_models_agree(a: &mut InferenceModel, b: &mut InferenceModel, x: &[f32], what: &str) {
        let width = a.input_width();
        let rows = x.len() / width;
        for (r, row) in x.chunks(width).enumerate() {
            let got = a.infer(row).to_vec();
            assert_same(&got, b.infer(row), &format!("{what}, input {r}"));
        }
        let view = TensorView::from_slice(x, &[rows, width]).unwrap();
        let got = a.infer_batch(view).data().to_vec();
        assert_same(&got, b.infer_batch(view).data(), &format!("{what}, batch"));
    }

    /// Masks set with `set_mask` on a model compiled with all-on mask steps give the bits of
    /// compiling the masked model, through `infer` and `infer_batch`, and changing them
    /// again (back to all on, to other neurons) keeps agreeing.
    #[test]
    fn set_mask_matches_masked_compile() {
        let mut params = ParamStore::new();
        let net = digits(&mut params, 15);
        let x = inputs(23, 64, 16);
        let mut model = InferenceModel::compile_with_masks(&net, &params, &[1, 2]).unwrap();
        let mut plain = InferenceModel::compile(&net, &params).unwrap();
        assert_models_agree(&mut model, &mut plain, &x, "all on");
        let cases = [
            (
                NeuronMask::from_off(32, &[0, 7, 31]),
                NeuronMask::from_off(10, &[3]),
            ),
            (NeuronMask::all_on(32), NeuronMask::from_on(10, &[1, 2])),
            (NeuronMask::all_off(32), NeuronMask::all_on(10)),
        ];
        for (i, (hidden, last)) in cases.into_iter().enumerate() {
            model.set_mask(1, &hidden).unwrap();
            model.set_mask(2, &last).unwrap();
            let mut masked = net.clone();
            masked.set_mask(1, hidden);
            masked.set_mask(2, last);
            let mut want = InferenceModel::compile(&masked, &params).unwrap();
            assert_models_agree(&mut model, &mut want, &x, &format!("case {i}"));
        }
    }

    /// `set_mask` on a clone changes the clone alone: the original's outputs stay the same
    /// bit for bit, and the two still share their weights.
    #[test]
    fn set_mask_on_clone_leaves_original() {
        let mut params = ParamStore::new();
        let net = digits(&mut params, 17);
        let x = inputs(9, 64, 18);
        let view = TensorView::from_slice(&x, &[9, 64]).unwrap();
        let mut original = InferenceModel::compile_with_masks(&net, &params, &[1]).unwrap();
        let before = original.infer_batch(view).data().to_vec();
        let rows_before: Vec<Vec<f32>> = x
            .chunks(64)
            .map(|row| original.infer(row).to_vec())
            .collect();
        let mut copy = original.clone();
        copy.set_mask(1, &NeuronMask::all_off(32)).unwrap();
        assert!(copy.shares_steps(&original));
        let changed = copy.infer_batch(view).data().to_vec();
        assert_ne!(changed, before);
        assert_same(original.infer_batch(view).data(), &before, "original batch");
        for (r, (row, want)) in x.chunks(64).zip(&rows_before).enumerate() {
            let got = original.infer(row).to_vec();
            assert_same(&got, want, &format!("original input {r}"));
        }
    }

    /// `refresh` keeps the mask steps `compile_with_masks` added, with the values
    /// `set_mask` gave them, whether the weights go in place or are recompiled (a clone
    /// shares them); masks the model has take the model's values in this instance only, and
    /// the clone's outputs stay the same bit for bit.
    #[test]
    fn refresh_keeps_added_mask_steps() {
        let mut params = ParamStore::new();
        let mut net = digits(&mut params, 27);
        net.set_mask(2, NeuronMask::all_on(10));
        let x = inputs(6, 64, 28);
        let view = TensorView::from_slice(&x, &[6, 64]).unwrap();
        let mut model = InferenceModel::compile_with_masks(&net, &params, &[1]).unwrap();
        let hidden = NeuronMask::from_off(32, &[2, 3, 30]);
        model.set_mask(1, &hidden).unwrap();
        let w = net.param("0.weight").unwrap();
        let want = |net: &Sequential, params: &ParamStore| {
            let mut masked = net.clone();
            masked.set_mask(1, hidden.clone());
            let mut m = InferenceModel::compile(&masked, params).unwrap();
            m.infer_batch(view).data().to_vec()
        };

        params.value_mut(w).data_mut()[0] += 0.5;
        assert!(model.refresh(&net, &params).unwrap());
        assert_same(
            model.infer_batch(view).data(),
            &want(&net, &params),
            "in place",
        );

        let mut copy = model.clone();
        let copy_before = copy.infer_batch(view).data().to_vec();
        params.value_mut(w).data_mut()[1] -= 0.5;
        net.set_mask(2, NeuronMask::from_off(10, &[7]));
        assert!(!model.refresh(&net, &params).unwrap());
        assert!(!model.shares_steps(&copy));
        let got = model.infer_batch(view).data().to_vec();
        assert_same(&got, &want(&net, &params), "recompiled");
        assert!(got.chunks(10).all(|row| row[7].to_bits() == 0));
        assert_same(copy.infer_batch(view).data(), &copy_before, "the clone");
        model.set_mask(1, &NeuronMask::all_on(32)).unwrap();
    }

    /// `refresh_in_place` updates a `Linear` without a bias in place, bit for bit a fresh
    /// compile, and returns `false` when the compiled layer and the current one disagree on
    /// having a bias (either way round), after which `refresh` gives the new network's
    /// outputs. A `fits_in_place` that ignored the bias would take the in-place path there
    /// and keep a missing or stale bias.
    #[test]
    fn refresh_in_place_without_a_bias() {
        // A 5 → 6 → ReLU → 3 network whose first `Linear` has a bias exactly when `biased`
        // is true; the output layer always has one. Biases are non-zero, so a dropped or
        // stale bias changes the outputs.
        let build = |biased: bool, seed: u64| {
            let mut params = ParamStore::new();
            let mut rng = Rng::seed(seed);
            let first = if biased {
                Linear::new(&mut params, 5, 6, &mut rng)
            } else {
                Linear::without_bias(&mut params, 5, 6, Init::HeUniform, &mut rng)
            };
            let last = Linear::new(&mut params, 6, 3, &mut rng);
            for id in [first.bias(), last.bias()].into_iter().flatten() {
                for b in params.value_mut(id).data_mut() {
                    *b = rng.range(0.1, 0.3) as f32;
                }
            }
            let net = Sequential::new().push(first).push(Relu).push(last);
            (net, params)
        };
        let x = inputs(9, 5, 43);
        let view = TensorView::from_slice(&x, &[9, 5]).unwrap();
        let fresh = |net: &Sequential, params: &ParamStore| {
            let mut model = InferenceModel::compile(net, params).unwrap();
            model.infer_batch(view).data().to_vec()
        };

        // (a) Bias-free to bias-free, new values: in place, bit for bit a fresh compile.
        let (bare, mut bare_params) = build(false, 44);
        let mut model = InferenceModel::compile(&bare, &bare_params).unwrap();
        let before = model.infer_batch(view).data().to_vec();
        for (_, id) in bare.parameter_names() {
            for v in bare_params.value_mut(id).data_mut() {
                *v = *v * 0.75 + 0.01;
            }
        }
        assert!(model.refresh_in_place(&bare, &bare_params));
        let got = model.infer_batch(view).data().to_vec();
        assert_same(&got, &fresh(&bare, &bare_params), "bias-free, in place");
        assert_ne!(got, before);

        // (b) Compiled bias-free, now a same-width layer with a bias: not in place, and
        // `refresh` gives that network's outputs.
        let (biased, biased_params) = build(true, 45);
        assert!(!model.refresh_in_place(&biased, &biased_params));
        model.refresh(&biased, &biased_params).unwrap();
        let got = model.infer_batch(view).data().to_vec();
        assert_same(&got, &fresh(&biased, &biased_params), "bias added");

        // (c) The reverse: compiled with a bias, now without one.
        let mut model = InferenceModel::compile(&biased, &biased_params).unwrap();
        assert!(!model.refresh_in_place(&bare, &bare_params));
        model.refresh(&bare, &bare_params).unwrap();
        let got = model.infer_batch(view).data().to_vec();
        assert_same(&got, &fresh(&bare, &bare_params), "bias removed");
    }

    /// `refresh_in_place` copies new weights, PReLU slopes and mask values into the
    /// compiled model, giving bit for bit the outputs of compiling afresh (a `Linear` fused
    /// with a ReLU, a per-channel PReLU, a model mask and a mask step `compile_with_masks`
    /// added, whose values it keeps). It changes nothing and returns `false` when a clone
    /// shares the weights, when a mask is added or removed, and when the layers differ.
    #[test]
    fn refresh_in_place_matches_compile() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(41);
        let prelu = Prelu::new(&mut params, 6);
        let slopes = prelu.weight();
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 5, 6, &mut rng))
            .push(prelu)
            .push(Linear::new(&mut params, 6, 7, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 7, 3, &mut rng));
        net.set_mask(1, NeuronMask::from_off(6, &[2]));
        let x = inputs(9, 5, 42);
        let view = TensorView::from_slice(&x, &[9, 5]).unwrap();
        let added = NeuronMask::from_off(7, &[4]);
        let mut model = InferenceModel::compile_with_masks(&net, &params, &[3]).unwrap();
        model.set_mask(3, &added).unwrap();
        let want = |net: &Sequential, params: &ParamStore| {
            let mut masked = net.clone();
            masked.set_mask(3, added.clone());
            InferenceModel::compile(&masked, params)
                .unwrap()
                .infer_batch(view)
                .data()
                .to_vec()
        };
        let before = model.infer_batch(view).data().to_vec();

        for (_, id) in net.parameter_names() {
            for v in params.value_mut(id).data_mut() {
                *v = *v * 0.75 + 0.01;
            }
        }
        params.value_mut(slopes).data_mut()[4] = -0.5;
        net.set_mask(1, NeuronMask::from_off(6, &[0, 5]));
        assert!(model.refresh_in_place(&net, &params));
        let got = model.infer_batch(view).data().to_vec();
        assert_same(&got, &want(&net, &params), "in place");
        assert_ne!(got, before);

        let copy = model.clone();
        params.value_mut(slopes).data_mut()[0] = 0.5;
        assert!(!model.refresh_in_place(&net, &params));
        assert_same(model.infer_batch(view).data(), &got, "shared, unchanged");
        drop(copy);

        let mut unmasked = net.clone();
        unmasked.clear_mask(1);
        assert!(!model.refresh_in_place(&unmasked, &params));
        let mut more = net.clone();
        more.set_mask(2, NeuronMask::all_on(7));
        assert!(!model.refresh_in_place(&more, &params));
        let mut other = ParamStore::new();
        let tanh = Sequential::new()
            .push(Linear::new(&mut other, 5, 6, &mut rng))
            .push(Tanh)
            .push(Linear::new(&mut other, 6, 7, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut other, 7, 3, &mut rng));
        assert!(!model.refresh_in_place(&tanh, &other));
        assert_same(model.infer_batch(view).data(), &got, "refused, unchanged");
        assert!(model.refresh_in_place(&net, &params));
        assert_same(
            model.infer_batch(view).data(),
            &want(&net, &params),
            "again",
        );
    }

    /// Linear(3 → 4) → Sigmoid → Linear(4 → 2), with weights from `Rng::seed(seed)`.
    fn sigmoid_net(params: &mut ParamStore, seed: u64) -> Sequential {
        let mut rng = Rng::seed(seed);
        Sequential::new()
            .push(Linear::new(params, 3, 4, &mut rng))
            .push(Sigmoid)
            .push(Linear::new(params, 4, 2, &mut rng))
    }

    /// On Linear → Sigmoid → Linear, `set_neuron_mask` on a model compiled with
    /// `compile_with_neuron_masks` puts the mask after the sigmoid: the off neuron reads
    /// exactly 0 at layer 1, the output is bit for bit that of compiling the network masked
    /// with `Sequential::set_neuron_mask`, and both differ from masking the `Linear`'s own
    /// output by layer (where the neuron reads `sigmoid(0) = 0.5`). Changing it back gives
    /// the plain model's bits.
    #[test]
    fn set_neuron_mask_after_sigmoid_is_zero() {
        let mut params = ParamStore::new();
        let net = sigmoid_net(&mut params, 31);
        let x = inputs(5, 3, 32);
        let off = NeuronMask::from_off(4, &[1]);
        let mut model = InferenceModel::compile_with_neuron_masks(&net, &params, &[0]).unwrap();
        model.set_neuron_mask(0, &off).unwrap();
        let (seen, out) = observe_all(&mut model, &x, 5);
        let hidden = &seen.iter().find(|s| s.0 == 1).expect("layer 1 observed").2;
        for row in hidden.chunks(4) {
            assert_eq!(row[1].to_bits(), 0.0f32.to_bits(), "{row:?}");
            assert!(row[0] > 0.0 && row[2] > 0.0);
        }
        let mut masked = net.clone();
        masked.set_neuron_mask(0, off.clone());
        let mut want = InferenceModel::compile(&masked, &params).unwrap();
        assert_models_agree(&mut model, &mut want, &x, "by Linear");

        let mut by_layer = InferenceModel::compile_with_masks(&net, &params, &[0]).unwrap();
        by_layer.set_mask(0, &off).unwrap();
        let (seen, wrong) = observe_all(&mut by_layer, &x, 5);
        let hidden = &seen.iter().find(|s| s.0 == 1).expect("layer 1 observed").2;
        assert!(hidden.chunks(4).all(|row| row[1] == 0.5), "{hidden:?}");
        assert_ne!(wrong, out);

        model.set_neuron_mask(0, &NeuronMask::all_on(4)).unwrap();
        let mut plain = InferenceModel::compile(&net, &params).unwrap();
        assert_models_agree(&mut model, &mut plain, &x, "all on again");
    }

    /// On Linear → Tanh → Sigmoid → Linear the mask step of the first `Linear`'s neurons
    /// sits after the `Sigmoid`, so an off neuron reads exactly 0 at layer 2.
    #[test]
    fn neuron_mask_step_after_chained_activations() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(33);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 3, 4, &mut rng))
            .push(Tanh)
            .push(Sigmoid)
            .push(Linear::new(&mut params, 4, 2, &mut rng));
        let x = inputs(4, 3, 34);
        let mut model = InferenceModel::compile_with_neuron_masks(&net, &params, &[0]).unwrap();
        model
            .set_neuron_mask(0, &NeuronMask::from_off(4, &[2]))
            .unwrap();
        let (seen, _) = observe_all(&mut model, &x, 4);
        let layers: Vec<usize> = seen.iter().map(|s| s.0).collect();
        assert_eq!(layers, [0, 1, 2, 3]);
        for row in seen[2].2.chunks(4) {
            assert_eq!(row[2].to_bits(), 0.0f32.to_bits(), "{row:?}");
            assert!(row[1] > 0.0);
        }
        assert!(seen[1].2.chunks(4).all(|row| row[2] != 0.0));
    }

    /// `compile_with_neuron_masks` with an index that is not a `Linear` is `InvalidData`;
    /// `set_neuron_mask` on a layer that is not a `Linear` or with the wrong width is
    /// `InvalidData`, and on a `Linear` without a mask step `Unsupported`, each naming the
    /// `Linear` and leaving the model unchanged. A neuron mask set on the network before
    /// compiling has a mask step without being listed.
    #[test]
    fn set_neuron_mask_errors() {
        let mut params = ParamStore::new();
        let mut net = sigmoid_net(&mut params, 35);
        match InferenceModel::compile_with_neuron_masks(&net, &params, &[0, 1]) {
            Err(Error::InvalidData(msg)) => assert!(msg.contains("layer 1"), "{msg}"),
            other => panic!("expected InvalidData, got {:?}", other.map(|_| ())),
        }
        let x = inputs(1, 3, 36);
        let mut model = InferenceModel::compile_with_neuron_masks(&net, &params, &[0]).unwrap();
        let before = model.infer(&x).to_vec();
        let cases: [(usize, NeuronMask, &str); 3] = [
            (1, NeuronMask::all_off(4), "layer 1 is not a Linear"),
            (0, NeuronMask::all_off(3), "the linear layer 0"),
            (2, NeuronMask::all_off(2), "the linear layer 2"),
        ];
        for (linear, mask, want) in cases {
            let got = model.set_neuron_mask(linear, &mask);
            let msg = match (&got, linear) {
                (Err(Error::Unsupported(msg)), 2) | (Err(Error::InvalidData(msg)), 0 | 1) => msg,
                _ => panic!("linear {linear}: {got:?}"),
            };
            assert!(msg.contains(want), "{msg}");
        }
        assert_same(model.infer(&x), &before, "after the failed calls");
        net.set_neuron_mask(2, NeuronMask::all_on(2));
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        model.set_neuron_mask(2, &NeuronMask::all_off(2)).unwrap();
        assert_eq!(model.infer(&x), &[0.0, 0.0]);
    }

    /// `set_mask` on a layer with no mask step is `Unsupported`, and a mask of the wrong
    /// width `InvalidData`, both naming the layer and leaving the model unchanged;
    /// `compile_with_masks` with a layer out of range is `InvalidData`.
    #[test]
    fn set_mask_without_mask_step_is_unsupported() {
        let mut params = ParamStore::new();
        let net = digits(&mut params, 19);
        let x = inputs(1, 64, 20);
        let mut model = InferenceModel::compile_with_masks(&net, &params, &[1]).unwrap();
        let before = model.infer(&x).to_vec();
        for layer in [0, 2, 3] {
            match model.set_mask(layer, &NeuronMask::all_off(10)) {
                Err(Error::Unsupported(msg)) => {
                    assert!(msg.contains(&format!("layer {layer}")), "{msg}");
                }
                other => panic!("layer {layer}: expected Unsupported, got {other:?}"),
            }
        }
        match model.set_mask(1, &NeuronMask::all_off(31)) {
            Err(Error::InvalidData(msg)) => assert!(msg.contains("layer 1"), "{msg}"),
            other => panic!("expected InvalidData, got {other:?}"),
        }
        assert_same(model.infer(&x), &before, "after the failed calls");
        let mut plain = InferenceModel::compile(&net, &params).unwrap();
        assert!(matches!(
            plain.set_mask(1, &NeuronMask::all_on(32)),
            Err(Error::Unsupported(_))
        ));
        match InferenceModel::compile_with_masks(&net, &params, &[1, 3]) {
            Err(Error::InvalidData(msg)) => assert!(msg.contains("layer 3"), "{msg}"),
            other => panic!("expected InvalidData, got {:?}", other.map(|_| ())),
        }
    }
}
