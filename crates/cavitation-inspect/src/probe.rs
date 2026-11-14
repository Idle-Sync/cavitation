//! `Probe`: one input row run forward and backward beside training, on a graph, gradients
//! and buffers of its own, so the page can show every layer's values and one neuron's
//! weights for that row without touching the run's parameters or gradients.

use cavitation_core::{Error, Graph, ParamGrads, ParamStore, TensorView};
use cavitation_nn::{Dataset, Loss, Sequential, Targets, TargetsView, Trace};

/// How many of a neuron's inputs [`NeuronValues::top_inputs`] keeps: those whose
/// contribution to its pre-activation is largest in size.
///
/// ```
/// assert_eq!(cavitation_inspect::probe::TOP_INPUTS, 10);
/// ```
pub const TOP_INPUTS: usize = 10;

/// One layer's values for the probed row, as [`Probe::run`] leaves them.
///
/// Every layer has its output `a` and, when the loss depends on it, `∂L/∂a`. A layer whose
/// every output depends only on the input at the same position (an activation, a `Prelu`:
/// see [`Layer::is_elementwise`]) also has its input `z`, `∂L/∂z` and, when it can say,
/// its local derivative f′(z); other layers (`Linear`, `Softmax`, ...) have none of the
/// three. A masked layer's `a` is the output after the mask.
///
/// [`Layer::is_elementwise`]: cavitation_nn::Layer::is_elementwise
///
/// # Examples
///
/// ```
/// let layer = cavitation_inspect::probe::LayerValues::default();
/// assert!(layer.a().is_empty() && layer.z().is_none());
/// ```
#[derive(Clone, Debug, Default)]
pub struct LayerValues {
    /// The layer's output.
    a: Vec<f32>,
    /// The layer's input, when the layer is element-wise.
    z: Vec<f32>,
    /// The local derivative at the input, when the layer gives one.
    df: Vec<f32>,
    /// The gradient of the loss with respect to the output.
    grad_a: Vec<f32>,
    /// The gradient of the loss with respect to the input, when the layer is element-wise.
    grad_z: Vec<f32>,
    /// Whether `z` holds values.
    has_z: bool,
    /// Whether `df` holds values.
    has_df: bool,
    /// Whether `grad_a` holds values.
    has_grad_a: bool,
    /// Whether `grad_z` holds values.
    has_grad_z: bool,
}

impl LayerValues {
    /// The layer's output `a` for the probed row.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::LayerValues::default().a().is_empty());
    /// ```
    pub fn a(&self) -> &[f32] {
        &self.a
    }

    /// The layer's input `z`, for an element-wise layer; `None` for any other.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::LayerValues::default().z().is_none());
    /// ```
    pub fn z(&self) -> Option<&[f32]> {
        self.has_z.then_some(&self.z[..])
    }

    /// The local derivative f′(z) from [`Layer::local_derivative`]; `None` for a layer
    /// that gives none (one that is not element-wise, or one of your own without it).
    ///
    /// [`Layer::local_derivative`]: cavitation_nn::Layer::local_derivative
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::LayerValues::default().df().is_none());
    /// ```
    pub fn df(&self) -> Option<&[f32]> {
        self.has_df.then_some(&self.df[..])
    }

    /// `∂L/∂a`, the gradient of the probed row's loss with respect to the output; `None`
    /// when the loss does not depend on it.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::LayerValues::default().grad_a().is_none());
    /// ```
    pub fn grad_a(&self) -> Option<&[f32]> {
        self.has_grad_a.then_some(&self.grad_a[..])
    }

    /// `∂L/∂z`, the gradient with respect to the input of an element-wise layer; `None` for
    /// any other layer, and for the network's input.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::LayerValues::default().grad_z().is_none());
    /// ```
    pub fn grad_z(&self) -> Option<&[f32]> {
        self.has_grad_z.then_some(&self.grad_z[..])
    }
}

/// What one input contributes to a neuron's pre-activation: `weight · x`.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::probe::Contribution;
/// let c = Contribution { input: 3, x: 0.5, weight: -2.0, product: -1.0 };
/// assert_eq!(c.x * c.weight, c.product);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Contribution {
    /// The input's index in the `Linear`'s input.
    pub input: usize,
    /// The input's value for the probed row.
    pub x: f32,
    /// The weight from that input to the neuron.
    pub weight: f32,
    /// `weight · x`.
    pub product: f32,
}

/// One neuron of the probed row, as [`Probe::neuron`] fills it: the `Linear` it comes
/// from, its weights, the size of its outgoing weights and the inputs that drive it most.
///
/// A neuron belongs to a `Linear` when its layer is that `Linear` or an element-wise
/// layer after it (with only element-wise layers between). A neuron of another layer
/// (`Softmax`, say) has no `Linear`: its weights are empty and its sums are absent.
///
/// # Examples
///
/// ```
/// let n = cavitation_inspect::probe::NeuronValues::default();
/// assert!(n.linear().is_none() && n.top_inputs().is_empty());
/// ```
#[derive(Clone, Debug, Default)]
pub struct NeuronValues {
    /// The layer the neuron was chosen in.
    layer: usize,
    /// The neuron's index in that layer's output.
    neuron: usize,
    /// The layer index of the `Linear` it comes from.
    linear: Option<usize>,
    /// Whether the neuron is switched on.
    on: bool,
    /// Its incoming weights, the `Linear`'s weight row.
    weights: Vec<f32>,
    /// Its bias, `None` without a `Linear` or when the `Linear` has no bias.
    bias: Option<f32>,
    /// `bias + Σ weight · x` for the probed row (`Σ weight · x` without a bias), summed in
    /// `f64`.
    pre_activation: f32,
    /// The Euclidean norm of its column in the next `Linear`'s weight.
    outgoing_norm: Option<f32>,
    /// The largest contributions, largest in size first.
    top: [Contribution; TOP_INPUTS],
    /// How many of `top` are filled.
    top_len: usize,
}

impl NeuronValues {
    /// The layer the neuron was chosen in.
    ///
    /// ```
    /// assert_eq!(cavitation_inspect::probe::NeuronValues::default().layer(), 0);
    /// ```
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// The neuron's index in its layer's output.
    ///
    /// ```
    /// assert_eq!(cavitation_inspect::probe::NeuronValues::default().neuron(), 0);
    /// ```
    pub fn neuron(&self) -> usize {
        self.neuron
    }

    /// The layer index of the `Linear` the neuron comes from, if any.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::NeuronValues::default().linear().is_none());
    /// ```
    pub fn linear(&self) -> Option<usize> {
        self.linear
    }

    /// Whether the neuron is switched on: its `Linear`'s neuron mask (or, for a neuron
    /// without a `Linear`, its layer's mask) does not switch it off.
    ///
    /// ```
    /// assert!(!cavitation_inspect::probe::NeuronValues::default().on());
    /// ```
    pub fn on(&self) -> bool {
        self.on
    }

    /// The neuron's incoming weights (its row of the `Linear`'s weight); empty without a
    /// `Linear`.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::NeuronValues::default().weights().is_empty());
    /// ```
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// The neuron's bias; `None` without a `Linear`, or when its `Linear` has no bias.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::NeuronValues::default().bias().is_none());
    /// ```
    pub fn bias(&self) -> Option<f32> {
        self.bias
    }

    /// `bias + Σ weight · x` for the probed row (`Σ weight · x` when the `Linear` has no
    /// bias), summed in `f64` from the weights, so it is the pre-activation before any mask;
    /// `None` without a `Linear`.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::NeuronValues::default().pre_activation().is_none());
    /// ```
    pub fn pre_activation(&self) -> Option<f32> {
        self.linear.map(|_| self.pre_activation)
    }

    /// The Euclidean norm of the neuron's outgoing weights (its column in the next
    /// `Linear`, after the element-wise layers that follow it); `None` when no `Linear`
    /// reads it directly.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::NeuronValues::default().outgoing_norm().is_none());
    /// ```
    pub fn outgoing_norm(&self) -> Option<f32> {
        self.outgoing_norm
    }

    /// Up to [`TOP_INPUTS`] inputs whose contribution `weight · x` is largest in size, the
    /// largest first (a NaN contribution counts as largest).
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::NeuronValues::default().top_inputs().is_empty());
    /// ```
    pub fn top_inputs(&self) -> &[Contribution] {
        &self.top[..self.top_len]
    }
}

/// One row of a data set run forward and backward with the training loss, on a [`Graph`],
/// a [`ParamGrads`] set made with [`ParamGrads::like`], a [`Trace`] and buffers of its own.
///
/// The run reads the parameter values in place and writes gradients only to its own set,
/// so the training run's parameters, gradients and optimiser state are untouched. Every
/// buffer is made on the first run and reused, so later runs of the same network allocate
/// nothing.
///
/// The probe's gradient set belongs to the [`ParamStore`] of its first run; to probe a
/// network whose parameters live in another store, use a new `Probe`.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_inspect::probe::Probe;
/// use cavitation_nn::{Dataset, Linear, Loss, Relu, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 3, 2, &mut rng));
/// let data = Dataset {
///     x: Tensor::from_vec(vec![0.5f32, -1.0, 1.0, 2.0], &[2, 2]).unwrap(),
///     targets: Targets::Classes(vec![1, 0]),
/// };
/// let mut probe = Probe::new();
/// probe.run(&net, &params, &data, 1, Loss::CrossEntropy)?;
/// assert_eq!(probe.layers().len(), 3);
/// assert!(probe.layers()[1].df().is_some()); // the ReLU
/// assert!(probe.layers()[0].df().is_none()); // a Linear
/// let neuron = probe.neuron(&net, &params, 1, 2)?;
/// assert_eq!(neuron.linear(), Some(0));
/// assert_eq!(neuron.weights().len(), 2);
/// # Ok::<(), cavitation_core::Error>(())
/// ```
#[derive(Debug, Default)]
pub struct Probe {
    /// The storage the probe's tape records on.
    graph: Graph<f32>,
    /// The probe's own gradients, made on the first run.
    grads: Option<ParamGrads<f32>>,
    /// Each layer's input and output on the probe's tape.
    trace: Trace,
    /// Each layer's values from the last run.
    layers: Vec<LayerValues>,
    /// The probed row's input.
    input: Vec<f32>,
    /// The probed row's value targets (empty for class targets).
    target: Vec<f32>,
    /// The probed row's class label.
    label: Option<usize>,
    /// The probed row's index.
    row: usize,
    /// The probed row's loss.
    loss: f32,
    /// Whether a run has filled the values.
    ran: bool,
    /// The last neuron filled by [`neuron`](Self::neuron).
    neuron: NeuronValues,
}

/// Whether `loss` takes class labels.
fn needs_classes(loss: Loss) -> bool {
    matches!(loss, Loss::CrossEntropy | Loss::Nll)
}

/// Copies `src` into `dst`, keeping `dst`'s buffer.
fn copy(dst: &mut Vec<f32>, src: &[f32]) {
    dst.clear();
    dst.extend_from_slice(src);
}

impl Probe {
    /// A probe with no buffers yet; the first run makes them.
    ///
    /// ```
    /// let probe = cavitation_inspect::probe::Probe::new();
    /// assert!(probe.layers().is_empty());
    /// ```
    pub fn new() -> Probe {
        Probe::default()
    }

    /// Runs row `row` of `data` through `model` with the values in `params`, records `loss`
    /// against the row's target as training does, and runs the backward pass into the
    /// probe's own gradients; then fills every layer's [`LayerValues`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`], with nothing changed, if `row` is not a row of
    /// `data`, if the kind of `data`'s targets is not the kind `loss` needs, if the row's
    /// class label is not below the model's output width, or if value targets are not as
    /// wide as the output.
    ///
    /// # Panics
    ///
    /// Panics as [`Sequential::forward`] does when `data`'s width does not suit the model,
    /// and if `params` is not the store of the probe's first run.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_inspect::probe::Probe;
    /// use cavitation_nn::{Dataset, Linear, Loss, Sequential, Targets};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2]).unwrap(),
    ///     targets: Targets::Values(Tensor::from_vec(vec![0.5f32], &[1, 1]).unwrap()),
    /// };
    /// let mut probe = Probe::new();
    /// assert!(probe.run(&net, &params, &data, 1, Loss::Mse).is_err());
    /// probe.run(&net, &params, &data, 0, Loss::Mse)?;
    /// assert_eq!(probe.layers()[0].grad_a().unwrap().len(), 1);
    /// # Ok::<(), cavitation_core::Error>(())
    /// ```
    pub fn run(
        &mut self,
        model: &Sequential,
        params: &ParamStore,
        data: &Dataset,
        row: usize,
        loss: Loss,
    ) -> Result<(), Error> {
        let rows = data.len();
        if row >= rows {
            return Err(Error::InvalidData(format!(
                "probe: row {row} is out of range for {rows} rows"
            )));
        }
        let classes = matches!(data.targets, Targets::Classes(_));
        if needs_classes(loss) != classes {
            return Err(Error::InvalidData(format!(
                "probe: Loss::{loss:?} does not take the {} targets of the probe rows",
                if classes { "class" } else { "value" }
            )));
        }
        let features = data.features();
        let x = TensorView::from_slice(
            &data.x.data()[row * features..(row + 1) * features],
            &[1, features],
        )
        .expect("one whole row of x has shape [1, features]");
        let grads = self
            .grads
            .get_or_insert_with(|| ParamGrads::like(&params.values));
        grads.zero_grad();
        let mut tape = self.graph.tape(&params.values);
        let input = tape.input(x);
        let out = model.forward_traced(&mut tape, input, &mut self.trace);
        let width = tape.shape(out).dims().last().copied().unwrap_or(0);
        let targets = match &data.targets {
            Targets::Classes(labels) => {
                let label = labels[row];
                if label >= width {
                    return Err(Error::InvalidData(format!(
                        "probe: the label of row {row} is {label}, but the model has {width} \
                         outputs"
                    )));
                }
                TargetsView::Classes(&labels[row..row + 1])
            }
            Targets::Values(t) => {
                let columns = t.shape().dims().last().copied().unwrap_or(0);
                if columns != width {
                    return Err(Error::InvalidData(format!(
                        "probe: the value targets have {columns} columns, but the model \
                         outputs {width}"
                    )));
                }
                TargetsView::Values(
                    TensorView::from_slice(
                        &t.data()[row * columns..(row + 1) * columns],
                        &[1, columns],
                    )
                    .expect("one whole row of the targets has shape [1, columns]"),
                )
            }
        };
        let l = loss.record(&mut tape, out, targets);
        self.loss = tape.value(l).data()[0];
        tape.backward(l, grads);

        self.layers.resize_with(model.len(), LayerValues::default);
        for (i, values) in self.layers.iter_mut().enumerate() {
            let layer = model.layer(i);
            let (zin, aout) = (self.trace.input(i), self.trace.output(i));
            copy(&mut values.a, tape.value(aout).data());
            values.has_grad_a = match tape.grad(aout) {
                Some(g) => {
                    copy(&mut values.grad_a, g.data());
                    true
                }
                None => false,
            };
            let elementwise = layer.is_elementwise();
            values.has_z = elementwise;
            values.has_grad_z = false;
            if elementwise {
                copy(&mut values.z, tape.value(zin).data());
                if let Some(g) = tape.grad(zin) {
                    copy(&mut values.grad_z, g.data());
                    values.has_grad_z = true;
                }
            }
            let inputs = tape.value(zin).data().len();
            values.df.resize(inputs, 0.0);
            values.has_df = layer.local_derivative(&tape, zin, &mut values.df);
        }
        drop(tape);
        copy(&mut self.input, x.data());
        match &data.targets {
            Targets::Classes(labels) => {
                self.label = Some(labels[row]);
                self.target.clear();
            }
            Targets::Values(t) => {
                self.label = None;
                let columns = width;
                copy(
                    &mut self.target,
                    &t.data()[row * columns..(row + 1) * columns],
                );
            }
        }
        self.row = row;
        self.ran = true;
        Ok(())
    }

    /// Each layer's values from the last run, in layer order; empty before the first run.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::Probe::new().layers().is_empty());
    /// ```
    pub fn layers(&self) -> &[LayerValues] {
        if self.ran { &self.layers } else { &[] }
    }

    /// The probed row's input values.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::Probe::new().input().is_empty());
    /// ```
    pub fn input(&self) -> &[f32] {
        &self.input
    }

    /// The index of the probed row.
    ///
    /// ```
    /// assert_eq!(cavitation_inspect::probe::Probe::new().row(), 0);
    /// ```
    pub fn row(&self) -> usize {
        self.row
    }

    /// The probed row's class label, for class targets.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::Probe::new().label().is_none());
    /// ```
    pub fn label(&self) -> Option<usize> {
        self.label
    }

    /// The probed row's value targets, for value targets; empty for class targets.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::Probe::new().target().is_empty());
    /// ```
    pub fn target(&self) -> &[f32] {
        &self.target
    }

    /// The probed row's loss, as training defines it.
    ///
    /// ```
    /// assert_eq!(cavitation_inspect::probe::Probe::new().loss(), 0.0);
    /// ```
    pub fn loss(&self) -> f32 {
        self.loss
    }

    /// The network's output for the probed row: the last layer's `a`.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::Probe::new().output().is_empty());
    /// ```
    pub fn output(&self) -> &[f32] {
        self.layers().last().map_or(&[], |l| l.a())
    }

    /// The index of the first largest output, ignoring NaN, for class targets; `None` for
    /// value targets, before a run, or when every output is NaN.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::Probe::new().predicted().is_none());
    /// ```
    pub fn predicted(&self) -> Option<usize> {
        self.label?;
        let mut best: Option<(usize, f32)> = None;
        for (i, &v) in self.output().iter().enumerate() {
            if !v.is_nan() && best.is_none_or(|(_, b)| v > b) {
                best = Some((i, v));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Fills and returns neuron `neuron` of layer `layer` for the last run: its `Linear`,
    /// whether it is on, its weights and bias, its pre-activation, the norm of its outgoing
    /// weights and its largest contributions. Reuses its buffers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if no run has been made with a model of this many
    /// layers, if `layer` is not a layer, or if `neuron` is not below the layer's width.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_inspect::probe::Probe;
    /// use cavitation_nn::{Dataset, Linear, Loss, Sequential, Targets, Tanh};
    /// let mut params = ParamStore::new();
    /// let mut rng = Rng::seed(0);
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 3, 4, &mut rng))
    ///     .push(Tanh)
    ///     .push(Linear::new(&mut params, 4, 2, &mut rng));
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![1.0f32, -2.0, 0.5], &[1, 3]).unwrap(),
    ///     targets: Targets::Classes(vec![1]),
    /// };
    /// let mut probe = Probe::new();
    /// probe.run(&net, &params, &data, 0, Loss::CrossEntropy)?;
    /// // The Tanh's input z is the neuron's pre-activation.
    /// let z = probe.layers()[1].z().unwrap()[3];
    /// let n = probe.neuron(&net, &params, 1, 3)?;
    /// assert_eq!(n.linear(), Some(0));
    /// assert_eq!(n.top_inputs().len(), 3);
    /// assert!(n.outgoing_norm().unwrap() > 0.0);
    /// assert!((n.pre_activation().unwrap() - z).abs() < 1e-5);
    /// assert!(probe.neuron(&net, &params, 1, 4).is_err());
    /// # Ok::<(), cavitation_core::Error>(())
    /// ```
    pub fn neuron(
        &mut self,
        model: &Sequential,
        params: &ParamStore,
        layer: usize,
        neuron: usize,
    ) -> Result<&NeuronValues, Error> {
        if !self.ran || self.layers.len() != model.len() {
            return Err(Error::InvalidData(
                "probe: no row of this model has been run".to_string(),
            ));
        }
        let layers = model.len();
        if layer >= layers {
            return Err(Error::InvalidData(format!(
                "probe: layer {layer} is out of range for {layers} layers"
            )));
        }
        let width = self.layers[layer].a.len();
        if neuron >= width {
            return Err(Error::InvalidData(format!(
                "probe: neuron {neuron} is out of range for layer {layer}, which has width \
                 {width}"
            )));
        }
        let linear = owning_linear(model, layer);
        let n = &mut self.neuron;
        n.layer = layer;
        n.neuron = neuron;
        n.linear = linear;
        n.weights.clear();
        n.top_len = 0;
        n.outgoing_norm = None;
        n.bias = None;
        n.pre_activation = 0.0;
        let Some(at) = linear else {
            n.on = model.mask(layer).is_none_or(|m| m.is_on(neuron));
            return Ok(&self.neuron);
        };
        let lin = model.linear(at).expect("owning_linear names a Linear");
        n.on = model.neuron_mask(at).is_none_or(|m| m.is_on(neuron));
        let inputs = lin.inputs();
        let weights = &params.value(lin.weight()).data()[neuron * inputs..(neuron + 1) * inputs];
        n.weights.extend_from_slice(weights);
        n.bias = lin.bias().map(|id| params.value(id).data()[neuron]);
        let x: &[f32] = if at == 0 {
            &self.input
        } else {
            &self.layers[at - 1].a
        };
        let mut sum = n.bias.map_or(0.0, f64::from);
        for (i, (&w, &v)) in weights.iter().zip(x).enumerate() {
            let product = w * v;
            sum += f64::from(w) * f64::from(v);
            insert_top(
                &mut n.top,
                &mut n.top_len,
                Contribution {
                    input: i,
                    x: v,
                    weight: w,
                    product,
                },
            );
        }
        n.pre_activation = sum as f32;
        let mut next = layer + 1;
        while next < layers && model.layer(next).is_elementwise() {
            next += 1;
        }
        if let Some(reader) = model.linear(next) {
            let w = params.value(reader.weight()).data();
            let columns = reader.inputs();
            if neuron < columns {
                let squares: f64 = (0..reader.outputs())
                    .map(|k| f64::from(w[k * columns + neuron]).powi(2))
                    .sum();
                n.outgoing_norm = Some(squares.sqrt() as f32);
            }
        }
        Ok(&self.neuron)
    }

    /// The neuron [`neuron`](Self::neuron) last filled.
    ///
    /// ```
    /// assert!(cavitation_inspect::probe::Probe::new().last_neuron().weights().is_empty());
    /// ```
    pub fn last_neuron(&self) -> &NeuronValues {
        &self.neuron
    }
}

/// The layer index of the `Linear` whose neurons layer `layer` holds: `layer` itself when it
/// is a `Linear`, or the `Linear` before it when only element-wise layers lie between.
fn owning_linear(model: &Sequential, layer: usize) -> Option<usize> {
    let mut at = layer;
    loop {
        if model.linear(at).is_some() {
            return Some(at);
        }
        if at == 0 || !model.layer(at).is_elementwise() {
            return None;
        }
        at -= 1;
    }
}

/// The size by which contributions are ranked: NaN ranks above everything.
fn rank(c: &Contribution) -> f32 {
    if c.product.is_nan() {
        f32::INFINITY
    } else {
        c.product.abs()
    }
}

/// Inserts `c` into the first `len` entries of `top`, kept largest first, if it ranks
/// among the largest [`TOP_INPUTS`].
fn insert_top(top: &mut [Contribution; TOP_INPUTS], len: &mut usize, c: Contribution) {
    let key = rank(&c);
    let mut at = *len;
    while at > 0 && rank(&top[at - 1]) < key {
        at -= 1;
    }
    if at >= TOP_INPUTS {
        return;
    }
    let end = (*len).min(TOP_INPUTS - 1);
    for j in (at..end).rev() {
        top[j + 1] = top[j];
    }
    top[at] = c;
    *len = (*len + 1).min(TOP_INPUTS);
}

/// Tests of the probe: the values it fills for each layer and for the chosen neuron.
#[cfg(test)]
mod tests {
    use super::*;

    /// The ranking keeps the largest ten in size, largest first.
    #[test]
    fn top_keeps_the_largest_ten() {
        let mut top = [Contribution::default(); TOP_INPUTS];
        let mut len = 0;
        let values = [
            0.5f32, -3.0, 2.0, 0.1, -0.2, 7.0, 1.0, -1.5, 0.3, 4.0, -6.0, 0.0, 2.5,
        ];
        for (i, &p) in values.iter().enumerate() {
            insert_top(
                &mut top,
                &mut len,
                Contribution {
                    input: i,
                    x: p,
                    weight: 1.0,
                    product: p,
                },
            );
        }
        assert_eq!(len, 10);
        let got: Vec<f32> = top.iter().map(|c| c.product).collect();
        assert_eq!(got, [7.0, -6.0, 4.0, -3.0, 2.5, 2.0, -1.5, 1.0, 0.5, 0.3]);
    }

    /// Without a full list every entry is kept, in order of size.
    #[test]
    fn top_with_few_inputs() {
        let mut top = [Contribution::default(); TOP_INPUTS];
        let mut len = 0;
        for (i, p) in [1.0f32, f32::NAN, -2.0].into_iter().enumerate() {
            insert_top(
                &mut top,
                &mut len,
                Contribution {
                    input: i,
                    x: p,
                    weight: 1.0,
                    product: p,
                },
            );
        }
        assert_eq!(len, 3);
        assert_eq!(
            top[..3].iter().map(|c| c.input).collect::<Vec<_>>(),
            [1, 2, 0]
        );
    }

    /// A neuron of a `Linear` without a bias reports no bias, and its pre-activation is
    /// `Σ weight · x` alone, the tape's value.
    #[test]
    fn neuron_without_a_bias() {
        use cavitation_core::{Rng, Tensor};
        use cavitation_nn::{Dataset, Init, Linear, Loss, Tanh, Targets};
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(4);
        let net = Sequential::new()
            .push(Linear::without_bias(
                &mut params,
                3,
                4,
                Init::XavierUniform,
                &mut rng,
            ))
            .push(Tanh)
            .push(Linear::new(&mut params, 4, 2, &mut rng));
        let data = Dataset {
            x: Tensor::from_vec(vec![1.0f32, -2.0, 0.5], &[1, 3]).unwrap(),
            targets: Targets::Classes(vec![1]),
        };
        let mut probe = Probe::new();
        probe
            .run(&net, &params, &data, 0, Loss::CrossEntropy)
            .unwrap();
        let z = probe.layers()[1].z().unwrap()[2];
        let n = probe.neuron(&net, &params, 1, 2).unwrap();
        assert_eq!(n.linear(), Some(0));
        assert_eq!(n.bias(), None);
        let w = params.value(net.linear(0).unwrap().weight()).data()[6..9].to_vec();
        let want = f64::from(w[0]) - 2.0 * f64::from(w[1]) + 0.5 * f64::from(w[2]);
        let got = n.pre_activation().unwrap();
        assert_eq!(got, want as f32);
        assert!((got - z).abs() < 1e-6, "{got} against the tape's {z}");
        let mut w = crate::json::JsonWriter::new();
        crate::wire::neuron(&mut w, 0, &probe, probe.last_neuron());
        assert!(
            w.as_str().contains(r#""bias":null,"has_bias":false"#),
            "{}",
            w.as_str()
        );
        // The output layer has a bias.
        assert!(probe.neuron(&net, &params, 2, 0).unwrap().bias().is_some());
    }
}
