//! `Sequential` and `Trace`: layers run one after another, with optional neuron masks on
//! their outputs, and a record of each layer's input and output.

use std::sync::Arc;

use cavitation_core::{ParamId, Real, Tape, Var};

use crate::activation::last_dim;
use crate::mask::LayerMask;
use crate::{Layer, Linear, NeuronMask};

/// Each layer's input and output [`Var`] from one forward pass. A masked layer's output is
/// the masked value.
///
/// Fill it with [`Sequential::forward_traced`]. Reuse one trace for every step: it keeps its
/// capacity, so tracing allocates nothing after the first step.
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Rng, Tensor};
/// use cavitation_nn::{Linear, Relu, Sequential, Trace};
/// let mut params = ParamStore::new();
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
///     .push(Relu);
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&params.values);
/// let x = tape.input(Tensor::full(&[4, 2], 1.0f32).view());
/// let mut trace = Trace::new();
/// let y = net.forward_traced(&mut tape, x, &mut trace);
/// assert_eq!(trace.len(), 2);
/// assert_eq!(trace.input(0), x);
/// assert_eq!(trace.output(1), y);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Trace {
    /// The input of each layer.
    inputs: Vec<Var>,
    /// The output of each layer.
    outputs: Vec<Var>,
}

impl Trace {
    /// An empty trace.
    ///
    /// ```
    /// use cavitation_nn::Trace;
    /// assert!(Trace::new().is_empty());
    /// ```
    pub fn new() -> Trace {
        Trace::default()
    }

    /// An empty trace with room for `layers` layers, so tracing a network of that many layers
    /// allocates nothing for the trace's layer list. A manual training loop that makes its trace
    /// this way and passes it to [`Sequential::forward_traced`] each step saves the first
    /// pass's allocation (a trace made with [`Trace::new`] and reused stops allocating after
    /// its first pass too).
    ///
    /// ```
    /// use cavitation_nn::Trace;
    /// let trace = Trace::with_capacity(5);
    /// assert!(trace.is_empty());
    /// ```
    pub fn with_capacity(layers: usize) -> Trace {
        Trace {
            inputs: Vec::with_capacity(layers),
            outputs: Vec::with_capacity(layers),
        }
    }

    /// The number of layers recorded.
    ///
    /// ```
    /// use cavitation_nn::Trace;
    /// assert_eq!(Trace::new().len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    /// Whether nothing is recorded.
    ///
    /// ```
    /// use cavitation_nn::Trace;
    /// assert!(Trace::new().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// How many layers the trace can record without allocating.
    ///
    /// ```
    /// use cavitation_nn::Trace;
    /// assert_eq!(Trace::new().capacity(), 0);
    /// ```
    pub fn capacity(&self) -> usize {
        self.inputs.capacity().min(self.outputs.capacity())
    }

    /// The input of layer `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not a recorded layer.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{Relu, Sequential, Trace};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::full(&[1, 2], 1.0f32).view());
    /// let mut trace = Trace::new();
    /// Sequential::new().push(Relu).forward_traced(&mut tape, x, &mut trace);
    /// assert_eq!(trace.input(0), x);
    /// ```
    pub fn input(&self, i: usize) -> Var {
        assert!(
            i < self.len(),
            "trace: layer {i} out of range for {} layers",
            self.len()
        );
        self.inputs[i]
    }

    /// The output of layer `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not a recorded layer.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{Relu, Sequential, Trace};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::full(&[1, 2], 1.0f32).view());
    /// let mut trace = Trace::new();
    /// let y = Sequential::new().push(Relu).forward_traced(&mut tape, x, &mut trace);
    /// assert_eq!(trace.output(0), y);
    /// ```
    pub fn output(&self, i: usize) -> Var {
        assert!(
            i < self.len(),
            "trace: layer {i} out of range for {} layers",
            self.len()
        );
        self.outputs[i]
    }
}

/// Layers run one after another, each taking the previous layer's output.
///
/// Its parameters are named `"{index}.{local name}"`, so layer 0's weight is `0.weight`, as
/// in a PyTorch `nn.Sequential`. A `Sequential` is not itself a layer, so they do not nest.
///
/// A `Sequential` is `Clone`. A clone shares its layers with the original, and they never
/// change after [`push`](Self::push), so a layer that holds its own interior mutability (a
/// `Mutex` field, say) is shared between the copies. The clone copies the neuron masks:
/// setting or clearing a mask on one copy leaves the other as it was. A layer keeps only
/// [`ParamId`]s, so for an independent network clone the
/// [`ParamStore`](cavitation_core::ParamStore) too: `params.clone()` copies every value, and
/// training the clone then leaves the original's values untouched.
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Rng, Tensor};
/// use cavitation_nn::{Linear, Relu, Sequential};
/// let mut params = ParamStore::new();
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
///     .push(Relu);
/// let copy = net.clone();
/// let copy_params = params.clone();
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&copy_params.values);
/// let x = tape.input(Tensor::full(&[1, 2], 1.0f32).view());
/// let y = copy.forward(&mut tape, x);
/// assert_eq!(copy.len(), net.len());
/// assert_eq!(tape.shape(y).dims(), &[1, 3]);
/// ```
///
/// Any layer can carry a [`NeuronMask`] ([`set_mask`](Self::set_mask)): the forward pass sets
/// the off neurons of that layer's output to exactly 0, and they pass no gradient back.
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Rng, Tensor};
/// use cavitation_nn::{Linear, Relu, Sequential};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 4, 8, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 8, 2, &mut rng));
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&params.values);
/// let x = tape.input(Tensor::full(&[5, 4], 0.5f32).view());
/// let y = net.forward(&mut tape, x);
/// assert_eq!(tape.shape(y).dims(), &[5, 2]);
/// ```
#[derive(Clone)]
pub struct Sequential<T: Real = f32> {
    /// The layers, in order. Shared between clones; a layer never changes after `push`.
    layers: Vec<Arc<dyn Layer<T>>>,
    /// The neuron mask on each layer's output, if any; one entry per layer.
    masks: Vec<Option<LayerMask<T>>>,
}

impl Sequential<f32> {
    /// An empty `f32` network.
    ///
    /// ```
    /// use cavitation_nn::Sequential;
    /// assert!(Sequential::new().is_empty());
    /// ```
    pub fn new() -> Sequential<f32> {
        Sequential::new_generic()
    }
}

impl Default for Sequential<f32> {
    /// An empty `f32` network.
    fn default() -> Self {
        Sequential::new()
    }
}

impl<T: Real> Sequential<T> {
    /// An empty network for any [`Real`] type (for example `f64`, for gradient checks).
    ///
    /// ```
    /// use cavitation_nn::Sequential;
    /// assert_eq!(Sequential::<f64>::new_generic().len(), 0);
    /// ```
    pub fn new_generic() -> Sequential<T> {
        Sequential {
            layers: Vec::new(),
            masks: Vec::new(),
        }
    }

    /// Adds `layer` at the end and returns the network.
    ///
    /// ```
    /// use cavitation_nn::{Relu, Sequential, Tanh};
    /// let net = Sequential::new().push(Relu).push(Tanh);
    /// assert_eq!(net.len(), 2);
    /// ```
    pub fn push(mut self, layer: impl Layer<T> + 'static) -> Sequential<T> {
        self.layers.push(Arc::new(layer));
        self.masks.push(None);
        self
    }

    /// Adds a layer already shared with another network at the end, so that both hold the
    /// same layer: for a copy that rebuilds some layers and keeps the rest.
    pub(crate) fn push_shared(mut self, layer: Arc<dyn Layer<T>>) -> Sequential<T> {
        self.layers.push(layer);
        self.masks.push(None);
        self
    }

    /// Layer `i`, shared.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()`.
    pub(crate) fn shared_layer(&self, i: usize) -> Arc<dyn Layer<T>> {
        Arc::clone(&self.layers[i])
    }

    /// The number of layers.
    ///
    /// ```
    /// use cavitation_nn::{Relu, Sequential};
    /// assert_eq!(Sequential::new().push(Relu).len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether there are no layers.
    ///
    /// ```
    /// use cavitation_nn::Sequential;
    /// assert!(Sequential::new().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Layer `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()`; the message names the index and the length.
    ///
    /// ```
    /// use cavitation_nn::{Relu, Sequential};
    /// let net = Sequential::new().push(Relu);
    /// assert_eq!(net.layer(0).name(), "relu");
    /// ```
    pub fn layer(&self, i: usize) -> &dyn Layer<T> {
        assert!(
            i < self.layers.len(),
            "sequential: layer index {i} out of range for {} layers",
            self.layers.len()
        );
        &*self.layers[i]
    }

    /// Puts `mask` on layer `layer`'s output, replacing any mask already there. From the next
    /// forward pass on, each off neuron's output (its position on the output's last axis) is
    /// exactly 0, even where the layer gives infinity or NaN, and its gradient is 0, so
    /// nothing flows back through it.
    ///
    /// This addresses a mask by layer index. To turn off the neurons of a `Linear`, address
    /// them by the `Linear` instead, with [`set_neuron_mask`](Self::set_neuron_mask) or
    /// [`set_neuron`](Self::set_neuron): they put the mask on
    /// [`neuron_layer`](Self::neuron_layer), after the activations that follow the `Linear`,
    /// so the neuron outputs 0 whatever the activation gives at 0. A compiled model has the
    /// same two routes, [`InferenceModel::set_mask`](crate::InferenceModel::set_mask) by layer
    /// and [`InferenceModel::set_neuron_mask`](crate::InferenceModel::set_neuron_mask) by
    /// `Linear`.
    ///
    /// The mask's width is checked against the layer's output at the next forward pass.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= len()`; the message names the index and the valid range.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Rng, Tensor};
    /// use cavitation_nn::{Linear, NeuronMask, Sequential, Sigmoid};
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    ///     .push(Sigmoid);
    /// let mut mask = NeuronMask::all_on(3);
    /// mask.set(1, false);
    /// net.set_mask(1, mask);
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&params.values);
    /// let x = tape.input(Tensor::full(&[1, 2], 1.0f32).view());
    /// let y = net.forward(&mut tape, x);
    /// assert_eq!(tape.value(y).data()[1], 0.0);
    /// ```
    pub fn set_mask(&mut self, layer: usize, mask: NeuronMask) {
        self.check_layer(layer, "mask");
        self.masks[layer] = Some(LayerMask::new(mask));
    }

    /// Removes the mask from layer `layer`, if it has one; the layer's output is used as it
    /// is again, and the forward pass records nothing extra for it.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= len()`; the message names the index and the valid range.
    ///
    /// ```
    /// use cavitation_nn::{NeuronMask, Relu, Sequential};
    /// let mut net = Sequential::new().push(Relu);
    /// net.set_mask(0, NeuronMask::all_on(4));
    /// net.clear_mask(0);
    /// assert!(net.mask(0).is_none());
    /// ```
    pub fn clear_mask(&mut self, layer: usize) {
        self.check_layer(layer, "clear the mask of");
        self.masks[layer] = None;
    }

    /// The mask on layer `layer`, if it has one.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= len()`; the message names the index and the valid range.
    ///
    /// ```
    /// use cavitation_nn::{NeuronMask, Relu, Sequential};
    /// let mut net = Sequential::new().push(Relu);
    /// assert!(net.mask(0).is_none());
    /// net.set_mask(0, NeuronMask::all_on(4));
    /// assert_eq!(net.mask(0).unwrap().width(), 4);
    /// ```
    pub fn mask(&self, layer: usize) -> Option<&NeuronMask> {
        self.check_layer(layer, "read the mask of");
        self.masks[layer].as_ref().map(|m| &m.mask)
    }

    /// Layer `i` as a [`Linear`], or `None` if it is another kind of layer or there is no
    /// layer `i`.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    ///     .push(Relu);
    /// assert_eq!(net.linear(0).unwrap().outputs(), 3);
    /// assert!(net.linear(1).is_none() && net.linear(9).is_none());
    /// ```
    pub fn linear(&self, i: usize) -> Option<&Linear> {
        self.layers.get(i)?.as_linear()
    }

    /// Every [`Linear`] layer with its layer index, in order.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    ///     .push(Relu)
    ///     .push(Linear::new(&mut params, 3, 1, &mut Rng::seed(1)));
    /// let at: Vec<usize> = net.linears().map(|(i, _)| i).collect();
    /// assert_eq!(at, [0, 2]);
    /// ```
    pub fn linears(&self) -> impl Iterator<Item = (usize, &Linear)> + '_ {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(i, layer)| layer.as_linear().map(|linear| (i, linear)))
    }

    /// The parameter named `name` in the form of [`parameter_names`](Self::parameter_names),
    /// `"{layer index}.{local name}"` such as `"2.bias"`, or `None` if there is no such
    /// parameter. It does not allocate for the built-in layers; a layer of your own that keeps
    /// the default [`Layer::parameter`] may.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Sequential};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
    /// let bias = layer.bias();
    /// let net = Sequential::new().push(layer);
    /// assert_eq!(net.param("0.bias"), Some(bias));
    /// assert_eq!(net.param("1.bias"), None);
    /// ```
    pub fn param(&self, name: &str) -> Option<ParamId> {
        let (index, local) = name.split_once('.')?;
        let digits = !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit());
        if !digits || (index.len() > 1 && index.starts_with('0')) {
            return None;
        }
        self.layers
            .get(index.parse::<usize>().ok()?)?
            .parameter(local)
    }

    /// The index of the layer whose output holds the neurons of the `Linear` at layer
    /// `linear`, as the next layer receives them: the last layer of the run of element-wise
    /// layers right after it (each one whose [`Layer::is_elementwise`] is true: an
    /// activation, a `Prelu`, or a layer of your own that says so), or the `Linear` itself
    /// when the layer after it is not element-wise. Masks on a hidden neuron go on that
    /// layer, so that the neuron's output is 0 whatever the activations give at 0: on
    /// `Linear → Tanh → Sigmoid` it is the `Sigmoid`, so an off neuron sends 0, not
    /// `sigmoid(0) = 0.5`.
    ///
    /// # Panics
    ///
    /// Panics if layer `linear` is not a `Linear`; the message names the index and the
    /// layer's kind.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    ///     .push(Relu)
    ///     .push(Linear::new(&mut params, 3, 1, &mut Rng::seed(1)));
    /// assert_eq!((net.neuron_layer(0), net.neuron_layer(2)), (1, 2));
    /// ```
    pub fn neuron_layer(&self, linear: usize) -> usize {
        self.check_linear(linear);
        let mut layer = linear;
        while self
            .layers
            .get(layer + 1)
            .is_some_and(|next| next.is_elementwise())
        {
            layer += 1;
        }
        layer
    }

    /// Puts `mask` on the neurons of the `Linear` at layer `linear`, on the layer
    /// [`neuron_layer`](Self::neuron_layer) names, replacing any mask there.
    ///
    /// Masks have three routes, one story: by layer index with
    /// [`set_mask`](Self::set_mask); by `Linear` with this method and
    /// [`set_neuron`](Self::set_neuron), which find the layer for you; and on a compiled
    /// model, [`InferenceModel::set_neuron_mask`](crate::InferenceModel::set_neuron_mask)
    /// (and [`Evaluator::set_neuron_mask`](crate::Evaluator::set_neuron_mask)), which change
    /// a mask per instance without allocating.
    ///
    /// # Panics
    ///
    /// Panics if layer `linear` is not a `Linear` (the message names the index and the
    /// layer's kind); if `mask`'s width is not the `Linear`'s output width (the message
    /// names the `Linear`, the mask's width and the output width); or if the `Linear`
    /// itself, or an element-wise layer between it and the layer that holds its neurons,
    /// carries a mask set with [`set_mask`](Self::set_mask), so that one neuron set never
    /// carries two masks (clear that one with [`clear_mask`](Self::clear_mask) first).
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, NeuronMask, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    ///     .push(Relu);
    /// net.set_neuron_mask(0, NeuronMask::from_off(3, &[1]));
    /// assert!(net.mask(1).is_some() && net.mask(0).is_none());
    /// ```
    pub fn set_neuron_mask(&mut self, linear: usize, mask: NeuronMask) {
        let layer = self.neuron_layer(linear);
        self.check_neuron_width(linear, mask.width());
        self.check_no_linear_mask(linear, layer);
        self.masks[layer] = Some(LayerMask::new(mask));
    }

    /// Copies `mask` into the existing mask on the neurons of the `Linear` at layer
    /// `linear`, without allocating. Returns `false`, changing nothing, if there is no mask
    /// there or its width differs from `mask`'s.
    ///
    /// # Panics
    ///
    /// Panics if layer `linear` is not a `Linear`, as [`neuron_layer`](Self::neuron_layer)
    /// does.
    pub(crate) fn copy_neuron_mask(&mut self, linear: usize, mask: &NeuronMask) -> bool {
        let layer = self.neuron_layer(linear);
        match &mut self.masks[layer] {
            Some(existing) if existing.mask.width() == mask.width() => {
                for j in 0..mask.width() {
                    if existing.mask.is_on(j) != mask.is_on(j) {
                        existing.set(j, mask.is_on(j));
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Puts `mask` on the neurons of the `Linear` at layer `linear`, as
    /// [`set_neuron_mask`](Self::set_neuron_mask) does but without its checks: for a copy
    /// that carries masks over from another model.
    ///
    /// # Panics
    ///
    /// Panics if layer `linear` is not a `Linear`, as [`neuron_layer`](Self::neuron_layer)
    /// does.
    pub(crate) fn put_neuron_mask(&mut self, linear: usize, mask: NeuronMask) {
        let layer = self.neuron_layer(linear);
        self.masks[layer] = Some(LayerMask::new(mask));
    }

    /// The mask on the neurons of the `Linear` at layer `linear`, if it has one: the mask on
    /// [`neuron_layer`](Self::neuron_layer). A mask put on the `Linear` itself with
    /// [`set_mask`](Self::set_mask), when its neurons are read at a later layer, is not the
    /// neuron mask and is not returned here; read it with [`mask`](Self::mask).
    ///
    /// # Panics
    ///
    /// Panics if layer `linear` is not a `Linear`; the message names the index and the
    /// layer's kind.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, NeuronMask, Sequential};
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
    /// assert!(net.neuron_mask(0).is_none());
    /// net.set_neuron_mask(0, NeuronMask::all_on(3));
    /// assert_eq!(net.neuron_mask(0).unwrap().width(), 3);
    /// ```
    pub fn neuron_mask(&self, linear: usize) -> Option<&NeuronMask> {
        self.masks[self.neuron_layer(linear)]
            .as_ref()
            .map(|m| &m.mask)
    }

    /// Removes the mask from the neurons of the `Linear` at layer `linear`, if it has one:
    /// the mask on [`neuron_layer`](Self::neuron_layer). A mask put on the `Linear` itself
    /// with [`set_mask`](Self::set_mask), when its neurons are read at a later layer, is not
    /// the neuron mask and stays; remove it with [`clear_mask`](Self::clear_mask).
    ///
    /// # Panics
    ///
    /// Panics if layer `linear` is not a `Linear`; the message names the index and the
    /// layer's kind.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, NeuronMask, Sequential};
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
    /// net.set_neuron_mask(0, NeuronMask::all_on(3));
    /// net.clear_neuron_mask(0);
    /// assert!(net.neuron_mask(0).is_none());
    /// ```
    pub fn clear_neuron_mask(&mut self, linear: usize) {
        let layer = self.neuron_layer(linear);
        self.masks[layer] = None;
    }

    /// Switches neuron `neuron` of the `Linear` at layer `linear` on or off. With a mask
    /// already there it changes that one neuron in place, without allocating; without one it
    /// first creates a mask with every neuron on.
    ///
    /// # Panics
    ///
    /// Panics if layer `linear` is not a `Linear` (the message names the index and the
    /// layer's kind); if `neuron` is not below the `Linear`'s output width, or a mask
    /// already there has another width (each message names the `Linear` and the widths);
    /// or if the `Linear` itself, or an element-wise layer between it and the layer that
    /// holds its neurons, carries a mask set with [`set_mask`](Self::set_mask), so that one
    /// neuron set never carries two masks (clear that one with
    /// [`clear_mask`](Self::clear_mask) first).
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    ///     .push(Relu);
    /// net.set_neuron(0, 2, false);
    /// assert_eq!(net.neuron_mask(0).unwrap().off().collect::<Vec<_>>(), [2]);
    /// ```
    pub fn set_neuron(&mut self, linear: usize, neuron: usize, on: bool) {
        let layer = self.neuron_layer(linear);
        let width = self.linear_outputs(linear);
        assert!(
            neuron < width,
            "sequential: neuron {neuron} is out of range for the linear layer {linear}, which \
             outputs width {width}"
        );
        if let Some(mask) = &self.masks[layer] {
            self.check_neuron_width(linear, mask.mask.width());
        }
        self.check_no_linear_mask(linear, layer);
        let mask =
            self.masks[layer].get_or_insert_with(|| LayerMask::new(NeuronMask::all_on(width)));
        mask.set(neuron, on);
    }

    /// The output width of the `Linear` at layer `linear`, which must be one.
    fn linear_outputs(&self, linear: usize) -> usize {
        self.layers[linear]
            .as_linear()
            .expect("the caller checked that the layer is a Linear")
            .outputs()
    }

    /// Panics, naming the `Linear`, `width` and the output width, if a neuron mask of
    /// `width` does not fit the `Linear` at layer `linear`.
    fn check_neuron_width(&self, linear: usize, width: usize) {
        let outputs = self.linear_outputs(linear);
        assert!(
            width == outputs,
            "sequential: a neuron mask of width {width} for the linear layer {linear}, which \
             outputs width {outputs}"
        );
    }

    /// Panics if a layer from the `Linear` at layer `linear` up to, but not including, the
    /// layer `layer` that holds its neurons (the `Linear` itself or an element-wise layer of
    /// the run after it) carries a mask, naming the `Linear`, that layer and `layer`.
    fn check_no_linear_mask(&self, linear: usize, layer: usize) {
        if let Some(other) = (linear..layer).find(|&l| self.masks[l].is_some()) {
            panic!(
                "sequential: the neurons of the linear layer {linear} are masked at layer \
                 {layer}, but layer {other} has a mask set with set_mask; clear it with \
                 clear_mask({other}) first, so that one neuron set carries one mask"
            );
        }
    }

    /// Panics, naming the index and the layer's kind, if layer `linear` is not a `Linear`.
    fn check_linear(&self, linear: usize) {
        self.check_layer(linear, "address neurons of");
        let layer = &self.layers[linear];
        assert!(
            layer.as_linear().is_some(),
            "sequential: layer {linear} is a {} layer, not a linear layer",
            layer.name()
        );
    }

    /// Panics, naming `action`, the index and the valid range, if `layer` is not a layer.
    fn check_layer(&self, layer: usize, action: &str) {
        let layers = self.layers.len();
        assert!(
            layer < layers,
            "sequential: cannot {action} layer {layer}, the model has {layers} layers \
             (valid indices 0..{layers})"
        );
    }

    /// Records every layer's forward pass on `tape`, each taking the previous output, and
    /// returns the last output (`x` itself for an empty network). A masked layer's output is
    /// recorded through [`Tape::keep_where`] with the mask as the condition; an unmasked
    /// layer records nothing extra.
    ///
    /// # Panics
    ///
    /// Panics if a layer's required input width differs from the width of the previous
    /// output, or a layer produces a width other than the one it reports; the message names
    /// the layer's index and name and both widths. Panics, after that layer's forward pass,
    /// if a layer's mask is not as wide as its output; the message names the layer and both
    /// widths. A layer may also panic on a shape it rejects.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Rng, Tensor};
    /// use cavitation_nn::{Linear, Sequential, Tanh};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    ///     .push(Tanh);
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&params.values);
    /// let x = tape.input(Tensor::full(&[1, 2], 1.0f32).view());
    /// let y = net.forward(&mut tape, x);
    /// assert_eq!(tape.shape(y).dims(), &[1, 3]);
    /// ```
    pub fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        self.run(tape, x, None)
    }

    /// Like [`forward`](Self::forward), and records each layer's input and output in `trace`
    /// (cleared first; its capacity is kept).
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as `forward`.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{Relu, Sequential, Trace};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::full(&[1, 2], 1.0f32).view());
    /// let mut trace = Trace::new();
    /// Sequential::new().push(Relu).forward_traced(&mut tape, x, &mut trace);
    /// assert_eq!(trace.len(), 1);
    /// ```
    pub fn forward_traced(&self, tape: &mut Tape<'_, T>, x: Var, trace: &mut Trace) -> Var {
        self.run(tape, x, Some(trace))
    }

    /// The forward pass, with the width checks and the optional trace.
    fn run(&self, tape: &mut Tape<'_, T>, x: Var, mut trace: Option<&mut Trace>) -> Var {
        if let Some(trace) = trace.as_deref_mut() {
            trace.inputs.clear();
            trace.outputs.clear();
            trace.inputs.reserve(self.layers.len());
            trace.outputs.reserve(self.layers.len());
        }
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            let width = last_dim(tape.shape(x));
            if let Some(expected) = layer.input_width()
                && expected != width
            {
                if i == 0 {
                    panic!(
                        "sequential: layer 0 ({}) expects input width {expected}, but the input has width {width}",
                        layer.name()
                    );
                }
                panic!(
                    "sequential: layer {i} ({}) expects input width {expected}, but layer {} ({}) outputs width {width}",
                    layer.name(),
                    i - 1,
                    self.layers[i - 1].name()
                );
            }
            let mut y = layer.forward(tape, x);
            let produced = last_dim(tape.shape(y));
            let reported = layer.output_width(width);
            assert!(
                produced == reported,
                "sequential: layer {i} ({}) produced output width {produced}, but reports {reported} for input width {width}",
                layer.name()
            );
            if let Some(mask) = &self.masks[i] {
                let masked = mask.mask.width();
                assert!(
                    masked == produced,
                    "sequential: the mask on layer {i} ({}) has width {masked}, but the layer outputs width {produced}",
                    layer.name()
                );
                let keep = tape.input(mask.keep.view());
                y = tape.keep_where(y, keep);
            }
            if let Some(trace) = trace.as_deref_mut() {
                trace.inputs.push(x);
                trace.outputs.push(y);
            }
            x = y;
        }
        x
    }

    /// Every parameter with its PyTorch-style name `"{layer index}.{local name}"` (for
    /// example `0.weight`, `2.bias`), layer by layer.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    ///     .push(Relu);
    /// let names: Vec<String> = net.parameter_names().into_iter().map(|(n, _)| n).collect();
    /// assert_eq!(names, ["0.weight", "0.bias"]);
    /// ```
    pub fn parameter_names(&self) -> Vec<(String, ParamId)> {
        let mut names = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (local, id) in layer.parameters() {
                names.push((format!("{i}.{local}"), id));
            }
        }
        names
    }
}

#[cfg(test)]
mod tests {
    //! Tests that a cloned `Sequential` shares its layers, copies its masks and trains
    //! independently of the original once the parameter store is cloned too.

    use cavitation_core::{Graph, ParamId, ParamStore, Real, Reduction, Rng, Tape, Tensor, Var};

    use crate::{
        Init, Layer, Linear, NeuronMask, Optimizer, Prelu, Relu, Sequential, Sgd, Sigmoid, Tanh,
        Trace,
    };

    /// The digits-shaped network, 64 → 32 → 10, with its parameters in `params`.
    fn digits_net(params: &mut ParamStore) -> Sequential {
        let mut rng = Rng::seed(1);
        Sequential::new()
            .push(Linear::with_init(
                params,
                64,
                32,
                Init::HeUniform,
                Init::Zeros,
                &mut rng,
            ))
            .push(Relu)
            .push(Linear::with_init(
                params,
                32,
                10,
                Init::HeUniform,
                Init::Zeros,
                &mut rng,
            ))
    }

    /// The network's output on `x`, as a vector.
    fn output(net: &Sequential, params: &ParamStore, x: &Tensor<f32>) -> Vec<f32> {
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let input = tape.input(x.view());
        let y = net.forward(&mut tape, input);
        tape.value(y).data().to_vec()
    }

    /// A cloned network and a cloned store give bit-identical outputs, and ten training steps
    /// on the clone leave the original store's values unchanged.
    #[test]
    fn cloned_network_is_independent() {
        let mut params = ParamStore::new();
        let net = digits_net(&mut params);
        let mut rng = Rng::seed(7);
        let data: Vec<f32> = (0..8 * 64).map(|_| rng.range(0.0, 1.0) as f32).collect();
        let x = Tensor::from_vec(data, &[8, 64]).unwrap();
        let labels: Vec<usize> = (0..8).collect();

        let copy = net.clone();
        let mut copy_params = params.clone();
        let before = output(&net, &params, &x);
        assert_eq!(output(&copy, &copy_params, &x), before);

        let mut optimizer = Sgd::new(0.1);
        for _ in 0..10 {
            let mut graph = Graph::new();
            let mut tape = graph.tape(&copy_params.values);
            let input = tape.input(x.view());
            let logits = copy.forward(&mut tape, input);
            let loss = tape.softmax_cross_entropy(logits, &labels, Reduction::Mean);
            tape.backward(loss, &mut copy_params.grads);
            drop(tape);
            optimizer.step(&mut copy_params);
            copy_params.zero_grad();
        }
        assert_eq!(output(&net, &params, &x), before);
        assert_ne!(output(&copy, &copy_params, &x), before);
    }

    /// Layers `Linear, Relu, Linear` as a network.
    fn lrl(params: &mut ParamStore) -> Sequential {
        Sequential::new()
            .push(Linear::new(params, 3, 4, &mut Rng::seed(1)))
            .push(Relu)
            .push(Linear::new(params, 4, 2, &mut Rng::seed(2)))
    }

    /// `linear` and `linears` find the `Linear` layers only.
    #[test]
    fn linear_by_index() {
        let net = lrl(&mut ParamStore::new());
        assert!(net.linear(0).is_some() && net.linear(2).is_some());
        assert!(net.linear(1).is_none() && net.linear(3).is_none());
        assert_eq!(net.linears().map(|(i, _)| i).collect::<Vec<_>>(), [0, 2]);
    }

    /// `"2.bias"` is layer 2's bias; unknown layers, locals and malformed names are `None`.
    #[test]
    fn param_by_name() {
        let mut params = ParamStore::new();
        let net = lrl(&mut params);
        assert_eq!(net.param("2.bias"), Some(net.linear(2).unwrap().bias()));
        assert_eq!(net.param("0.weight"), Some(net.linear(0).unwrap().weight()));
        for bad in [
            "7.weight", "x", "", ".", "1.weight", "2.scale", "02.bias", "+2.bias", "2",
        ] {
            assert_eq!(net.param(bad), None, "{bad:?}");
        }
        for (name, id) in net.parameter_names() {
            assert_eq!(net.param(&name), Some(id));
        }
    }

    /// The neuron mask of a hidden `Linear` goes on the activation after it; the output
    /// `Linear` has none after it, so it takes the mask itself.
    #[test]
    fn neuron_mask_goes_after_activation() {
        let mut net = lrl(&mut ParamStore::new());
        net.set_neuron_mask(0, NeuronMask::from_off(4, &[1]));
        assert!(net.mask(1).is_some() && net.mask(0).is_none());
        net.set_neuron_mask(2, NeuronMask::from_off(2, &[0]));
        assert!(net.mask(2).is_some());
        assert_eq!((net.neuron_layer(0), net.neuron_layer(2)), (1, 2));
        assert_eq!(net.neuron_mask(0).unwrap().width(), 4);
        net.clear_neuron_mask(0);
        assert!(net.mask(1).is_none() && net.neuron_mask(0).is_none());
    }

    /// A `Prelu` after a `Linear` carries its mask too.
    #[test]
    fn neuron_mask_goes_after_prelu() {
        let mut params = ParamStore::new();
        let net = Sequential::new()
            .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
            .push(Prelu::new(&mut params, 1));
        assert_eq!(net.neuron_layer(0), 1);
        let mut net = net;
        net.set_neuron_mask(0, NeuronMask::all_on(3));
        assert!(net.mask(1).is_some() && net.mask(0).is_none());
    }

    /// A user layer with the given name, element-wise or not as `elementwise` says.
    struct Custom {
        /// The layer's name.
        name: &'static str,
        /// What `is_elementwise` answers; `None` keeps the default.
        elementwise: Option<bool>,
    }

    impl<T: Real> Layer<T> for Custom {
        /// The configured name.
        fn name(&self) -> &str {
            self.name
        }
        /// Unchanged.
        fn output_width(&self, input_width: usize) -> usize {
            input_width
        }
        /// Passes the input through.
        fn forward(&self, _tape: &mut Tape<'_, T>, x: Var) -> Var {
            x
        }
        /// None.
        fn parameters(&self) -> Vec<(&'static str, ParamId)> {
            Vec::new()
        }
        /// The configured answer, or the default.
        fn is_elementwise(&self) -> bool {
            self.elementwise.unwrap_or(false)
        }
    }

    /// A user layer named "prelu" that is not element-wise does not take the mask.
    #[test]
    fn layer_named_prelu_is_not_assumed_elementwise() {
        let mut params = ParamStore::new();
        let net = Sequential::new()
            .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
            .push(Custom {
                name: "prelu",
                elementwise: None,
            });
        assert_eq!(net.neuron_layer(0), 0);
    }

    /// A user element-wise layer that overrides `is_elementwise` takes the mask.
    #[test]
    fn user_elementwise_layer_takes_the_mask() {
        let mut params = ParamStore::new();
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
            .push(Custom {
                name: "scale",
                elementwise: Some(true),
            });
        net.set_neuron(0, 1, false);
        assert!(net.mask(1).is_some() && net.mask(0).is_none());
    }

    /// A panic in `set_neuron` leaves the network without a mask.
    #[test]
    fn set_neuron_out_of_range_leaves_no_mask() {
        let mut net = lrl(&mut ParamStore::new());
        let hit =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| net.set_neuron(0, 4, false)));
        assert!(hit.is_err());
        assert!(net.mask(1).is_none());
    }

    /// A masked sigmoid neuron outputs exactly 0, not 0.5.
    #[test]
    fn neuron_mask_on_sigmoid_outputs_zero() {
        let mut params = ParamStore::new();
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 3, 4, &mut Rng::seed(5)))
            .push(Sigmoid)
            .push(Linear::new(&mut params, 4, 2, &mut Rng::seed(6)));
        net.set_neuron(0, 1, false);
        let x = Tensor::from_vec(vec![0.3f32, -0.8, 1.2, -0.5, 0.9, 0.1], &[2, 3]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xi = tape.input(x.view());
        let mut trace = Trace::new();
        net.forward_traced(&mut tape, xi, &mut trace);
        let out = tape.value(trace.output(1)).data();
        for row in 0..2 {
            assert_eq!(out[row * 4 + 1].to_bits(), 0.0f32.to_bits());
            assert!(out[row * 4] > 0.0 && out[row * 4 + 2] > 0.0);
        }
    }

    /// Addressing neurons of a layer that is not a `Linear` panics, naming the index and kind.
    #[test]
    #[should_panic(expected = "sequential: layer 1 is a relu layer, not a linear layer")]
    fn neuron_mask_on_non_linear_panics() {
        let mut net = lrl(&mut ParamStore::new());
        net.set_neuron_mask(1, NeuronMask::all_on(4));
    }

    /// A neuron past the width panics, naming the neuron, the `Linear` and its width.
    #[test]
    #[should_panic(
        expected = "neuron 4 is out of range for the linear layer 0, which outputs \
                               width 4"
    )]
    fn set_neuron_out_of_range_panics() {
        let mut net = lrl(&mut ParamStore::new());
        net.set_neuron(0, 4, false);
    }

    /// `set_neuron_mask` with a mask wider than the `Linear`'s output panics at once,
    /// naming the `Linear`, the mask's width and the output width, and sets nothing.
    #[test]
    fn set_neuron_mask_checks_the_width() {
        let mut net = lrl(&mut ParamStore::new());
        let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            net.set_neuron_mask(0, NeuronMask::all_on(5));
        }));
        let msg = *hit.unwrap_err().downcast::<String>().unwrap();
        assert!(
            msg.contains("width 5 for the linear layer 0, which outputs width 4"),
            "{msg}"
        );
        assert!(net.mask(1).is_none());
    }

    /// `set_neuron` on a neuron mask of the wrong width (put there by layer with `set_mask`)
    /// panics, naming the `Linear` and both widths, even for a neuron below both widths.
    #[test]
    #[should_panic(expected = "width 5 for the linear layer 0, which outputs width 4")]
    fn set_neuron_checks_an_existing_mask() {
        let mut net = lrl(&mut ParamStore::new());
        net.set_mask(1, NeuronMask::all_on(5));
        net.set_neuron(0, 1, false);
    }

    /// A mask set on the `Linear` itself, whose neurons are read at the ReLU after it, is
    /// not the neuron mask: `neuron_mask` does not return it, `clear_neuron_mask` leaves
    /// it, and `set_neuron` and `set_neuron_mask` refuse to add a second mask, naming the
    /// `Linear`.
    #[test]
    fn mask_on_the_linear_is_not_the_neuron_mask() {
        let mut net = lrl(&mut ParamStore::new());
        net.set_mask(0, NeuronMask::from_off(4, &[1]));
        assert!(net.neuron_mask(0).is_none());
        net.clear_neuron_mask(0);
        assert!(net.mask(0).is_some());
        let calls: [fn(&mut Sequential); 2] = [
            |n| n.set_neuron(0, 2, false),
            |n| n.set_neuron_mask(0, NeuronMask::all_on(4)),
        ];
        for call in calls {
            let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| call(&mut net)));
            let msg = *hit.unwrap_err().downcast::<String>().unwrap();
            assert!(msg.contains("clear_mask(0)"), "{msg}");
        }
        assert!(net.mask(1).is_none());
        net.clear_mask(0);
        net.set_neuron(0, 2, false);
        assert_eq!(net.neuron_mask(0).unwrap().off().collect::<Vec<_>>(), [2]);
    }

    /// On `Linear → Tanh → Sigmoid → Linear`, a mask set by layer on the `Tanh`, inside the
    /// run before the `Sigmoid` that holds the neurons, is refused the same way: neither
    /// `set_neuron` nor `set_neuron_mask` adds a second mask, and each message names the
    /// `Tanh`'s layer. Once it is cleared, the neuron mask goes on the `Sigmoid`.
    #[test]
    fn mask_inside_the_run_is_refused() {
        let mut params = ParamStore::new();
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 3, 4, &mut Rng::seed(5)))
            .push(Tanh)
            .push(Sigmoid)
            .push(Linear::new(&mut params, 4, 2, &mut Rng::seed(6)));
        net.set_mask(1, NeuronMask::from_off(4, &[0]));
        assert!(net.neuron_mask(0).is_none());
        let calls: [fn(&mut Sequential); 2] = [
            |n| n.set_neuron(0, 2, false),
            |n| n.set_neuron_mask(0, NeuronMask::all_on(4)),
        ];
        for call in calls {
            let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| call(&mut net)));
            let msg = *hit.unwrap_err().downcast::<String>().unwrap();
            assert!(
                msg.contains("layer 1 has a mask") && msg.contains("clear_mask(1)"),
                "{msg}"
            );
        }
        assert!(net.mask(2).is_none());
        net.clear_mask(1);
        net.set_neuron(0, 2, false);
        assert!(net.mask(2).is_some() && net.mask(1).is_none());
    }

    /// On `Linear → Tanh → Sigmoid → Linear` the neurons of the first `Linear` are read at
    /// the `Sigmoid`, the last of the element-wise run, so an off neuron sends exactly 0 to
    /// the next `Linear`, not `sigmoid(0) = 0.5`.
    #[test]
    fn neuron_mask_goes_after_chained_activations() {
        let mut params = ParamStore::new();
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 3, 4, &mut Rng::seed(5)))
            .push(Tanh)
            .push(Sigmoid)
            .push(Linear::new(&mut params, 4, 2, &mut Rng::seed(6)));
        assert_eq!((net.neuron_layer(0), net.neuron_layer(3)), (2, 3));
        net.set_neuron(0, 1, false);
        assert!(net.mask(2).is_some() && net.mask(1).is_none());
        let x = Tensor::from_vec(vec![0.3f32, -0.8, 1.2, -0.5, 0.9, 0.1], &[2, 3]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xi = tape.input(x.view());
        let mut trace = Trace::new();
        net.forward_traced(&mut tape, xi, &mut trace);
        let out = tape.value(trace.output(2)).data();
        for row in 0..2 {
            assert_eq!(out[row * 4 + 1].to_bits(), 0.0f32.to_bits());
            assert!(out[row * 4] > 0.0 && out[row * 4 + 2] > 0.0);
        }
    }

    /// Switching neurons one at a time gives the same network as setting the whole mask,
    /// including the stored 1/0 tensor (seen through the outputs).
    #[test]
    fn set_neuron_matches_set_neuron_mask() {
        let mut params = ParamStore::new();
        let mut a = lrl(&mut params);
        let mut b = a.clone();
        a.set_neuron(0, 1, false);
        a.set_neuron(0, 3, false);
        a.set_neuron(0, 3, true);
        a.set_neuron(0, 0, false);
        b.set_neuron_mask(0, NeuronMask::from_off(4, &[0, 1]));
        assert_eq!(a.neuron_mask(0), b.neuron_mask(0));
        let x = Tensor::from_vec(vec![0.3f32, -0.8, 1.2], &[1, 3]).unwrap();
        let run = |net: &Sequential| {
            let mut graph = Graph::new();
            let mut tape = graph.tape(&params.values);
            let xi = tape.input(x.view());
            let y = net.forward(&mut tape, xi);
            tape.value(y).data().to_vec()
        };
        assert_eq!(run(&a), run(&b));
    }

    /// A mask set before cloning is on both networks with the same contents; a mask set on
    /// either network afterwards is on that network only.
    #[test]
    fn clone_keeps_masks() {
        let mut net = Sequential::new().push(Relu).push(Relu);
        let mask = NeuronMask::all_on(4);
        net.set_mask(0, mask.clone());
        let mut copy = net.clone();
        assert_eq!(net.mask(0), Some(&mask));
        assert_eq!(copy.mask(0), Some(&mask));
        copy.set_mask(1, NeuronMask::all_on(2));
        assert!(copy.mask(1).is_some());
        assert!(net.mask(1).is_none());
        net.set_mask(1, NeuronMask::all_on(3));
        assert_eq!(copy.mask(1), Some(&NeuronMask::all_on(2)));
    }

    /// A layer that is not `Clone` works inside a cloned network, and both copies share it.
    #[test]
    fn user_layer_clones_without_clone() {
        /// The `ScaledTanh` layer of the `Layer` docs: `tanh(x)` times a learnable `alpha`.
        /// It is not `Clone`.
        struct ScaledTanh {
            /// The scale.
            alpha: ParamId,
        }

        impl<T: Real> Layer<T> for ScaledTanh {
            /// The layer's name.
            fn name(&self) -> &str {
                "scaled_tanh"
            }
            /// The output is as wide as the input.
            fn output_width(&self, input_width: usize) -> usize {
                input_width
            }
            /// `tanh(x)` times `alpha`.
            fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
                let t = tape.tanh(x);
                let alpha = tape.param(self.alpha);
                tape.mul(t, alpha)
            }
            /// The one parameter, `alpha`.
            fn parameters(&self) -> Vec<(&'static str, ParamId)> {
                vec![("alpha", self.alpha)]
            }
        }

        let mut params = ParamStore::new();
        let alpha = params.add(Tensor::from_vec(vec![1.5f32], &[1]).unwrap());
        let net = Sequential::new().push(ScaledTanh { alpha });
        let copy = net.clone();
        assert_eq!(copy.len(), 1);
        assert_eq!(copy.layer(0).name(), "scaled_tanh");
        assert_eq!(
            output(&copy, &params, &Tensor::full(&[1, 2], 0.5f32)),
            output(&net, &params, &Tensor::full(&[1, 2], 0.5f32))
        );
        assert!(std::ptr::eq(
            net.layer(0) as *const dyn Layer<f32> as *const u8,
            copy.layer(0) as *const dyn Layer<f32> as *const u8,
        ));
    }
}
