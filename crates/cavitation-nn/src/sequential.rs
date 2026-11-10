//! `Sequential` and `Trace`: layers run one after another, and a record of each layer's input
//! and output.

use cavitation_core::{ParamId, Real, Tape, Var};

use crate::Layer;
use crate::activation::last_dim;

/// Each layer's input and output [`Var`] from one forward pass.
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
pub struct Sequential<T: Real = f32> {
    /// The layers, in order.
    layers: Vec<Box<dyn Layer<T>>>,
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
        Sequential { layers: Vec::new() }
    }

    /// Adds `layer` at the end and returns the network.
    ///
    /// ```
    /// use cavitation_nn::{Relu, Sequential, Tanh};
    /// let net = Sequential::new().push(Relu).push(Tanh);
    /// assert_eq!(net.len(), 2);
    /// ```
    pub fn push(mut self, layer: impl Layer<T> + 'static) -> Sequential<T> {
        self.layers.push(Box::new(layer));
        self
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

    /// Records every layer's forward pass on `tape`, each taking the previous output, and
    /// returns the last output (`x` itself for an empty network).
    ///
    /// # Panics
    ///
    /// Panics if a layer's required input width differs from the width of the previous
    /// output, or a layer produces a width other than the one it reports; the message names
    /// the layer's index and name and both widths. A layer may also panic on a shape it
    /// rejects.
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
            let y = layer.forward(tape, x);
            let produced = last_dim(tape.shape(y));
            let reported = layer.output_width(width);
            assert!(
                produced == reported,
                "sequential: layer {i} ({}) produced output width {produced}, but reports {reported} for input width {width}",
                layer.name()
            );
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
