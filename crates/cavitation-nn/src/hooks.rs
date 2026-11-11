//! Forward and backward hooks: closures that see each layer's input, output, local
//! derivative and gradients during training.

use std::fmt;

use cavitation_core::{Real, Tape, TensorView};

use crate::{NeuronMask, Sequential, Trace};

/// What a hook sees of one layer in one step.
///
/// For an activation layer `a = f(z)` the five values are `z` (`input`), `a` (`output`),
/// `f′(z)` (`local_derivative`), `∂L/∂a` (`grad_output`) and `∂L/∂z` (`grad_input`). The
/// views borrow the tape and the hooks; copy what you want to keep.
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// use cavitation_nn::{Hooks, Sequential, Tanh, Trace};
/// let store = ParamStore::new();
/// let net = Sequential::new().push(Tanh);
/// let mut seen = Vec::new();
/// let mut hooks = Hooks::new();
/// hooks.on_forward(0, |e| seen.push((e.name.to_string(), e.local_derivative.unwrap().data()[0])));
/// let mut graph = Graph::new();
/// let mut trace = Trace::new();
/// let mut tape = graph.tape(&store.values);
/// let x = tape.input(Tensor::full(&[1, 1], 0.0f32).view());
/// net.forward_traced(&mut tape, x, &mut trace);
/// hooks.run_forward(&tape, &trace, &net);
/// drop(hooks);
/// assert_eq!(seen, [("tanh".to_string(), 1.0)]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct LayerEvent<'a, T: Real = f32> {
    /// The layer's index in the [`Sequential`].
    pub layer: usize,
    /// The layer's name, for example `"relu"`.
    pub name: &'a str,
    /// The layer's input.
    pub input: TensorView<'a, T>,
    /// The layer's output, as the next layer receives it: with a
    /// [`NeuronMask`](crate::NeuronMask) on the layer, the masked output (0 for each off
    /// neuron).
    pub output: TensorView<'a, T>,
    /// The derivative of each output value with respect to the input value at the same
    /// position, from [`Layer::local_derivative`](crate::Layer::local_derivative): `f′(z)`
    /// for an activation, 1 or the slope for `Prelu`, and 0 for each neuron a mask turns off
    /// (its output no longer depends on its input). `None` for layers whose derivative is
    /// not element-wise (`Linear`, `Softmax`, `LogSoftmax`, `Glu`).
    pub local_derivative: Option<TensorView<'a, T>>,
    /// The gradient of the loss with respect to the output. `None` in a forward event, and
    /// when the loss does not depend on the output.
    pub grad_output: Option<TensorView<'a, T>>,
    /// The gradient of the loss with respect to the input. `None` in a forward event, for
    /// the first layer (its input is the batch, which takes no gradient), and when the loss
    /// does not depend on the input.
    pub grad_input: Option<TensorView<'a, T>>,
}

/// Names one registered hook, so it can be removed with [`Hooks::remove`].
///
/// ```
/// use cavitation_nn::Hooks;
/// let mut hooks = Hooks::new();
/// let a = hooks.on_forward(0, |_| {});
/// let b = hooks.on_forward(0, |_| {});
/// assert_ne!(a, b);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HookId(u64);

/// A registered hook's callback.
type Callback<'h, T> = Box<dyn FnMut(&LayerEvent<'_, T>) + 'h>;

/// One registered hook.
struct Entry<'h, T: Real> {
    /// The id [`Hooks::on_forward`] or [`Hooks::on_backward`] returned.
    id: HookId,
    /// The layer it watches.
    layer: usize,
    /// The callback.
    f: Callback<'h, T>,
}

/// Closures run on chosen layers after the forward pass (forward hooks) or after the backward
/// pass (backward hooks) of each step, each seeing that layer's [`LayerEvent`].
///
/// Hooks read values; they cannot change the forward or the backward pass. Forward hooks run
/// in layer order and backward hooks in reverse layer order, the order of the passes; hooks on
/// the same layer run in the order they were registered. The lifetime `'h` lets a hook borrow
/// local variables, for example to push into a `Vec` that outlives the hooks.
///
/// The local derivative is computed into a buffer the hooks own, and only for layers that
/// have a hook. With no hooks registered, running them costs nothing; with hooks, nothing is
/// allocated once the buffer has grown to the largest layer output.
///
/// Pass them to [`Trainer::step_with_hooks`](crate::Trainer::step_with_hooks) or
/// [`fit_with_hooks`](crate::fit_with_hooks), or call [`run_forward`](Self::run_forward)
/// and [`run_backward`](Self::run_backward) in a loop of your own.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{
///     Dataset, FitConfig, Hooks, Linear, Loss, Relu, Sequential, Targets, fit_with_hooks,
/// };
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 4, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 4, 2, &mut rng));
/// let data = Dataset {
///     x: Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 0.0], &[2, 2]).unwrap(),
///     targets: Targets::Classes(vec![0, 1]),
/// };
/// // The share of ReLU outputs that pass a gradient, each step.
/// let mut live = Vec::new();
/// let mut hooks = Hooks::new();
/// hooks.on_backward(1, |e| {
///     let d = e.local_derivative.unwrap().data();
///     live.push(d.iter().filter(|&&v| v > 0.0).count() as f32 / d.len() as f32);
/// });
/// let config = FitConfig { epochs: 3, batch: 2, loss: Loss::CrossEntropy, ..FitConfig::default() };
/// fit_with_hooks(&net, &mut params, &data, config, &mut hooks).unwrap();
/// drop(hooks);
/// assert_eq!(live.len(), 3);
/// ```
pub struct Hooks<'h, T: Real = f32> {
    /// The forward hooks, in the order they were registered.
    forward: Vec<Entry<'h, T>>,
    /// The backward hooks, in the order they were registered.
    backward: Vec<Entry<'h, T>>,
    /// The id the next registered hook gets.
    next_id: u64,
    /// The buffer the local derivative of the layer being reported is computed into.
    derivative: Vec<T>,
}

impl<'h> Hooks<'h, f32> {
    /// No hooks, for an `f32` network.
    ///
    /// ```
    /// use cavitation_nn::Hooks;
    /// assert!(Hooks::new().is_empty());
    /// ```
    pub fn new() -> Hooks<'h, f32> {
        Hooks::new_generic()
    }
}

impl Default for Hooks<'_, f32> {
    /// No hooks, for an `f32` network.
    fn default() -> Self {
        Hooks::new()
    }
}

impl<T: Real> fmt::Debug for Hooks<'_, T> {
    /// Writes the layers each hook watches; the callbacks themselves are not shown.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layers = |entries: &[Entry<'_, T>]| entries.iter().map(|e| e.layer).collect::<Vec<_>>();
        f.debug_struct("Hooks")
            .field("forward", &layers(&self.forward))
            .field("backward", &layers(&self.backward))
            .finish()
    }
}

impl<'h, T: Real> Hooks<'h, T> {
    /// No hooks, for a network of any [`Real`] type (for example `f64`).
    ///
    /// ```
    /// use cavitation_nn::Hooks;
    /// assert!(Hooks::<f64>::new_generic().is_empty());
    /// ```
    pub fn new_generic() -> Hooks<'h, T> {
        Hooks {
            forward: Vec::new(),
            backward: Vec::new(),
            next_id: 0,
            derivative: Vec::new(),
        }
    }

    /// Whether no hook is registered.
    ///
    /// ```
    /// use cavitation_nn::Hooks;
    /// let mut hooks = Hooks::new();
    /// assert!(hooks.is_empty());
    /// hooks.on_backward(0, |_| {});
    /// assert!(!hooks.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty() && self.backward.is_empty()
    }

    /// Registers `f` to run on layer `layer` after each forward pass. The event carries the
    /// input, the output and the local derivative; its gradients are `None`. The index is
    /// checked against the model when the hooks run.
    ///
    /// ```
    /// use cavitation_nn::Hooks;
    /// let mut calls = 0;
    /// let mut hooks = Hooks::new();
    /// hooks.on_forward(2, |_| calls += 1);
    /// assert!(!hooks.is_empty());
    /// ```
    pub fn on_forward(&mut self, layer: usize, f: impl FnMut(&LayerEvent<'_, T>) + 'h) -> HookId {
        let id = self.next_id();
        self.forward.push(Entry {
            id,
            layer,
            f: Box::new(f),
        });
        id
    }

    /// Registers `f` to run on layer `layer` after each backward pass. The event carries
    /// everything a forward event does and the gradients of the loss with respect to the
    /// layer's output and input. The index is checked against the model when the hooks run.
    ///
    /// ```
    /// use cavitation_nn::Hooks;
    /// let mut norms = Vec::new();
    /// let mut hooks = Hooks::new();
    /// hooks.on_backward(0, |e| {
    ///     let g = e.grad_output.unwrap().data();
    ///     norms.push(g.iter().map(|v| v * v).sum::<f32>().sqrt());
    /// });
    /// ```
    pub fn on_backward(&mut self, layer: usize, f: impl FnMut(&LayerEvent<'_, T>) + 'h) -> HookId {
        let id = self.next_id();
        self.backward.push(Entry {
            id,
            layer,
            f: Box::new(f),
        });
        id
    }

    /// Removes the hook `id`; it does not run again. Removing a hook that is already gone
    /// does nothing.
    ///
    /// ```
    /// use cavitation_nn::Hooks;
    /// let mut hooks = Hooks::new();
    /// let id = hooks.on_forward(0, |_| {});
    /// hooks.remove(id);
    /// assert!(hooks.is_empty());
    /// ```
    pub fn remove(&mut self, id: HookId) {
        self.forward.retain(|e| e.id != id);
        self.backward.retain(|e| e.id != id);
    }

    /// Runs the forward hooks, in layer order, on the forward pass `trace` recorded on
    /// `tape` with `model`. Call it after [`Sequential::forward_traced`].
    ///
    /// # Panics
    ///
    /// Panics, before any hook runs, if a hook (forward or backward) is registered on a layer
    /// index the model does not have; the message names the index and the valid range. Also
    /// panics if `trace` does not hold one entry per layer of `model` or was not recorded on
    /// `tape`.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{Hooks, Relu, Sequential, Trace};
    /// let store = ParamStore::new();
    /// let net = Sequential::new().push(Relu);
    /// let mut outputs = Vec::new();
    /// let mut hooks = Hooks::new();
    /// hooks.on_forward(0, |e| outputs.extend_from_slice(e.output.data()));
    /// let mut graph = Graph::new();
    /// let mut trace = Trace::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::from_vec(vec![-1.0f32, 2.0], &[1, 2]).unwrap().view());
    /// net.forward_traced(&mut tape, x, &mut trace);
    /// hooks.run_forward(&tape, &trace, &net);
    /// drop(hooks);
    /// assert_eq!(outputs, [0.0, 2.0]);
    /// ```
    pub fn run_forward(&mut self, tape: &Tape<'_, T>, trace: &Trace, model: &Sequential<T>) {
        if self.is_empty() {
            return;
        }
        self.check(trace, model);
        let Hooks {
            forward,
            derivative,
            ..
        } = self;
        for layer in 0..model.len() {
            run_layer(forward, derivative, layer, tape, trace, model, false);
        }
    }

    /// Runs the backward hooks, in reverse layer order, after [`Tape::backward`] on a tape
    /// whose forward pass `trace` recorded with `model`.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`run_forward`](Self::run_forward).
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{Hooks, Sequential, Tanh, Trace};
    /// let mut store = ParamStore::new();
    /// let net = Sequential::new().push(Tanh);
    /// let mut grad = Vec::new();
    /// let mut hooks = Hooks::new();
    /// hooks.on_backward(0, |e| grad.extend_from_slice(e.grad_output.unwrap().data()));
    /// let mut graph = Graph::new();
    /// let mut trace = Trace::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::full(&[1, 2], 0.5f32).view());
    /// let y = net.forward_traced(&mut tape, x, &mut trace);
    /// let loss = tape.sum(y);
    /// tape.backward(loss, &mut store.grads);
    /// hooks.run_backward(&tape, &trace, &net);
    /// drop(hooks);
    /// assert_eq!(grad, [1.0, 1.0]);
    /// ```
    pub fn run_backward(&mut self, tape: &Tape<'_, T>, trace: &Trace, model: &Sequential<T>) {
        if self.is_empty() {
            return;
        }
        self.check(trace, model);
        let Hooks {
            backward,
            derivative,
            ..
        } = self;
        for layer in (0..model.len()).rev() {
            run_layer(backward, derivative, layer, tape, trace, model, true);
        }
    }

    /// Panics, naming the index and the valid range, if a hook (forward or backward) watches
    /// a layer `model` does not have. Needs no tape, so a training step can call it before
    /// any kernel runs.
    pub(crate) fn check_layers(&self, model: &Sequential<T>) {
        let layers = model.len();
        for entry in self.forward.iter().chain(&self.backward) {
            assert!(
                entry.layer < layers,
                "hooks: a hook is registered on layer {}, but the model has {layers} layers \
                 (valid indices 0..{layers})",
                entry.layer
            );
        }
    }

    /// A fresh id.
    fn next_id(&mut self) -> HookId {
        let id = HookId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Panics if a hook watches a layer `model` does not have, or `trace` does not match
    /// `model`.
    fn check(&self, trace: &Trace, model: &Sequential<T>) {
        self.check_layers(model);
        let layers = model.len();
        assert!(
            trace.len() == layers,
            "hooks: the trace records {} layers, but the model has {layers}",
            trace.len()
        );
    }
}

/// Runs every hook in `entries` that watches `layer`, with that layer's event; the local
/// derivative is computed into `derivative` only if there is such a hook, and set to 0 for
/// the neurons the layer's mask turns off. With `backward` the event carries the tape's
/// gradients.
fn run_layer<T: Real>(
    entries: &mut [Entry<'_, T>],
    derivative: &mut Vec<T>,
    layer: usize,
    tape: &Tape<'_, T>,
    trace: &Trace,
    model: &Sequential<T>,
    backward: bool,
) {
    if !entries.iter().any(|e| e.layer == layer) {
        return;
    }
    let l = model.layer(layer);
    let (x, y) = (trace.input(layer), trace.output(layer));
    let input = tape.value(x);
    derivative.resize(input.data().len(), T::ZERO);
    let local_derivative = if l.local_derivative(tape, x, derivative) {
        if let Some(mask) = model.mask(layer) {
            zero_masked(derivative, mask);
        }
        let view = TensorView::from_slice(&derivative[..], input.shape().dims());
        Some(view.expect("the derivative buffer has the input's length"))
    } else {
        None
    };
    let event = LayerEvent {
        layer,
        name: l.name(),
        input,
        output: tape.value(y),
        local_derivative,
        grad_output: if backward { tape.grad(y) } else { None },
        grad_input: if backward { tape.grad(x) } else { None },
    };
    for entry in entries.iter_mut().filter(|e| e.layer == layer) {
        (entry.f)(&event);
    }
}

/// Sets to 0 every value of `values` (rows of `mask.width()` values each, as an element-wise
/// layer's output is laid out) in a column `mask` turns off.
fn zero_masked<T: Real>(values: &mut [T], mask: &NeuronMask) {
    let width = mask.width();
    if width == 0 {
        return;
    }
    debug_assert_eq!(values.len() % width, 0);
    for row in values.chunks_exact_mut(width) {
        for (neuron, v) in row.iter_mut().enumerate() {
            if !mask.is_on(neuron) {
                *v = T::ZERO;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the hooks: the five values of an activation layer, the derivatives of ReLU
    //! and PReLU, the layers without an element-wise derivative, agreement with the tape's
    //! gradients, layer-index checks and training with hooks.

    use std::cell::RefCell;

    use super::*;
    use crate::{
        Batch, Batcher, Dataset, FitConfig, Glu, LeakyRelu, Linear, LogSoftmax, Loss,
        OptimizerConfig, Prelu, Relu, Sequential, Sgd, Softmax, Tanh, Targets, Trace, Trainer, fit,
        fit_with_hooks,
    };
    use cavitation_core::{Gate, Graph, ParamStore, Real, Reduction, Rng, Tensor};

    /// An owned copy of a [`LayerEvent`], so a test can look at it after the step.
    #[derive(Clone, Debug)]
    struct Seen<T: Real> {
        /// The layer's index.
        layer: usize,
        /// The layer's name.
        name: String,
        /// The layer's input.
        input: Vec<T>,
        /// The layer's output.
        output: Vec<T>,
        /// The local derivative, if the layer has one.
        local: Option<Vec<T>>,
        /// The gradient of the loss with respect to the output.
        grad_output: Option<Vec<T>>,
        /// The gradient of the loss with respect to the input.
        grad_input: Option<Vec<T>>,
    }

    /// Copies `e` into a [`Seen`].
    fn seen<T: Real>(e: &LayerEvent<'_, T>) -> Seen<T> {
        Seen {
            layer: e.layer,
            name: e.name.to_string(),
            input: e.input.data().to_vec(),
            output: e.output.data().to_vec(),
            local: e.local_derivative.map(|v| v.data().to_vec()),
            grad_output: e.grad_output.map(|v| v.data().to_vec()),
            grad_input: e.grad_input.map(|v| v.data().to_vec()),
        }
    }

    /// Asserts that `got` and `want` agree element by element within `1e-12` relative.
    fn assert_close(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-12 * w.abs().max(1.0),
                "{what}[{i}]: got {g}, want {w}"
            );
        }
    }

    /// A 2 → 2 linear layer followed by tanh, in `f64`, with a hook on the tanh layer: the
    /// forward event carries z, tanh(z) and 1 − tanh²(z) and no gradients; the backward event
    /// adds ∂L/∂a (the upstream gradient `c`, since L = Σ a·c) and ∂L/∂z = c · (1 − tanh²(z)).
    #[test]
    fn activation_hook_sees_all_five() {
        let mut params = ParamStore::<f64>::new_generic();
        let linear = Linear::new(&mut params, 2, 2, &mut Rng::seed(0));
        params
            .value_mut(linear.weight())
            .data_mut()
            .copy_from_slice(&[0.5, -0.25, 0.75, 1.0]);
        params
            .value_mut(linear.bias())
            .data_mut()
            .copy_from_slice(&[0.1, -0.2]);
        let net = Sequential::new_generic().push(linear).push(Tanh);
        let x = Tensor::from_vec(vec![1.0, 2.0, -0.5, 0.25], &[2, 2]).unwrap();
        let c = Tensor::from_vec(vec![1.0, -2.0, 0.5, 3.0], &[2, 2]).unwrap();

        // z = x Wᵀ + b, worked out by hand.
        let z = [0.1, 2.55, -0.2125, -0.325];
        let a: Vec<f64> = z.iter().map(|v: &f64| v.tanh()).collect();
        let d: Vec<f64> = a.iter().map(|v| 1.0 - v * v).collect();
        let dz: Vec<f64> = d.iter().zip(c.data()).map(|(d, c)| d * c).collect();

        let mut forward = Vec::new();
        let mut backward = Vec::new();
        {
            let mut hooks = Hooks::<f64>::new_generic();
            hooks.on_forward(1, |e| forward.push(seen(e)));
            hooks.on_backward(1, |e| backward.push(seen(e)));
            let mut graph = Graph::<f64>::new_generic();
            let mut trace = Trace::new();
            let mut tape = graph.tape(&params.values);
            let xv = tape.input(x.view());
            let out = net.forward_traced(&mut tape, xv, &mut trace);
            hooks.run_forward(&tape, &trace, &net);
            let cv = tape.input(c.view());
            let prod = tape.mul(out, cv);
            let loss = tape.sum(prod);
            tape.backward(loss, &mut params.grads);
            hooks.run_backward(&tape, &trace, &net);
        }

        assert_eq!(forward.len(), 1);
        let f = &forward[0];
        assert_eq!((f.layer, f.name.as_str()), (1, "tanh"));
        assert_close(&f.input, &z, "forward z");
        assert_close(&f.output, &a, "forward a");
        assert_close(f.local.as_ref().unwrap(), &d, "forward f'(z)");
        assert!(f.grad_output.is_none() && f.grad_input.is_none());

        assert_eq!(backward.len(), 1);
        let b = &backward[0];
        assert_eq!((b.layer, b.name.as_str()), (1, "tanh"));
        assert_close(&b.input, &z, "backward z");
        assert_close(&b.output, &a, "backward a");
        assert_close(b.local.as_ref().unwrap(), &d, "backward f'(z)");
        assert_close(b.grad_output.as_ref().unwrap(), c.data(), "dL/da");
        assert_close(b.grad_input.as_ref().unwrap(), &dz, "dL/dz");
    }

    /// The local derivative of ReLU is 1 where the input is above 0 and 0 elsewhere,
    /// including at exactly 0.
    #[test]
    fn relu_hook_derivative_is_mask() {
        let params = ParamStore::new();
        let net = Sequential::new().push(Relu);
        let x = Tensor::from_vec(vec![-1.0f32, 0.0, 2.0, -0.5, 3.0, 1e-30], &[2, 3]).unwrap();
        let mut local = Vec::new();
        {
            let mut hooks = Hooks::new();
            hooks.on_forward(0, |e| local = e.local_derivative.unwrap().data().to_vec());
            let mut graph = Graph::new();
            let mut trace = Trace::new();
            let mut tape = graph.tape(&params.values);
            let xv = tape.input(x.view());
            net.forward_traced(&mut tape, xv, &mut trace);
            hooks.run_forward(&tape, &trace, &net);
        }
        assert_eq!(local, [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
    }

    /// The local derivative of leaky ReLU is 1 above 0, the slope at and below 0, and NaN
    /// for a NaN input.
    #[test]
    fn leaky_relu_hook_derivative_is_one_slope_or_nan() {
        let params = ParamStore::new();
        let net = Sequential::new().push(LeakyRelu { slope: 0.2 });
        let x = Tensor::from_vec(vec![-1.0f32, 0.0, 2.0, f32::NAN], &[2, 2]).unwrap();
        let mut local = Vec::new();
        {
            let mut hooks = Hooks::new();
            hooks.on_forward(0, |e| local = e.local_derivative.unwrap().data().to_vec());
            let mut graph = Graph::new();
            let mut trace = Trace::new();
            let mut tape = graph.tape(&params.values);
            let xv = tape.input(x.view());
            net.forward_traced(&mut tape, xv, &mut trace);
            hooks.run_forward(&tape, &trace, &net);
        }
        assert_eq!(&local[..3], [0.2, 0.2, 1.0]);
        assert!(local[3].is_nan());
    }

    /// PReLU's local derivative is 1 above 0, the channel's slope at and below 0, and NaN
    /// for a NaN input.
    #[test]
    fn prelu_derivative_is_one_or_slope() {
        let mut params = ParamStore::new();
        let prelu = Prelu::new(&mut params, 2);
        params
            .value_mut(prelu.weight())
            .data_mut()
            .copy_from_slice(&[0.25, 0.5]);
        let net = Sequential::new().push(prelu);
        let x = Tensor::from_vec(vec![-2.0f32, 3.0, 0.0, -1.0, f32::NAN, 1.0], &[3, 2]).unwrap();
        let mut local = Vec::new();
        {
            let mut hooks = Hooks::new();
            hooks.on_forward(0, |e| local = e.local_derivative.unwrap().data().to_vec());
            let mut graph = Graph::new();
            let mut trace = Trace::new();
            let mut tape = graph.tape(&params.values);
            let xv = tape.input(x.view());
            net.forward_traced(&mut tape, xv, &mut trace);
            hooks.run_forward(&tape, &trace, &net);
        }
        assert_eq!(&local[..4], [0.25, 1.0, 0.25, 0.5]);
        assert!(local[4].is_nan());
        assert_eq!(local[5], 1.0);
    }

    /// `Linear`, `Softmax`, `LogSoftmax` and `Glu` are not element-wise functions of one
    /// value, so their events carry no local derivative.
    #[test]
    fn non_elementwise_layers_have_no_local_derivative() {
        let mut params = ParamStore::new();
        let net = Sequential::new()
            .push(Linear::new(&mut params, 4, 4, &mut Rng::seed(0)))
            .push(Softmax)
            .push(LogSoftmax)
            .push(Glu {
                gate: Gate::Sigmoid,
            });
        let x = Tensor::full(&[3, 4], 0.5f32);
        // Several hooks push into one list, so it sits in a `RefCell`.
        let names = RefCell::new(Vec::new());
        {
            let mut hooks = Hooks::new();
            for layer in 0..4 {
                hooks.on_forward(layer, |e| {
                    assert!(e.local_derivative.is_none(), "{}", e.name);
                    names.borrow_mut().push(e.name.to_string());
                });
            }
            let mut graph = Graph::new();
            let mut trace = Trace::new();
            let mut tape = graph.tape(&params.values);
            let xv = tape.input(x.view());
            net.forward_traced(&mut tape, xv, &mut trace);
            hooks.run_forward(&tape, &trace, &net);
        }
        assert_eq!(
            names.into_inner(),
            ["linear", "softmax", "log_softmax", "glu"]
        );
    }

    /// Backward hooks on every layer of a 3 → 4 → 2 network, run by
    /// `Trainer::step_with_hooks`, see exactly the gradients the tape holds for each layer's
    /// output and input, in reverse layer order; layer 0's input is a batch input and takes
    /// no gradient.
    #[test]
    fn backward_hook_matches_tape_grad() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(4);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 3, 4, &mut rng))
            .push(Tanh)
            .push(Linear::new(&mut params, 4, 2, &mut rng));
        let data = Dataset {
            x: Tensor::from_vec(
                vec![0.5f32, -1.0, 2.0, 0.25, 0.75, -0.5, -2.0, 1.5, 1.0],
                &[3, 3],
            )
            .unwrap(),
            targets: Targets::Values(
                Tensor::from_vec(vec![1.0f32, 0.0, -1.0, 0.5, 0.25, 2.0], &[3, 2]).unwrap(),
            ),
        };

        // The same forward and backward pass by hand, reading the tape's gradients.
        let mut want = Vec::new();
        {
            let mut graph = Graph::new();
            let mut trace = Trace::new();
            let mut tape = graph.tape(&params.values);
            let x = tape.input(data.x.view());
            let out = net.forward_traced(&mut tape, x, &mut trace);
            let Targets::Values(t) = &data.targets else {
                unreachable!()
            };
            let t = tape.input(t.view());
            let loss = tape.mse(out, t, Reduction::Mean);
            tape.backward(loss, &mut params.grads);
            for i in (0..net.len()).rev() {
                let grad = |v| tape.grad(v).map(|g| g.data().to_vec());
                want.push((i, grad(trace.output(i)), grad(trace.input(i))));
            }
        }
        params.zero_grad();

        // Several hooks push into one list, so it sits in a `RefCell`.
        let got = RefCell::new(Vec::new());
        {
            let mut hooks = Hooks::new();
            for layer in 0..net.len() {
                hooks.on_backward(layer, |e| {
                    got.borrow_mut().push((
                        e.layer,
                        e.grad_output.map(|g| g.data().to_vec()),
                        e.grad_input.map(|g| g.data().to_vec()),
                    ))
                });
            }
            let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 3);
            let mut batch = Batch::new(&data, 3);
            let view = Batcher::new(3, 3, None).gather(&data, 0..3, &mut batch);
            trainer
                .step_with_hooks(&net, &mut params, view, &mut hooks)
                .unwrap();
        }

        let got = got.into_inner();
        assert_eq!(got, want);
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), [2, 1, 0]);
        assert!(got.iter().all(|g| g.1.is_some()));
        assert!(got[2].2.is_none(), "layer 0's input is a batch input");
        assert_eq!(got[1].2, got[2].1, "layer 1's input is layer 0's output");
    }

    /// A hook on a layer the model does not have panics with the index and the valid range
    /// before any hook runs (the hook on layer 0 would panic with another message).
    #[test]
    #[should_panic(expected = "layer 5, but the model has 2 layers (valid indices 0..2)")]
    fn hook_on_missing_layer_panics() {
        let params = ParamStore::new();
        let net = Sequential::new().push(Relu).push(Tanh);
        let mut hooks = Hooks::new();
        hooks.on_forward(0, |_| panic!("a hook ran before the index check"));
        hooks.on_forward(5, |_| {});
        let mut graph = Graph::new();
        let mut trace = Trace::new();
        let mut tape = graph.tape(&params.values);
        let x = tape.input(Tensor::full(&[1, 2], 1.0f32).view());
        net.forward_traced(&mut tape, x, &mut trace);
        hooks.run_forward(&tape, &trace, &net);
    }

    /// `run_backward` checks the registered indices the same way.
    #[test]
    #[should_panic(expected = "layer 2, but the model has 2 layers (valid indices 0..2)")]
    fn backward_hook_on_missing_layer_panics() {
        let params = ParamStore::new();
        let net = Sequential::new().push(Relu).push(Tanh);
        let mut hooks = Hooks::new();
        hooks.on_backward(1, |_| panic!("a hook ran before the index check"));
        hooks.on_backward(2, |_| {});
        let mut graph = Graph::new();
        let mut trace = Trace::new();
        let mut tape = graph.tape(&params.values);
        let x = tape.input(Tensor::full(&[1, 2], 1.0f32).view());
        net.forward_traced(&mut tape, x, &mut trace);
        hooks.run_backward(&tape, &trace, &net);
    }

    /// A small classifier and data set for the `fit` tests: 8 rows of 3 features, 2 classes.
    fn fit_setup() -> (ParamStore, Sequential, Dataset) {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(7);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 3, 4, &mut rng))
            .push(Tanh)
            .push(Linear::new(&mut params, 4, 2, &mut rng));
        let x: Vec<f32> = (0..24).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let data = Dataset {
            x: Tensor::from_vec(x, &[8, 3]).unwrap(),
            targets: Targets::Classes(vec![0, 1, 1, 0, 1, 0, 0, 1]),
        };
        (params, net, data)
    }

    /// The `FitConfig` the `fit` tests use: 3 epochs of batch 3 (3 batches each, the last of
    /// 2 rows), SGD.
    fn fit_config() -> FitConfig {
        FitConfig {
            epochs: 3,
            batch: 3,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            ..FitConfig::default()
        }
    }

    /// `fit_with_hooks` runs each hook once per step and trains exactly as `fit` does, since
    /// hooks only read; a removed hook stops running.
    #[test]
    fn fit_with_hooks_runs_every_step_and_trains_like_fit() {
        let (mut plain, plain_net, data) = fit_setup();
        let plain_history = fit(&plain_net, &mut plain, &data, fit_config()).unwrap();

        let (mut hooked, net, data) = fit_setup();
        let mut forward_calls = 0;
        let mut backward_calls = 0;
        let mut removed_calls = 0;
        let history = {
            let mut hooks = Hooks::new();
            hooks.on_forward(1, |_| forward_calls += 1);
            hooks.on_backward(2, |_| backward_calls += 1);
            let removed = hooks.on_forward(0, |_| removed_calls += 1);
            hooks.remove(removed);
            fit_with_hooks(&net, &mut hooked, &data, fit_config(), &mut hooks).unwrap()
        };
        assert_eq!((forward_calls, backward_calls, removed_calls), (9, 9, 0));
        let losses = |h: &crate::History| h.epochs.iter().map(|e| e.mean_loss).collect::<Vec<_>>();
        assert_eq!(losses(&history), losses(&plain_history));
        let names = net.parameter_names().into_iter();
        for ((name, id), (_, plain_id)) in names.zip(plain_net.parameter_names()) {
            assert_eq!(
                hooked.value(id).data(),
                plain.value(plain_id).data(),
                "parameter {name}"
            );
        }
    }
}
