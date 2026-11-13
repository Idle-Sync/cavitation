//! Neuron masks: which neurons of a layer's output are switched on.

use cavitation_core::{Real, Tensor};

/// Which neurons of one layer's output are on. A neuron is one position of the output's last
/// axis; a mask set on a layer of a [`Sequential`](crate::Sequential) with
/// [`set_mask`](crate::Sequential::set_mask) turns its off neurons' outputs to exactly 0 and
/// stops their gradient, in training and in every forward pass.
///
/// A mask reaches a network by one of three routes:
///
/// - By layer index: [`Sequential::set_mask`](crate::Sequential::set_mask) puts it on the
///   output of the layer you name.
/// - By `Linear`: [`Sequential::set_neuron_mask`](crate::Sequential::set_neuron_mask) and
///   [`Sequential::set_neuron`](crate::Sequential::set_neuron) turn off a hidden neuron by
///   naming its `Linear`, and put the mask on
///   [`Sequential::neuron_layer`](crate::Sequential::neuron_layer), after the activations
///   that follow it. The neuron's output is then 0 whatever the activations give at 0, and
///   nothing flows back into the weights that feed it. This is the route to use for
///   neurons.
/// - Compiled: [`InferenceModel::set_neuron_mask`](crate::InferenceModel::set_neuron_mask)
///   (by `Linear`) and [`InferenceModel::set_mask`](crate::InferenceModel::set_mask) (by
///   layer index), and [`Evaluator::set_neuron_mask`](crate::Evaluator::set_neuron_mask),
///   change one instance's mask in place without allocating, for a sweep that switches
///   neurons again and again.
///
/// ```
/// use cavitation_nn::NeuronMask;
/// let mut mask = NeuronMask::all_on(4);
/// mask.set(2, false);
/// assert_eq!(mask.count_on(), 3);
/// assert!(!mask.is_on(2));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NeuronMask {
    /// Whether each neuron is on, by position on the last axis.
    on: Vec<bool>,
}

impl NeuronMask {
    /// A mask of `width` neurons, every one on.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// let mask = NeuronMask::all_on(3);
    /// assert_eq!((mask.width(), mask.count_on()), (3, 3));
    /// ```
    pub fn all_on(width: usize) -> NeuronMask {
        NeuronMask {
            on: vec![true; width],
        }
    }

    /// A mask of `width` neurons, every one off.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// let mask = NeuronMask::all_off(3);
    /// assert_eq!((mask.width(), mask.count_on()), (3, 0));
    /// ```
    pub fn all_off(width: usize) -> NeuronMask {
        NeuronMask {
            on: vec![false; width],
        }
    }

    /// A mask of `width` neurons with the neurons in `off` switched off and the rest on.
    /// Listing a neuron twice is the same as listing it once.
    ///
    /// # Panics
    ///
    /// Panics if a neuron in `off` is `>= width`; the message names the neuron and the width.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// let mask = NeuronMask::from_off(5, &[1, 3]);
    /// assert_eq!(mask.on().collect::<Vec<_>>(), [0, 2, 4]);
    /// ```
    pub fn from_off(width: usize, off: &[usize]) -> NeuronMask {
        let mut mask = NeuronMask::all_on(width);
        for &neuron in off {
            mask.check(neuron, "from_off");
            mask.on[neuron] = false;
        }
        mask
    }

    /// A mask of `width` neurons with only the neurons in `on` switched on.
    /// Listing a neuron twice is the same as listing it once.
    ///
    /// # Panics
    ///
    /// Panics if a neuron in `on` is `>= width`; the message names the neuron and the width.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// let mask = NeuronMask::from_on(5, &[0, 2, 4]);
    /// assert_eq!(mask.off().collect::<Vec<_>>(), [1, 3]);
    /// ```
    pub fn from_on(width: usize, on: &[usize]) -> NeuronMask {
        let mut mask = NeuronMask::all_off(width);
        for &neuron in on {
            mask.check(neuron, "from_on");
            mask.on[neuron] = true;
        }
        mask
    }

    /// The positions of the neurons that are on, in increasing order.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// let mask = NeuronMask::from_off(4, &[0]);
    /// assert_eq!(mask.on().collect::<Vec<_>>(), [1, 2, 3]);
    /// ```
    pub fn on(&self) -> impl Iterator<Item = usize> + '_ {
        self.on
            .iter()
            .enumerate()
            .filter(|(_, on)| **on)
            .map(|(i, _)| i)
    }

    /// The positions of the neurons that are off, in increasing order.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// let mask = NeuronMask::from_off(4, &[0, 3]);
    /// assert_eq!(mask.off().collect::<Vec<_>>(), [0, 3]);
    /// ```
    pub fn off(&self) -> impl Iterator<Item = usize> + '_ {
        self.on
            .iter()
            .enumerate()
            .filter(|(_, on)| !**on)
            .map(|(i, _)| i)
    }

    /// Turns neuron `neuron` on (`true`) or off (`false`).
    ///
    /// # Panics
    ///
    /// Panics if `neuron >= width()`; the message names the neuron and the width.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// let mut mask = NeuronMask::all_on(2);
    /// mask.set(0, false);
    /// mask.set(0, true);
    /// assert!(mask.is_on(0));
    /// ```
    pub fn set(&mut self, neuron: usize, on: bool) {
        self.check(neuron, "set");
        self.on[neuron] = on;
    }

    /// Whether neuron `neuron` is on.
    ///
    /// # Panics
    ///
    /// Panics if `neuron >= width()`; the message names the neuron and the width.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// let mut mask = NeuronMask::all_on(2);
    /// mask.set(1, false);
    /// assert!(mask.is_on(0) && !mask.is_on(1));
    /// ```
    pub fn is_on(&self, neuron: usize) -> bool {
        self.check(neuron, "is_on");
        self.on[neuron]
    }

    /// The number of neurons that are on.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// let mut mask = NeuronMask::all_on(5);
    /// mask.set(4, false);
    /// assert_eq!(mask.count_on(), 4);
    /// ```
    pub fn count_on(&self) -> usize {
        self.on.iter().filter(|&&on| on).count()
    }

    /// The number of neurons, on or off: the width of the layer output it fits.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// assert_eq!(NeuronMask::all_on(7).width(), 7);
    /// ```
    pub fn width(&self) -> usize {
        self.on.len()
    }

    /// Panics, naming `method`, the neuron and the width, if `neuron` is out of range.
    fn check(&self, neuron: usize, method: &str) {
        assert!(
            neuron < self.on.len(),
            "NeuronMask::{method}: neuron {neuron} is out of range for width {}",
            self.on.len()
        );
    }
}

impl FromIterator<bool> for NeuronMask {
    /// Collects one flag per neuron, `true` for on, so the mask's width is the number of flags.
    ///
    /// ```
    /// use cavitation_nn::NeuronMask;
    /// let mask: NeuronMask = [true, false, true].into_iter().collect();
    /// assert_eq!(mask.on().collect::<Vec<_>>(), [0, 2]);
    /// ```
    fn from_iter<I: IntoIterator<Item = bool>>(flags: I) -> NeuronMask {
        NeuronMask {
            on: flags.into_iter().collect(),
        }
    }
}

/// A mask stored on a layer of a [`Sequential`](crate::Sequential): the mask, and the same as
/// a `[width]` tensor of 1 (on) and 0 (off) that the forward pass records as the condition of
/// [`Tape::keep_where`](cavitation_core::Tape::keep_where).
#[derive(Clone, Debug)]
pub(crate) struct LayerMask<T: Real> {
    /// The mask as set.
    pub(crate) mask: NeuronMask,
    /// 1 for each neuron that is on, 0 for each that is off.
    pub(crate) keep: Tensor<T>,
}

impl<T: Real> LayerMask<T> {
    /// The stored form of `mask`.
    pub(crate) fn new(mask: NeuronMask) -> LayerMask<T> {
        let values = mask
            .on
            .iter()
            .map(|&on| if on { T::ONE } else { T::ZERO })
            .collect();
        let keep =
            Tensor::from_vec(values, &[mask.width()]).expect("a [width] tensor holds width values");
        LayerMask { mask, keep }
    }

    /// Turns neuron `neuron` on or off in the mask and in the tensor, in place.
    ///
    /// # Panics
    ///
    /// Panics if `neuron` is out of range for the mask's width.
    pub(crate) fn set(&mut self, neuron: usize, on: bool) {
        self.mask.set(neuron, on);
        self.keep.data_mut()[neuron] = if on { T::ONE } else { T::ZERO };
    }
}

#[cfg(test)]
mod tests {
    //! Tests for neuron masks: the mask itself, masked outputs and gradients, the hooks' view
    //! of a masked layer, and the checks on layer indices and widths.

    use std::cell::RefCell;

    use super::*;
    use crate::{Hooks, Init, Linear, Relu, Sequential, Sigmoid, Tanh, Trace};
    use cavitation_core::{Graph, ParamStore, Rng, Tensor};

    /// A fresh mask has every neuron on.
    #[test]
    fn all_on_counts_every_neuron() {
        let mask = NeuronMask::all_on(5);
        assert_eq!(mask.width(), 5);
        assert_eq!(mask.count_on(), 5);
        assert!((0..5).all(|i| mask.is_on(i)));
    }

    /// `from_off`, `from_on`, `all_off` and `collect` build the masks they name.
    #[test]
    fn mask_constructors() {
        let off = NeuronMask::from_off(5, &[1, 3]);
        assert_eq!(off.on().collect::<Vec<_>>(), [0, 2, 4]);
        assert_eq!(off.off().collect::<Vec<_>>(), [1, 3]);
        assert_eq!(
            NeuronMask::from_on(5, &[1, 3]).on().collect::<Vec<_>>(),
            [1, 3]
        );
        assert_eq!(
            NeuronMask::from_on(5, &[1, 3]),
            NeuronMask::from_off(5, &[0, 2, 4])
        );
        assert_eq!(NeuronMask::all_off(3).count_on(), 0);
        let collected: NeuronMask = [true, false, false, true].into_iter().collect();
        assert_eq!(collected, NeuronMask::from_off(4, &[1, 2]));
    }

    /// A neuron past the width panics in `from_off`, naming both.
    #[test]
    #[should_panic(expected = "NeuronMask::from_off: neuron 5 is out of range for width 5")]
    fn from_off_out_of_range_panics() {
        NeuronMask::from_off(5, &[5]);
    }

    /// `set` turns single neurons off and back on.
    #[test]
    fn set_turns_neurons_off_and_on() {
        let mut mask = NeuronMask::all_on(4);
        mask.set(1, false);
        mask.set(3, false);
        assert_eq!(mask.count_on(), 2);
        assert!(!mask.is_on(1) && !mask.is_on(3) && mask.is_on(0) && mask.is_on(2));
        mask.set(1, true);
        assert_eq!(mask.count_on(), 3);
        assert!(mask.is_on(1));
    }

    /// A neuron index at or past the width panics in `set`, naming both.
    #[test]
    #[should_panic(expected = "NeuronMask::set: neuron 4 is out of range for width 4")]
    fn set_out_of_range_panics() {
        NeuronMask::all_on(4).set(4, false);
    }

    /// A neuron index at or past the width panics in `is_on`, naming both.
    #[test]
    #[should_panic(expected = "NeuronMask::is_on: neuron 7 is out of range for width 2")]
    fn is_on_out_of_range_panics() {
        NeuronMask::all_on(2).is_on(7);
    }

    /// `mask` reads back what `set_mask` stored, and `None` after `clear_mask`.
    #[test]
    fn set_and_clear_mask_read_back() {
        let mut net = Sequential::new().push(Relu).push(Tanh);
        assert!(net.mask(1).is_none());
        let mut mask = NeuronMask::all_on(3);
        mask.set(0, false);
        net.set_mask(1, mask.clone());
        assert_eq!(net.mask(1), Some(&mask));
        assert!(net.mask(0).is_none());
        net.clear_mask(1);
        assert!(net.mask(1).is_none());
    }

    /// A 3 → 4 network ending in a sigmoid, from `Rng::seed(5)`.
    fn sigmoid_net(params: &mut ParamStore) -> Sequential {
        Sequential::new()
            .push(Linear::new(params, 3, 4, &mut Rng::seed(5)))
            .push(Sigmoid)
    }

    /// Two rows of three inputs.
    fn two_rows() -> Tensor<f32> {
        Tensor::from_vec(vec![0.3f32, -0.8, 1.2, -0.5, 0.9, 0.1], &[2, 3]).unwrap()
    }

    /// A sigmoid never outputs 0 on its own, so a 0 in column 1 can only come from the mask;
    /// it is exactly +0.0 in every row, and the other columns are untouched.
    #[test]
    fn masked_neuron_outputs_zero_even_for_sigmoid() {
        let mut params = ParamStore::new();
        let mut net = sigmoid_net(&mut params);
        let x = two_rows();
        let plain = {
            let mut graph = Graph::new();
            let mut tape = graph.tape(&params.values);
            let xi = tape.input(x.view());
            let y = net.forward(&mut tape, xi);
            tape.value(y).data().to_vec()
        };
        let mut mask = NeuronMask::all_on(4);
        mask.set(1, false);
        net.set_mask(1, mask);
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xi = tape.input(x.view());
        let y = net.forward(&mut tape, xi);
        let out = tape.value(y).data();
        for row in 0..2 {
            for col in 0..4 {
                let (got, before) = (out[row * 4 + col], plain[row * 4 + col]);
                if col == 1 {
                    assert_eq!(got.to_bits(), 0.0f32.to_bits(), "row {row}");
                    assert!(before > 0.0);
                } else {
                    assert_eq!(got.to_bits(), before.to_bits(), "row {row} column {col}");
                }
            }
        }
    }

    /// With every ReLU input positive, each hidden neuron passes gradient, except the masked
    /// one: the weight row and the bias entry feeding it get exactly 0, every other entry
    /// does not.
    #[test]
    fn masked_neuron_gets_no_gradient() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(0);
        let mut constant = |params: &mut ParamStore, i, o| {
            Linear::with_init(
                params,
                i,
                o,
                Init::Constant(0.5),
                Init::Constant(0.1),
                &mut rng,
            )
        };
        let first = constant(&mut params, 3, 4);
        let (weight, bias) = (first.weight(), first.bias());
        let second = constant(&mut params, 4, 2);
        let mut net = Sequential::new().push(first).push(Relu).push(second);
        let mut mask = NeuronMask::all_on(4);
        mask.set(2, false);
        net.set_mask(1, mask);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 0.5, 0.5, 0.5], &[2, 3]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xi = tape.input(x.view());
        let y = net.forward(&mut tape, xi);
        let loss = tape.sum(y);
        tape.backward(loss, &mut params.grads);
        drop(tape);
        let w = params.grad(weight).data();
        for neuron in 0..4 {
            let row = &w[neuron * 3..neuron * 3 + 3];
            if neuron == 2 {
                assert_eq!(row, &[0.0; 3]);
            } else {
                assert!(row.iter().all(|&g| g != 0.0), "neuron {neuron}: {row:?}");
            }
        }
        assert_eq!(params.grad(bias).data(), &[2.0, 2.0, 0.0, 2.0]);
    }

    /// Clearing a mask leaves no trace on the tape: the node count equals that of a network
    /// that was never masked, and a mask adds nodes.
    #[test]
    fn unmasked_records_no_extra_op() {
        let mut params = ParamStore::new();
        let mut net = sigmoid_net(&mut params);
        let x = two_rows();
        let count = |net: &Sequential| {
            let mut graph = Graph::new();
            let mut tape = graph.tape(&params.values);
            let xi = tape.input(x.view());
            net.forward(&mut tape, xi);
            tape.node_count()
        };
        let never = count(&net);
        net.set_mask(1, NeuronMask::all_on(4));
        let masked = count(&net);
        net.clear_mask(1);
        assert_eq!(count(&net), never);
        assert!(masked > never);
    }

    /// A mask on a layer the network does not have panics, naming the index and the range.
    #[test]
    #[should_panic(
        expected = "sequential: cannot mask layer 5, the model has 2 layers (valid indices 0..2)"
    )]
    fn mask_on_missing_layer_panics() {
        let mut net = Sequential::new().push(Relu).push(Tanh);
        net.set_mask(5, NeuronMask::all_on(3));
    }

    /// A width-3 mask on a 4-wide layer panics at the next forward, naming both widths.
    #[test]
    #[should_panic(
        expected = "sequential: the mask on layer 1 (sigmoid) has width 3, but the layer outputs width 4"
    )]
    fn mask_width_mismatch_panics() {
        let mut params = ParamStore::new();
        let mut net = sigmoid_net(&mut params);
        net.set_mask(1, NeuronMask::all_on(3));
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xi = tape.input(two_rows().view());
        net.forward(&mut tape, xi);
    }

    /// A masked neuron whose output is infinite becomes exactly 0 (a multiply by 0 would give
    /// NaN); the unmasked infinite neuron stays infinite.
    #[test]
    fn masked_infinite_output_is_zero() {
        let mut params = ParamStore::new();
        let layer = Linear::with_init(
            &mut params,
            1,
            2,
            Init::Constant(f64::from(f32::MAX)),
            Init::Zeros,
            &mut Rng::seed(0),
        );
        let mut net = Sequential::new().push(layer);
        let mut mask = NeuronMask::all_on(2);
        mask.set(0, false);
        net.set_mask(0, mask);
        let x = Tensor::from_vec(vec![10.0f32], &[1, 1]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xi = tape.input(x.view());
        let y = net.forward(&mut tape, xi);
        let out = tape.value(y).data();
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[1], f32::INFINITY);
    }

    /// For a masked neuron the hooks show an output of 0, a local derivative of 0 and an input
    /// gradient of 0; the trace's output is the masked one.
    #[test]
    fn masked_neuron_hook_shows_zero_derivative() {
        let mut params = ParamStore::new();
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 3, 4, &mut Rng::seed(9)))
            .push(Tanh);
        let mut mask = NeuronMask::all_on(4);
        mask.set(1, false);
        net.set_mask(1, mask);
        let forward = RefCell::new(Vec::new());
        let backward = RefCell::new(Vec::new());
        let mut hooks = Hooks::new();
        hooks.on_forward(1, |e| {
            forward.borrow_mut().push((
                e.output.data().to_vec(),
                e.local_derivative.unwrap().data().to_vec(),
            ));
        });
        hooks.on_backward(1, |e| {
            backward
                .borrow_mut()
                .push(e.grad_input.unwrap().data().to_vec());
        });
        let mut graph = Graph::new();
        let mut trace = Trace::new();
        let mut tape = graph.tape(&params.values);
        let xi = tape.input(two_rows().view());
        let y = net.forward_traced(&mut tape, xi, &mut trace);
        assert_eq!(trace.output(1), y);
        hooks.run_forward(&tape, &trace, &net);
        let loss = tape.sum(y);
        tape.backward(loss, &mut params.grads);
        hooks.run_backward(&tape, &trace, &net);
        drop(hooks);
        let (output, local) = &forward.borrow()[0];
        let grad_input = &backward.borrow()[0];
        for row in 0..2 {
            for col in 0..4 {
                let i = row * 4 + col;
                if col == 1 {
                    assert_eq!(output[i], 0.0, "output row {row}");
                    assert_eq!(local[i], 0.0, "derivative row {row}");
                    assert_eq!(grad_input[i], 0.0, "input gradient row {row}");
                } else {
                    assert!(output[i] != 0.0 && local[i] > 0.0 && grad_input[i] > 0.0);
                }
            }
        }
    }
}
