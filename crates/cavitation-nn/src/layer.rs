//! The `Layer` trait.

use cavitation_core::{ParamId, Real, Tape, Var};

/// One stage of a network: it records its forward pass on a tape and names its parameters.
///
/// The trait is object safe, so a network can hold its layers as `Box<dyn Layer<T>>`.
/// Layers keep only [`ParamId`]s, never values: the values live in the
/// [`ParamStore`](cavitation_core::ParamStore).
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::{Layer, Linear};
/// let mut params = ParamStore::new();
/// let layers: Vec<Box<dyn Layer<f32>>> = vec![
///     Box::new(Linear::new(&mut params, 4, 3, &mut Rng::seed(0))),
///     Box::new(Linear::new(&mut params, 3, 2, &mut Rng::seed(1))),
/// ];
/// let width = layers.iter().fold(4, |w, l| l.output_width(w));
/// assert_eq!(width, 2);
/// ```
pub trait Layer<T: Real>: Send + Sync {
    /// The layer's kind, for example `"linear"`.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Layer, Linear};
    /// let layer = Linear::new(&mut ParamStore::new(), 2, 2, &mut Rng::seed(0));
    /// assert_eq!(Layer::<f32>::name(&layer), "linear");
    /// ```
    fn name(&self) -> &str;

    /// The width of the layer's output for an input of `input_width` columns.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Layer, Linear};
    /// let layer = Linear::new(&mut ParamStore::new(), 2, 5, &mut Rng::seed(0));
    /// assert_eq!(Layer::<f32>::output_width(&layer, 2), 5);
    /// ```
    fn output_width(&self, input_width: usize) -> usize;

    /// Records the layer's forward pass on `tape` for the input `x` and returns its output.
    ///
    /// # Panics
    ///
    /// Panics if the shape of `x` does not suit the layer; the message names the layer, the
    /// width it expects and the shape it got.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Rng, Tensor};
    /// use cavitation_nn::{Layer, Linear};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&params.values);
    /// let x = tape.input(Tensor::full(&[4, 2], 1.0f32).view());
    /// let y = layer.forward(&mut tape, x);
    /// assert_eq!(tape.shape(y).dims(), &[4, 3]);
    /// ```
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var;

    /// The layer's parameters with their local names (for example `"weight"` and `"bias"`),
    /// in a fixed order. Meant for saving and listing, not for the training loop.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Layer, Linear};
    /// let layer = Linear::new(&mut ParamStore::new(), 2, 3, &mut Rng::seed(0));
    /// let names: Vec<_> = Layer::<f32>::parameters(&layer).iter().map(|p| p.0).collect();
    /// assert_eq!(names, ["weight", "bias"]);
    /// ```
    fn parameters(&self) -> Vec<(&'static str, ParamId)>;
}
