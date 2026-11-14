//! Measuring a model on held-out data: [`Evaluator`], which reuses its compiled model and
//! buffers from one call to the next, and the one-shot [`evaluate`].

use std::mem;

use cavitation_core::{Error, Graph, ParamStore, ParamValues, TensorView};

use super::fit::{argmax, check_dataset, check_model_and_loss, check_widths};
use super::{Dataset, Loss, Targets, TargetsView};
use crate::{InferenceModel, NeuronMask, Sequential};

/// What [`Evaluator::evaluate`] measures on a data set.
///
/// It is `#[non_exhaustive]`, so a later release can add a measure without breaking your
/// code: read its fields, and get one from [`evaluate`] or [`Evaluator::evaluate`].
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{Dataset, Evaluation, Linear, Loss, Sequential, Targets, evaluate};
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
/// let data = Dataset { x: Tensor::zeros(&[2, 2]), targets: Targets::Classes(vec![0, 0]) };
/// let e: Evaluation = evaluate(&net, &params, &data, Loss::CrossEntropy, 2).unwrap();
/// assert_eq!((e.accuracy, e.r2), (Some(1.0), None));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Evaluation {
    /// The mean loss over the rows, defined as [`EpochReport::mean_loss`] is: each batch's
    /// loss (as [`Trainer::step`] computes it) weighted by its row count.
    ///
    /// [`EpochReport::mean_loss`]: crate::EpochReport::mean_loss
    /// [`Trainer::step`]: crate::Trainer::step
    pub loss: f64,
    /// For class targets, the share of rows whose first largest output is at the row's
    /// label (a row whose outputs are all NaN counts as wrong); `None` for value targets.
    pub accuracy: Option<f64>,
    /// For value targets, the coefficient of determination R² of the outputs, as
    /// scikit-learn's `r2_score` with `multioutput='uniform_average'`; `None` for class
    /// targets. With one row every output has zero variance, so R² is 1 or 0, where
    /// scikit-learn returns NaN.
    pub r2: Option<f64>,
}

/// How the evaluator runs the model.
#[derive(Clone, Debug)]
enum Engine {
    /// Through the compiled model's batched path.
    Compiled(InferenceModel),
    /// On the tape, with a copy of the parameter values: for a model with a layer that has
    /// no inference form.
    Tape(ParamValues<f32>),
}

/// What [`Evaluator::reload`] did, for the tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reload {
    /// The compiled model took the new values in place.
    InPlace,
    /// The model was compiled afresh.
    Recompiled,
    /// The model runs on the tape.
    Tape,
}

/// Measures a model on data sets, again and again, without allocating once it has seen a
/// data set of that shape: the loss, and the accuracy (class targets) or R² (value
/// targets).
///
/// The model is compiled into an [`InferenceModel`] and each batch runs through
/// [`InferenceModel::infer_batch`]; a model with a layer that has no inference form (an
/// activation of your own, say) runs on the tape instead, with the same results within
/// rounding. Each batch's outputs then go onto a tape as an input and the loss is recorded
/// there exactly as [`Trainer::step`](crate::Trainer::step) records it, so
/// [`Evaluation::loss`] is the training loss's definition, row-weighted over the batches as
/// [`EpochReport::mean_loss`](crate::EpochReport::mean_loss) is.
///
/// The evaluator copies the weights when it is made: call [`refresh`](Self::refresh) after
/// training to pick up new ones. A clone shares the compiled weights and has buffers and
/// neuron masks of its own, so each thread can evaluate with its own clone.
///
/// To measure the network with neurons switched off, again and again (an ablation sweep),
/// make it with [`with_neuron_masks`](Self::with_neuron_masks) and switch them with
/// [`set_neuron_mask`](Self::set_neuron_mask), which allocates nothing on either engine.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{Dataset, Evaluator, Linear, Loss, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let layer = Linear::new(&mut params, 1, 1, &mut Rng::seed(0));
/// params.value_mut(layer.weight()).data_mut()[0] = 2.0;
/// params.value_mut(layer.bias()).data_mut()[0] = 1.0;
/// let net = Sequential::new().push(layer);
/// let data = Dataset {
///     x: Tensor::from_vec(vec![0.0f32, 1.0, 2.0], &[3, 1]).unwrap(),
///     targets: Targets::Values(Tensor::from_vec(vec![1.0f32, 3.0, 6.0], &[3, 1]).unwrap()),
/// };
/// let mut evaluator = Evaluator::new(&net, &params, Loss::Mse, 2).unwrap();
/// let e = evaluator.evaluate(&data).unwrap();
/// // The outputs are 1, 3 and 5: batch [0, 1] has loss 0, batch [2] has loss 1.
/// assert!((e.loss - 1.0 / 3.0).abs() < 1e-12);
/// assert!(e.accuracy.is_none() && e.r2.unwrap() > 0.8);
/// ```
pub struct Evaluator {
    /// The model, sharing its layers with the caller's, with the neuron masks this
    /// evaluator runs: the tape engine runs it, and both engines keep its masks equal to
    /// the ones in use, so that [`refresh`](Self::refresh) can carry them over.
    model: Sequential,
    /// The `Linear` layers given a neuron mask by
    /// [`with_neuron_masks`](Self::with_neuron_masks).
    linears: Vec<usize>,
    /// The loss to measure.
    loss: Loss,
    /// The number of rows run at a time.
    batch: usize,
    /// How the model runs.
    engine: Engine,
    /// The storage the loss tape (and, on the tape engine, the forward pass) records on.
    graph: Graph<f32>,
    /// An empty set of parameter values, for the loss tape of the compiled engine.
    no_params: ParamValues<f32>,
    /// The mean of each target output, for R².
    means: Vec<f64>,
    /// The sum of squared deviations of each target output from its mean, for R².
    ss_tot: Vec<f64>,
    /// The sum of squared errors of each output, for R².
    ss_res: Vec<f64>,
}

impl Clone for Evaluator {
    /// A copy that shares the compiled weights (see [`Evaluator::refresh`]) and has a graph
    /// and buffers of its own.
    fn clone(&self) -> Evaluator {
        Evaluator {
            model: self.model.clone(),
            linears: self.linears.clone(),
            loss: self.loss,
            batch: self.batch,
            engine: self.engine.clone(),
            graph: Graph::new(),
            no_params: self.no_params.clone(),
            means: self.means.clone(),
            ss_tot: self.ss_tot.clone(),
            ss_res: self.ss_res.clone(),
        }
    }
}

impl std::fmt::Debug for Evaluator {
    /// Shows the loss, the batch size and whether the model runs compiled or on the tape.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Evaluator")
            .field("loss", &self.loss)
            .field("batch", &self.batch)
            .field("compiled", &matches!(self.engine, Engine::Compiled(_)))
            .finish()
    }
}

impl Evaluator {
    /// An evaluator of `model` with the parameter values in `params` (copied: later changes
    /// reach it through [`refresh`](Self::refresh)), measuring `loss`, `batch` rows at a
    /// time.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if `batch` is 0, and [`Error::Unsupported`] if the model
    /// has no layers.
    ///
    /// # Panics
    ///
    /// Panics as [`InferenceModel::compile`] does when the layers' widths do not fit one
    /// another.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Evaluator, Linear, Loss, Sequential};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 4, 3, &mut Rng::seed(0)));
    /// assert!(Evaluator::new(&net, &params, Loss::CrossEntropy, 128).is_ok());
    /// assert!(Evaluator::new(&net, &params, Loss::CrossEntropy, 0).is_err());
    /// ```
    pub fn new(
        model: &Sequential,
        params: &ParamStore,
        loss: Loss,
        batch: usize,
    ) -> Result<Evaluator, Error> {
        Evaluator::with_neuron_masks(model, params, loss, batch, &[])
    }

    /// Like [`new`](Self::new), and gives the neurons of each `Linear` listed in `linears`
    /// (by layer index) a mask with every neuron on, at [`Sequential::neuron_layer`] of it,
    /// so that [`set_neuron_mask`](Self::set_neuron_mask) can switch them later without
    /// allocating. A `Linear` whose neurons already have a mask in `model` keeps it.
    ///
    /// On the compiled engine the masks are mask steps of the compiled model
    /// ([`InferenceModel::compile_with_neuron_masks`]); on the tape they are masks on the
    /// evaluator's own copy of the model. Either way the caller's model is not changed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if `batch` is 0 or an entry of `linears` is not the
    /// index of a `Linear` layer (the message names it), and [`Error::Unsupported`] if the
    /// model has no layers.
    ///
    /// # Panics
    ///
    /// Panics as [`InferenceModel::compile`] does when the layers' widths do not fit one
    /// another.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{
    ///     Dataset, Evaluator, Linear, Loss, NeuronMask, Relu, Sequential, Targets,
    /// };
    /// let mut params = ParamStore::new();
    /// let mut rng = Rng::seed(0);
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 4, &mut rng))
    ///     .push(Relu)
    ///     .push(Linear::new(&mut params, 4, 1, &mut rng));
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![1.0f32, -1.0, 0.5, 2.0], &[2, 2]).unwrap(),
    ///     targets: Targets::Values(Tensor::zeros(&[2, 1])),
    /// };
    /// let mut evaluator =
    ///     Evaluator::with_neuron_masks(&net, &params, Loss::Mse, 2, &[0]).unwrap();
    /// evaluator.set_neuron_mask(0, &NeuronMask::all_off(4)).unwrap();
    /// // With every hidden neuron off the output is the last bias, the same for each row.
    /// let off = evaluator.evaluate(&data).unwrap();
    /// let bias = params.value(net.linear(2).unwrap().bias()).data()[0] as f64;
    /// assert!((off.loss - bias * bias).abs() < 1e-6);
    /// ```
    pub fn with_neuron_masks(
        model: &Sequential,
        params: &ParamStore,
        loss: Loss,
        batch: usize,
        linears: &[usize],
    ) -> Result<Evaluator, Error> {
        if batch == 0 {
            return Err(Error::InvalidData(
                "the batch size must be at least 1".to_string(),
            ));
        }
        check_layers(model)?;
        check_linears(model, linears)?;
        let engine = match InferenceModel::compile_with_neuron_masks(model, params, linears) {
            Ok(compiled) => Engine::Compiled(compiled),
            Err(_) => Engine::Tape(params.values.clone()),
        };
        let mut copy = model.clone();
        for &linear in linears {
            if copy.neuron_mask(linear).is_none() {
                let width = copy.linear(linear).expect("checked above").outputs();
                copy.put_neuron_mask(linear, NeuronMask::all_on(width));
            }
        }
        Ok(Evaluator {
            model: copy,
            linears: linears.to_vec(),
            loss,
            batch,
            engine,
            graph: Graph::new(),
            no_params: ParamStore::new().values,
            means: Vec::new(),
            ss_tot: Vec::new(),
            ss_res: Vec::new(),
        })
    }

    /// Picks up new weights and masks: `model` and `params` replace the ones the evaluator
    /// was made or last refreshed with.
    ///
    /// When no clone of this evaluator shares its compiled weights and the new model
    /// compiles to the same layout (the same layers with the same widths, and masks on the
    /// same layers), the new weights are copied into the compiled model in place, and its
    /// buffers are kept. Otherwise (a clone shares the compiled weights, or a width changed,
    /// or a mask was added or removed) the model is compiled afresh for this evaluator
    /// alone, and a clone keeps evaluating the weights it had. The mask values are each
    /// evaluator's own, never shared with a clone: with masks on the same layers they are
    /// copied in place either way. A model with a layer that has no inference form runs on
    /// the tape, with a copy of the new values.
    ///
    /// The neuron masks [`with_neuron_masks`](Self::with_neuron_masks) added stay, with the
    /// values [`set_neuron_mask`](Self::set_neuron_mask) gave them, on either engine (and
    /// when the evaluator moves between the two), unless `model` has a neuron mask of its
    /// own on that `Linear`, which then replaces them; an added mask whose `Linear` changed
    /// width starts again with every neuron on. When the layer that holds a listed
    /// `Linear`'s neurons changes (an element-wise layer was added after it, say), the mask
    /// moves there with its values, and the model is compiled afresh so that no mask is
    /// left on the old layer.
    ///
    /// When `model` has the very layers the evaluator was last made or refreshed with (the
    /// model itself, or a clone of it, with only parameter values and mask values changed
    /// since, as during [`fit`](crate::fit)), the model compiles, no clone shares its
    /// weights and no mask was added or removed, `refresh` copies the new values in place
    /// and allocates nothing. Otherwise it allocates: it builds the new weights before
    /// copying them in, and on the tape it copies the parameter values afresh.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] if the model has no layers; the evaluator is then
    /// unchanged.
    ///
    /// # Panics
    ///
    /// Panics as [`InferenceModel::compile`] does when the layers' widths do not fit one
    /// another.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{Dataset, Evaluator, Linear, Loss, Sequential, Targets};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 1, 1, &mut Rng::seed(0));
    /// let (w, b) = (layer.weight(), layer.bias());
    /// let net = Sequential::new().push(layer);
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![1.0f32, 2.0], &[2, 1]).unwrap(),
    ///     targets: Targets::Values(Tensor::from_vec(vec![2.0f32, 4.0], &[2, 1]).unwrap()),
    /// };
    /// let mut evaluator = Evaluator::new(&net, &params, Loss::Mse, 2).unwrap();
    /// params.value_mut(w).data_mut()[0] = 2.0;
    /// params.value_mut(b).data_mut()[0] = 0.0;
    /// evaluator.refresh(&net, &params).unwrap();
    /// assert_eq!(evaluator.evaluate(&data).unwrap().loss, 0.0);
    /// ```
    pub fn refresh(&mut self, model: &Sequential, params: &ParamStore) -> Result<(), Error> {
        self.reload(model, params).map(|_| ())
    }

    /// The work of [`refresh`](Self::refresh), saying which way it went.
    fn reload(&mut self, model: &Sequential, params: &ParamStore) -> Result<Reload, Error> {
        check_layers(model)?;
        if self.reload_in_place(model, params) {
            return Ok(Reload::InPlace);
        }
        // The evaluator's own copy of `model`: each listed `Linear` without a neuron mask
        // in `model` gets the mask in use now (or every neuron on, if its width changed).
        let mut copy = model.clone();
        let mut carried = Vec::new();
        for &linear in &self.linears {
            let Some(layer) = model.linear(linear) else {
                continue;
            };
            if model.neuron_mask(linear).is_some() {
                continue;
            }
            let width = layer.outputs();
            let mask = (self.model.linear(linear).is_some())
                .then(|| self.model.neuron_mask(linear))
                .flatten()
                .filter(|m| m.width() == width)
                .cloned()
                .unwrap_or_else(|| NeuronMask::all_on(width));
            copy.put_neuron_mask(linear, mask.clone());
            carried.push((linear, mask));
        }
        let listed: Vec<usize> = carried.iter().map(|(l, _)| *l).collect();
        let mut reload = match &mut self.engine {
            Engine::Compiled(compiled) => match compiled.refresh(model, params) {
                Ok(true) => Reload::InPlace,
                Ok(false) => Reload::Recompiled,
                Err(_) => Reload::Tape,
            },
            Engine::Tape(_) => {
                match InferenceModel::compile_with_neuron_masks(model, params, &listed) {
                    Ok(compiled) => {
                        self.engine = Engine::Compiled(compiled);
                        Reload::Recompiled
                    }
                    Err(_) => Reload::Tape,
                }
            }
        };
        match &mut self.engine {
            Engine::Compiled(compiled) if reload != Reload::Tape => {
                // A step `refresh` kept has these values already; a fresh one gets them
                // here. If a listed `Linear`'s neurons moved to a layer the kept steps do
                // not cover (an element-wise layer was added after it), the old step would
                // still mask a layer the copy does not: compile afresh with steps where the
                // neurons are now, so the compiled model and the copy agree.
                let moved = carried
                    .iter()
                    .any(|(linear, mask)| compiled.set_neuron_mask(*linear, mask).is_err());
                if moved {
                    let mut fresh =
                        InferenceModel::compile_with_neuron_masks(model, params, &listed)?;
                    for (linear, mask) in &carried {
                        fresh
                            .set_neuron_mask(*linear, mask)
                            .expect("the listed Linear has a mask step of its width");
                    }
                    *compiled = fresh;
                    reload = Reload::Recompiled;
                }
            }
            Engine::Tape(values) => values.clone_from(&params.values),
            Engine::Compiled(_) => self.engine = Engine::Tape(params.values.clone()),
        }
        self.model = copy;
        Ok(reload)
    }

    /// The work of [`refresh`](Self::refresh) without allocating, when `model` has the same
    /// layers as the evaluator's copy and the masks sit where they did: copies the new
    /// weights into the compiled model and each of `model`'s mask values into the compiled
    /// model and the copy, as [`reload`](Self::reload) would. A mask added by
    /// [`with_neuron_masks`](Self::with_neuron_masks) on a `Linear` that `model` does not
    /// mask keeps its values. Returns `false`, changing nothing, when it cannot: the layers
    /// differ, a mask was added, removed or changed width, the model runs on the tape, or
    /// the compiled model refuses ([`InferenceModel::refresh_in_place`]).
    fn reload_in_place(&mut self, model: &Sequential, params: &ParamStore) -> bool {
        if !self.model.same_layers(model) {
            return false;
        }
        for i in 0..model.len() {
            let carried = (self.linears.iter())
                .any(|&l| model.neuron_layer(l) == i && model.neuron_mask(l).is_none());
            match (model.mask(i), self.model.mask(i)) {
                (Some(new), Some(old)) if new.width() == old.width() => {}
                (None, Some(_)) if carried => {}
                (None, None) if !carried => {}
                _ => return false,
            }
        }
        let Engine::Compiled(compiled) = &mut self.engine else {
            return false;
        };
        if !compiled.refresh_in_place(model, params) {
            return false;
        }
        for i in 0..model.len() {
            if let Some(mask) = model.mask(i) {
                self.model.copy_mask(i, mask);
            }
        }
        true
    }

    /// Replaces the mask on the neurons of the `Linear` at layer `linear` with `mask`: each
    /// neuron `mask` turns off outputs exactly 0, after the activations that follow the
    /// `Linear` (at [`Sequential::neuron_layer`]), from the next
    /// [`evaluate`](Self::evaluate) on. Only this evaluator changes; its clones keep their
    /// own masks. Allocates nothing, on the compiled engine and on the tape.
    ///
    /// The neurons must have a mask: one listed in
    /// [`with_neuron_masks`](Self::with_neuron_masks), or one the model had when the
    /// evaluator was made or refreshed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if layer `linear` is not a `Linear` or `mask`'s width
    /// differs from its output width, and [`Error::Unsupported`] if its neurons have no mask
    /// (each message names the `Linear`); the evaluator is then unchanged.
    ///
    /// ```
    /// use cavitation_core::{Error, ParamStore, Rng};
    /// use cavitation_nn::{Evaluator, Linear, Loss, NeuronMask, Relu, Sequential};
    /// let mut params = ParamStore::new();
    /// let mut rng = Rng::seed(0);
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 4, &mut rng))
    ///     .push(Relu)
    ///     .push(Linear::new(&mut params, 4, 3, &mut rng));
    /// let mut evaluator =
    ///     Evaluator::with_neuron_masks(&net, &params, Loss::CrossEntropy, 32, &[0]).unwrap();
    /// assert!(evaluator.set_neuron_mask(0, &NeuronMask::from_off(4, &[2])).is_ok());
    /// let wrong_width = evaluator.set_neuron_mask(0, &NeuronMask::all_on(3));
    /// assert!(matches!(wrong_width, Err(Error::InvalidData(_))));
    /// let not_listed = evaluator.set_neuron_mask(2, &NeuronMask::all_on(3));
    /// assert!(matches!(not_listed, Err(Error::Unsupported(_))));
    /// ```
    pub fn set_neuron_mask(&mut self, linear: usize, mask: &NeuronMask) -> Result<(), Error> {
        let Some(layer) = self.model.linear(linear) else {
            return Err(Error::InvalidData(format!(
                "evaluator: layer {linear} is not a Linear layer"
            )));
        };
        let width = layer.outputs();
        if mask.width() != width {
            return Err(Error::InvalidData(format!(
                "evaluator: a mask of width {} for the linear layer {linear}, which outputs \
                 width {width}",
                mask.width()
            )));
        }
        match &mut self.engine {
            Engine::Compiled(compiled) => {
                compiled.set_neuron_mask(linear, mask)?;
                // Keep the copy's masks equal to the ones in use, for `refresh`.
                self.model.copy_neuron_mask(linear, mask);
            }
            Engine::Tape(_) => {
                if !self.model.copy_neuron_mask(linear, mask) {
                    return Err(Error::Unsupported(format!(
                        "evaluator: the neurons of the linear layer {linear} have no mask; \
                         list it in with_neuron_masks"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Measures the model on `data`: the loss, and the accuracy (class targets) or R²
    /// (value targets), running the rows in order, `batch` at a time. Allocates nothing
    /// once it has evaluated a data set with the same target kind and output count, compiled
    /// or on the tape.
    ///
    /// Accuracy counts a row as right when its first largest output (ignoring NaN) is at
    /// its label. R² follows scikit-learn's `r2_score` with `multioutput='uniform_average'`:
    /// for each output, `1 - Σ(y - ŷ)² / Σ(y - ȳ)²` in `f64` over every row, then the mean
    /// over the outputs; an output whose targets do not vary scores 1 when it is predicted
    /// exactly and 0 otherwise. With one row every output has zero variance, so R² is 1 or
    /// 0, where scikit-learn returns NaN. R² is computed on the model's outputs as they are, so for
    /// a loss on raw scores (such as [`Loss::BinaryCrossEntropyWithLogits`]) it compares scores
    /// with targets.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`], as [`fit`](crate::fit) does before training, if
    /// `data.x` is not a matrix or has no rows, if the targets are not one per row, if the
    /// kind of targets is not the kind the loss needs, if binary cross-entropy targets are
    /// outside `[0, 1]`, if the inputs are not as wide as the model expects, if value
    /// targets are not as wide as the model's output, or if a class label is not below the
    /// model's output width (the message names the row, the label and the width).
    ///
    /// # Panics
    ///
    /// Panics as [`InferenceModel::infer_batch`] (or [`Sequential::forward`]) and the loss's
    /// tape operation do when the shapes do not fit.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{Dataset, Evaluator, Linear, Loss, Sequential, Targets};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
    /// let data = Dataset {
    ///     x: Tensor::zeros(&[4, 2]),
    ///     targets: Targets::Classes(vec![0, 1, 0, 1]),
    /// };
    /// let mut evaluator = Evaluator::new(&net, &params, Loss::CrossEntropy, 3).unwrap();
    /// let e = evaluator.evaluate(&data).unwrap();
    /// // Zero inputs and a zero bias give equal scores: every row predicts class 0, and the
    /// // loss is ln 2.
    /// assert_eq!(e.accuracy, Some(0.5));
    /// assert!((e.loss - 2f64.ln()).abs() < 1e-6);
    /// ```
    pub fn evaluate(&mut self, data: &Dataset) -> Result<Evaluation, Error> {
        check_dataset(data)?;
        check_model_and_loss(&self.model, self.loss, data)?;
        let mut means = mem::take(&mut self.means);
        let mut ss_tot = mem::take(&mut self.ss_tot);
        let mut ss_res = mem::take(&mut self.ss_res);
        if let Targets::Values(t) = &data.targets {
            target_spread(t.data(), t.shape().dims()[1], &mut means, &mut ss_tot);
            ss_res.clear();
            ss_res.resize(means.len(), 0.0);
        }
        let mut correct = 0usize;
        let ran = self.run(data, true, |start, out, targets| {
            let width = out.shape().dims()[1];
            match targets {
                TargetsView::Classes(labels) => {
                    for (i, (row, &label)) in out
                        .data()
                        .chunks_exact(width.max(1))
                        .zip(labels)
                        .enumerate()
                    {
                        check_label(start + i, label, width)?;
                        if argmax(row) == Some(label) {
                            correct += 1;
                        }
                    }
                }
                TargetsView::Values(values) => {
                    check_value_width(values, width)?;
                    for (y, chunk) in values
                        .data()
                        .chunks_exact(width.max(1))
                        .zip(out.data().chunks_exact(width.max(1)))
                    {
                        for ((s, &t), &p) in ss_res.iter_mut().zip(y).zip(chunk) {
                            let e = f64::from(t) - f64::from(p);
                            *s += e * e;
                        }
                    }
                }
            }
            Ok(())
        });
        let evaluation = ran.map(|loss| {
            let rows = data.len() as f64;
            match &data.targets {
                Targets::Classes(_) => Evaluation {
                    loss,
                    accuracy: Some(correct as f64 / rows),
                    r2: None,
                },
                Targets::Values(_) => Evaluation {
                    loss,
                    accuracy: None,
                    r2: Some(r2(&ss_res, &ss_tot)),
                },
            }
        });
        self.means = means;
        self.ss_tot = ss_tot;
        self.ss_res = ss_res;
        evaluation
    }

    /// Fills `out` (cleared first) with the index of every row of `data` the model gets
    /// wrong, in order: the rows whose first largest output is not at their label. Allocates
    /// nothing once `out` has room for them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if `data` has value targets, and as
    /// [`evaluate`](Self::evaluate) does for the data checks that do not involve the loss.
    ///
    /// # Panics
    ///
    /// Panics as [`evaluate`](Self::evaluate) does.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{Dataset, Evaluator, Linear, Loss, Sequential, Targets};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
    /// let data = Dataset {
    ///     x: Tensor::zeros(&[4, 2]),
    ///     targets: Targets::Classes(vec![0, 1, 0, 1]),
    /// };
    /// let mut evaluator = Evaluator::new(&net, &params, Loss::CrossEntropy, 3).unwrap();
    /// let mut wrong = Vec::new();
    /// evaluator.misclassified(&data, &mut wrong).unwrap();
    /// // Every row predicts class 0, so the rows labelled 1 are wrong.
    /// assert_eq!(wrong, [1, 3]);
    /// ```
    pub fn misclassified(&mut self, data: &Dataset, out: &mut Vec<usize>) -> Result<(), Error> {
        out.clear();
        check_dataset(data)?;
        if let Targets::Values(_) = data.targets {
            return Err(Error::InvalidData(
                "misclassified needs class targets, but the data has value targets".to_string(),
            ));
        }
        check_widths(&self.model, data)?;
        self.run(data, false, |start, outputs, targets| {
            let width = outputs.shape().dims()[1];
            if let TargetsView::Classes(labels) = targets {
                for (i, (row, &label)) in outputs
                    .data()
                    .chunks_exact(width.max(1))
                    .zip(labels)
                    .enumerate()
                {
                    check_label(start + i, label, width)?;
                    if argmax(row) != Some(label) {
                        out.push(start + i);
                    }
                }
            }
            Ok(())
        })
        .map(|_| ())
    }

    /// Runs the model over `data` in order, `batch` rows at a time, calling `visit` with
    /// each batch's first row, its outputs and its targets, then (when `with_loss`)
    /// recording the batch's loss. Returns the row-weighted mean loss (0 without the loss),
    /// or the first error `visit` returns.
    fn run(
        &mut self,
        data: &Dataset,
        with_loss: bool,
        mut visit: impl FnMut(usize, TensorView<'_, f32>, TargetsView<'_, f32>) -> Result<(), Error>,
    ) -> Result<f64, Error> {
        let (rows, features) = (data.len(), data.features());
        let Evaluator {
            model,
            loss,
            batch,
            engine,
            graph,
            no_params,
            ..
        } = self;
        let mut total = 0.0f64;
        for start in (0..rows).step_by(*batch) {
            let end = (start + *batch).min(rows);
            let n = end - start;
            let x = TensorView::from_slice(
                &data.x.data()[start * features..end * features],
                &[n, features],
            )
            .expect("a run of whole rows of x has the batch's shape");
            let targets = match &data.targets {
                Targets::Classes(labels) => TargetsView::Classes(&labels[start..end]),
                Targets::Values(t) => {
                    let width = t.shape().dims()[1];
                    TargetsView::Values(
                        TensorView::from_slice(&t.data()[start * width..end * width], &[n, width])
                            .expect("a run of whole rows of the targets has the batch's shape"),
                    )
                }
            };
            let value = match engine {
                Engine::Compiled(compiled) => {
                    let out = compiled.infer_batch(x);
                    visit(start, out, targets)?;
                    if !with_loss {
                        continue;
                    }
                    let mut tape = graph.tape(no_params);
                    let out = tape.input(out);
                    let l = loss.record(&mut tape, out, targets);
                    tape.value(l).data()[0]
                }
                Engine::Tape(values) => {
                    let mut tape = graph.tape(values);
                    let input = tape.input(x);
                    let out = model.forward(&mut tape, input);
                    visit(start, tape.value(out), targets)?;
                    if !with_loss {
                        continue;
                    }
                    let l = loss.record(&mut tape, out, targets);
                    tape.value(l).data()[0]
                }
            };
            total += f64::from(value) * n as f64;
        }
        Ok(total / rows as f64)
    }
}

/// Measures `model` with the values in `params` on `data`, `batch` rows at a time: an
/// [`Evaluator`] made for one call. Keep an evaluator instead to measure again without
/// compiling again.
///
/// # Errors
///
/// Returns the errors of [`Evaluator::new`] and [`Evaluator::evaluate`].
///
/// # Panics
///
/// Panics as [`Evaluator::new`] and [`Evaluator::evaluate`] do.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{Dataset, Linear, Loss, Sequential, Targets, evaluate};
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
/// let data = Dataset { x: Tensor::zeros(&[5, 2]), targets: Targets::Classes(vec![0; 5]) };
/// let e = evaluate(&net, &params, &data, Loss::CrossEntropy, 2).unwrap();
/// assert_eq!(e.accuracy, Some(1.0));
/// assert!((e.loss - 3f64.ln()).abs() < 1e-6);
/// ```
pub fn evaluate(
    model: &Sequential,
    params: &ParamStore,
    data: &Dataset,
    loss: Loss,
    batch: usize,
) -> Result<Evaluation, Error> {
    Evaluator::new(model, params, loss, batch)?.evaluate(data)
}

/// Returns [`Error::InvalidData`], naming the index, if an entry of `linears` is not the
/// index of a `Linear` layer of `model`.
fn check_linears(model: &Sequential, linears: &[usize]) -> Result<(), Error> {
    match linears.iter().find(|&&l| model.linear(l).is_none()) {
        Some(linear) => Err(Error::InvalidData(format!(
            "evaluator: layer {linear} is not a Linear layer"
        ))),
        None => Ok(()),
    }
}

/// Returns [`Error::Unsupported`] if `model` has no layers.
fn check_layers(model: &Sequential) -> Result<(), Error> {
    if model.is_empty() {
        return Err(Error::Unsupported(
            "evaluating a model with no layers".to_string(),
        ));
    }
    Ok(())
}

/// Returns [`Error::InvalidData`], naming the row, the label and the width, if `label` is
/// not below the model's output width.
fn check_label(row: usize, label: usize, width: usize) -> Result<(), Error> {
    if label >= width {
        return Err(Error::InvalidData(format!(
            "the label of row {row} is {label}, but the model has {width} outputs (classes 0..{width})"
        )));
    }
    Ok(())
}

/// Returns [`Error::InvalidData`], naming both widths, if a batch's value targets are not as
/// wide as the model's output.
fn check_value_width(values: TensorView<'_, f32>, width: usize) -> Result<(), Error> {
    let dims = values.shape().dims();
    if dims[1] != width {
        return Err(Error::InvalidData(format!(
            "the value targets are {} wide, but the model outputs {width} values per row",
            dims[1]
        )));
    }
    Ok(())
}

/// Fills `means` with the mean of each of the `width` columns of `targets` (row-major) and
/// `ss_tot` with each column's sum of squared deviations from its mean, both in `f64`.
fn target_spread(targets: &[f32], width: usize, means: &mut Vec<f64>, ss_tot: &mut Vec<f64>) {
    means.clear();
    means.resize(width, 0.0);
    ss_tot.clear();
    ss_tot.resize(width, 0.0);
    if width == 0 {
        return;
    }
    let rows = targets.len() / width;
    for row in targets.chunks_exact(width) {
        for (m, &t) in means.iter_mut().zip(row) {
            *m += f64::from(t);
        }
    }
    for m in means.iter_mut() {
        *m /= rows as f64;
    }
    for row in targets.chunks_exact(width) {
        for ((s, &m), &t) in ss_tot.iter_mut().zip(means.iter()).zip(row) {
            let d = f64::from(t) - m;
            *s += d * d;
        }
    }
}

/// R² averaged over the outputs from each output's sum of squared errors `ss_res` and sum
/// of squared deviations `ss_tot`: `1 - ss_res / ss_tot`, or for an output whose targets do
/// not vary, 1 when it is predicted exactly and 0 otherwise. With no outputs, 1.
fn r2(ss_res: &[f64], ss_tot: &[f64]) -> f64 {
    if ss_res.is_empty() {
        return 1.0;
    }
    let sum: f64 = ss_res
        .iter()
        .zip(ss_tot)
        .map(|(&res, &tot)| {
            if tot != 0.0 {
                1.0 - res / tot
            } else if res == 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .sum();
    sum / ss_res.len() as f64
}

#[cfg(test)]
mod tests {
    //! Tests for the evaluator: the loss against the tape for every loss, R² by hand,
    //! label checks, the tape fallback, `misclassified` and both ways `refresh` works.

    use super::*;
    use crate::inference::tests::layer_bounds;
    use crate::{Activation, ActivationLayer, Linear, LogSoftmax, Sigmoid, Tanh};
    use cavitation_core::{Rng, Tensor};

    /// Every [`Loss`], with settings for the ones that take them.
    fn every_loss() -> [Loss; 9] {
        [
            Loss::CrossEntropy,
            Loss::Nll,
            Loss::Mse,
            Loss::L1,
            Loss::Huber { delta: 0.5 },
            Loss::SmoothL1 { beta: 0.5 },
            Loss::BinaryCrossEntropy,
            Loss::BinaryCrossEntropyWithLogits,
            Loss::KlDiv,
        ]
    }

    /// A 6 → 8 tanh → 4 network from `Rng::seed(seed)` and `rows` rows of data suited to
    /// `loss`: class labels for the class losses, values of the right kind otherwise. The
    /// network ends in what the loss expects: log-probabilities for `Nll` and `KlDiv`,
    /// probabilities for binary cross-entropy, raw scores otherwise.
    fn case(loss: Loss, rows: usize, seed: u64) -> (Sequential, ParamStore, Dataset) {
        let mut rng = Rng::seed(seed);
        let mut params = ParamStore::new();
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 6, 8, &mut rng))
            .push(Tanh)
            .push(Linear::new(&mut params, 8, 4, &mut rng));
        let x: Vec<f32> = (0..rows * 6).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let x = Tensor::from_vec(x, &[rows, 6]).unwrap();
        let mut values = |f: &mut dyn FnMut(&mut Rng) -> f32| {
            let v: Vec<f32> = (0..rows * 4).map(|_| f(&mut rng)).collect();
            Tensor::from_vec(v, &[rows, 4]).unwrap()
        };
        let targets = match loss {
            Loss::CrossEntropy | Loss::Nll => {
                Targets::Classes((0..rows).map(|i| (i * 7 + i / 3) % 4).collect())
            }
            Loss::Mse | Loss::L1 | Loss::Huber { .. } | Loss::SmoothL1 { .. } => {
                Targets::Values(values(&mut |r| r.range(-1.5, 1.5) as f32))
            }
            Loss::BinaryCrossEntropy | Loss::BinaryCrossEntropyWithLogits => {
                Targets::Values(values(&mut |r| r.uniform_f32()))
            }
            Loss::KlDiv => {
                let mut t = values(&mut |r| r.range(0.05, 1.0) as f32);
                for row in t.data_mut().as_chunks_mut::<4>().0 {
                    let sum: f32 = row.iter().sum();
                    row.iter_mut().for_each(|v| *v /= sum);
                }
                Targets::Values(t)
            }
        };
        match loss {
            Loss::Nll | Loss::KlDiv => net = net.push(LogSoftmax),
            Loss::BinaryCrossEntropy => net = net.push(Sigmoid),
            _ => {}
        }
        (net, params, Dataset { x, targets })
    }

    /// The loss of `net` on `data` recorded on the tape for each batch of `batch` rows in
    /// order, as `Trainer::step` records it, and weighted by each batch's row count.
    fn tape_loss(
        net: &Sequential,
        params: &ParamStore,
        data: &Dataset,
        loss: Loss,
        batch: usize,
    ) -> f64 {
        let (rows, features) = (data.len(), data.features());
        let mut graph = Graph::new();
        let mut total = 0.0;
        for start in (0..rows).step_by(batch) {
            let end = (start + batch).min(rows);
            let n = end - start;
            let x = TensorView::from_slice(
                &data.x.data()[start * features..end * features],
                &[n, features],
            )
            .unwrap();
            let targets = match &data.targets {
                Targets::Classes(l) => TargetsView::Classes(&l[start..end]),
                Targets::Values(t) => {
                    let w = t.shape().dims()[1];
                    TargetsView::Values(
                        TensorView::from_slice(&t.data()[start * w..end * w], &[n, w]).unwrap(),
                    )
                }
            };
            let mut tape = graph.tape(&params.values);
            let input = tape.input(x);
            let out = net.forward(&mut tape, input);
            let l = loss.record(&mut tape, out, targets);
            total += f64::from(tape.value(l).data()[0]) * n as f64;
        }
        total / rows as f64
    }

    /// For every loss, 300 rows at batch 128 (a last batch of 44): the evaluator's loss is
    /// the row-weighted mean of the tape's batch losses within 1e-6 relative.
    #[test]
    fn evaluate_loss_matches_tape_for_every_loss() {
        for (k, loss) in every_loss().into_iter().enumerate() {
            let (net, params, data) = case(loss, 300, 10 + k as u64);
            let want = tape_loss(&net, &params, &data, loss, 128);
            let got = evaluate(&net, &params, &data, loss, 128).unwrap();
            assert!(
                (got.loss - want).abs() <= 1e-6 * want.abs().max(f64::MIN_POSITIVE),
                "Loss::{loss:?}: evaluate {} against the tape's {want}",
                got.loss
            );
            let classes = matches!(data.targets, Targets::Classes(_));
            assert_eq!(got.accuracy.is_some(), classes, "Loss::{loss:?}");
            assert_eq!(got.r2.is_some(), !classes, "Loss::{loss:?}");
        }
    }

    /// A 2 → 2 `Linear` with weight `w` (`[outputs, inputs]`, row-major) and bias `b`.
    fn linear_2x2(w: [f32; 4], b: [f32; 2]) -> (Sequential, ParamStore) {
        let mut params = ParamStore::new();
        let layer = Linear::from_values(
            &mut params,
            Tensor::from_vec(w.to_vec(), &[2, 2]).unwrap(),
            Tensor::from_vec(b.to_vec(), &[2]).unwrap(),
        )
        .unwrap();
        (Sequential::new().push(layer), params)
    }

    /// The R² of a 2 → 2 `Linear` (`w`, `b`) on inputs `x` against targets `y`, both
    /// `[4, 2]`.
    fn r2_of(w: [f32; 4], b: [f32; 2], x: [f32; 8], y: [f32; 8]) -> f64 {
        let (net, params) = linear_2x2(w, b);
        let data = Dataset {
            x: Tensor::from_vec(x.to_vec(), &[4, 2]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(y.to_vec(), &[4, 2]).unwrap()),
        };
        evaluate(&net, &params, &data, Loss::Mse, 3)
            .unwrap()
            .r2
            .unwrap()
    }

    /// R² by hand: perfect predictions give 1; predicting each output's mean gives 0; one
    /// perfect output and one at its mean average to 0.5; errors of 1 against a spread of 2
    /// give 0.5; an output whose targets do not vary gives 1 when predicted exactly and 0
    /// otherwise.
    #[test]
    fn r2_by_hand() {
        let x = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 6.0, 60.0];
        let identity = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(r2_of(identity, [0.0, 0.0], x, x), 1.0);
        // The column means are 3 and 30.
        assert_eq!(r2_of([0.0; 4], [3.0, 30.0], x, x), 0.0);
        assert_eq!(r2_of([1.0, 0.0, 0.0, 0.0], [0.0, 30.0], x, x), 0.5);
        // Output 0 predicts 1, 2, 3, 6 for targets 1, 2, 4, 5: errors 0, 0, 1, 1 against
        // deviations of 2, 1, 1, 2 from the mean of 3: R² = 1 - 2 / 10 = 0.8. Output 1 is
        // exact. The average is 0.9.
        let y = [1.0, 10.0, 2.0, 20.0, 4.0, 30.0, 5.0, 60.0];
        assert!((r2_of(identity, [0.0, 0.0], x, y) - 0.9).abs() < 1e-12);
        // Output 1's targets are all 7: predicted exactly (weight 0, bias 7) it scores 1,
        // predicted as 8 it scores 0; output 0 is exact both times.
        let flat = [1.0, 7.0, 2.0, 7.0, 3.0, 7.0, 6.0, 7.0];
        assert_eq!(r2_of([1.0, 0.0, 0.0, 0.0], [0.0, 7.0], x, flat), 1.0);
        assert_eq!(r2_of([1.0, 0.0, 0.0, 0.0], [0.0, 8.0], x, flat), 0.5);
        assert_eq!(r2(&[0.0, 1.0], &[0.0, 0.0]), 0.5);
        assert_eq!(r2(&[1.0], &[2.0]), 0.5);
    }

    /// A label not below the output width is `InvalidData` naming the row, the label and
    /// the width, from `evaluate`, `misclassified` and the per-batch check alike.
    #[test]
    fn label_out_of_range_is_invalid_data() {
        let (net, params, mut data) = case(Loss::CrossEntropy, 20, 3);
        if let Targets::Classes(labels) = &mut data.targets {
            labels[13] = 4;
        }
        let mut evaluator = Evaluator::new(&net, &params, Loss::CrossEntropy, 8).unwrap();
        let mut out = vec![99];
        let errors = [
            evaluator.evaluate(&data).unwrap_err(),
            evaluator.misclassified(&data, &mut out).unwrap_err(),
            check_label(13, 4, 4).unwrap_err(),
        ];
        for error in errors {
            match error {
                Error::InvalidData(msg) => {
                    assert!(msg.contains("row 13 is 4"), "{msg}");
                    assert!(msg.contains("4 outputs"), "{msg}");
                }
                other => panic!("expected InvalidData, got {other:?}"),
            }
        }
        assert!(out.is_empty());
    }

    /// Value targets of the wrong kind or width, and empty data, are `InvalidData` too.
    #[test]
    fn unsuitable_data_is_invalid_data() {
        let (net, params, data) = case(Loss::Mse, 10, 4);
        let mut evaluator = Evaluator::new(&net, &params, Loss::CrossEntropy, 4).unwrap();
        assert!(matches!(
            evaluator.evaluate(&data),
            Err(Error::InvalidData(_))
        ));
        assert!(matches!(
            evaluator.misclassified(&data, &mut Vec::new()),
            Err(Error::InvalidData(_))
        ));
        let empty = data.subset(0..0);
        assert!(matches!(
            evaluate(&net, &params, &empty, Loss::Mse, 4),
            Err(Error::InvalidData(_))
        ));
        assert!(matches!(
            Evaluator::new(&net, &params, Loss::Mse, 0),
            Err(Error::InvalidData(_))
        ));
        assert!(matches!(
            Evaluator::new(&Sequential::new(), &params, Loss::Mse, 4),
            Err(Error::Unsupported(_))
        ));
    }

    /// `tanh` written as an activation of your own, so it has no inference form.
    struct MyTanh;

    impl Activation<f32> for MyTanh {
        /// Returns `"my_tanh"`.
        fn name(&self) -> &str {
            "my_tanh"
        }
        /// `tanh(x)`.
        fn f(&self, x: f32) -> f32 {
            x.tanh()
        }
        /// `1 - tanh(x)²`, from the output.
        fn df(&self, _x: f32, y: f32) -> f32 {
            1.0 - y * y
        }
    }

    /// The rows of `data` whose two largest outputs of `net` on the tape lie within twice
    /// the inference bound of each other ([`layer_bounds`], the matvec and GEMM bound
    /// carried through the layers), so that the compiled path may rank them the other way.
    fn near_ties(net: &Sequential, params: &ParamStore, data: &Dataset) -> Vec<bool> {
        let (out, bound) = layer_bounds(net, params, data.x.data(), data.len())
            .pop()
            .unwrap();
        let width = out.len() / data.len();
        out.chunks(width)
            .zip(bound.chunks(width))
            .map(|(row, b)| {
                let mut order: Vec<usize> = (0..width).collect();
                order.sort_by(|&i, &j| row[j].total_cmp(&row[i]));
                let (top, next) = (order[0], order[1]);
                f64::from(row[top] - row[next]) <= 2.0 * (b[top] + b[next])
            })
            .collect()
    }

    /// A model with an activation of your own runs on the tape and evaluates as the same
    /// model with the built-in `Tanh` does compiled: the loss and R² within 1e-6. The tape
    /// gives the same outputs on both models, and the compiled path stays within the
    /// matvec and GEMM bound of them, so the argmax can differ only on a row whose two
    /// largest outputs are within that bound of each other: such near ties are left out of
    /// the comparison of `misclassified`, and the accuracies may differ by at most their
    /// share of the rows.
    #[test]
    fn tape_fallback_matches_compiled() {
        for loss in [Loss::CrossEntropy, Loss::Mse] {
            let mut rng = Rng::seed(21);
            let mut params = ParamStore::new();
            let a = Linear::new(&mut params, 6, 8, &mut rng);
            let b = Linear::new(&mut params, 8, 4, &mut rng);
            let builtin = Sequential::new().push(a.clone()).push(Tanh).push(b.clone());
            let own = Sequential::new()
                .push(a)
                .push(ActivationLayer::new(MyTanh))
                .push(b);
            let (_, _, data) = case(loss, 300, 22);
            let mut compiled = Evaluator::new(&builtin, &params, loss, 128).unwrap();
            let mut tape = Evaluator::new(&own, &params, loss, 128).unwrap();
            assert!(matches!(compiled.engine, Engine::Compiled(_)));
            assert!(matches!(tape.engine, Engine::Tape(_)));
            let (c, t) = (
                compiled.evaluate(&data).unwrap(),
                tape.evaluate(&data).unwrap(),
            );
            let close = |x: f64, y: f64| (x - y).abs() <= 1e-6 * x.abs().max(1.0);
            assert!(close(c.loss, t.loss), "{loss:?}: {c:?} against {t:?}");
            match (c.accuracy, t.accuracy, c.r2, t.r2) {
                (Some(x), Some(y), None, None) => {
                    let ties = near_ties(&builtin, &params, &data);
                    let n = ties.iter().filter(|&&t| t).count();
                    assert!(
                        (x - y).abs() <= n as f64 / 300.0 + 1e-12,
                        "{c:?} against {t:?}"
                    );
                    let (mut x, mut y) = (Vec::new(), Vec::new());
                    compiled.misclassified(&data, &mut x).unwrap();
                    tape.misclassified(&data, &mut y).unwrap();
                    x.retain(|&r| !ties[r]);
                    y.retain(|&r| !ties[r]);
                    assert_eq!(x, y, "{n} near ties left out");
                }
                (None, None, Some(x), Some(y)) => assert!(close(x, y), "{c:?} against {t:?}"),
                _ => panic!("{loss:?}: {c:?} against {t:?}"),
            }
        }
    }

    /// The same 6 → 8 → 4 network as `builtin` (built-in `Tanh`) and as `own` (`MyTanh`,
    /// so it runs on the tape), with 300 rows of class data.
    fn both_engines() -> (Sequential, Sequential, ParamStore, Dataset) {
        let (builtin, params, data) = case(Loss::CrossEntropy, 300, 40);
        let own = Sequential::new()
            .push(builtin.linear(0).unwrap().clone())
            .push(ActivationLayer::new(MyTanh))
            .push(builtin.linear(2).unwrap().clone());
        (builtin, own, params, data)
    }

    /// On each engine, an evaluator made with `with_neuron_masks` and switched with
    /// `set_neuron_mask` gives exactly the `evaluate` of the network with that neuron
    /// masked through `Sequential::set_neuron`, for every hidden neuron, then with all of
    /// them on again; the caller's network keeps no mask, and a clone keeps its own masks.
    #[test]
    fn neuron_masked_evaluator_matches_masked_evaluate() {
        let (builtin, own, params, data) = both_engines();
        let loss = Loss::CrossEntropy;
        for (net, compiled) in [(&builtin, true), (&own, false)] {
            let mut evaluator = Evaluator::with_neuron_masks(net, &params, loss, 64, &[0]).unwrap();
            assert_eq!(matches!(evaluator.engine, Engine::Compiled(_)), compiled);
            let plain = evaluate(net, &params, &data, loss, 64).unwrap();
            assert_eq!(evaluator.evaluate(&data).unwrap(), plain);
            for neuron in 0..8 {
                evaluator
                    .set_neuron_mask(0, &NeuronMask::from_off(8, &[neuron]))
                    .unwrap();
                let mut masked = net.clone();
                masked.set_neuron(0, neuron, false);
                let want = evaluate(&masked, &params, &data, loss, 64).unwrap();
                let got = evaluator.evaluate(&data).unwrap();
                assert_eq!(got, want, "compiled {compiled}, neuron {neuron}");
                assert_ne!(got, plain, "compiled {compiled}, neuron {neuron}");
            }
            assert!(net.mask(1).is_none());
            let mut copy = evaluator.clone();
            copy.set_neuron_mask(0, &NeuronMask::all_on(8)).unwrap();
            assert_eq!(copy.evaluate(&data).unwrap(), plain);
            assert_ne!(evaluator.evaluate(&data).unwrap(), plain);
        }
    }

    /// `refresh` keeps the masks `set_neuron_mask` gave, on either engine and when the
    /// evaluator moves from one to the other; a neuron mask on the refreshed network
    /// replaces them.
    #[test]
    fn refresh_keeps_neuron_masks() {
        let (builtin, own, mut params, data) = both_engines();
        let loss = Loss::CrossEntropy;
        let off = NeuronMask::from_off(8, &[1, 6]);
        let mut evaluator =
            Evaluator::with_neuron_masks(&builtin, &params, loss, 64, &[0]).unwrap();
        evaluator.set_neuron_mask(0, &off).unwrap();
        let w = builtin.param("0.weight").unwrap();
        let want = |net: &Sequential, params: &ParamStore, mask: &NeuronMask| {
            let mut masked = net.clone();
            masked.set_neuron_mask(0, mask.clone());
            evaluate(&masked, params, &data, loss, 64).unwrap()
        };
        params.value_mut(w).data_mut()[0] += 0.5;
        assert_eq!(
            evaluator.reload(&builtin, &params).unwrap(),
            Reload::InPlace
        );
        assert_eq!(
            evaluator.evaluate(&data).unwrap(),
            want(&builtin, &params, &off)
        );
        assert_eq!(evaluator.reload(&own, &params).unwrap(), Reload::Tape);
        assert_eq!(
            evaluator.evaluate(&data).unwrap(),
            want(&own, &params, &off)
        );
        params.value_mut(w).data_mut()[1] -= 0.5;
        assert_eq!(evaluator.reload(&own, &params).unwrap(), Reload::Tape);
        assert_eq!(
            evaluator.evaluate(&data).unwrap(),
            want(&own, &params, &off)
        );
        assert_eq!(
            evaluator.reload(&builtin, &params).unwrap(),
            Reload::Recompiled
        );
        assert_eq!(
            evaluator.evaluate(&data).unwrap(),
            want(&builtin, &params, &off)
        );
        let mut masked = builtin.clone();
        let other = NeuronMask::from_off(8, &[3]);
        masked.set_neuron_mask(0, other.clone());
        evaluator.refresh(&masked, &params).unwrap();
        assert_eq!(
            evaluator.evaluate(&data).unwrap(),
            want(&builtin, &params, &other)
        );
    }

    /// When a refresh adds a `Sigmoid` after the `Tanh`, the listed `Linear`'s neurons move
    /// from layer 1 to layer 2: the mask moves with its values, the model is compiled
    /// afresh (no step is left masking the `Tanh`), the evaluation equals `evaluate` on the
    /// new network with that neuron mask, and `set_neuron_mask` keeps working.
    #[test]
    fn refresh_moves_a_neuron_mask_with_its_layer() {
        let (net, params, data) = case(Loss::CrossEntropy, 200, 41);
        let loss = Loss::CrossEntropy;
        let off = NeuronMask::from_off(8, &[0, 5]);
        let mut evaluator = Evaluator::with_neuron_masks(&net, &params, loss, 64, &[0]).unwrap();
        evaluator.set_neuron_mask(0, &off).unwrap();
        let longer = Sequential::new()
            .push(net.linear(0).unwrap().clone())
            .push(Tanh)
            .push(Sigmoid)
            .push(net.linear(2).unwrap().clone());
        let want = |mask: &NeuronMask| {
            let mut masked = longer.clone();
            masked.set_neuron_mask(0, mask.clone());
            evaluate(&masked, &params, &data, loss, 64).unwrap()
        };
        assert_eq!(
            evaluator.reload(&longer, &params).unwrap(),
            Reload::Recompiled
        );
        assert!(matches!(evaluator.engine, Engine::Compiled(_)));
        assert_eq!(evaluator.evaluate(&data).unwrap(), want(&off));
        let other = NeuronMask::from_off(8, &[3]);
        evaluator.set_neuron_mask(0, &other).unwrap();
        assert_eq!(evaluator.evaluate(&data).unwrap(), want(&other));
        assert_eq!(evaluator.model.neuron_mask(0), Some(&other));
    }

    /// `with_neuron_masks` with an index that is not a `Linear` is `InvalidData`; on each
    /// engine `set_neuron_mask` on a layer that is not a `Linear` or with the wrong width is
    /// `InvalidData`, and on a `Linear` that was not listed `Unsupported`, each naming the
    /// layer, and the evaluation is unchanged.
    #[test]
    fn set_neuron_mask_errors_on_each_engine() {
        let (builtin, own, params, data) = both_engines();
        let loss = Loss::CrossEntropy;
        for net in [&builtin, &own] {
            match Evaluator::with_neuron_masks(net, &params, loss, 64, &[1]) {
                Err(Error::InvalidData(msg)) => assert!(msg.contains("layer 1"), "{msg}"),
                other => panic!("expected InvalidData, got {other:?}"),
            }
            let mut evaluator = Evaluator::with_neuron_masks(net, &params, loss, 64, &[0]).unwrap();
            let before = evaluator.evaluate(&data).unwrap();
            let cases: [(usize, NeuronMask); 3] = [
                (1, NeuronMask::all_off(8)),
                (0, NeuronMask::all_off(7)),
                (2, NeuronMask::all_off(4)),
            ];
            for (linear, mask) in cases {
                let got = evaluator.set_neuron_mask(linear, &mask);
                let msg = match (&got, linear) {
                    (Err(Error::Unsupported(msg)), 2) | (Err(Error::InvalidData(msg)), 0 | 1) => {
                        msg
                    }
                    _ => panic!("linear {linear}: {got:?}"),
                };
                assert!(msg.contains(&format!("layer {linear}")), "{msg}");
            }
            assert_eq!(evaluator.evaluate(&data).unwrap(), before);
        }
    }

    /// `misclassified` lists, in order, exactly the rows whose first largest output is not
    /// at the label, and agrees with the accuracy.
    #[test]
    fn misclassified_lists_the_wrong_rows() {
        let (net, params, data) = case(Loss::CrossEntropy, 300, 5);
        let mut evaluator = Evaluator::new(&net, &params, Loss::CrossEntropy, 128).unwrap();
        let mut wrong = Vec::new();
        evaluator.misclassified(&data, &mut wrong).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let input = tape.input(data.x.view());
        let out = net.forward(&mut tape, input);
        let Targets::Classes(labels) = &data.targets else {
            unreachable!()
        };
        let want: Vec<usize> = tape
            .value(out)
            .data()
            .chunks(4)
            .zip(labels)
            .enumerate()
            .filter(|(_, (row, l))| argmax(row) != Some(**l))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(wrong, want);
        assert!(!wrong.is_empty() && wrong.len() < 300);
        let accuracy = evaluator.evaluate(&data).unwrap().accuracy.unwrap();
        assert_eq!(accuracy, (300 - wrong.len()) as f64 / 300.0);
    }

    /// `refresh` copies new weights and mask values in place when no clone shares the
    /// compiled weights and the layout is unchanged; it recompiles when a clone shares them
    /// (the clone keeps the old weights, and its own masks when this evaluator's change),
    /// when a mask is added or cleared, or when the last layer gets wider; a model without
    /// an inference form moves to the tape and back. Each time the evaluation matches a
    /// fresh evaluator's.
    #[test]
    fn refresh_in_place_and_recompiled() {
        let loss = Loss::CrossEntropy;
        let (mut net, mut params, data) = case(loss, 200, 6);
        let mut evaluator = Evaluator::new(&net, &params, loss, 64).unwrap();
        let first = evaluator.evaluate(&data).unwrap();
        let w = net.param("0.weight").unwrap();
        let fresh =
            |net: &Sequential, params: &ParamStore| evaluate(net, params, &data, loss, 64).unwrap();

        params.value_mut(w).data_mut()[0] += 1.0;
        assert_eq!(evaluator.reload(&net, &params).unwrap(), Reload::InPlace);
        let second = evaluator.evaluate(&data).unwrap();
        assert_eq!(second, fresh(&net, &params));
        assert_ne!(second, first);

        let mut copy = evaluator.clone();
        let shares = |a: &Evaluator, b: &Evaluator| match (&a.engine, &b.engine) {
            (Engine::Compiled(x), Engine::Compiled(y)) => x.shares_steps(y),
            _ => false,
        };
        assert!(shares(&evaluator, &copy));
        params.value_mut(w).data_mut()[1] -= 2.0;
        assert_eq!(evaluator.reload(&net, &params).unwrap(), Reload::Recompiled);
        assert!(!shares(&evaluator, &copy));
        assert_eq!(evaluator.evaluate(&data).unwrap(), fresh(&net, &params));
        assert_eq!(copy.evaluate(&data).unwrap(), second);

        let mut mask = NeuronMask::all_on(8);
        mask.set(2, false);
        net.set_mask(1, mask.clone());
        assert_eq!(evaluator.reload(&net, &params).unwrap(), Reload::Recompiled);
        assert_eq!(evaluator.evaluate(&data).unwrap(), fresh(&net, &params));
        mask.set(5, false);
        net.set_mask(1, mask.clone());
        assert_eq!(evaluator.reload(&net, &params).unwrap(), Reload::InPlace);
        let masked = evaluator.evaluate(&data).unwrap();
        assert_eq!(masked, fresh(&net, &params));

        // A clone shares the weights but not the masks: new mask values reach this
        // evaluator (its weights recompiled) and leave the clone as it was.
        let mut copy = evaluator.clone();
        mask.set(0, false);
        net.set_mask(1, mask);
        assert_eq!(evaluator.reload(&net, &params).unwrap(), Reload::Recompiled);
        assert_eq!(evaluator.evaluate(&data).unwrap(), fresh(&net, &params));
        assert_ne!(evaluator.evaluate(&data).unwrap(), masked);
        assert_eq!(copy.evaluate(&data).unwrap(), masked);
        net.clear_mask(1);
        assert_eq!(evaluator.reload(&net, &params).unwrap(), Reload::Recompiled);
        assert_eq!(evaluator.evaluate(&data).unwrap(), fresh(&net, &params));

        let wide_last = Linear::new(&mut params, 8, 6, &mut Rng::seed(8));
        let wide = Sequential::new()
            .push(net.linear(0).unwrap().clone())
            .push(Tanh)
            .push(wide_last);
        assert_eq!(
            evaluator.reload(&wide, &params).unwrap(),
            Reload::Recompiled
        );
        let widened = evaluator.evaluate(&data).unwrap();
        assert_eq!(widened, fresh(&wide, &params));
        assert_ne!(widened, fresh(&net, &params));
        assert_eq!(evaluator.reload(&net, &params).unwrap(), Reload::Recompiled);
        assert_eq!(evaluator.evaluate(&data).unwrap(), fresh(&net, &params));

        let mut rng = Rng::seed(7);
        let mut other = ParamStore::new();
        let own = Sequential::new()
            .push(Linear::new(&mut other, 6, 8, &mut rng))
            .push(ActivationLayer::new(MyTanh))
            .push(Linear::new(&mut other, 8, 4, &mut rng));
        assert_eq!(evaluator.reload(&own, &other).unwrap(), Reload::Tape);
        assert_eq!(evaluator.evaluate(&data).unwrap(), fresh(&own, &other));
        assert_eq!(evaluator.reload(&net, &params).unwrap(), Reload::Recompiled);
        assert_eq!(evaluator.evaluate(&data).unwrap(), fresh(&net, &params));
        assert!(matches!(
            evaluator.refresh(&Sequential::new(), &params),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(evaluator.evaluate(&data).unwrap(), fresh(&net, &params));
    }
}
