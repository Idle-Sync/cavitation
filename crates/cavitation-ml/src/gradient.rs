//! The gradient path of linear and logistic regression: [`LinearRegression::fit_gradient`]
//! and [`LogisticRegression::fit_gradient`](crate::LogisticRegression::fit_gradient) train
//! one [`Linear`] layer by gradient descent (full-batch by default, or in minibatches) on the
//! same objective their exact paths solve, through one shared loop, and [`GradientConfig`]
//! says how.

use std::time::Instant;

use cavitation_core::{Error, Graph, ParamStore, Reduction, Rng};
use cavitation_nn::{
    Batch, Batcher, Dataset, EpochReport, Health, HealthMonitor, History, Linear, Loss, LrSchedule,
    OptimizerConfig, Sequential, StepInputs, TargetsView, Trace,
};

use crate::linear::{LinearModel, LinearRegression, Shape, check_alpha, check_values, targets};

/// How [`LinearRegression::fit_gradient`] and
/// [`LogisticRegression::fit_gradient`](crate::LogisticRegression::fit_gradient) train.
/// Taken by value, because it holds the health checks' sink.
///
/// The default is 1,000 epochs of full-batch gradient descent (every row in every step),
/// Adam at a learning rate of 0.5 without weight decay, cosine annealing towards 0, rows
/// shuffled with seed 0, weights initialised with seed 0, and the default [`Health`]
/// checks. On the Concrete training rows with standardised inputs (824 rows, 8 inputs) it
/// reaches the exact solution within `1e-3 · max(1, |exact|)` of every coefficient and
/// intercept, with and without a ridge penalty; the gap measured there is below 1e-5. On
/// the Breast Cancer training rows with standardised inputs (455 rows, 30 inputs), logistic
/// regression with `c = Some(1.0)` reaches its exact solution within about 5e-7.
///
/// Why full batches: the objective is convex, and the full gradient points at its one
/// minimum, so the iterates settle on it. A minibatch gradient carries the noise of the rows
/// it happened to draw; near the minimum that noise is as large as the gradient itself, and
/// the last iterate wanders with it. On the same Concrete rows, minibatches of 32 rows with
/// Adam stayed about 2e-3 away from the exact solution even after 3,000 to 5,000 epochs
/// (several seeds), where full batches reach 1e-5 in 1,000.
///
/// These defaults assume standardised inputs **and** value targets: standardise both with
/// [`Standardizer`](cavitation_nn::Standardizer) before training and fold the scaling back
/// in afterwards with `fold_into_input` and `fold_into_output` (the example on
/// [`LinearRegression::fit_gradient`] shows how). Adam moves each weight by about one
/// learning rate per step, so over 1,000 cosine-annealed epochs from 0.5 a weight travels
/// at most about 250 from where it starts: coefficients much larger than about
/// learning rate × epochs cannot be reached, and the result can be far from the exact
/// answer with no error returned. On the Concrete rows with standardised inputs but targets
/// multiplied by 100 (coefficients near 1,260) the defaults end about 2.3 away from the exact
/// solution, relative to `max(1, |exact|)`; unscaled inputs end as far off and make the
/// conditioning worse as well.
///
/// # Examples
///
/// ```
/// use cavitation_ml::GradientConfig;
/// let config = GradientConfig { batch: 32, epochs: 3000, ..GradientConfig::default() };
/// assert_eq!((config.epochs, config.batch), (3000, 32));
/// assert_eq!(GradientConfig::default().batch, usize::MAX);
/// ```
#[derive(Debug)]
pub struct GradientConfig {
    /// How many passes over the data.
    pub epochs: usize,
    /// The rows per step; the last batch of an epoch may be smaller. A batch at least as
    /// large as the data set, such as the default `usize::MAX`, is the whole data set in one
    /// step: full-batch gradient descent.
    pub batch: usize,
    /// The optimiser to build. Its weight decay, if any, applies to the intercept as well,
    /// on top of the regression's own `alpha`; leave it at 0 to minimise the regression's
    /// objective.
    pub optimizer: OptimizerConfig,
    /// How the learning rate changes from epoch to epoch, starting from the optimiser's.
    pub schedule: LrSchedule,
    /// The seed that shuffles the rows each epoch; `None` keeps the data's order. With a
    /// full batch, shuffling changes only the order of the sums.
    pub shuffle_seed: Option<u64>,
    /// The seed of the weights' initialisation (Xavier uniform, as [`Linear::new`]).
    pub init_seed: u64,
    /// The health checks run on every step and at the end of every epoch. The default turns
    /// the vanishing-gradient check off (`grad_norm_low: 0.0`, since no norm is below 0):
    /// least squares is convex, so a tiny gradient here means the fit has converged, and on
    /// noisy data the loss stops at the residual variance instead of falling the 10,000-fold
    /// that would otherwise silence the check. The non-finite and exploding checks stay on.
    pub health: Health,
}

impl Default for GradientConfig {
    /// 1,000 epochs, every row in one batch (`usize::MAX`), Adam at 0.5 without weight
    /// decay, cosine annealing to 0, shuffle seed 0, initialisation seed 0, and the default
    /// health checks with the vanishing-gradient check off.
    fn default() -> GradientConfig {
        GradientConfig {
            epochs: 1000,
            batch: usize::MAX,
            optimizer: OptimizerConfig::Adam {
                lr: 0.5,
                weight_decay: 0.0,
            },
            schedule: LrSchedule::Cosine { min_lr: 0.0 },
            shuffle_seed: Some(0),
            init_seed: 0,
            health: Health {
                grad_norm_low: 0.0,
                ..Health::default()
            },
        }
    }
}

impl LinearRegression {
    /// Fits the regression by gradient descent: trains one [`Linear`] layer (weights
    /// initialised with `config.init_seed`, bias starting at each target column's mean) on
    /// `data` and returns it as a [`LinearModel`] with each epoch's report.
    ///
    /// Each step minimises, on a batch of `b` rows (`b = n` for a full batch),
    ///
    /// ```text
    /// L = 1/(b·k) · Σ_batch ‖y − x·Wᵀ − c‖²  +  alpha/(n·k) · ‖W‖²
    /// ```
    ///
    /// the mean squared error over the batch's `b·k` values (`k` outputs, `n` rows in all)
    /// plus the ridge penalty on the weight only, both recorded on the tape. Its minimiser is
    /// the exact path's. An epoch splits the shuffled rows into `B` batches, and a row lands
    /// in batch `j` (of `b_j` rows) with probability `b_j / n`, where its squared error is
    /// weighted `1/(b_j·k)`; so each row's expected weight in an epoch's summed gradient is
    /// `Σ_j (b_j/n) · 1/(b_j·k) = B/(n·k)`, the same for every row even when the last batch
    /// is short. The penalty enters once per step, `B` times per epoch, with weight
    /// `B·alpha/(n·k)`. The expected gradient of an epoch is therefore
    /// `B/(n·k) · ∇(‖y − X·Wᵀ − c‖² + alpha·‖W‖²)`: a positive multiple of the gradient of
    /// scikit-learn's ridge objective, with the same minimiser. With a full batch (`B = 1`,
    /// `b = n`) shuffling has no effect: the argument holds for every epoch, not only in
    /// expectation, and each step's objective is exactly `1/(n·k)` times scikit-learn's.
    /// Without the `1/(n·k)` the penalty would be `n·k` times too strong next to a mean. With
    /// minibatches and `shuffle_seed: None`, a short last batch holds the same rows every
    /// epoch and weights them more, so the iterates are not centred on that minimiser.
    ///
    /// **Standardise first.** The defaults of [`GradientConfig`] assume standardised inputs
    /// and value targets. Standardise both with [`Standardizer`](cavitation_nn::Standardizer),
    /// train, then fold the scaling back into the layer with `fold_into_input` and
    /// `fold_into_output`, as the second example does. Adam moves each weight by about one
    /// learning rate per step, so with the defaults a coefficient much larger than about
    /// learning rate × epochs (about 250) cannot be reached: the call returns `Ok` with a
    /// model that can be far from the exact one. On the Concrete rows with standardised
    /// inputs but targets multiplied by 100, the defaults end about 2.3 away (relative to
    /// `max(1, |exact|)`). [`fit`](Self::fit) is the reference to compare with.
    ///
    /// Without `fit_intercept` the bias is held at 0: set to 0 at the start and after every
    /// step. Every step runs the [`HealthMonitor`] checks of `config.health`, as
    /// [`fit`](cavitation_nn::fit) does. The same seeds give the same model on every run on
    /// the same machine and kernel level, for any thread count.
    ///
    /// The answer is as close to the exact one as the training gets it: the exact path is
    /// faster and exact, and this path is here to show that gradient descent finds the same
    /// model, and as the start of models that have no exact solution.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] for the same data and `alpha` problems as
    /// [`fit`](Self::fit) checks before solving (not the rank checks: a rank-deficient
    /// problem has many minimisers, and gradient descent finds one of them), and if
    /// `config.batch` is 0. During training, with `config.health.on_non_finite` set to
    /// [`OnNonFinite::Stop`](cavitation_nn::OnNonFinite) (the default), [`Error::NonFinite`]
    /// at the first step whose loss or gradient is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::{GradientConfig, LinearRegression};
    /// use cavitation_nn::{Dataset, Targets};
    /// let x: Vec<f32> = (0..64).map(|i| i as f32 / 32.0 - 1.0).collect();
    /// let y: Vec<f32> = x.iter().map(|v| 3.0 * v - 1.0).collect();
    /// let data = Dataset {
    ///     x: Tensor::from_vec(x, &[64, 1]).unwrap(),
    ///     targets: Targets::Values(Tensor::from_vec(y, &[64, 1]).unwrap()),
    /// };
    /// let config = GradientConfig::default();
    /// let (model, history) = LinearRegression::default().fit_gradient(&data, config).unwrap();
    /// assert!((model.coef.data()[0] - 3.0).abs() < 1e-3);
    /// assert!((model.intercept[0] + 1.0).abs() < 1e-3);
    /// assert_eq!(history.epochs.len(), 1000);
    /// ```
    ///
    /// Inputs in the hundreds and targets in the tens of thousands: standardise both, train,
    /// fold the scaling back in, and compare with the exact path.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_ml::{GradientConfig, LinearModel, LinearRegression};
    /// use cavitation_nn::{Dataset, Standardizer, Targets};
    /// let mut rng = Rng::seed(1);
    /// let x: Vec<f32> = (0..200).map(|_| 500.0 + 100.0 * rng.normal() as f32).collect();
    /// let y: Vec<f32> = x.chunks(2).map(|r| 40.0 * r[0] - 7.0 * r[1] + 2000.0).collect();
    /// let raw = Dataset {
    ///     x: Tensor::from_vec(x, &[100, 2]).unwrap(),
    ///     targets: Targets::Values(Tensor::from_vec(y, &[100, 1]).unwrap()),
    /// };
    /// let exact = LinearRegression::default().fit(&raw).unwrap();
    ///
    /// let Targets::Values(t) = &raw.targets else { unreachable!() };
    /// let (inputs, outputs) = (Standardizer::fit(raw.x.view()), Standardizer::fit(t.view()));
    /// let mut data = raw.clone();
    /// inputs.transform_inputs(&mut data);
    /// outputs.transform_targets(&mut data).unwrap();
    /// let regression = LinearRegression::default();
    /// let (scaled, _) = regression.fit_gradient(&data, GradientConfig::default()).unwrap();
    ///
    /// let mut params = ParamStore::new();
    /// let net = scaled.to_sequential(&mut params);
    /// inputs.fold_into_input(&net, &mut params).unwrap();
    /// outputs.fold_into_output(&net, &mut params).unwrap();
    /// let model = LinearModel::from_sequential(&net, &params).unwrap();
    /// let ours = model.coef.data().iter().chain(&model.intercept);
    /// let theirs = exact.coef.data().iter().chain(&exact.intercept);
    /// for (g, e) in ours.zip(theirs) {
    ///     assert!((g - e).abs() <= 1e-3 * e.abs().max(1.0), "{g} against {e}");
    /// }
    /// ```
    pub fn fit_gradient(
        &self,
        data: &Dataset,
        config: GradientConfig,
    ) -> Result<(LinearModel, History), Error> {
        check_alpha(self.alpha)?;
        let Shape {
            rows,
            features,
            outputs,
        } = check_values(data)?;
        let bias = self.fit_intercept.then(|| {
            let y = targets(data).data();
            (0..outputs)
                .map(|q| {
                    let sum: f64 = y
                        .iter()
                        .skip(q)
                        .step_by(outputs)
                        .map(|&v| f64::from(v))
                        .sum();
                    (sum / rows as f64) as f32
                })
                .collect()
        });
        let penalty = (self.alpha / (rows as f64 * outputs as f64)) as f32;
        let setup = Setup {
            features,
            outputs,
            loss: StepLoss::Mse,
            penalty,
            fit_intercept: self.fit_intercept,
            bias,
        };
        let (model, params, history) = train(data, config, setup)?;
        Ok((LinearModel::from_sequential(&model, &params)?, history))
    }
}

/// The loss each step of [`train`] minimises, before the penalty on the weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StepLoss {
    /// The mean squared error against value targets, as [`Loss::Mse`].
    Mse,
    /// Binary cross-entropy on the logits against value targets of 0 and 1, as
    /// [`Loss::BinaryCrossEntropyWithLogits`].
    BceWithLogits,
    /// Softmax cross-entropy on the logits against class labels, as [`Loss::CrossEntropy`].
    CrossEntropy,
}

/// The layer [`train`] trains and the objective it trains it on.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Setup {
    /// The layer's input columns.
    pub(crate) features: usize,
    /// The layer's outputs.
    pub(crate) outputs: usize,
    /// The data term of each step's loss.
    pub(crate) loss: StepLoss,
    /// The factor on `‖W‖²` added to each step's loss (0 for none); the bias is never
    /// penalised.
    pub(crate) penalty: f32,
    /// Whether the bias is trained; without it the bias is held at 0.
    pub(crate) fit_intercept: bool,
    /// The bias to start from, one value per output; `None` keeps the layer's zeros.
    pub(crate) bias: Option<Vec<f32>>,
}

/// The gradient loop both regressions share: trains one [`Linear`] layer (weights
/// initialised with `config.init_seed`, bias from `setup.bias`) on `data` by
/// `config.optimizer` in batches of `config.batch` rows, minimising per step the mean of
/// `setup.loss` over the batch plus `setup.penalty · ‖W‖²`, with the [`HealthMonitor`]
/// checks of `config.health` on every step. Returns the network, its parameters and each
/// epoch's report (the row-weighted mean of the step losses, penalty included).
///
/// `data`'s targets must suit `setup.loss`: values for [`StepLoss::Mse`] and
/// [`StepLoss::BceWithLogits`] (`[rows, outputs]`), class labels below `setup.outputs` for
/// [`StepLoss::CrossEntropy`]; the callers check this.
///
/// # Errors
///
/// [`Error::InvalidData`] if `config.batch` is 0; the health checks' errors, such as
/// [`Error::NonFinite`], during training.
pub(crate) fn train(
    data: &Dataset,
    config: GradientConfig,
    setup: Setup,
) -> Result<(Sequential, ParamStore, History), Error> {
    if config.batch == 0 {
        return Err(Error::InvalidData(
            "GradientConfig: the batch size must be at least 1".to_string(),
        ));
    }
    let rows = data.len();
    let mut params = ParamStore::new();
    let linear = Linear::new(
        &mut params,
        setup.features,
        setup.outputs,
        &mut Rng::seed(config.init_seed),
    );
    let (weight, bias) = (linear.weight(), linear.bias());
    if let Some(start) = &setup.bias {
        params.value_mut(bias).data_mut().copy_from_slice(start);
    }
    let model = Sequential::new().push(linear);
    let names = model.parameter_names();
    let penalty = setup.penalty;
    let mut optimizer = config.optimizer.build();
    let health_loss = match setup.loss {
        StepLoss::Mse => Loss::Mse,
        StepLoss::BceWithLogits => Loss::BinaryCrossEntropyWithLogits,
        StepLoss::CrossEntropy => Loss::CrossEntropy,
    };
    let mut monitor = HealthMonitor::new(config.health).with_loss(health_loss);
    let (mut graph, mut trace) = (Graph::new(), Trace::with_capacity(1));
    let size = config.batch.min(rows);
    let mut batcher = Batcher::new(rows, size, config.shuffle_seed);
    let mut batch = Batch::new(data, size);
    let mut history = History {
        epochs: Vec::with_capacity(config.epochs.min(1024)),
    };
    let initial_lr = optimizer.learning_rate();
    let mut step = 0u64;
    for epoch in 1..=config.epochs {
        let lr = config
            .schedule
            .learning_rate(initial_lr, epoch - 1, config.epochs);
        optimizer.set_learning_rate(lr);
        let start = Instant::now();
        batcher.epoch();
        let mut total = 0.0f64;
        for range in batcher.batches() {
            step += 1;
            let batch_rows = range.len() as f64;
            let order = &batcher.order()[range.clone()];
            let view = batcher.gather(data, range.clone(), &mut batch);
            let checked = {
                let mut tape = graph.tape(&params.values);
                let x = tape.input(view.x);
                let out = model.forward_traced(&mut tape, x, &mut trace);
                let mut loss = match (setup.loss, view.targets) {
                    (StepLoss::Mse, TargetsView::Values(y)) => {
                        let y_var = tape.input(y);
                        tape.mse(out, y_var, Reduction::Mean)
                    }
                    (StepLoss::BceWithLogits, TargetsView::Values(y)) => {
                        let y_var = tape.input(y);
                        tape.binary_cross_entropy_with_logits(out, y_var, Reduction::Mean)
                    }
                    (StepLoss::CrossEntropy, TargetsView::Classes(labels)) => {
                        tape.softmax_cross_entropy(out, labels, Reduction::Mean)
                    }
                    _ => unreachable!("the callers pass targets that suit the loss"),
                };
                if penalty > 0.0 {
                    let w = tape.param(weight);
                    let squares = tape.mul(w, w);
                    let sum = tape.sum(squares);
                    let term = tape.scale(sum, penalty);
                    loss = tape.add(loss, term);
                }
                let value = tape.value(loss).data()[0];
                total += f64::from(value) * batch_rows;
                tape.backward(loss, &mut params.grads);
                let inputs = StepInputs {
                    x: view.x,
                    targets: view.targets,
                    rows: Some(order),
                };
                monitor.after_backward(step, value, &tape, &trace, &model, inputs)
            };
            let checked = checked.and_then(|()| {
                optimizer.step(&mut params);
                monitor.after_update(step, optimizer.last_stats(), &names)
            });
            params.zero_grad();
            if !setup.fit_intercept {
                params.value_mut(bias).data_mut().fill(0.0);
            }
            checked?;
        }
        monitor.end_epoch(epoch)?;
        history.epochs.push(EpochReport {
            epoch,
            mean_loss: total / rows as f64,
            rows,
            seconds: start.elapsed().as_secs_f64(),
            validation: None,
        });
    }
    if !setup.fit_intercept {
        params.value_mut(bias).data_mut().fill(0.0);
    }
    Ok((model, params, history))
}

#[cfg(test)]
mod tests {
    //! Tests for the gradient path's settings and errors; its agreement with the exact path
    //! is tested in `tests/gradient_path.rs`.

    use cavitation_core::{Rng, Tensor};
    use cavitation_nn::{Health, Targets};

    use super::*;

    /// Seeded standard-normal inputs `[rows, 2]` and targets `2·x0 − x1 + 3` plus noise.
    fn data(rows: usize) -> Dataset {
        let mut rng = Rng::seed(4);
        let x: Vec<f32> = (0..rows * 2).map(|_| rng.normal() as f32).collect();
        let y: Vec<f32> = x
            .chunks(2)
            .map(|r| 2.0 * r[0] - r[1] + 3.0 + 0.1 * rng.normal() as f32)
            .collect();
        Dataset {
            x: Tensor::from_vec(x, &[rows, 2]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(y, &[rows, 1]).unwrap()),
        }
    }

    /// A quiet default configuration.
    fn quiet() -> GradientConfig {
        GradientConfig {
            health: Health::quiet(),
            ..GradientConfig::default()
        }
    }

    /// Without an intercept the bias stays exactly 0, and the weights match the exact path.
    #[test]
    fn no_intercept_keeps_bias_zero() {
        let d = data(100);
        let r = LinearRegression {
            fit_intercept: false,
            alpha: 0.0,
        };
        let (model, _) = r.fit_gradient(&d, quiet()).unwrap();
        assert_eq!(model.intercept, [0.0]);
        let exact = r.fit(&d).unwrap();
        for (g, e) in model.coef.data().iter().zip(exact.coef.data()) {
            assert!((g - e).abs() <= 1e-3 * e.abs().max(1.0), "{g} against {e}");
        }
    }

    /// Minibatches run too (with the last batch short), and the same seeds give the same
    /// model; another shuffle seed gives a different one.
    #[test]
    fn minibatches_are_deterministic() {
        let d = data(50);
        let config = |seed| GradientConfig {
            batch: 16,
            epochs: 20,
            shuffle_seed: Some(seed),
            ..quiet()
        };
        let r = LinearRegression::default();
        let (a, ha) = r.fit_gradient(&d, config(1)).unwrap();
        let (b, _) = r.fit_gradient(&d, config(1)).unwrap();
        let (c, _) = r.fit_gradient(&d, config(2)).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(ha.epochs.len(), 20);
        assert!(ha.epochs[19].mean_loss < ha.epochs[0].mean_loss);
    }

    /// The penalty is recorded: the reported loss of a ridge fit is the mean squared error
    /// plus `alpha/(n·k)·‖W‖²`, so it is larger than the plain fit's, and the weights are
    /// smaller.
    #[test]
    fn ridge_penalty_shrinks() {
        let d = data(40);
        let plain = LinearRegression::default();
        let ridge = LinearRegression {
            alpha: 40.0,
            ..LinearRegression::default()
        };
        let (p, hp) = plain.fit_gradient(&d, quiet()).unwrap();
        let (q, hq) = ridge.fit_gradient(&d, quiet()).unwrap();
        assert!(q.coef.data()[0].abs() < p.coef.data()[0].abs());
        assert!(hq.epochs.last().unwrap().mean_loss > hp.epochs.last().unwrap().mean_loss);
        let exact = ridge.fit(&d).unwrap();
        for (g, e) in q.coef.data().iter().zip(exact.coef.data()) {
            assert!((g - e).abs() <= 1e-3 * e.abs().max(1.0), "{g} against {e}");
        }
    }

    /// Ridge with three outputs reaches the exact path within `1e-3 · max(1, |exact|)`: the
    /// penalty's `1/(n·k)` scaling holds for `k > 1` (with `1/n` alone it would be three
    /// times too strong).
    #[test]
    fn several_outputs_ridge_matches_exact() {
        let (rows, features, outputs) = (200, 4, 3);
        let mut rng = Rng::seed(9);
        let x: Vec<f32> = (0..rows * features).map(|_| rng.normal() as f32).collect();
        let w: Vec<f32> = (0..features * outputs)
            .map(|_| 2.0 * rng.normal() as f32)
            .collect();
        let mut y = Vec::with_capacity(rows * outputs);
        for row in x.chunks(features) {
            for q in 0..outputs {
                let s: f32 = (0..features).map(|j| row[j] * w[j * outputs + q]).sum();
                y.push(s + q as f32 + 0.5 * rng.normal() as f32);
            }
        }
        let d = Dataset {
            x: Tensor::from_vec(x, &[rows, features]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(y, &[rows, outputs]).unwrap()),
        };
        let ridge = LinearRegression {
            alpha: 10.0,
            ..LinearRegression::default()
        };
        let (g, _) = ridge.fit_gradient(&d, quiet()).unwrap();
        let exact = ridge.fit(&d).unwrap();
        let ours = g.coef.data().iter().chain(&g.intercept);
        let theirs = exact.coef.data().iter().chain(&exact.intercept);
        for (i, (a, e)) in ours.zip(theirs).enumerate() {
            assert!(
                (a - e).abs() <= 1e-3 * e.abs().max(1.0),
                "value {i}: {a} against {e}"
            );
        }
    }

    /// A zero batch, class targets and a negative `alpha` are `InvalidData`.
    #[test]
    fn bad_settings_are_errors() {
        let d = data(10);
        let zero = GradientConfig {
            batch: 0,
            ..quiet()
        };
        let err = LinearRegression::default()
            .fit_gradient(&d, zero)
            .unwrap_err();
        assert!(err.to_string().contains("batch size"), "{err}");
        let classes = Dataset {
            x: d.x.clone(),
            targets: Targets::Classes(vec![0; 10]),
        };
        assert!(
            LinearRegression::default()
                .fit_gradient(&classes, quiet())
                .is_err()
        );
        let negative = LinearRegression {
            alpha: -1.0,
            ..LinearRegression::default()
        };
        assert!(negative.fit_gradient(&d, quiet()).is_err());
    }

    /// Inputs far from unit scale with a learning rate too high for plain SGD overflow, and
    /// the default health checks stop with `Error::NonFinite` instead of returning NaN.
    #[test]
    fn divergence_stops() {
        let mut d = data(20);
        for v in d.x.data_mut() {
            *v *= 1e6;
        }
        let config = GradientConfig {
            optimizer: OptimizerConfig::Sgd {
                lr: 1.0,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            ..quiet()
        };
        let err = LinearRegression::default()
            .fit_gradient(&d, config)
            .unwrap_err();
        assert!(matches!(err, Error::NonFinite { .. }), "{err}");
    }
}
