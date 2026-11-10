//! Health checks during training: non-finite values, exploding and vanishing gradient norms,
//! clamped binary cross-entropy logarithms and dead ReLU neurons.

use std::fmt;

use cavitation_core::{Error, ParamId, Tape};

use crate::activation::last_dim;
use crate::{ActivationKind, GradStats, Sequential, Trace};

/// What the health checks do when the loss or a gradient is NaN or infinite.
///
/// # Examples
///
/// ```
/// use cavitation_nn::{Health, OnNonFinite};
/// assert_eq!(Health::default().on_non_finite, OnNonFinite::Stop);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnNonFinite {
    /// Stop training with [`Error::NonFinite`], naming the step and where the value was,
    /// after sending the matching [`Warning::NonFinite`] to the sink.
    Stop,
    /// Send a [`Warning::NonFinite`] to the sink and keep training.
    Warn,
}

/// Something the health checks noticed. Every warning names the step or epoch it belongs
/// to; its [`Display`](fmt::Display) form is one plain-English line.
///
/// # Examples
///
/// ```
/// use cavitation_nn::Warning;
/// let w = Warning::ClampedLogs { step: 4, count: 1 };
/// assert!(w.to_string().starts_with("step 4:"));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Warning {
    /// Binary cross-entropy met probabilities of exactly 0 or 1 and clamped their
    /// logarithms to -100 (see [`Tape::clamped_logs`]). Sent at most once per epoch.
    ClampedLogs {
        /// The step, counting from 1.
        step: u64,
        /// How many logarithms that step clamped.
        count: usize,
    },
    /// The loss or a gradient held a NaN or an infinity. With [`OnNonFinite::Warn`] it is
    /// sent at most once per epoch for each location; with [`OnNonFinite::Stop`] it is sent
    /// once, just before the step returns [`Error::NonFinite`] for the same location.
    NonFinite {
        /// The step, counting from 1.
        step: u64,
        /// `"loss"`, or the parameter's name, for example `"0.weight"`.
        location: String,
    },
    /// A parameter's gradient norm was above [`Health::grad_norm_high`]. Sent at most once
    /// per epoch for each parameter.
    Exploding {
        /// The step, counting from 1.
        step: u64,
        /// The parameter's name, for example `"0.weight"`.
        layer: String,
        /// The Euclidean norm of the parameter's gradient.
        norm: f64,
    },
    /// A parameter's gradient norm was below [`Health::grad_norm_low`]. Sent at most once
    /// per epoch for each parameter.
    Vanishing {
        /// The step, counting from 1.
        step: u64,
        /// The parameter's name, for example `"0.weight"`.
        layer: String,
        /// The Euclidean norm of the parameter's gradient.
        norm: f64,
    },
    /// Neurons of a ReLU or ReLU6 layer that output exactly 0 for every sample of the epoch.
    /// Neurons turned off by a [`NeuronMask`](crate::NeuronMask) are not counted. Sent to the
    /// sink every epoch; without a sink, printed only the first time for each layer.
    DeadNeurons {
        /// The epoch, as passed to [`HealthMonitor::end_epoch`].
        epoch: usize,
        /// The layer's index in the [`Sequential`].
        layer: usize,
        /// The dead neurons' positions on the layer's output, in increasing order.
        neurons: Vec<usize>,
    },
}

impl fmt::Display for Warning {
    /// Writes the warning as one plain-English line, starting with its step or epoch.
    ///
    /// ```
    /// use cavitation_nn::Warning;
    /// let w = Warning::NonFinite { step: 1, location: "loss".to_string() };
    /// assert_eq!(w.to_string(), "step 1: a NaN or infinite value in loss");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Warning::ClampedLogs { step, count } => write!(
                f,
                "step {step}: binary cross-entropy met {count} probabilities of exactly 0 or 1 \
                 and clamped their logarithms to -100"
            ),
            Warning::NonFinite { step, location } => {
                write!(f, "step {step}: a NaN or infinite value in {location}")
            }
            Warning::Exploding { step, layer, norm } => write!(
                f,
                "step {step}: the gradient norm of {layer} is {norm:.3e}, above the exploding limit"
            ),
            Warning::Vanishing { step, layer, norm } => write!(
                f,
                "step {step}: the gradient norm of {layer} is {norm:.3e}, below the vanishing limit"
            ),
            Warning::DeadNeurons {
                epoch,
                layer,
                neurons,
            } => write!(
                f,
                "epoch {epoch}: {} neurons of layer {layer} output 0 for every sample (dead ReLU): \
                 {neurons:?}",
                neurons.len()
            ),
        }
    }
}

/// Which health checks run during training, and where their warnings go.
///
/// The default stops on a non-finite loss or gradient, warns about gradient norms above
/// 1e3 or below 1e-7, reports dead ReLU neurons, and prints the warnings to standard error
/// as `cavitation: warning: <the warning>`. Printed, a layer's dead-neuron report appears
/// only the first time, so a network with a dead layer does not print a line every epoch;
/// a [`sink`](Health::sink) receives every epoch's report.
///
/// # Examples
///
/// ```
/// use cavitation_nn::{Health, OnNonFinite, Warning};
/// let health = Health {
///     on_non_finite: OnNonFinite::Warn,
///     dead_relu: false,
///     sink: Some(Box::new(|w: &Warning| println!("training: {w}"))),
///     ..Health::default()
/// };
/// assert_eq!(health.grad_norm_high, 1e3);
/// ```
pub struct Health {
    /// What to do when the loss or a gradient is NaN or infinite.
    pub on_non_finite: OnNonFinite,
    /// A gradient norm above this sends [`Warning::Exploding`].
    pub grad_norm_high: f64,
    /// A gradient norm below this sends [`Warning::Vanishing`].
    pub grad_norm_low: f64,
    /// Whether to look for dead ReLU and ReLU6 neurons (one pass over those layers' outputs
    /// each step).
    pub dead_relu: bool,
    /// Receives every warning; `None` prints them to standard error, prefixed
    /// `cavitation: warning:`, each layer's [`Warning::DeadNeurons`] only the first time.
    // The boxed callback is the plainest type for this field; an alias would only hide it.
    #[allow(clippy::type_complexity)]
    pub sink: Option<Box<dyn FnMut(&Warning)>>,
}

impl Default for Health {
    /// Stop on non-finite values, norm bounds 1e3 and 1e-7, dead ReLU checks on, warnings
    /// printed to standard error.
    fn default() -> Health {
        Health {
            on_non_finite: OnNonFinite::Stop,
            grad_norm_high: 1e3,
            grad_norm_low: 1e-7,
            dead_relu: true,
            sink: None,
        }
    }
}

impl fmt::Debug for Health {
    /// Writes every setting; the sink shows only whether there is one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Health")
            .field("on_non_finite", &self.on_non_finite)
            .field("grad_norm_high", &self.grad_norm_high)
            .field("grad_norm_low", &self.grad_norm_low)
            .field("dead_relu", &self.dead_relu)
            .field("sink", &self.sink.as_ref().map(|_| "FnMut"))
            .finish()
    }
}

/// Bit in a parameter's warned flags: [`Warning::NonFinite`] was sent this epoch.
const WARNED_NON_FINITE: u8 = 1;
/// Bit in a parameter's warned flags: [`Warning::Exploding`] was sent this epoch.
const WARNED_EXPLODING: u8 = 2;
/// Bit in a parameter's warned flags: [`Warning::Vanishing`] was sent this epoch.
const WARNED_VANISHING: u8 = 4;

/// Where one ReLU or ReLU6 layer's neurons sit in the monitor's `alive` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReluLayer {
    /// The layer's index in the model.
    layer: usize,
    /// The position of its first neuron in `alive`.
    offset: usize,
    /// Its output width.
    width: usize,
}

/// Runs the [`Health`] checks during training. [`Trainer`](crate::Trainer) and
/// [`fit`](crate::fit) run one for you; a hand-written loop calls it directly, three places
/// per epoch:
///
/// 1. [`after_backward`](Self::after_backward) after `tape.backward`, while the tape still
///    lives: the loss's finiteness, binary cross-entropy's clamped logarithms
///    ([`Tape::clamped_logs`]) and the ReLU outputs for the dead-neuron check. Run the
///    forward pass with [`Sequential::forward_traced`] so the monitor can find each layer's
///    output.
/// 2. [`after_update`](Self::after_update) after the tape is dropped and
///    [`Optimizer::step`](crate::Optimizer::step) has run: each gradient's finiteness and
///    norm, from the optimiser's [`GradStats`] (read with
///    [`last_stats`](crate::Optimizer::last_stats)), so the gradients are not read twice.
///    A gradient that is not finite while the loss is finite is therefore found after the
///    update has applied it.
/// 3. [`end_epoch`](Self::end_epoch) once each epoch ends: the dead-neuron report, and the
///    start of a new epoch for the once-per-epoch warnings.
///
/// After its first epoch the monitor allocates nothing while training is healthy; a
/// warning that names a parameter or lists neurons allocates its text when it is sent.
///
/// # Examples
///
/// ```
/// use cavitation_core::{Error, Graph, ParamStore, Reduction, Rng, Tensor};
/// use cavitation_nn::{Health, HealthMonitor, Linear, Optimizer, Sequential, Sgd, Trace};
/// let x = Tensor::from_vec(vec![0.0f32, 1.0, f32::NAN, 3.0], &[4, 1]).unwrap();
/// let y = Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap();
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
/// let names = net.parameter_names();
/// let mut monitor = HealthMonitor::new(Health::default());
/// let (mut graph, mut trace, mut optimizer) = (Graph::new(), Trace::new(), Sgd::new(0.05));
/// let mut train = || -> Result<(), Error> {
///     for step in 1..=10 {
///         let mut tape = graph.tape(&params.values);
///         let input = tape.input(x.view());
///         let target = tape.input(y.view());
///         let out = net.forward_traced(&mut tape, input, &mut trace);
///         let loss = tape.mse(out, target, Reduction::Mean);
///         let value = tape.value(loss).data()[0];
///         tape.backward(loss, &mut params.grads);
///         monitor.after_backward(step, value, &tape, &trace, &net)?;
///         drop(tape);
///         optimizer.step(&mut params);
///         monitor.after_update(step, optimizer.last_stats(), &names)?;
///         params.zero_grad();
///     }
///     monitor.end_epoch(1)
/// };
/// let err = train().unwrap_err();
/// assert_eq!(err.to_string(), "non-finite value at step 1 in loss");
/// ```
pub struct HealthMonitor {
    /// The checks and the sink.
    health: Health,
    /// Whether the non-finite loss warning was sent this epoch.
    loss_warned: bool,
    /// Whether the clamped-logarithm warning was sent this epoch.
    clamped_warned: bool,
    /// For each parameter, in the order of the names passed to `after_update`, which
    /// warnings were sent this epoch (the `WARNED_*` bits).
    warned: Vec<u8>,
    /// The ReLU and ReLU6 layers seen in the last step.
    relu_layers: Vec<ReluLayer>,
    /// For every neuron of those layers, whether it output anything but 0 this epoch (or is
    /// masked off, and so never reported).
    alive: Vec<bool>,
    /// Whether `alive` holds at least one step of this epoch.
    tracked: bool,
    /// The layers whose dead-neuron report was printed to standard error (with no sink).
    dead_printed: Vec<usize>,
}

impl HealthMonitor {
    /// A monitor running the checks of `health`.
    ///
    /// ```
    /// use cavitation_nn::{Health, HealthMonitor};
    /// let mut monitor = HealthMonitor::new(Health::default());
    /// assert!(monitor.end_epoch(1).is_ok());
    /// ```
    pub fn new(health: Health) -> HealthMonitor {
        HealthMonitor {
            health,
            loss_warned: false,
            clamped_warned: false,
            warned: Vec::new(),
            relu_layers: Vec::new(),
            alive: Vec::new(),
            tracked: false,
            dead_printed: Vec::new(),
        }
    }

    /// Sends `warning` to the sink, or prints it to standard error when there is none: each
    /// layer's dead-neuron report only the first time, saying that later epochs are not
    /// printed.
    fn emit(&mut self, warning: Warning) {
        match (self.health.sink.as_mut(), &warning) {
            (Some(sink), _) => sink(&warning),
            (None, Warning::DeadNeurons { layer, .. }) => {
                if !self.dead_printed.contains(layer) {
                    self.dead_printed.push(*layer);
                    eprintln!(
                        "cavitation: warning: {warning}; later epochs' reports for layer {layer} \
                         are not printed (a Health::sink receives every one)"
                    );
                }
            }
            (None, _) => eprintln!("cavitation: warning: {warning}"),
        }
    }

    /// Sends [`Warning::NonFinite`] for `location` at `step` to the sink and returns the
    /// matching [`Error::NonFinite`], for [`OnNonFinite::Stop`].
    fn stop(&mut self, step: u64, location: String) -> Error {
        self.emit(Warning::NonFinite {
            step,
            location: location.clone(),
        });
        Error::NonFinite { step, location }
    }

    /// The checks that need the tape, run after the backward pass of step `step` (counting
    /// from 1) and before the tape is dropped: `loss` must be finite, binary cross-entropy
    /// clamps send [`Warning::ClampedLogs`] (at most once per epoch), and, with
    /// [`Health::dead_relu`], each ReLU and ReLU6 layer's output in `trace` is checked for
    /// neurons that are not 0. `trace` must come from this step's
    /// [`Sequential::forward_traced`] of `model` on `tape`.
    ///
    /// # Errors
    ///
    /// With [`OnNonFinite::Stop`], returns [`Error::NonFinite`] with location `"loss"` if
    /// `loss` is NaN or infinite, after sending [`Warning::NonFinite`] to the sink. With
    /// [`OnNonFinite::Warn`] it only sends the warning (at most once per epoch) and returns
    /// `Ok`.
    ///
    /// # Panics
    ///
    /// May panic if `trace` was not recorded on `tape`.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{Health, HealthMonitor, Relu, Sequential, Trace};
    /// let params = ParamStore::new();
    /// let net = Sequential::new().push(Relu);
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&params.values);
    /// let x = tape.input(Tensor::full(&[2, 2], 1.0f32).view());
    /// let mut trace = Trace::new();
    /// net.forward_traced(&mut tape, x, &mut trace);
    /// let mut monitor = HealthMonitor::new(Health::default());
    /// assert!(monitor.after_backward(1, 4.0, &tape, &trace, &net).is_ok());
    /// let err = monitor.after_backward(2, f32::NAN, &tape, &trace, &net).unwrap_err();
    /// assert_eq!(err.to_string(), "non-finite value at step 2 in loss");
    /// ```
    pub fn after_backward(
        &mut self,
        step: u64,
        loss: f32,
        tape: &Tape<'_, f32>,
        trace: &Trace,
        model: &Sequential,
    ) -> Result<(), Error> {
        let clamped = tape.clamped_logs();
        if clamped > 0 && !self.clamped_warned {
            self.clamped_warned = true;
            self.emit(Warning::ClampedLogs {
                step,
                count: clamped,
            });
        }
        if !loss.is_finite() {
            match self.health.on_non_finite {
                OnNonFinite::Stop => {
                    return Err(self.stop(step, "loss".to_string()));
                }
                OnNonFinite::Warn if !self.loss_warned => {
                    self.loss_warned = true;
                    self.emit(Warning::NonFinite {
                        step,
                        location: "loss".to_string(),
                    });
                }
                OnNonFinite::Warn => {}
            }
        }
        if self.health.dead_relu {
            self.track_relu_outputs(tape, trace, model);
        }
        Ok(())
    }

    /// Marks every neuron of each ReLU and ReLU6 layer that output anything but 0 in this
    /// step, and every neuron masked off, as alive. Rebuilds the layer table when the
    /// layers or widths differ from the last step's; otherwise allocates nothing.
    fn track_relu_outputs(&mut self, tape: &Tape<'_, f32>, trace: &Trace, model: &Sequential) {
        let mut k = 0;
        let mut offset = 0;
        for i in 0..trace.len().min(model.len()) {
            let is_relu = model
                .layer(i)
                .activation()
                .and_then(|a| a.builtin())
                .is_some_and(|kind| matches!(kind, ActivationKind::Relu | ActivationKind::Relu6));
            if !is_relu {
                continue;
            }
            let out = tape.value(trace.output(i));
            let width = last_dim(*out.shape());
            let entry = ReluLayer {
                layer: i,
                offset,
                width,
            };
            if self.relu_layers.get(k) != Some(&entry) {
                self.relu_layers.truncate(k);
                self.relu_layers.push(entry);
                self.alive.truncate(offset);
                self.alive.resize(offset + width, false);
            }
            let alive = &mut self.alive[offset..offset + width];
            if width > 0 {
                for row in out.data().chunks_exact(width) {
                    for (a, &v) in alive.iter_mut().zip(row) {
                        *a |= v != 0.0;
                    }
                }
            }
            if let Some(mask) = model.mask(i) {
                for (j, a) in alive.iter_mut().enumerate() {
                    *a |= !mask.is_on(j);
                }
            }
            k += 1;
            offset += width;
        }
        self.relu_layers.truncate(k);
        self.alive.truncate(offset);
        self.tracked = true;
    }

    /// The checks on the gradients, run after [`Optimizer::step`](crate::Optimizer::step)
    /// of step `step` with that optimiser's [`last_stats`](crate::Optimizer::last_stats).
    /// `names` gives each parameter's name, as [`Sequential::parameter_names`] returns them
    /// (compute it once, before training). Each named parameter's gradient must be finite,
    /// and its norm between [`Health::grad_norm_low`] and [`Health::grad_norm_high`]: one
    /// outside sends [`Warning::Vanishing`] or [`Warning::Exploding`], at most once per
    /// epoch for each parameter.
    ///
    /// The update has already applied the gradients when this runs, so a non-finite
    /// gradient has reached the parameters by then.
    ///
    /// # Errors
    ///
    /// With [`OnNonFinite::Stop`], returns [`Error::NonFinite`] naming the first parameter
    /// (in the order of the store) whose gradient held a NaN or an infinity, after sending
    /// [`Warning::NonFinite`] to the sink. With [`OnNonFinite::Warn`] it sends
    /// [`Warning::NonFinite`] for each such parameter instead (at most once per epoch each)
    /// and returns `Ok`.
    ///
    /// # Panics
    ///
    /// Panics if a parameter in `names` was not in the optimiser's last step.
    ///
    /// ```
    /// use std::cell::Cell;
    /// use std::rc::Rc;
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Health, HealthMonitor, Linear, Optimizer, Sequential, Sgd, Warning};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut optimizer = Sgd::new(0.1);
    /// // No backward pass: nothing moves, and every gradient counts as 0 in the statistics.
    /// optimizer.step(&mut params);
    /// // The sink is `'static`, so it shares the count through an `Rc`.
    /// let vanishing = Rc::new(Cell::new(0));
    /// let count = Rc::clone(&vanishing);
    /// let mut monitor = HealthMonitor::new(Health {
    ///     sink: Some(Box::new(move |w: &Warning| {
    ///         if let Warning::Vanishing { .. } = w {
    ///             count.set(count.get() + 1);
    ///         }
    ///     })),
    ///     ..Health::default()
    /// });
    /// monitor.after_update(1, optimizer.last_stats(), &net.parameter_names()).unwrap();
    /// assert_eq!(vanishing.get(), 2);
    /// ```
    pub fn after_update(
        &mut self,
        step: u64,
        stats: &GradStats,
        names: &[(String, ParamId)],
    ) -> Result<(), Error> {
        if self.health.on_non_finite == OnNonFinite::Stop
            && let Some(id) = stats.first_non_finite()
        {
            let location = match names.iter().find(|(_, i)| *i == id) {
                Some((name, _)) => name.clone(),
                None => format!("{id:?}"),
            };
            return Err(self.stop(step, location));
        }
        self.warned.resize(names.len(), 0);
        let (high, low) = (self.health.grad_norm_high, self.health.grad_norm_low);
        for (k, (name, id)) in names.iter().enumerate() {
            // A sum of squares of finite `f32` values is finite in `f64`, so a norm that is
            // not finite means the gradient held a NaN or an infinity.
            let norm = stats.norm(*id);
            let bit = if !norm.is_finite() {
                WARNED_NON_FINITE
            } else if norm > high {
                WARNED_EXPLODING
            } else if norm < low {
                WARNED_VANISHING
            } else {
                continue;
            };
            if self.warned[k] & bit != 0 {
                continue;
            }
            self.warned[k] |= bit;
            let name = name.clone();
            let warning = match bit {
                WARNED_NON_FINITE => Warning::NonFinite {
                    step,
                    location: name,
                },
                WARNED_EXPLODING => Warning::Exploding {
                    step,
                    layer: name,
                    norm,
                },
                _ => Warning::Vanishing {
                    step,
                    layer: name,
                    norm,
                },
            };
            self.emit(warning);
        }
        Ok(())
    }

    /// Ends epoch `epoch`: with [`Health::dead_relu`], sends [`Warning::DeadNeurons`] for
    /// each ReLU or ReLU6 layer with neurons that output exactly 0 for every sample since the
    /// last `end_epoch` (masked-off neurons excepted), then starts a new epoch, so each
    /// once-per-epoch warning can be sent again.
    ///
    /// # Errors
    ///
    /// None of the end-of-epoch checks stops training, so this returns `Ok`; it returns a
    /// `Result` like the per-step calls so that a loop can use `?` on all three.
    ///
    /// ```
    /// use cavitation_nn::{Health, HealthMonitor};
    /// let mut monitor = HealthMonitor::new(Health::default());
    /// // No step ran this epoch, so there is nothing to report.
    /// assert!(monitor.end_epoch(1).is_ok());
    /// ```
    pub fn end_epoch(&mut self, epoch: usize) -> Result<(), Error> {
        if self.tracked {
            for k in 0..self.relu_layers.len() {
                let ReluLayer {
                    layer,
                    offset,
                    width,
                } = self.relu_layers[k];
                let alive = &self.alive[offset..offset + width];
                if alive.iter().all(|&a| a) {
                    continue;
                }
                let neurons = (0..width).filter(|&j| !alive[j]).collect();
                self.emit(Warning::DeadNeurons {
                    epoch,
                    layer,
                    neurons,
                });
            }
        }
        self.alive.fill(false);
        self.tracked = false;
        self.loss_warned = false;
        self.clamped_warned = false;
        self.warned.fill(0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the health checks, through `fit`, `Trainer` and the manual loop.

    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::{
        Batch, Batcher, Dataset, FitConfig, Linear, Loss, NeuronMask, Optimizer, OptimizerConfig,
        Relu, Sequential, Sgd, Tanh, Targets, Trainer, fit,
    };
    use cavitation_core::{Error, Graph, ParamStore, Reduction, Rng, Tensor, TensorView};

    /// The list a collecting sink writes to.
    type Seen = Rc<RefCell<Vec<Warning>>>;

    /// A warning sink, as [`Health::sink`] holds it.
    type Sink = Box<dyn FnMut(&Warning)>;

    /// A sink that keeps a copy of every warning it receives, and the list it keeps them in.
    fn collecting_sink() -> (Seen, Sink) {
        let seen = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&seen);
        (
            seen,
            Box::new(move |w: &Warning| sink.borrow_mut().push(w.clone())),
        )
    }

    /// The default checks with `on_non_finite` and the given sink.
    fn health(on_non_finite: OnNonFinite, sink: Sink) -> Health {
        Health {
            on_non_finite,
            sink: Some(sink),
            ..Health::default()
        }
    }

    /// 64 rows of 4 features in `[-1, 1)` from `Rng::seed(4)` with class labels 0 to 2;
    /// the first feature of row 0 is NaN.
    fn nan_dataset() -> Dataset {
        let mut rng = Rng::seed(4);
        let mut x: Vec<f32> = (0..64 * 4).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        x[0] = f32::NAN;
        Dataset {
            x: Tensor::from_vec(x, &[64, 4]).unwrap(),
            targets: Targets::Classes((0..64).map(|i| i % 3).collect()),
        }
    }

    /// A 4 → 8 tanh → 3 classifier from `Rng::seed(6)`.
    fn classifier(params: &mut ParamStore) -> Sequential {
        let mut rng = Rng::seed(6);
        Sequential::new()
            .push(Linear::new(params, 4, 8, &mut rng))
            .push(Tanh)
            .push(Linear::new(params, 8, 3, &mut rng))
    }

    /// The bits of every parameter value of `net`, in order.
    fn parameter_bits(net: &Sequential, params: &ParamStore) -> Vec<u32> {
        net.parameter_names()
            .into_iter()
            .flat_map(|(_, id)| {
                params
                    .value(id)
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Whether `err` is `Error::NonFinite` at `step` in `location`.
    fn is_non_finite(err: &Error, step: u64, location: &str) -> bool {
        matches!(err, Error::NonFinite { step: s, location: l } if *s == step && l == location)
    }

    /// A NaN in the first input row stops `fit` at step 1 with the loss named as the
    /// location, before the optimiser applies anything.
    #[test]
    fn nan_in_input_stops_training() {
        let data = nan_dataset();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let before = parameter_bits(&net, &params);
        let config = FitConfig {
            epochs: 2,
            batch: 32,
            shuffle_seed: None,
            ..FitConfig::default()
        };
        let err = fit(&net, &mut params, &data, config).unwrap_err();
        assert!(is_non_finite(&err, 1, "loss"), "{err:?}");
        assert_eq!(err.to_string(), "non-finite value at step 1 in loss");
        assert_eq!(parameter_bits(&net, &params), before);
    }

    /// With `OnNonFinite::Warn` the sink receives the non-finite loss (once per epoch) and
    /// the non-finite gradients, and training runs every epoch.
    #[test]
    fn warn_mode_continues() {
        let data = nan_dataset();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let (seen, sink) = collecting_sink();
        let config = FitConfig {
            epochs: 2,
            batch: 32,
            shuffle_seed: None,
            health: health(OnNonFinite::Warn, sink),
            ..FitConfig::default()
        };
        let history = fit(&net, &mut params, &data, config).unwrap();
        assert_eq!(history.epochs.len(), 2);
        let seen = seen.borrow();
        assert_eq!(
            seen[0],
            Warning::NonFinite {
                step: 1,
                location: "loss".to_string()
            }
        );
        let loss_steps: Vec<u64> = seen
            .iter()
            .filter_map(|w| match w {
                Warning::NonFinite { step, location } if location == "loss" => Some(*step),
                _ => None,
            })
            .collect();
        assert_eq!(loss_steps, [1, 3]);
        assert!(seen.contains(&Warning::NonFinite {
            step: 1,
            location: "0.weight".to_string()
        }));
    }

    /// One `Linear` 2 → 1 fitting a target of 1e4 from inputs [1, 1] with a learning rate of
    /// 1e-12, so nothing moves: the step-1 gradient norm of `0.weight` is far above 1e3, and
    /// the sink receives exactly one `Exploding` warning for it per epoch.
    #[test]
    fn exploding_gradient_warns_once_per_epoch() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 1, &mut Rng::seed(0));
        let (w, b) = (layer.weight(), layer.bias());
        let net = Sequential::new().push(layer);
        let y = f64::from(params.value(w).data().iter().sum::<f32>() + params.value(b).data()[0]);
        // Mean squared error over a batch of 2 rows: each weight's gradient is 2 (y − 1e4).
        let expected_norm = 2.0 * 2f64.sqrt() * (y - 1e4).abs();
        let data = Dataset {
            x: Tensor::full(&[4, 2], 1.0f32),
            targets: Targets::Values(Tensor::full(&[4, 1], 1e4f32)),
        };
        let (seen, sink) = collecting_sink();
        let config = FitConfig {
            epochs: 2,
            batch: 2,
            optimizer: OptimizerConfig::Sgd {
                lr: 1e-12,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            shuffle_seed: None,
            health: health(OnNonFinite::Warn, sink),
            ..FitConfig::default()
        };
        fit(&net, &mut params, &data, config).unwrap();
        let exploding: Vec<(u64, f64)> = seen
            .borrow()
            .iter()
            .filter_map(|w| match w {
                Warning::Exploding { step, layer, norm } if layer == "0.weight" => {
                    Some((*step, *norm))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            exploding.iter().map(|e| e.0).collect::<Vec<_>>(),
            [1, 3],
            "{exploding:?}"
        );
        let (_, norm) = exploding[0];
        assert!(norm > 1e3, "{norm}");
        assert!(
            (norm - expected_norm).abs() <= 1e-5 * expected_norm,
            "{norm} vs {expected_norm}"
        );
    }

    /// A 2 → 4 → ReLU → 1 network whose first bias is −100 (so the ReLU outputs 0 for every
    /// sample), with `masked` neurons of the ReLU layer turned off, and 8 rows to train on.
    fn dead_network(params: &mut ParamStore, masked: &[usize]) -> (Sequential, Dataset) {
        let mut rng = Rng::seed(3);
        let first = Linear::new(params, 2, 4, &mut rng);
        params.value_mut(first.bias()).data_mut().fill(-100.0);
        let mut net = Sequential::new()
            .push(first)
            .push(Relu)
            .push(Linear::new(params, 4, 1, &mut rng));
        if !masked.is_empty() {
            let mut mask = NeuronMask::all_on(4);
            for &n in masked {
                mask.set(n, false);
            }
            net.set_mask(1, mask);
        }
        let x: Vec<f32> = (0..16).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let data = Dataset {
            x: Tensor::from_vec(x, &[8, 2]).unwrap(),
            targets: Targets::Values(Tensor::full(&[8, 1], 0.5f32)),
        };
        (net, data)
    }

    /// The `DeadNeurons` warnings in `seen`.
    fn dead_reports(seen: &Seen) -> Vec<Warning> {
        seen.borrow()
            .iter()
            .filter(|w| matches!(w, Warning::DeadNeurons { .. }))
            .cloned()
            .collect()
    }

    /// A ReLU layer whose input is always about −100 has every neuron reported dead at the
    /// epoch's end; a neuron turned off by a mask is not reported.
    #[test]
    fn dead_relu_reported() {
        for (masked, dead) in [(vec![], vec![0, 1, 2, 3]), (vec![2], vec![0, 1, 3])] {
            let mut params = ParamStore::new();
            let (net, data) = dead_network(&mut params, &masked);
            let (seen, sink) = collecting_sink();
            let config = FitConfig {
                epochs: 1,
                batch: 4,
                optimizer: OptimizerConfig::Sgd {
                    lr: 0.1,
                    momentum: 0.0,
                    weight_decay: 0.0,
                },
                loss: Loss::Mse,
                health: health(OnNonFinite::Stop, sink),
                ..FitConfig::default()
            };
            fit(&net, &mut params, &data, config).unwrap();
            assert_eq!(
                dead_reports(&seen),
                [Warning::DeadNeurons {
                    epoch: 1,
                    layer: 1,
                    neurons: dead
                }],
                "masked {masked:?}"
            );
        }
    }

    /// Three epochs of `fit` on [`dead_network`] with `health`, using SGD and mean squared
    /// error; the network stays dead every epoch.
    fn fit_dead_network(health: Health) {
        let mut params = ParamStore::new();
        let (net, data) = dead_network(&mut params, &[]);
        let config = FitConfig {
            epochs: 3,
            batch: 4,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            health,
            ..FitConfig::default()
        };
        fit(&net, &mut params, &data, config).unwrap();
    }

    /// A sink receives the dead-neuron report of every epoch, not only the first.
    #[test]
    fn dead_relu_sink_receives_every_epoch() {
        let (seen, sink) = collecting_sink();
        fit_dead_network(health(OnNonFinite::Stop, sink));
        let epochs: Vec<usize> = dead_reports(&seen)
            .iter()
            .map(|w| match w {
                Warning::DeadNeurons { epoch, .. } => *epoch,
                _ => unreachable!("dead_reports keeps only DeadNeurons"),
            })
            .collect();
        assert_eq!(epochs, [1, 2, 3]);
    }

    /// Without a sink, a layer's dead neurons are printed to standard error once per run, with
    /// their count, and not again every epoch: the test binary re-runs itself as a child that
    /// trains the dead network for three epochs, and its standard error has one such line.
    #[test]
    fn dead_relu_prints_once_per_layer_without_a_sink() {
        let exe = std::env::current_exe().expect("the test binary's path");
        let out = std::process::Command::new(exe)
            .args([
                "--ignored",
                "--exact",
                "--nocapture",
                "--test-threads=1",
                "health::tests::dead_relu_printing_child",
            ])
            .output()
            .expect("the test binary runs");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "the child should run the helper and pass; stdout:\n{stdout}\nstderr:\n{stderr}"
        );
        let dead: Vec<&str> = stderr
            .lines()
            .filter(|l| l.starts_with("cavitation: warning:") && l.contains("dead ReLU"))
            .collect();
        assert_eq!(dead.len(), 1, "stderr:\n{stderr}");
        assert!(
            dead[0].starts_with(
                "cavitation: warning: epoch 1: 4 neurons of layer 1 output 0 for every sample"
            ) && dead[0].contains("later epochs"),
            "{}",
            dead[0]
        );
    }

    /// Helper for `dead_relu_prints_once_per_layer_without_a_sink`: trains the dead network
    /// with the default checks, which print to standard error. Ignored by default; it is run
    /// in a child process.
    #[test]
    #[ignore = "run in a child process by dead_relu_prints_once_per_layer_without_a_sink"]
    fn dead_relu_printing_child() {
        fit_dead_network(Health::default());
    }

    /// Runs the manual loop over the first `steps` batches of 32 rows of `data` with a
    /// monitor, stopping at the first error the monitor returns. Returns the steps finished.
    fn manual_epoch(
        net: &Sequential,
        params: &mut ParamStore,
        data: &Dataset,
        monitor: &mut HealthMonitor,
        steps: u64,
    ) -> Result<u64, Error> {
        let Targets::Classes(labels) = &data.targets else {
            unreachable!("the set has class labels")
        };
        let names = net.parameter_names();
        let mut optimizer = Sgd::new(0.1);
        let mut graph = Graph::new();
        let mut trace = crate::Trace::new();
        let features = data.features();
        for step in 1..=steps {
            let rows = (step as usize - 1) * 32..step as usize * 32;
            let x = TensorView::from_slice(
                &data.x.data()[rows.start * features..rows.end * features],
                &[32, features],
            )
            .unwrap();
            let mut tape = graph.tape(&params.values);
            let xi = tape.input(x);
            let out = net.forward_traced(&mut tape, xi, &mut trace);
            let loss = tape.softmax_cross_entropy(out, &labels[rows], Reduction::Mean);
            let value = tape.value(loss).data()[0];
            tape.backward(loss, &mut params.grads);
            monitor.after_backward(step, value, &tape, &trace, net)?;
            drop(tape);
            optimizer.step(params);
            monitor.after_update(step, optimizer.last_stats(), &names)?;
            params.zero_grad();
        }
        monitor.end_epoch(1)?;
        Ok(steps)
    }

    /// In the manual loop the monitor's `after_backward` returns the non-finite loss of step 1
    /// as an error; without the NaN the same loop finishes.
    #[test]
    fn manual_loop_monitor_stops_on_nan() {
        let data = nan_dataset();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let before = parameter_bits(&net, &params);
        let mut monitor = HealthMonitor::new(Health::default());
        let err = manual_epoch(&net, &mut params, &data, &mut monitor, 2).unwrap_err();
        assert!(is_non_finite(&err, 1, "loss"), "{err:?}");
        assert_eq!(parameter_bits(&net, &params), before);

        // The stopped loop left the NaN gradients of step 1 in place, so start afresh.
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let mut clean = nan_dataset();
        clean.x.data_mut()[0] = 0.25;
        let mut monitor = HealthMonitor::new(Health::default());
        assert_eq!(
            manual_epoch(&net, &mut params, &clean, &mut monitor, 2).unwrap(),
            2
        );
    }

    /// A non-finite gradient with a finite loss is reported by `after_update`, naming the
    /// parameter, both as the error and to the sink.
    #[test]
    fn non_finite_gradient_names_the_parameter() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 1, &mut Rng::seed(0));
        let bias = layer.bias();
        let net = Sequential::new().push(layer);
        params.grads.accumulate(bias).0.data_mut()[0] = f32::INFINITY;
        let mut optimizer = Sgd::new(0.1);
        optimizer.step(&mut params);
        let (seen, sink) = collecting_sink();
        let mut monitor = HealthMonitor::new(health(OnNonFinite::Stop, sink));
        let err = monitor
            .after_update(5, optimizer.last_stats(), &net.parameter_names())
            .unwrap_err();
        assert!(is_non_finite(&err, 5, "0.bias"), "{err:?}");
        assert_eq!(
            *seen.borrow(),
            [Warning::NonFinite {
                step: 5,
                location: "0.bias".to_string()
            }]
        );
    }

    /// Binary cross-entropy with one prediction of exactly 0 against a target of 1 gives the
    /// sink one `ClampedLogs` warning per epoch (two steps each), and training continues.
    #[test]
    fn clamped_bce_warns() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 1, 1, &mut Rng::seed(0));
        params.value_mut(layer.weight()).data_mut()[0] = 0.5;
        let net = Sequential::new().push(layer);
        // Predictions 0, 0.5, 0, 0.5: the zero against a target of 1 is clamped, and every
        // gradient is 0, so the predictions stay where they are.
        let data = Dataset {
            x: Tensor::from_vec(vec![0.0f32, 1.0, 0.0, 1.0], &[4, 1]).unwrap(),
            targets: Targets::Values(
                Tensor::from_vec(vec![1.0f32, 0.5, 1.0, 0.5], &[4, 1]).unwrap(),
            ),
        };
        let (seen, sink) = collecting_sink();
        let config = FitConfig {
            epochs: 2,
            batch: 2,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::BinaryCrossEntropy,
            shuffle_seed: None,
            health: health(OnNonFinite::Stop, sink),
            ..FitConfig::default()
        };
        let history = fit(&net, &mut params, &data, config).unwrap();
        assert_eq!(history.epochs.len(), 2);
        let clamped: Vec<Warning> = seen
            .borrow()
            .iter()
            .filter(|w| matches!(w, Warning::ClampedLogs { .. }))
            .cloned()
            .collect();
        assert_eq!(
            clamped,
            [
                Warning::ClampedLogs { step: 1, count: 1 },
                Warning::ClampedLogs { step: 3, count: 1 }
            ]
        );
    }

    /// Under `OnNonFinite::Stop`, `Trainer::step` returns the error at the step that met
    /// it, skips the update, counts the step and sends the warning to the sink; nothing is
    /// held back for `Trainer::end_epoch`.
    #[test]
    fn trainer_step_returns_the_error_at_the_step() {
        let data = nan_dataset();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let before = parameter_bits(&net, &params);
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 32);
        let (seen, sink) = collecting_sink();
        trainer.set_health(health(OnNonFinite::Stop, sink));
        let batcher = Batcher::new(64, 32, None);
        let mut batch = Batch::new(&data, 32);
        let err = trainer
            .step(&net, &mut params, batcher.gather(&data, 0..32, &mut batch))
            .unwrap_err();
        assert!(is_non_finite(&err, 1, "loss"), "{err:?}");
        assert_eq!(parameter_bits(&net, &params), before);
        assert_eq!(trainer.steps(), 1);
        assert_eq!(
            *seen.borrow(),
            [Warning::NonFinite {
                step: 1,
                location: "loss".to_string()
            }]
        );
        assert!(trainer.end_epoch(1).is_ok());
        // The second batch has no NaN, so the next step trains.
        let loss = trainer
            .step(&net, &mut params, batcher.gather(&data, 32..64, &mut batch))
            .unwrap();
        assert!(loss.is_finite());
        assert_ne!(parameter_bits(&net, &params), before);
    }

    /// Each warning reads as one plain-English line.
    #[test]
    fn warnings_display() {
        let lines = [
            Warning::ClampedLogs { step: 3, count: 2 }.to_string(),
            Warning::NonFinite {
                step: 1,
                location: "loss".to_string(),
            }
            .to_string(),
            Warning::Exploding {
                step: 2,
                layer: "0.weight".to_string(),
                norm: 2.5e4,
            }
            .to_string(),
            Warning::Vanishing {
                step: 2,
                layer: "2.bias".to_string(),
                norm: 0.0,
            }
            .to_string(),
            Warning::DeadNeurons {
                epoch: 1,
                layer: 1,
                neurons: vec![0, 3],
            }
            .to_string(),
        ];
        assert_eq!(
            lines,
            [
                "step 3: binary cross-entropy met 2 probabilities of exactly 0 or 1 and clamped their logarithms to -100",
                "step 1: a NaN or infinite value in loss",
                "step 2: the gradient norm of 0.weight is 2.500e4, above the exploding limit",
                "step 2: the gradient norm of 2.bias is 0.000e0, below the vanishing limit",
                "epoch 1: 2 neurons of layer 1 output 0 for every sample (dead ReLU): [0, 3]",
            ]
        );
    }
}
