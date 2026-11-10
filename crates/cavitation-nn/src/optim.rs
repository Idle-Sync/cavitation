//! Optimisers: SGD (with momentum and weight decay), Adam and AdamW, and the gradient
//! statistics they collect while they update.
//!
//! All arithmetic is done per element in `f64` and stored back as `T`, so an `f32` run
//! rounds once per stored value. Momentum and Adam's moment estimates are stored as `T`.

use cavitation_core::{ParamId, ParamStore, Real};

/// What the optimiser saw in the gradients during a step: for every parameter the sum of the
/// squares of its gradient (in `f64`) and whether every value was finite.
///
/// The optimiser fills it in the same pass that updates the parameters, so reading the
/// gradients costs no extra pass. A stale gradient (no backward pass since
/// [`ParamStore::zero_grad`]) counts as zeros. A `GradStats` can be reused from step to step;
/// after the first step it allocates nothing.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{GradStats, Optimizer, Sgd};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::zeros(&[2]));
/// store.grads.accumulate(p).0.data_mut().copy_from_slice(&[3.0, 4.0]);
/// let mut stats = GradStats::new();
/// Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
/// assert_eq!(stats.norm(p), 5.0);
/// assert!(stats.all_finite());
/// ```
#[derive(Clone, Debug, Default)]
pub struct GradStats {
    /// The parameters seen in the last step, in the order of the store.
    ids: Vec<ParamId>,
    /// The sum of squares of each parameter's gradient.
    sum_squares: Vec<f64>,
    /// Whether every gradient value of each parameter was finite.
    finite: Vec<bool>,
}

impl GradStats {
    /// Empty statistics, to be filled by [`Optimizer::step_with_stats`].
    ///
    /// ```
    /// let stats = cavitation_nn::GradStats::new();
    /// assert!(stats.all_finite());
    /// ```
    pub fn new() -> GradStats {
        GradStats::default()
    }

    /// Forgets the last step and makes room for `n` parameters.
    fn reset(&mut self, n: usize) {
        self.ids.clear();
        self.sum_squares.clear();
        self.finite.clear();
        self.ids.reserve(n);
        self.sum_squares.reserve(n);
        self.finite.reserve(n);
    }

    /// Records one parameter.
    fn push(&mut self, id: ParamId, sum_squares: f64, finite: bool) {
        self.ids.push(id);
        self.sum_squares.push(sum_squares);
        self.finite.push(finite);
    }

    /// The Euclidean norm of the gradient of `id` in the last step: the square root of its
    /// sum of squares.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not seen in the last step.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{GradStats, Optimizer, Sgd};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[1]));
    /// store.grads.accumulate(p).0.data_mut()[0] = -2.0;
    /// let mut stats = GradStats::new();
    /// Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
    /// assert_eq!(stats.norm(p), 2.0);
    /// ```
    pub fn norm(&self, id: ParamId) -> f64 {
        match self.ids.iter().position(|&i| i == id) {
            Some(k) => self.sum_squares[k].sqrt(),
            None => panic!("GradStats::norm: {id:?} was not seen in the last step"),
        }
    }

    /// Whether every gradient value of every parameter was finite.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{GradStats, Optimizer, Sgd};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[1]));
    /// store.grads.accumulate(p).0.data_mut()[0] = f32::NAN;
    /// let mut stats = GradStats::new();
    /// Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
    /// assert!(!stats.all_finite());
    /// ```
    pub fn all_finite(&self) -> bool {
        self.finite.iter().all(|&f| f)
    }

    /// The first parameter, in the order of the store, whose gradient held a NaN or an
    /// infinity; `None` when all were finite.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{GradStats, Optimizer, Sgd};
    /// let mut store = ParamStore::new();
    /// let a = store.add(Tensor::zeros(&[1]));
    /// let b = store.add(Tensor::zeros(&[1]));
    /// store.grads.accumulate(b).0.data_mut()[0] = f32::INFINITY;
    /// let mut stats = GradStats::new();
    /// Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
    /// assert_eq!(stats.first_non_finite(), Some(b));
    /// # let _ = a;
    /// ```
    pub fn first_non_finite(&self) -> Option<ParamId> {
        self.finite.iter().position(|&f| !f).map(|k| self.ids[k])
    }
}

/// Changes parameters from their gradients.
///
/// An optimiser keeps its state (momentum, moment estimates) in buffers it allocates on the
/// first step and reuses afterwards. The parameter store must not change its number or the
/// sizes of its parameters after that: a later step panics if it does.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{Optimizer, Sgd};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// store.grads.accumulate(p).0.data_mut()[0] = 0.5;
/// let mut opt = Sgd::new(0.1);
/// opt.step(&mut store);
/// assert_eq!(store.value(p).data(), &[0.95]);
/// ```
pub trait Optimizer<T: Real>: Send {
    /// Updates every parameter from its gradient and fills `stats` with the gradients'
    /// sums of squares and finiteness, in the same pass.
    ///
    /// # Panics
    ///
    /// Panics if the store's parameters differ in number or size from the first step.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{GradStats, Optimizer, Adam};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[2], 1.0f32));
    /// store.grads.accumulate(p).0.data_mut().fill(1.0);
    /// let mut stats = GradStats::new();
    /// Adam::new(0.01).step_with_stats(&mut store, &mut stats);
    /// assert!((stats.norm(p) - 2f64.sqrt()).abs() < 1e-12);
    /// ```
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats);

    /// Updates every parameter from its gradient. The statistics go into a [`GradStats`] the
    /// optimiser owns, allocated on the first step and reused, so after the first step this
    /// allocates nothing; read them with [`last_stats`](Optimizer::last_stats).
    ///
    /// # Panics
    ///
    /// Panics if the store's parameters differ in number or size from the first step.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{Optimizer, Sgd};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 1.0f32));
    /// store.grads.accumulate(p).0.data_mut()[0] = 1.0;
    /// let mut opt = Sgd::new(0.5);
    /// opt.step(&mut store);
    /// assert_eq!(store.value(p).data(), &[0.5]);
    /// assert_eq!(opt.last_stats().norm(p), 1.0);
    /// ```
    fn step(&mut self, params: &mut ParamStore<T>);

    /// The statistics of the last [`step`](Optimizer::step) (empty before the first). A step
    /// taken with [`step_with_stats`](Optimizer::step_with_stats) fills the caller's
    /// `GradStats` instead and leaves this one unchanged.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{Adam, Optimizer};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[2]));
    /// store.grads.accumulate(p).0.data_mut().copy_from_slice(&[3.0, 4.0]);
    /// let mut opt = Adam::new(0.01);
    /// opt.step(&mut store);
    /// assert_eq!(opt.last_stats().norm(p), 5.0);
    /// ```
    fn last_stats(&self) -> &GradStats;

    /// The current learning rate.
    ///
    /// ```
    /// use cavitation_nn::{Optimizer, Sgd};
    /// assert_eq!(Optimizer::<f32>::learning_rate(&Sgd::new(0.1)), 0.1);
    /// ```
    fn learning_rate(&self) -> f64;

    /// Sets the learning rate used by later steps.
    ///
    /// ```
    /// use cavitation_nn::{Optimizer, Sgd};
    /// let mut opt = Sgd::new(0.1);
    /// Optimizer::<f32>::set_learning_rate(&mut opt, 0.01);
    /// assert_eq!(Optimizer::<f32>::learning_rate(&opt), 0.01);
    /// ```
    fn set_learning_rate(&mut self, lr: f64);
}

impl<T: Real, O: Optimizer<T> + ?Sized> Optimizer<T> for Box<O> {
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats) {
        (**self).step_with_stats(params, stats);
    }

    fn step(&mut self, params: &mut ParamStore<T>) {
        (**self).step(params);
    }

    fn last_stats(&self) -> &GradStats {
        (**self).last_stats()
    }

    fn learning_rate(&self) -> f64 {
        (**self).learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f64) {
        (**self).set_learning_rate(lr);
    }
}

/// The parameter sizes seen on the first step, checked on every later step.
#[derive(Clone, Debug, Default)]
struct Sizes {
    /// Whether the first step has happened.
    seen: bool,
    /// The element count of each parameter, in the order of the store.
    lens: Vec<usize>,
}

impl Sizes {
    /// Records the sizes on the first call (returns `true`); on later calls panics unless the
    /// store has the same number of parameters of the same sizes (returns `false`). Allocates
    /// only on the first call.
    fn prepare<T: Real>(&mut self, name: &str, params: &mut ParamStore<T>) -> bool {
        if !self.seen {
            self.lens
                .extend(params.iter_mut().map(|(_, v, _)| v.data().len()));
            self.seen = true;
            return true;
        }
        let now = params.len();
        assert!(
            now == self.lens.len(),
            "{name}: the store has {now} parameters but the first step saw {}; \
             an optimiser's state is sized on its first step",
            self.lens.len()
        );
        for (k, (_, v, _)) in params.iter_mut().enumerate() {
            assert!(
                v.data().len() == self.lens[k],
                "{name}: parameter {k} has {} elements but the first step saw {}; \
                 an optimiser's state is sized on its first step",
                v.data().len(),
                self.lens[k]
            );
        }
        false
    }
}

/// Stochastic gradient descent with optional momentum and weight decay, with PyTorch's
/// `torch.optim.SGD` semantics (dampening 0, no Nesterov).
///
/// With weight decay `wd` the gradient becomes `g + wd·p`. With momentum `μ` the buffer is
/// `b ← μ·b + g` (`b = g` on the first step) and the parameter moves by `−lr·b`.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{Optimizer, Sgd};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// let mut opt = Sgd::new(0.1).with_momentum(0.9);
/// for _ in 0..2 {
///     store.zero_grad();
///     store.grads.accumulate(p).0.data_mut()[0] = 1.0;
///     opt.step(&mut store);
/// }
/// // b = 1, then b = 1.9: 1 - 0.1 - 0.19
/// assert!((store.value(p).data()[0] - 0.71).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct Sgd<T: Real = f32> {
    /// The learning rate.
    lr: f64,
    /// The momentum factor; 0 disables the buffer.
    momentum: f64,
    /// The weight decay added to the gradient.
    weight_decay: f64,
    /// The sizes seen on the first step.
    sizes: Sizes,
    /// The momentum buffer of each parameter (empty without momentum).
    buffers: Vec<Vec<T>>,
    /// The statistics of the last `step`.
    stats: GradStats,
}

impl<T: Real> Sgd<T> {
    /// Plain gradient descent with learning rate `lr`.
    ///
    /// ```
    /// let opt: cavitation_nn::Sgd = cavitation_nn::Sgd::new(0.1);
    /// # let _ = opt;
    /// ```
    pub fn new(lr: f64) -> Sgd<T> {
        Sgd {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            sizes: Sizes::default(),
            buffers: Vec::new(),
            stats: GradStats::new(),
        }
    }

    /// Sets the momentum factor.
    ///
    /// ```
    /// let opt: cavitation_nn::Sgd = cavitation_nn::Sgd::new(0.1).with_momentum(0.9);
    /// # let _ = opt;
    /// ```
    pub fn with_momentum(mut self, momentum: f64) -> Sgd<T> {
        self.momentum = momentum;
        self
    }

    /// Sets the weight decay, added to the gradient as `wd·p`.
    ///
    /// ```
    /// let opt: cavitation_nn::Sgd = cavitation_nn::Sgd::new(0.1).with_weight_decay(1e-4);
    /// # let _ = opt;
    /// ```
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Sgd<T> {
        self.weight_decay = weight_decay;
        self
    }
}

impl<T: Real> Optimizer<T> for Sgd<T> {
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats) {
        let first = self.sizes.prepare("Sgd", params);
        if first && self.momentum != 0.0 {
            self.buffers = self.sizes.lens.iter().map(|&n| vec![T::ZERO; n]).collect();
        }
        stats.reset(params.len());
        let (lr, mu, wd) = (self.lr, self.momentum, self.weight_decay);
        for (k, (id, value, grad)) in params.iter_mut().enumerate() {
            let (mut sum, mut finite) = (0.0f64, true);
            let values = value.data_mut();
            let grads = grad.data();
            for i in 0..values.len() {
                let g = grads[i].to_f64();
                sum += g * g;
                finite &= g.is_finite();
                let p = values[i].to_f64();
                let mut step = g;
                if wd != 0.0 {
                    step += wd * p;
                }
                if mu != 0.0 {
                    let buf = &mut self.buffers[k][i];
                    step = if first {
                        step
                    } else {
                        mu * buf.to_f64() + step
                    };
                    *buf = T::from_f64(step);
                }
                values[i] = T::from_f64(p - lr * step);
            }
            stats.push(id, sum, finite);
        }
    }

    fn step(&mut self, params: &mut ParamStore<T>) {
        let mut stats = std::mem::take(&mut self.stats);
        self.step_with_stats(params, &mut stats);
        self.stats = stats;
    }

    fn last_stats(&self) -> &GradStats {
        &self.stats
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// The state and update shared by [`Adam`] and [`AdamW`].
#[derive(Clone, Debug)]
struct AdamCore<T: Real> {
    /// The learning rate.
    lr: f64,
    /// The decay rate of the first moment.
    beta1: f64,
    /// The decay rate of the second moment.
    beta2: f64,
    /// Added to the square root of the second moment, outside the root.
    eps: f64,
    /// The weight decay.
    weight_decay: f64,
    /// The steps taken so far.
    steps: i32,
    /// The sizes seen on the first step.
    sizes: Sizes,
    /// The first moment of each parameter.
    m: Vec<Vec<T>>,
    /// The second moment of each parameter.
    v: Vec<Vec<T>>,
    /// The statistics of the last `step`.
    stats: GradStats,
}

impl<T: Real> AdamCore<T> {
    /// The defaults of `torch.optim.Adam` with learning rate `lr` and weight decay `wd`.
    fn new(lr: f64, weight_decay: f64) -> AdamCore<T> {
        AdamCore {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            steps: 0,
            sizes: Sizes::default(),
            m: Vec::new(),
            v: Vec::new(),
            stats: GradStats::new(),
        }
    }

    /// Checks the decay rates.
    ///
    /// # Panics
    ///
    /// Panics unless both are in `[0, 1)`.
    fn set_betas(&mut self, beta1: f64, beta2: f64) {
        assert!(
            (0.0..1.0).contains(&beta1) && (0.0..1.0).contains(&beta2),
            "Adam betas must be in [0, 1), got ({beta1}, {beta2})"
        );
        self.beta1 = beta1;
        self.beta2 = beta2;
    }

    /// One step. `decoupled` selects AdamW's decay (`p ← p·(1 − lr·wd)` before the update)
    /// over Adam's (`wd·p` added to the gradient).
    fn step(
        &mut self,
        name: &str,
        decoupled: bool,
        params: &mut ParamStore<T>,
        stats: &mut GradStats,
    ) {
        if self.sizes.prepare(name, params) {
            self.m = self.sizes.lens.iter().map(|&n| vec![T::ZERO; n]).collect();
            self.v = self.sizes.lens.iter().map(|&n| vec![T::ZERO; n]).collect();
        }
        self.steps += 1;
        stats.reset(params.len());
        let (lr, b1, b2, eps, wd) = (self.lr, self.beta1, self.beta2, self.eps, self.weight_decay);
        let correction1 = 1.0 - b1.powi(self.steps);
        let correction2 = 1.0 - b2.powi(self.steps);
        let shrink = 1.0 - lr * wd;
        for (k, (id, value, grad)) in params.iter_mut().enumerate() {
            let (mut sum, mut finite) = (0.0f64, true);
            let values = value.data_mut();
            let grads = grad.data();
            let (m, v) = (&mut self.m[k], &mut self.v[k]);
            for i in 0..values.len() {
                let raw = grads[i].to_f64();
                sum += raw * raw;
                finite &= raw.is_finite();
                let mut p = values[i].to_f64();
                let mut g = raw;
                if decoupled {
                    p *= shrink;
                } else if wd != 0.0 {
                    g += wd * p;
                }
                let mi = b1 * m[i].to_f64() + (1.0 - b1) * g;
                let vi = b2 * v[i].to_f64() + (1.0 - b2) * g * g;
                m[i] = T::from_f64(mi);
                v[i] = T::from_f64(vi);
                let m_hat = mi / correction1;
                let v_hat = vi / correction2;
                values[i] = T::from_f64(p - lr * m_hat / (v_hat.sqrt() + eps));
            }
            stats.push(id, sum, finite);
        }
    }
}

/// Adam (Kingma and Ba, 2015) with PyTorch's `torch.optim.Adam` semantics: bias-corrected
/// moments, `eps` added outside the square root, and weight decay added to the gradient.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{Adam, Optimizer};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// store.grads.accumulate(p).0.data_mut()[0] = 3.0;
/// Adam::new(0.01).step(&mut store);
/// // The first step moves by lr in the direction of the gradient.
/// assert!((store.value(p).data()[0] - 0.99).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct Adam<T: Real = f32> {
    /// The shared state and update.
    core: AdamCore<T>,
}

impl<T: Real> Adam<T> {
    /// Adam with learning rate `lr`, betas (0.9, 0.999), eps 1e-8 and no weight decay.
    ///
    /// ```
    /// let opt: cavitation_nn::Adam = cavitation_nn::Adam::new(0.001);
    /// # let _ = opt;
    /// ```
    pub fn new(lr: f64) -> Adam<T> {
        Adam {
            core: AdamCore::new(lr, 0.0),
        }
    }

    /// Sets the decay rates of the two moments.
    ///
    /// # Panics
    ///
    /// Panics unless both are in `[0, 1)`.
    ///
    /// ```
    /// let opt: cavitation_nn::Adam = cavitation_nn::Adam::new(0.001).with_betas(0.8, 0.99);
    /// # let _ = opt;
    /// ```
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Adam<T> {
        self.core.set_betas(beta1, beta2);
        self
    }

    /// Sets the constant added to the square root of the second moment.
    ///
    /// ```
    /// let opt: cavitation_nn::Adam = cavitation_nn::Adam::new(0.001).with_eps(1e-6);
    /// # let _ = opt;
    /// ```
    pub fn with_eps(mut self, eps: f64) -> Adam<T> {
        self.core.eps = eps;
        self
    }

    /// Sets the weight decay, added to the gradient as `wd·p` (coupled, so it is scaled by
    /// the adaptive step like any gradient).
    ///
    /// ```
    /// let opt: cavitation_nn::Adam = cavitation_nn::Adam::new(0.001).with_weight_decay(0.1);
    /// # let _ = opt;
    /// ```
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Adam<T> {
        self.core.weight_decay = weight_decay;
        self
    }
}

impl<T: Real> Optimizer<T> for Adam<T> {
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats) {
        self.core.step("Adam", false, params, stats);
    }

    fn step(&mut self, params: &mut ParamStore<T>) {
        let mut stats = std::mem::take(&mut self.core.stats);
        self.step_with_stats(params, &mut stats);
        self.core.stats = stats;
    }

    fn last_stats(&self) -> &GradStats {
        &self.core.stats
    }

    fn learning_rate(&self) -> f64 {
        self.core.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.core.lr = lr;
    }
}

/// AdamW (Loshchilov and Hutter, 2019) with PyTorch's `torch.optim.AdamW` semantics: the
/// weight decay is decoupled, `p ← p·(1 − lr·wd)` before the Adam update, and the default
/// weight decay is 0.01.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{AdamW, Optimizer};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// // No gradient: only the decay acts, 1 - 0.1 * 0.01 * 1.
/// AdamW::new(0.1).step(&mut store);
/// assert!((store.value(p).data()[0] - 0.999).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct AdamW<T: Real = f32> {
    /// The shared state and update.
    core: AdamCore<T>,
}

impl<T: Real> AdamW<T> {
    /// AdamW with learning rate `lr`, betas (0.9, 0.999), eps 1e-8 and weight decay 0.01.
    ///
    /// ```
    /// let opt: cavitation_nn::AdamW = cavitation_nn::AdamW::new(0.001);
    /// # let _ = opt;
    /// ```
    pub fn new(lr: f64) -> AdamW<T> {
        AdamW {
            core: AdamCore::new(lr, 0.01),
        }
    }

    /// Sets the decay rates of the two moments.
    ///
    /// # Panics
    ///
    /// Panics unless both are in `[0, 1)`.
    ///
    /// ```
    /// let opt: cavitation_nn::AdamW = cavitation_nn::AdamW::new(0.001).with_betas(0.8, 0.99);
    /// # let _ = opt;
    /// ```
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> AdamW<T> {
        self.core.set_betas(beta1, beta2);
        self
    }

    /// Sets the constant added to the square root of the second moment.
    ///
    /// ```
    /// let opt: cavitation_nn::AdamW = cavitation_nn::AdamW::new(0.001).with_eps(1e-6);
    /// # let _ = opt;
    /// ```
    pub fn with_eps(mut self, eps: f64) -> AdamW<T> {
        self.core.eps = eps;
        self
    }

    /// Sets the decoupled weight decay.
    ///
    /// ```
    /// let opt: cavitation_nn::AdamW = cavitation_nn::AdamW::new(0.001).with_weight_decay(0.1);
    /// # let _ = opt;
    /// ```
    pub fn with_weight_decay(mut self, weight_decay: f64) -> AdamW<T> {
        self.core.weight_decay = weight_decay;
        self
    }
}

impl<T: Real> Optimizer<T> for AdamW<T> {
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats) {
        self.core.step("AdamW", true, params, stats);
    }

    fn step(&mut self, params: &mut ParamStore<T>) {
        let mut stats = std::mem::take(&mut self.core.stats);
        self.step_with_stats(params, &mut stats);
        self.core.stats = stats;
    }

    fn last_stats(&self) -> &GradStats {
        &self.core.stats
    }

    fn learning_rate(&self) -> f64 {
        self.core.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.core.lr = lr;
    }
}

/// Tests for the optimisers.
#[cfg(test)]
mod tests {
    use super::*;
    use cavitation_core::Tensor;

    /// Sets the gradient of `p` to `values`.
    fn set_grad(store: &mut ParamStore, p: ParamId, values: &[f32]) {
        store
            .grads
            .accumulate(p)
            .0
            .data_mut()
            .copy_from_slice(values);
    }

    /// p = 1, g = 0.5, lr 0.1 gives 0.95.
    #[test]
    fn sgd_by_hand() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[1], 1.0f32));
        set_grad(&mut store, p, &[0.5]);
        Sgd::new(0.1).step(&mut store);
        assert!((store.value(p).data()[0] - 0.95).abs() < 1e-7);
    }

    /// The first Adam step moves every parameter by lr in the direction of its gradient.
    #[test]
    fn adam_first_step_moves_by_lr() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::from_vec(vec![1.0f32, -2.0, 0.5, 3.0], &[4]).unwrap());
        let grads = [0.3f32, -7.0, 1e-3, -200.0];
        set_grad(&mut store, p, &grads);
        let before = store.value(p).data().to_vec();
        Adam::new(0.01).step(&mut store);
        for i in 0..4 {
            let moved = store.value(p).data()[i] - before[i];
            assert!(
                (moved - (-0.01 * grads[i].signum())).abs() < 1e-6,
                "element {i}"
            );
        }
    }

    /// The statistics equal a direct computation, report the first non-finite parameter and
    /// treat a stale gradient as zeros.
    #[test]
    fn grad_stats_match_direct_computation() {
        let mut store = ParamStore::new();
        let a = store.add(Tensor::zeros(&[3]));
        let b = store.add(Tensor::zeros(&[2]));
        let c = store.add(Tensor::zeros(&[2]));
        let d = store.add(Tensor::zeros(&[1]));
        set_grad(&mut store, a, &[1.0, -2.0, 2.0]);
        set_grad(&mut store, c, &[f32::NAN, 1.0]);
        set_grad(&mut store, d, &[f32::NEG_INFINITY]);
        let mut stats = GradStats::new();
        Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
        assert_eq!(stats.norm(a), 3.0);
        assert_eq!(stats.norm(b), 0.0);
        assert!(stats.norm(c).is_nan());
        assert!(!stats.all_finite());
        assert_eq!(stats.first_non_finite(), Some(c));
        store.zero_grad();
        Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
        assert!(stats.all_finite());
        assert_eq!(stats.first_non_finite(), None);
        assert_eq!(stats.norm(a), 0.0);
    }

    /// Changing the number of parameters after the first step panics with a clear message.
    #[test]
    #[should_panic(expected = "the store has 2 parameters but the first step saw 1")]
    fn adding_a_parameter_after_the_first_step_panics() {
        let mut store = ParamStore::new();
        store.add(Tensor::zeros(&[2]));
        let mut opt = Adam::new(0.01);
        opt.step(&mut store);
        store.add(Tensor::zeros(&[2]));
        opt.step(&mut store);
    }

    /// Each optimiser returns the rate it was given, and its next step uses it: SGD moves by
    /// `lr·g`, Adam's first step moves by `lr`, and AdamW's decoupled decay shrinks by
    /// `lr·wd` (PyTorch's `p ← p·(1 − lr·wd)` with the group's current rate).
    #[test]
    fn set_learning_rate_round_trips_and_applies() {
        let one = || {
            let mut store = ParamStore::new();
            let p = store.add(Tensor::full(&[1], 1.0f32));
            (store, p)
        };

        let (mut store, p) = one();
        let mut sgd = Sgd::new(0.1);
        Optimizer::<f32>::set_learning_rate(&mut sgd, 0.25);
        assert_eq!(Optimizer::<f32>::learning_rate(&sgd), 0.25);
        set_grad(&mut store, p, &[1.0]);
        sgd.step(&mut store);
        assert_eq!(store.value(p).data(), &[0.75]);

        let (mut store, p) = one();
        let mut adam = Adam::new(0.1);
        Optimizer::<f32>::set_learning_rate(&mut adam, 0.03);
        assert_eq!(Optimizer::<f32>::learning_rate(&adam), 0.03);
        set_grad(&mut store, p, &[2.0]);
        adam.step(&mut store);
        assert!((store.value(p).data()[0] - 0.97).abs() < 1e-6);

        let (mut store, p) = one();
        let mut adamw = AdamW::new(0.1);
        Optimizer::<f32>::set_learning_rate(&mut adamw, 0.2);
        assert_eq!(Optimizer::<f32>::learning_rate(&adamw), 0.2);
        // No gradient: only the decay acts, 1 − 0.2 · 0.01.
        adamw.step(&mut store);
        assert!((store.value(p).data()[0] - 0.998).abs() < 1e-7);
    }

    /// A `Box<dyn Optimizer>` is an optimiser.
    #[test]
    fn boxed_optimizer_steps() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[1], 1.0f32));
        set_grad(&mut store, p, &[1.0]);
        let mut opt: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(0.5));
        opt.set_learning_rate(0.25);
        opt.step(&mut store);
        assert_eq!(opt.learning_rate(), 0.25);
        assert_eq!(store.value(p).data(), &[0.75]);
    }
}
