//! The optimiser kernel on the card against the CPU optimisers of `cavitation_nn`, as
//! `cavitation-gpu`'s tests check its WGSL kernel: SGD (plain, with weight decay, with
//! momentum, with both), Adam (with and without weight decay) and AdamW, 50 updates of seven
//! parameters whose lengths end anywhere in a block, some parameters skipping some steps.
//! Each step is checked from the card's own starting state against the exact result of the
//! CPU's arithmetic with the card's `f32` constants, within the bounds `OptimizerKernel`
//! states, and against the kernels' `f32` expressions evaluated on the host, bit for bit;
//! that exact arithmetic is pinned to the real CPU optimisers bit for bit. Skipped steps,
//! update counts, NaN and ±∞ gradients, the `cifar-cnn` network's 1,117,162 parameters,
//! 65,536 one-element parameters and the refusals are tested too.
//!
//! Every test but the CPU's skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1` makes
//! that a failure.

mod common;

use cavitation_core::{ParamId, ParamStore, Tensor};
use cavitation_cuda::{Cuda, CudaError, DeviceBuffer, OptimizerKernel, OptimizerRule};
use cavitation_nn::{Adam, AdamW, Optimizer, Sgd};
use common::{
    FLOOR, Seeded, U, assert_exact, assert_within, nans, panic_message, read, seeded_values, upload,
};

/// The parameters' lengths. Each block updates one parameter's run of 256 elements, and
/// these end it at every kind of place: a 232-element tail (1000), 37, a single element, a
/// 44-element tail (300), exactly one block (256), one past (257) and one short (255).
const LENS: [usize; 7] = [1000, 37, 1, 300, 256, 257, 255];

/// Updates per twin.
const STEPS: usize = 50;

/// The parameters of the `cifar-cnn` network in the order it adds them: four 3 × 3
/// convolutions without a bias, each followed by a batch norm's `γ` and `β`, then
/// `Linear(4096 → 256)` and `Linear(256 → 10)` with their biases; 1,117,162 floats.
const CIFAR: [usize; 16] = [
    3 * 32 * 9,
    32,
    32,
    32 * 32 * 9,
    32,
    32,
    32 * 64 * 9,
    64,
    64,
    64 * 64 * 9,
    64,
    64,
    4096 * 256,
    256,
    256 * 10,
    10,
];

/// The rules the twins run.
fn rules() -> Vec<OptimizerRule> {
    let sgd = |lr, momentum, weight_decay| OptimizerRule::Sgd {
        lr,
        momentum,
        weight_decay,
    };
    vec![
        sgd(0.1, 0.0, 0.0),
        sgd(0.05, 0.0, 1e-2),
        sgd(0.05, 0.9, 0.0),
        sgd(0.03, 0.9, 5e-3),
        OptimizerRule::adam(1e-3, 0.0),
        OptimizerRule::adam(1e-2, 1e-2),
        OptimizerRule::adam_w(1e-3, 1e-2),
        OptimizerRule::AdamW {
            lr: 3e-3,
            beta1: 0.8,
            beta2: 0.99,
            eps: 1e-6,
            weight_decay: 0.1,
        },
    ]
}

/// Whether parameter `k` has a gradient at step `t`: the single-element parameter misses
/// every fourth step, and the 256-element one the first three, so their update counts (and
/// Adam's bias corrections) run behind the others'.
fn active(k: usize, t: usize) -> bool {
    !(k == 2 && t % 4 == 1 || k == 4 && t < 3)
}

/// Step `t`'s gradients for `total` elements: seeded values spread over sixteen binades, with
/// every 53rd tiny enough (about 1e-22) that its square is subnormal, so Adam's second moment
/// reaches the flush to zero.
fn gradients_of(seed: u64, t: usize, total: usize) -> Vec<f32> {
    let mut rng = Seeded::new(seed * 1000 + t as u64);
    (0..total)
        .map(|i| {
            if i % 53 == 0 {
                return rng.uniform(-1.0, 1.0) * 1e-22;
            }
            let scale = 2f32.powi((rng.next_u64() >> 60) as i32 - 12);
            rng.uniform(-1.0, 1.0) * scale
        })
        .collect()
}

/// Step `t`'s gradients for [`LENS`].
fn gradients(seed: u64, t: usize) -> Vec<f32> {
    gradients_of(seed, t, LENS.iter().sum())
}

/// One step's constants for a parameter on its update `t` (from 1), in `f64`: either the
/// CPU's own (computed in `f64`) or the card's (each rounded to `f32` as the kernel's host
/// side rounds it).
#[derive(Clone, Copy, Debug)]
struct Constants {
    /// The learning rate.
    lr: f64,
    /// SGD's momentum.
    momentum: f64,
    /// The weight decay.
    wd: f64,
    /// Adam's first-moment decay.
    beta1: f64,
    /// Adam's second-moment decay.
    beta2: f64,
    /// `1 − β₁`.
    one_minus_beta1: f64,
    /// `1 − β₂`.
    one_minus_beta2: f64,
    /// Adam's eps.
    eps: f64,
    /// AdamW's `1 − lr·wd`.
    shrink: f64,
    /// `lr / (1 − β₁ᵗ)`.
    step_size: f64,
    /// `1 / √(1 − β₂ᵗ)`.
    inv_sqrt_correction2: f64,
}

impl Constants {
    /// The constants of `rule` for update `t`, rounded to `f32` when `card`.
    fn new(rule: &OptimizerRule, t: u32, card: bool) -> Constants {
        let (lr, momentum, wd, beta1, beta2, eps) = match *rule {
            OptimizerRule::Sgd {
                lr,
                momentum,
                weight_decay,
            } => (lr, momentum, weight_decay, 0.0, 0.0, 0.0),
            OptimizerRule::Adam {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            }
            | OptimizerRule::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => (lr, 0.0, weight_decay, beta1, beta2, eps),
        };
        let round = |v: f64| if card { f64::from(v as f32) } else { v };
        // The kernels take one minus each beta and the bias corrections from the beta as
        // they hold it, rounded to `f32`.
        let (beta1, beta2) = (round(beta1), round(beta2));
        Constants {
            lr: round(lr),
            momentum: round(momentum),
            wd: round(wd),
            beta1: round(beta1),
            beta2: round(beta2),
            one_minus_beta1: round(1.0 - beta1),
            one_minus_beta2: round(1.0 - beta2),
            eps: round(eps),
            shrink: round(1.0 - lr * wd),
            step_size: round(lr / (1.0 - beta1.powi(t as i32))),
            inv_sqrt_correction2: round(1.0 / (1.0 - beta2.powi(t as i32)).sqrt()),
        }
    }
}

/// The state the CPU stores for the next step: 0 below the smallest normal `f32`.
fn store_state(x: f64) -> f32 {
    if x.abs() < f64::from(f32::MIN_POSITIVE) {
        0.0
    } else {
        x as f32
    }
}

/// The state the kernels store for the next step: 0 below the smallest normal `f32`.
fn store_state_f32(x: f32) -> f32 {
    if x.to_bits() & 0x7f80_0000 == 0 {
        0.0
    } else {
        x
    }
}

/// The largest error of `got` against `reference` as a share of one rounding, `u·|p'|`
/// plus the floor.
fn one_rounding_share(got: &[f32], reference: &[f64]) -> f64 {
    got.iter()
        .zip(reference)
        .map(|(&g, &r)| (f64::from(g) - r).abs() / (U * r.abs() + FLOOR))
        .fold(0.0, f64::max)
}

/// Picks one of an [`Exact`]'s results and its bound.
type Pick = fn(&Exact) -> (f64, f64);

/// One element's update in exact (`f64`) arithmetic, with the bound of the kernels' `f32`
/// arithmetic against it for the value and the two state words.
#[derive(Clone, Copy, Debug, Default)]
struct Exact {
    /// The new value.
    value: f64,
    /// The new momentum buffer or first moment.
    m: f64,
    /// The new second moment.
    v: f64,
    /// The value's bound.
    value_bound: f64,
    /// The buffer's or first moment's bound.
    m_bound: f64,
    /// The second moment's bound.
    v_bound: f64,
}

/// The CPU optimisers' formulas for one element (`sgd_reference` and `adam_reference` in
/// `cavitation_nn::optim`) evaluated exactly in `f64`: the value `p`, its state `m` and `v`,
/// its gradient `g`, and whether this is its first update, with constants `c`. With the CPU's
/// own `f64` constants, stored as the CPU stores them, this is the CPU optimisers' bits
/// ([`the_exact_arithmetic_is_the_cpu_optimisers`]); the card's twins evaluate it with the
/// card's `f32` constants instead, so their bounds measure the kernels against the CPU's
/// formulas, and [`the_card_against_the_cpu_optimisers`] measures the drift the constants'
/// rounding adds. The bounds are those `OptimizerKernel`'s docs state, with 5% to spare for
/// second-order terms; plain SGD's is one rounding, `u·|p'|`.
fn exact(
    rule: &OptimizerRule,
    c: &Constants,
    first: bool,
    p: f32,
    m: f32,
    v: f32,
    g: f32,
) -> Exact {
    let (p, g) = (f64::from(p), f64::from(g));
    match rule {
        OptimizerRule::Sgd { .. } => {
            let decay = (c.wd * p).abs();
            let mut step = g;
            if c.wd != 0.0 {
                step += c.wd * p;
            }
            let s = step;
            let mut carried = 0.0;
            if c.momentum != 0.0 && !first {
                carried = (c.momentum * f64::from(m)).abs();
                step += c.momentum * f64::from(m);
            }
            let value = p - c.lr * step;
            let plain = c.wd == 0.0 && c.momentum == 0.0;
            let buffer = U * (step.abs() + carried + s.abs() + decay);
            let value_bound = if plain {
                U * value.abs()
            } else {
                1.05 * (U * (value.abs() + c.lr * step.abs()) + c.lr * buffer)
            };
            Exact {
                value,
                m: step,
                v: 0.0,
                value_bound,
                m_bound: 1.05 * buffer,
                v_bound: 0.0,
            }
        }
        _ => {
            let decoupled = matches!(rule, OptimizerRule::AdamW { .. });
            let decay = if decoupled {
                0.0
            } else {
                (c.wd * f64::from(p as f32)).abs()
            };
            let (mut p, mut g) = (p, g);
            if decoupled {
                p *= c.shrink;
            } else if c.wd != 0.0 {
                g += c.wd * p;
            }
            let mi = c.beta1 * f64::from(m) + c.one_minus_beta1 * g;
            let vi = c.beta2 * f64::from(v) + c.one_minus_beta2 * g * g;
            let d = vi.sqrt() * c.inv_sqrt_correction2 + c.eps;
            let value = p - c.step_size * mi / d;
            let a = c.one_minus_beta1 * g.abs();
            let shrunk = if decoupled { p.abs() } else { 0.0 };
            let ratio = (mi / d).abs();
            let m_error =
                U * (mi.abs() + c.beta1 * f64::from(m).abs() + 2.0 * a + c.one_minus_beta1 * decay);
            let v_error = U * (6.0 * vi + 2.0 * c.one_minus_beta2 * g.abs() * decay);
            let v_relative = if vi > 0.0 { v_error / vi } else { 0.0 };
            Exact {
                value,
                m: mi,
                v: vi,
                value_bound: 1.05
                    * (U * (value.abs() + shrunk)
                        + c.step_size * (ratio * (17.0 * U + v_relative / 2.0) + m_error / d)),
                m_bound: 1.05 * m_error,
                v_bound: 1.05 * v_error,
            }
        }
    }
}

/// The kernels' own `f32` expression for one element, evaluated on the host with the card's
/// constants: each `fmaf` one `mul_add`, rounded once; the value, the first state word and
/// the second.
fn kernel_f32(
    rule: &OptimizerRule,
    c: &Constants,
    first: bool,
    p: f32,
    m: f32,
    v: f32,
    g: f32,
) -> [f32; 3] {
    let f = |x: f64| x as f32;
    match rule {
        OptimizerRule::Sgd { .. } => {
            let mut step = g;
            if f(c.wd) != 0.0 {
                step = f(c.wd).mul_add(p, step);
            }
            if f(c.momentum) == 0.0 {
                return [(-f(c.lr)).mul_add(step, p), 0.0, 0.0];
            }
            if !first {
                step = f(c.momentum).mul_add(m, step);
            }
            [(-f(c.lr)).mul_add(step, p), store_state_f32(step), 0.0]
        }
        _ => {
            let (mut p, mut g) = (p, g);
            if matches!(rule, OptimizerRule::AdamW { .. }) {
                p *= f(c.shrink);
            } else if f(c.wd) != 0.0 {
                g = f(c.wd).mul_add(p, g);
            }
            let mi = f(c.beta1).mul_add(m, f(c.one_minus_beta1) * g);
            let vi = f(c.beta2).mul_add(v, f(c.one_minus_beta2) * (g * g));
            let d = vi.sqrt().mul_add(f(c.inv_sqrt_correction2), f(c.eps));
            [
                (-f(c.step_size)).mul_add(mi / d, p),
                store_state_f32(mi),
                store_state_f32(vi),
            ]
        }
    }
}

/// The values, momentum buffer (or first moment) and second moment after one step from
/// `state` with gradients `g` (inactive parameters unchanged), in the CPU's arithmetic with
/// its own `f64` constants, stored as the CPU stores them. `counts` are the parameters'
/// updates so far, advanced for the active ones.
fn mirror_step(
    rule: &OptimizerRule,
    counts: &mut [u32],
    t: usize,
    state: &mut [Vec<f32>; 3],
    g: &[f32],
) {
    let mut start = 0;
    for (k, &n) in LENS.iter().enumerate() {
        if active(k, t) {
            let c = Constants::new(rule, counts[k] + 1, false);
            for i in start..start + n {
                let e = exact(
                    rule,
                    &c,
                    counts[k] == 0,
                    state[0][i],
                    state[1][i],
                    state[2][i],
                    g[i],
                );
                state[0][i] = e.value as f32;
                state[1][i] = store_state(e.m);
                state[2][i] = store_state(e.v);
            }
            counts[k] += 1;
        }
        start += n;
    }
}

/// The real CPU optimiser for a rule, kept as its own type so its state can be read.
enum CpuOptimizer {
    /// `cavitation_nn::Sgd`.
    Sgd(Sgd),
    /// `cavitation_nn::Adam`.
    Adam(Adam),
    /// `cavitation_nn::AdamW`.
    AdamW(AdamW),
}

impl CpuOptimizer {
    /// The CPU optimiser for `rule`.
    fn new(rule: &OptimizerRule) -> CpuOptimizer {
        match *rule {
            OptimizerRule::Sgd {
                lr,
                momentum,
                weight_decay,
            } => CpuOptimizer::Sgd(
                Sgd::new(lr)
                    .with_momentum(momentum)
                    .with_weight_decay(weight_decay),
            ),
            OptimizerRule::Adam {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => CpuOptimizer::Adam(
                Adam::new(lr)
                    .with_betas(beta1, beta2)
                    .with_eps(eps)
                    .with_weight_decay(weight_decay),
            ),
            OptimizerRule::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => CpuOptimizer::AdamW(
                AdamW::new(lr)
                    .with_betas(beta1, beta2)
                    .with_eps(eps)
                    .with_weight_decay(weight_decay),
            ),
        }
    }

    /// One step of the CPU optimiser.
    fn step(&mut self, store: &mut ParamStore) {
        match self {
            CpuOptimizer::Sgd(o) => o.step(store),
            CpuOptimizer::Adam(o) => o.step(store),
            CpuOptimizer::AdamW(o) => o.step(store),
        }
    }

    /// The state of parameter `id`: the momentum buffer or first moment, and the second
    /// moment, each `None` when the optimiser keeps none or has not updated `id` yet.
    fn state(&self, id: ParamId) -> [Option<&[f32]>; 2] {
        match self {
            CpuOptimizer::Sgd(o) => [o.momentum(id), None],
            CpuOptimizer::Adam(o) => o
                .moments(id)
                .map_or([None, None], |(m, v)| [Some(m), Some(v)]),
            CpuOptimizer::AdamW(o) => o
                .moments(id)
                .map_or([None, None], |(m, v)| [Some(m), Some(v)]),
        }
    }
}

/// The bits of `v`.
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The arithmetic the card's twins check against is the CPU optimisers' own: run in `f64`
/// with the CPU's constants and stored as the CPU stores it, it gives the bits of
/// `cavitation_nn`'s `Sgd`, `Adam` and `AdamW` over 50 steps of every rule, values and
/// state, parameters that skip steps included. It runs on the CPU alone.
#[test]
fn the_exact_arithmetic_is_the_cpu_optimisers() {
    for (r, rule) in rules().iter().enumerate() {
        let total = LENS.iter().sum::<usize>();
        let start = seeded_values(r as u64, total, -1.0, 1.0);
        let mut state = [start.clone(), vec![0.0; total], vec![0.0; total]];
        let mut counts = [0u32; LENS.len()];
        let mut store = ParamStore::new();
        let mut offset = 0;
        let ids: Vec<_> = LENS
            .iter()
            .map(|&n| {
                let id =
                    store.add(Tensor::from_vec(start[offset..offset + n].to_vec(), &[n]).unwrap());
                offset += n;
                id
            })
            .collect();
        let mut opt = CpuOptimizer::new(rule);
        for t in 0..STEPS {
            let g = gradients(r as u64, t);
            mirror_step(rule, &mut counts, t, &mut state, &g);
            store.zero_grad();
            let mut offset = 0;
            for (k, (&id, &n)) in ids.iter().zip(&LENS).enumerate() {
                if active(k, t) {
                    store
                        .grads
                        .accumulate(id)
                        .0
                        .data_mut()
                        .copy_from_slice(&g[offset..offset + n]);
                }
                offset += n;
            }
            opt.step(&mut store);
            let mut offset = 0;
            for (k, &id) in ids.iter().enumerate() {
                let range = offset..offset + LENS[k];
                offset += LENS[k];
                assert_eq!(
                    bits(store.value(id).data()),
                    bits(&state[0][range.clone()]),
                    "{rule:?} step {t} parameter {k}'s values"
                );
                for (s, cpu) in opt.state(id).into_iter().enumerate() {
                    if let Some(cpu) = cpu {
                        assert_eq!(
                            bits(cpu),
                            bits(&state[s + 1][range.clone()]),
                            "{rule:?} step {t} parameter {k}'s state {s}"
                        );
                    }
                }
            }
        }
        let n = STEPS as u32;
        assert_eq!(counts, [n, n, 37, n, 47, n, n]);
        for (k, &id) in ids.iter().enumerate() {
            let kept = opt.state(id).iter().filter(|s| s.is_some()).count();
            assert_eq!(
                kept,
                rule.state_buffers(),
                "{rule:?} parameter {k} keeps its state"
            );
        }
    }
}

/// The kernel for `rule` with its buffers: values (holding `start`), gradients, state
/// buffers (zeros) and the finite flag (clear).
struct Twin<'c> {
    /// The kernel.
    kernel: OptimizerKernel<'c>,
    /// The parameters' values.
    values: DeviceBuffer<'c>,
    /// Their gradients, written each step.
    grads: DeviceBuffer<'c>,
    /// The rule's state buffers.
    state: Vec<DeviceBuffer<'c>>,
    /// The finite flag.
    flag: DeviceBuffer<'c>,
    /// The parameters' lengths.
    lens: Vec<usize>,
}

impl<'c> Twin<'c> {
    /// The twin of `rule` over parameters of `lens`, its values starting at `start`.
    fn new(cuda: &'c Cuda, rule: OptimizerRule, lens: &[usize], start: &[f32]) -> Twin<'c> {
        let kernel = OptimizerKernel::new(cuda, rule, lens).expect("optimizer");
        let n = start.len();
        Twin {
            kernel,
            values: upload(cuda, start),
            grads: nans(cuda, n),
            state: (0..rule.state_buffers())
                .map(|_| upload(cuda, &vec![0.0f32; n]))
                .collect(),
            flag: upload(cuda, &[0u32]),
            lens: lens.to_vec(),
        }
    }

    /// One step with gradients `g`, the active parameters `active` and the flag set to
    /// `flag`, committed as run when the flag is clear.
    fn step_with(&mut self, g: &[f32], flag: u32, active: &[bool]) {
        self.grads.write(g).expect("gradients");
        self.flag.write(&[flag]).expect("flag");
        let mut state: Vec<&mut DeviceBuffer<'c>> = self.state.iter_mut().collect();
        self.kernel
            .update(
                &mut self.values,
                &self.grads,
                &mut state,
                &self.flag,
                active,
            )
            .expect("update");
        let ran = self.kernel.commit().expect("commit");
        assert_eq!(ran, flag == 0, "the step ran under flag {flag:#x}");
    }

    /// One step of [`LENS`] at step `t`'s activity.
    fn step(&mut self, g: &[f32], flag: u32, t: usize) {
        let active: Vec<bool> = (0..self.lens.len()).map(|k| active(k, t)).collect();
        self.step_with(g, flag, &active);
    }

    /// The values and the state buffers (zeros for a state the rule does not keep).
    fn read(&self) -> [Vec<f32>; 3] {
        let n = self.lens.iter().sum::<usize>();
        let mut out = [read(&self.values, n), vec![0.0; n], vec![0.0; n]];
        for (k, s) in self.state.iter().enumerate() {
            out[k + 1] = read(s, n);
        }
        out
    }
}

/// Every rule, 50 steps on the card, each step checked from the card's own starting state:
/// every active element's value and state the bits of the kernels' `f32` expression on the
/// host, and within its bound of the exact result with the card's `f32` constants; every
/// inactive element unchanged bit for bit; the update counts those of the active steps.
/// Plain SGD's value is within one rounding. A second run of the same steps gives the same
/// bits.
#[test]
fn every_step_is_within_its_bound_of_the_cpu_arithmetic() {
    let Some(cuda) = Cuda::new() else { return };
    for (r, rule) in rules().into_iter().enumerate() {
        let total = LENS.iter().sum::<usize>();
        let start = seeded_values(100 + r as u64, total, -1.0, 1.0);
        let mut twin = Twin::new(&cuda, rule, &LENS, &start);
        let mut before = twin.read();
        let mut counts = [0u32; LENS.len()];
        let mut worst = [0.0f64; 3];
        let plain = matches!(
            rule,
            OptimizerRule::Sgd {
                momentum: 0.0,
                weight_decay: 0.0,
                ..
            }
        );
        let mut fused = 0.0f64;
        for t in 0..STEPS {
            let g = gradients(100 + r as u64, t);
            twin.step(&g, 0, t);
            let after = twin.read();
            let mut start = 0;
            for (k, &n) in LENS.iter().enumerate() {
                let range = start..start + n;
                start += n;
                if !active(k, t) {
                    for s in 0..3 {
                        assert_eq!(
                            bits(&after[s][range.clone()]),
                            bits(&before[s][range.clone()]),
                            "{rule:?} step {t}: inactive parameter {k} changed"
                        );
                    }
                    continue;
                }
                let c = Constants::new(&rule, counts[k] + 1, true);
                let first = counts[k] == 0;
                let at = |i: usize| (before[0][i], before[1][i], before[2][i], g[i]);
                let exacts: Vec<Exact> = range
                    .clone()
                    .map(|i| {
                        let (p, m, v, g) = at(i);
                        exact(&rule, &c, first, p, m, v, g)
                    })
                    .collect();
                let host: Vec<[f32; 3]> = range
                    .clone()
                    .map(|i| {
                        let (p, m, v, g) = at(i);
                        kernel_f32(&rule, &c, first, p, m, v, g)
                    })
                    .collect();
                let checks: [(&str, Pick); 3] = [
                    ("value", |e| (e.value, e.value_bound)),
                    ("first state", |e| (e.m, e.m_bound)),
                    ("second moment", |e| (e.v, e.v_bound)),
                ];
                for (s, (what, pick)) in checks.iter().enumerate().take(1 + rule.state_buffers()) {
                    let what = format!("{rule:?} step {t} parameter {k} {what}");
                    let want: Vec<f32> = host.iter().map(|h| h[s]).collect();
                    assert_exact(&after[s][range.clone()], &want, &format!("{what}, host"));
                    let reference: Vec<f64> = exacts.iter().map(|e| pick(e).0).collect();
                    let w = assert_within(
                        &after[s][range.clone()],
                        &reference,
                        |i| pick(&exacts[i]).1,
                        &what,
                    );
                    worst[s] = worst[s].max(w.ratio());
                    if s == 0 && plain {
                        fused = fused.max(one_rounding_share(&after[0][range.clone()], &reference));
                    }
                }
                counts[k] += 1;
            }
            before = after;
        }
        assert_eq!(twin.kernel.steps(), &counts, "{rule:?} update counts");
        println!(
            "{rule:?}: worst error as a share of its bound: value {:.3}, first state {:.3}, \
             second moment {:.3}",
            worst[0], worst[1], worst[2]
        );
        if plain {
            println!("{rule:?}: worst value error as a share of one rounding: {fused:.3}");
            assert!(fused <= 1.0, "plain SGD took more than one rounding");
        }

        let mut again = Twin::new(
            &cuda,
            rule,
            &LENS,
            &seeded_values(100 + r as u64, total, -1.0, 1.0),
        );
        for t in 0..STEPS {
            again.step(&gradients(100 + r as u64, t), 0, t);
        }
        let (a, b) = (again.read(), twin.read());
        for s in 0..3 {
            assert_eq!(
                bits(&a[s]),
                bits(&b[s]),
                "{rule:?}: a second run's buffer {s}"
            );
        }
    }
}

/// With the finite flag set (to 1, to the top bit alone or to a NaN's bits), a step changes
/// no value, no state and no update count; the next clear step updates as usual.
#[test]
fn a_set_flag_skips_the_step() {
    let Some(cuda) = Cuda::new() else { return };
    for rule in [
        OptimizerRule::Sgd {
            lr: 0.1,
            momentum: 0.0,
            weight_decay: 0.0,
        },
        OptimizerRule::Sgd {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
        },
        OptimizerRule::adam(1e-2, 0.0),
        OptimizerRule::adam_w(1e-2, 1e-2),
    ] {
        for flag in [1u32, 0x8000_0000, f32::NAN.to_bits()] {
            let total = LENS.iter().sum::<usize>();
            let mut twin = Twin::new(&cuda, rule, &LENS, &seeded_values(7, total, -1.0, 1.0));
            twin.step(&gradients(7, 0), 0, 3);
            let before = twin.read();
            let counts = twin.kernel.steps().to_vec();
            twin.step(&gradients(7, 1), flag, 3);
            let after = twin.read();
            for s in 0..3 {
                assert_eq!(
                    bits(&after[s]),
                    bits(&before[s]),
                    "{rule:?}, flag {flag:#x}: a skipped step changed buffer {s}"
                );
            }
            assert_eq!(twin.kernel.steps(), &counts[..]);
            twin.step(&gradients(7, 2), 0, 3);
            assert_ne!(bits(&twin.read()[0]), bits(&after[0]));
            assert!(
                twin.kernel
                    .steps()
                    .iter()
                    .zip(&counts)
                    .all(|(a, b)| a == &(b + 1))
            );
        }
    }
}

/// A step skipped before any parameter's first update leaves the first update as it was:
/// SGD's buffer still starts as the step itself and Adam's bias corrections still start at
/// `t = 1`, so a skip and then a step give the bits of that step alone.
#[test]
fn a_skip_before_the_first_update_changes_nothing() {
    let Some(cuda) = Cuda::new() else { return };
    let momentum = OptimizerRule::Sgd {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 1e-2,
    };
    for rule in [momentum, OptimizerRule::adam(1e-2, 1e-2)] {
        let total = LENS.iter().sum::<usize>();
        let start = seeded_values(8, total, -1.0, 1.0);
        let mut skipped = Twin::new(&cuda, rule, &LENS, &start);
        skipped.step(&gradients(8, 0), 1, 3);
        assert!(skipped.kernel.steps().iter().all(|&c| c == 0));
        skipped.step(&gradients(8, 1), 0, 3);
        let mut direct = Twin::new(&cuda, rule, &LENS, &start);
        direct.step(&gradients(8, 1), 0, 3);
        let (a, b) = (skipped.read(), direct.read());
        for s in 0..3 {
            assert_eq!(bits(&a[s]), bits(&b[s]), "{rule:?}: buffer {s}");
        }
        assert_eq!(skipped.kernel.steps(), direct.kernel.steps());
    }
}

/// NaN, +∞ and −∞ gradients with the flag clear: every value and state word is the bits of
/// the kernels' expression on the host (a NaN matching any NaN), NaN exactly where the CPU's
/// `f64` arithmetic gives NaN, and the elements around them are updated as usual.
#[test]
fn non_finite_gradients_go_where_the_cpus_go() {
    let Some(cuda) = Cuda::new() else { return };
    let total = LENS.iter().sum::<usize>();
    let mut g = gradients(11, 0);
    for (at, value) in [
        (3, f32::NAN),
        (40, f32::INFINITY),
        (1001, f32::NEG_INFINITY),
        (1500, f32::INFINITY),
    ] {
        g[at] = value;
    }
    for rule in rules() {
        let start = seeded_values(12, total, -1.0, 1.0);
        let mut twin = Twin::new(&cuda, rule, &LENS, &start);
        twin.step_with(&g, 0, &[true; LENS.len()]);
        let after = twin.read();
        let zeros = vec![0.0f32; total];
        let mut begin = 0;
        for (k, &n) in LENS.iter().enumerate() {
            let c = Constants::new(&rule, 1, true);
            let cpu = Constants::new(&rule, 1, false);
            for i in begin..begin + n {
                let host = kernel_f32(&rule, &c, true, start[i], zeros[i], zeros[i], g[i]);
                let e = exact(&rule, &cpu, true, start[i], 0.0, 0.0, g[i]);
                for (s, want) in host.iter().enumerate().take(1 + rule.state_buffers()) {
                    let what = format!("{rule:?} parameter {k} element {i} buffer {s}");
                    assert_exact(&[after[s][i]], &[*want], &what);
                    let cpu_value = [e.value, e.m, e.v][s];
                    assert_eq!(
                        after[s][i].is_nan(),
                        cpu_value.is_nan(),
                        "{what}: {} against the CPU's {cpu_value}",
                        after[s][i]
                    );
                }
            }
            begin += n;
        }
        assert!(after[0][3].is_nan(), "{rule:?}: a NaN gradient's value");
    }
}

/// The `cifar-cnn` network's 1,117,162 parameters in its 16 tensors, three steps of SGD with
/// momentum, Adam and AdamW: every value and state word the bits of the kernels' expression
/// on the host, and the counts three.
#[test]
fn the_cifar_network_parameters() {
    let Some(cuda) = Cuda::new() else { return };
    let total = CIFAR.iter().sum::<usize>();
    assert_eq!(total, 1_117_162);
    for rule in [
        OptimizerRule::Sgd {
            lr: 0.05,
            momentum: 0.9,
            weight_decay: 5e-4,
        },
        OptimizerRule::adam(1e-3, 0.0),
        OptimizerRule::adam_w(1e-3, 1e-2),
    ] {
        let start = seeded_values(13, total, -0.1, 0.1);
        let mut twin = Twin::new(&cuda, rule, &CIFAR, &start);
        let mut before = twin.read();
        for t in 0..3 {
            let g = gradients_of(14, t, total);
            twin.step_with(&g, 0, &[true; CIFAR.len()]);
            let after = twin.read();
            let c = Constants::new(&rule, t as u32 + 1, true);
            let host: Vec<[f32; 3]> = (0..total)
                .map(|i| {
                    kernel_f32(
                        &rule,
                        &c,
                        t == 0,
                        before[0][i],
                        before[1][i],
                        before[2][i],
                        g[i],
                    )
                })
                .collect();
            for s in 0..1 + rule.state_buffers() {
                let want: Vec<f32> = host.iter().map(|h| h[s]).collect();
                assert_exact(&after[s], &want, &format!("{rule:?} step {t} buffer {s}"));
            }
            before = after;
        }
        assert!(twin.kernel.steps().iter().all(|&c| c == 3));
    }
}

/// 65,536 one-element parameters: 65,536 blocks, one per parameter. Every parameter moves by
/// one plain SGD step, within one rounding, and counts one update.
#[test]
fn many_parameters() {
    let Some(cuda) = Cuda::new() else { return };
    let n = 1 << 16;
    let lens = vec![1; n];
    let rule = OptimizerRule::Sgd {
        lr: 0.5,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let start = seeded_values(9, n, -1.0, 1.0);
    let g = seeded_values(10, n, -1.0, 1.0);
    let mut twin = Twin::new(&cuda, rule, &lens, &start);
    twin.step_with(&g, 0, &vec![true; n]);
    let lr = f64::from(0.5f32);
    let reference: Vec<f64> = start
        .iter()
        .zip(&g)
        .map(|(&p, &g)| f64::from(p) - lr * f64::from(g))
        .collect();
    let got = read(&twin.values, n);
    assert_within(
        &got,
        &reference,
        |i| U * reference[i].abs(),
        "sgd over 65,536 parameters",
    );
    assert!(twin.kernel.steps().iter().all(|&c| c == 1));
}

/// The refusals: no parameters, an empty one, a setting that is not finite as an `f32`,
/// betas outside `[0, 1)` (a β₂ that rounds to 1 included) and a learning rate that is not
/// finite are `CudaError::Invalid`; so are a commit with nothing launched, an update before
/// the last one was committed, active flags of the wrong length, the wrong number of state
/// buffers, and values or state buffers other than the first update's. A buffer too short
/// panics before any launch, naming it.
#[test]
fn refusals() {
    let Some(cuda) = Cuda::new() else { return };
    let invalid = |r: Result<(), CudaError>, why: &str| match r {
        Err(CudaError::Invalid(text)) => assert!(text.contains(why), "{text}"),
        other => panic!("accepted where {why:?} was expected: {other:?}"),
    };
    let adam = OptimizerRule::adam(1e-3, 0.0);
    let new = |rule, lens: &[usize]| OptimizerKernel::new(&cuda, rule, lens).map(|_| ());
    invalid(new(adam, &[]), "at least one element");
    invalid(new(adam, &[3, 0]), "at least one element");
    let huge = OptimizerRule::Sgd {
        lr: 1e39,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    invalid(new(huge, &[3]), "finite as an f32");
    invalid(
        new(OptimizerRule::adam(f64::NAN, 0.0), &[3]),
        "finite as an f32",
    );
    for (beta1, beta2) in [(1.0, 0.999), (0.9, 0.999_999_99), (-0.1, 0.999)] {
        let betas = OptimizerRule::Adam {
            lr: 1e-3,
            beta1,
            beta2,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        invalid(new(betas, &[3]), "betas must be in [0, 1)");
    }
    invalid(new(adam, &[usize::MAX, 2]), "more than the");

    let mut kernel = OptimizerKernel::new(&cuda, adam, &[3, 5]).expect("optimizer");
    invalid(kernel.set_learning_rate(1e39), "finite as an f32");
    assert_eq!(kernel.rule(), adam);
    invalid(kernel.commit().map(|_| ()), "no update was launched");
    let [mut values, grads, mut m, mut v, mut other] = [(); 5].map(|_| upload(&cuda, &[0.0f32; 8]));
    let flag = upload(&cuda, &[0u32]);
    invalid(
        kernel.update(&mut values, &grads, &mut [&mut m, &mut v], &flag, &[true]),
        "1 active flags for 2 parameters",
    );
    invalid(
        kernel.update(&mut values, &grads, &mut [&mut m], &flag, &[true, true]),
        "keeps 2 state buffers, got 1",
    );
    let mut short = upload(&cuda, &[0.0f32; 7]);
    let message = panic_message(|| {
        let _ = kernel.update(
            &mut values,
            &short,
            &mut [&mut m, &mut v],
            &flag,
            &[true, true],
        );
    });
    assert!(
        message.contains("the gradients buffer holds 28 bytes, and 8 values need 32"),
        "{message}"
    );
    let message = panic_message(|| {
        let _ = kernel.update(
            &mut values,
            &grads,
            &mut [&mut m, &mut short],
            &flag,
            &[true, true],
        );
    });
    assert!(
        message.contains("the state buffer holds 28 bytes"),
        "{message}"
    );
    assert_eq!(kernel.steps(), &[0, 0], "nothing was launched");

    kernel
        .update(
            &mut values,
            &grads,
            &mut [&mut m, &mut v],
            &flag,
            &[true, false],
        )
        .expect("update");
    invalid(
        kernel.update(
            &mut values,
            &grads,
            &mut [&mut m, &mut v],
            &flag,
            &[true, true],
        ),
        "was not committed",
    );
    assert!(kernel.commit().expect("commit"));
    assert_eq!(kernel.steps(), &[1, 0]);
    invalid(
        kernel.update(
            &mut other,
            &grads,
            &mut [&mut m, &mut v],
            &flag,
            &[true, true],
        ),
        "the first one used",
    );
    invalid(
        kernel.update(
            &mut values,
            &grads,
            &mut [&mut m, &mut other],
            &flag,
            &[true, true],
        ),
        "the first one used",
    );
    invalid(
        kernel.update(
            &mut values,
            &grads,
            &mut [&mut v, &mut m],
            &flag,
            &[true, true],
        ),
        "the first one used",
    );
    kernel
        .update(
            &mut values,
            &grads,
            &mut [&mut m, &mut v],
            &flag,
            &[true, true],
        )
        .expect("the first update's buffers");
    assert!(kernel.commit().expect("commit"));
    assert_eq!(kernel.steps(), &[2, 1]);

    // lr = 1e38 is finite as an f32, but Adam's first step size lr / (1 − β₁) is not.
    let mut tiny = OptimizerKernel::new(&cuda, OptimizerRule::adam(1e38, 0.0), &[3, 5])
        .expect("a learning rate finite as an f32");
    let message = |r: Result<(), CudaError>| match r {
        Err(CudaError::Invalid(text)) => text,
        other => panic!("accepted: {other:?}"),
    };
    let text = message(tiny.update(
        &mut values,
        &grads,
        &mut [&mut m, &mut v],
        &flag,
        &[true, true],
    ));
    assert!(text.contains("step size inf"), "{text}");
    invalid(tiny.commit().map(|_| ()), "no update was launched");
    let shrink = OptimizerRule::adam_w(1e38, 1e10);
    let mut tiny = OptimizerKernel::new(&cuda, shrink, &[3, 5]).expect("settings finite as f32s");
    let text = message(tiny.update(
        &mut values,
        &grads,
        &mut [&mut m, &mut v],
        &flag,
        &[true, true],
    ));
    assert!(text.contains("is not finite as an f32"), "{text}");
}

/// The card's flag decides the count: `commit` reads back the flag the launch read, so a
/// step whose flag was set when it launched counts nothing even if the flag is cleared before
/// the commit, and a step launched under a clear flag counts even if it is set afterwards.
#[test]
fn commit_counts_from_the_flag_the_launch_read() {
    let Some(cuda) = Cuda::new() else { return };
    let rule = OptimizerRule::adam(1e-2, 0.0);
    let total = LENS.iter().sum::<usize>();
    let mut twin = Twin::new(&cuda, rule, &LENS, &seeded_values(15, total, -1.0, 1.0));
    let all = [true; LENS.len()];
    twin.grads.write(&gradients(15, 0)).expect("gradients");
    twin.flag.write(&[1u32]).expect("flag");
    let mut state: Vec<&mut DeviceBuffer<'_>> = twin.state.iter_mut().collect();
    twin.kernel
        .update(&mut twin.values, &twin.grads, &mut state, &twin.flag, &all)
        .expect("update");
    twin.flag.write(&[0u32]).expect("flag");
    assert!(!twin.kernel.commit().expect("commit"), "a skipped step ran");
    assert!(twin.kernel.steps().iter().all(|&c| c == 0));
    twin.kernel
        .update(&mut twin.values, &twin.grads, &mut state, &twin.flag, &all)
        .expect("update");
    twin.flag.write(&[1u32]).expect("flag");
    assert!(
        twin.kernel.commit().expect("commit"),
        "a clear step was skipped"
    );
    assert!(twin.kernel.steps().iter().all(|&c| c == 1));
}

/// The card against `cavitation_nn`'s `Sgd`, `Adam` and `AdamW` themselves, side by side for
/// 50 steps from the same seeded parameters and gradients, each on its own state: the card
/// rounds the settings and per-step constants to `f32` and the CPU keeps them in `f64`, so
/// the two drift apart. The worst difference of a value from the CPU's, relative to the
/// largest value at that step, over every rule, element and step, is printed and held within
/// 128u = 2⁻¹⁷; on an NVIDIA GTX 1660 Super it measured 2u to 7u for SGD and Adam and 50u
/// for AdamW, whose shrink factor `1 − lr·wd` loses the most to its `f32` rounding.
#[test]
fn the_card_against_the_cpu_optimisers() {
    let Some(cuda) = Cuda::new() else { return };
    let total = LENS.iter().sum::<usize>();
    let mut overall = 0.0f64;
    for (r, rule) in rules().into_iter().enumerate() {
        let start = seeded_values(300 + r as u64, total, -1.0, 1.0);
        let mut twin = Twin::new(&cuda, rule, &LENS, &start);
        let mut store = ParamStore::new();
        let mut offset = 0;
        let ids: Vec<_> = LENS
            .iter()
            .map(|&n| {
                let id =
                    store.add(Tensor::from_vec(start[offset..offset + n].to_vec(), &[n]).unwrap());
                offset += n;
                id
            })
            .collect();
        let mut opt = CpuOptimizer::new(&rule);
        let mut worst = 0.0f64;
        for t in 0..STEPS {
            let g = gradients(300 + r as u64, t);
            twin.step(&g, 0, t);
            store.zero_grad();
            let mut offset = 0;
            for (k, (&id, &n)) in ids.iter().zip(&LENS).enumerate() {
                if active(k, t) {
                    store
                        .grads
                        .accumulate(id)
                        .0
                        .data_mut()
                        .copy_from_slice(&g[offset..offset + n]);
                }
                offset += n;
            }
            opt.step(&mut store);
            let card = read(&twin.values, total);
            let cpu: Vec<f32> = ids
                .iter()
                .flat_map(|&id| store.value(id).data().to_vec())
                .collect();
            let scale = cpu.iter().fold(0.0f64, |m, &b| m.max(f64::from(b).abs()));
            for (&a, &b) in card.iter().zip(&cpu) {
                worst = worst.max((f64::from(a) - f64::from(b)).abs() / scale);
            }
        }
        println!(
            "{rule:?}: after {STEPS} steps the card's values are within {worst:.3e} of the CPU's, \
             relative to the largest value ({:.0}u)",
            worst / U
        );
        overall = overall.max(worst);
    }
    assert!(
        overall <= 128.0 * U,
        "the card drifted {overall:e} from the CPU optimisers"
    );
}
