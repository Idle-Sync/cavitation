//! The GPU optimiser kernel against the CPU optimisers of `cavitation_nn`, on every backend
//! this machine has: SGD (plain, with weight decay, with momentum, with both), Adam (with and
//! without weight decay) and AdamW, 50 updates of seven parameters whose lengths end
//! anywhere in a workgroup, some parameters skipping some steps, and a model of 65,536
//! one-element parameters, whose workgroups take two grid rows. Each step is checked from the GPU's own starting
//! state against the exact result of the CPU's arithmetic with the GPU's `f32` constants,
//! within the bounds `OptimizerKernel` states; that arithmetic is pinned to the real CPU
//! optimisers bit for bit. Skipped steps, update counts and buffer checks are tested too.
//! The GPU tests skip without a GPU, unless `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_core::{ParamId, ParamStore, Tensor};
use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
use cavitation_gpu::{Gpu, PooledBuffer};
use cavitation_nn::{Adam, AdamW, Optimizer, Sgd};
use common::{Kit, Seeded, U, assert_within, for_each_gpu, seeded_values};

/// The parameters' lengths. Each workgroup updates one parameter's run of 256 elements, and
/// these end it at every kind of place: a 232-element tail (1000), 37, a single element, a
/// 44-element tail (300), exactly one workgroup (256), one past (257) and one short (255).
const LENS: [usize; 7] = [1000, 37, 1, 300, 256, 257, 255];

/// Updates per twin.
const STEPS: usize = 50;

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

/// Step `t`'s gradients for all parameters: seeded values spread over sixteen binades, with
/// every 53rd tiny enough (about 1e-22) that its square is subnormal, so Adam's second moment
/// reaches the flush to zero.
fn gradients(seed: u64, t: usize) -> Vec<f32> {
    let mut rng = Seeded::new(seed * 1000 + t as u64);
    (0..LENS.iter().sum::<usize>())
        .map(|i| {
            if i % 53 == 0 {
                return rng.uniform(-1.0, 1.0) * 1e-22;
            }
            let scale = 2f32.powi((rng.next_u64() >> 60) as i32 - 12);
            rng.uniform(-1.0, 1.0) * scale
        })
        .collect()
}

/// One step's constants for a parameter on its update `t` (from 1), in `f64`: either the
/// CPU's own (computed in `f64`) or the GPU's (each rounded to `f32` as the kernel's host
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
    /// The constants of `rule` for update `t`, rounded to `f32` when `gpu`.
    fn new(rule: &OptimizerRule, t: u32, gpu: bool) -> Constants {
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
        let round = |v: f64| if gpu { f64::from(v as f32) } else { v };
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

/// Picks one of an [`Exact`]'s results and its bound.
type Pick = fn(&Exact) -> (f64, f64);

/// One element's update in exact (`f64`) arithmetic, with the bound of the GPU's `f32`
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

/// The CPU optimisers' arithmetic for one element (`sgd_reference` and `adam_reference` in
/// `cavitation_nn::optim`): the value `p`, its state `m` and `v`, its gradient `g`, and
/// whether this is its first update, with constants `c`. The bounds are those
/// `OptimizerKernel`'s docs state, with 5% to spare for second-order terms, except plain
/// SGD's one rounding, which has none.
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
            let mut step = g;
            if c.wd != 0.0 {
                step += c.wd * p;
            }
            let s = step;
            if c.momentum != 0.0 && !first {
                step += c.momentum * f64::from(m);
            }
            let value = p - c.lr * step;
            let plain = c.wd == 0.0 && c.momentum == 0.0;
            let value_bound = if plain {
                U * value.abs()
            } else {
                U * (value.abs() + c.lr * (step.abs() + s.abs()))
            };
            Exact {
                value,
                m: step,
                v: 0.0,
                value_bound: if plain {
                    value_bound
                } else {
                    1.05 * value_bound
                },
                m_bound: 1.05 * U * (step.abs() + s.abs()),
                v_bound: 0.0,
            }
        }
        _ => {
            let decoupled = matches!(rule, OptimizerRule::AdamW { .. });
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
            Exact {
                value,
                m: mi,
                v: vi,
                value_bound: 1.05
                    * U
                    * (value.abs()
                        + shrunk
                        + c.step_size * (19.0 * ratio + (mi.abs() + 2.0 * a) / d)),
                m_bound: 1.05 * U * (mi.abs() + 2.0 * a),
                v_bound: 1.05 * 6.0 * U * vi,
            }
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

/// The arithmetic the GPU twins check against is the CPU optimisers' own: run in `f64` with
/// the CPU's constants and stored as the CPU stores it, it gives the bits of
/// `cavitation_nn`'s `Sgd`, `Adam` and `AdamW` over 50 steps of every rule, values and
/// state, parameters that skip steps included.
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
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
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

/// A GPU kit for the twins, with room for a step's gradient upload and the kernel's
/// constants.
fn kit(gpu: &Gpu) -> Kit {
    Kit::new(gpu, 4 * LENS.iter().sum::<usize>() as u64 + (1 << 16))
}

/// The kernel for `rule` with its buffers: values (holding `start`), gradients, state
/// buffers, the finite flag (clear), and the bind group.
struct Twin {
    /// The kernel.
    kernel: OptimizerKernel,
    /// The parameters' values.
    values: PooledBuffer,
    /// Their gradients, uploaded each step.
    grads: PooledBuffer,
    /// The rule's state buffers.
    state: Vec<PooledBuffer>,
    /// The finite flag.
    flag: PooledBuffer,
    /// The bind group.
    group: cavitation_gpu::wgpu::BindGroup,
}

impl Twin {
    /// The twin of `rule`, its values starting at `start`.
    fn new(kit: &mut Kit, rule: OptimizerRule, start: &[f32]) -> Twin {
        let kernel =
            OptimizerKernel::new(&kit.gpu.clone(), &mut kit.cache, &mut kit.pool, rule, &LENS)
                .unwrap();
        let values = kit.upload(start);
        let grads = kit.zeros(start.len());
        let state: Vec<PooledBuffer> = (0..rule.state_buffers())
            .map(|_| kit.zeros(start.len()))
            .collect();
        let flag = kit.zeros(1);
        let refs: Vec<&PooledBuffer> = state.iter().collect();
        let group = kernel.bind_group(&mut kit.steps, &values, &grads, &refs, &flag);
        Twin {
            kernel,
            values,
            grads,
            state,
            flag,
            group,
        }
    }

    /// One step with gradients `g` and the flag set to `flag`, committed as run when the
    /// flag is clear.
    fn step(&mut self, kit: &mut Kit, g: &[f32], flag: u32, t: usize) {
        let active: Vec<bool> = (0..LENS.len()).map(|k| active(k, t)).collect();
        let bytes: Vec<u8> = g.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut step = kit.steps.begin();
        step.upload(self.grads.buffer(), 0, &bytes);
        step.upload(self.flag.buffer(), 0, &flag.to_le_bytes());
        self.kernel.record(&mut step, &self.group, &active);
        let _ = step.finish();
        self.kernel.commit(flag == 0);
    }

    /// The values and the state buffers (zeros for a state the rule does not keep).
    fn read(&self, kit: &mut Kit) -> [Vec<f32>; 3] {
        let n = LENS.iter().sum::<usize>();
        let mut out = [kit.read(&self.values), vec![0.0; n], vec![0.0; n]];
        for (k, s) in self.state.iter().enumerate() {
            out[k + 1] = kit.read(s);
        }
        out
    }
}

/// Every rule, 50 steps on the GPU, each step checked from the GPU's own starting state:
/// every active element's value and state within its bound of the exact result with the
/// GPU's `f32` constants (SGD without decay or momentum within one rounding), every inactive
/// element's unchanged bit for bit, and the update counts those of the active steps.
#[test]
fn every_step_is_within_its_bound_of_the_cpu_arithmetic() {
    for_each_gpu(|gpu| {
        for (r, rule) in rules().into_iter().enumerate() {
            let mut kit = kit(gpu);
            let total = LENS.iter().sum::<usize>();
            let start = seeded_values(100 + r as u64, total, -1.0, 1.0);
            let mut twin = Twin::new(&mut kit, rule, &start);
            let mut before = twin.read(&mut kit);
            let mut counts = [0u32; LENS.len()];
            let mut worst = [0.0f64; 3];
            for t in 0..STEPS {
                let g = gradients(100 + r as u64, t);
                twin.step(&mut kit, &g, 0, t);
                let after = twin.read(&mut kit);
                let mut start = 0;
                for (k, &n) in LENS.iter().enumerate() {
                    let range = start..start + n;
                    start += n;
                    if !active(k, t) {
                        for s in 0..3 {
                            let bits =
                                |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
                            assert_eq!(
                                bits(&after[s][range.clone()]),
                                bits(&before[s][range.clone()]),
                                "{rule:?} step {t}: inactive parameter {k} changed"
                            );
                        }
                        continue;
                    }
                    let c = Constants::new(&rule, counts[k] + 1, true);
                    let exacts: Vec<Exact> = range
                        .clone()
                        .map(|i| {
                            exact(
                                &rule,
                                &c,
                                counts[k] == 0,
                                before[0][i],
                                before[1][i],
                                before[2][i],
                                g[i],
                            )
                        })
                        .collect();
                    let checks: [(&str, Pick); 3] = [
                        ("value", |e| (e.value, e.value_bound)),
                        ("first state", |e| (e.m, e.m_bound)),
                        ("second moment", |e| (e.v, e.v_bound)),
                    ];
                    for (s, (what, pick)) in
                        checks.iter().enumerate().take(1 + rule.state_buffers())
                    {
                        let reference: Vec<f64> = exacts.iter().map(|e| pick(e).0).collect();
                        let w = assert_within(
                            &after[s][range.clone()],
                            &reference,
                            |i| pick(&exacts[i]).1,
                            &format!("{rule:?} step {t} parameter {k} {what}"),
                        );
                        worst[s] = worst[s].max(w.ratio());
                    }
                    counts[k] += 1;
                }
                before = after;
            }
            assert_eq!(twin.kernel.steps(), &counts, "{rule:?} update counts");
            println!(
                "{rule:?}: worst error as a share of its bound: value {:.3}, first state {:.3}, second moment {:.3}",
                worst[0], worst[1], worst[2]
            );
        }
    });
}

/// With the finite flag set, a step changes no value, no state and no update count; the
/// next clear step updates as usual.
#[test]
fn a_set_flag_skips_the_step() {
    for_each_gpu(|gpu| {
        for rule in [
            OptimizerRule::Sgd {
                lr: 0.1,
                momentum: 0.9,
                weight_decay: 0.0,
            },
            OptimizerRule::adam(1e-2, 0.0),
        ] {
            let mut kit = kit(gpu);
            let total = LENS.iter().sum::<usize>();
            let mut twin = Twin::new(&mut kit, rule, &seeded_values(7, total, -1.0, 1.0));
            twin.step(&mut kit, &gradients(7, 0), 0, 3);
            let before = twin.read(&mut kit);
            let counts = twin.kernel.steps().to_vec();
            twin.step(&mut kit, &gradients(7, 1), 1, 3);
            let after = twin.read(&mut kit);
            for s in 0..3 {
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
                assert_eq!(
                    bits(&after[s]),
                    bits(&before[s]),
                    "{rule:?}: a skipped step changed buffer {s}"
                );
            }
            assert_eq!(twin.kernel.steps(), &counts[..]);
            twin.step(&mut kit, &gradients(7, 2), 0, 3);
            assert_ne!(twin.read(&mut kit)[0], after[0]);
            assert!(
                twin.kernel
                    .steps()
                    .iter()
                    .zip(&counts)
                    .all(|(a, b)| a == &(b + 1))
            );
        }
    });
}

/// Building the kernel for the same rule again builds no pipeline, whatever the parameters'
/// sizes: sizes are in buffers, not in the pipeline.
#[test]
fn one_pipeline_per_rule() {
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu);
        let before = kit.cache.pipelines_built();
        let rule = OptimizerRule::adam(1e-3, 0.0);
        OptimizerKernel::new(&kit.gpu.clone(), &mut kit.cache, &mut kit.pool, rule, &[10]).unwrap();
        OptimizerKernel::new(&kit.gpu.clone(), &mut kit.cache, &mut kit.pool, rule, &LENS).unwrap();
        OptimizerKernel::new(
            &kit.gpu.clone(),
            &mut kit.cache,
            &mut kit.pool,
            OptimizerRule::adam(5e-4, 1e-2),
            &[3, 3],
        )
        .unwrap();
        assert_eq!(kit.cache.pipelines_built() - before, 1);
    });
}

/// A wrong-sized buffer or the wrong number of state buffers is refused when the bind group
/// is made, and recording twice without a commit is refused.
#[test]
fn misuse_is_refused() {
    for_each_gpu(|gpu| {
        let mut kit = kit(gpu);
        let rule = OptimizerRule::adam(1e-3, 0.0);
        let kernel = OptimizerKernel::new(
            &kit.gpu.clone(),
            &mut kit.cache,
            &mut kit.pool,
            rule,
            &[4, 4],
        )
        .unwrap();
        let (good, short, flag) = (kit.zeros(8), kit.zeros(7), kit.zeros(1));
        let message = |f: &mut dyn FnMut()| {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("refused");
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default()
        };
        let m = message(&mut || {
            kernel.bind_group(&mut kit.steps, &good, &short, &[&good, &good], &flag);
        });
        assert!(
            m.contains("the gradients holds 28 bytes, expected 32"),
            "{m}"
        );
        let m = message(&mut || {
            kernel.bind_group(&mut kit.steps, &good, &good, &[&good], &flag);
        });
        assert!(m.contains("keeps 2 state buffers, got 1"), "{m}");
        let mut twin = Twin::new(&mut kit, rule, &vec![0.0; LENS.iter().sum::<usize>()]);
        let mut step = kit.steps.begin();
        twin.kernel
            .record(&mut step, &twin.group, &[true; LENS.len()]);
        let m = message(&mut || {
            twin.kernel
                .record(&mut step, &twin.group, &[true; LENS.len()])
        });
        assert!(m.contains("was not committed"), "{m}");
    });
}

/// A step skipped before any parameter's first update leaves the first update as it was:
/// SGD's buffer still starts as the step itself and Adam's bias corrections still start at
/// `t = 1`, so a skip and then a step give the bits of that step alone.
#[test]
fn a_skip_before_the_first_update_changes_nothing() {
    for_each_gpu(|gpu| {
        let momentum = OptimizerRule::Sgd {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 1e-2,
        };
        for rule in [momentum, OptimizerRule::adam(1e-2, 1e-2)] {
            let mut kit = kit(gpu);
            let total = LENS.iter().sum::<usize>();
            let start = seeded_values(8, total, -1.0, 1.0);
            let mut skipped = Twin::new(&mut kit, rule, &start);
            skipped.step(&mut kit, &gradients(8, 0), 1, 3);
            assert!(skipped.kernel.steps().iter().all(|&c| c == 0));
            skipped.step(&mut kit, &gradients(8, 1), 0, 3);
            let mut direct = Twin::new(&mut kit, rule, &start);
            direct.step(&mut kit, &gradients(8, 1), 0, 3);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            let (a, b) = (skipped.read(&mut kit), direct.read(&mut kit));
            for s in 0..3 {
                assert_eq!(bits(&a[s]), bits(&b[s]), "{rule:?}: buffer {s}");
            }
            assert_eq!(skipped.kernel.steps(), direct.kernel.steps());
        }
    });
}

/// 65,536 one-element parameters: 65,536 workgroups, so the grid takes two rows and all but
/// one workgroup of the second row lie past the table. Every parameter moves by one plain
/// SGD step, within one rounding, and counts one update.
#[test]
fn many_parameters_cover_a_split_grid() {
    for_each_gpu(|gpu| {
        let n = 1 << 16;
        let lens = vec![1; n];
        let mut kit = Kit::new(gpu, 1 << 21);
        let rule = OptimizerRule::Sgd {
            lr: 0.5,
            momentum: 0.0,
            weight_decay: 0.0,
        };
        let mut kernel =
            OptimizerKernel::new(&kit.gpu.clone(), &mut kit.cache, &mut kit.pool, rule, &lens)
                .unwrap();
        let start = seeded_values(9, n, -1.0, 1.0);
        let g = seeded_values(10, n, -1.0, 1.0);
        let (values, grads, flag) = (kit.upload(&start), kit.upload(&g), kit.zeros(1));
        let group = kernel.bind_group(&mut kit.steps, &values, &grads, &[], &flag);
        let mut step = kit.steps.begin();
        kernel.record(&mut step, &group, &vec![true; n]);
        let _ = step.finish();
        kernel.commit(true);
        let lr = f64::from(0.5f32);
        let reference: Vec<f64> = start
            .iter()
            .zip(&g)
            .map(|(&p, &g)| f64::from(p) - lr * f64::from(g))
            .collect();
        assert_within(
            &kit.read(&values),
            &reference,
            |i| U * reference[i].abs(),
            "sgd over 65,536 parameters",
        );
        assert!(kernel.steps().iter().all(|&c| c == 1));
    });
}

/// The documented refusals of `new`, `record` and `commit`: no parameters, an empty one,
/// a setting that is not finite as an `f32`, betas outside `[0, 1)`, a commit with nothing
/// recorded, and active flags of the wrong length.
#[test]
fn refusals() {
    for_each_gpu(|gpu| {
        let message = |f: &mut dyn FnMut()| {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("refused");
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default()
        };
        let mut kit = kit(gpu);
        let gpu = kit.gpu.clone();
        let adam = OptimizerRule::adam(1e-3, 0.0);
        let mut new = |rule: OptimizerRule, lens: &[usize]| {
            message(&mut || {
                let _ = OptimizerKernel::new(&gpu, &mut kit.cache, &mut kit.pool, rule, lens);
            })
        };
        assert!(new(adam, &[]).contains("at least one element"));
        assert!(new(adam, &[3, 0]).contains("at least one element"));
        let huge = OptimizerRule::Sgd {
            lr: 1e39,
            momentum: 0.0,
            weight_decay: 0.0,
        };
        assert!(new(huge, &[3]).contains("finite as an f32"));
        let nan = OptimizerRule::adam(f64::NAN, 0.0);
        assert!(new(nan, &[3]).contains("finite as an f32"));
        let betas = OptimizerRule::Adam {
            lr: 1e-3,
            beta1: 1.0,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        assert!(new(betas, &[3]).contains("betas must be in [0, 1)"));
        let mut kernel =
            OptimizerKernel::new(&gpu, &mut kit.cache, &mut kit.pool, adam, &[3]).unwrap();
        assert!(message(&mut || kernel.commit(true)).contains("no step was recorded"));
        let [values, grads, m, v] = [0; 4].map(|_| kit.zeros(3));
        let flag = kit.zeros(1);
        let group = kernel.bind_group(&mut kit.steps, &values, &grads, &[&m, &v], &flag);
        let mut step = kit.steps.begin();
        let m = message(&mut || kernel.record(&mut step, &group, &[true, true]));
        assert!(m.contains("2 active flags for 1 parameters"), "{m}");
    });
}
