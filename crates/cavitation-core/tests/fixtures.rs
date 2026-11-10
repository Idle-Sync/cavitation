//! Compares every tape operation's output and input gradients with PyTorch's, from the JSON
//! fixtures in `tests/fixtures/ops/` written by `scripts/fixtures/make_fixtures.py`.
//!
//! Two bounds are used, and every failure message names the one that applied:
//!
//! - `elementwise`: ours (f32) against PyTorch's f32 result within `1e-6 + 1e-5 * |torch|`.
//!   Used for operations that compute each element from one input element.
//! - `gemm`: ours (f32) against PyTorch's float64 result within
//!   `2 * (k + 2) * 1.19e-7 * sum|terms| + 1e-30`, with `k` the length of the sum and
//!   `sum|terms|` the sum of the absolute values of the terms, computed here in f64 from the
//!   fixture's inputs. Used wherever a result is a sum, because PyTorch sums in a different
//!   order.

use std::path::PathBuf;
use std::sync::Mutex;

use cavitation_core::kernels::{Level, detected, level, set_level, supported};
use cavitation_core::{Gate, GeluApprox, Graph, ParamStore, Reduction, Tape, Tensor, Var};
use serde_json::Value;

/// A tensor read from a fixture: its shape and its values as f64.
struct Tens {
    /// The dimensions.
    shape: Vec<usize>,
    /// The values in row-major order.
    data: Vec<f64>,
}

/// One fixture file.
struct Fixture {
    /// The file name without its extension.
    name: String,
    /// The operation, spelled like the tape method.
    op: String,
    /// The operation's parameters.
    params: Value,
    /// The version of PyTorch that wrote the file.
    torch_version: String,
    /// The operation's inputs, as f32 values.
    inputs: Vec<Tens>,
    /// The gradient fed into the output.
    upstream: Tens,
    /// PyTorch's f32 output.
    output: Tens,
    /// PyTorch's f32 gradients, one per input.
    grads: Vec<Tens>,
    /// PyTorch's float64 output.
    output_f64: Tens,
    /// PyTorch's float64 gradients, one per input.
    grads_f64: Vec<Tens>,
}

/// Which bound a comparison uses.
enum Bound {
    /// `1e-6 + 1e-5 * |torch|` against PyTorch's f32 value.
    Elementwise,
    /// The GEMM bound against PyTorch's f64 value: the length of the sum and, per element,
    /// the sum of the absolute values of its terms.
    Gemm {
        /// The length of the sum.
        k: usize,
        /// `sum|terms|` for every element.
        sums: Vec<f64>,
    },
}

/// The directory holding the fixture files.
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ops")
}

/// The member `key` of a JSON object, panicking with the file name if it is missing.
fn field<'a>(v: &'a Value, key: &str, file: &str) -> &'a Value {
    v.get(key)
        .unwrap_or_else(|| panic!("{file}: missing field `{key}`"))
}

/// Reads a `{ "shape", "data" }` object.
fn parse_tensor(v: &Value, file: &str) -> Tens {
    let shape = field(v, "shape", file)
        .as_array()
        .unwrap_or_else(|| panic!("{file}: `shape` is not an array"))
        .iter()
        .map(|d| {
            d.as_u64()
                .unwrap_or_else(|| panic!("{file}: bad dimension")) as usize
        })
        .collect();
    let data = field(v, "data", file)
        .as_array()
        .unwrap_or_else(|| panic!("{file}: `data` is not an array"))
        .iter()
        .map(|x| x.as_f64().unwrap_or_else(|| panic!("{file}: bad number")))
        .collect();
    Tens { shape, data }
}

/// Reads an array of tensors.
fn parse_tensors(v: &Value, file: &str) -> Vec<Tens> {
    v.as_array()
        .unwrap_or_else(|| panic!("{file}: expected an array of tensors"))
        .iter()
        .map(|t| parse_tensor(t, file))
        .collect()
}

/// Reads every fixture file, in file-name order.
fn load_all() -> Vec<Fixture> {
    let dir = fixture_dir();
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read the fixture directory {}: {e}", dir.display()));
    let mut paths: Vec<PathBuf> = entries
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(path).expect("read fixture");
            let v: Value =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: invalid JSON: {e}"));
            let text_of = |key: &str| {
                field(&v, key, &name)
                    .as_str()
                    .unwrap_or_else(|| panic!("{name}: `{key}` is not a string"))
                    .to_string()
            };
            Fixture {
                op: text_of("op"),
                torch_version: text_of("torch_version"),
                params: field(&v, "params", &name).clone(),
                inputs: parse_tensors(field(&v, "inputs", &name), &name),
                upstream: parse_tensor(field(&v, "upstream", &name), &name),
                output: parse_tensor(field(&v, "output", &name), &name),
                grads: parse_tensors(field(&v, "grads", &name), &name),
                output_f64: parse_tensor(field(&v, "output_f64", &name), &name),
                grads_f64: parse_tensors(field(&v, "grads_f64", &name), &name),
                name,
            }
        })
        .collect()
}

/// A parameter as f64.
fn num(p: &Value, key: &str) -> f64 {
    p[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing number parameter `{key}` in {p}"))
}

/// A parameter as a string.
fn text<'a>(p: &'a Value, key: &str) -> &'a str {
    p[key]
        .as_str()
        .unwrap_or_else(|| panic!("missing string parameter `{key}` in {p}"))
}

/// The reduction a fixture names.
fn reduction(p: &Value) -> Reduction {
    match text(p, "reduction") {
        "mean" => Reduction::Mean,
        "sum" => Reduction::Sum,
        "none" => Reduction::None,
        "batchmean" => Reduction::BatchMean,
        other => panic!("unknown reduction {other:?}"),
    }
}

/// The class labels a fixture names.
fn labels(p: &Value) -> Vec<usize> {
    p["labels"]
        .as_array()
        .expect("labels parameter")
        .iter()
        .map(|l| l.as_u64().expect("label") as usize)
        .collect()
}

/// Records the fixture's operation on `tape` over `v` (one variable per input).
fn record(tape: &mut Tape<'_, f32>, f: &Fixture, v: &[Var]) -> Var {
    let p = &f.params;
    let f32_of = |key: &str| num(p, key) as f32;
    match f.op.as_str() {
        "add" => tape.add(v[0], v[1]),
        "sub" => tape.sub(v[0], v[1]),
        "mul" => tape.mul(v[0], v[1]),
        "scale" => tape.scale(v[0], f32_of("s")),
        "keep_where" => tape.keep_where(v[0], v[1]),
        "matmul" => tape.matmul(v[0], v[1]),
        "matmul_nt" => tape.matmul_nt(v[0], v[1]),
        "sum" => tape.sum(v[0]),
        "mean" => tape.mean(v[0]),
        "sum_axis" => tape.sum_axis(v[0], num(p, "axis") as usize, p["keep_dim"] == true),
        "mean_axis" => tape.mean_axis(v[0], num(p, "axis") as usize, p["keep_dim"] == true),
        "relu" => tape.relu(v[0]),
        "leaky_relu" => tape.leaky_relu(v[0], f32_of("slope")),
        "tanh" => tape.tanh(v[0]),
        "sigmoid" => tape.sigmoid(v[0]),
        "gelu" => tape.gelu(
            v[0],
            match text(p, "approximate") {
                "none" => GeluApprox::Exact,
                "tanh" => GeluApprox::Tanh,
                other => panic!("unknown gelu form {other:?}"),
            },
        ),
        "elu" => tape.elu(v[0], f32_of("alpha")),
        "selu" => tape.selu(v[0]),
        "celu" => tape.celu(v[0], f32_of("alpha")),
        "silu" => tape.silu(v[0]),
        "mish" => tape.mish(v[0]),
        "softplus" => tape.softplus(v[0], f32_of("beta"), f32_of("threshold")),
        "softsign" => tape.softsign(v[0]),
        "hardtanh" => tape.hardtanh(v[0], f32_of("min"), f32_of("max")),
        "relu6" => tape.relu6(v[0]),
        "hardsigmoid" => tape.hardsigmoid(v[0]),
        "hardswish" => tape.hardswish(v[0]),
        "log_sigmoid" => tape.log_sigmoid(v[0]),
        "prelu" => tape.prelu(v[0], v[1]),
        "glu" => tape.glu(
            v[0],
            match text(p, "gate") {
                "sigmoid" => Gate::Sigmoid,
                "gelu_exact" => Gate::Gelu(GeluApprox::Exact),
                "gelu_tanh" => Gate::Gelu(GeluApprox::Tanh),
                "relu" => Gate::Relu,
                "silu" => Gate::Silu,
                other => panic!("unknown gate {other:?}"),
            },
        ),
        "softmax" => tape.softmax(v[0]),
        "log_softmax" => tape.log_softmax(v[0]),
        "softmax_cross_entropy" => tape.softmax_cross_entropy(v[0], &labels(p), reduction(p)),
        "mse" => tape.mse(v[0], v[1], reduction(p)),
        "binary_cross_entropy" => tape.binary_cross_entropy(v[0], v[1], reduction(p)),
        "binary_cross_entropy_with_logits" => {
            tape.binary_cross_entropy_with_logits(v[0], v[1], reduction(p))
        }
        "nll" => tape.nll(v[0], &labels(p), reduction(p)),
        "l1" => tape.l1(v[0], v[1], reduction(p)),
        "huber" => tape.huber(v[0], v[1], f32_of("delta"), reduction(p)),
        "smooth_l1" => tape.smooth_l1(v[0], v[1], f32_of("beta"), reduction(p)),
        "kl_div" => tape.kl_div(v[0], v[1], reduction(p)),
        other => panic!("{}: no test code for operation {other:?}", f.name),
    }
}

/// Runs the fixture's operation on its f32 inputs and returns the output and the gradient of
/// each input for the loss `sum(output * upstream)`.
fn ours(f: &Fixture) -> (Vec<f32>, Vec<Vec<f32>>) {
    let to_tensor = |t: &Tens| {
        let data = t.data.iter().map(|&x| x as f32).collect();
        Tensor::from_vec(data, &t.shape).expect("fixture shape matches data")
    };
    let mut store = ParamStore::new();
    let ids: Vec<_> = f.inputs.iter().map(|t| store.add(to_tensor(t))).collect();
    let upstream = to_tensor(&f.upstream);
    let mut graph = Graph::new();
    let output;
    {
        let mut tape = graph.tape(&store.values);
        let vars: Vec<Var> = ids.iter().map(|&id| tape.param(id)).collect();
        let out = record(&mut tape, f, &vars);
        assert_eq!(
            tape.shape(out).dims(),
            &f.output.shape[..],
            "{}: output shape",
            f.name
        );
        output = tape.value(out).data().to_vec();
        let up = tape.input(upstream.view());
        let weighted = tape.mul(out, up);
        let loss = tape.sum(weighted);
        tape.backward(loss, &mut store.grads);
    }
    let grads = ids
        .iter()
        .map(|&id| store.grad(id).data().to_vec())
        .collect();
    (output, grads)
}

/// For each element of an output of shape `out`, the flat index of the element of an input of
/// shape `input` it was broadcast from (right-aligned, size-1 dimensions repeat).
fn broadcast_map(out: &[usize], input: &[usize]) -> Vec<usize> {
    let len: usize = out.iter().product();
    let offset = out.len() - input.len();
    (0..len)
        .map(|flat| {
            let mut rest = flat;
            let mut index = 0;
            let mut stride = 1;
            for d in (0..out.len()).rev() {
                let coord = rest % out[d];
                rest /= out[d];
                if d >= offset && input[d - offset] != 1 {
                    index += coord * stride;
                }
                if d >= offset {
                    stride *= input[d - offset];
                }
            }
            index
        })
        .collect()
}

/// The bound for a gradient that sums `weights` (the absolute value of each term) into
/// `len` input elements through `map`.
fn summed(len: usize, map: &[usize], weights: &[f64]) -> Bound {
    let k = map.len() / len.max(1);
    if k <= 1 {
        return Bound::Elementwise;
    }
    let mut sums = vec![0.0; len];
    for (&i, &w) in map.iter().zip(weights) {
        sums[i] += w;
    }
    Bound::Gemm { k, sums }
}

/// A bound with the same `k` for every element.
fn gemm(k: usize, sums: Vec<f64>) -> Bound {
    Bound::Gemm { k, sums }
}

/// The bounds for `add`, `sub` and `mul`: the output is elementwise, and a gradient is a sum
/// over every position its input was broadcast to.
fn binary_bounds(f: &Fixture) -> (Bound, Vec<Bound>) {
    let out = &f.output.shape;
    let up: Vec<f64> = f.upstream.data.iter().map(|x| x.abs()).collect();
    let grads = (0..2)
        .map(|i| {
            let map = broadcast_map(out, &f.inputs[i].shape);
            let weights: Vec<f64> = if f.op == "mul" {
                let other = broadcast_map(out, &f.inputs[1 - i].shape);
                (0..up.len())
                    .map(|o| up[o] * f.inputs[1 - i].data[other[o]].abs())
                    .collect()
            } else {
                up.clone()
            };
            summed(f.inputs[i].data.len(), &map, &weights)
        })
        .collect();
    (Bound::Elementwise, grads)
}

/// The bounds for `matmul` and `matmul_nt`.
fn matmul_bounds(f: &Fixture) -> (Bound, Vec<Bound>) {
    let nt = f.op == "matmul_nt";
    let (a, b) = (&f.inputs[0], &f.inputs[1]);
    let (m, k) = (a.shape[0], a.shape[1]);
    let n = if nt { b.shape[0] } else { b.shape[1] };
    let at = |i: usize, p: usize| a.data[i * k + p].abs();
    let bt = |p: usize, j: usize| {
        if nt {
            b.data[j * k + p].abs()
        } else {
            b.data[p * n + j].abs()
        }
    };
    let g = |i: usize, j: usize| f.upstream.data[i * n + j].abs();
    let mut out = vec![0.0; m * n];
    let mut da = vec![0.0; m * k];
    let mut db = vec![0.0; k * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                out[i * n + j] += at(i, p) * bt(p, j);
                da[i * k + p] += g(i, j) * bt(p, j);
                let db_index = if nt { j * k + p } else { p * n + j };
                db[db_index] += at(i, p) * g(i, j);
            }
        }
    }
    (gemm(k, out), vec![gemm(n, da), gemm(m, db)])
}

/// The bounds for `sum`, `mean`, `sum_axis` and `mean_axis` of a rank-2 input (and of any
/// input for `sum` and `mean`).
fn reduce_bounds(f: &Fixture) -> (Bound, Vec<Bound>) {
    let x = &f.inputs[0];
    let mean = f.op.starts_with("mean");
    let up: Vec<f64> = f.upstream.data.iter().map(|v| v.abs()).collect();
    let (len, out_of): (usize, Box<dyn Fn(usize) -> usize>) = if f.op.ends_with("axis") {
        let axis = num(&f.params, "axis") as usize;
        let cols = x.shape[1];
        (
            x.shape[axis],
            Box::new(move |i| if axis == 0 { i % cols } else { i / cols }),
        )
    } else {
        (x.data.len(), Box::new(|_| 0))
    };
    let scale = if mean { 1.0 / len as f64 } else { 1.0 };
    let mut sums = vec![0.0; f.output.data.len()];
    for (i, v) in x.data.iter().enumerate() {
        sums[out_of(i)] += v.abs() * scale;
    }
    let grad_sums = (0..x.data.len()).map(|i| up[out_of(i)] * scale).collect();
    (gemm(len, sums), vec![gemm(1, grad_sums)])
}

/// Row-wise softmax of a rank-2 f64 tensor.
fn softmax_rows(x: &Tens) -> Vec<f64> {
    let cols = x.shape[1];
    let mut y = Vec::with_capacity(x.data.len());
    for row in x.data.chunks(cols) {
        let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = row.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = e.iter().sum();
        y.extend(e.iter().map(|v| v / s));
    }
    y
}

/// The log-sum-exp of each row of a rank-2 f64 tensor, measured from the row maximum, and
/// the maximum.
fn row_lse(x: &Tens) -> Vec<(f64, f64)> {
    x.data
        .chunks(x.shape[1])
        .map(|row| {
            let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (m, row.iter().map(|v| (v - m).exp()).sum::<f64>().ln())
        })
        .collect()
}

/// The bounds for `softmax` and `log_softmax`: `k` is the row length.
fn softmax_bounds(f: &Fixture) -> (Bound, Vec<Bound>) {
    let x = &f.inputs[0];
    let cols = x.shape[1];
    let y = softmax_rows(x);
    let g: Vec<f64> = f.upstream.data.iter().map(|v| v.abs()).collect();
    if f.op == "softmax" {
        // dx = y (g - sum(g y)): the terms are y g and y y' g'.
        let grad = y
            .chunks(cols)
            .zip(g.chunks(cols))
            .flat_map(|(y, g)| {
                let dot: f64 = y.iter().zip(g).map(|(y, g)| y * g).sum();
                y.iter().zip(g).map(move |(y, g)| y * (g + dot))
            })
            .collect();
        (gemm(cols, y.clone()), vec![gemm(cols, grad)])
    } else {
        // out = x - max - lse; dx = g - y sum(g).
        let lse = row_lse(x);
        let out = x
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| (v - lse[i / cols].0).abs() + lse[i / cols].1.abs())
            .collect();
        let grad = y
            .chunks(cols)
            .zip(g.chunks(cols))
            .flat_map(|(y, g)| {
                let total: f64 = g.iter().sum();
                y.iter().zip(g).map(move |(y, g)| g + y * total)
            })
            .collect();
        (gemm(cols, out), vec![gemm(cols, grad)])
    }
}

/// What a loss needs for its bounds: per unreduced element, the sum of absolute terms of the
/// loss and of each input's gradient (before the upstream and the reduction's factor).
struct LossTerms {
    /// `sum|terms|` of the unreduced loss, per unreduced element.
    loss: Vec<f64>,
    /// `sum|terms|` of the gradient of each input, per input element.
    grads: Vec<Vec<f64>>,
    /// The unreduced element each input element belongs to.
    row_of: Vec<usize>,
}

/// The terms of an elementwise loss over inputs `p` and `t` of equal shape.
fn pairwise_terms(
    f: &Fixture,
    loss: impl Fn(f64, f64) -> f64,
    grad_p: impl Fn(f64, f64) -> f64,
    grad_t: impl Fn(f64, f64) -> f64,
) -> LossTerms {
    let (p, t) = (&f.inputs[0].data, &f.inputs[1].data);
    LossTerms {
        loss: p.iter().zip(t).map(|(&p, &t)| loss(p, t)).collect(),
        grads: vec![
            p.iter().zip(t).map(|(&p, &t)| grad_p(p, t)).collect(),
            p.iter().zip(t).map(|(&p, &t)| grad_t(p, t)).collect(),
        ],
        row_of: (0..p.len()).collect(),
    }
}

/// The terms of the loss named by the fixture's operation.
fn loss_terms(f: &Fixture) -> LossTerms {
    let p = &f.params;
    match f.op.as_str() {
        "mse" => pairwise_terms(
            f,
            |p, t| (p - t).powi(2),
            |p, t| 2.0 * (p - t).abs(),
            |p, t| 2.0 * (p - t).abs(),
        ),
        "l1" => pairwise_terms(f, |p, t| (p - t).abs(), |_, _| 1.0, |_, _| 1.0),
        "huber" => {
            let delta = num(p, "delta");
            pairwise_terms(
                f,
                |p, t| {
                    let d = (p - t).abs();
                    if d <= delta {
                        0.5 * d * d
                    } else {
                        delta * d + 0.5 * delta * delta
                    }
                },
                |p, t| (p - t).abs().min(delta),
                |p, t| (p - t).abs().min(delta),
            )
        }
        "smooth_l1" => {
            let beta = num(p, "beta");
            pairwise_terms(
                f,
                |p, t| {
                    let d = (p - t).abs();
                    if d < beta {
                        0.5 * d * d / beta
                    } else {
                        d + 0.5 * beta
                    }
                },
                |p, t| {
                    let d = (p - t).abs();
                    if d < beta { d / beta } else { 1.0 }
                },
                |p, t| {
                    let d = (p - t).abs();
                    if d < beta { d / beta } else { 1.0 }
                },
            )
        }
        "binary_cross_entropy" => pairwise_terms(
            f,
            |p, t| (t * p.ln()).abs() + ((1.0 - t) * (1.0 - p).ln()).abs(),
            |p, t| t.abs() / p + (1.0 - t).abs() / (1.0 - p),
            |p, _| (1.0 - p).ln().abs() + p.ln().abs(),
        ),
        "binary_cross_entropy_with_logits" => pairwise_terms(
            f,
            |z, t| z.max(0.0) + (z * t).abs() + (-z.abs()).exp().ln_1p(),
            |z, t| 1.0 / (1.0 + (-z).exp()) + t.abs(),
            |z, _| z.abs(),
        ),
        "kl_div" => pairwise_terms(
            f,
            |x, t| (t * t.ln()).abs() + (t * x).abs(),
            |_, t| t.abs(),
            |x, t| t.ln().abs() + x.abs() + 1.0,
        ),
        "nll" | "softmax_cross_entropy" => {
            let x = &f.inputs[0];
            let (batch, classes) = (x.shape[0], x.shape[1]);
            let ls = labels(p);
            let ce = f.op == "softmax_cross_entropy";
            let y = softmax_rows(x);
            let lse = row_lse(x);
            let loss = (0..batch)
                .map(|i| {
                    let xl = x.data[i * classes + ls[i]];
                    if ce {
                        (xl - lse[i].0).abs() + lse[i].1.abs()
                    } else {
                        xl.abs()
                    }
                })
                .collect();
            let grad = (0..batch * classes)
                .map(|e| {
                    let hit = f64::from(e % classes == ls[e / classes]);
                    if ce { y[e] + hit } else { hit }
                })
                .collect();
            LossTerms {
                loss,
                grads: vec![grad],
                row_of: (0..batch * classes).map(|e| e / classes).collect(),
            }
        }
        other => panic!("{}: no bound for {other:?}", f.name),
    }
}

/// The bounds for the losses. Unreduced, each element is one term (`k = 1`). Reduced, the
/// output sums the unreduced terms (`k` is their number) and a gradient is one term scaled
/// by the upstream and the reduction's factor (`k = 1`). Cross-entropy and NLL rows add the
/// length of the row to the output's `k` for the log-sum-exp.
fn loss_bounds(f: &Fixture) -> (Bound, Vec<Bound>) {
    let terms = loss_terms(f);
    let reduction = text(&f.params, "reduction");
    let n = terms.loss.len();
    let divisor = match reduction {
        "mean" => n as f64,
        "batchmean" => f.inputs[0].shape[0] as f64,
        _ => 1.0,
    };
    let row_extra = match f.op.as_str() {
        "softmax_cross_entropy" => f.inputs[0].shape[1],
        _ => 0,
    };
    let up = &f.upstream.data;
    let weight = |row: usize| {
        if reduction == "none" {
            up[row].abs()
        } else {
            up[0].abs() / divisor
        }
    };
    let out = if reduction == "none" {
        gemm(1 + row_extra, terms.loss.clone())
    } else {
        gemm(
            n + row_extra,
            vec![terms.loss.iter().sum::<f64>() / divisor],
        )
    };
    let grads = terms
        .grads
        .iter()
        .map(|g| {
            gemm(
                1 + row_extra,
                g.iter()
                    .enumerate()
                    .map(|(e, t)| t * weight(terms.row_of[e]))
                    .collect(),
            )
        })
        .collect();
    (out, grads)
}

/// The bounds for the fixture's output and each of its gradients.
fn bounds(f: &Fixture) -> (Bound, Vec<Bound>) {
    let all_elementwise = || {
        (
            Bound::Elementwise,
            f.inputs.iter().map(|_| Bound::Elementwise).collect(),
        )
    };
    match f.op.as_str() {
        "scale" | "relu" | "leaky_relu" | "tanh" | "sigmoid" | "gelu" | "elu" | "selu" | "celu"
        | "silu" | "mish" | "softplus" | "softsign" | "hardtanh" | "relu6" | "hardsigmoid"
        | "hardswish" | "log_sigmoid" | "glu" | "keep_where" => all_elementwise(),
        "prelu" => {
            // The slope's gradient sums over every position of its channel.
            let (x, slope) = (&f.inputs[0], &f.inputs[1]);
            let map: Vec<usize> = (0..x.data.len()).map(|i| i % slope.data.len()).collect();
            let weights: Vec<f64> = x
                .data
                .iter()
                .zip(&f.upstream.data)
                .map(|(x, g)| if *x > 0.0 { 0.0 } else { x.abs() * g.abs() })
                .collect();
            let slope_bound = Bound::Gemm {
                k: map.len() / slope.data.len(),
                sums: {
                    let mut s = vec![0.0; slope.data.len()];
                    for (&c, w) in map.iter().zip(&weights) {
                        s[c] += w;
                    }
                    s
                },
            };
            (Bound::Elementwise, vec![Bound::Elementwise, slope_bound])
        }
        "add" | "sub" | "mul" => binary_bounds(f),
        "matmul" | "matmul_nt" => matmul_bounds(f),
        "sum" | "mean" | "sum_axis" | "mean_axis" => reduce_bounds(f),
        "softmax" | "log_softmax" => softmax_bounds(f),
        _ => loss_bounds(f),
    }
}

/// Compares `ours` with PyTorch's values under `bound`, panicking with the operation, the
/// quantity, the element, both values and the bound at the first element outside it.
fn compare(f: &Fixture, what: &str, ours: &[f32], torch32: &Tens, torch64: &Tens, bound: &Bound) {
    let torch = match bound {
        Bound::Elementwise => torch32,
        Bound::Gemm { .. } => torch64,
    };
    assert_eq!(
        ours.len(),
        torch.data.len(),
        "{} ({}): {what} has {} elements, torch has {}",
        f.op,
        f.name,
        ours.len(),
        torch.data.len()
    );
    for (i, (&o, &t)) in ours.iter().zip(&torch.data).enumerate() {
        let (limit, label) = match bound {
            Bound::Elementwise => (
                1e-6 + 1e-5 * t.abs(),
                "elementwise bound 1e-6 + 1e-5*|torch|".to_string(),
            ),
            Bound::Gemm { k, sums } => (
                2.0 * (*k as f64 + 2.0) * 1.19e-7 * sums[i] + 1e-30,
                format!(
                    "gemm bound 2*(k+2)*1.19e-7*sum|terms| + 1e-30, k = {k}, sum|terms| = {:e}",
                    sums[i]
                ),
            ),
        };
        let diff = (f64::from(o) - t).abs();
        assert!(
            diff <= limit,
            "{} ({}) at level {}: {what}, element {i}: ours {o:e}, torch {t:e}, difference {diff:e} > {limit:e} ({label})",
            f.op,
            f.name,
            level()
        );
    }
}

/// Runs one fixture: the output and every input's gradient.
fn check(f: &Fixture) {
    let (out_bound, grad_bounds) = bounds(f);
    assert_eq!(
        grad_bounds.len(),
        f.inputs.len(),
        "{}: bounds per input",
        f.name
    );
    assert_eq!(
        f.grads.len(),
        f.inputs.len(),
        "{}: gradients per input",
        f.name
    );
    let (output, grads) = ours(f);
    compare(f, "output", &output, &f.output, &f.output_f64, &out_bound);
    for (i, bound) in grad_bounds.iter().enumerate() {
        let what = format!("gradient of input {i}");
        compare(f, &what, &grads[i], &f.grads[i], &f.grads_f64[i], bound);
    }
}

/// The arithmetic operations.
const ARITHMETIC: &[&str] = &["add", "sub", "mul", "scale"];
/// The select operation.
const SELECT: &[&str] = &["keep_where"];
/// The matrix products.
const MATMUL: &[&str] = &["matmul", "matmul_nt"];
/// The reductions.
const REDUCTIONS: &[&str] = &["sum", "mean", "sum_axis", "mean_axis"];
/// The activations, `prelu` and `glu`.
const ACTIVATIONS: &[&str] = &[
    "relu",
    "leaky_relu",
    "tanh",
    "sigmoid",
    "gelu",
    "elu",
    "selu",
    "celu",
    "silu",
    "mish",
    "softplus",
    "softsign",
    "hardtanh",
    "relu6",
    "hardsigmoid",
    "hardswish",
    "log_sigmoid",
    "prelu",
    "glu",
];
/// `softmax` and `log_softmax`.
const SOFTMAX: &[&str] = &["softmax", "log_softmax"];
/// The losses.
const LOSSES: &[&str] = &[
    "softmax_cross_entropy",
    "mse",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "nll",
    "l1",
    "huber",
    "smooth_l1",
    "kl_div",
];

/// Every operation list a comparison test checks.
const OP_GROUPS: &[&[&str]] = &[
    ARITHMETIC,
    SELECT,
    MATMUL,
    REDUCTIONS,
    ACTIVATIONS,
    SOFTMAX,
    LOSSES,
];

/// Every level, in the order of their numbers.
const ALL_LEVELS: [Level; 4] = [Level::Reference, Level::Sse42, Level::Avx2Fma, Level::Neon];

/// Serialises the tests that change the process-wide kernel level.
static LEVEL_LOCK: Mutex<()> = Mutex::new(());

/// Restores the kernel level it was created under when dropped, even during a panic.
struct RestoreLevel(Level);

impl Drop for RestoreLevel {
    /// Sets the saved level back.
    fn drop(&mut self) {
        set_level(self.0).expect("the original level is supported");
    }
}

/// Every level this CPU can run, in the order of `Level`'s numbers.
fn supported_levels() -> Vec<Level> {
    ALL_LEVELS
        .iter()
        .copied()
        .filter(|&l| supported(l))
        .collect()
}

/// The levels this CPU cannot run.
fn skipped_levels() -> Vec<Level> {
    ALL_LEVELS
        .iter()
        .copied()
        .filter(|&l| !supported(l))
        .collect()
}

/// Runs `body` once on every supported level, holding the level lock for the whole call and
/// restoring the original level afterwards, also when `body` panics.
fn on_every_level(mut body: impl FnMut(Level)) {
    let _lock = LEVEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = RestoreLevel(level());
    for l in supported_levels() {
        set_level(l).expect("a supported level can be set");
        body(l);
    }
}

/// Checks every fixture of the listed operations on every supported level, and that each
/// listed operation has a fixture.
fn check_ops(ops: &[&str]) {
    let all = load_all();
    for op in ops {
        assert!(
            all.iter().any(|f| f.op == *op),
            "no fixture for operation {op:?}"
        );
    }
    on_every_level(|_| {
        for f in all.iter().filter(|f| ops.contains(&f.op.as_str())) {
            check(f);
        }
    });
}

/// `add`, `sub`, `mul` and `scale`, with broadcasting, match PyTorch.
#[test]
fn arithmetic_matches_torch() {
    check_ops(ARITHMETIC);
}

/// `keep_where` matches `torch.where(keep != 0, x, 0)`, including its zero gradient for
/// `keep`.
#[test]
fn keep_where_matches_torch() {
    check_ops(SELECT);
}

/// `matmul` and `matmul_nt` match PyTorch's float64 result within the GEMM bound.
#[test]
fn matmul_matches_torch() {
    check_ops(MATMUL);
}

/// `sum`, `mean`, `sum_axis` and `mean_axis` match PyTorch's float64 result.
#[test]
fn reductions_match_torch() {
    check_ops(REDUCTIONS);
}

/// Every activation, `prelu` and `glu` match PyTorch.
#[test]
fn activations_match_torch() {
    check_ops(ACTIVATIONS);
}

/// `softmax` and `log_softmax` match PyTorch's float64 result.
#[test]
fn softmax_matches_torch() {
    check_ops(SOFTMAX);
}

/// Every loss, under every reduction, matches PyTorch's float64 result.
#[test]
fn losses_match_torch() {
    check_ops(LOSSES);
}

/// Every fixture file's operation is covered by one of the comparison tests.
#[test]
fn every_fixture_op_is_checked() {
    let unchecked: Vec<String> = load_all()
        .iter()
        .filter(|f| !OP_GROUPS.iter().any(|g| g.contains(&f.op.as_str())))
        .map(|f| format!("{} (operation {:?})", f.name, f.op))
        .collect();
    assert!(
        unchecked.is_empty(),
        "fixtures whose operation no comparison covers: {}",
        unchecked.join(", ")
    );
}

/// Prints the levels the comparisons ran on and the ones this CPU cannot run, and checks
/// that the reference level and the level the library chose by default are among them.
#[test]
fn levels_covered() {
    let _lock = LEVEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let run = supported_levels();
    let skipped = skipped_levels();
    let names = |ls: &[Level]| {
        ls.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("fixture levels run: {}", names(&run));
    println!(
        "fixture levels skipped (unsupported on this CPU): {}",
        if skipped.is_empty() {
            "none".to_string()
        } else {
            names(&skipped)
        }
    );
    assert!(run.contains(&Level::Reference), "reference always runs");
    assert!(
        run.contains(&detected()),
        "the detected level {} is among the levels run",
        detected()
    );
    assert_eq!(run.len() + skipped.len(), ALL_LEVELS.len());
}

/// Every loss has a fixture for each of the four reductions.
#[test]
fn every_loss_has_every_reduction() {
    let all = load_all();
    for op in LOSSES {
        for r in ["mean", "sum", "none", "batchmean"] {
            assert!(
                all.iter()
                    .any(|f| f.op == *op && text(&f.params, "reduction") == r),
                "no fixture for {op} with reduction {r}"
            );
        }
    }
}

/// Every fixture records the PyTorch version that wrote it, and all use the pinned one.
#[test]
fn fixture_versions_recorded() {
    let all = load_all();
    assert!(
        !all.is_empty(),
        "no fixtures found in {}",
        fixture_dir().display()
    );
    for f in &all {
        assert_eq!(f.torch_version, "2.15.0", "{}: torch_version", f.name);
    }
}
