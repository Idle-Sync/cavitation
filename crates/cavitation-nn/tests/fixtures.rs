//! Compares `Linear` with PyTorch's `nn.Linear` from the JSON fixture in
//! `tests/fixtures/layers/` written by `scripts/fixtures/make_fixtures.py`.
//!
//! The layer sums over its inputs, so ours (f32) is compared with PyTorch's float64 result
//! within `2 * (k + 2) * 1.19e-7 * sum|terms| + 1e-30`, with `k` the length of the sum and
//! `sum|terms|` the sum of the absolute values of the terms, computed here in f64 from the
//! fixture's inputs. The optimisers are compared with `torch.optim` from the fixtures in
//! `tests/fixtures/optim/`, on both update paths. A network whose first `Linear` has no bias
//! is loaded from PyTorch's `state_dict` in `tests/fixtures/state-dict/` and compared on its
//! output and its parameter gradients, with the same bound carried through the layers.

use std::path::PathBuf;
use std::sync::Mutex;

use cavitation_core::kernels::{Level, level, set_level, supported};
use cavitation_core::{Graph, ParamId, ParamStore, Rng, Tensor};
use cavitation_nn::{Adam, AdamW, Init, Layer, Linear, Optimizer, Relu, Sequential, Sgd, load};
use serde_json::Value;

/// A tensor read from a fixture: its shape and its values as f64.
struct Tens {
    /// The dimensions.
    shape: Vec<usize>,
    /// The values in row-major order.
    data: Vec<f64>,
}

/// Reads a `{ "shape", "data" }` object.
fn parse_tensor(v: &Value) -> Tens {
    let shape = v["shape"]
        .as_array()
        .expect("shape array")
        .iter()
        .map(|d| d.as_u64().expect("dimension") as usize)
        .collect();
    let data = v["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect();
    Tens { shape, data }
}

/// Reads an array of tensors.
fn parse_tensors(v: &Value) -> Vec<Tens> {
    v.as_array()
        .expect("array of tensors")
        .iter()
        .map(parse_tensor)
        .collect()
}

/// A tensor of f32 values from a fixture tensor.
fn to_tensor(t: &Tens) -> Tensor<f32> {
    let data = t.data.iter().map(|&x| x as f32).collect();
    Tensor::from_vec(data, &t.shape).expect("fixture shape matches data")
}

/// Compares `ours` with PyTorch's float64 values within the GEMM bound: `k` is the length of
/// each sum and `sums[i]` the sum of the absolute values of element `i`'s terms.
fn compare(what: &str, ours: &[f32], torch64: &Tens, k: usize, sums: &[f64]) {
    assert_eq!(ours.len(), torch64.data.len(), "{what}: element count");
    for (i, (&o, &t)) in ours.iter().zip(&torch64.data).enumerate() {
        let limit = 2.0 * (k as f64 + 2.0) * 1.19e-7 * sums[i] + 1e-30;
        let diff = (f64::from(o) - t).abs();
        assert!(
            diff <= limit,
            "linear at level {}: {what}, element {i}: ours {o:e}, torch {t:e}, difference {diff:e} > {limit:e} (gemm bound, k = {k}, sum|terms| = {:e})",
            level(),
            sums[i]
        );
    }
}

/// Loads PyTorch's weights into our `Linear`, runs forward and backward under the fixture's
/// upstream gradient and compares the output and the three gradients.
fn check_at_current_level() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/layers/linear.json");
    let text = std::fs::read_to_string(&path).expect("read fixture");
    let v: Value = serde_json::from_str(&text).expect("valid JSON");
    assert_eq!(v["op"], "linear");
    let inputs = parse_tensors(&v["inputs"]);
    let upstream = parse_tensor(&v["upstream"]);
    let output64 = parse_tensor(&v["output_f64"]);
    let grads64 = parse_tensors(&v["grads_f64"]);
    let (x, w, b) = (&inputs[0], &inputs[1], &inputs[2]);
    let (batch, n_in, n_out) = (x.shape[0], x.shape[1], w.shape[0]);
    assert_eq!(w.shape, [n_out, n_in]);
    assert_eq!(b.shape, [n_out]);
    assert_eq!(v["params"]["inputs"].as_u64(), Some(n_in as u64));
    assert_eq!(v["params"]["outputs"].as_u64(), Some(n_out as u64));

    let mut params = ParamStore::new();
    let layer = Linear::new(&mut params, n_in, n_out, &mut Rng::seed(0));
    *params.value_mut(layer.weight()) = to_tensor(w);
    *params.value_mut(layer.bias().unwrap()) = to_tensor(b);
    // The input is registered as a parameter so that it receives a gradient.
    let x_id = params.add(to_tensor(x));

    let mut graph = Graph::new();
    let output;
    {
        let mut tape = graph.tape(&params.values);
        let xv = tape.param(x_id);
        let y = layer.forward(&mut tape, xv);
        output = tape.value(y).data().to_vec();
        let up = tape.input(to_tensor(&upstream).view());
        let weighted = tape.mul(y, up);
        let loss = tape.sum(weighted);
        tape.backward(loss, &mut params.grads);
    }

    let abs = |t: &Tens| t.data.iter().map(|v| v.abs()).collect::<Vec<f64>>();
    let (xa, wa, up) = (abs(x), abs(w), abs(&upstream));
    // Output: y[i][j] = sum_p x[i][p] w[j][p] + b[j].
    let mut out_sums = vec![0.0; batch * n_out];
    // dx[i][p] = sum_j g[i][j] w[j][p].
    let mut dx_sums = vec![0.0; batch * n_in];
    // dw[j][p] = sum_i g[i][j] x[i][p].
    let mut dw_sums = vec![0.0; n_out * n_in];
    // db[j] = sum_i g[i][j].
    let mut db_sums = vec![0.0; n_out];
    for i in 0..batch {
        for j in 0..n_out {
            out_sums[i * n_out + j] += b.data[j].abs();
            db_sums[j] += up[i * n_out + j];
            for p in 0..n_in {
                out_sums[i * n_out + j] += xa[i * n_in + p] * wa[j * n_in + p];
                dx_sums[i * n_in + p] += up[i * n_out + j] * wa[j * n_in + p];
                dw_sums[j * n_in + p] += up[i * n_out + j] * xa[i * n_in + p];
            }
        }
    }
    compare("output", &output, &output64, n_in + 1, &out_sums);
    compare(
        "gradient of input",
        params.grad(x_id).data(),
        &grads64[0],
        n_out,
        &dx_sums,
    );
    compare(
        "gradient of weight",
        params.grad(layer.weight()).data(),
        &grads64[1],
        batch,
        &dw_sums,
    );
    compare(
        "gradient of bias",
        params.grad(layer.bias().unwrap()).data(),
        &grads64[2],
        batch,
        &db_sums,
    );
}

/// Every kernel level, in the order of their numbers.
const ALL_LEVELS: [Level; 5] = [
    Level::Reference,
    Level::Sse42,
    Level::Avx2Fma,
    Level::Neon,
    Level::Accelerate,
];

/// Serialises the tests that change the process-wide kernel level, so each one knows which
/// level, and so which path, the code it compares with PyTorch runs on.
static LEVEL_LOCK: Mutex<()> = Mutex::new(());

/// Restores the kernel level it was created under when dropped, even during a panic.
struct RestoreLevel(Level);

impl Drop for RestoreLevel {
    /// Sets the saved level back.
    fn drop(&mut self) {
        set_level(self.0).expect("the original level is supported");
    }
}

/// `Linear` matches `nn.Linear` (output and the gradients of input, weight and bias) on every
/// kernel level this CPU supports, holding the level lock throughout.
#[test]
fn linear_matches_pytorch() {
    let _lock = LEVEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = RestoreLevel(level());
    for l in ALL_LEVELS.into_iter().filter(|&l| supported(l)) {
        set_level(l).expect("a supported level can be set");
        check_at_current_level();
    }
}

/// The GEMM bound's factor for a sum of `k` terms, `2 · (k + 2) · 1.19e-7`.
fn gamma(k: usize) -> f64 {
    2.0 * (k as f64 + 2.0) * 1.19e-7
}

/// Compares `ours` with PyTorch's float64 values within `bound`, element by element.
fn compare_within(what: &str, ours: &[f32], torch64: &Tens, bound: &[f64]) {
    assert_eq!(ours.len(), torch64.data.len(), "{what}: element count");
    for (i, ((&o, &t), &b)) in ours.iter().zip(&torch64.data).zip(bound).enumerate() {
        let diff = (f64::from(o) - t).abs();
        assert!(
            diff <= b + 1e-30,
            "no-bias network at level {}: {what}, element {i}: ours {o:e}, torch {t:e}, difference {diff:e} > {b:e}",
            level()
        );
    }
}

/// Loads PyTorch's `Linear(4, 8, bias=False), ReLU, Linear(8, 3)` into the same network
/// built with [`Linear::without_bias`], runs forward and backward of `sum(output ·
/// upstream)` and compares the output and the gradients of `0.weight`, `2.weight` and
/// `2.bias` with PyTorch's float64 results.
///
/// Each bound is the GEMM bound of its own sum, `γ(k) · Σ|terms|` with `γ(k) = 2 · (k + 2)
/// · 1.19e-7`, plus the error carried in: the hidden pre-activation `z = x W₁ᵀ` (4 terms,
/// no bias) is within `e₁ = γ(4) Σ|x||W₁|`, and every `z` is further than `e₁` from 0 (the
/// test checks), so both sides take the same ReLU branch and `h = relu(z)` is within `e₁`
/// too. Then the output (`k = 9`, the bias one term) adds `Σ |W₂| e₁`; `∂/∂W₂ = upᵀ h`
/// (`k = 5` rows) adds `Σ |up| e₁`; `∂/∂b₂` sums the upstream over the rows; the hidden
/// gradient `g = up W₂` (`k = 3`) is within `γ(3) Σ|up||W₂|`, which `∂/∂W₁ = gᵀ x` carries
/// through `|x|`.
fn check_no_bias_network_at_current_level() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/state-dict");
    let text = std::fs::read_to_string(dir.join("mlp-no-bias-io.json")).expect("read fixture");
    let v: Value = serde_json::from_str(&text).expect("valid JSON");
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(0);
    let net = Sequential::new()
        .push(Linear::without_bias(
            &mut params,
            4,
            8,
            Init::XavierUniform,
            &mut rng,
        ))
        .push(Relu)
        .push(Linear::new(&mut params, 8, 3, &mut rng));
    let names: Vec<String> = net.parameter_names().into_iter().map(|(n, _)| n).collect();
    let keys: Vec<&str> = (v["keys"].as_array().expect("keys").iter())
        .map(|k| k.as_str().expect("key"))
        .collect();
    assert_eq!(names, keys);
    load(dir.join("mlp-no-bias.safetensors"), &net, &mut params).expect("load");

    let (x, up) = (parse_tensor(&v["input"]), parse_tensor(&v["upstream"]));
    let (rows, n_in, hidden, n_out) = (5, 4, 8, 3);
    assert_eq!(
        (x.shape.as_slice(), up.shape.as_slice()),
        (&[rows, n_in][..], &[rows, n_out][..])
    );
    let (first, last) = (net.linear(0).unwrap(), net.linear(2).unwrap());
    let (w1_id, w2_id) = (first.weight(), last.weight());
    let b2_id = last.bias().expect("the output layer has a bias");
    assert_eq!(first.bias(), None);

    let mut graph = Graph::new();
    let output;
    {
        let mut tape = graph.tape(&params.values);
        let xv = tape.input(to_tensor(&x).view());
        let y = net.forward(&mut tape, xv);
        output = tape.value(y).data().to_vec();
        let u = tape.input(to_tensor(&up).view());
        let weighted = tape.mul(y, u);
        let loss = tape.sum(weighted);
        tape.backward(loss, &mut params.grads);
    }

    let wide = |id: ParamId| -> Vec<f64> {
        params
            .value(id)
            .data()
            .iter()
            .map(|&w| f64::from(w))
            .collect()
    };
    let (w1, w2, b2) = (wide(w1_id), wide(w2_id), wide(b2_id));
    let (x, up) = (&x.data, &up.data);
    // The hidden layer, exactly, and its error bound.
    let mut z = vec![0.0f64; rows * hidden];
    let mut e1 = vec![0.0f64; rows * hidden];
    for r in 0..rows {
        for p in 0..hidden {
            let (mut sum, mut abs) = (0.0, 0.0);
            for q in 0..n_in {
                sum += x[r * n_in + q] * w1[p * n_in + q];
                abs += (x[r * n_in + q] * w1[p * n_in + q]).abs();
            }
            z[r * hidden + p] = sum;
            e1[r * hidden + p] = gamma(n_in) * abs;
            assert!(
                sum.abs() > e1[r * hidden + p],
                "row {r}, neuron {p} is at the kink"
            );
        }
    }
    let h: Vec<f64> = z.iter().map(|&v| v.max(0.0)).collect();
    let on = |i: usize| z[i] > 0.0;

    let mut out_bound = vec![0.0f64; rows * n_out];
    for r in 0..rows {
        for j in 0..n_out {
            let (mut abs, mut carried) = (b2[j].abs(), 0.0);
            for p in 0..hidden {
                abs += (h[r * hidden + p] * w2[j * hidden + p]).abs();
                carried += w2[j * hidden + p].abs() * e1[r * hidden + p];
            }
            out_bound[r * n_out + j] = gamma(hidden + 1) * abs + carried;
        }
    }
    let mut b2_bound = vec![0.0f64; n_out];
    let mut w2_bound = vec![0.0f64; n_out * hidden];
    for j in 0..n_out {
        b2_bound[j] = gamma(rows) * (0..rows).map(|r| up[r * n_out + j].abs()).sum::<f64>();
        for p in 0..hidden {
            let (mut abs, mut carried) = (0.0, 0.0);
            for r in 0..rows {
                abs += (up[r * n_out + j] * h[r * hidden + p]).abs();
                carried += up[r * n_out + j].abs() * e1[r * hidden + p];
            }
            w2_bound[j * hidden + p] = gamma(rows) * abs + carried;
        }
    }
    // The hidden gradient, through the ReLU, and its error bound.
    let mut g = vec![0.0f64; rows * hidden];
    let mut eg = vec![0.0f64; rows * hidden];
    for r in 0..rows {
        for p in 0..hidden {
            if !on(r * hidden + p) {
                continue;
            }
            let (mut sum, mut abs) = (0.0, 0.0);
            for j in 0..n_out {
                sum += up[r * n_out + j] * w2[j * hidden + p];
                abs += (up[r * n_out + j] * w2[j * hidden + p]).abs();
            }
            g[r * hidden + p] = sum;
            eg[r * hidden + p] = gamma(n_out) * abs;
        }
    }
    let mut w1_bound = vec![0.0f64; hidden * n_in];
    for p in 0..hidden {
        for q in 0..n_in {
            let (mut abs, mut carried) = (0.0, 0.0);
            for r in 0..rows {
                abs += (g[r * hidden + p] * x[r * n_in + q]).abs();
                carried += eg[r * hidden + p] * x[r * n_in + q].abs();
            }
            w1_bound[p * n_in + q] = gamma(rows) * abs + carried;
        }
    }

    let grads64 = &v["grads_f64"];
    compare_within(
        "output",
        &output,
        &parse_tensor(&v["output_f64"]),
        &out_bound,
    );
    let checks = [
        ("0.weight", w1_id, &w1_bound),
        ("2.weight", w2_id, &w2_bound),
        ("2.bias", b2_id, &b2_bound),
    ];
    for (name, id, bound) in checks {
        let torch = parse_tensor(&grads64[name]);
        assert_eq!(torch.shape, params.grad(id).shape().dims(), "{name}: shape");
        compare_within(
            &format!("gradient of {name}"),
            params.grad(id).data(),
            &torch,
            bound,
        );
    }
}

/// A network whose first `Linear` has no bias, loaded from PyTorch's `state_dict` of
/// `nn.Sequential(nn.Linear(4, 8, bias=False), nn.ReLU(), nn.Linear(8, 3))`, matches
/// PyTorch's output and parameter gradients on every kernel level this CPU supports.
#[test]
fn no_bias_network_matches_pytorch() {
    let _lock = LEVEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = RestoreLevel(level());
    for l in ALL_LEVELS.into_iter().filter(|&l| supported(l)) {
        set_level(l).expect("a supported level can be set");
        check_no_bias_network_at_current_level();
    }
}

/// Copies a fixture tensor's values over the parameter `id`.
fn set_value(params: &mut ParamStore, id: ParamId, t: &Tens) {
    params
        .value_mut(id)
        .data_mut()
        .copy_from_slice(to_tensor(t).data());
}

/// Checks `got` against PyTorch's `want` within `1e-6 + 1e-5 * |torch|`.
fn within(case: &str, step: usize, what: &str, got: &[f32], want: &Tens) {
    assert_eq!(got.len(), want.data.len(), "{case}: {what} length");
    for (i, (&g, &w)) in got.iter().zip(&want.data).enumerate() {
        let limit = 1e-6 + 1e-5 * w.abs();
        assert!(
            (f64::from(g) - w).abs() <= limit,
            "{case}, step {}: {what}[{i}] is {g:e}, torch has {w:e} (limit {limit:e})",
            step + 1
        );
    }
}

/// Builds the optimiser a fixture's `params` describe.
fn optimizer_for(v: &Value) -> Box<dyn Optimizer<f32>> {
    let p = &v["params"];
    let get = |key: &str| p[key].as_f64();
    let lr = get("lr").expect("lr");
    let wd = get("weight_decay");
    match v["optimizer"].as_str().expect("optimizer name") {
        "SGD" => {
            let mut o = Sgd::new(lr);
            if let Some(m) = get("momentum") {
                o = o.with_momentum(m);
            }
            if let Some(w) = wd {
                o = o.with_weight_decay(w);
            }
            Box::new(o)
        }
        "Adam" => {
            let o = Adam::new(lr);
            Box::new(match wd {
                Some(w) => o.with_weight_decay(w),
                None => o,
            })
        }
        "AdamW" => {
            let o = AdamW::new(lr);
            Box::new(match wd {
                Some(w) => o.with_weight_decay(w),
                None => o,
            })
        }
        other => panic!("unknown optimiser {other}"),
    }
}

/// Every optimiser fixture: ten steps on a weight and a bias, with the fixture's gradients,
/// agree with `torch.optim` after every step, on both update paths: the plain loop (run at
/// `Level::Reference`) and the chunked loop (run at the first faster level this CPU
/// supports, when there is one). The level lock is held throughout, so another test cannot
/// switch the path mid-run.
#[test]
fn optimizers_match_pytorch() {
    let _lock = LEVEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = RestoreLevel(level());
    let fast = ALL_LEVELS
        .into_iter()
        .find(|&l| l != Level::Reference && supported(l));
    for l in std::iter::once(Level::Reference).chain(fast) {
        set_level(l).expect("a supported level can be set");
        check_optimizers(l);
    }
}

/// Runs every optimiser fixture at the current kernel level `at`, naming it in failures.
fn check_optimizers(at: Level) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/optim");
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .expect("fixture directory")
        .map(|e| e.expect("entry").path())
        .collect();
    names.sort();
    assert_eq!(names.len(), 7, "expected seven optimiser fixtures");
    for path in names {
        let case = format!("{} at {at}", path.file_stem().unwrap().to_string_lossy());
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let mut opt = optimizer_for(&v);
        let mut params = ParamStore::new();
        let w = params.add(Tensor::zeros(&[4, 3]));
        let b = params.add(Tensor::zeros(&[3]));
        set_value(&mut params, w, &parse_tensor(&v["initial"]["weight"]));
        set_value(&mut params, b, &parse_tensor(&v["initial"]["bias"]));
        let grads = v["grads"].as_array().unwrap();
        let after = v["after"].as_array().unwrap();
        assert_eq!(grads.len(), 10);
        for (step, (g, a)) in grads.iter().zip(after).enumerate() {
            params.zero_grad();
            for (id, key) in [(w, "weight"), (b, "bias")] {
                let (mut buf, stale) = params.grads.accumulate(id);
                assert!(stale);
                buf.data_mut()
                    .copy_from_slice(to_tensor(&parse_tensor(&g[key])).data());
            }
            opt.step(&mut params);
            within(
                &case,
                step,
                "weight",
                params.value(w).data(),
                &parse_tensor(&a["weight"]),
            );
            within(
                &case,
                step,
                "bias",
                params.value(b).data(),
                &parse_tensor(&a["bias"]),
            );
        }
    }
}
