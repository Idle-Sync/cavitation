//! Compares `Linear` with PyTorch's `nn.Linear` from the JSON fixture in
//! `tests/fixtures/layers/` written by `scripts/fixtures/make_fixtures.py`.
//!
//! The layer sums over its inputs, so ours (f32) is compared with PyTorch's float64 result
//! within `2 * (k + 2) * 1.19e-7 * sum|terms| + 1e-30`, with `k` the length of the sum and
//! `sum|terms|` the sum of the absolute values of the terms, computed here in f64 from the
//! fixture's inputs. The optimisers are compared with `torch.optim` from the fixtures in
//! `tests/fixtures/optim/`, on both update paths.

use std::path::PathBuf;
use std::sync::Mutex;

use cavitation_core::kernels::{Level, level, set_level, supported};
use cavitation_core::{Graph, ParamId, ParamStore, Rng, Tensor};
use cavitation_nn::{Adam, AdamW, Layer, Linear, Optimizer, Sgd};
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
    *params.value_mut(layer.bias()) = to_tensor(b);
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
        params.grad(layer.bias()).data(),
        &grads64[2],
        batch,
        &db_sums,
    );
}

/// Every kernel level, in the order of their numbers.
const ALL_LEVELS: [Level; 4] = [Level::Reference, Level::Sse42, Level::Avx2Fma, Level::Neon];

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
    assert_eq!(names.len(), 5, "expected five optimiser fixtures");
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
