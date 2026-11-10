//! Compares `Linear` with PyTorch's `nn.Linear` from the JSON fixture in
//! `tests/fixtures/layers/` written by `scripts/fixtures/make_fixtures.py`.
//!
//! The layer sums over its inputs, so ours (f32) is compared with PyTorch's float64 result
//! within `2 * (k + 2) * 1.19e-7 * sum|terms| + 1e-30`, with `k` the length of the sum and
//! `sum|terms|` the sum of the absolute values of the terms, computed here in f64 from the
//! fixture's inputs.

use std::path::PathBuf;

use cavitation_core::kernels::{Level, level, set_level, supported};
use cavitation_core::{Graph, ParamStore, Rng, Tensor};
use cavitation_nn::{Layer, Linear};
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

/// Restores the kernel level it was created under when dropped, even during a panic.
struct RestoreLevel(Level);

impl Drop for RestoreLevel {
    /// Sets the saved level back.
    fn drop(&mut self) {
        set_level(self.0).expect("the original level is supported");
    }
}

/// `Linear` matches `nn.Linear` (output and the gradients of input, weight and bias) on every
/// kernel level this CPU supports.
#[test]
fn linear_matches_pytorch() {
    let _restore = RestoreLevel(level());
    for l in ALL_LEVELS.into_iter().filter(|&l| supported(l)) {
        set_level(l).expect("a supported level can be set");
        check_at_current_level();
    }
}
