// The element-wise kernels: one thread per element, each entry point one operation.
// Bindings: a (read), b (read), out (read-write), and a uniform with the length and the
// operation's constants. Binary operations read a and b; unary ones read a only (b is bound
// but unused); backward passes take the forward input or output in a and the gradient
// flowing back in b. Dropout is in dropout.wgsl, whose mask binding holds u32 words.
//
// NaN is tested by its bits, never with a comparison, because backends may fold x != x.
// tanh and the sigmoid are the CPU kernels' formulas, ported operation for operation: one
// exponential e^y for y <= 0 from the Cephes polynomial and a 2^n scale, a polynomial for
// tanh near 0. ELU's e^y - 1 and softplus's ln(1 + t) are the CPU kernels' too, and SELU,
// CELU, SiLU, Mish, softplus at any beta and log-sigmoid are built on them as the CPU builds
// them. The backward passes the CPU takes with std's exp use exp_neg. GELU's erf form uses
// musl's f32 erf with WGSL's exp. Those scalar functions are in activation.wgsl, which
// elementwise.rs appends to this text. Workgroups are numbered over two dimensions
// (w.x + w.y * n.x), as a split dispatch needs.

struct Params {
    n: u32,
    alpha: f32,
    scale: f32,
    // Always 0: see held().
    zero: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP: u32 = 256u;

// The element this thread computes, numbered over a split grid.
fn element(w: vec3<u32>, n: vec3<u32>, l: u32) -> u32 {
    return (w.x + w.y * n.x) * WORKGROUP + l;
}

@compute @workgroup_size(256)
fn add(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i < params.n) { out[i] = a[i] + b[i]; }
}

@compute @workgroup_size(256)
fn mul(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i < params.n) { out[i] = a[i] * b[i]; }
}

@compute @workgroup_size(256)
fn scale(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i < params.n) { out[i] = params.alpha * a[i]; }
}

// relu(x) = 0 for x <= 0, x otherwise, NaN for NaN (as the CPU's relu, which also turns -0
// into +0).
@compute @workgroup_size(256)
fn relu(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    out[i] = select(select(x, 0.0, x <= 0.0), x, is_nan(x));
}

// The gradient through relu: grad · (1 above 0, 0 at or below, NaN for NaN).
@compute @workgroup_size(256)
fn relu_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let d = select(select(0.0, 1.0, x > 0.0), x, is_nan(x));
    out[i] = b[i] * d;
}

// leaky_relu(x) = x above 0, alpha · x at or below; NaN stays NaN.
@compute @workgroup_size(256)
fn leaky_relu(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    out[i] = select(select(params.alpha * x, x, x > 0.0), x, is_nan(x));
}

// The gradient through leaky_relu: grad · (1 above 0, alpha at or below, NaN for NaN).
@compute @workgroup_size(256)
fn leaky_relu_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let d = select(select(params.alpha, 1.0, x > 0.0), x, is_nan(x));
    out[i] = b[i] * d;
}

@compute @workgroup_size(256)
fn tanh_forward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i < params.n) { out[i] = tanh_one(a[i]); }
}

// The gradient through tanh from its output y: grad · (1 − y²).
@compute @workgroup_size(256)
fn tanh_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let y = a[i];
    out[i] = b[i] * (1.0 - y * y);
}

@compute @workgroup_size(256)
fn sigmoid_forward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i < params.n) { out[i] = sigmoid_one(a[i]); }
}

// The gradient through the sigmoid from its output y: grad · y · (1 − y).
@compute @workgroup_size(256)
fn sigmoid_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let y = a[i];
    out[i] = b[i] * (y * (1.0 - y));
}

// GELU, tanh form: 0.5 · x · (1 + tanh(sqrt(2/pi) · (x + 0.044715 · x³))).
@compute @workgroup_size(256)
fn gelu_tanh_forward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let t = tanh_one(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x));
    out[i] = 0.5 * x * (1.0 + t);
}

// The gradient through GELU's tanh form from its input x.
@compute @workgroup_size(256)
fn gelu_tanh_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let x2 = x * x;
    let t = tanh_one(SQRT_2_OVER_PI * (x + GELU_CUBIC * x2 * x));
    // Where tanh saturates, 1 - t² is exactly 0 and the second term is dropped: 0 for any finite
    // du, and its limit where x² or du overflows (0 · ∞ would be NaN), as the CPU does.
    let first = 0.5 * (1.0 + t);
    let flat = 1.0 - t * t;
    if (flat == 0.0) {
        out[i] = b[i] * first;
        return;
    }
    let du = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * x2);
    out[i] = b[i] * (first + 0.5 * x * flat * du);
}

// GELU, erf form: 0.5 · x · (1 + erf(x / sqrt 2)).
@compute @workgroup_size(256)
fn gelu_erf_forward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    out[i] = 0.5 * x * (1.0 + erf_one(x * FRAC_1_SQRT_2));
}

// The gradient through GELU's erf form from its input x: grad · (Φ(x) + x · φ(x)).
@compute @workgroup_size(256)
fn gelu_erf_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let cdf = 0.5 * (1.0 + erf_one(x * FRAC_1_SQRT_2));
    let pdf = FRAC_1_SQRT_2PI * exp(-(0.5 * x * x));
    // Where the density underflows to 0 the x · φ term is dropped, as the CPU does: 0 for any
    // finite x, and its limit where x is infinite.
    if (pdf == 0.0) {
        out[i] = b[i] * cdf;
        return;
    }
    out[i] = b[i] * (cdf + x * pdf);
}

// ELU with alpha: x above 0, alpha · (e^x - 1) at or below.
@compute @workgroup_size(256)
fn elu(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    out[i] = elu_one(params.alpha, x);
}

// The gradient through ELU from its input x: grad · (1 above 0, alpha · e^x at or below,
// NaN for NaN).
@compute @workgroup_size(256)
fn elu_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    var d = 1.0;
    if (is_nan(x)) {
        d = x;
    } else if (x <= 0.0) {
        d = params.alpha * exp_neg(x);
    }
    out[i] = b[i] * d;
}

// SELU: scale · elu(x) with SELU's alpha.
@compute @workgroup_size(256)
fn selu(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    out[i] = SELU_SCALE * elu_one(SELU_ALPHA, x);
}

// The gradient through SELU from its input x: grad · (scale above 0, scale · alpha · e^x
// at or below, NaN for NaN).
@compute @workgroup_size(256)
fn selu_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    var d = SELU_SCALE;
    if (is_nan(x)) {
        d = x;
    } else if (x <= 0.0) {
        d = SELU_SCALE_ALPHA * exp_neg(x);
    }
    out[i] = b[i] * d;
}

// CELU with alpha > 0: x above 0, alpha · (e^(x / alpha) - 1) at or below, taken as the
// CPU's kernel takes it, as ELU with alpha of x / alpha.
@compute @workgroup_size(256)
fn celu(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let e = elu_one(params.alpha, x / params.alpha);
    out[i] = select(x, e, x <= 0.0 || is_nan(x));
}

// The gradient through CELU from its input x: grad · (1 above 0, e^(x / alpha) at or
// below, NaN for NaN).
@compute @workgroup_size(256)
fn celu_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    var d = 1.0;
    if (is_nan(x)) {
        d = x;
    } else if (x <= 0.0) {
        d = exp_neg(x / params.alpha);
    }
    out[i] = b[i] * d;
}

// SiLU: x · sigmoid(x).
@compute @workgroup_size(256)
fn silu(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    out[i] = x * sigmoid_one(x);
}

// The gradient through SiLU from its input x: grad · s · (1 + x · (1 - s)), s = sigmoid(x).
@compute @workgroup_size(256)
fn silu_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let s = sigmoid_one(x);
    out[i] = b[i] * (s * (1.0 + x * (1.0 - s)));
}

// Mish: x · tanh(softplus(x)), softplus at beta 1 and threshold 20 (x itself above 20).
@compute @workgroup_size(256)
fn mish(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let sp = select(softplus_one(x), x, x > MISH_THRESHOLD);
    out[i] = x * tanh_one(sp);
}

// The gradient through Mish from its input x: grad · (t + x · sigmoid(x) · (1 - t²)),
// t = tanh(softplus(x)).
@compute @workgroup_size(256)
fn mish_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let t = tanh_one(select(softplus_one(x), x, x > MISH_THRESHOLD));
    let s = sigmoid_one(x);
    out[i] = b[i] * (t + x * s * (1.0 - t * t));
}

// Softplus with beta (alpha) and threshold (scale): x where beta · x is above the
// threshold, ln(1 + e^(beta · x)) / beta at or below.
@compute @workgroup_size(256)
fn softplus(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let z = params.alpha * x;
    out[i] = select(softplus_one(z) / params.alpha, x, z > params.scale);
}

// The gradient through softplus from its input x: grad · (1 where beta · x is above the
// threshold, sigmoid(beta · x) at or below).
@compute @workgroup_size(256)
fn softplus_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let z = params.alpha * x;
    out[i] = b[i] * select(sigmoid_one(z), 1.0, z > params.scale);
}

// Softsign: x / (1 + |x|).
@compute @workgroup_size(256)
fn softsign(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    out[i] = x / (1.0 + abs(x));
}

// The gradient through softsign from its input x: grad · 1 / (1 + |x|)².
@compute @workgroup_size(256)
fn softsign_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let d = 1.0 + abs(x);
    out[i] = b[i] * (1.0 / (d * d));
}

// Hardtanh: x clamped to [min (alpha), max (scale)]; NaN stays NaN.
@compute @workgroup_size(256)
fn hardtanh(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let high = select(x, params.scale, x > params.scale);
    out[i] = select(select(high, params.alpha, x < params.alpha), x, is_nan(x));
}

// The gradient through hardtanh from its input x: grad · (1 strictly inside (min, max), 0
// at and beyond the bounds, NaN for NaN).
@compute @workgroup_size(256)
fn hardtanh_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let inside = x > params.alpha && x < params.scale;
    out[i] = b[i] * select(select(0.0, 1.0, inside), x, is_nan(x));
}

// The hard sigmoid: 0 up to -3, 1 from 3, x / 6 + 1/2 between; NaN stays NaN.
@compute @workgroup_size(256)
fn hardsigmoid(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let y = select(select(div_by_3h(x, 2.0) + 0.5, 1.0, x >= 3.0), 0.0, x <= -3.0);
    out[i] = select(y, x, is_nan(x));
}

// The gradient through the hard sigmoid from its input x: grad · (1/6 strictly inside
// (-3, 3), 0 elsewhere, NaN for NaN).
@compute @workgroup_size(256)
fn hardsigmoid_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let inside = x > -3.0 && x < 3.0;
    out[i] = b[i] * select(select(0.0, SIXTH, inside), x, is_nan(x));
}

// The hard swish: 0 up to -3, x from 3, x · (x + 3) / 6 between; NaN stays NaN.
@compute @workgroup_size(256)
fn hardswish(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let y = select(select(div_by_3h(held(x * (x + 3.0)), 2.0), x, x >= 3.0), 0.0, x <= -3.0);
    out[i] = select(y, x, is_nan(x));
}

// The gradient through the hard swish from its input x, with PyTorch's boundaries: grad ·
// (0 below -3, 1 above 3, x / 3 + 1/2 on [-3, 3], NaN for NaN).
@compute @workgroup_size(256)
fn hardswish_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    let d = select(select(div_by_3h(x, 1.0) + 0.5, 1.0, x > 3.0), 0.0, x < -3.0);
    out[i] = b[i] * select(d, x, is_nan(x));
}

// Log-sigmoid: ln sigmoid(x) = 0 - softplus(-x), as the CPU's kernel takes it.
@compute @workgroup_size(256)
fn log_sigmoid(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    out[i] = held(0.0) - softplus_one(-x);
}

// The gradient through log-sigmoid from its input x: grad · sigmoid(-x).
@compute @workgroup_size(256)
fn log_sigmoid_backward(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = element(w, g, l);
    if (i >= params.n) { return; }
    let x = a[i];
    out[i] = b[i] * sigmoid_one(-x);
}
