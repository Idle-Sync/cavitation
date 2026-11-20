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
// musl's f32 erf with WGSL's exp. Workgroups are numbered over two dimensions
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

// Whether x is NaN, from its bits.
fn is_nan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

// The element this thread computes, numbered over a split grid.
fn element(w: vec3<u32>, n: vec3<u32>, l: u32) -> u32 {
    return (w.x + w.y * n.x) * WORKGROUP + l;
}

// x itself, from its bits xor'd with the uniform's zero, which the shader compiler cannot
// see: a value passed through it is used as rounded. Without it a compiler may fuse a
// product into the add that reads it, simplify (1 + t) - 1 to t and 0 - s to -s, or
// regroup t - (m - 1) as (t + 1) - m, as the Vulkan driver and DXC do on the NVIDIA GTX
// 1660 Super, which a twin of the CPU's bits, or a compensated sum, cannot allow.
fn held(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ params.zero);
}

// x / (3h) for h = 1 or 2, correctly rounded as the CPU divides, for x / (3h) normal: a quotient q0
// from WGSL's division (which may be a product by the reciprocal, within a few ulp), and of q0 and
// the two floats on each side of it, the one whose remainder x - 3h q = (x - 2h q) - h q (held, so
// it is not regrouped) is smallest. Each step of the remainder is exact (h q is a power-of-two
// multiple, and each difference is of floats within a factor of 2), so the smallest remainder is
// the nearest quotient; x / 3 and x / 6 are never halfway between two floats. 0, infinities and NaN
// give WGSL's quotient.
fn div_by_3h(x: f32, h: f32) -> f32 {
    let q0 = held(x / (3.0 * h));
    var best = q0;
    var least = abs(held(x - 2.0 * h * q0) - h * q0);
    let bits = bitcast<u32>(q0);
    for (var k = 0u; k < 4u; k++) {
        let q = bitcast<f32>(select(bits - 1u - (k & 1u), bits + 1u + (k & 1u), k < 2u));
        let r = abs(held(x - 2.0 * h * q) - h * q);
        if (r < least) {
            least = r;
            best = q;
        }
    }
    return held(best);
}

// 2^k for -126 <= k <= 127, from its exponent bits.
fn pow2(k: i32) -> f32 {
    return bitcast<f32>(u32(k + 127) << 23u);
}

// e^y for y <= 0: y = k ln 2 + r, e^r from the Cephes expf polynomial, scaled by 2^k in two
// halves (a result below the normal range flushes on the GPU).
fn exp_neg(y_in: f32) -> f32 {
    let y = max(y_in, -104.0);
    let k = floor(y * 1.44269504 + 0.5);
    var r = y - k * 0.693359375;
    r = r - k * -2.12194440e-4;
    var p = 1.9875691e-4;
    p = p * r + 1.3981999e-3;
    p = p * r + 8.3334519e-3;
    p = p * r + 4.1665796e-2;
    p = p * r + 1.6666665e-1;
    p = p * r + 0.5;
    p = p * r * r + r + 1.0;
    let ki = i32(k);
    let k1 = ki >> 1u;
    return p * pow2(k1) * pow2(ki - k1);
}

// tanh(x), as the CPU computes it; NaN stays NaN and the sign of x is kept.
fn tanh_one(x: f32) -> f32 {
    if (is_nan(x)) {
        return x;
    }
    let s_bits = bitcast<u32>(x) & 0x80000000u;
    let ax = abs(x);
    let s = ax * ax;
    var q = -6.2741733e-3;
    q = q * s + 2.1071635e-2;
    q = q * s + -5.38523e-2;
    q = q * s + 1.3332586e-1;
    q = q * s + -3.333332e-1;
    let small = ax + ax * (s * q);
    let e = exp_neg(-2.0 * ax);
    let big = (1.0 - e) / (1.0 + e);
    let r = select(big, small, ax < 0.55);
    return bitcast<f32>(bitcast<u32>(r) | s_bits);
}

// The logistic sigmoid, as the CPU computes it; NaN stays NaN.
fn sigmoid_one(x: f32) -> f32 {
    if (is_nan(x)) {
        return x;
    }
    let e = exp_neg(-abs(x));
    let d = 1.0 + e;
    return select(e / d, 1.0 / d, x >= 0.0);
}

// e^y - 1 for y <= 0, as the CPU's expm1_neg: y = k ln 2 + r, q = e^r - 1 = r + r^2 P(r)
// from the Cephes polynomial, and q 2^k + (2^k - 1); -1 below -17.5, NaN for NaN.
fn expm1_neg(y: f32) -> f32 {
    if (is_nan(y)) {
        return y;
    }
    if (y < -17.5) {
        return -1.0;
    }
    let k = floor(y * 1.44269504 + 0.5);
    var r = y - k * 0.693359375;
    r = r - k * -2.12194440e-4;
    var p = 1.9875691e-4;
    p = p * r + 1.3981999e-3;
    p = p * r + 8.3334519e-3;
    p = p * r + 4.1665796e-2;
    p = p * r + 1.6666665e-1;
    p = p * r + 0.5;
    let q = p * r * r + r;
    let s = pow2(i32(k));
    return q * s + (s - 1.0);
}

// ln(1 + t) for t in [0, 1], as the CPU's log1p_unit: m = 1 + t rounded, its rounding error
// c = t - (m - 1), m = 2^k f with f in [sqrt(1/2), sqrt 2), ln f from the Cephes logf
// polynomial, and k ln 2 + ln f + c / m.
fn log1p_unit(t: f32) -> f32 {
    let m = held(1.0 + t);
    let c = t - held(m - 1.0);
    let high = m > 0x1.6a09e6p+0f;
    let f = select(m, m * 0.5, high);
    let k = select(0.0, 1.0, high);
    let x = f - 1.0;
    let z = x * x;
    var p = 7.0376836e-2;
    p = p * x + -1.1514610e-1;
    p = p * x + 1.1676999e-1;
    p = p * x + -1.2420141e-1;
    p = p * x + 1.4249323e-1;
    p = p * x + -1.6668058e-1;
    p = p * x + 2.0000715e-1;
    p = p * x + -2.4999994e-1;
    p = p * x + 3.3333330e-1;
    var y = x * z * p;
    y = y + k * -2.12194440e-4;
    y = y - 0.5 * z;
    var r = x + y;
    r = r + k * 0.693359375;
    return r + c / m;
}

// ln(1 + e^x), as the CPU's softplus kernel: max(x, 0) + ln(1 + e^-|x|); NaN stays NaN.
fn softplus_one(x: f32) -> f32 {
    if (is_nan(x)) {
        return x;
    }
    let l = log1p_unit(exp_neg(-abs(x)));
    return select(0.0, x, x > 0.0) + l;
}

// ELU with alpha, as the CPU's kernel: x above 0, alpha · x below 2^-25 in size (which keeps
// -0), alpha · (e^x - 1) otherwise; NaN stays NaN.
fn elu_one(alpha: f32, x: f32) -> f32 {
    if (is_nan(x) || x > 0.0) {
        return x;
    }
    if (abs(x) < 2.98023224e-8) {
        return alpha * x;
    }
    return alpha * expm1_neg(x);
}

// musl's f32 erf, with WGSL's exp in its tail.
fn erf_one(x: f32) -> f32 {
    let hx = bitcast<u32>(x);
    let negative = (hx >> 31u) != 0u;
    let ix = hx & 0x7fffffffu;
    if (ix >= 0x7f800000u) {
        if (is_nan(x)) {
            return x;
        }
        return select(1.0, -1.0, negative);
    }
    if (ix < 0x3f580000u) {
        if (ix < 0x38800000u) {
            // musl's extra branch below 2^-119 only avoids an underflow flag; here a flushed
            // 0.128 · x changes erf by at most its last bits, and GELU adds erf to 1.
            return x + 1.28379166e-01 * x;
        }
        let z = x * x;
        let r = 1.28379166e-01 + z * (-3.36030394e-01 + z * -1.86261395e-03);
        let s = 1.0 + z * (3.12324315e-01 + z * (2.16070414e-02 + z * -1.98859372e-03));
        return x + x * (r / s);
    }
    if (ix < 0x3fa00000u) {
        let s = abs(x) - 1.0;
        let p = 3.65041046e-06 + s * (4.15109307e-01 + s * (-2.09395722e-01 + s * 8.67677554e-02));
        let q = 1.0 + s * (4.95560974e-01 + s * (3.71248513e-01 + s * 3.92478965e-02));
        return select(8.42697144e-01 + p / q, -8.42697144e-01 - p / q, negative);
    }
    if (ix >= 0x40800000u) {
        return select(1.0, -1.0, negative);
    }
    let ax = abs(x);
    let s = 1.0 / (ax * ax);
    var rp: f32;
    var sp: f32;
    if (ix < 0x4036db8cu) {
        rp = -9.88156721e-03 + s * (-5.43658376e-01 + s * (-1.66828310e+00 + s * -6.91554189e-01));
        sp = 1.0 + s * (4.48581553e+00 + s * (4.10799170e+00 + s * 5.53855181e-01));
    } else {
        rp = -9.86496918e-03 + s * (-5.48049808e-01 + s * -1.84115684e+00);
        sp = 1.0 + s * (4.87132740e+00 + s * (3.04982710e+00 + s * -7.61900663e-01));
    }
    let z = bitcast<f32>(ix & 0xffffe000u);
    let r = exp(-z * z - 0.5625) * exp((z - ax) * (z + ax) + rp / sp);
    return select(1.0 - r / ax, r / ax - 1.0, negative);
}

// sqrt(2/pi) and GELU's cubic coefficient, rounded to f32 as the CPU rounds them.
const SQRT_2_OVER_PI: f32 = 0.7978846;
const GELU_CUBIC: f32 = 0.044715;
const FRAC_1_SQRT_2: f32 = 0.70710678;
const FRAC_1_SQRT_2PI: f32 = 0.39894228;
// SELU's scale and alpha rounded to f32, and their product rounded once, as the CPU has them.
const SELU_SCALE: f32 = 0x1.0cfabep+0f;
const SELU_ALPHA: f32 = 0x1.ac5afap+0f;
const SELU_SCALE_ALPHA: f32 = 0x1.c212ccp+0f;
// 1/6 rounded to f32, the hard sigmoid's slope.
const SIXTH: f32 = 0x1.555556p-3f;
// Mish's softplus threshold.
const MISH_THRESHOLD: f32 = 20.0;

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
