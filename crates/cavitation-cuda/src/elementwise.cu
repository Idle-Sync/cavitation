// The element-wise kernels: one thread per element, each kernel one operation, all with the
// parameters (a, b, out, n, alpha, scale). Binary operations read a and b; unary ones read a
// only (b is passed but unused); backward passes take the forward input or output in a and
// the gradient flowing back in b. A port of cavitation-gpu's elementwise.wgsl, operation for
// operation, compiled without fused multiply-add; NVRTC neither regroups nor simplifies
// floating-point operations and divides correctly rounded, so the WGSL's guards against
// those (held, the corrected division by 3 and 6) are plain operations here.
//
// NaN is tested by its bits. tanh and the sigmoid are the CPU kernels' formulas: one
// exponential e^y for y <= 0 from the Cephes polynomial and a 2^n scale, a polynomial for
// tanh near 0. ELU's e^y - 1 and softplus's ln(1 + t) are the CPU kernels' too, and SELU,
// CELU, SiLU, Mish, softplus at any beta and log-sigmoid are built on them as the CPU builds
// them. The backward passes the CPU takes with std's exp use exp_neg. GELU's erf form uses
// musl's f32 erf with CUDA's expf. Those scalar functions are in activation.cu, which
// elementwise.rs compiles before this text.

// The element this thread computes; the kernel returns when it is past the end.
#define ELEMENT                                                   \
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;                \
    if (i >= n) {                                                 \
        return;                                                   \
    }

#define KERNEL(name)                                                                  \
    extern "C" __global__ void name(const float *a, const float *b, float *out, u32 n, \
                                    float alpha, float scale)

KERNEL(add) { ELEMENT out[i] = a[i] + b[i]; }

KERNEL(mul) { ELEMENT out[i] = a[i] * b[i]; }

KERNEL(scale) { ELEMENT out[i] = alpha * a[i]; }

// relu(x) = 0 for x <= 0, x otherwise, NaN for NaN (as the CPU's relu, which also turns -0
// into +0).
KERNEL(relu) {
    ELEMENT float x = a[i];
    out[i] = is_nan(x) ? x : (x <= 0.0f ? 0.0f : x);
}

// The gradient through relu: grad · (1 above 0, 0 at or below, NaN for NaN).
KERNEL(relu_backward) {
    ELEMENT float x = a[i];
    float d = is_nan(x) ? x : (x > 0.0f ? 1.0f : 0.0f);
    out[i] = b[i] * d;
}

// leaky_relu(x) = x above 0, alpha · x at or below; NaN stays NaN.
KERNEL(leaky_relu) {
    ELEMENT float x = a[i];
    out[i] = is_nan(x) ? x : (x > 0.0f ? x : alpha * x);
}

// The gradient through leaky_relu: grad · (1 above 0, alpha at or below, NaN for NaN).
KERNEL(leaky_relu_backward) {
    ELEMENT float x = a[i];
    float d = is_nan(x) ? x : (x > 0.0f ? 1.0f : alpha);
    out[i] = b[i] * d;
}

KERNEL(tanh_forward) { ELEMENT out[i] = tanh_one(a[i]); }

// The gradient through tanh from its output y: grad · (1 − y²).
KERNEL(tanh_backward) {
    ELEMENT float y = a[i];
    out[i] = b[i] * (1.0f - y * y);
}

KERNEL(sigmoid_forward) { ELEMENT out[i] = sigmoid_one(a[i]); }

// The gradient through the sigmoid from its output y: grad · y · (1 − y).
KERNEL(sigmoid_backward) {
    ELEMENT float y = a[i];
    out[i] = b[i] * (y * (1.0f - y));
}

// GELU, tanh form: 0.5 · x · (1 + tanh(sqrt(2/pi) · (x + 0.044715 · x³))).
KERNEL(gelu_tanh_forward) {
    ELEMENT float x = a[i];
    float t = tanh_one(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x));
    out[i] = 0.5f * x * (1.0f + t);
}

// The gradient through GELU's tanh form from its input x.
KERNEL(gelu_tanh_backward) {
    ELEMENT float x = a[i];
    float x2 = x * x;
    float t = tanh_one(SQRT_2_OVER_PI * (x + GELU_CUBIC * x2 * x));
    // Where tanh saturates, 1 - t² is exactly 0 and the second term is dropped: 0 for any
    // finite du, and its limit where x² or du overflows (0 · ∞ would be NaN), as the CPU does.
    float first = 0.5f * (1.0f + t);
    float flat = 1.0f - t * t;
    if (flat == 0.0f) {
        out[i] = b[i] * first;
        return;
    }
    float du = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_CUBIC * x2);
    out[i] = b[i] * (first + 0.5f * x * flat * du);
}

// GELU, erf form: 0.5 · x · (1 + erf(x / sqrt 2)).
KERNEL(gelu_erf_forward) {
    ELEMENT float x = a[i];
    out[i] = 0.5f * x * (1.0f + erf_one(x * FRAC_1_SQRT_2));
}

// The gradient through GELU's erf form from its input x: grad · (Φ(x) + x · φ(x)).
KERNEL(gelu_erf_backward) {
    ELEMENT float x = a[i];
    float cdf = 0.5f * (1.0f + erf_one(x * FRAC_1_SQRT_2));
    float pdf = FRAC_1_SQRT_2PI * expf(-(0.5f * x * x));
    // Where the density underflows to 0 the x · φ term is dropped, as the CPU does: 0 for any
    // finite x, and its limit where x is infinite.
    if (pdf == 0.0f) {
        out[i] = b[i] * cdf;
        return;
    }
    out[i] = b[i] * (cdf + x * pdf);
}

// ELU with alpha: x above 0, alpha · (e^x - 1) at or below.
KERNEL(elu) { ELEMENT out[i] = elu_one(alpha, a[i]); }

// The gradient through ELU from its input x: grad · (1 above 0, alpha · e^x at or below, NaN
// for NaN).
KERNEL(elu_backward) {
    ELEMENT float x = a[i];
    float d = is_nan(x) ? x : (x > 0.0f ? 1.0f : alpha * exp_neg(x));
    out[i] = b[i] * d;
}

// SELU: scale · elu(x) with SELU's alpha.
KERNEL(selu) { ELEMENT out[i] = SELU_SCALE * elu_one(SELU_ALPHA, a[i]); }

// The gradient through SELU from its input x: grad · (scale above 0, scale · alpha · e^x at
// or below, NaN for NaN).
KERNEL(selu_backward) {
    ELEMENT float x = a[i];
    float d = is_nan(x) ? x : (x > 0.0f ? SELU_SCALE : SELU_SCALE_ALPHA * exp_neg(x));
    out[i] = b[i] * d;
}

// CELU with alpha > 0: x above 0, alpha · (e^(x / alpha) - 1) at or below, taken as the CPU's
// kernel takes it, as ELU with alpha of x / alpha.
KERNEL(celu) {
    ELEMENT float x = a[i];
    float e = elu_one(alpha, x / alpha);
    out[i] = (x <= 0.0f || is_nan(x)) ? e : x;
}

// The gradient through CELU from its input x: grad · (1 above 0, e^(x / alpha) at or below,
// NaN for NaN).
KERNEL(celu_backward) {
    ELEMENT float x = a[i];
    float d = is_nan(x) ? x : (x > 0.0f ? 1.0f : exp_neg(x / alpha));
    out[i] = b[i] * d;
}

// SiLU: x · sigmoid(x).
KERNEL(silu) {
    ELEMENT float x = a[i];
    out[i] = x * sigmoid_one(x);
}

// The gradient through SiLU from its input x: grad · s · (1 + x · (1 - s)), s = sigmoid(x).
KERNEL(silu_backward) {
    ELEMENT float x = a[i];
    float s = sigmoid_one(x);
    out[i] = b[i] * (s * (1.0f + x * (1.0f - s)));
}

// Mish: x · tanh(softplus(x)), softplus at beta 1 and threshold 20 (x itself above 20).
KERNEL(mish) {
    ELEMENT float x = a[i];
    float sp = x > MISH_THRESHOLD ? x : softplus_one(x);
    out[i] = x * tanh_one(sp);
}

// The gradient through Mish from its input x: grad · (t + x · sigmoid(x) · (1 - t²)),
// t = tanh(softplus(x)).
KERNEL(mish_backward) {
    ELEMENT float x = a[i];
    float t = tanh_one(x > MISH_THRESHOLD ? x : softplus_one(x));
    float s = sigmoid_one(x);
    out[i] = b[i] * (t + x * s * (1.0f - t * t));
}

// Softplus with beta (alpha) and threshold (scale): x where beta · x is above the threshold,
// ln(1 + e^(beta · x)) / beta at or below.
KERNEL(softplus) {
    ELEMENT float x = a[i];
    float z = alpha * x;
    out[i] = z > scale ? x : softplus_one(z) / alpha;
}

// The gradient through softplus from its input x: grad · (1 where beta · x is above the
// threshold, sigmoid(beta · x) at or below).
KERNEL(softplus_backward) {
    ELEMENT float x = a[i];
    float z = alpha * x;
    out[i] = b[i] * (z > scale ? 1.0f : sigmoid_one(z));
}

// Softsign: x / (1 + |x|).
KERNEL(softsign) {
    ELEMENT float x = a[i];
    out[i] = x / (1.0f + fabsf(x));
}

// The gradient through softsign from its input x: grad · 1 / (1 + |x|)².
KERNEL(softsign_backward) {
    ELEMENT float x = a[i];
    float d = 1.0f + fabsf(x);
    out[i] = b[i] * (1.0f / (d * d));
}

// Hardtanh: x clamped to [min (alpha), max (scale)]; NaN stays NaN.
KERNEL(hardtanh) {
    ELEMENT float x = a[i];
    out[i] = is_nan(x) ? x : (x < alpha ? alpha : (x > scale ? scale : x));
}

// The gradient through hardtanh from its input x: grad · (1 strictly inside (min, max), 0 at
// and beyond the bounds, NaN for NaN).
KERNEL(hardtanh_backward) {
    ELEMENT float x = a[i];
    float d = is_nan(x) ? x : ((x > alpha && x < scale) ? 1.0f : 0.0f);
    out[i] = b[i] * d;
}

// The hard sigmoid: 0 up to -3, 1 from 3, x / 6 + 1/2 between; NaN stays NaN.
KERNEL(hardsigmoid) {
    ELEMENT float x = a[i];
    out[i] = is_nan(x) ? x : (x <= -3.0f ? 0.0f : (x >= 3.0f ? 1.0f : x / 6.0f + 0.5f));
}

// The gradient through the hard sigmoid from its input x: grad · (1/6 strictly inside
// (-3, 3), 0 elsewhere, NaN for NaN).
KERNEL(hardsigmoid_backward) {
    ELEMENT float x = a[i];
    float d = is_nan(x) ? x : ((x > -3.0f && x < 3.0f) ? SIXTH : 0.0f);
    out[i] = b[i] * d;
}

// The hard swish: 0 up to -3, x from 3, x · (x + 3) / 6 between; NaN stays NaN.
KERNEL(hardswish) {
    ELEMENT float x = a[i];
    out[i] = is_nan(x) ? x : (x <= -3.0f ? 0.0f : (x >= 3.0f ? x : x * (x + 3.0f) / 6.0f));
}

// The gradient through the hard swish from its input x, with PyTorch's boundaries: grad · (0
// below -3, 1 above 3, x / 3 + 1/2 on [-3, 3], NaN for NaN).
KERNEL(hardswish_backward) {
    ELEMENT float x = a[i];
    float d = is_nan(x) ? x : (x < -3.0f ? 0.0f : (x > 3.0f ? 1.0f : x / 3.0f + 0.5f));
    out[i] = b[i] * d;
}

// Log-sigmoid: ln sigmoid(x) = 0 - softplus(-x), as the CPU's kernel takes it.
KERNEL(log_sigmoid) {
    ELEMENT float x = a[i];
    out[i] = 0.0f - softplus_one(-x);
}

// The gradient through log-sigmoid from its input x: grad · sigmoid(-x).
KERNEL(log_sigmoid_backward) {
    ELEMENT float x = a[i];
    out[i] = b[i] * sigmoid_one(-x);
}
