// The element-wise kernels: one thread per element, each kernel one operation, all with the
// parameters (a, b, out, n, alpha). Binary operations read a and b; unary ones read a only
// (b is passed but unused); backward passes take the forward input or output in a and the
// gradient flowing back in b. A port of cavitation-gpu's elementwise.wgsl, operation for
// operation, compiled without fused multiply-add.
//
// NaN is tested by its bits. tanh and the sigmoid are the CPU kernels' formulas: one
// exponential e^y for y <= 0 from the Cephes polynomial and a 2^n scale, a polynomial for
// tanh near 0. GELU's erf form uses musl's f32 erf with CUDA's expf.

typedef unsigned int u32;

// Whether x is NaN, from its bits.
__device__ __forceinline__ bool is_nan(float x) {
    return (__float_as_uint(x) & 0x7fffffffu) > 0x7f800000u;
}

// 2^k for -126 <= k <= 127, from its exponent bits.
__device__ __forceinline__ float pow2(int k) {
    return __uint_as_float(((u32)(k + 127)) << 23);
}

// e^y for y <= 0: y = k ln 2 + r, e^r from the Cephes expf polynomial, scaled by 2^k in two
// halves.
__device__ float exp_neg(float y_in) {
    float y = y_in < -104.0f ? -104.0f : y_in;
    float k = floorf(y * 1.44269504f + 0.5f);
    float r = y - k * 0.693359375f;
    r = r - k * -2.12194440e-4f;
    float p = 1.9875691e-4f;
    p = p * r + 1.3981999e-3f;
    p = p * r + 8.3334519e-3f;
    p = p * r + 4.1665796e-2f;
    p = p * r + 1.6666665e-1f;
    p = p * r + 0.5f;
    p = p * r * r + r + 1.0f;
    int ki = (int)k;
    int k1 = ki >> 1;
    return p * pow2(k1) * pow2(ki - k1);
}

// tanh(x), as the CPU computes it; NaN stays NaN and the sign of x is kept.
__device__ float tanh_one(float x) {
    if (is_nan(x)) {
        return x;
    }
    u32 s_bits = __float_as_uint(x) & 0x80000000u;
    float ax = fabsf(x);
    float s = ax * ax;
    float q = -6.2741733e-3f;
    q = q * s + 2.1071635e-2f;
    q = q * s + -5.38523e-2f;
    q = q * s + 1.3332586e-1f;
    q = q * s + -3.333332e-1f;
    float small = ax + ax * (s * q);
    float e = exp_neg(-2.0f * ax);
    float big = (1.0f - e) / (1.0f + e);
    float r = ax < 0.55f ? small : big;
    return __uint_as_float(__float_as_uint(r) | s_bits);
}

// The logistic sigmoid, as the CPU computes it; NaN stays NaN.
__device__ float sigmoid_one(float x) {
    if (is_nan(x)) {
        return x;
    }
    float e = exp_neg(-fabsf(x));
    float d = 1.0f + e;
    return x >= 0.0f ? 1.0f / d : e / d;
}

// musl's f32 erf, with CUDA's expf in its tail.
__device__ float erf_one(float x) {
    u32 hx = __float_as_uint(x);
    bool negative = (hx >> 31) != 0u;
    u32 ix = hx & 0x7fffffffu;
    if (ix >= 0x7f800000u) {
        if (is_nan(x)) {
            return x;
        }
        return negative ? -1.0f : 1.0f;
    }
    if (ix < 0x3f580000u) {
        if (ix < 0x38800000u) {
            return x + 1.28379166e-01f * x;
        }
        float z = x * x;
        float r = 1.28379166e-01f + z * (-3.36030394e-01f + z * -1.86261395e-03f);
        float s = 1.0f + z * (3.12324315e-01f + z * (2.16070414e-02f + z * -1.98859372e-03f));
        return x + x * (r / s);
    }
    if (ix < 0x3fa00000u) {
        float s = fabsf(x) - 1.0f;
        float p = 3.65041046e-06f + s * (4.15109307e-01f + s * (-2.09395722e-01f + s * 8.67677554e-02f));
        float q = 1.0f + s * (4.95560974e-01f + s * (3.71248513e-01f + s * 3.92478965e-02f));
        return negative ? -8.42697144e-01f - p / q : 8.42697144e-01f + p / q;
    }
    if (ix >= 0x40800000u) {
        return negative ? -1.0f : 1.0f;
    }
    float ax = fabsf(x);
    float s = 1.0f / (ax * ax);
    float rp;
    float sp;
    if (ix < 0x4036db8cu) {
        rp = -9.88156721e-03f + s * (-5.43658376e-01f + s * (-1.66828310e+00f + s * -6.91554189e-01f));
        sp = 1.0f + s * (4.48581553e+00f + s * (4.10799170e+00f + s * 5.53855181e-01f));
    } else {
        rp = -9.86496918e-03f + s * (-5.48049808e-01f + s * -1.84115684e+00f);
        sp = 1.0f + s * (4.87132740e+00f + s * (3.04982710e+00f + s * -7.61900663e-01f));
    }
    float z = __uint_as_float(ix & 0xffffe000u);
    float r = expf(-z * z - 0.5625f) * expf((z - ax) * (z + ax) + rp / sp);
    return negative ? r / ax - 1.0f : 1.0f - r / ax;
}

// sqrt(2/pi) and GELU's cubic coefficient, rounded to f32 as the CPU rounds them.
#define SQRT_2_OVER_PI 0.7978846f
#define GELU_CUBIC 0.044715f
#define FRAC_1_SQRT_2 0.70710678f
#define FRAC_1_SQRT_2PI 0.39894228f

// The element this thread computes; the kernel returns when it is past the end.
#define ELEMENT                                                   \
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;                \
    if (i >= n) {                                                 \
        return;                                                   \
    }

#define KERNEL(name)                                                                  \
    extern "C" __global__ void name(const float *a, const float *b, float *out, u32 n, \
                                    float alpha)

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
