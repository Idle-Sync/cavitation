// The scalar functions the element-wise and the GLU kernels share, a port of
// cavitation-gpu's activation.wgsl compiled before each of them without fused multiply-add:
// NaN by its bits, the exponential, tanh and the sigmoid as the CPU kernels compute them,
// e^y - 1, ln(1 + t), softplus, ELU, musl's erf with CUDA's expf, and the activations'
// constants.

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

// e^y - 1 for y <= 0, as the CPU's expm1_neg: y = k ln 2 + r, q = e^r - 1 = r + r^2 P(r)
// from the Cephes polynomial, and q 2^k + (2^k - 1); -1 below -17.5, NaN for NaN.
__device__ float expm1_neg(float y) {
    if (is_nan(y)) {
        return y;
    }
    if (y < -17.5f) {
        return -1.0f;
    }
    float k = floorf(y * 1.44269504f + 0.5f);
    float r = y - k * 0.693359375f;
    r = r - k * -2.12194440e-4f;
    float p = 1.9875691e-4f;
    p = p * r + 1.3981999e-3f;
    p = p * r + 8.3334519e-3f;
    p = p * r + 4.1665796e-2f;
    p = p * r + 1.6666665e-1f;
    p = p * r + 0.5f;
    float q = p * r * r + r;
    float s = pow2((int)k);
    return q * s + (s - 1.0f);
}

// ln(1 + t) for t in [0, 1], as the CPU's log1p_unit: m = 1 + t rounded, its rounding error
// c = t - (m - 1), m = 2^k f with f in [sqrt(1/2), sqrt 2), ln f from the Cephes logf
// polynomial, and k ln 2 + ln f + c / m.
__device__ float log1p_unit(float t) {
    float m = 1.0f + t;
    float c = t - (m - 1.0f);
    bool high = m > 1.41421354f;
    float f = high ? m * 0.5f : m;
    float k = high ? 1.0f : 0.0f;
    float x = f - 1.0f;
    float z = x * x;
    float p = 7.0376836e-2f;
    p = p * x + -1.1514610e-1f;
    p = p * x + 1.1676999e-1f;
    p = p * x + -1.2420141e-1f;
    p = p * x + 1.4249323e-1f;
    p = p * x + -1.6668058e-1f;
    p = p * x + 2.0000715e-1f;
    p = p * x + -2.4999994e-1f;
    p = p * x + 3.3333330e-1f;
    float y = x * z * p;
    y = y + k * -2.12194440e-4f;
    y = y - 0.5f * z;
    float r = x + y;
    r = r + k * 0.693359375f;
    return r + c / m;
}

// ln(1 + e^x), as the CPU's softplus kernel: max(x, 0) + ln(1 + e^-|x|); NaN stays NaN.
__device__ float softplus_one(float x) {
    if (is_nan(x)) {
        return x;
    }
    float l = log1p_unit(exp_neg(-fabsf(x)));
    return (x > 0.0f ? x : 0.0f) + l;
}

// ELU with alpha, as the CPU's kernel: x above 0, alpha · x below 2^-25 in size (which keeps
// -0), alpha · (e^x - 1) otherwise; NaN stays NaN.
__device__ float elu_one(float alpha, float x) {
    if (is_nan(x) || x > 0.0f) {
        return x;
    }
    if (fabsf(x) < 2.98023224e-8f) {
        return alpha * x;
    }
    return alpha * expm1_neg(x);
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
            // musl's extra branch below 2^-119 only avoids an underflow flag; this formula
            // differs from it in at most the last bits of a tiny erf, and GELU adds erf to 1.
            return x + 1.28379166e-01f * x;
        }
        float z = x * x;
        float r = 1.28379166e-01f + z * (-3.36030394e-01f + z * -1.86261395e-03f);
        float s = 1.0f + z * (3.12324315e-01f + z * (2.16070414e-02f + z * -1.98859372e-03f));
        return x + x * (r / s);
    }
    if (ix < 0x3fa00000u) {
        float s = fabsf(x) - 1.0f;
        float p = 3.65041046e-06f
                  + s * (4.15109307e-01f + s * (-2.09395722e-01f + s * 8.67677554e-02f));
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
        rp = -9.88156721e-03f
             + s * (-5.43658376e-01f + s * (-1.66828310e+00f + s * -6.91554189e-01f));
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
// SELU's scale and alpha rounded to f32, and their product rounded once, as the CPU has them.
#define SELU_SCALE 1.05070102f
#define SELU_ALPHA 1.67326319f
#define SELU_SCALE_ALPHA 1.75809932f
// 1/6 rounded to f32, the hard sigmoid's slope.
#define SIXTH 0.166666672f
// Mish's softplus threshold.
#define MISH_THRESHOLD 20.0f

