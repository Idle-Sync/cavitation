// Softmax and log-softmax over each row of a [rows][width] batch, forward and backward: a port
// of cavitation-gpu's kernels/softmax.rs, operation for operation, compiled without fused
// multiply-add. One thread per row, which reads the row's values in the CPU's order. Every
// kernel takes (const float *a, const float *b, float *out, u32 rows, u32 width): a is the
// forward input, or the forward output for a backward pass, and b the gradient flowing back
// (a again, unused, forward).

typedef unsigned int u32;

// The bits of the NaN a bad row gets, and of -inf.
#define NAN_BITS 0x7fc00000u
#define NEG_INF_BITS 0xff800000u

// Whether x is -inf, from its bits.
__device__ __forceinline__ bool is_neg_inf(float x) {
    return __float_as_uint(x) == NEG_INF_BITS;
}

// Writes NaN over the row of a at base into out and returns true where the CPU gives a NaN
// row: a value in it is NaN or +inf, or every value is -inf.
__device__ bool bad_row(const float *a, float *out, u32 base, u32 width) {
    bool bad = false;
    bool finite = false;
    for (u32 j = 0u; j < width; j = j + 1u) {
        float z = a[base + j];
        if (is_neg_inf(z)) {
            continue;
        }
        bad = bad || (__float_as_uint(z) & 0x7f800000u) == 0x7f800000u;
        finite = true;
    }
    bad = bad || !finite;
    if (bad) {
        float nan = __uint_as_float(NAN_BITS);
        for (u32 j = 0u; j < width; j = j + 1u) {
            out[base + j] = nan;
        }
    }
    return bad;
}

// The row's maximum over its values other than -inf (a row bad_row passed has one).
__device__ float row_max(const float *a, u32 base, u32 width) {
    float m = 0.0f;
    bool seen = false;
    for (u32 j = 0u; j < width; j = j + 1u) {
        float z = a[base + j];
        if (is_neg_inf(z)) {
            continue;
        }
        m = seen ? fmaxf(m, z) : z;
        seen = true;
    }
    return m;
}

// e^(z - m), and 0 for a z of -inf, as the CPU's exp(-inf) gives.
__device__ __forceinline__ float exp_from(float z, float m) {
    return is_neg_inf(z) ? 0.0f : expf(z - m);
}

// ln(1 + e) for e >= 0: 2 atanh(e / (2 + e)) from its series to y^15 for e <= 1, logf(1 + e)
// above.
__device__ float ln_1p_series(float e) {
    if (e > 1.0f) {
        return logf(1.0f + e);
    }
    float y = e / (2.0f + e);
    float q = y * y;
    float t = 1.0f / 15.0f;
    t = 1.0f / 13.0f + q * t;
    t = 1.0f / 11.0f + q * t;
    t = 1.0f / 9.0f + q * t;
    t = 1.0f / 7.0f + q * t;
    t = 1.0f / 5.0f + q * t;
    t = 1.0f / 3.0f + q * t;
    t = 1.0f + q * t;
    return 2.0f * y * t;
}

// e^(z - max) / sum of e^(z - max), the terms kept in out and summed in order.
extern "C" __global__ void softmax(const float *a, const float *, float *out, u32 rows,
                                   u32 width) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    u32 base = i * width;
    if (bad_row(a, out, base, width)) {
        return;
    }
    float m = row_max(a, base, width);
    float s = 0.0f;
    for (u32 j = 0u; j < width; j = j + 1u) {
        float e = exp_from(a[base + j], m);
        out[base + j] = e;
        s = s + e;
    }
    for (u32 j = 0u; j < width; j = j + 1u) {
        out[base + j] = out[base + j] / s;
    }
}

// z - max - ln(1 + e), e the sum of e^(z - max) without the first maximum's term.
extern "C" __global__ void log_softmax(const float *a, const float *, float *out, u32 rows,
                                       u32 width) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    u32 base = i * width;
    if (bad_row(a, out, base, width)) {
        return;
    }
    float m = row_max(a, base, width);
    u32 k = 0u;
    for (u32 j = 0u; j < width; j = j + 1u) {
        if (a[base + j] == m) {
            k = j;
            break;
        }
    }
    float e = 0.0f;
    for (u32 j = 0u; j < width; j = j + 1u) {
        if (j != k) {
            e = e + exp_from(a[base + j], m);
        }
    }
    float lse = ln_1p_series(e);
    for (u32 j = 0u; j < width; j = j + 1u) {
        float z = a[base + j];
        // -inf - max - lse is -inf.
        out[base + j] = is_neg_inf(z) ? __uint_as_float(NEG_INF_BITS) : z - m - lse;
    }
}

// y * (g - sum of y * g), with y the forward output in a and g in b, as the CPU computes it.
extern "C" __global__ void softmax_backward(const float *a, const float *b, float *out, u32 rows,
                                            u32 width) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    u32 base = i * width;
    float dot = 0.0f;
    for (u32 j = 0u; j < width; j = j + 1u) {
        dot = dot + a[base + j] * b[base + j];
    }
    for (u32 j = 0u; j < width; j = j + 1u) {
        out[base + j] = a[base + j] * (b[base + j] - dot);
    }
}

// g - e^y * sum of g, with y the forward output in a and g in b; e^y is 0 for a y of -inf.
extern "C" __global__ void log_softmax_backward(const float *a, const float *b, float *out,
                                                u32 rows, u32 width) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    u32 base = i * width;
    float total = 0.0f;
    for (u32 j = 0u; j < width; j = j + 1u) {
        total = total + b[base + j];
    }
    for (u32 j = 0u; j < width; j = j + 1u) {
        float p = exp_from(a[base + j], 0.0f);
        out[base + j] = b[base + j] - p * total;
    }
}
