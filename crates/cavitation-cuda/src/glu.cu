// Gated linear units over a [rows][width] batch of even width: a port of cavitation-gpu's
// kernels/glu.rs, operation for operation, written after activation.cu's text and compiled
// with it, without fused multiply-add. One thread takes the pair (a, b) of row r, column j:
// a = x[r * width + j] and b = x[r * width + half + j], half = width / 2. The forward pass
// writes a * gate(b); the backward pass writes g * gate(b) into a's place and
// (g * a) * gate'(b) into b's. Every kernel takes (const float *x, const float *g, float *out,
// u32 pairs, u32 half), g being x again, unused, forward. The gates are 0 the sigmoid, 1
// GELU's tanh form, 2 its erf form, 3 ReLU and 4 SiLU, with the element-wise kernels' values
// and derivatives.

// The gate's value at v, as the element-wise kernels compute it.
template <int GATE> __device__ float glu_gate(float v) {
    switch (GATE) {
    case 0:
        return sigmoid_one(v);
    case 1: {
        float t = tanh_one(SQRT_2_OVER_PI * (v + GELU_CUBIC * v * v * v));
        return 0.5f * v * (1.0f + t);
    }
    case 2:
        return 0.5f * v * (1.0f + erf_one(v * FRAC_1_SQRT_2));
    case 3:
        return is_nan(v) ? v : (v <= 0.0f ? 0.0f : v);
    default:
        return v * sigmoid_one(v);
    }
}

// The gate's derivative at v, as the element-wise kernels' backward passes take it.
template <int GATE> __device__ float glu_derivative(float v) {
    switch (GATE) {
    case 0: {
        float s = sigmoid_one(v);
        return s * (1.0f - s);
    }
    case 1: {
        float v2 = v * v;
        float t = tanh_one(SQRT_2_OVER_PI * (v + GELU_CUBIC * v2 * v));
        // Where tanh saturates the second term is dropped, as the CPU does.
        float first = 0.5f * (1.0f + t);
        float flat = 1.0f - t * t;
        if (flat == 0.0f) {
            return first;
        }
        float du = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_CUBIC * v2);
        return first + 0.5f * v * flat * du;
    }
    case 2: {
        float cdf = 0.5f * (1.0f + erf_one(v * FRAC_1_SQRT_2));
        float pdf = FRAC_1_SQRT_2PI * expf(-(0.5f * v * v));
        // Where the density underflows the x * phi term is dropped, as the CPU does.
        if (pdf == 0.0f) {
            return cdf;
        }
        return cdf + v * pdf;
    }
    case 3:
        return is_nan(v) ? v : (v > 0.0f ? 1.0f : 0.0f);
    default: {
        float s = sigmoid_one(v);
        return s * (1.0f + v * (1.0f - s));
    }
    }
}

// a * gate(b).
template <int GATE>
__device__ void glu_forward(const float *x, float *out, u32 pairs, u32 half) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) {
        return;
    }
    u32 k = i + (i / half) * half;
    out[i] = x[k] * glu_gate<GATE>(x[k + half]);
}

// g * gate(b) into a's place and (g * a) * gate'(b) into b's.
template <int GATE>
__device__ void glu_backward(const float *x, const float *g, float *out, u32 pairs, u32 half) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) {
        return;
    }
    u32 k = i + (i / half) * half;
    float a = x[k];
    float b = x[k + half];
    float gi = g[i];
    out[k] = gi * glu_gate<GATE>(b);
    out[k + half] = (gi * a) * glu_derivative<GATE>(b);
}

// The forward and backward kernels of one gate.
#define GLU_KERNELS(name, gate)                                                               \
    extern "C" __global__ void glu_forward_##name(const float *x, const float *g, float *out,  \
                                                  u32 pairs, u32 half) {                     \
        glu_forward<gate>(x, out, pairs, half);                                               \
    }                                                                                         \
    extern "C" __global__ void glu_backward_##name(const float *x, const float *g, float *out, \
                                                   u32 pairs, u32 half) {                    \
        glu_backward<gate>(x, g, out, pairs, half);                                           \
    }

GLU_KERNELS(sigmoid, 0)
GLU_KERNELS(gelu_tanh, 1)
GLU_KERNELS(gelu_erf, 2)
GLU_KERNELS(relu, 3)
GLU_KERNELS(silu, 4)
