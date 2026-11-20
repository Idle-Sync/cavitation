// PReLU over a [rows][channels] batch whose value i takes slope i % channels: a port of
// cavitation-gpu's kernels/prelu.rs, operation for operation, written after reduce.cu's text
// and compiled with it, without fused multiply-add. The forward pass and the input's gradient
// run one thread per value and take (const float *x, const float *slope, const float *g,
// float *out, u32 n, u32 channels), g being x again, unused, forward. The slopes' gradient
// is a channel sum of g * x over the values not above 0, in reduce.cu's fixed tree.

// Whether v is NaN, from its bits.
__device__ __forceinline__ bool prelu_is_nan(float v) {
    return (__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u;
}

// x above 0, slope * x at or below; NaN stays NaN.
extern "C" __global__ void prelu_forward(const float *x, const float *slope, const float *g,
                                         float *out, u32 n, u32 channels) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float v = x[i];
    float s = slope[i % channels];
    out[i] = prelu_is_nan(v) ? v : (v > 0.0f ? v : s * v);
}

// g * (1 above 0, the slope at or below, NaN for a NaN x).
extern "C" __global__ void prelu_input_grad(const float *x, const float *slope, const float *g,
                                            float *out, u32 n, u32 channels) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float v = x[i];
    float s = slope[i % channels];
    out[i] = g[i] * (prelu_is_nan(v) ? v : (v > 0.0f ? 1.0f : s));
}

// The first pass of the slopes' gradient: the term of value i is g * x where x is not above 0
// (a NaN x included, which carries into the sum) and 0 elsewhere.
extern "C" __global__ void prelu_slope_partials(const float *x, const float *g,
                                                float *partials, u32 outer, u32 channels,
                                                u32 inner, u32 chunks) {
    channel_partials(
        [=](u32 i, u32) {
            float v = x[i];
            return v > 0.0f ? 0.0f : g[i] * v;
        },
        partials, outer, channels, inner, chunks);
}
