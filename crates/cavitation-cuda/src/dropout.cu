// Dropout and the neuron mask with a bit-packed mask: a port of cavitation-gpu's
// dropout.wgsl, operation for operation, compiled without fused multiply-add. Element i's bit
// is bit i % 32 of mask word i / 32, the low bit first.
//
// dropout, forward and backward alike: out = a * m with m = scale where the bit is set and 0
// where it is clear, as the CPU multiplies by its multipliers (so a dropped -3 is -0 and a
// dropped NaN stays NaN). keep_where selects a where the bit is set and +0 where it is clear,
// as Tape::keep_where does (so a cleared -3 or NaN is +0), forward and backward alike.

typedef unsigned int u32;

// Whether element i's bit is set in the mask.
__device__ __forceinline__ bool kept(const u32 *mask, u32 i) {
    return ((mask[i / 32u] >> (i % 32u)) & 1u) != 0u;
}

// a times scale where the mask's bit is set and times 0 where it is clear.
extern "C" __global__ void dropout(const float *a, const u32 *mask, float *out, u32 n,
                                   float scale) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    out[i] = a[i] * (kept(mask, i) ? scale : 0.0f);
}

// a where the mask's bit is set, +0 where it is clear.
extern "C" __global__ void keep_where(const float *a, const u32 *mask, float *out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    out[i] = kept(mask, i) ? a[i] : 0.0f;
}
