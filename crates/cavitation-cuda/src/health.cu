// The health values a training step reads back: a port of the kernels in cavitation-gpu's
// kernels/health.rs, operation for operation, written after reduce.cu's text and compiled
// with it, without fused multiply-add. Every count and sum runs in reduce.cu's fixed tree, so
// two runs give the same bits. NaN and infinity are tested by their bits.

// The first pass of a sum of squares, g^2 for each value of the tensor that starts `start`
// values into g.
extern "C" __global__ void squares_partials(const float *g, u32 start, float *partials,
                                            u32 outer, u32 channels, u32 inner, u32 chunks) {
    channel_partials(
        [=](u32 i, u32) {
            float v = g[start + i];
            return v * v;
        },
        partials, outer, channels, inner, chunks);
}

// The first pass of a count of non-finite values: 1 for each value of the tensor that starts
// `start` values into g whose exponent bits are all set (NaN or +-inf), 0 otherwise.
extern "C" __global__ void non_finite_partials(const u32 *g, u32 start, float *partials,
                                               u32 outer, u32 channels, u32 inner,
                                               u32 chunks) {
    channel_partials(
        [=](u32 i, u32) {
            return (g[start + i] & 0x7f800000u) == 0x7f800000u ? 1.0f : 0.0f;
        },
        partials, outer, channels, inner, chunks);
}

// The first pass of a count of non-zero values per channel: 1 for each value whose bits
// other than the sign are not all 0 (anything but +-0, NaN and subnormals included).
extern "C" __global__ void non_zero_partials(const u32 *y, float *partials, u32 outer,
                                             u32 channels, u32 inner, u32 chunks) {
    channel_partials([=](u32 i, u32) { return (y[i] & 0x7fffffffu) != 0u ? 1.0f : 0.0f; },
                     partials, outer, channels, inner, chunks);
}

// The second pass of one channel's sum, in one thread: adds its chunks' partial sums in chunk
// order into out[at].
extern "C" __global__ void merge_at(const float *partials, float *out, u32 at, u32 chunks) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    float acc = partials[0];
    for (u32 k = 1u; k < chunks; k = k + 1u) {
        acc = acc + partials[k];
    }
    out[at] = acc;
}

// The finite flag, in one thread: 0 when the loss at loss[loss_at] has an exponent that is not
// all ones and every tensor's count of non-finite values is 0, 1 otherwise, written to
// flag[0] as a u32.
extern "C" __global__ void finite_fold(const float *counts, const u32 *loss, u32 loss_at,
                                       u32 tensors, u32 *flag) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    bool bad = (loss[loss_at] & 0x7f800000u) == 0x7f800000u;
    for (u32 k = 0u; k < tensors; k = k + 1u) {
        bad = bad || counts[k] > 0.0f;
    }
    flag[0] = bad ? 1u : 0u;
}

// The dead-ReLU bits, one thread per word: bit b of word w is set when channel 32 w + b had a
// non-zero value, OR-ed into the old word when accumulating; word w goes to bits[offset + w].
extern "C" __global__ void dead_relu_pack(const float *counts, u32 *bits, u32 channels,
                                          u32 offset, u32 accumulate) {
    u32 w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= (channels + 31u) / 32u) {
        return;
    }
    u32 word = 0u;
    for (u32 b = 0u; b < 32u; b = b + 1u) {
        u32 c = w * 32u + b;
        if (c < channels && counts[c] > 0.0f) {
            word = word | (1u << b);
        }
    }
    if (accumulate != 0u) {
        word = word | bits[offset + w];
    }
    bits[offset + w] = word;
}
