// Per-channel sums over tensors laid out [outer][channels][inner], in a fixed tree: a port of
// the two passes of cavitation-gpu's kernels/reduce.rs (partials_pass and the channel sums'
// merge), operation for operation. The kernels that sum their own terms (batch norm's) are
// written after this text, in the same source.
//
// The first pass's block (chunk, c) sums the terms of values chunk * CHUNK onwards of channel
// c, taken in the order j = b * inner + p: each of its BLOCK threads adds every BLOCK-th term
// of the chunk in order, starting from +0, then the threads' sums meet in a fixed halving
// tree. The second pass adds each channel's chunks in order. The tree depends on the shape
// alone, so two runs give the same bits.

typedef unsigned int u32;

// Threads per block of every kernel here; the halving tree needs a power of two.
#define BLOCK 256u

// Values of one channel each thread of the first pass adds, BLOCK apart: a chunk is
// BLOCK * PER_LANE values.
#define PER_LANE 16u

// The first pass for block (chunk, c) = (blockIdx.x, blockIdx.y), summing term(i, c) for
// value i of channel c, into partials[c * chunks + chunk].
template <typename Term>
__device__ void channel_partials(Term term, float *partials, u32 outer, u32 channels,
                                 u32 inner, u32 chunks) {
    __shared__ float lanes[BLOCK];
    u32 chunk = blockIdx.x;
    u32 c = blockIdx.y;
    u32 lane = threadIdx.x;
    u32 len = outer * inner;
    u32 first = chunk * BLOCK * PER_LANE + lane;
    float acc = 0.0f;
    for (u32 k = 0u; k < PER_LANE; k = k + 1u) {
        u32 j = first + k * BLOCK;
        if (j < len) {
            u32 b = j / inner;
            u32 i = (b * channels + c) * inner + j % inner;
            acc = acc + term(i, c);
        }
    }
    lanes[lane] = acc;
    __syncthreads();
    for (u32 width = BLOCK / 2u; width > 0u; width = width / 2u) {
        if (lane < width) {
            lanes[lane] = lanes[lane] + lanes[lane + width];
        }
        __syncthreads();
    }
    if (lane == 0u) {
        partials[c * chunks + chunk] = lanes[0];
    }
}

// The first pass of the channel sums: the terms are the values themselves.
extern "C" __global__ void sum_partials(const float *x, float *partials, u32 outer,
                                        u32 channels, u32 inner, u32 chunks) {
    channel_partials([=](u32 i, u32) { return x[i]; }, partials, outer, channels, inner,
                     chunks);
}

// The second pass, one thread per channel: adds the channel's chunks' partial sums in chunk
// order into sums[c].
extern "C" __global__ void merge(const float *partials, float *sums, u32 channels,
                                 u32 chunks) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    u32 base = c * chunks;
    float acc = partials[base];
    for (u32 k = 1u; k < chunks; k = k + 1u) {
        acc = acc + partials[base + k];
    }
    sums[c] = acc;
}
