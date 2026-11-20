// The health values a training step reads back: the terms and trees of the kernels in
// cavitation-gpu's kernels/health.rs, operation for operation, written after reduce.cu's text
// and compiled with it, without fused multiply-add. Every count and sum runs in reduce.cu's
// fixed tree, so two runs give the same bits; the tensors of a flat buffer are summed in one
// launch per pass instead of two launches per tensor. NaN and infinity are tested by their
// bits.

// The first pass of a count of non-zero values per channel: 1 for each value whose bits
// other than the sign are not all 0 (anything but +-0, NaN and subnormals included).
extern "C" __global__ void non_zero_partials(const u32 *y, float *partials, u32 outer,
                                             u32 channels, u32 inner, u32 chunks) {
    channel_partials([=](u32 i, u32) { return (y[i] & 0x7fffffffu) != 0u ? 1.0f : 0.0f; },
                     partials, outer, channels, inner, chunks);
}

// The sum of term(j) over values j = chunk * BLOCK * PER_LANE onwards of a tensor of len
// values, in channel_partials's tree for a tensor of one channel ([1][1][len], where value j
// is term j): each thread adds every BLOCK-th term from +0, then the halving tree. Every
// thread of the block calls it and gets the sum; lanes is the block's own shared scratch.
template <typename Term>
__device__ float chunk_sum(Term term, u32 len, u32 chunk, float *lanes) {
    u32 lane = threadIdx.x;
    u32 first = chunk * BLOCK * PER_LANE + lane;
    float acc = 0.0f;
    for (u32 k = 0u; k < PER_LANE; k = k + 1u) {
        u32 j = first + k * BLOCK;
        if (j < len) {
            acc = acc + term(j);
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
    return lanes[0];
}

// The first pass of every tensor of a flat buffer at once: block b is chunk b - base of the
// tensor k whose words table[3k..3k + 3] = (start, len, base) hold the last base at most b.
// With bit 0 of want it writes the chunk's sum of squares to squares[b], with bit 1 its count
// of values whose exponent bits are all set to non_finite[b]: the sums squares_partials and
// non_finite_partials give for that chunk, tensor by tensor.
extern "C" __global__ void flat_partials(const float *g, const u32 *table, u32 tensors,
                                         float *squares, float *non_finite, u32 want) {
    __shared__ float lanes_squares[BLOCK];
    __shared__ float lanes_non_finite[BLOCK];
    u32 b = blockIdx.x;
    u32 k = 0u;
    while (k + 1u < tensors && table[3u * (k + 1u) + 2u] <= b) {
        k = k + 1u;
    }
    u32 start = table[3u * k];
    u32 len = table[3u * k + 1u];
    u32 chunk = b - table[3u * k + 2u];
    if ((want & 1u) != 0u) {
        float s = chunk_sum(
            [=](u32 j) {
                float v = g[start + j];
                return v * v;
            },
            len, chunk, lanes_squares);
        if (threadIdx.x == 0u) {
            squares[b] = s;
        }
    }
    if ((want & 2u) != 0u) {
        const u32 *bits = (const u32 *)g;
        float n = chunk_sum(
            [=](u32 j) {
                return (bits[start + j] & 0x7f800000u) == 0x7f800000u ? 1.0f : 0.0f;
            },
            len, chunk, lanes_non_finite);
        if (threadIdx.x == 0u) {
            non_finite[b] = n;
        }
    }
}

// The second pass of every tensor, in one block: thread k adds tensor k's chunks' partial sums
// in chunk order, as merge_at does, into out[at + k] (chunks base_k up to base_k+1, the last
// up to total). With fold set, thread 0 then writes the finite flag as finite_fold does, from
// these sums as the counts and the loss at loss[loss_at], to flag[0].
extern "C" __global__ void flat_merge(const float *partials, const u32 *table, u32 tensors,
                                      u32 total, float *out, u32 at, u32 fold, const u32 *loss,
                                      u32 loss_at, u32 *flag) {
    for (u32 k = threadIdx.x; k < tensors; k = k + BLOCK) {
        u32 base = table[3u * k + 2u];
        u32 end = k + 1u < tensors ? table[3u * (k + 1u) + 2u] : total;
        float acc = partials[base];
        for (u32 c = base + 1u; c < end; c = c + 1u) {
            acc = acc + partials[c];
        }
        out[at + k] = acc;
    }
    if (fold == 0u) {
        return;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        bool bad = (loss[loss_at] & 0x7f800000u) == 0x7f800000u;
        for (u32 k = 0u; k < tensors; k = k + 1u) {
            bad = bad || out[at + k] > 0.0f;
        }
        flag[0] = bad ? 1u : 0u;
    }
}

// The dead-ReLU merge and packing in one pass, one thread per word: for each of its channels
// c, adds the channel's chunks in chunk order as merge does into counts[c], and sets bit b of
// word w when channel 32 w + b's count is above 0, as dead_relu_pack does.
extern "C" __global__ void dead_relu_merge_pack(const float *partials, float *counts, u32 *bits,
                                                u32 channels, u32 chunks, u32 offset,
                                                u32 accumulate) {
    u32 w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= (channels + 31u) / 32u) {
        return;
    }
    u32 word = 0u;
    for (u32 b = 0u; b < 32u; b = b + 1u) {
        u32 c = w * 32u + b;
        if (c < channels) {
            u32 base = c * chunks;
            float acc = partials[base];
            for (u32 k = 1u; k < chunks; k = k + 1u) {
                acc = acc + partials[base + k];
            }
            counts[c] = acc;
            if (acc > 0.0f) {
                word = word | (1u << b);
            }
        }
    }
    if (accumulate != 0u) {
        word = word | bits[offset + w];
    }
    bits[offset + w] = word;
}
