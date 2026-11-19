// The training losses: a port of the kernels in cavitation-gpu's kernels/loss.rs, operation
// for operation, written after reduce.cu's text and compiled with it, without fused
// multiply-add. Each loss's mean is a channel sum of one channel in reduce.cu's fixed tree,
// divided by the count rounded to f32.

// The bits of the NaN a bad cross-entropy row gets.
#define NAN_BITS 0x7fc00000u

// Softmax cross-entropy, one thread per row: the row's loss ln S - (z_label - max) and its
// gradient w * (softmax - onehot). S = 1 + e, where e sums e^(z - max) over every class but
// the first maximum's; ln S is 2 atanh(e / (2 + e)) from its series to y^15 for e <= 1, which
// keeps e's low bits that 1 + e rounded to f32 would drop, and logf(1 + e) above. A row with
// a logit whose exponent bits are all set (NaN or +-inf), or with a label not below the
// classes, gets NaN for its loss and its whole gradient row.
extern "C" __global__ void ce_rows(const float *logits, const u32 *labels, float *grad,
                                   float *row_loss, u32 batch, u32 classes, float w) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) {
        return;
    }
    u32 base = i * classes;
    u32 label = labels[i];
    bool bad = label >= classes;
    float m = logits[base];
    for (u32 j = 0u; j < classes; j = j + 1u) {
        float z = logits[base + j];
        bad = bad || (__float_as_uint(z) & 0x7f800000u) == 0x7f800000u;
        m = fmaxf(m, z);
    }
    if (bad) {
        float nan = __uint_as_float(NAN_BITS);
        row_loss[i] = nan;
        for (u32 j = 0u; j < classes; j = j + 1u) {
            grad[base + j] = nan;
        }
        return;
    }
    u32 k = 0u;
    for (u32 j = 0u; j < classes; j = j + 1u) {
        if (logits[base + j] == m) {
            k = j;
            break;
        }
    }
    float e = 0.0f;
    for (u32 j = 0u; j < classes; j = j + 1u) {
        if (j != k) {
            e = e + expf(logits[base + j] - m);
        }
    }
    float lse;
    if (e <= 1.0f) {
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
        lse = 2.0f * y * t;
    } else {
        lse = logf(1.0f + e);
    }
    row_loss[i] = -(logits[base + label] - m - lse);
    for (u32 j = 0u; j < classes; j = j + 1u) {
        float p = expf(logits[base + j] - m - lse);
        float onehot = j == label ? 1.0f : 0.0f;
        grad[base + j] = w * (p - onehot);
    }
}

// The first pass of the squared errors' sum, (p - t)^2 summed as one channel.
extern "C" __global__ void squared_partials(const float *pred, const float *target,
                                            float *partials, u32 outer, u32 channels,
                                            u32 inner, u32 chunks) {
    channel_partials(
        [=](u32 i, u32) {
            float d = pred[i] - target[i];
            return d * d;
        },
        partials, outer, channels, inner, chunks);
}

// The mean squared error's gradient, one thread per value: 2 * (p - t) * w, as the CPU
// computes it.
extern "C" __global__ void mse_grad(const float *pred, const float *target, float *grad, u32 n,
                                    float w) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    grad[i] = 2.0f * (pred[i] - target[i]) * w;
}

// The second pass of a loss's mean, one thread: adds the chunks' partial sums in chunk order,
// as reduce.cu's merge does, divides by the count and writes the loss to out[offset].
extern "C" __global__ void mean_merge(const float *partials, float *out, u32 offset, u32 chunks,
                                      float divisor) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0u) {
        return;
    }
    float acc = partials[0];
    for (u32 k = 1u; k < chunks; k = k + 1u) {
        acc = acc + partials[k];
    }
    out[offset] = acc / divisor;
}
