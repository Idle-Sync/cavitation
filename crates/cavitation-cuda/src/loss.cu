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

// The bits of +infinity and of the sign.
#define INF_BITS 0x7f800000u
#define SIGN_BIT 0x80000000u

// The negative log-likelihood, one thread per row: the row's loss -lp[label] (its sign bit
// flipped) and its gradient, -w at the label and +0 elsewhere, as the CPU's bits. A row with a
// NaN gets NaN for its loss and keeps its finite gradient; a row whose label is not below the
// classes gets NaN for its loss and its whole gradient row.
extern "C" __global__ void nll_rows(const u32 *log_probs, const u32 *labels, u32 *grad,
                                    u32 *row_loss, u32 batch, u32 classes, float w) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) {
        return;
    }
    u32 base = i * classes;
    u32 label = labels[i];
    if (label >= classes) {
        row_loss[i] = NAN_BITS;
        for (u32 j = 0u; j < classes; j = j + 1u) {
            grad[base + j] = NAN_BITS;
        }
        return;
    }
    bool nan = false;
    for (u32 j = 0u; j < classes; j = j + 1u) {
        nan = nan || (log_probs[base + j] & 0x7fffffffu) > INF_BITS;
    }
    row_loss[i] = nan ? NAN_BITS : log_probs[base + label] ^ SIGN_BIT;
    u32 minus_w = __float_as_uint(w) ^ SIGN_BIT;
    for (u32 j = 0u; j < classes; j = j + 1u) {
        grad[base + j] = j == label ? minus_w : 0u;
    }
}

// The bits of pred[i] - target[i], its NaN and infinite cases decided from the inputs' bits.
__device__ u32 difference_bits(const float *pred, const float *target, u32 i) {
    u32 pb = __float_as_uint(pred[i]);
    u32 tb = __float_as_uint(target[i]);
    u32 pm = pb & 0x7fffffffu;
    u32 tm = tb & 0x7fffffffu;
    if (pm > INF_BITS || tm > INF_BITS) {
        return NAN_BITS;
    }
    if (pm == INF_BITS && tm == INF_BITS) {
        return pb != tb ? pb : NAN_BITS;
    }
    if (pm == INF_BITS) {
        return pb;
    }
    if (tm == INF_BITS) {
        return tb ^ SIGN_BIT;
    }
    return __float_as_uint(pred[i] - target[i]);
}

// The L1 (kind 0), Huber (kind 1) and smooth L1 (kind 2) losses with threshold param, one
// thread per value: each value's loss into terms and the prediction's gradient into grad. NaN
// gives NaN for both; an infinite difference takes the linear branch, a loss of +inf (NaN with
// an infinite threshold, for Huber and smooth L1) and the gradient sign(d) * w (Huber's
// sign(d) * (param * w)). A finite difference runs the CPU's expressions in its order.
extern "C" __global__ void regression_terms(const float *pred, const float *target, u32 *grad,
                                            u32 *terms, u32 n, float w, u32 kind,
                                            float param) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    u32 db = difference_bits(pred, target, i);
    u32 magnitude = db & 0x7fffffffu;
    if (magnitude > INF_BITS) {
        terms[i] = NAN_BITS;
        grad[i] = NAN_BITS;
        return;
    }
    u32 sign = db & SIGN_BIT;
    u32 wb = __float_as_uint(w);
    if (magnitude == INF_BITS) {
        bool infinite_param = (__float_as_uint(param) & 0x7fffffffu) == INF_BITS;
        if (kind == 0u) {
            terms[i] = INF_BITS;
            grad[i] = wb ^ sign;
        } else if (kind == 1u) {
            terms[i] = infinite_param ? NAN_BITS : INF_BITS;
            grad[i] = (infinite_param ? INF_BITS : __float_as_uint(param * w)) ^ sign;
        } else {
            terms[i] = infinite_param ? NAN_BITS : INF_BITS;
            grad[i] = wb ^ sign;
        }
        return;
    }
    float d = __uint_as_float(db);
    float a = fabsf(d);
    // sign(d) * w, with sign(+-0) = 0 giving +0, as the CPU's sign.
    u32 unit = d > 0.0f ? wb : (d < 0.0f ? wb ^ SIGN_BIT : 0u);
    if (kind == 0u) {
        terms[i] = __float_as_uint(a);
        grad[i] = unit;
    } else if (kind == 1u) {
        if (a < param) {
            terms[i] = __float_as_uint(0.5f * d * d);
            grad[i] = __float_as_uint(d * w);
        } else {
            terms[i] = __float_as_uint(param * (a - 0.5f * param));
            grad[i] = __float_as_uint((d > 0.0f ? param : -param) * w);
        }
    } else {
        if (a < param) {
            terms[i] = __float_as_uint(0.5f * d * d / param);
            grad[i] = __float_as_uint(d / param * w);
        } else {
            terms[i] = __float_as_uint(a - 0.5f * param);
            grad[i] = unit;
        }
    }
}
