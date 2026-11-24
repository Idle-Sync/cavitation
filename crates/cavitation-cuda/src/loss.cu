// The training losses: a port of the kernels in cavitation-gpu's kernels/loss.rs, operation
// for operation, written after reduce.cu's text and compiled with it, without fused
// multiply-add. Each loss's mean is a channel sum of one channel in reduce.cu's fixed tree,
// divided by the count (or, for a batch mean, the batch) rounded to f32.

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

// The bits of 1.0 and of the magnitude.
#define ONE_BITS 0x3f800000u
#define MAGNITUDE 0x7fffffffu
// The magnitude bits of 2^64: from there on p (1 - p) rounds to -infinity in f32, since 1 - p
// rounds to -p and p^2 reaches 2^128; just below, p^2 rounds to a finite float.
#define PRODUCT_OVERFLOWS 0x5f800000u

// 2 atanh(y), from its series to y^15.
__device__ float atanh2(float y) {
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

// ln x for the positive finite x whose bits are b, subnormals included: k ln 2 + 2 atanh(y)
// with y = (f - 1) / (f + 1) for x = f 2^k, f within [sqrt(1/2), sqrt(2)], ln 2 split in two
// so that k ln2_hi is exact.
__device__ float ln_bits(u32 b) {
    int e;
    u32 m;
    u32 field = b >> 23u;
    if (field == 0u) {
        u32 s = (u32)__clz(b) - 8u;
        m = (b << s) & 0x7fffffu;
        e = -126 - (int)s;
    } else {
        m = b & 0x7fffffu;
        e = (int)field - 127;
    }
    float f = __uint_as_float(ONE_BITS | m);
    if (f > 1.41421356f) {
        f = 0.5f * f;
        e = e + 1;
    }
    float k = (float)e;
    return k * 0.693145751953125f + (k * 1.4286068203094172e-6f + atanh2((f - 1.0f) / (f + 1.0f)));
}

// The value whose bits are b times 2^32, exactly (infinite past the largest float).
__device__ float scaled(u32 b) {
    u32 sign = b & SIGN_BIT;
    u32 field = (b >> 23u) & 0xffu;
    if (field == 0u) {
        float v = (float)(b & 0x7fffffu) * __uint_as_float(10u << 23u);
        return __uint_as_float(__float_as_uint(v) | sign);
    }
    if (field >= 223u) {
        return __uint_as_float(INF_BITS | sign);
    }
    return __uint_as_float(b + (32u << 23u));
}

// Binary cross-entropy of probabilities for one value: its loss, its clamp count (1.0 where
// the CPU's tape counts a clamped logarithm) and its gradient times w. Each logarithm is
// clamped to -100 where the CPU's is: p of +-0 or of at most 26 * 2^-149, and 1 - p of 0.
// Where |p| < 2^-62 the gradient's quotient is taken on p and t scaled by 2^32.
__device__ void bce_probabilities(u32 pb, u32 tb, float w, u32 *term, u32 *g, float *count) {
    u32 pm = pb & MAGNITUDE;
    u32 tm = tb & MAGNITUDE;
    bool p_clamped = pm == 0u || pb <= 26u;
    bool q_clamped = pb == ONE_BITS;
    *count = (p_clamped && tm != 0u) || (q_clamped && tb != ONE_BITS) ? 1.0f : 0.0f;
    float p = __uint_as_float(pb);
    float t = __uint_as_float(tb);
    // ln p or ln(1 - p) is NaN outside [-0, 1], and an infinite target meets a zero logarithm
    // or two of one sign with weights of opposite signs.
    if ((pb <= ONE_BITS || pb == SIGN_BIT) && tm != INF_BITS) {
        float lp = p_clamped ? -100.0f : ln_bits(pb);
        float lq = q_clamped ? -100.0f : ln_bits(__float_as_uint(1.0f - p));
        *term = __float_as_uint(-(t * lp + (1.0f - t) * lq));
    } else {
        *term = NAN_BITS;
    }
    if (pm == INF_BITS) {
        *g = NAN_BITS;
    } else if (tm == INF_BITS && pm >= PRODUCT_OVERFLOWS) {
        // p (1 - p) overflows to -infinity in f32 for |p| from 2^64, and infinity over
        // infinity is NaN.
        *g = NAN_BITS;
    } else if (tm == INF_BITS) {
        // -t times infinity over 1, or over p (1 - p), negative for p below 0 or above 1.
        bool negative = !p_clamped && !q_clamped && ((pb & SIGN_BIT) != 0u || pb > ONE_BITS);
        *g = INF_BITS | (((tb ^ SIGN_BIT) & SIGN_BIT) ^ (negative ? SIGN_BIT : 0u));
    } else if (p_clamped) {
        *g = __float_as_uint((1.0f - t) * w);
    } else if (q_clamped) {
        *g = __float_as_uint(-t * w);
    } else if (pm < (65u << 23u)) {
        float ps = scaled(pb);
        *g = __float_as_uint((ps - scaled(tb)) / ps * w);
    } else {
        // The WGSL kernel scales both sides by 2^-64 from |p (1 - p)| = 2^64, for a division
        // through the reciprocal; NVRTC divides correctly rounded, so the CPU's quotient needs
        // no scaling here.
        *g = __float_as_uint((p - t) / (p * (1.0f - p)) * w);
    }
}

// Binary cross-entropy of logits for one value: max(z, 0) - z t + ln(1 + e) with
// e = e^(-|z|) and ln(1 + e) from the series of 2 atanh(e / (2 + e)), and the gradient
// (sigma(z) - t) w with the CPU's stable sigmoid on the same e. An infinite logit or target
// takes the CPU's infinite or NaN loss and gradient from the bits.
__device__ void bce_logits(u32 zb, u32 tb, float w, u32 *term, u32 *g) {
    u32 zm = zb & MAGNITUDE;
    u32 tm = tb & MAGNITUDE;
    float t = __uint_as_float(tb);
    if (zm == INF_BITS || tm == INF_BITS) {
        // max(z, 0) - z t with an infinite z or t: infinity times 0 is NaN, as is infinity
        // minus infinity.
        if (zb == INF_BITS) {
            *term = (tb & SIGN_BIT) != 0u && tm != 0u ? INF_BITS : NAN_BITS;
        } else if (zm == INF_BITS) {
            *term = tm == 0u ? NAN_BITS : (INF_BITS | (tb & SIGN_BIT));
        } else {
            *term = zm == 0u ? NAN_BITS : (INF_BITS | ((zb ^ tb ^ SIGN_BIT) & SIGN_BIT));
        }
        if (tm == INF_BITS) {
            *g = INF_BITS | ((tb ^ SIGN_BIT) & SIGN_BIT);
        } else {
            *g = __float_as_uint(((zb == INF_BITS ? 1.0f : 0.0f) - t) * w);
        }
        return;
    }
    float z = __uint_as_float(zb);
    float e = expf(-fabsf(z));
    float positive = z > 0.0f ? z : 0.0f;
    *term = __float_as_uint(positive - z * t + atanh2(e / (2.0f + e)));
    float sigma = z >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    *g = __float_as_uint((sigma - t) * w);
}

// Binary cross-entropy, one thread per value: each value's loss into terms, the prediction's
// gradient into grad (both as bits) and its clamp count into clamps. logits is 0 for
// probabilities and 1 for logits. NaN in either input gives NaN for the loss and the gradient.
extern "C" __global__ void bce_terms(const u32 *pred, const u32 *target, u32 *grad, u32 *terms,
                                     float *clamps, u32 n, float w, u32 logits) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    u32 pb = pred[i];
    u32 tb = target[i];
    if ((pb & MAGNITUDE) > INF_BITS || (tb & MAGNITUDE) > INF_BITS) {
        terms[i] = NAN_BITS;
        grad[i] = NAN_BITS;
        clamps[i] = 0.0f;
        return;
    }
    if (logits == 0u) {
        bce_probabilities(pb, tb, w, &terms[i], &grad[i], &clamps[i]);
    } else {
        clamps[i] = 0.0f;
        bce_logits(pb, tb, w, &terms[i], &grad[i]);
    }
}

// 2^-32, which undoes scaled().
#define UNSCALE 2.3283064365386963e-10f

// The KL divergence t (ln t - x) of one value for the bits of x and t, both not NaN: 0 for a
// target of +-0, NaN for a negative target, +inf for a target of +inf (NaN against x = +inf),
// the other sign's infinity for an infinite x. ln t comes from the bits, and a subnormal
// target's product is taken on t 2^32 and scaled back, as the WGSL kernel does.
__device__ u32 kl_term(u32 xb, u32 tb) {
    if ((tb & MAGNITUDE) == 0u) {
        return 0u;
    }
    if ((tb & SIGN_BIT) != 0u) {
        return NAN_BITS;
    }
    if (tb == INF_BITS) {
        return xb == INF_BITS ? NAN_BITS : INF_BITS;
    }
    if ((xb & MAGNITUDE) == INF_BITS) {
        return INF_BITS | ((xb ^ SIGN_BIT) & SIGN_BIT);
    }
    float d = ln_bits(tb) - __uint_as_float(xb);
    if ((tb >> 23u) == 0u) {
        return __float_as_uint(scaled(tb) * d * UNSCALE);
    }
    return __float_as_uint(__uint_as_float(tb) * d);
}

// KL divergence, one thread per value: each value's loss t (ln t - x) into terms and the
// log-probabilities' gradient -t w into grad, both as bits, +0 where t is +-0, the gradient's
// NaN and infinities taken from the target's bits. NaN in either input gives a NaN loss, also
// where t is 0.
extern "C" __global__ void kl_terms(const u32 *log_probs, const u32 *target, u32 *grad,
                                    u32 *terms, u32 n, float w) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    u32 xb = log_probs[i];
    u32 tb = target[i];
    u32 tm = tb & MAGNITUDE;
    if (tm == 0u) {
        grad[i] = 0u;
    } else if (tm > INF_BITS) {
        grad[i] = NAN_BITS;
    } else if (tm == INF_BITS) {
        grad[i] = INF_BITS | ((tb ^ SIGN_BIT) & SIGN_BIT);
    } else {
        grad[i] = __float_as_uint(__uint_as_float(tb) * w) ^ SIGN_BIT;
    }
    terms[i] = (xb & MAGNITUDE) > INF_BITS || tm > INF_BITS ? NAN_BITS : kl_term(xb, tb);
}
