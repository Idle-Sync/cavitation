// SGD (with momentum and weight decay), Adam and AdamW: a port of the kernels in
// cavitation-gpu's kernels/optimizer.rs, operation for operation, compiled without fused
// multiply-add, so the only fused operations are the explicit fmaf calls, each rounded once.
//
// One block updates one parameter's run of up to BLOCK elements of the flat buffers. The
// table holds, for each block, its parameter and the first element it updates. The constants
// buffer holds SHARED floats, then four words per parameter: its end in the flat buffers, its
// flags (bit 0 set when it updates this step, bit 1 on its first update), and its Adam bias
// corrections for this step, step_size = lr / (1 - beta1^t) and 1 / sqrt(1 - beta2^t). Every
// kernel does nothing when the step's finite flag is set, and its first thread copies the
// flag to `seen`, so the host counts the step from the flag the launch read.

typedef unsigned int u32;

// Floats of the shared constants at the start of the constants buffer: lr, momentum, weight
// decay, beta1, beta2, 1 - beta1, 1 - beta2, eps and AdamW's shrink 1 - lr * wd.
#define SHARED 9u

// The shared constants' positions.
#define LR 0u
#define MOMENTUM 1u
#define WEIGHT_DECAY 2u
#define BETA1 3u
#define BETA2 4u
#define ONE_MINUS_BETA1 5u
#define ONE_MINUS_BETA2 6u
#define EPS 7u
#define SHRINK 8u

// A moment stored for the next step: 0 below the smallest normal f32, as the CPU optimisers
// store their state.
__device__ float store_state(float x) {
    return (__float_as_uint(x) & 0x7f800000u) == 0u ? 0.0f : x;
}

// The element this thread updates, or -1 when there is none: the step's finite flag is set,
// the block lies past the table, the parameter does not update this step, or the element is
// past the parameter's end. `info` is set to the parameter's four words.
__device__ long long element(const u32 *table, const float *consts, const u32 *finite,
                             u32 groups, u32 *seen, const float **info) {
    u32 flag = finite[0];
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        seen[0] = flag;
    }
    if (flag != 0u) {
        return -1;
    }
    u32 w = blockIdx.x;
    if (w >= groups) {
        return -1;
    }
    u32 param = table[2u * w];
    u32 i = table[2u * w + 1u] + threadIdx.x;
    *info = consts + SHARED + 4u * param;
    u32 end = __float_as_uint((*info)[0]);
    u32 flags = __float_as_uint((*info)[1]);
    if ((flags & 1u) == 0u || i >= end) {
        return -1;
    }
    return (long long)i;
}

// Whether this is the parameter's first update.
__device__ bool first_update(const float *info) {
    return (__float_as_uint(info[1]) & 2u) != 0u;
}

// SGD without momentum: p = p - lr * (g + wd * p), the decay term left out when wd is 0, as
// the CPU leaves it out.
extern "C" __global__ void sgd(float *values, const float *grads, const u32 *table,
                               const float *consts, const u32 *finite, u32 groups, u32 *seen) {
    const float *info;
    long long at = element(table, consts, finite, groups, seen, &info);
    if (at < 0) {
        return;
    }
    u32 i = (u32)at;
    float p = values[i];
    float step = grads[i];
    float wd = consts[WEIGHT_DECAY];
    if (wd != 0.0f) {
        step = fmaf(wd, p, step);
    }
    values[i] = fmaf(-consts[LR], step, p);
}

// SGD with momentum: b = mu * b + (g + wd * p), b set to the step itself on a parameter's
// first update, then p = p - lr * b.
extern "C" __global__ void sgd_momentum(float *values, const float *grads, const u32 *table,
                                        const float *consts, const u32 *finite, u32 groups,
                                        u32 *seen, float *buffer) {
    const float *info;
    long long at = element(table, consts, finite, groups, seen, &info);
    if (at < 0) {
        return;
    }
    u32 i = (u32)at;
    float p = values[i];
    float step = grads[i];
    float wd = consts[WEIGHT_DECAY];
    if (wd != 0.0f) {
        step = fmaf(wd, p, step);
    }
    if (!first_update(info)) {
        step = fmaf(consts[MOMENTUM], buffer[i], step);
    }
    buffer[i] = store_state(step);
    values[i] = fmaf(-consts[LR], step, p);
}

// Adam, or AdamW when `decoupled`: AdamW first shrinks p by 1 - lr * wd, Adam adds wd * p to
// the gradient; then the moments m = beta1 * m + (1 - beta1) * g and
// v = beta2 * v + (1 - beta2) * g^2 are stored and
// p = p - step_size * m / (sqrt(v) * c2 + eps), with the parameter's own bias corrections.
extern "C" __global__ void adam(float *values, const float *grads, const u32 *table,
                                const float *consts, const u32 *finite, u32 groups, u32 *seen,
                                float *m, float *v, u32 decoupled) {
    const float *info;
    long long at = element(table, consts, finite, groups, seen, &info);
    if (at < 0) {
        return;
    }
    u32 i = (u32)at;
    float p = values[i];
    float g = grads[i];
    float wd = consts[WEIGHT_DECAY];
    if (decoupled != 0u) {
        p = p * consts[SHRINK];
    } else if (wd != 0.0f) {
        g = fmaf(wd, p, g);
    }
    float mi = fmaf(consts[BETA1], m[i], consts[ONE_MINUS_BETA1] * g);
    float vi = fmaf(consts[BETA2], v[i], consts[ONE_MINUS_BETA2] * (g * g));
    m[i] = store_state(mi);
    v[i] = store_state(vi);
    float denominator = fmaf(sqrtf(vi), info[3], consts[EPS]);
    values[i] = fmaf(-info[2], mi / denominator, p);
}
