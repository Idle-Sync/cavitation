// 2-D batch normalisation: a port of the kernels in cavitation-gpu's kernels/batch_norm.rs,
// operation for operation, written after reduce.cu's text and compiled with it, without fused
// multiply-add. Where the WGSL kernels take `inverseSqrt`, these take the CPU's expression,
// 1 / sqrt(v); NVRTC's division and square root are correctly rounded, so every kernel but the
// sums computes the CPU's own f32 expression from the same inputs.
//
// The statistics buffer holds 3 * C values, [mean | inv | unbiased]: the mean,
// 1 / sqrt(var + eps) and the unbiased variance (0 for running statistics).

// The first pass of the centred squares, (x - mean)^2 summed per channel.
extern "C" __global__ void centred_partials(const float *x, const float *stats,
                                            float *partials, u32 outer, u32 channels,
                                            u32 inner, u32 chunks) {
    channel_partials(
        [=](u32 i, u32 c) {
            float d = x[i] - stats[c];
            return d * d;
        },
        partials, outer, channels, inner, chunks);
}

// The first pass of dy * xhat = dy * ((x - mean) * inv) summed per channel: gamma's gradient,
// and part of x's.
extern "C" __global__ void dy_xhat_partials(const float *dy, const float *x,
                                            const float *stats, float *partials, u32 outer,
                                            u32 channels, u32 inner, u32 chunks) {
    channel_partials(
        [=](u32 i, u32 c) { return dy[i] * ((x[i] - stats[c]) * stats[channels + c]); },
        partials, outer, channels, inner, chunks);
}

// The batch mean, one thread per channel: the sum divided by m.
extern "C" __global__ void stats_mean(const float *sums, float *stats, u32 channels, float m) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    stats[c] = sums[c] / m;
}

// inv = 1 / sqrt(sq / m + eps) and the unbiased variance sq / (m - 1) from the sum of centred
// squares, one thread per channel.
extern "C" __global__ void stats_finish(const float *squares, float *stats, u32 channels,
                                        float m, float m_minus_1, float eps) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    float sq = squares[c];
    stats[channels + c] = 1.0f / sqrtf(sq / m + eps);
    stats[2u * channels + c] = sq / m_minus_1;
}

// The statistics from running ones, one thread per channel: the running mean, then
// 1 / sqrt(var + eps), then 0.
extern "C" __global__ void stats_running(const float *mean, const float *var, float *stats,
                                         u32 channels, float eps) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    stats[c] = mean[c];
    stats[channels + c] = 1.0f / sqrtf(var[c] + eps);
    stats[2u * channels + c] = 0.0f;
}

// Normalisation, one thread per value: y = (x - mean) * inv * gamma + beta, with gamma = 1 and beta = 0
// when the layer is not affine, as the CPU computes it (so a -0 becomes +0 there too).
extern "C" __global__ void normalise(const float *x, const float *stats, const float *weight,
                                     const float *bias, float *y, u32 total, u32 channels,
                                     u32 inner, u32 affine) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    u32 c = (i / inner) % channels;
    float g = 1.0f;
    float s = 0.0f;
    if (affine != 0u) {
        g = weight[c];
        s = bias[c];
    }
    y[i] = (x[i] - stats[c]) * stats[channels + c] * g + s;
}

// The input gradient, one thread per value, from sum dy and sum dy * xhat per channel: with
// g = gamma (1 without), dx = inv / m * (m * dy*g - sum(dy)*g - xhat * sum(dy*xhat)*g)
// through batch statistics, dx = dy * g * inv through running ones.
extern "C" __global__ void backward(const float *grad_out, const float *x, const float *stats,
                                    const float *weight, const float *sum_dy,
                                    const float *sum_dy_xhat, float *grad_in, u32 total,
                                    u32 channels, u32 inner, u32 affine, u32 batch, float m) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    u32 c = (i / inner) % channels;
    float inv = stats[channels + c];
    float g = 1.0f;
    if (affine != 0u) {
        g = weight[c];
    }
    float gy = grad_out[i] * g;
    if (batch == 0u) {
        grad_in[i] = gy * inv;
        return;
    }
    float xhat = (x[i] - stats[c]) * inv;
    float sg = sum_dy[c] * g;
    float sgx = sum_dy_xhat[c] * g;
    grad_in[i] = inv / m * (m * gy - sg - xhat * sgx);
}

// The running statistics' update, one thread per channel, unless the step's finite flag is
// set: r = momentum * batch + keep * r for the mean and the unbiased variance.
extern "C" __global__ void running_update(const float *stats, const u32 *finite,
                                          float *running_mean, float *running_var,
                                          u32 channels, float momentum, float keep) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (finite[0] != 0u || c >= channels) {
        return;
    }
    running_mean[c] = momentum * stats[c] + keep * running_mean[c];
    running_var[c] = momentum * stats[2u * channels + c] + keep * running_var[c];
}
