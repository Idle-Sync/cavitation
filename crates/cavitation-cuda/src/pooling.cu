// 2-D max and average pooling, forward and backward: a port of the four WGSL kernels in
// cavitation-gpu's kernels/pooling.rs, operation for operation, compiled without fused
// multiply-add. The host puts the geometry in front of this text as macros (FORWARD_TOTAL and
// BACKWARD_TOTAL, the outputs and the input pixels of the batch; H, W, OH, OW, KH, KW, SH, SW,
// PH, PW; DH and DW for max pooling, COUNT_PAD and DIVISOR for average pooling), as the WGSL
// kernels take it in override constants, so one compiled module serves one geometry and
// batch.
//
// One thread per value: per output forward, per input pixel backward. The backward passes are
// gathers, each input pixel adding its outputs' gradients in the CPU's order, so they need no
// atomics and give the same bits on every run.

typedef unsigned int u32;

// The thread's value t; the kernel returns when it is past the total values.
#define VALUE(total)                                   \
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;     \
    if (t >= (total)) {                                \
        return;                                        \
    }

// Whether the float with these bits is NaN.
__device__ __forceinline__ bool is_nan(u32 bits) {
    return (bits & 0x7fffffffu) > 0x7f800000u;
}

// The f32 bits as an unsigned key in the floats' order: -inf lowest, +inf highest.
__device__ __forceinline__ u32 order_key(u32 bits) {
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

// Whether the float with bits v is greater than the one with bits best, neither NaN.
__device__ __forceinline__ bool greater(u32 v, u32 best) {
    bool both_zero = ((v | best) & 0x7fffffffu) == 0u;
    return !both_zero && order_key(v) > order_key(best);
}

// The first output along an axis whose window can reach padded position p.
__device__ __forceinline__ u32 first_output(u32 p, u32 reach, u32 stride) {
    return p > reach ? (p - reach + stride - 1u) / stride : 0u;
}

#ifdef DH

// Max pooling forward, one thread per output: scans its window's in-image taps in row-major
// order, keeping the first tap and then each later one that is greater or NaN, unless the
// value kept so far is NaN and the new one is not. Values are compared as integers that order
// the f32 bits as the floats are ordered (the two zeros equal), and the winner's bits are
// copied unchanged; its winner is its position in the plane.
extern "C" __global__ void max_forward(const u32 *x, u32 *out, u32 *winners) {
    VALUE(FORWARD_TOTAL)
    u32 ox = t % OW;
    u32 oy = (t / OW) % OH;
    u32 base = (t / (OW * OH)) * H * W;
    u32 best = 0u;
    u32 at = 0xffffffffu;
    for (u32 a = 0u; a < KH; a = a + 1u) {
        u32 py = oy * SH + a * DH;
        if (py < PH || py - PH >= H) {
            continue;
        }
        u32 y = py - PH;
        for (u32 b = 0u; b < KW; b = b + 1u) {
            u32 px = ox * SW + b * DW;
            if (px < PW || px - PW >= W) {
                continue;
            }
            u32 pos = y * W + px - PW;
            u32 v = x[base + pos];
            bool take = at == 0xffffffffu || is_nan(v);
            if (!take && !is_nan(best)) {
                take = greater(v, best);
            }
            if (take) {
                best = v;
                at = pos;
            }
        }
    }
    out[t] = best;
    winners[t] = at;
}

// Max pooling backward, one thread per input pixel: adds to the pixel's gradient the gradient
// of each output that pixel won, those outputs taken in plane order, as the CPU adds them.
extern "C" __global__ void max_backward(const float *grad_out, const u32 *winners,
                                        float *grad_in) {
    VALUE(BACKWARD_TOTAL)
    u32 col = t % W;
    u32 row = (t / W) % H;
    u32 plane = t / (W * H);
    u32 pos = row * W + col;
    u32 py = row + PH;
    u32 px = col + PW;
    u32 y0 = first_output(py, (KH - 1u) * DH, SH);
    u32 y1 = min(py / SH, OH - 1u);
    u32 x0 = first_output(px, (KW - 1u) * DW, SW);
    u32 x1 = min(px / SW, OW - 1u);
    float acc = grad_in[t];
    for (u32 oy = y0; oy <= y1; oy = oy + 1u) {
        for (u32 ox = x0; ox <= x1; ox = ox + 1u) {
            u32 o = (plane * OH + oy) * OW + ox;
            if (winners[o] == pos) {
                acc = acc + grad_out[o];
            }
        }
    }
    grad_in[t] = acc;
}

#else

// Along one axis, in padded coordinates (the image's row r is r + P): the window's length
// counting the padding, and its first and one-past-last in-image positions, as avg_window in
// the CPU kernel computes them.
__device__ __forceinline__ uint3 window(u32 o, u32 k, u32 s, u32 p, u32 size) {
    u32 start = o * s;
    u32 end = min(start + k, size + 2u * p);
    u32 lo = max(start, p);
    return make_uint3(end - start, lo, max(min(end, size + p), lo));
}

// An output's divisor: the override, else the window counting the padding, else the pixels
// summed.
__device__ __forceinline__ float divisor(uint3 r, uint3 c) {
    if (DIVISOR != 0u) {
        return (float)DIVISOR;
    }
    if (COUNT_PAD) {
        return (float)(r.x * c.x);
    }
    return (float)((r.z - r.y) * (c.z - c.y));
}

// Average pooling forward, one thread per output: the sum of its window's in-image pixels in
// row-major order, divided by its divisor.
extern "C" __global__ void avg_forward(const float *x, float *out) {
    VALUE(FORWARD_TOTAL)
    u32 ox = t % OW;
    u32 oy = (t / OW) % OH;
    u32 base = (t / (OW * OH)) * H * W;
    uint3 r = window(oy, KH, SH, PH, H);
    uint3 c = window(ox, KW, SW, PW, W);
    float acc = 0.0f;
    for (u32 y = r.y; y < r.z; y = y + 1u) {
        for (u32 xx = c.y; xx < c.z; xx = xx + 1u) {
            acc = acc + x[base + (y - PH) * W + xx - PW];
        }
    }
    out[t] = acc / divisor(r, c);
}

// Average pooling backward, one thread per input pixel: adds to the pixel's gradient each
// covering output's gradient divided by that output's divisor, the outputs in plane order, as
// the CPU adds them.
extern "C" __global__ void avg_backward(const float *grad_out, float *grad_in) {
    VALUE(BACKWARD_TOTAL)
    u32 col = t % W;
    u32 row = (t / W) % H;
    u32 plane = t / (W * H);
    u32 py = row + PH;
    u32 px = col + PW;
    u32 y0 = first_output(py, KH - 1u, SH);
    u32 y1 = min(py / SH, OH - 1u);
    u32 x0 = first_output(px, KW - 1u, SW);
    u32 x1 = min(px / SW, OW - 1u);
    float acc = grad_in[t];
    for (u32 oy = y0; oy <= y1; oy = oy + 1u) {
        uint3 r = window(oy, KH, SH, PH, H);
        for (u32 ox = x0; ox <= x1; ox = ox + 1u) {
            uint3 c = window(ox, KW, SW, PW, W);
            acc = acc + grad_out[(plane * OH + oy) * OW + ox] / divisor(r, c);
        }
    }
    grad_in[t] = acc;
}

#endif
