// The convolution's input gradient as an implicit GEMM: the loaders that go between
// gemm_head.wgsl and gemm_store.wgsl. Product z is image z: dX (in × h·w) = W' (in ×
// out·kh·kw) · dY' (out·kh·kw × h·w). W'[c][kk] is the weight w[o][c][i][j] at tap
// kk = (o · kh + i) · kw + j. dY'[kk][col] is the output gradient of channel o at the window
// whose tap (i, j) lands on input position col = iy · w + ix, or 0 where no window does (the
// stride steps over it, or the window would start in the padding past the image). Each input
// gradient sums its taps in increasing kk, the CPU reference's order, gathered rather than
// scattered, so no two threads write one value and nothing needs atomics. A is the weight
// [out][in][kh][kw], B the output gradients [batch][out][oh][ow], C the input gradients
// [batch][in][h][w].

struct ConvGrad {
    cin: u32,
    h: u32,
    w: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    oh: u32,
    ow: u32,
    taps: u32,
    out_image: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(4) var<uniform> conv: ConvGrad;

// The index meaning "no window covers this position".
const NONE: u32 = 0xffffffffu;

// W'[row][kk]: the weight of output channel o = kk / taps on input channel row.
fn load_a(z: u32, row: u32, kk: u32) -> f32 {
    if (row >= dims.m || kk >= dims.k) {
        return 0.0;
    }
    let o = kk / conv.taps;
    let tap = kk - o * conv.taps;
    return a[(o * conv.cin + row) * conv.taps + tap];
}

// Along one axis, the output position q whose tap t lands on input position p, q · stride +
// t · dilation = p + padding, or NONE. Unsigned, so the subtraction is tested before it is
// taken. A q past the output is returned as is and refused by the caller.
fn source(p: u32, t: u32, stride: u32, padding: u32, dilation: u32) -> u32 {
    let padded = p + padding;
    let reach = t * dilation;
    if (padded < reach) {
        return NONE;
    }
    let q = padded - reach;
    if (q % stride != 0u) {
        return NONE;
    }
    return q / stride;
}

// dY'[kk][col] of image z.
fn load_b(z: u32, kk: u32, col: u32) -> f32 {
    if (kk >= dims.k || col >= dims.n) {
        return 0.0;
    }
    let o = kk / conv.taps;
    let tap = kk - o * conv.taps;
    let i = tap / conv.kw;
    let j = tap - i * conv.kw;
    let iy = col / conv.w;
    let ix = col - iy * conv.w;
    let oy = source(iy, i, conv.sh, conv.ph, conv.dh);
    let ox = source(ix, j, conv.sw, conv.pw, conv.dw);
    if (oy >= conv.oh || ox >= conv.ow) {
        return 0.0;
    }
    return b[z * conv.out_image + (o * conv.oh + oy) * conv.ow + ox];
}
