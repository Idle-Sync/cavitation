// The convolution forward pass as an implicit GEMM: the loaders and store that go between
// gemm_head.wgsl and gemm_core.wgsl. Product z is image z: C (out × oh·ow) = W (out ×
// in·kh·kw) · X̂ (in·kh·kw × oh·ow), with X̂ the image's windows, read from the image as the
// tiles are filled and never stored. A is the weight [out][in][kh][kw], shared by every
// image; B is the input images [batch][in][h][w]; C is the output images [batch][out][oh][ow].
// Taps outside the image (padding) read 0, so they add fma(w, 0, acc) = acc.

struct Conv {
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
    ow: u32,
    taps: u32,
    image: u32,
    has_bias: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(4) var<uniform> conv: Conv;
@group(0) @binding(5) var<storage, read> bias: array<f32>;

// W[row][kk]: output channel row, tap kk = (channel · kh + i) · kw + j.
fn load_a(z: u32, row: u32, kk: u32) -> f32 {
    if (row >= dims.m || kk >= dims.k) {
        return 0.0;
    }
    return a[row * dims.k + kk];
}

// X̂[kk][col] of image z: the input value under tap kk of the window at output position col
// = oy · ow + ox, or 0 where the tap falls in the padding.
fn load_b(z: u32, kk: u32, col: u32) -> f32 {
    if (kk >= dims.k || col >= dims.n) {
        return 0.0;
    }
    let ch = kk / conv.taps;
    let tap = kk - ch * conv.taps;
    let i = tap / conv.kw;
    let j = tap - i * conv.kw;
    let oy = col / conv.ow;
    let ox = col - oy * conv.ow;
    // Coordinates in the padded image; unsigned, so the padding is tested before it is taken off.
    let py = oy * conv.sh + i * conv.dh;
    let px = ox * conv.sw + j * conv.dw;
    if (py < conv.ph || px < conv.pw) {
        return 0.0;
    }
    let iy = py - conv.ph;
    let ix = px - conv.pw;
    if (iy >= conv.h || ix >= conv.w) {
        return 0.0;
    }
    return b[z * conv.image + (ch * conv.h + iy) * conv.w + ix];
}

// Writes one output of image z, the sum plus the channel's bias when there is one: the CPU
// reference's acc + bias[o], one rounding.
fn store_one(z: u32, row: u32, column: u32, value: f32) {
    if (row >= dims.m || column >= dims.n) {
        return;
    }
    var out = value;
    if (conv.has_bias != 0u) {
        out = out + bias[row];
    }
    c[z * dims.c_batch + row * dims.n + column] = out;
}

// Writes the four outputs of one of a thread's rows: columns col, col + 16, col + 32 and
// col + 48.
fn store_row(z: u32, row: u32, col: u32, v: vec4<f32>) {
    store_one(z, row, col, v.x);
    store_one(z, row, col + 16u, v.y);
    store_one(z, row, col + 32u, v.z);
    store_one(z, row, col + 48u, v.w);
}
