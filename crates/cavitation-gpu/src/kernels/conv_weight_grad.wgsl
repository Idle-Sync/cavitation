// The convolution's weight gradient as an implicit GEMM: the loaders that go between
// gemm_head.wgsl and gemm_store.wgsl. dW (out × in·kh·kw) = dY (out × batch·oh·ow) · X̂
// (batch·oh·ow × in·kh·kw), with X̂ every image's windows, read from the images as the tiles
// are filled. The sum over the batch's positions is cut into fixed blocks of dims.k
// positions, g = n · oh·ow + oy · ow + ox counted across images: product z is block z, which
// adds its positions in increasing g, the CPU reference's order, into its own partial
// gradient, C[z][o][kk]. A second pass adds the partials in block order. A is the output
// gradients [batch][out][oh][ow], B the input images [batch][in][h][w], C the partials
// [blocks][out][in][kh][kw]. Taps in the padding read 0, so they add fma(dy, 0, acc) = acc.

struct ConvWeightGrad {
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
    plane: u32,
    taps: u32,
    cout: u32,
    positions: u32,
    pad0: u32,
}

@group(0) @binding(4) var<uniform> conv: ConvWeightGrad;

// dY[row][kk] of block z: the output gradient of channel row at position g = z · k + kk, or
// 0 past the batch's last position.
fn load_a(z: u32, row: u32, kk: u32) -> f32 {
    let g = z * dims.k + kk;
    if (row >= dims.m || kk >= dims.k || g >= conv.positions) {
        return 0.0;
    }
    let n = g / conv.plane;
    let p = g - n * conv.plane;
    return a[(n * conv.cout + row) * conv.plane + p];
}

// X̂[kk][col] of block z: the input value under tap col = (channel · kh + i) · kw + j of the
// window at position g = z · k + kk, or 0 where the tap falls in the padding or g is past the
// batch's last position.
fn load_b(z: u32, kk: u32, col: u32) -> f32 {
    let g = z * dims.k + kk;
    if (kk >= dims.k || col >= dims.n || g >= conv.positions) {
        return 0.0;
    }
    let n = g / conv.plane;
    let p = g - n * conv.plane;
    let oy = p / conv.ow;
    let ox = p - oy * conv.ow;
    let ch = col / conv.taps;
    let tap = col - ch * conv.taps;
    let i = tap / conv.kw;
    let j = tap - i * conv.kw;
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
    return b[((n * conv.cin + ch) * conv.h + iy) * conv.w + ix];
}
