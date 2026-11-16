// The tiled GEMM's default store, after the loaders and before gemm_core.wgsl: C[row][column]
// of product z, with the leading dimension, offset and batch stride of Dims, plus beta times
// the old value. A kernel that adds something else (a convolution's bias) supplies store_one
// and store_row instead, with the same signatures.

// Writes one output of product z's C, if it is inside C, adding beta times the old value
// unless beta is 0 (tested by its bits, so either zero overwrites C without reading it).
fn store_one(z: u32, row: u32, column: u32, value: f32) {
    if (row >= dims.m || column >= dims.n) {
        return;
    }
    let at = dims.c_offset + z * dims.c_batch + row * dims.ldc + column;
    var out = value;
    if ((bitcast<u32>(dims.beta) & 0x7fffffffu) != 0u) {
        out = out + dims.beta * c[at];
    }
    c[at] = out;
}

// Writes the four outputs of one of a thread's rows: columns col, col + 16, col + 32 and
// col + 48.
fn store_row(z: u32, row: u32, col: u32, v: vec4<f32>) {
    store_one(z, row, col, v.x);
    store_one(z, row, col + 16u, v.y);
    store_one(z, row, col + 32u, v.z);
    store_one(z, row, col + 48u, v.w);
}
