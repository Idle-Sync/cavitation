// The tiled GEMM's default loaders: A[row][kk] and B[kk][col] of product z from storage,
// with the leading dimensions, offsets and batch strides of Dims, or 0 outside the matrix.
// A kernel that builds one operand on the fly (an implicit-GEMM convolution reading image
// windows) supplies these two functions instead, with the same signatures.

fn load_a(z: u32, row: u32, kk: u32) -> f32 {
    if (row >= dims.m || kk >= dims.k) {
        return 0.0;
    }
    let base = dims.a_offset + z * dims.a_batch;
    if (TRANS_A) {
        return a[base + kk * dims.lda + row];
    }
    return a[base + row * dims.lda + kk];
}

fn load_b(z: u32, kk: u32, col: u32) -> f32 {
    if (kk >= dims.k || col >= dims.n) {
        return 0.0;
    }
    let base = dims.b_offset + z * dims.b_batch;
    if (TRANS_B) {
        return b[base + col * dims.ldb + kk];
    }
    return b[base + kk * dims.ldb + col];
}
