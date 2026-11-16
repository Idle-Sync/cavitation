// The tiled GEMM's declarations: the transposes, the Dims uniform, the bindings and the
// shared tiles. A product is C = A·B + beta·C with A (m × k) and B (k × n), each stored
// row-major or transposed, with leading dimensions, element offsets and, for a batch of
// products along the grid's z axis, per-batch element strides; C (m × n) is row-major.
// gemm_load.wgsl supplies load_a and load_b and gemm_store.wgsl store_one and store_row, which
// another kernel (the implicit-GEMM convolution, conv_forward.wgsl) can replace with its own;
// gemm_core.wgsl is the tiled product itself.

override TRANS_A: bool = false;
override TRANS_B: bool = false;

struct Dims {
    m: u32,
    n: u32,
    k: u32,
    beta: f32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    batch: u32,
    a_offset: u32,
    b_offset: u32,
    c_offset: u32,
    pad0: u32,
    a_batch: u32,
    b_batch: u32,
    c_batch: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const TILE: u32 = 64u;
const STEP: u32 = 16u;
// The row stride of the shared tiles: 64 plus 2 floats of padding, so that writes with k
// varying fastest across threads (A row-major, B transposed) fall in distinct banks.
const PAD: u32 = 66u;

// As[kk * PAD + r] holds A[row0 + r][t + kk]; Bs[kk * PAD + col] holds B[t + kk][col0 + col].
var<workgroup> As: array<f32, 1056>;
var<workgroup> Bs: array<f32, 1056>;
