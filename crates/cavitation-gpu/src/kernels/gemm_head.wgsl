// The tiled GEMM's declarations: the transposes, the Dims uniform and the bindings. A product is C = A·B + beta·C with A (m × k) and B (k × n), each stored
// row-major or transposed, with leading dimensions, element offsets and, for a batch of
// products along the grid's z axis, per-batch element strides; C (m × n) is row-major.
// gemm_load.wgsl supplies load_a and load_b and gemm_store.wgsl store_one and store_row, which
// another kernel (the implicit-GEMM convolution, conv_forward.wgsl) can replace with its own;
// gemm_core.wgsl is the tiled product itself, with its shared tiles, and gemm_core_wide.wgsl
// the same product on a 32 × 128 tile, for a C of at most 32 rows.

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
