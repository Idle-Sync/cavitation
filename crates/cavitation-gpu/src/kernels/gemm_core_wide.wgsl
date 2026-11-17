// The tiled GEMM's product for a C of few rows, in place of gemm_core.wgsl: a workgroup of
// 16 × 16 threads computes a 32 × 128 tile of product z = the workgroup's z, each thread a
// 2 × 8 block strided by 16 in both directions (rows ty + 16i, columns tx + 16j). With
// m ≤ 32 one row of tiles holds every row of C, where gemm_core.wgsl's 64-row tile would
// leave half of each thread's accumulators adding zeros. k is walked in steps of 16 through
// the two padded shared tiles, and each output's products are added in increasing k with
// fma, exactly as in gemm_core.wgsl, so an output has the same bits from either core. Loads
// past an edge read 0, so any m, n and k work.

const ROWS: u32 = 32u;
const COLS: u32 = 128u;
const STEP: u32 = 16u;
// The row strides of the shared tiles: the tile's width plus 2 floats of padding, so that
// writes with k varying fastest across threads fall in distinct banks.
const PAD_A: u32 = 34u;
const PAD_B: u32 = 130u;

// As[kk * PAD_A + r] holds A[row0 + r][t + kk]; Bs[kk * PAD_B + col] holds B[t + kk][col0 + col].
var<workgroup> As: array<f32, 544>;
var<workgroup> Bs: array<f32, 2080>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let z = wg.z;
    let row0 = wg.y * ROWS;
    let col0 = wg.x * COLS;
    // Row lid.y: columns lid.x + 16j for j < 4, then for 4 <= j < 8.
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    // Row lid.y + 16, the same columns.
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var t = 0u; t < dims.k; t = t + STEP) {
        // 512 values of A's tile, 2 per thread, and 2048 of B's, 8 per thread, read so that
        // neighbouring threads read neighbouring addresses in either layout.
        for (var i = 0u; i < 2u; i = i + 1u) {
            let idx = li + 256u * i;
            var r = idx / STEP;
            var ka = idx % STEP;
            if (TRANS_A) {
                r = idx % ROWS;
                ka = idx / ROWS;
            }
            As[ka * PAD_A + r] = load_a(z, row0 + r, t + ka);
        }
        for (var i = 0u; i < 8u; i = i + 1u) {
            let idx = li + 256u * i;
            var kb = idx / COLS;
            var col = idx % COLS;
            if (TRANS_B) {
                kb = idx % STEP;
                col = idx / STEP;
            }
            Bs[kb * PAD_B + col] = load_b(z, t + kb, col0 + col);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < STEP; kk = kk + 1u) {
            let oa = kk * PAD_A;
            let ob = kk * PAD_B + lid.x;
            let b0 = vec4<f32>(Bs[ob], Bs[ob + 16u], Bs[ob + 32u], Bs[ob + 48u]);
            let b1 = vec4<f32>(Bs[ob + 64u], Bs[ob + 80u], Bs[ob + 96u], Bs[ob + 112u]);
            let a0 = vec4<f32>(As[oa + lid.y]);
            let a1 = vec4<f32>(As[oa + lid.y + 16u]);
            acc0 = fma(a0, b0, acc0);
            acc1 = fma(a0, b1, acc1);
            acc2 = fma(a1, b0, acc2);
            acc3 = fma(a1, b1, acc3);
        }
        workgroupBarrier();
    }
    let col = col0 + lid.x;
    store_row(z, row0 + lid.y, col, acc0);
    store_row(z, row0 + lid.y, col + 64u, acc1);
    store_row(z, row0 + lid.y + 16u, col, acc2);
    store_row(z, row0 + lid.y + 16u, col + 64u, acc3);
}
