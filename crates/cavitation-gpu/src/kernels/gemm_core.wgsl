// The tiled GEMM's product, after gemm_head.wgsl and a pair of loaders (gemm_load.wgsl or a
// kernel's own). A workgroup of 16 × 16 threads computes a 64 × 64 tile of product z = the
// workgroup's z, each thread a 4 × 4 block strided by 16 in both directions (rows ty + 16i,
// columns tx + 16j), so its shared-memory reads are conflict-free and its stores coalesce.
// k is walked in steps of 16 through the two padded shared tiles; the accumulators are four
// named vec4s, never an indexed private array. Each output's products are added in
// increasing k with fma; loads past an edge read 0, so any m, n and k work.

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

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let z = wg.z;
    let row0 = wg.y * TILE;
    let col0 = wg.x * TILE;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var t = 0u; t < dims.k; t = t + STEP) {
        // 1024 values of each tile, 4 per thread, read so that neighbouring threads read
        // neighbouring addresses in either layout.
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = li + 256u * i;
            var r = idx / STEP;
            var ka = idx % STEP;
            if (TRANS_A) {
                r = idx % TILE;
                ka = idx / TILE;
            }
            As[ka * PAD + r] = load_a(z, row0 + r, t + ka);
            var kb = idx / TILE;
            var col = idx % TILE;
            if (TRANS_B) {
                kb = idx % STEP;
                col = idx / STEP;
            }
            Bs[kb * PAD + col] = load_b(z, t + kb, col0 + col);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < STEP; kk = kk + 1u) {
            let o = kk * PAD;
            let bv = vec4<f32>(Bs[o + lid.x], Bs[o + lid.x + 16u], Bs[o + lid.x + 32u], Bs[o + lid.x + 48u]);
            acc0 = fma(vec4<f32>(As[o + lid.y]), bv, acc0);
            acc1 = fma(vec4<f32>(As[o + lid.y + 16u]), bv, acc1);
            acc2 = fma(vec4<f32>(As[o + lid.y + 32u]), bv, acc2);
            acc3 = fma(vec4<f32>(As[o + lid.y + 48u]), bv, acc3);
        }
        workgroupBarrier();
    }
    let col = col0 + lid.x;
    store_row(z, row0 + lid.y, col, acc0);
    store_row(z, row0 + lid.y + 16u, col, acc1);
    store_row(z, row0 + lid.y + 32u, col, acc2);
    store_row(z, row0 + lid.y + 48u, col, acc3);
}
