// The tiled GEMM: C = A·B + beta·C, with A (m × k) and B (k × n) each stored row-major or
// transposed, chosen by the TRANS_A and TRANS_B overrides, and C (m × n) row-major.
//
// A workgroup of 16 × 16 threads computes a 64 × 64 tile of C, each thread a 4 × 4 block
// strided by 16 in both directions (rows ty + 16i, columns tx + 16j), so its shared-memory
// reads are conflict-free and its stores coalesce. k is walked in steps of 16 through two
// shared tiles of 16 × 64 floats (padded to 16 × 66, 8.25 KiB); the accumulators are four named vec4s, never an
// indexed private array. Each output's products are added in increasing k, one fused
// multiply-add each; loads past an edge read 0, so any m, n and k work.

override TRANS_A: bool = false;
override TRANS_B: bool = false;

struct Dims {
    m: u32,
    n: u32,
    k: u32,
    beta: f32,
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

// A[row][kk], or 0 outside A.
fn load_a(row: u32, kk: u32) -> f32 {
    if (row >= dims.m || kk >= dims.k) {
        return 0.0;
    }
    if (TRANS_A) {
        return a[kk * dims.m + row];
    }
    return a[row * dims.k + kk];
}

// B[kk][col], or 0 outside B.
fn load_b(kk: u32, col: u32) -> f32 {
    if (kk >= dims.k || col >= dims.n) {
        return 0.0;
    }
    if (TRANS_B) {
        return b[col * dims.k + kk];
    }
    return b[kk * dims.n + col];
}

// Writes one output of C, if it is inside C, adding beta times the old value unless beta
// is 0 (tested by its bits, so either zero overwrites C without reading it).
fn store_one(row: u32, column: u32, value: f32) {
    if (row >= dims.m || column >= dims.n) {
        return;
    }
    let at = row * dims.n + column;
    var out = value;
    if ((bitcast<u32>(dims.beta) & 0x7fffffffu) != 0u) {
        out = out + dims.beta * c[at];
    }
    c[at] = out;
}

// Writes the four outputs of one of a thread's rows: columns col, col + 16, col + 32 and
// col + 48.
fn store_row(row: u32, col: u32, v: vec4<f32>) {
    store_one(row, col, v.x);
    store_one(row, col + 16u, v.y);
    store_one(row, col + 32u, v.z);
    store_one(row, col + 48u, v.w);
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
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
            As[ka * PAD + r] = load_a(row0 + r, t + ka);
            var kb = idx / TILE;
            var col = idx % TILE;
            if (TRANS_B) {
                kb = idx % STEP;
                col = idx / STEP;
            }
            Bs[kb * PAD + col] = load_b(t + kb, col0 + col);
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
    store_row(row0 + lid.y, col, acc0);
    store_row(row0 + lid.y + 16u, col, acc1);
    store_row(row0 + lid.y + 32u, col, acc2);
    store_row(row0 + lid.y + 48u, col, acc3);
}
