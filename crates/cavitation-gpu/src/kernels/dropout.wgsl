// Dropout with a bit-packed mask, forward and backward alike: out = a · m with m = scale where
// bit i (bit i % 32 of word i / 32) of the mask is set and 0 where it is clear, as the CPU
// computes it (so a dropped −3 is −0 and a dropped NaN stays NaN). scale is 1 / f32(1 − p), as
// the CPU rounds it. The bindings match the element-wise kernels' layout, with the
// mask in place of b.

struct Params {
    n: u32,
    alpha: f32,
    scale: f32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn dropout(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {
    let i = (w.x + w.y * g.x) * 256u + l;
    if (i >= params.n) { return; }
    let keep = ((mask[i / 32u] >> (i % 32u)) & 1u) != 0u;
    out[i] = a[i] * select(0.0, params.scale, keep);
}