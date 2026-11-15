//! Prints every GPU adapter wgpu offers on this machine, whether cavitation would use it,
//! and for each usable backend its features and limits. It then runs probe shaders that
//! show how this backend's shader compiler treats floating point:
//! - whether `a * b + c` is fused into one rounding;
//! - whether a sum is reassociated;
//! - whether subnormals are flushed;
//! - how NaN behaves under comparisons, `max` and a bit test.
//!
//! Run with `cargo run --release -p cavitation-gpu --example gpu_probe`.

use cavitation_gpu::{
    Backend, BufferPool, Gpu, PipelineCache, Shader, StepEncoder, StepSizes, wgpu,
};

/// The probe: each output is one floating-point question, asked on inputs read from a
/// buffer so the compiler cannot answer it at compile time.
///
/// Inputs: `x[0..3]` = a, b, c for a fused multiply-add test; `x[3..6]` = p, q, r for
/// reassociation (`(p + q) + r` against `p + (q + r)`); `x[6]` a subnormal; `x[7]` a quiet
/// NaN; `x[8]` zero.
const PROBE: Shader = Shader {
    name: "gpu probe",
    source: "
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read_write> y: array<f32>;
        @compute @workgroup_size(1)
        fn main() {
            y[0] = x[0] * x[1] + x[2];
            y[1] = fma(x[0], x[1], x[2]);
            y[2] = (x[3] + x[4]) + x[5];
            y[3] = x[3] + (x[4] + x[5]);
            y[4] = x[6] + x[8];
            y[5] = x[6] * 2.0;
            let n = x[7];
            y[6] = select(0.0, 1.0, n != n);
            y[7] = max(n, 0.0);
            y[8] = max(0.0, n);
            y[9] = select(0.0, 1.0, ((bitcast<u32>(n) >> 23u) & 0xffu) == 0xffu);
            y[10] = select(0.0, 1.0, n > 0.0 || n <= 0.0);
            y[11] = bitcast<f32>(bitcast<u32>(n * 1.0));
        }
    ",
};

/// The number of probe outputs.
const OUTPUTS: usize = 12;

/// Lists every adapter on every backend wgpu was built with, and the reason cavitation
/// would leave it out, if any.
fn list_adapters() {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::all();
    let instance = wgpu::Instance::new(descriptor);
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    println!("adapters wgpu offers: {}", adapters.len());
    for adapter in &adapters {
        let i = adapter.get_info();
        println!(
            "  {:?} | {} | {:?} | vendor 0x{:04x} device 0x{:04x} | driver {} {}",
            i.backend, i.name, i.device_type, i.vendor, i.device, i.driver, i.driver_info
        );
    }
    for backend in Backend::ALL {
        match Gpu::open(&[backend]) {
            Ok(gpu) => println!("cavitation on {backend}: {}", gpu.info().name),
            Err(err) => println!("cavitation on {backend}: {err}"),
        }
    }
}

/// Prints the features and limits of `gpu`'s adapter that the kernels care about.
fn print_capabilities(gpu: &Gpu) {
    let features = gpu.adapter().features();
    let limits = gpu.adapter().limits();
    for (name, feature) in [
        ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY),
        (
            "TIMESTAMP_QUERY_INSIDE_ENCODERS",
            wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        ),
        (
            "TIMESTAMP_QUERY_INSIDE_PASSES",
            wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        ),
        ("SHADER_F16", wgpu::Features::SHADER_F16),
        ("SUBGROUP", wgpu::Features::SUBGROUP),
        (
            "SHADER_FLOAT32_ATOMIC",
            wgpu::Features::SHADER_FLOAT32_ATOMIC,
        ),
    ] {
        println!("  adapter {name}: {}", features.contains(feature));
    }
    println!(
        "  adapter limits: workgroup memory {} B, invocations {}, workgroup size {}/{}/{}, \
         workgroups per dimension {}, storage binding {} B, buffer {} B, storage buffers per stage {}",
        limits.max_compute_workgroup_storage_size,
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroup_size_y,
        limits.max_compute_workgroup_size_z,
        limits.max_compute_workgroups_per_dimension,
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
        limits.max_storage_buffers_per_shader_stage,
    );
    println!(
        "  timestamp period: {} ns",
        gpu.queue().get_timestamp_period()
    );
}

/// Runs the probe shader on `gpu` and prints what each output says.
fn probe_floating_point(gpu: &Gpu) {
    let a = 1.0f32 + 2f32.powi(-12);
    let c = -(1.0f32 + 2f32.powi(-11));
    let (p, q, r) = (1.0f32, 2f32.powi(-24), 2f32.powi(-24));
    let subnormal = f32::from_bits(0x0000_1000);
    let nan = f32::from_bits(0x7fc0_0000);
    let inputs = [a, a, c, p, q, r, subnormal, nan, 0.0];
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let mut pool = BufferPool::new(gpu);
    let mut cache = PipelineCache::new(gpu);
    let mut steps = StepEncoder::new(gpu, StepSizes::default());
    let pipeline = match cache.get(PROBE, "main", &[]) {
        Ok(pipeline) => pipeline,
        Err(err) => {
            println!("  probe shader: {err}");
            return;
        }
    };
    let x = pool.take(4 * inputs.len() as u64, usage);
    let y = pool.take(4 * OUTPUTS as u64, usage);
    let group = steps.bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y.binding(),
            },
        ],
    });
    let bytes: Vec<u8> = inputs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut step = steps.begin();
    step.upload(x.buffer(), 0, &bytes);
    step.dispatch(&pipeline, &group, 1);
    step.read_back(y.buffer(), 0, 4 * OUTPUTS as u64);
    let output = step.finish();
    let (words, _) = output.bytes().as_chunks::<4>();
    let o: Vec<f32> = words.iter().map(|w| f32::from_le_bytes(*w)).collect();
    let unfused = a * a + c;
    println!(
        "  a*b+c = {:e} (CPU unfused {:e}, fused {:e}): {}",
        o[0],
        unfused,
        a.mul_add(a, c),
        if o[0] == unfused {
            "not fused"
        } else {
            "fused into one rounding"
        }
    );
    println!("  fma(a, b, c) = {:e}", o[1]);
    println!(
        "  (p+q)+r = {:e}, p+(q+r) = {:e} (CPU {:e} and {:e}): {}",
        o[2],
        o[3],
        (p + q) + r,
        p + (q + r),
        if o[2] == (p + q) + r && o[3] == p + (q + r) {
            "order kept"
        } else {
            "reassociated"
        }
    );
    println!(
        "  subnormal {subnormal:e}: x + 0 = {:e}, x * 2 = {:e}: {}",
        o[4],
        o[5],
        if o[4] == 0.0 || o[5] == 0.0 {
            "flushed"
        } else {
            "kept"
        }
    );
    println!(
        "  NaN: (n != n) = {}, max(n, 0) = {}, max(0, n) = {}, exponent-bits test = {}, \
         (n > 0 || n <= 0) = {}, n * 1 bits = 0x{:08x}",
        o[6],
        o[7],
        o[8],
        o[9],
        o[10],
        o[11].to_bits()
    );
}

/// Lists the adapters, then probes each backend cavitation can open here.
fn main() {
    list_adapters();
    for backend in Backend::ALL {
        let Ok(gpu) = Gpu::open(&[backend]) else {
            continue;
        };
        println!("\n{}", gpu.info());
        print_capabilities(&gpu);
        probe_floating_point(&gpu);
    }
}
