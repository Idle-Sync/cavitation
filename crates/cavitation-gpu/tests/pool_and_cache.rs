//! The buffer pool and the pipeline cache on this machine's GPU: warm steps create no
//! buffers and build no pipelines, a pooled buffer binds only the bytes asked for, `override`
//! values make distinct pipelines (including a workgroup array sized by one), and broken
//! shaders are errors. Each test skips without a GPU, unless `CAVITATION_GPU_REQUIRED=1`
//! makes that a failure.

use cavitation_gpu::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, wgpu};

/// Writes `arrayLength` of its first binding, as a float, to its second binding.
const LENGTH: Shader = Shader {
    name: "length",
    source: "
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read_write> y: array<f32>;
        @compute @workgroup_size(1)
        fn main() { y[0] = f32(arrayLength(&x)); }
    ",
};

/// Reverses each block of `TILE` values through a workgroup array of `TILE` floats, doubling
/// them: the array's size comes from an `override`.
const REVERSE: Shader = Shader {
    name: "reverse",
    source: "
        override TILE: u32 = 64u;
        var<workgroup> tile: array<f32, TILE>;
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read_write> y: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(local_invocation_index) i: u32) {
            if (i < TILE) { tile[i] = 2.0 * x[i]; }
            workgroupBarrier();
            if (i < TILE) { y[i] = tile[TILE - 1u - i]; }
        }
    ",
};

/// Usage for an input buffer: written from the host, read by shaders.
const INPUT: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);

/// Usage for an output buffer: written by shaders, copied out.
const OUTPUT: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC);

/// The GPU, or `None` (after saying so) to skip the test.
fn gpu() -> Option<Gpu> {
    let gpu = Gpu::new();
    if gpu.is_none() {
        println!("no GPU: skipped");
    }
    gpu
}

/// Runs one workgroup of `pipeline` with `x` at binding 0 and `y` at binding 1, and returns
/// the first `count` floats of `y`.
fn run(
    gpu: &Gpu,
    pipeline: &wgpu::ComputePipeline,
    x: &PooledBuffer,
    y: &PooledBuffer,
    count: usize,
) -> Vec<f32> {
    let device = gpu.device();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
    let bytes = (count * 4) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(y.buffer(), 0, &readback, 0, bytes);
    gpu.queue().submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| {
        result.expect("map the readback buffer")
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("wait for the GPU");
    let data = slice.get_mapped_range().expect("read the mapped range");
    let (words, _) = data.as_chunks::<4>();
    words.iter().map(|w| f32::from_le_bytes(*w)).collect()
}

/// Writes `values` to `buffer`.
fn write(gpu: &Gpu, buffer: &PooledBuffer, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.queue().write_buffer(buffer.buffer(), 0, &bytes);
}

/// A hundred steps that take and give back the same buffers create them in the first step
/// only.
#[test]
fn warm_steps_create_no_buffers() {
    let Some(gpu) = gpu() else { return };
    let mut pool = BufferPool::new(&gpu);
    let sizes = [
        (4000, INPUT),
        (300, INPUT),
        (1 << 20, OUTPUT),
        (12, OUTPUT),
        (
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        ),
    ];
    let mut after_first = 0;
    for step in 0..100 {
        let taken: Vec<PooledBuffer> = sizes
            .iter()
            .map(|&(len, usage)| pool.take(len, usage))
            .collect();
        for (buffer, &(len, _)) in taken.iter().zip(&sizes) {
            assert_eq!(buffer.len(), len);
            assert!(buffer.buffer().size() >= len);
        }
        for buffer in taken {
            pool.give(buffer);
        }
        if step == 0 {
            after_first = pool.created();
        }
    }
    assert_eq!(after_first, sizes.len());
    assert_eq!(pool.created(), sizes.len());
    assert_eq!(pool.free_buffers(), sizes.len());
}

/// Two buffers of one class and usage taken at once are two buffers; given back, both are
/// reused, and a different usage is never mixed in.
#[test]
fn reuse_is_by_class_and_usage() {
    let Some(gpu) = gpu() else { return };
    let mut pool = BufferPool::new(&gpu);
    let a = pool.take(600, INPUT);
    let b = pool.take(1000, INPUT);
    assert_eq!(pool.created(), 2);
    pool.give(a);
    pool.give(b);
    let c = pool.take(516, INPUT);
    let d = pool.take(1024, INPUT);
    assert_eq!(pool.created(), 2);
    let e = pool.take(600, OUTPUT);
    assert_eq!(pool.created(), 3);
    assert_eq!(pool.created_bytes(), 3 * 1024);
    drop((c, d, e));
}

/// A shader sees only the bytes asked for, not the size class: `arrayLength` of a 1000-byte
/// request in a 1024-byte buffer is 250.
#[test]
fn binding_covers_the_bytes_asked_for() {
    let Some(gpu) = gpu() else { return };
    let mut pool = BufferPool::new(&gpu);
    let mut cache = PipelineCache::new(&gpu);
    let pipeline = cache
        .get(LENGTH, "main", &[])
        .expect("build the length shader");
    let x = pool.take(1000, INPUT);
    let y = pool.take(4, OUTPUT);
    assert_eq!(x.buffer().size(), 1024);
    assert_eq!(run(&gpu, &pipeline, &x, &y, 1), [250.0]);
}

/// The same key gives the same pipeline without building again, whatever the order of the
/// `override` values; a different value builds a second pipeline from the same module.
#[test]
fn same_key_same_pipeline() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    let first = cache
        .get(REVERSE, "main", &[("TILE", 32.0)])
        .expect("build");
    for _ in 0..100 {
        let again = cache
            .get(REVERSE, "main", &[("TILE", 32.0)])
            .expect("cached");
        assert_eq!(again, first);
    }
    assert_eq!(cache.pipelines_built(), 1);
    let default = cache.get(REVERSE, "main", &[]).expect("build the default");
    assert_ne!(default, first);
    assert_eq!(cache.pipelines_built(), 2);
    assert_eq!(cache.modules_built(), 1);
}

/// A workgroup array sized by an `override` works at the default size and at an overridden
/// one, each giving its own exact result.
#[test]
fn override_sized_workgroup_array() {
    let Some(gpu) = gpu() else { return };
    let mut pool = BufferPool::new(&gpu);
    let mut cache = PipelineCache::new(&gpu);
    let x = pool.take(64 * 4, INPUT);
    let y = pool.take(64 * 4, OUTPUT);
    let input: Vec<f32> = (0..64).map(|i| i as f32 - 20.5).collect();
    write(&gpu, &x, &input);
    for tile in [64usize, 32, 7] {
        let constants: &[(&str, f64)] = if tile == 64 {
            &[]
        } else {
            &[("TILE", tile as f64)]
        };
        let pipeline = cache.get(REVERSE, "main", constants).expect("build");
        let got = run(&gpu, &pipeline, &x, &y, tile);
        let want: Vec<f32> = (0..tile).map(|i| 2.0 * input[tile - 1 - i]).collect();
        assert_eq!(got, want, "TILE = {tile}");
    }
}

/// A shader that does not parse, a missing entry point and an unknown `override` are each
/// an error naming the shader, and nothing is cached for them.
#[test]
fn broken_shaders_are_errors() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    let broken = Shader {
        name: "broken",
        source: "fn main( {",
    };
    for (shader, entry, constants, detail) in [
        (broken, "main", &[][..], "expected identifier"),
        (LENGTH, "missing", &[][..], "entry point 'missing'"),
        (
            REVERSE,
            "main",
            &[("NO_SUCH_CONSTANT", 1.0)][..],
            "NO_SUCH_CONSTANT",
        ),
    ] {
        let err = cache
            .get(shader, entry, constants)
            .expect_err("must not build");
        assert!(
            matches!(&err, GpuError::Shader { shader: s, .. } if s.starts_with(shader.name)),
            "{err}"
        );
        assert!(err.to_string().contains(detail), "{err}");
        println!("{err}");
    }
    assert_eq!(cache.pipelines_built(), 0);
    assert_eq!(cache.modules_built(), 0);
}

/// Two shaders with one name but different sources are a programming error.
#[test]
fn one_name_one_source() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    cache.get(LENGTH, "main", &[]).expect("build");
    let impostor = Shader {
        name: LENGTH.name,
        source: REVERSE.source,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.get(impostor, "main", &[])
    }));
    assert!(result.is_err());
}

/// A buffer from one pool cannot be given to another.
#[test]
fn give_to_another_pool_panics() {
    let Some(gpu) = gpu() else { return };
    let mut first = BufferPool::new(&gpu);
    let mut second = BufferPool::new(&gpu);
    let buffer = first.take(256, INPUT);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| second.give(buffer)));
    assert!(result.is_err());
    assert_eq!(second.free_buffers(), 0);
}

/// A length that is not whole 4-byte words is refused, as is one above the device's
/// largest storage binding.
#[test]
fn unusable_lengths_panic() {
    let Some(gpu) = gpu() else { return };
    let mut pool = BufferPool::new(&gpu);
    let too_long = gpu.info().limits.max_storage_buffer_binding_size + 4;
    for len in [10, 2, too_long] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let buffer = pool.take(len, INPUT);
            pool.give(buffer);
        }));
        assert!(result.is_err(), "{len}");
    }
    assert_eq!(pool.created(), 0);
}

/// A name whose first use failed to build still holds its source: a second shader under
/// that name is a programming error too.
#[test]
fn one_name_one_source_after_a_failure() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    let broken = Shader {
        name: "same name",
        source: "fn main( {",
    };
    assert!(cache.get(broken, "main", &[]).is_err());
    let other = Shader {
        name: "same name",
        source: LENGTH.source,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.get(other, "main", &[])
    }));
    assert!(result.is_err());
}

/// A NaN `override` value is an error, not a pipeline.
#[test]
fn nan_override_is_an_error() {
    let Some(gpu) = gpu() else { return };
    let mut cache = PipelineCache::new(&gpu);
    const SCALE: Shader = Shader {
        name: "scale",
        source: "
            override FACTOR: f32 = 1.0;
            @group(0) @binding(0) var<storage, read_write> x: array<f32>;
            @compute @workgroup_size(1)
            fn main() { x[0] = FACTOR * x[0]; }
        ",
    };
    let err = cache
        .get(SCALE, "main", &[("FACTOR", f64::NAN)])
        .expect_err("NaN must not build");
    assert!(matches!(err, GpuError::Shader { .. }), "{err}");
    assert!(cache.get(SCALE, "main", &[("FACTOR", 2.0)]).is_ok());
    assert_eq!(cache.pipelines_built(), 1);
}

/// Two specialisations of one shader built with one explicit layout run from one bind group,
/// each with its own exact result.
#[test]
fn explicit_layout_shares_bind_groups() {
    let Some(gpu) = gpu() else { return };
    let device = gpu.device();
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[storage(0, true), storage(1, false)],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&group)],
        immediate_size: 0,
    });
    let mut pool = BufferPool::new(&gpu);
    let mut cache = PipelineCache::new(&gpu);
    let x = pool.take(64 * 4, INPUT);
    let y = pool.take(64 * 4, OUTPUT);
    let input: Vec<f32> = (0..64).map(|i| 0.25 * i as f32).collect();
    write(&gpu, &x, &input);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &group,
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
    for tile in [64usize, 16] {
        let pipeline = cache
            .get_with_layout(REVERSE, "main", &[("TILE", tile as f64)], &layout)
            .expect("build");
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (tile * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(y.buffer(), 0, &readback, 0, (tile * 4) as u64);
        gpu.queue().submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| {
            result.expect("map the readback buffer")
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for the GPU");
        let data = slice.get_mapped_range().expect("read the mapped range");
        let (words, _) = data.as_chunks::<4>();
        let got: Vec<f32> = words.iter().map(|w| f32::from_le_bytes(*w)).collect();
        let want: Vec<f32> = (0..tile).map(|i| 2.0 * input[tile - 1 - i]).collect();
        assert_eq!(got, want, "TILE = {tile}");
    }
    // The same key with the layout is cached; without it, it is another pipeline.
    cache
        .get_with_layout(REVERSE, "main", &[("TILE", 16.0)], &layout)
        .expect("cached");
    assert_eq!(cache.pipelines_built(), 2);
    cache
        .get(REVERSE, "main", &[("TILE", 16.0)])
        .expect("build");
    assert_eq!(cache.pipelines_built(), 3);
    pool.give(x);
    pool.give(y);
}
