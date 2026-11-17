//! Opens the GPU on this machine, if it has one, on the default backend and on each backend,
//! prints what was chosen and runs one small round trip through it. Each test skips without
//! a GPU, unless `CAVITATION_GPU_REQUIRED=1` makes that a failure.

use cavitation_gpu::{Backend, Gpu, wgpu};

/// Writes four numbers to a GPU buffer, copies them to a readable buffer and reads them back.
fn round_trip(gpu: &Gpu) {
    let data: Vec<u8> = [1.0f32, -2.0, 3.5, 1e-3]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let device = gpu.device();
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: data.len() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: data.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    gpu.queue().write_buffer(&storage, 0, &data);
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, data.len() as u64);
    gpu.queue().submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| {
        result.expect("map the readback buffer")
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("wait for the copy");
    assert_eq!(
        &*slice.get_mapped_range().expect("read the mapped range"),
        &data[..]
    );
}

/// The default GPU opens with at most 32 KiB of workgroup memory, prints its information
/// and copies data exactly.
#[test]
fn default_gpu() {
    let Some(gpu) = Gpu::new() else {
        match Gpu::open(Backend::on_this_platform()) {
            Err(err) => println!("no GPU, skipped: {err}"),
            Ok(_) => println!("no GPU at first, skipped; a second attempt found one"),
        }
        return;
    };
    println!("{}", gpu.info());
    let info = gpu.info();
    assert!(info.limits.max_compute_workgroup_storage_size <= 32 * 1024);
    assert_eq!(info.dx12_compiler.is_some(), info.backend == Backend::Dx12);
    assert_eq!(gpu.adapter().get_info().name, info.name);
    assert_eq!(
        gpu.device()
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY),
        info.timestamps
    );
    assert_eq!(
        gpu.device()
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
        info.encoder_timestamps
    );
    round_trip(&gpu);
}

/// Each backend this machine has opens on itself and copies data exactly. With
/// `CAVITATION_GPU_REQUIRED=1`, every backend this platform tries must open, except those
/// `CAVITATION_GPU_BACKEND` excludes.
#[test]
fn every_backend() {
    let mut opened = 0;
    for backend in Backend::ALL {
        let Some(gpu) = Gpu::with_backend(backend) else {
            println!("{backend}: none");
            continue;
        };
        println!("{backend}:\n{}", gpu.info());
        assert_eq!(gpu.info().backend, backend);
        round_trip(&gpu);
        opened += 1;
    }
    if opened == 0 {
        println!("no GPU: skipped");
    }
}
