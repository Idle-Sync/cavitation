//! The bind group helpers the kernels with several storage buffers share: layouts and bind
//! groups from a list of bindings, the length and aliasing checks every call makes, and
//! uniforms uploaded in a step of their own.

use crate::{BufferPool, Gpu, PooledBuffer, StepEncoder};

/// A bind group layout entry for a compute buffer at `binding`, used as `kind` says.
fn entry(binding: u32, kind: Kind) -> wgpu::BindGroupLayoutEntry {
    let ty = match kind {
        Kind::Read => wgpu::BufferBindingType::Storage { read_only: true },
        Kind::Write => wgpu::BufferBindingType::Storage { read_only: false },
        Kind::Uniform(_) => wgpu::BufferBindingType::Uniform,
    };
    let min_binding_size = match kind {
        Kind::Uniform(size) => std::num::NonZeroU64::new(size),
        _ => None,
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size,
        },
        count: None,
    }
}

/// How a binding is used.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    /// A read-only storage buffer.
    Read,
    /// A read-write storage buffer.
    Write,
    /// A uniform of this many bytes.
    Uniform(u64),
}

/// A bind group layout and its pipeline layout, with the bindings `kinds` at 0, 1, ….
pub(crate) fn layout(
    gpu: &Gpu,
    label: &str,
    kinds: &[Kind],
) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    let device = gpu.device();
    let entries: Vec<_> = kinds
        .iter()
        .enumerate()
        .map(|(k, &kind)| entry(k as u32, kind))
        .collect();
    let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&group)],
        immediate_size: 0,
    });
    (group, pipeline)
}

/// A counted bind group on `layout` with `buffers` at bindings 0, 1, ….
pub(crate) fn bind(
    steps: &mut StepEncoder,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&PooledBuffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(k, b)| wgpu::BindGroupEntry {
            binding: k as u32,
            resource: b.binding(),
        })
        .collect();
    steps.bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

/// Panics, naming the `kernel` and `what`, unless `buffer` holds at least `values` 4-byte
/// values.
pub(crate) fn check_len(kernel: &str, what: &str, buffer: &PooledBuffer, values: u64) {
    assert!(
        buffer.len() >= 4 * values,
        "{kernel}: the {what} needs {values} values ({} bytes), but its buffer has {} bytes",
        4 * values,
        buffer.len()
    );
}

/// Panics, naming the `kernel`, unless the buffer it writes, `written` (named), is none of
/// the buffers `read` (named) that it is used with: wgpu refuses a dispatch that writes a
/// buffer it also reads, and an alias between them is a mistake in the caller either way.
pub(crate) fn check_distinct(
    kernel: &str,
    written: (&str, &PooledBuffer),
    read: &[(&str, &PooledBuffer)],
) {
    for (name, buffer) in read {
        assert!(
            written.1.buffer() != buffer.buffer(),
            "{kernel}: the {} and the {name} must be different buffers",
            written.0
        );
    }
}

/// A uniform buffer from `pool` holding `bytes`, uploaded in a step of its own.
pub(crate) fn upload_uniform(
    steps: &mut StepEncoder,
    pool: &mut BufferPool,
    bytes: &[u8],
) -> PooledBuffer {
    let buffer = pool.take(
        bytes.len() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let mut step = steps.begin();
    step.upload(buffer.buffer(), 0, bytes);
    let _ = step.finish();
    buffer
}

/// Little-endian bytes of `words`.
pub(crate) fn word_bytes<const N: usize>(words: [u32; N]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
