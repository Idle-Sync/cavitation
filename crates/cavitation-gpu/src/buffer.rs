//! GPU buffers from a pool, by size class: a buffer handed back is reused for the next
//! request of its class and usage, and the pool counts every buffer it creates, so a test
//! can check that warm steps create none.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::Gpu;

/// The smallest size class, in bytes. Smaller requests get a buffer of this size.
const SMALLEST_CLASS: u64 = 256;

/// Up to this size, classes are powers of two; above it, quarter steps between powers of
/// two, so a large buffer is at most a quarter larger than asked for.
const FINE_CLASSES_FROM: u64 = 4 << 20;

/// The id the next [`BufferPool`] gets, so a buffer can be matched to its pool.
static NEXT_POOL: AtomicU64 = AtomicU64::new(0);

/// A GPU buffer from a [`BufferPool`]: a wgpu buffer of the request's size class, of which
/// the first [`len`](PooledBuffer::len) bytes are the ones asked for.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::{BufferPool, Gpu, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut pool = BufferPool::new(&gpu);
///     let buffer = pool.take(1000, wgpu::BufferUsages::STORAGE);
///     assert_eq!(buffer.len(), 1000);
///     assert_eq!(buffer.buffer().size(), 1024); // the size class
///     pool.give(buffer);
/// }
/// ```
#[derive(Debug)]
pub struct PooledBuffer {
    /// The wgpu buffer, of the size class's size.
    buffer: wgpu::Buffer,
    /// The number of bytes asked for.
    len: u64,
    /// The id of the pool it came from.
    pool: u64,
}

impl PooledBuffer {
    /// The wgpu buffer. It can be larger than [`len`](PooledBuffer::len): bind it with
    /// [`binding`](PooledBuffer::binding) so a shader's `arrayLength` sees only the bytes
    /// asked for.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let buffer = pool.take(64, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE);
    ///     gpu.queue().write_buffer(buffer.buffer(), 0, &[0u8; 64]);
    /// }
    /// ```
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// The number of bytes asked for, a multiple of 4.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     assert_eq!(pool.take(12, wgpu::BufferUsages::STORAGE).len(), 12);
    /// }
    /// ```
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether no bytes were asked for. Never true: [`BufferPool::take`] refuses a size of
    /// zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     assert!(!pool.take(4, wgpu::BufferUsages::STORAGE).is_empty());
    /// }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// A binding of the first [`len`](PooledBuffer::len) bytes, for a bind group entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let buffer = pool.take(40, wgpu::BufferUsages::STORAGE);
    ///     let entry = wgpu::BindGroupEntry { binding: 0, resource: buffer.binding() };
    ///     # let _ = entry;
    /// }
    /// ```
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: NonZeroU64::new(self.len),
        })
    }
}

/// GPU buffers by size class and usage, reused once handed back.
///
/// [`take`](BufferPool::take) returns a buffer whose size is the request's size class,
/// reusing one handed back with [`give`](BufferPool::give) when there is one of that class
/// and usage, and creating one otherwise. Up to 4 MiB the classes are powers of two (at
/// least 256 bytes); above, they are quarter steps between powers of two (5, 6, 7 and
/// 8 MiB, then 10, 12, 14 and 16 MiB, and so on), so a buffer is at most twice its
/// request when small and a quarter more when large. Usages must match exactly to be
/// reused, so tensors are best given one usage. [`created`](BufferPool::created) counts
/// the buffers created, so code that takes and gives back the same sizes each step can
/// show it creates nothing after its first step. A buffer is not cleared when reused.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::{BufferPool, Gpu, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut pool = BufferPool::new(&gpu);
///     for _ in 0..10 {
///         let a = pool.take(4000, wgpu::BufferUsages::STORAGE);
///         let b = pool.take(300, wgpu::BufferUsages::STORAGE);
///         pool.give(a);
///         pool.give(b);
///     }
///     assert_eq!(pool.created(), 2);
/// }
/// ```
#[derive(Debug)]
pub struct BufferPool {
    /// The GPU the buffers are created on.
    gpu: Gpu,
    /// This pool's id, carried by its buffers.
    id: u64,
    /// The buffers handed back, by usage and size class.
    free: HashMap<(wgpu::BufferUsages, u64), Vec<wgpu::Buffer>>,
    /// The number of buffers created so far.
    created: usize,
    /// The bytes in the buffers created so far.
    created_bytes: u64,
    /// The number of buffers taken and not yet given back.
    outstanding: usize,
}

impl BufferPool {
    /// An empty pool on `gpu`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     assert_eq!(BufferPool::new(&gpu).created(), 0);
    /// }
    /// ```
    pub fn new(gpu: &Gpu) -> BufferPool {
        BufferPool {
            gpu: gpu.clone(),
            id: NEXT_POOL.fetch_add(1, Ordering::Relaxed),
            free: HashMap::new(),
            created: 0,
            created_bytes: 0,
            outstanding: 0,
        }
    }

    /// A buffer of at least `len` bytes with `usage`, reused if one of its size class and
    /// usage was handed back, created otherwise.
    ///
    /// # Panics
    ///
    /// If `len` is 0 or not a multiple of 4 (storage bindings and buffer writes need
    /// whole 4-byte words); if it is larger than the device's largest storage binding (for
    /// a `STORAGE` usage), largest uniform binding (for `UNIFORM`) or largest buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    ///     let first = pool.take(500, usage);
    ///     pool.give(first);
    ///     let again = pool.take(400, usage); // same class (512 bytes): reused
    ///     assert_eq!(again.len(), 400);
    ///     assert_eq!(pool.created(), 1);
    /// }
    /// ```
    #[must_use = "a buffer taken and dropped is never reused; give it back"]
    pub fn take(&mut self, len: u64, usage: wgpu::BufferUsages) -> PooledBuffer {
        assert!(len > 0, "a pooled buffer needs at least one byte");
        assert!(
            len.is_multiple_of(4),
            "a pooled buffer's length must be a multiple of 4 bytes, not {len}"
        );
        let limits = &self.gpu.info().limits;
        let mut max = limits.max_buffer_size;
        if usage.contains(wgpu::BufferUsages::STORAGE) {
            max = max.min(limits.max_storage_buffer_binding_size);
        }
        if usage.contains(wgpu::BufferUsages::UNIFORM) {
            max = max.min(limits.max_uniform_buffer_binding_size);
        }
        assert!(
            len <= max,
            "a buffer of {len} bytes is larger than this device allows for {usage:?} ({max} bytes)"
        );
        // The class can exceed the device's largest buffer when the request is close to it;
        // such a buffer is made at the request's size, which is then its own class.
        let class = size_class(len);
        let class = if class <= limits.max_buffer_size {
            class
        } else {
            len
        };
        let reused = self.free.get_mut(&(usage, class)).and_then(Vec::pop);
        let buffer = reused.unwrap_or_else(|| {
            self.created += 1;
            self.created_bytes += class;
            self.gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("cavitation pooled buffer"),
                size: class,
                usage,
                mapped_at_creation: false,
            })
        });
        self.outstanding += 1;
        PooledBuffer {
            buffer,
            len,
            pool: self.id,
        }
    }

    /// Hands a buffer back for reuse by a later [`take`](BufferPool::take) of its size
    /// class and usage.
    ///
    /// Work already submitted keeps the buffer alive, and GPU commands recorded after it
    /// (in the same encoder or later ones) are ordered after the commands recorded before.
    /// A `Queue::write_buffer` is different: it runs before every command buffer of the
    /// next submission. So a buffer given back while a recorded but unsubmitted encoder
    /// still reads it must not be written with `write_buffer` until that encoder has been
    /// submitted.
    ///
    /// # Panics
    ///
    /// If the buffer came from another pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let buffer = pool.take(8, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
    ///     pool.give(buffer);
    ///     assert_eq!(pool.free_buffers(), 1);
    /// }
    /// ```
    pub fn give(&mut self, buffer: PooledBuffer) {
        assert_eq!(buffer.pool, self.id, "this buffer came from another pool");
        self.outstanding -= 1;
        self.free
            .entry((buffer.buffer.usage(), buffer.buffer.size()))
            .or_default()
            .push(buffer.buffer);
    }

    /// The number of buffers this pool has created.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let a = pool.take(12, wgpu::BufferUsages::STORAGE);
    ///     let b = pool.take(12, wgpu::BufferUsages::STORAGE); // `a` is still out
    ///     assert_eq!(pool.created(), 2);
    ///     pool.give(a);
    ///     pool.give(b);
    /// }
    /// ```
    pub fn created(&self) -> usize {
        self.created
    }

    /// The bytes in the buffers this pool has created (their size classes, added up).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let a = pool.take(1000, wgpu::BufferUsages::STORAGE);
    ///     assert_eq!(pool.created_bytes(), 1024);
    ///     pool.give(a);
    /// }
    /// ```
    pub fn created_bytes(&self) -> u64 {
        self.created_bytes
    }

    /// The number of buffers handed back and not yet taken again.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     assert_eq!(pool.free_buffers(), 0);
    ///     let a = pool.take(12, wgpu::BufferUsages::STORAGE);
    ///     pool.give(a);
    ///     assert_eq!(pool.free_buffers(), 1);
    /// }
    /// ```
    pub fn free_buffers(&self) -> usize {
        self.free.values().map(Vec::len).sum()
    }

    /// The number of buffers taken and not yet given back.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{BufferPool, Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let a = pool.take(12, wgpu::BufferUsages::STORAGE);
    ///     assert_eq!(pool.outstanding(), 1);
    ///     pool.give(a);
    ///     assert_eq!(pool.outstanding(), 0);
    /// }
    /// ```
    pub fn outstanding(&self) -> usize {
        self.outstanding
    }
}

/// The size class of a request of `len` bytes: the next power of two (at least 256) up to
/// 4 MiB, and above that the next quarter step between powers of two.
///
/// # Panics
///
/// If `len` is above 2^63, which has no power of two above it in a `u64`.
fn size_class(len: u64) -> u64 {
    let power = len
        .max(SMALLEST_CLASS)
        .checked_next_power_of_two()
        .unwrap_or_else(|| panic!("no size class holds {len} bytes"));
    if power <= FINE_CLASSES_FROM {
        return power;
    }
    let step = power / 8;
    len.div_ceil(step) * step
}

#[cfg(test)]
mod tests {
    //! The size classes; the pool itself is tested on a GPU in `tests/pool_and_cache.rs`.

    use super::*;

    /// Requests round up to a power of two of at least 256 bytes up to 4 MiB, then to
    /// quarter steps; a class is its own class, and never more than twice the request
    /// (a quarter more above 4 MiB).
    #[test]
    fn size_classes() {
        const MIB: u64 = 1 << 20;
        for (len, class) in [
            (4, 256),
            (256, 256),
            (260, 512),
            (1000, 1024),
            (1028, 2048),
            (4 * MIB, 4 * MIB),
            (4 * MIB + 4, 5 * MIB),
            (5 * MIB, 5 * MIB),
            (7 * MIB + 4, 8 * MIB),
            (8 * MIB + 4, 10 * MIB),
            (1100 * MIB, 1280 * MIB),
            (2048 * MIB, 2048 * MIB),
            (2048 * MIB + 4, 2560 * MIB),
        ] {
            assert_eq!(size_class(len), class, "{len}");
            assert_eq!(size_class(class), class, "{class}");
            assert!(class.is_multiple_of(256), "{class}");
            if len > FINE_CLASSES_FROM {
                assert!(class - len <= len / 4, "{len} -> {class}");
            }
        }
    }

    /// A request above 2^63 has no class.
    #[test]
    #[should_panic(expected = "no size class holds")]
    fn too_large_for_a_class() {
        size_class((1 << 63) + 4);
    }
}
