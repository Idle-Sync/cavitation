//! Apple's Metal under wgpu: the `MTLDevice` and command queue wgpu's Metal backend runs on,
//! pooled buffers whose `MTLBuffer` is known, and Metal work (Metal Performance Shaders)
//! recorded into a step between its wgpu passes, on the same queue and the same buffers.
//!
//! wgpu refuses raw Metal commands in an encoder that records its own passes, and keeps a
//! buffer's `MTLBuffer` to itself. So the buffers are made here, on wgpu's `MTLDevice`, and
//! handed to wgpu (`create_buffer_from_hal`); and a step that holds Metal work is cut into
//! wgpu command buffers and Metal command buffers, committed in order on wgpu's own queue
//! when the step is finished. Metal orders the command buffers of one queue and tracks the
//! buffers' hazards across them (wgpu's buffers and these are made with Metal's default,
//! tracked hazards), so a pass sees what the Metal work before it wrote.
//!
//! [`CAVITATION_MPS`](MPS_VAR)`=0` turns all of it off: the buffers are wgpu's own and
//! every kernel is WGSL, as on Vulkan and DX12.

/// The environment variable that, set to `0`, keeps Metal's own kernels off: wgpu's
/// buffers and the WGSL kernels only. Unset, or `1`, uses them where the adapter is Metal.
pub const MPS_VAR: &str = "CAVITATION_MPS";

/// Whether `value`, the value of [`MPS_VAR`], allows Metal's own kernels.
///
/// # Errors
///
/// Returns a message for a value other than `0` or `1`.
#[cfg_attr(
    not(target_os = "macos"),
    allow(dead_code, reason = "read only where there is Metal")
)]
pub(crate) fn mps_allowed(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => Err(format!("{MPS_VAR} takes 0 or 1, not `{other}`")),
    }
}

/// Whether [`MPS_VAR`] allows Metal's own kernels.
///
/// # Panics
///
/// If it has a value other than `0` or `1`.
#[cfg_attr(
    not(target_os = "macos"),
    allow(dead_code, reason = "read only where there is Metal")
)]
fn mps_on() -> bool {
    let value = std::env::var(MPS_VAR).ok();
    mps_allowed(value.as_deref()).unwrap_or_else(|e| panic!("{e}"))
}

#[cfg(target_os = "macos")]
pub(crate) use apple::*;
#[cfg(not(target_os = "macos"))]
pub(crate) use elsewhere::*;

/// Where there is no Metal: types with no values, so the code that holds them compiles
/// everywhere and their `Option`s are always `None`.
#[cfg(not(target_os = "macos"))]
mod elsewhere {
    /// Metal's device and queue under wgpu; there are none here.
    #[derive(Debug)]
    pub(crate) enum Metal {}

    /// A pooled buffer's `MTLBuffer`; there are none here.
    #[derive(Debug)]
    pub(crate) enum RawBuffer {}

    impl Metal {
        /// No Metal here, whatever `CAVITATION_MPS` says.
        pub(crate) fn open(_: &wgpu::Device, _: &wgpu::Queue) -> Option<Metal> {
            None
        }

        /// Never called: there is no `Metal` value.
        pub(crate) fn create_buffer(
            &self,
            _: &wgpu::Device,
            _: u64,
            _: wgpu::BufferUsages,
        ) -> Option<(wgpu::Buffer, RawBuffer)> {
            match *self {}
        }
    }

    /// One part of a step, in the order it runs: here always a wgpu command buffer.
    #[derive(Debug)]
    pub(crate) enum Piece {
        /// A finished wgpu command buffer.
        Wgpu(wgpu::CommandBuffer),
    }

    /// Metal work committed and not yet waited on; there is none here.
    #[derive(Debug)]
    pub(crate) enum Committed {}

    /// Submits `pieces`, in order, on `queue`.
    pub(crate) fn submit(
        _: Option<&Metal>,
        queue: &wgpu::Queue,
        pieces: &mut Vec<Piece>,
        _: &mut Vec<Committed>,
    ) {
        queue.submit(pieces.drain(..).map(|Piece::Wgpu(c)| c));
    }

    /// Nothing to wait for.
    pub(crate) fn wait(_: &mut Vec<Committed>, _: &mut Vec<(usize, Option<f64>)>) {}
}

/// On macOS: wgpu's `MTLDevice` and queue, buffers made on them, and Metal work committed in
/// a step's order.
#[cfg(target_os = "macos")]
mod apple {
    use std::fmt;
    use std::sync::Arc;

    use objc2::Message;
    use objc2::rc::{Retained, autoreleasepool};
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSRange;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
        MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLResourceOptions,
    };
    use objc2_metal_performance_shaders::MPSCommandBuffer;
    use wgpu::hal::api::Metal as HalMetal;

    /// wgpu's `MTLDevice` and command queue, retained.
    pub(crate) struct Metal {
        /// The device wgpu's Metal backend opened.
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        /// The queue wgpu submits to: Metal work is committed on it, in order with wgpu's.
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    }

    impl fmt::Debug for Metal {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Metal")
                .field("device", &self.device.name())
                .finish_non_exhaustive()
        }
    }

    /// A pooled buffer's `MTLBuffer`: the one wgpu's buffer wraps.
    pub(crate) struct RawBuffer(Retained<ProtocolObject<dyn MTLBuffer>>);

    // SAFETY: Metal resources may be used from any thread; wgpu's own Metal buffer type is
    // `Send` and `Sync` over the same object.
    unsafe impl Send for RawBuffer {}
    // SAFETY: as for `Send`.
    unsafe impl Sync for RawBuffer {}

    impl fmt::Debug for RawBuffer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("RawBuffer").field(&self.0.length()).finish()
        }
    }

    impl RawBuffer {
        /// The `MTLBuffer`.
        pub(crate) fn get(&self) -> &ProtocolObject<dyn MTLBuffer> {
            &self.0
        }
    }

    impl Metal {
        /// wgpu's `MTLDevice`.
        pub(crate) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
            &self.device
        }

        /// wgpu's `MTLDevice` and queue when `device` and `queue` are wgpu's Metal backend and
        /// [`MPS_VAR`](super::MPS_VAR) allows it; `None` on another backend or with
        /// `CAVITATION_MPS=0`.
        ///
        /// # Panics
        ///
        /// If `CAVITATION_MPS` has a value other than `0` or `1`.
        pub(crate) fn open(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Metal> {
            if !super::mps_on() {
                return None;
            }
            // SAFETY: the raw device is only retained here, never destroyed.
            let hal_device = unsafe { device.as_hal::<HalMetal>() }?;
            let raw_device = hal_device.raw_device().clone();
            // SAFETY: the raw queue is only retained and committed to, never destroyed.
            let hal_queue = unsafe { queue.as_hal::<HalMetal>() }?;
            let raw_queue = hal_queue.as_raw().retain();
            Some(Metal {
                device: raw_device,
                queue: raw_queue,
            })
        }

        /// A buffer of `size` bytes for `usage`, made on the `MTLDevice` in private storage
        /// with tracked hazards, filled with zeros (wgpu fills a new buffer with zeros before
        /// its first read) and handed to wgpu; with its `MTLBuffer`. `None` for a usage other
        /// than storage, uniform and copies, which wgpu makes itself: mapping needs shared
        /// storage, and wgpu checks and pads index, vertex and indirect buffers.
        ///
        /// # Panics
        ///
        /// If Metal refuses the buffer, or the fill fails, as wgpu panics on a buffer it
        /// cannot make.
        pub(crate) fn create_buffer(
            &self,
            device: &wgpu::Device,
            size: u64,
            usage: wgpu::BufferUsages,
        ) -> Option<(wgpu::Buffer, RawBuffer)> {
            let ours = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            if usage.is_empty() || !ours.contains(usage) {
                return None;
            }
            let len = usize::try_from(size).expect("a buffer size the host can address");
            let raw = self
                .device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModePrivate)
                .unwrap_or_else(|| panic!("Metal refused a buffer of {size} bytes"));
            autoreleasepool(|_| {
                let cb = self.queue.commandBuffer().expect("a Metal command buffer");
                let blit = cb.blitCommandEncoder().expect("a Metal blit encoder");
                blit.fillBuffer_range_value(&raw, NSRange::new(0, len), 0);
                blit.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                assert!(
                    cb.status() == MTLCommandBufferStatus::Completed,
                    "zero-filling a new buffer of {size} bytes failed: {:?}",
                    cb.error().map(|e| e.localizedDescription().to_string())
                );
            });
            // SAFETY: `raw` is a buffer of `size` bytes on wgpu's own device, and no other
            // owner destroys it.
            let hal = unsafe { wgpu::hal::metal::Device::buffer_from_raw(raw.clone(), size) };
            let descriptor = wgpu::BufferDescriptor {
                label: Some("cavitation pooled buffer"),
                size,
                usage,
                mapped_at_creation: false,
            };
            // SAFETY: `hal` was made on this device's `MTLDevice`, with `size` bytes in
            // private storage, which storage, uniform and copy usages allow; it holds zeros,
            // the fill having completed, so wgpu may take it as initialised.
            let buffer = unsafe { device.create_buffer_from_hal::<HalMetal>(hal, &descriptor) };
            Some((buffer, RawBuffer(raw)))
        }
    }

    /// Metal work for a step. It runs after the step's work recorded before it and before
    /// the work recorded after it.
    pub(crate) trait MetalOp: fmt::Debug + Send + Sync {
        /// Encodes the work into `cb`'s current command buffer
        /// (`rootCommandBuffer`): it ends every encoder it opens and does not commit `cb`,
        /// which the step does after; it may commit and continue (as MPSGraph does).
        fn encode(&self, cb: &MPSCommandBuffer);
    }

    /// One part of a step, in the order it runs.
    #[derive(Debug)]
    pub(crate) enum Piece {
        /// A finished wgpu command buffer.
        Wgpu(wgpu::CommandBuffer),
        /// Metal work, with the index of its profile entry when it is timed.
        Metal(Arc<dyn MetalOp>, Option<usize>),
    }

    /// Metal work committed, to wait for and check after the step.
    pub(crate) struct Committed {
        /// The first command buffer the work was encoded into.
        first: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        /// The last, which the step commits: the same as `first` unless the work committed
        /// and continued.
        last: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        /// The profile entry it is timed in, if any.
        entry: Option<usize>,
        /// The work, kept until it has run.
        op: Arc<dyn MetalOp>,
    }

    impl fmt::Debug for Committed {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Committed")
                .field("op", &self.op)
                .field("entry", &self.entry)
                .finish_non_exhaustive()
        }
    }

    // SAFETY: command buffers may be waited on and read from any thread once committed.
    unsafe impl Send for Committed {}
    // SAFETY: as for `Send`.
    unsafe impl Sync for Committed {}

    /// Submits `pieces` in order: each run of wgpu command buffers in one `submit` on
    /// `queue`, each piece of Metal work encoded and committed on `metal`'s queue, which is
    /// the same queue. The Metal work goes to `committed`.
    ///
    /// # Panics
    ///
    /// If there is Metal work and no `metal`.
    pub(crate) fn submit(
        metal: Option<&Metal>,
        queue: &wgpu::Queue,
        pieces: &mut Vec<Piece>,
        committed: &mut Vec<Committed>,
    ) {
        let mut pieces = pieces.drain(..).peekable();
        while pieces.peek().is_some() {
            queue.submit(std::iter::from_fn(|| {
                if matches!(pieces.peek(), Some(Piece::Wgpu(_))) {
                    let Some(Piece::Wgpu(c)) = pieces.next() else {
                        unreachable!("just looked")
                    };
                    Some(c)
                } else {
                    None
                }
            }));
            while let Some(Piece::Metal(..)) = pieces.peek() {
                let Some(Piece::Metal(op, entry)) = pieces.next() else {
                    unreachable!("just looked")
                };
                let metal = metal.expect("Metal work is recorded only where there is Metal");
                autoreleasepool(|_| {
                    // SAFETY: the queue is wgpu's, alive for as long as `metal`.
                    let cb =
                        unsafe { MPSCommandBuffer::commandBufferFromCommandQueue(&metal.queue) };
                    // SAFETY: an MPS command buffer always has a current command buffer.
                    let first = unsafe { cb.rootCommandBuffer() };
                    op.encode(&cb);
                    // SAFETY: as above; after a commit and continue it is the new one.
                    let last = unsafe { cb.rootCommandBuffer() };
                    cb.commit();
                    committed.push(Committed {
                        first,
                        last,
                        entry,
                        op,
                    });
                });
            }
        }
    }

    /// Waits for each piece of `committed` Metal work, and gives each timed one's GPU time
    /// in nanoseconds (from the start of its first command buffer to the end of its last)
    /// in `times`, as (entry, time).
    ///
    /// # Panics
    ///
    /// If the first or the last command buffer of a piece did not complete, with Metal's
    /// error. A command buffer the work committed in between (by committing and continuing)
    /// is not checked: MPS hands it no further out.
    pub(crate) fn wait(committed: &mut Vec<Committed>, times: &mut Vec<(usize, Option<f64>)>) {
        for c in committed.drain(..) {
            for cb in [&c.first, &c.last] {
                cb.waitUntilCompleted();
                assert!(
                    cb.status() == MTLCommandBufferStatus::Completed,
                    "Metal work {:?} failed: {:?}",
                    c.op,
                    cb.error().map(|e| e.localizedDescription().to_string())
                );
            }
            if let Some(entry) = c.entry {
                let seconds = c.last.GPUEndTime() - c.first.GPUStartTime();
                times.push((entry, (seconds >= 0.0).then_some(seconds * 1e9)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The `CAVITATION_MPS` values; and on macOS, Metal work recorded in a step running in
    //! the step's order on the pooled buffers, and timed in a profiled step.

    use super::*;

    /// `0` turns Metal's kernels off, `1` and no value leave them on, and anything else is an
    /// error that names the values.
    #[test]
    fn mps_values() {
        assert_eq!(mps_allowed(None), Ok(true));
        assert_eq!(mps_allowed(Some("1")), Ok(true));
        assert_eq!(mps_allowed(Some("0")), Ok(false));
        for bad in ["", "yes", "2", " 0"] {
            let err = mps_allowed(Some(bad)).unwrap_err();
            assert!(err.contains("0 or 1"), "{bad:?}: {err}");
        }
    }

    /// Metal work in steps, on the pooled buffers, on a Metal adapter.
    #[cfg(target_os = "macos")]
    mod on_metal {
        use std::sync::Arc;

        use objc2::Message;
        use objc2::rc::Retained;
        use objc2::runtime::ProtocolObject;
        use objc2_metal::{MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};
        use objc2_metal_performance_shaders::MPSCommandBuffer;

        use super::super::MetalOp;
        use crate::{Backend, BufferPool, Gpu, PooledBuffer, StepEncoder, StepSizes};

        /// Metal work that copies `len` bytes from one buffer to another with blits, in two
        /// halves with a commit and continue between them when `split`, as MPSGraph may do.
        #[derive(Debug)]
        struct Copy {
            /// The buffer read.
            from: Retained<ProtocolObject<dyn MTLBuffer>>,
            /// The buffer written.
            to: Retained<ProtocolObject<dyn MTLBuffer>>,
            /// The bytes copied.
            len: usize,
            /// Whether to commit and continue between the halves.
            split: bool,
        }

        impl Copy {
            /// Blits `len` bytes at `at` into `cb`'s current command buffer.
            fn blit(&self, cb: &MPSCommandBuffer, at: usize, len: usize) {
                // SAFETY: an MPS command buffer always has a current command buffer.
                let raw = unsafe { cb.rootCommandBuffer() };
                let blit = raw.blitCommandEncoder().expect("a blit encoder");
                // SAFETY: both buffers hold at least `at + len` bytes.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &self.from, at, &self.to, at, len,
                    );
                }
                blit.endEncoding();
            }
        }

        // SAFETY: Metal buffers may be used from any thread.
        unsafe impl Send for Copy {}
        // SAFETY: as for `Send`.
        unsafe impl Sync for Copy {}

        impl MetalOp for Copy {
            fn encode(&self, cb: &MPSCommandBuffer) {
                if self.split {
                    let half = self.len / 2;
                    self.blit(cb, 0, half);
                    // SAFETY: the first half's encoder is ended.
                    let before = unsafe { cb.rootCommandBuffer() };
                    // SAFETY: as above.
                    unsafe { cb.commitAndContinue() };
                    // SAFETY: as above.
                    let after = unsafe { cb.rootCommandBuffer() };
                    assert!(!std::ptr::eq(&*before, &*after), "a new command buffer");
                    self.blit(cb, half, self.len - half);
                } else {
                    self.blit(cb, 0, self.len);
                }
            }
        }

        /// The Metal GPU with its `MTLDevice`, or `None` (and the test checks nothing) without
        /// one or with `CAVITATION_MPS=0`.
        fn metal_gpu() -> Option<Gpu> {
            Gpu::with_backend(Backend::Metal).filter(|g| g.metal().is_some())
        }

        /// The `MTLBuffer` of a pooled buffer.
        fn raw(b: &PooledBuffer) -> Retained<ProtocolObject<dyn MTLBuffer>> {
            b.raw()
                .expect("a pooled storage buffer on Metal")
                .get()
                .retain()
        }

        /// The little-endian bytes of `v`.
        fn bytes(v: &[u32]) -> Vec<u8> {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        }

        /// Storage buffers from the pool have their `MTLBuffer`, read as zeros (as fresh
        /// pages would without the fill, so the fill itself is not told apart here), and keep
        /// it when reused; a buffer that maps is wgpu's own.
        #[test]
        fn pooled_buffers_are_metal_buffers() {
            let Some(gpu) = metal_gpu() else { return };
            let mut pool = BufferPool::new(&gpu);
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let a = pool.take(1024, usage);
            let ptr = raw(&a);
            assert_eq!(ptr.length(), 1024);
            let mut steps = StepEncoder::new(&gpu, StepSizes::default());
            let mut step = steps.begin();
            step.read_back(a.buffer(), 0, 1024);
            assert!(step.finish().bytes().iter().all(|&b| b == 0));
            pool.give(a);
            let again = pool.take(1000, usage);
            assert!(std::ptr::eq(&*raw(&again), &*ptr));
            assert_eq!(pool.created(), 1);
            let mapped = pool.take(
                64,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            assert!(mapped.raw().is_none());
            pool.give(again);
            pool.give(mapped);
        }

        /// Metal work runs after the commands recorded before it and before those recorded
        /// after it: an upload, a Metal copy, a wgpu copy of what the Metal copy wrote and a
        /// second upload over the first buffer read back as recorded, in four steps in a row,
        /// every other one with a commit and continue inside the Metal work.
        #[test]
        fn metal_work_runs_in_step_order() {
            let Some(gpu) = metal_gpu() else { return };
            let mut pool = BufferPool::new(&gpu);
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let n = 4096;
            let (a, b, c) = (
                pool.take(4 * n, usage),
                pool.take(4 * n, usage),
                pool.take(4 * n, usage),
            );
            let copies = [false, true].map(|split| {
                Arc::new(Copy {
                    from: raw(&a),
                    to: raw(&b),
                    len: 4 * n as usize,
                    split,
                })
            });
            let mut steps = StepEncoder::new(
                &gpu,
                StepSizes {
                    upload_bytes: 8 * n,
                    readback_bytes: 12 * n,
                    ..StepSizes::default()
                },
            );
            for round in 0..4u32 {
                let first: Vec<u32> = (0..n as u32).map(|i| i * 3 + round).collect();
                let second: Vec<u32> = (0..n as u32).map(|i| i ^ 0x5a5a).collect();
                let mut step = steps.begin();
                step.upload(a.buffer(), 0, &bytes(&first));
                step.metal(copies[round as usize % 2].clone());
                step.copy(b.buffer(), 0, c.buffer(), 0, 4 * n);
                step.upload(a.buffer(), 0, &bytes(&second));
                step.read_back(a.buffer(), 0, 4 * n);
                step.read_back(b.buffer(), 0, 4 * n);
                step.read_back(c.buffer(), 0, 4 * n);
                let out = step.finish();
                let want = [bytes(&second), bytes(&first), bytes(&first)].concat();
                assert_eq!(out.bytes(), &want[..], "round {round}");
            }
            for buffer in [a, b, c] {
                pool.give(buffer);
            }
        }

        /// A profiled step times Metal work as a pass of its span, from its first command
        /// buffer's start to its last's end across a commit and continue; work outside a span
        /// is not timed.
        #[test]
        fn profiled_metal_work_is_timed() {
            let Some(gpu) = metal_gpu() else { return };
            if !gpu.info().timestamps {
                return;
            }
            let mut pool = BufferPool::new(&gpu);
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let len = 64 << 20;
            let (a, b) = (pool.take(len, usage), pool.take(len, usage));
            let copy = Arc::new(Copy {
                from: raw(&a),
                to: raw(&b),
                len: len as usize,
                split: true,
            });
            let mut steps = StepEncoder::new(&gpu, StepSizes::default());
            steps.set_profiling(true);
            let mut step = steps.begin();
            // Outside any span the work runs and is not timed.
            step.metal(copy.clone());
            step.span("before");
            step.clear(a.buffer(), 0, 256);
            step.span("metal");
            step.metal(copy.clone());
            let out = step.finish();
            let spans = out.spans();
            assert_eq!(spans[1].0, "metal");
            let span = spans[1].1.expect("the Metal work is timed");
            // The span's timed pass, still empty, is taken over by the work.
            let timed = out.timed();
            assert_eq!(timed.iter().filter(|t| t.span == 1).count(), 1);
            let last = timed.last().unwrap();
            assert_eq!((last.span, last.kind), (1, crate::TimedKind::Pass));
            let metal = last.nanoseconds.expect("the Metal work's own time");
            // Copying 64 MiB takes more than 10 µs on any Apple GPU.
            assert!(
                metal > 1e4 && span == metal,
                "{metal} ns of a span of {span} ns"
            );
            pool.give(a);
            pool.give(b);
        }

        /// Metal work recorded after other commands in a span gets an entry of its own, and
        /// work that takes over a later span's pass times that span, not an earlier entry.
        #[test]
        fn later_metal_work_times_its_own_span() {
            let Some(gpu) = metal_gpu() else { return };
            if !gpu.info().timestamps {
                return;
            }
            let mut pool = BufferPool::new(&gpu);
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let len = 64 << 20;
            let (a, b) = (pool.take(len, usage), pool.take(len, usage));
            let small = Arc::new(Copy {
                from: raw(&a),
                to: raw(&b),
                len: 256,
                split: false,
            });
            let big = Arc::new(Copy {
                from: raw(&a),
                to: raw(&b),
                len: len as usize,
                split: false,
            });
            let mut steps = StepEncoder::new(&gpu, StepSizes::default());
            steps.set_profiling(true);
            let mut step = steps.begin();
            step.span("first");
            step.clear(a.buffer(), 0, 256);
            step.metal(small);
            step.span("second");
            step.metal(big);
            let out = step.finish();
            let timed = out.timed();
            let per_span = |s| timed.iter().filter(|t| t.span == s).count();
            assert_eq!((per_span(0), per_span(1)), (2, 1), "{timed:?}");
            let second = out.spans()[1].1.expect("the second span is timed");
            let own = timed.iter().find(|t| t.span == 1).unwrap().nanoseconds;
            // Copying 64 MiB takes more than 10 µs on any Apple GPU.
            assert!(second > 1e4 && own == Some(second), "{timed:?}");
            let first = out.spans()[0].1.expect("the first span is timed");
            assert!(first < second, "{first} ns against {second} ns");
            pool.give(a);
            pool.give(b);
        }

        /// A step dropped before it is finished leaves nothing behind: the next step runs only
        /// its own commands and times only its own work.
        #[test]
        fn an_abandoned_step_leaves_nothing() {
            let Some(gpu) = metal_gpu() else { return };
            let mut pool = BufferPool::new(&gpu);
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let (a, b) = (pool.take(256, usage), pool.take(256, usage));
            let copy = Arc::new(Copy {
                from: raw(&a),
                to: raw(&b),
                len: 256,
                split: false,
            });
            let mut steps = StepEncoder::new(&gpu, StepSizes::default());
            steps.set_profiling(true);
            {
                let mut step = steps.begin();
                step.upload(a.buffer(), 0, &bytes(&[7; 64]));
                step.span("dropped");
                step.metal(copy.clone());
            }
            let mut step = steps.begin();
            step.span("kept");
            step.read_back(b.buffer(), 0, 256);
            let out = step.finish();
            assert!(out.bytes().iter().all(|&x| x == 0), "the dropped copy ran");
            assert_eq!(out.spans().len(), 1);
            pool.give(a);
            pool.give(b);
        }
    }
}
