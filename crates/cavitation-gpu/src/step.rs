//! The step encoder: everything one training or inference step sends to the GPU, recorded
//! in one command encoder and submitted once; a profiled step's timestamps resolve in a
//! second submission after it. Uploads go through a mapped staging buffer,
//! results come back through one readback buffer, dispatches can be split over two
//! dimensions, and when profiling each named span is timed on the GPU: its compute passes,
//! and its copies where the device can write timestamps between passes.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::Gpu;

/// Upload, copy and readback offsets are kept to this alignment, wgpu's `MAP_ALIGNMENT`
/// (copies need only 4).
const ALIGN: u64 = wgpu::MAP_ALIGNMENT;

/// The largest number of workgroups in one dimension of a dispatch: wgpu's default
/// `max_compute_workgroups_per_dimension`, which the device is opened with.
pub const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// The sizes a [`StepEncoder`] is made with: they fix the buffers it holds, so a step
/// never creates one.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::StepSizes;
///
/// let sizes = StepSizes { upload_bytes: 1 << 20, ..StepSizes::default() };
/// assert_eq!(sizes.spans, 16);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepSizes {
    /// The most bytes one step uploads, in all its [`Step::upload`] calls together, each
    /// call's bytes rounded up to a multiple of 8.
    pub upload_bytes: u64,
    /// The most bytes one step reads back, in all its [`Step::read_back`] calls together,
    /// each call's bytes rounded up to a multiple of 8.
    pub readback_bytes: u64,
    /// The most spans one step has.
    pub spans: u32,
    /// The most timed compute passes one profiled step has: one per span, and one more
    /// for each span's dispatches that follow an upload, clear or readback recorded inside
    /// that span (a copy ends the pass it interrupts). With encoder timestamps, each run of
    /// copies in a span is timed too; a span begins with a pass, so a step never has more
    /// copy runs than passes, and room is kept for as many. The timestamps must fit wgpu's
    /// 4,096 queries: at most 2,048 timed passes, or 1,024 with encoder timestamps.
    pub timed_passes: u32,
}

impl Default for StepSizes {
    /// 1 MiB of uploads, 4 KiB of readback, 16 spans and 64 timed passes.
    fn default() -> StepSizes {
        StepSizes {
            upload_bytes: 1 << 20,
            readback_bytes: 4096,
            spans: 16,
            timed_passes: 64,
        }
    }
}

/// The query set for span timestamps and the buffer it resolves into.
#[derive(Debug)]
struct Timestamps {
    /// Two timestamps per timed entry, its beginning and end, at `2e` and `2e + 1`.
    queries: wgpu::QuerySet,
    /// Where the queries resolve (`QUERY_RESOLVE | COPY_SRC`), copied into the readback.
    resolve: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period: f64,
    /// Whether timestamps can be written on the encoder, so copy runs are timed.
    encoder: bool,
}

/// What a profiled step can time on this encoder, from [`StepEncoder::timing`]. What is not
/// timed is said by name, and its time is `None`, never 0.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::{Gpu, StepEncoder, StepSizes, Timing};
///
/// if let Some(gpu) = Gpu::new() {
///     let steps = StepEncoder::new(&gpu, StepSizes { timed_passes: 0, ..StepSizes::default() });
///     assert!(matches!(steps.timing(), Timing::Untimed { .. }));
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Timing {
    /// Nothing is timed, for the reason given.
    Untimed {
        /// Why: the missing feature or setting, by name.
        why: &'static str,
    },
    /// Compute passes are timed; copies (uploads, clears, readbacks) are not, for the reason
    /// given.
    Passes {
        /// Why copies are not timed: the missing feature, by name.
        copies_untimed: &'static str,
    },
    /// Compute passes and the runs of copies between them inside a span are timed.
    PassesAndCopies,
}

/// What one timed entry of a profiled step covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimedKind {
    /// One compute pass, timed from its beginning to its end.
    Pass,
    /// One run of consecutive copies (uploads, clears, readbacks) between passes inside a
    /// span, timed by timestamps written on the encoder before and after it. Copies before
    /// a step's first span are not timed.
    Copies,
}

/// One timed entry of a profiled step: a compute pass or a run of copies, in the order
/// recorded, from [`StepOutput::timed`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timed {
    /// The index of the span it belongs to, in [`StepOutput::spans`].
    pub span: usize,
    /// A pass or a run of copies.
    pub kind: TimedKind,
    /// Its GPU time in nanoseconds, from its own two timestamps, or `None` if its end reads
    /// before its beginning.
    pub nanoseconds: Option<f64>,
}

/// The timed entries of the step being recorded and their query indices: entry `e` writes
/// queries `2e` and `2e + 1`. Kept apart from the encoder so the order of entries is tested
/// without a GPU.
#[derive(Debug)]
struct Timeline {
    /// The entries in the order they were begun; their times are filled in when the step
    /// finishes.
    entries: Vec<Timed>,
    /// The timed passes so far.
    passes: u32,
    /// Whether the last entry is a run of copies not yet closed.
    open_copies: bool,
}

impl Timeline {
    /// An empty timeline with room for `passes` timed passes and as many copy runs.
    fn with_capacity(passes: u32) -> Timeline {
        Timeline {
            entries: Vec::with_capacity(2 * passes as usize),
            passes: 0,
            open_copies: false,
        }
    }

    /// Empties it for a new step.
    fn clear(&mut self) {
        self.entries.clear();
        self.passes = 0;
        self.open_copies = false;
    }

    /// Begins a timed pass of `span`, with at most `max` timed passes a step, and returns
    /// its entry. An open run of copies must be closed first.
    ///
    /// # Panics
    ///
    /// If the step would have more than `max` timed passes.
    fn pass(&mut self, span: usize, max: u32) -> u32 {
        assert!(
            self.passes < max,
            "this profiled step has more than the {max} timed passes its encoder was made for"
        );
        debug_assert!(!self.open_copies, "a run of copies is closed before a pass");
        self.passes += 1;
        self.push(span, TimedKind::Pass)
    }

    /// Opens a run of copies in `span` if none is open, and returns the query to write
    /// before its first copy.
    fn open_copies(&mut self, span: usize) -> Option<u32> {
        if self.open_copies {
            return None;
        }
        let runs = self.entries.len() as u32 - self.passes;
        assert!(
            runs < self.passes,
            "a run of copies follows a pass of its span, so runs never outnumber passes"
        );
        self.open_copies = true;
        Some(2 * self.push(span, TimedKind::Copies))
    }

    /// Closes the open run of copies, if any, and returns the query to write after its last
    /// copy.
    fn close_copies(&mut self) -> Option<u32> {
        if !std::mem::replace(&mut self.open_copies, false) {
            return None;
        }
        Some(2 * (self.entries.len() as u32 - 1) + 1)
    }

    /// Adds an entry and returns its index.
    fn push(&mut self, span: usize, kind: TimedKind) -> u32 {
        self.entries.push(Timed {
            span,
            kind,
            nanoseconds: None,
        });
        self.entries.len() as u32 - 1
    }
}

/// Records and submits steps on one GPU, holding every buffer a step needs.
///
/// [`begin`](StepEncoder::begin) starts a [`Step`], which records uploads, dispatches,
/// clears and readbacks into one command encoder, and [`Step::finish`] submits it, waits
/// for the GPU and gives the bytes read back. The staging buffer for uploads, the readback
/// buffer and the timestamp queries are made once, by [`new`](StepEncoder::new), sized by
/// [`StepSizes`]; [`buffers_made`](StepEncoder::buffers_made) counts them, and bind groups
/// made through [`bind_group`](StepEncoder::bind_group) are counted too, so a test can
/// show that warm steps make neither.
///
/// With [`set_profiling`](StepEncoder::set_profiling) on and timestamps available, each
/// [`Step::span`] runs in compute passes of its own with timestamps at their beginnings and
/// ends. Where the device can also write timestamps on the encoder, each run of copies
/// (uploads, clears, readbacks) inside a span is timed between two of them.
/// [`StepOutput::timed`] gives each pass's and copy run's time, and [`StepOutput::spans`]
/// each span's, the sum of its entries' times; [`timing`](StepEncoder::timing) says which
/// are timed, and names what is missing for the rest. A profiled step has a pass boundary
/// per span, so its timings describe a step structured for profiling. Without profiling,
/// dispatches share passes and spans only label them.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::{Gpu, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
///         label: None,
///         size: 16,
///         usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
///         mapped_at_creation: false,
///     });
///     let data: Vec<u8> = (1..=16).collect();
///     let mut step = steps.begin();
///     step.upload(&buffer, 0, &data);
///     step.read_back(&buffer, 0, 16);
///     let output = step.finish();
///     assert_eq!(output.bytes(), &data[..]);
/// }
/// ```
#[derive(Debug)]
pub struct StepEncoder {
    /// The GPU the steps run on.
    gpu: Gpu,
    /// The sizes the buffers were made with.
    sizes: StepSizes,
    /// The staging buffer for uploads (`MAP_WRITE | COPY_SRC`), mapped between steps.
    staging: wgpu::Buffer,
    /// Set when the staging buffer is mapped and ready to be written.
    staging_mapped: Arc<AtomicBool>,
    /// The readback buffer (`MAP_READ | COPY_DST`): read-back bytes, then timestamps.
    readback: wgpu::Buffer,
    /// The span timestamps, when the device has `TIMESTAMP_QUERY` and the sizes allow a
    /// timed pass.
    timestamps: Option<Timestamps>,
    /// Whether steps are profiled.
    profiling: bool,
    /// The bytes of the last step's readback, reused from step to step.
    host: Vec<u8>,
    /// The last step's spans: name and GPU nanoseconds, reused from step to step.
    spans: Vec<(&'static str, Option<f64>)>,
    /// The timed passes and copy runs of the last step.
    timeline: Timeline,
    /// The number of buffers this encoder has made.
    buffers: usize,
    /// The number of bind groups made through [`StepEncoder::bind_group`].
    bind_groups: usize,
    /// The number of steps submitted.
    submitted: u64,
}

impl StepEncoder {
    /// A step encoder on `gpu`, with its staging buffer, readback buffer and timestamp
    /// queries made now, sized by `sizes`.
    ///
    /// # Panics
    ///
    /// If `sizes.upload_bytes` or `sizes.readback_bytes` is 0 or larger than the device's
    /// largest buffer, or if the timestamps of `sizes.timed_passes` need more than wgpu's
    /// 4,096 queries (above 2,048 timed passes, or 1,024 with encoder timestamps).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     assert_eq!(steps.bind_groups_made(), 0);
    /// }
    /// ```
    pub fn new(gpu: &Gpu, sizes: StepSizes) -> StepEncoder {
        let device = gpu.device();
        let max = gpu.info().limits.max_buffer_size;
        let upload = round_up(sizes.upload_bytes, ALIGN);
        let encoder_timestamps = gpu.info().encoder_timestamps;
        // Two 8-byte timestamps per entry: a pass, and with encoder timestamps a copy run.
        let entries = u64::from(sizes.timed_passes) * if encoder_timestamps { 2 } else { 1 };
        let timestamp_bytes = entries * 16;
        assert!(
            !gpu.info().timestamps || 2 * entries <= u64::from(wgpu::QUERY_SET_MAX_QUERIES),
            "{} timed passes need {} timestamp queries, more than wgpu's {}",
            sizes.timed_passes,
            2 * entries,
            wgpu::QUERY_SET_MAX_QUERIES
        );
        let readback_bytes = round_up(sizes.readback_bytes, ALIGN) + timestamp_bytes;
        assert!(
            upload > 0 && upload <= max,
            "upload size {} must be between 1 byte and the device's largest buffer ({max} bytes)",
            sizes.upload_bytes
        );
        assert!(
            sizes.readback_bytes > 0 && readback_bytes <= max,
            "readback size {} must be between 1 byte and the device's largest buffer ({max} bytes)",
            sizes.readback_bytes
        );
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cavitation upload staging"),
            size: upload,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cavitation readback"),
            size: readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut buffers = 2;
        let timestamps = (gpu.info().timestamps && sizes.timed_passes > 0).then(|| {
            buffers += 1;
            Timestamps {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("cavitation span timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2 * entries as u32,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cavitation timestamp resolve"),
                    size: timestamp_bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                period: f64::from(gpu.queue().get_timestamp_period()),
                encoder: encoder_timestamps,
            }
        });
        StepEncoder {
            gpu: gpu.clone(),
            sizes,
            staging,
            staging_mapped: Arc::new(AtomicBool::new(true)),
            readback,
            timestamps,
            profiling: false,
            host: Vec::with_capacity(readback_bytes as usize),
            spans: Vec::with_capacity(sizes.spans as usize),
            timeline: Timeline::with_capacity(sizes.timed_passes),
            buffers,
            bind_groups: 0,
            submitted: 0,
        }
    }

    /// Turns profiling on or off for the steps that follow. With it on, each span runs in
    /// compute passes of its own, timed on the GPU when the device has timestamps
    /// ([`GpuInfo::timestamps`](crate::GpuInfo::timestamps)).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     steps.set_profiling(true);
    ///     assert!(steps.profiling());
    /// }
    /// ```
    pub fn set_profiling(&mut self, on: bool) {
        self.profiling = on;
    }

    /// What a profiled step on this encoder times, and by name what it cannot: nothing
    /// without `TIMESTAMP_QUERY` or with [`StepSizes::timed_passes`] 0, and its compute
    /// passes but not its copies without `TIMESTAMP_QUERY_INSIDE_ENCODERS` (Metal on Apple
    /// GPUs). It describes the device and sizes, whether or not profiling is on.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes, Timing};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     match steps.timing() {
    ///         Timing::PassesAndCopies => assert!(gpu.info().encoder_timestamps),
    ///         Timing::Passes { copies_untimed } => println!("copies untimed: {copies_untimed}"),
    ///         Timing::Untimed { why } => println!("untimed: {why}"),
    ///     }
    /// }
    /// ```
    pub fn timing(&self) -> Timing {
        match &self.timestamps {
            None if !self.gpu.info().timestamps => Timing::Untimed {
                why: "the adapter does not offer TIMESTAMP_QUERY",
            },
            None => Timing::Untimed {
                why: "the encoder was made with StepSizes::timed_passes 0",
            },
            Some(t) if !t.encoder => Timing::Passes {
                copies_untimed: "the adapter does not offer TIMESTAMP_QUERY_INSIDE_ENCODERS",
            },
            Some(_) => Timing::PassesAndCopies,
        }
    }

    /// Whether steps are profiled.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     assert!(!StepEncoder::new(&gpu, StepSizes::default()).profiling());
    /// }
    /// ```
    pub fn profiling(&self) -> bool {
        self.profiling
    }

    /// Makes a bind group on this encoder's device and counts it. Bind groups are meant to
    /// be made once, when a model is compiled, and kept: the count shows that steps make
    /// none.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let layout = gpu.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    ///         label: None,
    ///         entries: &[],
    ///     });
    ///     let _group = steps.bind_group(&wgpu::BindGroupDescriptor {
    ///         label: None,
    ///         layout: &layout,
    ///         entries: &[],
    ///     });
    ///     assert_eq!(steps.bind_groups_made(), 1);
    /// }
    /// ```
    pub fn bind_group(&mut self, descriptor: &wgpu::BindGroupDescriptor<'_>) -> wgpu::BindGroup {
        self.bind_groups += 1;
        self.gpu.device().create_bind_group(descriptor)
    }

    /// The number of bind groups made through [`bind_group`](StepEncoder::bind_group).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     assert_eq!(StepEncoder::new(&gpu, StepSizes::default()).bind_groups_made(), 0);
    /// }
    /// ```
    pub fn bind_groups_made(&self) -> usize {
        self.bind_groups
    }

    /// The number of buffers this encoder has made: the staging and readback buffers, and
    /// the timestamp resolve buffer when the device has timestamps. All are made by
    /// [`new`](StepEncoder::new); steps make none.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let made = steps.buffers_made();
    ///     steps.begin().finish();
    ///     assert_eq!(steps.buffers_made(), made);
    /// }
    /// ```
    pub fn buffers_made(&self) -> usize {
        self.buffers
    }

    /// The number of steps submitted so far.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     steps.begin().finish();
    ///     assert_eq!(steps.steps_submitted(), 1);
    /// }
    /// ```
    pub fn steps_submitted(&self) -> u64 {
        self.submitted
    }

    /// The sizes this encoder was made with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     assert_eq!(steps.sizes(), StepSizes::default());
    /// }
    /// ```
    pub fn sizes(&self) -> StepSizes {
        self.sizes
    }

    /// Starts recording a step. Its staging buffer is ready to write: if the GPU has not
    /// finished with it, this waits. A step dropped without [`Step::finish`] is discarded.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let output = steps.begin().finish();
    ///     assert!(output.bytes().is_empty());
    /// }
    /// ```
    #[must_use = "a step does nothing until it is finished"]
    pub fn begin(&mut self) -> Step<'_> {
        while !self.staging_mapped.load(Ordering::Acquire) {
            self.gpu
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("wait for the staging buffer");
        }
        let encoder = self
            .gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cavitation step"),
            });
        self.spans.clear();
        self.timeline.clear();
        Step {
            steps: self,
            encoder: Some(encoder),
            pass: None,
            span: None,
            uploaded: 0,
            read: 0,
            read_end: 0,
        }
    }
}

/// One step being recorded: uploads, clears, dispatches in spans, and readbacks, run in
/// the order recorded. Made by [`StepEncoder::begin`] and submitted by
/// [`finish`](Step::finish).
///
/// Commands run in the order they are recorded, so an upload recorded after a dispatch
/// reaches the buffer after that dispatch has read it.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let mut step = steps.begin();
///     step.span("forward");
///     step.span("backward");
///     let output = step.finish();
///     assert_eq!(output.spans().len(), 2);
/// }
/// ```
#[must_use = "a step does nothing until it is finished"]
pub struct Step<'a> {
    /// The encoder this step belongs to.
    steps: &'a mut StepEncoder,
    /// The command encoder, taken by [`Step::finish`].
    encoder: Option<wgpu::CommandEncoder>,
    /// The compute pass being recorded, if any.
    pass: Option<wgpu::ComputePass<'static>>,
    /// The index of the span being recorded, if any.
    span: Option<usize>,
    /// The bytes of the staging buffer used so far, each upload rounded up to 8 bytes.
    uploaded: u64,
    /// The bytes of the readback buffer used so far, each read rounded up to 8 bytes.
    read: u64,
    /// The end of the last read, not rounded: the length of the bytes returned.
    read_end: u64,
}

impl std::fmt::Debug for Step<'_> {
    /// Writes how much of the staging and readback buffers the step has used.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Step")
            .field("uploaded", &self.uploaded)
            .field("read", &self.read)
            .field("spans", &self.steps.spans.len())
            .finish_non_exhaustive()
    }
}

impl<'a> Step<'a> {
    /// Copies `data` into `buffer` at `offset`, through this step's staging buffer, when
    /// the step runs (in order with the step's other commands).
    ///
    /// # Panics
    ///
    /// If `data` is empty or its length or `offset` is not a multiple of 4, or if the
    /// step's uploads together, each rounded up to 8 bytes, exceed
    /// [`StepSizes::upload_bytes`] rounded up to 8.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
    ///         label: None,
    ///         size: 8,
    ///         usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    ///         mapped_at_creation: false,
    ///     });
    ///     let mut step = steps.begin();
    ///     step.upload(&buffer, 4, &7u32.to_le_bytes());
    ///     step.read_back(&buffer, 4, 4);
    ///     assert_eq!(step.finish().bytes(), &7u32.to_le_bytes());
    /// }
    /// ```
    pub fn upload(&mut self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        let len = data.len() as u64;
        assert!(
            len > 0 && len.is_multiple_of(4) && offset.is_multiple_of(4),
            "uploads need a whole number of 4-byte words at a 4-byte offset, not {len} bytes at {offset}"
        );
        let start = self.uploaded;
        let end = start + len;
        assert!(
            end <= self.steps.staging.size(),
            "this step uploads more than the {} bytes (rounded up to 8) its encoder was made for",
            self.steps.sizes.upload_bytes
        );
        self.steps
            .staging
            .get_mapped_range_mut(start..end)
            .expect("the staging buffer is mapped")
            .copy_from_slice(data);
        self.begin_copies();
        let staging = &self.steps.staging;
        self.encoder
            .as_mut()
            .expect("the step is being recorded")
            .copy_buffer_to_buffer(staging, start, buffer, offset, len);
        self.uploaded = round_up(end, ALIGN);
    }

    /// Fills `len` bytes of `buffer` from `offset` with zeros when the step runs, in order
    /// with the step's other commands.
    ///
    /// # Panics
    ///
    /// If `len` or `offset` is not a multiple of 4, or the range is outside the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
    ///         label: None,
    ///         size: 8,
    ///         usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    ///         mapped_at_creation: false,
    ///     });
    ///     let mut step = steps.begin();
    ///     step.upload(&buffer, 0, &[9; 8]);
    ///     step.clear(&buffer, 0, 4);
    ///     step.read_back(&buffer, 0, 8);
    ///     assert_eq!(step.finish().bytes(), &[0, 0, 0, 0, 9, 9, 9, 9]);
    /// }
    /// ```
    pub fn clear(&mut self, buffer: &wgpu::Buffer, offset: u64, len: u64) {
        assert!(
            len.is_multiple_of(4) && offset.is_multiple_of(4) && offset + len <= buffer.size(),
            "a clear needs a whole number of 4-byte words at a 4-byte offset inside the buffer, not {len} bytes at {offset}"
        );
        self.begin_copies();
        self.encoder
            .as_mut()
            .expect("the step is being recorded")
            .clear_buffer(buffer, offset, Some(len));
    }

    /// Starts a span named `name`: when profiling, the dispatches that follow run in timed
    /// compute passes of their own until the next span; otherwise the name only labels
    /// them, and they share the current pass.
    ///
    /// # Panics
    ///
    /// If the step has more spans than [`StepSizes::spans`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     steps.set_profiling(true);
    ///     let mut step = steps.begin();
    ///     step.span("conv1 forward");
    ///     let output = step.finish();
    ///     assert_eq!(output.spans()[0].0, "conv1 forward");
    /// }
    /// ```
    pub fn span(&mut self, name: &'static str) {
        let index = self.steps.spans.len();
        assert!(
            index < self.steps.sizes.spans as usize,
            "this step has more than the {} spans its encoder was made for",
            self.steps.sizes.spans
        );
        self.close_copies();
        self.steps.spans.push((name, None));
        self.span = Some(index);
        if self.steps.profiling {
            self.end_pass();
            self.begin_pass();
        }
    }

    /// Dispatches `workgroups` workgroups of `pipeline` in one dimension, with `bind_group`
    /// as group 0.
    ///
    /// # Panics
    ///
    /// If `workgroups` is 0 or above [`MAX_WORKGROUPS_PER_DIMENSION`]; a kernel written for
    /// more uses [`dispatch_split`](Step::dispatch_split).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, PipelineCache, Shader, StepEncoder, StepSizes};
    ///
    /// const NOTHING: Shader = Shader {
    ///     name: "nothing",
    ///     source: "@compute @workgroup_size(64) fn main() {}",
    /// };
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     let pipeline = cache.get(NOTHING, "main", &[]).unwrap();
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let group = steps.bind_group(&cavitation_gpu::wgpu::BindGroupDescriptor {
    ///         label: None,
    ///         layout: &pipeline.get_bind_group_layout(0),
    ///         entries: &[],
    ///     });
    ///     let mut step = steps.begin();
    ///     step.dispatch(&pipeline, &group, 1000);
    ///     step.finish();
    /// }
    /// ```
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
    ) {
        assert!(
            workgroups <= MAX_WORKGROUPS_PER_DIMENSION,
            "{workgroups} workgroups do not fit in one dimension; a kernel that numbers its \
             workgroups over two uses dispatch_split"
        );
        self.dispatch_grid(pipeline, bind_group, [workgroups, 1, 1]);
    }

    /// Dispatches `workgroups` workgroups of `pipeline` with `bind_group` as group 0, over
    /// two dimensions when there are more than [`MAX_WORKGROUPS_PER_DIMENSION`].
    ///
    /// Above the limit the grid is `x` = [`MAX_WORKGROUPS_PER_DIMENSION`] by
    /// `y` = `ceil(workgroups / x)`, so the kernel must number its workgroup as
    /// `workgroup_id.x + workgroup_id.y * num_workgroups.x` and skip those at or past
    /// `workgroups` (the last row can reach beyond it). A kernel that uses
    /// `global_invocation_id.x` alone would repeat the first row's work instead.
    ///
    /// # Panics
    ///
    /// If `workgroups` is 0, or more than [`MAX_WORKGROUPS_PER_DIMENSION`] squared.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, PipelineCache, Shader, StepEncoder, StepSizes};
    ///
    /// const NOTHING: Shader = Shader {
    ///     name: "nothing",
    ///     source: "@compute @workgroup_size(64)
    ///              fn main(@builtin(workgroup_id) w: vec3<u32>,
    ///                      @builtin(num_workgroups) n: vec3<u32>) {
    ///                  let group = w.x + w.y * n.x;
    ///              }",
    /// };
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     let pipeline = cache.get(NOTHING, "main", &[]).unwrap();
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let group = steps.bind_group(&cavitation_gpu::wgpu::BindGroupDescriptor {
    ///         label: None,
    ///         layout: &pipeline.get_bind_group_layout(0),
    ///         entries: &[],
    ///     });
    ///     let mut step = steps.begin();
    ///     step.dispatch_split(&pipeline, &group, 100_000); // 65,535 × 2 workgroups
    ///     step.finish();
    /// }
    /// ```
    pub fn dispatch_split(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
    ) {
        let (x, y) = grid(workgroups);
        self.dispatch_grid(pipeline, bind_group, [x, y, 1]);
    }

    /// Dispatches exactly the grid `[x, y, z]` of `pipeline` with `bind_group` as group 0.
    ///
    /// # Panics
    ///
    /// If any dimension is 0 or above [`MAX_WORKGROUPS_PER_DIMENSION`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, PipelineCache, Shader, StepEncoder, StepSizes};
    ///
    /// const NOTHING: Shader = Shader {
    ///     name: "nothing",
    ///     source: "@compute @workgroup_size(8, 8) fn main() {}",
    /// };
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     let pipeline = cache.get(NOTHING, "main", &[]).unwrap();
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let group = steps.bind_group(&cavitation_gpu::wgpu::BindGroupDescriptor {
    ///         label: None,
    ///         layout: &pipeline.get_bind_group_layout(0),
    ///         entries: &[],
    ///     });
    ///     let mut step = steps.begin();
    ///     step.dispatch_grid(&pipeline, &group, [16, 16, 1]);
    ///     step.finish();
    /// }
    /// ```
    pub fn dispatch_grid(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        [x, y, z]: [u32; 3],
    ) {
        for n in [x, y, z] {
            assert!(
                (1..=MAX_WORKGROUPS_PER_DIMENSION).contains(&n),
                "a dispatch dimension must be between 1 and {MAX_WORKGROUPS_PER_DIMENSION}, not {n}"
            );
        }
        if self.pass.is_none() {
            self.begin_pass();
        }
        let pass = self.pass.as_mut().expect("a pass was just begun");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }

    /// Copies `len` bytes of `buffer` from `offset` into this step's readback, after the
    /// commands recorded so far, and returns where they will be in
    /// [`StepOutput::bytes`]: after those of earlier `read_back` calls in the step, each
    /// part starting at a multiple of 8 bytes.
    ///
    /// # Panics
    ///
    /// If `len` is 0 or `len` or `offset` is not a multiple of 4, or if the step's reads
    /// together, each rounded up to 8 bytes, exceed [`StepSizes::readback_bytes`] rounded
    /// up to 8.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
    ///         label: None,
    ///         size: 4,
    ///         usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    ///         mapped_at_creation: false,
    ///     });
    ///     let mut step = steps.begin();
    ///     step.upload(&buffer, 0, &[1, 2, 3, 4]);
    ///     let at = step.read_back(&buffer, 0, 4);
    ///     let output = step.finish();
    ///     assert_eq!(&output.bytes()[at..at + 4], &[1, 2, 3, 4]);
    /// }
    /// ```
    pub fn read_back(&mut self, buffer: &wgpu::Buffer, offset: u64, len: u64) -> usize {
        assert!(
            len > 0 && len.is_multiple_of(4) && offset.is_multiple_of(4),
            "readbacks need a whole number of 4-byte words at a 4-byte offset, not {len} bytes at {offset}"
        );
        let start = self.read;
        let end = start + len;
        assert!(
            end <= round_up(self.steps.sizes.readback_bytes, ALIGN),
            "this step reads back more than the {} bytes (rounded up to 8) its encoder was made for",
            self.steps.sizes.readback_bytes
        );
        self.begin_copies();
        let readback = &self.steps.readback;
        self.encoder
            .as_mut()
            .expect("the step is being recorded")
            .copy_buffer_to_buffer(buffer, offset, readback, start, len);
        self.read = round_up(end, ALIGN);
        self.read_end = end;
        start as usize
    }

    /// Submits the step, waits for the GPU to finish it, and returns the bytes read back
    /// and the spans' GPU times. The staging buffer is mapped again for the next step.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let output = steps.begin().finish();
    ///     assert!(output.bytes().is_empty() && output.spans().is_empty());
    /// }
    /// ```
    pub fn finish(mut self) -> StepOutput<'a> {
        self.end_pass();
        self.close_copies();
        let Step {
            steps,
            encoder,
            read_end: user_bytes,
            ..
        } = self;
        let encoder = encoder.expect("the step is being recorded");
        let entries = steps.timeline.entries.len() as u64;
        let timestamp_at = round_up(steps.sizes.readback_bytes, ALIGN);
        // The timestamps resolve in a submission of their own, after the step's has
        // finished: on Metal, a resolve in the step's own command buffer, or in a second one
        // submitted with it, reads the counter samples of the step before.
        let resolve = steps.timestamps.as_ref().filter(|_| entries > 0).map(|t| {
            let mut resolve =
                steps
                    .gpu
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("cavitation timestamp resolve"),
                    });
            resolve.resolve_query_set(&t.queries, 0..(2 * entries) as u32, &t.resolve, 0);
            resolve.copy_buffer_to_buffer(
                &t.resolve,
                0,
                &steps.readback,
                timestamp_at,
                16 * entries,
            );
            resolve.finish()
        });
        steps.staging_mapped.store(false, Ordering::Release);
        steps.staging.unmap();
        steps.gpu.queue().submit([encoder.finish()]);
        if let Some(resolve) = resolve {
            steps
                .gpu
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("wait for the step");
            steps.gpu.queue().submit([resolve]);
        }
        steps.submitted += 1;
        let flag = Arc::clone(&steps.staging_mapped);
        steps
            .staging
            .map_async(wgpu::MapMode::Write, .., move |result| {
                result.expect("map the staging buffer for writing");
                flag.store(true, Ordering::Release);
            });
        let map_end = if entries > 0 {
            timestamp_at + 16 * entries
        } else {
            user_bytes
        };
        if map_end > 0 {
            steps
                .readback
                .map_async(wgpu::MapMode::Read, ..map_end, |result| {
                    result.expect("map the readback buffer")
                });
        }
        steps
            .gpu
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for the step");
        steps.host.clear();
        if map_end > 0 {
            {
                let view = steps
                    .readback
                    .get_mapped_range(..map_end)
                    .expect("the readback buffer is mapped");
                steps.host.extend_from_slice(&view[..user_bytes as usize]);
                if let Some(t) = steps.timestamps.as_ref().filter(|_| entries > 0) {
                    let (ticks, _) = view[timestamp_at as usize..map_end as usize].as_chunks::<8>();
                    add_span_times(
                        &mut steps.spans,
                        &mut steps.timeline.entries,
                        ticks,
                        t.period,
                    );
                }
            }
            steps.readback.unmap();
        }
        StepOutput {
            bytes: &steps.host,
            spans: &steps.spans,
            timed: &steps.timeline.entries,
        }
    }

    /// Begins a compute pass, timed when profiling with timestamps inside a span.
    ///
    /// # Panics
    ///
    /// If the step has more timed passes than [`StepSizes::timed_passes`].
    fn begin_pass(&mut self) {
        self.close_copies();
        let timed = self
            .steps
            .timestamps
            .as_ref()
            .filter(|_| self.steps.profiling)
            .zip(self.span);
        let max = self.steps.sizes.timed_passes;
        let timeline = &mut self.steps.timeline;
        let writes = timed.map(|(t, span)| {
            let index = timeline.pass(span, max);
            wgpu::ComputePassTimestampWrites {
                query_set: &t.queries,
                beginning_of_pass_write_index: Some(2 * index),
                end_of_pass_write_index: Some(2 * index + 1),
            }
        });
        let label = self.span.map(|s| self.steps.spans[s].0);
        let encoder = self.encoder.as_mut().expect("the step is being recorded");
        let pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label.unwrap_or("cavitation step")),
            timestamp_writes: writes,
        });
        self.pass = Some(pass.forget_lifetime());
    }

    /// Ends the compute pass being recorded, if any, so copies can be recorded.
    fn end_pass(&mut self) {
        self.pass = None;
    }

    /// Ends the compute pass being recorded, if any, before a copy; when profiling with
    /// encoder timestamps inside a span, opens a timed run of copies unless one is open.
    fn begin_copies(&mut self) {
        self.end_pass();
        let Some(span) = self.span else { return };
        let profiling = self.steps.profiling;
        let Some(t) = self
            .steps
            .timestamps
            .as_ref()
            .filter(|t| t.encoder && profiling)
        else {
            return;
        };
        if let Some(query) = self.steps.timeline.open_copies(span) {
            self.encoder
                .as_mut()
                .expect("the step is being recorded")
                .write_timestamp(&t.queries, query);
        }
    }

    /// Closes the open run of copies, if any, with a timestamp after its last copy.
    fn close_copies(&mut self) {
        if let Some(query) = self.steps.timeline.close_copies() {
            let t = self
                .steps
                .timestamps
                .as_ref()
                .expect("a run of copies is opened only with encoder timestamps");
            self.encoder
                .as_mut()
                .expect("the step is being recorded")
                .write_timestamp(&t.queries, query);
        }
    }
}

/// Sets each timed entry's GPU time from the resolved timestamp `ticks` (two per entry, in
/// entry order) and the nanoseconds per tick, from the entry's beginning to its end, and each
/// span's time, the sum of its entries' times. Only the two timestamps of one entry are
/// compared: on Metal, timestamps of passes separated by a copy are not on one timeline (a
/// later pass can end before an earlier one begins). An entry whose end reads before its
/// beginning has no time, and neither has its span: no time is made up as 0. A span with
/// no timed entry keeps `None`.
fn add_span_times(
    spans: &mut [(&'static str, Option<f64>)],
    entries: &mut [Timed],
    ticks: &[[u8; 8]],
    period: f64,
) {
    for (e, entry) in entries.iter_mut().enumerate() {
        let begin = u64::from_le_bytes(ticks[2 * e]);
        let end = u64::from_le_bytes(ticks[2 * e + 1]);
        entry.nanoseconds = (end >= begin).then(|| (end - begin) as f64 * period);
    }
    for (s, span) in spans.iter_mut().enumerate() {
        let mut mine = entries.iter().filter(|e| e.span == s).peekable();
        span.1 = if mine.peek().is_some() {
            mine.map(|e| e.nanoseconds).sum()
        } else {
            None
        };
    }
}

/// What a finished [`Step`] gives back: the bytes read back and the spans with their GPU
/// times. It borrows the [`StepEncoder`], whose buffers it reads, until it is dropped.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let mut step = steps.begin();
///     step.span("all");
///     let output = step.finish();
///     let (name, nanoseconds) = output.spans()[0];
///     assert_eq!(name, "all");
///     assert!(nanoseconds.is_none()); // not profiling
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct StepOutput<'a> {
    /// The bytes read back, in the order of the step's `read_back` calls.
    bytes: &'a [u8],
    /// The spans in order, with their GPU nanoseconds when the step was profiled with
    /// timestamps.
    spans: &'a [(&'static str, Option<f64>)],
    /// The timed passes and copy runs in order, when the step was profiled with timestamps.
    timed: &'a [Timed],
}

impl<'a> StepOutput<'a> {
    /// The bytes read back: each [`Step::read_back`] call's at the offset it returned. The
    /// padding between parts holds no data.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     assert!(steps.begin().finish().bytes().is_empty());
    /// }
    /// ```
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// The step's spans in order, each with its GPU time in nanoseconds when the step was
    /// profiled and the device has timestamps: the sum of its passes' times, and of its copy
    /// runs' where [`StepEncoder::timing`] is [`Timing::PassesAndCopies`]. `None` when it was
    /// not timed, or when one of its entries' timestamps read backwards.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     steps.set_profiling(true);
    ///     let mut step = steps.begin();
    ///     step.span("idle");
    ///     let output = step.finish();
    ///     if gpu.info().timestamps {
    ///         assert!(output.spans()[0].1.is_some());
    ///     }
    /// }
    /// ```
    pub fn spans(&self) -> &'a [(&'static str, Option<f64>)] {
        self.spans
    }

    /// The step's timed passes, and copy runs where the device has encoder timestamps, in
    /// the order recorded, each with its span and its own GPU time. Empty when the step was
    /// not profiled or nothing could be timed ([`StepEncoder::timing`] says why).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, StepEncoder, StepSizes, TimedKind, Timing};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     steps.set_profiling(true);
    ///     let timing = steps.timing();
    ///     let mut step = steps.begin();
    ///     step.span("idle");
    ///     let output = step.finish();
    ///     if !matches!(timing, Timing::Untimed { .. }) {
    ///         assert_eq!(output.timed().len(), 1);
    ///         assert_eq!(output.timed()[0].kind, TimedKind::Pass);
    ///     }
    /// }
    /// ```
    pub fn timed(&self) -> &'a [Timed] {
        self.timed
    }
}

/// `value` rounded up to a multiple of `align`.
fn round_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

/// The grid for `workgroups` workgroups: one row up to [`MAX_WORKGROUPS_PER_DIMENSION`],
/// otherwise full rows of that many.
///
/// # Panics
///
/// If `workgroups` is 0 or needs more rows than one dimension allows.
fn grid(workgroups: u32) -> (u32, u32) {
    assert!(workgroups > 0, "a dispatch needs at least one workgroup");
    if workgroups <= MAX_WORKGROUPS_PER_DIMENSION {
        return (workgroups, 1);
    }
    let rows = workgroups.div_ceil(MAX_WORKGROUPS_PER_DIMENSION);
    assert!(
        rows <= MAX_WORKGROUPS_PER_DIMENSION,
        "{workgroups} workgroups do not fit in two dimensions"
    );
    (MAX_WORKGROUPS_PER_DIMENSION, rows)
}

#[cfg(test)]
mod tests {
    //! The grid split, the rounding and the span times; the encoder itself is tested on a
    //! GPU in `tests/step.rs` and `tests/step_alloc.rs`.

    use super::*;

    /// Up to the limit a dispatch is one row; above, full rows of the limit, enough to hold
    /// every workgroup, and never a row too many.
    #[test]
    fn grids() {
        const M: u32 = MAX_WORKGROUPS_PER_DIMENSION;
        for (n, want) in [
            (1, (1, 1)),
            (M, (M, 1)),
            (M + 1, (M, 2)),
            (78_125, (M, 2)),
            (2 * M, (M, 2)),
            (2 * M + 1, (M, 3)),
            (u32::MAX / 2, (M, 32_769)),
        ] {
            let (x, y) = grid(n);
            assert_eq!((x, y), want, "{n}");
            assert!(u64::from(x) * u64::from(y) >= u64::from(n));
            assert!(u64::from(x) * u64::from(y - 1) < u64::from(n));
        }
    }

    /// No workgroups is refused.
    #[test]
    #[should_panic(expected = "at least one workgroup")]
    fn empty_grid() {
        grid(0);
    }

    /// Rounding up to 8 bytes.
    #[test]
    fn rounding() {
        assert_eq!(round_up(0, 8), 0);
        assert_eq!(round_up(4, 8), 8);
        assert_eq!(round_up(8, 8), 8);
        assert_eq!(round_up(12, 8), 16);
    }

    /// The ticks of `pairs` (begin, end) as resolved timestamps.
    fn ticks(pairs: &[(u64, u64)]) -> Vec<[u8; 8]> {
        pairs
            .iter()
            .flat_map(|&(b, e)| [b.to_le_bytes(), e.to_le_bytes()])
            .collect()
    }

    /// Entries for `spans` and `kinds`, not yet timed.
    fn entries(of: &[(usize, TimedKind)]) -> Vec<Timed> {
        of.iter()
            .map(|&(span, kind)| Timed {
                span,
                kind,
                nanoseconds: None,
            })
            .collect()
    }

    /// Each entry's time comes from its own two timestamps, whatever the order of
    /// timestamps across entries, and a span's is the sum of its entries', copy runs
    /// included; a span without an entry has none, and a span with no time between its
    /// beginning and end has 0.
    #[test]
    fn span_times_add_their_entries() {
        use TimedKind::{Copies, Pass};
        let mut spans = vec![("a", None), ("b", None), ("c", None), ("d", None)];
        // a: pass, copies, pass, the second pass timed before the first, as Metal can
        // report across a copy. b: one pass. d: one pass of no measurable time.
        let mut timed = entries(&[(0, Pass), (0, Copies), (0, Pass), (1, Pass), (3, Pass)]);
        let raw = ticks(&[(500, 510), (510, 520), (100, 130), (300, 301), (700, 700)]);
        add_span_times(&mut spans, &mut timed, &raw, 2.0);
        let times: Vec<Option<f64>> = timed.iter().map(|t| t.nanoseconds).collect();
        assert_eq!(
            times,
            [Some(20.0), Some(20.0), Some(60.0), Some(2.0), Some(0.0)]
        );
        assert_eq!(
            spans,
            [
                ("a", Some(100.0)),
                ("b", Some(2.0)),
                ("c", None),
                ("d", Some(0.0))
            ]
        );
    }

    /// An entry whose end reads before its beginning has no time, never a made-up 0, and
    /// its span has none either, whether the backwards entry comes before or after a good
    /// one.
    #[test]
    fn backwards_timestamps_give_none() {
        use TimedKind::{Copies, Pass};
        for raw in [
            ticks(&[(500, 400), (100, 130)]),
            ticks(&[(100, 130), (500, 400)]),
        ] {
            let mut spans = vec![("a", None)];
            let mut timed = entries(&[(0, Pass), (0, Copies)]);
            add_span_times(&mut spans, &mut timed, &raw, 1.0);
            assert_eq!(spans, [("a", None)], "{timed:?}");
            assert_eq!(timed.iter().filter(|t| t.nanoseconds.is_none()).count(), 1);
            assert!(timed.iter().all(|t| t.nanoseconds != Some(0.0)));
        }
    }

    /// The recording order of one profiled step with encoder timestamps: span, upload,
    /// dispatch, two readbacks, a second span with a clear, then finish. Each entry takes
    /// queries 2e and 2e + 1, consecutive copies share one run, every run opened is closed,
    /// and runs never outnumber passes.
    #[test]
    fn timeline_pairs_every_query() {
        use TimedKind::{Copies, Pass};
        let mut line = Timeline::with_capacity(4);
        let mut written = Vec::new();
        // span "a" begins a pass.
        written.extend(line.close_copies());
        let e = line.pass(0, 4);
        written.extend([2 * e, 2 * e + 1]);
        // upload: a run opens.
        written.extend(line.open_copies(0));
        // dispatch: the run closes and a pass begins.
        written.extend(line.close_copies());
        let e = line.pass(0, 4);
        written.extend([2 * e, 2 * e + 1]);
        // two readbacks: one run.
        written.extend(line.open_copies(0));
        assert_eq!(line.open_copies(0), None);
        // span "b": the run closes, a pass begins; a clear opens a run.
        written.extend(line.close_copies());
        let e = line.pass(1, 4);
        written.extend([2 * e, 2 * e + 1]);
        written.extend(line.open_copies(1));
        // finish closes it.
        written.extend(line.close_copies());
        assert_eq!(line.close_copies(), None);
        let kinds: Vec<(usize, TimedKind)> =
            line.entries.iter().map(|t| (t.span, t.kind)).collect();
        assert_eq!(
            kinds,
            [
                (0, Pass),
                (0, Copies),
                (0, Pass),
                (0, Copies),
                (1, Pass),
                (1, Copies)
            ]
        );
        written.sort_unstable();
        let want: Vec<u32> = (0..2 * line.entries.len() as u32).collect();
        assert_eq!(written, want, "every query written once");
        line.clear();
        assert!(line.entries.is_empty() && line.close_copies().is_none());
    }

    /// More timed passes than the encoder was made for are refused, naming the limit.
    #[test]
    #[should_panic(expected = "more than the 1 timed passes")]
    fn timeline_refuses_a_pass_too_many() {
        let mut line = Timeline::with_capacity(1);
        line.pass(0, 1);
        line.pass(0, 1);
    }
}
