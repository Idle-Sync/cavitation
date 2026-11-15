//! The step encoder: everything one training or inference step sends to the GPU, recorded
//! in one command encoder and submitted once. Uploads go through a mapped staging buffer,
//! results come back through one readback buffer, dispatches can be split over two
//! dimensions, and when profiling each named span is timed on the GPU.

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
    /// that span (a copy ends the pass it interrupts).
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
    /// Two timestamps per timed pass: its beginning and end.
    queries: wgpu::QuerySet,
    /// Where the queries resolve (`QUERY_RESOLVE | COPY_SRC`), copied into the readback.
    resolve: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period: f64,
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
/// ends, and [`StepOutput::spans`] gives each span's GPU time, from the beginning of its first pass to the end of its last. A
/// profiled step has a pass boundary per span, so its timings describe a step structured
/// for profiling. Without profiling, dispatches share passes and spans only label them.
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
    /// The span timestamps, when the device has `TIMESTAMP_QUERY`.
    timestamps: Option<Timestamps>,
    /// Whether steps are profiled.
    profiling: bool,
    /// The bytes of the last step's readback, reused from step to step.
    host: Vec<u8>,
    /// The last step's spans: name and GPU nanoseconds, reused from step to step.
    spans: Vec<(&'static str, Option<f64>)>,
    /// For each timed pass of the last step, the span it belongs to.
    passes: Vec<usize>,
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
    /// largest buffer.
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
        let timestamp_bytes = u64::from(sizes.timed_passes) * 16;
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
                    count: 2 * sizes.timed_passes,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cavitation timestamp resolve"),
                    size: timestamp_bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                period: f64::from(gpu.queue().get_timestamp_period()),
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
            passes: Vec::with_capacity(sizes.timed_passes as usize),
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
        self.passes.clear();
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
        self.end_pass();
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
        self.end_pass();
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
        self.end_pass();
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
        let Step {
            steps,
            encoder,
            read_end: user_bytes,
            ..
        } = self;
        let mut encoder = encoder.expect("the step is being recorded");
        let passes = steps.passes.len() as u64;
        let timestamp_at = round_up(steps.sizes.readback_bytes, ALIGN);
        if let Some(t) = steps.timestamps.as_ref().filter(|_| passes > 0) {
            encoder.resolve_query_set(&t.queries, 0..(2 * passes) as u32, &t.resolve, 0);
            encoder.copy_buffer_to_buffer(
                &t.resolve,
                0,
                &steps.readback,
                timestamp_at,
                16 * passes,
            );
        }
        steps.staging_mapped.store(false, Ordering::Release);
        steps.staging.unmap();
        steps.gpu.queue().submit([encoder.finish()]);
        steps.submitted += 1;
        let flag = Arc::clone(&steps.staging_mapped);
        steps
            .staging
            .map_async(wgpu::MapMode::Write, .., move |result| {
                result.expect("map the staging buffer for writing");
                flag.store(true, Ordering::Release);
            });
        let map_end = if passes > 0 {
            timestamp_at + 16 * passes
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
                if let Some(t) = steps.timestamps.as_ref().filter(|_| passes > 0) {
                    let (ticks, _) = view[timestamp_at as usize..map_end as usize].as_chunks::<8>();
                    add_span_times(&mut steps.spans, &steps.passes, ticks, t.period);
                }
            }
            steps.readback.unmap();
        }
        StepOutput {
            bytes: &steps.host,
            spans: &steps.spans,
        }
    }

    /// Begins a compute pass, timed when profiling with timestamps inside a span.
    ///
    /// # Panics
    ///
    /// If the step has more timed passes than [`StepSizes::timed_passes`].
    fn begin_pass(&mut self) {
        let timed = self
            .steps
            .timestamps
            .as_ref()
            .filter(|_| self.steps.profiling)
            .zip(self.span);
        let writes = timed.map(|(t, span)| {
            let index = self.steps.passes.len() as u32;
            assert!(
                index < self.steps.sizes.timed_passes,
                "this profiled step has more than the {} timed passes its encoder was made for",
                self.steps.sizes.timed_passes
            );
            self.steps.passes.push(span);
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
}

/// Sets each span's GPU time from the resolved timestamp `ticks` (two per pass, in pass
/// order) and the nanoseconds per tick: from the beginning of its first timed pass to the
/// end of its last, so the commands between its passes (an upload inside the span) count
/// too, and the result does not depend on how a backend times work between passes. A span's
/// passes are consecutive, since spans are recorded one after another. A span with no timed
/// pass, or with a pass whose end tick is before its beginning, keeps `None`.
fn add_span_times(
    spans: &mut [(&'static str, Option<f64>)],
    passes: &[usize],
    ticks: &[[u8; 8]],
    period: f64,
) {
    let tick = |i: usize| u64::from_le_bytes(ticks[i]);
    let mut first = 0;
    while first < passes.len() {
        let span = passes[first];
        let mut last = first;
        let mut ordered = tick(2 * first + 1) >= tick(2 * first);
        while last + 1 < passes.len() && passes[last + 1] == span {
            last += 1;
            ordered &= tick(2 * last + 1) >= tick(2 * last);
        }
        let (begin, end) = (tick(2 * first), tick(2 * last + 1));
        spans[span].1 = (ordered && end >= begin).then(|| (end - begin) as f64 * period);
        first = last + 1;
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

    /// The step's spans in order, each with its GPU time in nanoseconds (from its first pass's
    /// beginning to its last pass's end) when the step was profiled and the device has timestamps, `None`
    /// otherwise.
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

    /// A span's time runs from its first pass's beginning to its last pass's end, the gap
    /// between its passes included; a span without a pass, or with a pass whose ticks run
    /// backwards, has none.
    #[test]
    fn span_times_cover_all_their_passes() {
        let mut spans = vec![("a", None), ("b", None), ("c", None), ("d", None)];
        // Passes: a, a, b, d; d's ticks run backwards.
        let passes = [0, 0, 1, 3];
        let raw: [u64; 8] = [100, 110, 200, 230, 300, 301, 500, 400];
        let ticks: Vec<[u8; 8]> = raw.iter().map(|t| t.to_le_bytes()).collect();
        add_span_times(&mut spans, &passes, &ticks, 2.0);
        assert_eq!(
            spans,
            [
                ("a", Some(260.0)),
                ("b", Some(2.0)),
                ("c", None),
                ("d", None)
            ]
        );
    }
}
