//! The forward pass of a [`GpuModel`] in evaluation mode: one list of kernel calls built when
//! the model is compiled, recorded into one step per batch, and the model's answers through
//! it: [`GpuModel::infer_batch`], [`GpuModel::evaluate`] and [`GpuModel::accuracy`].

use cavitation_core::{Error, TensorView};
use cavitation_gpu::kernels::{
    AvgPool2d, BatchNorm2d, BatchNormForward, BatchStats, BiasAdd, BiasAddCall, ChannelDims,
    ConvCall, ConvDims, ConvForward, Elementwise, ElementwiseCall, ElementwiseKernels, Gemm,
    GemmCall, GemmDims, MaxPool2d,
};
use cavitation_gpu::{
    BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Step, StepEncoder, wgpu,
};

use super::model::{Act, CompiledLayer, GpuMask, GpuModel, GpuTensor, Kind};
use crate::train::{Evaluation, measure};
use crate::{Dataset, Loss, Sequential, Targets};

/// The kernels every model shares, built once per model.
#[derive(Debug)]
pub(crate) struct Kernels {
    /// The tiled GEMM, for `Linear`.
    pub(crate) gemm: Gemm,
    /// The bias add, for `Linear`.
    pub(crate) bias: BiasAdd,
    /// The convolution's forward pass.
    pub(crate) conv: ConvForward,
    /// The element-wise operations: the activations and the masks.
    pub(crate) elementwise: ElementwiseKernels,
    /// Batch normalisation.
    pub(crate) batch_norm: BatchNorm2d,
}

impl Kernels {
    /// Builds every shared kernel on `gpu` through `cache`.
    pub(crate) fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<Kernels, GpuError> {
        Ok(Kernels {
            gemm: Gemm::new(gpu, cache)?,
            bias: BiasAdd::new(gpu, cache)?,
            conv: ConvForward::new(gpu, cache)?,
            elementwise: ElementwiseKernels::new(gpu, cache)?,
            batch_norm: BatchNorm2d::new(gpu, cache)?,
        })
    }
}

/// One prepared kernel call of a forward pass.
#[derive(Debug)]
enum Op {
    /// A GEMM.
    Gemm(GemmCall),
    /// A bias added per column.
    Bias(BiasAddCall),
    /// A convolution.
    Conv(ConvCall),
    /// An activation or a mask.
    Elementwise(ElementwiseCall),
    /// Max pooling with its kernel and bind group.
    MaxPool(MaxPool2d, wgpu::BindGroup),
    /// Average pooling with its kernel and bind group.
    AvgPool(AvgPool2d, wgpu::BindGroup),
    /// Batch normalisation, boxed: its prepared call is several times the size of the others.
    BatchNorm(Box<BatchNormForward>),
}

/// A forward pass compiled for the model's batch: its kernel calls in order and the buffers
/// they read and write.
#[derive(Debug)]
pub(crate) struct Forward {
    /// The calls, in order.
    ops: Vec<Op>,
    /// The values: the input first, then each output a call writes.
    values: Vec<PooledBuffer>,
    /// The index in `values` of the model's output.
    output: usize,
    /// For each layer, the index in `values` of its output as the next layer reads it (after
    /// its mask, if it has one).
    #[allow(dead_code)] // The backward pass reads it; so do the tests.
    pub(crate) layer_outputs: Vec<usize>,
    /// Buffers the calls use besides the values: max pooling's winners and batch norm's
    /// statistics, held here as long as the calls that bind them.
    #[allow(dead_code)] // Read only through the bind groups.
    scratch: Vec<PooledBuffer>,
    /// Whether a batch-norm layer normalises with the batch's own statistics (it keeps no
    /// running ones), so that every row of the batch counts.
    batch_statistics: bool,
}

/// What the forward pass is built from: the model's compiled layers and GPU state.
pub(crate) struct Parts<'a> {
    /// The layers.
    pub(crate) layers: &'a [CompiledLayer],
    /// The trained parameters.
    pub(crate) params: &'a [GpuTensor],
    /// The running statistics.
    pub(crate) buffers: &'a [GpuTensor],
    /// The neuron masks.
    pub(crate) masks: &'a [GpuMask],
    /// Rows per batch.
    pub(crate) batch: usize,
    /// The width of one input row.
    pub(crate) input_width: usize,
}

/// The usage of every value buffer: bound as storage, uploaded into and read back.
fn value_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
}

/// `n` as a `u32`, for a size [`GpuModel::compile`] has checked fits a binding.
fn u32_of(n: usize) -> u32 {
    u32::try_from(n).expect("GpuModel::compile checks every size against the GPU's bindings")
}

impl Forward {
    /// Prepares the evaluation-mode forward pass of `parts`: dropout and flatten pass their
    /// input on, batch norm uses its running statistics (or, without them, the batch's), and
    /// each mask is applied to its layer's output.
    pub(crate) fn evaluation(
        parts: &Parts<'_>,
        kernels: &mut Kernels,
        cache: &mut PipelineCache,
        pool: &mut BufferPool,
        steps: &mut StepEncoder,
    ) -> Result<Forward, GpuError> {
        let batch = parts.batch;
        let mut values = vec![pool.take(4 * (batch * parts.input_width) as u64, value_usage())];
        let mut ops = Vec::new();
        let mut scratch = Vec::new();
        let mut batch_statistics = false;
        let mut x = 0;
        let mut layer_outputs = Vec::with_capacity(parts.layers.len());
        for layer in parts.layers {
            let n = batch * layer.outputs;
            let new = |values: &mut Vec<PooledBuffer>, pool: &mut BufferPool| {
                values.push(pool.take(4 * n as u64, value_usage()));
                values.len() - 1
            };
            let out = match layer.kind {
                Kind::Linear {
                    inputs,
                    outputs,
                    weight,
                    bias,
                } => {
                    let out = new(&mut values, pool);
                    let dims = GemmDims {
                        trans_b: true,
                        ..GemmDims::new(u32_of(batch), u32_of(outputs), u32_of(inputs))
                    };
                    let w = &parts.params[weight].buffer;
                    ops.push(Op::Gemm(kernels.gemm.call(
                        steps,
                        pool,
                        dims,
                        &values[x],
                        w,
                        &values[out],
                    )));
                    if let Some(b) = bias {
                        let dims = ChannelDims::rows(u32_of(batch), u32_of(outputs));
                        let b = &parts.params[b].buffer;
                        ops.push(Op::Bias(kernels.bias.call(
                            steps,
                            pool,
                            dims,
                            b,
                            &values[out],
                        )));
                    }
                    out
                }
                Kind::Conv2d { geom, weight, bias } => {
                    let out = new(&mut values, pool);
                    let pair = |(a, b): (usize, usize)| (u32_of(a), u32_of(b));
                    let dims = ConvDims {
                        stride: pair(geom.stride()),
                        padding: pair(geom.padding()),
                        dilation: pair(geom.dilation()),
                        ..ConvDims::new(
                            u32_of(batch),
                            u32_of(geom.in_channels()),
                            u32_of(geom.out_channels()),
                            pair(geom.input()),
                            pair(geom.kernel()),
                        )
                    };
                    let w = &parts.params[weight].buffer;
                    let b = bias.map(|b| &parts.params[b].buffer);
                    ops.push(Op::Conv(kernels.conv.call(
                        steps,
                        pool,
                        dims,
                        &values[x],
                        w,
                        b,
                        &values[out],
                    )));
                    out
                }
                Kind::Activation(act) => {
                    let out = new(&mut values, pool);
                    let (op, alpha) = match act {
                        Act::Relu => (Elementwise::Relu, 0.0),
                        Act::LeakyRelu(slope) => (Elementwise::LeakyRelu, slope as f32),
                        Act::Tanh => (Elementwise::Tanh, 0.0),
                        Act::Sigmoid => (Elementwise::Sigmoid, 0.0),
                        Act::GeluTanh => (Elementwise::GeluTanh, 0.0),
                        Act::GeluErf => (Elementwise::GeluErf, 0.0),
                    };
                    let call = kernels.elementwise.call(
                        steps,
                        pool,
                        op,
                        u32_of(n),
                        alpha,
                        0.0,
                        &values[x],
                        None,
                        &values[out],
                    );
                    ops.push(Op::Elementwise(call));
                    out
                }
                Kind::MaxPool2d(geom) => {
                    let out = new(&mut values, pool);
                    let kernel = MaxPool2d::new(cache, &geom, batch)?;
                    scratch.push(pool.take(4 * n as u64, wgpu::BufferUsages::STORAGE));
                    let winners = scratch.last().expect("just pushed");
                    let group = kernel.forward_bind_group(steps, &values[x], &values[out], winners);
                    ops.push(Op::MaxPool(kernel, group));
                    out
                }
                Kind::AvgPool2d {
                    geom,
                    count_include_pad,
                    divisor_override,
                } => {
                    let out = new(&mut values, pool);
                    let kernel =
                        AvgPool2d::new(cache, &geom, batch, count_include_pad, divisor_override)?;
                    let group = kernel.forward_bind_group(steps, &values[x], &values[out]);
                    ops.push(Op::AvgPool(kernel, group));
                    out
                }
                Kind::BatchNorm2d(norm) => {
                    let out = new(&mut values, pool);
                    let c = norm.channels;
                    scratch.push(pool.take(4 * 3 * c as u64, wgpu::BufferUsages::STORAGE));
                    let stats = scratch.last().expect("just pushed");
                    let from = match norm.running {
                        Some(r) => BatchStats::Running {
                            mean: &parts.buffers[r.mean].buffer,
                            var: &parts.buffers[r.var].buffer,
                        },
                        None => {
                            batch_statistics = true;
                            BatchStats::Batch
                        }
                    };
                    let affine = match (norm.weight, norm.bias) {
                        (Some(w), Some(b)) => {
                            Some((&parts.params[w].buffer, &parts.params[b].buffer))
                        }
                        _ => None,
                    };
                    let dims = ChannelDims::new(u32_of(batch), u32_of(c), u32_of(norm.spatial));
                    let call = kernels.batch_norm.forward_call(
                        steps,
                        pool,
                        dims,
                        norm.eps,
                        from,
                        &values[x],
                        affine,
                        &values[out],
                        stats,
                    );
                    ops.push(Op::BatchNorm(Box::new(call)));
                    out
                }
                Kind::Dropout(_) | Kind::Identity => x,
            };
            x = match layer.mask {
                Some(m) => {
                    let masked = new(&mut values, pool);
                    let call = kernels.elementwise.call(
                        steps,
                        pool,
                        Elementwise::KeepWhere,
                        u32_of(n),
                        0.0,
                        0.0,
                        &values[out],
                        Some(&parts.masks[m].bits),
                        &values[masked],
                    );
                    ops.push(Op::Elementwise(call));
                    masked
                }
                None => out,
            };
            layer_outputs.push(x);
        }
        // The GEMM's wide tile, for the calls of this pass that use it: its pipelines are
        // built only for a model that has such a call.
        for op in &ops {
            match op {
                Op::Gemm(call) => {
                    kernels.gemm.ensure_wide(cache, call)?;
                }
                Op::Conv(call) => {
                    kernels.conv.ensure_wide(cache, call)?;
                }
                _ => {}
            }
        }
        Ok(Forward {
            ops,
            values,
            output: x,
            layer_outputs,
            scratch,
            batch_statistics,
        })
    }

    /// Records every call in `step`, in order.
    fn record(&self, kernels: &Kernels, step: &mut Step<'_>) {
        for op in &self.ops {
            match op {
                Op::Gemm(call) => kernels.gemm.record(step, call),
                Op::Bias(call) => kernels.bias.record(step, call),
                Op::Conv(call) => kernels.conv.record(step, call),
                Op::Elementwise(call) => kernels.elementwise.record(step, call),
                Op::MaxPool(kernel, group) => kernel.forward(step, group),
                Op::AvgPool(kernel, group) => kernel.forward(step, group),
                Op::BatchNorm(call) => kernels.batch_norm.record_forward(step, call),
            }
        }
    }
}

impl GpuModel {
    /// Runs a batch of inputs, shape `[rows, input_width]` with `rows` from 1 to
    /// [`batch`](Self::batch), through the model in evaluation mode, and returns the
    /// outputs, shape `[rows, output_width]` (in the model's own buffer, valid until the next
    /// call). Evaluation mode is the CPU tape's: dropout passes its input on, and batch norm
    /// uses its running statistics, or the batch's when it keeps none. A batch is one GPU
    /// step: one upload, the layers, one readback.
    ///
    /// Each row's outputs depend on that row alone, so fewer rows than the compiled batch run
    /// as a full batch whose other rows are not read. The exception is a batch-norm layer
    /// that keeps no running statistics: it normalises over every row, so it needs a full
    /// batch.
    ///
    /// # Panics
    ///
    /// If `inputs` is not rank 2 with [`input_width`](Self::input_width) columns and 1 to
    /// [`batch`](Self::batch) rows, or has fewer rows than the batch while a batch-norm layer
    /// uses the batch's statistics; the message names the shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{GpuModel, Linear, Relu, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 3, 4, &mut Rng::seed(0)))
    ///     .push(Relu);
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut model = GpuModel::compile(&net, &params, 8, &gpu).unwrap();
    ///     let x = Tensor::full(&[5, 3], 0.5f32);
    ///     let y = model.infer_batch(x.view());
    ///     assert_eq!(y.shape().dims(), &[5, 4]);
    ///     assert!(y.data().iter().all(|&v| v >= 0.0));
    /// }
    /// ```
    pub fn infer_batch(&mut self, inputs: TensorView<'_, f32>) -> TensorView<'_, f32> {
        let dims = inputs.shape().dims();
        let rows = match *dims {
            [rows, width] if width == self.input_width && (1..=self.batch).contains(&rows) => rows,
            _ => panic!(
                "GpuModel::infer_batch: the inputs must be [1..={}, {}], got {dims:?}",
                self.batch, self.input_width
            ),
        };
        assert!(
            rows == self.batch || !self.eval.batch_statistics,
            "GpuModel::infer_batch: a batch-norm layer of this model normalises with the \
             batch's statistics, so it needs all {} rows, got {dims:?}",
            self.batch
        );
        let out_len = rows * self.output_width;
        let bytes: Vec<u8> = inputs.data().iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut step = self.steps.begin();
        step.upload(self.eval.values[0].buffer(), 0, &bytes);
        self.eval.record(&self.kernels, &mut step);
        let at = step.read_back(
            self.eval.values[self.eval.output].buffer(),
            0,
            4 * out_len as u64,
        );
        let output = step.finish();
        let (words, _) = output.bytes()[at..at + 4 * out_len].as_chunks::<4>();
        self.host.clear();
        self.host
            .extend(words.iter().map(|b| f32::from_le_bytes(*b)));
        TensorView::from_slice(&self.host, &[rows, self.output_width])
            .expect("the outputs hold rows × output_width values")
    }

    /// Measures the model on `data` in evaluation mode, [`batch`](Self::batch) rows at a
    /// time: the mean `loss`, and the accuracy for class targets or R² for value targets, as
    /// [`evaluate`](crate::evaluate) measures them on the CPU. `model` is the
    /// [`Sequential`] this model was compiled from, for the checks on the data.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] as [`evaluate`](crate::evaluate) returns it: the data does not
    /// suit the model or the loss (the message says how), its rows are not as wide as the
    /// model's input, or a batch-norm layer that keeps no running statistics would see a
    /// last batch shorter than [`batch`](Self::batch).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{Dataset, GpuModel, Linear, Loss, Sequential, Targets};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
    /// let data = Dataset { x: Tensor::zeros(&[5, 2]), targets: Targets::Classes(vec![0, 1, 0, 1, 0]) };
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut model = GpuModel::compile(&net, &params, 2, &gpu).unwrap();
    ///     let e = model.evaluate(&net, &data, Loss::CrossEntropy).unwrap();
    ///     // Zero inputs and a zero bias give equal outputs, so every row predicts class 0.
    ///     assert_eq!(e.accuracy, Some(0.6));
    /// }
    /// ```
    pub fn evaluate(
        &mut self,
        model: &Sequential,
        data: &Dataset,
        loss: Loss,
    ) -> Result<Evaluation, Error> {
        self.check_rows(data)?;
        let width = self.output_width;
        measure(model, data, loss, self.batch, width, |x, out| {
            out.extend_from_slice(self.infer_batch(x).data());
        })
    }

    /// The share of the rows of `data` whose first largest output (NaN ignored) is at the
    /// row's label, in evaluation mode, as [`accuracy`](crate::accuracy) measures it on the
    /// CPU.
    ///
    /// # Panics
    ///
    /// As [`accuracy`](crate::accuracy) does: `data` has value targets, no rows, not one
    /// label per row, or a label not below the model's output width. Also if its rows are not
    /// as wide as the model's input, or a batch-norm layer that keeps no running statistics
    /// would see a short last batch.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{Dataset, GpuModel, Linear, Sequential, Targets};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
    /// let data = Dataset { x: Tensor::zeros(&[4, 2]), targets: Targets::Classes(vec![0, 1, 0, 1]) };
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut model = GpuModel::compile(&net, &params, 3, &gpu).unwrap();
    ///     assert_eq!(model.accuracy(&data), 0.5);
    /// }
    /// ```
    pub fn accuracy(&mut self, data: &Dataset) -> f64 {
        let Targets::Classes(labels) = &data.targets else {
            panic!("GpuModel::accuracy: the data has value targets; accuracy needs class labels");
        };
        let (rows, features) = (data.len(), data.features());
        assert!(rows > 0, "GpuModel::accuracy: the data has no rows");
        assert_eq!(
            labels.len(),
            rows,
            "GpuModel::accuracy: the data has {rows} rows but {} labels",
            labels.len()
        );
        if let Err(e) = self.check_rows(data) {
            panic!("GpuModel::accuracy: {e}");
        }
        let classes = self.output_width;
        let mut correct = 0usize;
        for start in (0..rows).step_by(self.batch) {
            let end = (start + self.batch).min(rows);
            let x = TensorView::from_slice(
                &data.x.data()[start * features..end * features],
                &[end - start, features],
            )
            .expect("a run of whole rows of x has the batch's shape");
            let scores = self.infer_batch(x);
            for (i, (row, &label)) in scores
                .data()
                .chunks(classes)
                .zip(&labels[start..end])
                .enumerate()
            {
                assert!(
                    label < classes,
                    "GpuModel::accuracy: the label of row {} is {label}, but the model has \
                     {classes} outputs (classes 0..{classes})",
                    start + i
                );
                if crate::train::argmax(row) == Some(label) {
                    correct += 1;
                }
            }
        }
        correct as f64 / rows as f64
    }

    /// Runs one full batch `x` through the evaluation pass, as [`infer_batch`](Self::infer_batch)
    /// does, and reads back every layer's output: for the tests, which check each layer on
    /// its own input. Uses an encoder of its own, sized for the readback.
    #[cfg(test)]
    pub(crate) fn infer_layers(&mut self, x: &[f32]) -> Vec<Vec<f32>> {
        use cavitation_gpu::StepSizes;
        assert_eq!(x.len(), self.batch * self.input_width);
        let lens: Vec<usize> = self.layers.iter().map(|l| self.batch * l.outputs).collect();
        let total: usize = lens.iter().map(|n| 4 * n + 8).sum();
        let sizes = StepSizes {
            upload_bytes: 4 * x.len() as u64,
            readback_bytes: total as u64,
            ..StepSizes::default()
        };
        let mut steps = StepEncoder::new(&self.gpu, sizes);
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut step = steps.begin();
        step.upload(self.eval.values[0].buffer(), 0, &bytes);
        self.eval.record(&self.kernels, &mut step);
        let at: Vec<usize> = (self.eval.layer_outputs.iter().zip(&lens))
            .map(|(&v, &n)| step.read_back(self.eval.values[v].buffer(), 0, 4 * n as u64))
            .collect();
        let output = step.finish();
        (at.iter().zip(&lens))
            .map(|(&at, &n)| {
                let (words, _) = output.bytes()[at..at + 4 * n].as_chunks::<4>();
                words.iter().map(|b| f32::from_le_bytes(*b)).collect()
            })
            .collect()
    }

    /// Returns [`Error::InvalidData`] if the rows of `data` are not as wide as the model's
    /// input, or the last batch would be short while a batch-norm layer uses the batch's
    /// statistics.
    pub(crate) fn check_rows(&self, data: &Dataset) -> Result<(), Error> {
        let dims = data.x.shape().dims();
        if dims.len() != 2 || dims[1] != self.input_width {
            return Err(Error::InvalidData(format!(
                "the inputs are {dims:?}, but the GPU model takes rows of {} values",
                self.input_width
            )));
        }
        if self.eval.batch_statistics && !dims[0].is_multiple_of(self.batch) {
            return Err(Error::InvalidData(format!(
                "a batch-norm layer of the GPU model normalises with the batch's statistics, so \
                 the {} rows must be whole batches of {}",
                dims[0], self.batch
            )));
        }
        Ok(())
    }
}
