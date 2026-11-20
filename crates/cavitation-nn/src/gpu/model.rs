//! `GpuModel`: a [`Sequential`] compiled for the GPU at one batch size. Compiling checks
//! every layer against the forms the GPU runs, refusing the others by index and name, and
//! uploads the parameters, the batch-norm running statistics and the neuron masks.

use cavitation_core::{Error, ParamId, ParamStore};
use cavitation_gpu::{BufferPool, Gpu, PipelineCache, PooledBuffer, StepEncoder, StepSizes, wgpu};

use super::forward::{Forward, Kernels, Parts};

use crate::Sequential;
use crate::plan::Plan;
#[cfg(test)]
use crate::plan::pack_rows;
pub(crate) use crate::plan::{Act, CompiledLayer, Kind};

/// A parameter or buffer held on the GPU under the id it has in the [`ParamStore`].
#[derive(Debug)]
pub(crate) struct GpuTensor {
    /// The id in the store.
    pub(crate) id: ParamId,
    /// The number of `f32` values.
    pub(crate) len: usize,
    /// The values on the GPU, `4 · len` bytes.
    pub(crate) buffer: PooledBuffer,
}

/// A neuron mask compiled for the batch: one bit per value of the layer's output, set where
/// the value is kept.
#[derive(Debug)]
#[allow(dead_code)] // `layer` is read by the tests and the backward pass.
pub(crate) struct GpuMask {
    /// The [`Sequential`] layer whose output it masks.
    pub(crate) layer: usize,
    /// The bits, `ceil(batch · width / 32)` words: bit `k % 32` of word `k / 32` for value
    /// `k` of the batch, row after row.
    pub(crate) bits: PooledBuffer,
}

/// A [`Sequential`] compiled for the GPU, for batches of a fixed number of rows: its layers
/// as the GPU runs them, with its parameters, batch-norm running statistics and neuron
/// masks held on the GPU.
///
/// [`compile`](GpuModel::compile) refuses a model with a layer the GPU does not run, naming the
/// layer's index and name. The layers it runs are [`Linear`](crate::Linear),
/// [`Conv2d`](crate::Conv2d), [`MaxPool2d`](crate::MaxPool2d), [`AvgPool2d`](crate::AvgPool2d),
/// [`AdaptiveAvgPool2d`](crate::AdaptiveAvgPool2d),
/// [`BatchNorm2d`](crate::BatchNorm2d), [`Dropout`](crate::Dropout), [`Flatten`](crate::Flatten),
/// [`Identity`](crate::Identity) and every built-in activation: [`Relu`](crate::Relu),
/// [`LeakyRelu`](crate::LeakyRelu), [`Tanh`](crate::Tanh), [`Sigmoid`](crate::Sigmoid),
/// [`Gelu`](crate::Gelu) (both forms), [`Elu`](crate::Elu), [`Selu`](crate::Selu),
/// [`Celu`](crate::Celu) (with a positive `alpha`), [`Silu`](crate::Silu), [`Mish`](crate::Mish),
/// [`Softplus`](crate::Softplus), [`Softsign`](crate::Softsign), [`Hardtanh`](crate::Hardtanh)
/// (with `min <= max`), [`Relu6`](crate::Relu6), [`Hardsigmoid`](crate::Hardsigmoid),
/// [`Hardswish`](crate::Hardswish) and [`LogSigmoid`](crate::LogSigmoid), each as an
/// [`ActivationLayer`](crate::ActivationLayer) too, [`Softmax`](crate::Softmax) and
/// [`LogSoftmax`](crate::LogSoftmax) over each row, [`Prelu`](crate::Prelu) with one slope per
/// column or one shared slope (at most 65,535), its slopes trained as parameters, and
/// [`Glu`](crate::Glu) with every gate, over an even width. Neuron masks set on the model when
/// it is compiled are compiled in; a mask set later needs a new compile.
///
/// A batch-norm layer's `num_batches_tracked` stays on the host, since only the host
/// decides whether a step's running-statistics update ran.
///
/// # Examples
///
/// Without a GPU, [`Gpu::new`] returns `None` and the example does nothing:
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_gpu::Gpu;
/// use cavitation_nn::{GpuModel, Linear, Relu, Sequential};
///
/// let mut params = ParamStore::new();
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 4, 8, &mut Rng::seed(0)))
///     .push(Relu)
///     .push(Linear::new(&mut params, 8, 2, &mut Rng::seed(1)));
/// if let Some(gpu) = Gpu::new() {
///     let model = GpuModel::compile(&net, &params, 32, &gpu).unwrap();
///     assert_eq!((model.batch(), model.input_width(), model.output_width()), (32, 4, 2));
/// }
/// ```
#[derive(Debug)]
pub struct GpuModel {
    /// The GPU the model is on.
    pub(super) gpu: Gpu,
    /// The pool every buffer of the model comes from.
    #[allow(dead_code)] // The trainer takes its buffers from it.
    pub(super) pool: BufferPool,
    /// The pipelines of the model's kernels.
    #[allow(dead_code)] // The trainer builds its kernels through it.
    pub(super) cache: PipelineCache,
    /// The kernels every layer shares.
    pub(super) kernels: Kernels,
    /// The encoder every step of the model is recorded in.
    pub(super) steps: StepEncoder,
    /// Rows per batch.
    pub(super) batch: usize,
    /// The width of one input row.
    pub(super) input_width: usize,
    /// The width of one output row.
    pub(super) output_width: usize,
    /// The layers, one per layer of the [`Sequential`].
    pub(super) layers: Vec<CompiledLayer>,
    /// The trained parameters.
    pub(super) params: Vec<GpuTensor>,
    /// The batch-norm running statistics.
    pub(super) buffers: Vec<GpuTensor>,
    /// Each batch-norm batch count, with its value, kept on the host.
    pub(super) counts: Vec<(ParamId, f32)>,
    /// The neuron masks, by increasing layer.
    #[allow(dead_code)] // The trainer's backward pass binds them.
    pub(super) masks: Vec<GpuMask>,
    /// The forward pass in evaluation mode.
    pub(super) eval: Forward,
    /// The last batch's outputs, read back.
    pub(super) host: Vec<f32>,
}

impl GpuModel {
    /// Compiles `model`, with its values in `params`, for batches of `batch` rows on `gpu`,
    /// and uploads its parameters, batch-norm running statistics and neuron masks.
    ///
    /// # Errors
    ///
    /// - [`Error::Unsupported`] if the model has no layers, or a layer is not one the GPU
    ///   runs (see [`GpuModel`]) or has parameters its GPU form does not use; the message
    ///   names the layer's index and name. Also if no layer fixes the input width, or a
    ///   parameter or a batch of a layer's inputs or outputs is larger than the GPU binds,
    ///   the model has a convolution and `batch` is above 65,535 or a softmax and `batch` is
    ///   above 16,776,960, a PReLU has more than 65,535 slopes, takes more than 2³¹ values
    ///   per batch or its slope gradient would sum more than 268,431,360 values per slope, a
    ///   GLU takes more than 4,294,967,295 values per batch, an adaptive average pooling
    ///   reads or writes more than 2³¹ values per batch or has an axis whose
    ///   `(size + 1) · (out + 1)` is above 4,294,967,295, a `Linear`'s batch, inputs or
    ///   outputs or a convolution's output pixels or channels are above 4,194,240, a batch
    ///   norm's eps is below 2⁻¹²⁶ in `f32` (and not 0), or the parameters, running
    ///   statistics and masks together are larger than one GPU buffer.
    /// - [`Error::InvalidData`] if `batch` is 0, a layer's input width is not the width the
    ///   layer before it outputs, a layer outputs width 0 or another width than it reports,
    ///   a GLU's input width is odd (the CPU's GLU panics on it), a mask is not as wide as its
    ///   layer's output, a batch count does not hold one value,
    ///   a PReLU's weight does not hold one value per slope, a batch norm that keeps no
    ///   running statistics would see one value per channel, or a layer's inference form is
    ///   an average pooling that is dilated or divides by 0 (`AvgPool2d::with_options`
    ///   refuses both with `InvalidData`, `AvgPool2d::new` panics on a dilated geometry, and
    ///   `InferenceModel::compile` panics on such a form), or an adaptive average pooling with
    ///   a size of 0 (as `AdaptiveAvgPool2d::new` refuses it); the message names the layer
    ///   (and both widths, where two differ).
    ///
    /// # Panics
    ///
    /// If an id of the model's layers is not in `params`.
    ///
    /// # Examples
    ///
    /// A layer the GPU does not run, such as a layer of your own, is refused by index and
    /// name:
    ///
    /// ```
    /// use cavitation_core::{ParamId, ParamStore, Real, Rng, Tape, Var};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{GpuModel, Layer, Linear, Sequential};
    ///
    /// /// `x²`, a layer of your own: it has no inference form, so no GPU runs it.
    /// struct Square;
    ///
    /// impl<T: Real> Layer<T> for Square {
    ///     fn name(&self) -> &str {
    ///         "square"
    ///     }
    ///     fn output_width(&self, input_width: usize) -> usize {
    ///         input_width
    ///     }
    ///     fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
    ///         tape.mul(x, x)
    ///     }
    ///     fn parameters(&self) -> Vec<(&'static str, ParamId)> {
    ///         Vec::new()
    ///     }
    /// }
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 4, 4, &mut Rng::seed(0)))
    ///     .push(Square);
    /// if let Some(gpu) = Gpu::new() {
    ///     let err = GpuModel::compile(&net, &params, 8, &gpu).unwrap_err();
    ///     assert!(err.to_string().contains("layer 1 (square)"), "{err}");
    /// }
    /// ```
    pub fn compile(
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
        batch: usize,
        gpu: &Gpu,
    ) -> Result<GpuModel, Error> {
        let plan = Plan::build(model, params, batch)?;
        let limits = &gpu.info().limits;
        let largest = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        let bytes = |values: usize| 4 * values as u64;
        for (i, layer) in plan.layers.iter().enumerate() {
            let rows = (batch * layer.inputs).max(batch * layer.outputs);
            if bytes(rows) > largest {
                return Err(Error::Unsupported(format!(
                    "GpuModel::compile: layer {i} ({}) reads or writes {} bytes per batch of \
                     {batch}, more than the {largest} this GPU binds",
                    model.layer(i).name(),
                    bytes(rows)
                )));
            }
        }
        let values = &params.values;
        for i in 0..model.len() {
            let layer = model.layer(i);
            for (local, id) in layer.parameters().into_iter().chain(layer.buffers()) {
                let len = values.get(id).data().len();
                if bytes(len) > largest {
                    return Err(Error::Unsupported(format!(
                        "GpuModel::compile: layer {i} ({})'s {local} holds {} bytes, more than \
                         the {largest} this GPU binds",
                        layer.name(),
                        bytes(len)
                    )));
                }
            }
        }

        let round = |n: u64| n.div_ceil(8) * 8;
        let state_bytes: u64 = (plan.params.iter().chain(&plan.buffers))
            .map(|&id| round(bytes(values.get(id).data().len())))
            .sum();
        let mask_bytes: u64 = plan.masks.iter().map(|(_, w)| round(bytes(w.len()))).sum();
        if state_bytes + mask_bytes > limits.max_buffer_size {
            return Err(Error::Unsupported(format!(
                "GpuModel::compile: the parameters, running statistics and masks hold {} bytes \
                 together, more than the {} of one buffer on this GPU, which uploads them",
                state_bytes + mask_bytes,
                limits.max_buffer_size
            )));
        }
        let input_bytes = round(bytes(batch * plan.input_width));
        let output_width = plan.layers.last().map_or(plan.input_width, |l| l.outputs);
        let output_bytes = round(bytes(batch * output_width));
        let sizes = StepSizes {
            // At least 4 KiB: the kernels' uniforms go up through this encoder too.
            upload_bytes: (state_bytes + mask_bytes).max(input_bytes).max(4096),
            readback_bytes: (state_bytes + mask_bytes).max(output_bytes),
            ..StepSizes::default()
        };
        let mut pool = BufferPool::new(gpu);
        let mut steps = StepEncoder::new(gpu, sizes);
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mut tensor = |id: ParamId| {
            let len = values.get(id).data().len();
            GpuTensor {
                id,
                len,
                buffer: pool.take(bytes(len).max(4), usage),
            }
        };
        let gpu_params: Vec<GpuTensor> = plan.params.iter().map(|&id| tensor(id)).collect();
        let gpu_buffers: Vec<GpuTensor> = plan.buffers.iter().map(|&id| tensor(id)).collect();
        let masks: Vec<GpuMask> = (plan.masks.iter())
            .map(|(layer, words)| GpuMask {
                layer: *layer,
                bits: pool.take(bytes(words.len()), usage),
            })
            .collect();

        let mut step = steps.begin();
        for t in gpu_params.iter().chain(&gpu_buffers) {
            if t.len > 0 {
                step.upload(t.buffer.buffer(), 0, &f32_bytes(values.get(t.id).data()));
            }
        }
        for (mask, (_, words)) in masks.iter().zip(&plan.masks) {
            step.upload(mask.bits.buffer(), 0, &u32_bytes(words));
        }
        let _ = step.finish();

        let counts = (plan.counts.iter())
            .map(|&id| (id, values.get(id).data()[0]))
            .collect();
        let shader = |e| Error::Unsupported(format!("GpuModel::compile: {e}"));
        let mut cache = PipelineCache::new(gpu);
        let mut kernels = Kernels::new(gpu, &mut cache).map_err(shader)?;
        let parts = Parts {
            layers: &plan.layers,
            params: &gpu_params,
            buffers: &gpu_buffers,
            masks: &masks,
            batch,
            input_width: plan.input_width,
        };
        let eval = Forward::evaluation(&parts, &mut kernels, &mut cache, &mut pool, &mut steps)
            .map_err(shader)?;
        Ok(GpuModel {
            gpu: gpu.clone(),
            pool,
            cache,
            kernels,
            steps,
            batch,
            input_width: plan.input_width,
            output_width,
            layers: plan.layers,
            params: gpu_params,
            buffers: gpu_buffers,
            counts,
            masks,
            eval,
            host: Vec::with_capacity(batch * output_width),
        })
    }

    /// The number of rows in each batch the model was compiled for.
    ///
    /// # Examples
    ///
    /// See [`GpuModel`].
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The width of one input row.
    ///
    /// # Examples
    ///
    /// See [`GpuModel`].
    pub fn input_width(&self) -> usize {
        self.input_width
    }

    /// The width of one output row.
    ///
    /// # Examples
    ///
    /// See [`GpuModel`].
    pub fn output_width(&self) -> usize {
        self.output_width
    }

    /// The GPU the model is on.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{GpuModel, Linear, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// if let Some(gpu) = Gpu::new() {
    ///     let model = GpuModel::compile(&net, &params, 4, &gpu).unwrap();
    ///     assert_eq!(model.gpu().info().name, gpu.info().name);
    /// }
    /// ```
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The compiled layers, one per layer of the [`Sequential`].
    #[allow(dead_code)] // The forward pass and the trainer read it.
    pub(crate) fn layers(&self) -> &[CompiledLayer] {
        &self.layers
    }

    /// Copies the parameters and running statistics on the GPU, and the batch counts kept
    /// on the host, into `params`, in one step with one readback.
    ///
    /// # Panics
    ///
    /// If an id of the model is not in `params`, or holds another number of values there.
    #[allow(dead_code)] // The trainer's `sync_to` calls it.
    pub(crate) fn download_to(&mut self, params: &mut ParamStore<f32>) {
        let mut step = self.steps.begin();
        let mut at = Vec::with_capacity(self.params.len() + self.buffers.len());
        for t in self.params.iter().chain(&self.buffers) {
            at.push((t.len > 0).then(|| step.read_back(t.buffer.buffer(), 0, 4 * t.len as u64)));
        }
        let output = step.finish();
        let bytes = output.bytes();
        for (t, at) in self.params.iter().chain(&self.buffers).zip(at) {
            let Some(at) = at else { continue };
            let data = params.value_mut(t.id).data_mut();
            assert_eq!(
                data.len(),
                t.len,
                "GpuModel: parameter {} holds {} values in the store and {} on the GPU",
                t.id.position(),
                data.len(),
                t.len
            );
            let (words, _) = bytes[at..at + 4 * t.len].as_chunks::<4>();
            for (v, b) in data.iter_mut().zip(words) {
                *v = f32::from_le_bytes(*b);
            }
        }
        for &(id, count) in &self.counts {
            params.value_mut(id).data_mut()[0] = count;
        }
    }

    /// Reads back the packed bits of each neuron mask, by increasing layer, with the layer.
    #[cfg(test)]
    fn mask_words(&mut self) -> Vec<(usize, Vec<u32>)> {
        let mut step = self.steps.begin();
        let at: Vec<usize> = (self.masks.iter())
            .map(|m| step.read_back(m.bits.buffer(), 0, m.bits.len()))
            .collect();
        let output = step.finish();
        let bytes = output.bytes();
        (self.masks.iter().zip(at))
            .map(|(m, at)| {
                let (words, _) = bytes[at..at + m.bits.len() as usize].as_chunks::<4>();
                let words = words.iter().map(|b| u32::from_le_bytes(*b)).collect();
                (m.layer, words)
            })
            .collect()
    }
}

/// The little-endian bytes of `values`.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The little-endian bytes of `words`.
fn u32_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Layer refusals, the plan's layout, the masks' bit packing and the uploads' exact round
/// trip.
#[cfg(test)]
mod tests {
    use cavitation_core::{
        Conv2dGeom, Gate, GeluApprox, ParamId, ParamStore, Pool2dGeom, Real, Rng, Tape, Var,
    };
    use cavitation_gpu::{Backend, Gpu};

    use super::{Act, GpuModel, Kind, Plan, pack_rows};
    use crate::InferenceLayer;
    use crate::{
        AdaptiveAvgPool2d, AvgPool2d, BatchNorm2d, Celu, Conv2d, Dropout, Flatten, Gelu, Glu,
        Hardtanh, Init, Layer, LeakyRelu, Linear, LogSoftmax, MaxPool2d, NeuronMask, Relu,
        Sequential, Sigmoid, Softmax, Tanh,
    };

    /// A layer with no inference form.
    struct Opaque;

    impl<T: Real> Layer<T> for Opaque {
        fn name(&self) -> &str {
            "opaque"
        }
        fn output_width(&self, input_width: usize) -> usize {
            input_width
        }
        fn forward(&self, _: &mut Tape<'_, T>, x: Var) -> Var {
            x
        }
        fn parameters(&self) -> Vec<(&'static str, ParamId)> {
            Vec::new()
        }
    }

    /// A layer of your own with no parameters whose inference form is the one it holds, as a
    /// layer outside this crate could declare any form.
    struct Formed(InferenceLayer);

    impl<T: Real> Layer<T> for Formed {
        fn name(&self) -> &str {
            "formed"
        }
        fn output_width(&self, input_width: usize) -> usize {
            input_width
        }
        fn forward(&self, _: &mut Tape<'_, T>, x: Var) -> Var {
            x
        }
        fn parameters(&self) -> Vec<(&'static str, ParamId)> {
            Vec::new()
        }
        fn inference(&self, _: &cavitation_core::ParamValues<f32>) -> Option<InferenceLayer> {
            Some(self.0.clone())
        }
    }

    /// A layer that passes its input on and fixes an input width more than half of `usize`.
    struct Huge;

    impl<T: Real> Layer<T> for Huge {
        fn name(&self) -> &str {
            "huge"
        }
        fn output_width(&self, input_width: usize) -> usize {
            input_width
        }
        fn input_width(&self) -> Option<usize> {
            Some(usize::MAX / 2 + 1)
        }
        fn forward(&self, _: &mut Tape<'_, T>, x: Var) -> Var {
            x
        }
        fn parameters(&self) -> Vec<(&'static str, ParamId)> {
            Vec::new()
        }
        fn inference(
            &self,
            _: &cavitation_core::ParamValues<f32>,
        ) -> Option<crate::InferenceLayer> {
            Some(crate::InferenceLayer::Identity)
        }
    }

    /// Runs `body` on each GPU this platform has, one per backend, and returns how many ran.
    fn for_each_gpu(mut body: impl FnMut(&Gpu)) -> usize {
        let mut ran = 0;
        for &backend in Backend::on_this_platform() {
            if let Some(gpu) = Gpu::with_backend(backend) {
                println!("on {backend}: {}", gpu.info().name);
                body(&gpu);
                ran += 1;
            }
        }
        if ran == 0 {
            println!("no GPU: skipped");
        }
        ran
    }

    /// A small CNN with every kind of layer the GPU runs, two of them masked:
    /// 1 × 6 × 6 images through a convolution, batch norm, ReLU, max pooling, average
    /// pooling, dropout, flatten, two `Linear`s and each activation.
    fn every_kind(params: &mut ParamStore) -> Sequential {
        let mut rng = Rng::seed(7);
        let conv = Conv2dGeom::with_options(1, 3, (6, 6), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
        let mut net = Sequential::new()
            .push(Conv2d::new(params, conv, &mut rng))
            .push(BatchNorm2d::new(params, 3, (6, 6)))
            .push(Relu)
            .push(MaxPool2d::new(Pool2dGeom::new(3, (6, 6), (2, 2)).unwrap()))
            .push(AvgPool2d::new(
                Pool2dGeom::with_options(3, (3, 3), (2, 2), (1, 1), (1, 1), (1, 1), false).unwrap(),
            ))
            .push(Dropout::new(0.25).unwrap())
            .push(Flatten)
            .push(Linear::new(params, 3 * 4 * 4, 5, &mut rng))
            .push(LeakyRelu { slope: 0.125 })
            .push(Tanh)
            .push(Sigmoid)
            .push(Gelu::default())
            .push(Gelu {
                approx: GeluApprox::Tanh,
            })
            .push(Linear::without_bias(
                params,
                5,
                2,
                Init::TorchDefault,
                &mut rng,
            ));
        net.set_mask(2, NeuronMask::from_off(3 * 6 * 6, &[0, 5, 107]));
        net.set_neuron_mask(7, NeuronMask::from_off(5, &[1, 3]));
        net
    }

    /// Each layer kind is recognised, with the widths flowing through, the parameters and
    /// buffers each laid out once and the batch count kept apart.
    #[test]
    fn plan_reads_every_kind() {
        let mut params = ParamStore::new();
        let net = every_kind(&mut params);
        let plan = Plan::build(&net, &params, 3).unwrap();
        assert_eq!(plan.input_width, 36);
        let widths: Vec<(usize, usize)> =
            plan.layers.iter().map(|l| (l.inputs, l.outputs)).collect();
        assert_eq!(
            widths[..5],
            [(36, 108), (108, 108), (108, 108), (108, 27), (27, 48)]
        );
        assert_eq!(widths[13], (5, 2));
        assert!(matches!(
            plan.layers[0].kind,
            Kind::Conv2d {
                weight: 0,
                bias: Some(1),
                ..
            }
        ));
        let Kind::BatchNorm2d(norm) = plan.layers[1].kind else {
            panic!("{:?}", plan.layers[1].kind)
        };
        assert_eq!(
            (norm.weight, norm.bias, norm.channels, norm.spatial),
            (Some(2), Some(3), 3, 36)
        );
        let running = norm.running.unwrap();
        assert_eq!(
            (running.mean, running.var, running.count, running.momentum),
            (0, 1, Some(0), Some(0.1))
        );
        assert_eq!(plan.layers[5].kind, Kind::Dropout(0.25));
        assert_eq!(plan.layers[6].kind, Kind::Identity);
        let acts: Vec<Kind> = plan.layers[8..13].iter().map(|l| l.kind).collect();
        assert_eq!(
            acts,
            [
                Kind::Activation(Act::LeakyRelu(0.125)),
                Kind::Activation(Act::Tanh),
                Kind::Activation(Act::Sigmoid),
                Kind::Activation(Act::GeluErf),
                Kind::Activation(Act::GeluTanh),
            ]
        );
        assert!(matches!(
            plan.layers[13].kind,
            Kind::Linear {
                inputs: 5,
                outputs: 2,
                weight: 6,
                bias: None
            }
        ));
        assert_eq!(
            (plan.params.len(), plan.buffers.len(), plan.counts.len()),
            (7, 2, 1)
        );
        // Layer 7's neurons are masked after the run of activations that follows it.
        let masked: Vec<(usize, Option<usize>)> = (plan.layers.iter().enumerate())
            .filter(|(_, l)| l.mask.is_some())
            .map(|(i, l)| (i, l.mask))
            .collect();
        assert_eq!(masked, [(2, Some(0)), (12, Some(1))]);
        assert_eq!(plan.masks[1].0, 12);
    }

    /// A layer the GPU does not run is refused with its index and name, whatever the GPU.
    #[test]
    fn refusals_name_the_layer() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(0);
        let linear = |params: &mut ParamStore, rng: &mut Rng| Linear::new(params, 4, 3, rng);
        let cases: Vec<(Sequential, &str)> = vec![
            (
                Sequential::new()
                    .push(linear(&mut params, &mut rng))
                    .push(Formed(InferenceLayer::Linear {
                        w_t: vec![0.0; 9],
                        bias: None,
                        inputs: 3,
                        outputs: 3,
                    })),
                "unsupported: GpuModel::compile: layer 1 (formed) is not a layer the GPU runs",
            ),
            (
                Sequential::new()
                    .push(linear(&mut params, &mut rng))
                    .push(Glu {
                        gate: Gate::Sigmoid,
                    }),
                "invalid data: GpuModel::compile: layer 1 (glu) is a GLU over width 3, which must \
                 be even",
            ),
            (
                Sequential::new()
                    .push(linear(&mut params, &mut rng))
                    .push(Celu { alpha: -1.0 }),
                "unsupported: GpuModel::compile: layer 1 (celu) is a CELU with alpha -1; the GPU runs a positive alpha only",
            ),
            (
                Sequential::new()
                    .push(linear(&mut params, &mut rng))
                    .push(Hardtanh {
                        min: 1.0,
                        max: -1.0,
                    }),
                "unsupported: GpuModel::compile: layer 1 (hardtanh) is a hardtanh with min 1 and max -1; min must not exceed max",
            ),
            (
                Sequential::new()
                    .push(linear(&mut params, &mut rng))
                    .push(Hardtanh {
                        min: f64::NAN,
                        max: 1.0,
                    }),
                "unsupported: GpuModel::compile: layer 1 (hardtanh) is a hardtanh with min NaN and max 1; min must not exceed max",
            ),
            (
                Sequential::new()
                    .push(linear(&mut params, &mut rng))
                    .push(Relu)
                    .push(Opaque),
                "unsupported: GpuModel::compile: layer 2 (opaque) has no inference form, so the GPU cannot run it",
            ),
            (
                Sequential::new().push(Relu),
                "unsupported: GpuModel::compile: no layer of the model fixes its input width",
            ),
            (
                Sequential::new(),
                "unsupported: GpuModel::compile: the model has no layers",
            ),
            (
                Sequential::new()
                    .push(linear(&mut params, &mut rng))
                    .push(Linear::new(&mut params, 5, 1, &mut rng)),
                "invalid data: GpuModel::compile: layer 1 (linear) expects input width 5, but its input has width 3",
            ),
        ];
        for (net, message) in cases {
            let err = Plan::build(&net, &params, 8).unwrap_err();
            assert_eq!(err.to_string(), message);
        }
        let net = Sequential::new().push(linear(&mut params, &mut rng));
        // A GEMM's sides, batch norm on one value per channel, and an eps f32 flushes.
        let err = Plan::build(&net, &params, 4_194_241).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unsupported: GpuModel::compile: layer 0 (linear) has 4194241 rows per batch, more \
             than the 4194240 the GPU's GEMM takes"
        );
        let mut store = ParamStore::new();
        let batch_stats =
            BatchNorm2d::with_options(&mut store, 2, (1, 1), 1e-5, Some(0.1), true, false).unwrap();
        let err = Plan::build(&Sequential::new().push(batch_stats), &store, 1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid data: GpuModel::compile: layer 0 (batch_norm2d) normalises with the batch's \
             statistics, which need more than one value per channel; a batch of 1 gives 1"
        );
        let tiny_eps =
            BatchNorm2d::with_options(&mut store, 2, (2, 2), 1e-40, Some(0.1), true, true).unwrap();
        let err = Plan::build(&Sequential::new().push(tiny_eps), &store, 4).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unsupported: GpuModel::compile: layer 0 (batch_norm2d) has eps 1e-40, which the \
             GPU's f32 arithmetic may flush to 0; use 0 or at least 2⁻¹²⁶"
        );
        let err = Plan::build(&net, &params, 0).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid data: GpuModel::compile: a batch needs at least one row"
        );
        // Without a convolution, a batch may pass 65,535 rows; with one, it may not.
        assert!(Plan::build(&net, &params, 65_536).is_ok());
        let conv = Conv2dGeom::new(1, 1, (2, 2), (1, 1)).unwrap();
        let net = Sequential::new().push(Conv2d::new(&mut params, conv, &mut rng));
        let err = Plan::build(&net, &params, 65_536).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unsupported: GpuModel::compile: layer 0 (conv2d) is a convolution, which takes at \
             most 65535 images per batch on the GPU, not 65536"
        );
        // A softmax takes one thread per row, 65,535 workgroups of 256.
        let mut store = ParamStore::new();
        let norm =
            BatchNorm2d::with_options(&mut store, 2, (2, 2), 1e-5, Some(0.1), true, true).unwrap();
        for (net, log) in [
            (Sequential::new().push(norm.clone()).push(Softmax), false),
            (Sequential::new().push(norm).push(LogSoftmax), true),
        ] {
            let plan = Plan::build(&net, &store, 16_776_960).unwrap();
            assert_eq!(plan.layers[1].kind, Kind::Softmax { log });
            let err = Plan::build(&net, &store, 16_776_961).unwrap_err();
            let name = net.layer(1).name().to_string();
            assert_eq!(
                err.to_string(),
                format!(
                    "unsupported: GpuModel::compile: layer 1 ({name}) is a softmax, which takes \
                     at most 16776960 rows per batch on the GPU, not 16776961"
                )
            );
        }
    }

    /// Adaptive average pooling is planned with its sizes and widths; the pooling forms the CPU
    /// refuses (a dilated average pooling or a divisor of 0, which `AvgPool2d::with_options`
    /// refuses with `InvalidData`, and an adaptive pooling with a size of 0, which
    /// `AdaptiveAvgPool2d::new` refuses) are invalid data, and an adaptive pooling beyond the
    /// kernels' 32-bit arithmetic or 2³¹ values per batch is unsupported, each naming the layer.
    #[test]
    fn pooling_forms_are_checked() {
        let mut params = ParamStore::new();
        let net = Sequential::new()
            .push(AdaptiveAvgPool2d::new(3, (7, 5), (3, 2)).unwrap())
            .push(Linear::new(&mut params, 18, 2, &mut Rng::seed(0)));
        let plan = Plan::build(&net, &params, 4).unwrap();
        assert_eq!(plan.input_width, 105);
        assert_eq!(
            plan.layers[0].kind,
            Kind::AdaptiveAvgPool2d {
                channels: 3,
                input: (7, 5),
                output: (3, 2)
            }
        );
        assert_eq!((plan.layers[0].inputs, plan.layers[0].outputs), (105, 18));
        let dilated = Pool2dGeom::with_options(1, (6, 6), (2, 2), (2, 2), (0, 0), (2, 2), false);
        let avg = |geom, divisor_override| InferenceLayer::AvgPool2d {
            geom,
            count_include_pad: true,
            divisor_override,
        };
        let adaptive = |input, output| InferenceLayer::AdaptiveAvgPool2d {
            channels: 1,
            input,
            output,
        };
        let plain = Pool2dGeom::new(1, (6, 6), (2, 2)).unwrap();
        let cases = [
            (
                avg(dilated.unwrap(), None),
                1,
                "invalid data: GpuModel::compile: layer 0 (formed) is an average pooling with \
                 dilation (2, 2); average pooling takes none",
            ),
            (
                avg(plain, Some(0)),
                1,
                "invalid data: GpuModel::compile: layer 0 (formed) is an average pooling with a \
                 divisor of 0",
            ),
            (
                adaptive((4, 4), (0, 2)),
                1,
                "invalid data: GpuModel::compile: layer 0 (formed) is an adaptive average \
                 pooling with channels 1, input (4, 4) and output (0, 2); every count and size \
                 must be at least 1",
            ),
            (
                adaptive((70_000, 1), (70_000, 1)),
                1,
                "unsupported: GpuModel::compile: layer 0 (formed) is an adaptive average pooling \
                 of (70000, 1) to (70000, 1), whose window arithmetic does not fit the GPU's \
                 32-bit integers",
            ),
            (
                adaptive((2, 1), (2, 1)),
                (1 << 30) + 1,
                "unsupported: GpuModel::compile: layer 0 (formed) is an adaptive average pooling \
                 over more than the 2147483648 values per batch the GPU takes",
            ),
        ];
        for (form, batch, message) in cases {
            let net = Sequential::new().push(Formed(form));
            let err = Plan::build(&net, &ParamStore::new(), batch).unwrap_err();
            assert_eq!(err.to_string(), message);
        }
        let net = Sequential::new().push(Formed(adaptive((2, 1), (2, 1))));
        assert!(Plan::build(&net, &ParamStore::new(), 1 << 30).is_ok());
    }

    /// The masks' bits repeat each row's for every row of the batch, low bit first.
    #[test]
    fn masks_pack_row_after_row() {
        let on = [true, false, true];
        // Bits 0, 2, 3, 5, 6, 8, ... of 11 rows: 33 values over two words.
        let words = pack_rows(&on, 11);
        assert_eq!(words.len(), 2);
        assert_eq!(
            words[0],
            (0..32)
                .filter(|k| k % 3 != 1)
                .map(|k| 1u32 << k)
                .sum::<u32>()
        );
        assert_eq!(words[1], 1);
    }

    /// Every parameter, running statistic and mask comes back from the GPU bit for bit,
    /// odd values included, and the batch count as the host kept it.
    #[test]
    fn uploads_read_back_exactly() {
        let mut params = ParamStore::new();
        let net = every_kind(&mut params);
        let ids: Vec<ParamId> = net.state_names().into_iter().map(|(_, id)| id).collect();
        // Values a copy must keep: −0, a NaN with a payload, the smallest subnormal, the
        // largest float and infinity, written over the start of each tensor.
        let odd = [
            -0.0,
            f32::from_bits(0x7fc1_2345),
            f32::from_bits(1),
            f32::MAX,
            f32::NEG_INFINITY,
        ];
        for &id in &ids {
            for (v, &o) in params.value_mut(id).data_mut().iter_mut().zip(&odd) {
                *v = o;
            }
        }
        let count = (net.layer(1).as_batch_norm2d())
            .and_then(|bn| bn.running())
            .and_then(|r| r.count)
            .unwrap();
        assert!(ids.contains(&count));
        params.value_mut(count).data_mut()[0] = 17.0;
        let want: Vec<Vec<u32>> = ids
            .iter()
            .map(|&id| {
                params
                    .value(id)
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect()
            })
            .collect();
        for_each_gpu(|gpu| {
            let mut model = GpuModel::compile(&net, &params, 3, gpu).unwrap();
            let mut back = params.clone();
            for &id in &ids {
                back.value_mut(id).data_mut().fill(1.5);
            }
            model.download_to(&mut back);
            for (k, &id) in ids.iter().enumerate() {
                let got: Vec<u32> = back.value(id).data().iter().map(|v| v.to_bits()).collect();
                assert_eq!(got, want[k], "parameter {}", id.position());
            }
            let masks = model.mask_words();
            assert_eq!(masks.len(), 2);
            let on2: Vec<bool> = (0..108).map(|j| ![0, 5, 107].contains(&j)).collect();
            let on12: Vec<bool> = (0..5).map(|j| ![1, 3].contains(&j)).collect();
            assert_eq!(masks[0], (2, pack_rows(&on2, 3)));
            assert_eq!(masks[1], (12, pack_rows(&on12, 3)));
        });
    }

    /// A PReLU is planned with its slopes as a trained parameter, one per column or one
    /// shared; a PReLU of more than 65,535 slopes is refused on both engines, one whose
    /// weight does not hold one value per slope is invalid, the wgpu engine refuses a slope
    /// gradient over more than 268,431,360 values per slope, which CUDA takes, and both refuse
    /// more than 2³¹ values per batch.
    #[test]
    fn prelu_is_planned_with_its_limits() {
        use crate::Prelu;
        use crate::plan::Engine;
        use cavitation_core::Tensor;

        let mut params = ParamStore::new();
        let mut rng = Rng::seed(4);
        let linear = Linear::new(&mut params, 4, 3, &mut rng);
        let (per_column, shared) = (Prelu::new(&mut params, 3), Prelu::new(&mut params, 1));
        let net = Sequential::new().push(linear).push(per_column).push(shared);
        let plan = Plan::build(&net, &params, 8).unwrap();
        assert_eq!(
            plan.layers[1].kind,
            Kind::Prelu {
                channels: 3,
                weight: 2
            }
        );
        assert_eq!(
            plan.layers[2].kind,
            Kind::Prelu {
                channels: 1,
                weight: 3
            }
        );
        assert_eq!(plan.params[2..], [per_column.weight(), shared.weight()]);

        let mut store = ParamStore::new();
        let wide = Sequential::new().push(Prelu::new(&mut store, 65_536));
        for engine in [Engine::WGPU, Engine::CUDA] {
            let err = Plan::build_for(&wide, &store, 2, engine).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "unsupported: {}: layer 0 (prelu) is a PReLU with 65536 slopes; the GPU \
                     takes at most 65535",
                    engine.name
                )
            );
        }

        let mut store = ParamStore::new();
        let prelu = Prelu::new(&mut store, 3);
        *store.value_mut(prelu.weight()) = Tensor::from_vec(vec![0.5; 4], &[4]).unwrap();
        let net = Sequential::new().push(prelu);
        let err = Plan::build(&net, &store, 2).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid data: GpuModel::compile: layer 0 (prelu) is a PReLU with 3 slopes, but its \
             weight holds 4 values"
        );

        // A shared slope sums every value of the batch; one per column, the batch's rows.
        let mut store = ParamStore::new();
        let wide_linear = Linear::new(&mut store, 2, 4096, &mut rng);
        let net = Sequential::new()
            .push(wide_linear)
            .push(Prelu::new(&mut store, 1));
        assert!(Plan::build(&net, &store, 65_535).is_ok());
        let err = Plan::build(&net, &store, 65_536).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unsupported: GpuModel::compile: layer 1 (prelu) is a PReLU whose slope gradient \
             sums 268435456 values per slope, more than the 268431360 the GPU takes"
        );
        assert!(Plan::build_for(&net, &store, 65_536, Engine::CUDA).is_ok());
        let net = Sequential::new().push(Prelu::new(&mut store, 2));
        assert!(Plan::build(&net, &store, 268_431_360).is_ok());
        assert!(Plan::build(&net, &store, 268_431_361).is_err());
        assert!(Plan::build_for(&net, &store, 268_431_361, Engine::CUDA).is_ok());
        // Either engine takes at most 2³¹ values per batch through a PReLU.
        let net = Sequential::new().push(Prelu::new(&mut store, 65_535));
        for engine in [Engine::WGPU, Engine::CUDA] {
            assert!(Plan::build_for(&net, &store, 32_768, engine).is_ok());
            let err = Plan::build_for(&net, &store, 32_769, engine).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "unsupported: {}: layer 0 (prelu) is a PReLU over 32769 rows of 65535 \
                     values, more than the 2147483648 per batch the GPU takes",
                    engine.name
                )
            );
        }
    }

    /// A GLU is planned with every gate, halving the width: a GLU before the first layer that
    /// fixes a width doubles the model's input width for each GLU, an odd width is invalid on
    /// both engines, as the CPU's GLU panics on it, and both refuse more than 4,294,967,295
    /// values per batch, which the kernels' 32-bit indices do not hold.
    #[test]
    fn glu_is_planned_with_its_widths() {
        use crate::plan::Engine;

        let mut params = ParamStore::new();
        let mut rng = Rng::seed(5);
        for gate in [
            Gate::Sigmoid,
            Gate::Gelu(GeluApprox::Tanh),
            Gate::Gelu(GeluApprox::Exact),
            Gate::Relu,
            Gate::Silu,
        ] {
            let net = Sequential::new()
                .push(Linear::new(&mut params, 4, 6, &mut rng))
                .push(Glu { gate })
                .push(Linear::new(&mut params, 3, 2, &mut rng));
            let plan = Plan::build(&net, &params, 8).unwrap();
            assert_eq!(plan.layers[1].kind, Kind::Glu(gate));
            assert_eq!((plan.layers[1].inputs, plan.layers[1].outputs), (6, 3));
        }

        let net = Sequential::new()
            .push(Glu { gate: Gate::Silu })
            .push(Glu {
                gate: Gate::Sigmoid,
            })
            .push(Linear::new(&mut params, 3, 2, &mut rng));
        let plan = Plan::build(&net, &params, 8).unwrap();
        assert_eq!(plan.input_width, 12);
        let widths: Vec<(usize, usize)> =
            plan.layers.iter().map(|l| (l.inputs, l.outputs)).collect();
        assert_eq!(widths, [(12, 6), (6, 3), (3, 2)]);
        let alone = Sequential::new().push(Glu { gate: Gate::Relu });
        let err = Plan::build(&alone, &params, 8).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unsupported: GpuModel::compile: no layer of the model fixes its input width"
        );
        let doubled = Sequential::new().push(Glu { gate: Gate::Relu }).push(Huge);
        let err = Plan::build(&doubled, &params, 8).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "unsupported: GpuModel::compile: layer 1 (huge) takes input width {}, which the \
                 GLUs before it double past what memory can index",
                usize::MAX / 2 + 1
            )
        );

        let odd = Sequential::new()
            .push(Linear::new(&mut params, 4, 5, &mut rng))
            .push(Glu { gate: Gate::Relu });
        for engine in [Engine::WGPU, Engine::CUDA] {
            let err = Plan::build_for(&odd, &params, 8, engine).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "invalid data: {}: layer 1 (glu) is a GLU over width 5, which must be even",
                    engine.name
                )
            );
        }

        let wide = Sequential::new()
            .push(Linear::new(&mut params, 2, 4096, &mut rng))
            .push(Glu { gate: Gate::Relu });
        for engine in [Engine::WGPU, Engine::CUDA] {
            assert!(Plan::build_for(&wide, &params, 1_048_575, engine).is_ok());
            let err = Plan::build_for(&wide, &params, 1_048_576, engine).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "unsupported: {}: layer 1 (glu) is a GLU over 1048576 rows of 4096 values, \
                     more than the 4294967295 per batch the GPU takes",
                    engine.name
                )
            );
        }
    }

    /// A GLU's buffers are held to the largest buffer the GPU binds, as every layer's are: a
    /// leading GLU over 4,096 values a row, at the first batch whose input is larger, is
    /// refused when the model compiles, before anything is allocated.
    #[test]
    fn glu_buffers_are_held_to_the_gpus_binding_limit() {
        let mut params = ParamStore::new();
        let net = Sequential::new()
            .push(Glu { gate: Gate::Relu })
            .push(Linear::new(&mut params, 2048, 1, &mut Rng::seed(8)));
        for_each_gpu(|gpu| {
            let limits = &gpu.info().limits;
            let largest = limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size);
            let batch = (largest / (4 * 4096) + 1) as usize;
            let err = GpuModel::compile(&net, &params, batch, gpu).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "unsupported: GpuModel::compile: layer 0 (glu) reads or writes {} bytes per \
                     batch of {batch}, more than the {largest} this GPU binds",
                    4 * 4096 * batch
                )
            );
        });
    }
}
