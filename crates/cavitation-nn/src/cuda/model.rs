//! `CudaModel`: a [`Sequential`] compiled for an NVIDIA card through CUDA at one batch size.
//! Compiling checks every layer against the forms the card runs, refusing the others by
//! index and name, and uploads the parameters, the batch-norm running statistics and the
//! neuron masks.

use cavitation_core::{Error, ParamId, ParamStore};
use cavitation_cuda::{Cuda, CudaError, Dropout, Elementwise, Glu, Prelu, RowSoftmax};

use super::pass::{Forward, Parts, Shared, Store};
use crate::Sequential;
use crate::plan::{CompiledLayer, Engine, Plan};

/// A parameter or buffer held on the card under the id it has in the [`ParamStore`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaTensor {
    /// The id in the store.
    pub(crate) id: ParamId,
    /// The number of `f32` values.
    pub(crate) len: usize,
    /// Its buffer, an index into the model's store.
    pub(crate) at: usize,
}

/// `who: e` as [`Error::Unsupported`]: what the card or NVIDIA's libraries refused.
pub(crate) fn refused(who: &str, e: CudaError) -> Error {
    Error::Unsupported(format!("{who}: {e}"))
}

/// The result of a call to the card that is expected to work once the model compiled;
/// a failure is the card's, not the caller's.
///
/// # Panics
///
/// If `result` is an error, naming `who` and the error.
pub(crate) fn ran<T>(who: &str, result: Result<T, CudaError>) -> T {
    match result {
        Ok(value) => value,
        Err(e) => panic!("{who}: the card reported an error: {e}"),
    }
}

/// A [`Sequential`] compiled for an NVIDIA card through CUDA, for batches of a fixed number of
/// rows: its layers as the card runs them, with its parameters, batch-norm running statistics
/// and neuron masks held on the card.
///
/// It runs the layers [`GpuModel`](crate::GpuModel) runs, from the same checks:
/// [`Linear`](crate::Linear) (through cuBLAS), [`Conv2d`](crate::Conv2d) (through cuDNN),
/// [`MaxPool2d`](crate::MaxPool2d), [`AvgPool2d`](crate::AvgPool2d),
/// [`BatchNorm2d`](crate::BatchNorm2d), [`Dropout`](crate::Dropout), [`Flatten`](crate::Flatten),
/// [`Identity`](crate::Identity) and every built-in activation (ReLU, leaky ReLU, tanh, the
/// sigmoid, GELU in both forms, ELU, SELU, CELU with a positive `alpha`, SiLU, Mish, softplus,
/// softsign, hardtanh with `min <= max`, ReLU6, the hard sigmoid, the hard swish and log-sigmoid),
/// [`Softmax`](crate::Softmax) and [`LogSoftmax`](crate::LogSoftmax) over each row,
/// [`Prelu`](crate::Prelu) with one slope per column or one shared slope (at most 65,535), its
/// slopes trained as parameters, and [`Glu`](crate::Glu) with every gate, over an even width.
/// [`compile`](CudaModel::compile) refuses any other layer, naming its index and name. Neuron masks
/// set on the model when it is compiled are compiled in; a mask set later needs a new compile.
///
/// A batch-norm layer's `num_batches_tracked` stays on the host, since only the host
/// decides whether a step's running-statistics update ran.
///
/// The model borrows the [`Cuda`] it was compiled on, which threads may share. The model itself
/// is neither `Send` nor `Sync`: it holds its kernels' handles and its buffers, so it is used
/// on the thread that compiled it.
///
/// # Examples
///
/// Without CUDA, [`Cuda::new`] returns `None` and the example does nothing:
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_cuda::Cuda;
/// use cavitation_nn::{CudaModel, Linear, Relu, Sequential};
///
/// let mut params = ParamStore::new();
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 4, 8, &mut Rng::seed(0)))
///     .push(Relu)
///     .push(Linear::new(&mut params, 8, 2, &mut Rng::seed(1)));
/// if let Some(cuda) = Cuda::new() {
///     let model = CudaModel::compile(&net, &params, 32, &cuda).unwrap();
///     assert_eq!((model.batch(), model.input_width(), model.output_width()), (32, 4, 2));
/// }
/// ```
pub struct CudaModel<'c> {
    /// The CUDA the model is on.
    pub(super) cuda: &'c Cuda,
    /// Every buffer of the model and of its trainer.
    pub(super) store: Store<'c>,
    /// The element-wise kernels every pass shares.
    pub(super) shared: Shared<'c>,
    /// Rows per batch.
    pub(super) batch: usize,
    /// The width of one input row.
    pub(super) input_width: usize,
    /// The width of one output row.
    pub(super) output_width: usize,
    /// The layers, one per layer of the [`Sequential`].
    pub(super) layers: Vec<CompiledLayer>,
    /// The trained parameters.
    pub(super) params: Vec<CudaTensor>,
    /// The batch-norm running statistics.
    pub(super) buffers: Vec<CudaTensor>,
    /// Each batch-norm batch count, with its value, kept on the host.
    pub(super) counts: Vec<(ParamId, f32)>,
    /// The neuron masks' bits, by increasing layer, with their layer.
    pub(super) masks: Vec<(usize, usize)>,
    /// The forward pass in evaluation mode.
    pub(super) eval: Forward<'c>,
    /// The last batch's outputs, read back.
    pub(super) host: Vec<f32>,
}

impl std::fmt::Debug for CudaModel<'_> {
    /// Writes the card, the batch and the widths.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaModel")
            .field("card", &self.cuda.info().name)
            .field("batch", &self.batch)
            .field("input_width", &self.input_width)
            .field("output_width", &self.output_width)
            .field("layers", &self.layers.len())
            .finish_non_exhaustive()
    }
}

impl<'c> CudaModel<'c> {
    /// Compiles `model`, with its values in `params`, for batches of `batch` rows on `cuda`'s
    /// card, and uploads its parameters, batch-norm running statistics and neuron masks.
    ///
    /// # Errors
    ///
    /// - [`Error::Unsupported`] if the model has no layers, or a layer is not one the card
    ///   runs (see [`CudaModel`]) or has parameters its GPU form does not use; the message
    ///   names the layer's index and name. Also if no layer fixes the input width, or if the
    ///   card or NVIDIA's libraries refuse a layer's shape or its memory: a size cuDNN's or
    ///   cuBLAS's `int` does not hold, a convolution with no deterministic algorithm that
    ///   uses no tensor cores, a batch norm of more than 65,535 channels, or memory the card
    ///   does not have. Also a PReLU of more than 65,535 slopes or of more than 2³¹ values per
    ///   batch, and a GLU of more than 4,294,967,295 values per batch.
    /// - [`Error::InvalidData`] if `batch` is 0, a layer's input width is not the width the
    ///   layer before it outputs, a layer outputs width 0 or another width than it reports,
    ///   a GLU's input width is odd (the CPU's GLU panics on it), a mask is not as wide as its
    ///   layer's output, a batch count does not hold one value,
    ///   a PReLU's weight does not hold one value per slope, or a batch norm that keeps no
    ///   running statistics would see one value per channel; the message names the layer (and
    ///   both widths, where two differ).
    ///
    /// # Panics
    ///
    /// If an id of the model's layers is not in `params`.
    ///
    /// # Examples
    ///
    /// A layer the card does not run is refused by index and name:
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{AdaptiveAvgPool2d, CudaModel, Linear, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 4, 4, &mut Rng::seed(0)))
    ///     .push(AdaptiveAvgPool2d::new(1, (2, 2), (1, 1)).unwrap());
    /// if let Some(cuda) = Cuda::new() {
    ///     let err = CudaModel::compile(&net, &params, 8, &cuda).unwrap_err();
    ///     assert!(err.to_string().contains("layer 1 (adaptive_avg_pool2d)"), "{err}");
    /// }
    /// ```
    pub fn compile(
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
        batch: usize,
        cuda: &'c Cuda,
    ) -> Result<CudaModel<'c>, Error> {
        const WHO: &str = "CudaModel::compile";
        let plan = Plan::build_for(model, params, batch, Engine::CUDA)?;
        let refuse = |e| refused(WHO, e);
        let values = &params.values;
        let mut store = Store::new(cuda);
        let tensor = |id: ParamId, store: &mut Store<'c>| {
            let data = values.get(id).data();
            let at = store.values(data.len())?;
            store.get_mut(at).write(data)?;
            Ok::<_, CudaError>(CudaTensor {
                id,
                len: data.len(),
                at,
            })
        };
        let gpu_params = (plan.params.iter())
            .map(|&id| tensor(id, &mut store))
            .collect::<Result<Vec<_>, _>>()
            .map_err(refuse)?;
        let gpu_buffers = (plan.buffers.iter())
            .map(|&id| tensor(id, &mut store))
            .collect::<Result<Vec<_>, _>>()
            .map_err(refuse)?;
        let mut masks = Vec::with_capacity(plan.masks.len());
        for (layer, words) in &plan.masks {
            let at = store.values(words.len()).map_err(refuse)?;
            store.get_mut(at).write(words).map_err(refuse)?;
            masks.push((*layer, at));
        }
        let counts = (plan.counts.iter())
            .map(|&id| (id, values.get(id).data()[0]))
            .collect();
        let shared = Shared {
            elementwise: Elementwise::new(cuda).map_err(refuse)?,
            dropout: Dropout::new(cuda).map_err(refuse)?,
            softmax: RowSoftmax::new(cuda).map_err(refuse)?,
            prelu: Prelu::new(cuda).map_err(refuse)?,
            glu: Glu::new(cuda).map_err(refuse)?,
        };
        let param_at: Vec<usize> = gpu_params.iter().map(|t| t.at).collect();
        let buffer_at: Vec<usize> = gpu_buffers.iter().map(|t| t.at).collect();
        let mask_at: Vec<usize> = masks.iter().map(|&(_, at)| at).collect();
        let parts = Parts {
            layers: &plan.layers,
            params: &param_at,
            buffers: &buffer_at,
            masks: &mask_at,
            input_width: plan.input_width,
        };
        let eval = Forward::build(&mut store, &parts, batch, false).map_err(refuse)?;
        let output_width = plan.layers.last().map_or(plan.input_width, |l| l.outputs);
        Ok(CudaModel {
            cuda,
            store,
            shared,
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
    /// See [`CudaModel`].
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The width of one input row.
    ///
    /// # Examples
    ///
    /// See [`CudaModel`].
    pub fn input_width(&self) -> usize {
        self.input_width
    }

    /// The width of one output row.
    ///
    /// # Examples
    ///
    /// See [`CudaModel`].
    pub fn output_width(&self) -> usize {
        self.output_width
    }

    /// The CUDA the model is on.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{CudaModel, Linear, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// if let Some(cuda) = Cuda::new() {
    ///     let model = CudaModel::compile(&net, &params, 4, &cuda).unwrap();
    ///     assert_eq!(model.cuda().info().name, cuda.info().name);
    /// }
    /// ```
    pub fn cuda(&self) -> &'c Cuda {
        self.cuda
    }

    /// Copies the parameters and running statistics on the card, and the batch counts kept
    /// on the host, into `params`.
    ///
    /// # Panics
    ///
    /// If an id of the model is not in `params`, or holds another number of values there, or
    /// the card reports an error.
    pub(crate) fn download_to(&mut self, params: &mut ParamStore<f32>) {
        for t in self.params.iter().chain(&self.buffers) {
            let data = params.value_mut(t.id).data_mut();
            assert_eq!(
                data.len(),
                t.len,
                "CudaModel: parameter {} holds {} values in the store and {} on the card",
                t.id.position(),
                data.len(),
                t.len
            );
            ran("CudaModel", self.store.get(t.at).read(data));
        }
        for &(id, count) in &self.counts {
            params.value_mut(id).data_mut()[0] = count;
        }
    }

    /// Reads back the packed bits of each neuron mask, by increasing layer, with the layer.
    #[cfg(test)]
    pub(crate) fn mask_words(&self) -> Vec<(usize, Vec<u32>)> {
        (self.masks.iter())
            .map(|&(layer, at)| {
                let b = self.store.get(at);
                let mut words = vec![0u32; b.bytes() / 4];
                b.read(&mut words).unwrap();
                (layer, words)
            })
            .collect()
    }
}
