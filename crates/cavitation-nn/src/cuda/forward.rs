//! A [`CudaModel`]'s answers through its evaluation-mode forward pass:
//! [`CudaModel::infer_batch`], [`CudaModel::evaluate`] and [`CudaModel::accuracy`].

use cavitation_core::{Error, TensorView};

use super::model::{CudaModel, ran};
use super::pass::run;
use crate::train::{Evaluation, measure};
use crate::{Dataset, Loss, Sequential, Targets};

impl CudaModel<'_> {
    /// Runs a batch of inputs, shape `[rows, input_width]` with `rows` from 1 to
    /// [`batch`](Self::batch), through the model in evaluation mode, and returns the
    /// outputs, shape `[rows, output_width]` (in the model's own buffer, valid until the next
    /// call). Evaluation mode is the CPU tape's: dropout passes its input on, and batch norm
    /// uses its running statistics, or the batch's when it keeps none. A batch is one upload,
    /// the layers, and one readback.
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
    /// uses the batch's statistics; the message names the shape. Also if the card reports an
    /// error.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{CudaModel, Linear, Relu, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 3, 4, &mut Rng::seed(0)))
    ///     .push(Relu);
    /// if let Some(cuda) = Cuda::new() {
    ///     let mut model = CudaModel::compile(&net, &params, 8, &cuda).unwrap();
    ///     let x = Tensor::full(&[5, 3], 0.5f32);
    ///     let y = model.infer_batch(x.view());
    ///     assert_eq!(y.shape().dims(), &[5, 4]);
    ///     assert!(y.data().iter().all(|&v| v >= 0.0));
    /// }
    /// ```
    pub fn infer_batch(&mut self, inputs: TensorView<'_, f32>) -> TensorView<'_, f32> {
        const WHO: &str = "CudaModel::infer_batch";
        let dims = inputs.shape().dims();
        let rows = match *dims {
            [rows, width] if width == self.input_width && (1..=self.batch).contains(&rows) => rows,
            _ => panic!(
                "{WHO}: the inputs must be [1..={}, {}], got {dims:?}",
                self.batch, self.input_width
            ),
        };
        assert!(
            rows == self.batch || !self.eval.batch_statistics,
            "{WHO}: a batch-norm layer of this model normalises with the batch's statistics, \
             so it needs all {} rows, got {dims:?}",
            self.batch
        );
        let out_len = rows * self.output_width;
        let store = &self.store;
        ran(WHO, store.get_mut(self.eval.input).write(inputs.data()));
        ran(
            WHO,
            run(&self.eval.ops, &mut self.eval.kernels, &self.shared, store),
        );
        self.host.clear();
        self.host.resize(out_len, 0.0);
        ran(WHO, store.get(self.eval.output).read(&mut self.host));
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
    /// # Panics
    ///
    /// If the card reports an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{CudaModel, Dataset, Linear, Loss, Sequential, Targets};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
    /// let data = Dataset { x: Tensor::zeros(&[5, 2]), targets: Targets::Classes(vec![0, 1, 0, 1, 0]) };
    /// if let Some(cuda) = Cuda::new() {
    ///     let mut model = CudaModel::compile(&net, &params, 2, &cuda).unwrap();
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
        let batch = self.batch;
        measure(model, data, loss, batch, width, |x, out| {
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
    /// as wide as the model's input, a batch-norm layer that keeps no running statistics
    /// would see a short last batch, or the card reports an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{CudaModel, Dataset, Linear, Sequential, Targets};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
    /// let data = Dataset { x: Tensor::zeros(&[4, 2]), targets: Targets::Classes(vec![0, 1, 0, 1]) };
    /// if let Some(cuda) = Cuda::new() {
    ///     let mut model = CudaModel::compile(&net, &params, 3, &cuda).unwrap();
    ///     assert_eq!(model.accuracy(&data), 0.5);
    /// }
    /// ```
    pub fn accuracy(&mut self, data: &Dataset) -> f64 {
        let Targets::Classes(labels) = &data.targets else {
            panic!("CudaModel::accuracy: the data has value targets; accuracy needs class labels");
        };
        let (rows, features) = (data.len(), data.features());
        assert!(rows > 0, "CudaModel::accuracy: the data has no rows");
        assert_eq!(
            labels.len(),
            rows,
            "CudaModel::accuracy: the data has {rows} rows but {} labels",
            labels.len()
        );
        if let Err(e) = self.check_rows(data) {
            panic!("CudaModel::accuracy: {e}");
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
                    "CudaModel::accuracy: the label of row {} is {label}, but the model has \
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

    /// Runs one full batch `x` through the evaluation pass and reads back every layer's
    /// output: for the tests.
    #[cfg(test)]
    pub(crate) fn infer_layers(&mut self, x: &[f32]) -> Vec<Vec<f32>> {
        assert_eq!(x.len(), self.batch * self.input_width);
        self.store.get_mut(self.eval.input).write(x).unwrap();
        run(
            &self.eval.ops,
            &mut self.eval.kernels,
            &self.shared,
            &self.store,
        )
        .unwrap();
        (self.eval.outputs.iter().zip(&self.layers))
            .map(|(&at, layer)| {
                let mut out = vec![0.0f32; self.batch * layer.outputs];
                self.store.get(at).read(&mut out).unwrap();
                out
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
                "the inputs are {dims:?}, but the CUDA model takes rows of {} values",
                self.input_width
            )));
        }
        if self.eval.batch_statistics && !dims[0].is_multiple_of(self.batch) {
            return Err(Error::InvalidData(format!(
                "a batch-norm layer of the CUDA model normalises with the batch's statistics, \
                 so the {} rows must be whole batches of {}",
                dims[0], self.batch
            )));
        }
        Ok(())
    }
}
