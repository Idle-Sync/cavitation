//! The fully connected layer's passes through cuBLAS: `y = x · Wᵀ + b` and its gradients.

use std::ffi::c_int;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::sys::{CUBLAS_OP_N, CUBLAS_OP_T};

/// The shape of a fully connected layer over a batch: the input `[batch, inputs]`, the
/// weight `[outputs, inputs]` and the output `[batch, outputs]`, all row-major, as the
/// CPU's `Linear` stores them.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::LinearShape;
///
/// let shape = LinearShape { batch: 128, inputs: 4096, outputs: 256 };
/// assert_eq!(shape.w_len(), 4096 * 256);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearShape {
    /// Rows in the batch.
    pub batch: u32,
    /// Input columns.
    pub inputs: u32,
    /// Output columns.
    pub outputs: u32,
}

impl LinearShape {
    /// The input's length in floats: batch × inputs.
    pub fn x_len(&self) -> usize {
        self.batch as usize * self.inputs as usize
    }

    /// The weight's length in floats: outputs × inputs.
    pub fn w_len(&self) -> usize {
        self.outputs as usize * self.inputs as usize
    }

    /// The output's length in floats: batch × outputs.
    pub fn y_len(&self) -> usize {
        self.batch as usize * self.outputs as usize
    }
}

/// One fully connected layer shape on the card, through cuBLAS's `f32` matrix products in
/// pedantic math (the prescribed precision in every phase, no tensor cores): the forward
/// pass `y = x · Wᵀ + b`, the input gradient `dx = dy · W`, the weight gradient
/// `dW = dyᵀ · x` and the bias gradient, `dy` summed over the batch.
///
/// cuBLAS publishes no error bound for how it orders a product's sums, so how far each pass
/// is from the exact result is measured, not proven: see this crate's `tests/linear.rs`.
///
/// Every pass is queued on the card and returns without waiting for it. A layer is set up
/// for one batch size: a run whose last batch is shorter needs a second one for it.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, Linear, LinearShape};
///
/// if let Some(cuda) = Cuda::new() {
///     // Two rows of two inputs, three outputs.
///     let shape = LinearShape { batch: 2, inputs: 2, outputs: 3 };
///     let layer = Linear::new(&cuda, shape).unwrap();
///     let mut x = cuda.buffer(4 * 4).unwrap();
///     let mut w = cuda.buffer(6 * 4).unwrap();
///     let mut b = cuda.buffer(3 * 4).unwrap();
///     let mut y = cuda.buffer(6 * 4).unwrap();
///     x.write(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
///     w.write(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
///     b.write(&[0.5f32, 0.0, -1.0]).unwrap();
///     layer.forward(&x, &w, Some(&b), &mut y).unwrap();
///     let mut out = [0.0f32; 6];
///     y.read(&mut out).unwrap();
///     assert_eq!(out, [1.5, 2.0, 2.0, 3.5, 4.0, 6.0]);
/// }
/// ```
#[derive(Debug)]
pub struct Linear<'c> {
    /// The CUDA it runs on.
    cuda: &'c Cuda,
    /// Its shape.
    shape: LinearShape,
    /// The shape's sizes as cuBLAS's `int`: batch, inputs, outputs.
    sizes: [c_int; 3],
    /// `batch` ones: the bias is added as `ones · bᵀ`, and its gradient is `dyᵀ · ones`.
    ones: DeviceBuffer<'c>,
}

/// `1.0` and `0.0` as cuBLAS's host scaling factors.
const ONE: f32 = 1.0;
/// See [`ONE`].
const ZERO: f32 = 0.0;

impl<'c> Linear<'c> {
    /// Sets up the layer `shape` on `cuda`'s card.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if a size is zero or a matrix does not fit cuBLAS's `int`;
    /// [`CudaError::Call`] if the ones cannot be allocated.
    pub fn new(cuda: &'c Cuda, shape: LinearShape) -> Result<Linear<'c>, CudaError> {
        let invalid = |why: &str| CudaError::Invalid(format!("linear layer {shape:?}: {why}"));
        if shape.batch == 0 || shape.inputs == 0 || shape.outputs == 0 {
            return Err(invalid("every size must be positive"));
        }
        let int =
            |v: usize| c_int::try_from(v).map_err(|_| invalid("a matrix is too large for cuBLAS"));
        for len in [shape.x_len(), shape.w_len(), shape.y_len()] {
            int(len)?;
        }
        let sizes = [
            int(shape.batch as usize)?,
            int(shape.inputs as usize)?,
            int(shape.outputs as usize)?,
        ];
        let mut ones = cuda.buffer(shape.batch as usize * 4)?;
        ones.write(&vec![1.0f32; shape.batch as usize])?;
        Ok(Linear {
            cuda,
            shape,
            sizes,
            ones,
        })
    }

    /// The layer's shape.
    pub fn shape(&self) -> LinearShape {
        self.shape
    }

    /// Asserts that `buffer` holds at least `floats` floats.
    fn fits(buffer: &DeviceBuffer<'_>, floats: usize, what: &str) {
        let needs = floats.saturating_mul(4);
        assert!(
            buffer.bytes() >= needs,
            "the {what} buffer holds {} bytes, and the layer needs {needs}",
            buffer.bytes()
        );
    }

    /// One `C = alpha · op(A) · op(B) + beta · C` on column-major matrices.
    ///
    /// # Safety
    ///
    /// The buffers must hold the matrices the sizes and leading dimensions describe, and
    /// `c` must not be `a` or `b`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        &self,
        what: &'static str,
        trans: (c_int, c_int),
        (m, n, k): (c_int, c_int, c_int),
        (a, lda): (&DeviceBuffer<'_>, c_int),
        (b, ldb): (&DeviceBuffer<'_>, c_int),
        beta: f32,
        (c, ldc): (&mut DeviceBuffer<'_>, c_int),
    ) -> Result<(), CudaError> {
        // SAFETY: the handle is live and the scalars are host floats, as the handle's
        // default pointer mode reads them; the buffers are as this function's contract asks.
        let status = unsafe {
            (self.cuda.cublas.cublasSgemm_v2)(
                self.cuda.cublas_handle,
                trans.0,
                trans.1,
                m,
                n,
                k,
                &ONE,
                a.pointer,
                lda,
                b.pointer,
                ldb,
                &beta,
                c.pointer,
                ldc,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(self.cuda.cublas_error(what, status))
        }
    }

    /// `y = x · Wᵀ`, plus `bias` on every row when given: `x` `[batch, inputs]`, `w`
    /// `[outputs, inputs]`, `bias` `[outputs]`, `y` `[batch, outputs]`, row-major. Every
    /// output is written. With a bias, each output is the bias plus the product's sum.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if cuBLAS refuses the call.
    ///
    /// # Panics
    ///
    /// If a buffer is smaller than its matrix.
    pub fn forward(
        &self,
        x: &DeviceBuffer<'_>,
        w: &DeviceBuffer<'_>,
        bias: Option<&DeviceBuffer<'_>>,
        y: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let s = &self.shape;
        Self::fits(x, s.x_len(), "input");
        Self::fits(w, s.w_len(), "weight");
        Self::fits(y, s.y_len(), "output");
        if let Some(b) = bias {
            Self::fits(b, s.outputs as usize, "bias");
        }
        self.cuda.bind()?;
        let [batch, inputs, outputs] = self.sizes;
        // Row-major `y` is the column-major `outputs × batch` matrix `yᵀ = W · xᵀ`, where
        // row-major `W` is column-major `Wᵀ` (inputs × outputs) and row-major `x` is
        // column-major `xᵀ` (inputs × batch).
        let beta = if let Some(b) = bias {
            // `yᵀ = b · onesᵀ`, products of one and exact.
            // SAFETY: `b` holds `outputs` floats, the ones `batch`, and `y` `outputs ×
            // batch`, borrowed mutably.
            unsafe {
                self.gemm(
                    "cublasSgemm_v2",
                    (CUBLAS_OP_N, CUBLAS_OP_N),
                    (outputs, batch, 1),
                    (b, outputs),
                    (&self.ones, 1),
                    0.0,
                    (y, outputs),
                )
            }?;
            1.0
        } else {
            0.0
        };
        // SAFETY: `w` holds `inputs × outputs` floats, `x` `inputs × batch` and `y`
        // `outputs × batch`, borrowed mutably; with beta 0 `y` is not read.
        unsafe {
            self.gemm(
                "cublasSgemm_v2",
                (CUBLAS_OP_T, CUBLAS_OP_N),
                (outputs, batch, inputs),
                (w, inputs),
                (x, inputs),
                beta,
                (y, outputs),
            )
        }
    }

    /// `dx = dy · W`, the input's gradient: `w` `[outputs, inputs]`, `dy` `[batch,
    /// outputs]`, `dx` `[batch, inputs]`, row-major. Every value of `dx` is written.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if cuBLAS refuses the call.
    ///
    /// # Panics
    ///
    /// If a buffer is smaller than its matrix.
    pub fn input_grad(
        &self,
        w: &DeviceBuffer<'_>,
        dy: &DeviceBuffer<'_>,
        dx: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let s = &self.shape;
        Self::fits(w, s.w_len(), "weight");
        Self::fits(dy, s.y_len(), "output gradient");
        Self::fits(dx, s.x_len(), "input gradient");
        self.cuda.bind()?;
        let [batch, inputs, outputs] = self.sizes;
        // Column-major `dxᵀ` (inputs × batch) = `Wᵀ` (inputs × outputs) · `dyᵀ`.
        // SAFETY: `w` holds `inputs × outputs` floats, `dy` `outputs × batch` and `dx`
        // `inputs × batch`, borrowed mutably; with beta 0 `dx` is not read.
        unsafe {
            self.gemm(
                "cublasSgemm_v2",
                (CUBLAS_OP_N, CUBLAS_OP_N),
                (inputs, batch, outputs),
                (w, inputs),
                (dy, outputs),
                0.0,
                (dx, inputs),
            )
        }
    }

    /// `dW = dyᵀ · x`, the weight's gradient summed over the batch: `x` `[batch, inputs]`,
    /// `dy` `[batch, outputs]`, `dw` `[outputs, inputs]`, row-major. Every value of `dw` is
    /// written.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if cuBLAS refuses the call.
    ///
    /// # Panics
    ///
    /// If a buffer is smaller than its matrix.
    pub fn weight_grad(
        &self,
        x: &DeviceBuffer<'_>,
        dy: &DeviceBuffer<'_>,
        dw: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let s = &self.shape;
        Self::fits(x, s.x_len(), "input");
        Self::fits(dy, s.y_len(), "output gradient");
        Self::fits(dw, s.w_len(), "weight gradient");
        self.cuda.bind()?;
        let [batch, inputs, outputs] = self.sizes;
        // Column-major `dWᵀ` (inputs × outputs) = `xᵀ` (inputs × batch) · `dy` (batch ×
        // outputs), the transpose of column-major `dyᵀ`.
        // SAFETY: `x` holds `inputs × batch` floats, `dy` `outputs × batch` and `dw`
        // `inputs × outputs`, borrowed mutably; with beta 0 `dw` is not read.
        unsafe {
            self.gemm(
                "cublasSgemm_v2",
                (CUBLAS_OP_N, CUBLAS_OP_T),
                (inputs, outputs, batch),
                (x, inputs),
                (dy, outputs),
                0.0,
                (dw, inputs),
            )
        }
    }

    /// `db`, the bias's gradient: `dy` (`[batch, outputs]`, row-major) summed over the
    /// batch, one value per output. Every value of `db` is written.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if cuBLAS refuses the call.
    ///
    /// # Panics
    ///
    /// If a buffer is smaller than its matrix.
    pub fn bias_grad(
        &self,
        dy: &DeviceBuffer<'_>,
        db: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let s = &self.shape;
        Self::fits(dy, s.y_len(), "output gradient");
        Self::fits(db, s.outputs as usize, "bias gradient");
        self.cuda.bind()?;
        let [batch, _, outputs] = self.sizes;
        // `db = dyᵀ · ones`, with column-major `dyᵀ` (outputs × batch).
        // SAFETY: the handle is live, `dy` holds `outputs × batch` floats, the ones
        // `batch`, and `db` `outputs`, borrowed mutably; with beta 0 `db` is not read.
        let status = unsafe {
            (self.cuda.cublas.cublasSgemv_v2)(
                self.cuda.cublas_handle,
                CUBLAS_OP_N,
                outputs,
                batch,
                &ONE,
                dy.pointer,
                outputs,
                self.ones.pointer,
                1,
                &ZERO,
                db.pointer,
                1,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(self.cuda.cublas_error("cublasSgemv_v2", status))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lengths are the row-major matrices' element counts.
    #[test]
    fn lengths() {
        let s = LinearShape {
            batch: 3,
            inputs: 5,
            outputs: 2,
        };
        assert_eq!((s.x_len(), s.w_len(), s.y_len()), (15, 10, 6));
    }
}
