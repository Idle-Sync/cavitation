//! What the CUDA engine's passes are made of: the [`Store`] of the card's buffers a model and
//! its trainer share, each layer's kernel calls as [`Op`]s on store indices, and the forward
//! pass of a compiled model for one number of rows, in evaluation or training mode.

use std::cell::{Ref, RefCell, RefMut};

use cavitation_core::{Conv2dGeom, Pool2dGeom};
use cavitation_cuda::{
    AvgPool2d, BatchNorm2d, BatchStats, ChannelDims, Conv2d, ConvShape, Cuda, CudaError,
    DeviceBuffer, Dropout, Elementwise, ElementwiseOp, Linear, LinearShape, MaxPool2d, PoolShape,
};

use crate::plan::{Act, CompiledLayer, Kind};

/// Every buffer a model and its trainer hold on the card, by index: a kernel call names its
/// buffers by their indices, and borrows them for the call alone.
pub(crate) struct Store<'c> {
    /// The CUDA the buffers are on.
    cuda: &'c Cuda,
    /// The buffers.
    buffers: Vec<RefCell<DeviceBuffer<'c>>>,
}

impl<'c> Store<'c> {
    /// An empty store on `cuda`.
    pub(crate) fn new(cuda: &'c Cuda) -> Store<'c> {
        Store {
            cuda,
            buffers: Vec::new(),
        }
    }

    /// The CUDA the buffers are on.
    pub(crate) fn cuda(&self) -> &'c Cuda {
        self.cuda
    }

    /// A new zeroed buffer of `n` four-byte values, and its index.
    pub(crate) fn values(&mut self, n: usize) -> Result<usize, CudaError> {
        let buffer = self.cuda.buffer(4 * n)?;
        self.buffers.push(RefCell::new(buffer));
        Ok(self.buffers.len() - 1)
    }

    /// Buffer `i`, to read.
    pub(crate) fn get(&self, i: usize) -> Ref<'_, DeviceBuffer<'c>> {
        self.buffers[i].borrow()
    }

    /// Buffer `i`, to write.
    pub(crate) fn get_mut(&self, i: usize) -> RefMut<'_, DeviceBuffer<'c>> {
        self.buffers[i].borrow_mut()
    }

    /// Copies `n` four-byte values of buffer `from`, starting at value `from_at`, into
    /// buffer `to` at value `at`.
    pub(crate) fn copy(
        &self,
        (to, at): (usize, usize),
        (from, from_at): (usize, usize),
        n: usize,
    ) -> Result<(), CudaError> {
        self.get_mut(to)
            .copy_from(4 * at, &self.get(from), 4 * from_at, 4 * n)
    }
}

/// The element-wise kernels every pass of a model shares.
#[derive(Debug)]
pub(crate) struct Shared<'c> {
    /// The activations, forward and backward.
    pub(crate) elementwise: Elementwise<'c>,
    /// Dropout and the neuron masks.
    pub(crate) dropout: Dropout<'c>,
}

/// The kernels of one pass that are set up for its shapes, which its [`Op`]s name by index.
#[derive(Debug, Default)]
pub(crate) struct Kernels<'c> {
    /// Each `Linear`'s.
    pub(crate) linears: Vec<Linear<'c>>,
    /// Each convolution's.
    pub(crate) convs: Vec<Conv2d<'c>>,
    /// Each max pooling's.
    pub(crate) max_pools: Vec<MaxPool2d<'c>>,
    /// Each average pooling's.
    pub(crate) avg_pools: Vec<AvgPool2d<'c>>,
    /// Each batch norm's, which remembers its last forward pass for the backward pass and the
    /// running update.
    pub(crate) norms: Vec<BatchNorm2d<'c>>,
}

/// One kernel call, with its kernel (`k`, an index into [`Kernels`]) and its buffers
/// (indices into the [`Store`]). The names are the passes': `x` an input, `y` an output, `w`
/// and `b` a weight and a bias, and `d` before a name its gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::missing_docs_in_private_items)] // The fields are named as the enum says.
pub(crate) enum Op {
    /// `y = x · wᵀ + b`.
    Linear {
        k: usize,
        x: usize,
        w: usize,
        b: Option<usize>,
        y: usize,
    },
    /// `dw = dyᵀ · x`.
    LinearWeightGrad {
        k: usize,
        x: usize,
        dy: usize,
        dw: usize,
    },
    /// `db`, `dy` summed over the rows.
    LinearBiasGrad { k: usize, dy: usize, db: usize },
    /// `dx = dy · w`.
    LinearInputGrad {
        k: usize,
        w: usize,
        dy: usize,
        dx: usize,
    },
    /// The convolution, with its bias.
    Conv {
        k: usize,
        x: usize,
        w: usize,
        b: Option<usize>,
        y: usize,
    },
    /// The convolution's weight gradient.
    ConvWeightGrad {
        k: usize,
        x: usize,
        dy: usize,
        dw: usize,
    },
    /// The convolution's bias gradient.
    ConvBiasGrad { k: usize, dy: usize, db: usize },
    /// The convolution's input gradient.
    ConvInputGrad {
        k: usize,
        w: usize,
        dy: usize,
        dx: usize,
    },
    /// The element-wise operation `op` of `n` values of `a` (and `b`), with `alpha`: an
    /// activation or its backward pass.
    Elementwise {
        op: ElementwiseOp,
        alpha: f32,
        n: usize,
        a: usize,
        b: Option<usize>,
        y: usize,
    },
    /// Dropout of `n` values with the bits `mask` and the scale `scale`, forward or backward.
    Dropout {
        n: usize,
        x: usize,
        mask: usize,
        y: usize,
        scale: f32,
    },
    /// A neuron mask's bits `mask` on `n` values, forward or backward.
    KeepWhere {
        n: usize,
        x: usize,
        mask: usize,
        y: usize,
    },
    /// Max pooling, with each output's winner.
    MaxPool {
        k: usize,
        x: usize,
        y: usize,
        winners: usize,
    },
    /// Max pooling's backward pass, which adds to `dx`.
    MaxPoolBackward {
        k: usize,
        dy: usize,
        winners: usize,
        dx: usize,
    },
    /// Average pooling.
    AvgPool { k: usize, x: usize, y: usize },
    /// Average pooling's backward pass, which adds to `dx`.
    AvgPoolBackward { k: usize, dy: usize, dx: usize },
    /// Batch norm with `eps`, from the `running` statistics `(mean, var)` or, with `None`,
    /// the batch's, with `affine` `(γ, β)` and the statistics written to `stats`.
    NormForward {
        k: usize,
        eps: f64,
        running: Option<(usize, usize)>,
        x: usize,
        affine: Option<(usize, usize)>,
        y: usize,
        stats: usize,
    },
    /// Batch norm's backward pass, with `affine` `(γ, dγ, dβ)` for an affine layer.
    NormBackward {
        k: usize,
        x: usize,
        stats: usize,
        dy: usize,
        dx: usize,
        affine: Option<(usize, usize, usize)>,
    },
}

/// Runs `ops` in order.
pub(crate) fn run(
    ops: &[Op],
    kernels: &mut Kernels<'_>,
    shared: &Shared<'_>,
    s: &Store<'_>,
) -> Result<(), CudaError> {
    for op in ops {
        run_op(op, kernels, shared, s)?;
    }
    Ok(())
}

/// Runs one call.
fn run_op(
    op: &Op,
    kernels: &mut Kernels<'_>,
    shared: &Shared<'_>,
    s: &Store<'_>,
) -> Result<(), CudaError> {
    match *op {
        Op::Linear { k, x, w, b, y } => {
            let bias = b.map(|b| s.get(b));
            kernels.linears[k].forward(&s.get(x), &s.get(w), bias.as_deref(), &mut s.get_mut(y))
        }
        Op::LinearWeightGrad { k, x, dy, dw } => {
            kernels.linears[k].weight_grad(&s.get(x), &s.get(dy), &mut s.get_mut(dw))
        }
        Op::LinearBiasGrad { k, dy, db } => {
            kernels.linears[k].bias_grad(&s.get(dy), &mut s.get_mut(db))
        }
        Op::LinearInputGrad { k, w, dy, dx } => {
            kernels.linears[k].input_grad(&s.get(w), &s.get(dy), &mut s.get_mut(dx))
        }
        Op::Conv { k, x, w, b, y } => {
            let bias = b.map(|b| s.get(b));
            kernels.convs[k].forward(&s.get(x), &s.get(w), bias.as_deref(), &mut s.get_mut(y))
        }
        Op::ConvWeightGrad { k, x, dy, dw } => {
            kernels.convs[k].weight_grad(&s.get(x), &s.get(dy), &mut s.get_mut(dw))
        }
        Op::ConvBiasGrad { k, dy, db } => {
            kernels.convs[k].bias_grad(&s.get(dy), &mut s.get_mut(db))
        }
        Op::ConvInputGrad { k, w, dy, dx } => {
            kernels.convs[k].input_grad(&s.get(w), &s.get(dy), &mut s.get_mut(dx))
        }
        Op::Elementwise {
            op,
            alpha,
            n,
            a,
            b,
            y,
        } => {
            let second = b.map(|b| s.get(b));
            shared.elementwise.run(
                op,
                &s.get(a),
                second.as_deref(),
                &mut s.get_mut(y),
                n,
                alpha,
            )
        }
        Op::Dropout {
            n,
            x,
            mask,
            y,
            scale,
        } => shared
            .dropout
            .dropout(&s.get(x), &s.get(mask), &mut s.get_mut(y), n, scale),
        Op::KeepWhere { n, x, mask, y } => {
            shared
                .dropout
                .keep_where(&s.get(x), &s.get(mask), &mut s.get_mut(y), n)
        }
        Op::MaxPool { k, x, y, winners } => {
            kernels.max_pools[k].forward(&s.get(x), &mut s.get_mut(y), &mut s.get_mut(winners))
        }
        Op::MaxPoolBackward { k, dy, winners, dx } => {
            kernels.max_pools[k].backward(&s.get(dy), &s.get(winners), &mut s.get_mut(dx))
        }
        Op::AvgPool { k, x, y } => kernels.avg_pools[k].forward(&s.get(x), &mut s.get_mut(y)),
        Op::AvgPoolBackward { k, dy, dx } => {
            kernels.avg_pools[k].backward(&s.get(dy), &mut s.get_mut(dx))
        }
        Op::NormForward {
            k,
            eps,
            running,
            x,
            affine,
            y,
            stats,
        } => {
            let running = running.map(|(m, v)| (s.get(m), s.get(v)));
            let from = match &running {
                Some((mean, var)) => BatchStats::Running { mean, var },
                None => BatchStats::Batch,
            };
            let affine = affine.map(|(w, b)| (s.get(w), s.get(b)));
            kernels.norms[k].forward(
                eps,
                from,
                &s.get(x),
                affine.as_ref().map(|(w, b)| (&**w, &**b)),
                &mut s.get_mut(y),
                &mut s.get_mut(stats),
            )
        }
        Op::NormBackward {
            k,
            x,
            stats,
            dy,
            dx,
            affine,
        } => {
            let mut affine = affine.map(|(w, dw, db)| (s.get(w), s.get_mut(dw), s.get_mut(db)));
            kernels.norms[k].backward(
                &s.get(x),
                &s.get(stats),
                &s.get(dy),
                &mut s.get_mut(dx),
                affine
                    .as_mut()
                    .map(|(w, dw, db)| (&**w, &mut **dw, &mut **db)),
            )
        }
    }
}

/// A dropout layer's mask for a step: the layer's stream, drop probability and the packed
/// bits on the card.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DropoutMask {
    /// The tape's random stream for the layer: its rank among the model's dropout layers.
    pub(crate) stream: u64,
    /// The drop probability.
    pub(crate) p: f64,
    /// The values the mask covers: the pass's rows times the layer's width.
    pub(crate) len: usize,
    /// The bits, one per value, an index into the store.
    pub(crate) bits: usize,
}

/// What the backward pass needs of one layer's forward pass.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Saved {
    /// The layer's input, an index into the store.
    pub(crate) input: usize,
    /// Its output before its mask.
    pub(crate) pre: usize,
    /// Its kernel's index in [`Kernels`], for a `Linear`, a convolution, a pooling or a
    /// batch norm.
    pub(crate) kernel: Option<usize>,
    /// Max pooling's winners, or batch norm's statistics, in the store.
    pub(crate) scratch: Option<usize>,
    /// Its dropout mask's index in [`Forward::dropouts`].
    pub(crate) dropout: Option<usize>,
}

/// The store indices of a model's state, which a pass reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Parts<'a> {
    /// The layers.
    pub(crate) layers: &'a [CompiledLayer],
    /// Each trained parameter's buffer.
    pub(crate) params: &'a [usize],
    /// Each running statistic's buffer.
    pub(crate) buffers: &'a [usize],
    /// Each neuron mask's bits.
    pub(crate) masks: &'a [usize],
    /// The width of one input row.
    pub(crate) input_width: usize,
}

/// A forward pass of a model for `rows` rows: its kernels, its calls in order and the buffers
/// they read and write.
#[derive(Debug)]
pub(crate) struct Forward<'c> {
    /// The kernels set up for the rows.
    pub(crate) kernels: Kernels<'c>,
    /// The calls, in order.
    pub(crate) ops: Vec<Op>,
    /// Where each layer's calls end in `ops`, in layer order.
    pub(crate) ends: Vec<usize>,
    /// The input, `rows × input_width` values.
    pub(crate) input: usize,
    /// The model's output.
    pub(crate) output: usize,
    /// Each layer's output as the next layer reads it (after its mask, if it has one).
    #[allow(dead_code)] // The tests read it.
    pub(crate) outputs: Vec<usize>,
    /// What each layer's backward pass needs.
    pub(crate) saved: Vec<Saved>,
    /// The dropout masks, in the model's order (training only).
    pub(crate) dropouts: Vec<DropoutMask>,
    /// Whether a batch norm normalises with the batch's statistics in evaluation (it keeps
    /// no running ones), so that every row counts.
    pub(crate) batch_statistics: bool,
}

/// `n` as a `u32`, for a size of a layer, or [`CudaError::Invalid`] naming it.
fn u32_of(n: usize, what: &str) -> Result<u32, CudaError> {
    u32::try_from(n).map_err(|_| CudaError::Invalid(format!("{what} {n} does not fit a u32")))
}

/// A `(height, width)` as `u32`s.
fn pair(p: (usize, usize), what: &str) -> Result<(u32, u32), CudaError> {
    Ok((u32_of(p.0, what)?, u32_of(p.1, what)?))
}

/// The convolution `geom` over `rows` images.
pub(crate) fn conv_shape(rows: usize, geom: &Conv2dGeom) -> Result<ConvShape, CudaError> {
    Ok(ConvShape {
        stride: pair(geom.stride(), "a stride")?,
        padding: pair(geom.padding(), "a padding")?,
        dilation: pair(geom.dilation(), "a dilation")?,
        ..ConvShape::new(
            u32_of(rows, "a batch of")?,
            u32_of(geom.in_channels(), "a channel count")?,
            u32_of(geom.out_channels(), "a channel count")?,
            pair(geom.input(), "an image size")?,
            pair(geom.kernel(), "a kernel size")?,
        )
    })
}

/// The pooling `geom` over `rows` images.
fn pool_shape(rows: usize, geom: &Pool2dGeom) -> Result<PoolShape, CudaError> {
    Ok(PoolShape {
        stride: pair(geom.stride(), "a stride")?,
        padding: pair(geom.padding(), "a padding")?,
        dilation: pair(geom.dilation(), "a dilation")?,
        ceil_mode: geom.ceil_mode(),
        ..PoolShape::new(
            u32_of(rows, "a batch of")?,
            u32_of(geom.channels(), "a channel count")?,
            pair(geom.input(), "an image size")?,
            pair(geom.kernel(), "a window size")?,
        )
    })
}

/// A batch norm's shape over `rows` images.
pub(crate) fn norm_dims(
    rows: usize,
    channels: usize,
    spatial: usize,
) -> Result<ChannelDims, CudaError> {
    Ok(ChannelDims::new(
        u32_of(rows, "a batch of")?,
        u32_of(channels, "a channel count")?,
        u32_of(spatial, "a channel's size")?,
    ))
}

/// An activation's forward operation and its `alpha`.
pub(crate) fn forward_op(act: Act) -> (ElementwiseOp, f32) {
    match act {
        Act::Relu => (ElementwiseOp::Relu, 0.0),
        Act::LeakyRelu(slope) => (ElementwiseOp::LeakyRelu, slope as f32),
        Act::Tanh => (ElementwiseOp::Tanh, 0.0),
        Act::Sigmoid => (ElementwiseOp::Sigmoid, 0.0),
        Act::GeluTanh => (ElementwiseOp::GeluTanh, 0.0),
        Act::GeluErf => (ElementwiseOp::GeluErf, 0.0),
    }
}

/// An activation's backward operation, its `alpha`, and whether it reads the forward output
/// (tanh and the sigmoid) rather than the input.
pub(crate) fn backward_op(act: Act) -> (ElementwiseOp, f32, bool) {
    match act {
        Act::Relu => (ElementwiseOp::ReluBackward, 0.0, false),
        Act::LeakyRelu(slope) => (ElementwiseOp::LeakyReluBackward, slope as f32, false),
        Act::Tanh => (ElementwiseOp::TanhBackward, 0.0, true),
        Act::Sigmoid => (ElementwiseOp::SigmoidBackward, 0.0, true),
        Act::GeluTanh => (ElementwiseOp::GeluTanhBackward, 0.0, false),
        Act::GeluErf => (ElementwiseOp::GeluErfBackward, 0.0, false),
    }
}

impl<'c> Forward<'c> {
    /// Sets up the forward pass of `parts` for `rows` rows on the store's card, with its
    /// buffers added to `store`. In evaluation mode dropout, flatten and identity pass their
    /// input on and batch norm uses its running statistics (or, without them, the batch's);
    /// in training mode dropout multiplies by a mask the step uploads and batch norm uses
    /// the batch's statistics. Each neuron mask is applied to its layer's output.
    pub(crate) fn build(
        store: &mut Store<'c>,
        parts: &Parts<'_>,
        rows: usize,
        training: bool,
    ) -> Result<Forward<'c>, CudaError> {
        let cuda = store.cuda();
        let mut k = Kernels::default();
        let input = store.values(rows * parts.input_width)?;
        let mut ops = Vec::new();
        let mut ends = Vec::with_capacity(parts.layers.len());
        let mut outputs = Vec::with_capacity(parts.layers.len());
        let mut saved = Vec::with_capacity(parts.layers.len());
        let mut dropouts = Vec::new();
        let mut batch_statistics = false;
        let mut x = input;
        for layer in parts.layers {
            let n = rows * layer.outputs;
            let mut s = Saved {
                input: x,
                pre: x,
                ..Saved::default()
            };
            let out = match layer.kind {
                Kind::Linear {
                    inputs,
                    outputs,
                    weight,
                    bias,
                } => {
                    let shape = LinearShape {
                        batch: u32_of(rows, "a batch of")?,
                        inputs: u32_of(inputs, "a width of")?,
                        outputs: u32_of(outputs, "a width of")?,
                    };
                    k.linears.push(Linear::new(cuda, shape)?);
                    let y = store.values(n)?;
                    s.kernel = Some(k.linears.len() - 1);
                    ops.push(Op::Linear {
                        k: k.linears.len() - 1,
                        x,
                        w: parts.params[weight],
                        b: bias.map(|b| parts.params[b]),
                        y,
                    });
                    y
                }
                Kind::Conv2d { geom, weight, bias } => {
                    k.convs.push(Conv2d::new(cuda, conv_shape(rows, &geom)?)?);
                    let y = store.values(n)?;
                    s.kernel = Some(k.convs.len() - 1);
                    ops.push(Op::Conv {
                        k: k.convs.len() - 1,
                        x,
                        w: parts.params[weight],
                        b: bias.map(|b| parts.params[b]),
                        y,
                    });
                    y
                }
                Kind::Activation(act) => {
                    let y = store.values(n)?;
                    let (op, alpha) = forward_op(act);
                    ops.push(Op::Elementwise {
                        op,
                        alpha,
                        n,
                        a: x,
                        b: None,
                        y,
                    });
                    y
                }
                Kind::MaxPool2d(geom) => {
                    k.max_pools
                        .push(MaxPool2d::new(cuda, pool_shape(rows, &geom)?)?);
                    let (y, winners) = (store.values(n)?, store.values(n)?);
                    s.kernel = Some(k.max_pools.len() - 1);
                    s.scratch = Some(winners);
                    ops.push(Op::MaxPool {
                        k: k.max_pools.len() - 1,
                        x,
                        y,
                        winners,
                    });
                    y
                }
                Kind::AvgPool2d {
                    geom,
                    count_include_pad,
                    divisor_override,
                } => {
                    let divisor = divisor_override
                        .map(|d| u32_of(d, "a divisor of"))
                        .transpose()?;
                    let shape = pool_shape(rows, &geom)?;
                    k.avg_pools
                        .push(AvgPool2d::new(cuda, shape, count_include_pad, divisor)?);
                    let y = store.values(n)?;
                    s.kernel = Some(k.avg_pools.len() - 1);
                    ops.push(Op::AvgPool {
                        k: k.avg_pools.len() - 1,
                        x,
                        y,
                    });
                    y
                }
                Kind::BatchNorm2d(norm) => {
                    let dims = norm_dims(rows, norm.channels, norm.spatial)?;
                    k.norms.push(BatchNorm2d::new(cuda, dims)?);
                    let (y, stats) = (store.values(n)?, store.values(3 * norm.channels)?);
                    let running = match norm.running {
                        Some(r) if !training => Some((parts.buffers[r.mean], parts.buffers[r.var])),
                        _ => {
                            batch_statistics |= !training;
                            None
                        }
                    };
                    let affine = match (norm.weight, norm.bias) {
                        (Some(w), Some(b)) => Some((parts.params[w], parts.params[b])),
                        _ => None,
                    };
                    s.kernel = Some(k.norms.len() - 1);
                    s.scratch = Some(stats);
                    ops.push(Op::NormForward {
                        k: k.norms.len() - 1,
                        eps: norm.eps,
                        running,
                        x,
                        affine,
                        y,
                        stats,
                    });
                    y
                }
                Kind::Dropout(p) if training => {
                    let bits = store.values(Dropout::mask_words(n))?;
                    dropouts.push(DropoutMask {
                        stream: dropouts.len() as u64,
                        p,
                        len: n,
                        bits,
                    });
                    let y = store.values(n)?;
                    s.dropout = Some(dropouts.len() - 1);
                    ops.push(Op::Dropout {
                        n,
                        x,
                        mask: bits,
                        y,
                        scale: Dropout::scale(p)?,
                    });
                    y
                }
                Kind::Dropout(_) | Kind::Identity => x,
            };
            s.pre = out;
            x = match layer.mask {
                Some(m) => {
                    let y = store.values(n)?;
                    ops.push(Op::KeepWhere {
                        n,
                        x: out,
                        mask: parts.masks[m],
                        y,
                    });
                    y
                }
                None => out,
            };
            outputs.push(x);
            saved.push(s);
            ends.push(ops.len());
        }
        Ok(Forward {
            kernels: k,
            ops,
            ends,
            input,
            output: x,
            outputs,
            saved,
            dropouts,
            batch_statistics,
        })
    }
}
