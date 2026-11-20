//! The plan both GPU engines compile a [`Sequential`] from, with the `gpu` or the `cuda`
//! feature: every layer checked against the forms a GPU runs and refused by index and name
//! otherwise, the widths flowing through, the parameters, running statistics and batch counts
//! laid out once each, and the neuron masks packed for the batch. Nothing here touches a
//! device.

use std::collections::HashMap;

use cavitation_core::{Conv2dGeom, Error, GeluApprox, ParamId, ParamStore, Pool2dGeom};

use crate::layer::private::Token;
use crate::{ActivationKind, InferenceLayer, Layer, Sequential};

/// The most images a batch through a convolution may hold on the wgpu engine: its
/// convolution kernels dispatch one row of workgroups per image, and a dispatch dimension
/// holds at most 65,535.
const MAX_CONV_BATCH: usize = 65_535;

/// The most rows or columns a GEMM's output may have on the wgpu engine: 65,535 tiles of 64,
/// one workgroup each along a dispatch dimension.
const MAX_GEMM_SIDE: usize = 65_535 * 64;

/// The most rows a batch through a softmax or log-softmax may hold on the wgpu engine: its
/// kernels run one thread per row, in one dispatch dimension of 65,535 workgroups of 256.
const MAX_SOFTMAX_ROWS: usize = 65_535 * 256;

/// The engine a plan is built for: the name its messages start with, and whether the wgpu
/// engine's own limits apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Engine {
    /// What the messages name, such as `GpuModel::compile`.
    pub(crate) name: &'static str,
    /// Whether the wgpu kernels' limits are checked: a convolution's and a softmax's batch,
    /// a GEMM's sides, and a batch norm's eps that `f32` arithmetic flushing subnormals
    /// would lose.
    pub(crate) wgpu_limits: bool,
}

impl Engine {
    /// The wgpu engine, `GpuModel`.
    pub(crate) const WGPU: Engine = Engine {
        name: "GpuModel::compile",
        wgpu_limits: true,
    };

    /// The CUDA engine, `CudaModel`.
    #[allow(dead_code)] // Used with the `cuda` feature only.
    pub(crate) const CUDA: Engine = Engine {
        name: "CudaModel::compile",
        wgpu_limits: false,
    };
}

/// An element-wise activation a GPU runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Act {
    /// `max(x, 0)`.
    Relu,
    /// `x` above 0, `slope · x` elsewhere, with the slope as the CPU layer holds it.
    LeakyRelu(f64),
    /// The hyperbolic tangent.
    Tanh,
    /// The logistic sigmoid.
    Sigmoid,
    /// GELU's tanh form.
    GeluTanh,
    /// GELU's erf form.
    GeluErf,
    /// `x` above 0, `alpha · (eˣ − 1)` elsewhere.
    Elu(f64),
    /// The scaled ELU.
    Selu,
    /// `x` above 0, `alpha · (e^(x/alpha) − 1)` elsewhere, with a positive `alpha`.
    Celu(f64),
    /// `x · σ(x)`.
    Silu,
    /// `x · tanh(softplus(x))`.
    Mish,
    /// `ln(1 + e^(beta·x)) / beta`, and `x` once `beta · x` passes the threshold.
    Softplus {
        /// The sharpness.
        beta: f64,
        /// The value of `beta · x` above which the output is `x`.
        threshold: f64,
    },
    /// `x / (1 + |x|)`.
    Softsign,
    /// `min(max(x, 0), 6)`, which the hardtanh kernels run, apart from hardtanh since the
    /// dead-ReLU check reads it.
    Relu6,
    /// `x` clamped to `[min, max]`.
    Hardtanh {
        /// The lower bound.
        min: f64,
        /// The upper bound.
        max: f64,
    },
    /// The piecewise-linear sigmoid.
    Hardsigmoid,
    /// `x · hardsigmoid(x)`.
    Hardswish,
    /// `ln σ(x)`.
    LogSigmoid,
}

impl Act {
    /// The two settings an element-wise kernel takes, rounded to `f32` as the CPU's layer
    /// rounds them: the slope, `alpha` or `beta` first, and the threshold or the upper bound
    /// second (the lower bound first for hardtanh); 0 where the activation has none.
    pub(crate) fn settings(self) -> (f32, f32) {
        match self {
            Act::LeakyRelu(alpha) | Act::Elu(alpha) | Act::Celu(alpha) => (alpha as f32, 0.0),
            Act::Softplus { beta, threshold } => (beta as f32, threshold as f32),
            Act::Hardtanh { min, max } => (min as f32, max as f32),
            Act::Relu6 => (0.0, 6.0),
            Act::Relu
            | Act::Tanh
            | Act::Sigmoid
            | Act::GeluTanh
            | Act::GeluErf
            | Act::Selu
            | Act::Silu
            | Act::Mish
            | Act::Softsign
            | Act::Hardsigmoid
            | Act::Hardswish
            | Act::LogSigmoid => (0.0, 0.0),
        }
    }
}

/// A batch-norm layer's running statistics on a GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Running {
    /// The running mean's index in the plan's buffers.
    pub(crate) mean: usize,
    /// The running variance's index in the plan's buffers.
    pub(crate) var: usize,
    /// The batch count's index in the plan's counts, if the layer keeps one.
    pub(crate) count: Option<usize>,
    /// The weight of each batch's statistics, or `None` for the cumulative average.
    pub(crate) momentum: Option<f64>,
}

/// A batch-norm layer as a GPU runs it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Norm {
    /// Channels of each image.
    pub(crate) channels: usize,
    /// Values per channel of one image.
    pub(crate) spatial: usize,
    /// Added to the variance before its square root.
    pub(crate) eps: f64,
    /// γ's index in the plan's parameters, if the layer is affine.
    pub(crate) weight: Option<usize>,
    /// β's index in the plan's parameters, if the layer is affine.
    pub(crate) bias: Option<usize>,
    /// The running statistics, if the layer keeps them.
    pub(crate) running: Option<Running>,
}

/// What one layer computes on a GPU, with its parameters as indices into the plan's
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind {
    /// `y = x · wᵀ + bias`, with the weight `[outputs, inputs]` as the CPU stores it.
    Linear {
        /// Input columns.
        inputs: usize,
        /// Output columns.
        outputs: usize,
        /// The weight's index.
        weight: usize,
        /// The bias's index, if the layer has one.
        bias: Option<usize>,
    },
    /// An element-wise activation.
    Activation(Act),
    /// Softmax over each row, or log-softmax if `log`.
    Softmax {
        /// Whether it is the log-softmax.
        log: bool,
    },
    /// A 2-D convolution.
    Conv2d {
        /// Its shape.
        geom: Conv2dGeom,
        /// The weight's index, `[out, in, kh, kw]`.
        weight: usize,
        /// The bias's index, if the layer has one.
        bias: Option<usize>,
    },
    /// 2-D max pooling.
    MaxPool2d(Pool2dGeom),
    /// 2-D average pooling.
    AvgPool2d {
        /// Its shape, never dilated.
        geom: Pool2dGeom,
        /// Whether the padding counts towards the divisor.
        count_include_pad: bool,
        /// The divisor used in place of the window's size, if any.
        divisor_override: Option<usize>,
    },
    /// 2-D batch normalisation.
    BatchNorm2d(Norm),
    /// Dropout with this drop probability while training; the identity in evaluation.
    Dropout(f64),
    /// The input passed on unchanged: `Flatten` and `Identity`.
    Identity,
}

impl Kind {
    /// The input width the kind fixes, if it fixes one.
    fn declared_input(&self) -> Option<usize> {
        match self {
            Kind::Linear { inputs, .. } => Some(*inputs),
            Kind::Conv2d { geom, .. } => Some(geom.in_width()),
            Kind::MaxPool2d(geom) | Kind::AvgPool2d { geom, .. } => Some(geom.in_width()),
            Kind::BatchNorm2d(norm) => Some(norm.channels * norm.spatial),
            Kind::Activation(_) | Kind::Softmax { .. } | Kind::Dropout(_) | Kind::Identity => None,
        }
    }

    /// The output width for an input of `width` columns.
    fn output_width(&self, width: usize) -> usize {
        match self {
            Kind::Linear { outputs, .. } => *outputs,
            Kind::Conv2d { geom, .. } => geom.out_width(),
            Kind::MaxPool2d(geom) | Kind::AvgPool2d { geom, .. } => geom.out_width(),
            Kind::Activation(_)
            | Kind::Softmax { .. }
            | Kind::BatchNorm2d(_)
            | Kind::Dropout(_)
            | Kind::Identity => width,
        }
    }
}

/// One layer of the compiled model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CompiledLayer {
    /// What it computes.
    pub(crate) kind: Kind,
    /// The width of each row it reads.
    pub(crate) inputs: usize,
    /// The width of each row it writes.
    pub(crate) outputs: usize,
    /// The index in the plan's masks of the neuron mask on its output, if it has one.
    pub(crate) mask: Option<usize>,
}

/// A layer list checked and laid out, before anything is put on a GPU.
#[derive(Debug)]
pub(crate) struct Plan {
    /// The layers, in order.
    pub(crate) layers: Vec<CompiledLayer>,
    /// The width of one input row.
    pub(crate) input_width: usize,
    /// The trained parameters, in the order the layers first use them.
    pub(crate) params: Vec<ParamId>,
    /// The batch-norm running statistics, in the order the layers first use them.
    pub(crate) buffers: Vec<ParamId>,
    /// The batch-norm batch counts.
    pub(crate) counts: Vec<ParamId>,
    /// For each neuron mask, its layer and the packed bits for the batch.
    pub(crate) masks: Vec<(usize, Vec<u32>)>,
}

impl Plan {
    /// [`Plan::build_for`] the wgpu engine.
    #[allow(dead_code)] // Used with the `gpu` feature only.
    pub(crate) fn build(
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
        batch: usize,
    ) -> Result<Plan, Error> {
        Plan::build_for(model, params, batch, Engine::WGPU)
    }

    /// Checks every layer of `model` and lays out its parameters, buffers and masks for
    /// batches of `batch` rows, for `engine`, whose name starts each message.
    ///
    /// # Errors
    ///
    /// - [`Error::Unsupported`] if the model has no layers, a layer is not one a GPU runs or
    ///   has parameters its GPU form does not use (by index and name), an id is used in two
    ///   roles, no layer fixes the input width, or (on the wgpu engine) a convolution's batch
    ///   is above 65,535, a softmax's above 16,776,960, a GEMM's side above 4,194,240, or a
    ///   batch norm's eps below 2⁻¹²⁶ in `f32` and not 0.
    /// - [`Error::InvalidData`] if `batch` is 0, a layer's input width is not the width the
    ///   layer before it outputs, a layer outputs width 0 or another width than it reports, a
    ///   mask is not as wide as its layer's output, a batch count does not hold one value, or
    ///   a batch norm that keeps no running statistics would see one value per channel.
    pub(crate) fn build_for(
        model: &Sequential<f32>,
        params: &ParamStore<f32>,
        batch: usize,
        engine: Engine,
    ) -> Result<Plan, Error> {
        let who = engine.name;
        if model.is_empty() {
            return Err(Error::Unsupported(format!(
                "{who}: the model has no layers"
            )));
        }
        if batch == 0 {
            return Err(Error::InvalidData(format!(
                "{who}: a batch needs at least one row"
            )));
        }
        let mut slots = Slots::default();
        let mut kinds = Vec::with_capacity(model.len());
        for i in 0..model.len() {
            kinds.push(kind(model.layer(i), i, params, &mut slots, who)?);
        }
        let input_width = (0..model.len())
            .find_map(|i| model.layer(i).input_width().or(kinds[i].declared_input()))
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "{who}: no layer of the model fixes its input width"
                ))
            })?;
        let mut layers = Vec::with_capacity(model.len());
        let mut masks = Vec::new();
        let mut width = input_width;
        for (i, kind) in kinds.into_iter().enumerate() {
            let layer = model.layer(i);
            let name = layer.name();
            if let Some(expected) = layer.input_width().or(kind.declared_input())
                && expected != width
            {
                return Err(Error::InvalidData(format!(
                    "{who}: layer {i} ({name}) expects input width {expected}, but its input \
                     has width {width}"
                )));
            }
            if let Kind::BatchNorm2d(norm) = kind
                && norm.running.is_none()
                && batch * norm.spatial < 2
            {
                return Err(Error::InvalidData(format!(
                    "{who}: layer {i} ({name}) normalises with the batch's statistics, which \
                     need more than one value per channel; a batch of {batch} gives {}",
                    batch * norm.spatial
                )));
            }
            if engine.wgpu_limits {
                check_wgpu_limits(&kind, batch, i, name, who)?;
            }
            let outputs = kind.output_width(width);
            if outputs == 0 {
                return Err(Error::InvalidData(format!(
                    "{who}: layer {i} ({name}) outputs width 0"
                )));
            }
            let reported = layer.output_width(width);
            if outputs != reported {
                return Err(Error::InvalidData(format!(
                    "{who}: layer {i} ({name}) outputs width {outputs} on the GPU, but reports \
                     {reported} for input width {width}"
                )));
            }
            let mask = match model.mask(i) {
                Some(m) if m.width() != outputs => {
                    return Err(Error::InvalidData(format!(
                        "{who}: the mask on layer {i} ({name}) has width {}, but the layer \
                         outputs width {outputs}",
                        m.width()
                    )));
                }
                Some(m) => {
                    let on: Vec<bool> = (0..outputs).map(|j| m.is_on(j)).collect();
                    masks.push((i, pack_rows(&on, batch)));
                    Some(masks.len() - 1)
                }
                None => None,
            };
            layers.push(CompiledLayer {
                kind,
                inputs: width,
                outputs,
                mask,
            });
            width = outputs;
        }
        Ok(Plan {
            layers,
            input_width,
            params: slots.params,
            buffers: slots.buffers,
            counts: slots.counts,
            masks,
        })
    }
}

/// Refuses what the wgpu kernels cannot run: a convolution over more than 65,535 images, a
/// softmax over more than 16,776,960 rows, a GEMM side above 4,194,240, and a batch norm's
/// eps that `f32` arithmetic flushing subnormals may turn into 0.
fn check_wgpu_limits(
    kind: &Kind,
    batch: usize,
    i: usize,
    name: &str,
    who: &str,
) -> Result<(), Error> {
    if matches!(kind, Kind::Softmax { .. }) && batch > MAX_SOFTMAX_ROWS {
        return Err(Error::Unsupported(format!(
            "{who}: layer {i} ({name}) is a softmax, which takes at most {MAX_SOFTMAX_ROWS} \
             rows per batch on the GPU, not {batch}"
        )));
    }
    if matches!(kind, Kind::Conv2d { .. }) && batch > MAX_CONV_BATCH {
        return Err(Error::Unsupported(format!(
            "{who}: layer {i} ({name}) is a convolution, which takes at most {MAX_CONV_BATCH} \
             images per batch on the GPU, not {batch}"
        )));
    }
    let too_wide = |what: &str, n: usize| {
        Err(Error::Unsupported(format!(
            "{who}: layer {i} ({name}) has {n} {what}, more than the {MAX_GEMM_SIDE} the GPU's \
             GEMM takes"
        )))
    };
    match *kind {
        Kind::Linear {
            inputs, outputs, ..
        } => {
            if batch > MAX_GEMM_SIDE {
                return too_wide("rows per batch", batch);
            }
            if outputs > MAX_GEMM_SIDE {
                return too_wide("outputs", outputs);
            }
            // The backward pass's input gradient is a GEMM with `inputs` columns.
            if inputs > MAX_GEMM_SIDE {
                return too_wide("inputs", inputs);
            }
        }
        Kind::Conv2d { geom, .. } => {
            let (oh, ow) = geom.out_size();
            if oh * ow > MAX_GEMM_SIDE {
                return too_wide("output pixels per channel", oh * ow);
            }
            if geom.out_channels() > MAX_GEMM_SIDE {
                return too_wide("output channels", geom.out_channels());
            }
        }
        Kind::BatchNorm2d(norm) if norm.eps != 0.0 && (norm.eps as f32) < f32::MIN_POSITIVE => {
            return Err(Error::Unsupported(format!(
                "{who}: layer {i} ({name}) has eps {:e}, which the GPU's f32 arithmetic may \
                 flush to 0; use 0 or at least 2⁻¹²⁶",
                norm.eps
            )));
        }
        _ => {}
    }
    Ok(())
}

/// The ids a plan holds, each once, with its index.
#[derive(Debug, Default)]
struct Slots {
    /// The trained parameters, in order of first use.
    params: Vec<ParamId>,
    /// The running statistics, in order of first use.
    buffers: Vec<ParamId>,
    /// The batch counts, in order of first use.
    counts: Vec<ParamId>,
    /// Each id's role and index.
    index: HashMap<ParamId, (Role, usize)>,
}

/// What an id is to a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// A parameter the optimiser trains.
    Param,
    /// A running statistic.
    Buffer,
    /// A batch count, kept on the host.
    Count,
}

impl Slots {
    /// The index of `id` in its role's list, added there on its first use. A layer that uses
    /// an id in another role than an earlier layer did is refused, naming it.
    fn slot(
        &mut self,
        id: ParamId,
        role: Role,
        i: usize,
        name: &str,
        who: &str,
    ) -> Result<usize, Error> {
        if let Some(&(had, at)) = self.index.get(&id) {
            if had != role {
                return Err(Error::Unsupported(format!(
                    "{who}: layer {i} ({name}) uses parameter {} as a {role:?}, which an \
                     earlier layer uses as a {had:?}",
                    id.position()
                )));
            }
            return Ok(at);
        }
        let list = match role {
            Role::Param => &mut self.params,
            Role::Buffer => &mut self.buffers,
            Role::Count => &mut self.counts,
        };
        list.push(id);
        self.index.insert(id, (role, list.len() - 1));
        Ok(list.len() - 1)
    }
}

/// The GPU form of `layer`, layer `i` of its model, with its ids added to `slots`; `who`
/// starts each message.
fn kind(
    layer: &dyn Layer<f32>,
    i: usize,
    params: &ParamStore<f32>,
    slots: &mut Slots,
    who: &str,
) -> Result<Kind, Error> {
    let name = layer.name();
    let refuse = |why: &str| {
        Err(Error::Unsupported(format!(
            "{who}: layer {i} ({name}) {why}"
        )))
    };
    if let Some(linear) = layer.as_linear() {
        return Ok(Kind::Linear {
            inputs: linear.inputs(),
            outputs: linear.outputs(),
            weight: slots.slot(linear.weight(), Role::Param, i, name, who)?,
            bias: (linear.bias())
                .map(|b| slots.slot(b, Role::Param, i, name, who))
                .transpose()?,
        });
    }
    if let Some(conv) = layer.as_conv2d() {
        return Ok(Kind::Conv2d {
            geom: conv.geom(),
            weight: slots.slot(conv.weight(), Role::Param, i, name, who)?,
            bias: (conv.bias())
                .map(|b| slots.slot(b, Role::Param, i, name, who))
                .transpose()?,
        });
    }
    if let Some(bn) = layer.as_batch_norm2d() {
        let param = |id: Option<ParamId>, slots: &mut Slots| {
            id.map(|id| slots.slot(id, Role::Param, i, name, who))
                .transpose()
        };
        let weight = param(bn.weight(), slots)?;
        let bias = param(bn.bias(), slots)?;
        let running = match bn.running() {
            Some(r) => Some(Running {
                mean: slots.slot(r.mean, Role::Buffer, i, name, who)?,
                var: slots.slot(r.var, Role::Buffer, i, name, who)?,
                count: match r.count {
                    Some(c) if params.values.get(c).data().len() != 1 => {
                        return Err(Error::InvalidData(format!(
                            "{who}: layer {i} ({name})'s num_batches_tracked holds {} values, \
                             not 1",
                            params.values.get(c).data().len()
                        )));
                    }
                    Some(c) => Some(slots.slot(c, Role::Count, i, name, who)?),
                    None => None,
                },
                momentum: r.momentum,
            }),
            None => None,
        };
        return Ok(Kind::BatchNorm2d(Norm {
            channels: bn.channels(),
            spatial: bn.spatial(),
            eps: bn.eps(),
            weight,
            bias,
            running,
        }));
    }
    if !layer.parameters().is_empty() || !layer.buffers().is_empty() {
        return refuse("has parameters, and the GPU runs no layer of its kind");
    }
    if let Some(p) = layer.dropout_form(Token(())) {
        return Ok(Kind::Dropout(p));
    }
    let act = match layer.inference(&params.values) {
        Some(InferenceLayer::Identity) => return Ok(Kind::Identity),
        Some(InferenceLayer::MaxPool2d(geom)) => return Ok(Kind::MaxPool2d(geom)),
        Some(InferenceLayer::AvgPool2d {
            geom,
            count_include_pad,
            divisor_override,
        }) => {
            if geom.dilation() != (1, 1) {
                return refuse(&format!(
                    "is an average pooling with dilation {:?}; it takes none",
                    geom.dilation()
                ));
            }
            if divisor_override == Some(0) {
                return refuse("is an average pooling with a divisor of 0");
            }
            return Ok(Kind::AvgPool2d {
                geom,
                count_include_pad,
                divisor_override,
            });
        }
        Some(InferenceLayer::Activation(kind)) => kind,
        Some(InferenceLayer::Softmax) => return Ok(Kind::Softmax { log: false }),
        Some(InferenceLayer::LogSoftmax) => return Ok(Kind::Softmax { log: true }),
        Some(_) => return refuse("is not a layer the GPU runs"),
        None => return refuse("has no inference form, so the GPU cannot run it"),
    };
    let act = match act {
        ActivationKind::Relu => Act::Relu,
        ActivationKind::LeakyRelu(slope) => Act::LeakyRelu(slope),
        ActivationKind::Tanh => Act::Tanh,
        ActivationKind::Sigmoid => Act::Sigmoid,
        ActivationKind::Gelu(GeluApprox::Tanh) => Act::GeluTanh,
        ActivationKind::Gelu(GeluApprox::Exact) => Act::GeluErf,
        ActivationKind::Elu(alpha) => Act::Elu(alpha),
        ActivationKind::Selu => Act::Selu,
        ActivationKind::Celu(alpha) => {
            // The kernels need `x / alpha` at or below 0 wherever `x` is, as the CPU's kernel
            // has it for a positive alpha; for any other alpha the CPU takes std's `exp_m1`.
            if !(alpha > 0.0 && (alpha as f32) > 0.0) {
                return refuse(&format!(
                    "is a CELU with alpha {alpha}; the GPU runs a positive alpha only"
                ));
            }
            Act::Celu(alpha)
        }
        ActivationKind::Silu => Act::Silu,
        ActivationKind::Mish => Act::Mish,
        ActivationKind::Softplus { beta, threshold } => Act::Softplus { beta, threshold },
        ActivationKind::Softsign => Act::Softsign,
        ActivationKind::Hardtanh { min, max } => {
            // The CPU's `Tape::hardtanh` panics on these, so no device trains them.
            if min.is_nan() || max.is_nan() || min > max {
                return refuse(&format!(
                    "is a hardtanh with min {min} and max {max}; min must not exceed max"
                ));
            }
            Act::Hardtanh { min, max }
        }
        ActivationKind::Relu6 => Act::Relu6,
        ActivationKind::Hardsigmoid => Act::Hardsigmoid,
        ActivationKind::Hardswish => Act::Hardswish,
        ActivationKind::LogSigmoid => Act::LogSigmoid,
    };
    Ok(Kind::Activation(act))
}

/// The bits of `on` repeated for `rows` rows, packed 32 to a word, low bit first.
pub(crate) fn pack_rows(on: &[bool], rows: usize) -> Vec<u32> {
    let n = on.len() * rows;
    let mut words = vec![0u32; n.div_ceil(32)];
    for k in 0..n {
        if on[k % on.len()] {
            words[k / 32] |= 1 << (k % 32);
        }
    }
    words
}
