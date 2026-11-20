//! The 2-D convolution on Metal through Apple's MPSGraph: the forward pass with its bias,
//! the input gradient and the weight gradient, each compiled once for its shape into an
//! `MPSGraphExecutable` that reads and writes the pooled buffers in place (NCHW images,
//! OIHW weights), recorded into a step as Metal work.
//!
//! Each executable is compiled at full `f32` precision (`MPSGraphReducedPrecisionFastMath`
//! none, so no `f16` Winograd intermediates) and waits for its compilation to finish:
//! freeing an executable whose compilation is still running on MPSGraph's own queue crashes.

use std::fmt;
use std::sync::Arc;

use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
use objc2_metal_performance_shaders::{MPSCommandBuffer, MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphCompilationDescriptor, MPSGraphConvolution2DOpDescriptor, MPSGraphDevice,
    MPSGraphExecutable, MPSGraphPaddingStyle, MPSGraphReducedPrecisionFastMath, MPSGraphShapedType,
    MPSGraphTensor, MPSGraphTensorData, MPSGraphTensorNamedDataLayout,
};

use super::conv::ConvDims;
use crate::metal::MetalOp;
use crate::{Gpu, PooledBuffer};

/// One convolution pass compiled for its shape, with the tensors it reads and the one it
/// writes bound to their buffers.
pub(crate) struct ConvGraph {
    /// Which pass, for messages.
    what: &'static str,
    /// The shape, for messages.
    dims: ConvDims,
    /// The compiled pass.
    exe: Retained<MPSGraphExecutable>,
    /// The inputs, in the executable's feed order.
    inputs: Retained<NSArray<MPSGraphTensorData>>,
    /// The output, written in place.
    results: Retained<NSArray<MPSGraphTensorData>>,
}

// SAFETY: the executable and tensor data are only encoded from, one step at a time (a step
// holds its encoder by `&mut`); MPSGraph executables may be used from any thread.
unsafe impl Send for ConvGraph {}
// SAFETY: as for `Send`: shared references only ever encode, never concurrently.
unsafe impl Sync for ConvGraph {}

impl fmt::Debug for ConvGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConvGraph")
            .field("what", &self.what)
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}

impl MetalOp for ConvGraph {
    fn encode(&self, cb: &MPSCommandBuffer) {
        // SAFETY: the inputs and the result are bound to buffers that live as long as the
        // call that holds this graph, which a step records only while it is alive, in the
        // executable's feed order and shapes.
        unsafe {
            self.exe
                .encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                    cb,
                    &self.inputs,
                    Some(&self.results),
                    None,
                );
        }
    }
}

/// An MPS shape from dimensions.
fn shape(dims: &[u32]) -> Retained<MPSShape> {
    let numbers: Vec<Retained<NSNumber>> = dims
        .iter()
        .map(|&d| NSNumber::new_usize(d as usize))
        .collect();
    NSArray::from_retained_slice(&numbers)
}

/// A placeholder of `dims` in `f32`.
fn placeholder(graph: &MPSGraph, dims: &[u32]) -> Retained<MPSGraphTensor> {
    // SAFETY: a placeholder of a static shape and a plain data type.
    unsafe {
        graph.placeholderWithShape_dataType_name(Some(&shape(dims)), MPSDataType::Float32, None)
    }
}

/// `buffer`'s first elements as a tensor of `dims` in `f32`.
fn tensor_data(buffer: &PooledBuffer, dims: &[u32]) -> Option<Retained<MPSGraphTensorData>> {
    let raw = buffer.raw()?;
    // SAFETY: the buffer holds at least the tensor's `f32`s (the call checked its length);
    // the tensor data retains it, and the call that holds the tensor data keeps it from
    // being given back to the pool.
    Some(unsafe {
        MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
            MPSGraphTensorData::alloc(),
            raw.get(),
            &shape(dims),
            MPSDataType::Float32,
        )
    })
}

/// The convolution's descriptor: strides, dilations and symmetric padding of `dims`, NCHW
/// images and OIHW weights.
fn descriptor(dims: &ConvDims) -> Retained<MPSGraphConvolution2DOpDescriptor> {
    let u = |v: u32| v as usize;
    // SAFETY: plain values; every one of them is at least 1 or a padding.
    unsafe {
        MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
            u(dims.stride.1),
            u(dims.stride.0),
            u(dims.dilation.1),
            u(dims.dilation.0),
            1,
            u(dims.padding.1),
            u(dims.padding.1),
            u(dims.padding.0),
            u(dims.padding.0),
            MPSGraphPaddingStyle::Explicit,
            MPSGraphTensorNamedDataLayout::NCHW,
            MPSGraphTensorNamedDataLayout::OIHW,
        )
    }
    .expect("a convolution descriptor of valid values")
}

/// Compiles `out` of `graph` for `feeds` (placeholder, shape and buffer each) on `gpu`'s
/// device and binds it to the buffers; `None` without Metal or when a buffer has no
/// `MTLBuffer`.
fn compile(
    gpu: &Gpu,
    graph: &MPSGraph,
    feeds: &[(&MPSGraphTensor, &[u32], &PooledBuffer)],
    out: (&MPSGraphTensor, &[u32], &PooledBuffer),
    what: &'static str,
    dims: ConvDims,
) -> Option<Arc<ConvGraph>> {
    let metal = gpu.metal()?;
    let data: Vec<Retained<MPSGraphTensorData>> = feeds
        .iter()
        .map(|(_, d, b)| tensor_data(b, d))
        .collect::<Option<_>>()?;
    let result = tensor_data(out.2, out.1)?;
    // SAFETY: the device is wgpu's own `MTLDevice`, alive for as long as `gpu`.
    let device = unsafe { MPSGraphDevice::deviceWithMTLDevice(metal.device()) };
    let types: Vec<Retained<MPSGraphShapedType>> = feeds
        .iter()
        // SAFETY: a shaped type of a static shape and a plain data type.
        .map(|(_, d, _)| unsafe {
            MPSGraphShapedType::initWithShape_dataType(
                MPSGraphShapedType::alloc(),
                Some(&shape(d)),
                MPSDataType::Float32,
            )
        })
        .collect();
    let keys: Vec<&MPSGraphTensor> = feeds.iter().map(|f| f.0).collect();
    let values: Vec<&MPSGraphShapedType> = types.iter().map(|t| &**t).collect();
    let feed_types = NSDictionary::from_slices(&keys, &values);
    // SAFETY: a fresh descriptor, set before it is used.
    let options = unsafe { MPSGraphCompilationDescriptor::new() };
    // SAFETY: as above.
    unsafe {
        options.setReducedPrecisionFastMath(MPSGraphReducedPrecisionFastMath::None);
        options.setWaitForCompilationCompletion(true);
    }
    // SAFETY: every placeholder `out` depends on is fed, with its shape and type.
    let exe = unsafe {
        graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
            Some(&device),
            &feed_types,
            &NSArray::from_slice(&[out.0]),
            None,
            Some(&options),
        )
    };
    // SAFETY: a compiled executable lists its feeds.
    let order = unsafe { exe.feedTensors() }.expect("a compiled executable has its feeds");
    let inputs: Vec<&MPSGraphTensorData> = order
        .iter()
        .map(|t| {
            let at = feeds
                .iter()
                .position(|f| std::ptr::eq(f.0, &*t))
                .expect("the executable's feeds are the ones given");
            &*data[at]
        })
        .collect();
    Some(Arc::new(ConvGraph {
        what,
        dims,
        exe,
        inputs: NSArray::from_slice(&inputs),
        results: NSArray::from_slice(&[&*result]),
    }))
}

/// The four dimensions of `dims`'s images, weights and outputs.
fn shapes(dims: &ConvDims) -> ([u32; 4], [u32; 4], [u32; 4]) {
    let (oh, ow) = dims.out_size().expect("the call checked the window");
    (
        [dims.batch, dims.in_channels, dims.input.0, dims.input.1],
        [
            dims.out_channels,
            dims.in_channels,
            dims.kernel.0,
            dims.kernel.1,
        ],
        [dims.batch, dims.out_channels, oh, ow],
    )
}

/// The forward pass `out = conv(x, weight) + bias` through MPSGraph, or `None` where the
/// GPU is not Metal's (or `CAVITATION_MPS=0`) or a buffer has no `MTLBuffer`.
pub(crate) fn forward(
    gpu: &Gpu,
    dims: ConvDims,
    x: &PooledBuffer,
    weight: &PooledBuffer,
    bias: Option<&PooledBuffer>,
    out: &PooledBuffer,
) -> Option<Arc<ConvGraph>> {
    gpu.metal()?;
    autoreleasepool(|_| {
        let (xs, ws, ys) = shapes(&dims);
        let bs = [1, dims.out_channels, 1, 1];
        // SAFETY: a new, empty graph.
        let graph = unsafe { MPSGraph::new() };
        let (px, pw) = (placeholder(&graph, &xs), placeholder(&graph, &ws));
        let pb = placeholder(&graph, &bs);
        // SAFETY: placeholders of this graph, of the shapes the descriptor convolves.
        let conv = unsafe {
            graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                &px,
                &pw,
                &descriptor(&dims),
                None,
            )
        };
        let mut feeds = vec![(&*px, &xs[..], x), (&*pw, &ws[..], weight)];
        let y = match bias {
            Some(b) => {
                feeds.push((&*pb, &bs[..], b));
                // SAFETY: tensors of this graph, the bias broadcast over the images.
                unsafe { graph.additionWithPrimaryTensor_secondaryTensor_name(&conv, &pb, None) }
            }
            None => conv,
        };
        compile(gpu, &graph, &feeds, (&y, &ys, out), "forward", dims)
    })
}

/// The input gradient `dx` from `weight` and the output gradient `dy` through MPSGraph, or
/// `None` as for [`forward`].
pub(crate) fn input_grad(
    gpu: &Gpu,
    dims: ConvDims,
    weight: &PooledBuffer,
    dy: &PooledBuffer,
    dx: &PooledBuffer,
) -> Option<Arc<ConvGraph>> {
    gpu.metal()?;
    autoreleasepool(|_| {
        let (xs, ws, ys) = shapes(&dims);
        // SAFETY: a new, empty graph.
        let graph = unsafe { MPSGraph::new() };
        let (pdy, pw) = (placeholder(&graph, &ys), placeholder(&graph, &ws));
        // SAFETY: placeholders of this graph, of the forward convolution's shapes.
        let grad = unsafe {
            graph.convolution2DDataGradientWithIncomingGradientTensor_weightsTensor_outputShape_forwardConvolutionDescriptor_name(
                &pdy,
                &pw,
                &shape(&xs),
                &descriptor(&dims),
                None,
            )
        };
        let feeds = [(&*pdy, &ys[..], dy), (&*pw, &ws[..], weight)];
        compile(
            gpu,
            &graph,
            &feeds,
            (&grad, &xs, dx),
            "input gradient",
            dims,
        )
    })
}

/// The weight gradient `dw` from the images `x` and the output gradient `dy` through
/// MPSGraph, or `None` as for [`forward`].
pub(crate) fn weight_grad(
    gpu: &Gpu,
    dims: ConvDims,
    x: &PooledBuffer,
    dy: &PooledBuffer,
    dw: &PooledBuffer,
) -> Option<Arc<ConvGraph>> {
    gpu.metal()?;
    autoreleasepool(|_| {
        let (xs, ws, ys) = shapes(&dims);
        // SAFETY: a new, empty graph.
        let graph = unsafe { MPSGraph::new() };
        let (pdy, px) = (placeholder(&graph, &ys), placeholder(&graph, &xs));
        // SAFETY: placeholders of this graph, of the forward convolution's shapes.
        let grad = unsafe {
            graph.convolution2DWeightsGradientWithIncomingGradientTensor_sourceTensor_outputShape_forwardConvolutionDescriptor_name(
                &pdy,
                &px,
                &shape(&ws),
                &descriptor(&dims),
                None,
            )
        };
        let feeds = [(&*pdy, &ys[..], dy), (&*px, &xs[..], x)];
        compile(
            gpu,
            &graph,
            &feeds,
            (&grad, &ws, dw),
            "weight gradient",
            dims,
        )
    })
}
