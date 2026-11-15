//! The per-channel kernels against their CPU twins, on every backend this machine has: bias
//! add bit for bit (one addition per value, as the CPU's broadcast `add`, which it is checked
//! against too), and channel sums exactly on small integers and within the bound of a sum in
//! any order on seeded values, over matrices and NCHW batches: one value, one chunk exactly,
//! a partial last chunk, the CIFAR network's 131,072 values per channel, and a bias add over
//! more values than one grid row holds. The tests skip without a GPU, unless
//! `CAVITATION_GPU_REQUIRED=1`.

mod common;

use cavitation_core::{Graph, ParamStore, Tensor};
use cavitation_gpu::kernels::{BiasAdd, ChannelDims, ChannelSums};
use common::{
    Kit, assert_exact, assert_within, exact_values, for_each_gpu, seeded_values, sum_bound,
};

/// The shapes both kernels run: a single value, a small matrix, an NCHW batch of odd sizes,
/// one channel of exactly one chunk (4,096 values) and of one value more, a batch whose
/// channels span several chunks, and the CIFAR network's first convolution output at batch
/// 128 (131,072 values per channel).
fn shapes() -> Vec<ChannelDims> {
    vec![
        ChannelDims::new(1, 1, 1),
        ChannelDims::rows(3, 5),
        ChannelDims::new(2, 7, 9),
        ChannelDims::new(1, 1, 4096),
        ChannelDims::new(1, 1, 4097),
        ChannelDims::new(3, 4, 5000),
        ChannelDims::rows(1000, 10),
        ChannelDims::new(128, 32, 1024),
    ]
}

/// A kit whose steps can move a tensor of `dims`.
fn kit(gpu: &cavitation_gpu::Gpu, dims: ChannelDims) -> Kit {
    Kit::new(gpu, 4 * dims.len() + 4096)
}

/// The GPU's bias add of `bias` into `x`.
fn gpu_bias_add(
    kit: &mut Kit,
    kernel: &BiasAdd,
    dims: ChannelDims,
    bias: &[f32],
    x: &[f32],
) -> Vec<f32> {
    let (b, out) = (kit.upload(bias), kit.upload(x));
    let call = kernel.call(&mut kit.steps, &mut kit.pool, dims, &b, &out);
    let mut step = kit.steps.begin();
    kernel.record(&mut step, &call);
    let _ = step.finish();
    let result = kit.read(&out);
    call.release(&mut kit.pool);
    kit.pool.give(b);
    kit.pool.give(out);
    result
}

/// The GPU's channel sums of `x`.
fn gpu_sums(kit: &mut Kit, kernel: &ChannelSums, dims: ChannelDims, x: &[f32]) -> Vec<f32> {
    let (xb, out) = (kit.upload(x), kit.zeros(dims.channels as usize));
    let call = kernel.call(&mut kit.steps, &mut kit.pool, dims, &xb, &out);
    let mut step = kit.steps.begin();
    kernel.record(&mut step, &call);
    let _ = step.finish();
    let result = kit.read(&out);
    call.release(&mut kit.pool);
    kit.pool.give(xb);
    kit.pool.give(out);
    result
}

/// The channel of value `i` of a tensor shaped `dims`.
fn channel(dims: ChannelDims, i: usize) -> usize {
    i / dims.inner as usize % dims.channels as usize
}

/// Bias add on seeded values: every value is the CPU's `f32` addition of its channel's bias,
/// bit for bit.
#[test]
fn bias_add_twin() {
    for_each_gpu(|gpu| {
        for (s, dims) in shapes().into_iter().enumerate() {
            let mut kit = kit(gpu, dims);
            let kernel = BiasAdd::new(gpu, &mut kit.cache).unwrap();
            let s = s as u64;
            let bias = seeded_values(10 + s, dims.channels as usize, -2.0, 2.0);
            let x = seeded_values(20 + s, dims.len() as usize, -2.0, 2.0);
            let want: Vec<f32> = x
                .iter()
                .enumerate()
                .map(|(i, v)| v + bias[channel(dims, i)])
                .collect();
            assert_exact(
                &gpu_bias_add(&mut kit, &kernel, dims, &bias, &x),
                &want,
                &format!("bias add {dims:?}"),
            );
        }
    });
}

/// The CPU arithmetic bias add is checked against is the CPU tape's own: a `[N]` bias added
/// to an `[M, N]` input by `Tape::add` gives the same bits.
#[test]
fn bias_add_is_the_tapes_broadcast_add() {
    for_each_gpu(|gpu| {
        let dims = ChannelDims::rows(37, 11);
        let mut kit = kit(gpu, dims);
        let kernel = BiasAdd::new(gpu, &mut kit.cache).unwrap();
        let bias = seeded_values(30, 11, -2.0, 2.0);
        let x = seeded_values(31, 37 * 11, -2.0, 2.0);
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let xs = Tensor::from_vec(x.clone(), &[37, 11]).unwrap();
        let bs = Tensor::from_vec(bias.clone(), &[11]).unwrap();
        let (xv, bv) = (tape.input(xs.view()), tape.input(bs.view()));
        let sum = tape.add(xv, bv);
        let want = tape.value(sum).data().to_vec();
        assert_exact(
            &gpu_bias_add(&mut kit, &kernel, dims, &bias, &x),
            &want,
            "bias add against Tape::add",
        );
    });
}

/// More values than one grid row of workgroups holds (17 × 1,000,000): bias add numbers its
/// workgroups over two rows and adds to every value exactly once.
#[test]
fn bias_add_covers_a_split_grid() {
    for_each_gpu(|gpu| {
        let dims = ChannelDims::new(17, 2, 500_000);
        let mut kit = kit(gpu, dims);
        let kernel = BiasAdd::new(gpu, &mut kit.cache).unwrap();
        let x = exact_values(40, dims.len() as usize);
        let bias = [1.0, -2.0];
        let want: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v + bias[channel(dims, i)])
            .collect();
        assert_exact(
            &gpu_bias_add(&mut kit, &kernel, dims, &bias, &x),
            &want,
            "bias add over two grid rows",
        );
    });
}

/// Channel sums on small integers: every sum is exact in any order, so the GPU's fixed tree
/// gives the CPU's in-order `f32` sum bit for bit, on every shape.
#[test]
fn channel_sums_exact_twin() {
    for_each_gpu(|gpu| {
        for (s, dims) in shapes().into_iter().enumerate() {
            let mut kit = kit(gpu, dims);
            let kernel = ChannelSums::new(gpu, &mut kit.cache).unwrap();
            let x = exact_values(50 + s as u64, dims.len() as usize);
            let mut want = vec![0.0f32; dims.channels as usize];
            for (i, v) in x.iter().enumerate() {
                want[channel(dims, i)] += v;
            }
            assert_exact(
                &gpu_sums(&mut kit, &kernel, dims, &x),
                &want,
                &format!("channel sums {dims:?}"),
            );
        }
    });
}

/// Channel sums on seeded values, against the exact (`f64`) sums of the same `f32` values:
/// each within `2(L + 2) · u · Σ|x|` for `L = outer · inner` values, the bound of a sum in any
/// order; and the same bits on a second run, the tree being fixed.
#[test]
fn channel_sums_rounding_twin() {
    for_each_gpu(|gpu| {
        for (s, dims) in shapes().into_iter().enumerate() {
            let mut kit = kit(gpu, dims);
            let kernel = ChannelSums::new(gpu, &mut kit.cache).unwrap();
            let x = seeded_values(60 + s as u64, dims.len() as usize, -1.0, 1.0);
            let (mut exact, mut abs) = (
                vec![0.0f64; dims.channels as usize],
                vec![0.0f64; dims.channels as usize],
            );
            for (i, &v) in x.iter().enumerate() {
                exact[channel(dims, i)] += f64::from(v);
                abs[channel(dims, i)] += f64::from(v).abs();
            }
            let got = gpu_sums(&mut kit, &kernel, dims, &x);
            let l = dims.summed() as usize;
            let worst = assert_within(
                &got,
                &exact,
                |c| sum_bound(l, abs[c]),
                &format!("channel sums {dims:?}"),
            );
            println!(
                "channel sums {dims:?}: worst error {:.3} of its bound",
                worst.ratio()
            );
            let again = gpu_sums(&mut kit, &kernel, dims, &x);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            assert_eq!(
                bits(&again),
                bits(&got),
                "channel sums {dims:?} on a second run"
            );
        }
    });
}

/// The documented refusals of both calls: a zero dimension, more than 2³¹ values, more than
/// 65,535 channels or chunks for the sums, a buffer too short, and the same buffer as input
/// and output.
#[test]
fn refusals() {
    for_each_gpu(|gpu| {
        let message = |f: &mut dyn FnMut()| {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("refused");
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default()
        };
        let mut kit = kit(gpu, ChannelDims::new(1, 1, 64));
        let bias_add = BiasAdd::new(gpu, &mut kit.cache).unwrap();
        let sums = ChannelSums::new(gpu, &mut kit.cache).unwrap();
        let (a, b) = (kit.zeros(64), kit.zeros(64));
        let mut bias = |dims: ChannelDims,
                        x: &cavitation_gpu::PooledBuffer,
                        y: &cavitation_gpu::PooledBuffer| {
            message(&mut || {
                let _ = bias_add.call(&mut kit.steps, &mut kit.pool, dims, x, y);
            })
        };
        assert!(bias(ChannelDims::new(0, 1, 1), &a, &b).contains("above 0"));
        assert!(
            bias(ChannelDims::new(1 << 16, 1 << 16, 1), &a, &b)
                .contains("more than the 2147483648")
        );
        assert!(bias(ChannelDims::new(1, 1, 65), &a, &b).contains("the tensor needs 65 f32s"));
        assert!(bias(ChannelDims::new(1, 65, 1), &a, &b).contains("the bias needs 65 f32s"));
        assert!(bias(ChannelDims::new(1, 1, 64), &a, &a).contains("different buffers"));
        let mut sum = |dims: ChannelDims,
                       x: &cavitation_gpu::PooledBuffer,
                       y: &cavitation_gpu::PooledBuffer| {
            message(&mut || {
                let _ = sums.call(&mut kit.steps, &mut kit.pool, dims, x, y);
            })
        };
        assert!(sum(ChannelDims::new(1, 0, 1), &a, &b).contains("above 0"));
        assert!(
            sum(ChannelDims::new(1, 70_000, 1), &a, &b).contains("more than 65535 in a dimension")
        );
        assert!(
            sum(ChannelDims::new(1, 1, 1 << 29), &a, &b).contains("more than 65535 in a dimension")
        );
        assert!(sum(ChannelDims::new(1, 1, 65), &a, &b).contains("the tensor needs 65 f32s"));
        assert!(sum(ChannelDims::new(1, 65, 1), &a, &b).contains("the tensor needs 65 f32s"));
        assert!(sum(ChannelDims::new(1, 1, 64), &a, &a).contains("different buffers"));
    });
}
