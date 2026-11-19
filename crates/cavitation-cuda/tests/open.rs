//! Opens CUDA on this machine, if it can be used, and runs memory round trips and a
//! run-time compiled kernel through it. Each test skips without CUDA, unless
//! `CAVITATION_CUDA_REQUIRED=1` makes that a failure.

use cavitation_cuda::{Arg, Cuda, CudaError, MAX_KERNEL_ARGS};

/// `x[i] = a * x[i] + y[i]` over the first `n` elements, and a kernel that adds its integer
/// arguments into one output.
const SOURCE: &str = r#"
    extern "C" __global__ void scale_add(float *x, const float *y, float a, unsigned int n) {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) x[i] = a * x[i] + y[i];
    }

    extern "C" __global__ void sum_args(int *out, int a, unsigned int b, int c) {
        if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = a + (int)b + c;
    }
"#;

/// Opening either works or says why in a sentence; the versions are ones the crate is
/// written for.
#[test]
fn opens_or_says_why() {
    match Cuda::open() {
        Ok(cuda) => {
            let info = cuda.info();
            println!("{info}");
            assert!(!info.name.is_empty());
            assert!(info.memory > 0);
            assert_eq!(info.cudnn.0, 9);
            assert!([12, 13].contains(&info.cudnn_cuda.0));
            assert_eq!(info.cublas.0, info.cudnn_cuda.0);
            assert_eq!(info.nvrtc.0, info.cudnn_cuda.0);
            assert_eq!(info.libraries.len(), 3);
            let free = cuda.free_memory().expect("read the free memory");
            assert!(free > 0 && free <= info.memory);
        }
        Err(why) => {
            println!("CUDA is not available: {why}");
            assert!(!why.to_string().is_empty());
            // `new` must agree, and panics here when CUDA is required.
            assert!(Cuda::new().is_none());
        }
    }
}

/// Two `Cuda`s on one thread share the card's context, and each keeps working after the
/// other is dropped.
#[test]
fn opens_twice() {
    let Some(first) = Cuda::new() else { return };
    let second = Cuda::new().expect("open CUDA a second time");
    let mut a = first.buffer(4).expect("allocate");
    drop(second);
    a.write(&[5u32]).expect("write after the other is dropped");
    let mut back = [0u32];
    a.read(&mut back).expect("read");
    assert_eq!(back, [5]);
}

/// A buffer starts as zeros and gives back the bits written to it, for each scalar type
/// and for a part of its length.
#[test]
fn buffers_round_trip() {
    let Some(cuda) = Cuda::new() else { return };
    let mut buffer = cuda.buffer(6 * 4).expect("allocate");
    assert_eq!(buffer.bytes(), 24);
    let mut zeros = [1.0f32; 6];
    buffer.read(&mut zeros).expect("read");
    assert_eq!(zeros, [0.0; 6]);

    let values = [
        1.0f32,
        -0.0,
        f32::MIN_POSITIVE / 2.0,
        f32::INFINITY,
        f32::NAN,
        1e-3,
    ];
    buffer.write(&values).expect("write");
    let mut back = [0.0f32; 6];
    buffer.read(&mut back).expect("read");
    assert_eq!(back.map(f32::to_bits), values.map(f32::to_bits));

    buffer.write(&[u32::MAX, 7]).expect("write a part");
    let mut bits = [0u32; 3];
    buffer.read(&mut bits).expect("read a part");
    assert_eq!(bits, [u32::MAX, 7, values[2].to_bits()]);

    buffer.write(&[-1i32]).expect("write");
    let mut signed = [0i32; 1];
    buffer.read(&mut signed).expect("read");
    assert_eq!(signed, [-1]);

    let empty = cuda.buffer(0).expect("allocate nothing");
    assert_eq!(empty.bytes(), 0);
    empty.read::<f32>(&mut []).expect("read nothing");
}

/// A kernel compiled at run time computes `a * x + y` with the host's bits: the product is
/// rounded before the sum, on inputs where a fused multiply-add gives another result.
#[test]
fn a_kernel_matches_the_host_without_fusing() {
    let Some(cuda) = Cuda::new() else { return };
    let module = cuda.compile("open test", SOURCE).expect("compile");
    let scale_add = module.kernel("scale_add").expect("find the kernel");

    let n = 1000usize;
    let e = f32::EPSILON; // 2^-23
    let a = 1.0 + 2048.0 * e; // 1 + 2^-12
    let mut xs: Vec<f32> = (0..n).map(|i| 0.37 * i as f32 - 91.5).collect();
    let mut ys: Vec<f32> = (0..n).map(|i| 1.0 / (1.0 + i as f32)).collect();
    // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11, so the two-step sum is 0
    // where a fused one is 2^-24.
    xs[0] = a;
    ys[0] = -(1.0 + 4096.0 * e);
    let expected: Vec<f32> = xs.iter().zip(&ys).map(|(x, y)| a * x + y).collect();
    assert_eq!(expected[0], 0.0);
    assert_ne!(a.mul_add(xs[0], ys[0]), 0.0);

    let mut x = cuda.buffer(n * 4).expect("allocate");
    let mut y = cuda.buffer(n * 4).expect("allocate");
    x.write(&xs).expect("write");
    y.write(&ys).expect("write");
    let blocks = n.div_ceil(128) as u32;
    // SAFETY: the kernel takes two float pointers, a float and an unsigned count, and
    // touches only the first `n` floats of each buffer, which hold `n`.
    unsafe {
        scale_add.launch(
            [blocks, 1, 1],
            [128, 1, 1],
            &[
                Arg::Buffer(&x),
                Arg::Buffer(&y),
                Arg::F32(a),
                Arg::U32(n as u32),
            ],
        )
    }
    .expect("launch");
    cuda.synchronize().expect("wait for the kernel");
    let mut got = vec![0.0f32; n];
    x.read(&mut got).expect("read");
    for i in 0..n {
        assert_eq!(got[i].to_bits(), expected[i].to_bits(), "element {i}");
    }
}

/// Signed, unsigned and negative integer arguments reach the kernel as written.
#[test]
fn integer_arguments_arrive() {
    let Some(cuda) = Cuda::new() else { return };
    let module = cuda.compile("open test", SOURCE).expect("compile");
    let sum_args = module.kernel("sum_args").expect("find the kernel");
    let out = cuda.buffer(4).expect("allocate");
    // SAFETY: the kernel takes an int pointer, an int, an unsigned int and an int, and
    // writes one int to the buffer, which holds one.
    unsafe {
        sum_args.launch(
            [1, 1, 1],
            [1, 1, 1],
            &[
                Arg::Buffer(&out),
                Arg::I32(-1000),
                Arg::U32(250),
                Arg::I32(7),
            ],
        )
    }
    .expect("launch");
    let mut got = [0i32];
    out.read(&mut got).expect("read");
    assert_eq!(got, [-743]);
}

/// A source that does not compile gives the compiler's log, which names the source and
/// the line; a kernel that is not in the module and a launch the driver refuses are errors
/// that name the call.
#[test]
fn failures_are_reported() {
    let Some(cuda) = Cuda::new() else { return };
    let broken = "extern \"C\" __global__ void f(float *x) {\n    x[0] = undeclared;\n}\n";
    match cuda.compile("broken source", broken) {
        Err(CudaError::Compile { name, log }) => {
            assert_eq!(name, "broken source");
            assert!(log.contains("undeclared"), "{log}");
            assert!(log.contains("broken source(2)"), "{log}");
        }
        other => panic!("expected a compile error, got {other:?}"),
    }
    assert!(matches!(
        cuda.compile("nul", "a\0b"),
        Err(CudaError::Compile { .. })
    ));

    let module = cuda.compile("open test", SOURCE).expect("compile");
    let missing = module.kernel("no_such_kernel").unwrap_err();
    assert!(
        missing
            .to_string()
            .starts_with("cuModuleGetFunction failed: CUDA_ERROR_NOT_FOUND"),
        "{missing}"
    );

    let sum_args = module.kernel("sum_args").expect("find the kernel");
    let out = cuda.buffer(4).expect("allocate");
    let args = [Arg::Buffer(&out), Arg::I32(1), Arg::U32(2), Arg::I32(3)];
    // SAFETY: the arguments are the kernel's, as in `integer_arguments_arrive`; a grid
    // with no blocks is refused before anything runs.
    let refused = unsafe { sum_args.launch([0, 1, 1], [1, 1, 1], &args) }.unwrap_err();
    assert!(
        refused
            .to_string()
            .starts_with("cuLaunchKernel failed: CUDA_ERROR_INVALID_VALUE"),
        "{refused}"
    );
    // The context still works after the refusals.
    cuda.synchronize().expect("synchronize");
    assert!(MAX_KERNEL_ARGS >= args.len());
}
