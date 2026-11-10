# Changelog

Notable changes by release. Releases are git tags `vX.Y.Z`. New work goes under Unreleased.

## Unreleased

- Cargo workspace with the core, nn, facade, datasets and bench crates, and shared lints.
- `Real`, `Shape`, `Tensor` (with views) and `Error` in `cavitation-core`.
- `SplitMix64` and `Rng` (xoshiro256++) in `cavitation-core`.
- Strided matrix views (`MatRef`, `MatMut`) and reference `gemm` and `matvec` kernels in `cavitation-core`.
- Reverse-mode autograd in `cavitation-core`: `Graph`, `Tape` and `Var`, the `ParamStore` for parameters and their gradients, and the first operation, `add`.
- `gradcheck` and `gradcheck_params`: a numerical gradient check in `f64` (central differences) with a report of every element outside the tolerance.
- `sub`, `mul` and `scale` on the tape, and NumPy-style broadcasting for `add`, `sub` and `mul` (with a fast path for the bias add).
- `matmul` and `matmul_nt` on the tape, with gradients computed on strided views, and `Real::gemm`.
- `sum`, `sum_axis`, `mean`, `mean_axis`, `relu`, `leaky_relu`, `tanh`, `sigmoid` and `gelu` (exact and tanh forms, `GeluApprox`) on the tape, and `Real::erf` (ported from FreeBSD msun).
