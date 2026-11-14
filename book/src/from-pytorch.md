# Coming from PyTorch

Cavitation follows PyTorch's definitions where a model moved between the two has to give the same outputs: the activation formulas and constants, the weight layout and the parameter names. Where a convention only changes how a program behaves, it picks the one that shows problems instead of hiding them. This chapter lists both: what to expect to match, and where a PyTorch habit gives a different answer or none.

The program is `crates/cavitation/examples/from_pytorch.rs`. Run it with:

```sh
cargo run --release --example from_pytorch -p cavitation
```

The printed blocks below come from that command, run on an Intel Celeron N5105 (4 cores) at kernel level `sse4.2`, rustc 1.99.1.

## The digits split

`cavitation_datasets::digits::load` returns the digits as 1,437 training rows and 360 test rows:

```rust,ignore
{{#include ../../crates/cavitation/examples/from_pytorch.rs:split}}
```

```text
train 1437 rows, test 360 rows
```

The rows are the 1,797 of scikit-learn's copy of the data, in a fixed shuffled order: a Fisher-Yates shuffle driven by the mulberry32 generator with seed 42. `data/digits/README.md` gives the order as a short Python snippet, so a PyTorch script can rebuild the same split and train on the same rows. Each row has 64 pixel counts divided by 16, so they lie in 0 to 1, and a class label 0 to 9.

## The mean loss of an epoch

Each batch's loss is the mean over that batch's rows. The mean over the epoch is then either the average of the batch means or the mean over all rows. They are the same only when every batch has the same size. With 1,437 rows in batches of 100 the last batch has 37 rows, and a loop that adds up `loss.item()` and divides by the number of batches counts those 37 rows as much as 100.

`EpochReport::mean_loss` is the mean over rows: each batch's loss is weighted by its row count. This loop does both sums by hand, with `Trainer::step` standing in for the forward, backward and optimiser calls of a PyTorch loop:

```rust,ignore
{{#include ../../crates/cavitation/examples/from_pytorch.rs:loop}}
```

`fit` on the same network, the same batches and no shuffling reports the row-weighted figure:

```rust,ignore
{{#include ../../crates/cavitation/examples/from_pytorch.rs:fit}}
```

```text
average of the batch means 2.319727, row-weighted 2.330126
fit's mean_loss 2.330126
```

The example asserts that `mean_loss` equals the row-weighted figure to 1e-9. To get the same figure in PyTorch, weight each batch:

```python
total, rows = 0.0, 0
for x, y in loader:
    loss = criterion(model(x), y)  # mean over the batch's rows
    total += loss.item() * len(x)
    rows += len(x)
    # ... backward and step ...
mean_loss = total / rows
```

The two figures differ by 0.5% here because this is the first epoch of an untrained network, where the loss changes fast from batch to batch; they converge as the loss settles.

## Activation defaults

The activations that take a parameter default to PyTorch's values, so `LeakyRelu::default()` is `nn.LeakyReLU()` and so on. The example asserts each one:

```rust,ignore
{{#include ../../crates/cavitation/examples/from_pytorch.rs:defaults}}
```

| Cavitation | PyTorch | Default |
|---|---|---|
| `LeakyRelu` | `nn.LeakyReLU()` | `slope` 0.01 |
| `Elu` | `nn.ELU()` | `alpha` 1.0 |
| `Celu` | `nn.CELU()` | `alpha` 1.0 |
| `Gelu` | `nn.GELU()` | exact (erf) form, `approximate='none'`; `GeluApprox::Tanh` is `approximate='tanh'` |
| `Softplus` | `nn.Softplus()` | `beta` 1, `threshold` 20 |
| `Hardtanh` | `nn.Hardtanh()` | `min` -1, `max` 1 |
| `Prelu` | `nn.PReLU()` | slopes start at 0.25 |

`Prelu` names its slopes `weight`, as PyTorch does, so a PyTorch `state_dict` with PReLU slopes loads into it.

The formulas match; the last bits of a value need not. In f32, `Tanh` and `Sigmoid` run SIMD kernels whose results are within 3 ulp of the correctly rounded value, a relative error of at most about 3.6e-7. PyTorch's CPU kernels make their own approximations, so a transferred network's outputs agree to rounding, not bit for bit.

## Where Cavitation differs on purpose

**Masks act on a neuron's final output.** PyTorch has no neuron mask; you would multiply an output by a 0/1 tensor, or write a hook. In Cavitation a neuron mask, set with `Sequential::set_neuron_mask` or `set_neuron`, replaces the neuron's output with exactly 0 and passes no gradient through it. It selects instead of multiplying, so an infinite output gives 0 rather than NaN. The mask is not placed on the `Linear` itself but at `Sequential::neuron_layer`, after the activations that follow it.

**Neuron masks land after the last activation of a run.** A mask on a `Linear`'s own output would be undone by an activation that maps 0 to something else: a sigmoid turns a masked 0 into 0.5. So `neuron_layer` is the last of the element-wise layers that follow the `Linear`, where the neuron's output is final. `Sequential::set_mask` puts a mask on a layer by its own index, for the rare case where you want it there.

**Health checks run on every step.** PyTorch trains on whatever the loss gives, and a NaN spreads through every weight in a few steps without a message. Cavitation checks the loss and the gradients of each step for NaN and infinity, and by default stops with `Error::NonFinite` naming the step and where the value was. It also warns about exploding and vanishing gradients, and reports dead ReLU neurons. The settings are in `Health`; `Health::quiet` turns the printing off, and `OnNonFinite::Warn` keeps training after a non-finite value. For the same reason ReLU passes a NaN through as NaN instead of turning it into 0.

**Data errors are `InvalidData`.** `fit` checks the data against the model and the loss before it trains: the kind of targets, every class label below the output width, and the input width. A problem is an `Error` (`Error::InvalidData`, or `Error::ShapeMismatch` for the input width), not a failure in the middle of epoch 3. A mean-squared-error loss on class labels is a typical case.

**Shape errors panic.** Building a network and stepping it by hand is code you write next to the model, so a wrong shape inside it is a programming error: `Trainer::step` and the `Sequential` mask calls panic with a message naming the shapes. The calls that a program drives from data or from a user's input, such as `InferenceModel::set_mask` and `Evaluator::set_neuron_mask`, return a `Result`.

## Saving and loading

`save` writes every parameter as an `F32` `.safetensors` file, named by `Sequential::parameter_names`: `0.weight`, `0.bias`, `2.weight` and so on, where the number is the layer's index in the `Sequential`. A `Linear` stores its weight as `[outputs, inputs]`, the layout of PyTorch's `nn.Linear`, so the file is what `safetensors.torch.save_file(model.state_dict(), path)` writes for the same `nn.Sequential`, and loads there too. `load` reads `F32` files, and `F16`, `BF16` and `F64` ones too: half precision widens to `F32` exactly, and `F64` rounds to the nearest `F32`, as `.float()` does in PyTorch.

```rust,ignore
{{#include ../../crates/cavitation/examples/from_pytorch.rs:save}}
```

```text
0.weight: [32, 64]
0.bias: [32]
2.weight: [10, 32]
2.bias: [10]
```

`load` is strict. Every parameter must be in the file with its exact shape and type, and the file must have no other key. One `Error::Keys` lists every missing, unexpected and wrong-shape key together, and says when a weight looks transposed. Nothing is written into the parameters unless every check passes. A file made by `torch.save` (a zip archive or a pickle) is refused with a message that says to save with safetensors instead.

A layer built with `nn.Linear(inputs, outputs, bias=False)` is `Linear::without_bias(&mut params, inputs, outputs, init, &mut rng)` in Rust, or `Linear::from_weight` when you already have the weight. It computes `x · Wᵀ`, its only key is `{i}.weight`, and its `state_dict` loads with `load`. A `Linear::new` in its place would ask for a `{i}.bias` the file does not have, and `load` would report it as missing. `save` writes a bias-free layer the same way, and PyTorch loads the file into the `bias=False` network.

`load_lenient` ignores the keys the model does not have and returns them, so you can see what was dropped, for example the slopes of a PReLU loaded into a network that uses ReLU. For a file whose names differ from the model's, because of a `module.` prefix from `DataParallel` or because layers sit at other indices, `load_with` takes `LoadOptions` with a renaming function, and `LoadOptions::strip_prefix` covers the prefix.

Training in PyTorch and running in Cavitation then takes three steps: build the same `Sequential` in Rust, `load` the file, and compare an output on a few rows before trusting it.
