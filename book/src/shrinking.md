# Shrinking a network

A trained network is often larger than it needs to be. This chapter switches neurons off to see what each one is worth, lets `shrink` find how many the digits network can lose, removes them for good with `compact`, and ends with a question pruning raises: is it the kept neurons' starting weights that make the small network work, or only its size?

The program is `crates/cavitation/examples/shrink_digits.rs`. Run it with:

```sh
cargo run --release --example shrink_digits -p cavitation -- --ablate --lottery
```

Without the two flags it only trains, shrinks, compacts and saves. The tools are in `cavitation::cavitation_nn::shrink`; they are not in the prelude.

The printed blocks in this chapter come from that command, run on an Intel Celeron N5105 (4 cores) at kernel level `sse4.2`, rustc 1.99.1. The accuracies are the same on every run there; on other kernel levels the rounding differs and so can the figures.

## The network and the rows it never sees

The network is the digits network of the known result: 64 inputs (an 8×8 image), 256 and 128 ReLU neurons, 10 outputs, 50,826 parameters, trained with Adam at 0.002 in batches of 32 for 20 epochs. Here it is trained on the first 1,149 of the 1,437 training digits only:

```rust,ignore
{{#include ../../crates/cavitation/examples/shrink_digits.rs:split}}
```

The other 288 rows are the *validation set*. Every decision below, which neurons matter and when to stop removing them, is measured on those rows. The 360 test digits are only looked at once the decisions are made.

The rows have to be held out *before* the first training, not after. A network keeps answering the rows it was trained on correctly long after it has started to fail on new ones, so a stop rule measured on training rows sees nothing change until it is far too late. The test set is held back for the opposite reason: a size chosen by looking at the test digits has been tuned to them, and the test accuracy no longer tells you how the network does on new data.

## Switching one neuron off by hand

A neuron mask switches neurons off without changing any weight: the neuron's output is replaced by exactly 0 on every row. `set_neuron_mask` takes the index of the `Linear` whose neurons it masks (layer 0 here, the first hidden layer; layer 2 is the second) and puts the mask after that layer's activation, where the neuron's output is final:

```rust,ignore
{{#include ../../crates/cavitation/examples/shrink_digits.rs:by_hand}}
```

The example prints:

```text
neuron 0 of layer 0 off: validation loss 0.0487 -> 0.0512, accuracy 98.96% -> 98.96%
```

The loss rises a little, and not one of the 288 answers changes. A masked neuron also passes no gradient, to its incoming weights or its outgoing ones, so retraining with a fresh optimiser and no weight decay leaves its weights as they were, and clearing the mask brings it back.

## The ablation sweep

`ablate` does the same for every neuron of the layers you list, one at a time, and ranks them by what each one's absence costs, the most costly first. For class targets the cost is the accuracy points lost, counted in whole rows (one row of 288 is 0.35 points):

```rust,ignore
{{#include ../../crates/cavitation/examples/shrink_digits.rs:sweep}}
```

```text
layer 0 neuron 152: +1.04 points
layer 0 neuron 199: +1.04 points
layer 2 neuron   2: +1.04 points
layer 0 neuron  63: +0.69 points
layer 0 neuron  69: +0.69 points
of 384 neurons: 61 cost accuracy alone, 318 change no answer, 5 help when off
```

No single neuron is worth more than three validation rows, and 318 of the 384 change no answer at all when they are the only one missing. That does not mean 318 of them can go. The network spreads what it knows across its neurons, so each one is backed up by others; remove many together and the backups go too. Ablation answers "what does this neuron do in the network as it is", which is the right question for reading a network, not for cutting most of it away. Measures of activity tell you about the same thing from the other side: `activity` reports each neuron's mean, mean absolute value and the share of rows on which it fires, and a neuron that never fires on the training rows can be removed for free.

## `shrink` and its stop rule

`shrink` removes neurons a step at a time and lets the network recover after each step:

```rust,ignore
{{#include ../../crates/cavitation/examples/shrink_digits.rs:shrink}}
```

Each step removes, from each hidden layer, neurons up to the next level: 10%, 20%, … 70% of the layer's width, then 75%, 80%, … 95%, where the cliff usually is and finer steps pay. Within a layer it removes first the neurons that never fire on the training rows, then those with the smallest outgoing weights (the norm of the neuron's column in the next layer's weight: a neuron the next layer barely listens to). Then it retrains for 3 epochs with a fresh Adam at 1e-3 and measures the network on the validation set. The ranking is computed again at every step, on the network as the last retraining left it.

A removed neuron takes its average output with it: the next layer loses not only what the neuron says about each row but the constant it adds on average, and the retraining has to make that up. With `ShrinkConfig { fold_means: true, .. }` each removed neuron's mean output on the training rows is added into the next layer's bias before retraining (`b[i] += W[i, j] · mean_j`), measured on the network with the earlier layers of the step already shrunk. For a neuron whose output never changes the fold is exact, and the network's outputs stay as they were. It is off by default; it is for runs that retrain little or not at all, where nothing else makes up the lost average.

A step *meets* the criterion when its validation accuracy is no more than 1 point below the unpruned network's. The run keeps the latest step that met it, not the step before the first miss, because on a few hundred rows one unlucky row can dip below the line and a later step recover; it ends after two misses in a row or at the last level. With fewer validation rows than the threshold can resolve (one row worth more than half a point here, so fewer than 200 rows), `ShrinkReport::warnings` says so.

This is the curve of that run, with the test accuracy of every step's network beside it, from `cavitation-bench shrink` (run 2, `ShrinkConfig::default()`), measured on an Intel Celeron N5105 (4 cores) at kernel level `sse4.2`:

| removed | kept | parameters | validation | test | met |
|---|---|---|---|---|---|
| none | 256 + 128 | 50,826 | 285/288 (98.96%) | 352/360 (97.78%) | |
| 10% | 231 + 116 | 43,097 | 286/288 (99.31%) | 351/360 (97.50%) | yes |
| 20% | 205 + 103 | 35,583 | 284/288 (98.61%) | 352/360 (97.78%) | yes |
| 30% | 180 + 90 | 28,900 | 284/288 (98.61%) | 352/360 (97.78%) | yes |
| 40% | 154 + 77 | 22,725 | 284/288 (98.61%) | 351/360 (97.50%) | yes |
| 50% | 128 + 64 | 17,226 | 286/288 (99.31%) | 352/360 (97.78%) | yes |
| 60% | 103 + 52 | 12,633 | 283/288 (98.26%) | 353/360 (98.06%) | yes |
| 70% | 77 + 39 | 8,447 | 284/288 (98.61%) | 350/360 (97.22%) | yes |
| 75% | 64 + 32 | 6,570 | 283/288 (98.26%) | 351/360 (97.50%) | yes |
| 80% | 52 + 26 | 5,028 | 282/288 (97.92%) | 350/360 (97.22%) | no |
| 85% | 39 + 20 | 3,545 | 280/288 (97.22%) | 348/360 (96.67%) | no |

The run keeps 75%: 64 + 32 neurons, 6,570 parameters, 7.7 times fewer than the full network, at 351 of 360 test digits against 352 unpruned. At 80% it is 3 rows down on validation, one more than the 1-point line allows (1 point of 288 rows is 2.88 rows). The validation loss rises more than threefold, from 0.030 at 10% to 0.106 at 80% (with a dip at 50%, to 0.039), while the accuracy holds; that is why the default criterion is an accuracy drop and not a loss rise, which would stop long before the answers change. For value targets, which have no accuracy, you give `Criterion::LossIncrease` yourself.

The benchmark's run 1 ranks by outgoing-weight norm only (`dead_first: false`) and runs every level (`patience: None`). It also keeps 75%, at 354/360, and shows where the cliff is: 84.38% on validation at 90% removed, 53.82% at 95%. The figures are the same on every run, since the seeds are fixed; on a CPU with another kernel level (AVX2, NEON or Accelerate) the rounding differs and so can the curve. The known-result test `shrink_known_result` runs both and checks them:

```sh
cargo test --release -p cavitation --test known_results -- --ignored shrink_known_result --nocapture
```

## Compaction

A masked network is no faster: its matrices are as large as before, and the masks add a step. `compact` builds the smaller dense network: it keeps the rows of each hidden `Linear`'s weight and bias for the neurons that are on, and the matching columns of the next `Linear`'s weight. The result is an ordinary `Sequential` with its own `ParamStore`, with the same parameter names, so it saves, loads, trains and inspects like any other:

```rust,ignore
{{#include ../../crates/cavitation/examples/shrink_digits.rs:compact}}
```

A switched-off neuron adds an exact 0 to every sum in the next layer, and adding 0 changes no rounding, so on the `reference` and `sse4.2` kernel levels the compacted network gives the masked one's outputs bit for bit through `InferenceModel::infer`, and through the batched paths as long as no sum is longer than 256 terms. On the other levels the vector kernels round by position, and the outputs agree within the rounding bound that `compact`'s documentation states. `cavitation-bench shrink` checks it on the 360 test digits: 0 of 3,600 outputs differ in their bits, through `infer` and through `infer_batch`, for the 80% step and for the chosen network.

The same benchmark times one test digit through `InferenceModel::infer`, 20,000 warm-up calls per network, then 301 rounds of 1,000 calls of each network in turn. Measured on an Intel Celeron N5105 (4 cores; pinned to one, so 1 thread), kernel `sse4.2`, rustc 1.99.1, with `cargo build --release -p cavitation-bench && taskset -c 2 target/release/cavitation-bench shrink` (the chosen network is run 1's, the same 64 + 32 size as the defaults choose):

| network | parameters | median µs per call (p10–p90) | speed-up |
|---|---|---|---|
| full, 256 + 128 | 50,826 | 6.808 (6.646–7.850) | 1.00× |
| full, with the chosen masks on (64 + 32 on) | 50,826 | 7.095 (6.941–8.720) | 0.96× |
| the chosen network compacted, 64 + 32 | 6,570 | 1.006 (0.984–1.054) | 6.77× |

The compacted network has 7.7 times fewer parameters and is 6.77 times faster. The speed-up is smaller than the parameter ratio: every call has a fixed cost for its layer steps and copies, which counts for more the smaller the network.

## Is it the weights, or the size?

The chosen network got its weights by training the full network for 20 epochs and then retraining for 3 epochs after each of the 8 cuts up to the one kept: 44 epochs in all. Two other networks of the same size, 64 + 32, can be trained from the start with the same recipe for the same 44 epochs, and the example's `--lottery` flag does both:

- the kept neurons with the weights they had *before* training: `compact` applied to the initial parameters with the final masks, then trained alone;
- a network of the same size from new random weights.

The first is the experiment of the lottery ticket hypothesis (Frankle and Carbin, 2019), which says a large network contains small ones whose starting weights train well on their own. They pruned single weights by magnitude, not whole neurons; here a "ticket" is a set of neurons. Saving the starting point takes one line before training:

```rust,ignore
{{#include ../../crates/cavitation/examples/shrink_digits.rs:initial}}
```

and the ticket is the final masks on those weights:

```rust,ignore
{{#include ../../crates/cavitation/examples/shrink_digits.rs:lottery}}
```

The second needs no starting point saved. `shrink::train_found_size` takes the pruned network as `shrink` left it, builds the network `compact` would make of it with every weight started again (here as the example's `network` starts them: He uniform weights, zero biases, seed 1), trains it with the `FitConfig` you give, and measures both networks on the rows you give, with the recipe's loss:

```rust,ignore
{{#include ../../crates/cavitation/examples/shrink_digits.rs:scratch}}
```

`found.pruned` and `found.scratch` are the two networks' `Evaluation`s, and `found.model` and `found.params` the fresh network, trained. Measure on rows neither network was chosen or trained on, the test digits here: the size was picked on the validation rows, which flatter the pruned network.

Measured on an Intel Celeron N5105 (4 cores) at kernel level `sse4.2`, rustc 1.99.1, with `cargo run --release --example shrink_digits -p cavitation -- --ablate --lottery` (the accuracies are the same on every run there):

| 64 + 32 network, 44 epochs of training in all | test accuracy |
|---|---|
| pruned with `shrink` and retrained | 351/360 (97.50%) |
| kept neurons from their initial weights, trained alone | 349/360 (96.94%) |
| new random weights, trained alone | 351/360 (97.50%) |

Given the same number of epochs, the network trained from new random weights matches the pruned one, and the ticket is 2 digits behind both. On this problem what pruning found was the size: once you know that 64 + 32 neurons are enough, you can train a network of that size directly. That is one seed of one problem, so read it as an example of the comparison, not a law; on other problems the pruned network can come out ahead. Run the comparison on your own problem, with equal training, before you settle on the pruned network. Compare the training too: an unequal number of epochs can make either side look better than it is.

## Try it

- Set `ShrinkConfig::criterion` to `Some(Criterion::AccuracyDrop { points: 3.0 })` and watch the run go further down the curve, towards the cliff.
- Try `Ranking::Activity` or `Ranking::Random(seed)` instead of the default, and compare where each one stops.
- Load the saved network with `load` into a `Sequential` of widths 64, 64, 32, 10 and check that it gives the chosen network's test accuracy.
