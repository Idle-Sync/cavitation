# Image models

Convolution, pooling, batch normalisation and dropout run on the same tape, in the same `fit`, and in the same `InferenceModel` as the dense layers. This chapter covers:

- how an image is laid out as a row;
- a LeNet-5-style network trained on MNIST;
- what changes between training and evaluation;
- loading a CNN trained in PyTorch;
- a small steering model trained on synthetic road images.

The programs are three examples in `crates/cavitation/examples/`: `mnist_cnn.rs`, `load_torch_cnn.rs` and `steering_cnn.rs`.

## Images as rows

A `Dataset` holds one image per row, and a row holds the image's values in PyTorch's order: channel by channel, each channel row by row. A 3 × 32 × 32 colour image is a row of 3,072 values: the 1,024 red values, then the green, then the blue. A batch of images is then a `[batch, channels · height · width]` matrix, the same shape every other layer takes. So batching, shuffling, masks, hooks, `save` and `load` work on image models without change.

An image layer carries its geometry, because a row does not say which of its values sit next to each other:

- `Conv2dGeom` holds a convolution's channels in and out, the image size, the kernel, the stride, the padding and the dilation.
- `Pool2dGeom` holds a pooling's channels, image size, window, stride, padding, dilation and `ceil_mode`.

Each one checks the sizes when it is made and computes the output size, as PyTorch's formulas do. A layer's output is again a flat row, in the same channel-by-channel order, so the next image layer reads it with the geometry of that output. `Flatten` changes nothing; it is there so that a network reads like its PyTorch twin and its layer indices, and so its state-dict keys, match.

## A CNN on MNIST

`mnist_cnn` builds LeNet-5's layout: two 5 × 5 convolutions, each followed by ReLU and 2 × 2 max pooling, then three `Linear` layers. Each comment gives the shape of the output, channels by height by width:

```rust,ignore
{{#include ../../crates/cavitation/examples/mnist_cnn.rs:network}}
```

`Conv2d::new` initialises the weight and bias as PyTorch's `nn.Conv2d` does. It draws them uniformly within `1 / √(in · kh · kw)`. The network has 61,706 parameters.

It trains with `fit`, Adam at 1e-3 and batches of 64, and prints the test accuracy after each epoch:

```rust,ignore
{{#include ../../crates/cavitation/examples/mnist_cnn.rs:fit}}
```

The run turns off the health checks' dead-ReLU report. The report counts each position of a convolution's output as a neuron. The blank border of a digit leaves many positions at 0 for a whole epoch, but the channel there is not dead.

Run it with `scripts/data/get-mnist.sh` once, then `cargo run --release --example mnist_cnn -p cavitation`. On an Intel Celeron N5105 (4 cores, kernel level `sse4.2`, 4 threads, with other work running on the machine) it printed:

```text
epoch 1: mean loss 0.2423, test accuracy 97.75%, 17.0 s
epoch 2: mean loss 0.0680, test accuracy 98.26%, 20.0 s
epoch 3: mean loss 0.0499, test accuracy 98.43%, 18.0 s
epoch 4: mean loss 0.0383, test accuracy 98.49%, 17.4 s
epoch 5: mean loss 0.0323, test accuracy 98.85%, 17.7 s
```

`cavitation-bench mnist-cnn` trains the same network beside PyTorch, from the same starting weights and in the same batch order, and prints both sides' epochs.

## Training and evaluation

Dropout and batch normalisation behave differently while training. A training tape, `graph.training_tape(&params.values, seed, step)`, records the training behaviour. The plain `graph.tape(&params.values)` records evaluation. `fit` and `Trainer` open a training tape for every step. `evaluate`, `accuracy`, `Evaluator` and `InferenceModel` always evaluate.

- **`Dropout(p)`.** While training it sets each value to 0 with probability `p` and scales the others by `1 / (1 − p)`. In evaluation it passes values through. Its masks come from `FitConfig::dropout_seed`, the step's number and the dropout layer's place on the tape. So a run gives the same masks at any thread count and on any machine, but not PyTorch's masks.
- **`BatchNorm2d`.** While training it normalises each channel by the batch's mean and variance. In evaluation it uses its running mean and variance. The running statistics are buffers, not parameters: they take no gradient and the optimiser never touches them. A training tape only reads the parameter values, so batch norm records each batch's mean and unbiased variance on the graph, and `graph.apply_buffer_updates(&mut params.values)` writes them into the buffers after the step. `fit` and `Trainer` make that call for you. In a training loop of your own, call it after the optimiser's step, as the README's manual loop does.

A convolution right before a batch norm needs no bias. Batch norm subtracts each channel's mean, which cancels any per-channel bias. The bias then gets a gradient of 0 up to rounding, and the health checks report it as vanishing. Build such a convolution with `Conv2d::with_init(params, geom, Init::TorchDefault, None, rng)`, which is PyTorch's `bias=False`.

## Loading a CNN from PyTorch

A PyTorch `nn.Sequential` built from these layers saves its `state_dict` with `safetensors.torch.save_file`. Build the same layers in the same order in Cavitation, and `load` reads that file. The keys are the layer's index followed by the tensor's name, so the layer positions must match. That includes the parameter-free layers (`ReLU`, `MaxPool2d`, `Dropout`, `Flatten`), which take an index too.

`load_torch_cnn` loads the CIFAR-10 network the library's fixtures keep. It has four `Conv2d(3 × 3, padding 1, bias=False) → BatchNorm2d → ReLU` blocks, max pooling after each pair, dropout and two `Linear`s. The weights are seeded and untrained, and the file was saved from PyTorch in F16. The layout:

```rust,ignore
{{#include ../../crates/cavitation/examples/load_torch_cnn.rs:network}}
```

`inspect_file` lists what the file holds before anything is loaded:

```rust,ignore
{{#include ../../crates/cavitation/examples/load_torch_cnn.rs:inspect}}
```

The batch norm at index 1 brings five tensors. Its `weight` and `bias` are γ and β, loaded as parameters. Its `running_mean`, `running_var` and `num_batches_tracked` are loaded into the layer's buffers. PyTorch stores the count as an `I64` scalar, and `save` writes it the same way. F16 and BF16 values widen to F32 exactly when loaded.

```rust,ignore
{{#include ../../crates/cavitation/examples/load_torch_cnn.rs:load}}
```

The example then runs eight images through `InferenceModel` and compares the outputs with PyTorch's, which were computed in float64 with the same weights and stored beside the file:

```rust,ignore
{{#include ../../crates/cavitation/examples/load_torch_cnn.rs:run}}
```

`load_torch_cnn` prints, at the reference kernel level on one thread:

```text
28 tensors in the file, for example:
  1.num_batches_tracked  I64  []
  1.bias                 F16  [32]
  1.running_mean         F16  [32]
  1.running_var          F16  [32]
  1.weight               F16  [32]
image 0: class 1 here, 1 in PyTorch
image 1: class 1 here, 1 in PyTorch
image 2: class 1 here, 1 in PyTorch
image 3: class 1 here, 1 in PyTorch
image 4: class 1 here, 1 in PyTorch
image 5: class 1 here, 1 in PyTorch
image 6: class 1 here, 1 in PyTorch
image 7: class 1 here, 1 in PyTorch
largest difference from PyTorch's float64 output: 6.9e-8
```

The network is untrained, so it puts every image in the same class. What the run shows is that the two sides agree on all 80 outputs to within 6.9e-8. The fixture tests check the same network layer by layer, each layer within its own rounding bound, at every kernel level.

## Steering from road images

`cavitation_datasets::steering::generate(n, seed)` draws 32 × 32 grey images of a road edge, bright road on its right and darker ground on its left:

- The edge leans by an angle between −30° and 30°, and that angle in degrees is the target.
- Each image also gets a random brightness offset and pixel noise, so the network has to find the edge rather than read one pixel.

`steering::network` is a small CNN of two strided convolutions, a pooling and two `Linear`s, with 9,633 parameters. `steering::recipe` trains it with Adam on the mean squared error of the angle. The example trains on 2,000 images and measures the mean absolute error on 500 others after each epoch:

```rust,ignore
{{#include ../../crates/cavitation/examples/steering_cnn.rs}}
```

`steering_cnn` prints, at the reference kernel level on one thread:

```text
epoch 1: train loss 186.158, test mean absolute error 5.43°
epoch 2: train loss 23.563, test mean absolute error 3.18°
epoch 3: train loss 9.112, test mean absolute error 1.62°
epoch 4: train loss 3.853, test mean absolute error 1.12°
epoch 5: train loss 2.073, test mean absolute error 0.87°
epoch 6: train loss 1.632, test mean absolute error 0.73°
epoch 7: train loss 1.199, test mean absolute error 0.75°
epoch 8: train loss 1.069, test mean absolute error 0.80°
angle   15.9°, network   15.1°
angle   22.8°, network   23.4°
angle   18.9°, network   20.4°
angle   13.7°, network   12.9°
angle    2.0°, network    1.1°
final test mean absolute error 0.80°
```

After 8 epochs the network reads the angle within 0.8° on average, from images it has not seen. A known-result test, `steering_known_result`, requires this run to stay below 1.5°.

## What is not here yet

The shrink tools work on dense networks and refuse a model with an image layer; shrinking a CNN by channel is a later feature. Batch norm runs as its own step in `InferenceModel` rather than folded into the convolution before it, which keeps the compiled model's outputs equal to the tape's.
