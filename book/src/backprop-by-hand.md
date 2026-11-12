# Backprop by hand

A network with one hidden layer learns XOR. This chapter writes out its forward pass, the local derivative of each layer and the gradients that backpropagation computes, then reads every one of those numbers out of the running program with hooks.

The program is `crates/cavitation/examples/xor.rs`. Run it with:

```sh
cargo run --release --example xor -p cavitation
```

## The problem

XOR has two inputs and one output, which is 1 when exactly one input is 1:

| x₁ | x₂ | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No straight line separates the two cases whose output is 1 from the two whose output is 0, so a single linear layer cannot learn it, whatever its weights. A hidden layer with a non-linear activation can: it bends the plane so that a line in the hidden layer's space does the job.

## The network

```rust,ignore
{{#include ../../crates/cavitation/examples/xor.rs:network}}
```

`Sequential` runs its layers in order and numbers them from 0:

| Layer | What it is | Output for the batch of 4 |
|---|---|---|
| 0 | `Linear`, 2 inputs to 8 outputs: weight `0.weight` of shape [8, 2], bias `0.bias` of shape [8] | z₁, shape [4, 8] |
| 1 | `Tanh` | a₁, shape [4, 8] |
| 2 | `Linear`, 8 inputs to 1 output: weight `2.weight` of shape [1, 8], bias `2.bias` of shape [1] | z₂, shape [4, 1] |
| 3 | `Sigmoid` | ŷ, shape [4, 1] |

That is 2·8 + 8 + 8·1 + 1 = 33 parameters. A `Linear` layer stores its weight as [outputs, inputs], as PyTorch does, so the parameter names and shapes are the ones a PyTorch `state_dict` of the same network would have. The four XOR cases go in together as one batch: `x` is a [4, 2] tensor, one row per case, and `y` is [4, 1].

## The forward pass

Written out for the whole batch, with Wᵀ the transpose of W:

```text
z₁ = x W₁ᵀ + b₁        [4, 2] × [2, 8] + [8]  →  [4, 8]
a₁ = tanh(z₁)          element by element     →  [4, 8]
z₂ = a₁ W₂ᵀ + b₂       [4, 8] × [8, 1] + [1]  →  [4, 1]
ŷ  = σ(z₂)             σ(z) = 1 / (1 + e⁻ᶻ)   →  [4, 1]
```

The bias is added to every row: that is the broadcasting bias add.

The loss is the mean squared error over the four outputs:

```text
L = ¼ · Σᵢ (ŷᵢ − yᵢ)²
```

## Local derivatives

Backpropagation needs, for each layer, how its output changes when its input changes. These local derivatives use only values the forward pass already computed:

```text
tanh:     ∂a/∂z = 1 − tanh²(z) = 1 − a²
sigmoid:  ∂ŷ/∂z = σ(z) · (1 − σ(z)) = ŷ · (1 − ŷ)
linear:   z = a Wᵀ + b, so ∂z/∂a is W, ∂z/∂W is a, and ∂z/∂b is 1
loss:     ∂L/∂ŷᵢ = ½ · (ŷᵢ − yᵢ)
```

The activations act on each value on its own, so their local derivative is one number per value, f′(z), the same shape as z. This is what a hook reports as `local_derivative`. A `Linear` layer mixes its inputs, so its derivative is a matrix (its weight), and a hook reports no element-wise derivative for it.

## Gradients: the chain rule, layer by layer

The gradient of the loss with respect to any value is the gradient with respect to the value it feeds into, times the local derivative between them. Starting at the loss and walking back through the layers, with ⊙ for element-by-element multiplication:

```text
∂L/∂ŷ  = ½ · (ŷ − y)                    [4, 1]
∂L/∂z₂ = ∂L/∂ŷ ⊙ ŷ · (1 − ŷ)            [4, 1]   through the sigmoid
∂L/∂W₂ = (∂L/∂z₂)ᵀ a₁                   [1, 8]   gradient of 2.weight
∂L/∂b₂ = sum of the rows of ∂L/∂z₂      [1]      gradient of 2.bias
∂L/∂a₁ = ∂L/∂z₂ W₂                      [4, 8]   back through the second Linear
∂L/∂z₁ = ∂L/∂a₁ ⊙ (1 − a₁²)             [4, 8]   through the tanh
∂L/∂W₁ = (∂L/∂z₁)ᵀ x                    [8, 2]   gradient of 0.weight
∂L/∂b₁ = sum of the rows of ∂L/∂z₁      [8]      gradient of 0.bias
```

Two patterns repeat. Through an activation, the gradient is multiplied element by element by f′(z). Through a `Linear` layer, the gradient with respect to its weight is the gradient at its output times its input, summed over the batch, and the gradient passed further back is the gradient at its output times its weight. The input `x` is data, not a parameter, so nothing computes ∂L/∂x.

Every gradient has the shape of the value it belongs to. The weight gradients add up the contribution of all four cases; that sum is why the batch dimension disappears from them.

## The training loop

```rust,ignore
{{#include ../../crates/cavitation/examples/xor.rs:loop}}
```

Each epoch is one step on the full batch:

1. `graph.tape(&params.values)` opens a tape. Each operation recorded on it computes its value straight away and remembers what its backward pass needs. The `Graph` keeps the buffers between steps, so after the first steps recording allocates nothing.
2. `tape.input` records the batch, `net.forward_traced` records the four layers (the trace keeps each layer's input and output for the hooks and health checks), and `tape.mse` records the loss.
3. `tape.backward(loss, &mut params.grads)` walks the tape from the loss back to the start, applying the rules of the previous section, and writes ∂L/∂W and ∂L/∂b for every parameter into `params.grads`.
4. The monitor checks that the loss and the gradients are finite and warns about gradients that explode or vanish.
5. `drop(tape)` ends the tape's borrow of `params.values`, so that `optimizer.step` can change them. Plain SGD subtracts the learning rate times each gradient: W ← W − 0.5 · ∂L/∂W.
6. `params.zero_grad()` marks the gradients stale, so the next `backward` overwrites them instead of adding to them.

The loss falls from 0.26 to under 0.001 over the 2,000 steps.

## Reading it through hooks

After training, the example runs one more forward and backward pass with a hook on every layer. A forward hook sees the layer's input and output and, for an activation, f′(z). A backward hook also sees ∂L/∂(output) and ∂L/∂(input). The example prints only the last of the four rows, the one for the input (1, 1):

```rust,ignore
{{#include ../../crates/cavitation/examples/xor.rs:hooks}}
```

Hooks run when you ask, on a tape that has been recorded (`run_forward`) and differentiated (`run_backward`):

```rust,ignore
{{#include ../../crates/cavitation/examples/xor.rs:run}}
```

Finally the example prints the parameter gradients that `backward` left in `params.grads`:

```rust,ignore
{{#include ../../crates/cavitation/examples/xor.rs:grads}}
```

The whole output:

```text
epoch    1: loss 0.261309
epoch  400: loss 0.006758
epoch  800: loss 0.002248
epoch 1200: loss 0.001296
epoch 1600: loss 0.000898
epoch 2000: loss 0.000682

Hook output for the input (1, 1), after training:
forward  layer 0 (linear)
  input         : [  1.000,   1.000]
  output        : [  0.514,  -0.734,  -1.755,   0.639,  -3.501,  -1.741,  -2.875,  -2.708]
forward  layer 1 (tanh)
  input  (z)    : [  0.514,  -0.734,  -1.755,   0.639,  -3.501,  -1.741,  -2.875,  -2.708]
  output (a)    : [  0.473,  -0.626,  -0.942,   0.564,  -0.998,  -0.940,  -0.994,  -0.991]
  f'(z)         : [  0.776,   0.609,   0.113,   0.682,   0.004,   0.116,   0.013,   0.018]
forward  layer 2 (linear)
  input         : [  0.473,  -0.626,  -0.942,   0.564,  -0.998,  -0.940,  -0.994,  -0.991]
  output        : [ -3.406]
forward  layer 3 (sigmoid)
  input  (z)    : [ -3.406]
  output (a)    : [  0.032]
  f'(z)         : [  0.031]
backward layer 3 (sigmoid)
  dL/d(output)  : [  1.61e-2]
  dL/d(input)   : [  4.99e-4]
backward layer 2 (linear)
  dL/d(output)  : [  4.99e-4]
  dL/d(input)   : [ -2.63e-4,   2.63e-4,   1.86e-3,  -2.48e-4,  -1.05e-3,   2.03e-3,  -6.70e-4,  -5.27e-4]
backward layer 1 (tanh)
  dL/d(output)  : [ -2.63e-4,   2.63e-4,   1.86e-3,  -2.48e-4,  -1.05e-3,   2.03e-3,  -6.70e-4,  -5.27e-4]
  dL/d(input)   : [ -2.04e-4,   1.60e-4,   2.10e-4,  -1.69e-4,  -3.80e-6,   2.35e-4,  -8.47e-6,  -9.30e-6]
backward layer 0 (linear)
  dL/d(output)  : [ -2.04e-4,   1.60e-4,   2.10e-4,  -1.69e-4,  -3.80e-6,   2.35e-4,  -8.47e-6,  -9.30e-6]

Parameter gradients over the whole batch:
  0.weight: [ -3.58e-6,  -3.10e-5,   7.59e-5,  -4.35e-6,   2.07e-4,  -1.60e-4,  -3.06e-5,  -9.23e-6,   9.13e-5,   9.63e-5,  -1.87e-4,   2.32e-4,   5.97e-5,   7.22e-5,   5.62e-5,   6.11e-5]
  0.bias  : [  1.18e-4,  -3.10e-5,   5.83e-5,   7.72e-5,  -3.34e-5,   7.04e-5,   5.19e-6,   2.03e-5]
  2.weight: [  1.35e-4,  -1.00e-4,  -4.37e-4,   1.10e-4,   1.85e-4,  -5.39e-4,   1.31e-4,   1.22e-4]
  2.bias  : [ -1.07e-4]

Predictions (targets 0, 1, 1, 0):
  [  0.015,   0.974,   0.972,   0.032]
health: no warnings
```

The seeds are fixed, so the program prints these numbers every time. On a CPU that runs a different kernel level (AVX2, NEON or Accelerate) the last digits can differ, because those kernels round and add the products of a matrix multiplication in a different order.

## Checking the numbers by hand

Every backward line follows from the forward lines and the rules above.

**The loss gradient.** For (1, 1) the target is 0 and the prediction ŷ is 0.032, so ∂L/∂ŷ = ½ · (0.032 − 0) = 0.016. The hook prints 1.61e-2.

**Through the sigmoid.** f′(z) is ŷ · (1 − ŷ) = 0.032 · 0.968 = 0.031, the forward hook's `f'(z)`. Then ∂L/∂z₂ = 0.0161 · 0.031 = 4.99e-4, the sigmoid's `dL/d(input)`.

**Through the second `Linear`.** Its `dL/d(output)` is the sigmoid's `dL/d(input)`: one layer's input is the previous layer's output, so the two gradients are the same tensor. Its `dL/d(input)` is ∂L/∂z₂ times its weight, one number per hidden neuron, so dividing that line by 4.99e-4 gives back the weight `2.weight`: the first neuron's weight is −2.63e-4 / 4.99e-4 ≈ −0.53.

**Through the tanh.** Each value of ∂L/∂z₁ is ∂L/∂a₁ times f′(z₁) at the same position. For the first neuron: −2.63e-4 · 0.776 = −2.04e-4. Check f′ too: 1 − a² = 1 − 0.473² = 0.776.

**The first layer.** It gets ∂L/∂z₁ and has no `dL/d(input)`, because its input is the batch.

**The parameter gradients.** These add up all four rows, so the single row printed above is one of four contributions. Two of them can be seen directly. The row for (0, 0) adds nothing to `0.weight`, because ∂L/∂W₁ multiplies by x, which is 0 there. And `2.bias` is the sum of ∂L/∂z₂ over the four rows: the (1, 1) row adds 4.99e-4, and the other three bring the total to −1.07e-4.

## What the numbers show

Look at the fifth hidden neuron. Its z is −3.501, deep in the flat part of tanh, so f′(z) is 1 − 0.998² ≈ 0.004. A gradient of −1.05e-3 arrives at its output and only −3.80e-6 leaves through its input, about 280 times smaller. A saturated neuron passes almost no gradient back, so the weights feeding it barely move. In a deep network this happens layer after layer and the gradient vanishes; the health checks warn when a parameter's gradient norm falls below 1e-7.

The output layer has the same effect: at ŷ = 0.032 the sigmoid's f′ is 0.031, so the loss gradient is cut to about a thirtieth before it reaches the network. That is part of why classifiers pair a sigmoid or softmax with cross-entropy instead of squared error: the cross-entropy gradient with respect to z is ŷ − y, with no f′ to shrink it.

## Try it

- Change `Tanh` to `Relu` and run again. Neurons whose z is negative for all four inputs have f′(z) = 0 and receive no gradient at all; the health checks report them as dead.
- Change the learning rate in `Sgd::new(0.5)` and watch how fast the loss falls, or whether it falls.
- Turn a hidden neuron off with `NeuronMask` and `Sequential::set_mask` (the network must then be declared `let mut net`). Its output becomes exactly 0, its f′ is reported as 0, and no gradient passes through it.
