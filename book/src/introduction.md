# Introduction

What this guide covers, what it assumes and how to run its code.

Cavitation is a neural network library in Rust that keeps nothing hidden. A training step is a tape of recorded operations, and every value on it can be read: each layer's input and output, the local derivative of every activation, and the gradient of the loss with respect to every value and every parameter.

This guide uses that to explain how networks learn. Each chapter takes a real program from the repository, works out by hand what it computes, and then reads the same numbers out of the running program so you can check the hand calculation against it.

## What you need

- Rust 1.99 or newer, and a clone of the repository: <https://github.com/Idle-Sync/cavitation>.
- The chain rule from calculus, and matrix multiplication. Nothing else is assumed.

## Running the code

Every program in the guide is an example in `crates/cavitation/examples/`. The guide includes its code straight from those files, so what you read is what runs. From the repository root:

```sh
cargo run --release --example xor -p cavitation
```

## Chapters

1. [Backprop by hand](backprop-by-hand.md): a small network learns XOR. The forward pass, the local derivatives, the gradients of backpropagation, and all of them read through hooks.
