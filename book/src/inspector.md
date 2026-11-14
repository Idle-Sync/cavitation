# Watching a network train

The inspector shows a training run in your browser while it happens: the speed, the loss and the held-out accuracy, every layer's values for one input, one neuron opened up with the chain rule written out in its own numbers, and where each step's time goes. When training ends it keeps the trained network open, so you can switch neurons off, rank them by what each costs, and shrink the network from the page. This chapter runs it on the digits network, reads each panel, and measures what watching costs.

The program is `crates/cavitation/examples/inspect_digits.rs`. Run it with:

```sh
cargo run --release --example inspect_digits -p cavitation --features inspect
```

and open the address it prints. Training starts when the page connects, or after 60 seconds without one. The digits network trains quickly, so add `-- --epochs 300` if you want to watch it longer.

The printed blocks in this chapter come from that command, run on an Intel Celeron N5105 (4 cores) at kernel level `sse4.2`, rustc 1.99.1. With fixed seeds and no commands from the page, the inspector does not change training: the parameters are those of plain `fit`, bit for bit, so these figures are the same on every run there.

## Turning it on

The inspector is in its own crate, `cavitation-inspect`, which `cavitation` re-exports as `cavitation::cavitation_inspect` behind the `inspect` feature. The feature is off by default, so a program that does not ask for it does not compile the inspector at all:

```toml
[dependencies]
cavitation = { git = "https://github.com/Idle-Sync/cavitation", features = ["inspect"] }
```

It needs nothing outside the standard library: the server is a small HTTP and Server-Sent Events server on `std::net`, and the page is one HTML file built into the program, which loads nothing from the internet.

## Starting the inspector

`Inspector::start` listens on `127.0.0.1:7878`, which only this machine can reach, and `url` gives the address to open:

```rust,ignore
{{#include ../../crates/cavitation/examples/inspect_digits.rs:start}}
```

The configuration names what the page should show:

- `probe`: the rows the page can run through the network one at a time. Here they are the 360 test digits. Without it the page uses the first training rows, and says so.
- `input_image`: `Some((8, 8))` draws each input, and each first-layer neuron's incoming weights, as an 8×8 picture.
- `class_names` and `title` label the outputs and the page.
- `eval_rows` (1,000 by default): how many probe rows the evaluation measures at the end of an epoch. Here all 360 test digits. The evaluation is held out only when `probe` is given; without it the rows are training rows, and the page labels the accuracy and loss as training figures (the `model` event's `probe_is_training`).
- `snapshot_hz` (4 by default): how many times a second the page is sent new values.
- `address`: another port, or `127.0.0.1:0` to let the system pick a free one, which `url` then reports.

Training under the inspector is `cavitation_inspect::fit` in place of `fit`, with the same `FitConfig`:

```rust,ignore
{{#include ../../crates/cavitation/examples/inspect_digits.rs:fit}}
```

Between steps it carries out what the page asks for and, while a page is connected, sends it a snapshot of the run. It takes the network as `&mut` because the page can switch neurons off; any switches are still on the network when `fit` returns.

## What the page shows

The top band is the run's speed and progress:

- **samples/s**: training rows per second, counted over the training steps only, so the inspector's own work between steps and any pause are left out.
- **GFLOPS**: that rate times the floating-point operations of one training row in the `Linear` layers.
- **accuracy** and **loss**: the last evaluation on the probe rows (held out here, since `probe` is given), and the last batch's loss beside the evaluation's loss. When the run has a validation set (`FitConfig::validation`), a second line under each shows its accuracy (or R²) and loss from the last epoch that measured it.
- **epoch** and **steps**, with the learning rate.

The evaluation runs at the end of an epoch, only while a page is connected, and only when the time since the last one is at least 200 times what that one took (`EVALUATION_FACTOR`), so it takes at most about 0.5% of the run. The last epoch always gets one, and when training ends the page gets one last snapshot, so it shows the final step and epoch. Here the evaluation rows are the test digits, which is fine for watching; to choose anything by them (when to stop, how far to shrink) you would hold out other rows instead, as the chapter on shrinking explains.

The **controls** pause and resume the run and set the learning rate. A new rate replaces the schedule for the rest of the run. A paused run resumes by itself after 30 seconds with no page connected, so a closed tab never leaves your program waiting.

**The network** panel runs one probe row through the network, forward and backward, and draws every layer as a grid of cells, one per neuron. A switch picks which value the cells show:

- **a**, the neuron's output;
- **z**, its value before the activation;
- **∂L/∂z**, the loss's gradient at z;
- **f′(z)**, the activation's slope at z.

The arrows step through the probe rows, and "Next mistake" jumps to the next row the last evaluation got wrong. The output layer is drawn as bars with the predicted share of each class.

The probe runs on its own graph and gradient buffers, beside training, so it never changes the parameters, the gradients or the optimiser's state.

Click a cell and **the neuron** panel opens it up:

- its five values;
- the outgoing weight norm (how strongly the next layer listens to it);
- the activation curve with the tangent at z;
- its incoming weights as an image;
- the inputs that add the most to its z for this row, with the bias and the sum.

## The chain rule on one neuron

The neuron panel writes out one step of backpropagation: `∂L/∂z = ∂L/∂a × f′(z)`. The gradient that reaches a neuron's output from the layer above, times the slope of its activation, is the gradient at its input. The example prints the same step after training, for one neuron of the first hidden layer (layer 1, the ReLU) on the first test digit the network gets wrong:

```rust,ignore
{{#include ../../crates/cavitation/examples/inspect_digits.rs:probe}}
```

It picks, among the neurons that fire on that digit, the one with the steepest gradient, and prints:

```text
trained 20 epochs: last epoch's loss 0.0026, test accuracy 97.78%
test digit 8: label 8, predicted 5, loss 1.291
layer 1 (ReLU), neuron 50: z = 0.09567, a = 0.09567, f'(z) = 1.000
dL/da = sum over the 128 neurons of layer 2 of w[j][50] * dL/dy[j] = 1.489 (the probe's dL/da: 1.489); largest terms neuron 108: 0.09916, neuron 115: 0.09619, neuron 53: 0.07078
dL/dz = dL/da * f'(z) = 1.489 * 1.000 = 1.489 (the probe's dL/dz: 1.489)
its strongest input is pixel (1, 5): x = 1.000, w = -0.3328; dL/dw = dL/dz * x = 1.489 * 1.000 = 1.489
on the page: probe row 8, then click neuron 50 of layer 1
```

Read it from the top:

- **The forward pass.** The digit is an 8 the network calls a 5. Neuron 50 adds up its weighted inputs to z = 0.09567. The ReLU passes a positive z through unchanged, so a = z, and its slope there is 1.
- **The gradient from above.** ∂L/∂a, how much the loss changes as the neuron's output changes, is the sum over the 128 neurons of the next `Linear` that read this output. Each term is the weight from neuron 50 to neuron j times the gradient at neuron j. The example adds those 128 terms itself and gets the probe's 1.489. No single term is large (the largest is 0.099); many small ones in the same direction add up.
- **One step down.** ∂L/∂z = ∂L/∂a × f′(z) = 1.489 × 1. For a ReLU that fires, the gradient passes straight through. For one that does not fire on this row, f′(z) is 0 and nothing passes. A neuron for which that is true on every row never learns again: that is a dead neuron.
- **To a weight.** The same rule goes on to the weights. z = Σ w·x + b, so ∂L/∂w for each input is ∂L/∂z × x. Pixel (1, 5) is fully lit (x = 1), so that weight's gradient is 1.489 too. Gradient descent lowers each weight by the learning rate times its gradient, which here lowers z, and with it this neuron's push towards the wrong answer.

On the page, step the probe to row 8 and click neuron 50 of layer 1 to see the same numbers.

## Where the time goes

The profiler times one sampled step, layer by layer: the forward pass, the loss, the backward pass, and the optimiser's update. The other steps are not timed at all, so profiling costs nothing between samples.

## Switching neurons and the shrink lab

When `fit` returns, `explore` keeps the trained network open on the page until you press Finish:

```rust,ignore
{{#include ../../crates/cavitation/examples/inspect_digits.rs:explore}}
```

In explore mode:

- **A switch** recomputes the probe at once. A switched-off neuron outputs exactly 0 and sends no gradient back; its cell is crossed out.
- **Rank neurons** switches each neuron of the chosen hidden layer off in turn and measures what its absence costs on the probe rows. The neurons are listed cheapest first, each as soon as it is measured.
- **Shrink** runs the shrink tool, `cavitation_nn::shrink::shrink`, with the configuration the program gives (here the defaults, with the last 288 training rows as the validation set). The curve of parameters against validation accuracy is drawn level by level. **Cancel** stops after the level in progress and keeps the best level so far.

The chapter on shrinking explains the ranking, the levels and the stop rule. Masks you set in explore mode stay on the network when `explore` returns. `explore` also returns after 30 seconds with no page connected.

## Saving a view

A view can be saved as one HTML file with the run so far inside it: the model, the latest snapshot, the recent epochs, and any ranking and shrink curve. It opens in a browser with no server, with the controls off. The example saves one when explore mode ends:

```rust,ignore
{{#include ../../crates/cavitation/examples/inspect_digits.rs:export}}
```

From a running page, `/export?token=…` (the page's address, with its path changed from / to /export) downloads the same file and writes nothing on the machine that trains. The file contains the network's weights for the probed neuron and the probe row's values, so share it as you would share the data.

## The token

The address the inspector prints ends in `?token=` and 32 hexadecimal digits, a 128-bit number made for this run. The event stream, every command and the export require it, so another user of the machine, or a program that only knows the port, can neither watch the run nor change it. Treat the address as a password while the run lasts.

The server also refuses:

- a request whose `Host` is not its own address, so a web page cannot reach it through a DNS name that points at your machine;
- a request from a page on another site (a foreign `Origin`).

It never allows another site to read its answers, and a `GET` never changes anything. The page and the token travel over plain HTTP, which is why the inspector listens only on 127.0.0.1 unless you name another address.

## What it costs

**Off, it costs nothing.** Without the `inspect` feature the inspector is not compiled. Plain `fit` and `Trainer` contain no inspector code at all. The only addition to the training step is one branch for the optional step profile, which plain `fit` never asks for.

**On, it costs little.** Between steps, the inspector:

- reads the client count from an atomic;
- checks for commands with a lock it never waits for;
- reads the clock once, only while a page is connected, to decide whether a snapshot is due.

With a page connected it also captures a snapshot at most `snapshot_hz` times a second: the probe row forward and backward, the neuron, one profiled step. It publishes the snapshot with a lock it never waits for: if a page's connection holds the lock, the snapshot is taken again at the next step. After the first steps, none of this allocates memory on the training thread, sampled steps and neuron switches included: each snapshot's buffer is made large enough, the first time, for any values of the same shape. The `no_alloc` test of `cavitation-inspect` checks that. Only the epoch-end events copy their text, once for each connected page, at the epoch's end, or at the next step if a page's connection held the lock then.

Most of the inspector's time on the training thread goes to the held-out evaluation at epoch ends: at most about 0.5% of the run by its budget, plus the one at the last epoch. The rest is small and bounded: a few snapshots a second, each one probe row forward and backward plus the page's JSON, and the `epoch` event at each epoch's end. Fewer `eval_rows` makes the evaluation cheaper. With no page connected, nothing is evaluated or captured.

Each "on" mode is meant to stay within 2% of training without the inspector. To measure it on your machine, run:

```sh
cargo build --release -p cavitation-bench
target/release/cavitation-bench inspect
```

It trains the digits benchmark (the 50k network, SGD at 0.01, batch 128, 100 epochs per run), with the 360 test digits as the probe set and the default `eval_rows` and `snapshot_hz`. First it trains for at least 30 seconds, past the power limit after which a low-power CPU slows down. Then it runs 8 rounds of four modes, in an order rotated every round, so each mode runs first, second, third and last equally often:

- off (plain `fit`);
- off again, to show how much two identical runs differ;
- on, with one client reading the event stream;
- on, with one client connected and not reading.

It prints every round's times and each mode's ratio to off (off's time over the mode's: 1 means no cost, 0.98 means 2% slower), and the median of each. It also prints the spread of the two identical off modes: the median, over the rounds, of how far their ratio is from 1. That is the noise every round's ratio carries. A spread above 1% means the machine was too busy for the medians to resolve 2%: close other programs and run it again.

## Try it

- Pause a run, switch off the neuron the chain rule above was about, and resume: watch the accuracy and the other neurons of the layer as the network learns to do without it.
- Set the learning rate to 0.5 mid-run and watch the loss and the f′(z) view: ReLU neurons whose z goes negative for every row stop learning, and their cells go dark.
- In explore mode, rank the first hidden layer and switch off the ten cheapest neurons by hand, then press Shrink and compare.
