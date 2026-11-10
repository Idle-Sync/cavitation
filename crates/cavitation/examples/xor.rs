//! Trains a small network on XOR with a hand-written training loop, then uses hooks to print
//! what every layer saw and computed.
//!
//! Run it with `cargo run --release --example xor -p cavitation`.

use cavitation::prelude::*;
use cavitation::{Error, Reduction, Trace};

/// Prints the values of a tensor view as one short line, to three decimals.
fn row(values: &[f32]) -> String {
    let items: Vec<String> = values.iter().map(|v| format!("{v:7.3}")).collect();
    format!("[{}]", items.join(", "))
}

/// The last of the four rows of a batch-of-4 tensor's data, the one for the input (1, 1).
fn tail(data: &[f32]) -> &[f32] {
    &data[data.len() - data.len() / 4..]
}

/// Like `row`, in scientific notation, for gradients that are too small for fixed decimals.
fn sci(values: &[f32]) -> String {
    let items: Vec<String> = values.iter().map(|v| format!("{v:9.2e}")).collect();
    format!("[{}]", items.join(", "))
}

/// Trains 2 -> 8 tanh -> 1 sigmoid on XOR, then reports every layer's z, a, f'(z) and gradients.
fn main() -> Result<(), Error> {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(1);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 2, 8, &mut rng))
        .push(Tanh)
        .push(Linear::new(&mut params, 8, 1, &mut rng))
        .push(Sigmoid);
    let x = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2])?;
    let y = Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 0.0], &[4, 1])?;

    let names = net.parameter_names();
    let mut optimizer = Sgd::new(0.5);
    let mut monitor = HealthMonitor::new(Health::default());
    let mut graph = Graph::new();
    let mut trace = Trace::new();

    // Full-batch training: one step per epoch. Each step records the forward pass on a tape,
    // computes the loss, runs backward into `params.grads`, checks health while the tape is
    // alive, then drops the tape so the optimiser can update the parameters.
    for epoch in 1..=2000 {
        let step = epoch as u64;
        let mut tape = graph.tape(&params.values);
        let input = tape.input(x.view());
        let target = tape.input(y.view());
        let out = net.forward_traced(&mut tape, input, &mut trace);
        let loss = tape.mse(out, target, Reduction::Mean);
        let value = tape.value(loss).data()[0];
        tape.backward(loss, &mut params.grads);
        monitor.after_backward(step, value, &tape, &trace, &net)?;
        drop(tape);
        optimizer.step(&mut params);
        monitor.after_update(step, optimizer.last_stats(), &names)?;
        params.zero_grad();
        monitor.end_epoch(epoch)?;
        if epoch % 400 == 0 || epoch == 1 {
            println!("epoch {epoch:4}: loss {value:.6}");
        }
    }

    // One more pass, with a hook on every layer. Forward hooks see the layer's input (z for
    // an activation), its output (a) and the local derivative f'(z); backward hooks also see
    // the gradients of the loss with respect to the layer's output and input.
    let mut hooks = Hooks::new();
    for layer in 0..net.len() {
        hooks.on_forward(layer, move |e| {
            println!("forward  layer {layer} ({})", e.name);
            println!("  input  (z)    : {}", row(tail(e.input.data())));
            println!("  output (a)    : {}", row(tail(e.output.data())));
            if let Some(d) = &e.local_derivative {
                println!("  f'(z)         : {}", row(tail(d.data())));
            }
        });
        hooks.on_backward(layer, move |e| {
            println!("backward layer {layer} ({})", e.name);
            if let Some(g) = &e.grad_output {
                println!("  dL/d(output)  : {}", sci(tail(g.data())));
            }
            if let Some(g) = &e.grad_input {
                println!("  dL/d(input)   : {}", sci(tail(g.data())));
            }
        });
    }
    let mut tape = graph.tape(&params.values);
    let input = tape.input(x.view());
    let target = tape.input(y.view());
    let out = net.forward_traced(&mut tape, input, &mut trace);
    let loss = tape.mse(out, target, Reduction::Mean);
    tape.backward(loss, &mut params.grads);
    println!("\nHook output for the input (1, 1), after training:");
    hooks.run_forward(&tape, &trace, &net);
    hooks.run_backward(&tape, &trace, &net);

    println!("\nPredictions (targets 0, 1, 1, 0):");
    println!("  {}", row(tape.value(out).data()));
    Ok(())
}
