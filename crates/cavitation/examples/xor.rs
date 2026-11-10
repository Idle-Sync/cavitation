//! Trains a small network on XOR with a hand-written training loop, then uses hooks to print
//! what every layer saw and computed.
//!
//! The guide's chapter "Backprop by hand" walks through this file; the `ANCHOR` comments
//! mark the parts it includes.
//!
//! Run it with `cargo run --release --example xor -p cavitation`.

use std::cell::RefCell;
use std::rc::Rc;

use cavitation::prelude::*;

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
    // ANCHOR: network
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(1);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 2, 8, &mut rng))
        .push(Tanh)
        .push(Linear::new(&mut params, 8, 1, &mut rng))
        .push(Sigmoid);
    let x = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2])?;
    let y = Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 0.0], &[4, 1])?;
    // ANCHOR_END: network

    // ANCHOR: loop
    let names = net.parameter_names();
    let mut optimizer = Sgd::new(0.5);
    // Health checks stay on. Warnings (for example a gradient norm outside the bounds) are
    // collected here and summarised at the end instead of printed in the middle of the run.
    let warnings: Rc<RefCell<Vec<String>>> = Rc::default();
    let sink = Rc::clone(&warnings);
    let mut monitor = HealthMonitor::new(Health {
        sink: Some(Box::new(move |w: &Warning| {
            sink.borrow_mut().push(w.to_string())
        })),
        ..Health::default()
    });
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
    // ANCHOR_END: loop

    // One more pass, with a hook on every layer. Forward hooks see the layer's input (z for
    // an activation), its output (a) and the local derivative f'(z); backward hooks also see
    // the gradients of the loss with respect to the layer's output and input.
    // ANCHOR: hooks
    let mut hooks = Hooks::new();
    for layer in 0..net.len() {
        hooks.on_forward(layer, move |e| {
            println!("forward  layer {layer} ({})", e.name);
            // z, a and f'(z) are the names for an activation layer; the layers that
            // carry no element-wise derivative are the Linear layers.
            let (z, a) = if e.local_derivative.is_some() {
                ("input  (z)", "output (a)")
            } else {
                ("input     ", "output    ")
            };
            println!("  {z}    : {}", row(tail(e.input.data())));
            println!("  {a}    : {}", row(tail(e.output.data())));
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
    // ANCHOR_END: hooks
    // ANCHOR: run
    let mut tape = graph.tape(&params.values);
    let input = tape.input(x.view());
    let target = tape.input(y.view());
    let out = net.forward_traced(&mut tape, input, &mut trace);
    let loss = tape.mse(out, target, Reduction::Mean);
    tape.backward(loss, &mut params.grads);
    println!("\nHook output for the input (1, 1), after training:");
    hooks.run_forward(&tape, &trace, &net);
    hooks.run_backward(&tape, &trace, &net);
    // ANCHOR_END: run

    // The gradients of the parameters, summed over the four inputs, as `backward` left them
    // in `params.grads`. The optimiser would subtract 0.5 times each of these.
    // ANCHOR: grads
    println!("\nParameter gradients over the whole batch:");
    for (name, id) in &names {
        println!("  {name:8}: {}", sci(params.grad(*id).data()));
    }
    // ANCHOR_END: grads

    println!("\nPredictions (targets 0, 1, 1, 0):");
    println!("  {}", row(tape.value(out).data()));
    let warnings = warnings.borrow();
    match warnings.last() {
        Some(last) => println!("health: {} warnings; last: {last}", warnings.len()),
        None => println!("health: no warnings"),
    }
    Ok(())
}
