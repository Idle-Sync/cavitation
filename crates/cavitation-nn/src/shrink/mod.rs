//! Shrinking a trained network: finding the neurons it can do without.
//!
//! [`activity()`] measures how each neuron of chosen `Linear` layers behaves over a data
//! set: its mean, mean absolute value, firing share and maximum, read where the neuron's
//! output is final. Its statistics show which neurons the network might do without; switch
//! them off by `Linear` with [`Sequential::set_neuron_mask`](crate::Sequential::set_neuron_mask),
//! and measure the result with an [`Evaluator`](crate::Evaluator).
//!
//! The items here are reached through this module, `cavitation_nn::shrink`, and are not in
//! the crate root or the `cavitation` prelude.

mod activity;

pub use activity::{LayerActivity, activity};
