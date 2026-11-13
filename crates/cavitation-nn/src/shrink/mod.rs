//! Shrinking a trained network: finding the neurons it can do without.
//!
//! [`activity()`] measures how each neuron of chosen `Linear` layers behaves over a data
//! set: its mean, mean absolute value, firing share and maximum, read where the neuron's
//! output is final. Its statistics show which neurons the network might do without; switch
//! them off by `Linear` with [`Sequential::set_neuron_mask`](crate::Sequential::set_neuron_mask),
//! and measure the result with an [`Evaluator`](crate::Evaluator). [`compact()`] then
//! removes the off neurons for good: it rebuilds the network as a smaller dense one that
//! gives the masked network's outputs bit for bit through
//! [`InferenceModel::infer`](crate::InferenceModel::infer).
//!
//! The items here are reached through this module, `cavitation_nn::shrink`, and are not in
//! the crate root or the `cavitation` prelude.

mod activity;
mod compact;

pub use activity::{LayerActivity, activity};
pub use compact::compact;
