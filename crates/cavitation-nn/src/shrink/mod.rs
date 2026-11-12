//! Shrinking a trained network. [`activity()`] measures how each neuron of chosen `Linear`
//! layers behaves over a data set, on the inference path; its statistics show which neurons
//! the network might do without.

mod activity;

pub use activity::{LayerActivity, activity};
