//! Shrinking a trained network: finding the neurons it can do without.
//!
//! [`activity()`] measures how each neuron of chosen `Linear` layers behaves over a data
//! set: its mean, mean absolute value, firing share and maximum, read where the neuron's
//! output is final. Its statistics show which neurons the network might do without; switch
//! them off by `Linear` with [`Sequential::set_neuron_mask`](crate::Sequential::set_neuron_mask),
//! and measure the result with an [`Evaluator`](crate::Evaluator); [`ablate()`] does that
//! for every neuron in turn and ranks them by what each one's absence costs. [`compact()`] then
//! removes the off neurons for good: it rebuilds the network as a smaller dense one that
//! gives the masked network's outputs through
//! [`InferenceModel::infer`](crate::InferenceModel::infer), bit for bit on the `reference`
//! and `sse4.2` kernel levels and within the rounding bound [`compact()`] states on the
//! others. [`shrink()`] does the whole job: it removes neurons a step at a time, retrains
//! after each step and stops by a rule measured on a validation set, to find the smallest
//! network that is still as good as the one you started from, optionally folding each
//! removed neuron's mean output into the next layer's bias
//! ([`ShrinkConfig::fold_means`]). [`train_found_size()`] trains a network of the size it
//! found from new starting values and measures both, to show whether pruning bought more
//! than the size.
//!
//! The items here are reached through this module, `cavitation_nn::shrink`, and are not in
//! the crate root or the `cavitation` prelude.

mod ablate;
mod activity;
mod compact;
mod prune;
mod scratch;

pub use ablate::{Ablation, ablate, ablate_controlled, ablate_with};
pub use activity::{LayerActivity, activity};
pub use compact::compact;
pub use prune::{
    Criterion, RankContext, Ranking, RetrainConfig, ShrinkConfig, ShrinkReport, ShrinkStep,
    ShrinkWarning, Validation, shrink,
};
pub use scratch::{FoundSize, FromScratch, train_found_size};

use cavitation_core::Real;

use crate::Sequential;

/// Which neurons of the `Linear` at layer `linear` are on: a neuron is off when any mask in
/// its run (the `Linear` itself and the element-wise layers after it, up to
/// [`Sequential::neuron_layer`]) switches it off. One entry per output of the `Linear`; a
/// mask entry past that width is ignored. Every tool here reads the on neurons this way.
///
/// # Panics
///
/// Panics if layer `linear` is not a `Linear`.
pub(crate) fn neurons_on<T: Real>(model: &Sequential<T>, linear: usize) -> Vec<bool> {
    let width = model
        .linear(linear)
        .expect("neurons_on: the layer is a Linear")
        .outputs();
    let mut on = vec![true; width];
    for layer in linear..=model.neuron_layer(linear) {
        if let Some(mask) = model.mask(layer) {
            for neuron in mask.off().filter(|&n| n < width) {
                on[neuron] = false;
            }
        }
    }
    on
}
