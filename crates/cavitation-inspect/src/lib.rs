//! A live view of a cavitation training run in the browser.
//!
//! [`Inspector`] is a small HTTP and Server-Sent Events server built on `std::net` alone,
//! with no dependency outside the cavitation workspace. It listens on `127.0.0.1:7878` by
//! default, reachable only from this machine; [`Inspector::url`] gives the address to open,
//! which carries a token made for this run.
//!
//! The pieces, each usable on its own:
//!
//! - [`sse::Hub`]: where the training side publishes. Snapshots (the run's statistics, the
//!   probed row, the selected neuron, a step's profile) keep only their latest value and
//!   are published without ever waiting for a lock; discrete events (the model, epochs,
//!   status changes, ablation and shrink results) are queued for every client and never
//!   skipped.
//! - [`commands`]: what the page asks for (switch a neuron, change the learning rate,
//!   pause, resume, ...), queued for the training side to read between steps.
//! - [`http`]: the request parser and the checks every request passes.
//! - [`json::JsonWriter`]: JSON without NaN or infinity, safe inside an HTML `<script>`.
//! - [`token::new_token`]: the per-run 128-bit token.
//!
//! # Security
//!
//! The event stream carries the network's values and the commands change training, so the
//! server refuses what does not come from its own page: a request whose `Host` is not
//! `127.0.0.1:<port>` or `localhost:<port>` (DNS rebinding), a request whose `Origin` is
//! another site, and any request to `/events`, `/api/*` or `/export` without the token. No
//! response carries `Access-Control-Allow-Origin`, `GET` never changes anything, and every
//! connection has a time limit for its request and for each write.
//!
//! # Examples
//!
//! ```
//! use cavitation_inspect::{Inspector, InspectorConfig};
//! use cavitation_inspect::sse::Snapshot;
//!
//! let inspector = Inspector::start(InspectorConfig {
//!     address: "127.0.0.1:0".parse().unwrap(),
//!     ..InspectorConfig::default()
//! })?;
//! println!("open {}", inspector.url());
//! inspector.hub().publish_snapshot(Snapshot::Stats, |w| {
//!     w.begin_object().key("step").uint(1).key("loss").number(0.5).end_object();
//! });
//! while let Some(command) = inspector.try_command() {
//!     println!("the page asked for {command:?}");
//! }
//! # Ok::<(), cavitation_core::Error>(())
//! ```

pub mod commands;
pub mod http;
pub mod json;
mod server;
pub mod sse;
pub mod token;

pub use server::{Inspector, InspectorConfig};
