//! `Hub`: what the training side publishes and each Server-Sent Events client's writer
//! loop.
//!
//! Two kinds of message reach the page:
//!
//! - **Snapshots** ([`Snapshot`]: `stats`, `probe`, `neuron`, `profile`) describe the run
//!   now. Only the latest of each matters, so each lives in one slot with a version number;
//!   a client that falls behind skips to the newest.
//! - **Events** ([`Event`]: `model`, `epoch`, `status`, `ablation`, `shrink`) each say
//!   something that happened once, so each client has a bounded queue of them and none is
//!   skipped.
//!
//! Everything is behind one `Mutex` with one `Condvar`. The training side publishes
//! snapshots with `try_lock` and never waits; each client's writer thread waits on the
//! condition variable, copies what is new under the lock and writes outside it.

use std::collections::VecDeque;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError, TryLockError};
use std::time::{Duration, Instant};

use crate::http;
use crate::json::JsonWriter;

/// How often an idle stream sends a comment line, so that proxies and browsers keep it open.
///
/// ```
/// assert_eq!(cavitation_inspect::sse::KEEPALIVE.as_secs(), 15);
/// ```
pub const KEEPALIVE: Duration = Duration::from_secs(15);

/// How often a waiting writer checks whether its client has closed the connection.
const CHECK: Duration = Duration::from_millis(100);

/// A kind of snapshot: only the latest of each is sent.
///
/// # Examples
///
/// ```
/// assert_eq!(cavitation_inspect::sse::Snapshot::Stats.name(), "stats");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Snapshot {
    /// Step, epoch, speed, loss, learning rate, paused.
    Stats,
    /// The probed row's values through every layer.
    Probe,
    /// The selected neuron's values and weights.
    Neuron,
    /// The time spent in each layer of a profiled step.
    Profile,
}

impl Snapshot {
    /// Every kind, in slot order.
    const ALL: [Snapshot; 4] = [
        Snapshot::Stats,
        Snapshot::Probe,
        Snapshot::Neuron,
        Snapshot::Profile,
    ];

    /// The event name the page listens for.
    ///
    /// ```
    /// assert_eq!(cavitation_inspect::sse::Snapshot::Profile.name(), "profile");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            Snapshot::Stats => "stats",
            Snapshot::Probe => "probe",
            Snapshot::Neuron => "neuron",
            Snapshot::Profile => "profile",
        }
    }

    /// The slot this kind is kept in.
    fn slot(self) -> usize {
        self as usize
    }
}

/// A kind of discrete event: every one is delivered, in order.
///
/// # Examples
///
/// ```
/// assert_eq!(cavitation_inspect::sse::Event::Ablation.name(), "ablation");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// The network's description; the latest one is also sent first to each new client.
    Model,
    /// An epoch ended.
    Epoch,
    /// The run's state changed (training, paused, explore, finished), with a message.
    Status,
    /// One neuron's ablation result.
    Ablation,
    /// One step of the shrink tool.
    Shrink,
}

impl Event {
    /// The event name the page listens for.
    ///
    /// ```
    /// assert_eq!(cavitation_inspect::sse::Event::Model.name(), "model");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            Event::Model => "model",
            Event::Epoch => "epoch",
            Event::Status => "status",
            Event::Ablation => "ablation",
            Event::Shrink => "shrink",
        }
    }
}

/// The latest payload of one kind of snapshot.
#[derive(Debug, Default)]
struct Slot {
    /// Raised by one for every publish; 0 means never published.
    version: u64,
    /// The payload, rewritten in place by each publish.
    json: JsonWriter,
}

/// One connected client's queue of events not yet written.
#[derive(Debug)]
struct Client {
    /// The client's identifier, given at registration.
    id: u64,
    /// Events waiting for the client's writer, oldest first.
    queue: VecDeque<(Event, String)>,
}

/// Everything behind the hub's lock.
#[derive(Debug)]
struct State {
    /// Set when the server shuts down; nothing is delivered after it.
    closed: bool,
    /// The most clients at once.
    max_clients: usize,
    /// The most events one client's queue holds.
    queue_capacity: usize,
    /// The longest a blocking event publish waits for room in a client's queue.
    write_timeout: Duration,
    /// One slot per [`Snapshot`] kind.
    slots: [Slot; 4],
    /// Where an event's payload is written before it is copied to each queue.
    scratch: JsonWriter,
    /// The latest `model` event's payload, queued first for every new client.
    model: Option<String>,
    /// The connected clients.
    clients: Vec<Client>,
    /// The identifier the next client gets.
    next_id: u64,
}

impl State {
    /// The position of client `id` in the list, if it is still connected.
    fn position(&self, id: u64) -> Option<usize> {
        self.clients.iter().position(|c| c.id == id)
    }
}

/// The meeting point of the training side and the clients' writer threads.
///
/// The training side calls [`publish_snapshot`](Self::publish_snapshot), which never waits,
/// and [`publish_event`](Self::publish_event); the server registers each `/events`
/// connection and runs its writer loop. A client whose queue overflows, or whose connection
/// accepts no bytes for the write timeout, is dropped.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::sse::{Event, Hub, Snapshot};
/// use std::time::Duration;
/// let hub = Hub::new(8, 256, Duration::from_secs(2));
/// assert!(hub.publish_snapshot(Snapshot::Stats, |w| {
///     w.begin_object().key("step").uint(10).end_object();
/// }));
/// assert!(hub.publish_event(Event::Status, |w| { w.string("training"); }, false));
/// assert_eq!(hub.clients(), 0);
/// ```
#[derive(Debug)]
pub struct Hub {
    /// The slots, the queues and the client list.
    state: Mutex<State>,
    /// Signalled whenever something is published, a queue is drained, a client leaves, or
    /// the hub closes.
    changed: Condvar,
    /// Mirrors `State::closed` for loops that poll without taking the lock.
    stopping: AtomicBool,
}

impl Hub {
    /// A hub for at most `max_clients` clients at once, each with a queue of
    /// `client_queue` events. A blocking event publish waits at most `write_timeout` for a
    /// client whose queue is full before dropping it; the server passes its own write
    /// timeout.
    ///
    /// ```
    /// let hub = cavitation_inspect::sse::Hub::new(8, 256, std::time::Duration::from_secs(2));
    /// assert_eq!(hub.clients(), 0);
    /// ```
    pub fn new(max_clients: usize, client_queue: usize, write_timeout: Duration) -> Hub {
        Hub {
            state: Mutex::new(State {
                closed: false,
                max_clients,
                queue_capacity: client_queue,
                write_timeout,
                slots: Default::default(),
                scratch: JsonWriter::new(),
                model: None,
                clients: Vec::new(),
                next_id: 0,
            }),
            changed: Condvar::new(),
            stopping: AtomicBool::new(false),
        }
    }

    /// Takes the lock, recovering it if a thread panicked while holding it (the state stays
    /// consistent: every change under the lock is a single assignment or push).
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Takes the lock only if it is free now.
    fn try_lock(&self) -> Option<MutexGuard<'_, State>> {
        match self.state.try_lock() {
            Ok(guard) => Some(guard),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// Replaces the latest snapshot of kind `which` with what `fill` writes, without
    /// waiting: returns `false`, and calls nothing, when another thread holds the lock at
    /// this moment. The training side skips that snapshot; a newer one follows.
    ///
    /// `fill` runs under the lock and gets the slot's own writer, emptied, so a payload of
    /// the same size allocates nothing.
    ///
    /// ```
    /// use cavitation_inspect::sse::{Hub, Snapshot};
    /// use std::time::Duration;
    /// let hub = Hub::new(8, 256, Duration::from_secs(2));
    /// assert!(hub.publish_snapshot(Snapshot::Probe, |w| { w.f32s(&[0.5, 1.0]); }));
    /// ```
    pub fn publish_snapshot(&self, which: Snapshot, fill: impl FnOnce(&mut JsonWriter)) -> bool {
        let Some(mut state) = self.try_lock() else {
            return false;
        };
        if state.closed {
            return false;
        }
        let slot = &mut state.slots[which.slot()];
        slot.json.clear();
        fill(&mut slot.json);
        slot.version += 1;
        drop(state);
        self.changed.notify_all();
        true
    }

    /// Appends the event that `fill` writes to every client's queue; returns whether it was
    /// published.
    ///
    /// With `blocking: false` the lock is taken with `try_lock`: `false` means another
    /// thread held it, nothing was published, and the caller tries again at its next step.
    /// A client whose queue is full is then dropped, so a client that stopped reading never
    /// holds up the training loop.
    ///
    /// With `blocking: true` the call waits for the lock, and while any client's queue is
    /// full it waits for that client's writer to drain it, so a client that keeps up misses
    /// no event. Explore mode publishes this way. The wait is bounded: a client whose queue
    /// is still full after the hub's write timeout (from the start of this call) is dropped,
    /// so a client that reads a little at a time, and so never trips its writer's own
    /// timeout, cannot hold the publisher longer than that.
    ///
    /// Either way the result is `false` once the hub is closed. A `model` event is also
    /// kept and sent first to every client that connects later.
    ///
    /// ```
    /// use cavitation_inspect::sse::{Event, Hub};
    /// use std::time::Duration;
    /// let hub = Hub::new(8, 256, Duration::from_secs(2));
    /// assert!(hub.publish_event(Event::Epoch, |w| {
    ///     w.begin_object().key("epoch").uint(1).end_object();
    /// }, true));
    /// ```
    pub fn publish_event(
        &self,
        which: Event,
        fill: impl FnOnce(&mut JsonWriter),
        blocking: bool,
    ) -> bool {
        let mut state = if blocking {
            let start = Instant::now();
            let mut state = self.lock();
            loop {
                let capacity = state.queue_capacity;
                if state.closed || state.clients.iter().all(|c| c.queue.len() < capacity) {
                    break state;
                }
                let left = state.write_timeout.saturating_sub(start.elapsed());
                if left.is_zero() {
                    // The clients still full are dropped below, before the event is queued.
                    break state;
                }
                state = self
                    .changed
                    .wait_timeout(state, left)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        } else {
            match self.try_lock() {
                Some(state) => state,
                None => return false,
            }
        };
        if state.closed {
            return false;
        }
        let capacity = state.queue_capacity;
        state.clients.retain(|c| c.queue.len() < capacity);
        let State {
            scratch,
            clients,
            model,
            ..
        } = &mut *state;
        scratch.clear();
        fill(scratch);
        for client in clients.iter_mut() {
            client
                .queue
                .push_back((which, scratch.as_str().to_string()));
        }
        if which == Event::Model {
            *model = Some(scratch.as_str().to_string());
        }
        drop(state);
        self.changed.notify_all();
        true
    }

    /// The number of connected clients.
    ///
    /// ```
    /// let hub = cavitation_inspect::sse::Hub::new(8, 256, std::time::Duration::from_secs(2));
    /// assert_eq!(hub.clients(), 0);
    /// ```
    pub fn clients(&self) -> usize {
        self.lock().clients.len()
    }

    /// Closes the hub: every writer loop ends, waiting publishers return `false`, and
    /// nothing is published again.
    ///
    /// ```
    /// use cavitation_inspect::sse::{Hub, Snapshot};
    /// use std::time::Duration;
    /// let hub = Hub::new(8, 256, Duration::from_secs(2));
    /// hub.close();
    /// assert!(!hub.publish_snapshot(Snapshot::Stats, |w| { w.null(); }));
    /// ```
    pub fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        self.stopping.store(true, Ordering::Release);
        drop(state);
        self.changed.notify_all();
    }

    /// Whether [`close`](Self::close) has been called; read without the lock.
    pub(crate) fn stopping(&self) -> &AtomicBool {
        &self.stopping
    }

    /// Adds a client; `None` when `max_clients` are already connected or the hub is
    /// closed. The new client's queue starts with the latest `model` event, if any, and it
    /// will be sent every snapshot published so far.
    pub(crate) fn register(&self) -> Option<u64> {
        let mut state = self.lock();
        if state.closed || state.clients.len() >= state.max_clients {
            return None;
        }
        let id = state.next_id;
        state.next_id += 1;
        let mut queue = VecDeque::new();
        if let Some(model) = &state.model {
            queue.push_back((Event::Model, model.clone()));
        }
        state.clients.push(Client { id, queue });
        Some(id)
    }

    /// Removes client `id`, if it is still connected, and wakes any publisher waiting for
    /// its queue.
    pub(crate) fn unregister(&self, id: u64) {
        let mut state = self.lock();
        if let Some(i) = state.position(id) {
            state.clients.swap_remove(i);
        }
        drop(state);
        self.changed.notify_all();
    }

    /// Runs client `id`'s writer loop on `stream` (non-blocking, the response head already
    /// sent) until the client leaves, a write makes no progress for `write_timeout`, the
    /// client is dropped by a publisher, or the hub closes; then unregisters the client.
    ///
    /// Each pass waits for a queued event or a newer snapshot, copies them out under the
    /// lock and writes them outside it, events first. An idle stream gets a comment line
    /// every [`KEEPALIVE`]; while waiting, the loop checks every [`CHECK`] whether the
    /// client has closed its connection, so a closed tab is noticed within a fraction of a
    /// second.
    pub(crate) fn serve(&self, id: u64, stream: &mut TcpStream, write_timeout: Duration) {
        let mut seen = [0u64; 4];
        let mut fresh = [false; 4];
        let mut payloads: [String; 4] = Default::default();
        let mut events: VecDeque<(Event, String)> = VecDeque::new();
        let mut out: Vec<u8> = Vec::new();
        let mut last_write = Instant::now();
        loop {
            let mut check_peer = false;
            {
                let mut state = self.lock();
                loop {
                    if state.closed {
                        return;
                    }
                    let Some(i) = state.position(id) else {
                        return;
                    };
                    let newer = Snapshot::ALL
                        .iter()
                        .any(|s| state.slots[s.slot()].version > seen[s.slot()]);
                    if !state.clients[i].queue.is_empty() || newer {
                        std::mem::swap(&mut state.clients[i].queue, &mut events);
                        for s in Snapshot::ALL {
                            let slot = &state.slots[s.slot()];
                            if slot.version > seen[s.slot()] {
                                seen[s.slot()] = slot.version;
                                let payload = &mut payloads[s.slot()];
                                payload.clear();
                                payload.push_str(slot.json.as_str());
                                fresh[s.slot()] = true;
                            }
                        }
                        break;
                    }
                    if last_write.elapsed() >= KEEPALIVE {
                        break;
                    }
                    let (guard, wait) = self
                        .changed
                        .wait_timeout(state, CHECK)
                        .unwrap_or_else(PoisonError::into_inner);
                    state = guard;
                    if wait.timed_out() {
                        check_peer = true;
                        break;
                    }
                }
            }
            // A queue was drained: a blocked publisher may go on.
            self.changed.notify_all();
            if check_peer {
                if peer_closed(stream) {
                    self.unregister(id);
                    return;
                }
                if last_write.elapsed() < KEEPALIVE {
                    continue;
                }
            }
            out.clear();
            for (event, data) in events.drain(..) {
                frame(&mut out, event.name(), &data);
            }
            for s in Snapshot::ALL {
                if std::mem::take(&mut fresh[s.slot()]) {
                    frame(&mut out, s.name(), &payloads[s.slot()]);
                }
            }
            if out.is_empty() {
                out.extend_from_slice(b": keep-alive\n\n");
            }
            if http::write_all(stream, &out, write_timeout, &self.stopping).is_err() {
                self.unregister(id);
                return;
            }
            last_write = Instant::now();
        }
    }
}

/// Appends one Server-Sent Events message. The payload is JSON from [`JsonWriter`], which
/// never contains a raw line break, so it fits on one `data:` line.
fn frame(out: &mut Vec<u8>, name: &str, data: &str) {
    out.extend_from_slice(b"event: ");
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(b"\ndata: ");
    out.extend_from_slice(data.as_bytes());
    out.extend_from_slice(b"\n\n");
}

/// Whether the client has closed a non-blocking connection (end of stream or an error);
/// bytes it sent are read and ignored.
fn peer_closed(stream: &mut TcpStream) -> bool {
    use std::io::Read;
    let mut buf = [0u8; 256];
    match stream.read(&mut buf) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => !matches!(
            e.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
        ),
    }
}
