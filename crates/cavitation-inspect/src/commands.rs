//! `Command` and `CommandQueue`: what the page asks the training run to do, waiting to be
//! read between steps.

use std::collections::VecDeque;
use std::sync::{Mutex, PoisonError, TryLockError};

/// How many commands can wait before the server answers 503.
///
/// ```
/// use cavitation_inspect::commands::{Command, CommandQueue, QUEUE_CAPACITY};
/// let queue = CommandQueue::new(QUEUE_CAPACITY);
/// for _ in 0..QUEUE_CAPACITY {
///     assert!(queue.push(Command::Pause));
/// }
/// assert!(!queue.push(Command::Pause));
/// ```
pub const QUEUE_CAPACITY: usize = 64;

/// The largest learning rate the page may set. A rate from the page above it is refused
/// with a `status` message: a rate far beyond any useful one ends a run at its next step
/// with a non-finite loss.
///
/// ```
/// assert_eq!(cavitation_inspect::commands::MAX_LEARNING_RATE, 10.0);
/// ```
pub const MAX_LEARNING_RATE: f64 = 10.0;

/// A request from the page, sent as `POST /api/...` and read by the training side between
/// steps.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::commands::Command;
/// let c = Command::SetMask { linear: 0, neuron: 5, on: false };
/// assert_ne!(c, Command::Pause);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Switch neuron `neuron` of the `linear`-th `Linear` layer on or off
    /// (`POST /api/mask?linear=&neuron=&on=`, `on` is `0` or `1`). Training refuses the
    /// output `Linear` (one no later `Linear` reads), whose neurons are the network's answer.
    SetMask {
        /// Which `Linear` layer, counting only `Linear`s, from 0.
        linear: usize,
        /// Which of its output neurons.
        neuron: usize,
        /// `true` to switch it on, `false` to switch it off.
        on: bool,
    },
    /// Use this learning rate from now on (`POST /api/lr?value=`; finite and above 0, and
    /// at most [`MAX_LEARNING_RATE`] for training to apply it).
    SetLearningRate(f64),
    /// Pause training after the current step (`POST /api/pause`).
    Pause,
    /// Resume a paused run (`POST /api/resume`).
    Resume,
    /// Show the probe row with this index (`POST /api/probe?index=`).
    SelectProbe(usize),
    /// Show this neuron in the neuron panel (`POST /api/neuron?layer=&neuron=`).
    SelectNeuron {
        /// The layer's index in the model.
        layer: usize,
        /// The neuron's index in that layer's output.
        neuron: usize,
    },
    /// Ablate every neuron of the `linear`-th `Linear` layer in turn (`POST /api/ablate?linear=`).
    Ablate {
        /// Which `Linear` layer, counting only `Linear`s, from 0.
        linear: usize,
    },
    /// Run the shrink tool (`POST /api/shrink`).
    Shrink,
    /// Cancel a running ablation or shrink (`POST /api/cancel`).
    Cancel,
    /// Stop serving a trained network and return (`POST /api/finish`).
    Finish,
}

/// A bounded first-in, first-out queue of [`Command`]s.
///
/// The server pushes with [`push`](Self::push), which refuses a command when the queue is
/// full; the training side reads with [`try_pop`](Self::try_pop), which never waits for the
/// lock.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::commands::{Command, CommandQueue};
/// let queue = CommandQueue::new(1);
/// assert!(queue.push(Command::Pause));
/// assert!(!queue.push(Command::Resume));
/// assert_eq!(queue.try_pop(), Some(Command::Pause));
/// assert_eq!(queue.try_pop(), None);
/// ```
#[derive(Debug)]
pub struct CommandQueue {
    /// The waiting commands, oldest first.
    commands: Mutex<VecDeque<Command>>,
    /// The most commands that can wait.
    capacity: usize,
}

impl CommandQueue {
    /// An empty queue that holds up to `capacity` commands.
    ///
    /// ```
    /// let queue = cavitation_inspect::commands::CommandQueue::new(64);
    /// assert!(queue.is_empty());
    /// ```
    pub fn new(capacity: usize) -> CommandQueue {
        CommandQueue {
            commands: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    /// Adds a command at the back; `false` (and the command dropped) when the queue is full.
    ///
    /// ```
    /// let queue = cavitation_inspect::commands::CommandQueue::new(2);
    /// assert!(queue.push(cavitation_inspect::commands::Command::Shrink));
    /// assert_eq!(queue.len(), 1);
    /// ```
    pub fn push(&self, command: Command) -> bool {
        let mut commands = self.commands.lock().unwrap_or_else(PoisonError::into_inner);
        if commands.len() >= self.capacity {
            return false;
        }
        commands.push_back(command);
        true
    }

    /// Takes the oldest command, without waiting: `None` when the queue is empty or another
    /// thread holds its lock at this moment (the caller tries again at its next step).
    ///
    /// ```
    /// use cavitation_inspect::commands::{Command, CommandQueue};
    /// let queue = CommandQueue::new(4);
    /// queue.push(Command::Cancel);
    /// queue.push(Command::Finish);
    /// assert_eq!(queue.try_pop(), Some(Command::Cancel));
    /// ```
    pub fn try_pop(&self) -> Option<Command> {
        let mut commands = match self.commands.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        commands.pop_front()
    }

    /// How many commands are waiting.
    ///
    /// ```
    /// let queue = cavitation_inspect::commands::CommandQueue::new(4);
    /// assert_eq!(queue.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.commands
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether no command is waiting.
    ///
    /// ```
    /// let queue = cavitation_inspect::commands::CommandQueue::new(4);
    /// assert!(queue.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
