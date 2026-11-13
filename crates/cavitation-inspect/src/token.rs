//! `new_token`: the per-run secret that the inspector's URL carries.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Counts the tokens made in this process, so that no two inputs to the hashers repeat.
static MADE: AtomicU64 = AtomicU64::new(0);

/// A new 128-bit secret as 32 lowercase hexadecimal digits.
///
/// Each half is the output of a hasher from its own `std::collections::hash_map::RandomState`.
/// The standard library seeds those hashers' 128-bit SipHash keys from the operating
/// system's random number source, once per thread, and gives each new `RandomState`
/// different keys; the hashed input adds the clock and a process-wide counter. This keeps
/// the crate free of dependencies while giving every run a token another local user or
/// another site cannot guess.
///
/// # Examples
///
/// ```
/// let token = cavitation_inspect::token::new_token();
/// assert_eq!(token.len(), 32);
/// assert!(token.bytes().all(|b| b.is_ascii_hexdigit()));
/// assert_ne!(token, cavitation_inspect::token::new_token());
/// ```
pub fn new_token() -> String {
    let count = MADE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let half = |salt: u8| {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u8(salt);
        hasher.write_u64(count);
        hasher.write_u128(nanos);
        hasher.finish()
    };
    format!("{:016x}{:016x}", half(0), half(1))
}
