//! `new_token`: the per-run secret that the inspector's URL carries.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Counts the tokens made in this process, so that no two inputs to the hashers repeat.
static MADE: AtomicU64 = AtomicU64::new(0);

/// A new 128-bit secret as 32 lowercase hexadecimal digits.
///
/// Both halves come from SipHash keyed through `std::collections::hash_map::RandomState`.
/// The standard library draws one 128-bit seed per thread from the operating system's
/// random number source and gives each new `RandomState` that seed with one of its two key
/// words increased by one, so the two hashers have related keys from the same seed, not
/// independent ones. The token therefore carries the seed's 128 bits of entropy, the same
/// seed the thread's `HashMap`s use (their outputs are never exposed). The hashed input
/// adds a salt per half, a process-wide counter and the clock, so no two calls hash the
/// same input. This keeps the crate free of dependencies while giving every run a token
/// another local user or another site cannot guess.
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
