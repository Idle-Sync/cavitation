//! `Mulberry32`: the standard mulberry32 32-bit generator, reproduced exactly from its
//! common JavaScript form. It shuffles the scikit-learn digits deterministically.

/// The mulberry32 generator. Its sequence matches the JavaScript function
/// `rng(seed)` (which uses `|0` and `Math.imul`) value for value.
///
/// # Examples
///
/// ```
/// use cavitation_datasets::Mulberry32;
/// let mut rng = Mulberry32::new(42);
/// assert_eq!(format!("{:.10}", rng.next_f64()), "0.6011037519");
/// ```
#[derive(Clone, Debug)]
pub struct Mulberry32 {
    /// The running state, advanced by a fixed constant on every call.
    state: u32,
}

impl Mulberry32 {
    /// Creates a generator from a 32-bit seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_datasets::Mulberry32;
    /// let mut a = Mulberry32::new(1);
    /// let mut b = Mulberry32::new(1);
    /// assert_eq!(a.next_u32(), b.next_u32());
    /// ```
    pub fn new(seed: u32) -> Mulberry32 {
        Mulberry32 { state: seed }
    }

    /// Returns the next 32-bit output, before the division by 2^32.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_datasets::Mulberry32;
    /// assert_eq!(Mulberry32::new(42).next_u32(), 2581720956);
    /// ```
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6D2B_79F5);
        let seed = self.state;
        let mut t = (seed ^ (seed >> 15)).wrapping_mul(1 | seed);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        t ^ (t >> 14)
    }

    /// Returns the next value in `[0, 1)`: the 32-bit output divided by 2^32.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_datasets::Mulberry32;
    /// let x = Mulberry32::new(7).next_f64();
    /// assert!((0.0..1.0).contains(&x));
    /// ```
    pub fn next_f64(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4_294_967_296.0
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the mulberry32 generator against known outputs.

    use super::*;

    /// Seed 42 gives the values the standard JavaScript mulberry32 gives.
    #[test]
    fn mulberry32_matches_javascript() {
        let mut rng = Mulberry32::new(42);
        let raw: Vec<u32> = (0..3).map(|_| rng.next_u32()).collect();
        assert_eq!(raw, [2581720956, 1925393290, 3661312704]);
        let mut rng = Mulberry32::new(42);
        for want in ["0.6011037519", "0.4482905590", "0.8524657935"] {
            assert_eq!(format!("{:.10}", rng.next_f64()), want);
        }
    }
}
