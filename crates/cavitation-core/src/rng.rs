//! `SplitMix64` and `Rng` (xoshiro256++): the random number generators used for weight
//! initialisation, shuffling and sampling.

/// The SplitMix64 generator. It is used to turn one 64-bit seed into the four state words
/// of [`Rng`], and is a fine generator on its own for short sequences.
///
/// # Examples
///
/// ```
/// use cavitation_core::SplitMix64;
/// let mut sm = SplitMix64::new(0);
/// assert_eq!(sm.next_u64(), 0xE220A8397B1DCDAF);
/// ```
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    /// The running counter; it advances by a fixed odd constant on every call.
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator whose first output is determined by `seed`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::SplitMix64;
    /// let mut a = SplitMix64::new(7);
    /// let mut b = SplitMix64::new(7);
    /// assert_eq!(a.next_u64(), b.next_u64());
    /// ```
    pub fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    /// Returns the next 64-bit output.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::SplitMix64;
    /// let mut sm = SplitMix64::new(1);
    /// assert_ne!(sm.next_u64(), sm.next_u64());
    /// ```
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The xoshiro256++ generator with helpers for uniform, normal and bounded integer draws.
///
/// The same seed always gives the same sequence on every machine.
///
/// # Examples
///
/// ```
/// use cavitation_core::Rng;
/// let mut rng = Rng::seed(42);
/// let x = rng.uniform_f32();
/// assert!((0.0..1.0).contains(&x));
/// ```
#[derive(Clone, Debug)]
pub struct Rng {
    /// The four 64-bit state words; never all zero.
    s: [u64; 4],
    /// The second value of the last Box-Muller pair, waiting to be returned by `normal`.
    cached_normal: Option<f64>,
}

impl Rng {
    /// Creates a generator whose four state words are the first four outputs of
    /// `SplitMix64::new(seed)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Rng;
    /// let mut a = Rng::seed(42);
    /// let mut b = Rng::seed(42);
    /// assert_eq!(a.next_u64(), b.next_u64());
    /// ```
    pub fn seed(seed: u64) -> Rng {
        let mut sm = SplitMix64::new(seed);
        let s = [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()];
        Rng {
            s,
            cached_normal: None,
        }
    }

    /// Returns the next 64-bit output.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Rng;
    /// let mut rng = Rng::seed(42);
    /// assert_eq!(rng.next_u64(), 15021278609987233951);
    /// ```
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Returns a uniform `f32` in `[0, 1)` built from the top 24 bits of one output.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Rng;
    /// let mut rng = Rng::seed(42);
    /// assert_eq!(rng.uniform_f32(), 0.8143051266670227);
    /// ```
    pub fn uniform_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    /// Returns a uniform `f64` in `[0, 1)` built from the top 53 bits of one output.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Rng;
    /// let x = Rng::seed(1).uniform_f64();
    /// assert!((0.0..1.0).contains(&x));
    /// ```
    pub fn uniform_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Returns a uniform `f64` in `[lo, hi)` when `lo < hi`, and `lo` when `lo == hi`.
    ///
    /// Rounding can push `lo + (hi - lo) * u` up to `hi` for the largest draws; the result
    /// is then the largest `f64` below `hi` (but not below `lo`), so `hi` is never returned.
    /// With reversed bounds (`lo > hi`) the draw is `lo + (hi - lo) * u`, a value in
    /// `(hi, lo]`, with no clamp.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Rng;
    /// let x = Rng::seed(1).range(-2.0, 3.0);
    /// assert!((-2.0..3.0).contains(&x));
    /// ```
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = self.uniform_f64();
        Rng::range_from(lo, hi, u)
    }

    /// Maps the unit draw `u` in `[0, 1)` onto `[lo, hi)`, stepping back below `hi` when
    /// rounding reaches it and never going below `lo`. The step back applies only when
    /// `lo < hi`; reversed bounds return the plain mapping.
    fn range_from(lo: f64, hi: f64, u: f64) -> f64 {
        let x = lo + (hi - lo) * u;
        if lo < hi && x >= hi {
            hi.next_down().max(lo)
        } else {
            x
        }
    }

    /// Returns a standard normal draw (mean 0, variance 1) by the Box-Muller transform.
    /// Each transform makes two values; the second is kept for the next call.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Rng;
    /// let z = Rng::seed(1).normal();
    /// assert!(z.is_finite());
    /// ```
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.cached_normal.take() {
            return z;
        }
        // 1 - u is in (0, 1], so the logarithm is finite.
        let u1 = 1.0 - self.uniform_f64();
        let u2 = self.uniform_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.cached_normal = Some(r * angle.sin());
        r * angle.cos()
    }

    /// Returns a uniform integer in `0..n` without modulo bias (Lemire's method).
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Rng;
    /// let k = Rng::seed(1).below(7);
    /// assert!(k < 7);
    /// ```
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "Rng::below: n must be positive, got 0");
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        if (m as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
            }
        }
        (m >> 64) as u64
    }

    /// Shuffles `items` in place with Fisher-Yates, walking from the end.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Rng;
    /// let mut v = [1, 2, 3, 4, 5];
    /// Rng::seed(1).shuffle(&mut v);
    /// v.sort();
    /// assert_eq!(v, [1, 2, 3, 4, 5]);
    /// ```
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the random number generators: known published values and seeding.

    use super::*;

    /// SplitMix64 with seed 0 gives the published first output.
    #[test]
    fn splitmix_known_value() {
        assert_eq!(SplitMix64::new(0).next_u64(), 0xE220A8397B1DCDAF);
    }

    /// Seeding with 42 sets the xoshiro state to the first four SplitMix64 outputs.
    #[test]
    fn splitmix_seeds_xoshiro() {
        assert_eq!(
            Rng::seed(42).s,
            [
                0xbdd732262feb6e95,
                0x28efe333b266f103,
                0x47526757130f9f52,
                0x581ce1ff0e4ae394
            ]
        );
    }

    /// The first xoshiro256++ outputs for seed 42.
    #[test]
    fn xoshiro_first_outputs() {
        let mut rng = Rng::seed(42);
        assert_eq!(rng.next_u64(), 15021278609987233951);
        assert_eq!(rng.next_u64(), 5881210131331364753);
        assert_eq!(rng.next_u64(), 18149643915985481100);
    }

    /// The first f32 draws for seed 42 are exactly these values.
    #[test]
    fn uniform_f32_first() {
        let mut rng = Rng::seed(42);
        assert_eq!(rng.uniform_f32(), 0.814_305_1_f32);
        assert_eq!(rng.uniform_f32(), 0.318_821_f32);
        assert_eq!(rng.uniform_f32(), 0.983_894_1_f32);
    }

    /// The normal draws have mean near 0 and variance near 1.
    #[test]
    fn normal_moments() {
        let mut rng = Rng::seed(1);
        let n = 100_000;
        let xs: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
    }

    /// `below(7)` stays under 7 and reaches every value.
    #[test]
    fn below_is_in_range() {
        let mut rng = Rng::seed(3);
        let mut seen = [false; 7];
        for _ in 0..1000 {
            let k = rng.below(7);
            assert!(k < 7);
            seen[k as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    /// Even the largest draw never reaches `hi`, whatever the rounding, and never goes
    /// below `lo`.
    #[test]
    fn range_never_returns_hi() {
        let u = 1.0 - f64::EPSILON / 2.0;
        for (lo, hi) in [
            (0.0, 1.0),
            (1.0, 1.0 + f64::EPSILON * 4.0),
            (-1e-300, 1e-300),
            (1e15, 1e15 + 1.0),
        ] {
            let x = Rng::range_from(lo, hi, u);
            assert!(x < hi && x >= lo, "range_from({lo}, {hi}) gave {x}");
        }
    }

    /// Reversed bounds keep the plain `lo + (hi - lo) * u` mapping, as in v0.1, instead of
    /// collapsing to `lo`.
    #[test]
    fn range_reversed_bounds_stays_uniform() {
        for u in [0.0, 0.3, 0.7] {
            assert_eq!(Rng::range_from(3.0, 1.0, u), 3.0 + (1.0 - 3.0) * u);
        }
    }

    /// With equal bounds the only possible value is the bound itself.
    #[test]
    fn range_equal_bounds_returns_lo() {
        assert_eq!(Rng::seed(3).range(2.5, 2.5), 2.5);
    }
}
