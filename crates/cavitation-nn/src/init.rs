//! `Init`: how a freshly created parameter tensor is filled.

use cavitation_core::{Real, Rng, Tensor};

/// How a parameter tensor is filled with its starting values.
///
/// The formulas are PyTorch's: `XavierUniform` and `XavierNormal` are
/// `xavier_uniform_` and `xavier_normal_` with gain 1; `HeUniform` and `HeNormal` are
/// `kaiming_uniform_` and `kaiming_normal_` with `nonlinearity='relu'` and `mode='fan_in'`.
///
/// ```
/// use cavitation_core::{Rng, Tensor};
/// use cavitation_nn::Init;
/// let mut w = Tensor::<f32>::zeros(&[3, 2]);
/// Init::XavierUniform.fill(&mut w, 2, 3, &mut Rng::seed(0));
/// let bound = (6.0f32 / 5.0).sqrt();
/// assert!(w.data().iter().all(|v| v.abs() <= bound));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Init {
    /// Uniform in `±sqrt(6 / (fan_in + fan_out))`.
    XavierUniform,
    /// Normal with standard deviation `sqrt(2 / (fan_in + fan_out))`.
    XavierNormal,
    /// Uniform in `±sqrt(6 / fan_in)`.
    HeUniform,
    /// Normal with standard deviation `sqrt(2 / fan_in)`.
    HeNormal,
    /// Every element 0. Does not use the generator.
    Zeros,
    /// Every element the given value. Does not use the generator.
    Constant(f64),
    /// Normal with mean 0 and the given standard deviation, whatever the fans: PyTorch's
    /// `normal_(0, std)`, as recipes that say `std = 0.02` use. A deviation of 0 gives
    /// zeros (still one draw per element).
    ///
    /// ```
    /// use cavitation_core::{Rng, Tensor};
    /// use cavitation_nn::Init;
    /// let mut w = Tensor::<f32>::zeros(&[256, 256]);
    /// Init::Normal { std: 0.5 }.fill(&mut w, 256, 256, &mut Rng::seed(0));
    /// let n = w.data().len() as f64;
    /// let var = w.data().iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / n;
    /// assert!((var.sqrt() - 0.5).abs() < 0.01);
    /// ```
    Normal {
        /// The standard deviation: finite and not negative.
        std: f64,
    },
}

impl Init {
    /// Fills `t` in place. Elements are drawn in row-major order, one draw each, so a
    /// seed reproduces the same values exactly. `Zeros` and `Constant` do not touch `rng`.
    ///
    /// `fan_in` is the number of inputs of the layer the tensor belongs to and `fan_out` the
    /// number of outputs (for a bias, the layer's `fan_in` and `fan_out` as well).
    ///
    /// # Panics
    ///
    /// Panics if the formula needs a fan that is 0 (it would divide by zero), or if
    /// [`Init::Normal`]'s deviation is negative, infinite or NaN.
    ///
    /// ```
    /// use cavitation_core::{Rng, Tensor};
    /// use cavitation_nn::Init;
    /// let mut b = Tensor::<f32>::zeros(&[4]);
    /// Init::Constant(0.5).fill(&mut b, 3, 4, &mut Rng::seed(0));
    /// assert_eq!(b.data(), &[0.5; 4]);
    /// ```
    pub fn fill<T: Real>(&self, t: &mut Tensor<T>, fan_in: usize, fan_out: usize, rng: &mut Rng) {
        let fans = |n: usize| {
            assert!(
                n > 0,
                "Init::{self:?}: fan_in {fan_in} and fan_out {fan_out} must be positive for this formula"
            );
            n as f64
        };
        match *self {
            Init::Zeros => t.data_mut().fill(T::ZERO),
            Init::Constant(c) => t.data_mut().fill(T::from_f64(c)),
            Init::XavierUniform => {
                let bound = (6.0 / fans(fan_in + fan_out)).sqrt();
                uniform(t, bound, rng);
            }
            Init::XavierNormal => {
                let std = (2.0 / fans(fan_in + fan_out)).sqrt();
                normal(t, std, rng);
            }
            Init::HeUniform => {
                let bound = (6.0 / fans(fan_in)).sqrt();
                uniform(t, bound, rng);
            }
            Init::HeNormal => {
                let std = (2.0 / fans(fan_in)).sqrt();
                normal(t, std, rng);
            }
            Init::Normal { std } => {
                assert!(
                    std.is_finite() && std >= 0.0,
                    "Init::Normal: std must be finite and not negative, got {std}"
                );
                normal(t, std, rng);
            }
        }
    }
}

/// Fills `t` with uniform draws in `[-bound, bound)`, in row-major order.
fn uniform<T: Real>(t: &mut Tensor<T>, bound: f64, rng: &mut Rng) {
    for v in t.data_mut() {
        *v = T::from_f64(rng.range(-bound, bound));
    }
}

/// Fills `t` with normal draws of standard deviation `std`, in row-major order.
fn normal<T: Real>(t: &mut Tensor<T>, std: f64, rng: &mut Rng) {
    for v in t.data_mut() {
        *v = T::from_f64(rng.normal() * std);
    }
}

/// Tests of the initialisers.
#[cfg(test)]
mod tests {
    use super::*;
    use cavitation_core::{Rng, Tensor};

    /// He uniform draws stay within `±sqrt(6 / fan_in)` and average to about zero.
    #[test]
    fn init_bounds() {
        let mut t = Tensor::<f32>::zeros(&[1024, 784]);
        Init::HeUniform.fill(&mut t, 784, 1024, &mut Rng::seed(1));
        let bound = (6.0f64 / 784.0).sqrt();
        assert!((bound - 0.0874818).abs() < 1e-6);
        let mut sum = 0.0f64;
        for &v in t.data() {
            assert!(f64::from(v).abs() <= bound, "{v} outside ±{bound}");
            sum += f64::from(v);
        }
        let mean = sum / t.data().len() as f64;
        assert!(mean.abs() < 0.002, "mean {mean}");
    }

    /// `Normal { std }` draws have mean about 0 and standard deviation `std` within 1%,
    /// whatever the fans, and are the He normal draws scaled: the same generator draws, so
    /// `Normal { std: sqrt(2 / fan_in) }` gives `HeNormal`'s values bit for bit.
    #[test]
    fn normal_with_a_chosen_std() {
        for std in [0.02, 0.5, 3.0] {
            let mut t = Tensor::<f32>::zeros(&[512, 300]);
            Init::Normal { std }.fill(&mut t, 1, 1, &mut Rng::seed(5));
            let n = t.data().len() as f64;
            let mean = t.data().iter().map(|&v| f64::from(v)).sum::<f64>() / n;
            let var = t
                .data()
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / n;
            assert!(mean.abs() < 0.01 * std, "std {std}: mean {mean}");
            assert!(
                (var.sqrt() - std).abs() < 0.01 * std,
                "std {std}: {}",
                var.sqrt()
            );
        }
        let (mut he, mut chosen) = (Tensor::<f32>::zeros(&[8, 6]), Tensor::<f32>::zeros(&[8, 6]));
        Init::HeNormal.fill(&mut he, 6, 8, &mut Rng::seed(7));
        Init::Normal {
            std: (2.0f64 / 6.0).sqrt(),
        }
        .fill(&mut chosen, 6, 8, &mut Rng::seed(7));
        assert_eq!(he.data(), chosen.data());
        let mut zero = Tensor::<f32>::full(&[4], 1.0);
        Init::Normal { std: 0.0 }.fill(&mut zero, 0, 0, &mut Rng::seed(1));
        assert!(zero.data().iter().all(|&v| v == 0.0));
    }

    /// A negative or NaN deviation panics, naming the variant and the value.
    #[test]
    #[should_panic(expected = "Init::Normal: std must be finite and not negative, got -1")]
    fn normal_with_a_negative_std_panics() {
        Init::Normal { std: -1.0 }.fill(&mut Tensor::<f32>::zeros(&[2]), 1, 1, &mut Rng::seed(0));
    }

    /// He normal draws have standard deviation `sqrt(2 / fan_in)` within 1%.
    #[test]
    fn init_normal_std() {
        let mut t = Tensor::<f32>::zeros(&[1024, 784]);
        Init::HeNormal.fill(&mut t, 784, 1024, &mut Rng::seed(2));
        let n = t.data().len() as f64;
        let mean = t.data().iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = t
            .data()
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        let want = (2.0f64 / 784.0).sqrt();
        assert!(
            (var.sqrt() - want).abs() < 0.01 * want,
            "std {}",
            var.sqrt()
        );
    }

    /// Xavier uniform and normal use `fan_in + fan_out`.
    #[test]
    fn xavier_scales() {
        let mut t = Tensor::<f64>::zeros_generic(&[200, 100]);
        Init::XavierUniform.fill(&mut t, 100, 200, &mut Rng::seed(3));
        let bound = (6.0f64 / 300.0).sqrt();
        assert!(t.data().iter().all(|v| v.abs() <= bound));
        assert!(t.data().iter().any(|v| v.abs() > 0.9 * bound));
        Init::XavierNormal.fill(&mut t, 100, 200, &mut Rng::seed(3));
        let n = t.data().len() as f64;
        let std = (t.data().iter().map(|v| v * v).sum::<f64>() / n).sqrt();
        let want = (2.0f64 / 300.0).sqrt();
        assert!((std - want).abs() < 0.03 * want, "std {std}");
    }

    /// `Zeros` and `Constant` fill without touching the generator; the same seed gives the
    /// same values.
    #[test]
    fn constants_and_reproducibility() {
        let mut rng = Rng::seed(4);
        let mut before = Rng::seed(4);
        let mut t = Tensor::<f32>::full(&[3, 2], 9.0);
        Init::Constant(0.25).fill(&mut t, 2, 3, &mut rng);
        assert_eq!(t.data(), &[0.25; 6]);
        Init::Zeros.fill(&mut t, 2, 3, &mut rng);
        assert_eq!(t.data(), &[0.0; 6]);
        assert_eq!(rng.next_u64(), before.next_u64());

        let draw = |seed| {
            let mut t = Tensor::<f32>::zeros(&[4, 4]);
            Init::XavierNormal.fill(&mut t, 4, 4, &mut Rng::seed(seed));
            t
        };
        assert_eq!(draw(7).data(), draw(7).data());
        assert_ne!(draw(7).data(), draw(8).data());
    }
}
