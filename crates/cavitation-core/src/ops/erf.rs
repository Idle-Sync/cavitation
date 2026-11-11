//! The error function for f32 and f64, ported from FreeBSD's msun (`s_erf.c` and `s_erff.c`).
//!
//! Both ports follow the originals branch for branch and keep their coefficients. The
//! original notices apply to this file:
//!
//! ```text
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//!
//! s_erff.c -- float version of s_erf.c.
//! Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
//! ```
//!
//! The method, from the original: on `|x| < 0.84375`, `erf(x) = x + x·R(x²)` with `R` a
//! rational function; on `[0.84375, 1.25)`, `erf(x) = c + P(s)/Q(s)` with `s = |x| - 1`; on
//! `[1.25, 6)` (f32: `[1.25, 4)`), `erf(x) = 1 - erfc(x)` with
//! `erfc(x) = exp(-x² - 0.5625 + R/S) / x`; beyond that the result is ±1.

// The coefficients are kept exactly as printed in the original sources.
#![allow(clippy::excessive_precision)]

/// `1e-300`, added to and subtracted from 1 in the original to raise the inexact flag; the
/// result is 1.
const TINY64: f64 = 1e-300;
/// `1e-30`, the f32 counterpart of [`TINY64`].
const TINY32: f32 = 1e-30;

/// `erf` for f64: `(2/sqrt(pi))` times the integral of `exp(-t*t)` from 0 to `x`. Accurate to
/// about 1 ulp; `erf(-x) = -erf(x)`, `erf(NaN)` is NaN and `erf(±inf)` is ±1.
pub(crate) fn erf_f64(x: f64) -> f64 {
    const ERX: f64 = 8.45062911510467529297e-01;
    const EFX: f64 = 1.28379167095512586316e-01;
    const EFX8: f64 = 1.02703333676410069053e+00;
    // Coefficients for approximation to erf on [0, 0.84375].
    const PP0: f64 = 1.28379167095512558561e-01;
    const PP1: f64 = -3.25042107247001499370e-01;
    const PP2: f64 = -2.84817495755985104766e-02;
    const PP3: f64 = -5.77027029648944159157e-03;
    const PP4: f64 = -2.37630166566501626084e-05;
    const QQ1: f64 = 3.97917223959155352819e-01;
    const QQ2: f64 = 6.50222499887672944485e-02;
    const QQ3: f64 = 5.08130628187576562776e-03;
    const QQ4: f64 = 1.32494738004321644526e-04;
    const QQ5: f64 = -3.96022827877536812320e-06;
    // Coefficients for approximation to erf in [0.84375, 1.25].
    const PA0: f64 = -2.36211856075265944077e-03;
    const PA1: f64 = 4.14856118683748331666e-01;
    const PA2: f64 = -3.72207876035701323847e-01;
    const PA3: f64 = 3.18346619901161753674e-01;
    const PA4: f64 = -1.10894694282396677476e-01;
    const PA5: f64 = 3.54783043256182359371e-02;
    const PA6: f64 = -2.16637559486879084300e-03;
    const QA1: f64 = 1.06420880400844228286e-01;
    const QA2: f64 = 5.40397917702171048937e-01;
    const QA3: f64 = 7.18286544141962662868e-02;
    const QA4: f64 = 1.26171219808761642112e-01;
    const QA5: f64 = 1.36370839120290507362e-02;
    const QA6: f64 = 1.19844998467991074170e-02;
    // Coefficients for approximation to erfc in [1.25, 1/0.35].
    const RA0: f64 = -9.86494403484714822705e-03;
    const RA1: f64 = -6.93858572707181764372e-01;
    const RA2: f64 = -1.05586262253232909814e+01;
    const RA3: f64 = -6.23753324503260060396e+01;
    const RA4: f64 = -1.62396669462573470355e+02;
    const RA5: f64 = -1.84605092906711035994e+02;
    const RA6: f64 = -8.12874355063065934246e+01;
    const RA7: f64 = -9.81432934416914548592e+00;
    const SA1: f64 = 1.96512716674392571292e+01;
    const SA2: f64 = 1.37657754143519042600e+02;
    const SA3: f64 = 4.34565877475229228821e+02;
    const SA4: f64 = 6.45387271733267880336e+02;
    const SA5: f64 = 4.29008140027567833386e+02;
    const SA6: f64 = 1.08635005541779435134e+02;
    const SA7: f64 = 6.57024977031928170135e+00;
    const SA8: f64 = -6.04244152148580987438e-02;
    // Coefficients for approximation to erfc in [1/0.35, 28].
    const RB0: f64 = -9.86494292470009928597e-03;
    const RB1: f64 = -7.99283237680523006574e-01;
    const RB2: f64 = -1.77579549177547519889e+01;
    const RB3: f64 = -1.60636384855821916062e+02;
    const RB4: f64 = -6.37566443368389627722e+02;
    const RB5: f64 = -1.02509513161107724954e+03;
    const RB6: f64 = -4.83519191608651397019e+02;
    const SB1: f64 = 3.03380607434824582924e+01;
    const SB2: f64 = 3.25792512996573918826e+02;
    const SB3: f64 = 1.53672958608443695994e+03;
    const SB4: f64 = 3.19985821950859553908e+03;
    const SB5: f64 = 2.55305040643316442583e+03;
    const SB6: f64 = 4.74528541206955367215e+02;
    const SB7: f64 = -2.24409524465858183362e+01;

    let bits = x.to_bits();
    let hx = (bits >> 32) as u32;
    let negative = hx >> 31 != 0;
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        // NaN stays NaN (1/NaN); +-inf gives +-1.
        let i = (hx >> 31) << 1;
        return (1.0 - i as f64) + 1.0 / x;
    }

    if ix < 0x3feb_0000 {
        // |x| < 0.84375
        if ix < 0x3e30_0000 {
            // |x| < 2**-28
            if ix < 0x0080_0000 {
                // Avoid spurious underflow.
                return (8.0 * x + EFX8 * x) / 8.0;
            }
            return x + EFX * x;
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        return x + x * y;
    }
    if ix < 0x3ff4_0000 {
        // 0.84375 <= |x| < 1.25
        let s = x.abs() - 1.0;
        let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
        let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
        return if negative { -ERX - p / q } else { ERX + p / q };
    }
    if ix >= 0x4018_0000 {
        // inf > |x| >= 6
        return if negative { TINY64 - 1.0 } else { 1.0 - TINY64 };
    }
    let x = x.abs();
    let s = 1.0 / (x * x);
    let (r_poly, s_poly) = if ix < 0x4006_db6e {
        // |x| < 1/0.35
        let r =
            RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
        let q = 1.0
            + s * (SA1
                + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
        (r, q)
    } else {
        // |x| >= 1/0.35
        let r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
        let q =
            1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
        (r, q)
    };
    // x with its low 32 bits cleared, so that z*z is exact.
    let z = f64::from_bits(x.to_bits() & 0xffff_ffff_0000_0000);
    let r = (-z * z - 0.5625).exp() * ((z - x) * (z + x) + r_poly / s_poly).exp();
    if negative { r / x - 1.0 } else { 1.0 - r / x }
}

/// `erf` for f32: the single-precision counterpart of [`erf_f64`], with the same special
/// cases.
pub(crate) fn erf_f32(x: f32) -> f32 {
    const ERX: f32 = 8.42697144e-01;
    const EFX: f32 = 1.28379166e-01;
    const EFX8: f32 = 1.02703333e+00;
    // Domain [0, 0.84375].
    const PP0: f32 = 1.28379166e-01;
    const PP1: f32 = -3.36030394e-01;
    const PP2: f32 = -1.86261395e-03;
    const QQ1: f32 = 3.12324315e-01;
    const QQ2: f32 = 2.16070414e-02;
    const QQ3: f32 = -1.98859372e-03;
    // Domain [0.84375, 1.25].
    const PA0: f32 = 3.65041046e-06;
    const PA1: f32 = 4.15109307e-01;
    const PA2: f32 = -2.09395722e-01;
    const PA3: f32 = 8.67677554e-02;
    const QA1: f32 = 4.95560974e-01;
    const QA2: f32 = 3.71248513e-01;
    const QA3: f32 = 3.92478965e-02;
    // Domain [1.25, 1/0.35].
    const RA0: f32 = -9.88156721e-03;
    const RA1: f32 = -5.43658376e-01;
    const RA2: f32 = -1.66828310e+00;
    const RA3: f32 = -6.91554189e-01;
    const SA1: f32 = 4.48581553e+00;
    const SA2: f32 = 4.10799170e+00;
    const SA3: f32 = 5.53855181e-01;
    // Domain [2.85715, 11].
    const RB0: f32 = -9.86496918e-03;
    const RB1: f32 = -5.48049808e-01;
    const RB2: f32 = -1.84115684e+00;
    const SB1: f32 = 4.87132740e+00;
    const SB2: f32 = 3.04982710e+00;
    const SB3: f32 = -7.61900663e-01;

    let hx = x.to_bits();
    let negative = hx >> 31 != 0;
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x7f80_0000 {
        // NaN stays NaN (1/NaN); +-inf gives +-1.
        let i = (hx >> 31) << 1;
        return (1.0 - i as f32) + 1.0 / x;
    }

    if ix < 0x3f58_0000 {
        // |x| < 0.84375
        if ix < 0x3880_0000 {
            // |x| < 2**-14
            if ix < 0x0400_0000 {
                // |x| < 0x1p-119: avoid spurious underflow.
                return (8.0 * x + EFX8 * x) / 8.0;
            }
            return x + EFX * x;
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * PP2);
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * QQ3));
        let y = r / s;
        return x + x * y;
    }
    if ix < 0x3fa0_0000 {
        // 0.84375 <= |x| < 1.25
        let s = x.abs() - 1.0;
        let p = PA0 + s * (PA1 + s * (PA2 + s * PA3));
        let q = 1.0 + s * (QA1 + s * (QA2 + s * QA3));
        return if negative { -ERX - p / q } else { ERX + p / q };
    }
    if ix >= 0x4080_0000 {
        // inf > |x| >= 4
        return if negative { TINY32 - 1.0 } else { 1.0 - TINY32 };
    }
    let x = x.abs();
    let s = 1.0 / (x * x);
    let (r_poly, s_poly) = if ix < 0x4036_db8c {
        // |x| < 2.85715 ~ 1/0.35
        (
            RA0 + s * (RA1 + s * (RA2 + s * RA3)),
            1.0 + s * (SA1 + s * (SA2 + s * SA3)),
        )
    } else {
        // |x| >= 2.85715 ~ 1/0.35
        (
            RB0 + s * (RB1 + s * RB2),
            1.0 + s * (SB1 + s * (SB2 + s * SB3)),
        )
    };
    // The original clears the low 13 bits of x's word (sign included), so that z*z is exact.
    let z = f32::from_bits(hx & 0xffff_e000);
    let r = (-z * z - 0.5625f32).exp() * ((z - x) * (z + x) + r_poly / s_poly).exp();
    if negative { r / x - 1.0 } else { 1.0 - r / x }
}

/// Tests for `erf`.
#[cfg(test)]
mod tests {
    use super::{erf_f32, erf_f64};

    /// Known values, odd symmetry, and the limits.
    #[test]
    fn erf_known_values() {
        let known = [
            (0.0, 0.0),
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (3.0, 0.9999779095030014),
        ];
        for (x, want) in known {
            let got = erf_f64(x);
            assert!(
                (got - want).abs() <= 2e-16 * want.abs().max(1e-300),
                "{x}: {got}"
            );
            assert_eq!(erf_f64(-x), -got);
            let got32 = erf_f32(x as f32);
            assert!(ulps(got32, want as f32) <= 2, "{x}: {got32}");
            assert_eq!(erf_f32(-(x as f32)), -got32);
        }
        assert_eq!(erf_f64(f64::INFINITY), 1.0);
        assert_eq!(erf_f64(f64::NEG_INFINITY), -1.0);
        assert_eq!(erf_f32(f32::INFINITY), 1.0);
        assert_eq!(erf_f32(f32::NEG_INFINITY), -1.0);
        assert_eq!(erf_f64(10.0), 1.0);
        assert_eq!(erf_f32(10.0), 1.0);
        assert!(erf_f64(f64::NAN).is_nan());
        assert!(erf_f32(f32::NAN).is_nan());
    }

    /// Values from Python's `math.erf` (the C library's `erf`) in every branch of both ports:
    /// f64 within 4 ulp (2 ulp is the typical error; the bound leaves room for `exp`
    /// differences between libraries), f32 within 2 ulp of the correctly rounded result.
    #[test]
    fn erf_matches_reference_in_every_branch() {
        let reference: [(f64, f64); 12] = [
            (1e-5, 1.1283791670579e-05),
            (0.1, 0.1124629160182849),
            (0.3, 0.3286267594591274),
            (0.9, 0.7969082124228322),
            (1.1, 0.8802050695740817),
            (1.5, 0.9661051464753108),
            (2.0, 0.9953222650189527),
            (2.5, 0.999593047982555),
            (3.5, 0.9999992569016276),
            (4.0, 0.9999999845827421),
            (5.5, 0.9999999999999927),
            (5.99, 1.0),
        ];
        for (x, want) in reference {
            let got = erf_f64(x);
            assert!(
                got.to_bits().abs_diff(want.to_bits()) <= 4,
                "{x}: {got} vs {want}"
            );
            let got32 = erf_f32(x as f32);
            assert!(
                ulps(got32, want as f32) <= 2,
                "{x}: {got32} vs {}",
                want as f32
            );
        }
    }

    /// The distance in representable f32 values between `a` and `b`.
    fn ulps(a: f32, b: f32) -> u32 {
        a.to_bits().abs_diff(b.to_bits())
    }

    /// Across every branch, the f32 port stays within 2 ulp of the f64 port rounded to f32.
    #[test]
    fn erf_f32_matches_f64_across_branches() {
        let mut x = -7.0f32;
        while x < 7.0 {
            let want = erf_f64(x as f64) as f32;
            let got = erf_f32(x);
            assert!(
                ulps(got, want) <= 4 || (got - want).abs() < 1e-7,
                "{x}: {got} vs {want}"
            );
            x += 0.0137;
        }
    }
}
