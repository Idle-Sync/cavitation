//! Datasets for the cavitation examples, tests and benchmark. This crate is not published.
//!
//! - [`breast_cancer`]: the UCI Breast Cancer Wisconsin (Diagnostic) data set, 569 rows of 30
//!   features and a class (0 malignant, 1 benign), shuffled into 455 training and 114 test
//!   rows.
//! - [`cifar10`]: loads CIFAR-10 from the six batch files of its binary version, which
//!   `scripts/data/get-cifar10.sh` downloads.
//! - [`concrete`]: the UCI Concrete Compressive Strength data set, 1,030 rows of eight inputs
//!   and the strength in MPa, shuffled into 824 training and 206 test rows.
//! - [`digits`]: the 1,797 8 by 8 digit images from scikit-learn, shuffled with
//!   [`Mulberry32`] (seed 42) into 1,437 training and 360 test rows.
//! - [`mnist`]: loads MNIST from its four uncompressed IDX files, which
//!   `scripts/data/get-mnist.sh` downloads.
//! - [`mulberry32`]: the mulberry32 generator, which shuffles the digits, Concrete and Breast
//!   Cancer splits (seed 42, the same shuffle for each).
//!
//! The digits, Concrete and Breast Cancer files are in the repository's `data/` folder, each
//! with a README giving its source, licence and checksum.

pub mod breast_cancer;
pub mod cifar10;
pub mod concrete;
pub mod digits;
pub mod mnist;
pub mod mulberry32;
#[cfg(test)]
mod sha256;
mod tabular;

pub use mulberry32::Mulberry32;

#[cfg(test)]
mod tests {
    //! The Concrete and Breast Cancer splits: fixed by the seed, and taken from the pinned rows.

    use crate::digits::split_order;

    /// Loading twice gives the same split, and the shuffled order starts with the entries the
    /// data READMEs give.
    #[test]
    fn splits_deterministic() {
        assert_eq!(split_order(1030)[..5], [476, 747, 84, 914, 207]);
        assert_eq!(split_order(1030)[824..829], [289, 495, 705, 693, 523]);
        assert_eq!(split_order(1030)[..824].iter().sum::<usize>(), 427_018);
        assert_eq!(split_order(569)[..5], [259, 560, 417, 460, 288]);
        assert_eq!(split_order(569)[455..460], [404, 287, 94, 154, 301]);
        assert_eq!(split_order(569)[..455].iter().sum::<usize>(), 130_532);
        assert_eq!(
            crate::concrete::load().unwrap(),
            crate::concrete::load().unwrap()
        );
        assert_eq!(
            crate::breast_cancer::load().unwrap(),
            crate::breast_cancer::load().unwrap()
        );
    }
}
