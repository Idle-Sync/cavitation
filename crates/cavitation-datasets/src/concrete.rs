//! The UCI Concrete Compressive Strength data set: 1,030 concrete mixtures with eight inputs
//! and the measured strength in MPa, split into train and test.

use std::path::PathBuf;

use cavitation_core::{Error, Tensor};
use cavitation_nn::{Dataset, Targets};

use crate::tabular;

/// The number of rows in `concrete.csv`.
const ROWS: usize = 1030;
/// The inputs per row.
const FEATURES: usize = 8;
/// How many shuffled rows go to the train set; the rest are the test set.
const TRAIN_ROWS: usize = 824;
/// The header line of `concrete.csv`.
const HEADER: &str = "cement,blast_furnace_slag,fly_ash,water,superplasticizer,\
                      coarse_aggregate,fine_aggregate,age_days,strength_mpa";

/// Where `concrete.csv` lives in the repository checkout.
fn csv_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/concrete/concrete.csv")
}

/// Loads the Concrete Compressive Strength data set from `data/concrete/concrete.csv`,
/// shuffled with [`Mulberry32`](crate::Mulberry32) (seed 42) exactly as the digits split
/// ([`digits::load`](crate::digits::load)) and split into 824 training rows and 206 test rows.
///
/// Each row has eight features, unscaled: cement, blast furnace slag, fly ash, water,
/// superplasticizer, coarse aggregate and fine aggregate in kilograms per cubic metre, then
/// the age in days. The target is the compressive strength in MPa, as
/// [`Targets::Values`] of shape `[rows, 1]`. Values are read as `f64` and rounded to `f32`.
/// The file is found relative to this crate's source directory, so the function works from a
/// checkout of the repository. `data/concrete/README.md` gives the source, the licence and the
/// first entries of the shuffled order.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be read and [`Error::Format`] naming the line if
/// the header differs, a row is not nine finite numbers, or there are not 1,030 rows.
///
/// # Examples
///
/// ```
/// let (train, test) = cavitation_datasets::concrete::load().unwrap();
/// assert_eq!((train.len(), test.len()), (824, 206));
/// assert_eq!(train.features(), 8);
/// ```
pub fn load() -> Result<(Dataset, Dataset), Error> {
    let table = tabular::read(&csv_path(), HEADER, ROWS, FEATURES)?;
    tabular::split(&table, TRAIN_ROWS, |y| {
        let values: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        Ok(Targets::Values(Tensor::from_vec(values, &[y.len(), 1])?))
    })
}

#[cfg(test)]
mod tests {
    //! Tests for the Concrete loader: the counts, the targets and the data file itself.

    use super::*;
    use crate::sha256::sha256_hex;

    /// 824 train and 206 test rows of 8 features, positive strengths as `[rows, 1]` values,
    /// and the committed file is the one the fixtures were made from.
    #[test]
    fn concrete_counts_and_checksum() {
        let (train, test) = load().unwrap();
        assert_eq!((train.len(), test.len()), (824, 206));
        assert_eq!((train.features(), test.features()), (8, 8));
        for set in [&train, &test] {
            let Targets::Values(y) = &set.targets else {
                panic!("concrete targets must be values");
            };
            assert_eq!(y.shape().dims(), &[set.len(), 1]);
            assert!(y.data().iter().all(|&v| v > 0.0 && v < 100.0));
            assert!(set.x.data().iter().all(|&v| v >= 0.0));
        }
        // Training row 0 is source row 476 (line 478 of the file, after the header).
        assert_eq!(
            &train.x.data()[..8],
            &[446.0, 24.0, 79.0, 162.0, 11.64, 967.0, 712.0, 3.0]
        );
        let Targets::Values(y) = &train.targets else {
            unreachable!("checked above");
        };
        assert_eq!(y.data()[0], 25.021084039999998_f64 as f32);
        let bytes = std::fs::read(csv_path()).unwrap();
        assert_eq!(
            sha256_hex(&bytes),
            "858bd88ee4817fa50a3915312403ea425d91c34927e77c8bcfc334eee6544995"
        );
    }
}
