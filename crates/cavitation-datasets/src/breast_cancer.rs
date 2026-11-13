//! The UCI Breast Cancer Wisconsin (Diagnostic) data set: 569 breast masses with 30 features
//! each and the diagnosis, split into train and test.

use std::path::PathBuf;

use cavitation_core::Error;
use cavitation_nn::{Dataset, Targets};

use crate::tabular;

/// The number of rows in `breast-cancer.csv`.
const ROWS: usize = 569;
/// The features per row.
const FEATURES: usize = 30;
/// How many shuffled rows go to the train set; the rest are the test set.
const TRAIN_ROWS: usize = 455;
/// The ten nuclear measurements, in file order within each group of ten columns.
const MEASUREMENTS: [&str; 10] = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
];

/// Where `breast-cancer.csv` lives in the repository checkout.
fn csv_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/breast-cancer/breast-cancer.csv")
}

/// The header line of `breast-cancer.csv`: `mean_<m>` for each measurement, then
/// `<m>_error`, then `worst_<m>`, then `target`.
fn header() -> String {
    let mut names: Vec<String> = Vec::with_capacity(FEATURES + 1);
    names.extend(MEASUREMENTS.iter().map(|m| format!("mean_{m}")));
    names.extend(MEASUREMENTS.iter().map(|m| format!("{m}_error")));
    names.extend(MEASUREMENTS.iter().map(|m| format!("worst_{m}")));
    names.push("target".to_string());
    names.join(",")
}

/// Loads the Breast Cancer Wisconsin (Diagnostic) data set from
/// `data/breast-cancer/breast-cancer.csv`, shuffled with [`Mulberry32`](crate::Mulberry32)
/// (seed 42) exactly as the digits split ([`digits::load`](crate::digits::load)) and split
/// into 455 training rows and 114 test rows.
///
/// Each row has 30 features, unscaled: the mean, the standard error and the worst value of
/// ten measurements of the cell nuclei in a fine needle aspirate. The target is the class,
/// as [`Targets::Classes`]: 0 for malignant and 1 for benign, as scikit-learn numbers them.
/// Values are read as `f64` and rounded to `f32`. The file is found relative to this crate's
/// source directory, so the function works from a checkout of the repository.
/// `data/breast-cancer/README.md` gives the source, the licence and the first entries of the
/// shuffled order.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be read and [`Error::Format`] naming the line if
/// the header differs, a row is not 31 finite numbers, a class is not 0 or 1, or there are
/// not 569 rows.
///
/// # Examples
///
/// ```
/// let (train, test) = cavitation_datasets::breast_cancer::load().unwrap();
/// assert_eq!((train.len(), test.len()), (455, 114));
/// assert_eq!(train.features(), 30);
/// ```
pub fn load() -> Result<(Dataset, Dataset), Error> {
    let path = csv_path();
    let table = tabular::read(&path, &header(), ROWS, FEATURES)?;
    if let Some(row) = table.target.iter().position(|&v| v != 0.0 && v != 1.0) {
        return Err(Error::Format(format!(
            "{}, line {}: the class is {}, expected 0 or 1",
            path.display(),
            row + 2,
            table.target[row]
        )));
    }
    tabular::split(&table, TRAIN_ROWS, |y| {
        Ok(Targets::Classes(y.iter().map(|&v| v as usize).collect()))
    })
}

#[cfg(test)]
mod tests {
    //! Tests for the Breast Cancer loader: the counts, the classes and the data file itself.

    use super::*;
    use crate::sha256::sha256_hex;

    /// 455 train and 114 test rows of 30 features, 212 malignant and 357 benign rows in all,
    /// and the committed file is the one the fixtures were made from.
    #[test]
    fn breast_cancer_counts_and_checksum() {
        let (train, test) = load().unwrap();
        assert_eq!((train.len(), test.len()), (455, 114));
        assert_eq!((train.features(), test.features()), (30, 30));
        let mut counts = [0usize; 2];
        for set in [&train, &test] {
            let Targets::Classes(labels) = &set.targets else {
                panic!("breast cancer targets must be class labels");
            };
            assert_eq!(labels.len(), set.len());
            for &label in labels {
                counts[label] += 1;
            }
            assert!(set.x.data().iter().all(|&v| v >= 0.0));
        }
        assert_eq!(counts, [212, 357]);
        // Training row 0 is source row 259 (line 261 of the file, after the header), malignant.
        assert_eq!(&train.x.data()[..3], &[15.53, 33.56, 103.7]);
        let Targets::Classes(labels) = &train.targets else {
            unreachable!("checked above");
        };
        assert_eq!(labels[0], 0);
        let bytes = std::fs::read(csv_path()).unwrap();
        assert_eq!(
            sha256_hex(&bytes),
            "432ff316e7bfb60b70a275064b4401315cc39f09c9099d031013a23647e98687"
        );
    }
}
