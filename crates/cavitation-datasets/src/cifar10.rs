//! The CIFAR-10 loader: reads the binary version's six batch files into train and test sets.

use std::path::Path;

use cavitation_core::{Error, Tensor};
use cavitation_nn::{Dataset, Targets};

/// Pixel bytes in one image: 32 by 32 pixels, three colour planes.
pub const FEATURES: usize = 3 * 32 * 32;

/// Bytes in one record: the label byte, then the [`FEATURES`] pixel bytes.
pub const RECORD: usize = 1 + FEATURES;

/// The number of classes; every label is below it.
pub const CLASSES: usize = 10;

/// The five training batch files, in the order their records are loaded.
const TRAIN_FILES: [&str; 5] = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];

/// The test batch file.
const TEST_FILE: &str = "test_batch.bin";

/// Reads a whole file; a missing file's error says how to download CIFAR-10.
fn read(path: &Path) -> Result<Vec<u8>, Error> {
    std::fs::read(path).map_err(|source| {
        let source = if source.kind() == std::io::ErrorKind::NotFound {
            std::io::Error::new(
                source.kind(),
                format!("{source}; run scripts/data/get-cifar10.sh to download CIFAR-10"),
            )
        } else {
            source
        };
        Error::Io {
            path: path.to_path_buf(),
            source,
        }
    })
}

/// Appends the records of one batch file's `bytes` to `pixels` (each byte divided by 255)
/// and `labels`, checking that the file is a whole, positive number of [`RECORD`]-byte
/// records and that every label is below [`CLASSES`].
fn parse_batch(
    path: &Path,
    bytes: &[u8],
    pixels: &mut Vec<f32>,
    labels: &mut Vec<usize>,
) -> Result<(), Error> {
    let bad = |what: String| Error::Format(format!("{}: {what}", path.display()));
    if bytes.is_empty() || !bytes.len().is_multiple_of(RECORD) {
        return Err(bad(format!(
            "file is {} bytes, not a whole number of {RECORD}-byte records",
            bytes.len()
        )));
    }
    let records = bytes.len() / RECORD;
    pixels.reserve(records * FEATURES);
    labels.reserve(records);
    for (i, record) in bytes.as_chunks::<RECORD>().0.iter().enumerate() {
        let label = usize::from(record[0]);
        if label >= CLASSES {
            return Err(bad(format!(
                "record {i} has label {label}, expected 0 to {}",
                CLASSES - 1
            )));
        }
        labels.push(label);
        pixels.extend(record[1..].iter().map(|&p| f32::from(p) / 255.0));
    }
    Ok(())
}

/// Reads the batch files at `paths`, in order, into one dataset.
fn load_files(paths: &[&Path]) -> Result<Dataset, Error> {
    let (mut pixels, mut labels) = (Vec::new(), Vec::new());
    for path in paths {
        parse_batch(path, &read(path)?, &mut pixels, &mut labels)?;
    }
    Ok(Dataset {
        x: Tensor::from_vec(pixels, &[labels.len(), FEATURES])?,
        targets: Targets::Classes(labels),
    })
}

/// Loads one CIFAR-10 batch file of the binary version, such as `data_batch_1.bin`.
///
/// Each record becomes a row of [`FEATURES`] (3,072) features, the pixel bytes divided by
/// 255 in the file's order (the 1,024 red values of the 32 by 32 image row by row, then the
/// green, then the blue), and its label byte a class ([`Targets::Classes`]).
///
/// # Errors
///
/// Returns [`Error::Io`] naming the file if it cannot be read; for a missing file the message
/// points to `scripts/data/get-cifar10.sh`. Returns [`Error::Format`] naming the file if it
/// is empty or not a whole number of [`RECORD`] (3,073) bytes long, or if a label is 10 or
/// more.
///
/// # Examples
///
/// ```
/// use cavitation_datasets::cifar10::{load_batch, FEATURES, RECORD};
/// use cavitation_nn::Targets;
///
/// // One record: label 3, every pixel byte 255.
/// let mut record = vec![255u8; RECORD];
/// record[0] = 3;
/// let path = std::env::temp_dir().join(format!("cifar10-doc-{}.bin", std::process::id()));
/// std::fs::write(&path, &record).unwrap();
/// let batch = load_batch(&path).unwrap();
/// std::fs::remove_file(&path).unwrap();
/// assert_eq!(batch.x.shape().dims(), &[1, FEATURES]);
/// assert!(batch.x.data().iter().all(|&p| p == 1.0));
/// assert!(matches!(&batch.targets, Targets::Classes(labels) if labels == &[3]));
/// ```
pub fn load_batch(path: &Path) -> Result<Dataset, Error> {
    load_files(&[path])
}

/// Loads CIFAR-10 from the binary version's batch files in `dir` (`data_batch_1.bin` to
/// `data_batch_5.bin` and `test_batch.bin`), as `(train, test)`.
///
/// The training set is the five training batches in order (50,000 images in the published
/// files) and the test set is the test batch (10,000). Each image is a row of [`FEATURES`]
/// (3,072) features, as [`load_batch`] reads it. `scripts/data/get-cifar10.sh` downloads and
/// unpacks the files into `data/cifar10/`.
///
/// # Errors
///
/// Returns the first error of [`load_batch`] on any of the six files.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// // The files are not in the repository; a missing directory is an error, not a panic.
/// assert!(cavitation_datasets::cifar10::load(Path::new("no-such-directory")).is_err());
/// ```
pub fn load(dir: &Path) -> Result<(Dataset, Dataset), Error> {
    let train: Vec<_> = TRAIN_FILES.iter().map(|name| dir.join(name)).collect();
    let train: Vec<&Path> = train.iter().map(|p| p.as_path()).collect();
    let test = dir.join(TEST_FILE);
    Ok((load_files(&train)?, load_files(&[&test])?))
}

#[cfg(test)]
mod tests {
    //! Tests for the CIFAR-10 loader that need no CIFAR-10 download.

    use super::*;
    use std::path::PathBuf;

    /// A fresh empty directory under the system temp folder, named for the test.
    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("cavitation-cifar10-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A record with `label` and pixel byte `j` equal to `(j * 7 + seed) % 256`.
    fn record(label: u8, seed: usize) -> Vec<u8> {
        let mut r = vec![label];
        r.extend((0..FEATURES).map(|j| ((j * 7 + seed) % 256) as u8));
        r
    }

    /// Writes the six batch files with `train` records in each training batch and `test` in
    /// the test batch; the labels count up from 0, modulo 10, across each set.
    fn write_set(dir: &Path, train: usize, test: usize) {
        let mut next = 0usize;
        for name in TRAIN_FILES {
            let bytes: Vec<u8> = (0..train)
                .flat_map(|_| {
                    next += 1;
                    record(((next - 1) % 10) as u8, next)
                })
                .collect();
            std::fs::write(dir.join(name), bytes).unwrap();
        }
        let bytes: Vec<u8> = (0..test).flat_map(|i| record((i % 10) as u8, i)).collect();
        std::fs::write(dir.join(TEST_FILE), bytes).unwrap();
    }

    /// A two-record file gives two rows of 3,072 pixels divided by 255 in file order, and
    /// the two labels.
    #[test]
    fn cifar_reads_known_record() {
        let dir = temp_dir("known");
        let path = dir.join("two.bin");
        let mut bytes = record(9, 0);
        bytes.extend(record(0, 5));
        std::fs::write(&path, &bytes).unwrap();
        let batch = load_batch(&path).unwrap();
        assert_eq!(batch.x.shape().dims(), &[2, FEATURES]);
        let x = batch.x.data();
        assert_eq!(&x[..3], &[0.0, 7.0 / 255.0, 14.0 / 255.0]);
        // The last blue pixel of the first image, and the first red pixel of the second.
        assert_eq!(x[FEATURES - 1], ((3071 * 7) % 256) as f32 / 255.0);
        assert_eq!(x[FEATURES], 5.0 / 255.0);
        assert!(matches!(&batch.targets, Targets::Classes(l) if l == &[9, 0]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A file that ends inside a record, or is empty, is a `Format` error naming the file and
    /// its length.
    #[test]
    fn cifar_rejects_truncated_record() {
        let dir = temp_dir("truncated");
        let path = dir.join("short.bin");
        let mut bytes = record(1, 0);
        bytes.extend(&record(2, 0)[..100]);
        std::fs::write(&path, &bytes).unwrap();
        let err = load_batch(&path).unwrap_err();
        assert!(matches!(err, Error::Format(_)), "{err:?}");
        let text = err.to_string();
        assert!(text.contains("short.bin"), "{text}");
        assert!(text.contains("3173 bytes"), "{text}");
        assert!(text.contains("3073-byte records"), "{text}");

        std::fs::write(&path, []).unwrap();
        assert!(matches!(load_batch(&path), Err(Error::Format(_))));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A label of 10 or more is a `Format` error naming the file, the record and the label.
    #[test]
    fn cifar_rejects_label_above_9() {
        let dir = temp_dir("label");
        write_set(&dir, 2, 2);
        let mut bytes = record(4, 0);
        bytes.extend(record(10, 0));
        std::fs::write(dir.join("data_batch_3.bin"), &bytes).unwrap();
        let err = load(&dir).unwrap_err();
        assert!(matches!(err, Error::Format(_)), "{err:?}");
        let text = err.to_string();
        assert!(text.contains("data_batch_3.bin"), "{text}");
        assert!(text.contains("record 1 has label 10"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The five training batches load in order into one set, the test batch into another.
    #[test]
    fn cifar_loads_train_and_test() {
        let dir = temp_dir("set");
        write_set(&dir, 2, 3);
        let (train, test) = load(&dir).unwrap();
        assert_eq!((train.len(), test.len()), (10, 3));
        assert_eq!(train.features(), FEATURES);
        let Targets::Classes(labels) = &train.targets else {
            panic!("CIFAR-10 targets must be class labels");
        };
        assert_eq!(labels, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        // The first pixel of the fourth record (data_batch_2.bin's second) has seed 4.
        assert_eq!(train.x.data()[3 * FEATURES], 4.0 / 255.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A missing directory is an `Io` error naming the first file, and its message points to
    /// the download script.
    #[test]
    fn cifar_missing_dir_error_mentions_script() {
        let err = load(Path::new("/nonexistent-cavitation-dir/cifar10")).unwrap_err();
        assert!(matches!(err, Error::Io { .. }), "{err:?}");
        let text = err.to_string();
        assert!(text.contains("data_batch_1.bin"), "{text}");
        assert!(text.contains("scripts/data/get-cifar10.sh"), "{text}");
    }

    /// If the real CIFAR-10 files are present, they load with the published sizes.
    #[test]
    fn cifar_real_files_if_present() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/cifar10");
        if !dir.join(TEST_FILE).exists() {
            return;
        }
        let (train, test) = load(&dir).unwrap();
        assert_eq!((train.len(), test.len()), (50_000, 10_000));
        assert_eq!(train.features(), FEATURES);
    }
}
