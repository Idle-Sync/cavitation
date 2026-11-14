//! The MNIST loader: reads the four uncompressed IDX files into train and test sets.

use std::path::Path;

use cavitation_core::{Error, Tensor};
use cavitation_nn::{Dataset, Targets};

/// Reads a whole file; a missing file's error says how to download MNIST.
fn read(path: &Path) -> Result<Vec<u8>, Error> {
    std::fs::read(path).map_err(|source| {
        let source = if source.kind() == std::io::ErrorKind::NotFound {
            std::io::Error::new(
                source.kind(),
                format!("{source}; run scripts/data/get-mnist.sh to download MNIST"),
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

/// Splits an IDX file into its dimensions and payload, checking the magic number (2049
/// for labels, 2051 for images), the number of dimensions and the payload length.
fn parse_idx(
    path: &Path,
    bytes: &[u8],
    magic: u32,
    dims: usize,
) -> Result<(Vec<usize>, Vec<u8>), Error> {
    let bad = |what: String| Error::Format(format!("{}: {what}", path.display()));
    let header = 4 + 4 * dims;
    if bytes.len() < header {
        return Err(bad(format!(
            "file is {} bytes, shorter than its {header}-byte header",
            bytes.len()
        )));
    }
    let word =
        |at: usize| u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    if word(0) != magic {
        return Err(bad(format!("magic number {}, expected {magic}", word(0))));
    }
    let sizes: Vec<usize> = (0..dims).map(|d| word(4 + 4 * d) as usize).collect();
    let expected = sizes
        .iter()
        .try_fold(1usize, |total, &size| total.checked_mul(size))
        .ok_or_else(|| bad(format!("dimensions {sizes:?} overflow")))?;
    if bytes.len() - header != expected {
        return Err(bad(format!(
            "{} data bytes for dimensions {sizes:?}, expected {expected}",
            bytes.len() - header
        )));
    }
    Ok((sizes, bytes[header..].to_vec()))
}

/// The side of an MNIST image, in pixels.
const SIDE: usize = 28;

/// Loads one split: `{prefix}-images-idx3-ubyte` and `{prefix}-labels-idx1-ubyte`.
fn load_split(dir: &Path, prefix: &str) -> Result<Dataset, Error> {
    let images_path = dir.join(format!("{prefix}-images-idx3-ubyte"));
    let labels_path = dir.join(format!("{prefix}-labels-idx1-ubyte"));
    let (image_dims, pixels) = parse_idx(&images_path, &read(&images_path)?, 2051, 3)?;
    let (label_dims, labels) = parse_idx(&labels_path, &read(&labels_path)?, 2049, 1)?;
    // MNIST's images are 28 by 28; checking the size here also bounds `rows · cols` when
    // the file holds no image, so the product below cannot overflow.
    if image_dims[1..] != [SIDE, SIDE] {
        return Err(Error::Format(format!(
            "{}: images are {} by {}, expected {SIDE} by {SIDE}",
            images_path.display(),
            image_dims[1],
            image_dims[2]
        )));
    }
    if image_dims[0] != label_dims[0] {
        return Err(Error::Format(format!(
            "{} has {} images but {} has {} labels",
            images_path.display(),
            image_dims[0],
            labels_path.display(),
            label_dims[0]
        )));
    }
    let n = image_dims[0];
    let x: Vec<f32> = pixels.iter().map(|&p| f32::from(p) / 255.0).collect();
    Ok(Dataset {
        x: Tensor::from_vec(x, &[n, image_dims[1] * image_dims[2]])?,
        targets: Targets::Classes(labels.iter().map(|&l| usize::from(l)).collect()),
    })
}

/// Loads MNIST from the four uncompressed IDX files in `dir`
/// (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte` and
/// `t10k-labels-idx1-ubyte`), as `(train, test)`.
///
/// Each image becomes a row of 784 features, the pixel bytes divided by 255, and each label a
/// class ([`Targets::Classes`]). `scripts/data/get-mnist.sh` downloads and unpacks the
/// files into `data/mnist/`.
///
/// # Errors
///
/// Returns [`Error::Io`] naming the file if one cannot be read; for a missing file the message
/// points to `scripts/data/get-mnist.sh`. Returns [`Error::Format`] if a file has the wrong
/// magic number, a length that does not match its dimensions (including dimensions whose
/// product overflows `usize`), images that are not 28 by 28, or if the image and label counts
/// differ.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// // The files are not in the repository; a missing directory is an error, not a panic.
/// assert!(cavitation_datasets::mnist::load(Path::new("no-such-directory")).is_err());
/// ```
pub fn load(dir: &Path) -> Result<(Dataset, Dataset), Error> {
    Ok((load_split(dir, "train")?, load_split(dir, "t10k")?))
}

#[cfg(test)]
mod tests {
    //! Tests for the MNIST loader that need no MNIST download.

    use super::*;
    use cavitation_core::Error;
    use cavitation_nn::Targets;
    use std::path::{Path, PathBuf};

    /// A fresh empty directory under the system temp folder, named for the test.
    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("cavitation-mnist-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes an IDX file: magic, then each dimension as big-endian u32, then the bytes.
    fn write_idx(path: &Path, magic: u32, dims: &[u32], bytes: &[u8]) {
        let mut out = magic.to_be_bytes().to_vec();
        for d in dims {
            out.extend(d.to_be_bytes());
        }
        out.extend(bytes);
        std::fs::write(path, out).unwrap();
    }

    /// Writes the four files with `train` and `test` images of 2 by 2 pixels.
    fn write_set(dir: &Path, train: usize, test: usize) {
        for (prefix, n) in [("train", train), ("t10k", test)] {
            let pixels: Vec<u8> = (0..n * 784).map(|i| (i * 51 % 256) as u8).collect();
            let labels: Vec<u8> = (0..n).map(|i| (i % 10) as u8).collect();
            write_idx(
                &dir.join(format!("{prefix}-images-idx3-ubyte")),
                2051,
                &[n as u32, 28, 28],
                &pixels,
            );
            write_idx(
                &dir.join(format!("{prefix}-labels-idx1-ubyte")),
                2049,
                &[n as u32],
                &labels,
            );
        }
    }

    /// A missing directory is an `Io` error naming the file, and its message points to the
    /// download script.
    #[test]
    fn mnist_missing_dir_error_mentions_script() {
        let err = load(Path::new("/nonexistent-cavitation-dir/mnist")).unwrap_err();
        assert!(matches!(err, Error::Io { .. }), "{err:?}");
        let text = err.to_string();
        assert!(text.contains("train-images-idx3-ubyte"), "{text}");
        assert!(text.contains("scripts/data/get-mnist.sh"), "{text}");
    }

    /// Pixels are divided by 255, labels become classes, train and test stay separate.
    #[test]
    fn mnist_parses_idx_files() {
        let dir = temp_dir("parse");
        write_set(&dir, 3, 2);
        let (train, test) = load(&dir).unwrap();
        assert_eq!((train.len(), test.len()), (3, 2));
        assert_eq!(train.x.shape().dims(), &[3, 784]);
        assert_eq!(
            &train.x.data()[..5],
            &[
                0.0,
                51.0 / 255.0,
                102.0 / 255.0,
                153.0 / 255.0,
                204.0 / 255.0
            ]
        );
        let Targets::Classes(labels) = &test.targets else {
            panic!("MNIST targets must be class labels");
        };
        assert_eq!(labels, &[0, 1]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A wrong magic number, a truncated file and mismatched counts are `Format` errors.
    #[test]
    fn mnist_rejects_bad_files() {
        let dir = temp_dir("bad");
        write_set(&dir, 3, 2);
        write_idx(&dir.join("train-labels-idx1-ubyte"), 2051, &[3], &[0, 1, 2]);
        assert!(matches!(load(&dir), Err(Error::Format(_))));

        write_set(&dir, 3, 2);
        write_idx(
            &dir.join("t10k-images-idx3-ubyte"),
            2051,
            &[2, 28, 28],
            &[0; 2 * 784 - 1],
        );
        assert!(matches!(load(&dir), Err(Error::Format(_))));

        write_set(&dir, 3, 2);
        write_idx(&dir.join("train-labels-idx1-ubyte"), 2049, &[2], &[0, 1]);
        assert!(matches!(load(&dir), Err(Error::Format(_))));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A header whose dimensions overflow `usize` is a `Format` error naming the dimensions,
    /// not a panic (debug) or a wrapped length (release).
    #[test]
    fn overflowing_dimensions_are_a_format_error() {
        let dir = temp_dir("overflow");
        write_set(&dir, 3, 2);
        write_idx(
            &dir.join("train-images-idx3-ubyte"),
            2051,
            &[u32::MAX, u32::MAX, u32::MAX],
            &[0; 16],
        );
        let err = load(&dir).unwrap_err();
        assert!(matches!(err, Error::Format(_)), "{err:?}");
        let text = err.to_string();
        assert!(text.contains("overflow"), "{text}");
        assert!(text.contains("train-images-idx3-ubyte"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A header of 0 images with rows and columns of `u32::MAX`, which the length check
    /// does not bound, and images of another size are `Format` errors naming the file and
    /// the size.
    #[test]
    fn images_must_be_28_by_28() {
        let dir = temp_dir("side");
        for (dims, bytes) in [
            ([0, u32::MAX, u32::MAX], 0usize),
            ([3, 2, 2], 12),
            ([3, 28, 27], 3 * 28 * 27),
        ] {
            write_set(&dir, 3, 2);
            write_idx(
                &dir.join("train-images-idx3-ubyte"),
                2051,
                &dims,
                &vec![0; bytes],
            );
            if dims[0] == 0 {
                write_idx(&dir.join("train-labels-idx1-ubyte"), 2049, &[0], &[]);
            }
            let err = load(&dir).unwrap_err();
            assert!(matches!(err, Error::Format(_)), "{dims:?}: {err:?}");
            let text = err.to_string();
            assert!(text.contains("train-images-idx3-ubyte"), "{text}");
            assert!(
                text.contains(&format!("images are {} by {}", dims[1], dims[2])),
                "{text}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// If the real MNIST files are present, they load with the standard sizes.
    #[test]
    fn mnist_real_files_if_present() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/mnist");
        if !dir.join("train-images-idx3-ubyte").exists() {
            return;
        }
        let (train, test) = load(&dir).unwrap();
        assert_eq!((train.len(), test.len()), (60_000, 10_000));
        assert_eq!(train.features(), 784);
    }
}
