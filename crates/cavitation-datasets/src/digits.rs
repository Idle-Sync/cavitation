//! The scikit-learn digits data set: 1,797 8 by 8 images, split into train and test.

use std::path::PathBuf;

use cavitation_core::{Error, Tensor};
use cavitation_nn::{Dataset, Targets};

use crate::Mulberry32;

/// The number of rows in `digits.csv`.
const ROWS: usize = 1797;
/// The pixels per image (8 by 8).
const PIXELS: usize = 64;
/// How many shuffled rows go to the train set; the rest are the test set.
const TRAIN_ROWS: usize = 1437;

/// Where `digits.csv` lives in the repository checkout.
fn csv_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/digits/digits.csv")
}

/// The shuffled order of the source rows: Fisher-Yates from the end, with
/// `k = floor(r * (i + 1))` for each `r` from [`Mulberry32::new(42)`](Mulberry32).
fn split_order(n: usize) -> Vec<usize> {
    let mut rng = Mulberry32::new(42);
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let k = (rng.next_f64() * (i + 1) as f64) as usize;
        order.swap(i, k);
    }
    order
}

/// Loads the digits data set from `data/digits/digits.csv`, shuffled with
/// [`Mulberry32`] (seed 42) and split into 1,437 training rows and 360 test rows.
///
/// Each row has 64 features, the 8 by 8 pixel counts divided by 16 (so in 0 to 1), and a
/// class label 0 to 9 ([`Targets::Classes`]). The file is found relative to this crate's
/// source directory, so the function works from a checkout of the repository.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be read and [`Error::Format`] if a row is not
/// 64 pixel values and a label.
///
/// # Examples
///
/// ```
/// let (train, test) = cavitation_datasets::digits::load().unwrap();
/// assert_eq!((train.len(), test.len()), (1437, 360));
/// assert_eq!(train.features(), 64);
/// ```
pub fn load() -> Result<(Dataset, Dataset), Error> {
    let path = csv_path();
    let text = std::fs::read_to_string(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    let mut pixels: Vec<f32> = Vec::with_capacity(ROWS * PIXELS);
    let mut labels: Vec<usize> = Vec::with_capacity(ROWS);
    for (line_no, line) in text.lines().enumerate() {
        let bad =
            |what: &str| Error::Format(format!("{}, line {}: {what}", path.display(), line_no + 1));
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != PIXELS + 1 {
            return Err(bad(&format!(
                "expected {} fields, found {}",
                PIXELS + 1,
                fields.len()
            )));
        }
        for field in &fields[..PIXELS] {
            let value: f32 = field
                .trim()
                .parse()
                .map_err(|_| bad("pixel is not a number"))?;
            pixels.push(value / 16.0);
        }
        let label: usize = fields[PIXELS]
            .trim()
            .parse()
            .map_err(|_| bad("label is not a digit"))?;
        if label > 9 {
            return Err(bad("label is not a digit"));
        }
        labels.push(label);
    }
    if labels.len() != ROWS {
        return Err(Error::Format(format!(
            "{}: expected {ROWS} rows, found {}",
            path.display(),
            labels.len()
        )));
    }
    let order = split_order(ROWS);
    let build = |rows: &[usize]| -> Result<Dataset, Error> {
        let mut x = Vec::with_capacity(rows.len() * PIXELS);
        let mut y = Vec::with_capacity(rows.len());
        for &i in rows {
            x.extend_from_slice(&pixels[i * PIXELS..(i + 1) * PIXELS]);
            y.push(labels[i]);
        }
        Ok(Dataset {
            x: Tensor::from_vec(x, &[rows.len(), PIXELS])?,
            targets: Targets::Classes(y),
        })
    };
    Ok((build(&order[..TRAIN_ROWS])?, build(&order[TRAIN_ROWS..])?))
}

#[cfg(test)]
mod tests {
    //! Tests for the digits loader: the split, the counts and the data file itself.

    use super::*;
    use cavitation_nn::Targets;

    /// Reads the CSV independently of the loader: `(pixels, label)` per source row.
    fn source_rows() -> Vec<(Vec<f32>, usize)> {
        let text = std::fs::read_to_string(csv_path()).unwrap();
        text.lines()
            .map(|line| {
                let v: Vec<f32> = line.split(',').map(|f| f.parse().unwrap()).collect();
                (v[..64].iter().map(|p| p / 16.0).collect(), v[64] as usize)
            })
            .collect()
    }

    /// The shuffle reproduces the reference split: the first rows of each side come from
    /// the pinned source rows, and the pinned indices sum to the pinned total.
    #[test]
    fn digits_split_matches_reference() {
        let order = split_order(1797);
        assert_eq!(order[..5], [1784, 96, 434, 1205, 1703]);
        assert_eq!(order[1437..1442], [1231, 388, 1595, 389, 1767]);
        assert_eq!(order[1796], 1080);
        assert_eq!(order[..1437].iter().sum::<usize>(), 1_292_761);

        let source = source_rows();
        let (train, test) = load().unwrap();
        for (set, offset, count) in [(&train, 0, 1437), (&test, 1437, 360)] {
            let Targets::Classes(labels) = &set.targets else {
                panic!("digits targets must be class labels");
            };
            for row in 0..count {
                let (pixels, label) = &source[order[offset + row]];
                assert_eq!(&set.x.data()[row * 64..(row + 1) * 64], &pixels[..]);
                assert_eq!(labels[row], *label);
            }
        }
    }

    /// 1,437 train and 360 test rows of 64 features in [0, 1], labels 0 to 9 all present.
    #[test]
    fn digits_counts() {
        let (train, test) = load().unwrap();
        assert_eq!((train.len(), test.len()), (1437, 360));
        assert_eq!((train.features(), test.features()), (64, 64));
        assert_eq!(train.x.shape().dims(), &[1437, 64]);
        for set in [&train, &test] {
            assert!(set.x.data().iter().all(|p| (0.0..=1.0).contains(p)));
            let Targets::Classes(labels) = &set.targets else {
                panic!("digits targets must be class labels");
            };
            assert_eq!(labels.len(), set.len());
            for digit in 0..10 {
                assert!(labels.contains(&digit), "digit {digit} missing");
            }
            assert!(labels.iter().all(|&l| l < 10));
        }
    }

    /// The committed data file is the one the split indices were derived from.
    #[test]
    fn digits_csv_checksum() {
        let bytes = std::fs::read(csv_path()).unwrap();
        assert_eq!(
            sha256_hex(&bytes),
            "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8"
        );
    }

    /// SHA-256 (FIPS 180-4) of `data` as lower-case hex.
    fn sha256_hex(data: &[u8]) -> String {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut msg = data.to_vec();
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend(((data.len() as u64) * 8).to_be_bytes());
        for chunk in msg.chunks(64) {
            let mut w = [0u32; 64];
            for (i, word) in chunk.chunks(4).enumerate() {
                w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut v = h;
            for i in 0..64 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let t1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                let t2 = s0.wrapping_add(maj);
                v = [
                    t1.wrapping_add(t2),
                    v[0],
                    v[1],
                    v[2],
                    v[3].wrapping_add(t1),
                    v[4],
                    v[5],
                    v[6],
                ];
            }
            for i in 0..8 {
                h[i] = h[i].wrapping_add(v[i]);
            }
        }
        h.iter().map(|x| format!("{x:08x}")).collect()
    }

    /// The hash helper agrees with the published SHA-256 of "abc".
    #[test]
    fn sha256_known_value() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
