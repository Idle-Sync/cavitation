//! The scikit-learn digits data set: 1,797 8 by 8 images, split into train and test, and
//! the known result's network and training recipe on it, which the known-result tests, the
//! shrink example and the shrink benchmark share.

use std::path::PathBuf;

use cavitation_core::{Error, ParamStore, Rng, Tensor};
use cavitation_nn::{
    Dataset, FitConfig, Health, Init, Linear, OptimizerConfig, Relu, Sequential, Targets,
};

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

/// The shuffled order of `n` source rows, shared by every split in this crate: Fisher-Yates
/// from the end, with `k = floor(r * (i + 1))` for each `r` from
/// [`Mulberry32::new(42)`](Mulberry32).
pub(crate) fn split_order(n: usize) -> Vec<usize> {
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
/// The split can be reproduced outside this repository. Start from the rows of `digits.csv` in
/// file order, numbered 0 to 1,796. Draw numbers `r` in `[0, 1)` from mulberry32 with seed 42
/// (see [`Mulberry32`]). Then for `i` from 1,796 down to 1, let `k = floor(r · (i + 1))`, with
/// a fresh `r` each time, and swap entries `i` and `k` of the order. The first 1,437 entries
/// of the order are the training rows, in that order, and the other 360 are the test rows.
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

/// The training rows the shrink runs train on: the first 1,149 of the 1,437 training digits.
/// The other 288 are held out as their validation set.
///
/// ```
/// let (train, _) = cavitation_datasets::digits::load().unwrap();
/// let (fit_rows, held_out) = train.split_at(cavitation_datasets::digits::FIT_ROWS);
/// assert_eq!((fit_rows.len(), held_out.len()), (1149, 288));
/// ```
pub const FIT_ROWS: usize = 1149;

/// The known result's network for the digits, `64 → hidden[0] → hidden[1] → 10` with ReLU
/// between the `Linear`s, He uniform weights and zero biases drawn from `Rng::seed(seed)` in
/// layer order. The known result is `network(params, [256, 128], 1)`.
///
/// ```
/// use cavitation_core::ParamStore;
/// let mut params = ParamStore::new();
/// let net = cavitation_datasets::digits::network(&mut params, [256, 128], 1);
/// let count: usize = net
///     .parameter_names()
///     .iter()
///     .map(|(_, id)| params.value(*id).data().len())
///     .sum();
/// assert_eq!((net.len(), count), (5, 50_826));
/// ```
pub fn network(params: &mut ParamStore, hidden: [usize; 2], seed: u64) -> Sequential {
    let mut rng = Rng::seed(seed);
    let mut he = |inputs, outputs| {
        Linear::with_init(
            params,
            inputs,
            outputs,
            Init::HeUniform,
            Init::Zeros,
            &mut rng,
        )
    };
    Sequential::new()
        .push(he(PIXELS, hidden[0]))
        .push(Relu)
        .push(he(hidden[0], hidden[1]))
        .push(Relu)
        .push(he(hidden[1], 10))
}

/// The known result's training recipe for `epochs` epochs (20 in the known result): Adam at
/// learning rate 0.002, batch 32, cross-entropy, shuffle seed 7, a constant rate, quiet
/// health checks, no callback and no validation set.
///
/// ```
/// let config = cavitation_datasets::digits::recipe(20);
/// assert_eq!((config.epochs, config.batch, config.shuffle_seed), (20, 32, Some(7)));
/// ```
pub fn recipe(epochs: usize) -> FitConfig<'static> {
    FitConfig {
        epochs,
        batch: 32,
        optimizer: OptimizerConfig::Adam {
            lr: 0.002,
            weight_decay: 0.0,
        },
        shuffle_seed: Some(7),
        health: Health::quiet(),
        ..FitConfig::default()
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the digits loader: the split, the counts and the data file itself.

    use super::*;
    use crate::sha256::sha256_hex;
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
}
