//! Data for training: datasets, targets, batches and the [`Batcher`] that walks a data set
//! in batches.

use std::ops::Range;

use cavitation_core::{Real, Rng, Tensor, TensorView};

/// What each row of a [`Dataset`] should be predicted as.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_nn::Targets;
/// let classes: Targets = Targets::Classes(vec![0, 2, 1]);
/// let values = Targets::Values(Tensor::zeros(&[3, 2]));
/// assert!(matches!(classes, Targets::Classes(_)));
/// assert!(matches!(values, Targets::Values(_)));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Targets<T: Real = f32> {
    /// One class label per row, for cross-entropy.
    Classes(Vec<usize>),
    /// A `[rows, outputs]` tensor of target values, for example for mean squared error.
    Values(Tensor<T>),
}

/// Inputs and their targets.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_nn::{Dataset, Targets};
/// let data = Dataset {
///     x: Tensor::zeros(&[4, 3]),
///     targets: Targets::Classes(vec![0, 1, 0, 1]),
/// };
/// assert_eq!((data.len(), data.features()), (4, 3));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset<T: Real = f32> {
    /// The inputs, a `[rows, features]` tensor.
    pub x: Tensor<T>,
    /// The targets, one per row of `x`.
    pub targets: Targets<T>,
}

impl<T: Real> Dataset<T> {
    /// The number of rows.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not a matrix.
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::{Dataset, Targets};
    /// let data = Dataset { x: Tensor::zeros(&[5, 2]), targets: Targets::Classes(vec![0; 5]) };
    /// assert_eq!(data.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.matrix_dims().0
    }

    /// Whether there are no rows.
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::{Dataset, Targets};
    /// let data = Dataset { x: Tensor::zeros(&[0, 2]), targets: Targets::Classes(vec![]) };
    /// assert!(data.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of input features per row.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not a matrix.
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::{Dataset, Targets};
    /// let data = Dataset { x: Tensor::zeros(&[5, 2]), targets: Targets::Classes(vec![0; 5]) };
    /// assert_eq!(data.features(), 2);
    /// ```
    pub fn features(&self) -> usize {
        self.matrix_dims().1
    }

    /// Splits the data at row `n`: a copy of rows `0..n` and a copy of rows `n..len`, with
    /// their targets. A held-out set is the second part of a shuffled data set, or of one
    /// whose rows are already in no particular order.
    ///
    /// # Panics
    ///
    /// Panics if `n` is above [`len`](Self::len), and as [`subset`](Self::subset) does.
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::{Dataset, Targets};
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], &[5, 1]).unwrap(),
    ///     targets: Targets::Classes(vec![0, 1, 0, 1, 0]),
    /// };
    /// let (train, test) = data.split_at(3);
    /// assert_eq!((train.len(), test.len()), (3, 2));
    /// assert_eq!(test.x.data(), &[3.0, 4.0]);
    /// assert_eq!(test.targets, Targets::Classes(vec![1, 0]));
    /// ```
    pub fn split_at(&self, n: usize) -> (Dataset<T>, Dataset<T>) {
        let rows = self.len();
        assert!(
            n <= rows,
            "Dataset::split_at: cannot split {rows} rows at row {n}"
        );
        (self.subset(0..n), self.subset(n..rows))
    }

    /// A copy of the rows `rows` (start included, end excluded), with their targets.
    ///
    /// # Panics
    ///
    /// Panics, naming the range and the row count, if `rows` starts after it ends or ends
    /// past [`len`](Self::len); panics if `x` or value targets are not matrices, or if the
    /// targets do not have one entry per row.
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::{Dataset, Targets};
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[3, 2]).unwrap(),
    ///     targets: Targets::Values(Tensor::from_vec(vec![10.0f32, 20.0, 30.0], &[3, 1]).unwrap()),
    /// };
    /// let middle = data.subset(1..2);
    /// assert_eq!(middle.x.data(), &[2.0, 3.0]);
    /// assert_eq!(middle.targets, Targets::Values(Tensor::from_vec(vec![20.0f32], &[1, 1]).unwrap()));
    /// ```
    pub fn subset(&self, rows: Range<usize>) -> Dataset<T> {
        let (n, features) = self.matrix_dims();
        assert!(
            rows.start <= rows.end && rows.end <= n,
            "Dataset::subset: rows {}..{} are outside the {n} rows",
            rows.start,
            rows.end
        );
        let count = rows.len();
        let x = Tensor::from_vec(
            self.x.data()[rows.start * features..rows.end * features].to_vec(),
            &[count, features],
        )
        .expect("whole rows of x have the subset's shape");
        let targets = match &self.targets {
            Targets::Classes(labels) => {
                assert_eq!(
                    labels.len(),
                    n,
                    "Dataset::subset: there are {n} rows but {} labels",
                    labels.len()
                );
                Targets::Classes(labels[rows].to_vec())
            }
            Targets::Values(t) => match t.shape().dims() {
                &[r, outputs] if r == n => Targets::Values(
                    Tensor::from_vec(
                        t.data()[rows.start * outputs..rows.end * outputs].to_vec(),
                        &[count, outputs],
                    )
                    .expect("whole rows of the targets have the subset's shape"),
                ),
                dims => panic!(
                    "Dataset::subset: the value targets are {dims:?}, but the inputs are \
                     [{n}, {features}]"
                ),
            },
        };
        Dataset { x, targets }
    }

    /// The `(rows, features)` of `x`, panicking if it is not a matrix.
    fn matrix_dims(&self) -> (usize, usize) {
        match self.x.shape().dims() {
            &[rows, features] => (rows, features),
            dims => panic!("Dataset: x must have shape [rows, features], found {dims:?}"),
        }
    }
}

/// A batch's worth of targets, borrowed from a [`Batch`].
///
/// # Examples
///
/// ```
/// use cavitation_nn::TargetsView;
/// let view: TargetsView = TargetsView::Classes(&[3, 1]);
/// assert!(matches!(view, TargetsView::Classes(&[3, 1])));
/// ```
#[derive(Clone, Copy, Debug)]
pub enum TargetsView<'a, T: Real = f32> {
    /// One class label per row.
    Classes(&'a [usize]),
    /// A `[rows, outputs]` view of target values.
    Values(TensorView<'a, T>),
}

/// The rows of one batch, borrowed from a [`Batch`].
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_nn::{Batch, Batcher, Dataset, Targets};
/// let data = Dataset { x: Tensor::zeros(&[5, 2]), targets: Targets::Classes(vec![0; 5]) };
/// let mut out = Batch::new(&data, 4);
/// let view = Batcher::new(5, 4, None).gather(&data, 4..5, &mut out);
/// assert_eq!(view.x.shape().dims(), &[1, 2]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BatchView<'a, T: Real = f32> {
    /// The inputs, `[rows, features]`.
    pub x: TensorView<'a, T>,
    /// The targets for those rows.
    pub targets: TargetsView<'a, T>,
}

/// Owned storage for the target part of a [`Batch`].
#[derive(Clone, Debug)]
enum BatchTargets<T: Real> {
    /// Room for one label per row.
    Classes(Vec<usize>),
    /// Room for a `[batch, outputs]` tensor.
    Values(Tensor<T>),
}

/// Buffers for one batch at full size, filled by [`Batcher::gather`] and reused for every
/// batch, so the smaller last batch needs no new allocation.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_nn::{Batch, Dataset, Targets};
/// let data = Dataset { x: Tensor::zeros(&[10, 3]), targets: Targets::Classes(vec![0; 10]) };
/// let _batch = Batch::new(&data, 4);
/// ```
#[derive(Clone, Debug)]
pub struct Batch<T: Real = f32> {
    /// Room for `batch` rows of inputs, `[batch, features]`.
    x: Tensor<T>,
    /// Room for the same rows' targets.
    targets: BatchTargets<T>,
}

impl<T: Real> Batch<T> {
    /// Allocates buffers for `batch` rows shaped like the rows of `data`.
    ///
    /// # Panics
    ///
    /// Panics if `batch` is 0, if `data.x` is not a matrix, or if value targets are not a
    /// matrix.
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::{Batch, Dataset, Targets};
    /// let data = Dataset {
    ///     x: Tensor::zeros(&[10, 3]),
    ///     targets: Targets::Values(Tensor::zeros(&[10, 2])),
    /// };
    /// let _batch = Batch::new(&data, 4);
    /// ```
    pub fn new(data: &Dataset<T>, batch: usize) -> Batch<T> {
        assert!(batch > 0, "Batch::new: the batch size must be at least 1");
        let targets = match &data.targets {
            Targets::Classes(_) => BatchTargets::Classes(vec![0; batch]),
            Targets::Values(t) => match t.shape().dims() {
                &[_, outputs] => BatchTargets::Values(Tensor::zeros_generic(&[batch, outputs])),
                dims => panic!(
                    "Batch::new: value targets must have shape [rows, outputs], found {dims:?}"
                ),
            },
        };
        Batch {
            x: Tensor::zeros_generic(&[batch, data.features()]),
            targets,
        }
    }
}

/// Walks a data set in batches, reshuffling each epoch.
///
/// The order is a permutation of the row indices. [`epoch`](Batcher::epoch) reshuffles it
/// (when the batcher has a seed), [`batches`](Batcher::batches) lists the batch ranges
/// over it and [`gather`](Batcher::gather) copies one batch's rows into a [`Batch`]. None of
/// these allocates.
///
/// # Examples
///
/// ```
/// use cavitation_nn::Batcher;
/// let mut batcher = Batcher::new(10, 4, Some(1));
/// batcher.epoch();
/// let sizes: Vec<usize> = batcher.batches().map(|r| r.len()).collect();
/// assert_eq!(sizes, [4, 4, 2]);
/// ```
#[derive(Clone, Debug)]
pub struct Batcher {
    /// The batch size; the last batch of an epoch may be smaller.
    batch: usize,
    /// The current order: `order[position]` is the row index at that position.
    order: Vec<usize>,
    /// The shuffling generator; `None` keeps the original order.
    rng: Option<Rng>,
}

impl Batcher {
    /// A batcher over `n` rows in batches of `batch`. With `Some(seed)` each
    /// [`epoch`](Batcher::epoch) shuffles with [`Rng::shuffle`]; with `None` the order stays
    /// `0..n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` or `batch` is 0.
    ///
    /// ```
    /// use cavitation_nn::Batcher;
    /// let batcher = Batcher::new(1437, 128, Some(7));
    /// assert_eq!(batcher.batches().count(), 12);
    /// ```
    pub fn new(n: usize, batch: usize, shuffle_seed: Option<u64>) -> Batcher {
        assert!(n > 0, "Batcher::new: there must be at least one row");
        assert!(batch > 0, "Batcher::new: the batch size must be at least 1");
        Batcher {
            batch,
            order: (0..n).collect(),
            rng: shuffle_seed.map(Rng::seed),
        }
    }

    /// Starts an epoch: reshuffles the order if the batcher has a seed, otherwise does
    /// nothing. Call it before each pass over [`batches`](Batcher::batches).
    ///
    /// ```
    /// use cavitation_nn::Batcher;
    /// let mut batcher = Batcher::new(5, 2, Some(3));
    /// batcher.epoch();
    /// let mut order = batcher.order().to_vec();
    /// order.sort();
    /// assert_eq!(order, [0, 1, 2, 3, 4]);
    /// ```
    pub fn epoch(&mut self) {
        if let Some(rng) = &mut self.rng {
            rng.shuffle(&mut self.order);
        }
    }

    /// The row index at each position of the current order.
    ///
    /// ```
    /// use cavitation_nn::Batcher;
    /// assert_eq!(Batcher::new(3, 2, None).order(), [0, 1, 2]);
    /// ```
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// The position ranges of one epoch's batches: full batches, then a smaller one if
    /// the row count is not a multiple of the batch size.
    ///
    /// ```
    /// use cavitation_nn::Batcher;
    /// let ranges: Vec<_> = Batcher::new(10, 4, None).batches().collect();
    /// assert_eq!(ranges, [0..4, 4..8, 8..10]);
    /// ```
    pub fn batches(&self) -> impl Iterator<Item = Range<usize>> + use<> {
        let (n, batch) = (self.order.len(), self.batch);
        (0..n.div_ceil(batch)).map(move |i| i * batch..((i + 1) * batch).min(n))
    }

    /// Copies the rows at the order positions `range` (one item of
    /// [`batches`](Batcher::batches)) into `out` and returns views of the filled part.
    ///
    /// # Panics
    ///
    /// Panics, naming the values, if `range` reaches past the rows, if `data` does not have
    /// the batcher's row count, if `range` is longer than the batch size `out` was made for,
    /// or if `out` was made for a different kind of data (feature count, target kind or
    /// output count).
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::{Batch, Batcher, Dataset, Targets};
    /// let data = Dataset {
    ///     x: Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3, 1]).unwrap(),
    ///     targets: Targets::Classes(vec![0, 1, 0]),
    /// };
    /// let batcher = Batcher::new(3, 2, None);
    /// let mut out = Batch::new(&data, 2);
    /// let view = batcher.gather(&data, 2..3, &mut out);
    /// assert_eq!(view.x.data(), &[3.0]);
    /// ```
    pub fn gather<'b, T: Real>(
        &self,
        data: &Dataset<T>,
        range: Range<usize>,
        out: &'b mut Batch<T>,
    ) -> BatchView<'b, T> {
        let n = self.order.len();
        assert!(
            range.start <= range.end && range.end <= n,
            "Batcher::gather: range {}..{} is outside the {n} rows",
            range.start,
            range.end
        );
        assert_eq!(
            data.len(),
            n,
            "Batcher::gather: the data has {} rows, the batcher {n}",
            data.len()
        );
        let rows = range.len();
        let features = data.features();
        let capacity = out.x.shape().dims()[0];
        assert!(
            rows <= capacity && out.x.shape().dims()[1] == features,
            "Batcher::gather: {rows} rows of {features} features do not fit a batch of {:?}",
            out.x.shape().dims()
        );
        let picked = &self.order[range];
        for (k, &i) in picked.iter().enumerate() {
            out.x.data_mut()[k * features..(k + 1) * features]
                .copy_from_slice(&data.x.data()[i * features..(i + 1) * features]);
        }
        let x = TensorView::from_slice(&out.x.data()[..rows * features], &[rows, features])
            .expect("a prefix of the batch buffer has the gathered shape");
        let targets = match (&data.targets, &mut out.targets) {
            (Targets::Classes(src), BatchTargets::Classes(dst)) => {
                for (k, &i) in picked.iter().enumerate() {
                    dst[k] = src[i];
                }
                TargetsView::Classes(&dst[..rows])
            }
            (Targets::Values(src), BatchTargets::Values(dst)) => {
                let outputs = src.shape().dims()[1];
                assert_eq!(
                    dst.shape().dims()[1],
                    outputs,
                    "Batcher::gather: the batch has {} outputs, the data {outputs}",
                    dst.shape().dims()[1]
                );
                for (k, &i) in picked.iter().enumerate() {
                    dst.data_mut()[k * outputs..(k + 1) * outputs]
                        .copy_from_slice(&src.data()[i * outputs..(i + 1) * outputs]);
                }
                TargetsView::Values(
                    TensorView::from_slice(&dst.data()[..rows * outputs], &[rows, outputs])
                        .expect("a prefix of the target buffer has the gathered shape"),
                )
            }
            _ => panic!("Batcher::gather: the batch was made for a different kind of targets"),
        };
        BatchView { x, targets }
    }
}

#[cfg(test)]
pub(super) mod tests {
    //! Tests for the batcher and for gathering rows into a batch.

    use super::*;

    /// A dataset of `n` rows and 3 features where row `i` is `[i, i + 0.25, i + 0.5]` and its
    /// class is `i % 10`.
    pub(in crate::train) fn classes_dataset(n: usize) -> Dataset<f32> {
        let mut x = Vec::new();
        for i in 0..n {
            x.extend([i as f32, i as f32 + 0.25, i as f32 + 0.5]);
        }
        Dataset {
            x: Tensor::from_vec(x, &[n, 3]).unwrap(),
            targets: Targets::Classes((0..n).map(|i| i % 10).collect()),
        }
    }

    /// With n = 1,437 and batch 128 an epoch has 12 batches, the last of 29 rows, and every
    /// index appears exactly once, shuffled or not.
    #[test]
    fn batcher_covers_every_index_once_per_epoch() {
        for seed in [None, Some(7)] {
            let mut batcher = Batcher::new(1437, 128, seed);
            for _ in 0..3 {
                batcher.epoch();
                let ranges: Vec<_> = batcher.batches().collect();
                assert_eq!(ranges.len(), 12);
                assert_eq!(ranges[11].len(), 29);
                assert!(ranges[..11].iter().all(|r| r.len() == 128));
                let mut seen = vec![0u32; 1437];
                for r in ranges {
                    for &i in &batcher.order()[r] {
                        seen[i] += 1;
                    }
                }
                assert!(seen.iter().all(|&c| c == 1));
            }
        }
    }

    /// The same seed gives the same orders epoch after epoch; another seed gives another
    /// order; the order really is shuffled.
    #[test]
    fn batcher_shuffle_deterministic() {
        let mut a = Batcher::new(200, 32, Some(1));
        let mut b = Batcher::new(200, 32, Some(1));
        let mut c = Batcher::new(200, 32, Some(2));
        let identity: Vec<usize> = (0..200).collect();
        let mut previous = identity.clone();
        for _ in 0..3 {
            a.epoch();
            b.epoch();
            c.epoch();
            assert_eq!(a.order(), b.order());
            assert_ne!(a.order(), c.order());
            assert_ne!(a.order(), identity);
            assert_ne!(a.order(), previous);
            previous = a.order().to_vec();
        }
    }

    /// Without a seed `epoch` leaves the original order.
    #[test]
    fn batcher_without_seed_keeps_order() {
        let mut batcher = Batcher::new(10, 4, None);
        let identity: Vec<usize> = (0..10).collect();
        for _ in 0..3 {
            batcher.epoch();
            assert_eq!(batcher.order(), identity);
        }
        let ranges: Vec<_> = batcher.batches().collect();
        assert_eq!(ranges, [0..4, 4..8, 8..10]);
    }

    /// Gathering copies the rows named by the order and returns views of the filled prefix,
    /// so the smaller last batch reuses the full-size buffer.
    #[test]
    fn gather_copies_rows_and_returns_prefix_views() {
        let data = classes_dataset(10);
        let mut batcher = Batcher::new(10, 4, Some(3));
        batcher.epoch();
        let mut out = Batch::new(&data, 4);
        for range in batcher.batches() {
            let rows = batcher.order()[range.clone()].to_vec();
            let view = batcher.gather(&data, range, &mut out);
            assert_eq!(view.x.shape().dims(), &[rows.len(), 3]);
            for (k, &i) in rows.iter().enumerate() {
                assert_eq!(
                    &view.x.data()[k * 3..k * 3 + 3],
                    &data.x.data()[i * 3..i * 3 + 3]
                );
            }
            match view.targets {
                TargetsView::Classes(c) => {
                    assert_eq!(c, rows.iter().map(|i| i % 10).collect::<Vec<_>>());
                }
                TargetsView::Values(_) => panic!("expected class targets"),
            }
        }
    }

    /// Value targets are gathered row by row like the inputs.
    #[test]
    fn gather_copies_value_targets() {
        let data = Dataset {
            x: Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(vec![10.0f32, 20.0, 30.0], &[3, 1]).unwrap()),
        };
        assert_eq!((data.len(), data.features()), (3, 2));
        let batcher = Batcher::new(3, 2, None);
        let mut out = Batch::new(&data, 2);
        let view = batcher.gather(&data, 1..3, &mut out);
        assert_eq!(view.x.data(), &[3.0, 4.0, 5.0, 6.0]);
        match view.targets {
            TargetsView::Values(t) => {
                assert_eq!(t.shape().dims(), &[2, 1]);
                assert_eq!(t.data(), &[20.0, 30.0]);
            }
            TargetsView::Classes(_) => panic!("expected value targets"),
        }
    }

    /// `split_at` and `subset` copy whole rows with their targets, class or value; splitting
    /// at 0 or at the row count gives an empty part.
    #[test]
    fn split_at_and_subset() {
        let data = classes_dataset(10);
        let (a, b) = data.split_at(7);
        assert_eq!((a.len(), b.len()), (7, 3));
        assert_eq!(&b.x.data()[..3], &[7.0, 7.25, 7.5]);
        assert_eq!(b.targets, Targets::Classes(vec![7, 8, 9]));
        assert_eq!(a, data.subset(0..7));
        let (empty, all) = data.split_at(0);
        assert!(empty.is_empty() && all == data);
        assert!(data.split_at(10).1.is_empty());
        let values = Dataset {
            x: Tensor::from_vec((0..12).map(|i| i as f32).collect(), &[4, 3]).unwrap(),
            targets: Targets::Values(
                Tensor::from_vec((0..8).map(|i| i as f32 * 10.0).collect(), &[4, 2]).unwrap(),
            ),
        };
        let middle = values.subset(1..3);
        assert_eq!(middle.x.shape().dims(), &[2, 3]);
        assert_eq!(middle.x.data(), &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            middle.targets,
            Targets::Values(Tensor::from_vec(vec![20.0, 30.0, 40.0, 50.0], &[2, 2]).unwrap())
        );
    }

    /// Splitting past the row count panics naming both.
    #[test]
    #[should_panic(expected = "cannot split 10 rows at row 11")]
    fn split_at_past_the_end_panics() {
        let _ = classes_dataset(10).split_at(11);
    }

    /// A subset reaching past the rows panics naming the range.
    #[test]
    #[should_panic(expected = "rows 8..12 are outside the 10 rows")]
    fn subset_past_the_end_panics() {
        let _ = classes_dataset(10).subset(8..12);
    }

    /// A range past the end of the data panics with a message naming it.
    #[test]
    #[should_panic(expected = "range 8..12")]
    fn gather_rejects_range_past_the_end() {
        let data = classes_dataset(10);
        let batcher = Batcher::new(10, 4, None);
        let mut out = Batch::new(&data, 4);
        let _ = batcher.gather(&data, 8..12, &mut out);
    }
}
