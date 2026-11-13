//! The shipped Householder QR solver against its reference twin, and the solver's behaviour
//! on small problems worked by hand, rank-deficient and badly scaled matrices, and bad shapes.

use cavitation_core::{Error, Rng, Tensor};
use cavitation_ml::linalg::{Qr, lstsq, qr_reference};

/// A `rows` by `cols` tensor of standard normal draws from `rng`.
fn gaussian(rng: &mut Rng, rows: usize, cols: usize) -> Tensor<f64> {
    let data = (0..rows * cols).map(|_| rng.normal()).collect();
    Tensor::from_vec(data, &[rows, cols]).unwrap()
}

/// The `(rows, cols)` pairs of the twin tests: rows in {1, 2, 7, 64, 513}, cols in
/// {1, 3, 16, 65}, with `rows >= cols` and, where `cols > 1`, `rows >= 2 * cols`.
fn problem_sizes() -> Vec<(usize, usize)> {
    let mut sizes = Vec::new();
    for rows in [1, 2, 7, 64, 513] {
        for cols in [1, 3, 16, 65] {
            if rows >= cols && (cols == 1 || rows >= 2 * cols) {
                sizes.push((rows, cols));
            }
        }
    }
    sizes
}

/// The Frobenius norm of a slice of values.
fn norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The worked example: fitting `y = c0 + c1 * x` through (0, 1), (1, 2), (2, 4) gives
/// c0 = 5/6 and c1 = 3/2, by the normal equations [[3, 3], [3, 5]] c = [7, 10].
#[test]
fn qr_by_hand() {
    let a = Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0], &[3, 2]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3, 1]).unwrap();
    let expected = [0.8333333333333334, 1.5];
    let shipped = lstsq(a.view(), b.view()).unwrap();
    let reference = qr_reference::lstsq(a.view(), b.view()).unwrap();
    for x in [&shipped, &reference] {
        assert_eq!(x.shape().dims(), &[2, 1]);
        for (got, want) in x.data().iter().zip(expected) {
            assert!((got - want).abs() <= 1e-15, "{got} against {want}");
        }
    }
}

/// On seeded Gaussian problems every element of the shipped solution is within
/// `1e-10 · max(1, |reference|)` of the reference twin's.
#[test]
fn shipped_matches_reference() {
    let mut rng = Rng::seed(17);
    let mut worst = 0.0f64;
    for (rows, cols) in problem_sizes() {
        for k in [1, 3] {
            let a = gaussian(&mut rng, rows, cols);
            let b = gaussian(&mut rng, rows, k);
            let shipped = lstsq(a.view(), b.view()).unwrap();
            let reference = qr_reference::lstsq(a.view(), b.view()).unwrap();
            assert_eq!(shipped.shape().dims(), &[cols, k]);
            assert_eq!(reference.shape().dims(), &[cols, k]);
            for (s, r) in shipped.data().iter().zip(reference.data()) {
                let bound = 1e-10 * r.abs().max(1.0);
                assert!(
                    (s - r).abs() <= bound,
                    "rows {rows}, cols {cols}, k {k}: shipped {s}, reference {r}"
                );
                worst = worst.max((s - r).abs() / r.abs().max(1.0));
            }
        }
    }
    println!("largest |shipped - reference| / max(1, |reference|): {worst:e}");
}

/// The residual of the least-squares solution is orthogonal to every column of `a`:
/// `|aᵀ(b − a·x)| ≤ 1e-10 · ‖a‖·‖b‖` element by element, for the shipped solver and the twin.
#[test]
fn residual_is_orthogonal() {
    let mut rng = Rng::seed(3);
    for (rows, cols) in problem_sizes() {
        for k in [1, 3] {
            let a = gaussian(&mut rng, rows, cols);
            let b = gaussian(&mut rng, rows, k);
            let bound = 1e-10 * norm(a.data()) * norm(b.data());
            let shipped = lstsq(a.view(), b.view()).unwrap();
            let reference = qr_reference::lstsq(a.view(), b.view()).unwrap();
            for x in [&shipped, &reference] {
                let (ad, bd, xd) = (a.data(), b.data(), x.data());
                for q in 0..k {
                    let residual: Vec<f64> = (0..rows)
                        .map(|i| {
                            bd[i * k + q]
                                - (0..cols)
                                    .map(|j| ad[i * cols + j] * xd[j * k + q])
                                    .sum::<f64>()
                        })
                        .collect();
                    for j in 0..cols {
                        let dot: f64 = (0..rows).map(|i| ad[i * cols + j] * residual[i]).sum();
                        assert!(
                            dot.abs() <= bound,
                            "rows {rows}, cols {cols}, column {j}: aᵀr = {dot}, bound {bound}"
                        );
                    }
                }
            }
        }
    }
}

/// Column 2 is column 0 plus column 1, so `lstsq` refuses the matrix and names column 2,
/// and both factorisations report it as the first dependent column.
#[test]
fn rank_deficient_names_the_column() {
    let rows = 6;
    let mut data = Vec::new();
    for i in 0..rows {
        let (c0, c1) = (1.0, i as f64 * 0.5 - 1.0);
        data.extend([c0, c1, c0 + c1, (i * i) as f64]);
    }
    let a = Tensor::from_vec(data, &[rows, 4]).unwrap();
    let b = Tensor::from_vec((0..rows).map(|i| i as f64).collect(), &[rows, 1]).unwrap();
    assert_eq!(Qr::new(a.view()).unwrap().dependent_column(), Some(2));
    assert_eq!(
        qr_reference::Qr::new(a.view()).unwrap().dependent_column(),
        Some(2)
    );
    for result in [
        lstsq(a.view(), b.view()),
        qr_reference::lstsq(a.view(), b.view()),
    ] {
        match result {
            Err(Error::InvalidData(msg)) => assert!(msg.contains("column 2"), "{msg}"),
            other => panic!("expected InvalidData naming column 2, got {other:?}"),
        }
    }
}

/// A zero column has norm 0, and `0 ≤ 0`, so it counts as dependent, wherever it is.
#[test]
fn zero_column_is_dependent() {
    let first = Tensor::from_vec(vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0], &[3, 2]).unwrap();
    let middle =
        Tensor::from_vec(vec![1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 3.0, 0.0, 5.0], &[3, 3]).unwrap();
    for (a, j) in [(&first, 0), (&middle, 1)] {
        let shipped = Qr::new(a.view()).unwrap();
        let reference = qr_reference::Qr::new(a.view()).unwrap();
        assert_eq!(shipped.dependent_column(), Some(j));
        assert_eq!(reference.dependent_column(), Some(j));
        assert_eq!(shipped.column_norms()[j], 0.0);
        assert!(shipped.r_diagonal().iter().all(|r| r.is_finite()));
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        let err = lstsq(a.view(), b.view()).unwrap_err().to_string();
        assert!(err.contains(&format!("column {j}")), "{err}");
    }
}

/// Column 1 is 1e-12 times a column independent of column 0. The test is relative to each
/// column's own norm, so the tiny column is not dependent and its coefficient is recovered.
#[test]
fn wildly_scaled_columns_are_not_dependent() {
    let rows = 5;
    let mut data = Vec::new();
    let mut b = Vec::new();
    for i in 0..rows {
        let (c0, c1) = (1.0, 1e-12 * (i as f64 - 2.0));
        data.extend([c0, c1]);
        b.push(3.0 * c0 + 7.0e12 * c1);
    }
    let a = Tensor::from_vec(data, &[rows, 2]).unwrap();
    let b = Tensor::from_vec(b, &[rows, 1]).unwrap();
    assert_eq!(Qr::new(a.view()).unwrap().dependent_column(), None);
    assert_eq!(
        qr_reference::Qr::new(a.view()).unwrap().dependent_column(),
        None
    );
    for x in [
        lstsq(a.view(), b.view()).unwrap(),
        qr_reference::lstsq(a.view(), b.view()).unwrap(),
    ] {
        assert!((x.data()[0] - 3.0).abs() <= 1e-12, "{:?}", x.data());
        assert!(
            (x.data()[1] - 7.0e12).abs() <= 1e-12 * 7.0e12,
            "{:?}",
            x.data()
        );
    }
}

/// A matrix with more columns than rows has no unique least-squares solution and is refused.
#[test]
fn wide_matrix_is_an_error() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    for result in [
        Qr::new(a.view()).map(|_| ()),
        qr_reference::Qr::new(a.view()).map(|_| ()),
    ] {
        match result {
            Err(Error::InvalidData(msg)) => {
                assert!(msg.contains("more columns than rows"), "{msg}")
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }
}

/// The smallest problem: `2 x = 6` gives `x = 3`, with `R = [-2]` by the sign convention.
#[test]
fn single_row_single_column() {
    let a = Tensor::from_vec(vec![2.0], &[1, 1]).unwrap();
    let b = Tensor::from_vec(vec![6.0], &[1, 1]).unwrap();
    assert_eq!(lstsq(a.view(), b.view()).unwrap().data(), &[3.0]);
    assert_eq!(
        qr_reference::lstsq(a.view(), b.view()).unwrap().data(),
        &[3.0]
    );
    assert_eq!(Qr::new(a.view()).unwrap().r_diagonal(), &[-2.0]);
    assert_eq!(
        qr_reference::Qr::new(a.view()).unwrap().r_diagonal(),
        &[-2.0]
    );
}

/// The two factorisations use the same sign convention, so their R diagonals and column
/// norms agree in sign and closely in value.
#[test]
fn r_diagonals_agree() {
    let mut rng = Rng::seed(5);
    for (rows, cols) in problem_sizes() {
        let a = gaussian(&mut rng, rows, cols);
        let shipped = Qr::new(a.view()).unwrap();
        let reference = qr_reference::Qr::new(a.view()).unwrap();
        for (s, r) in shipped.r_diagonal().iter().zip(reference.r_diagonal()) {
            assert!((s - r).abs() <= 1e-12 * r.abs().max(1.0), "{s} against {r}");
        }
        for (s, r) in shipped.column_norms().iter().zip(reference.column_norms()) {
            assert!((s - r).abs() <= 1e-13 * r, "{s} against {r}");
        }
    }
}

/// Wrong shapes and non-finite values are data errors, not panics: a matrix that is not
/// two-dimensional, a right-hand side with the wrong number of rows, and a NaN.
#[test]
fn bad_inputs_are_errors() {
    let vector = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    assert!(matches!(Qr::new(vector.view()), Err(Error::InvalidData(_))));
    assert!(matches!(
        qr_reference::Qr::new(vector.view()),
        Err(Error::InvalidData(_))
    ));
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let short = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
    assert!(matches!(
        Qr::new(a.view()).unwrap().solve(short.view()),
        Err(Error::InvalidData(_))
    ));
    assert!(matches!(
        qr_reference::Qr::new(a.view()).unwrap().solve(short.view()),
        Err(Error::InvalidData(_))
    ));
    let nan = Tensor::from_vec(vec![1.0, f64::NAN, 3.0], &[3, 1]).unwrap();
    assert!(matches!(Qr::new(nan.view()), Err(Error::InvalidData(_))));
    assert!(matches!(
        lstsq(a.view(), nan.view()),
        Err(Error::InvalidData(_))
    ));
    assert!(matches!(
        qr_reference::lstsq(a.view(), nan.view()),
        Err(Error::InvalidData(_))
    ));
}
