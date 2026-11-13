//! Reading the comma-separated tables with a header line that the Concrete and Breast Cancer
//! loaders share, and splitting their rows into train and test sets.

use std::path::Path;

use cavitation_core::{Error, Tensor};
use cavitation_nn::{Dataset, Targets};

use crate::digits::split_order;

/// The rows of a table: every column but the last as features, the last as the target.
pub(crate) struct Table {
    /// The features, row-major, each read as `f64` and rounded to `f32`.
    pub(crate) x: Vec<f32>,
    /// The last column of every row, as `f64`.
    pub(crate) target: Vec<f64>,
    /// The number of features per row.
    pub(crate) features: usize,
}

/// Reads the table at `path`: a header line equal to `header`, then exactly `rows` lines of
/// `features + 1` finite numbers separated by commas.
///
/// # Errors
///
/// [`Error::Io`] if the file cannot be read; [`Error::Format`] naming the file and the line
/// if the header differs, a line has the wrong number of fields or a field that is not a
/// finite number, or the row count is not `rows`.
pub(crate) fn read(
    path: &Path,
    header: &str,
    rows: usize,
    features: usize,
) -> Result<Table, Error> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut lines = text.lines();
    if lines.next() != Some(header) {
        return Err(Error::Format(format!(
            "{}, line 1: expected the header {header}",
            path.display()
        )));
    }
    let mut x = Vec::with_capacity(rows * features);
    let mut target = Vec::with_capacity(rows);
    for (index, line) in lines.enumerate() {
        let bad =
            |what: &str| Error::Format(format!("{}, line {}: {what}", path.display(), index + 2));
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != features + 1 {
            return Err(bad(&format!(
                "expected {} fields, found {}",
                features + 1,
                fields.len()
            )));
        }
        for (column, field) in fields.iter().enumerate() {
            let value: f64 = field
                .trim()
                .parse()
                .ok()
                .filter(|v: &f64| v.is_finite())
                .ok_or_else(|| bad(&format!("field {} is not a finite number", column + 1)))?;
            if column < features {
                x.push(value as f32);
            } else {
                target.push(value);
            }
        }
    }
    if target.len() != rows {
        return Err(Error::Format(format!(
            "{}: expected {rows} rows, found {}",
            path.display(),
            target.len()
        )));
    }
    Ok(Table {
        x,
        target,
        features,
    })
}

/// Shuffles the table's rows with [`split_order`] and returns the first `train_rows` of the
/// order as the training set and the rest as the test set, with targets made by `targets`
/// from the chosen rows' target values, in order.
///
/// # Errors
///
/// Whatever `targets` returns, and [`Error`] from building the tensors.
pub(crate) fn split(
    table: &Table,
    train_rows: usize,
    targets: impl Fn(&[f64]) -> Result<Targets, Error>,
) -> Result<(Dataset, Dataset), Error> {
    let order = split_order(table.target.len());
    let features = table.features;
    let build = |rows: &[usize]| -> Result<Dataset, Error> {
        let mut x = Vec::with_capacity(rows.len() * features);
        let mut y = Vec::with_capacity(rows.len());
        for &i in rows {
            x.extend_from_slice(&table.x[i * features..(i + 1) * features]);
            y.push(table.target[i]);
        }
        Ok(Dataset {
            x: Tensor::from_vec(x, &[rows.len(), features])?,
            targets: targets(&y)?,
        })
    };
    Ok((build(&order[..train_rows])?, build(&order[train_rows..])?))
}

#[cfg(test)]
mod tests {
    //! The table reader on small files: a good one, and each kind of bad one.

    use super::*;
    use std::path::PathBuf;

    /// Writes `text` to a fresh file in the system temporary folder and returns its path.
    fn temp_file(name: &str, text: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "cavitation-tabular-{}-{name}.csv",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        path
    }

    /// A good table reads into features (rounded to f32) and targets (kept in f64).
    #[test]
    fn table_reads_rows() {
        let path = temp_file("good", "a,b,y\n1,2.5,0.1\n3,4,1\n");
        let table = read(&path, "a,b,y", 2, 2).unwrap();
        assert_eq!(table.x, [1.0, 2.5, 3.0, 4.0]);
        assert_eq!(table.target, [0.1, 1.0]);
        std::fs::remove_file(path).unwrap();
    }

    /// A wrong header, a short line, a word, a NaN and a missing row each name the problem.
    #[test]
    fn table_errors_name_the_line() {
        for (name, text, rows, expected) in [
            ("header", "a,c,y\n1,2,3\n", 1, "line 1: expected the header"),
            (
                "fields",
                "a,b,y\n1,2\n",
                1,
                "line 2: expected 3 fields, found 2",
            ),
            (
                "word",
                "a,b,y\n1,x,3\n",
                1,
                "line 2: field 2 is not a finite number",
            ),
            (
                "nan",
                "a,b,y\n1,2,3\n1,2,NaN\n",
                2,
                "line 3: field 3 is not a finite number",
            ),
            ("rows", "a,b,y\n1,2,3\n", 2, "expected 2 rows, found 1"),
        ] {
            let path = temp_file(name, text);
            let err = read(&path, "a,b,y", rows, 2).err().unwrap();
            assert!(matches!(err, Error::Format(_)), "{name}: {err:?}");
            assert!(err.to_string().contains(expected), "{name}: {err}");
            std::fs::remove_file(path).unwrap();
        }
    }

    /// A missing file is an I/O error naming the file.
    #[test]
    fn table_missing_file_is_io() {
        let path = std::env::temp_dir().join("cavitation-tabular-no-such-file.csv");
        let err = read(&path, "a", 1, 0).err().unwrap();
        assert!(matches!(err, Error::Io { .. }), "{err:?}");
    }
}
