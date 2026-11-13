# Breast Cancer Wisconsin (Diagnostic)

`breast-cancer.csv` is the UCI "Breast Cancer Wisconsin (Diagnostic)" data set: 569 breast
masses, each described by 30 features computed from a digitised image of a fine needle
aspirate, and the diagnosis. The first line is a header; then one row per mass, in the order
of the original `wdbc.data`:

- the mean, the standard error and the worst (mean of the three largest) value of ten
  measurements of the cell nuclei: radius, texture, perimeter, area, smoothness,
  compactness, concavity, concave points, symmetry and fractal dimension (columns
  `mean_radius` … `mean_fractal_dimension`, `radius_error` … `fractal_dimension_error`,
  `worst_radius` … `worst_fractal_dimension`);
- `target`: 0 for malignant (212 rows) and 1 for benign (357 rows), as scikit-learn numbers
  the classes.

The ID column of `wdbc.data` is dropped; the feature values are copied as text, unchanged.
The rows, values and classes are the same as the copy bundled with scikit-learn
(`sklearn.datasets.load_breast_cancer`), which `scripts/fixtures/make_sklearn_fixtures.py`
checks.

SHA-256 of `breast-cancer.csv`: `432ff316e7bfb60b70a275064b4401315cc39f09c9099d031013a23647e98687`

## How the file was made

`scripts/data/make_tabular_csvs.py` reads `wdbc.data` from
<https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip>,
after checking its SHA-256
(`d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`); its docstring gives
the command.

## The train and test split

`cavitation_datasets::breast_cancer::load` shuffles the 569 rows with mulberry32 (seed 42),
exactly as the digits split is made (see `data/digits/README.md`, with 569 rows in place of
1,797), and takes the first 455 entries of the order as the training rows and the other 114
as the test rows. The first five entries are 259, 560, 417, 460 and 288. Values are read as
64-bit floats and stored as 32-bit floats.

## Source and citation

- <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>
- W. H. Wolberg, W. N. Street and O. L. Mangasarian (1995). Image analysis and machine
  learning applied to breast cancer diagnosis and prognosis. Analytical and Quantitative
  Cytology and Histology, 17(2), 77–87.
- W. N. Street, W. H. Wolberg and O. L. Mangasarian (1993). Nuclear feature extraction for
  breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging:
  Science and Technology, volume 1905, 861–870.
- Wolberg, W., Mangasarian, O., Street, N. and Street, W. (1993). Breast Cancer Wisconsin
  (Diagnostic) [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5DW2B>

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the data set's UCI
page: <https://creativecommons.org/licenses/by/4.0/>.
