"""Writes data/concrete/concrete.csv and data/breast-cancer/breast-cancer.csv from the archives
the UCI Machine Learning Repository publishes, so the two committed files can be rebuilt.

Download the two archives first (into an empty folder of your choice):

    https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip
    https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip

then run, from the repository root, with the fixture environment (which has `xlrd` to read the
concrete workbook; see scripts/fixtures/requirements.txt):

    scripts/fixtures/.venv/bin/python -I scripts/data/make_tabular_csvs.py \\
        --concrete-zip PATH/concrete+compressive+strength.zip \\
        --breast-cancer-zip PATH/breast+cancer+wisconsin+diagnostic.zip

The script checks the SHA-256 of the data file inside each archive against the copy the
committed CSVs were made from, then writes:

- `concrete.csv`: a header line, then the 1,030 rows of the first sheet of
  `Concrete_Data.xls` in sheet order: the eight inputs (kilograms per cubic metre, and the age
  in days) and the compressive strength in MPa. Whole numbers are written without a decimal
  point; every other value is Python's shortest text that reads back as the same float64.
- `breast-cancer.csv`: a header line, then the 569 rows of `wdbc.data` in file order: the 30
  features exactly as the text in that file, then the class, 0 for malignant (M) and 1 for
  benign (B), as scikit-learn numbers them. The ID column is dropped.

Lines end with a single newline. Every check exits with status 1 and a message.
"""

import argparse
import hashlib
import io
import os
import sys
import zipfile
from typing import NoReturn

import xlrd

# SHA-256 of the data files inside the UCI archives the committed CSVs were made from.
CONCRETE_XLS = ("Concrete_Data.xls", "710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4")
WDBC_DATA = ("wdbc.data", "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a")

CONCRETE_HEADER = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age_days",
    "strength_mpa",
]

# The 30 features of wdbc.data in file order: the mean, the standard error and the worst value
# of ten measurements of the cell nuclei.
MEASUREMENTS = [
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
]
BREAST_CANCER_HEADER = (
    [f"mean_{m}" for m in MEASUREMENTS]
    + [f"{m}_error" for m in MEASUREMENTS]
    + [f"worst_{m}" for m in MEASUREMENTS]
    + ["target"]
)


def fail(message) -> NoReturn:
    """Prints `message` to standard error and exits with status 1."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def member(zip_path, name, sha256):
    """Returns the bytes of `name` inside the archive at `zip_path`, after checking its SHA-256."""
    with zipfile.ZipFile(zip_path) as archive:
        try:
            data = archive.read(name)
        except KeyError:
            fail(f"{zip_path} has no {name}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        fail(f"{name} in {zip_path} has SHA-256 {digest}, expected {sha256}")
    return data


def number_text(value):
    """Whole numbers without a decimal point, anything else as the shortest round-trip text."""
    if value == int(value):
        return str(int(value))
    return repr(value)


def concrete_rows(xls_bytes):
    """The 1,030 data rows of the concrete workbook's first sheet, as lists of text."""
    book = xlrd.open_workbook(file_contents=xls_bytes)
    sheet = book.sheet_by_index(0)
    if sheet.nrows != 1031 or sheet.ncols != 9:
        fail(f"concrete sheet is {sheet.nrows} by {sheet.ncols}, expected 1031 by 9")
    rows = []
    for r in range(1, sheet.nrows):
        cells = sheet.row(r)
        if any(cell.ctype != xlrd.XL_CELL_NUMBER for cell in cells):
            fail(f"concrete sheet row {r + 1} has a cell that is not a number")
        rows.append([number_text(cell.value) for cell in cells])
    return rows


def breast_cancer_rows(data_bytes):
    """The 569 rows of wdbc.data as lists of text: 30 features, then 0 (M) or 1 (B)."""
    rows = []
    for line_no, line in enumerate(data_bytes.decode("ascii").splitlines(), start=1):
        fields = line.strip().split(",")
        if len(fields) != 32:
            fail(f"wdbc.data line {line_no} has {len(fields)} fields, expected 32")
        classes = {"M": "0", "B": "1"}
        if fields[1] not in classes:
            fail(f"wdbc.data line {line_no} has class {fields[1]!r}, expected M or B")
        for text in fields[2:]:
            try:
                float(text)
            except ValueError:
                fail(f"wdbc.data line {line_no} has {text!r}, which is not a number")
        rows.append(fields[2:] + [classes[fields[1]]])
    if len(rows) != 569:
        fail(f"wdbc.data has {len(rows)} rows, expected 569")
    return rows


def write_csv(path, header, rows):
    """Writes `header` and `rows` as comma-separated lines ending in a newline."""
    out = io.StringIO()
    out.write(",".join(header) + "\n")
    for row in rows:
        out.write(",".join(row) + "\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(out.getvalue())
    print(f"wrote {path}: {len(rows)} rows, SHA-256 {hashlib.sha256(out.getvalue().encode()).hexdigest()}")


def main():
    """Reads the two archives and writes the two CSV files under `--out`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--concrete-zip", required=True)
    parser.add_argument("--breast-cancer-zip", required=True)
    parser.add_argument("--out", default="data", help="the data folder (default: data)")
    args = parser.parse_args()
    concrete = concrete_rows(member(args.concrete_zip, *CONCRETE_XLS))
    write_csv(os.path.join(args.out, "concrete", "concrete.csv"), CONCRETE_HEADER, concrete)
    breast_cancer = breast_cancer_rows(member(args.breast_cancer_zip, *WDBC_DATA))
    write_csv(
        os.path.join(args.out, "breast-cancer", "breast-cancer.csv"),
        BREAST_CANCER_HEADER,
        breast_cancer,
    )


if __name__ == "__main__":
    main()
