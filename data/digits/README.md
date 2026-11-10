# Optical recognition of handwritten digits

`digits.csv` is the copy of this data set that ships with scikit-learn 1.5.2
(the file digits.csv.gz in its datasets/data folder, decompressed): 1,797 rows, one per image. Each row
has 64 integer pixel values in 0 to 16 (an 8 by 8 image, row by row), then the digit 0 to 9
as the last column. There is no header.

SHA-256 of `digits.csv`: `6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8`

## Source and citation

The data is the test set of the UCI "Optical Recognition of Handwritten Digits" data set,
created by E. Alpaydin and C. Kaynak (Bogazici University, 1998):

- <https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits>
- E. Alpaydin, C. Kaynak (1998). Cascading Classifiers. Kybernetika.
- C. Kaynak (1995). Methods of Combining Multiple Classifiers and Their Applications to
  Handwritten Digit Recognition. MSc thesis, Bogazici University.

The images were made with NIST preprocessing programs: 32 by 32 bitmaps are divided into
non-overlapping 4 by 4 blocks and the "on" pixels in each block are counted.

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0), as published by the UCI Machine
Learning Repository: <https://creativecommons.org/licenses/by/4.0/>.
