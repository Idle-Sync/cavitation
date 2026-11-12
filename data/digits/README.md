# Optical recognition of handwritten digits

`digits.csv` is the copy of this data set that ships with scikit-learn 1.5.2
(the file digits.csv.gz in its datasets/data folder, decompressed): 1,797 rows, one per image. Each row
has 64 integer pixel values in 0 to 16 (an 8 by 8 image, row by row), then the digit 0 to 9
as the last column. There is no header.

SHA-256 of `digits.csv`: `6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8`

## The train and test split

The library's loader shuffles the 1,797 rows and takes the first 1,437 as the training set and
the other 360 as the test set. To rebuild the split elsewhere: number the rows of `digits.csv`
0 to 1,796 in file order; draw numbers `r` in `[0, 1)` from the mulberry32 generator with seed
42 (the 32-bit output divided by 2^32); then for `i` from 1,796 down to 1, let
`k = floor(r * (i + 1))` with a fresh `r` each time, and swap entries `i` and `k` of the order.
Entries 0 to 1,436 of the order are the training rows, in that order, and entries 1,437 to
1,796 are the test rows. The first five entries are 1784, 96, 434, 1205 and 1703.

```python
def mulberry32(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((state ^ (state >> 15)) * (1 | state)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        yield (t ^ (t >> 14)) / 4294967296

rng = mulberry32(42)
order = list(range(1797))
for i in range(1796, 0, -1):
    k = int(next(rng) * (i + 1))
    order[i], order[k] = order[k], order[i]
train, test = order[:1437], order[1437:]
```

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
