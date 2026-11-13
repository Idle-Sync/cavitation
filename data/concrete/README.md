# Concrete compressive strength

`concrete.csv` is the UCI "Concrete Compressive Strength" data set: 1,030 concrete mixtures,
each with eight inputs and the compressive strength measured in the laboratory. The first
line is a header; then one row per mixture, in the order of the original workbook:

| Column | Unit |
|---|---|
| `cement` | kg in a cubic metre of mixture |
| `blast_furnace_slag` | kg in a cubic metre of mixture |
| `fly_ash` | kg in a cubic metre of mixture |
| `water` | kg in a cubic metre of mixture |
| `superplasticizer` | kg in a cubic metre of mixture |
| `coarse_aggregate` | kg in a cubic metre of mixture |
| `fine_aggregate` | kg in a cubic metre of mixture |
| `age_days` | days (1 to 365) |
| `strength_mpa` | compressive strength, MPa (the target) |

SHA-256 of `concrete.csv`: `858bd88ee4817fa50a3915312403ea425d91c34927e77c8bcfc334eee6544995`

## How the file was made

UCI publishes the data as an Excel workbook, `Concrete_Data.xls`, inside
<https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip>.
`scripts/data/make_tabular_csvs.py` reads its first sheet with `xlrd` and writes every value
unchanged: whole numbers without a decimal point, the others as the shortest decimal text that
reads back as the same 64-bit float (so `79.98611076`, not a rounded `79.99`). The script
checks the workbook's SHA-256
(`710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4`) first; its docstring
gives the command.

## The train and test split

`cavitation_datasets::concrete::load` shuffles the 1,030 rows with mulberry32 (seed 42),
exactly as the digits split is made (see `data/digits/README.md`, with 1,030 rows in place
of 1,797), and takes the first 824 entries of the order as the training rows and the other
206 as the test rows. The first five entries are 476, 747, 84, 914 and 207. Values are read
as 64-bit floats and stored as 32-bit floats.

## Source and citation

- <https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength>
- I-Cheng Yeh (1998). Modeling of strength of high-performance concrete using artificial
  neural networks. Cement and Concrete Research, 28(12), 1797–1808.
  <https://doi.org/10.1016/S0008-8846(98)00165-3>
- Yeh, I. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5PK67>

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the data set's UCI
page: <https://creativecommons.org/licenses/by/4.0/>.
