# Measuring a classifier

Accuracy is the share of rows a model gets right, and on its own it can mislead. If 87.5% of the rows belong to one class, a model that always answers that class scores 87.5% and finds none of the others. This chapter trains a digits classifier, then reads its confusion matrix and classification report and scores it as a yes-or-no detector with ROC AUC and average precision. It also shows the baseline that accuracy flatters.

The program is `crates/cavitation/examples/classification_metrics.rs`. Run it with:

```sh
cargo run --release --example classification_metrics -p cavitation
```

The metrics are in `cavitation::cavitation_nn::metrics`, and `confusion_matrix` is in the prelude. The printed blocks below come from that command, run on an Intel Celeron N5105 (4 cores) at kernel level `sse4.2`, rustc 1.99.1. It takes under a second to run (0.75 s here, measured with `time`, after building). It trains the 64 → 256 → 128 → 10 ReLU network with the digits recipe: 20 epochs of Adam, batch 32.

## The confusion matrix and the report

`confusion_matrix` runs the model over the data and counts each row by its true class (down) and its predicted class (across). The prediction is the class of the first largest output, which is the rule `evaluate` uses for accuracy, so the matrix's accuracy is `evaluate`'s. Printing the matrix gives the counts, then a report in the shape of scikit-learn's `classification_report`:

```rust,ignore
{{#include ../../crates/cavitation/examples/classification_metrics.rs:report}}
```

```text
 true \ pred        0        1        2        3        4        5        6        7        8        9
           0       41        0        0        0        0        0        0        0        0        0
           1        0       28        0        0        0        0        0        0        0        0
           2        0        0       40        0        0        0        0        0        0        0
           3        0        0        0       30        0        1        0        0        0        0
           4        0        0        0        0       35        0        0        0        0        0
           5        0        0        0        0        0       37        0        0        0        1
           6        0        0        0        0        0        1       33        0        0        0
           7        0        0        0        0        0        0        0       41        0        1
           8        0        2        1        0        0        1        0        0       22        0
           9        0        0        0        0        0        0        0        1        0       44

       class precision    recall        f1   support
           0    1.0000    1.0000    1.0000        41
           1    0.9333    1.0000    0.9655        28
           2    0.9756    1.0000    0.9877        40
           3    1.0000    0.9677    0.9836        31
           4    1.0000    1.0000    1.0000        35
           5    0.9250    0.9737    0.9487        38
           6    1.0000    0.9706    0.9851        34
           7    0.9762    0.9762    0.9762        42
           8    1.0000    0.8462    0.9167        26
           9    0.9565    0.9778    0.9670        45
    accuracy                        0.9750       360
   macro avg    0.9767    0.9712    0.9730       360
weighted avg    0.9760    0.9750    0.9748       360
```

The accuracy is 97.5%, and the report shows where the other 2.5% went. Each row of the report reads one class:

- **Precision** is the share of the rows predicted as the class that really are the class: `tp / (tp + fp)`. Class 5 has 0.925, because three rows of other classes were called 5.
- **Recall** is the share of the class's rows the model found: `tp / (tp + fn)`. Class 8 has 0.846: 4 of its 26 test rows went to 1, 2 and 5. That is the model's weak spot, and the 97.5% accuracy hides it.
- **F1** is their harmonic mean, `2·P·R / (P + R)`, which is low when either one is. `f_beta(class, beta)` weights recall `beta` times as much as precision. F2 (`beta = 2`) suits a task where missing a case costs more than a false alarm, such as screening.
- **Support** is the number of rows of the class.

The last two lines average over the classes. The **macro** average treats every class alike, so a rare class that does badly pulls it down as much as a common one. The **weighted** average weights each class by its support, so it stays close to accuracy. A third, **micro**, pools the counts of every class into one rate (`Average::Micro`). When every row has a prediction, micro precision and micro recall both equal accuracy.

Two conventions match scikit-learn's defaults:

- A rate whose denominator is 0, such as the precision of a class the model never predicts, is 0.
- The averages cover the classes that appear among the targets or the predictions.

A row whose outputs are all NaN has no prediction. It counts against its class's recall and against accuracy, and the table gives it a `none` column.

## One class as a yes-or-no question

Precision and recall above depend on one decision rule: the largest output wins. A detector usually has a threshold to tune instead. "Is it a 9?" can be answered with the network's probability of 9 and any cut-off. Two measures summarise every cut-off at once:

```rust,ignore
{{#include ../../crates/cavitation/examples/classification_metrics.rs:nine}}
```

```text
is it a 9?  network: ROC AUC 0.9988, average precision 0.9902
```

`Evaluator::class_scores` gives each row's softmax probability of the class.

- **ROC AUC** is the area under the curve of true positive rate against false positive rate as the threshold falls. It equals the chance that a random 9 scores higher than a random other digit, so 0.5 is guessing and 1 is a perfect ranking.
- **Average precision** sums the precision at each threshold, weighted by how much recall rose there. It is the more telling of the two when positives are rare: it ignores the many easy negatives that ROC AUC counts, and asks how clean the top of the ranking is.

Both group tied scores, as scikit-learn does.

## The baseline that accuracy flatters

12.5% of the test digits are 9s. A detector that always answers "no" needs no model at all:

```rust,ignore
{{#include ../../crates/cavitation/examples/classification_metrics.rs:baseline}}
```

```text
always "no": accuracy 0.8750, recall of 9 0.0000, macro F1 0.4667
```

Its 87.5% accuracy sounds respectable, and its recall of 9 is 0: it never finds one. The macro F1 of 0.47 says so plainly, since the "9" class scores 0 and the "not 9" class scores 0.93. When the classes are imbalanced, report recall (or F-beta) per class and the macro average next to accuracy. For a detector, report average precision.

## The same numbers elsewhere

`LogisticModel::confusion` in `cavitation-ml` gives the same `ConfusionMatrix` for a logistic regression, and `ConfusionMatrix::from_labels` builds one from any two lists of labels. The metrics are checked against scikit-learn 1.9.1 on seven cases: imbalanced, balanced and perfect classes; tied, rare, perfect and reversed scores. Every count matches exactly, and every rate to 1e-12.
