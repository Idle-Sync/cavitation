# Regression, exactly and by gradient

When a straight line or a plane explains the data, you do not need to train anything: the best line has a formula, and a computer finds it in one step. This chapter fits two data sets that way and then by gradient descent, which is how every network in the other chapters learns, and shows that both end at the same answer. It covers least squares and how QR solves it, Newton's method for logistic regression, when the gradient path stops short, why the inputs are standardised first and how that scaling is folded back into the model, and what R² measures.

The programs are `crates/cavitation/examples/linear_regression.rs` (the Concrete Compressive Strength data: eight mixture inputs, the strength in MPa) and `crates/cavitation/examples/logistic_regression.rs` (the Breast Cancer Wisconsin data: 30 measurements per tumour, malignant or benign). Run them with:

```sh
cargo run --release --example linear_regression -p cavitation
cargo run --release --example logistic_regression -p cavitation
```

The models are in `cavitation::cavitation_ml`, and `Standardizer` is in the prelude. The printed blocks in this chapter come from these two commands, run on an Intel Celeron N5105 (4 cores) at kernel level `sse4.2`, rustc 1.99.1. Each takes under a second to run (0.2 s here, measured with `time`, after building).

## Least squares

A linear model predicts `ŷ = x·w + b`: each input times a weight, summed, plus an intercept. *Least squares* picks the weights that make the squared errors `Σ (ŷ − y)²` over the training rows as small as possible. Ridge regression adds a penalty `α·Σ w²` to that sum, which pulls the weights towards 0 and keeps them finite when two inputs say nearly the same thing; the intercept is not penalised.

The textbook route is to solve the *normal equations* `XᵀX·w = Xᵀy`. It works, but forming `XᵀX` squares the condition number of the problem, so a problem that is merely awkward becomes one that loses half its digits. `LinearRegression::fit` takes the route that does not: Householder QR. It turns the input matrix `X` into a product `Q·R`, where `Q` only rotates and reflects (it preserves lengths, so it cannot amplify any error) and `R` is upper triangular. Rotating the problem by `Qᵀ` leaves the least-squares problem the same, and with a triangular `R` the weights fall out one at a time from the last to the first, by back substitution. Ridge is the same solve with the rows `√α·I` stacked under `X` and zeros under `y`, because `‖y − Xw‖² + ‖0 − √α·w‖²` is exactly the ridge sum. Everything is computed in `f64`.

QR also tells you when there is no single answer: if a column is a combination of the columns before it, `R` has a zero on its diagonal there, and `fit` returns an error naming the column instead of one answer out of many. Setting `alpha` above 0 makes the problem solvable again.

## The gradient path

`fit_gradient` solves the same problem the way a network would: a `Linear` layer, the mean squared error as the loss, and `GradientConfig` (by default 1,000 full-batch epochs of Adam at learning rate 0.5, cosine annealing down to 0). This is the path to use when the model is one layer of a larger network, or when the data does not fit the exact solver. For the plain linear problem the exact path is faster and exact; the gradient path is here so you can see the two meet.

## Standardise first

The gradient path has a limit the exact path does not: a weight can move only about as far as the learning rate times the epochs, so it cannot reach coefficients much larger than that. Raw Concrete has inputs in the hundreds (kilograms of cement) and a target in tens (MPa). Standardising puts every column on the same footing: subtract its mean and divide by its standard deviation. The statistics come from the *train* rows only, and the test rows are scaled with them, so nothing about the test rows leaks into the fit. The value targets are standardised too:

```rust,ignore
{{#include ../../crates/cavitation/examples/linear_regression.rs:standardise}}
```

Then every fit is a single call:

```rust,ignore
{{#include ../../crates/cavitation/examples/linear_regression.rs:fit}}
```

## Both paths meet at the same answer

Ordinary least squares and ridge (α = 1), exact and by gradient, on the standardised Concrete rows. The coefficients are in standard deviations of strength per standard deviation of input:

```text
Coefficients on the standardised data (strength in standard deviations):
                     OLS exact    OLS grad ridge exact  ridge grad
cement                  0.7543      0.7543      0.7406      0.7406
blast furnace slag      0.5466      0.5466      0.5333      0.5333
fly ash                 0.3358      0.3358      0.3238      0.3238
water                  -0.1722     -0.1722     -0.1811     -0.1811
superplasticizer        0.1337      0.1337      0.1337      0.1337
coarse aggregate        0.1036      0.1036      0.0944      0.0944
fine aggregate          0.1031      0.1031      0.0913      0.0913
age                     0.4430      0.4430      0.4418      0.4418
intercept               0.0000      0.0000      0.0000      0.0000
largest difference, gradient against exact (relative to max(1, |exact|)): OLS 2.6e-8, ridge 6.6e-8
```

The two paths agree to the fourth decimal in every cell; the largest difference over all coefficients and intercepts is 2.6e-8 (OLS) and 6.6e-8 (ridge), measured as `|gradient − exact| / max(1, |exact|)`. The library's own check asks for 1e-3. Ridge pulls most weights a little towards 0 (cement from 0.7543 to 0.7406); the water weight, which is negative, goes from -0.1722 to -0.1811.

They agree because the problem has one lowest point. The squared error is a bowl in the weights, with no other valley, so a gradient method that is going downhill and takes small enough steps at the end must arrive where the formula puts the bottom. Logistic regression below is the same: its objective is also a single bowl.

### When the gradient path stops short

The exact answer is the bottom of the bowl; the gradient path is only as good as its steps. Run on the *raw* inputs and strengths, the same default configuration does not get there:

```rust,ignore
{{#include ../../crates/cavitation/examples/linear_regression.rs:unscaled}}
```

```text
on the raw data the gradient path ends 2.2e0 from the exact answer (largest difference as above); the exact path does not care about scale
```

A relative difference of 2.2 is a coefficient that is off by more than its own size, and the call still returns `Ok`: nothing in the training failed, it just did not arrive. (With the health checks on, the run prints an exploding-weights warning on epoch after epoch; the example silences it to keep its output short.) The exact path fits the raw data without trouble. So the rule is to standardise before the gradient path, and to fold the scaling back in afterwards.

The same thing happens when the penalty is weak. In logistic regression `c` is the inverse penalty strength, so a large `c` is a weak penalty, and the weights are allowed to grow before the penalty stops them. On standardised Breast Cancer data:

```text
455 train rows, 114 test rows, 30 measurements each
c = 1:
  exact     10 Newton iterations      test accuracy 0.9737  objective 33.140037
  gradient 1000 epochs                 test accuracy 0.9737  objective 33.140037
  largest coefficient difference (relative to max(1, |exact|)): 4.8e-7
c = 10:
  exact     12 Newton iterations      test accuracy 0.9649  objective 215.446382
  gradient 1000 epochs                 test accuracy 0.9649  objective 215.446382
  largest coefficient difference (relative to max(1, |exact|)): 5.7e-5
```

At `c = 1` the paths end 4.8e-7 apart; at `c = 10` they end 5.7e-5 apart, about a hundred times further, though both are still inside 1e-3 and the objectives agree to every printed digit. The weaker the penalty, the flatter the bottom of the bowl and the longer the gradient path takes to settle in it; with no penalty at all and classes that can be told apart perfectly, there is no bottom, and the exact path says so (`Error::InvalidData`, "separable") instead of returning weights that grow without bound.

## Logistic regression and Newton's method

Logistic regression predicts the probability of class 1 as `p = σ(x·w + b)`, where `σ` squeezes any number into 0 to 1. It is fitted by minimising the log-loss, plus the L2 penalty `½‖w‖²` with each row's loss weighted by `c`. The `objective` printed above is exactly `c · Σ log-loss + ½‖w‖²` on the training rows, and the intercept is not penalised.

Unlike least squares, there is no formula for the best weights, but there is a recipe that gets there in a handful of steps, *Newton's method*: at the current weights, replace the objective by the bowl that matches it in height, slope and curvature, and jump to that bowl's bottom. Here the bowl's bottom turns out to be a weighted least-squares problem. For each row, with `p = σ(x·w)`, take the weight `p(1 − p)` (a row the model is unsure about counts more than one it has settled) and the working response `x·w + (y − p)/(p(1 − p))`, and solve for the new weights. This is also called iteratively reweighted least squares, and `LogisticRegression::fit` solves each step with the same QR as above. A step that would not lower the objective is halved until it does. On the standardised Breast Cancer rows it takes 10 iterations at `c = 1` and 12 at `c = 10`; the gradient path takes 1,000 epochs for the same answer, and reaches the same test accuracy, 0.9737 and 0.9649.

```rust,ignore
{{#include ../../crates/cavitation/examples/logistic_regression.rs:fit}}
```

## Standardisation and folding

The standardised model answers questions asked in standard deviations. To use it on a new mixture you would have to standardise the inputs, predict, and then undo the target scaling, carrying the means and scales along with the weights. `fold_into_input` and `fold_into_output` remove that: the first folds the input scaling into the layer's weights, `W' = W/σ` and `b' = b − W'μ`, and the second folds the target scaling into the last layer, `W' = σW` and `b' = σb + μ`. They are the same arithmetic as the scaling, moved into the numbers, so the network takes raw inputs and answers in MPa:

```rust,ignore
{{#include ../../crates/cavitation/examples/linear_regression.rs:fold}}
```

Fold once: folding the same parameters again divides by the scale again. The example checks the folded network against the unfolded pipeline (standardise, predict, undo the scaling) on the raw test rows, and saves it as a `.safetensors` file that loads as a PyTorch `nn.Linear`:

```rust,ignore
{{#include ../../crates/cavitation/examples/linear_regression.rs:save}}
```

```text
folded network on the raw test rows: R² 0.6029; its answers differ from the unfolded pipeline's by at most 1.9e-5 MPa
first test row: 7.32 MPa measured, 26.14 MPa predicted
```

The folded answers differ from the unfolded ones by at most 1.9e-5 MPa: not zero, because each is rounded to `f32` at a different point, but far below anything a strength measurement can resolve.

## R²

R² says how much better the model is than always answering the mean of the targets. It is `1 − SSres/SStot`, where `SSres` is the sum of the squared errors of the model and `SStot` the sum of the squared distances of the targets from their mean. A model that does no better than the mean scores 0, a perfect one scores 1, and a model worse than the mean scores below 0. It does not change when the targets are rescaled, which is why the standardised and the folded model score alike. Two routes compute it: `LinearModel::score`, and `evaluate` on the `Sequential`, whose `Evaluation::r2` is set for value targets:

```rust,ignore
{{#include ../../crates/cavitation/examples/linear_regression.rs:score}}
```

```text
R² of the exact OLS fit:
  by LinearModel::score        train 0.6176, test 0.6029
  by evaluate on the Sequential train 0.6176, test 0.6029
```

The two routes agree to every printed digit. The test R², 0.6029, is a little below the train R², 0.6176, which is what a fitted model usually shows on rows it has not seen. About 60% of the variation in concrete strength is explained by these eight inputs and a straight line; the rest is what a line cannot see, which is where a network with hidden layers starts to earn its keep.
