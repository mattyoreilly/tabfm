# Fit a zero-shot TabFM model

`tabfm()` loads the pretrained TabFM v1.0.0 weights and conditions them
on your training data. Because TabFM learns in-context — a single
forward pass over the table — there is no feature engineering, no
hyperparameter tuning, and no retraining.

Loading the weights is the slow part, so it happens once per session for
each `task`/`backend` combination and the model is reused by later fits;
repeated calls to
[predict()](https://mattyoreilly.github.io/tabfm/reference/predict.tabfm.md)
are cheap.

Fitted models survive
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html)/[`readRDS()`](https://rdrr.io/r/base/readRDS.html):
because a zero-shot fit holds no learned state beyond the training data,
a restored fit is re-conditioned on its stored data automatically the
first time you call
[predict()](https://mattyoreilly.github.io/tabfm/reference/predict.tabfm.md)
in the new session (which also pays that session's one-time weights
load).

## Usage

``` r
tabfm(
  x,
  y,
  task = c("classification", "regression"),
  backend = c("jax", "pytorch")
)
```

## Arguments

- x:

  A data frame of predictors. Character and factor columns are handled
  natively by TabFM; no dummy coding is needed.

- y:

  The response: a character vector or factor for classification, a
  numeric vector for regression. Must have one value per row of `x`.

- task:

  Type of model to load: `"classification"` (the default) or
  `"regression"`.

- backend:

  Compute backend: `"jax"` or `"pytorch"`. Must match the backend you
  installed with
  [`install_tabfm()`](https://mattyoreilly.github.io/tabfm/reference/install_tabfm.md).

## Value

A model object of class `"tabfm"`: a list carrying the training data
(`x`, `y`), `task`, `backend`, `features` (the training column names),
and a cache holding the fitted Python estimator. Use
[predict()](https://mattyoreilly.github.io/tabfm/reference/predict.tabfm.md)
to generate predictions from it.

## See also

[`predict.tabfm()`](https://mattyoreilly.github.io/tabfm/reference/predict.tabfm.md)
to make predictions;
[`install_tabfm()`](https://mattyoreilly.github.io/tabfm/reference/install_tabfm.md)
for one-time setup.

Other modelling:
[`predict.tabfm()`](https://mattyoreilly.github.io/tabfm/reference/predict.tabfm.md)

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- tabfm(iris[-5], iris$Species)
fit

predict(fit, head(iris[-5]))
predict(fit, head(iris[-5]), type = "prob")

fit <- tabfm(mtcars[-1], mtcars$mpg, task = "regression")
predict(fit, head(mtcars[-1]))
} # }
```
