# Predict from a TabFM model

Generate predictions — class labels, class probabilities, or numeric
values — from a model fitted with
[`tabfm()`](https://mattyoreilly.github.io/tabfm/reference/tabfm.md).

## Usage

``` r
# S3 method for class 'tabfm'
predict(object, newdata, type = c("response", "prob"), ...)
```

## Arguments

- object:

  A `"tabfm"` object returned by
  [`tabfm()`](https://mattyoreilly.github.io/tabfm/reference/tabfm.md).

- newdata:

  A data frame with exactly the same columns as the training data.

- type:

  What to return: `"response"` (the default) for class labels or numeric
  predictions, or `"prob"` for a matrix of class probabilities
  (classification only).

- ...:

  Ignored, for compatibility with the
  [predict()](https://rdrr.io/r/stats/predict.html) generic.

## Value

For `type = "response"`, a vector with one prediction per row of
`newdata`: character class labels for classification, numeric values for
regression. For `type = "prob"`, a numeric matrix with one row per row
of `newdata` and one column per class, named by class.

## See also

[`tabfm()`](https://mattyoreilly.github.io/tabfm/reference/tabfm.md)

Other modelling:
[`tabfm()`](https://mattyoreilly.github.io/tabfm/reference/tabfm.md)

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- tabfm(iris[-5], iris$Species)

# class labels
predict(fit, head(iris[-5]))

# class probabilities, one column per class
predict(fit, head(iris[-5]), type = "prob")
} # }
```
