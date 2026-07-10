# Get started with tabfm

TabFM is a foundation model for tabular data from Google Research. Where
a traditional workflow makes you engineer features, pick a model, and
tune hyperparameters, TabFM does none of that: it was pretrained on a
large corpus of tables, and it predicts your table *in-context*, the way
a large language model completes a prompt. You hand it training rows, it
reads them in a single forward pass, and it predicts the rows you’re
interested in.

This vignette shows the whole workflow. There isn’t much of one — that’s
the point.

``` r

library(tabfm)
```

*(Code in this vignette isn’t evaluated when the package is built,
because it needs a local Python setup and a one-time weights download.
Every output shown is real, captured from a live session.)*

## One-time setup

TabFM’s inference code is a Python package, which tabfm drives through
reticulate.
[`install_tabfm()`](https://mattyoreilly.github.io/tabfm/reference/install_tabfm.md)
creates a dedicated virtualenv (it needs Python \>= 3.11 and will tell
you if it can’t find one) and installs everything:

``` r

install_tabfm()            # JAX backend, CPU
# install_tabfm("pytorch") # PyTorch backend, use on a GPU machine
```

The first time you fit a model, the pretrained weights are downloaded
from the Hugging Face Hub and cached in `~/.cache/huggingface/`. You may
see a warning about unauthenticated requests; it’s harmless. To silence
it (and download faster), put a free Hugging Face [access
token](https://huggingface.co/settings/tokens) in your `~/.Renviron` as
`HF_TOKEN=hf_...`.

Once the weights are cached, you can speed up each session’s first fit —
no account needed — by adding `HF_HUB_OFFLINE=1` (load straight from the
cache, skipping the Hub’s revalidation requests) and
`JAX_COMPILATION_CACHE_DIR=~/.cache/jax` (reuse compiled XLA artifacts
across sessions) to `~/.Renviron`. Unset `HF_HUB_OFFLINE` whenever you
need to download weights you don’t have yet.

## Classification

[`tabfm()`](https://mattyoreilly.github.io/tabfm/reference/tabfm.md)
takes a data frame of predictors and a response vector, like
[`lm()`](https://rdrr.io/r/stats/lm.html)’s cousins everywhere.
Character and factor columns are handled natively — no dummy coding,
scaling, or imputation required:

``` r

fit <- tabfm(iris[-5], iris$Species)
fit
#> TabFM v1.0.0 zero-shot classification model | 4 features
```

“Fitting” is instant: TabFM learns your table in-context rather than
training on it. The loaded model stays in memory, so you can predict as
many times as you like without paying the load cost again.

[`predict()`](https://rdrr.io/r/stats/predict.html) returns class labels
by default:

``` r

predict(fit, head(iris[-5]))
#> [1] "setosa" "setosa" "setosa" "setosa" "setosa" "setosa"
```

Use `type = "prob"` for a probability matrix, one column per class:

``` r

predict(fit, head(iris[-5]), type = "prob")
#>         setosa   versicolor    virginica
#> [1,] 0.9997242 1.378794e-04 1.378794e-04
#> [2,] 1.0000000 1.902007e-08 1.902007e-08
#> [3,] 1.0000000 1.902007e-08 1.902007e-08
#> [4,] 0.9999967 1.619624e-06 1.619624e-06
#> [5,] 0.9998604 1.378898e-04 1.619404e-06
#> [6,] 0.9997242 1.378794e-04 1.378794e-04
```

## Regression

Pass a numeric response and `task = "regression"`:

``` r

fit <- tabfm(mtcars[-1], mtcars$mpg, task = "regression")
predict(fit, head(mtcars[-1]))
#> [1] 21.00013 21.00013 22.76699 21.38826 18.68293 18.10942
```

## When should you reach for TabFM?

TabFM shines when you want a strong baseline *now*: no preprocessing
pipeline, no tuning grid, no cross-validation loop. It is a particularly
good fit for small-to-medium tables where fitting a bespoke model risks
overfitting the validation set.

It is not the right tool when:

- you need to squeeze out accuracy on one specific dataset — TabFM is
  zero-shot, so there is nothing to tune. Reach for
  [tidymodels](https://www.tidymodels.org) and boosted trees.
- your data doesn’t fit in memory alongside the model, since every
  prediction re-reads the training table in-context.
- you can’t ship a Python runtime. Predictions always run through the
  Python package.
