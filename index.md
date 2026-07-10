# tabfm

The goal of tabfm is to make
[TabFM](https://github.com/google-research/tabfm), Google Research’s
zero-shot foundation model for tabular data, easy to use from R. You
give it a data frame and a response; it gives you classification or
regression predictions with **no feature engineering, no hyperparameter
tuning, and no retraining** — the model predicts unseen tables
in-context, in a single forward pass. See the [announcement blog
post](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
for details on how it works.

tabfm wraps the official Python package with
[reticulate](https://rstudio.github.io/reticulate/), and exposes the
familiar R modelling interface: a fitting function plus
[`predict()`](https://rdrr.io/r/stats/predict.html).

## Installation

You can install the development version of tabfm like so:

``` r

# install.packages("remotes")
remotes::install_github("mattyoreilly/tabfm")
```

Then, one time only, install the underlying Python package into a
dedicated virtualenv:

``` r

library(tabfm)
install_tabfm()            # JAX backend, CPU
# install_tabfm("pytorch") # PyTorch backend, CPU/GPU
```

## Example

Fitting is instant — TabFM learns your table in-context rather than
training on it — and the loaded model stays in memory, so repeated
predictions are cheap. Character and factor columns are handled
natively; you never need to dummy-code.

``` r

library(tabfm)

fit <- tabfm(iris[-5], iris$Species)
fit
#> TabFM v1.0.0 zero-shot classification model | 4 features

predict(fit, head(iris[-5]))
#> [1] "setosa" "setosa" "setosa" "setosa" "setosa" "setosa"

predict(fit, head(iris[-5]), type = "prob")
#>         setosa   versicolor    virginica
#> [1,] 0.9997242 1.378794e-04 1.378794e-04
#> [2,] 1.0000000 1.902007e-08 1.902007e-08
#> [3,] 1.0000000 1.902007e-08 1.902007e-08
#> [4,] 0.9999967 1.619624e-06 1.619624e-06
#> [5,] 0.9998604 1.378898e-04 1.619404e-06
#> [6,] 0.9997242 1.378794e-04 1.378794e-04
```

Regression works the same way — pass a numeric response and
`task = "regression"`:

``` r

fit <- tabfm(mtcars[-1], mtcars$mpg, task = "regression")
predict(fit, head(mtcars[-1]))
#> [1] 21.00013 21.00013 22.76699 21.38826 18.68293 18.10942
```

## Making it fast

The pretrained weights are cached after the first download, but two
optional lines in your `~/.Renviron` (no Hugging Face account needed)
make cold starts noticeably faster:

``` R
HF_HUB_OFFLINE=1
JAX_COMPILATION_CACHE_DIR=~/.cache/jax
```

- `HF_HUB_OFFLINE=1` loads weights straight from the local cache instead
  of revalidating them against the Hugging Face Hub on each session’s
  first fit — unauthenticated requests get the slowest rate-limit tier,
  so this skips the wait entirely. Unset it whenever you need to
  download weights you don’t have yet.
- `JAX_COMPILATION_CACHE_DIR` lets new sessions reuse the compiled XLA
  artifacts from previous ones instead of recompiling the model graph.

Within a session, the loaded model is cached automatically: only the
first
[`tabfm()`](https://mattyoreilly.github.io/tabfm/reference/tabfm.md)
call per task pays the load cost.

## Limitations

- Predictions require a working Python installation;
  [`install_tabfm()`](https://mattyoreilly.github.io/tabfm/reference/install_tabfm.md)
  sets everything up via reticulate.
- The first
  [`tabfm()`](https://mattyoreilly.github.io/tabfm/reference/tabfm.md)
  call downloads the pretrained weights from the Hugging Face Hub
  (cached afterwards in `~/.cache/huggingface/`). You may see a warning
  about unauthenticated requests; it’s harmless, but you can silence it
  and get faster downloads by putting a free Hugging Face [access
  token](https://huggingface.co/settings/tokens) in your `~/.Renviron`:
  `HF_TOKEN=hf_...`
- TabFM is zero-shot: there is nothing to tune, but also no way to trade
  compute for accuracy on a specific dataset. If you need that, reach
  for [tidymodels](https://www.tidymodels.org).
