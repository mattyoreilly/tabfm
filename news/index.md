# Changelog

## tabfm 0.1.0

- Initial release.
- Fitted models survive
  [`saveRDS()`](https://rdrr.io/r/base/readRDS.html)/[`readRDS()`](https://rdrr.io/r/base/readRDS.html):
  a restored fit re-conditions on its stored training data automatically
  on the first [`predict()`](https://rdrr.io/r/stats/predict.html) of
  the new session.
- [`tabfm()`](https://mattyoreilly.github.io/tabfm/reference/tabfm.md)
  fits zero-shot classification and regression models using Google
  Research’s TabFM foundation model.
- [`predict()`](https://rdrr.io/r/stats/predict.html) method returns
  class labels, class probabilities (`type = "prob"`), or numeric
  predictions.
- [`install_tabfm()`](https://mattyoreilly.github.io/tabfm/reference/install_tabfm.md)
  performs one-time setup of the underlying Python package into a
  dedicated virtualenv, requiring Python \>= 3.11.
