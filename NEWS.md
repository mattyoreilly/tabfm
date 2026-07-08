# tabfm 0.1.0

* Initial release.
* Fitted models survive `saveRDS()`/`readRDS()`: a restored fit
  re-conditions on its stored training data automatically on the first
  `predict()` of the new session.
* `tabfm()` fits zero-shot classification and regression models using
  Google Research's TabFM foundation model.
* `predict()` method returns class labels, class probabilities
  (`type = "prob"`), or numeric predictions.
* `install_tabfm()` performs one-time setup of the underlying Python
  package into a dedicated virtualenv, requiring Python >= 3.11.
