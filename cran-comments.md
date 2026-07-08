# cran-comments

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

## Notes for reviewers

* All examples for `tabfm()` and `predict.tabfm()` are wrapped in
  `\dontrun{}` because running them requires a local Python installation
  with the 'TabFM' package plus a one-time multi-hundred-MB download of
  pretrained model weights. For the same reason the vignette is not
  evaluated at build time and the integration tests are opt-in via the
  `TABFM_TEST_FULL` environment variable; the remaining unit tests run
  everywhere without Python.
* `install_tabfm()` creates a Python virtualenv and downloads packages
  only when the user calls it explicitly; nothing is written to the
  user's home directory on load, attach, or during checks.
