# Real-model integration tests are opt-in: a fit loads the pretrained
# weights, which is far too slow for routine test runs and R CMD check.
# Run them deliberately with: Sys.setenv(TABFM_TEST_FULL = "1")
skip_if_no_tabfm <- function() {
  testthat::skip_if_not(
    nzchar(Sys.getenv("TABFM_TEST_FULL")),
    "set TABFM_TEST_FULL=1 to run slow integration tests"
  )
  have <- tryCatch(
    reticulate::py_module_available("tabfm"),
    error = function(e) FALSE
  )
  testthat::skip_if_not(have, "tabfm Python module not available")
}

# A fitted-model stand-in for exercising R-side predict() validation
# without Python.
fake_fit <- function(task = "classification", features = c("a", "b")) {
  structure(list(task = task, features = features), class = "tabfm")
}
