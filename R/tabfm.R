# ponytail: thin reticulate wrapper, one file; model loads once per tabfm()
# call and stays in memory for repeated predict()s

tabfm_py <- NULL

# loaded models, keyed by task/backend: checkpoint load + JAX compilation is
# the expensive part, so pay it once per session, not once per fit
the <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  reticulate::use_virtualenv("r-tabfm", required = FALSE)
  tabfm_py <<- reticulate::import("tabfm", delay_load = TRUE)
}

tabfm_model <- function(task, backend) {
  key <- paste(task, backend, sep = "/")
  if (is.null(the[[key]])) {
    quiet_hf_nag()
    the[[key]] <- tabfm_py[[paste0("tabfm_v1_0_0_", backend)]]$load(model_type = task)
  }
  the[[key]]
}

# The HF Hub nags about unauthenticated downloads on every weights load, via
# both warnings and logging. It's advisory only (rate limits), and the fix
# (HF_TOKEN) is documented in the README, so drop just that message.
quiet_hf_nag <- function() {
  if (isTRUE(the$quieted)) return(invisible())
  reticulate::py_run_string(paste(
    "import warnings, logging",
    "warnings.filterwarnings('ignore', message='.*unauthenticated requests.*')",
    "logging.getLogger('huggingface_hub').addFilter(",
    "    lambda r: 'unauthenticated requests' not in r.getMessage())",
    sep = "\n"
  ))
  the$quieted <- TRUE
  invisible()
}

#' Install the TabFM Python package
#'
#' @description
#' `install_tabfm()` installs the underlying
#' [TabFM Python package](https://github.com/google-research/tabfm) into a
#' dedicated virtualenv via [reticulate::py_install()]. You only need to run
#' it once per machine (or once per backend).
#'
#' @param backend Compute backend: `"jax"` (the default, CPU) or `"pytorch"`
#'   (CPU/GPU).
#' @param envname Name of the virtualenv to install into. The default,
#'   `"r-tabfm"`, is discovered automatically by reticulate when the package
#'   loads.
#' @param ... Additional arguments passed on to [reticulate::py_install()].
#' @returns `NULL`, invisibly. Called for its side effect.
#' @family setup
#' @examples
#' \dontrun{
#' install_tabfm()            # JAX backend, CPU
#' install_tabfm("pytorch")   # PyTorch backend, CPU/GPU
#' }
#' @seealso [tabfm()] to fit a model once installation is complete.
#' @export
install_tabfm <- function(backend = c("jax", "pytorch"), envname = "r-tabfm", ...) {
  backend <- match.arg(backend)

  # tabfm needs Python >= 3.11; drop any stale env built with an older one
  # (e.g. the macOS system Python 3.9)
  if (reticulate::virtualenv_exists(envname)) {
    py <- reticulate::virtualenv_python(envname)
    ver <- sub("^Python ", "", system2(py, "--version", stdout = TRUE))
    if (numeric_version(ver) < "3.11") {
      message("Recreating '", envname, "': its Python ", ver,
              " is too old for tabfm (needs >= 3.11).")
      reticulate::virtualenv_remove(envname, confirm = FALSE)
    }
  }
  if (!reticulate::virtualenv_exists(envname)) {
    # GUI R sessions often don't see Homebrew/pyenv Pythons on their PATH;
    # fall back to a reticulate-managed Python rather than erroring
    if (is.null(reticulate::virtualenv_starter(">=3.11"))) {
      message("No Python >= 3.11 found; installing one via reticulate ",
              "(one-time, may take a few minutes).")
      reticulate::install_python()
    }
    reticulate::virtualenv_create(envname, version = ">=3.11")
  }

  reticulate::py_install(
    paste0("tabfm[", backend, "] @ git+https://github.com/google-research/tabfm.git"),
    envname = envname, pip = TRUE, ...
  )
  invisible(NULL)
}

#' Fit a zero-shot TabFM model
#'
#' @description
#' `tabfm()` loads the pretrained TabFM v1.0.0 weights and conditions them on
#' your training data. Because TabFM learns in-context — a single forward
#' pass over the table — there is no feature engineering, no hyperparameter
#' tuning, and no retraining.
#'
#' Loading the weights is the slow part, so it happens once per session for
#' each `task`/`backend` combination and the model is reused by later fits;
#' repeated calls to [predict()][predict.tabfm] are cheap.
#'
#' Fitted models survive [saveRDS()]/[readRDS()]: because a zero-shot fit
#' holds no learned state beyond the training data, a restored fit is
#' re-conditioned on its stored data automatically the first time you call
#' [predict()][predict.tabfm] in the new session (which also pays that
#' session's one-time weights load).
#'
#' @param x A data frame of predictors. Character and factor columns are
#'   handled natively by TabFM; no dummy coding is needed.
#' @param y The response: a character vector or factor for classification, a
#'   numeric vector for regression. Must have one value per row of `x`.
#' @param task Type of model to load: `"classification"` (the default) or
#'   `"regression"`.
#' @param backend Compute backend: `"jax"` or `"pytorch"`. Must match the
#'   backend you installed with [install_tabfm()].
#' @returns A model object of class `"tabfm"`: a list carrying the training
#'   data (`x`, `y`), `task`, `backend`, `features` (the training column
#'   names), and a cache holding the fitted Python estimator. Use
#'   [predict()][predict.tabfm] to generate predictions from it.
#' @seealso [predict.tabfm()] to make predictions; [install_tabfm()] for
#'   one-time setup.
#' @family modelling
#' @examples
#' \dontrun{
#' fit <- tabfm(iris[-5], iris$Species)
#' fit
#'
#' predict(fit, head(iris[-5]))
#' predict(fit, head(iris[-5]), type = "prob")
#'
#' fit <- tabfm(mtcars[-1], mtcars$mpg, task = "regression")
#' predict(fit, head(mtcars[-1]))
#' }
#' @export
tabfm <- function(x, y, task = c("classification", "regression"),
                  backend = c("jax", "pytorch")) {
  task <- match.arg(task)
  backend <- match.arg(backend)
  if (!is.data.frame(x)) {
    stop("`x` must be a data frame, not a ", class(x)[1], ".")
  }
  if (nrow(x) != length(y)) {
    stop("`y` must have one value per row of `x`: ",
         "`x` has ", nrow(x), " rows but `y` has length ", length(y), ".")
  }
  if (is.factor(y)) y <- as.character(y)

  obj <- structure(
    list(
      x = x, y = y, task = task, backend = backend, features = names(x),
      # est lives in an environment so a refit after readRDS() persists
      cache = new.env(parent = emptyenv())
    ),
    class = "tabfm"
  )
  obj$cache$est <- new_est(obj)
  obj
}

new_est <- function(object) {
  model <- tabfm_model(object$task, object$backend)
  est <- if (object$task == "classification") {
    tabfm_py$TabFMClassifier(model = model)
  } else {
    tabfm_py$TabFMRegressor(model = model)
  }
  est$fit(object$x, object$y)
  est
}

# A TabFM fit holds no learned state beyond the training data, so a fit
# restored with readRDS() (whose Python object is a dead pointer) can be
# rebuilt transparently by re-conditioning on the stored data.
tabfm_est <- function(object) {
  est <- object$cache$est
  if (is.null(est) || reticulate::py_is_null_xptr(est)) {
    est <- new_est(object)
    object$cache$est <- est
  }
  est
}

#' Predict from a TabFM model
#'
#' @description
#' Generate predictions — class labels, class probabilities, or numeric
#' values — from a model fitted with [tabfm()].
#'
#' @param object A `"tabfm"` object returned by [tabfm()].
#' @param newdata A data frame with exactly the same columns as the training
#'   data.
#' @param type What to return: `"response"` (the default) for class labels or
#'   numeric predictions, or `"prob"` for a matrix of class probabilities
#'   (classification only).
#' @param ... Ignored, for compatibility with the [predict()][stats::predict]
#'   generic.
#' @returns For `type = "response"`, a vector with one prediction per row of
#'   `newdata`: character class labels for classification, numeric values for
#'   regression. For `type = "prob"`, a numeric matrix with one row per row of
#'   `newdata` and one column per class, named by class.
#' @seealso [tabfm()]
#' @family modelling
#' @examples
#' \dontrun{
#' fit <- tabfm(iris[-5], iris$Species)
#'
#' # class labels
#' predict(fit, head(iris[-5]))
#'
#' # class probabilities, one column per class
#' predict(fit, head(iris[-5]), type = "prob")
#' }
#' @importFrom stats predict
#' @export
predict.tabfm <- function(object, newdata, type = c("response", "prob"), ...) {
  type <- match.arg(type)
  if (type == "prob" && object$task != "classification") {
    stop("`type = \"prob\"` is only available for classification models.")
  }
  if (!is.data.frame(newdata)) {
    stop("`newdata` must be a data frame, not a ", class(newdata)[1], ".")
  }
  if (!identical(names(newdata), object$features)) {
    stop("`newdata` must have the same columns as the training data, ",
         "in the same order: ", paste(object$features, collapse = ", "), ".")
  }

  est <- tabfm_est(object)
  if (type == "prob") {
    p <- est$predict_proba(newdata)
    colnames(p) <- as.character(est$classes_)
    p
  } else {
    # numpy 1-d arrays come back as R arrays with a dim attribute; drop it
    as.vector(est$predict(newdata))
  }
}

#' @export
print.tabfm <- function(x, ...) {
  n <- length(x$features)
  cat("TabFM v1.0.0 zero-shot", x$task, "model |",
      n, ngettext(n, "feature", "features"), "\n")
  invisible(x)
}
