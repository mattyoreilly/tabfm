# R-side behaviour: input validation, predict() contract, print method.
# None of these touch Python.

test_that("tabfm() validates x and y before touching Python", {
  expect_snapshot(error = TRUE, {
    tabfm(1:3, 1:3)
    tabfm(data.frame(a = 1:2), y = 1:3)
    tabfm(data.frame(a = 1:2), y = 1:2, task = "banana")
    tabfm(data.frame(a = 1:2), y = 1:2, backend = "tensorflow")
  })
})

test_that("predict() validates type, task, and columns before touching Python", {
  expect_snapshot(error = TRUE, {
    predict(fake_fit("regression", "a"), data.frame(a = 1), type = "prob")
    predict(fake_fit(), data.frame(a = 1), type = "banana")
    predict(fake_fit(features = c("a", "b")), data.frame(wrong = 1))
    predict(fake_fit(), "not a data frame")
  })
})

test_that("predict() requires columns in training order", {
  # ponytail: name *order* is enforced, not just the set; relax to
  # setequal + reorder if this bites in practice
  expect_error(
    predict(fake_fit(features = c("a", "b")), data.frame(b = 1, a = 1)),
    "same columns"
  )
})

test_that("tabfm objects survive serialization structurally", {
  obj <- structure(
    list(x = data.frame(a = 1:2), y = c("u", "v"), task = "classification",
         backend = "jax", features = "a", cache = new.env(parent = emptyenv())),
    class = "tabfm"
  )
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(obj, path)
  restored <- readRDS(path)
  expect_identical(restored$x, obj$x)
  expect_identical(restored$y, obj$y)
  expect_true(is.environment(restored$cache))
  expect_null(restored$cache$est)
})

test_that("print() describes the model", {
  expect_snapshot({
    print(fake_fit("classification", features = letters[1:4]))
    print(fake_fit("regression", features = "sqft"))
  })
})

# Integration: the real Python path. Skipped unless install_tabfm() has
# been run on this machine.

test_that("classification round trip works", {
  skip_if_no_tabfm()

  fit <- tabfm(iris[-5], iris$Species)
  expect_s3_class(fit, "tabfm")
  expect_identical(fit$task, "classification")
  expect_identical(fit$features, names(iris)[-5])

  preds <- predict(fit, head(iris[-5]))
  expect_length(preds, 6L)
  expect_true(all(preds %in% levels(iris$Species)))

  probs <- predict(fit, head(iris[-5]), type = "prob")
  expect_true(is.matrix(probs))
  expect_identical(dim(probs), c(6L, 3L))
  expect_setequal(colnames(probs), levels(iris$Species))
  expect_equal(rowSums(probs), rep(1, 6), tolerance = 1e-4)
  expect_true(all(probs >= 0 & probs <= 1))

  # predicted label agrees with the highest-probability column
  expect_identical(preds, colnames(probs)[max.col(probs)])
})

test_that("fits survive saveRDS/readRDS and predict identically", {
  skip_if_no_tabfm()

  fit <- tabfm(iris[-5], iris$Species)
  before <- predict(fit, head(iris[-5]))

  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(fit, path)
  restored <- readRDS(path)

  # the Python estimator does not survive serialization ...
  expect_true(reticulate::py_is_null_xptr(restored$cache$est))
  # ... but predict() re-conditions on the stored data transparently
  expect_identical(predict(restored, head(iris[-5])), before)
})

test_that("regression round trip works", {
  skip_if_no_tabfm()

  fit <- tabfm(mtcars[-1], mtcars$mpg, task = "regression")
  preds <- predict(fit, head(mtcars[-1]))
  expect_type(preds, "double")
  expect_length(preds, 6L)
  expect_true(all(is.finite(preds)))
})

test_that("character predictors are handled without encoding", {
  skip_if_no_tabfm()

  x <- data.frame(
    age = c(25, 45, 35, 50),
    job = c("engineer", "manager", "engineer", "manager")
  )
  y <- c("low", "high", "low", "high")
  fit <- tabfm(x, y)
  expect_true(all(predict(fit, x) %in% c("low", "high")))
})
