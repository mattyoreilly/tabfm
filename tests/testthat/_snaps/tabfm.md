# tabfm() validates x and y before touching Python

    Code
      tabfm(1:3, 1:3)
    Condition
      Error in `tabfm()`:
      ! `x` must be a data frame, not a integer.
    Code
      tabfm(data.frame(a = 1:2), y = 1:3)
    Condition
      Error in `tabfm()`:
      ! `y` must have one value per row of `x`: `x` has 2 rows but `y` has length 3.
    Code
      tabfm(data.frame(a = 1:2), y = 1:2, task = "banana")
    Condition
      Error in `match.arg()`:
      ! 'arg' should be one of "classification", "regression"
    Code
      tabfm(data.frame(a = 1:2), y = 1:2, backend = "tensorflow")
    Condition
      Error in `match.arg()`:
      ! 'arg' should be one of "jax", "pytorch"

# predict() validates type, task, and columns before touching Python

    Code
      predict(fake_fit("regression", "a"), data.frame(a = 1), type = "prob")
    Condition
      Error in `predict.tabfm()`:
      ! `type = "prob"` is only available for classification models.
    Code
      predict(fake_fit(), data.frame(a = 1), type = "banana")
    Condition
      Error in `match.arg()`:
      ! 'arg' should be one of "response", "prob"
    Code
      predict(fake_fit(features = c("a", "b")), data.frame(wrong = 1))
    Condition
      Error in `predict.tabfm()`:
      ! `newdata` must have the same columns as the training data, in the same order: a, b.
    Code
      predict(fake_fit(), "not a data frame")
    Condition
      Error in `predict.tabfm()`:
      ! `newdata` must be a data frame, not a character.

# print() describes the model

    Code
      print(fake_fit("classification", features = letters[1:4]))
    Output
      TabFM v1.0.0 zero-shot classification model | 4 features 
    Code
      print(fake_fit("regression", features = "sqft"))
    Output
      TabFM v1.0.0 zero-shot regression model | 1 feature 

