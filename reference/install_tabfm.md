# Install the TabFM Python package

`install_tabfm()` installs the underlying [TabFM Python
package](https://github.com/google-research/tabfm) into a dedicated
virtualenv via
[`reticulate::py_install()`](https://rstudio.github.io/reticulate/reference/py_install.html).
You only need to run it once per machine (or once per backend).

## Usage

``` r
install_tabfm(backend = c("jax", "pytorch"), envname = "r-tabfm", ...)
```

## Arguments

- backend:

  Compute backend: `"jax"` (the default, CPU) or `"pytorch"` (CPU/GPU).

- envname:

  Name of the virtualenv to install into. The default, `"r-tabfm"`, is
  discovered automatically by reticulate when the package loads.

- ...:

  Additional arguments passed on to
  [`reticulate::py_install()`](https://rstudio.github.io/reticulate/reference/py_install.html).

## Value

`NULL`, invisibly. Called for its side effect.

## See also

[`tabfm()`](https://mattyoreilly.github.io/tabfm/reference/tabfm.md) to
fit a model once installation is complete.

## Examples

``` r
if (FALSE) { # \dontrun{
install_tabfm()            # JAX backend, CPU
install_tabfm("pytorch")   # PyTorch backend, CPU/GPU
} # }
```
