# Examples

Each runs against the hosted service: run `gf.login()` first, or set
`GF_API_KEY`.

- `predictors.py`: predictors as code, deployed with `gf deploy predictors.py`,
  or run as a script, with `gf.predictors.deploy`.
- `adhoc.py`: score rows with a catalog model, the context sent once by digest.
- `estimators.py`: the scikit-learn style `Classifier` and `Forecaster`.
