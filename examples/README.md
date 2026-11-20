# Examples

Each runs against the hosted service: log in first with `gf login`, or set
`GF_API_KEY`.

- `churn/`: a predictor directory. Run `gf check examples/churn`, then
  `gf eval examples/churn` and `gf push examples/churn`.
- `adhoc.py`: score rows with a catalog model and context rows sent with the call.
- `estimators.py`: the scikit-learn style `Classifier` and `Forecaster`.
