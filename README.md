# pac-owl
Materials for "PAC-Bayes bound based Outcome Weighted Learning"

## Requirements and Setup
```
# clone the repository
git clone git@github.com:shutech2001/DOLCE.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter.
poetry env info --path
```

## Numerical Experiments (PAC-Bayes robust OWL)

The script below implements:
- S0-S8-style simulation scenarios (including certificate miscalibration variants)
- `owl_standard`, `owl_clipped`, `owl_robust_map`
- `pac_nominal_gibbs` (U=0 analogue) and `pac_robust_gibbs`
- Metrics for true value, lower-tail safety (`q05`, `CVaR10`), uncertainty exposure, and PAC-Bayes LCB coverage/tightness
- Automatic plot generation (`output_dir/plots/*.png` and `*.pdf`)

### Run a smoke test
```bash
poetry run python experiments/run_pac_owl_experiments.py \
  --scenarios S0,S2,S5 \
  --n-train 200,500 \
  --replications 3 \
  --n-eval 5000 \
  --etas 0.3,1.0 \
  --deltas 0.2,0.1 \
  --output-dir outputs_smoke
```

### Run the full suite
```bash
poetry run python experiments/run_pac_owl_experiments.py \
  --scenarios all \
  --n-train 200,500,1000 \
  --replications 50 \
  --n-eval 30000 \
  --output-dir outputs/main
```

To skip plotting:
```bash
poetry run python experiments/run_pac_owl_experiments.py --no-make-plots
```

### Outputs
- `manifest.json`: run configuration and scenario definitions
- `raw_metrics.csv`: per-replication raw results
- `summary_by_setting.csv`: grouped means/std by `(scenario, n_train, method, eta, delta)`
- `lcb_summary.csv`: grouped LCB diagnostics for robust PAC-Bayes method
- `plots/fig_n_vs_true_value.(png|pdf)`: n vs true value
- `plots/fig_coverage_vs_n.(png|pdf)`: LCB coverage vs n
- `plots/fig_exposure_vs_true_scatter.(png|pdf)`: uncertainty exposure vs true value
- `plots/fig_eta_sweep_tradeoff.(png|pdf)`: eta sweep trade-off (true value vs exposure)
