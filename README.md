# PROWL

Materials for "**[PAC-Bayesian Reward-Certified Outcome Weighted Learning](https://arxiv.org/abs/2604.NNNNN)**".

## What is This Repository?

This repository includes an implementation of PROWL, a PAC-Bayesian reward-certified extension of outcome weighted learning that uses one-sided uncertainty certificates to replace optimistic proxy rewards with certified lower rewards and learn individualized treatment rules by maximizing a finite-sample lower bound on latent policy value, together with the numerical experiments and actual-data experiments reported in the paper.

### Requirements and Setup

```sh
# clone the repository
git clone git@github.com:shutech2001/PROWL.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter.
poetry env info --path
```

### Executing Experiments

#### Numerical Experiments

```sh
# run all synthetic experiments in the paper
poetry run python experiments/run_num_experiments.py --experiment all

# run one synthetic block at a time
poetry run python experiments/run_num_experiments.py --experiment rho_sweep
poetry run python experiments/run_num_experiments.py --experiment n_sweep
poetry run python experiments/run_num_experiments.py --experiment certificate_diagnostics
poetry run python experiments/run_num_experiments.py --experiment split_free_ablation
```

Arguments for `experiments/run_num_experiments.py`:

**General**

- `--experiment`
  - experiment block to run: `rho_sweep`, `n_sweep`, `certificate_diagnostics`, `split_free_ablation`, or `all` (default: `all`).
- `--output-dir`
  - directory where synthetic results, manifests, tables, and plots are written (default: `outputs_tmp/prowl_targeted_experiments`).
- `--n-jobs`
  - number of parallel workers (default: `1`).
- `--progress`
  - progress display mode: `auto`, `tqdm`, `plain`, or `quiet` (default: `auto`).
- `--plots-only`
  - regenerate plots and summaries from cached result files without rerunning simulations (default: `False`).
- `--n-test`
  - independent test-sample size used for evaluation (default: `10000`).
- `--seed`
  - global random seed (default: `42`).

**Certificate and Propensity**

- `--certificate-mode`
  - certificate source used by the simulator, typically `oracle` or `estimated` (default: `oracle`).
- `--certificate-scale`
  - multiplicative scale applied to the oracle certificate before clipping to `[0, 1]` (default: `1.0`).
- `--n-certificate-auxiliary`
  - auxiliary sample size used when fitting an estimated certificate (default: `400`).
- `--propensity-mode`
  - logging-propensity specification passed to the simulator; `auto` uses scenario-dependent defaults (default: `auto`).
- `--propensity-strength`
  - strength parameter for non-constant logging propensities (default: `1.0`).

**Common Model / PROWL Settings**

- `--delta`
  - PAC-Bayes confidence level used in exact-value lower-confidence-bound tuning (default: `0.1`).
- `--prior-sd`
  - prior standard deviation for the PROWL particle library (default: `5.0`).
- `--score-bound`
  - bound used to squash linear scores in bounded-score policies (default: `3.0`).
- `--penalty-grid`
  - comma-separated penalty grid for OWL and RWL tuning (default: `0.001,0.01,0.1,1.0`).
- `--q-penalty-grid`
  - comma-separated ridge-penalty grid for Q-learning tuning (default: `0.001,0.01,0.1,1.0`).
- `--prowl-deployment`
  - deployment rule used when evaluating PROWL: `auto`, `map`, `mean_rule`, or `gibbs`; `auto` resolves to `map` in this script (default: `auto`).
- `--policy-tree-depth`
  - maximum depth of the Policy Tree baseline (default: `2`).
- `--policy-tree-min-node-size`
  - minimum number of observations allowed in a Policy Tree leaf (default: `20`).
- `--policy-tree-split-step`
  - stride used when subsampling candidate split positions in Policy Tree fitting (default: `25`).
- `--policy-tree-max-features`
  - maximum number of features inspected per tree split; `0` means no restriction (default: `0`).
- `--n-anchor-particles`
  - number of anchor particles used to seed the PROWL candidate library (default: `2`).
- `--n-prior-particles`
  - number of prior particles drawn for the PROWL candidate library (default: `32`).
- `--n-local-particles`
  - number of local perturbation particles generated around each anchor (default: `4`).
- `--local-particle-scale`
  - scale of local perturbations around anchor particles (default: `0.3`).
- `--nuisance-l2-penalty`
  - ridge penalty used for nuisance reward regressions (default: `1e-6`).
- `--eta-grid`
  - comma-separated grid of PAC-Bayes learning rates for PROWL (default: `0.125,0.25,0.5,1.0,2.0,4.0,8.0`).
- `--gamma-grid`
  - comma-separated grid of exact-value temperature parameters for PROWL lower-confidence-bound selection (default: `0.125,0.25,0.5,1.0,2.0,4.0,8.0`).

`**rho_sweep` Settings**

- `--rho-scenario`
  - optional single scenario override for the `rho_sweep` experiment (default: `None`).
- `--rho-scenarios`
  - comma-separated scenario IDs used in the `rho_sweep` experiment (default: `linear_scope_invariant,clinical_triage_nuisance_conflict`).
- `--rho-n-total`
  - logged training-sample size used in the `rho_sweep` experiment (default: `200`).
- `--rho-replications`
  - number of Monte Carlo replications for each `rho` value (default: `30`).
- `--rho-grid`
  - comma-separated uncertainty grid for the `rho_sweep` experiment (default: `0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0`).

`**n_sweep` Settings**

- `--n-sweep-scenarios`
  - comma-separated scenario IDs used in the `n_sweep` experiment (default: `linear_scope_invariant,clinical_triage_nuisance_conflict`).
- `--n-sweep-grid`
  - comma-separated sample-size grid for the `n_sweep` experiment (default: `100,200,500,1000,2000`).
- `--n-sweep-replications`
  - number of Monte Carlo replications for each sample size (default: `30`).
- `--n-sweep-rho`
  - fixed uncertainty level used in the `n_sweep` experiment (default: `1.5`).

`**certificate_diagnostics` Settings**

- `--certificate-diagnostic-scenarios`
  - comma-separated scenario IDs used in certificate diagnostics (default: `linear_scope_invariant,clinical_triage_nuisance_conflict`).
- `--certificate-diagnostic-n-total`
  - logged training-sample size used in certificate diagnostics (default: `1000`).
- `--certificate-diagnostic-rho-grid`
  - comma-separated uncertainty grid used in certificate diagnostics (default: `0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0`).
- `--certificate-diagnostic-replications`
  - number of replications per certificate-diagnostic configuration (default: `30`).

`**split_free_ablation` Settings**

- `--split-free-scenarios`
  - comma-separated scenario IDs used in the split-free ablation (default: `linear_scope_invariant,clinical_triage_nuisance_conflict`).
- `--split-free-grid`
  - comma-separated sample-size grid used in the split-free ablation (default: `100,200,500,1000,2000`).
- `--split-free-replications`
  - number of replications per split-free ablation configuration (default: `30`).
- `--split-free-rho`
  - fixed uncertainty level used in the split-free ablation (default: `1.5`).

#### Actual-Data Experiments

```sh
# reproduce the ELAIA-1 analysis
poetry run python experiments/run_elaia_real_data.py
```

Arguments for `experiments/run_elaia_real_data.py`:

**Data and Split Settings**

- `--data-path`
  - path to the deidentified ELAIA-1 CSV file (default: `real_world/AKI-alert-trial/ELAIA-1_deidentified_data_10-6-2020.csv`).
- `--output-dir`
  - directory where actual-data tables, plots, appendix outputs, and manifests are written (default: `outputs/elaia1_real_data`).
- `--n-splits`
  - number of repeated train/test splits (default: `30`).
- `--n-jobs`
  - number of parallel workers (default: `1`).
- `--test-size`
  - held-out fraction in each repeated split (default: `0.30`).
- `--seed`
  - global random seed (default: `20260401`).
- `--subsample-n`
  - optional stratified subsample size; `0` keeps the full dataset (default: `0`).

**Utility / Certificate Settings**

- `--rho`
  - preference-uncertainty radius used in the main actual-data analysis (default: `1.0`).
- `--rho-grid`
  - comma-separated grid used for the appendix `rho`-sensitivity analysis (default: `0,0.5,1,1.5,2`).
- `--main-weights`
  - comma-separated baseline weights for the main 3-component hard-clinical utility (default: `0.60,0.25,0.15`).
- `--main-deltas`
  - comma-separated uncertainty tolerances for the main 3-component utility (default: `0.10,0.05,0.05`).
- `--patient4-weights`
  - comma-separated baseline weights for the 4-component patient-centered sensitivity utility (default: `0.55,0.20,0.15,0.10`).
- `--patient4-deltas`
  - comma-separated uncertainty tolerances for the 4-component patient-centered sensitivity utility (default: `0.10,0.05,0.05,0.05`).

**Tuning and Evaluation Settings**

- `--penalty-grid`
  - comma-separated regularization grid for OWL, Q-learning, and RWL (default: `1e-3,1e-2,1e-1,1`).
- `--eta-grid`
  - comma-separated PROWL learning-rate grid (default: `0.125,0.25,0.5,1,2,4,8`).
- `--gamma-grid`
  - comma-separated exact-value temperature grid for PROWL tuning (default: `0.125,0.25,0.5,1,2,4,8`).
- `--delta`
  - PAC-Bayes confidence level used in PROWL exact-value tuning (default: `0.1`).
- `--prior-sd`
  - prior standard deviation for the PROWL particle library (default: `5.0`).
- `--score-bound`
  - bound used to squash linear scores in bounded-score policies (default: `3.0`).
- `--nuisance-l2-penalty`
  - ridge penalty used in PROWL nuisance regressions (default: `1e-3`).
- `--eval-nuisance-l2-penalty`
  - ridge penalty used in evaluation nuisance models such as AIPW outcome regressions (default: `1e-3`).

**Policy Tree and PROWL Library Settings**

- `--policy-tree-depth`
  - maximum tree depth used when Policy Tree is fitted in auxiliary analyses (default: `2`).
- `--policy-tree-depth-sensitivity`
  - reserved alternative tree depth for sensitivity analyses; currently parsed but not used in the main execution path (default: `3`).
- `--policy-tree-min-node-size`
  - minimum number of observations allowed in a Policy Tree leaf (default: `200`).
- `--policy-tree-split-step`
  - stride used when subsampling candidate split positions in Policy Tree fitting (default: `1`).
- `--policy-tree-max-features`
  - maximum number of features inspected per tree split; `0` means no restriction (default: `0`).
- `--n-anchor-particles`
  - number of anchor particles used to seed the PROWL candidate library (default: `16`).
- `--n-prior-particles`
  - number of prior particles drawn for the PROWL candidate library (default: `256`).
- `--n-local-particles`
  - number of local perturbation particles generated around each anchor (default: `48`).
- `--local-particle-scale`
  - scale of local perturbations around anchor particles (default: `0.30`).

**Optional Appendix Toggles**

- `--skip-rho-sweep`
  - skip the appendix sensitivity analysis over `rho` (default: `False`).
- `--skip-no-hospital-sensitivity`
  - skip the appendix analysis that removes hospital covariates (default: `False`).
- `--skip-deployment-sensitivity`
  - skip the appendix comparison of PROWL deployment rules (default: `False`).
- `--skip-patient-centered-sensitivity`
  - skip the appendix analysis based on the 4-component patient-centered utility (default: `False`).
- `--quick`
  - smoke-test mode that reduces the split count and posterior-library size and disables appendix-heavy analyses (default: `False`).

### File Description

- `README.md`
  - repository overview, environment setup, experiment commands, and file guide.
- `pyproject.toml`
  - project metadata and Poetry dependency specification.
- `poetry.lock`
  - locked dependency versions for reproducible environments.
- `LICENSE`
  - repository license.
- `experiments/run_num_experiments.py`
  - driver for the synthetic experiments in the paper, including the `rho`-sweep, `N`-sweep, certificate diagnostics, split-free ablation, and figure/table generation.
- `experiments/run_elaia_real_data.py`
  - driver for the ELAIA-1 actual-data experiments, repeated-split evaluation, summary tables, and appendix sensitivity analyses.
- `pac_owl/__init__.py`
  - package marker for the `pac_owl` namespace.
- `pac_owl/estimators/__init__.py`
  - re-exports the main estimator classes and fitting utilities used throughout the experiments.
- `pac_owl/estimators/owl.py`
  - linear OWL baseline implemented as weighted hinge-loss classification with cross-validated penalty tuning.
  - Citation: Zhao, Y.-Q., Zeng, D., Laber, E. B., and Kosorok, M. R. (2015). New statistical learning methods for estimating optimal dynamic treatment regimes. J. Amer. Statist. Assoc., 110(510):583--598.
- `pac_owl/estimators/rwl.py`
  - linear residual weighted learning baseline with treatment-free residual regression and smoothed-ramp optimization.
  - Citation: Zhou, X., Mayer-Hamblett, N., Khan, U., and Kosorok, M. R. (2017). Residual weighted learning for estimating individualized treatment rules. J. Amer. Statist. Assoc., 112(517):169-187.
- `pac_owl/estimators/q_learning.py`
  - one-stage linear Q-learning baseline with separate main-effect and blip-effect feature maps.
  - Citation: Qian, M. and Murphy, S. A. (2011). Performance guarantees for individualized treatment rules. Ann. Statist., 39(2):1180-1210.
  - Citation: Schulte, P. J., Tsiatis, A. A., Laber, E. B., and Davidian, M. (2014). Q- and A-learning methods for estimating optimal dynamic treatment regimes. Statist. Sci., 29(4):640-661.
- `pac_owl/estimators/policy_tree.py`
  - depth-limited policy tree baseline together with doubly robust score-matrix utilities.
  - Citation: Sverdrup, E., Kanodia, A., Zhou, Z., Athey, S., and Wager, S. (2020). policytree: Policy learning via doubly robust empirical welfare maximization over trees. J. Open Source Softw., 5(50):2232.
- `pac_owl/estimators/prowl.py`
  - **This is our method.**
  - fits arm-specific or treatment-free nuisance models for certified lower rewards.
  - builds a finite policy-particle library and computes PAC-Bayesian / generalized Bayes posterior weights over candidate rules.
  - tunes the learning rate by exact-value lower confidence bounds and supports MAP, posterior-mean-rule, and Gibbs-style deployment.
- `pac_owl/simulation/__init__.py`
  - re-exports the simulation interfaces used by the experiment scripts.
- `pac_owl/simulation/paper_scenarios.py`
  - synthetic data generator for the paper's two benchmark scenarios, including oracle / estimated certificates, logged-bandit sampling, and oracle evaluation utilities.

## Citation

```bibtex
@article{ishikawa2026pac,
    author={Ishikawa, Yuya and Tamano, Shu},
    journal={arXiv preprint arXiv:2604.NNNNN},
    title={{PAC-Bayesian} Reward-Certified Outcome Weighted Learning},
    year={2026},
}
```

## Contact

If you have any question, please feel free to contact: [tamano-shu212@g.ecc.u-tokyo.ac.jp](mailto:tamano-shu212@g.ecc.u-tokyo.ac.jp)