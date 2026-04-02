# PROWL
Materials for "[**PAC-Bayesian Reward-Certified Outcome Weighted Learning**](https://arxiv.org/abs/2604.NNNNN)".

## What is This Repository?

This repository includes an implementation of PROWL, ..., as described in our paper. It also contains the numerical experiments and actual data experiments presented in the paper.

### Requirements and Setup
```
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
- `python experiments/run_num_experiments.py`

- `python experiments/run_elaia_real_data.py`

### File Description

- pac_owl/estimators
  - owl.py
    - Citation: Zhao, Y.-Q., Zeng, D., Laber, E. B., and Kosorok, M. R. (2015). New statistical learning methods for estimating optimal dynamic treatment regimes. J. Amer. Statist. Assoc., 110(510):583--598.
  - rwl.py
    - Citation: Zhou, X., Mayer-Hamblett, N., Khan, U., and Kosorok, M. R. (2017). Residual weighted learning for estimating individualized treatment rules. J. Amer. Statist. Assoc., 112(517):169-187.
  - q_learning.py
    - Citation: Qian, M. and Murphy, S.A. (2011). Performance guarantees for individualized treatment rules. Ann. Statist., 39(2):1180-1210.
    - Citation: Schulte, P. J., Tsiatis, A. A., Laber, E. B., and Davidian, M. (2014). Q- and A-learning methods for estimating optimal dynamic treatment regimes. Statist. Sci., 29(4):640-661.
  - policy_tree.py
    - Citation: Sverdrup, E., Kanodia, A., Zhou, Z., Athey, S., and Wager, S. (2020). policytree: Policy learning via doubly robust empirical welfare maximization over trees. J. Open Source Softw., 5(50):2232.
  - prowl.py
    - __This is our method.__

## Citation
```
@article{ishikawa2026pac,
    author={Ishikawa, Yuya and Tamano, Shu},
    journal={arXiv preprint arXiv:2604.NNNNN},
    title={{PAC-Bayesian} Reward-Certified Outcome Weighted Learning},
    year={2025},
}
```

## Contact

If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp