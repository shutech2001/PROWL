#!/usr/bin/env python3
"""Numerical experiments for PAC-Bayes based robust Outcome Weighted Learning.

This script implements:
1) Simulation scenarios (S0-S8 style) with controllable reward misspecification.
2) Baselines: standard OWL, clipped OWL, robust-MAP OWL.
3) PAC-Bayes learners: nominal Gibbs (U=0 analogue) and robust Gibbs.
4) Evaluation metrics for value, lower-tail safety, uncertainty exposure, and
   PAC-Bayes LCB coverage/tightness.

Only numpy + Python standard library are used so the script is portable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


NUM_EPS = 1e-12
METHOD_LINE_STYLES: dict[str, dict[str, Any]] = {
    "owl_standard": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "owl_clipped": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "owl_robust_map": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    "pac_nominal_gibbs": {"color": "#9467bd", "marker": "D", "linestyle": ":"},
    "pac_robust_gibbs": {"color": "#d62728", "marker": "v", "linestyle": "-"},
    "oracle_true_owl": {"color": "#7f7f7f", "marker": "P", "linestyle": "--"},
}
DELTA_LINE_STYLES: list[dict[str, Any]] = [
    {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    {"color": "#d62728", "marker": "D", "linestyle": ":"},
]


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    description: str
    p: int = 20
    cov_rho: float = 0.3
    propensity_mode: str = "rct"  # "rct" | "obs"
    rct_rho: float = 0.5
    obs_scale: float = 1.0
    eps_floor: float = 0.2
    misspec_mode: str = "none"  # "none" | "homo" | "hetero" | "treatment"
    bias_scale: float = 0.0
    outlier_prob: float = 0.0
    outlier_scale: float = 0.0
    true_concentration: float = 25.0
    nonlinear_reward: bool = False
    cert_mode: str = "oracle"  # "oracle" | "estimated"
    cert_scale: float = 1.0


@dataclass
class LoggedData:
    x: np.ndarray
    a: np.ndarray
    pi: np.ndarray
    pi1: np.ndarray
    r_obs: np.ndarray
    r_star: np.ndarray
    u: np.ndarray
    underline_r: np.ndarray
    mu1: np.ndarray
    mu_minus1: np.ndarray
    misspec_bias: np.ndarray
    epsilon: float


@dataclass
class FeatureMap:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        return np.concatenate([np.ones((x.shape[0], 1)), z], axis=1)


def parse_float_list(raw: str) -> list[float]:
    return [float(tok.strip()) for tok in raw.split(",") if tok.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(tok.strip()) for tok in raw.split(",") if tok.strip()]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits)
    z = np.exp(logits - m)
    return z / np.sum(z)


def safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def safe_cvar(values: np.ndarray, alpha: float = 0.1) -> float:
    if values.size == 0:
        return float("nan")
    threshold = np.quantile(values, alpha)
    tail = values[values <= threshold]
    if tail.size == 0:
        return float(threshold)
    return float(np.mean(tail))


def nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    mask = ~np.isnan(arr)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(arr[mask]))


def nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    mask = ~np.isnan(arr)
    n = int(np.sum(mask))
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    return float(np.std(arr[mask], ddof=1))


def scenario_library() -> dict[str, ScenarioConfig]:
    return {
        "S0": ScenarioConfig(
            name="S0",
            description="No misspecification sanity check (R=R*), U=0.",
            propensity_mode="rct",
            rct_rho=0.5,
            eps_floor=0.5,
            misspec_mode="none",
            bias_scale=0.0,
            cert_mode="oracle",
            cert_scale=1.0,
        ),
        "S1": ScenarioConfig(
            name="S1",
            description="Homoskedastic optimistic misspecification, oracle certificate.",
            propensity_mode="rct",
            rct_rho=0.5,
            eps_floor=0.5,
            misspec_mode="homo",
            bias_scale=0.20,
            cert_mode="oracle",
            cert_scale=1.0,
        ),
        "S2": ScenarioConfig(
            name="S2",
            description="Heteroskedastic policy-dependent misspecification hotspot.",
            propensity_mode="obs",
            obs_scale=1.0,
            eps_floor=0.2,
            misspec_mode="hetero",
            bias_scale=0.40,
            cert_mode="oracle",
            cert_scale=1.0,
        ),
        "S3": ScenarioConfig(
            name="S3",
            description="Treatment-dependent misspecification strength.",
            propensity_mode="rct",
            rct_rho=0.7,
            eps_floor=0.3,
            misspec_mode="treatment",
            bias_scale=0.35,
            cert_mode="oracle",
            cert_scale=1.0,
        ),
        "S4": ScenarioConfig(
            name="S4",
            description="Heavy-tail/outlier stress with uncertainty spikes.",
            propensity_mode="obs",
            obs_scale=1.1,
            eps_floor=0.2,
            misspec_mode="hetero",
            bias_scale=0.35,
            outlier_prob=0.03,
            outlier_scale=0.70,
            true_concentration=3.0,
            cert_mode="oracle",
            cert_scale=1.0,
        ),
        "S5": ScenarioConfig(
            name="S5",
            description="Weak positivity (small epsilon) with heteroskedastic misspec.",
            propensity_mode="obs",
            obs_scale=2.0,
            eps_floor=0.05,
            misspec_mode="hetero",
            bias_scale=0.40,
            cert_mode="oracle",
            cert_scale=1.0,
        ),
        "S6_under": ScenarioConfig(
            name="S6_under",
            description="Certificate underestimation (anti-conservative, c<1).",
            propensity_mode="obs",
            obs_scale=1.0,
            eps_floor=0.2,
            misspec_mode="hetero",
            bias_scale=0.40,
            cert_mode="oracle",
            cert_scale=0.6,
        ),
        "S6_over": ScenarioConfig(
            name="S6_over",
            description="Certificate overestimation (conservative, c>1).",
            propensity_mode="obs",
            obs_scale=1.0,
            eps_floor=0.2,
            misspec_mode="hetero",
            bias_scale=0.40,
            cert_mode="oracle",
            cert_scale=1.4,
        ),
        "S7": ScenarioConfig(
            name="S7",
            description="Estimated certificate (cross-fitted surrogate).",
            propensity_mode="obs",
            obs_scale=1.0,
            eps_floor=0.2,
            misspec_mode="hetero",
            bias_scale=0.40,
            cert_mode="estimated",
            cert_scale=1.0,
        ),
        "S8": ScenarioConfig(
            name="S8",
            description="Model misspecification: nonlinear true boundary vs linear learner.",
            propensity_mode="obs",
            obs_scale=1.0,
            eps_floor=0.2,
            misspec_mode="hetero",
            bias_scale=0.35,
            nonlinear_reward=True,
            cert_mode="oracle",
            cert_scale=1.0,
        ),
    }


def ar1_covariance(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    return rho ** np.abs(np.subtract.outer(idx, idx))


def sample_covariates(n: int, p: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    cov = ar1_covariance(p, rho)
    return rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)


def true_reward_means(x: np.ndarray, nonlinear_reward: bool) -> tuple[np.ndarray, np.ndarray]:
    x1 = x[:, 0]
    x2 = x[:, 1] if x.shape[1] > 1 else 0.0
    x3 = x[:, 2] if x.shape[1] > 2 else 0.0
    x4 = x[:, 3] if x.shape[1] > 3 else 0.0

    base = 0.4 * x1 - 0.3 * x2 + 0.2 * x3
    tau = -0.9 * np.tanh(x1) + 0.35 * x2

    if nonlinear_reward:
        base = base + 0.5 * np.sin(x1) - 0.25 * x2 * x3
        tau = tau + 0.35 * np.sin(x3) + 0.2 * x1 * x4

    mu1 = sigmoid(base + tau)
    mu_minus1 = sigmoid(base - tau)
    return mu1, mu_minus1


def propensity_probs(x: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    if cfg.propensity_mode == "rct":
        return np.full(x.shape[0], cfg.rct_rho, dtype=float)
    logits = cfg.obs_scale * (0.9 * x[:, 0] - 0.6 * x[:, 1] + 0.25 * x[:, 2])
    pi1 = sigmoid(logits)
    return np.clip(pi1, cfg.eps_floor, 1.0 - cfg.eps_floor)


def sample_true_reward(mu: np.ndarray, concentration: float, rng: np.random.Generator) -> np.ndarray:
    kappa = max(concentration, 1.0)
    alpha = np.clip(mu * kappa, 1e-3, None)
    beta = np.clip((1.0 - mu) * kappa, 1e-3, None)
    return rng.beta(alpha, beta)


def sample_misspec_bias(
    x: np.ndarray, a: np.ndarray, cfg: ScenarioConfig, rng: np.random.Generator
) -> np.ndarray:
    n = x.shape[0]
    if cfg.misspec_mode == "none":
        bias = np.zeros(n, dtype=float)
    elif cfg.misspec_mode == "homo":
        bias = cfg.bias_scale * rng.uniform(0.0, 1.0, size=n)
    elif cfg.misspec_mode == "hetero":
        hotspot = sigmoid(2.2 * x[:, 0] + 0.5 * x[:, 1])
        bias = cfg.bias_scale * (0.1 + 0.9 * hotspot) * (a == 1).astype(float)
        bias += 0.05 * cfg.bias_scale * (a == -1).astype(float)
    elif cfg.misspec_mode == "treatment":
        arm1 = (0.2 + 0.8 * sigmoid(1.8 * x[:, 0])) * (a == 1).astype(float)
        armm1 = 0.2 * (a == -1).astype(float)
        bias = cfg.bias_scale * (arm1 + armm1)
    else:
        raise ValueError(f"Unknown misspec_mode: {cfg.misspec_mode}")

    if cfg.outlier_prob > 0.0 and cfg.outlier_scale > 0.0:
        spikes = (rng.uniform(0.0, 1.0, size=n) < cfg.outlier_prob).astype(float)
        bias = bias + cfg.outlier_scale * spikes

    return np.clip(bias, 0.0, 1.0)


def fit_feature_map(x: np.ndarray) -> FeatureMap:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return FeatureMap(mean=mean, std=std)


def fit_ridge_regression(
    x: np.ndarray, y: np.ndarray, l2: float = 1e-3
) -> np.ndarray:
    d = x.shape[1]
    a = x.T @ x + l2 * np.eye(d)
    b = x.T @ y
    return np.linalg.solve(a, b)


def estimate_certificate_crossfit(
    x: np.ndarray,
    a: np.ndarray,
    r_obs: np.ndarray,
    u_target: np.ndarray,
    rng: np.random.Generator,
    l2: float = 1e-3,
) -> np.ndarray:
    n, p = x.shape
    idx = rng.permutation(n)
    split = n // 2
    folds = [idx[:split], idx[split:]]
    u_hat = np.zeros(n, dtype=float)

    ax = a[:, None] * x[:, : min(5, p)]
    z = np.concatenate(
        [
            np.ones((n, 1)),
            x,
            a[:, None],
            ax,
            r_obs[:, None],
        ],
        axis=1,
    )

    for holdout in [0, 1]:
        te = folds[holdout]
        tr = folds[1 - holdout]

        beta = fit_ridge_regression(z[tr], u_target[tr], l2=l2)
        train_resid = u_target[tr] - z[tr] @ beta
        margin = max(0.0, float(np.quantile(train_resid, 0.9)))
        pred = z[te] @ beta + margin
        u_hat[te] = np.clip(pred, 0.0, 1.0)

    return u_hat


def simulate_logged_data(
    cfg: ScenarioConfig, n: int, rng: np.random.Generator
) -> LoggedData:
    x = sample_covariates(n=n, p=cfg.p, rho=cfg.cov_rho, rng=rng)
    pi1 = propensity_probs(x, cfg)
    draw = rng.uniform(0.0, 1.0, size=n)
    a = np.where(draw < pi1, 1, -1)
    pi = np.where(a == 1, pi1, 1.0 - pi1)

    mu1, mu_minus1 = true_reward_means(x, nonlinear_reward=cfg.nonlinear_reward)
    mu_a = np.where(a == 1, mu1, mu_minus1)
    r_star = sample_true_reward(mu=mu_a, concentration=cfg.true_concentration, rng=rng)

    misspec_bias = sample_misspec_bias(x=x, a=a, cfg=cfg, rng=rng)
    r_obs = np.clip(r_star + misspec_bias, 0.0, 1.0)

    if cfg.cert_mode == "oracle":
        u = np.clip(cfg.cert_scale * misspec_bias, 0.0, 1.0)
    elif cfg.cert_mode == "estimated":
        u_hat = estimate_certificate_crossfit(
            x=x, a=a, r_obs=r_obs, u_target=misspec_bias, rng=rng
        )
        u = np.clip(cfg.cert_scale * u_hat, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown cert_mode: {cfg.cert_mode}")

    underline_r = np.clip(r_obs - u, 0.0, 1.0)
    # Use the known design lower bound (population-level) for PAC-Bayes scaling.
    # This matches the theoretical assumption pi(a|x) >= epsilon for all (x, a),
    # instead of a sample-dependent minimum.
    if cfg.propensity_mode == "rct":
        epsilon = min(cfg.rct_rho, 1.0 - cfg.rct_rho)
    else:
        epsilon = cfg.eps_floor

    return LoggedData(
        x=x,
        a=a,
        pi=pi,
        pi1=pi1,
        r_obs=r_obs,
        r_star=r_star,
        u=u,
        underline_r=underline_r,
        mu1=mu1,
        mu_minus1=mu_minus1,
        misspec_bias=misspec_bias,
        epsilon=epsilon,
    )


def fit_weighted_hinge(
    x_feat: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    l2: float,
    n_iter: int,
    lr0: float,
) -> np.ndarray:
    n, d = x_feat.shape
    beta = np.zeros(d, dtype=float)
    y = y.astype(float)
    w = np.maximum(w, 0.0)

    for t in range(1, n_iter + 1):
        margins = y * (x_feat @ beta)
        active = margins < 1.0
        if np.any(active):
            grad = -np.sum(
                (w[active] * y[active])[:, None] * x_feat[active], axis=0
            ) / n
        else:
            grad = np.zeros_like(beta)
        grad += l2 * beta
        lr = lr0 / math.sqrt(t)
        beta -= lr * grad

    return beta


def empirical_value_from_beta(
    beta: np.ndarray, x_feat: np.ndarray, a: np.ndarray, reward_over_pi: np.ndarray
) -> float:
    match = (a * (x_feat @ beta) > 0.0).astype(float)
    return float(np.mean(reward_over_pi * match))


def tune_and_fit_hinge(
    x_feat: np.ndarray,
    a: np.ndarray,
    w_train: np.ndarray,
    w_score: np.ndarray,
    lambda_grid: list[float],
    n_iter: int,
    lr0: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    n = x_feat.shape[0]
    perm = rng.permutation(n)
    split = max(1, int(0.75 * n))
    tr = perm[:split]
    va = perm[split:]
    if va.size == 0:
        va = tr

    best_lambda = lambda_grid[0]
    best_score = -float("inf")
    for l2 in lambda_grid:
        beta = fit_weighted_hinge(
            x_feat=x_feat[tr],
            y=a[tr],
            w=w_train[tr],
            l2=l2,
            n_iter=n_iter,
            lr0=lr0,
        )
        score = empirical_value_from_beta(
            beta=beta, x_feat=x_feat[va], a=a[va], reward_over_pi=w_score[va]
        )
        if score > best_score:
            best_score = score
            best_lambda = l2

    beta_full = fit_weighted_hinge(
        x_feat=x_feat,
        y=a,
        w=w_train,
        l2=best_lambda,
        n_iter=n_iter,
        lr0=lr0,
    )
    return beta_full, best_lambda


def build_candidate_betas(
    dim: int,
    anchors: list[np.ndarray],
    rng: np.random.Generator,
    prior_sd: float,
    n_prior: int,
    n_local_per_anchor: int,
    local_sd: float,
) -> np.ndarray:
    chunks = [np.zeros((1, dim), dtype=float)]
    for beta in anchors:
        chunks.append(beta[None, :])
        if n_local_per_anchor > 0:
            noise = rng.normal(0.0, local_sd, size=(n_local_per_anchor, dim))
            chunks.append(beta[None, :] + noise)
    if n_prior > 0:
        chunks.append(rng.normal(0.0, prior_sd, size=(n_prior, dim)))
    return np.vstack(chunks)


def candidate_prior_logprob(candidates: np.ndarray, prior_sd: float) -> np.ndarray:
    sq_norm = np.sum(candidates**2, axis=1)
    return -0.5 * sq_norm / (prior_sd**2 + NUM_EPS)


def candidate_scores(
    candidates: np.ndarray,
    x_feat: np.ndarray,
    a: np.ndarray,
    r_obs_over_pi: np.ndarray,
    r_rob_over_pi: np.ndarray,
    u_over_pi: np.ndarray,
) -> dict[str, np.ndarray]:
    margins = (a[:, None] * (x_feat @ candidates.T)) > 0.0
    match = margins.astype(float)
    v_nom = np.mean(r_obs_over_pi[:, None] * match, axis=0)
    v_rob = np.mean(r_rob_over_pi[:, None] * match, axis=0)
    gamma_u = np.mean(u_over_pi[:, None] * match, axis=0)
    match_rate = np.mean(match, axis=0)
    return {
        "v_nom": v_nom,
        "v_rob": v_rob,
        "gamma_u": gamma_u,
        "match_rate": match_rate,
    }


def gibbs_posterior_from_scores(
    scores: np.ndarray,
    log_prior: np.ndarray,
    epsilon: float,
    eta: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    tau = epsilon * eta * n
    log_post_unnorm = log_prior + tau * scores
    q = softmax(log_post_unnorm)
    p0 = softmax(log_prior)
    return q, p0


def kl_categorical(q: np.ndarray, p: np.ndarray) -> float:
    return float(np.sum(q * (np.log(q + NUM_EPS) - np.log(p + NUM_EPS))))


def evaluate_deterministic_policy(
    beta: np.ndarray,
    x_feat_eval: np.ndarray,
    mu1_eval: np.ndarray,
    mu_minus1_eval: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    act = np.where(x_feat_eval @ beta >= 0.0, 1, -1)
    reward = np.where(act == 1, mu1_eval, mu_minus1_eval)
    value = float(np.mean(reward))
    q05 = safe_quantile(reward, 0.05)
    cvar10 = safe_cvar(reward, 0.1)
    return value, q05, cvar10, act


def evaluate_randomized_policy(
    candidates: np.ndarray,
    q: np.ndarray,
    x_feat_eval: np.ndarray,
    mu1_eval: np.ndarray,
    mu_minus1_eval: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    acts = np.where(x_feat_eval @ candidates.T >= 0.0, 1, -1)
    p1 = (acts == 1).astype(float) @ q
    reward = p1 * mu1_eval + (1.0 - p1) * mu_minus1_eval
    value = float(np.mean(reward))
    q05 = safe_quantile(reward, 0.05)
    cvar10 = safe_cvar(reward, 0.1)
    return value, q05, cvar10, p1


def policy_train_diagnostics_deterministic(
    beta: np.ndarray,
    x_feat: np.ndarray,
    a: np.ndarray,
    r_obs_over_pi: np.ndarray,
    r_rob_over_pi: np.ndarray,
    r_star_over_pi: np.ndarray,
    u_over_pi: np.ndarray,
) -> dict[str, float]:
    match = (a * (x_feat @ beta) > 0.0).astype(float)
    match_mass = float(np.mean(match))
    selected_u_raw = float(np.sum((u_over_pi * match)) / (np.sum(match) + NUM_EPS))
    return {
        "emp_nominal_value": float(np.mean(r_obs_over_pi * match)),
        "emp_robust_value": float(np.mean(r_rob_over_pi * match)),
        "emp_true_value": float(np.mean(r_star_over_pi * match)),
        "gamma_u": float(np.mean(u_over_pi * match)),
        "match_rate": match_mass,
        "selected_u_raw": selected_u_raw,
    }


def policy_train_diagnostics_randomized(
    candidates: np.ndarray,
    q: np.ndarray,
    x_feat: np.ndarray,
    a: np.ndarray,
    r_obs_over_pi: np.ndarray,
    r_rob_over_pi: np.ndarray,
    r_star_over_pi: np.ndarray,
    u_over_pi: np.ndarray,
) -> dict[str, float]:
    match_prob = ((a[:, None] * (x_feat @ candidates.T)) > 0.0).astype(float) @ q
    match_mass = float(np.mean(match_prob))
    selected_u_raw = float(
        np.sum(u_over_pi * match_prob) / (np.sum(match_prob) + NUM_EPS)
    )
    return {
        "emp_nominal_value": float(np.mean(r_obs_over_pi * match_prob)),
        "emp_robust_value": float(np.mean(r_rob_over_pi * match_prob)),
        "emp_true_value": float(np.mean(r_star_over_pi * match_prob)),
        "gamma_u": float(np.mean(u_over_pi * match_prob)),
        "match_rate": match_mass,
        "selected_u_raw": selected_u_raw,
    }


def hotspot_rate_deterministic(x_eval: np.ndarray, actions: np.ndarray) -> float:
    hotspot = (x_eval[:, 0] > 0.0).astype(float)
    return float(np.mean(hotspot * (actions == 1).astype(float)))


def hotspot_rate_randomized(x_eval: np.ndarray, p_action1: np.ndarray) -> float:
    hotspot = (x_eval[:, 0] > 0.0).astype(float)
    return float(np.mean(hotspot * p_action1))


def run_single_replication(
    cfg: ScenarioConfig,
    n_train: int,
    n_eval: int,
    etas: list[float],
    deltas: list[float],
    lambda_grid: list[float],
    n_iter: int,
    lr0: float,
    clip_threshold: float,
    prior_sd: float,
    n_prior_candidates: int,
    n_local_candidates: int,
    local_sd: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    data = simulate_logged_data(cfg=cfg, n=n_train, rng=rng)

    x_eval = sample_covariates(n=n_eval, p=cfg.p, rho=cfg.cov_rho, rng=rng)
    mu1_eval, mu_minus1_eval = true_reward_means(
        x_eval, nonlinear_reward=cfg.nonlinear_reward
    )

    fmap = fit_feature_map(data.x)
    x_feat = fmap.transform(data.x)
    x_feat_eval = fmap.transform(x_eval)

    r_obs_over_pi = data.r_obs / data.pi
    r_rob_over_pi = data.underline_r / data.pi
    r_star_over_pi = data.r_star / data.pi
    u_over_pi = data.u / data.pi

    beta_std, lambda_std = tune_and_fit_hinge(
        x_feat=x_feat,
        a=data.a,
        w_train=r_obs_over_pi,
        w_score=r_obs_over_pi,
        lambda_grid=lambda_grid,
        n_iter=n_iter,
        lr0=lr0,
        rng=rng,
    )
    beta_clip, lambda_clip = tune_and_fit_hinge(
        x_feat=x_feat,
        a=data.a,
        w_train=np.minimum(r_obs_over_pi, clip_threshold),
        w_score=np.minimum(r_obs_over_pi, clip_threshold),
        lambda_grid=lambda_grid,
        n_iter=n_iter,
        lr0=lr0,
        rng=rng,
    )
    beta_rob, lambda_rob = tune_and_fit_hinge(
        x_feat=x_feat,
        a=data.a,
        w_train=r_rob_over_pi,
        w_score=r_rob_over_pi,
        lambda_grid=lambda_grid,
        n_iter=n_iter,
        lr0=lr0,
        rng=rng,
    )
    beta_oracle, lambda_oracle = tune_and_fit_hinge(
        x_feat=x_feat,
        a=data.a,
        w_train=r_star_over_pi,
        w_score=r_star_over_pi,
        lambda_grid=lambda_grid,
        n_iter=n_iter,
        lr0=lr0,
        rng=rng,
    )

    deterministic = {
        "owl_standard": (beta_std, lambda_std),
        "owl_clipped": (beta_clip, lambda_clip),
        "owl_robust_map": (beta_rob, lambda_rob),
        "oracle_true_owl": (beta_oracle, lambda_oracle),
    }

    rows: list[dict[str, Any]] = []
    det_results: dict[str, float] = {}
    oracle_true_value = float("nan")

    for method, (beta, picked_lambda) in deterministic.items():
        true_value, q05, cvar10, act_eval = evaluate_deterministic_policy(
            beta=beta,
            x_feat_eval=x_feat_eval,
            mu1_eval=mu1_eval,
            mu_minus1_eval=mu_minus1_eval,
        )
        diag = policy_train_diagnostics_deterministic(
            beta=beta,
            x_feat=x_feat,
            a=data.a,
            r_obs_over_pi=r_obs_over_pi,
            r_rob_over_pi=r_rob_over_pi,
            r_star_over_pi=r_star_over_pi,
            u_over_pi=u_over_pi,
        )
        hotspot = hotspot_rate_deterministic(x_eval=x_eval, actions=act_eval)
        if method == "oracle_true_owl":
            oracle_true_value = true_value
        det_results[method] = true_value

        rows.append(
            {
                "scenario": cfg.name,
                "method": method,
                "n_train": n_train,
                "eta": float("nan"),
                "delta": float("nan"),
                "lambda": picked_lambda,
                "true_value": true_value,
                "true_regret": float("nan"),  # filled after oracle is known
                "q05_value": q05,
                "cvar10_value": cvar10,
                "emp_nominal_value": diag["emp_nominal_value"],
                "emp_robust_value": diag["emp_robust_value"],
                "emp_true_value": diag["emp_true_value"],
                "gamma_u": diag["gamma_u"],
                "match_rate": diag["match_rate"],
                "selected_u_raw": diag["selected_u_raw"],
                "hotspot_treat1_rate": hotspot,
                "lcb": float("nan"),
                "coverage": float("nan"),
                "gap": float("nan"),
                "epsilon": data.epsilon,
            }
        )

    anchors = [beta_std, beta_clip, beta_rob]
    candidates = build_candidate_betas(
        dim=x_feat.shape[1],
        anchors=anchors,
        rng=rng,
        prior_sd=prior_sd,
        n_prior=n_prior_candidates,
        n_local_per_anchor=n_local_candidates,
        local_sd=local_sd,
    )
    log_p0 = candidate_prior_logprob(candidates=candidates, prior_sd=prior_sd)
    score_dict = candidate_scores(
        candidates=candidates,
        x_feat=x_feat,
        a=data.a,
        r_obs_over_pi=r_obs_over_pi,
        r_rob_over_pi=r_rob_over_pi,
        u_over_pi=u_over_pi,
    )

    for eta in etas:
        q_nom, p0 = gibbs_posterior_from_scores(
            scores=score_dict["v_nom"],
            log_prior=log_p0,
            epsilon=data.epsilon,
            eta=eta,
            n=n_train,
        )
        q_rob, _ = gibbs_posterior_from_scores(
            scores=score_dict["v_rob"],
            log_prior=log_p0,
            epsilon=data.epsilon,
            eta=eta,
            n=n_train,
        )
        kl_nom = kl_categorical(q=q_nom, p=p0)
        kl_rob = kl_categorical(q=q_rob, p=p0)

        true_value_nom, q05_nom, cvar10_nom, p1_nom = evaluate_randomized_policy(
            candidates=candidates,
            q=q_nom,
            x_feat_eval=x_feat_eval,
            mu1_eval=mu1_eval,
            mu_minus1_eval=mu_minus1_eval,
        )
        diag_nom = policy_train_diagnostics_randomized(
            candidates=candidates,
            q=q_nom,
            x_feat=x_feat,
            a=data.a,
            r_obs_over_pi=r_obs_over_pi,
            r_rob_over_pi=r_rob_over_pi,
            r_star_over_pi=r_star_over_pi,
            u_over_pi=u_over_pi,
        )
        rows.append(
            {
                "scenario": cfg.name,
                "method": "pac_nominal_gibbs",
                "n_train": n_train,
                "eta": eta,
                "delta": float("nan"),
                "lambda": float("nan"),
                "true_value": true_value_nom,
                "true_regret": float("nan"),
                "q05_value": q05_nom,
                "cvar10_value": cvar10_nom,
                "emp_nominal_value": diag_nom["emp_nominal_value"],
                "emp_robust_value": diag_nom["emp_robust_value"],
                "emp_true_value": diag_nom["emp_true_value"],
                "gamma_u": diag_nom["gamma_u"],
                "match_rate": diag_nom["match_rate"],
                "selected_u_raw": diag_nom["selected_u_raw"],
                "hotspot_treat1_rate": hotspot_rate_randomized(
                    x_eval=x_eval, p_action1=p1_nom
                ),
                "lcb": float("nan"),
                "coverage": float("nan"),
                "gap": float("nan"),
                "epsilon": data.epsilon,
                "kl_to_prior": kl_nom,
            }
        )

        true_value_rob, q05_rob, cvar10_rob, p1_rob = evaluate_randomized_policy(
            candidates=candidates,
            q=q_rob,
            x_feat_eval=x_feat_eval,
            mu1_eval=mu1_eval,
            mu_minus1_eval=mu_minus1_eval,
        )
        diag_rob = policy_train_diagnostics_randomized(
            candidates=candidates,
            q=q_rob,
            x_feat=x_feat,
            a=data.a,
            r_obs_over_pi=r_obs_over_pi,
            r_rob_over_pi=r_rob_over_pi,
            r_star_over_pi=r_star_over_pi,
            u_over_pi=u_over_pi,
        )

        vhat_rob_q = float(np.dot(q_rob, score_dict["v_rob"]))
        for delta in deltas:
            penalty = (
                (kl_rob + math.log(1.0 / delta)) / (eta * n_train) + (eta / 8.0)
            )
            lcb = vhat_rob_q - penalty / data.epsilon
            rows.append(
                {
                    "scenario": cfg.name,
                    "method": "pac_robust_gibbs",
                    "n_train": n_train,
                    "eta": eta,
                    "delta": delta,
                    "lambda": float("nan"),
                    "true_value": true_value_rob,
                    "true_regret": float("nan"),
                    "q05_value": q05_rob,
                    "cvar10_value": cvar10_rob,
                    "emp_nominal_value": diag_rob["emp_nominal_value"],
                    "emp_robust_value": diag_rob["emp_robust_value"],
                    "emp_true_value": diag_rob["emp_true_value"],
                    "gamma_u": diag_rob["gamma_u"],
                    "match_rate": diag_rob["match_rate"],
                    "selected_u_raw": diag_rob["selected_u_raw"],
                    "hotspot_treat1_rate": hotspot_rate_randomized(
                        x_eval=x_eval, p_action1=p1_rob
                    ),
                    "lcb": lcb,
                    "coverage": float(true_value_rob >= lcb),
                    "gap": true_value_rob - lcb,
                    "epsilon": data.epsilon,
                    "kl_to_prior": kl_rob,
                }
            )

    for row in rows:
        row["true_regret"] = oracle_true_value - row["true_value"]

    return rows


def group_summary(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    metric_keys: list[str],
) -> list[dict[str, Any]]:
    def norm_key(v: Any) -> Any:
        if isinstance(v, float) and math.isnan(v):
            return "__nan__"
        return v

    def denorm_key(v: Any) -> Any:
        if v == "__nan__":
            return float("nan")
        return v

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(norm_key(row[k]) for k in group_keys)
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        agg: dict[str, Any] = {k: denorm_key(v) for k, v in zip(group_keys, key)}
        agg["n_rep"] = len(group_rows)
        for m in metric_keys:
            values = [float(r.get(m, float("nan"))) for r in group_rows]
            agg[f"{m}_mean"] = nanmean(values)
            agg[f"{m}_std"] = nanstd(values)
        out.append(agg)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    preferred = [
        "scenario",
        "method",
        "n_train",
        "eta",
        "delta",
        "lambda",
        "true_value",
        "true_regret",
        "q05_value",
        "cvar10_value",
        "emp_nominal_value",
        "emp_robust_value",
        "emp_true_value",
        "gamma_u",
        "match_rate",
        "selected_u_raw",
        "hotspot_treat1_rate",
        "lcb",
        "coverage",
        "gap",
        "epsilon",
        "kl_to_prior",
        "n_rep",
    ]
    fieldnames = [k for k in preferred if k in all_keys] + sorted(all_keys - set(preferred))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pick_reference_value(values: list[float], target: float, fallback: float) -> float:
    if not values:
        return fallback
    return min(values, key=lambda v: abs(v - target))


def setup_matplotlib(plot_dir: Path) -> Any:
    # Keep matplotlib cache in a writable directory to avoid runtime warnings.
    import os
    import tempfile

    mplconfig = Path(tempfile.gettempdir()) / "pac_owl_mplconfig"
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig))

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.title_fontsize": 12,
            "mathtext.fontset": "stix",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def nan_to_num(v: float, default: float = float("nan")) -> float:
    if isinstance(v, float) and math.isnan(v):
        return default
    return float(v)


def aggregate_metric_by_n(
    rows: list[dict[str, Any]],
    scenario: str,
    method: str,
    metric_key: str,
    eta: float | None = None,
    delta: float | None = None,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_n: dict[int, list[float]] = {}
    for row in rows:
        if row["scenario"] != scenario or row["method"] != method:
            continue
        if eta is not None:
            row_eta = row.get("eta", float("nan"))
            if math.isnan(row_eta) or abs(row_eta - eta) > tol:
                continue
        if delta is not None:
            row_delta = row.get("delta", float("nan"))
            if math.isnan(row_delta) or abs(row_delta - delta) > tol:
                continue
        n_train = int(row["n_train"])
        by_n.setdefault(n_train, []).append(float(row.get(metric_key, float("nan"))))

    if not by_n:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    n_sorted = np.array(sorted(by_n.keys()), dtype=int)
    means = np.array([nanmean(by_n[n]) for n in n_sorted], dtype=float)
    stds = np.array([nanstd(by_n[n]) for n in n_sorted], dtype=float)
    return n_sorted, means, stds


def aggregate_metric_by_eta(
    rows: list[dict[str, Any]],
    scenario: str,
    method: str,
    metric_key: str,
    n_train: int,
    delta: float | None = None,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_eta: dict[float, list[float]] = {}
    for row in rows:
        if row["scenario"] != scenario or row["method"] != method:
            continue
        if int(row["n_train"]) != int(n_train):
            continue
        row_eta = row.get("eta", float("nan"))
        if math.isnan(row_eta):
            continue
        if delta is not None:
            row_delta = row.get("delta", float("nan"))
            if math.isnan(row_delta) or abs(row_delta - delta) > tol:
                continue
        by_eta.setdefault(float(row_eta), []).append(float(row.get(metric_key, float("nan"))))

    if not by_eta:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    eta_sorted = np.array(sorted(by_eta.keys()), dtype=float)
    means = np.array([nanmean(by_eta[e]) for e in eta_sorted], dtype=float)
    stds = np.array([nanstd(by_eta[e]) for e in eta_sorted], dtype=float)
    return eta_sorted, means, stds


def build_subplot_grid(n_items: int) -> tuple[int, int]:
    if n_items <= 1:
        return 1, 1
    if n_items <= 2:
        return 1, 2
    if n_items <= 4:
        return 2, 2
    cols = 3
    rows = int(math.ceil(n_items / cols))
    return rows, cols


def collect_legend_items(axes_arr: np.ndarray) -> tuple[list[Any], list[str]]:
    handles: list[Any] = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in axes_arr:
        h_list, l_list = ax.get_legend_handles_labels()
        for h, l in zip(h_list, l_list):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    return handles, labels


def finalize_figure_layout(
    fig: Any,
    axes_arr: np.ndarray,
    title: str,
    legend_ncol: int | None = None,
) -> None:
    handles, labels = collect_legend_items(axes_arr)
    top = 0.90
    if handles and legend_ncol is not None and legend_ncol > 0:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(legend_ncol, len(labels)),
            frameon=False,
            handlelength=2.6,
            columnspacing=1.4,
        )
        top = 0.895
    fig.suptitle(title, y=0.985, fontsize=15)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, top))


def save_plot_files(fig: Any, plot_dir: Path, stem: str) -> list[Path]:
    paths = [plot_dir / f"{stem}.png", plot_dir / f"{stem}.pdf"]
    for path in paths:
        if path.suffix == ".png":
            fig.savefig(path, dpi=160, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    return paths


def plot_true_value_vs_n(
    plt: Any,
    rows: list[dict[str, Any]],
    scenarios: list[str],
    eta_ref: float,
    delta_ref: float,
    plot_dir: Path,
) -> list[Path]:
    method_specs = [
        ("owl_standard", "OWL"),
        ("owl_clipped", "OWL clipped"),
        ("owl_robust_map", "Robust MAP"),
        ("pac_nominal_gibbs", f"PAC nominal (eta={eta_ref:g})"),
        ("pac_robust_gibbs", f"PAC robust (eta={eta_ref:g}, delta={delta_ref:g})"),
        ("oracle_true_owl", "Oracle upper ref"),
    ]

    rows_n, cols_n = build_subplot_grid(len(scenarios))
    fig, axes = plt.subplots(
        rows_n, cols_n, figsize=(5.0 * cols_n, 3.8 * rows_n), constrained_layout=False
    )
    axes_arr = np.asarray(axes).reshape(-1)

    for idx, scenario in enumerate(scenarios):
        ax = axes_arr[idx]
        for method, label in method_specs:
            if method == "pac_nominal_gibbs":
                n_vals, means, stds = aggregate_metric_by_n(
                    rows=rows,
                    scenario=scenario,
                    method=method,
                    metric_key="true_value",
                    eta=eta_ref,
                )
            elif method == "pac_robust_gibbs":
                n_vals, means, stds = aggregate_metric_by_n(
                    rows=rows,
                    scenario=scenario,
                    method=method,
                    metric_key="true_value",
                    eta=eta_ref,
                    delta=delta_ref,
                )
            else:
                n_vals, means, stds = aggregate_metric_by_n(
                    rows=rows,
                    scenario=scenario,
                    method=method,
                    metric_key="true_value",
                )
            if n_vals.size == 0:
                continue
            style = METHOD_LINE_STYLES[method]
            ax.plot(
                n_vals,
                means,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=1.8,
                markersize=5.0,
                label=label,
            )
            ax.fill_between(
                n_vals, means - stds, means + stds, alpha=0.10, color=style["color"]
            )
        ax.set_title(scenario)
        ax.set_xlabel("n_train")
        ax.set_ylabel("True value")
        ax.grid(True, alpha=0.3)
    for j in range(len(scenarios), axes_arr.size):
        axes_arr[j].axis("off")
    finalize_figure_layout(
        fig=fig,
        axes_arr=axes_arr,
        title="n vs True Policy Value (mean ± sd)",
        legend_ncol=3,
    )
    out_paths = save_plot_files(fig=fig, plot_dir=plot_dir, stem="fig_n_vs_true_value")
    plt.close(fig)
    return out_paths


def plot_coverage_vs_n(
    plt: Any,
    rows: list[dict[str, Any]],
    scenarios: list[str],
    eta_ref: float,
    deltas: list[float],
    plot_dir: Path,
) -> list[Path]:
    rows_n, cols_n = build_subplot_grid(len(scenarios))
    fig, axes = plt.subplots(
        rows_n, cols_n, figsize=(5.0 * cols_n, 3.6 * rows_n), constrained_layout=False
    )
    axes_arr = np.asarray(axes).reshape(-1)

    for idx, scenario in enumerate(scenarios):
        ax = axes_arr[idx]
        for d_idx, delta in enumerate(deltas):
            n_vals, means, stds = aggregate_metric_by_n(
                rows=rows,
                scenario=scenario,
                method="pac_robust_gibbs",
                metric_key="coverage",
                eta=eta_ref,
                delta=delta,
            )
            if n_vals.size == 0:
                continue
            style = DELTA_LINE_STYLES[d_idx % len(DELTA_LINE_STYLES)]
            ax.plot(
                n_vals,
                means,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=1.8,
                markersize=5.0,
                label=f"delta={delta:g} (target {1-delta:.2f})",
            )
            ax.fill_between(
                n_vals, means - stds, means + stds, alpha=0.10, color=style["color"]
            )
            ax.axhline(
                1.0 - delta,
                linestyle=":",
                color=style["color"],
                alpha=0.45,
                linewidth=1.0,
            )
        ax.set_title(scenario)
        ax.set_xlabel("n_train")
        ax.set_ylabel("Coverage")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
    for j in range(len(scenarios), axes_arr.size):
        axes_arr[j].axis("off")
    finalize_figure_layout(
        fig=fig,
        axes_arr=axes_arr,
        title=f"PAC-Bayes LCB Coverage vs n (eta={eta_ref:g})",
        legend_ncol=2,
    )
    out_paths = save_plot_files(fig=fig, plot_dir=plot_dir, stem="fig_coverage_vs_n")
    plt.close(fig)
    return out_paths


def plot_exposure_scatter(
    plt: Any,
    rows: list[dict[str, Any]],
    scenarios: list[str],
    eta_ref: float,
    delta_ref: float,
    plot_dir: Path,
) -> list[Path]:
    labels = {
        "owl_standard": "OWL",
        "owl_clipped": "OWL clipped",
        "owl_robust_map": "Robust MAP",
        "pac_robust_gibbs": "PAC robust",
    }
    methods = list(labels.keys())

    rows_n, cols_n = build_subplot_grid(len(scenarios))
    fig, axes = plt.subplots(
        rows_n, cols_n, figsize=(5.0 * cols_n, 3.8 * rows_n), constrained_layout=False
    )
    axes_arr = np.asarray(axes).reshape(-1)

    for idx, scenario in enumerate(scenarios):
        ax = axes_arr[idx]
        for method in methods:
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                if row["scenario"] != scenario or row["method"] != method:
                    continue
                if method == "pac_robust_gibbs":
                    row_eta = row.get("eta", float("nan"))
                    row_delta = row.get("delta", float("nan"))
                    if (
                        math.isnan(row_eta)
                        or math.isnan(row_delta)
                        or abs(row_eta - eta_ref) > 1e-9
                        or abs(row_delta - delta_ref) > 1e-9
                    ):
                        continue
                xs.append(nan_to_num(float(row.get("gamma_u", float("nan")))))
                ys.append(nan_to_num(float(row.get("true_value", float("nan")))))
            if not xs:
                continue
            ax.scatter(
                xs,
                ys,
                s=24,
                alpha=0.55,
                label=labels[method],
                color=METHOD_LINE_STYLES[method]["color"],
                marker=METHOD_LINE_STYLES[method]["marker"],
                edgecolors="none",
            )
        ax.set_title(scenario)
        ax.set_xlabel("Uncertainty exposure (gamma_u)")
        ax.set_ylabel("True value")
        ax.grid(True, alpha=0.25)
    for j in range(len(scenarios), axes_arr.size):
        axes_arr[j].axis("off")
    finalize_figure_layout(
        fig=fig,
        axes_arr=axes_arr,
        title="Exposure vs True Value",
        legend_ncol=4,
    )
    out_paths = save_plot_files(
        fig=fig, plot_dir=plot_dir, stem="fig_exposure_vs_true_scatter"
    )
    plt.close(fig)
    return out_paths


def plot_eta_sweep_tradeoff(
    plt: Any,
    rows: list[dict[str, Any]],
    scenarios: list[str],
    n_ref: int,
    delta_ref: float,
    plot_dir: Path,
) -> list[Path]:
    rows_n, cols_n = build_subplot_grid(len(scenarios))
    fig, axes = plt.subplots(
        rows_n, cols_n, figsize=(5.0 * cols_n, 3.8 * rows_n), constrained_layout=False
    )
    axes_arr = np.asarray(axes).reshape(-1)

    for idx, scenario in enumerate(scenarios):
        ax = axes_arr[idx]
        etas, true_mean, _ = aggregate_metric_by_eta(
            rows=rows,
            scenario=scenario,
            method="pac_robust_gibbs",
            metric_key="true_value",
            n_train=n_ref,
            delta=delta_ref,
        )
        _, expo_mean, _ = aggregate_metric_by_eta(
            rows=rows,
            scenario=scenario,
            method="pac_robust_gibbs",
            metric_key="gamma_u",
            n_train=n_ref,
            delta=delta_ref,
        )
        if etas.size == 0:
            continue
        ax.plot(
            etas,
            true_mean,
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=5.0,
            color="#1f77b4",
            label="true value",
        )
        ax.set_xscale("log")
        ax.set_xlabel("eta (log-scale)")
        ax.set_ylabel("True value", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(
            etas,
            expo_mean,
            marker="s",
            linestyle="--",
            linewidth=1.6,
            markersize=5.0,
            color="#d62728",
            label="gamma_u",
        )
        ax2.set_ylabel("Uncertainty exposure", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title(f"{scenario} (n={n_ref}, delta={delta_ref:g})")

    for j in range(len(scenarios), axes_arr.size):
        axes_arr[j].axis("off")
    fig.suptitle("eta sweep: efficiency vs robustness trade-off", y=0.985, fontsize=15)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    out_paths = save_plot_files(fig=fig, plot_dir=plot_dir, stem="fig_eta_sweep_tradeoff")
    plt.close(fig)
    return out_paths


def generate_plots(
    rows_all: list[dict[str, Any]],
    selected_scenarios: list[str],
    etas: list[float],
    deltas: list[float],
    n_train_list: list[int],
    out_dir: Path,
) -> list[Path]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib(plot_dir=plot_dir)

    eta_ref = pick_reference_value(etas, target=1.0, fallback=1.0)
    delta_ref = pick_reference_value(deltas, target=0.1, fallback=0.1)
    n_ref = max(n_train_list) if n_train_list else 0

    out_paths: list[Path] = []
    out_paths.extend(
        plot_true_value_vs_n(
        plt=plt,
        rows=rows_all,
        scenarios=selected_scenarios,
        eta_ref=eta_ref,
        delta_ref=delta_ref,
        plot_dir=plot_dir,
        )
    )
    out_paths.extend(
        plot_coverage_vs_n(
        plt=plt,
        rows=rows_all,
        scenarios=selected_scenarios,
        eta_ref=eta_ref,
        deltas=deltas,
        plot_dir=plot_dir,
        )
    )
    out_paths.extend(
        plot_exposure_scatter(
        plt=plt,
        rows=rows_all,
        scenarios=selected_scenarios,
        eta_ref=eta_ref,
        delta_ref=delta_ref,
        plot_dir=plot_dir,
        )
    )
    out_paths.extend(
        plot_eta_sweep_tradeoff(
        plt=plt,
        rows=rows_all,
        scenarios=selected_scenarios,
        n_ref=n_ref,
        delta_ref=delta_ref,
        plot_dir=plot_dir,
        )
    )
    return out_paths


def run(args: argparse.Namespace) -> None:
    scenarios = scenario_library()
    if args.scenarios.lower() == "all":
        selected = list(scenarios.keys())
    else:
        selected = [tok.strip() for tok in args.scenarios.split(",") if tok.strip()]
        unknown = [name for name in selected if name not in scenarios]
        if unknown:
            raise ValueError(f"Unknown scenarios: {unknown}. Known: {list(scenarios.keys())}")

    n_train_list = parse_int_list(args.n_train)
    etas = parse_float_list(args.etas)
    deltas = parse_float_list(args.deltas)
    lambda_grid = parse_float_list(args.lambda_grid)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "replications": args.replications,
        "n_eval": args.n_eval,
        "n_train": n_train_list,
        "etas": etas,
        "deltas": deltas,
        "lambda_grid": lambda_grid,
        "clip_threshold": args.clip_threshold,
        "prior_sd": args.prior_sd,
        "n_prior_candidates": args.n_prior_candidates,
        "n_local_candidates": args.n_local_candidates,
        "local_sd": args.local_sd,
        "n_iter": args.n_iter,
        "lr0": args.lr0,
        "scenarios": [asdict(scenarios[name]) for name in selected],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    rng = np.random.default_rng(args.seed)
    rows_all: list[dict[str, Any]] = []
    total_jobs = len(selected) * len(n_train_list) * args.replications
    done = 0
    for scenario_name in selected:
        cfg = scenarios[scenario_name]
        for n_train in n_train_list:
            for rep in range(args.replications):
                rep_seed = int(rng.integers(0, 2**32 - 1))
                rep_rng = np.random.default_rng(rep_seed)
                rows = run_single_replication(
                    cfg=cfg,
                    n_train=n_train,
                    n_eval=args.n_eval,
                    etas=etas,
                    deltas=deltas,
                    lambda_grid=lambda_grid,
                    n_iter=args.n_iter,
                    lr0=args.lr0,
                    clip_threshold=args.clip_threshold,
                    prior_sd=args.prior_sd,
                    n_prior_candidates=args.n_prior_candidates,
                    n_local_candidates=args.n_local_candidates,
                    local_sd=args.local_sd,
                    rng=rep_rng,
                )
                for row in rows:
                    row["replication"] = rep
                    row["rep_seed"] = rep_seed
                rows_all.extend(rows)
                done += 1
                if done % max(1, total_jobs // 20) == 0 or done == total_jobs:
                    print(f"[progress] {done}/{total_jobs}")

    write_csv(out_dir / "raw_metrics.csv", rows_all)

    metric_keys = [
        "true_value",
        "true_regret",
        "q05_value",
        "cvar10_value",
        "emp_nominal_value",
        "emp_robust_value",
        "emp_true_value",
        "gamma_u",
        "selected_u_raw",
        "hotspot_treat1_rate",
        "lcb",
        "coverage",
        "gap",
        "kl_to_prior",
        "epsilon",
    ]
    summary = group_summary(
        rows=rows_all,
        group_keys=["scenario", "n_train", "method", "eta", "delta"],
        metric_keys=metric_keys,
    )
    write_csv(out_dir / "summary_by_setting.csv", summary)

    lcb_rows = [
        row for row in rows_all if row["method"] == "pac_robust_gibbs" and not math.isnan(row["delta"])
    ]
    lcb_summary = group_summary(
        rows=lcb_rows,
        group_keys=["scenario", "n_train", "eta", "delta"],
        metric_keys=["coverage", "gap", "lcb", "true_value", "emp_robust_value"],
    )
    write_csv(out_dir / "lcb_summary.csv", lcb_summary)

    plot_paths: list[Path] = []
    if args.make_plots:
        plot_paths = generate_plots(
            rows_all=rows_all,
            selected_scenarios=selected,
            etas=etas,
            deltas=deltas,
            n_train_list=n_train_list,
            out_dir=out_dir,
        )

    print(f"Saved: {out_dir / 'raw_metrics.csv'}")
    print(f"Saved: {out_dir / 'summary_by_setting.csv'}")
    print(f"Saved: {out_dir / 'lcb_summary.csv'}")
    for path in plot_paths:
        print(f"Saved: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PAC-Bayes robust OWL simulation experiments."
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help="Comma-separated scenario names or 'all'.",
    )
    parser.add_argument(
        "--n-train",
        type=str,
        default="200,500,1000",
        help="Comma-separated training sample sizes.",
    )
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--etas",
        type=str,
        default="0.3,1.0,3.0",
        help="Comma-separated PAC-Bayes eta values.",
    )
    parser.add_argument(
        "--deltas",
        type=str,
        default="0.2,0.1,0.05",
        help="Comma-separated PAC-Bayes delta values.",
    )
    parser.add_argument(
        "--lambda-grid",
        type=str,
        default="0.001,0.01,0.1,1.0",
        help="Comma-separated L2 regularization grid for hinge learners.",
    )
    parser.add_argument("--n-iter", type=int, default=800)
    parser.add_argument("--lr0", type=float, default=0.15)
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=10.0,
        help="Weight clipping threshold for clipped OWL baseline.",
    )

    parser.add_argument("--prior-sd", type=float, default=1.0)
    parser.add_argument("--n-prior-candidates", type=int, default=256)
    parser.add_argument("--n-local-candidates", type=int, default=48)
    parser.add_argument("--local-sd", type=float, default=0.35)
    parser.add_argument("--output-dir", type=str, default="outputs/main")
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to generate PNG plots under output_dir/plots.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    run(parser.parse_args())
