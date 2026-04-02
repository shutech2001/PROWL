from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pac_owl.estimators import (
    PROWL,
    StandardizedFeatureMap,
    double_robust_score_matrix,
    fit_arm_value_nuisance_model,
    fit_policy_tree,
    tune_linear_q_learning,
    tune_linear_residual_weighted_learning,
    tune_weighted_hinge_policy,
)
from pac_owl.estimators.prowl import ParticleLibraryConfig, feature_scores
from pac_owl.simulation.paper_scenarios import (
    GeneratedDataset,
    PotentialOutcomeSample,
    ScenarioDefinition,
    available_scenarios,
    evaluate_policy_on_potential_outcomes,
    generate_logged_dataset,
)

FONT_SIZE_BOOST = 4
MARKER_SIZE_BOOST = 1.5
LEGEND_Y_SHIFT = 0.03
MANUSCRIPT_COMPARISON_SCENARIOS = (
    "linear_scope_invariant",
    "clinical_triage_nuisance_conflict",
)
SCENARIO_DISPLAY_LABEL_OVERRIDES = {
    "linear_scope_invariant": "Scenario 1",
    "clinical_triage_nuisance_conflict": "Scenario 2",
}
METHOD_FAMILY_ORDER = (
    "PROWL",
    "PROWL (U=0)",
    "OWL",
    "RWL",
    "Q-learning",
    "Policy Tree",
)
METHOD_FAMILY_DISPLAY_LABELS = {
    "PROWL": "PROWL",
    "PROWL (U=0)": r"PROWL ($U=0$)",
    "OWL": "OWL",
    "RWL": "RWL",
    "Q-learning": "Q-learning",
    "Policy Tree": "Policy Tree",
}
METHOD_FAMILY_COLORS = {
    "PROWL": "#111111",
    "PROWL (U=0)": "#7f7f7f",
    "OWL": "#1f77b4",
    "Q-learning": "#ff7f0e",
    "RWL": "#2ca02c",
    "Policy Tree": "#d62728",
}
METHOD_FAMILY_MARKERS = {
    "PROWL": "o",
    "PROWL (U=0)": "X",
    "OWL": "s",
    "Q-learning": "^",
    "RWL": "D",
    "Policy Tree": "P",
}
METHOD_FAMILY_LINE_STYLES = {
    "PROWL": "-",
    "PROWL (U=0)": "--",
    "OWL": "-.",
    "Q-learning": ":",
    "RWL": (0, (5, 1)),
    "Policy Tree": (0, (3, 1, 1, 1)),
}
METHOD_FAMILY_LINEWIDTHS = {
    "PROWL": 4.8,
    "PROWL (U=0)": 4.0,
    "OWL": 4.0,
    "Q-learning": 4.0,
    "RWL": 4.0,
    "Policy Tree": 4.0,
}
METHOD_FAMILY_ZORDERS = {
    "PROWL": 5,
    "PROWL (U=0)": 3,
    "OWL": 3,
    "Q-learning": 3,
    "RWL": 3,
    "Policy Tree": 3,
}

matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 22 + FONT_SIZE_BOOST,
        "axes.titlesize": 24 + FONT_SIZE_BOOST,
        "figure.titlesize": 28 + FONT_SIZE_BOOST,
        "xtick.labelsize": 19 + FONT_SIZE_BOOST,
        "ytick.labelsize": 19 + FONT_SIZE_BOOST,
        "legend.fontsize": 24 + FONT_SIZE_BOOST,
        "legend.title_fontsize": 24 + FONT_SIZE_BOOST,
        "axes.unicode_minus": False,
    }
)

EPS = 1e-12
RAW_BASELINE_METHODS = ("OWL(R)", "Q-learning(R)", "RWL(R)", "Policy Tree(R)")
LOWER_PLUGIN_METHODS = (
    "OWL(underline R)",
    "Q-learning(underline R)",
    "RWL(underline R)",
    "Policy Tree(underline R)",
)
PROWL_U0_METHOD = "PROWL (U=0)"
RHO_LABEL = r"$\rho$"
N_TOTAL_LABEL = r"$N$"
UNDERLINE_R_LABEL = r"$\underline{R}$"
DEFAULT_BENCHMARK_DEPLOYMENT = "map"
DEFAULT_RHO_GRID = "0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0"
DEFAULT_N_GRID = "100,200,500,1000,2000"
DEFAULT_N_SWEEP_GRID = "100,200,500,1000,2000"
DEFAULT_ETA_TUNING_GRID = "0.125,0.25,0.5,1.0,2.0,4.0,8.0"
DEFAULT_GAMMA_GRID = DEFAULT_ETA_TUNING_GRID
DEFAULT_RHO_REPLICATIONS = 30
DEFAULT_N_REPLICATIONS = 30
DEFAULT_CERTIFICATE_DIAGNOSTIC_REPLICATIONS = 30
DEFAULT_SPLIT_FREE_REPLICATIONS = 30
DEFAULT_MAIN_SCENARIOS = (
    "linear_scope_invariant",
    "clinical_triage_nuisance_conflict",
)
DEFAULT_RHO_SCENARIOS = DEFAULT_MAIN_SCENARIOS
DEFAULT_N_SWEEP_SCENARIOS = DEFAULT_MAIN_SCENARIOS
DEFAULT_CERTIFICATE_DIAGNOSTIC_SCENARIOS = DEFAULT_MAIN_SCENARIOS
DEFAULT_SPLIT_FREE_SCENARIOS = DEFAULT_MAIN_SCENARIOS
METRIC_DISPLAY_LABELS = {
    "target_value": "Target Value",
    "proxy_value": "Proxy Value",
    "robust_value": "Certified Value",
    "target_regret": "Target Regret",
    "proxy_regret": "Proxy Regret",
    "robust_regret": "Robust Regret",
    "proxy_target_gap": "Proxy - Target Gap",
    "target_robust_gap": "Target - Certified Gap",
    "proxy_robust_gap": "Proxy - Certified Gap",
    "policy_mean_certificate": r"Policy-Induced $\\mathbb{E}[U]$",
    "policy_mean_bias": "Policy-Induced Mean Bias",
    "policy_certificate_slack": "Certificate Slack",
    "policy_certificate_validity_rate": "Policy Validity Rate",
    "policy_clipping_rate": "Policy Clipping Rate",
    "train_mean_certificate": r"Logged $\\mathbb{E}[U]$",
    "train_mean_bias": "Logged Mean Bias",
    "train_clipping_rate": "Logged Clipping Rate",
    "certificate_validity_rate_test": "Test Validity Rate",
}
SCENARIO_GROUPS = {
    "linear_scope_invariant": "basic",
    "clinical_triage_nuisance_conflict": "complex",
}


def parse_float_grid(raw: str) -> list[float]:
    return [float(token.strip()) for token in str(raw).split(",") if token.strip()]


def parse_int_grid(raw: str) -> list[int]:
    return [int(token.strip()) for token in str(raw).split(",") if token.strip()]


def parse_str_grid(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def rho_scenario_ids(args: argparse.Namespace) -> list[str]:
    if getattr(args, "rho_scenario", None):
        return [str(args.rho_scenario)]
    return parse_str_grid(args.rho_scenarios)


def scenario_group_for_id(scenario_id: str) -> str:
    return SCENARIO_GROUPS.get(scenario_id, "other")


def normalize_certificate_mode(raw: str) -> str:
    mode = str(raw).strip().lower()
    if mode == "given":
        return "oracle"
    return mode


def scenario_display_label(scenario_id: str) -> str:
    if str(scenario_id) in SCENARIO_DISPLAY_LABEL_OVERRIDES:
        return SCENARIO_DISPLAY_LABEL_OVERRIDES[str(scenario_id)]
    scenario = scenario_library().get(str(scenario_id))
    return scenario.display_name if scenario is not None else str(scenario_id).replace("_", " ")


def method_family_name(method: str) -> str:
    mapping = {
        "PROWL (tuned eta)": "PROWL",
        PROWL_U0_METHOD: "PROWL (U=0)",
        "OWL(R)": "OWL",
        "Q-learning(R)": "Q-learning",
        "RWL(R)": "RWL",
        "Policy Tree(R)": "Policy Tree",
        "OWL(underline R)": "OWL",
        "Q-learning(underline R)": "Q-learning",
        "RWL(underline R)": "RWL",
        "Policy Tree(underline R)": "Policy Tree",
    }
    return mapping.get(method, method)


def method_display_label(method: str) -> str:
    mapping = {
        "PROWL (tuned eta)": "PROWL",
        PROWL_U0_METHOD: r"PROWL ($U=0$)",
        "OWL(R)": r"OWL ($R$)",
        "Q-learning(R)": r"Q-learning ($R$)",
        "RWL(R)": r"RWL ($R$)",
        "Policy Tree(R)": r"Policy Tree ($R$)",
        "OWL(underline R)": rf"OWL ({UNDERLINE_R_LABEL})",
        "Q-learning(underline R)": rf"Q-learning ({UNDERLINE_R_LABEL})",
        "RWL(underline R)": rf"RWL ({UNDERLINE_R_LABEL})",
        "Policy Tree(underline R)": rf"Policy Tree ({UNDERLINE_R_LABEL})",
    }
    return mapping.get(method, method)


def baseline_methods_for_variant(comparator_variant: str) -> tuple[str, ...]:
    return RAW_BASELINE_METHODS if comparator_variant == "raw" else LOWER_PLUGIN_METHODS


def comparator_method_order(comparator_variant: str, *, shared_legend_order: bool = False) -> list[str]:
    baseline_methods = baseline_methods_for_variant(comparator_variant)
    if shared_legend_order:
        baseline_order = [baseline_methods[0], baseline_methods[2], baseline_methods[1], baseline_methods[3]]
    else:
        baseline_order = list(baseline_methods)
    return ["PROWL (tuned eta)", PROWL_U0_METHOD, *baseline_order]


def manuscript_comparison_scenario_ids(available_scenario_ids: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for scenario_id in MANUSCRIPT_COMPARISON_SCENARIOS:
        if scenario_id in available_scenario_ids and scenario_id not in seen:
            ordered.append(scenario_id)
            seen.add(scenario_id)
    for scenario_id in available_scenario_ids:
        if scenario_id not in seen:
            ordered.append(scenario_id)
            seen.add(scenario_id)
        if len(ordered) >= 2:
            break
    return ordered[:2]


def reward_variant(method: str) -> str:
    method_str = str(method)
    if method_str.startswith("PROWL"):
        if "U=0" in method_str:
            return "prowl_u0"
        return "prowl"
    if "(R" in method_str and "underline R" not in method_str:
        return "raw"
    if "underline R" in method_str:
        return "lower_plugin"
    return "other"


def deployment_mode_name(raw: str) -> str:
    return {
        "map": "map",
        "mean_rule": "deterministic",
        "gibbs": "stochastic",
    }[str(raw)]


def resolve_experiment_prowl_deployment(raw: str, experiment: str) -> str:
    raw_value = str(raw).strip().lower()
    if raw_value != "auto":
        return raw_value
    return DEFAULT_BENCHMARK_DEPLOYMENT


def scenario_library() -> dict[str, ScenarioDefinition]:
    return available_scenarios()


def fit_policy_tree_lower(
    *,
    x: np.ndarray,
    a: np.ndarray,
    reward: np.ndarray,
    propensity: np.ndarray,
    feature_map_factory: Callable[[], StandardizedFeatureMap],
    depth: int,
    min_node_size: int,
    split_step: int,
    max_features: int | None,
) -> Any:
    nuisance = fit_arm_value_nuisance_model(
        x=x,
        treatment=a,
        lower_reward=reward,
        feature_map_factory=feature_map_factory,
    )
    mu_pos, mu_neg = nuisance.predict(x)
    p_treat_one = np.where(a == 1.0, propensity, 1.0 - propensity)
    score_matrix = double_robust_score_matrix(
        outcome=reward,
        action=a,
        action_values=(1, -1),
        conditional_means=np.column_stack([mu_pos, mu_neg]),
        action_probabilities=np.column_stack([p_treat_one, 1.0 - p_treat_one]),
    )
    return fit_policy_tree(
        x=x,
        scores=score_matrix,
        depth=depth,
        actions=(1, -1),
        min_node_size=min_node_size,
        split_step=split_step,
        max_features=max_features,
    )


def evaluate_policy(
    policy: Any,
    dataset: GeneratedDataset,
    *,
    deployment_mode: str,
) -> dict[str, float]:
    p_treat_one = policy_treat_one_probability(policy, dataset, deployment_mode=deployment_mode)
    metrics = evaluate_policy_on_potential_outcomes(
        sample=dataset.test,
        p_treat_one=p_treat_one,
    )
    metrics.update(policy_certificate_diagnostics(dataset.test, p_treat_one))
    return metrics


def policy_treat_one_probability(
    policy: Any,
    dataset: GeneratedDataset,
    *,
    deployment_mode: str,
) -> np.ndarray:
    if deployment_mode == "map" and all(
        hasattr(policy, attr) for attr in ("posterior_weights", "candidates", "feature_map", "score_bound")
    ):
        idx = int(np.argmax(np.asarray(policy.posterior_weights, dtype=float)))
        phi = policy.feature_map.transform(dataset.test.x)
        score = feature_scores(
            phi=phi,
            candidates=np.asarray(policy.candidates, dtype=float)[[idx]],
            score_bound=float(policy.score_bound),
        ).reshape(-1)
        return (score >= 0.0).astype(float)
    if deployment_mode == "deterministic" and hasattr(policy, "deterministic_action"):
        return (np.asarray(policy.deterministic_action(dataset.test.x), dtype=float).reshape(-1) == 1.0).astype(float)
    return np.clip(np.asarray(policy.action_probability(dataset.test.x), dtype=float).reshape(-1), 0.0, 1.0)


def policy_certificate_diagnostics(
    sample: PotentialOutcomeSample,
    p_treat_one: np.ndarray,
) -> dict[str, float]:
    p = np.clip(np.asarray(p_treat_one, dtype=float).reshape(-1), 0.0, 1.0)
    q = 1.0 - p
    selected_certificate = p * sample.certificate_pos + q * sample.certificate_neg
    selected_bias = p * sample.bias_pos + q * sample.bias_neg
    clipping_indicator = p * (sample.reward_pos < sample.certificate_pos).astype(float) + q * (
        sample.reward_neg < sample.certificate_neg
    ).astype(float)
    validity_indicator = p * (sample.bias_pos <= sample.certificate_pos).astype(float) + q * (
        sample.bias_neg <= sample.certificate_neg
    ).astype(float)
    proxy_value = float(np.mean(p * sample.mean_reward_pos + q * sample.mean_reward_neg))
    target_value = float(np.mean(p * sample.mean_reward_star_pos + q * sample.mean_reward_star_neg))
    robust_value = float(np.mean(p * sample.mean_lower_reward_pos + q * sample.mean_lower_reward_neg))
    return {
        "policy_mean_certificate": float(np.mean(selected_certificate)),
        "policy_mean_bias": float(np.mean(selected_bias)),
        "policy_certificate_slack": float(np.mean(selected_certificate - selected_bias)),
        "policy_certificate_validity_rate": float(np.mean(validity_indicator)),
        "policy_clipping_rate": float(np.mean(clipping_indicator)),
        "proxy_target_gap": proxy_value - target_value,
        "target_robust_gap": target_value - robust_value,
        "proxy_robust_gap": proxy_value - robust_value,
    }


def stratified_honest_split(
    treatment: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    treatment = np.asarray(treatment, dtype=float).reshape(-1)
    nuisance_parts: list[np.ndarray] = []
    policy_parts: list[np.ndarray] = []
    for arm in (-1.0, 1.0):
        idx = np.flatnonzero(treatment == arm)
        if idx.size == 0:
            continue
        perm = np.asarray(idx, dtype=int)
        rng.shuffle(perm)
        split = max(1, perm.size // 2)
        if split >= perm.size:
            split = perm.size - 1
        if split <= 0:
            continue
        nuisance_parts.append(perm[:split])
        policy_parts.append(perm[split:])
    nuisance_idx = np.sort(np.concatenate(nuisance_parts)) if nuisance_parts else np.empty(0, dtype=int)
    policy_idx = np.sort(np.concatenate(policy_parts)) if policy_parts else np.empty(0, dtype=int)
    if nuisance_idx.size == 0 or policy_idx.size == 0:
        n_obs = treatment.shape[0]
        perm = rng.permutation(n_obs)
        split = max(1, n_obs // 2)
        split = min(split, n_obs - 1)
        nuisance_idx = np.sort(perm[:split])
        policy_idx = np.sort(perm[split:])
    return nuisance_idx, policy_idx


def particle_library_config_from_settings(settings: dict[str, Any]) -> ParticleLibraryConfig:
    return ParticleLibraryConfig(
        n_anchor_particles=int(settings["n_anchor_particles"]),
        n_prior_samples=int(settings["n_prior_particles"]),
        n_local_samples_per_anchor=int(settings["n_local_particles"]),
        local_scale=float(settings["local_particle_scale"]),
    )


def build_dataset(
    *,
    scenario_id: str,
    n_total: int,
    n_test: int,
    rho: float,
    certificate_mode: str,
    certificate_scale: float,
    seed: int,
    n_certificate_auxiliary: int,
    propensity_mode: str,
    propensity_strength: float,
) -> GeneratedDataset:
    scenario = scenario_library()[scenario_id]
    resolved_propensity_mode = propensity_mode
    if str(propensity_mode).strip().lower() == "auto":
        resolved_propensity_mode = "constant" if scenario_group_for_id(scenario_id) == "basic" else "linear"
    return generate_logged_dataset(
        scenario=scenario,
        n_train=int(n_total),
        n_test=int(n_test),
        rho=float(rho),
        certificate_mode=normalize_certificate_mode(certificate_mode),
        certificate_scale=float(certificate_scale),
        rng=np.random.default_rng(int(seed)),
        n_certificate_auxiliary=int(n_certificate_auxiliary),
        propensity_mode=resolved_propensity_mode,
        propensity_strength=float(propensity_strength),
    )


def add_result_row(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    scenario: ScenarioDefinition,
    dataset: GeneratedDataset,
    replication: int,
    method: str,
    policy: Any,
    runtime_sec: float,
    deployment_mode: str,
    task_metadata: dict[str, Any],
    fit_info: dict[str, Any] | None = None,
) -> None:
    metrics = evaluate_policy(policy, dataset, deployment_mode=deployment_mode)
    row = {
        "experiment": experiment,
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.display_name,
        "scenario_group": scenario_group_for_id(scenario.scenario_id),
        "geometry": scenario.geometry,
        "uncertainty": scenario.uncertainty,
        "n_total": int(dataset.train.x.shape[0]),
        "rho": float(dataset.rho),
        "certificate_mode": dataset.certificate_mode,
        "certificate_scale": float(dataset.certificate_scale),
        "propensity_mode": dataset.propensity_mode,
        "propensity_strength": float(dataset.propensity_strength),
        "certificate_validity_rate_test": dataset.certificate_validity_rate_test,
        "train_mean_certificate": float(np.mean(dataset.train.certificate)),
        "train_mean_bias": float(np.mean(dataset.train.bias)),
        "train_certificate_slack": float(np.mean(dataset.train.certificate - dataset.train.bias)),
        "train_clipping_rate": float(np.mean(dataset.train.reward < dataset.train.certificate)),
        "replication": int(replication),
        "method": method,
        "reward_variant": reward_variant(method),
        "deployment_mode": deployment_mode,
        "runtime_sec": float(runtime_sec),
    }
    row.update(task_metadata)
    row.update(metrics)
    if fit_info:
        row.update(fit_info)
    rows.append(row)


def fit_baseline_rows(
    *,
    experiment: str,
    dataset: GeneratedDataset,
    scenario: ScenarioDefinition,
    settings: dict[str, Any],
    replication_seed: int,
    task_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    feature_map_factory = scenario.feature_map_factory
    penalty_grid = parse_float_grid(settings["penalty_grid"])
    q_penalty_grid = parse_float_grid(settings["q_penalty_grid"])
    train = dataset.train
    rows: list[dict[str, Any]] = []
    reward_over_propensity = train.reward / np.maximum(train.pi, EPS)
    lower_reward_over_propensity = train.lower_reward / np.maximum(train.pi, EPS)

    start = time.perf_counter()
    owl_policy, owl_penalty = tune_weighted_hinge_policy(
        x=train.x,
        fit_labels=train.a,
        fit_weights=reward_over_propensity,
        observed_a=train.a,
        evaluation_reward=train.reward,
        propensity=train.pi,
        feature_map_factory=feature_map_factory,
        penalty_grid=penalty_grid,
        random_state=replication_seed + 3,
        score_bound=None,
    )
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="OWL(R)",
        policy=owl_policy,
        runtime_sec=time.perf_counter() - start,
        deployment_mode="deterministic",
        task_metadata=task_metadata,
        fit_info={"selected_penalty": owl_penalty},
    )

    start = time.perf_counter()
    owl_policy, owl_penalty = tune_weighted_hinge_policy(
        x=train.x,
        fit_labels=train.a,
        fit_weights=lower_reward_over_propensity,
        observed_a=train.a,
        evaluation_reward=train.lower_reward,
        propensity=train.pi,
        feature_map_factory=feature_map_factory,
        penalty_grid=penalty_grid,
        random_state=replication_seed + 5,
        score_bound=None,
    )
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="OWL(underline R)",
        policy=owl_policy,
        runtime_sec=time.perf_counter() - start,
        deployment_mode="deterministic",
        task_metadata=task_metadata,
        fit_info={"selected_penalty": owl_penalty},
    )

    start = time.perf_counter()
    q_policy, q_penalty = tune_linear_q_learning(
        x=train.x,
        a=train.a,
        y=train.reward,
        main_feature_map_factory=feature_map_factory,
        blip_feature_map_factory=feature_map_factory,
        penalty_grid=q_penalty_grid,
        random_state=replication_seed + 7,
    )
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="Q-learning(R)",
        policy=q_policy,
        runtime_sec=time.perf_counter() - start,
        deployment_mode="deterministic",
        task_metadata=task_metadata,
        fit_info={"selected_penalty": q_penalty},
    )

    start = time.perf_counter()
    q_policy, q_penalty = tune_linear_q_learning(
        x=train.x,
        a=train.a,
        y=train.lower_reward,
        main_feature_map_factory=feature_map_factory,
        blip_feature_map_factory=feature_map_factory,
        penalty_grid=q_penalty_grid,
        random_state=replication_seed + 11,
    )
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="Q-learning(underline R)",
        policy=q_policy,
        runtime_sec=time.perf_counter() - start,
        deployment_mode="deterministic",
        task_metadata=task_metadata,
        fit_info={"selected_penalty": q_penalty},
    )

    start = time.perf_counter()
    rwl_policy, rwl_penalty = tune_linear_residual_weighted_learning(
        x=train.x,
        a=train.a,
        y=train.reward,
        propensity=train.pi,
        feature_map_factory=feature_map_factory,
        residual_feature_map_factory=feature_map_factory,
        penalty_grid=penalty_grid,
        random_state=replication_seed + 17,
    )
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="RWL(R)",
        policy=rwl_policy,
        runtime_sec=time.perf_counter() - start,
        deployment_mode="deterministic",
        task_metadata=task_metadata,
        fit_info={"selected_penalty": rwl_penalty},
    )

    start = time.perf_counter()
    rwl_policy, rwl_penalty = tune_linear_residual_weighted_learning(
        x=train.x,
        a=train.a,
        y=train.lower_reward,
        propensity=train.pi,
        feature_map_factory=feature_map_factory,
        residual_feature_map_factory=feature_map_factory,
        penalty_grid=penalty_grid,
        random_state=replication_seed + 23,
    )
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="RWL(underline R)",
        policy=rwl_policy,
        runtime_sec=time.perf_counter() - start,
        deployment_mode="deterministic",
        task_metadata=task_metadata,
        fit_info={"selected_penalty": rwl_penalty},
    )

    start = time.perf_counter()
    tree_policy = fit_policy_tree_lower(
        x=train.x,
        a=train.a,
        reward=train.reward,
        propensity=train.pi,
        feature_map_factory=feature_map_factory,
        depth=int(settings["policy_tree_depth"]),
        min_node_size=int(settings["policy_tree_min_node_size"]),
        split_step=int(settings["policy_tree_split_step"]),
        max_features=(
            None if int(settings["policy_tree_max_features"]) <= 0 else int(settings["policy_tree_max_features"])
        ),
    )
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="Policy Tree(R)",
        policy=tree_policy,
        runtime_sec=time.perf_counter() - start,
        deployment_mode="deterministic",
        task_metadata=task_metadata,
    )

    start = time.perf_counter()
    tree_policy = fit_policy_tree_lower(
        x=train.x,
        a=train.a,
        reward=train.lower_reward,
        propensity=train.pi,
        feature_map_factory=feature_map_factory,
        depth=int(settings["policy_tree_depth"]),
        min_node_size=int(settings["policy_tree_min_node_size"]),
        split_step=int(settings["policy_tree_split_step"]),
        max_features=(
            None if int(settings["policy_tree_max_features"]) <= 0 else int(settings["policy_tree_max_features"])
        ),
    )
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="Policy Tree(underline R)",
        policy=tree_policy,
        runtime_sec=time.perf_counter() - start,
        deployment_mode="deterministic",
        task_metadata=task_metadata,
    )
    return rows


def fit_baselines_and_tuned_prowl(
    *,
    experiment: str,
    dataset: GeneratedDataset,
    scenario: ScenarioDefinition,
    settings: dict[str, Any],
    replication_seed: int,
    task_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    feature_map_factory = scenario.feature_map_factory
    nuisance_feature_map_factory = scenario.nuisance_feature_map_factory or feature_map_factory
    deployment_mode = deployment_mode_name(settings["prowl_deployment"])
    eta_grid = parse_float_grid(settings["eta_grid"])
    gamma_grid = parse_float_grid(settings["gamma_grid"])
    train = dataset.train
    nuisance_model = fit_arm_value_nuisance_model(
        x=train.x,
        treatment=train.a,
        lower_reward=train.lower_reward,
        feature_map_factory=nuisance_feature_map_factory,
        l2_penalty=float(settings["nuisance_l2_penalty"]),
    )
    particle_cfg = particle_library_config_from_settings(settings)
    rows = fit_baseline_rows(
        experiment=experiment,
        dataset=dataset,
        scenario=scenario,
        settings=settings,
        replication_seed=replication_seed,
        task_metadata=task_metadata,
    )

    start = time.perf_counter()
    prowl_u0_nuisance = fit_arm_value_nuisance_model(
        x=train.x,
        treatment=train.a,
        lower_reward=train.reward,
        feature_map_factory=nuisance_feature_map_factory,
        l2_penalty=float(settings["nuisance_l2_penalty"]),
    )
    prowl_u0 = PROWL(
        eta_grid=eta_grid,
        gamma_grid=gamma_grid,
        selection_mode="lcb",
        delta=float(settings["delta"]),
        prior_sd=float(settings["prior_sd"]),
        score_bound=float(settings["score_bound"]),
        feature_map=feature_map_factory(),
        nuisance_feature_map_factory=nuisance_feature_map_factory,
        nuisance_l2_penalty=float(settings["nuisance_l2_penalty"]),
        particle_library_config=particle_cfg,
        random_state=replication_seed + 47,
    ).fit(
        x=train.x,
        treatment=train.a,
        reward=train.reward,
        propensity=train.pi,
        lower_reward=train.reward,
        nuisance_model=prowl_u0_nuisance,
        method_label=PROWL_U0_METHOD,
    )
    u0_diag = next(item for item in prowl_u0.diagnostics if abs(item.eta - prowl_u0.eta) <= 1e-12)
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method=PROWL_U0_METHOD,
        policy=prowl_u0,
        runtime_sec=time.perf_counter() - start,
        deployment_mode=deployment_mode,
        task_metadata=task_metadata,
        fit_info={
            "eta": prowl_u0.eta,
            "gamma": prowl_u0.gamma,
            "lcb": u0_diag.exact_value_lcb,
            "kl_to_prior": u0_diag.kl_to_prior,
            "selected_penalty": math.nan,
            "uses_certificate": False,
        },
    )

    start = time.perf_counter()
    prowl = PROWL(
        eta_grid=eta_grid,
        gamma_grid=gamma_grid,
        selection_mode="lcb",
        delta=float(settings["delta"]),
        prior_sd=float(settings["prior_sd"]),
        score_bound=float(settings["score_bound"]),
        feature_map=feature_map_factory(),
        nuisance_feature_map_factory=nuisance_feature_map_factory,
        nuisance_l2_penalty=float(settings["nuisance_l2_penalty"]),
        particle_library_config=particle_cfg,
        random_state=replication_seed + 53,
    ).fit(
        x=train.x,
        treatment=train.a,
        reward=train.reward,
        propensity=train.pi,
        lower_reward=train.lower_reward,
        nuisance_model=nuisance_model,
        method_label="PROWL (tuned eta)",
    )
    tuned_diag = next(item for item in prowl.diagnostics if abs(item.eta - prowl.eta) <= 1e-12)
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="PROWL (tuned eta)",
        policy=prowl,
        runtime_sec=time.perf_counter() - start,
        deployment_mode=deployment_mode,
        task_metadata=task_metadata,
        fit_info={
            "eta": prowl.eta,
            "gamma": prowl.gamma,
            "lcb": tuned_diag.exact_value_lcb,
            "kl_to_prior": tuned_diag.kl_to_prior,
            "selected_penalty": math.nan,
            "uses_certificate": True,
        },
    )
    return rows


def fit_split_free_comparison_rows(
    *,
    experiment: str,
    dataset: GeneratedDataset,
    scenario: ScenarioDefinition,
    settings: dict[str, Any],
    replication_seed: int,
    task_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    feature_map_factory = scenario.feature_map_factory
    nuisance_feature_map_factory = scenario.nuisance_feature_map_factory or feature_map_factory
    deployment_mode = deployment_mode_name(settings["prowl_deployment"])
    eta_grid = parse_float_grid(settings["eta_grid"])
    gamma_grid = parse_float_grid(settings["gamma_grid"])
    particle_cfg = particle_library_config_from_settings(settings)
    train = dataset.train
    rows: list[dict[str, Any]] = []

    split_free_nuisance = fit_arm_value_nuisance_model(
        x=train.x,
        treatment=train.a,
        lower_reward=train.lower_reward,
        feature_map_factory=nuisance_feature_map_factory,
        l2_penalty=float(settings["nuisance_l2_penalty"]),
    )
    start = time.perf_counter()
    split_free = PROWL(
        eta_grid=eta_grid,
        gamma_grid=gamma_grid,
        selection_mode="lcb",
        delta=float(settings["delta"]),
        prior_sd=float(settings["prior_sd"]),
        score_bound=float(settings["score_bound"]),
        feature_map=feature_map_factory(),
        nuisance_feature_map_factory=nuisance_feature_map_factory,
        nuisance_l2_penalty=float(settings["nuisance_l2_penalty"]),
        particle_library_config=particle_cfg,
        random_state=replication_seed + 401,
    ).fit(
        x=train.x,
        treatment=train.a,
        reward=train.reward,
        propensity=train.pi,
        lower_reward=train.lower_reward,
        nuisance_model=split_free_nuisance,
        method_label="PROWL (split-free)",
    )
    diag = next(item for item in split_free.diagnostics if abs(item.eta - split_free.eta) <= 1e-12)
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="PROWL (split-free)",
        policy=split_free,
        runtime_sec=time.perf_counter() - start,
        deployment_mode=deployment_mode,
        task_metadata=task_metadata,
        fit_info={
            "sample_split": False,
            "policy_sample_size": int(train.x.shape[0]),
            "nuisance_sample_size": int(train.x.shape[0]),
            "eta": split_free.eta,
            "gamma": split_free.gamma,
            "lcb": diag.exact_value_lcb,
        },
    )

    nuisance_idx, policy_idx = stratified_honest_split(train.a, rng=np.random.default_rng(replication_seed + 409))
    nuisance_train_x = train.x[nuisance_idx]
    nuisance_train_a = train.a[nuisance_idx]
    nuisance_train_lower = train.lower_reward[nuisance_idx]
    policy_train_x = train.x[policy_idx]
    policy_train_a = train.a[policy_idx]
    policy_train_reward = train.reward[policy_idx]
    policy_train_lower = train.lower_reward[policy_idx]
    policy_train_pi = train.pi[policy_idx]
    honest_nuisance = fit_arm_value_nuisance_model(
        x=nuisance_train_x,
        treatment=nuisance_train_a,
        lower_reward=nuisance_train_lower,
        feature_map_factory=nuisance_feature_map_factory,
        l2_penalty=float(settings["nuisance_l2_penalty"]),
    )
    start = time.perf_counter()
    honest = PROWL(
        eta_grid=eta_grid,
        gamma_grid=gamma_grid,
        selection_mode="lcb",
        delta=float(settings["delta"]),
        prior_sd=float(settings["prior_sd"]),
        score_bound=float(settings["score_bound"]),
        feature_map=feature_map_factory(),
        nuisance_feature_map_factory=nuisance_feature_map_factory,
        nuisance_l2_penalty=float(settings["nuisance_l2_penalty"]),
        particle_library_config=particle_cfg,
        random_state=replication_seed + 419,
    ).fit(
        x=policy_train_x,
        treatment=policy_train_a,
        reward=policy_train_reward,
        propensity=policy_train_pi,
        lower_reward=policy_train_lower,
        nuisance_model=honest_nuisance,
        method_label="PROWL (sample-split)",
    )
    diag = next(item for item in honest.diagnostics if abs(item.eta - honest.eta) <= 1e-12)
    add_result_row(
        rows,
        experiment=experiment,
        scenario=scenario,
        dataset=dataset,
        replication=replication_seed,
        method="PROWL (sample-split)",
        policy=honest,
        runtime_sec=time.perf_counter() - start,
        deployment_mode=deployment_mode,
        task_metadata=task_metadata,
        fit_info={
            "sample_split": True,
            "policy_sample_size": int(policy_idx.size),
            "nuisance_sample_size": int(nuisance_idx.size),
            "eta": honest.eta,
            "gamma": honest.gamma,
            "lcb": diag.exact_value_lcb,
        },
    )
    return rows


def execute_tasks(
    *,
    tasks: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    n_jobs: int,
    progress_mode: str,
    progress_label: str,
) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    total = len(tasks)
    mode = str(progress_mode).strip().lower()
    use_tqdm = mode in {"auto", "tqdm"}
    disable_tqdm = mode == "quiet"

    def emit_update(pbar: tqdm | None, completed: int, progress: str) -> None:
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(str(progress), refresh=False)
        elif mode == "plain":
            print(f"[{completed}/{total}] {progress_label}: {progress}", flush=True)

    def emit_message(pbar: tqdm | None, message: str) -> None:
        if mode == "quiet":
            return
        if pbar is not None:
            pbar.write(str(message))
        else:
            print(str(message), flush=True)

    completed = 0
    with tqdm(
        total=total,
        desc=progress_label,
        unit="task",
        dynamic_ncols=True,
        smoothing=0.1,
        disable=disable_tqdm or not use_tqdm,
    ) as pbar:
        active_pbar = pbar if use_tqdm and not disable_tqdm else None
        if int(n_jobs) <= 1:
            for task in tasks:
                result = worker(task)
                all_rows.extend(result["rows"])
                completed += 1
                emit_update(active_pbar, completed, result["progress"])
        else:
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=int(n_jobs)) as executor:
                    future_to_task = {executor.submit(worker, task): task for task in tasks}
                    for future in concurrent.futures.as_completed(future_to_task):
                        result = future.result()
                        all_rows.extend(result["rows"])
                        completed += 1
                        emit_update(active_pbar, completed, result["progress"])
            except (OSError, PermissionError, NotImplementedError):
                emit_message(active_pbar, "process-based parallelism is unavailable; falling back to threads.")
                with concurrent.futures.ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
                    future_to_task = {executor.submit(worker, task): task for task in tasks}
                    for future in concurrent.futures.as_completed(future_to_task):
                        result = future.result()
                        all_rows.extend(result["rows"])
                        completed += 1
                        emit_update(active_pbar, completed, result["progress"])
    return pd.DataFrame(all_rows)


def family_comparison_replication_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    join_cols = [
        "experiment",
        "scenario_id",
        "scenario_name",
        "scenario_group",
        "geometry",
        "uncertainty",
        "n_total",
        "rho",
        "certificate_mode",
        "certificate_scale",
        "propensity_mode",
        "propensity_strength",
        "replication",
    ]
    prowl = raw_df.loc[
        raw_df["method"] == "PROWL (tuned eta)",
        join_cols
        + [
            "method",
            "deployment_mode",
            "target_value",
            "robust_value",
            "target_regret",
            "robust_regret",
        ],
    ].rename(
        columns={
            "method": "reference_method",
            "deployment_mode": "reference_deployment_mode",
            "target_value": "reference_target_value",
            "robust_value": "reference_robust_value",
            "target_regret": "reference_target_regret",
            "robust_regret": "reference_robust_regret",
        }
    )
    comps = raw_df.loc[raw_df["reward_variant"].isin(["raw", "lower_plugin"])].copy()
    comps["comparator_variant"] = comps["reward_variant"]
    merged = prowl.merge(
        comps[
            join_cols
            + [
                "method",
                "deployment_mode",
                "comparator_variant",
                "target_value",
                "robust_value",
                "target_regret",
                "robust_regret",
            ]
        ],
        on=join_cols,
        how="inner",
    )
    merged = merged.loc[merged["method"] != "PROWL (tuned eta)"].copy()
    merged["target_value_improvement"] = merged["reference_target_value"] - merged["target_value"]
    merged["robust_value_improvement"] = merged["reference_robust_value"] - merged["robust_value"]
    merged["target_regret_improvement"] = merged["target_regret"] - merged["reference_target_regret"]
    merged["robust_regret_improvement"] = merged["robust_regret"] - merged["reference_robust_regret"]
    return (
        merged.groupby(
            join_cols + ["reference_deployment_mode", "comparator_variant"],
            dropna=False,
        )
        .agg(
            n_comparators=("method", "nunique"),
            target_value_improvement=("target_value_improvement", "mean"),
            robust_value_improvement=("robust_value_improvement", "mean"),
            target_regret_improvement=("target_regret_improvement", "mean"),
            robust_regret_improvement=("robust_regret_improvement", "mean"),
            min_target_value_improvement=("target_value_improvement", "min"),
            min_robust_value_improvement=("robust_value_improvement", "min"),
            min_target_regret_improvement=("target_regret_improvement", "min"),
            min_robust_regret_improvement=("robust_regret_improvement", "min"),
        )
        .reset_index()
        .sort_values(["scenario_id", "rho", "n_total", "replication", "comparator_variant"])
        .reset_index(drop=True)
    )


def family_comparison_summary(rep_df: pd.DataFrame) -> pd.DataFrame:
    if rep_df.empty:
        return rep_df
    group_cols = [
        "experiment",
        "scenario_id",
        "scenario_name",
        "scenario_group",
        "geometry",
        "uncertainty",
        "n_total",
        "rho",
        "certificate_mode",
        "certificate_scale",
        "propensity_mode",
        "propensity_strength",
        "reference_deployment_mode",
        "comparator_variant",
    ]
    return (
        rep_df.groupby(group_cols, dropna=False)
        .agg(
            n_replications=("replication", "count"),
            n_comparators=("n_comparators", "mean"),
            target_value_improvement=("target_value_improvement", "mean"),
            target_value_improvement_se=("target_value_improvement", "sem"),
            robust_value_improvement=("robust_value_improvement", "mean"),
            robust_value_improvement_se=("robust_value_improvement", "sem"),
            target_regret_improvement=("target_regret_improvement", "mean"),
            target_regret_improvement_se=("target_regret_improvement", "sem"),
            robust_regret_improvement=("robust_regret_improvement", "mean"),
            robust_regret_improvement_se=("robust_regret_improvement", "sem"),
            min_target_regret_improvement=("min_target_regret_improvement", "mean"),
            min_target_regret_improvement_se=("min_target_regret_improvement", "sem"),
            min_robust_regret_improvement=("min_robust_regret_improvement", "mean"),
            min_robust_regret_improvement_se=("min_robust_regret_improvement", "sem"),
            always_family_target_positive=(
                "target_regret_improvement",
                lambda values: bool(np.all(np.asarray(values) > 0.0)),
            ),
            always_family_robust_positive=(
                "robust_regret_improvement",
                lambda values: bool(np.all(np.asarray(values) > 0.0)),
            ),
            always_individual_target_positive=(
                "min_target_regret_improvement",
                lambda values: bool(np.all(np.asarray(values) > 0.0)),
            ),
            always_individual_robust_positive=(
                "min_robust_regret_improvement",
                lambda values: bool(np.all(np.asarray(values) > 0.0)),
            ),
        )
        .reset_index()
        .sort_values(["scenario_id", "rho", "n_total", "comparator_variant"])
        .reset_index(drop=True)
    )


def summarize_metrics(
    raw_df: pd.DataFrame,
    *,
    group_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    agg_kwargs: dict[str, Any] = {"n_replications": ("replication", "count")}
    for metric in metrics:
        agg_kwargs[metric] = (metric, "mean")
        agg_kwargs[f"{metric}_se"] = (metric, "sem")
    return (
        raw_df.groupby(group_cols, dropna=False)
        .agg(**agg_kwargs)
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def generic_pairwise_gap_summary(
    raw_df: pd.DataFrame,
    *,
    group_cols: list[str],
    method_pairs: list[tuple[str, str, str]],
    metrics: list[str],
) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    pivot = raw_df.pivot_table(
        index=group_cols + ["replication"],
        columns="method",
        values=metrics,
    )
    rows: list[dict[str, Any]] = []
    for idx, values in pivot.iterrows():
        if not isinstance(idx, tuple):
            idx = (idx,)
        base = {name: value for name, value in zip(group_cols + ["replication"], idx, strict=True)}
        for left, right, label in method_pairs:
            row = dict(base)
            row["gap_label"] = label
            valid_metric = False
            for metric in metrics:
                left_value = values.get((metric, left), math.nan)
                right_value = values.get((metric, right), math.nan)
                if np.isfinite(left_value) and np.isfinite(right_value):
                    row[f"{metric}_gap"] = float(left_value - right_value)
                    valid_metric = True
                else:
                    row[f"{metric}_gap"] = math.nan
            if valid_metric:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    gap_df = pd.DataFrame(rows)
    agg_kwargs: dict[str, Any] = {"n_replications": ("replication", "count")}
    for metric in metrics:
        gap_col = f"{metric}_gap"
        agg_kwargs[gap_col] = (gap_col, "mean")
        agg_kwargs[f"{gap_col}_positive_rate"] = (
            gap_col,
            lambda s, col=gap_col: float(np.mean(np.asarray(s, dtype=float) > 0.0)),
        )
    return (
        gap_df.groupby(group_cols + ["gap_label"], dropna=False)
        .agg(**agg_kwargs)
        .reset_index()
        .sort_values(group_cols + ["gap_label"])
        .reset_index(drop=True)
    )


def generic_method_style(method: str, idx: int) -> dict[str, Any]:
    family = method_family_name(method)
    if family in METHOD_FAMILY_COLORS:
        return {
            "color": METHOD_FAMILY_COLORS[family],
            "marker": METHOD_FAMILY_MARKERS[family],
            "linestyle": METHOD_FAMILY_LINE_STYLES[family],
            "linewidth": METHOD_FAMILY_LINEWIDTHS[family],
            "zorder": METHOD_FAMILY_ZORDERS[family],
        }
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    return {
        "color": color_cycle[idx % len(color_cycle)],
        "marker": markers[idx % len(markers)],
        "linestyle": linestyles[idx % len(linestyles)],
        "linewidth": 4.0,
        "zorder": 3,
    }


def plot_generic_method_grid(
    summary_df: pd.DataFrame,
    *,
    scenario_ids: list[str],
    x_col: str,
    x_label: str,
    metric_cols: list[str],
    method_order: list[str] | None,
    legend_ncol: int | None = None,
    output_path: Path,
) -> None:
    if summary_df.empty or not scenario_ids:
        return
    n_rows = len(metric_cols)
    n_cols = len(scenario_ids)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.8 * n_cols, 5.8 * n_rows),
        constrained_layout=True,
        sharex="col",
        sharey="row",
    )
    axes = np.asarray(axes)
    if n_rows == 1 and n_cols == 1:
        axes = axes.reshape(1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    methods = method_order or sorted(summary_df["method"].unique())
    style_map = {method: generic_method_style(method, idx) for idx, method in enumerate(methods)}
    for col_idx, scenario_id in enumerate(scenario_ids):
        frame = summary_df.loc[summary_df["scenario_id"] == scenario_id].copy()
        if frame.empty:
            for row_idx in range(n_rows):
                axes[row_idx, col_idx].set_visible(False)
            continue
        for row_idx, metric_col in enumerate(metric_cols):
            ax = axes[row_idx, col_idx]
            for method in methods:
                method_frame = frame.loc[frame["method"] == method].sort_values(x_col)
                if method_frame.empty:
                    continue
                style = style_map[method]
                x = method_frame[x_col].to_numpy(dtype=float)
                y = method_frame[metric_col].to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=11.0 + MARKER_SIZE_BOOST,
                    markeredgewidth=1.6,
                    zorder=style["zorder"],
                )
            ax.grid(alpha=0.25, linewidth=0.95)
            if row_idx == 0:
                ax.set_title(scenario_display_label(scenario_id))
            if col_idx == 0:
                ax.set_ylabel(METRIC_DISPLAY_LABELS.get(metric_col, metric_col.replace("_", " ").title()))
            if row_idx == n_rows - 1:
                ax.set_xlabel(x_label)
            if x_col == "n_total":
                ax.set_xscale("log")
                ax.set_xticks(sorted(frame[x_col].unique()))
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    legend_handles = []
    legend_labels = []
    for method in methods:
        style = style_map[method]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=11.0 + MARKER_SIZE_BOOST,
                markeredgewidth=1.6,
            )
        )
        legend_labels.append(method_display_label(method))
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(int(legend_ncol) if legend_ncol is not None else 4, len(legend_labels)),
        bbox_to_anchor=(0.5, 1.08 + LEGEND_Y_SHIFT),
        frameon=False,
        fontsize=22 + FONT_SIZE_BOOST,
        handlelength=2.8,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def require_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"plots-only requested, but required file is missing: {path}")
    return pd.read_csv(path)


def method_regret_summary(
    raw_df: pd.DataFrame,
    *,
    scenario_id: str,
    x_col: str,
    comparator_variant: str,
) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    baseline_methods = RAW_BASELINE_METHODS if comparator_variant == "raw" else LOWER_PLUGIN_METHODS
    methods = ["PROWL (tuned eta)", PROWL_U0_METHOD, *baseline_methods]
    frame = raw_df.loc[
        (raw_df["scenario_id"] == scenario_id) & (raw_df["method"].isin(methods)),
        ["scenario_id", x_col, "method", "target_regret", "robust_regret", "replication"],
    ].copy()
    if frame.empty:
        return frame
    return (
        frame.groupby(["scenario_id", x_col, "method"], dropna=False)
        .agg(
            n_replications=("replication", "count"),
            target_regret=("target_regret", "mean"),
            target_regret_se=("target_regret", "sem"),
            robust_regret=("robust_regret", "mean"),
            robust_regret_se=("robust_regret", "sem"),
        )
        .reset_index()
        .sort_values([x_col, "method"])
        .reset_index(drop=True)
    )


def plot_method_regret_comparison(
    summary_df: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    scenario_id: str,
    comparator_variant: str,
    output_path: Path,
) -> None:
    if summary_df.empty:
        return
    method_order = comparator_method_order(comparator_variant)

    fig, axes = plt.subplots(1, 2, figsize=(19.0, 8.2), constrained_layout=True, sharex=True)
    metric_specs = [
        ("target_regret", "Target Regret"),
        ("robust_regret", "Robust Regret"),
    ]
    for ax, (metric_col, panel_title) in zip(axes, metric_specs, strict=True):
        for method in method_order:
            frame = summary_df.loc[summary_df["method"] == method].sort_values(x_col)
            if frame.empty:
                continue
            x = frame[x_col].to_numpy(dtype=float)
            mean = frame[metric_col].to_numpy(dtype=float)
            family_name = method_family_name(method)
            ax.plot(
                x,
                mean,
                color=METHOD_FAMILY_COLORS[family_name],
                marker=METHOD_FAMILY_MARKERS[family_name],
                linestyle=METHOD_FAMILY_LINE_STYLES[family_name],
                linewidth=METHOD_FAMILY_LINEWIDTHS[family_name],
                markersize=12.5 + MARKER_SIZE_BOOST,
                markeredgewidth=1.8,
                label=method_display_label(method),
                zorder=METHOD_FAMILY_ZORDERS[family_name],
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(panel_title, rotation=90, labelpad=14)
        ax.grid(alpha=0.25, linewidth=1.0)
        if x_col == "n_total":
            ax.set_xscale("log")
            ax.set_xticks(sorted(summary_df[x_col].unique()))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.15 + LEGEND_Y_SHIFT),
            frameon=False,
            fontsize=24 + FONT_SIZE_BOOST,
            handlelength=3.0,
            columnspacing=1.6,
            handletextpad=0.7,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_two_panel_method_regret_scenario(
    raw_summary_df: pd.DataFrame,
    lower_summary_df: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    scenario_id: str,
    panel_specs: list[tuple[str, str, str, str]],
    output_path: Path,
) -> None:
    if not scenario_id:
        return
    summary_by_variant = {
        "raw": raw_summary_df,
        "lower_plugin": lower_summary_df,
    }
    fig, axes = plt.subplots(1, len(panel_specs), figsize=(18.8, 7.9), constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (metric_col, metric_label, comparator_variant, family_label) in zip(axes, panel_specs, strict=True):
        summary_df = summary_by_variant[comparator_variant]
        method_order = comparator_method_order(comparator_variant, shared_legend_order=True)
        frame = summary_df.loc[summary_df["scenario_id"] == scenario_id].sort_values(x_col)
        if frame.empty:
            ax.set_visible(False)
            continue
        panel_label = f"{metric_label} ({family_label} family)"
        for method in method_order:
            method_frame = frame.loc[frame["method"] == method].sort_values(x_col)
            if method_frame.empty:
                continue
            x = method_frame[x_col].to_numpy(dtype=float)
            mean = method_frame[metric_col].to_numpy(dtype=float)
            family_name = method_family_name(method)
            ax.plot(
                x,
                mean,
                color=METHOD_FAMILY_COLORS[family_name],
                marker=METHOD_FAMILY_MARKERS[family_name],
                linestyle=METHOD_FAMILY_LINE_STYLES[family_name],
                linewidth=METHOD_FAMILY_LINEWIDTHS[family_name],
                markersize=12.5 + MARKER_SIZE_BOOST,
                markeredgewidth=1.8,
                zorder=METHOD_FAMILY_ZORDERS[family_name],
            )
        ax.grid(alpha=0.25, linewidth=1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(panel_label, rotation=90, labelpad=14)
        if x_col == "n_total":
            ax.set_xscale("log")
            ax.set_xticks(sorted(frame[x_col].unique()))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_FAMILY_COLORS[label],
            marker=METHOD_FAMILY_MARKERS[label],
            linestyle=METHOD_FAMILY_LINE_STYLES[label],
            linewidth=METHOD_FAMILY_LINEWIDTHS[label],
            markersize=12.5 + MARKER_SIZE_BOOST,
            markeredgewidth=1.8,
            label=METHOD_FAMILY_DISPLAY_LABELS[label],
        )
        for label in METHOD_FAMILY_ORDER
    ]
    fig.legend(
        legend_handles,
        [METHOD_FAMILY_DISPLAY_LABELS[label] for label in METHOD_FAMILY_ORDER],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.15 + LEGEND_Y_SHIFT),
        frameon=False,
        fontsize=24 + FONT_SIZE_BOOST,
        handlelength=3.0,
        columnspacing=1.6,
        handletextpad=0.7,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_two_panel_family_improvement(
    summary_df: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    title: str,
    output_path: Path,
) -> None:
    if summary_df.empty:
        return
    variants = [
        ("raw", r"PROWL vs. $R$ baselines"),
        ("lower_plugin", rf"PROWL vs. {UNDERLINE_R_LABEL} plug-in baselines"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 7.6), constrained_layout=True, sharey=True)
    for ax, (variant, panel_title) in zip(axes, variants, strict=True):
        frame = summary_df.loc[summary_df["comparator_variant"] == variant].sort_values(x_col)
        if frame.empty:
            ax.set_visible(False)
            continue
        x = frame[x_col].to_numpy(dtype=float)
        target_mean = frame["target_regret_improvement"].to_numpy(dtype=float)
        robust_mean = frame["robust_regret_improvement"].to_numpy(dtype=float)
        ax.axhline(0.0, color="0.55", linewidth=1.2, linestyle="--")
        ax.plot(
            x,
            target_mean,
            color="#1f77b4",
            marker="o",
            linestyle="-",
            linewidth=4.2,
            markersize=11.0 + MARKER_SIZE_BOOST,
            label="Target Regret Improvement",
        )
        ax.plot(
            x,
            robust_mean,
            color="#d62728",
            marker="s",
            linestyle="--",
            linewidth=4.2,
            markersize=11.0 + MARKER_SIZE_BOOST,
            label="Robust Regret Improvement",
        )
        ax.set_title(panel_title)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.25, linewidth=0.95)
        if x_col == "n_total":
            ax.set_xscale("log")
            ax.set_xticks(sorted(frame[x_col].unique()))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        all_target = bool(frame["always_individual_target_positive"].all())
        all_robust = bool(frame["always_individual_robust_positive"].all())
        ax.text(
            0.02,
            0.03,
            f"All-point target dominance: {'yes' if all_target else 'no'}\nAll-point robust dominance: {'yes' if all_robust else 'no'}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14 + FONT_SIZE_BOOST,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
    axes[0].set_ylabel("Baseline Regret - PROWL Regret")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.12 + LEGEND_Y_SHIFT),
            frameon=False,
            fontsize=24 + FONT_SIZE_BOOST,
            handlelength=3.0,
            columnspacing=1.8,
            handletextpad=0.7,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def rho_sweep_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    rho_grid = parse_float_grid(args.rho_grid)
    scenario_ids = rho_scenario_ids(args)
    settings = dict(vars(args))
    settings["prowl_deployment"] = resolve_experiment_prowl_deployment(str(args.prowl_deployment), "rho_sweep")
    for scenario_idx, scenario_id in enumerate(scenario_ids):
        for rho_idx, rho in enumerate(rho_grid):
            for replication in range(int(args.rho_replications)):
                seed = int(
                    args.seed
                    + 10_000 * replication
                    + 100_000 * rho_idx
                    + 1_000_000 * args.rho_n_total
                    + 100_000_000 * scenario_idx
                )
                tasks.append(
                    {
                        "experiment": "rho_sweep",
                        "scenario_id": scenario_id,
                        "n_total": int(args.rho_n_total),
                        "rho": float(rho),
                        "replication": int(replication),
                        "seed": seed,
                        "settings": settings,
                    }
                )
    return tasks


def n_sweep_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    scenario_ids = parse_str_grid(args.n_sweep_scenarios)
    n_grid = parse_int_grid(args.n_sweep_grid)
    settings = dict(vars(args))
    settings["prowl_deployment"] = resolve_experiment_prowl_deployment(str(args.prowl_deployment), "n_sweep")
    for scenario_idx, scenario_id in enumerate(scenario_ids):
        for n_idx, n_total in enumerate(n_grid):
            for replication in range(int(args.n_sweep_replications)):
                seed = int(
                    args.seed + 10_000 * replication + 1_000_000 * n_idx + 100_000_000 * scenario_idx + 1_000 * n_total
                )
                tasks.append(
                    {
                        "experiment": "n_sweep",
                        "scenario_id": scenario_id,
                        "n_total": int(n_total),
                        "rho": float(args.n_sweep_rho),
                        "replication": int(replication),
                        "seed": seed,
                        "settings": settings,
                    }
                )
    return tasks


def certificate_diagnostic_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    scenario_ids = parse_str_grid(args.certificate_diagnostic_scenarios)
    rho_grid = parse_float_grid(args.certificate_diagnostic_rho_grid)
    settings = dict(vars(args))
    settings["prowl_deployment"] = resolve_experiment_prowl_deployment(
        str(args.prowl_deployment), "certificate_diagnostics"
    )
    for scenario_idx, scenario_id in enumerate(scenario_ids):
        for rho_idx, rho in enumerate(rho_grid):
            for replication in range(int(args.certificate_diagnostic_replications)):
                seed = int(
                    args.seed
                    + 10_000 * replication
                    + 100_000 * rho_idx
                    + 1_000_000 * int(args.certificate_diagnostic_n_total)
                    + 100_000_000 * scenario_idx
                )
                tasks.append(
                    {
                        "experiment": "certificate_diagnostics",
                        "scenario_id": scenario_id,
                        "n_total": int(args.certificate_diagnostic_n_total),
                        "rho": float(rho),
                        "replication": int(replication),
                        "seed": seed,
                        "settings": settings,
                    }
                )
    return tasks


def split_free_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    scenario_ids = parse_str_grid(args.split_free_scenarios)
    n_grid = parse_int_grid(args.split_free_grid)
    settings = dict(vars(args))
    settings["prowl_deployment"] = resolve_experiment_prowl_deployment(
        str(args.prowl_deployment), "split_free_ablation"
    )
    for scenario_idx, scenario_id in enumerate(scenario_ids):
        for n_idx, n_total in enumerate(n_grid):
            for replication in range(int(args.split_free_replications)):
                seed = int(
                    args.seed + 10_000 * replication + 1_000_000 * n_idx + 100_000_000 * scenario_idx + 1_000 * n_total
                )
                tasks.append(
                    {
                        "experiment": "split_free_ablation",
                        "scenario_id": scenario_id,
                        "n_total": int(n_total),
                        "rho": float(args.split_free_rho),
                        "replication": int(replication),
                        "seed": seed,
                        "settings": settings,
                    }
                )
    return tasks


def _run_family_task(task: dict[str, Any]) -> dict[str, Any]:
    settings = dict(task["settings"])
    dataset = build_dataset(
        scenario_id=task["scenario_id"],
        n_total=int(task["n_total"]),
        n_test=int(settings["n_test"]),
        rho=float(task["rho"]),
        certificate_mode=str(settings["certificate_mode"]),
        certificate_scale=float(settings["certificate_scale"]),
        seed=int(task["seed"]),
        n_certificate_auxiliary=int(settings["n_certificate_auxiliary"]),
        propensity_mode=str(settings["propensity_mode"]),
        propensity_strength=float(settings["propensity_strength"]),
    )
    scenario = scenario_library()[task["scenario_id"]]
    rows = fit_baselines_and_tuned_prowl(
        experiment=str(task["experiment"]),
        dataset=dataset,
        scenario=scenario,
        settings=settings,
        replication_seed=int(task["seed"]),
        task_metadata={
            "sweep_axis": "rho" if task["experiment"] == "rho_sweep" else "n_total",
            "sweep_value": float(task["rho"]) if task["experiment"] == "rho_sweep" else int(task["n_total"]),
        },
    )
    return {
        "rows": rows,
        "progress": f"{task['experiment']} {task['scenario_id']} value={task['rho'] if task['experiment'] == 'rho_sweep' else task['n_total']} rep={task['replication'] + 1}",
    }


def _run_certificate_diagnostic_task(task: dict[str, Any]) -> dict[str, Any]:
    settings = dict(task["settings"])
    dataset = build_dataset(
        scenario_id=task["scenario_id"],
        n_total=int(task["n_total"]),
        n_test=int(settings["n_test"]),
        rho=float(task["rho"]),
        certificate_mode=str(settings["certificate_mode"]),
        certificate_scale=float(settings["certificate_scale"]),
        seed=int(task["seed"]),
        n_certificate_auxiliary=int(settings["n_certificate_auxiliary"]),
        propensity_mode=str(settings["propensity_mode"]),
        propensity_strength=float(settings["propensity_strength"]),
    )
    scenario = scenario_library()[task["scenario_id"]]
    rows = fit_baselines_and_tuned_prowl(
        experiment="certificate_diagnostics",
        dataset=dataset,
        scenario=scenario,
        settings=settings,
        replication_seed=int(task["seed"]),
        task_metadata={"sweep_axis": "rho", "sweep_value": float(task["rho"])},
    )
    return {
        "rows": rows,
        "progress": f"certificate_diagnostics {task['scenario_id']} rho={task['rho']} rep={task['replication'] + 1}",
    }


def _run_split_free_task(task: dict[str, Any]) -> dict[str, Any]:
    settings = dict(task["settings"])
    dataset = build_dataset(
        scenario_id=task["scenario_id"],
        n_total=int(task["n_total"]),
        n_test=int(settings["n_test"]),
        rho=float(task["rho"]),
        certificate_mode=str(settings["certificate_mode"]),
        certificate_scale=float(settings["certificate_scale"]),
        seed=int(task["seed"]),
        n_certificate_auxiliary=int(settings["n_certificate_auxiliary"]),
        propensity_mode=str(settings["propensity_mode"]),
        propensity_strength=float(settings["propensity_strength"]),
    )
    scenario = scenario_library()[task["scenario_id"]]
    rows = fit_split_free_comparison_rows(
        experiment="split_free_ablation",
        dataset=dataset,
        scenario=scenario,
        settings=settings,
        replication_seed=int(task["seed"]),
        task_metadata={"sweep_axis": "n_total", "sweep_value": int(task["n_total"])},
    )
    return {
        "rows": rows,
        "progress": f"split_free_ablation {task['scenario_id']} N={task['n_total']} rep={task['replication'] + 1}",
    }


def run_rho_sweep(args: argparse.Namespace, output_root: Path) -> None:
    output_dir = output_root / "rho_sweep"
    raw_path = output_dir / "raw_metrics.csv"
    rep_path = output_dir / "family_replication_comparison.csv"
    summary_path = output_dir / "family_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.plots_only):
        raw_df = require_existing_csv(raw_path)
        if rep_path.exists():
            rep_df = pd.read_csv(rep_path)
        else:
            rep_df = family_comparison_replication_df(raw_df)
            rep_df.to_csv(rep_path, index=False)
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = family_comparison_summary(rep_df)
            summary_df.to_csv(summary_path, index=False)
    else:
        tasks = rho_sweep_tasks(args)
        raw_df = execute_tasks(
            tasks=tasks,
            worker=_run_family_task,
            n_jobs=int(args.n_jobs),
            progress_mode=str(args.progress),
            progress_label="rho_sweep",
        )
        rep_df = family_comparison_replication_df(raw_df)
        summary_df = family_comparison_summary(rep_df)
        raw_df.to_csv(raw_path, index=False)
        rep_df.to_csv(rep_path, index=False)
        summary_df.to_csv(summary_path, index=False)

    scenario_ids = rho_scenario_ids(args)
    resolved_deployment = resolve_experiment_prowl_deployment(str(args.prowl_deployment), "rho_sweep")
    raw_method_frames: list[pd.DataFrame] = []
    lower_method_frames: list[pd.DataFrame] = []
    for scenario_id in scenario_ids:
        raw_method_summary = method_regret_summary(
            raw_df,
            scenario_id=str(scenario_id),
            x_col="rho",
            comparator_variant="raw",
        )
        lower_method_summary = method_regret_summary(
            raw_df,
            scenario_id=str(scenario_id),
            x_col="rho",
            comparator_variant="lower_plugin",
        )
        raw_method_frames.append(raw_method_summary)
        lower_method_frames.append(lower_method_summary)
        raw_method_summary.to_csv(output_dir / f"{scenario_id}_rho_method_summary_raw.csv", index=False)
        lower_method_summary.to_csv(output_dir / f"{scenario_id}_rho_method_summary_lower_plugin.csv", index=False)
        plot_method_regret_comparison(
            raw_method_summary,
            x_col="rho",
            x_label=RHO_LABEL,
            scenario_id=str(scenario_id),
            comparator_variant="raw",
            output_path=output_dir / "plots" / f"{scenario_id}_rho_sweep_vs_r_baselines.png",
        )
        plot_method_regret_comparison(
            lower_method_summary,
            x_col="rho",
            x_label=RHO_LABEL,
            scenario_id=str(scenario_id),
            comparator_variant="lower_plugin",
            output_path=output_dir / "plots" / f"{scenario_id}_rho_sweep_vs_lower_plugin_baselines.png",
        )
    raw_method_summary_all = pd.concat(raw_method_frames, ignore_index=True)
    lower_method_summary_all = pd.concat(lower_method_frames, ignore_index=True)
    raw_method_summary_all.to_csv(output_dir / "rho_method_summary_raw.csv", index=False)
    lower_method_summary_all.to_csv(output_dir / "rho_method_summary_lower_plugin.csv", index=False)
    comparison_scenario_ids = manuscript_comparison_scenario_ids(scenario_ids)
    for scenario_id in comparison_scenario_ids:
        plot_two_panel_method_regret_scenario(
            raw_method_summary_all,
            lower_method_summary_all,
            x_col="rho",
            x_label=RHO_LABEL,
            scenario_id=str(scenario_id),
            panel_specs=[
                ("target_regret", "Target Regret", "raw", r"$R$"),
                ("robust_regret", "Robust Regret", "lower_plugin", UNDERLINE_R_LABEL),
            ],
            output_path=output_dir / "plots" / f"{scenario_id}_rho_sweep_target_raw_robust_lower_plugin.png",
        )
        plot_two_panel_method_regret_scenario(
            raw_method_summary_all,
            lower_method_summary_all,
            x_col="rho",
            x_label=RHO_LABEL,
            scenario_id=str(scenario_id),
            panel_specs=[
                ("target_regret", "Target Regret", "lower_plugin", UNDERLINE_R_LABEL),
                ("robust_regret", "Robust Regret", "raw", r"$R$"),
            ],
            output_path=output_dir / "plots" / f"{scenario_id}_rho_sweep_target_lower_plugin_robust_raw.png",
        )
    write_manifest(
        output_dir / "manifest.json",
        {
            "experiment": "rho_sweep",
            "scenario_ids": scenario_ids,
            "n_total": int(args.rho_n_total),
            "n_replications": int(args.rho_replications),
            "rho_grid": parse_float_grid(args.rho_grid),
            "prowl_deployment": resolved_deployment,
            "rows": int(raw_df.shape[0]),
            "plots_only": bool(args.plots_only),
        },
    )


def run_n_sweep(args: argparse.Namespace, output_root: Path) -> None:
    output_dir = output_root / "n_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_metrics.csv"
    rep_path = output_dir / "family_replication_comparison.csv"
    summary_path = output_dir / "family_summary.csv"
    raw_method_path = output_dir / "n_method_summary_raw.csv"
    lower_method_path = output_dir / "n_method_summary_lower_plugin.csv"
    if bool(args.plots_only):
        raw_df = require_existing_csv(raw_path)
        if rep_path.exists():
            rep_df = pd.read_csv(rep_path)
        else:
            rep_df = family_comparison_replication_df(raw_df)
            rep_df.to_csv(rep_path, index=False)
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = family_comparison_summary(rep_df)
            summary_df.to_csv(summary_path, index=False)
    else:
        tasks = n_sweep_tasks(args)
        raw_df = execute_tasks(
            tasks=tasks,
            worker=_run_family_task,
            n_jobs=int(args.n_jobs),
            progress_mode=str(args.progress),
            progress_label="n_sweep",
        )
        rep_df = family_comparison_replication_df(raw_df)
        summary_df = family_comparison_summary(rep_df)
        raw_df.to_csv(raw_path, index=False)
        rep_df.to_csv(rep_path, index=False)
        summary_df.to_csv(summary_path, index=False)
    raw_method_frames: list[pd.DataFrame] = []
    lower_method_frames: list[pd.DataFrame] = []
    resolved_deployment = resolve_experiment_prowl_deployment(str(args.prowl_deployment), "n_sweep")
    for scenario_id in parse_str_grid(args.n_sweep_scenarios):
        raw_method_summary = method_regret_summary(
            raw_df,
            scenario_id=str(scenario_id),
            x_col="n_total",
            comparator_variant="raw",
        )
        lower_method_summary = method_regret_summary(
            raw_df,
            scenario_id=str(scenario_id),
            x_col="n_total",
            comparator_variant="lower_plugin",
        )
        raw_method_frames.append(raw_method_summary)
        lower_method_frames.append(lower_method_summary)
        raw_method_summary.to_csv(output_dir / f"{scenario_id}_n_method_summary_raw.csv", index=False)
        lower_method_summary.to_csv(output_dir / f"{scenario_id}_n_method_summary_lower_plugin.csv", index=False)
        plot_method_regret_comparison(
            raw_method_summary,
            x_col="n_total",
            x_label=N_TOTAL_LABEL,
            scenario_id=str(scenario_id),
            comparator_variant="raw",
            output_path=output_dir / "plots" / f"{scenario_id}_n_sweep_vs_r_baselines.png",
        )
        plot_method_regret_comparison(
            lower_method_summary,
            x_col="n_total",
            x_label=N_TOTAL_LABEL,
            scenario_id=str(scenario_id),
            comparator_variant="lower_plugin",
            output_path=output_dir / "plots" / f"{scenario_id}_n_sweep_vs_lower_plugin_baselines.png",
        )
        plot_two_panel_family_improvement(
            summary_df.loc[summary_df["scenario_id"] == scenario_id].copy(),
            x_col="n_total",
            x_label=N_TOTAL_LABEL,
            title=f"{scenario_display_label(scenario_id)}: PROWL improvement across {N_TOTAL_LABEL}",
            output_path=output_dir / "plots" / f"{scenario_id}_n_sweep.png",
        )
    if raw_method_frames:
        raw_method_summary_all = pd.concat(raw_method_frames, ignore_index=True)
        raw_method_summary_all.to_csv(raw_method_path, index=False)
    else:
        raw_method_summary_all = pd.DataFrame()
    if lower_method_frames:
        lower_method_summary_all = pd.concat(lower_method_frames, ignore_index=True)
        lower_method_summary_all.to_csv(lower_method_path, index=False)
    else:
        lower_method_summary_all = pd.DataFrame()
    comparison_scenario_ids = manuscript_comparison_scenario_ids(parse_str_grid(args.n_sweep_scenarios))
    for scenario_id in comparison_scenario_ids:
        plot_two_panel_method_regret_scenario(
            raw_method_summary_all,
            lower_method_summary_all,
            x_col="n_total",
            x_label=N_TOTAL_LABEL,
            scenario_id=str(scenario_id),
            panel_specs=[
                ("target_regret", "Target Regret", "raw", r"$R$"),
                ("robust_regret", "Robust Regret", "lower_plugin", UNDERLINE_R_LABEL),
            ],
            output_path=output_dir / "plots" / f"{scenario_id}_n_sweep_target_raw_robust_lower_plugin.png",
        )
        plot_two_panel_method_regret_scenario(
            raw_method_summary_all,
            lower_method_summary_all,
            x_col="n_total",
            x_label=N_TOTAL_LABEL,
            scenario_id=str(scenario_id),
            panel_specs=[
                ("target_regret", "Target Regret", "lower_plugin", UNDERLINE_R_LABEL),
                ("robust_regret", "Robust Regret", "raw", r"$R$"),
            ],
            output_path=output_dir / "plots" / f"{scenario_id}_n_sweep_target_lower_plugin_robust_raw.png",
        )
    write_manifest(
        output_dir / "manifest.json",
        {
            "experiment": "n_sweep",
            "scenario_ids": parse_str_grid(args.n_sweep_scenarios),
            "n_grid": parse_int_grid(args.n_sweep_grid),
            "n_replications": int(args.n_sweep_replications),
            "rho": float(args.n_sweep_rho),
            "prowl_deployment": resolved_deployment,
            "rows": int(raw_df.shape[0]),
            "plots_only": bool(args.plots_only),
        },
    )


def run_certificate_diagnostics(args: argparse.Namespace, output_root: Path) -> None:
    output_dir = output_root / "certificate_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_metrics.csv"
    policy_summary_path = output_dir / "policy_summary.csv"
    dataset_summary_path = output_dir / "dataset_summary.csv"
    if bool(args.plots_only):
        raw_df = require_existing_csv(raw_path)
        policy_summary = (
            pd.read_csv(policy_summary_path)
            if policy_summary_path.exists()
            else summarize_metrics(
                raw_df,
                group_cols=["scenario_id", "scenario_name", "scenario_group", "rho", "method"],
                metrics=[
                    "target_value",
                    "proxy_value",
                    "robust_value",
                    "target_regret",
                    "robust_regret",
                    "policy_mean_certificate",
                    "policy_mean_bias",
                    "policy_certificate_slack",
                    "policy_certificate_validity_rate",
                    "policy_clipping_rate",
                    "proxy_target_gap",
                    "target_robust_gap",
                    "proxy_robust_gap",
                ],
            )
        )
        if not policy_summary_path.exists():
            policy_summary.to_csv(policy_summary_path, index=False)
        dataset_summary = (
            pd.read_csv(dataset_summary_path)
            if dataset_summary_path.exists()
            else summarize_metrics(
                raw_df.drop_duplicates(subset=["scenario_id", "rho", "replication"]),
                group_cols=["scenario_id", "scenario_name", "scenario_group", "rho"],
                metrics=[
                    "train_mean_certificate",
                    "train_mean_bias",
                    "train_certificate_slack",
                    "train_clipping_rate",
                    "certificate_validity_rate_test",
                ],
            )
        )
        if not dataset_summary_path.exists():
            dataset_summary.to_csv(dataset_summary_path, index=False)
    else:
        tasks = certificate_diagnostic_tasks(args)
        raw_df = execute_tasks(
            tasks=tasks,
            worker=_run_certificate_diagnostic_task,
            n_jobs=int(args.n_jobs),
            progress_mode=str(args.progress),
            progress_label="certificate_diagnostics",
        )
        policy_summary = summarize_metrics(
            raw_df,
            group_cols=["scenario_id", "scenario_name", "scenario_group", "rho", "method"],
            metrics=[
                "target_value",
                "proxy_value",
                "robust_value",
                "target_regret",
                "robust_regret",
                "policy_mean_certificate",
                "policy_mean_bias",
                "policy_certificate_slack",
                "policy_certificate_validity_rate",
                "policy_clipping_rate",
                "proxy_target_gap",
                "target_robust_gap",
                "proxy_robust_gap",
            ],
        )
        dataset_summary = summarize_metrics(
            raw_df.drop_duplicates(subset=["scenario_id", "rho", "replication"]),
            group_cols=["scenario_id", "scenario_name", "scenario_group", "rho"],
            metrics=[
                "train_mean_certificate",
                "train_mean_bias",
                "train_certificate_slack",
                "train_clipping_rate",
                "certificate_validity_rate_test",
            ],
        )
        raw_df.to_csv(raw_path, index=False)
        policy_summary.to_csv(policy_summary_path, index=False)
        dataset_summary.to_csv(dataset_summary_path, index=False)
    plot_generic_method_grid(
        policy_summary,
        scenario_ids=parse_str_grid(args.certificate_diagnostic_scenarios),
        x_col="rho",
        x_label=RHO_LABEL,
        metric_cols=["proxy_target_gap", "target_robust_gap"],
        method_order=[
            "PROWL (tuned eta)",
            PROWL_U0_METHOD,
            "OWL(underline R)",
            "Q-learning(underline R)",
            "RWL(underline R)",
            "Policy Tree(underline R)",
        ],
        legend_ncol=3,
        output_path=output_dir / "plots" / "certificate_value_gaps.png",
    )
    plot_generic_method_grid(
        dataset_summary.assign(method="dataset"),
        scenario_ids=parse_str_grid(args.certificate_diagnostic_scenarios),
        x_col="rho",
        x_label=RHO_LABEL,
        metric_cols=["train_mean_certificate", "train_clipping_rate", "certificate_validity_rate_test"],
        method_order=["dataset"],
        output_path=output_dir / "plots" / "certificate_dataset_diagnostics.png",
    )
    write_manifest(
        output_dir / "manifest.json",
        {
            "experiment": "certificate_diagnostics",
            "scenario_ids": parse_str_grid(args.certificate_diagnostic_scenarios),
            "n_total": int(args.certificate_diagnostic_n_total),
            "rho_grid": parse_float_grid(args.certificate_diagnostic_rho_grid),
            "n_replications": int(args.certificate_diagnostic_replications),
            "rows": int(raw_df.shape[0]),
            "plots_only": bool(args.plots_only),
        },
    )


def run_split_free_ablation(args: argparse.Namespace, output_root: Path) -> None:
    output_dir = output_root / "split_free_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_metrics.csv"
    summary_path = output_dir / "summary.csv"
    gap_path = output_dir / "pairwise_gap_summary.csv"
    if bool(args.plots_only):
        raw_df = require_existing_csv(raw_path)
        summary_df = (
            pd.read_csv(summary_path)
            if summary_path.exists()
            else summarize_metrics(
                raw_df,
                group_cols=["scenario_id", "scenario_name", "scenario_group", "n_total", "method", "sample_split"],
                metrics=["target_regret", "robust_regret", "target_value", "robust_value"],
            )
        )
        if not summary_path.exists():
            summary_df.to_csv(summary_path, index=False)
        gap_df = (
            pd.read_csv(gap_path)
            if gap_path.exists()
            else generic_pairwise_gap_summary(
                raw_df,
                group_cols=["scenario_id", "scenario_name", "scenario_group", "n_total"],
                method_pairs=[("PROWL (sample-split)", "PROWL (split-free)", "Sample-split - Split-free")],
                metrics=["target_regret", "robust_regret"],
            )
        )
        if not gap_path.exists():
            gap_df.to_csv(gap_path, index=False)
    else:
        tasks = split_free_tasks(args)
        raw_df = execute_tasks(
            tasks=tasks,
            worker=_run_split_free_task,
            n_jobs=int(args.n_jobs),
            progress_mode=str(args.progress),
            progress_label="split_free_ablation",
        )
        summary_df = summarize_metrics(
            raw_df,
            group_cols=["scenario_id", "scenario_name", "scenario_group", "n_total", "method", "sample_split"],
            metrics=["target_regret", "robust_regret", "target_value", "robust_value"],
        )
        gap_df = generic_pairwise_gap_summary(
            raw_df,
            group_cols=["scenario_id", "scenario_name", "scenario_group", "n_total"],
            method_pairs=[("PROWL (sample-split)", "PROWL (split-free)", "Sample-split - Split-free")],
            metrics=["target_regret", "robust_regret"],
        )
        raw_df.to_csv(raw_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        gap_df.to_csv(gap_path, index=False)
    plot_generic_method_grid(
        summary_df,
        scenario_ids=parse_str_grid(args.split_free_scenarios),
        x_col="n_total",
        x_label=N_TOTAL_LABEL,
        metric_cols=["target_regret", "robust_regret"],
        method_order=["PROWL (split-free)", "PROWL (sample-split)"],
        output_path=output_dir / "plots" / "split_free_ablation.png",
    )
    write_manifest(
        output_dir / "manifest.json",
        {
            "experiment": "split_free_ablation",
            "scenario_ids": parse_str_grid(args.split_free_scenarios),
            "n_grid": parse_int_grid(args.split_free_grid),
            "n_replications": int(args.split_free_replications),
            "rho": float(args.split_free_rho),
            "rows": int(raw_df.shape[0]),
            "plots_only": bool(args.plots_only),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=(
            "rho_sweep",
            "n_sweep",
            "certificate_diagnostics",
            "split_free_ablation",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs_tmp" / "prowl_targeted_experiments"))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--progress", choices=("auto", "tqdm", "plain", "quiet"), default="auto")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--certificate-mode", default="oracle")
    parser.add_argument("--certificate-scale", type=float, default=1.0)
    parser.add_argument("--n-certificate-auxiliary", type=int, default=400)
    parser.add_argument("--propensity-mode", default="auto")
    parser.add_argument("--propensity-strength", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--prior-sd", type=float, default=5.0)
    parser.add_argument("--score-bound", type=float, default=3.0)
    parser.add_argument("--penalty-grid", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--q-penalty-grid", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--prowl-deployment", choices=("auto", "map", "mean_rule", "gibbs"), default="auto")
    parser.add_argument("--policy-tree-depth", type=int, default=2)
    parser.add_argument("--policy-tree-min-node-size", type=int, default=20)
    parser.add_argument("--policy-tree-split-step", type=int, default=25)
    parser.add_argument("--policy-tree-max-features", type=int, default=0)
    parser.add_argument("--n-anchor-particles", type=int, default=2)
    parser.add_argument("--n-prior-particles", type=int, default=32)
    parser.add_argument("--n-local-particles", type=int, default=4)
    parser.add_argument("--local-particle-scale", type=float, default=0.3)
    parser.add_argument("--nuisance-l2-penalty", type=float, default=1e-6)
    parser.add_argument("--eta-grid", default=DEFAULT_ETA_TUNING_GRID)
    parser.add_argument("--gamma-grid", default=DEFAULT_GAMMA_GRID)

    parser.add_argument("--rho-scenario", default=None)
    parser.add_argument("--rho-scenarios", default=",".join(DEFAULT_RHO_SCENARIOS))
    parser.add_argument("--rho-n-total", type=int, default=200)
    parser.add_argument("--rho-replications", type=int, default=DEFAULT_RHO_REPLICATIONS)
    parser.add_argument("--rho-grid", default=DEFAULT_RHO_GRID)

    parser.add_argument("--n-sweep-scenarios", default=",".join(DEFAULT_N_SWEEP_SCENARIOS))
    parser.add_argument("--n-sweep-grid", default=DEFAULT_N_SWEEP_GRID)
    parser.add_argument("--n-sweep-replications", type=int, default=DEFAULT_N_REPLICATIONS)
    parser.add_argument("--n-sweep-rho", type=float, default=1.5)

    parser.add_argument(
        "--certificate-diagnostic-scenarios", default=",".join(DEFAULT_CERTIFICATE_DIAGNOSTIC_SCENARIOS)
    )
    parser.add_argument("--certificate-diagnostic-n-total", type=int, default=1000)
    parser.add_argument("--certificate-diagnostic-rho-grid", default=DEFAULT_RHO_GRID)
    parser.add_argument(
        "--certificate-diagnostic-replications", type=int, default=DEFAULT_CERTIFICATE_DIAGNOSTIC_REPLICATIONS
    )

    parser.add_argument("--split-free-scenarios", default=",".join(DEFAULT_SPLIT_FREE_SCENARIOS))
    parser.add_argument("--split-free-grid", default=DEFAULT_N_SWEEP_GRID)
    parser.add_argument("--split-free-replications", type=int, default=DEFAULT_SPLIT_FREE_REPLICATIONS)
    parser.add_argument("--split-free-rho", type=float, default=1.5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    start = time.perf_counter()

    if args.experiment in {"rho_sweep", "all"}:
        run_rho_sweep(args, output_root)
    if args.experiment in {"n_sweep", "all"}:
        run_n_sweep(args, output_root)
    if args.experiment in {"certificate_diagnostics", "all"}:
        run_certificate_diagnostics(args, output_root)
    if args.experiment in {"split_free_ablation", "all"}:
        run_split_free_ablation(args, output_root)

    duration = time.perf_counter() - start
    write_manifest(
        output_root / "manifest.json",
        {
            "experiment": args.experiment,
            "completed_at_seconds": duration,
            "settings": vars(args),
        },
    )
    print(f"Completed {args.experiment} in {duration:0.1f}s", flush=True)
    print(f"Outputs written under {output_root}", flush=True)


if __name__ == "__main__":
    main()
