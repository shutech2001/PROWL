from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / ".mpl-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from tqdm.auto import tqdm

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pac_owl.estimators import (  # noqa: E402
    PROWL,
    StandardizedFeatureMap,
    fit_arm_value_nuisance_model,
    fit_linear_q_learning,
    fit_linear_residual_weighted_learning,
    fit_weighted_hinge_policy,
)
from pac_owl.estimators.prowl import ParticleLibraryConfig, feature_scores  # noqa: E402


EPS = 1e-12
PROPENSITY = 0.5
FONT_SIZE_BOOST = 8
MARKER_SIZE_BOOST = 3.0
LINE_WIDTH_BOOST = 0.8
HOSPITAL_LEVELS = (1, 2, 3, 4, 5, 6)
TEACHING_HOSPITALS = frozenset({1, 2, 3, 4})
NON_TEACHING_HOSPITALS = frozenset({5, 6})
METHOD_ORDER = (
    "Never alert",
    "Always alert",
    "OWL(R)",
    "Q-learning(R)",
    "RWL(R)",
    "OWL(underline R)",
    "Q-learning(underline R)",
    "RWL(underline R)",
    "PROWL (U=0)",
    "PROWL",
)
RAW_BASELINE_METHODS = ("OWL(R)", "Q-learning(R)", "RWL(R)")
LOWER_PLUGIN_METHODS = (
    "OWL(underline R)",
    "Q-learning(underline R)",
    "RWL(underline R)",
)
PROWL_METHODS = ("PROWL (U=0)", "PROWL")
DEFAULT_MAIN_WEIGHTS = (0.60, 0.25, 0.15)
DEFAULT_MAIN_DELTAS = (0.10, 0.05, 0.05)
DEFAULT_PATIENT4_WEIGHTS = (0.55, 0.20, 0.15, 0.10)
# The 4-component deltas are not specified in the design text; this default
# mirrors the main utility and keeps the fourth preference axis modest.
DEFAULT_PATIENT4_DELTAS = (0.10, 0.05, 0.05, 0.05)
_POLICY_SHARED_STATE: dict[str, Any] | None = None
_DEPLOYMENT_SHARED_STATE: dict[str, Any] | None = None

MAIN_COVARIATE_COLUMNS = (
    "hospital",
    "admit_medical",
    "emergency_room",
    "icu",
    "ward",
    "age",
    "age_over_90",
    "sex",
    "race",
    "ethnicity",
    "aki_duration",
    "aki_to_rand",
    "time_to_rand",
    "baseline_creat",
    "mincreat48",
    "creat_at_rand",
    "initial_egfr",
    "anion_gap_at_rand",
    "bicarbonate_at_rand",
    "bun_at_rand",
    "chloride_at_rand",
    "hemoglobin_at_rand",
    "plateletcount_at_rand",
    "potassium_at_rand",
    "pulse_at_rand",
    "resp_at_rand",
    "sodium_at_rand",
    "systolic_at_rand",
    "diastolic_at_rand",
    "sofa",
    "bnp_flag",
    "chf",
    "ckd_icd",
    "copd",
    "diabetes",
    "hypertension",
    "liver_disease",
    "malignancy",
    "elx_score",
    "acearbreninpre24",
    "aminopre24",
    "nsaidpre24",
    "prior_acearbrenin72",
    "prior_contraststudy72",
    "prior_ctsurgery7",
    "prior_nsaid72",
    "prior_ppi72",
    "other_alert_burden",
)
CONTINUOUS_COVARIATE_COLUMNS = (
    "age",
    "aki_duration",
    "aki_to_rand",
    "time_to_rand",
    "baseline_creat",
    "mincreat48",
    "creat_at_rand",
    "initial_egfr",
    "anion_gap_at_rand",
    "bicarbonate_at_rand",
    "bun_at_rand",
    "chloride_at_rand",
    "hemoglobin_at_rand",
    "plateletcount_at_rand",
    "potassium_at_rand",
    "pulse_at_rand",
    "resp_at_rand",
    "sodium_at_rand",
    "systolic_at_rand",
    "diastolic_at_rand",
    "sofa",
    "elx_score",
    "other_alert_burden",
)
SELECTED_INTERACTION_CONTINUOUS = (
    "sofa",
    "baseline_creat",
    "initial_egfr",
    "age",
    "aki_duration",
    "other_alert_burden",
)
TABLE1_COLUMNS = (
    "certified_value",
    "nominal_value",
    "composite_free_value",
    "mortality_risk",
    "alert_rate",
)
TABLE_S2_COLUMNS = (
    "mortality_risk",
    "dialysis_risk",
    "aki_progression_risk",
    "composite_outcome_risk",
    "discharge_to_home_rate",
)
ALL_EVAL_METRICS = (
    "certified_value",
    "nominal_value",
    "composite_free_value",
    "mortality_risk",
    "dialysis_risk",
    "aki_progression_risk",
    "composite_outcome_risk",
    "discharge_to_home_rate",
    "alert_rate",
)

METHOD_COLORS = {
    "Never alert": "#4d4d4d",
    "Always alert": "#bdbdbd",
    "OWL(R)": "#1f77b4",
    "Q-learning(R)": "#ff7f0e",
    "RWL(R)": "#2ca02c",
    "OWL(underline R)": "#6baed6",
    "Q-learning(underline R)": "#fdae6b",
    "RWL(underline R)": "#74c476",
    "PROWL (U=0)": "#7f7f7f",
    "PROWL": "#111111",
    "PROWL (mean-rule)": "#111111",
    "PROWL (MAP)": "#6b6b6b",
    "PROWL (Gibbs)": "#b0b0b0",
}
METHOD_MARKERS = {
    "Never alert": "o",
    "Always alert": "o",
    "OWL(R)": "s",
    "Q-learning(R)": "^",
    "RWL(R)": "D",
    "OWL(underline R)": "s",
    "Q-learning(underline R)": "^",
    "RWL(underline R)": "D",
    "PROWL (U=0)": "X",
    "PROWL": "o",
    "PROWL (mean-rule)": "o",
    "PROWL (MAP)": "s",
    "PROWL (Gibbs)": "^",
}


@dataclass(frozen=True)
class UtilitySpecification:
    name: str
    component_columns: tuple[str, ...]
    complement_columns: frozenset[str]
    weights: tuple[float, ...]
    deltas: tuple[float, ...]


@dataclass
class RewardBundle:
    utility_spec: str
    rho: float
    nominal_reward: np.ndarray
    lower_reward: np.ndarray
    certificate_gap: np.ndarray
    composite_free: np.ndarray
    death14: np.ndarray
    dialysis14: np.ndarray
    aki_progression14: np.ndarray
    composite_outcome: np.ndarray
    discharge_to_home: np.ndarray


@dataclass
class ConstantTreatmentPolicy:
    action: int

    def predict_action(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.full(x.shape[0], float(self.action), dtype=float)

    def action_probability(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.full(x.shape[0], 1.0 if self.action == 1 else 0.0, dtype=float)


@dataclass
class PolicyFit:
    method: str
    policy: Any
    runtime_sec: float
    fit_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NuisanceBasisBuilder:
    main_dim: int
    squared_indices: tuple[int, ...]
    interaction_left_indices: tuple[int, ...]
    interaction_right_indices: tuple[int, ...]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        parts = [x[:, : self.main_dim]]
        if self.squared_indices:
            parts.append(x[:, self.squared_indices] ** 2)
        for left in self.interaction_left_indices:
            left_col = x[:, left]
            for right in self.interaction_right_indices:
                parts.append((left_col * x[:, right])[:, None])
        return np.concatenate(parts, axis=1)


@dataclass
class FoldPreprocessor:
    include_hospital: bool
    hospital_levels: tuple[int, ...]
    continuous_columns: tuple[str, ...]
    binary_columns: tuple[str, ...]
    hospital_mode_: float
    categorical_modes_: dict[str, float]
    continuous_medians_: dict[str, float]
    continuous_means_: dict[str, float]
    continuous_scales_: dict[str, float]
    missing_indicator_columns_: tuple[str, ...]
    feature_names_: tuple[str, ...]

    @classmethod
    def fit(cls, train_df: pd.DataFrame, *, include_hospital: bool) -> "FoldPreprocessor":
        missing = [column for column in MAIN_COVARIATE_COLUMNS if column not in train_df.columns]
        if missing:
            raise ValueError(f"ELAIA-1 data is missing required columns: {missing}")
        binary_columns = tuple(
            column
            for column in MAIN_COVARIATE_COLUMNS
            if column not in CONTINUOUS_COVARIATE_COLUMNS and column != "hospital"
        )
        continuous_columns = tuple(CONTINUOUS_COVARIATE_COLUMNS)
        hospital_mode = float(train_df["hospital"].mode(dropna=True).iloc[0])
        categorical_modes = {}
        for column in binary_columns:
            mode = train_df[column].mode(dropna=True)
            categorical_modes[column] = float(mode.iloc[0]) if not mode.empty else 0.0

        continuous_medians = {}
        continuous_means = {}
        continuous_scales = {}
        missing_indicator_columns: list[str] = []
        for column in continuous_columns:
            observed = pd.to_numeric(train_df[column], errors="coerce")
            median = float(observed.median())
            filled = observed.fillna(median)
            mean = float(filled.mean())
            scale = float(filled.std(ddof=0))
            continuous_medians[column] = median
            continuous_means[column] = mean
            continuous_scales[column] = 1.0 if (not np.isfinite(scale) or scale < 1e-8) else scale
            if bool(observed.isna().any()):
                missing_indicator_columns.append(column)

        feature_names: list[str] = []
        if include_hospital:
            feature_names.extend([f"hospital_{level}" for level in HOSPITAL_LEVELS])
        feature_names.extend(binary_columns)
        for column in continuous_columns:
            feature_names.append(column)
            if column in missing_indicator_columns:
                feature_names.append(f"{column}__missing")

        return cls(
            include_hospital=bool(include_hospital),
            hospital_levels=tuple(HOSPITAL_LEVELS),
            continuous_columns=continuous_columns,
            binary_columns=binary_columns,
            hospital_mode_=hospital_mode,
            categorical_modes_=categorical_modes,
            continuous_medians_=continuous_medians,
            continuous_means_=continuous_means,
            continuous_scales_=continuous_scales,
            missing_indicator_columns_=tuple(missing_indicator_columns),
            feature_names_=tuple(feature_names),
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.feature_names_

    @property
    def feature_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.feature_names_)}

    @property
    def base_continuous_feature_names(self) -> tuple[str, ...]:
        return tuple(self.continuous_columns)

    @property
    def hospital_feature_names(self) -> tuple[str, ...]:
        if not self.include_hospital:
            return ()
        return tuple(f"hospital_{level}" for level in self.hospital_levels)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        pieces: list[np.ndarray] = []

        if self.include_hospital:
            hospital = (
                pd.to_numeric(frame["hospital"], errors="coerce").fillna(self.hospital_mode_).to_numpy(dtype=float)
            )
            for level in self.hospital_levels:
                pieces.append((hospital == float(level)).astype(float)[:, None])

        for column in self.binary_columns:
            filled = (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(self.categorical_modes_[column])
                .to_numpy(dtype=float)
            )
            pieces.append(filled[:, None])

        for column in self.continuous_columns:
            observed = pd.to_numeric(frame[column], errors="coerce")
            missing = observed.isna().to_numpy(dtype=float)
            filled = observed.fillna(self.continuous_medians_[column]).to_numpy(dtype=float)
            standardized = (filled - self.continuous_means_[column]) / self.continuous_scales_[column]
            pieces.append(standardized[:, None])
            if column in self.missing_indicator_columns_:
                pieces.append(missing[:, None])

        if not pieces:
            return np.empty((len(frame), 0), dtype=float)
        return np.concatenate(pieces, axis=1)

    def policy_feature_map_factory(self) -> Callable[[], StandardizedFeatureMap]:
        return lambda: StandardizedFeatureMap(add_intercept=True, standardize=False)

    def nuisance_feature_map_factory(self) -> Callable[[], StandardizedFeatureMap]:
        index = self.feature_index
        squared_indices = tuple(index[name] for name in self.base_continuous_feature_names if name in index)
        interaction_left_names = list(self.hospital_feature_names)
        for name in ("icu", "ward"):
            if name in index:
                interaction_left_names.append(name)
        interaction_right_names = [name for name in SELECTED_INTERACTION_CONTINUOUS if name in index]
        basis_builder = NuisanceBasisBuilder(
            main_dim=len(self.feature_names_),
            squared_indices=squared_indices,
            interaction_left_indices=tuple(index[name] for name in interaction_left_names if name in index),
            interaction_right_indices=tuple(index[name] for name in interaction_right_names if name in index),
        )
        return lambda: StandardizedFeatureMap(basis_fn=basis_builder, add_intercept=True, standardize=False)


def parse_float_grid(raw: str) -> list[float]:
    return [float(token.strip()) for token in str(raw).split(",") if token.strip()]


def latex_ready() -> bool:
    if shutil.which("latex") is None or shutil.which("kpsewhich") is None:
        return False
    for package in ("newtxtext.sty", "newtxmath.sty"):
        proc = subprocess.run(
            ["kpsewhich", package],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return False
    return True


def configure_matplotlib() -> None:
    use_tex = latex_ready()
    params: dict[str, Any] = {
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
    if use_tex:
        params.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}",
            }
        )
    else:
        params["text.usetex"] = False
    matplotlib.rcParams.update(params)


def add_top_center_legend(
    fig: plt.Figure,
    handles: Sequence[Any],
    labels: Sequence[str],
    *,
    ncol: int,
    y: float = 1.02,
) -> None:
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        borderaxespad=0.0,
        ncol=ncol,
        columnspacing=1.1,
        handletextpad=0.5,
    )


def utility_specifications(args: argparse.Namespace) -> dict[str, UtilitySpecification]:
    return {
        "hard_clinical_3": UtilitySpecification(
            name="hard_clinical_3",
            component_columns=("death14", "dialysis14", "aki_progression14"),
            complement_columns=frozenset({"death14", "dialysis14", "aki_progression14"}),
            weights=tuple(parse_float_grid(args.main_weights)),
            deltas=tuple(parse_float_grid(args.main_deltas)),
        ),
        "patient_centered_4": UtilitySpecification(
            name="patient_centered_4",
            component_columns=("death14", "dialysis14", "aki_progression14", "discharge_to_home"),
            complement_columns=frozenset({"death14", "dialysis14", "aki_progression14"}),
            weights=tuple(parse_float_grid(args.patient4_weights)),
            deltas=tuple(parse_float_grid(args.patient4_deltas)),
        ),
    }


def load_elaia_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "id",
        "alert",
        "hospital",
        "death14",
        "dialysis14",
        "aki_progression14",
        "composite_outcome",
        "discharge_to_home",
        *MAIN_COVARIATE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"ELAIA-1 file is missing required columns: {missing}")
    return frame.copy()


@lru_cache(maxsize=None)
def solve_lower_reward_for_pattern(
    pattern: tuple[int, ...],
    weights: tuple[float, ...],
    deltas: tuple[float, ...],
    rho: float,
) -> float:
    if len(pattern) != len(weights) or len(pattern) != len(deltas):
        raise ValueError("pattern, weights, and deltas must have the same length.")
    if float(rho) <= 0.0:
        return float(np.dot(np.asarray(pattern, dtype=float), np.asarray(weights, dtype=float)))
    if np.sum(pattern) <= 0:
        return 0.0
    c = np.asarray(pattern, dtype=float)
    lower_bounds = np.maximum(0.0, np.asarray(weights, dtype=float) - float(rho) * np.asarray(deltas, dtype=float))
    upper_bounds = np.minimum(1.0, np.asarray(weights, dtype=float) + float(rho) * np.asarray(deltas, dtype=float))
    bounds = [(float(lb), float(ub)) for lb, ub in zip(lower_bounds, upper_bounds, strict=True)]
    result = linprog(
        c=c,
        A_eq=np.ones((1, len(pattern)), dtype=float),
        b_eq=np.array([1.0], dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError("Failed to solve lower-reward LP for pattern " f"{pattern} with rho={rho}: {result.message}")
    return float(result.fun)


def reward_bundle_from_spec(frame: pd.DataFrame, spec: UtilitySpecification, *, rho: float) -> RewardBundle:
    components: list[np.ndarray] = []
    for column in spec.component_columns:
        raw = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if column in spec.complement_columns:
            components.append(1.0 - raw)
        else:
            components.append(raw)
    component_matrix = np.column_stack(components)
    weights = np.asarray(spec.weights, dtype=float)
    nominal = component_matrix @ weights
    if float(rho) <= 0.0:
        lower = nominal.copy()
    else:
        patterns, inverse = np.unique(component_matrix.astype(int), axis=0, return_inverse=True)
        lower_lookup = np.array(
            [
                solve_lower_reward_for_pattern(
                    tuple(int(value) for value in pattern.tolist()),
                    tuple(float(value) for value in spec.weights),
                    tuple(float(value) for value in spec.deltas),
                    float(rho),
                )
                for pattern in patterns
            ],
            dtype=float,
        )
        lower = lower_lookup[inverse]
    certificate_gap = nominal - lower
    return RewardBundle(
        utility_spec=spec.name,
        rho=float(rho),
        nominal_reward=np.clip(nominal, 0.0, 1.0),
        lower_reward=np.clip(lower, 0.0, 1.0),
        certificate_gap=np.clip(certificate_gap, 0.0, 1.0),
        composite_free=1.0
        - pd.to_numeric(frame["composite_outcome"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        death14=pd.to_numeric(frame["death14"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        dialysis14=pd.to_numeric(frame["dialysis14"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        aki_progression14=pd.to_numeric(frame["aki_progression14"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        composite_outcome=pd.to_numeric(frame["composite_outcome"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        discharge_to_home=pd.to_numeric(frame["discharge_to_home"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    )


def action_from_probability(prob_treat_one: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob_treat_one, dtype=float).reshape(-1)
    return np.where(prob >= 0.5, 1.0, -1.0)


def deterministic_action(policy: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(policy, "predict_action"):
        return np.asarray(policy.predict_action(x), dtype=float).reshape(-1)
    if hasattr(policy, "deterministic_action"):
        return np.asarray(policy.deterministic_action(x), dtype=float).reshape(-1)
    if hasattr(policy, "action_probability"):
        return action_from_probability(np.asarray(policy.action_probability(x), dtype=float))
    raise TypeError(f"Policy type {type(policy)!r} does not expose a deterministic prediction interface.")


def policy_treat_probability(policy: Any, x: np.ndarray, *, deployment: str) -> np.ndarray:
    mode = str(deployment).strip().lower()
    if mode in {"mean_rule", "deterministic"}:
        return (deterministic_action(policy, x) == 1.0).astype(float)
    if mode == "map":
        if all(hasattr(policy, attr) for attr in ("posterior_weights", "candidates", "feature_map", "score_bound")):
            idx = int(np.argmax(np.asarray(policy.posterior_weights, dtype=float)))
            phi = policy.feature_map.transform(x)
            score = feature_scores(
                phi=phi,
                candidates=np.asarray(policy.candidates, dtype=float)[[idx]],
                score_bound=float(policy.score_bound),
            ).reshape(-1)
            return (score >= 0.0).astype(float)
        return (deterministic_action(policy, x) == 1.0).astype(float)
    if mode == "gibbs":
        if hasattr(policy, "action_probability"):
            return np.clip(np.asarray(policy.action_probability(x), dtype=float).reshape(-1), 0.0, 1.0)
        return (deterministic_action(policy, x) == 1.0).astype(float)
    raise ValueError(f"Unknown deployment mode: {deployment}")


def fit_outcome_nuisance(
    *,
    x: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    feature_map_factory: Callable[[], StandardizedFeatureMap],
    l2_penalty: float,
) -> Any:
    return fit_arm_value_nuisance_model(
        x=x,
        treatment=treatment,
        lower_reward=outcome,
        feature_map_factory=feature_map_factory,
        l2_penalty=float(l2_penalty),
    )


def aipw_value(
    *,
    prob_treat_one: np.ndarray,
    observed_treatment: np.ndarray,
    outcome: np.ndarray,
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    propensity: float = PROPENSITY,
) -> float:
    p = np.clip(np.asarray(prob_treat_one, dtype=float).reshape(-1), 0.0, 1.0)
    a = np.asarray(observed_treatment, dtype=float).reshape(-1)
    y = np.asarray(outcome, dtype=float).reshape(-1)
    mu_pos = np.clip(np.asarray(mu_pos, dtype=float).reshape(-1), 0.0, 1.0)
    mu_neg = np.clip(np.asarray(mu_neg, dtype=float).reshape(-1), 0.0, 1.0)
    mu_policy = p * mu_pos + (1.0 - p) * mu_neg
    correction = ((a == 1.0).astype(float) * p * (y - mu_pos) / max(float(propensity), EPS)) + (
        (a == -1.0).astype(float) * (1.0 - p) * (y - mu_neg) / max(1.0 - float(propensity), EPS)
    )
    return float(np.mean(mu_policy + correction))


def ridge_coefficients(design: np.ndarray, outcome: np.ndarray, l2_penalty: float) -> np.ndarray:
    gram = design.T @ design + float(l2_penalty) * np.eye(design.shape[1], dtype=float)
    rhs = design.T @ outcome
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


@dataclass
class FeatureMapRidgeModel:
    feature_map: StandardizedFeatureMap
    coefficients: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.feature_map.transform(x) @ self.coefficients, 0.0, 1.0)


def fit_control_risk_model(
    *,
    x: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    feature_map_factory: Callable[[], StandardizedFeatureMap],
    l2_penalty: float,
) -> FeatureMapRidgeModel:
    control = np.asarray(treatment, dtype=float) == -1.0
    if int(np.sum(control)) == 0:
        raise ValueError("Control-only risk model requires at least one control observation.")
    fmap = feature_map_factory()
    design = fmap.fit_transform(np.asarray(x[control], dtype=float))
    coefficients = ridge_coefficients(design, np.asarray(outcome[control], dtype=float), l2_penalty=float(l2_penalty))
    return FeatureMapRidgeModel(feature_map=fmap, coefficients=coefficients)


def fit_owl_policy(
    *,
    x: np.ndarray,
    treatment: np.ndarray,
    reward: np.ndarray,
    penalty: float,
    feature_map_factory: Callable[[], StandardizedFeatureMap],
) -> Any:
    return fit_weighted_hinge_policy(
        x=x,
        fit_labels=treatment,
        fit_weights=np.asarray(reward, dtype=float) / PROPENSITY,
        feature_map=feature_map_factory(),
        l2_penalty=float(penalty),
        score_bound=None,
    )


def fit_q_policy(
    *,
    x: np.ndarray,
    treatment: np.ndarray,
    reward: np.ndarray,
    penalty: float,
    feature_map_factory: Callable[[], StandardizedFeatureMap],
) -> Any:
    return fit_linear_q_learning(
        x=x,
        a=treatment,
        y=reward,
        main_feature_map=feature_map_factory(),
        blip_feature_map=feature_map_factory(),
        l2_penalty=float(penalty),
    )


def fit_rwl_policy(
    *,
    x: np.ndarray,
    treatment: np.ndarray,
    reward: np.ndarray,
    penalty: float,
    feature_map_factory: Callable[[], StandardizedFeatureMap],
    nuisance_feature_map_factory: Callable[[], StandardizedFeatureMap],
    nuisance_l2_penalty: float,
) -> Any:
    return fit_linear_residual_weighted_learning(
        x=x,
        a=treatment,
        y=reward,
        propensity=np.full(x.shape[0], PROPENSITY, dtype=float),
        feature_map=feature_map_factory(),
        residual_feature_map=nuisance_feature_map_factory(),
        l2_penalty=float(penalty),
        residual_regression_penalty=float(nuisance_l2_penalty),
    )


def tune_penalty_via_aipw(
    *,
    family: str,
    x: np.ndarray,
    treatment: np.ndarray,
    reward: np.ndarray,
    penalty_grid: Sequence[float],
    feature_map_factory: Callable[[], StandardizedFeatureMap],
    nuisance_feature_map_factory: Callable[[], StandardizedFeatureMap],
    eval_nuisance_l2_penalty: float,
    nuisance_l2_penalty: float,
    random_state: int,
    cv_folds: int = 3,
) -> tuple[Any, float, dict[float, float]]:
    x = np.asarray(x, dtype=float)
    treatment = np.asarray(treatment, dtype=float).reshape(-1)
    reward = np.asarray(reward, dtype=float).reshape(-1)
    penalties = [float(value) for value in penalty_grid]
    if not penalties:
        raise ValueError("penalty_grid must contain at least one value.")

    def fit_candidate(x_fit: np.ndarray, a_fit: np.ndarray, y_fit: np.ndarray, penalty: float) -> Any:
        if family == "owl":
            return fit_owl_policy(
                x=x_fit,
                treatment=a_fit,
                reward=y_fit,
                penalty=penalty,
                feature_map_factory=feature_map_factory,
            )
        if family == "q":
            return fit_q_policy(
                x=x_fit,
                treatment=a_fit,
                reward=y_fit,
                penalty=penalty,
                feature_map_factory=feature_map_factory,
            )
        if family == "rwl":
            return fit_rwl_policy(
                x=x_fit,
                treatment=a_fit,
                reward=y_fit,
                penalty=penalty,
                feature_map_factory=feature_map_factory,
                nuisance_feature_map_factory=nuisance_feature_map_factory,
                nuisance_l2_penalty=nuisance_l2_penalty,
            )
        raise ValueError(f"Unknown family: {family}")

    if x.shape[0] < 2 or len(penalties) == 1:
        penalty = penalties[0]
        return fit_candidate(x, treatment, reward, penalty), penalty, {penalty: math.nan}

    splitter = KFold(
        n_splits=min(max(2, int(cv_folds)), x.shape[0]),
        shuffle=True,
        random_state=int(random_state),
    )
    scores: dict[float, float] = {}
    for penalty in penalties:
        fold_values: list[float] = []
        for train_idx, val_idx in splitter.split(x):
            try:
                policy = fit_candidate(x[train_idx], treatment[train_idx], reward[train_idx], penalty)
                nuisance = fit_outcome_nuisance(
                    x=x[train_idx],
                    treatment=treatment[train_idx],
                    outcome=reward[train_idx],
                    feature_map_factory=nuisance_feature_map_factory,
                    l2_penalty=eval_nuisance_l2_penalty,
                )
                mu_pos, mu_neg = nuisance.predict(x[val_idx])
                prob = policy_treat_probability(policy, x[val_idx], deployment="deterministic")
                fold_values.append(
                    aipw_value(
                        prob_treat_one=prob,
                        observed_treatment=treatment[val_idx],
                        outcome=reward[val_idx],
                        mu_pos=mu_pos,
                        mu_neg=mu_neg,
                        propensity=PROPENSITY,
                    )
                )
            except RuntimeError:
                fold_values.append(-float("inf"))
        scores[penalty] = float(np.mean(fold_values))

    best_penalty = max(scores, key=scores.get)
    policy = fit_candidate(x, treatment, reward, best_penalty)
    return policy, best_penalty, scores


def fit_requested_method(
    *,
    method: str,
    x_train: np.ndarray,
    treatment_train: np.ndarray,
    nominal_reward_train: np.ndarray,
    lower_reward_train: np.ndarray,
    feature_map_factory: Callable[[], StandardizedFeatureMap],
    nuisance_feature_map_factory: Callable[[], StandardizedFeatureMap],
    penalty_grid: Sequence[float],
    policy_tree_depth: int,
    policy_tree_min_node_size: int,
    policy_tree_split_step: int,
    policy_tree_max_features: int | None,
    eval_nuisance_l2_penalty: float,
    nuisance_l2_penalty: float,
    eta_grid: Sequence[float],
    gamma_grid: Sequence[float],
    delta: float,
    prior_sd: float,
    score_bound: float,
    particle_library_config: ParticleLibraryConfig,
    random_state: int,
) -> PolicyFit:
    start = time.perf_counter()
    fit_info: dict[str, Any] = {}
    if method == "Never alert":
        return PolicyFit(
            method=method, policy=ConstantTreatmentPolicy(action=-1), runtime_sec=time.perf_counter() - start
        )
    if method == "Always alert":
        return PolicyFit(
            method=method, policy=ConstantTreatmentPolicy(action=1), runtime_sec=time.perf_counter() - start
        )

    if method == "OWL(R)":
        policy, best_penalty, cv_scores = tune_penalty_via_aipw(
            family="owl",
            x=x_train,
            treatment=treatment_train,
            reward=nominal_reward_train,
            penalty_grid=penalty_grid,
            feature_map_factory=feature_map_factory,
            nuisance_feature_map_factory=nuisance_feature_map_factory,
            eval_nuisance_l2_penalty=eval_nuisance_l2_penalty,
            nuisance_l2_penalty=nuisance_l2_penalty,
            random_state=random_state + 3,
        )
        fit_info = {"selected_penalty": best_penalty, "cv_mean_aipw": cv_scores[best_penalty]}
        return PolicyFit(method=method, policy=policy, runtime_sec=time.perf_counter() - start, fit_info=fit_info)
    if method == "OWL(underline R)":
        policy, best_penalty, cv_scores = tune_penalty_via_aipw(
            family="owl",
            x=x_train,
            treatment=treatment_train,
            reward=lower_reward_train,
            penalty_grid=penalty_grid,
            feature_map_factory=feature_map_factory,
            nuisance_feature_map_factory=nuisance_feature_map_factory,
            eval_nuisance_l2_penalty=eval_nuisance_l2_penalty,
            nuisance_l2_penalty=nuisance_l2_penalty,
            random_state=random_state + 5,
        )
        fit_info = {"selected_penalty": best_penalty, "cv_mean_aipw": cv_scores[best_penalty]}
        return PolicyFit(method=method, policy=policy, runtime_sec=time.perf_counter() - start, fit_info=fit_info)
    if method == "Q-learning(R)":
        policy, best_penalty, cv_scores = tune_penalty_via_aipw(
            family="q",
            x=x_train,
            treatment=treatment_train,
            reward=nominal_reward_train,
            penalty_grid=penalty_grid,
            feature_map_factory=feature_map_factory,
            nuisance_feature_map_factory=nuisance_feature_map_factory,
            eval_nuisance_l2_penalty=eval_nuisance_l2_penalty,
            nuisance_l2_penalty=nuisance_l2_penalty,
            random_state=random_state + 7,
        )
        fit_info = {"selected_penalty": best_penalty, "cv_mean_aipw": cv_scores[best_penalty]}
        return PolicyFit(method=method, policy=policy, runtime_sec=time.perf_counter() - start, fit_info=fit_info)
    if method == "Q-learning(underline R)":
        policy, best_penalty, cv_scores = tune_penalty_via_aipw(
            family="q",
            x=x_train,
            treatment=treatment_train,
            reward=lower_reward_train,
            penalty_grid=penalty_grid,
            feature_map_factory=feature_map_factory,
            nuisance_feature_map_factory=nuisance_feature_map_factory,
            eval_nuisance_l2_penalty=eval_nuisance_l2_penalty,
            nuisance_l2_penalty=nuisance_l2_penalty,
            random_state=random_state + 11,
        )
        fit_info = {"selected_penalty": best_penalty, "cv_mean_aipw": cv_scores[best_penalty]}
        return PolicyFit(method=method, policy=policy, runtime_sec=time.perf_counter() - start, fit_info=fit_info)
    if method == "RWL(R)":
        policy, best_penalty, cv_scores = tune_penalty_via_aipw(
            family="rwl",
            x=x_train,
            treatment=treatment_train,
            reward=nominal_reward_train,
            penalty_grid=penalty_grid,
            feature_map_factory=feature_map_factory,
            nuisance_feature_map_factory=nuisance_feature_map_factory,
            eval_nuisance_l2_penalty=eval_nuisance_l2_penalty,
            nuisance_l2_penalty=nuisance_l2_penalty,
            random_state=random_state + 13,
        )
        fit_info = {"selected_penalty": best_penalty, "cv_mean_aipw": cv_scores[best_penalty]}
        return PolicyFit(method=method, policy=policy, runtime_sec=time.perf_counter() - start, fit_info=fit_info)
    if method == "RWL(underline R)":
        policy, best_penalty, cv_scores = tune_penalty_via_aipw(
            family="rwl",
            x=x_train,
            treatment=treatment_train,
            reward=lower_reward_train,
            penalty_grid=penalty_grid,
            feature_map_factory=feature_map_factory,
            nuisance_feature_map_factory=nuisance_feature_map_factory,
            eval_nuisance_l2_penalty=eval_nuisance_l2_penalty,
            nuisance_l2_penalty=nuisance_l2_penalty,
            random_state=random_state + 17,
        )
        fit_info = {"selected_penalty": best_penalty, "cv_mean_aipw": cv_scores[best_penalty]}
        return PolicyFit(method=method, policy=policy, runtime_sec=time.perf_counter() - start, fit_info=fit_info)
    if method == "PROWL (U=0)":
        nuisance_model = fit_outcome_nuisance(
            x=x_train,
            treatment=treatment_train,
            outcome=nominal_reward_train,
            feature_map_factory=nuisance_feature_map_factory,
            l2_penalty=nuisance_l2_penalty,
        )
        policy = PROWL(
            eta_grid=eta_grid,
            gamma_grid=gamma_grid,
            selection_mode="lcb",
            delta=delta,
            prior_sd=prior_sd,
            score_bound=score_bound,
            feature_map=feature_map_factory(),
            nuisance_feature_map_factory=nuisance_feature_map_factory,
            nuisance_l2_penalty=nuisance_l2_penalty,
            particle_library_config=particle_library_config,
            random_state=random_state + 19,
        ).fit(
            x=x_train,
            treatment=treatment_train,
            reward=nominal_reward_train,
            propensity=PROPENSITY,
            lower_reward=nominal_reward_train,
            nuisance_model=nuisance_model,
            method_label=method,
        )
        fit_info = {"eta": policy.eta, "gamma": policy.gamma}
        return PolicyFit(method=method, policy=policy, runtime_sec=time.perf_counter() - start, fit_info=fit_info)
    if method == "PROWL":
        nuisance_model = fit_outcome_nuisance(
            x=x_train,
            treatment=treatment_train,
            outcome=lower_reward_train,
            feature_map_factory=nuisance_feature_map_factory,
            l2_penalty=nuisance_l2_penalty,
        )
        policy = PROWL(
            eta_grid=eta_grid,
            gamma_grid=gamma_grid,
            selection_mode="lcb",
            delta=delta,
            prior_sd=prior_sd,
            score_bound=score_bound,
            feature_map=feature_map_factory(),
            nuisance_feature_map_factory=nuisance_feature_map_factory,
            nuisance_l2_penalty=nuisance_l2_penalty,
            particle_library_config=particle_library_config,
            random_state=random_state + 23,
        ).fit(
            x=x_train,
            treatment=treatment_train,
            reward=nominal_reward_train,
            propensity=PROPENSITY,
            lower_reward=lower_reward_train,
            nuisance_model=nuisance_model,
            method_label=method,
        )
        fit_info = {"eta": policy.eta, "gamma": policy.gamma}
        return PolicyFit(method=method, policy=policy, runtime_sec=time.perf_counter() - start, fit_info=fit_info)
    raise ValueError(f"Unknown method: {method}")


def evaluate_policy_bundle(
    *,
    policy_fit: PolicyFit,
    x_test: np.ndarray,
    treatment_test: np.ndarray,
    eval_outcomes: Mapping[str, np.ndarray],
    eval_nuisance_models: Mapping[str, Any],
    deployment: str,
) -> dict[str, float]:
    prob = policy_treat_probability(policy_fit.policy, x_test, deployment=deployment)
    metrics = {"alert_rate": float(np.mean(prob))}
    for metric, outcome in eval_outcomes.items():
        nuisance = eval_nuisance_models[metric]
        mu_pos, mu_neg = nuisance.predict(x_test)
        metrics[metric] = aipw_value(
            prob_treat_one=prob,
            observed_treatment=treatment_test,
            outcome=outcome,
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            propensity=PROPENSITY,
        )
    return metrics


def quintile_labels(n_levels: int = 5) -> list[str]:
    return [f"Q{idx}" for idx in range(1, n_levels + 1)]


def risk_quintiles(values: np.ndarray, n_levels: int = 5) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=float)).rank(method="first")
    bins = pd.qcut(ranks, q=n_levels, labels=quintile_labels(n_levels))
    return bins.astype(str).to_numpy()


def select_best_method(summary_df: pd.DataFrame, candidates: Sequence[str]) -> str:
    frame = summary_df.loc[summary_df["method"].isin(candidates)].copy()
    if frame.empty:
        raise ValueError(f"No candidate methods found in summary for selection: {list(candidates)}")
    frame = frame.sort_values(["certified_value_mean", "method"], ascending=[False, True]).reset_index(drop=True)
    return str(frame.loc[0, "method"])


def summarize_split_metrics(
    raw_df: pd.DataFrame, *, group_cols: Sequence[str], metric_cols: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, frame in raw_df.groupby(list(group_cols), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(group_cols, group_key, strict=True)}
        row["n_splits"] = int(frame["split"].nunique()) if "split" in frame.columns else int(len(frame))
        for metric in metric_cols:
            values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
            mean = float(np.nanmean(values))
            if values.size <= 1:
                se = 0.0
            else:
                se = float(np.nanstd(values, ddof=1) / math.sqrt(np.sum(np.isfinite(values))))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_se"] = se
            row[f"{metric}_ci_low"] = mean - 1.96 * se
            row[f"{metric}_ci_high"] = mean + 1.96 * se
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference_summary(
    raw_df: pd.DataFrame,
    *,
    reference_method: str,
    metric_cols: Sequence[str],
    group_cols: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    index_cols = list(group_cols) + ["split"]
    for group_key, frame in raw_df.groupby(list(group_cols), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        ref = frame.loc[frame["method"] == reference_method]
        if ref.empty:
            continue
        ref = ref.set_index("split")
        for method, method_frame in frame.groupby("method", dropna=False):
            if method == reference_method:
                continue
            aligned = method_frame.set_index("split").join(
                ref[[*metric_cols]],
                how="inner",
                rsuffix="__ref",
            )
            if aligned.empty:
                continue
            row = {column: value for column, value in zip(group_cols, group_key, strict=True)}
            row["method"] = method
            row["reference_method"] = reference_method
            row["n_splits"] = int(aligned.shape[0])
            for metric in metric_cols:
                diff = pd.to_numeric(aligned[metric], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(
                    aligned[f"{metric}__ref"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                mean = float(np.nanmean(diff))
                se = 0.0 if diff.size <= 1 else float(np.nanstd(diff, ddof=1) / math.sqrt(np.sum(np.isfinite(diff))))
                row[f"{metric}_diff_mean"] = mean
                row[f"{metric}_diff_se"] = se
                row[f"{metric}_diff_ci_low"] = mean - 1.96 * se
                row[f"{metric}_diff_ci_high"] = mean + 1.96 * se
            rows.append(row)
    return pd.DataFrame(rows)


def format_mean_se(mean: float, se: float) -> str:
    return f"{mean:.3f} ({se:.3f})"


def method_display_label(method: str) -> str:
    mapping = {
        "Never alert": "Never alert",
        "Always alert": "Always alert",
        "OWL(R)": r"OWL ($R$)",
        "Q-learning(R)": r"Q-learning ($R$)",
        "RWL(R)": r"RWL ($R$)",
        "OWL(underline R)": r"OWL ($\underline{R}$)",
        "Q-learning(underline R)": r"Q-learning ($\underline{R}$)",
        "RWL(underline R)": r"RWL ($\underline{R}$)",
        "PROWL (U=0)": r"PROWL ($U=0$)",
        "PROWL": "PROWL",
        "PROWL (mean-rule)": "PROWL (mean-rule)",
        "PROWL (MAP)": "PROWL (MAP)",
        "PROWL (Gibbs)": "PROWL (Gibbs)",
    }
    return mapping.get(method, method)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_table1_outputs(summary_df: pd.DataFrame, paired_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = summary_df.copy()
    never_delta = paired_df.loc[
        paired_df["reference_method"] == "Never alert",
        ["method", "certified_value_diff_mean", "certified_value_diff_se"],
    ].rename(
        columns={
            "certified_value_diff_mean": "delta_vs_never_certified_mean",
            "certified_value_diff_se": "delta_vs_never_certified_se",
        }
    )
    frame = frame.merge(never_delta, on="method", how="left")
    frame = frame.set_index("method").loc[list(METHOD_ORDER)].reset_index()
    frame["is_best_certified"] = frame["certified_value_mean"] == float(frame["certified_value_mean"].max())
    numeric = frame[
        [
            "method",
            "certified_value_mean",
            "certified_value_se",
            "nominal_value_mean",
            "nominal_value_se",
            "composite_free_value_mean",
            "composite_free_value_se",
            "mortality_risk_mean",
            "mortality_risk_se",
            "alert_rate_mean",
            "alert_rate_se",
            "delta_vs_never_certified_mean",
            "delta_vs_never_certified_se",
            "is_best_certified",
        ]
    ].copy()
    numeric.to_csv(output_dir / "table1_cross_fitted_policy_comparison_numeric.csv", index=False)

    display = pd.DataFrame(
        {
            "Method": [method_display_label(method) for method in frame["method"]],
            "Certified value": [
                f"**{format_mean_se(mean, se)}**" if best else format_mean_se(mean, se)
                for mean, se, best in zip(
                    frame["certified_value_mean"],
                    frame["certified_value_se"],
                    frame["is_best_certified"],
                    strict=True,
                )
            ],
            "Nominal value": [
                format_mean_se(mean, se)
                for mean, se in zip(frame["nominal_value_mean"], frame["nominal_value_se"], strict=True)
            ],
            "Composite-free value": [
                format_mean_se(mean, se)
                for mean, se in zip(frame["composite_free_value_mean"], frame["composite_free_value_se"], strict=True)
            ],
            "Mortality risk": [
                format_mean_se(mean, se)
                for mean, se in zip(frame["mortality_risk_mean"], frame["mortality_risk_se"], strict=True)
            ],
            "Alert rate": [
                format_mean_se(mean, se)
                for mean, se in zip(frame["alert_rate_mean"], frame["alert_rate_se"], strict=True)
            ],
            "Delta vs Never": [
                "" if not np.isfinite(mean) else format_mean_se(mean, se)
                for mean, se in zip(
                    frame["delta_vs_never_certified_mean"],
                    frame["delta_vs_never_certified_se"],
                    strict=True,
                )
            ],
        }
    )
    write_markdown_table(display, output_dir / "table1_cross_fitted_policy_comparison.md")
    return frame


def build_table_s2_outputs(summary_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = summary_df.set_index("method").loc[list(METHOD_ORDER)].reset_index()
    numeric = frame[
        [
            "method",
            "mortality_risk_mean",
            "mortality_risk_se",
            "dialysis_risk_mean",
            "dialysis_risk_se",
            "aki_progression_risk_mean",
            "aki_progression_risk_se",
            "composite_outcome_risk_mean",
            "composite_outcome_risk_se",
            "discharge_to_home_rate_mean",
            "discharge_to_home_rate_se",
        ]
    ].copy()
    numeric.to_csv(output_dir / "table_s2_component_outcome_decomposition_numeric.csv", index=False)
    display = pd.DataFrame(
        {
            "Method": [method_display_label(method) for method in frame["method"]],
            "death14 risk": [
                format_mean_se(mean, se)
                for mean, se in zip(frame["mortality_risk_mean"], frame["mortality_risk_se"], strict=True)
            ],
            "dialysis14 risk": [
                format_mean_se(mean, se)
                for mean, se in zip(frame["dialysis_risk_mean"], frame["dialysis_risk_se"], strict=True)
            ],
            "aki_progression14 risk": [
                format_mean_se(mean, se)
                for mean, se in zip(frame["aki_progression_risk_mean"], frame["aki_progression_risk_se"], strict=True)
            ],
            "composite_outcome risk": [
                format_mean_se(mean, se)
                for mean, se in zip(
                    frame["composite_outcome_risk_mean"], frame["composite_outcome_risk_se"], strict=True
                )
            ],
            "discharge_to_home rate": [
                format_mean_se(mean, se)
                for mean, se in zip(
                    frame["discharge_to_home_rate_mean"], frame["discharge_to_home_rate_se"], strict=True
                )
            ],
        }
    )
    write_markdown_table(display, output_dir / "table_s2_component_outcome_decomposition.md")
    return frame


def classify_variable_type(column: str, frame: pd.DataFrame) -> str:
    if column == "id":
        return "identifier"
    if column == "alert":
        return "treatment"
    if column == "hospital":
        return "categorical"
    if column in CONTINUOUS_COVARIATE_COLUMNS:
        return "continuous"
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) <= 2 and set(np.round(values).astype(int)).issubset({0, 1}):
        return "binary"
    if len(values) <= 10 and np.all(np.abs(values - np.round(values)) < 1e-8):
        return "categorical"
    return "continuous"


def is_pre_randomization(column: str) -> bool:
    if column in MAIN_COVARIATE_COLUMNS:
        return True
    if column in {"id", "alert"}:
        return False
    if column.endswith("post24") or "_post_" in column or column.endswith("post28"):
        return False
    if column.startswith("time_to_") and column != "time_to_rand":
        return False
    if column in {
        "consult14",
        "aki_documentation",
        "death14",
        "dialysis14",
        "aki_progression14",
        "composite_outcome",
        "discharge_to_home",
        "los_since_alert",
        "direct_cost",
        "total_cost",
        "duration_of_alert",
        "duration",
        "unique_providers",
        "max_stage",
    }:
        return False
    if column in {"time_from_trial_to_rand", "aki_to_rand", "time_to_rand"}:
        return True
    if column.endswith("_at_rand") or column.startswith("prior_") or column.endswith("pre24"):
        return True
    return False


def variable_note(column: str) -> str:
    if column in MAIN_COVARIATE_COLUMNS:
        if column == "hospital":
            return "Main covariate; encoded as six one-hot indicators."
        if column in CONTINUOUS_COVARIATE_COLUMNS:
            return "Main covariate; median imputation, train-fold z-score, missing indicator if needed."
        return "Main covariate; mode imputation within the outer train fold."
    if column == "alert":
        return "Randomized treatment indicator; coded to A in {-1,+1}."
    if column == "id":
        return "Identifier; excluded from policy learning."
    if column == "max_stage":
        return "Excluded in the real-data policy model because of leakage concerns."
    if column.endswith("post24") or "_post_" in column or column.endswith("post28"):
        return "Post-randomization process variable; excluded."
    if column.startswith("time_to_") and column != "time_to_rand":
        return "Post-randomization event time; excluded."
    if column in {"death14", "dialysis14", "aki_progression14", "composite_outcome", "discharge_to_home"}:
        return "Outcome/reward component; used for evaluation, not as X."
    if column in {"consult14", "aki_documentation"}:
        return "Post-randomization care-process variable; excluded."
    if column in {"los_since_alert", "direct_cost", "total_cost", "duration_of_alert", "duration", "unique_providers"}:
        return "Encounter-level post-randomization/utilization summary; excluded."
    if column == "wbcc_at_rand":
        return "Pre-randomization lab not included in the prespecified main covariate set."
    if column == "time_from_trial_to_rand":
        return "Calendar-time trend excluded from the prespecified main covariate set."
    return "Excluded from the prespecified main policy covariate set."


def build_table_s1(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        rows.append(
            {
                "variable": column,
                "used_in_main_policy_model": "yes" if column in MAIN_COVARIATE_COLUMNS else "no",
                "pre_randomization": "yes" if is_pre_randomization(column) else "no",
                "variable_type": classify_variable_type(column, frame),
                "missing_pct": float(100.0 * frame[column].isna().mean()),
                "note": variable_note(column),
            }
        )
    result = pd.DataFrame(rows).sort_values(["used_in_main_policy_model", "variable"], ascending=[False, True])
    result.to_csv(output_dir / "table_s1_included_excluded_variables_missingness.csv", index=False)
    return result


def summary_frame_to_point_range_plot(
    summary_df: pd.DataFrame,
    *,
    metric_specs: Sequence[tuple[str, str]],
    methods: Sequence[str],
    output_path: Path,
    panel_width: float = 9.6,
    height_per_method: float = 0.52,
    show_reference_line_at_zero: bool = False,
) -> None:
    ordered = summary_df.set_index("method").loc[list(methods)].reset_index()
    fig_height = max(4.8, 1.8 + height_per_method * len(methods))
    fig, axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=(panel_width * len(metric_specs), fig_height),
        constrained_layout=True,
        sharey=True,
    )
    if len(metric_specs) == 1:
        axes = [axes]
    y_pos = np.arange(len(methods))[::-1]
    display_methods = [method_display_label(method) for method in methods]
    for ax, (metric, xlabel) in zip(axes, metric_specs, strict=True):
        ci_lows: list[float] = []
        ci_highs: list[float] = []
        for idx, method in enumerate(methods):
            row = ordered.loc[ordered["method"] == method].iloc[0]
            mean = float(row[f"{metric}_mean"])
            se = float(row[f"{metric}_se"])
            ci_low = mean - 1.96 * se
            ci_high = mean + 1.96 * se
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)
            ax.hlines(
                y=y_pos[idx],
                xmin=ci_low,
                xmax=ci_high,
                color=METHOD_COLORS.get(method, "#333333"),
                linewidth=2.8 + LINE_WIDTH_BOOST,
            )
            ax.plot(
                mean,
                y_pos[idx],
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, "#333333"),
                markersize=9.0 + MARKER_SIZE_BOOST,
                linestyle="none",
            )
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        if ci_lows and ci_highs:
            x_min = float(np.min(ci_lows))
            x_max = float(np.max(ci_highs))
            span = max(x_max - x_min, 1e-3)
            pad = max(0.01, 0.08 * span)
            ax.set_xlim(x_min - pad, x_max + pad)
        if show_reference_line_at_zero:
            ax.axvline(0.0, color="#cccccc", linewidth=1.1 + LINE_WIDTH_BOOST, linestyle="--")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(display_methods)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_main_figure1(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary_frame_to_point_range_plot(
        summary_df,
        metric_specs=(
            ("certified_value", "Estimated certified value"),
            ("composite_free_value", "Estimated composite-free value"),
        ),
        methods=METHOD_ORDER,
        output_path=output_dir / "figure1_point_range",
    )


def plot_figure2_allocation(allocation_df: pd.DataFrame, *, methods: Sequence[str], output_dir: Path) -> None:
    frame = allocation_df.loc[allocation_df["method"].isin(methods)].copy()
    frame = (
        frame.groupby(["method", "risk_quintile", "teaching_status"], dropna=False)
        .agg(alert_rate=("alert_rate", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, len(methods), figsize=(8.2 * len(methods), 6.4), constrained_layout=True, sharey=True)
    if len(methods) == 1:
        axes = [axes]
    quintiles = quintile_labels(5)
    x = np.arange(len(quintiles))
    for ax, method in zip(axes, methods, strict=True):
        panel = frame.loc[frame["method"] == method]
        for status, color in (("Teaching", "#1f77b4"), ("Non-teaching", "#d62728")):
            sub = panel.loc[panel["teaching_status"] == status].set_index("risk_quintile").reindex(quintiles)
            ax.plot(
                x,
                sub["alert_rate"].to_numpy(dtype=float),
                marker="o",
                linewidth=3.2 + LINE_WIDTH_BOOST,
                markersize=8.2 + MARKER_SIZE_BOOST,
                color=color,
                label=status,
            )
        ax.set_title(method_display_label(method))
        ax.set_xticks(x)
        ax.set_xticklabels(quintiles)
        ax.set_xlabel("Baseline-risk quintile")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Recommended alert rate")
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linewidth=3.2 + LINE_WIDTH_BOOST,
            markersize=8.2 + MARKER_SIZE_BOOST,
            label=status,
        )
        for status, color in (("Teaching", "#1f77b4"), ("Non-teaching", "#d62728"))
    ]
    add_top_center_legend(
        fig,
        legend_handles,
        [str(handle.get_label()) for handle in legend_handles],
        ncol=2,
        y=1.04,
    )
    fig.savefig((output_dir / "figure2_policy_allocation").with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig((output_dir / "figure2_policy_allocation").with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_hospital_forest(frame: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for hospital in HOSPITAL_LEVELS:
        subset = frame.loc[frame["hospital"] == hospital].copy()
        treat = subset.loc[subset["alert"] == 1, "composite_outcome"].to_numpy(dtype=float)
        ctrl = subset.loc[subset["alert"] == 0, "composite_outcome"].to_numpy(dtype=float)
        p1 = float(np.mean(treat))
        p0 = float(np.mean(ctrl))
        rd = p1 - p0
        se = math.sqrt((p1 * (1.0 - p1) / max(treat.size, 1)) + (p0 * (1.0 - p0) / max(ctrl.size, 1)))
        rows.append(
            {
                "hospital": hospital,
                "teaching_status": "Teaching" if hospital in TEACHING_HOSPITALS else "Non-teaching",
                "risk_difference": rd,
                "ci_low": rd - 1.96 * se,
                "ci_high": rd + 1.96 * se,
            }
        )
    forest = pd.DataFrame(rows).sort_values("hospital", ascending=False).reset_index(drop=True)
    forest.to_csv(output_dir / "figure_s1_hospital_itt_heterogeneity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 6.8), constrained_layout=True)
    y_pos = np.arange(forest.shape[0])
    colors = ["#1f77b4" if status == "Teaching" else "#d62728" for status in forest["teaching_status"]]
    for y, (_, row) in enumerate(forest.iterrows()):
        ax.hlines(y, row["ci_low"], row["ci_high"], color=colors[y], linewidth=2.8 + LINE_WIDTH_BOOST)
        ax.plot(row["risk_difference"], y, marker="o", color=colors[y], markersize=8.2 + MARKER_SIZE_BOOST)
    ax.axvline(0.0, color="#999999", linestyle="--", linewidth=1.2 + LINE_WIDTH_BOOST)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Hospital {value}" for value in forest["hospital"]], fontsize=20)
    ax.set_xlabel(
        "Risk difference of alert vs control on composite outcome",
        fontsize=22,
    )
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig((output_dir / "figure_s1_hospital_itt_heterogeneity").with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig((output_dir / "figure_s1_hospital_itt_heterogeneity").with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_rho_sweep(summary_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("certified_value", "Certified value"),
        ("composite_free_value", "Composite-free value"),
        ("alert_rate", "Alert rate"),
    ]
    methods = list(summary_df["method"].drop_duplicates())
    fig, axes = plt.subplots(1, len(metrics), figsize=(8.7 * len(metrics), 6.4), constrained_layout=True, sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        for method in methods:
            frame = summary_df.loc[summary_df["method"] == method].sort_values("rho")
            if frame.empty:
                continue
            x = frame["rho"].to_numpy(dtype=float)
            mean = frame[f"{metric}_mean"].to_numpy(dtype=float)
            se = frame[f"{metric}_se"].to_numpy(dtype=float)
            color = METHOD_COLORS.get(method, "#333333")
            ax.plot(
                x,
                mean,
                color=color,
                marker=METHOD_MARKERS.get(method, "o"),
                linewidth=3.0 + LINE_WIDTH_BOOST,
                markersize=8.0 + MARKER_SIZE_BOOST,
                label=method_display_label(method),
            )
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    add_top_center_legend(fig, handles, labels, ncol=max(1, len(methods)), y=1.04)
    fig.savefig((output_dir / "figure_s2_rho_sweep").with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig((output_dir / "figure_s2_rho_sweep").with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_no_hospital_sensitivity(summary_df: pd.DataFrame, output_dir: Path) -> None:
    methods = list(METHOD_ORDER)
    metrics = [
        ("certified_value", "Certified value"),
        ("composite_free_value", "Composite-free value"),
        ("alert_rate", "Alert rate"),
    ]
    covariate_order = ["main", "no_hospital"]
    colors = {"main": "#111111", "no_hospital": "#d95f02"}
    labels = {"main": "Main covariates", "no_hospital": "Without hospital"}
    markers = {"main": "o", "no_hospital": "s"}
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(9.6 * len(metrics), max(5.4, 0.52 * len(methods) + 1.8)),
        constrained_layout=True,
        sharey=True,
    )
    y_pos = np.arange(len(methods))[::-1]
    display_methods = [method_display_label(method) for method in methods]
    for ax, (metric, xlabel) in zip(axes, metrics, strict=True):
        for offset, covariate_set in enumerate(covariate_order):
            sub = (
                summary_df.loc[summary_df["covariate_set"] == covariate_set]
                .set_index("method")
                .loc[methods]
                .reset_index()
            )
            delta = -0.14 if covariate_set == "main" else 0.14
            for idx, (_, row) in enumerate(sub.iterrows()):
                mean = float(row[f"{metric}_mean"])
                se = float(row[f"{metric}_se"])
                y = y_pos[idx] + delta
                ax.hlines(
                    y,
                    mean - 1.96 * se,
                    mean + 1.96 * se,
                    color=colors[covariate_set],
                    linewidth=2.4 + LINE_WIDTH_BOOST,
                )
                ax.plot(
                    mean,
                    y,
                    marker=markers[covariate_set],
                    color=colors[covariate_set],
                    markersize=7.2 + MARKER_SIZE_BOOST,
                    linestyle="none",
                )
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(display_methods)
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[name],
            marker=markers[name],
            linestyle="-",
            linewidth=2.4 + LINE_WIDTH_BOOST,
            markersize=7.2 + MARKER_SIZE_BOOST,
            label=labels[name],
        )
        for name in covariate_order
    ]
    add_top_center_legend(fig, handles, [labels[name] for name in covariate_order], ncol=len(covariate_order), y=1.04)
    fig.savefig((output_dir / "figure_s3_no_hospital_sensitivity").with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig((output_dir / "figure_s3_no_hospital_sensitivity").with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_deployment_sensitivity(summary_df: pd.DataFrame, output_dir: Path) -> None:
    methods = ["PROWL (mean-rule)", "PROWL (MAP)", "PROWL (Gibbs)"]
    summary_frame_to_point_range_plot(
        summary_df,
        metric_specs=(
            ("certified_value", "Certified value"),
            ("composite_free_value", "Composite-free value"),
            ("alert_rate", "Alert rate"),
        ),
        methods=methods,
        output_path=output_dir / "figure_s4_deployment_sensitivity",
    )


def teaching_status_from_hospital(hospital_values: np.ndarray) -> np.ndarray:
    hospital = np.asarray(hospital_values, dtype=float).astype(int)
    return np.where(np.isin(hospital, list(TEACHING_HOSPITALS)), "Teaching", "Non-teaching")


def evaluation_outcome_dict(bundle: RewardBundle) -> dict[str, np.ndarray]:
    return {
        "certified_value": bundle.lower_reward,
        "nominal_value": bundle.nominal_reward,
        "composite_free_value": bundle.composite_free,
        "mortality_risk": bundle.death14,
        "dialysis_risk": bundle.dialysis14,
        "aki_progression_risk": bundle.aki_progression14,
        "composite_outcome_risk": bundle.composite_outcome,
        "discharge_to_home_rate": bundle.discharge_to_home,
    }


def build_split_iterator(frame: pd.DataFrame, *, n_splits: int, test_size: float, random_state: int):
    labels = frame["hospital"].astype(str) + "__" + frame["alert"].astype(str)
    splitter = StratifiedShuffleSplit(
        n_splits=int(n_splits),
        test_size=float(test_size),
        random_state=int(random_state),
    )
    return splitter.split(frame, labels)


def maybe_stratified_subsample(frame: pd.DataFrame, *, target_n: int, random_state: int) -> pd.DataFrame:
    n_total = int(frame.shape[0])
    if target_n <= 0 or target_n >= n_total:
        return frame.reset_index(drop=True).copy()

    strat_labels = frame["hospital"].astype(str) + "__" + frame["alert"].astype(str)
    counts = strat_labels.value_counts().sort_index()
    exact = counts / float(counts.sum()) * int(target_n)
    take = np.floor(exact).astype(int)
    take = np.maximum(take, 1)
    remainder = int(target_n) - int(take.sum())
    if remainder > 0:
        fractional = (exact - np.floor(exact)).sort_values(ascending=False)
        for label in fractional.index[:remainder]:
            take.loc[label] += 1
    elif remainder < 0:
        fractional = (exact - np.floor(exact)).sort_values(ascending=True)
        for label in fractional.index:
            if remainder == 0:
                break
            if take.loc[label] > 1:
                take.loc[label] -= 1
                remainder += 1

    sampled_parts: list[pd.DataFrame] = []
    for label, n_group in take.items():
        group = frame.loc[strat_labels == label]
        n_draw = min(int(n_group), int(group.shape[0]))
        label_seed = int(sum(ord(char) for char in str(label)))
        sampled_parts.append(group.sample(n=n_draw, replace=False, random_state=int(random_state) + label_seed))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=int(random_state)).reset_index(drop=True)


def experiment_settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "penalty_grid": tuple(parse_float_grid(args.penalty_grid)),
        "policy_tree_depth": int(args.policy_tree_depth),
        "policy_tree_min_node_size": int(args.policy_tree_min_node_size),
        "policy_tree_split_step": int(args.policy_tree_split_step),
        "policy_tree_max_features": (
            None if int(args.policy_tree_max_features) <= 0 else int(args.policy_tree_max_features)
        ),
        "eval_nuisance_l2_penalty": float(args.eval_nuisance_l2_penalty),
        "nuisance_l2_penalty": float(args.nuisance_l2_penalty),
        "eta_grid": tuple(parse_float_grid(args.eta_grid)),
        "gamma_grid": tuple(parse_float_grid(args.gamma_grid)),
        "delta": float(args.delta),
        "prior_sd": float(args.prior_sd),
        "score_bound": float(args.score_bound),
        "particle_library_config": ParticleLibraryConfig(
            n_anchor_particles=int(args.n_anchor_particles),
            n_prior_samples=int(args.n_prior_particles),
            n_local_samples_per_anchor=int(args.n_local_particles),
            local_scale=float(args.local_particle_scale),
        ),
    }


def build_split_tasks(
    frame: pd.DataFrame, *, n_splits: int, test_size: float, random_state: int
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    splitter = build_split_iterator(frame, n_splits=n_splits, test_size=test_size, random_state=random_state)
    for split_id, (train_idx, test_idx) in enumerate(splitter, start=1):
        tasks.append(
            {
                "split_id": int(split_id),
                "train_idx": np.asarray(train_idx, dtype=int),
                "test_idx": np.asarray(test_idx, dtype=int),
            }
        )
    return tasks


def _set_policy_shared_state(shared_state: dict[str, Any]) -> None:
    global _POLICY_SHARED_STATE
    _POLICY_SHARED_STATE = shared_state


def _set_deployment_shared_state(shared_state: dict[str, Any]) -> None:
    global _DEPLOYMENT_SHARED_STATE
    _DEPLOYMENT_SHARED_STATE = shared_state


def execute_tasks_with_optional_parallelism(
    *,
    tasks: Sequence[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    initializer: Callable[[dict[str, Any]], None],
    shared_state: dict[str, Any],
    n_jobs: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    completed_results: list[dict[str, Any]] = []
    total = len(tasks)
    if total == 0:
        return completed_results

    max_workers = max(1, min(int(n_jobs), total))
    with tqdm(total=total, desc=progress_label, unit="split", dynamic_ncols=True) as pbar:
        if max_workers <= 1:
            initializer(shared_state)
            for task in tasks:
                completed_results.append(worker(task))
                pbar.update(1)
            return completed_results

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=initializer,
                initargs=(shared_state,),
            ) as executor:
                future_to_task = {executor.submit(worker, task): task for task in tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    completed_results.append(future.result())
                    pbar.update(1)
            return completed_results
        except (OSError, PermissionError, NotImplementedError):
            initializer(shared_state)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(worker, task): task for task in tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    completed_results.append(future.result())
                    pbar.update(1)
            return completed_results


def _run_policy_split_task(task: dict[str, Any]) -> dict[str, Any]:
    if _POLICY_SHARED_STATE is None:
        raise RuntimeError("Policy worker shared state was not initialized.")
    shared = _POLICY_SHARED_STATE
    frame = shared["frame"]
    reward_bundle = shared["reward_bundle"]
    settings = shared["settings"]
    analysis = shared["analysis"]
    covariate_set = shared["covariate_set"]
    include_hospital = bool(shared["include_hospital"])
    methods = tuple(shared["methods"])
    deployment_override = dict(shared["deployment_override"])
    effective_tree_depth = int(shared["effective_tree_depth"])

    split_id = int(task["split_id"])
    train_idx = np.asarray(task["train_idx"], dtype=int)
    test_idx = np.asarray(task["test_idx"], dtype=int)
    train_df = frame.iloc[train_idx].reset_index(drop=True)
    test_df = frame.iloc[test_idx].reset_index(drop=True)
    prep = FoldPreprocessor.fit(train_df, include_hospital=include_hospital)
    x_train = prep.transform(train_df)
    x_test = prep.transform(test_df)
    treatment_train = np.where(train_df["alert"].to_numpy(dtype=float) == 1.0, 1.0, -1.0)
    treatment_test = np.where(test_df["alert"].to_numpy(dtype=float) == 1.0, 1.0, -1.0)

    nominal_reward_train = reward_bundle.nominal_reward[train_idx]
    lower_reward_train = reward_bundle.lower_reward[train_idx]
    eval_outcomes = evaluation_outcome_dict(reward_bundle)
    eval_outcomes_test = {key: value[test_idx] for key, value in eval_outcomes.items()}
    eval_nuisance_models = {
        metric: fit_outcome_nuisance(
            x=x_train,
            treatment=treatment_train,
            outcome=reward_bundle_value[train_idx],
            feature_map_factory=prep.nuisance_feature_map_factory(),
            l2_penalty=settings["eval_nuisance_l2_penalty"],
        )
        for metric, reward_bundle_value in eval_outcomes.items()
    }

    fit_results: dict[str, PolicyFit] = {}
    for method in methods:
        fit_results[method] = fit_requested_method(
            method=method,
            x_train=x_train,
            treatment_train=treatment_train,
            nominal_reward_train=nominal_reward_train,
            lower_reward_train=lower_reward_train,
            feature_map_factory=prep.policy_feature_map_factory(),
            nuisance_feature_map_factory=prep.nuisance_feature_map_factory(),
            penalty_grid=settings["penalty_grid"],
            policy_tree_depth=effective_tree_depth,
            policy_tree_min_node_size=settings["policy_tree_min_node_size"],
            policy_tree_split_step=settings["policy_tree_split_step"],
            policy_tree_max_features=settings["policy_tree_max_features"],
            eval_nuisance_l2_penalty=settings["eval_nuisance_l2_penalty"],
            nuisance_l2_penalty=settings["nuisance_l2_penalty"],
            eta_grid=settings["eta_grid"],
            gamma_grid=settings["gamma_grid"],
            delta=settings["delta"],
            prior_sd=settings["prior_sd"],
            score_bound=settings["score_bound"],
            particle_library_config=settings["particle_library_config"],
            random_state=settings["seed"] + 101 * split_id,
        )

    risk_model = fit_control_risk_model(
        x=x_train,
        treatment=treatment_train,
        outcome=reward_bundle.composite_outcome[train_idx],
        feature_map_factory=prep.nuisance_feature_map_factory(),
        l2_penalty=settings["eval_nuisance_l2_penalty"],
    )
    risk_quintile = risk_quintiles(risk_model.predict(x_test), n_levels=5)
    teaching_status = teaching_status_from_hospital(test_df["hospital"].to_numpy(dtype=float))

    raw_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    for method, fit_result in fit_results.items():
        deployment = deployment_override.get(method, "mean_rule")
        deployment_for_eval = "deterministic" if deployment == "mean_rule" else deployment
        metrics = evaluate_policy_bundle(
            policy_fit=fit_result,
            x_test=x_test,
            treatment_test=treatment_test,
            eval_outcomes=eval_outcomes_test,
            eval_nuisance_models=eval_nuisance_models,
            deployment=deployment_for_eval,
        )
        row = {
            "analysis": analysis,
            "covariate_set": covariate_set,
            "utility_spec": reward_bundle.utility_spec,
            "rho": reward_bundle.rho,
            "split": split_id,
            "method": method,
            "deployment": deployment,
            "runtime_sec": fit_result.runtime_sec,
        }
        row.update(fit_result.fit_info)
        row.update(metrics)
        raw_rows.append(row)

        prob = policy_treat_probability(fit_result.policy, x_test, deployment=deployment_for_eval)
        for quintile in quintile_labels(5):
            for status in ("Teaching", "Non-teaching"):
                mask = (risk_quintile == quintile) & (teaching_status == status)
                allocation_rows.append(
                    {
                        "analysis": analysis,
                        "covariate_set": covariate_set,
                        "utility_spec": reward_bundle.utility_spec,
                        "rho": reward_bundle.rho,
                        "split": split_id,
                        "method": method,
                        "risk_quintile": quintile,
                        "teaching_status": status,
                        "alert_rate": float(np.mean(prob[mask])) if np.any(mask) else math.nan,
                    }
                )
    return {"raw_rows": raw_rows, "allocation_rows": allocation_rows}


def _run_deployment_split_task(task: dict[str, Any]) -> dict[str, Any]:
    if _DEPLOYMENT_SHARED_STATE is None:
        raise RuntimeError("Deployment worker shared state was not initialized.")
    shared = _DEPLOYMENT_SHARED_STATE
    frame = shared["frame"]
    reward_bundle = shared["reward_bundle"]
    settings = shared["settings"]

    split_id = int(task["split_id"])
    train_idx = np.asarray(task["train_idx"], dtype=int)
    test_idx = np.asarray(task["test_idx"], dtype=int)
    train_df = frame.iloc[train_idx].reset_index(drop=True)
    test_df = frame.iloc[test_idx].reset_index(drop=True)
    prep = FoldPreprocessor.fit(train_df, include_hospital=True)
    x_train = prep.transform(train_df)
    x_test = prep.transform(test_df)
    treatment_train = np.where(train_df["alert"].to_numpy(dtype=float) == 1.0, 1.0, -1.0)
    treatment_test = np.where(test_df["alert"].to_numpy(dtype=float) == 1.0, 1.0, -1.0)
    nominal_reward_train = reward_bundle.nominal_reward[train_idx]
    lower_reward_train = reward_bundle.lower_reward[train_idx]
    eval_outcomes = evaluation_outcome_dict(reward_bundle)
    eval_outcomes_test = {key: value[test_idx] for key, value in eval_outcomes.items()}
    eval_nuisance_models = {
        metric: fit_outcome_nuisance(
            x=x_train,
            treatment=treatment_train,
            outcome=reward_bundle_value[train_idx],
            feature_map_factory=prep.nuisance_feature_map_factory(),
            l2_penalty=settings["eval_nuisance_l2_penalty"],
        )
        for metric, reward_bundle_value in eval_outcomes.items()
    }
    fit_result = fit_requested_method(
        method="PROWL",
        x_train=x_train,
        treatment_train=treatment_train,
        nominal_reward_train=nominal_reward_train,
        lower_reward_train=lower_reward_train,
        feature_map_factory=prep.policy_feature_map_factory(),
        nuisance_feature_map_factory=prep.nuisance_feature_map_factory(),
        penalty_grid=settings["penalty_grid"],
        policy_tree_depth=settings["policy_tree_depth"],
        policy_tree_min_node_size=settings["policy_tree_min_node_size"],
        policy_tree_split_step=settings["policy_tree_split_step"],
        policy_tree_max_features=settings["policy_tree_max_features"],
        eval_nuisance_l2_penalty=settings["eval_nuisance_l2_penalty"],
        nuisance_l2_penalty=settings["nuisance_l2_penalty"],
        eta_grid=settings["eta_grid"],
        gamma_grid=settings["gamma_grid"],
        delta=settings["delta"],
        prior_sd=settings["prior_sd"],
        score_bound=settings["score_bound"],
        particle_library_config=settings["particle_library_config"],
        random_state=settings["seed"] + 401 * split_id,
    )
    deployment_specs = {
        "PROWL (mean-rule)": "deterministic",
        "PROWL (MAP)": "map",
        "PROWL (Gibbs)": "gibbs",
    }

    raw_rows: list[dict[str, Any]] = []
    for label, deployment in deployment_specs.items():
        metrics = evaluate_policy_bundle(
            policy_fit=fit_result,
            x_test=x_test,
            treatment_test=treatment_test,
            eval_outcomes=eval_outcomes_test,
            eval_nuisance_models=eval_nuisance_models,
            deployment=deployment,
        )
        row = {
            "analysis": "deployment",
            "covariate_set": "main",
            "utility_spec": reward_bundle.utility_spec,
            "rho": reward_bundle.rho,
            "split": split_id,
            "method": label,
            "deployment": deployment,
            "runtime_sec": fit_result.runtime_sec,
            "eta": fit_result.fit_info.get("eta"),
            "gamma": fit_result.fit_info.get("gamma"),
        }
        row.update(metrics)
        raw_rows.append(row)
    return {"raw_rows": raw_rows}


def run_policy_experiment(
    *,
    frame: pd.DataFrame,
    reward_bundle: RewardBundle,
    args: argparse.Namespace,
    analysis: str,
    covariate_set: str,
    methods: Sequence[str],
    include_hospital: bool,
    deployment_map: Mapping[str, str] | None = None,
    policy_tree_depth: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deployment_override = dict(deployment_map or {})
    effective_tree_depth = int(args.policy_tree_depth if policy_tree_depth is None else policy_tree_depth)
    tasks = build_split_tasks(frame, n_splits=args.n_splits, test_size=args.test_size, random_state=args.seed)
    shared_state = {
        "frame": frame,
        "reward_bundle": reward_bundle,
        "settings": experiment_settings_from_args(args),
        "analysis": analysis,
        "covariate_set": covariate_set,
        "include_hospital": include_hospital,
        "methods": tuple(methods),
        "deployment_override": deployment_override,
        "effective_tree_depth": effective_tree_depth,
    }
    results = execute_tasks_with_optional_parallelism(
        tasks=tasks,
        worker=_run_policy_split_task,
        initializer=_set_policy_shared_state,
        shared_state=shared_state,
        n_jobs=int(args.n_jobs),
        progress_label=f"{analysis}:{covariate_set}",
    )
    raw_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    for result in results:
        raw_rows.extend(result["raw_rows"])
        allocation_rows.extend(result["allocation_rows"])
    return pd.DataFrame(raw_rows), pd.DataFrame(allocation_rows)


def run_deployment_sensitivity(
    *,
    frame: pd.DataFrame,
    reward_bundle: RewardBundle,
    args: argparse.Namespace,
) -> pd.DataFrame:
    tasks = build_split_tasks(frame, n_splits=args.n_splits, test_size=args.test_size, random_state=args.seed)
    shared_state = {
        "frame": frame,
        "reward_bundle": reward_bundle,
        "settings": experiment_settings_from_args(args),
    }
    results = execute_tasks_with_optional_parallelism(
        tasks=tasks,
        worker=_run_deployment_split_task,
        initializer=_set_deployment_shared_state,
        shared_state=shared_state,
        n_jobs=int(args.n_jobs),
        progress_label="deployment",
    )
    raw_rows: list[dict[str, Any]] = []
    for result in results:
        raw_rows.extend(result["raw_rows"])
    return pd.DataFrame(raw_rows)


def save_manifest(args: argparse.Namespace, output_dir: Path) -> None:
    manifest = vars(args).copy()
    manifest["main_covariate_columns"] = list(MAIN_COVARIATE_COLUMNS)
    manifest["continuous_covariate_columns"] = list(CONTINUOUS_COVARIATE_COLUMNS)
    manifest["hospital_levels"] = list(HOSPITAL_LEVELS)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "real_world" / "AKI-alert-trial" / "ELAIA-1_deidentified_data_10-6-2020.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "elaia1_real_data"),
    )
    parser.add_argument("--n-splits", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument("--subsample-n", type=int, default=0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--rho-grid", default="0,0.5,1,1.5,2")
    parser.add_argument("--main-weights", default="0.60,0.25,0.15")
    parser.add_argument("--main-deltas", default="0.10,0.05,0.05")
    parser.add_argument("--patient4-weights", default="0.55,0.20,0.15,0.10")
    parser.add_argument("--patient4-deltas", default="0.10,0.05,0.05,0.05")
    parser.add_argument("--penalty-grid", default="1e-3,1e-2,1e-1,1")
    parser.add_argument("--eta-grid", default="0.125,0.25,0.5,1,2,4,8")
    parser.add_argument("--gamma-grid", default="0.125,0.25,0.5,1,2,4,8")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--prior-sd", type=float, default=5.0)
    parser.add_argument("--score-bound", type=float, default=3.0)
    parser.add_argument("--nuisance-l2-penalty", type=float, default=1e-3)
    parser.add_argument("--eval-nuisance-l2-penalty", type=float, default=1e-3)
    parser.add_argument("--policy-tree-depth", type=int, default=2)
    parser.add_argument("--policy-tree-depth-sensitivity", type=int, default=3)
    parser.add_argument("--policy-tree-min-node-size", type=int, default=200)
    parser.add_argument("--policy-tree-split-step", type=int, default=1)
    parser.add_argument("--policy-tree-max-features", type=int, default=0)
    parser.add_argument("--n-anchor-particles", type=int, default=16)
    parser.add_argument("--n-prior-particles", type=int, default=256)
    parser.add_argument("--n-local-particles", type=int, default=48)
    parser.add_argument("--local-particle-scale", type=float, default=0.30)
    parser.add_argument("--skip-rho-sweep", action="store_true")
    parser.add_argument("--skip-no-hospital-sensitivity", action="store_true")
    parser.add_argument("--skip-deployment-sensitivity", action="store_true")
    parser.add_argument("--skip-patient-centered-sensitivity", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: fewer splits and lighter appendix workload.",
    )
    args = parser.parse_args()
    if args.quick:
        args.n_splits = min(args.n_splits, 2)
        args.subsample_n = args.subsample_n or 1200
        args.rho_grid = "0,1"
        args.penalty_grid = "1e-2"
        args.eta_grid = "0.25,1,4"
        args.gamma_grid = "0.25,1,4"
        args.policy_tree_min_node_size = min(args.policy_tree_min_node_size, 80)
        args.n_anchor_particles = min(args.n_anchor_particles, 8)
        args.n_prior_particles = min(args.n_prior_particles, 64)
        args.n_local_particles = min(args.n_local_particles, 12)
        args.skip_rho_sweep = True
        args.skip_no_hospital_sensitivity = True
        args.skip_deployment_sensitivity = True
        args.skip_patient_centered_sensitivity = True
    return args


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    appendix_dir = output_dir / "appendix"
    for directory in (plots_dir, tables_dir, appendix_dir):
        directory.mkdir(parents=True, exist_ok=True)
    save_manifest(args, output_dir)

    frame = load_elaia_data(args.data_path)
    frame = maybe_stratified_subsample(frame, target_n=int(args.subsample_n), random_state=int(args.seed))
    build_table_s1(frame, appendix_dir)
    plot_hospital_forest(frame, appendix_dir)

    specs = utility_specifications(args)
    main_bundle = reward_bundle_from_spec(frame, specs["hard_clinical_3"], rho=float(args.rho))

    main_raw, main_alloc = run_policy_experiment(
        frame=frame,
        reward_bundle=main_bundle,
        args=args,
        analysis="main",
        covariate_set="main",
        methods=METHOD_ORDER,
        include_hospital=True,
    )
    main_raw.to_csv(output_dir / "main_split_metrics.csv", index=False)
    main_alloc.to_csv(output_dir / "main_allocation_split_metrics.csv", index=False)

    main_summary = summarize_split_metrics(
        main_raw,
        group_cols=("analysis", "covariate_set", "utility_spec", "rho", "method"),
        metric_cols=ALL_EVAL_METRICS,
    )
    main_summary.to_csv(output_dir / "main_summary.csv", index=False)
    paired_vs_never = paired_difference_summary(
        main_raw,
        reference_method="Never alert",
        metric_cols=TABLE1_COLUMNS,
        group_cols=("analysis", "covariate_set", "utility_spec", "rho"),
    )
    paired_vs_never.to_csv(output_dir / "main_paired_differences_vs_never.csv", index=False)
    main_method_summary = main_summary.loc[
        (main_summary["analysis"] == "main") & (main_summary["covariate_set"] == "main")
    ].copy()
    table1_frame = build_table1_outputs(main_method_summary, paired_vs_never, tables_dir)
    build_table_s2_outputs(main_method_summary, appendix_dir)
    plot_main_figure1(main_method_summary, plots_dir)

    best_raw_method = select_best_method(main_method_summary, RAW_BASELINE_METHODS)
    best_lower_method = select_best_method(main_method_summary, LOWER_PLUGIN_METHODS)
    plot_figure2_allocation(main_alloc, methods=("PROWL", best_raw_method, best_lower_method), output_dir=plots_dir)

    if not args.skip_rho_sweep:
        rho_rows: list[pd.DataFrame] = []
        selected_methods = ("PROWL", "PROWL (U=0)", best_raw_method, best_lower_method)
        for rho in parse_float_grid(args.rho_grid):
            bundle = reward_bundle_from_spec(frame, specs["hard_clinical_3"], rho=float(rho))
            raw_df, _ = run_policy_experiment(
                frame=frame,
                reward_bundle=bundle,
                args=args,
                analysis="rho_sweep",
                covariate_set="main",
                methods=selected_methods,
                include_hospital=True,
            )
            rho_rows.append(raw_df)
        rho_raw = pd.concat(rho_rows, ignore_index=True)
        rho_raw.to_csv(appendix_dir / "rho_sweep_split_metrics.csv", index=False)
        rho_summary = summarize_split_metrics(
            rho_raw,
            group_cols=("analysis", "covariate_set", "utility_spec", "rho", "method"),
            metric_cols=("certified_value", "composite_free_value", "alert_rate"),
        )
        rho_summary.to_csv(appendix_dir / "rho_sweep_summary.csv", index=False)
        plot_rho_sweep(rho_summary, appendix_dir)

    if not args.skip_no_hospital_sensitivity:
        no_hospital_raw, _ = run_policy_experiment(
            frame=frame,
            reward_bundle=main_bundle,
            args=args,
            analysis="no_hospital",
            covariate_set="no_hospital",
            methods=METHOD_ORDER,
            include_hospital=False,
        )
        no_hospital_raw.to_csv(appendix_dir / "no_hospital_split_metrics.csv", index=False)
        no_hospital_summary = summarize_split_metrics(
            pd.concat([main_raw, no_hospital_raw], ignore_index=True),
            group_cols=("analysis", "covariate_set", "utility_spec", "rho", "method"),
            metric_cols=("certified_value", "composite_free_value", "alert_rate"),
        )
        no_hospital_summary.to_csv(appendix_dir / "no_hospital_summary.csv", index=False)
        plot_no_hospital_sensitivity(
            no_hospital_summary.loc[no_hospital_summary["covariate_set"].isin(["main", "no_hospital"])].copy(),
            appendix_dir,
        )

    if not args.skip_deployment_sensitivity:
        deployment_raw = run_deployment_sensitivity(frame=frame, reward_bundle=main_bundle, args=args)
        deployment_raw.to_csv(appendix_dir / "deployment_split_metrics.csv", index=False)
        deployment_summary = summarize_split_metrics(
            deployment_raw,
            group_cols=("analysis", "covariate_set", "utility_spec", "rho", "method"),
            metric_cols=("certified_value", "composite_free_value", "alert_rate"),
        )
        deployment_summary.to_csv(appendix_dir / "deployment_summary.csv", index=False)
        plot_deployment_sensitivity(deployment_summary, appendix_dir)

    if not args.skip_patient_centered_sensitivity:
        patient4_bundle = reward_bundle_from_spec(frame, specs["patient_centered_4"], rho=float(args.rho))
        patient4_raw, _ = run_policy_experiment(
            frame=frame,
            reward_bundle=patient4_bundle,
            args=args,
            analysis="patient_centered_4",
            covariate_set="main",
            methods=METHOD_ORDER,
            include_hospital=True,
        )
        patient4_raw.to_csv(appendix_dir / "patient_centered_4_split_metrics.csv", index=False)
        patient4_summary = summarize_split_metrics(
            patient4_raw,
            group_cols=("analysis", "covariate_set", "utility_spec", "rho", "method"),
            metric_cols=ALL_EVAL_METRICS,
        )
        patient4_summary.to_csv(appendix_dir / "patient_centered_4_summary.csv", index=False)


if __name__ == "__main__":
    main()
