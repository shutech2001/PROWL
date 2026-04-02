from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from pac_owl.estimators import StandardizedFeatureMap

BasisFactory = Callable[[], StandardizedFeatureMap]


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    display_name: str
    description: str
    feature_map_factory: BasisFactory
    source: str
    geometry: str
    uncertainty: str
    p: int | None = None
    nuisance_feature_map_factory: BasisFactory | None = None


@dataclass
class CertificateModel:
    feature_map: StandardizedFeatureMap
    coefficients: NDArray
    residual_quantile: float

    def predict(self, x: NDArray, a: NDArray) -> NDArray:
        phi = self.feature_map.transform(x)
        a = np.asarray(a, dtype=float).reshape(-1, 1)
        design = np.concatenate([np.ones((x.shape[0], 1)), a, phi, a * phi], axis=1)
        fitted = design @ self.coefficients
        return np.clip(np.maximum(fitted, 0.0) + self.residual_quantile, 0.0, 1.0)


@dataclass
class ObservedBanditSample:
    x: NDArray
    a: NDArray
    pi: NDArray
    reward: NDArray
    reward_star: NDArray
    bias: NDArray
    certificate: NDArray
    lower_reward: NDArray
    oracle_action: NDArray
    robust_oracle_action: NDArray
    proxy_oracle_action: NDArray


@dataclass
class PotentialOutcomeSample:
    x: NDArray
    reward_pos: NDArray
    reward_neg: NDArray
    reward_star_pos: NDArray
    reward_star_neg: NDArray
    mean_reward_pos: NDArray
    mean_reward_neg: NDArray
    mean_reward_star_pos: NDArray
    mean_reward_star_neg: NDArray
    bias_pos: NDArray
    bias_neg: NDArray
    certificate_pos: NDArray
    certificate_neg: NDArray
    lower_reward_pos: NDArray
    lower_reward_neg: NDArray
    mean_lower_reward_pos: NDArray
    mean_lower_reward_neg: NDArray
    oracle_action: NDArray
    robust_oracle_action: NDArray
    proxy_oracle_action: NDArray


@dataclass
class GeneratedDataset:
    scenario: ScenarioDefinition
    rho: float
    certificate_mode: str
    certificate_scale: float
    train: ObservedBanditSample
    test: PotentialOutcomeSample
    best_constant_target_value: float
    best_constant_robust_value: float
    best_constant_proxy_value: float
    target_oracle_value: float
    robust_oracle_value: float
    proxy_oracle_value: float
    certificate_model: CertificateModel | None = None
    certificate_validity_rate_test: float | None = None
    propensity_mode: str = "constant"
    propensity_strength: float = 1.0


def linear_feature_map_factory() -> StandardizedFeatureMap:
    return StandardizedFeatureMap()


def clinical_triage_feature_map_factory() -> StandardizedFeatureMap:
    def basis_fn(z: NDArray) -> NDArray:
        x1 = z[:, [0]]
        x2 = z[:, [1]]
        x3 = z[:, [2]]
        x4 = z[:, [3]]
        x5 = z[:, [4]]
        x6 = z[:, [5]]
        x7 = z[:, [6]]
        x8 = z[:, [7]]
        return np.concatenate(
            [
                z,
                x1**2,
                x2**2,
                x1 * x2,
                np.sin(np.pi * x1),
                np.sin(np.pi * x2),
                x3**2,
                x4**2,
                x3 * x4,
                np.sin(np.pi * x3),
                np.sin(np.pi * x4),
                x5**2,
                x6**2,
                x5 * x6,
                np.maximum(x5, 0.0),
                np.maximum(x6, 0.0),
                x7**2,
                x8**2,
                x7 * x8,
                np.maximum(x7, 0.0),
                x1 * x3,
                x1 * x4,
                x2 * x3,
                x2 * x4,
                x4 * x6,
                x5 * x6,
                x5 * x7,
                x6 * x7,
                x4 * x7,
                x3 * x8,
            ],
            axis=1,
        )

    return StandardizedFeatureMap(basis_fn=basis_fn)


def available_scenarios() -> dict[str, ScenarioDefinition]:
    return {
        "clinical_triage_nuisance_conflict": ScenarioDefinition(
            scenario_id="clinical_triage_nuisance_conflict",
            display_name="Clinical Triage Nuisance Conflict",
            description="Moderate-risk patients benefit from treatment, but frailty, operational strain, and surrogate-discordant biomarkers create treated-arm lower-value penalties and triage selection into clinically ambiguous cases near the decision boundary; all methods share the same rich clinical feature class, so differences come from the estimator.",  # noqa: E501
            feature_map_factory=clinical_triage_feature_map_factory,
            nuisance_feature_map_factory=clinical_triage_feature_map_factory,
            source="synthetic",
            geometry="clinical_gatekeeping",
            uncertainty="clinical_triage_conflict",
            p=8,
        ),
        "linear_scope_invariant": ScenarioDefinition(
            scenario_id="linear_scope_invariant",
            display_name="Linear-Scope Invariant",
            description="Comparator-scope control with linear main effects, linear treatment effects, and policy-invariant uncertainty under a linear feature map.",  # noqa: E501
            feature_map_factory=linear_feature_map_factory,
            source="synthetic",
            geometry="linear_scope",
            uncertainty="invariant",
            p=2,
        ),
    }


def _sample_synthetic_covariates(n: int, p: int, rng: np.random.Generator) -> NDArray:
    return rng.uniform(-1.0, 1.0, size=(n, p))


def _sample_synthetic_covariates_for_scenario(
    scenario: ScenarioDefinition,
    n: int,
    rng: np.random.Generator,
) -> NDArray:
    if scenario.scenario_id != "clinical_triage_nuisance_conflict":
        return _sample_synthetic_covariates(n, int(scenario.p), rng)
    x = rng.uniform(-1.0, 1.0, size=(n, int(scenario.p)))
    x[:, 4] = np.clip(0.50 * x[:, 0] - 0.22 * x[:, 1] + 0.14 * x[:, 2] + 0.32 * rng.normal(size=n), -1.0, 1.0)
    x[:, 5] = np.clip(-0.30 * x[:, 1] + 0.26 * x[:, 2] + 0.18 * x[:, 4] + 0.34 * rng.normal(size=n), -1.0, 1.0)
    x[:, 6] = np.clip(0.68 * x[:, 4] + 0.40 * x[:, 5] + 0.30 * x[:, 0] + 0.28 * rng.normal(size=n), -1.0, 1.0)
    x[:, 7] = np.clip(
        0.46 * x[:, 2] - 0.28 * x[:, 3] + 0.60 * x[:, 6] + 0.18 * x[:, 4] + 0.30 * rng.normal(size=n), -1.0, 1.0
    )
    return x


def _bounded_main_effect(scenario: ScenarioDefinition, x: NDArray) -> NDArray:
    x1 = x[:, 0]
    x2 = x[:, 1]
    if scenario.scenario_id == "clinical_triage_nuisance_conflict":
        raw = (
            0.20 * np.sin(1.2 * np.pi * x[:, 2])
            + 0.14 * np.cos(0.8 * np.pi * x[:, 3])
            + 0.10 * x[:, 4]
            - 0.08 * x[:, 5]
            + 0.06 * x[:, 6] * x[:, 7]
        )
        return np.clip(raw, -1.0, 1.0)
    if scenario.geometry == "linear_scope":
        return np.clip(0.38 * x1 - 0.22 * x2, -1.0, 1.0)
    raise ValueError(f"Unknown scenario for main effect: {scenario.scenario_id}")


def _bounded_treatment_effect(scenario: ScenarioDefinition, x: NDArray) -> NDArray:
    x1 = x[:, 0]
    x2 = x[:, 1]
    if scenario.scenario_id == "clinical_triage_nuisance_conflict":
        risk = 0.90 * x[:, 0] - 0.64 * x[:, 1] + 0.45 * np.sin(np.pi * x[:, 2]) + 0.24 * x[:, 3]
        moderate_window = np.exp(-2.9 * (risk - 0.02) ** 2)
        biomarker_alignment = (
            0.14 * np.tanh(x[:, 2] - 0.85 * x[:, 3]) - 0.04 * np.maximum(x[:, 4], 0.0) - 0.03 * np.maximum(x[:, 5], 0.0)
        )
        tau = 1.00 * np.tanh(2.05 * (moderate_window + biomarker_alignment - 0.60))
    elif scenario.geometry == "linear_scope":
        tau = 0.72 * (x1 + 0.65 * x2) / 1.65
    else:
        raise ValueError(f"Unknown scenario for treatment effect: {scenario.scenario_id}")
    return np.clip(tau, -1.0, 1.0)


def _uncertainty_envelope(scenario: ScenarioDefinition, x: NDArray) -> tuple[NDArray, NDArray]:
    if scenario.scenario_id == "clinical_triage_nuisance_conflict":
        risk = 0.94 * x[:, 0] - 0.70 * x[:, 1] + 0.48 * np.sin(np.pi * x[:, 2]) + 0.24 * x[:, 3]
        high_risk = 1.0 / (1.0 + np.exp(-2.6 * (risk - 0.02)))
        boundary_band = np.exp(-8.6 * (risk - 0.02) ** 2)
        frailty = 0.78 * np.maximum(x[:, 4], 0.0) + 0.58 * np.maximum(x[:, 5], 0.0)
        operational_strain = 1.0 / (1.0 + np.exp(-2.4 * (1.18 * x[:, 6] + 0.60 * x[:, 4] - 0.12)))
        surrogate_discordance = 1.0 / (
            1.0
            + np.exp(
                -2.8
                * (
                    1.28 * x[:, 7]
                    + 0.92 * x[:, 2] * x[:, 3]
                    + 0.78 * x[:, 4] * x[:, 6]
                    + 0.48 * x[:, 5] * x[:, 7]
                    - 0.06
                )
            )
        )
        treated_only = 1.0 / (
            1.0
            + np.exp(
                -2.5 * (1.08 * x[:, 6] + 0.96 * x[:, 7] + 0.92 * x[:, 4] * x[:, 6] + 0.70 * x[:, 5] * x[:, 7] - 0.08)
            )
        )
        control_only = 1.0 / (1.0 + np.exp(-1.9 * (0.48 * x[:, 5] - 0.52 * x[:, 6] - 0.28 * x[:, 7] - 0.10)))
        u_pos = (
            0.010
            + 0.020 * high_risk
            + boundary_band
            * (0.16 * frailty + 0.17 * surrogate_discordance + 0.11 * operational_strain + 0.19 * treated_only)
            + 0.075 * treated_only
            + 0.022 * frailty * treated_only
        )
        u_neg = 0.002 + 0.006 * high_risk + boundary_band * (0.010 * control_only) + 0.004 * control_only
        return np.clip(u_pos, 0.0, 0.32), np.clip(u_neg, 0.0, 0.06)

    x1 = x[:, 0]
    x2 = x[:, 1]
    local_region = ((x1 > 0.0) & (x2 > 0.0)).astype(float)
    if scenario.uncertainty == "invariant":
        base = 0.02 + 0.04 * local_region
        return base.copy(), base.copy()
    raise ValueError(f"Unknown uncertainty regime: {scenario.uncertainty}")


def _generate_potential_outcomes(
    scenario: ScenarioDefinition,
    x: NDArray,
    rho: float,
    rng: np.random.Generator,
) -> PotentialOutcomeSample:
    m = _bounded_main_effect(scenario, x)
    tau = _bounded_treatment_effect(scenario, x)
    u_pos_base, u_neg_base = _uncertainty_envelope(scenario, x)
    uncertainty_cap = 0.32 if scenario.scenario_id == "clinical_triage_nuisance_conflict" else 0.10
    u_pos = np.clip(float(rho) * u_pos_base, 0.0, uncertainty_cap)
    u_neg = np.clip(float(rho) * u_neg_base, 0.0, uncertainty_cap)

    if scenario.scenario_id == "clinical_triage_nuisance_conflict":
        pos_bias_mean = 0.92
        neg_bias_mean = 0.14
    else:
        pos_bias_mean = 0.50
        neg_bias_mean = 0.50

    mean_reward_star_pos = np.clip(0.5 + 0.15 * m + 0.15 * tau, 0.0, 1.0)
    mean_reward_star_neg = np.clip(0.5 + 0.15 * m - 0.15 * tau, 0.0, 1.0)
    mean_reward_pos = np.clip(mean_reward_star_pos + pos_bias_mean * u_pos, 0.0, 1.0)
    mean_reward_neg = np.clip(mean_reward_star_neg + neg_bias_mean * u_neg, 0.0, 1.0)

    if scenario.scenario_id == "clinical_triage_nuisance_conflict":
        pos_noise_scale = 0.08 + 0.68 * u_pos
        neg_noise_scale = 0.05 + 0.04 * u_neg
        reward_star_pos = np.clip(
            mean_reward_star_pos + pos_noise_scale * rng.uniform(-1.0, 1.0, size=x.shape[0]),
            0.0,
            1.0,
        )
        reward_star_neg = np.clip(
            mean_reward_star_neg + neg_noise_scale * rng.uniform(-1.0, 1.0, size=x.shape[0]),
            0.0,
            1.0,
        )
    else:
        reward_star_pos = np.clip(mean_reward_star_pos + 0.10 * rng.uniform(-1.0, 1.0, size=x.shape[0]), 0.0, 1.0)
        reward_star_neg = np.clip(mean_reward_star_neg + 0.10 * rng.uniform(-1.0, 1.0, size=x.shape[0]), 0.0, 1.0)
    if scenario.scenario_id == "clinical_triage_nuisance_conflict":
        bias_pos = u_pos * rng.beta(7.0, 1.0, size=x.shape[0])
        bias_neg = u_neg * rng.beta(1.08, 5.2, size=x.shape[0])
    else:
        bias_pos = u_pos * rng.uniform(0.0, 1.0, size=x.shape[0])
        bias_neg = u_neg * rng.uniform(0.0, 1.0, size=x.shape[0])
    reward_pos = np.clip(reward_star_pos + bias_pos, 0.0, 1.0)
    reward_neg = np.clip(reward_star_neg + bias_neg, 0.0, 1.0)

    oracle_action = np.where(mean_reward_star_pos >= mean_reward_star_neg, 1, -1)
    proxy_oracle_action = np.where(mean_reward_pos >= mean_reward_neg, 1, -1)
    return PotentialOutcomeSample(
        x=x,
        reward_pos=reward_pos,
        reward_neg=reward_neg,
        reward_star_pos=reward_star_pos,
        reward_star_neg=reward_star_neg,
        mean_reward_pos=mean_reward_pos,
        mean_reward_neg=mean_reward_neg,
        mean_reward_star_pos=mean_reward_star_pos,
        mean_reward_star_neg=mean_reward_star_neg,
        bias_pos=bias_pos,
        bias_neg=bias_neg,
        certificate_pos=u_pos.copy(),
        certificate_neg=u_neg.copy(),
        lower_reward_pos=reward_pos.copy(),
        lower_reward_neg=reward_neg.copy(),
        mean_lower_reward_pos=mean_reward_pos.copy(),
        mean_lower_reward_neg=mean_reward_neg.copy(),
        oracle_action=oracle_action,
        robust_oracle_action=proxy_oracle_action.copy(),
        proxy_oracle_action=proxy_oracle_action,
    )


def _propensity_treat_one(
    scenario: ScenarioDefinition,
    x: NDArray,
    *,
    propensity_mode: str,
    propensity_strength: float,
) -> NDArray:
    mode = propensity_mode.strip().lower()
    if mode in {"constant", "balanced"}:
        return np.full(x.shape[0], 0.5, dtype=float)
    strength = float(propensity_strength)
    if scenario.scenario_id == "clinical_triage_nuisance_conflict":
        risk = 0.84 * x[:, 0] - 0.60 * x[:, 1] + 0.42 * np.sin(np.pi * x[:, 2]) + 0.20 * x[:, 3]
        logit = strength * (
            0.20 * risk
            + 1.35 * np.maximum(x[:, 4], 0.0)
            + 1.10 * np.maximum(x[:, 5], 0.0)
            + 1.55 * x[:, 6]
            + 1.25 * x[:, 7]
            + 1.55 * x[:, 4] * x[:, 6]
            + 0.95 * x[:, 5] * x[:, 7]
            - 0.40 * x[:, 2] * x[:, 3]
        )
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.clip(p, 0.01, 0.99)
    x1 = x[:, 0]
    x2 = x[:, 1] if x.shape[1] > 1 else 0.0
    logit = strength * (0.8 * x1 - 0.6 * x2)
    p = 1.0 / (1.0 + np.exp(-logit))
    return np.clip(p, 0.20, 0.80)


def fit_data_derived_certificate(
    scenario: ScenarioDefinition,
    *,
    rho: float,
    n_auxiliary: int,
    rng: np.random.Generator,
    propensity_mode: str = "constant",
    propensity_strength: float = 1.0,
) -> CertificateModel:
    if n_auxiliary < 20:
        raise ValueError("n_auxiliary must be at least 20 for the data-derived certificate study.")
    auxiliary = generate_logged_dataset(
        scenario=scenario,
        n_train=n_auxiliary,
        n_test=1,
        rho=rho,
        certificate_mode="oracle",
        certificate_scale=1.0,
        rng=rng,
        n_certificate_auxiliary=n_auxiliary,
        propensity_mode=propensity_mode,
        propensity_strength=propensity_strength,
    ).train
    x_aux = auxiliary.x
    a_aux = auxiliary.a
    d_aux = auxiliary.bias
    split = n_auxiliary // 2
    perm = rng.permutation(n_auxiliary)
    fit_idx = perm[:split]
    cal_idx = perm[split:]
    basis_map = scenario.feature_map_factory()
    phi_fit = basis_map.fit_transform(x_aux[fit_idx])
    a_fit = a_aux[fit_idx][:, None]
    design_fit = np.concatenate([np.ones((phi_fit.shape[0], 1)), a_fit, phi_fit, a_fit * phi_fit], axis=1)
    ridge = 1e-6 * np.eye(design_fit.shape[1], dtype=float)
    coefficients = np.linalg.solve(design_fit.T @ design_fit + ridge, design_fit.T @ d_aux[fit_idx])
    phi_cal = basis_map.transform(x_aux[cal_idx])
    a_cal = a_aux[cal_idx][:, None]
    design_cal = np.concatenate([np.ones((phi_cal.shape[0], 1)), a_cal, phi_cal, a_cal * phi_cal], axis=1)
    residuals = d_aux[cal_idx] - np.maximum(design_cal @ coefficients, 0.0)
    residual_quantile = float(np.quantile(residuals, 0.95))
    return CertificateModel(
        feature_map=basis_map,
        coefficients=coefficients,
        residual_quantile=max(residual_quantile, 0.0),
    )


def _attach_certificate(
    potential: PotentialOutcomeSample,
    *,
    certificate_mode: str,
    certificate_scale: float,
    certificate_model: CertificateModel | None,
) -> tuple[PotentialOutcomeSample, float | None]:
    if certificate_mode == "oracle":
        certificate_pos = np.clip(certificate_scale * potential.certificate_pos, 0.0, 1.0)
        certificate_neg = np.clip(certificate_scale * potential.certificate_neg, 0.0, 1.0)
    elif certificate_mode == "estimated":
        if certificate_model is None:
            raise ValueError("certificate_model is required when certificate_mode='estimated'.")
        a_pos = np.ones(potential.x.shape[0], dtype=float)
        a_neg = -np.ones(potential.x.shape[0], dtype=float)
        certificate_pos = certificate_model.predict(potential.x, a_pos)
        certificate_neg = certificate_model.predict(potential.x, a_neg)
    else:
        raise ValueError(f"Unknown certificate_mode: {certificate_mode}")

    lower_pos = np.clip(potential.reward_pos - certificate_pos, 0.0, 1.0)
    lower_neg = np.clip(potential.reward_neg - certificate_neg, 0.0, 1.0)
    mean_lower_pos = np.clip(potential.mean_reward_pos - certificate_pos, 0.0, 1.0)
    mean_lower_neg = np.clip(potential.mean_reward_neg - certificate_neg, 0.0, 1.0)
    robust_oracle_action = np.where(mean_lower_pos >= mean_lower_neg, 1, -1)
    validity = float(np.mean((potential.bias_pos <= certificate_pos) & (potential.bias_neg <= certificate_neg)))
    attached = PotentialOutcomeSample(
        x=potential.x,
        reward_pos=potential.reward_pos,
        reward_neg=potential.reward_neg,
        reward_star_pos=potential.reward_star_pos,
        reward_star_neg=potential.reward_star_neg,
        mean_reward_pos=potential.mean_reward_pos,
        mean_reward_neg=potential.mean_reward_neg,
        mean_reward_star_pos=potential.mean_reward_star_pos,
        mean_reward_star_neg=potential.mean_reward_star_neg,
        bias_pos=potential.bias_pos,
        bias_neg=potential.bias_neg,
        certificate_pos=certificate_pos,
        certificate_neg=certificate_neg,
        lower_reward_pos=lower_pos,
        lower_reward_neg=lower_neg,
        mean_lower_reward_pos=mean_lower_pos,
        mean_lower_reward_neg=mean_lower_neg,
        oracle_action=potential.oracle_action,
        robust_oracle_action=robust_oracle_action,
        proxy_oracle_action=potential.proxy_oracle_action,
    )
    return attached, validity


def generate_logged_dataset(
    *,
    scenario: ScenarioDefinition,
    n_train: int,
    n_test: int,
    rho: float,
    certificate_mode: str,
    certificate_scale: float,
    rng: np.random.Generator,
    n_certificate_auxiliary: int = 400,
    propensity_mode: str = "constant",
    propensity_strength: float = 1.0,
) -> GeneratedDataset:
    if scenario.p is None:
        raise ValueError(f"Scenario {scenario.scenario_id} is missing its covariate dimension.")
    x_train = _sample_synthetic_covariates_for_scenario(scenario, n_train, rng)
    x_test = _sample_synthetic_covariates_for_scenario(scenario, n_test, rng)

    certificate_model = None
    if certificate_mode == "estimated":
        certificate_model = fit_data_derived_certificate(
            scenario=scenario,
            rho=rho,
            n_auxiliary=n_certificate_auxiliary,
            rng=np.random.default_rng(int(rng.integers(1_000_000_000))),
            propensity_mode=propensity_mode,
            propensity_strength=propensity_strength,
        )

    potential_train, _ = _attach_certificate(
        _generate_potential_outcomes(scenario, x_train, rho=rho, rng=rng),
        certificate_mode=certificate_mode,
        certificate_scale=certificate_scale,
        certificate_model=certificate_model,
    )
    potential_test, validity = _attach_certificate(
        _generate_potential_outcomes(scenario, x_test, rho=rho, rng=rng),
        certificate_mode=certificate_mode,
        certificate_scale=certificate_scale,
        certificate_model=certificate_model,
    )

    p_treat_one = _propensity_treat_one(
        scenario,
        x_train,
        propensity_mode=propensity_mode,
        propensity_strength=propensity_strength,
    )
    choose_pos = rng.uniform(size=n_train) < p_treat_one
    a = np.where(choose_pos, 1, -1).astype(int)
    pi = np.where(choose_pos, p_treat_one, 1.0 - p_treat_one)
    train = ObservedBanditSample(
        x=x_train,
        a=a.astype(float),
        pi=pi,
        reward=np.where(choose_pos, potential_train.reward_pos, potential_train.reward_neg),
        reward_star=np.where(choose_pos, potential_train.reward_star_pos, potential_train.reward_star_neg),
        bias=np.where(choose_pos, potential_train.bias_pos, potential_train.bias_neg),
        certificate=np.where(choose_pos, potential_train.certificate_pos, potential_train.certificate_neg),
        lower_reward=np.where(choose_pos, potential_train.lower_reward_pos, potential_train.lower_reward_neg),
        oracle_action=potential_train.oracle_action,
        robust_oracle_action=potential_train.robust_oracle_action,
        proxy_oracle_action=potential_train.proxy_oracle_action,
    )

    target_oracle_value = float(
        np.mean(np.maximum(potential_test.mean_reward_star_pos, potential_test.mean_reward_star_neg))
    )
    robust_oracle_value = float(
        np.mean(np.maximum(potential_test.mean_lower_reward_pos, potential_test.mean_lower_reward_neg))
    )
    proxy_oracle_value = float(np.mean(np.maximum(potential_test.mean_reward_pos, potential_test.mean_reward_neg)))
    best_constant_target_value = float(
        max(np.mean(potential_test.mean_reward_star_pos), np.mean(potential_test.mean_reward_star_neg))
    )
    best_constant_robust_value = float(
        max(np.mean(potential_test.mean_lower_reward_pos), np.mean(potential_test.mean_lower_reward_neg))
    )
    best_constant_proxy_value = float(
        max(np.mean(potential_test.mean_reward_pos), np.mean(potential_test.mean_reward_neg))
    )

    return GeneratedDataset(
        scenario=scenario,
        rho=float(rho),
        certificate_mode=certificate_mode,
        certificate_scale=float(certificate_scale),
        train=train,
        test=potential_test,
        best_constant_target_value=best_constant_target_value,
        best_constant_robust_value=best_constant_robust_value,
        best_constant_proxy_value=best_constant_proxy_value,
        target_oracle_value=target_oracle_value,
        robust_oracle_value=robust_oracle_value,
        proxy_oracle_value=proxy_oracle_value,
        certificate_model=certificate_model,
        certificate_validity_rate_test=validity,
        propensity_mode=propensity_mode,
        propensity_strength=float(propensity_strength),
    )


def _misclassification_rate(p_treat_one: NDArray, oracle_action: NDArray) -> float:
    """Compute the misclassification rate.

    Args:
        p_treat_one (NDArray): The action probability.
        oracle_action (NDArray): The oracle action.

    Returns:
        float: The misclassification rate.
    """
    p_treat_one = np.asarray(p_treat_one, dtype=float).reshape(-1)
    oracle_action = np.asarray(oracle_action, dtype=float).reshape(-1)
    return float(np.mean(np.where(oracle_action == 1.0, 1.0 - p_treat_one, p_treat_one)))


def evaluate_policy_on_potential_outcomes(
    *,
    sample: PotentialOutcomeSample,
    p_treat_one: Optional[NDArray] = None,
    deterministic_action: Optional[NDArray] = None,
) -> dict[str, float]:
    if p_treat_one is None and deterministic_action is None:
        raise ValueError("Provide either p_treat_one or deterministic_action.")
    if p_treat_one is None:
        action = np.asarray(deterministic_action, dtype=float).reshape(-1)
        p_treat_one = np.where(action == 1.0, 1.0, 0.0)
    else:
        p_treat_one = np.asarray(p_treat_one, dtype=float).reshape(-1)
    p_treat_one = np.clip(p_treat_one, 0.0, 1.0)
    p_treat_neg = 1.0 - p_treat_one

    target_value = float(np.mean(p_treat_one * sample.mean_reward_star_pos + p_treat_neg * sample.mean_reward_star_neg))
    proxy_value = float(np.mean(p_treat_one * sample.mean_reward_pos + p_treat_neg * sample.mean_reward_neg))
    robust_value = float(
        np.mean(p_treat_one * sample.mean_lower_reward_pos + p_treat_neg * sample.mean_lower_reward_neg)
    )
    target_oracle_value = float(np.mean(np.maximum(sample.mean_reward_star_pos, sample.mean_reward_star_neg)))
    proxy_oracle_value = float(np.mean(np.maximum(sample.mean_reward_pos, sample.mean_reward_neg)))
    robust_oracle_value = float(np.mean(np.maximum(sample.mean_lower_reward_pos, sample.mean_lower_reward_neg)))
    target_misclassification = _misclassification_rate(p_treat_one, sample.oracle_action)
    proxy_misclassification = _misclassification_rate(p_treat_one, sample.proxy_oracle_action)
    robust_misclassification = _misclassification_rate(p_treat_one, sample.robust_oracle_action)
    return {
        "target_value": target_value,
        "target_regret": target_oracle_value - target_value,
        "proxy_value": proxy_value,
        "proxy_regret": proxy_oracle_value - proxy_value,
        "robust_value": robust_value,
        "robust_regret": robust_oracle_value - robust_value,
        "target_misclassification": target_misclassification,
        "proxy_misclassification": proxy_misclassification,
        "robust_misclassification": robust_misclassification,
        "target_agreement": 1.0 - target_misclassification,
        "proxy_agreement": 1.0 - proxy_misclassification,
        "robust_agreement": 1.0 - robust_misclassification,
    }
