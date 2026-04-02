from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore
from scipy.special import expit  # type: ignore

BasisFn = Callable[[NDArray], NDArray]
FeatureMapFactory = Callable[[], "StandardizedFeatureMap"]
EPS = 1e-12


def _as_1d_float_array(values: NDArray, n_expected: Optional[int] = None) -> NDArray:
    """Convert a 1D array to a 1D float array.

    Args:
        values (NDArray): The input array.
        n_expected (Optional[int], optional): The expected length of the array. Defaults to None.

    Raises:
        ValueError: If the length of the array is not equal to the expected length.

    Returns:
        NDArray: The 1D float array.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if n_expected is not None and arr.shape[0] != n_expected:
        raise ValueError(f"Expected length {n_expected}, got {arr.shape[0]}.")
    return arr


def _softmax(log_weights: NDArray) -> NDArray:
    """Compute the softmax of a log-weights array.

    Args:
        log_weights (NDArray): The log-weights array.

    Returns:
        NDArray: The softmax of the log-weights array.
    """
    shifted = np.asarray(log_weights, dtype=float) - float(np.max(log_weights))
    weights = np.exp(shifted)
    return weights / np.maximum(np.sum(weights), EPS)


def _maurer_xi(n: int) -> float:
    """Compute the Maurer xi constant.

    Args:
        n (int): The number of observations.

    Returns:
        float: The Maurer xi constant.
    """
    n_eff = max(1, int(n))
    return float(math.exp(1.0 / (12.0 * n_eff)) * math.sqrt(math.pi * n_eff / 2.0) + 2.0)


def _score_from_linear(linear_score: NDArray, score_bound: Optional[float]) -> Tuple[NDArray, NDArray]:
    """Compute the score from a linear score.

    Args:
        linear_score (NDArray): The linear score.
        score_bound (Optional[float]): The score bound.

    Returns:
        Tuple[NDArray, NDArray]: The score and the derivative of the score.
    """
    linear = np.asarray(linear_score, dtype=float)
    if score_bound is None:
        return linear, np.ones_like(linear)
    bound = float(score_bound)
    bounded = bound * np.tanh(linear / max(bound, EPS))
    derivative = 1.0 - np.tanh(linear / max(bound, EPS)) ** 2
    return bounded, derivative


def _solve_weighted_ridge(
    design: NDArray,
    outcome: NDArray,
    weight: NDArray,
    l2_penalty: float,
    *,
    penalize_intercept: bool,
) -> NDArray:
    """Solve a weighted ridge system with a pseudoinverse fallback.

    Args:
        design (NDArray): The design matrix.
        outcome (NDArray): The outcome vector.
        weight (NDArray): The weight vector.
        l2_penalty (float): The L2 penalty.
        penalize_intercept (bool): Whether to penalize the intercept.

    Returns:
        NDArray: The coefficients.
    """
    weighted_design = design * weight[:, None]
    gram = weighted_design.T @ design
    if l2_penalty > 0.0:
        penalty = float(l2_penalty) * np.eye(design.shape[1], dtype=float)
        if not penalize_intercept and design.shape[1] > 0:
            penalty[0, 0] = 0.0
        gram = gram + penalty
    rhs = weighted_design.T @ outcome
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


@dataclass
class StandardizedFeatureMap:
    """Standardize covariates, then apply an optional basis expansion."""

    basis_fn: Optional[BasisFn] = None
    add_intercept: bool = True
    standardize: bool = True
    mean_: Optional[NDArray] = field(default=None, init=False, repr=False)
    scale_: Optional[NDArray] = field(default=None, init=False, repr=False)

    def fit(self, x: NDArray) -> "StandardizedFeatureMap":
        """Fit the feature map to the data.

        Args:
            x (NDArray): The data.

        Returns:
            "StandardizedFeatureMap": The fitted feature map.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array.")
        if self.standardize:
            mean = np.mean(x, axis=0)
            scale = np.std(x, axis=0)
            scale = np.where(scale < 1e-8, 1.0, scale)
        else:
            mean = np.zeros(x.shape[1], dtype=float)
            scale = np.ones(x.shape[1], dtype=float)
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, x: NDArray) -> NDArray:
        """Transform the data using the feature map.

        Args:
            x (NDArray): The data.

        Returns:
            NDArray: The transformed data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Feature map must be fit before calling transform.")
        x = np.asarray(x, dtype=float)
        z = (x - self.mean_) / self.scale_
        base = z if self.basis_fn is None else np.asarray(self.basis_fn(z), dtype=float)
        if base.ndim != 2:
            raise ValueError("basis_fn must return a 2D array.")
        if self.add_intercept:
            return np.concatenate([np.ones((base.shape[0], 1), dtype=float), base], axis=1)
        return base

    def fit_transform(self, x: NDArray) -> NDArray:
        """Fit the feature map to the data and transform the data.

        Args:
            x (NDArray): The data.

        Returns:
            NDArray: The transformed data.
        """
        return self.fit(x).transform(x)

    def clone(self) -> "StandardizedFeatureMap":
        """Clone the feature map.

        Returns:
            "StandardizedFeatureMap": The cloned feature map.
        """
        return copy.deepcopy(self)


@dataclass
class TreatmentFreeBaselineModel:
    """A treatment-free baseline model."""

    feature_map: StandardizedFeatureMap
    coefficients: NDArray

    def predict(self, x: NDArray) -> NDArray:
        """Predict the outcome using the baseline model.

        Args:
            x (NDArray): The data.

        Returns:
            NDArray: The predicted outcome.
        """
        pred = self.feature_map.transform(x) @ self.coefficients
        return np.clip(pred, 0.0, 1.0)

    def clone(self) -> "TreatmentFreeBaselineModel":
        """Clone the baseline model.

        Returns:
            "TreatmentFreeBaselineModel": The cloned baseline model.
        """
        return copy.deepcopy(self)


@dataclass(frozen=True)
class ParticleLibraryConfig:
    n_anchor_particles: int = 16
    n_prior_samples: int = 256
    n_local_samples_per_anchor: int = 48
    local_scale: float = 0.30


def fit_treatment_free_baseline_model(
    x: NDArray,
    y: NDArray,
    propensity: NDArray,
    *,
    feature_map: StandardizedFeatureMap,
    l2_penalty: float = 1e-6,
) -> TreatmentFreeBaselineModel:
    """Fit a treatment-free baseline model.

    Args:
        x (NDArray): The input array.
        y (NDArray): The outcome array.
        propensity (NDArray): The propensity array.
        feature_map (StandardizedFeatureMap): The feature map.
        l2_penalty (float): The L2 penalty.

    Returns:
        TreatmentFreeBaselineModel: The fitted treatment-free baseline model.
    """
    x = np.asarray(x, dtype=float)
    y = _as_1d_float_array(y, n_expected=x.shape[0])
    propensity = _as_1d_float_array(propensity, n_expected=x.shape[0])
    fmap = feature_map.clone()
    design = fmap.fit_transform(x)
    weights = 1.0 / np.maximum(2.0 * propensity, EPS)
    coefficients = _solve_weighted_ridge(
        design=design,
        outcome=y,
        weight=weights,
        l2_penalty=l2_penalty,
        penalize_intercept=False,
    )
    return TreatmentFreeBaselineModel(feature_map=fmap, coefficients=coefficients)


def fit_weighted_hinge_map(
    phi: NDArray,
    treatment: NDArray,
    sample_weight: NDArray,
    l2_penalty: float,
    score_bound: Optional[float],
    smooth_hinge_scale: float = 25.0,
    start: Optional[NDArray] = None,
    maxiter: int = 2000,
) -> NDArray:
    """Fit a weighted hinge map.

    Args:
        phi (NDArray): The feature array.
        treatment (NDArray): The treatment array.
        sample_weight (NDArray): The sample weight array.
        l2_penalty (float): The L2 penalty.
        score_bound (Optional[float]): The score bound.
        smooth_hinge_scale (float): The smooth hinge scale.
        start (Optional[NDArray]): The start array.
        maxiter (int): The maximum number of iterations.

    Returns:
        NDArray: The fitted weighted hinge map.
    """
    treatment = _as_1d_float_array(treatment, n_expected=phi.shape[0])
    sample_weight = np.maximum(_as_1d_float_array(sample_weight, n_expected=phi.shape[0]), 0.0)
    if start is None:
        start = np.zeros(phi.shape[1], dtype=float)
    else:
        start = np.asarray(start, dtype=float)
    scale = float(max(smooth_hinge_scale, 1.0))
    n_obs = max(1, phi.shape[0])

    def objective(beta: NDArray) -> Tuple[float, NDArray]:
        linear = phi @ beta
        score, dscore_dlinear = _score_from_linear(linear, score_bound)
        margin = 1.0 - treatment * score
        soft_hinge = np.logaddexp(0.0, scale * margin) / scale
        sigmoid_margin = expit(scale * margin)
        common = -(sample_weight * sigmoid_margin * treatment * dscore_dlinear) / n_obs
        grad = phi.T @ common + l2_penalty * beta
        value = float(np.mean(sample_weight * soft_hinge) + 0.5 * l2_penalty * np.sum(beta**2))
        return value, grad

    def run_lbfgs(x0: NDArray, iter_budget: int):
        return minimize(
            fun=lambda beta: objective(beta),
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(iter_budget), "maxls": 100},
        )

    result = run_lbfgs(start, maxiter)
    if result.success:
        return np.asarray(result.x, dtype=float)

    message = str(result.message)
    candidate = result
    if "TOTAL NO. OF ITERATIONS REACHED LIMIT" in message:
        retry = run_lbfgs(np.asarray(result.x, dtype=float), maxiter * 4)
        if retry.success:
            return np.asarray(retry.x, dtype=float)
        candidate = retry if np.isfinite(retry.fun) else result
        message = str(candidate.message)

    value, grad = objective(np.asarray(candidate.x, dtype=float))
    grad_inf = float(np.max(np.abs(grad)))
    if np.isfinite(value) and grad_inf <= 1e-3:
        return np.asarray(candidate.x, dtype=float)

    raise RuntimeError(
        "Weighted hinge optimization failed: " f"{message}; objective={value:.6g}; grad_inf={grad_inf:.6g}"
    )


def feature_scores(phi: NDArray, candidates: NDArray, score_bound: Optional[float]) -> NDArray:
    """Compute the feature scores.

    Args:
        phi (NDArray): The feature array.
        candidates (NDArray): The candidates array.
        score_bound (Optional[float]): The score bound.

    Returns:
        NDArray: The feature scores.
    """
    linear = np.asarray(phi, dtype=float) @ np.asarray(candidates, dtype=float).T
    scores, _ = _score_from_linear(linear, score_bound)
    return scores


def empirical_hinge_risk_matrix(scores: NDArray, treatment: NDArray, reward_over_propensity: NDArray) -> NDArray:
    """Compute the empirical hinge risk matrix.

    Args:
        scores (NDArray): The scores array.
        treatment (NDArray): The treatment array.
        reward_over_propensity (NDArray): The reward over propensity array.

    Returns:
        NDArray: The empirical hinge risk matrix.
    """
    margins = 1.0 - np.asarray(treatment, dtype=float)[:, None] * np.asarray(scores, dtype=float)
    losses = np.asarray(reward_over_propensity, dtype=float)[:, None] * np.maximum(margins, 0.0)
    return np.mean(losses, axis=0)


def empirical_exact_value_matrix(scores: NDArray, treatment: NDArray, reward_over_propensity: NDArray) -> NDArray:
    """Compute the empirical exact value matrix.

    Args:
        scores (NDArray): The scores array.
        treatment (NDArray): The treatment array.
        reward_over_propensity (NDArray): The reward over propensity array.

    Returns:
        NDArray: The empirical exact value matrix.
    """
    matches = (np.asarray(treatment, dtype=float)[:, None] * np.asarray(scores, dtype=float) >= 0.0).astype(float)
    return np.mean(np.asarray(reward_over_propensity, dtype=float)[:, None] * matches, axis=0)


def posterior_weights_from_general_bayes(
    *,
    log_prior: NDArray,
    empirical_risks: NDArray,
    eta: float,
    n_obs: int,
) -> NDArray:
    """Compute the posterior weights from general Bayes.

    Args:
        log_prior (NDArray): The log prior array.
        empirical_risks (NDArray): The empirical risks array.
        eta (float): The eta value.
        n_obs (int): The number of observations.

    Returns:
        NDArray: The posterior weights.
    """
    log_weights = np.asarray(log_prior, dtype=float) - float(eta) * max(1, int(n_obs)) * np.asarray(
        empirical_risks,
        dtype=float,
    )
    return _softmax(log_weights)


def kl_categorical(q: NDArray, p: NDArray) -> float:
    """Compute the KL categorical.

    Args:
        q (NDArray): The q array.
        p (NDArray): The p array.

    Returns:
        float: The KL categorical.
    """
    q = np.maximum(np.asarray(q, dtype=float), EPS)
    p = np.maximum(np.asarray(p, dtype=float), EPS)
    return float(np.sum(q * (np.log(q) - np.log(p))))


def empirical_mean_lower_bound(
    *,
    empirical_mean: float,
    value_upper_bound: float,
    n_obs: int,
    delta: float,
) -> float:
    """Compute the empirical mean lower bound.

    Args:
        empirical_mean (float): The empirical mean.
        value_upper_bound (float): The value upper bound.
        n_obs (int): The number of observations.
        delta (float): The delta value.

    Returns:
        float: The empirical mean lower bound.
    """
    return float(
        empirical_mean
        - float(value_upper_bound) * math.sqrt(math.log(1.0 / max(delta, EPS)) / (2.0 * max(1, int(n_obs))))
    )


def bounded_loss_catoni_ucb(
    *,
    empirical_loss: float,
    kl_to_prior: float,
    eta: float,
    n_obs: int,
    delta: float,
) -> float:
    """Compute the bounded loss Catoni UCB.

    Args:
        empirical_loss (float): The empirical loss.
        kl_to_prior (float): The KL to prior.
        eta (float): The eta value.
        n_obs (int): The number of observations.
        delta (float): The delta value.

    Returns:
        float: The bounded loss Catoni UCB.
    """
    empirical_loss = float(np.clip(empirical_loss, 0.0, 1.0))
    c_delta = (kl_to_prior + math.log(_maurer_xi(n_obs) / max(delta, EPS))) / max(1, int(n_obs))
    numerator = -math.expm1(-c_delta - eta * empirical_loss)
    denominator = -math.expm1(-eta)
    if denominator <= EPS:
        return float("inf")
    return float(numerator / denominator)


def surrogate_risk_ucb(
    *,
    posterior_empirical_risk: float,
    kl_to_prior: float,
    eta: float,
    n_obs: int,
    delta: float,
    epsilon: float,
    score_bound: float,
) -> float:
    """Compute the surrogate risk UCB.

    Args:
        posterior_empirical_risk (float): The posterior empirical risk.
        kl_to_prior (float): The KL to prior.
        eta (float): The eta value.
        n_obs (int): The number of observations.
        delta (float): The delta value.
        epsilon (float): The epsilon value.
        score_bound (float): The score bound.

    Returns:
        float: The surrogate risk UCB.
    """
    c_b = (1.0 + score_bound) / epsilon
    c_delta = (kl_to_prior + math.log(_maurer_xi(n_obs) / max(delta, EPS))) / max(1, int(n_obs))
    numerator = -math.expm1(-c_delta - eta * posterior_empirical_risk)
    denominator = -math.expm1(-eta * c_b)
    if denominator <= EPS:
        return float("inf")
    return float(c_b * numerator / denominator)


def surrogate_target_value_lcb(
    *,
    c_hat: float,
    posterior_empirical_risk: float,
    kl_to_prior: float,
    eta: float,
    n_obs: int,
    delta: float,
    epsilon: float,
    score_bound: float,
) -> float:
    """Compute the surrogate target value LCB.

    Args:
        c_hat (float): The c hat value.
        posterior_empirical_risk (float): The posterior empirical risk.
        kl_to_prior (float): The KL to prior.
        eta (float): The eta value.
        n_obs (int): The number of observations.
        delta (float): The delta value.
        epsilon (float): The epsilon value.
        score_bound (float): The score bound.

    Returns:
        float: The surrogate target value LCB.
    """
    constant_lb = empirical_mean_lower_bound(
        empirical_mean=c_hat,
        value_upper_bound=1.0 / epsilon,
        n_obs=n_obs,
        delta=delta / 2.0,
    )
    ucb = surrogate_risk_ucb(
        posterior_empirical_risk=posterior_empirical_risk,
        kl_to_prior=kl_to_prior,
        eta=eta,
        n_obs=n_obs,
        delta=delta / 2.0,
        epsilon=epsilon,
        score_bound=score_bound,
    )
    return float(constant_lb - ucb)


def residualized_surrogate_target_value_lcb(
    *,
    baseline_empirical_mean: float,
    positive_part_empirical_mean: float,
    posterior_empirical_risk: float,
    kl_to_prior: float,
    eta: float,
    n_obs: int,
    delta: float,
    epsilon: float,
    score_bound: float,
) -> float:
    """Compute the residualized surrogate target value LCB.

    Args:
        baseline_empirical_mean (float): The baseline empirical mean.
        positive_part_empirical_mean (float): The positive part empirical mean.
        posterior_empirical_risk (float): The posterior empirical risk.
        kl_to_prior (float): The KL to prior.
        eta (float): The eta value.
        n_obs (int): The number of observations.
        delta (float): The delta value.
        epsilon (float): The epsilon value.
        score_bound (float): The score bound.

    Returns:
        float: The residualized surrogate target value LCB.
    """
    baseline_lb = empirical_mean_lower_bound(
        empirical_mean=baseline_empirical_mean,
        value_upper_bound=1.0,
        n_obs=n_obs,
        delta=delta / 4.0,
    )
    positive_part_lb = empirical_mean_lower_bound(
        empirical_mean=positive_part_empirical_mean,
        value_upper_bound=1.0 / epsilon,
        n_obs=n_obs,
        delta=delta / 4.0,
    )
    ucb = surrogate_risk_ucb(
        posterior_empirical_risk=posterior_empirical_risk,
        kl_to_prior=kl_to_prior,
        eta=eta,
        n_obs=n_obs,
        delta=delta / 2.0,
        epsilon=epsilon,
        score_bound=score_bound,
    )
    return float(baseline_lb + positive_part_lb - ucb)


def exact_value_lcb(
    *,
    posterior_empirical_exact_value: float,
    kl_to_prior: float,
    eta: float,
    n_obs: int,
    delta: float,
    epsilon: float,
) -> float:
    """Compute the exact value LCB.

    Args:
        posterior_empirical_exact_value (float): The posterior empirical exact value.
        kl_to_prior (float): The KL to prior.
        eta (float): The eta value.
        n_obs (int): The number of observations.
        delta (float): The delta value.
        epsilon (float): The epsilon value.

    Returns:
        float: The exact value LCB.
    """
    loss_ucb = bounded_loss_catoni_ucb(
        empirical_loss=1.0 - epsilon * posterior_empirical_exact_value,
        kl_to_prior=kl_to_prior,
        eta=eta,
        n_obs=n_obs,
        delta=delta,
    )
    return float((1.0 - loss_ucb) / epsilon)


def residualized_exact_value_lcb(
    *,
    baseline_empirical_mean: float,
    posterior_empirical_centered_delta: float,
    kl_to_prior: float,
    eta: float,
    n_obs: int,
    delta: float,
    epsilon: float,
) -> float:
    """Compute the residualized exact value LCB.

    Args:
        baseline_empirical_mean (float): The baseline empirical mean.
        posterior_empirical_centered_delta (float): The posterior empirical centered delta.
        kl_to_prior (float): The KL to prior.
        eta (float): The eta value.
        n_obs (int): The number of observations.
        delta (float): The delta value.
        epsilon (float): The epsilon value.

    Returns:
        float: The residualized exact value LCB.
    """
    baseline_lb = empirical_mean_lower_bound(
        empirical_mean=baseline_empirical_mean,
        value_upper_bound=1.0,
        n_obs=n_obs,
        delta=delta / 2.0,
    )
    loss_ucb = bounded_loss_catoni_ucb(
        empirical_loss=0.5 - 0.5 * epsilon * posterior_empirical_centered_delta,
        kl_to_prior=kl_to_prior,
        eta=eta,
        n_obs=n_obs,
        delta=delta / 2.0,
    )
    centered_delta_lb = float((1.0 - 2.0 * loss_ucb) / epsilon)
    return float(baseline_lb + centered_delta_lb)


def _solve_ridge(
    design: NDArray,
    outcome: NDArray,
    *,
    l2_penalty: float,
    penalize_intercept: bool,
) -> NDArray:
    """Solve the ridge regression.

    Args:
        design (NDArray): The design matrix.
        outcome (NDArray): The outcome array.
        l2_penalty (float): The L2 penalty.
        penalize_intercept (bool): Whether to penalize the intercept.

    Returns:
        NDArray: The coefficients.
    """
    gram = design.T @ design
    penalty = float(max(l2_penalty, 0.0)) * np.eye(design.shape[1], dtype=float)
    if not penalize_intercept and penalty.shape[0] > 0:
        penalty[0, 0] = 0.0
    rhs = design.T @ outcome
    try:
        return np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram + penalty) @ rhs


@dataclass
class ConstantRegressor:
    value: float

    def predict(self, x: NDArray) -> NDArray:
        """Predict the outcome using the constant regressor.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The predicted outcome.
        """
        x = np.asarray(x, dtype=float)
        return np.full(x.shape[0], float(self.value), dtype=float)


@dataclass
class FeatureMapRidgeRegressor:
    feature_map: StandardizedFeatureMap
    coefficients: NDArray

    def predict(self, x: NDArray) -> NDArray:
        """Predict the outcome using the feature map ridge regressor.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The predicted outcome.
        """
        return self.feature_map.transform(x) @ self.coefficients


@dataclass
class ArmValueNuisanceModel:
    """Arm-specific regressions for E[underline R | X, A=a]."""

    reg_pos: ConstantRegressor | FeatureMapRidgeRegressor
    reg_neg: ConstantRegressor | FeatureMapRidgeRegressor

    def predict(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        """Predict the outcome using the arm value nuisance model.

        Args:
            x (NDArray): The input array.

        Returns:
            Tuple[NDArray, NDArray]: The predicted outcome.
        """
        x = np.asarray(x, dtype=float)
        nu_pos = np.clip(self.reg_pos.predict(x), 0.0, 1.0)
        nu_neg = np.clip(self.reg_neg.predict(x), 0.0, 1.0)
        return nu_pos, nu_neg

    def clone(self) -> "ArmValueNuisanceModel":
        """Clone the arm value nuisance model.

        Returns:
            "ArmValueNuisanceModel": The cloned arm value nuisance model.
        """
        return copy.deepcopy(self)


def _fit_feature_map_ridge_regressor(
    x: NDArray,
    y: NDArray,
    *,
    feature_map: StandardizedFeatureMap,
    l2_penalty: float,
) -> ConstantRegressor | FeatureMapRidgeRegressor:
    """Fit the feature map ridge regressor.

    Args:
        x (NDArray): The input array.
        y (NDArray): The outcome array.
        feature_map (StandardizedFeatureMap): The feature map.
        l2_penalty (float): The L2 penalty.

    Returns:
        ConstantRegressor | FeatureMapRidgeRegressor: The fitted feature map ridge regressor.
    """
    x = np.asarray(x, dtype=float)
    y = _as_1d_float_array(y, n_expected=x.shape[0])
    if y.size == 0:
        return ConstantRegressor(0.0)
    if y.size == 1:
        return ConstantRegressor(float(y[0]))
    fmap = feature_map.clone()
    design = fmap.fit_transform(x)
    coefficients = _solve_ridge(
        design=design,
        outcome=y,
        l2_penalty=l2_penalty,
        penalize_intercept=False,
    )
    return FeatureMapRidgeRegressor(feature_map=fmap, coefficients=coefficients)


def fit_arm_value_nuisance_model(
    x: NDArray,
    treatment: NDArray,
    lower_reward: NDArray,
    *,
    feature_map_factory: Optional[FeatureMapFactory] = None,
    l2_penalty: float = 1e-6,
) -> ArmValueNuisanceModel:
    """Fit separate nuisance regressions for the two treatment arms.

    In the split-free PROWL workflow this is fit on the same policy-learning
    sample used to optimize the policy. An external auxiliary sample can still
    be supplied when a deliberately honest nuisance fit is desired.

    Args:
        x (NDArray): The input array.
        treatment (NDArray): The treatment array.
        lower_reward (NDArray): The lower reward array.
        feature_map_factory (Optional[FeatureMapFactory]): The feature map factory.
        l2_penalty (float): The L2 penalty.

    Returns:
        ArmValueNuisanceModel: The fitted arm value nuisance model.
    """

    x = np.asarray(x, dtype=float)
    treatment = _as_1d_float_array(treatment, n_expected=x.shape[0])
    lower_reward = _as_1d_float_array(lower_reward, n_expected=x.shape[0])
    if np.any(~np.isin(treatment, [-1.0, 1.0])):
        raise ValueError("Treatment must take values in {-1, 1}.")
    factory = feature_map_factory or StandardizedFeatureMap
    reg_pos = _fit_feature_map_ridge_regressor(
        x=x[treatment == 1.0],
        y=lower_reward[treatment == 1.0],
        feature_map=factory(),
        l2_penalty=l2_penalty,
    )
    reg_neg = _fit_feature_map_ridge_regressor(
        x=x[treatment == -1.0],
        y=lower_reward[treatment == -1.0],
        feature_map=factory(),
        l2_penalty=l2_penalty,
    )
    return ArmValueNuisanceModel(reg_pos=reg_pos, reg_neg=reg_neg)


def fit_treatment_free_nuisance_model(
    x: NDArray,
    lower_reward: NDArray,
    propensity: NDArray,
    *,
    feature_map_factory: Optional[FeatureMapFactory] = None,
    l2_penalty: float = 1e-6,
) -> ArmValueNuisanceModel:
    """Fit a shared treatment-free nuisance model for both treatment arms.

    Args:
        x (NDArray): The input array.
        lower_reward (NDArray): The lower reward array.
        propensity (NDArray): The propensity array.
        feature_map_factory (Optional[FeatureMapFactory]): The feature map factory.
        l2_penalty (float): The L2 penalty.

    Returns:
        ArmValueNuisanceModel: The fitted arm value nuisance model.
    """
    x = np.asarray(x, dtype=float)
    lower_reward = _as_1d_float_array(lower_reward, n_expected=x.shape[0])
    propensity = _as_1d_float_array(propensity, n_expected=x.shape[0])
    factory = feature_map_factory or StandardizedFeatureMap
    reg = fit_treatment_free_baseline_model(
        x=x,
        y=lower_reward,
        propensity=propensity,
        feature_map=factory(),
        l2_penalty=l2_penalty,
    )
    return ArmValueNuisanceModel(reg_pos=copy.deepcopy(reg), reg_neg=copy.deepcopy(reg))


@dataclass(frozen=True)
class PROWLObjectiveTerms:
    gamma_pos: NDArray
    gamma_neg: NDArray
    signed_label: NDArray
    abs_weight: NDArray
    advantage: NDArray


@dataclass(frozen=True)
class PROWLDiagnostic:
    eta: float
    gamma: float
    posterior_empirical_hinge_risk: float
    posterior_empirical_exact_value: float
    exact_value_lcb: float
    kl_to_prior: float


@dataclass
class PROWLResult:
    method_label: str
    selection_mode: str
    eta: float
    gamma: float
    epsilon: float
    score_bound: float
    feature_map: StandardizedFeatureMap
    nuisance_model: ArmValueNuisanceModel
    candidates: NDArray
    prior_weights: NDArray
    posterior_weights: NDArray
    posterior_weights_by_eta: Dict[float, NDArray]
    empirical_hinge_risks: NDArray
    empirical_exact_values: NDArray
    diagnostics: List[PROWLDiagnostic]

    def candidate_scores(self, x: NDArray) -> NDArray:
        phi = self.feature_map.transform(x)
        scores, _ = _score_from_linear(phi @ self.candidates.T, self.score_bound)
        return scores

    def posterior_mean_score(self, x: NDArray) -> NDArray:
        return self.candidate_scores(x) @ self.posterior_weights

    def action_probability(self, x: NDArray) -> NDArray:
        scores = self.candidate_scores(x)
        p_treat_one = (scores >= 0.0).astype(float) @ self.posterior_weights
        return np.clip(p_treat_one, 0.0, 1.0)

    def deterministic_action(self, x: NDArray) -> NDArray:
        return np.where(self.posterior_mean_score(x) >= 0.0, 1, -1)

    def result_for_eta(self, eta: float) -> "PROWLResult":
        eta_value = float(eta)
        if eta_value not in self.posterior_weights_by_eta:
            raise KeyError(f"eta={eta_value} is not available in posterior_weights_by_eta.")
        diag = next((item for item in self.diagnostics if abs(item.eta - eta_value) <= 1e-12), None)
        if diag is None:
            raise KeyError(f"eta={eta_value} is not available in diagnostics.")
        result = copy.deepcopy(self)
        result.eta = diag.eta
        result.gamma = diag.gamma
        result.posterior_weights = np.asarray(self.posterior_weights_by_eta[diag.eta], dtype=float).copy()
        return result


def empirical_prowl_exact_value_matrix(scores: NDArray, gamma_pos: NDArray, gamma_neg: NDArray) -> NDArray:
    """Compute the empirical PROWL exact value matrix.

    Args:
        scores (NDArray): The scores array.
        gamma_pos (NDArray): The gamma positive array.
        gamma_neg (NDArray): The gamma negative array.

    Returns:
        NDArray: The empirical PROWL exact value matrix.
    """
    return np.mean(np.where(scores >= 0.0, gamma_pos[:, None], gamma_neg[:, None]), axis=0)


def prowl_exact_value_lcb(
    *,
    posterior_empirical_exact_value: float,
    kl_to_prior: float,
    gamma: float,
    n_obs: int,
    delta: float,
    epsilon: float,
) -> float:
    """Compute the PROWL exact value LCB.

    Args:
        posterior_empirical_exact_value (float): The posterior empirical exact value.
        kl_to_prior (float): The KL to prior.
        gamma (float): The gamma value.
        n_obs (int): The number of observations.
        delta (float): The delta value.
        epsilon (float): The epsilon value.

    Returns:
        float: The PROWL exact value LCB.
    """
    c_eps = 1.0 / max(float(epsilon), EPS)
    k_eps = 2.0 * c_eps - 1.0
    loss_hat = (c_eps - float(posterior_empirical_exact_value)) / max(k_eps, EPS)
    loss_hat = float(np.clip(loss_hat, 0.0, 1.0))
    c_delta = (float(kl_to_prior) + math.log(_maurer_xi(n_obs) / max(float(delta), EPS))) / max(1, int(n_obs))
    numerator = -math.expm1(-c_delta - float(gamma) * loss_hat)
    denominator = -math.expm1(-float(gamma))
    if denominator <= EPS:
        return float("-inf")
    return float(c_eps - k_eps * numerator / denominator)


class PROWL:
    """Practical nuisance-adjusted PAC-Bayes OWL with exact-value tuning.

    Args:
        eta (float | None): The eta value.
        eta_grid (Sequence[float] | None): The eta grid.
        gamma (float | None): The gamma value.
        gamma_grid (Sequence[float] | None): The gamma grid.
        selection_mode (str): The selection mode.
        delta (float): The delta value.
        prior_sd (float): The prior standard deviation.
        score_bound (float): The score bound.
        feature_map (StandardizedFeatureMap | None): The feature map.
        nuisance_feature_map_factory (Optional[FeatureMapFactory]): The nuisance feature map factory.
        nuisance_l2_penalty (float): The nuisance L2 penalty.
        library_penalty_grid (Sequence[float] | None): The library penalty grid.
        include_treatment_free_anchors (bool): Whether to include treatment-free anchors.
        include_policy_anchors (bool): Whether to include policy anchors.
        particle_library_config (ParticleLibraryConfig | None): The particle library config.
        random_state (int | None): The random state.
    """

    def __init__(
        self,
        *,
        eta: Optional[float] = None,
        eta_grid: Sequence[float] | None = None,
        gamma: Optional[float] = None,
        gamma_grid: Optional[Sequence[float]] = None,
        selection_mode: str = "lcb",
        delta: float = 0.1,
        prior_sd: float = 5.0,
        score_bound: float = 3.0,
        feature_map: Optional[StandardizedFeatureMap] = None,
        nuisance_feature_map_factory: Optional[FeatureMapFactory] = None,
        nuisance_l2_penalty: float = 1e-6,
        library_penalty_grid: Optional[Sequence[float]] = None,
        include_treatment_free_anchors: bool = True,
        include_policy_anchors: bool = True,
        particle_library_config: Optional[ParticleLibraryConfig] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.eta = None if eta is None else float(eta)
        self.eta_grid = tuple(sorted({float(v) for v in (eta_grid or []) if float(v) > 0.0}))
        self.gamma = None if gamma is None else float(gamma)
        self.gamma_grid = tuple(sorted({float(v) for v in (gamma_grid or []) if float(v) > 0.0}))
        self.selection_mode = str(selection_mode)
        self.delta = float(delta)
        self.prior_sd = float(prior_sd)
        self.score_bound = float(score_bound)
        if self.score_bound <= 0.0:
            raise ValueError("score_bound must be positive for bounded-score PROWL.")
        self.feature_map = feature_map or StandardizedFeatureMap()
        self.nuisance_feature_map_factory = nuisance_feature_map_factory or StandardizedFeatureMap
        self.nuisance_l2_penalty = float(nuisance_l2_penalty)
        self.library_penalty_grid = tuple(
            sorted({float(v) for v in (library_penalty_grid or (1e-4, 1e-3, 1e-2, 1e-1, 1.0)) if float(v) >= 0.0})
        )
        self.include_treatment_free_anchors = bool(include_treatment_free_anchors)
        self.include_policy_anchors = bool(include_policy_anchors)
        self.particle_library_config = particle_library_config or ParticleLibraryConfig()
        self.random_state = random_state
        self.result_: PROWLResult | None = None

    def fit(
        self,
        x: NDArray,
        treatment: NDArray,
        reward: NDArray,
        propensity: NDArray | float,
        *,
        certificate_values: Optional[NDArray] = None,
        lower_reward: Optional[NDArray] = None,
        epsilon: Optional[float] = None,
        nuisance_model: Optional[ArmValueNuisanceModel] = None,
        auxiliary_x: Optional[NDArray] = None,
        auxiliary_treatment: Optional[NDArray] = None,
        auxiliary_lower_reward: Optional[NDArray] = None,
        auxiliary_propensity: Optional[NDArray | float] = None,
        candidate_betas: Optional[NDArray] = None,
        method_label: str = "PROWL",
    ) -> PROWLResult:
        """Fit the PROWL.

        Args:
            x (NDArray): The input array.
            treatment (NDArray): The treatment array.
            reward (NDArray): The reward array.
            propensity (NDArray | float): The propensity array.
            certificate_values (Optional[NDArray]): The certificate values.
            lower_reward (Optional[NDArray]): The lower reward array.
            epsilon (Optional[float]): The epsilon value.
            nuisance_model (Optional[ArmValueNuisanceModel]): The nuisance model.
            auxiliary_x (Optional[NDArray]): The auxiliary x array.
            auxiliary_treatment (Optional[NDArray]): The auxiliary treatment array.
            auxiliary_lower_reward (Optional[NDArray]): The auxiliary lower reward array.
            auxiliary_propensity (Optional[NDArray | float]): The auxiliary propensity array.
            candidate_betas (Optional[NDArray]): The candidate betas.
            method_label (str): The method label.

        Returns:
            PROWLResult: The PROWL result.
        """
        x = np.asarray(x, dtype=float)
        treatment = _as_1d_float_array(treatment, n_expected=x.shape[0])
        reward = _as_1d_float_array(reward, n_expected=x.shape[0])
        if np.any(~np.isin(treatment, [-1.0, 1.0])):
            raise ValueError("Treatment must take values in {-1, 1}.")
        propensity_arr = self._normalize_propensity(propensity, n=x.shape[0])
        if lower_reward is None:
            if certificate_values is None:
                raise ValueError("Provide either lower_reward or certificate_values.")
            certificate_values = _as_1d_float_array(certificate_values, n_expected=x.shape[0])
            lower_reward_arr = np.clip(reward - certificate_values, 0.0, 1.0)
        else:
            lower_reward_arr = np.clip(_as_1d_float_array(lower_reward, n_expected=x.shape[0]), 0.0, 1.0)
        epsilon_val = float(np.min(propensity_arr) if epsilon is None else epsilon)
        if epsilon_val <= 0.0:
            raise ValueError("epsilon must be positive.")

        has_external_auxiliary = (
            auxiliary_x is not None and auxiliary_treatment is not None and auxiliary_lower_reward is not None
        )
        if nuisance_model is None:
            nuisance_model = fit_arm_value_nuisance_model(
                x=np.asarray(auxiliary_x, dtype=float) if has_external_auxiliary else x,
                treatment=np.asarray(auxiliary_treatment, dtype=float) if has_external_auxiliary else treatment,
                lower_reward=(
                    np.asarray(auxiliary_lower_reward, dtype=float) if has_external_auxiliary else lower_reward_arr
                ),
                feature_map_factory=self.nuisance_feature_map_factory,
                l2_penalty=self.nuisance_l2_penalty,
            )
        else:
            nuisance_model = nuisance_model.clone()

        eta_grid = self._resolve_eta_grid()
        gamma_grid = self._resolve_gamma_grid(eta_grid)
        objective_terms = self._prepare_objective_terms(
            x=x,
            treatment=treatment,
            lower_reward=lower_reward_arr,
            propensity=propensity_arr,
            nuisance_model=nuisance_model,
        )
        auxiliary_objective_terms = objective_terms
        auxiliary_propensity_arr = propensity_arr
        feature_fit_x = x
        if has_external_auxiliary:
            auxiliary_x = np.asarray(auxiliary_x, dtype=float)
            auxiliary_propensity_arr = (
                np.full(auxiliary_x.shape[0], epsilon_val, dtype=float)
                if auxiliary_propensity is None
                else self._normalize_propensity(auxiliary_propensity, n=auxiliary_x.shape[0])
            )
            auxiliary_objective_terms = self._prepare_objective_terms(
                x=auxiliary_x,
                treatment=np.asarray(auxiliary_treatment, dtype=float),
                lower_reward=np.asarray(auxiliary_lower_reward, dtype=float),
                propensity=auxiliary_propensity_arr,
                nuisance_model=nuisance_model,
            )
            feature_fit_x = auxiliary_x
        else:
            auxiliary_x = x
            auxiliary_treatment = treatment
            auxiliary_lower_reward = lower_reward_arr
        fit_state = self._fit_once(
            x=x,
            eta_grid=eta_grid,
            gamma_grid=gamma_grid,
            epsilon=epsilon_val,
            feature_fit_x=feature_fit_x,
            objective_terms=objective_terms,
            auxiliary_x=auxiliary_x,
            auxiliary_treatment=None if auxiliary_treatment is None else np.asarray(auxiliary_treatment, dtype=float),
            auxiliary_lower_reward=(
                None if auxiliary_lower_reward is None else np.asarray(auxiliary_lower_reward, dtype=float)
            ),
            auxiliary_propensity=None if auxiliary_propensity is None else auxiliary_propensity_arr,
            auxiliary_objective_terms=auxiliary_objective_terms,
            candidate_betas=candidate_betas,
            rng=np.random.default_rng(self.random_state),
        )

        selected = self._select_diagnostic(fit_state["diagnostics"])
        selected_idx = fit_state["eta_to_index"][selected.eta]
        self.result_ = PROWLResult(
            method_label=method_label,
            selection_mode=self.selection_mode,
            eta=selected.eta,
            gamma=selected.gamma,
            epsilon=epsilon_val,
            score_bound=self.score_bound,
            feature_map=fit_state["feature_map"],
            nuisance_model=nuisance_model,
            candidates=fit_state["candidates"],
            prior_weights=fit_state["prior_weights"],
            posterior_weights=fit_state["posterior_weights_by_eta"][selected_idx],
            posterior_weights_by_eta={
                diagnostic.eta: fit_state["posterior_weights_by_eta"][fit_state["eta_to_index"][diagnostic.eta]]
                for diagnostic in fit_state["diagnostics"]
            },
            empirical_hinge_risks=fit_state["empirical_hinge_risks"],
            empirical_exact_values=fit_state["empirical_exact_values"],
            diagnostics=list(fit_state["diagnostics"]),
        )
        return self.result_

    def _normalize_propensity(self, propensity: NDArray | float, n: int) -> NDArray:
        if np.isscalar(propensity):
            return np.full(n, float(propensity), dtype=float)
        return _as_1d_float_array(propensity, n_expected=n)

    def _resolve_eta_grid(self) -> List[float]:
        if self.selection_mode == "fixed":
            if self.eta is None or self.eta <= 0.0:
                raise ValueError("A positive eta is required when selection_mode='fixed'.")
            return [self.eta]
        if not self.eta_grid:
            raise ValueError("A positive eta_grid is required for adaptive eta selection.")
        return list(self.eta_grid)

    def _resolve_gamma_grid(self, eta_grid: Sequence[float]) -> List[float]:
        if self.selection_mode == "fixed" and self.gamma is not None:
            return [self.gamma]
        if self.gamma_grid:
            return list(self.gamma_grid)
        return [float(value) for value in eta_grid]

    def _prepare_objective_terms(
        self,
        *,
        x: NDArray,
        treatment: NDArray,
        lower_reward: NDArray,
        propensity: NDArray,
        nuisance_model: ArmValueNuisanceModel,
    ) -> PROWLObjectiveTerms:
        nu_pos, nu_neg = nuisance_model.predict(x)
        propensity = np.maximum(_as_1d_float_array(propensity, n_expected=x.shape[0]), EPS)
        treat_pos = (treatment == 1.0).astype(float)
        treat_neg = (treatment == -1.0).astype(float)
        gamma_pos = nu_pos + treat_pos * (lower_reward - nu_pos) / propensity
        gamma_neg = nu_neg + treat_neg * (lower_reward - nu_neg) / propensity
        advantage = gamma_pos - gamma_neg
        signed_label = np.where(advantage >= 0.0, 1.0, -1.0)
        abs_weight = np.abs(advantage)
        return PROWLObjectiveTerms(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            signed_label=signed_label,
            abs_weight=abs_weight,
            advantage=advantage,
        )

    def _build_candidate_library(
        self,
        dim: int,
        rng: np.random.Generator,
        anchor_betas: NDArray | None = None,
    ) -> NDArray:
        cfg = self.particle_library_config
        anchor_particles = [np.zeros(dim, dtype=float)]
        if anchor_betas is not None and np.asarray(anchor_betas).size > 0:
            anchor_particles.extend(np.asarray(anchor_betas, dtype=float))
        if cfg.n_anchor_particles > 0:
            anchor_particles.extend(
                np.asarray(rng.normal(0.0, self.prior_sd, size=(cfg.n_anchor_particles, dim)), dtype=float)
            )
        particles = [np.vstack(anchor_particles)]
        for anchor in anchor_particles:
            if cfg.n_local_samples_per_anchor <= 0:
                continue
            noise = rng.normal(
                loc=0.0,
                scale=cfg.local_scale,
                size=(cfg.n_local_samples_per_anchor, dim),
            )
            particles.append(anchor[None, :] + noise)
        if cfg.n_prior_samples > 0:
            particles.append(np.asarray(rng.normal(0.0, self.prior_sd, size=(cfg.n_prior_samples, dim)), dtype=float))
        return np.vstack(particles)

    def _auxiliary_anchor_candidates(
        self,
        *,
        phi_aux: NDArray,
        objective_terms: PROWLObjectiveTerms,
    ) -> NDArray:
        anchors: list[NDArray] = []
        target_direction = phi_aux.T @ objective_terms.advantage / max(1, phi_aux.shape[0])
        if np.any(np.isfinite(target_direction)):
            anchors.append(np.asarray(target_direction, dtype=float))
        for penalty in self.library_penalty_grid:
            beta = fit_weighted_hinge_map(
                phi=phi_aux,
                treatment=objective_terms.signed_label,
                sample_weight=objective_terms.abs_weight,
                l2_penalty=float(penalty),
                score_bound=self.score_bound,
            )
            anchors.append(np.asarray(beta, dtype=float))
        if not anchors:
            return np.empty((0, phi_aux.shape[1]), dtype=float)
        return np.vstack(anchors)

    def _treatment_free_anchor_candidates(
        self,
        *,
        auxiliary_x: NDArray,
        auxiliary_treatment: NDArray,
        auxiliary_lower_reward: NDArray,
        auxiliary_propensity: NDArray,
        phi_aux: NDArray,
    ) -> NDArray:
        nuisance = _fit_feature_map_ridge_regressor(
            x=auxiliary_x,
            y=auxiliary_lower_reward,
            feature_map=self.nuisance_feature_map_factory(),
            l2_penalty=self.nuisance_l2_penalty,
        )
        centered = np.asarray(auxiliary_lower_reward, dtype=float) - np.clip(nuisance.predict(auxiliary_x), 0.0, 1.0)
        signed_label = np.asarray(auxiliary_treatment, dtype=float) * np.where(centered >= 0.0, 1.0, -1.0)
        abs_weight = np.abs(centered) / np.maximum(np.asarray(auxiliary_propensity, dtype=float), EPS)
        signed_advantage = (
            np.asarray(auxiliary_treatment, dtype=float)
            * centered
            / np.maximum(
                np.asarray(auxiliary_propensity, dtype=float),
                EPS,
            )
        )
        anchors: list[NDArray] = []
        target_direction = phi_aux.T @ signed_advantage / max(1, phi_aux.shape[0])
        if np.any(np.isfinite(target_direction)):
            anchors.append(np.asarray(target_direction, dtype=float))
        for penalty in self.library_penalty_grid:
            beta = fit_weighted_hinge_map(
                phi=phi_aux,
                treatment=signed_label,
                sample_weight=abs_weight,
                l2_penalty=float(penalty),
                score_bound=self.score_bound,
            )
            anchors.append(np.asarray(beta, dtype=float))
        if not anchors:
            return np.empty((0, phi_aux.shape[1]), dtype=float)
        return np.vstack(anchors)

    def _policy_anchor_candidates(
        self,
        *,
        auxiliary_x: NDArray,
        auxiliary_treatment: NDArray,
        auxiliary_lower_reward: NDArray,
        auxiliary_propensity: NDArray,
    ) -> NDArray:
        from .q_learning import fit_linear_q_learning
        from .rwl import fit_linear_residual_weighted_learning

        anchors: list[NDArray] = []
        for penalty in self.library_penalty_grid:
            q_policy = fit_linear_q_learning(
                x=auxiliary_x,
                a=auxiliary_treatment,
                y=auxiliary_lower_reward,
                main_feature_map=self.feature_map.clone(),
                blip_feature_map=self.feature_map.clone(),
                l2_penalty=float(penalty),
            )
            anchors.append(np.asarray(q_policy.psi, dtype=float))
            rwl_policy = fit_linear_residual_weighted_learning(
                x=auxiliary_x,
                a=auxiliary_treatment,
                y=auxiliary_lower_reward,
                propensity=auxiliary_propensity,
                feature_map=self.feature_map.clone(),
                residual_feature_map=self.nuisance_feature_map_factory(),
                l2_penalty=float(penalty),
                residual_regression_penalty=self.nuisance_l2_penalty,
            )
            anchors.append(np.asarray(rwl_policy.beta, dtype=float))
        if not anchors:
            return np.empty((0, 0), dtype=float)
        return np.vstack(anchors)

    def _prior_weights(self, candidates: NDArray) -> Tuple[NDArray, NDArray]:
        log_prior = -0.5 * np.sum(np.asarray(candidates, dtype=float) ** 2, axis=1) / max(self.prior_sd**2, EPS)
        return log_prior, _softmax(log_prior)

    def _fit_once(
        self,
        *,
        x: NDArray,
        eta_grid: Sequence[float],
        gamma_grid: Sequence[float],
        epsilon: float,
        feature_fit_x: NDArray,
        objective_terms: PROWLObjectiveTerms,
        auxiliary_x: NDArray | None,
        auxiliary_treatment: NDArray | None,
        auxiliary_lower_reward: NDArray | None,
        auxiliary_propensity: NDArray | None,
        auxiliary_objective_terms: PROWLObjectiveTerms | None,
        candidate_betas: NDArray | None,
        rng: np.random.Generator,
    ) -> Dict[str, object]:
        feature_map = self.feature_map.clone()
        feature_map.fit(feature_fit_x)
        phi = feature_map.transform(x)
        anchor_blocks: List[NDArray] = []
        if auxiliary_x is not None and auxiliary_objective_terms is not None:
            phi_aux = feature_map.transform(auxiliary_x)
            anchor_blocks.append(
                self._auxiliary_anchor_candidates(
                    phi_aux=phi_aux,
                    objective_terms=auxiliary_objective_terms,
                )
            )
            if (
                self.include_treatment_free_anchors
                and auxiliary_treatment is not None
                and auxiliary_lower_reward is not None
                and auxiliary_propensity is not None
            ):
                anchor_blocks.append(
                    self._treatment_free_anchor_candidates(
                        auxiliary_x=auxiliary_x,
                        auxiliary_treatment=auxiliary_treatment,
                        auxiliary_lower_reward=auxiliary_lower_reward,
                        auxiliary_propensity=auxiliary_propensity,
                        phi_aux=phi_aux,
                    )
                )
            if (
                self.include_policy_anchors
                and auxiliary_treatment is not None
                and auxiliary_lower_reward is not None
                and auxiliary_propensity is not None
            ):
                anchor_blocks.append(
                    self._policy_anchor_candidates(
                        auxiliary_x=auxiliary_x,
                        auxiliary_treatment=auxiliary_treatment,
                        auxiliary_lower_reward=auxiliary_lower_reward,
                        auxiliary_propensity=auxiliary_propensity,
                    )
                )
        anchor_betas = None
        if anchor_blocks:
            nonempty = [block for block in anchor_blocks if np.asarray(block).size > 0]
            if nonempty:
                anchor_betas = np.unique(np.vstack(nonempty), axis=0)
        candidates = (
            np.asarray(candidate_betas, dtype=float)
            if candidate_betas is not None
            else self._build_candidate_library(phi.shape[1], rng=rng, anchor_betas=anchor_betas)
        )
        log_prior, prior_weights = self._prior_weights(candidates)
        score_matrix = feature_scores(phi=phi, candidates=candidates, score_bound=self.score_bound)
        hinge_risks = empirical_hinge_risk_matrix(
            scores=score_matrix,
            treatment=objective_terms.signed_label,
            reward_over_propensity=objective_terms.abs_weight,
        )
        exact_values = empirical_prowl_exact_value_matrix(
            scores=score_matrix,
            gamma_pos=objective_terms.gamma_pos,
            gamma_neg=objective_terms.gamma_neg,
        )
        diagnostics: List[PROWLDiagnostic] = []
        posterior_weights_by_eta: List[NDArray] = []
        eta_to_index: Dict[float, int] = {}

        for idx, eta in enumerate(float(value) for value in eta_grid):
            q = posterior_weights_from_general_bayes(
                log_prior=log_prior,
                empirical_risks=hinge_risks,
                eta=eta,
                n_obs=phi.shape[0],
            )
            posterior_weights_by_eta.append(q)
            eta_to_index[eta] = idx
            posterior_hinge = float(np.dot(q, hinge_risks))
            posterior_exact = float(np.dot(q, exact_values))
            kl_value = kl_categorical(q=q, p=prior_weights)
            best_gamma = None
            best_lcb = -float("inf")
            for gamma in gamma_grid:
                lcb = prowl_exact_value_lcb(
                    posterior_empirical_exact_value=posterior_exact,
                    kl_to_prior=kl_value,
                    gamma=float(gamma),
                    n_obs=phi.shape[0],
                    delta=self.delta,
                    epsilon=epsilon,
                )
                if lcb > best_lcb:
                    best_lcb = lcb
                    best_gamma = float(gamma)
            if best_gamma is None:
                raise RuntimeError("gamma_grid must contain at least one positive candidate.")
            diagnostics.append(
                PROWLDiagnostic(
                    eta=eta,
                    gamma=best_gamma,
                    posterior_empirical_hinge_risk=posterior_hinge,
                    posterior_empirical_exact_value=posterior_exact,
                    exact_value_lcb=best_lcb,
                    kl_to_prior=kl_value,
                )
            )

        return {
            "feature_map": feature_map,
            "candidates": candidates,
            "prior_weights": prior_weights,
            "posterior_weights_by_eta": posterior_weights_by_eta,
            "empirical_hinge_risks": hinge_risks,
            "empirical_exact_values": exact_values,
            "diagnostics": diagnostics,
            "eta_to_index": eta_to_index,
        }

    def _select_diagnostic(self, diagnostics: Sequence[PROWLDiagnostic]) -> PROWLDiagnostic:
        if self.selection_mode == "fixed":
            return diagnostics[0]
        if self.selection_mode == "lcb":
            return max(diagnostics, key=lambda item: item.exact_value_lcb)
        raise ValueError(f"Unknown selection_mode: {self.selection_mode}")


__all__ = [
    "ParticleLibraryConfig",
    "ArmValueNuisanceModel",
    "PROWL",
    "PROWLDiagnostic",
    "PROWLObjectiveTerms",
    "PROWLResult",
    "StandardizedFeatureMap",
    "TreatmentFreeBaselineModel",
    "empirical_exact_value_matrix",
    "empirical_prowl_exact_value_matrix",
    "exact_value_lcb",
    "fit_treatment_free_baseline_model",
    "fit_weighted_hinge_map",
    "feature_scores",
    "fit_arm_value_nuisance_model",
    "fit_treatment_free_nuisance_model",
    "kl_categorical",
    "prowl_exact_value_lcb",
]
