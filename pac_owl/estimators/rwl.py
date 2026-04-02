from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore
from sklearn.model_selection import KFold  # type: ignore

from .prowl import StandardizedFeatureMap

FeatureMapFactory = Callable[[], StandardizedFeatureMap]
EPS = 1e-12


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


def _phi_s(u: NDArray, s: float) -> NDArray:
    """Smoothed ramp loss T(u) used by linear RWL

    Args:
        u (NDArray): The input array.
        s (float): The slope parameter.

    Returns:
        NDArray: The smoothed ramp loss.
    """
    out = np.zeros_like(u, dtype=float)
    mask_linear = u < (s - 1.0)
    mask_quadratic = (~mask_linear) & (u < s)
    out[mask_linear] = 2.0 * s - 2.0 * u[mask_linear] - 1.0
    out[mask_quadratic] = (s - u[mask_quadratic]) ** 2
    return out


def _phi_s_prime(u: NDArray, s: float) -> NDArray:
    """Derivative of smoothed ramp loss T(u) used by linear RWL

    Args:
        u (NDArray): The input array.
        s (float): The slope parameter.

    Returns:
        NDArray: The derivative of the smoothed ramp loss.
    """
    out = np.zeros_like(u, dtype=float)
    mask_linear = u < (s - 1.0)
    mask_quadratic = (~mask_linear) & (u < s)
    out[mask_linear] = -2.0
    out[mask_quadratic] = 2.0 * (u[mask_quadratic] - s)
    return out


def smoothed_ramp_loss(u: NDArray) -> NDArray:
    """Smoothed ramp loss T(u) used by linear RWL

    Args:
        u (NDArray): The input array.

    Returns:
        NDArray: The smoothed ramp loss.
    """
    u = np.asarray(u, dtype=float)
    return _phi_s(u, 1.0) - _phi_s(u, 0.0)


@dataclass
class WeightedMainEffectResidualModel:
    """Treatment-free weighted main-effects regression for RWL residuals."""

    feature_map: StandardizedFeatureMap
    coefficients: NDArray

    def predict(self, x: NDArray) -> NDArray:
        """Predict the outcome using the weighted main-effects residual model.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The predicted outcome.
        """
        return self.feature_map.transform(x) @ self.coefficients

    def residuals(self, x: NDArray, y: NDArray) -> NDArray:
        """Compute the residuals using the weighted main-effects residual model.

        Args:
            x (NDArray): The input array.
            y (NDArray): The outcome array.

        Returns:
            NDArray: The residuals.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        return y - self.predict(x)

    def clone(self) -> "WeightedMainEffectResidualModel":
        """Clone the weighted main-effects residual model.

        Returns:
            "WeightedMainEffectResidualModel": The cloned weighted main-effects residual model.
        """
        return copy.deepcopy(self)


@dataclass
class LinearResidualWeightedLearningPolicy:
    """Linear RWL policy with a treatment-free residual model."""

    feature_map: StandardizedFeatureMap
    beta: NDArray
    residual_model: WeightedMainEffectResidualModel
    l2_penalty: float
    converged: bool
    n_dc_iterations: int

    def decision_function(self, x: NDArray) -> NDArray:
        """Compute the decision function for the linear RWL policy.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The decision function.
        """
        return self.feature_map.transform(x) @ self.beta

    def action_probability(self, x: NDArray) -> NDArray:
        """Compute the action probability for the linear RWL policy.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The action probability.
        """
        return (self.decision_function(x) >= 0.0).astype(float)

    def estimated_residuals(self, x: NDArray, y: NDArray) -> NDArray:
        """Compute the estimated residuals for the linear RWL policy.

        Args:
            x (NDArray): The input array.
            y (NDArray): The outcome array.

        Returns:
            NDArray: The estimated residuals.
        """
        return self.residual_model.residuals(x, y)

    def treatment_matching_factor(self, x: NDArray, a: NDArray, propensity: NDArray) -> float:
        """Compute the treatment matching factor for the linear RWL policy.

        Args:
            x (NDArray): The input array.
            a (NDArray): The treatment array.
            propensity (NDArray): The propensity array.

        Returns:
            float: The treatment matching factor.
        """
        p_treat_one = self.action_probability(x)
        a = np.asarray(a, dtype=float).reshape(-1)
        propensity = np.asarray(propensity, dtype=float).reshape(-1)
        match_prob = np.where(a == 1.0, p_treat_one, 1.0 - p_treat_one)
        return float(np.mean(match_prob / np.maximum(propensity, EPS)))

    def clone(self) -> "LinearResidualWeightedLearningPolicy":
        """Clone the linear RWL policy.

        Returns:
            "LinearResidualWeightedLearningPolicy": The cloned linear RWL policy.
        """
        return copy.deepcopy(self)


def fit_weighted_main_effect_residual_model(
    x: NDArray,
    y: NDArray,
    propensity: NDArray,
    *,
    feature_map: StandardizedFeatureMap,
    l2_penalty: float = 1e-6,
) -> WeightedMainEffectResidualModel:
    """Fit the treatment-free regression used to construct RWL residuals.

    Args:
        x (NDArray): The input array.
        y (NDArray): The outcome array.
        propensity (NDArray): The propensity array.
        feature_map (StandardizedFeatureMap): The feature map.
        l2_penalty (float): The L2 penalty.

    Returns:
        WeightedMainEffectResidualModel: The fitted weighted main-effects residual model.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    propensity = np.asarray(propensity, dtype=float).reshape(-1)
    if x.shape[0] != y.shape[0] or x.shape[0] != propensity.shape[0]:
        raise ValueError("x, y, and propensity must have the same number of rows.")

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
    return WeightedMainEffectResidualModel(feature_map=fmap, coefficients=coefficients)


def _solve_rwl_convex_subproblem(
    phi: NDArray,
    treatment: NDArray,
    residual: NDArray,
    propensity: NDArray,
    beta_dc: NDArray,
    *,
    l2_penalty: float,
    penalize_intercept: bool,
    start: NDArray,
    maxiter: int,
) -> NDArray:
    """Solve the convex subproblem for the linear RWL policy.

    Args:
        phi (NDArray): The feature matrix.
        treatment (NDArray): The treatment array.
        residual (NDArray): The residual array.
        propensity (NDArray): The propensity array.
        beta_dc (NDArray): The beta_dc array.
        l2_penalty (float): The L2 penalty.
        penalize_intercept (bool): Whether to penalize the intercept.
        start (NDArray): The starting point.
        maxiter (int): The maximum number of iterations.

    Returns:
        NDArray: The coefficients.
    """
    abs_weight = np.abs(residual) / np.maximum(propensity, EPS) / max(1, phi.shape[0])
    positive = residual > 0.0
    negative = residual < 0.0
    penalty_mask = np.ones(phi.shape[1], dtype=float)
    if not penalize_intercept and phi.shape[1] > 0:
        penalty_mask[0] = 0.0

    def objective(beta: NDArray) -> Tuple[float, NDArray]:
        """Compute the objective function for the convex subproblem.

        Args:
            beta (NDArray): The coefficients.

        Returns:
            Tuple[float, NDArray]: The objective value and gradient.
        """
        margin = treatment * (phi @ beta)
        convex_loss = np.zeros_like(margin)
        convex_grad = np.zeros_like(margin)
        if np.any(positive):
            convex_loss[positive] = _phi_s(margin[positive], 1.0)
            convex_grad[positive] = _phi_s_prime(margin[positive], 1.0)
        if np.any(negative):
            convex_loss[negative] = _phi_s(margin[negative], 0.0)
            convex_grad[negative] = _phi_s_prime(margin[negative], 0.0)
        weighted_linear = (abs_weight * convex_grad + beta_dc) * treatment
        grad = phi.T @ weighted_linear + l2_penalty * penalty_mask * beta
        value = float(
            np.sum(abs_weight * convex_loss + beta_dc * margin) + 0.5 * l2_penalty * np.sum((penalty_mask * beta) ** 2)
        )
        return value, grad

    def run_lbfgs(x0: NDArray, iter_budget: int):
        """Run L-BFGS-B optimization.

        Args:
            x0 (NDArray): The starting point.
            iter_budget (int): The maximum number of iterations.

        Returns:
            OptimizeResult: The optimization result.
        """
        return minimize(
            fun=lambda beta: objective(beta),
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(iter_budget),
                "maxls": 100,
            },
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

    value, grad = objective(np.asarray(candidate.x, dtype=float))
    grad_inf = float(np.max(np.abs(grad)))
    if np.isfinite(value) and grad_inf <= 1e-4:
        return np.asarray(candidate.x, dtype=float)

    raise RuntimeError(
        "RWL convex subproblem optimization failed: "
        f"{candidate.message}; objective={value:.6g}; grad_inf={grad_inf:.6g}"
    )


def _fit_linear_rwl_dc(
    phi: NDArray,
    treatment: NDArray,
    residual: NDArray,
    propensity: NDArray,
    *,
    l2_penalty: float,
    penalize_intercept: bool,
    dc_tolerance: float,
    max_dc_iterations: int,
    subproblem_maxiter: int,
    start: Optional[NDArray],
) -> Tuple[NDArray, bool, int]:
    """Fit the linear RWL policy.

    Args:
        phi (NDArray): The feature matrix.
        treatment (NDArray): The treatment array.
        residual (NDArray): The residual array.
        propensity (NDArray): The propensity array.
        l2_penalty (float): The L2 penalty.
        penalize_intercept (bool): Whether to penalize the intercept.
        dc_tolerance (float): The tolerance for the DC algorithm.
        max_dc_iterations (int): The maximum number of iterations for the DC algorithm.
        subproblem_maxiter (int): The maximum number of iterations for the subproblem.
        start (Optional[NDArray]): The starting point.

    Returns:
        Tuple[NDArray, bool, int]: The coefficients, convergence flag, and number of iterations.
    """
    n_obs = max(1, phi.shape[0])
    abs_weight = np.abs(residual) / np.maximum(propensity, EPS) / n_obs
    beta_dc = 2.0 * abs_weight * (residual < 0.0)
    beta = np.zeros(phi.shape[1], dtype=float) if start is None else np.asarray(start, dtype=float).copy()
    converged = False

    for iteration in range(1, max_dc_iterations + 1):
        beta = _solve_rwl_convex_subproblem(
            phi=phi,
            treatment=treatment,
            residual=residual,
            propensity=propensity,
            beta_dc=beta_dc,
            l2_penalty=l2_penalty,
            penalize_intercept=penalize_intercept,
            start=beta,
            maxiter=subproblem_maxiter,
        )
        margin = treatment * (phi @ beta)
        beta_dc_new = np.zeros_like(beta_dc)
        negative = residual < 0.0
        positive = residual > 0.0
        if np.any(negative):
            beta_dc_new[negative] = -abs_weight[negative] * _phi_s_prime(margin[negative], 1.0)
        if np.any(positive):
            beta_dc_new[positive] = -abs_weight[positive] * _phi_s_prime(margin[positive], 0.0)
        if float(np.max(np.abs(beta_dc_new - beta_dc))) < dc_tolerance:
            converged = True
            return beta, converged, iteration
        beta_dc = beta_dc_new
    return beta, converged, max_dc_iterations


def fit_linear_residual_weighted_learning(
    x: NDArray,
    a: NDArray,
    y: NDArray,
    propensity: NDArray,
    *,
    feature_map: StandardizedFeatureMap,
    residual_feature_map: Optional[StandardizedFeatureMap] = None,
    l2_penalty: float = 0.01,
    residual_regression_penalty: float = 1e-6,
    penalize_intercept: bool = False,
    dc_tolerance: float = 1e-8,
    max_dc_iterations: int = 50,
    subproblem_maxiter: int = 500,
    start: Optional[NDArray] = None,
) -> LinearResidualWeightedLearningPolicy:
    """Fit linear RWL with smoothed ramp loss via a d.c. algorithm.

    Args:
        x (NDArray): The input array.
        a (NDArray): The treatment array.
        y (NDArray): The outcome array.
        propensity (NDArray): The propensity array.
        feature_map (StandardizedFeatureMap): The feature map.
        residual_feature_map (Optional[StandardizedFeatureMap]): The residual feature map.
        l2_penalty (float): The L2 penalty.
        residual_regression_penalty (float): The residual regression penalty.
        penalize_intercept (bool): Whether to penalize the intercept.
        dc_tolerance (float): The tolerance for the DC algorithm.
        max_dc_iterations (int): The maximum number of iterations for the DC algorithm.
        subproblem_maxiter (int): The maximum number of iterations for the subproblem.
        start (Optional[NDArray]): The starting point.

    Returns:
        LinearResidualWeightedLearningPolicy: The fitted linear RWL policy.
    """
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    propensity = np.asarray(propensity, dtype=float).reshape(-1)
    if x.shape[0] != a.shape[0] or x.shape[0] != y.shape[0] or x.shape[0] != propensity.shape[0]:
        raise ValueError("x, a, y, and propensity must have the same number of rows.")
    if np.any(~np.isin(a, [-1.0, 1.0])):
        raise ValueError("RWL expects treatments coded as -1/+1.")

    residual_model = fit_weighted_main_effect_residual_model(
        x=x,
        y=y,
        propensity=propensity,
        feature_map=residual_feature_map or StandardizedFeatureMap(),
        l2_penalty=residual_regression_penalty,
    )
    residual = residual_model.residuals(x, y)
    fmap = feature_map.clone()
    phi = fmap.fit_transform(x)
    beta, converged, n_iter = _fit_linear_rwl_dc(
        phi=phi,
        treatment=a,
        residual=residual,
        propensity=propensity,
        l2_penalty=float(l2_penalty),
        penalize_intercept=penalize_intercept,
        dc_tolerance=float(dc_tolerance),
        max_dc_iterations=int(max_dc_iterations),
        subproblem_maxiter=int(subproblem_maxiter),
        start=start,
    )
    return LinearResidualWeightedLearningPolicy(
        feature_map=fmap,
        beta=beta,
        residual_model=residual_model,
        l2_penalty=float(l2_penalty),
        converged=converged,
        n_dc_iterations=n_iter,
    )


def _stabilized_value_estimate(
    *,
    p_treat_one: NDArray,
    a: NDArray,
    reward: NDArray,
    propensity: NDArray,
) -> float:
    """Compute the stabilized value estimate for the linear RWL policy.

    Args:
        p_treat_one (NDArray): The action probability.
        a (NDArray): The treatment array.
        reward (NDArray): The reward array.
        propensity (NDArray): The propensity array.

    Returns:
        float: The stabilized value estimate.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    reward = np.asarray(reward, dtype=float).reshape(-1)
    propensity = np.asarray(propensity, dtype=float).reshape(-1)
    p_treat_one = np.asarray(p_treat_one, dtype=float).reshape(-1)
    match_prob = np.where(a == 1.0, p_treat_one, 1.0 - p_treat_one)
    denom = float(np.mean(match_prob / np.maximum(propensity, EPS)))
    if denom <= EPS:
        return -float("inf")
    numer = float(np.mean(reward * match_prob / np.maximum(propensity, EPS)))
    return numer / denom


def tune_linear_residual_weighted_learning(
    x: NDArray,
    a: NDArray,
    y: NDArray,
    propensity: NDArray,
    *,
    feature_map_factory: FeatureMapFactory,
    penalty_grid: Sequence[float],
    random_state: int,
    residual_feature_map_factory: Optional[FeatureMapFactory] = None,
    cv_folds: int = 5,
    residual_regression_penalty: float = 1e-6,
    penalize_intercept: bool = False,
    dc_tolerance: float = 1e-8,
    max_dc_iterations: int = 50,
    subproblem_maxiter: int = 500,
) -> Tuple[LinearResidualWeightedLearningPolicy, float]:
    """Tune the linear RWL penalty by cross-validated stabilized value."""

    penalty_values = [float(value) for value in penalty_grid]
    if not penalty_values:
        raise ValueError("penalty_grid must contain at least one candidate penalty.")

    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    propensity = np.asarray(propensity, dtype=float).reshape(-1)
    residual_map_factory = residual_feature_map_factory or (lambda: StandardizedFeatureMap())

    if x.shape[0] < 2 or len(penalty_values) == 1:
        best_penalty = penalty_values[0]
        policy = fit_linear_residual_weighted_learning(
            x=x,
            a=a,
            y=y,
            propensity=propensity,
            feature_map=feature_map_factory(),
            residual_feature_map=residual_map_factory(),
            l2_penalty=best_penalty,
            residual_regression_penalty=residual_regression_penalty,
            penalize_intercept=penalize_intercept,
            dc_tolerance=dc_tolerance,
            max_dc_iterations=max_dc_iterations,
            subproblem_maxiter=subproblem_maxiter,
        )
        return policy, best_penalty

    cv = KFold(n_splits=min(max(2, cv_folds), x.shape[0]), shuffle=True, random_state=random_state)
    scores: dict[float, float] = {}
    for penalty in penalty_values:
        fold_scores = []
        for train_idx, val_idx in cv.split(x):
            policy = fit_linear_residual_weighted_learning(
                x=x[train_idx],
                a=a[train_idx],
                y=y[train_idx],
                propensity=propensity[train_idx],
                feature_map=feature_map_factory(),
                residual_feature_map=residual_map_factory(),
                l2_penalty=penalty,
                residual_regression_penalty=residual_regression_penalty,
                penalize_intercept=penalize_intercept,
                dc_tolerance=dc_tolerance,
                max_dc_iterations=max_dc_iterations,
                subproblem_maxiter=subproblem_maxiter,
            )
            fold_scores.append(
                _stabilized_value_estimate(
                    p_treat_one=policy.action_probability(x[val_idx]),
                    a=a[val_idx],
                    reward=y[val_idx],
                    propensity=propensity[val_idx],
                )
            )
        scores[penalty] = float(np.mean(fold_scores))

    best_penalty = max(scores, key=lambda x: scores[x])
    fitted = fit_linear_residual_weighted_learning(
        x=x,
        a=a,
        y=y,
        propensity=propensity,
        feature_map=feature_map_factory(),
        residual_feature_map=residual_map_factory(),
        l2_penalty=best_penalty,
        residual_regression_penalty=residual_regression_penalty,
        penalize_intercept=penalize_intercept,
        dc_tolerance=dc_tolerance,
        max_dc_iterations=max_dc_iterations,
        subproblem_maxiter=subproblem_maxiter,
    )
    return fitted, best_penalty
