from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold  # type: ignore

from .prowl import StandardizedFeatureMap, feature_scores, fit_weighted_hinge_map


FeatureMapFactory = Callable[[], StandardizedFeatureMap]
EPS = 1e-12


@dataclass
class DeterministicScorePolicy:
    """Deterministic score policy.

    Args:
        feature_map (StandardizedFeatureMap): The feature map.
        beta (NDArray): The coefficients.
        score_bound (Optional[float]): The score bound.
    """

    feature_map: StandardizedFeatureMap
    beta: NDArray
    score_bound: Optional[float]

    def decision_function(self, x: NDArray) -> NDArray:
        """Compute the decision function.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The decision function.
        """
        phi = self.feature_map.transform(x)
        if self.score_bound is None:
            return (phi @ self.beta).reshape(-1)
        return feature_scores(phi, self.beta[None, :], self.score_bound).reshape(-1)

    def predict_action(self, x: NDArray) -> NDArray:
        """Compute the predicted action.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The predicted action.
        """
        score = self.decision_function(x)
        return np.where(score >= 0.0, 1.0, -1.0)

    def action_probability(self, x: NDArray) -> NDArray:
        """Compute the action probability.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The action probability.
        """
        return (self.predict_action(x) == 1.0).astype(float)

    def clone(self) -> "DeterministicScorePolicy":
        """Clone the deterministic score policy.

        Returns:
            DeterministicScorePolicy: The cloned deterministic score policy.
        """
        return copy.deepcopy(self)


def _ips_policy_value(
    *,
    score: NDArray,
    observed_a: NDArray,
    evaluation_reward: NDArray,
    propensity: NDArray,
) -> float:
    """Compute the IPS policy value.

    Args:
        score (NDArray): The score.
        observed_a (NDArray): The observed action.
        evaluation_reward (NDArray): The evaluation reward.
        propensity (NDArray): The propensity.
    """
    score = np.asarray(score, dtype=float).reshape(-1)
    observed_a = np.asarray(observed_a, dtype=float).reshape(-1)
    evaluation_reward = np.asarray(evaluation_reward, dtype=float).reshape(-1)
    propensity = np.asarray(propensity, dtype=float).reshape(-1)
    predicted_a = np.where(score >= 0.0, 1.0, -1.0)
    match = (predicted_a == observed_a).astype(float)
    return float(np.mean(evaluation_reward * match / np.maximum(propensity, EPS)))


def fit_weighted_hinge_policy(
    *,
    x: NDArray,
    fit_labels: NDArray,
    fit_weights: NDArray,
    feature_map: StandardizedFeatureMap,
    l2_penalty: float,
    score_bound: Optional[float],
) -> DeterministicScorePolicy:
    """Fit the weighted hinge policy.

    Args:
        x (NDArray): The input array.
        fit_labels (NDArray): The fit labels.
        fit_weights (NDArray): The fit weights.
        feature_map (StandardizedFeatureMap): The feature map.
        l2_penalty (float): The L2 penalty.
        score_bound (Optional[float]): The score bound.

    Returns:
        DeterministicScorePolicy: The fitted deterministic score policy.
    """
    fmap = feature_map.clone()
    phi = fmap.fit_transform(np.asarray(x, dtype=float))
    beta = fit_weighted_hinge_map(
        phi=phi,
        treatment=np.asarray(fit_labels, dtype=float).reshape(-1),
        sample_weight=np.asarray(fit_weights, dtype=float).reshape(-1),
        l2_penalty=float(l2_penalty),
        score_bound=score_bound,
    )
    return DeterministicScorePolicy(feature_map=fmap, beta=beta, score_bound=score_bound)


def tune_weighted_hinge_policy(
    *,
    x: NDArray,
    fit_labels: NDArray,
    fit_weights: NDArray,
    observed_a: NDArray,
    evaluation_reward: NDArray,
    propensity: NDArray,
    feature_map_factory: FeatureMapFactory,
    penalty_grid: Sequence[float],
    random_state: int,
    score_bound: Optional[float],
) -> Tuple[DeterministicScorePolicy, float]:
    """Tune the weighted hinge policy.

    Args:
        x (NDArray): The input array.
        fit_labels (NDArray): The fit labels.
        fit_weights (NDArray): The fit weights.
        observed_a (NDArray): The observed action.
        evaluation_reward (NDArray): The evaluation reward.
        propensity (NDArray): The propensity.
        feature_map_factory (FeatureMapFactory): The feature map factory.
        penalty_grid (Sequence[float]): The penalty grid.
        random_state (int): The random state.
        score_bound (Optional[float]): The score bound.

    Returns:
        Tuple[DeterministicScorePolicy, float]: The tuned deterministic score policy and the best penalty.
    """

    x = np.asarray(x, dtype=float)
    fit_labels = np.asarray(fit_labels, dtype=float).reshape(-1)
    fit_weights = np.asarray(fit_weights, dtype=float).reshape(-1)
    observed_a = np.asarray(observed_a, dtype=float).reshape(-1)
    evaluation_reward = np.asarray(evaluation_reward, dtype=float).reshape(-1)
    propensity = np.asarray(propensity, dtype=float).reshape(-1)
    penalty_values = [float(value) for value in penalty_grid]
    if not penalty_values:
        raise ValueError("penalty_grid must contain at least one candidate penalty.")
    if x.shape[0] < 2 or len(penalty_values) == 1:
        best_penalty = penalty_values[0]
        return (
            fit_weighted_hinge_policy(
                x=x,
                fit_labels=fit_labels,
                fit_weights=fit_weights,
                feature_map=feature_map_factory(),
                l2_penalty=best_penalty,
                score_bound=score_bound,
            ),
            best_penalty,
        )

    cv = KFold(n_splits=min(5, x.shape[0]), shuffle=True, random_state=random_state)
    best_penalty = penalty_values[0]
    best_score = -float("inf")
    for penalty in penalty_values:
        fold_scores = []
        for train_idx, val_idx in cv.split(x):
            policy = fit_weighted_hinge_policy(
                x=x[train_idx],
                fit_labels=fit_labels[train_idx],
                fit_weights=fit_weights[train_idx],
                feature_map=feature_map_factory(),
                l2_penalty=penalty,
                score_bound=score_bound,
            )
            score_val = policy.decision_function(x[val_idx])
            fold_scores.append(
                _ips_policy_value(
                    score=score_val,
                    observed_a=observed_a[val_idx],
                    evaluation_reward=evaluation_reward[val_idx],
                    propensity=propensity[val_idx],
                )
            )
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_penalty = penalty

    fitted = fit_weighted_hinge_policy(
        x=x,
        fit_labels=fit_labels,
        fit_weights=fit_weights,
        feature_map=feature_map_factory(),
        l2_penalty=best_penalty,
        score_bound=score_bound,
    )
    return fitted, best_penalty


__all__ = [
    "DeterministicScorePolicy",
    "fit_weighted_hinge_policy",
    "tune_weighted_hinge_policy",
]
