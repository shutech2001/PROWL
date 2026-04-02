from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold  # type: ignore

from .prowl import StandardizedFeatureMap

FeatureMapFactory = Callable[[], StandardizedFeatureMap]
EPS = 1e-12


def _solve_weighted_ridge(
    design: NDArray,
    outcome: NDArray,
    weight: NDArray,
    l2_penalty: float,
) -> NDArray:
    """Solve a weighted ridge system with a pseudoinverse fallback.

    Args:
        design (NDArray): The design matrix.
        outcome (NDArray): The outcome vector.
        weight (NDArray): The weight vector.
        l2_penalty (float): The L2 penalty.

    Returns:
        NDArray: The coefficients.
    """

    weighted_design = design * weight[:, None]
    gram = weighted_design.T @ design
    if l2_penalty > 0.0:
        gram = gram + float(l2_penalty) * np.eye(design.shape[1], dtype=float)
    rhs = weighted_design.T @ outcome
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


@dataclass
class LinearQLearningPolicy:
    """One-stage linear Q-learning policy with separate main and blip terms.

    Args:
        main_feature_map (StandardizedFeatureMap): The main feature map.
        blip_feature_map (StandardizedFeatureMap): The blip feature map.
        beta (NDArray): The main coefficients.
        psi (NDArray): The blip coefficients.
    """

    main_feature_map: StandardizedFeatureMap
    blip_feature_map: StandardizedFeatureMap
    beta: NDArray
    psi: NDArray

    def main_features(self, x: NDArray) -> NDArray:
        """Compute the main features.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The main features.
        """
        return self.main_feature_map.transform(x)

    def blip_features(self, x: NDArray) -> NDArray:
        """Compute the blip features.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The blip features.
        """
        return self.blip_feature_map.transform(x)

    def q_values(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        """Compute the Q-values.

        Args:
            x (NDArray): The input array.

        Returns:
            Tuple[NDArray, NDArray]: The Q-values.
        """
        h0 = self.main_features(x)
        h1 = self.blip_features(x)
        main = h0 @ self.beta
        blip = h1 @ self.psi
        return main + blip, main - blip

    def observed_q(self, x: NDArray, a: NDArray) -> NDArray:
        """Compute the observed Q-value.

        Args:
            x (NDArray): The input array.
            a (NDArray): The treatment array.

        Returns:
            NDArray: The observed Q-value.
        """
        q_pos, q_neg = self.q_values(x)
        a = np.asarray(a, dtype=float).reshape(-1)
        return np.where(a == 1.0, q_pos, q_neg)

    def blip(self, x: NDArray) -> NDArray:
        """Compute the blip.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The blip.
        """
        return self.blip_features(x) @ self.psi

    def action_probability(self, x: NDArray) -> NDArray:
        """Compute the action probability.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The action probability.
        """
        q_pos, q_neg = self.q_values(x)
        return (q_pos >= q_neg).astype(float)

    def clone(self) -> "LinearQLearningPolicy":
        """Clone the linear Q-learning policy.

        Returns:
            LinearQLearningPolicy: The cloned linear Q-learning policy.
        """
        return copy.deepcopy(self)


def fit_linear_q_learning(
    x: NDArray,
    a: NDArray,
    y: NDArray,
    *,
    main_feature_map: StandardizedFeatureMap,
    blip_feature_map: StandardizedFeatureMap,
    l2_penalty: float = 0.0,
    sample_weight: Optional[NDArray] = None,
) -> LinearQLearningPolicy:
    """Fit Q(h, a) = beta^T h0 + (psi^T h1) a by weighted ridge least squares.

    Args:
        x (NDArray): The input array.
        a (NDArray): The treatment array.
        y (NDArray): The outcome array.
        main_feature_map (StandardizedFeatureMap): The main feature map.
        blip_feature_map (StandardizedFeatureMap): The blip feature map.
        l2_penalty (float): The L2 penalty.
        sample_weight (NDArray): The sample weight.

    Returns:
        LinearQLearningPolicy: The fitted linear Q-learning policy.
    """

    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape[0] != a.shape[0] or x.shape[0] != y.shape[0]:
        raise ValueError("x, a, and y must have the same number of rows.")
    if np.any(~np.isin(a, [-1.0, 1.0])):
        raise ValueError("Q-learning expects treatments coded as -1/+1.")

    main_map = main_feature_map.clone()
    blip_map = blip_feature_map.clone()
    h0 = main_map.fit_transform(x)
    h1 = blip_map.fit_transform(x)
    design = np.concatenate([h0, a[:, None] * h1], axis=1)

    if sample_weight is None:
        weight = np.ones(x.shape[0], dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weight.shape[0] != x.shape[0]:
            raise ValueError("sample_weight must match the number of rows in x.")
        weight = np.maximum(weight, 0.0)

    coefficients = _solve_weighted_ridge(
        design=design,
        outcome=y,
        weight=weight,
        l2_penalty=l2_penalty,
    )
    dim_main = h0.shape[1]
    beta = coefficients[:dim_main]
    psi = coefficients[dim_main:]
    return LinearQLearningPolicy(
        main_feature_map=main_map,
        blip_feature_map=blip_map,
        beta=beta,
        psi=psi,
    )


def tune_linear_q_learning(
    x: NDArray,
    a: NDArray,
    y: NDArray,
    *,
    main_feature_map_factory: FeatureMapFactory,
    blip_feature_map_factory: FeatureMapFactory,
    penalty_grid: Sequence[float],
    random_state: int,
    sample_weight: Optional[NDArray] = None,
) -> tuple[LinearQLearningPolicy, float]:
    """Tune the ridge penalty by cross-validated Q-function prediction error.

    Args:
        x (NDArray): The input array.
        a (NDArray): The treatment array.
        y (NDArray): The outcome array.
        main_feature_map_factory (FeatureMapFactory): The main feature map factory.
        blip_feature_map_factory (FeatureMapFactory): The blip feature map factory.
        penalty_grid (Sequence[float]): The penalty grid.
        random_state (int): The random state.
        sample_weight (Optional[NDArray]): The sample weight.

    Returns:
        tuple[LinearQLearningPolicy, float]: The fitted linear Q-learning policy and the best penalty.
    """

    penalty_values = [float(value) for value in penalty_grid]
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if not penalty_values:
        raise ValueError("penalty_grid must contain at least one candidate penalty.")
    if x.shape[0] < 2 or len(penalty_values) == 1:
        best_penalty = penalty_values[0]
        fitted = fit_linear_q_learning(
            x=x,
            a=a,
            y=y,
            main_feature_map=main_feature_map_factory(),
            blip_feature_map=blip_feature_map_factory(),
            l2_penalty=best_penalty,
            sample_weight=weight,
        )
        return fitted, best_penalty

    cv = KFold(n_splits=min(5, x.shape[0]), shuffle=True, random_state=random_state)

    scores: dict[float, float] = {}
    for penalty in penalty_values:
        fold_losses = []
        for train_idx, val_idx in cv.split(x):
            policy = fit_linear_q_learning(
                x=x[train_idx],
                a=a[train_idx],
                y=y[train_idx],
                main_feature_map=main_feature_map_factory(),
                blip_feature_map=blip_feature_map_factory(),
                l2_penalty=penalty,
                sample_weight=None if weight is None else weight[train_idx],
            )
            residual = y[val_idx] - policy.observed_q(x[val_idx], a[val_idx])
            if weight is None:
                fold_loss = float(np.mean(residual**2))
            else:
                fold_weight = np.maximum(weight[val_idx], 0.0)
                fold_loss = float(np.sum(fold_weight * residual**2) / max(np.sum(fold_weight), EPS))
            fold_losses.append(fold_loss)
        scores[penalty] = float(np.mean(fold_losses))

    best_penalty = min(scores, key=lambda x: scores[x])
    fitted = fit_linear_q_learning(
        x=x,
        a=a,
        y=y,
        main_feature_map=main_feature_map_factory(),
        blip_feature_map=blip_feature_map_factory(),
        l2_penalty=best_penalty,
        sample_weight=weight,
    )
    return fitted, best_penalty
