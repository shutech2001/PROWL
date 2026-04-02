from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12


@dataclass
class PolicyTreeNode:
    reward: float
    action_index: Optional[int]
    split_feature: Optional[int]
    split_threshold: Optional[float]
    left: Optional["PolicyTreeNode"]
    right: Optional["PolicyTreeNode"]
    n_samples: int

    @property
    def is_leaf(self) -> bool:
        return self.action_index is not None


@dataclass
class PolicyTreePolicy:
    actions: NDArray
    depth: int
    root: PolicyTreeNode
    min_node_size: int
    split_step: int
    max_features: int | None

    def predict_action(self, x: NDArray) -> NDArray:
        """Compute the predicted action.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The predicted action.
        """
        x = np.asarray(x, dtype=float)
        return np.asarray([self._predict_one(row, self.root) for row in x], dtype=self.actions.dtype)

    def action_probability(self, x: NDArray) -> NDArray:
        """Compute the action probability.

        Args:
            x (NDArray): The input array.

        Returns:
            NDArray: The action probability.
        """
        if self.actions.shape[0] != 2 or not np.all(np.isin(self.actions, [-1, 1])):
            raise ValueError("action_probability is only defined for binary actions {-1, 1}.")
        action = self.predict_action(x).reshape(-1)
        return (action == 1).astype(float)

    def clone(self) -> "PolicyTreePolicy":
        """Clone the policy tree policy.

        Returns:
            PolicyTreePolicy: The cloned policy tree policy.
        """
        return copy.deepcopy(self)

    def _predict_one(self, x_row: NDArray, node: PolicyTreeNode) -> NDArray:
        """Compute the predicted action for a single row of the input array.

        Args:
            x_row (NDArray): The input row.
            node (PolicyTreeNode): The current node.

        Returns:
            NDArray: The predicted action.
        """
        if node.is_leaf:
            assert node.action_index is not None
            return self.actions[int(node.action_index)]

        assert node.split_feature is not None
        assert node.split_threshold is not None

        if x_row[int(node.split_feature)] <= float(node.split_threshold):
            if node.left is None:
                raise ValueError("left child is None.")
            return self._predict_one(x_row, node.left)
        if node.right is None:
            raise ValueError("right child is None.")
        return self._predict_one(x_row, node.right)


def double_robust_score_matrix(
    outcome: NDArray,
    action: NDArray,
    action_values: Sequence[float | int],
    conditional_means: NDArray,
    *,
    action_probabilities: NDArray | float,
) -> NDArray:
    """Construct a doubly robust score matrix.

    Args:
        outcome (NDArray): The observed outcome vector.
        action (NDArray): The observed action labels.
        action_values (Sequence[float | int]): The column ordering for the score matrix.
        conditional_means (NDArray): The conditional means matrix.
        action_probabilities (NDArray | float): The action probabilities.

    Returns:
        NDArray: The doubly robust score matrix.
    """

    outcome = np.asarray(outcome, dtype=float).reshape(-1)
    action = np.asarray(action).reshape(-1)
    means = np.asarray(conditional_means, dtype=float)
    if means.ndim != 2 or means.shape[0] != outcome.shape[0]:
        raise ValueError("conditional_means must have shape (n_obs, n_actions).")
    actions = np.asarray(action_values)
    if actions.shape[0] != means.shape[1]:
        raise ValueError("action_values must align with conditional_means columns.")

    if np.isscalar(action_probabilities):
        fill_value = means.dtype.type(float(action_probabilities))
        propensity = np.full(means.shape, fill_value, dtype=means.dtype)
    else:
        propensity = np.asarray(action_probabilities, dtype=float)

    scores = means.copy()
    for action_idx, action_value in enumerate(actions):
        mask = action == action_value
        if not np.any(mask):
            continue
        scores[mask, action_idx] += (outcome[mask] - means[mask, action_idx]) / np.maximum(
            propensity[mask, action_idx], EPS
        )
    return scores


def _leaf_node(scores: NDArray) -> PolicyTreeNode:
    """Compute the leaf node.

    Args:
        scores (NDArray): The scores.

    Returns:
        PolicyTreeNode: The leaf node.
    """
    action_rewards = np.sum(scores, axis=0)
    action_index = int(np.argmax(action_rewards))
    return PolicyTreeNode(
        reward=float(action_rewards[action_index]),
        action_index=action_index,
        split_feature=None,
        split_threshold=None,
        left=None,
        right=None,
        n_samples=int(scores.shape[0]),
    )


def _candidate_split_positions(
    sorted_feature: NDArray,
    *,
    min_node_size: int,
    split_step: int,
) -> NDArray:
    """Compute the candidate split positions.

    Args:
        sorted_feature (NDArray): The sorted feature.
        min_node_size (int): The minimum node size.
        split_step (int): The split step.

    Returns:
        NDArray: The candidate split positions.
    """
    n_obs = sorted_feature.shape[0]
    if n_obs < 2 * min_node_size:
        return np.empty(0, dtype=int)
    valid = np.flatnonzero(sorted_feature[:-1] < sorted_feature[1:]) + 1
    if valid.size == 0:
        return valid
    valid = valid[(valid >= min_node_size) & (valid <= n_obs - min_node_size)]
    if valid.size == 0:
        return valid
    step = max(1, int(split_step))
    reduced = valid[::step]
    if reduced.size == 0 or reduced[-1] != valid[-1]:
        reduced = np.concatenate([reduced, valid[[-1]]]) if valid.size > 0 else reduced
    return np.unique(reduced)


def _best_depth1_split_for_feature(
    x_col: NDArray,
    scores: NDArray,
    *,
    feature_index: int,
    min_node_size: int,
    split_step: int,
) -> Optional[PolicyTreeNode]:
    """Compute the best depth 1 split for a feature.

    Args:
        x_col (NDArray): The feature column.
        scores (NDArray): The scores.
        feature_index (int): The feature index.
        min_node_size (int): The minimum node size.
        split_step (int): The split step.

    Returns:
        Optional[PolicyTreeNode]: The best depth 1 split.
    """
    order = np.argsort(x_col, kind="mergesort")
    x_sorted = x_col[order]
    scores_sorted = scores[order]
    split_positions = _candidate_split_positions(
        x_sorted,
        min_node_size=min_node_size,
        split_step=split_step,
    )
    if split_positions.size == 0:
        return None

    cumulative = np.cumsum(scores_sorted, axis=0)
    total = cumulative[-1]
    best_tree: Optional[PolicyTreeNode] = None
    for position in split_positions:
        left_rewards = cumulative[position - 1]
        right_rewards = total - left_rewards
        left_action = int(np.argmax(left_rewards))
        right_action = int(np.argmax(right_rewards))
        total_reward = float(left_rewards[left_action] + right_rewards[right_action])
        threshold = float(0.5 * (x_sorted[position - 1] + x_sorted[position]))
        tree = PolicyTreeNode(
            reward=total_reward,
            action_index=None,
            split_feature=feature_index,
            split_threshold=threshold,
            left=PolicyTreeNode(
                reward=float(left_rewards[left_action]),
                action_index=left_action,
                split_feature=None,
                split_threshold=None,
                left=None,
                right=None,
                n_samples=int(position),
            ),
            right=PolicyTreeNode(
                reward=float(right_rewards[right_action]),
                action_index=right_action,
                split_feature=None,
                split_threshold=None,
                left=None,
                right=None,
                n_samples=int(scores.shape[0] - position),
            ),
            n_samples=int(scores.shape[0]),
        )
        if best_tree is None or tree.reward > best_tree.reward:
            best_tree = tree
    return best_tree


def _screen_features(
    x: NDArray,
    scores: NDArray,
    *,
    min_node_size: int,
    split_step: int,
    max_features: Optional[int],
) -> NDArray:
    """Compute the screened features.

    Args:
        x (NDArray): The input array.
        scores (NDArray): The scores.
        min_node_size (int): The minimum node size.
        split_step (int): The split step.
        max_features (Optional[int]): The maximum features.

    Returns:
        NDArray: The screened features.
    """
    p = x.shape[1]
    if max_features is None or max_features >= p:
        return np.arange(p, dtype=int)
    leaf_reward = _leaf_node(scores).reward
    gains = np.full(p, -np.inf, dtype=float)
    for feature in range(p):
        split_tree = _best_depth1_split_for_feature(
            x[:, feature],
            scores,
            feature_index=feature,
            min_node_size=min_node_size,
            split_step=split_step,
        )
        if split_tree is not None:
            gains[feature] = split_tree.reward - leaf_reward
    ranked = np.argsort(-gains, kind="mergesort")
    finite = ranked[np.isfinite(gains[ranked])]
    if finite.size == 0:
        return np.arange(min(max_features, p), dtype=int)
    return np.sort(finite[:max_features])


def _search_policy_tree(
    x: NDArray,
    scores: NDArray,
    *,
    depth: int,
    min_node_size: int,
    split_step: int,
    max_features: Optional[int],
) -> PolicyTreeNode:
    """Compute the policy tree.

    Args:
        x (NDArray): The input array.
        scores (NDArray): The scores.
        depth (int): The depth.
        min_node_size (int): The minimum node size.
        split_step (int): The split step.
        max_features (Optional[int]): The maximum features.

    Returns:
        PolicyTreeNode: The policy tree.
    """
    leaf = _leaf_node(scores)
    if depth <= 0 or x.shape[0] < 2 * min_node_size:
        return leaf
    if depth == 1:
        best_split = None
        for feature in range(x.shape[1]):
            candidate = _best_depth1_split_for_feature(
                x[:, feature],
                scores,
                feature_index=feature,
                min_node_size=min_node_size,
                split_step=split_step,
            )
            if candidate is not None and (best_split is None or candidate.reward > best_split.reward):
                best_split = candidate
        return leaf if best_split is None or best_split.reward <= leaf.reward else best_split

    feature_candidates = _screen_features(
        x,
        scores,
        min_node_size=min_node_size,
        split_step=split_step,
        max_features=max_features,
    )
    best_tree = leaf
    for feature in feature_candidates:
        order = np.argsort(x[:, feature], kind="mergesort")
        x_sorted = x[order]
        scores_sorted = scores[order]
        positions = _candidate_split_positions(
            x_sorted[:, feature],
            min_node_size=min_node_size,
            split_step=split_step,
        )
        for position in positions:
            left_tree = _search_policy_tree(
                x_sorted[:position],
                scores_sorted[:position],
                depth=depth - 1,
                min_node_size=min_node_size,
                split_step=split_step,
                max_features=max_features,
            )
            right_tree = _search_policy_tree(
                x_sorted[position:],
                scores_sorted[position:],
                depth=depth - 1,
                min_node_size=min_node_size,
                split_step=split_step,
                max_features=max_features,
            )
            reward = left_tree.reward + right_tree.reward
            if reward <= best_tree.reward:
                continue
            threshold = float(0.5 * (x_sorted[position - 1, feature] + x_sorted[position, feature]))
            best_tree = PolicyTreeNode(
                reward=float(reward),
                action_index=None,
                split_feature=int(feature),
                split_threshold=threshold,
                left=left_tree,
                right=right_tree,
                n_samples=int(x.shape[0]),
            )
    return best_tree


def fit_policy_tree(
    x: NDArray,
    scores: NDArray,
    *,
    depth: int,
    actions: Sequence[float | int],
    min_node_size: int = 5,
    split_step: int = 1,
    max_features: int | None = None,
) -> PolicyTreePolicy:
    """Fit a shallow policy tree by empirical welfare maximization.

    Args:
        x (NDArray): The input array.
        scores (NDArray): The scores.
        depth (int): The depth.
        actions (Sequence[float | int]): The actions.
        min_node_size (int): The minimum node size.
        split_step (int): The split step.
        max_features (Optional[int]): The maximum features.

    Returns:
        PolicyTreePolicy: The fitted policy tree policy.
    """

    x = np.asarray(x, dtype=float)
    scores = np.asarray(scores, dtype=float)
    actions_arr = np.asarray(actions)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")
    if scores.ndim != 2 or scores.shape[0] != x.shape[0]:
        raise ValueError("scores must have shape (n_obs, n_actions).")
    if actions_arr.shape[0] != scores.shape[1]:
        raise ValueError("actions must align with the number of columns in scores.")

    root = _search_policy_tree(
        x,
        scores,
        depth=int(max(0, depth)),
        min_node_size=int(max(1, min_node_size)),
        split_step=int(max(1, split_step)),
        max_features=max_features,
    )
    return PolicyTreePolicy(
        actions=actions_arr,
        depth=int(max(0, depth)),
        root=root,
        min_node_size=int(max(1, min_node_size)),
        split_step=int(max(1, split_step)),
        max_features=None if max_features is None else int(max_features),
    )
