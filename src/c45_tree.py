
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd


def entropy(y: pd.Series) -> float:
    """Calculate Shannon entropy for a collection of class labels."""
    if len(y) == 0:
        return 0.0

    counts = Counter(y)
    total = len(y)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count > 0
    )


def information_gain(
    y: pd.Series,
    left_y: pd.Series,
    right_y: pd.Series,
) -> float:
    """Calculate information gain from a binary split."""
    total = len(y)

    if total == 0:
        return 0.0

    parent_entropy = entropy(y)
    weighted_child_entropy = (
        len(left_y) / total * entropy(left_y)
        + len(right_y) / total * entropy(right_y)
    )

    return parent_entropy - weighted_child_entropy


def split_information(left_size: int, right_size: int) -> float:
    """Calculate split information for a binary split."""
    total = left_size + right_size

    if total == 0:
        return 0.0

    result = 0.0

    for size in (left_size, right_size):
        if size > 0:
            probability = size / total
            result -= probability * math.log2(probability)

    return result


def gain_ratio(
    gain: float,
    left_size: int,
    right_size: int,
) -> float:
    """Calculate gain ratio while handling zero split information."""
    split_info = split_information(left_size, right_size)

    if split_info == 0:
        return 0.0

    return gain / split_info


@dataclass
class TreeNode:
    """A node in the C4.5-style decision tree."""

    prediction: Any
    feature: str | None = None
    threshold: float | None = None
    split_value: Any = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class C45DecisionTree:
    """A C4.5-style binary decision tree using gain ratio."""

    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root: TreeNode | None = None
        self.feature_names: list[str] = []

    @staticmethod
    def _majority_class(y: pd.Series) -> Any:
        return y.value_counts().idxmax()

    def _best_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[str, str, Any, float] | None:
        candidates = []

        for feature in X.columns:
            values = X[feature]

            if pd.api.types.is_numeric_dtype(values):
                unique_values = sorted(values.dropna().unique())
                thresholds = [
                    (a + b) / 2
                    for a, b in zip(unique_values, unique_values[1:])
                ]

                for threshold in thresholds:
                    left_mask = values <= threshold
                    right_mask = values > threshold

                    left_y = y.loc[left_mask]
                    right_y = y.loc[right_mask]

                    if (
                        len(left_y) < self.min_samples_leaf
                        or len(right_y) < self.min_samples_leaf
                    ):
                        continue

                    gain = information_gain(y, left_y, right_y)
                    ratio = gain_ratio(gain, len(left_y), len(right_y))

                    candidates.append(
                        (feature, "numeric", threshold, gain, ratio)
                    )

            else:
                for category in values.dropna().unique():
                    left_mask = values == category
                    right_mask = values != category

                    left_y = y.loc[left_mask]
                    right_y = y.loc[right_mask]

                    if (
                        len(left_y) < self.min_samples_leaf
                        or len(right_y) < self.min_samples_leaf
                    ):
                        continue

                    gain = information_gain(y, left_y, right_y)
                    ratio = gain_ratio(gain, len(left_y), len(right_y))

                    candidates.append(
                        (feature, "categorical", category, gain, ratio)
                    )

        if not candidates:
            return None

        # C4.5-inspired safeguard against choosing a split with
        # very low information gain but an artificially high gain ratio.
        positive_gains = [
            candidate[3]
            for candidate in candidates
            if candidate[3] > 0
        ]

        if not positive_gains:
            return None

        average_gain = sum(positive_gains) / len(positive_gains)

        eligible = [
            candidate
            for candidate in candidates
            if candidate[3] >= average_gain
        ]

        if not eligible:
            return None

        best = max(eligible, key=lambda candidate: candidate[4])

        if best[4] <= 0:
            return None

        feature, split_type, value, _, ratio = best
        return feature, split_type, value, ratio

    def _build_tree(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        depth: int,
    ) -> TreeNode:
        majority = self._majority_class(y)

        if (
            y.nunique() == 1
            or depth >= self.max_depth
            or len(y) < self.min_samples_split
            or X.empty
        ):
            return TreeNode(prediction=majority)

        split = self._best_split(X, y)

        if split is None:
            return TreeNode(prediction=majority)

        feature, split_type, value, _ = split

        node = TreeNode(
            prediction=majority,
            feature=feature,
            threshold=float(value) if split_type == "numeric" else None,
            split_value=value if split_type == "categorical" else None,
        )

        if split_type == "numeric":
            left_mask = X[feature] <= value
            right_mask = X[feature] > value
        else:
            left_mask = X[feature] == value
            right_mask = X[feature] != value

        X_left = X.loc[left_mask].drop(columns=[feature])
        X_right = X.loc[right_mask].drop(columns=[feature])

        y_left = y.loc[left_mask]
        y_right = y.loc[right_mask]

        if (
            len(y_left) < self.min_samples_leaf
            or len(y_right) < self.min_samples_leaf
        ):
            return TreeNode(prediction=majority)

        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)

        return node

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the decision tree."""
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")

        if X.empty or len(y) == 0:
            raise ValueError("Training data must not be empty.")

        if X.isnull().any().any() or y.isnull().any():
            raise ValueError(
                "Missing values are not supported yet. "
                "Handle them before training."
            )

        if not all(
            pd.api.types.is_numeric_dtype(X[column])
            or pd.api.types.is_object_dtype(X[column])
            or isinstance(X[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X[column])
            for column in X.columns
        ):
            raise ValueError("Unsupported feature data type.")

        self.feature_names = list(X.columns)

        # Reset both indices to ensure positional alignment.
        X_clean = X.reset_index(drop=True).copy()
        y_clean = y.reset_index(drop=True).copy()

        self.root = self._build_tree(X_clean, y_clean, 0)
        return self

    def predict_one(self, row: pd.Series) -> Any:
        """Predict the grade category for one student."""
        if self.root is None:
            raise ValueError("Train the model by calling fit() first.")

        node = self.root

        while node.feature is not None:
            feature = node.feature

            if feature not in row.index:
                return node.prediction

            value = row[feature]

            if node.threshold is not None:
                go_left = value <= node.threshold
            else:
                go_left = value == node.split_value

            next_node = node.left if go_left else node.right

            if next_node is None:
                return node.prediction

            node = next_node

        return node.prediction

    def predict(self, X: pd.DataFrame) -> list[Any]:
        """Predict grade categories for multiple students."""
        return [
            self.predict_one(row)
            for _, row in X.iterrows()
        ]
