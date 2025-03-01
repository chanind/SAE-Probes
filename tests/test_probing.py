"""Tests for probing functionality."""

import numpy as np
import torch

from sae_probes.probing import (
    BaselineProbeResults,
    ProbeConfig,
    select_features,
    train_baseline_probe,
    train_k_sparse_probe,
)


class TestFeatureSelection:
    """Test feature selection functionality."""

    def test_select_features(self):
        """Test feature selection based on class difference."""
        # Create synthetic data
        X_train = torch.tensor(
            [
                [1.0, 0.0, 3.0, 4.0],  # Class 0
                [2.0, 0.0, 2.0, 5.0],  # Class 0
                [5.0, 0.0, 1.0, 6.0],  # Class 1
                [6.0, 0.0, 0.0, 7.0],  # Class 1
            ]
        )
        y_train = np.array([0, 0, 1, 1])

        # Expected feature importance: feature 0 (4.0), feature 2 (-2.0), feature 3 (2.0), feature 1 (0.0)
        # So the order should be: 0, 2, 3, 1

        # Select top 2 features
        indices = select_features(X_train, y_train, 2)
        assert indices == [0, 2]

        # Select all features
        indices = select_features(X_train, y_train, 4)
        assert indices == [0, 2, 3, 1]


class TestProbeTraining:
    """Test probe training functionality."""

    def test_probe_training(self):
        """Test probe training with synthetic data."""
        # Create synthetic data with two classes that are linearly separable
        np.random.seed(42)

        # Class 0: features 0 and 1 are small
        X_train_0 = np.random.normal(0, 1, (50, 10))
        X_train_0[:, 0:2] = np.random.normal(-3, 1, (50, 2))

        # Class 1: features 0 and 1 are large
        X_train_1 = np.random.normal(0, 1, (50, 10))
        X_train_1[:, 0:2] = np.random.normal(3, 1, (50, 2))

        X_train = np.vstack([X_train_0, X_train_1])
        y_train = np.array([0] * 50 + [1] * 50)

        # Create test data in the same way
        X_test_0 = np.random.normal(0, 1, (20, 10))
        X_test_0[:, 0:2] = np.random.normal(-3, 1, (20, 2))

        X_test_1 = np.random.normal(0, 1, (20, 10))
        X_test_1[:, 0:2] = np.random.normal(3, 1, (20, 2))

        X_test = np.vstack([X_test_0, X_test_1])
        y_test = np.array([0] * 20 + [1] * 20)

        # Train probe
        config = ProbeConfig(
            reg_type="l2",
            k_values=[1, 2, 5, 10],
            binarize=False,
            seed=42,
        )

        results = train_k_sparse_probe(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
        )

        # Check results
        assert len(results) == 4  # One for each k value

        # Top features should be 0 and 1
        assert 0 in results[1].feature_indices
        assert 0 in results[2].feature_indices and 1 in results[2].feature_indices

        # AUC should be high since data is linearly separable
        assert results[1].auc > 0.9
        assert results[2].auc > 0.9

        # More features should not significantly improve performance
        # since only the first two features are meaningful
        assert abs(results[3].auc - results[2].auc) < 0.1


class TestBaselineProbeTraining:
    """Test baseline probe training functionality."""

    def test_baseline_probe_training(self):
        """Test baseline probe training with synthetic data."""
        # Create synthetic data with two classes that are linearly separable
        np.random.seed(42)

        # Class 0: features concentrated in certain dimensions
        X_train_0 = np.random.normal(0, 1, (50, 100))
        X_train_0[:, 0:5] = np.random.normal(-3, 1, (50, 5))

        # Class 1: features concentrated in certain dimensions
        X_train_1 = np.random.normal(0, 1, (50, 100))
        X_train_1[:, 0:5] = np.random.normal(3, 1, (50, 5))

        X_train = np.vstack([X_train_0, X_train_1])
        y_train = np.array([0] * 50 + [1] * 50)

        # Create test data in the same way
        X_test_0 = np.random.normal(0, 1, (20, 100))
        X_test_0[:, 0:5] = np.random.normal(-3, 1, (20, 5))

        X_test_1 = np.random.normal(0, 1, (20, 100))
        X_test_1[:, 0:5] = np.random.normal(3, 1, (20, 5))

        X_test = np.vstack([X_test_0, X_test_1])
        y_test = np.array([0] * 20 + [1] * 20)

        # Test with L2 regularization
        results_l2 = train_baseline_probe(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            reg_type="l2",
            seed=42,
        )

        # Test with L1 regularization
        results_l1 = train_baseline_probe(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            reg_type="l1",
            seed=42,
        )

        # Check results
        assert isinstance(results_l2, BaselineProbeResults)
        assert isinstance(results_l1, BaselineProbeResults)

        # Both should have high AUC since data is linearly separable
        assert results_l2.auc > 0.9
        assert results_l1.auc > 0.9

        # L1 regularization should select fewer features
        assert np.sum(abs(results_l1.model.coef_) > 0) < np.sum(
            abs(results_l2.model.coef_) > 0
        )
