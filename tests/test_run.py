"""Tests for the main run module."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sae_probes.probing
from sae_probes.probing import ProbeConfig, select_features, train_probe


class TestProbeComparison:
    """Tests comparing our implementation with equivalent code."""

    def test_feature_selection(self):
        """Test that feature selection produces expected results."""
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

        # Reference implementation
        X_train_diff = X_train[y_train == 1].mean(dim=0) - X_train[y_train == 0].mean(
            dim=0
        )
        sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
        reference_indices = sorted_indices[:2].tolist()

        # Our implementation
        our_indices = select_features(X_train, y_train, 2)

        # Check that they match
        assert reference_indices == our_indices, (
            f"Expected {reference_indices}, got {our_indices}"
        )

    def test_probe_training(self):
        """Test that probe training produces expected results."""
        # Create synthetic data
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

        # Get top 2 features
        X_train_diff = X_train[y_train == 1].mean(axis=0) - X_train[y_train == 0].mean(
            axis=0
        )
        sorted_indices = np.argsort(np.abs(X_train_diff))[::-1]
        top_indices = sorted_indices[:2]

        X_train_filtered = X_train[:, top_indices]
        X_test_filtered = X_test[:, top_indices]

        # Reference implementation
        C_values = np.logspace(-4, 4, 20)
        best_auc = -1
        best_model = None

        for C in C_values:
            model = LogisticRegression(
                penalty="l2",
                C=C,
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            )

            model.fit(X_train_filtered, y_train)
            y_pred_proba = model.predict_proba(X_test_filtered)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)

            if auc > best_auc:
                best_auc = auc
                best_model = model

        # Calculate reference AUC
        y_pred_proba = best_model.predict_proba(X_test_filtered)[:, 1]
        reference_auc = roc_auc_score(y_test, y_pred_proba)

        # Our implementation
        config = ProbeConfig(
            reg_type="l2",
            k_values=[2],
            binarize=False,
            seed=42,
        )

        # Create a mock feature selector
        class MockFeatureSelector:
            def __call__(self, X, y, k):  # noqa: ARG002
                return top_indices.tolist()

        # Save original function to restore later
        original_select_features = sae_probes.probing.select_features

        try:
            # Replace with mock function for this test
            sae_probes.probing.select_features = MockFeatureSelector()

            # Train probe with our implementation
            results = train_probe(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                config=config,
            )
        finally:
            # Restore original function
            sae_probes.probing.select_features = original_select_features

        # Check that AUC is close
        assert abs(reference_auc - results[0].auc) < 1e-10, (
            f"Expected AUC {reference_auc}, got {results[0].auc}"
        )
