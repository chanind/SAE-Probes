# It appears there isn't a direct old equivalent of utils_training in the _comparison folder based on filename.
# If there are specific functions from the old utils_training.py (perhaps the one in _comparison/utils_training.py)
# that have counterparts in the new sae_probes.utils_training, we can add comparison tests.
# For now, let's add a placeholder or a simple test for a function if one exists.

# from sae_probes import utils_training as new_utils_training
# from tests._comparison import utils_training as old_utils_training # if applicable

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeavePOut, StratifiedKFold

from sae_probes import utils_training as new_utils_training
from tests._comparison import utils_training as old_utils_training


@pytest.mark.parametrize(
    "n_samples, expected_cv_type",
    [
        (10, LeavePOut),  # n_samples <= 12
        (50, StratifiedKFold),  # 12 < n_samples < 128
        (200, list),  # n_samples >= 128 (custom split)
    ],
)
def test_get_cv_behavior(n_samples: int, expected_cv_type: type):
    """Test that get_cv returns the correct cross-validator type or structure."""
    X_train = np.random.rand(n_samples, 5)  # 5 features

    new_cv = new_utils_training.get_cv(X_train)
    old_cv = old_utils_training.get_cv(X_train)

    assert isinstance(new_cv, expected_cv_type)
    assert isinstance(old_cv, expected_cv_type)

    if expected_cv_type == LeavePOut:
        assert new_cv.p == 2
        assert old_cv.p == 2
    elif expected_cv_type == StratifiedKFold:
        assert new_cv.n_splits == 6
        assert new_cv.shuffle is True
        assert new_cv.random_state == 42
        assert old_cv.n_splits == 6
        assert old_cv.shuffle is True
        assert old_cv.random_state == 42
    elif expected_cv_type == list:
        assert len(new_cv) == 1
        assert len(new_cv[0]) == 2
        val_size = min(int(0.2 * n_samples), 100)
        train_size = n_samples - val_size
        assert len(new_cv[0][0]) == train_size
        assert len(new_cv[0][1]) == val_size

        assert len(old_cv) == 1
        assert len(old_cv[0]) == 2
        assert len(old_cv[0][0]) == train_size
        assert len(old_cv[0][1]) == val_size


@pytest.mark.parametrize(
    "n_samples, cv_type_param",
    [
        (20, "lpo"),  # LeavePOut with enough samples for valid splits
        (20, "stratified"),  # StratifiedKFold
        (150, "custom"),  # Custom list-based split
        (
            5,
            "lpo_few_samples",
        ),  # LeavePOut with very few samples (some val might be single class)
    ],
)
def test_get_splits_behavior_and_comparison(n_samples: int, cv_type_param: str):
    """Test get_splits ensures validation sets have both classes and compares old/new."""
    X_train = np.random.rand(n_samples, 3)
    # Create y_train such that some splits might initially be problematic if not filtered
    if n_samples > 4:
        y_train = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
        if n_samples % 2 == 1:  # ensure we have enough of one class for LPO with p=2
            y_train[0] = 0
            y_train[1] = 0
    else:  # too few samples to guarantee two classes in LPO p=2 validation
        y_train = np.array([0] * n_samples)

    if cv_type_param == "lpo":
        cv = LeavePOut(2)
    elif cv_type_param == "lpo_few_samples":
        # This case is tricky for LPO(2) if we want to ensure both classes in val
        # For very small n_samples, get_splits might return an empty list if no val split has 2 classes
        # Here we make y_train such that it's possible to get valid splits
        y_train = (
            np.array([0, 0, 1, 1] + [0] * (n_samples - 4))
            if n_samples >= 4
            else np.array([0] * n_samples)
        )

        cv = LeavePOut(2)

    elif cv_type_param == "stratified":
        cv = (
            StratifiedKFold(
                n_splits=min(n_samples // 2, 3), shuffle=True, random_state=42
            )
            if n_samples >= 4
            else LeavePOut(1)
        )  # StratifiedKFold needs min 2 samples per class for n_splits > 1
        if n_samples < 4:  # Make y such that StratifiedKFold can work or LPO takes over
            y_train = np.array([0, 1] * (n_samples // 2) + [0] * (n_samples % 2))

    else:  # custom
        val_size = min(int(0.2 * n_samples), 100)
        train_size = n_samples - val_size
        cv = [(list(range(train_size)), list(range(train_size, n_samples)))]

    np.random.shuffle(y_train)  # Shuffle to make it more realistic

    new_splits = new_utils_training.get_splits(cv, X_train, y_train)
    old_splits = old_utils_training.get_splits(cv, X_train, y_train)

    assert len(new_splits) == len(old_splits)

    for (new_train_idx, new_val_idx), (old_train_idx, old_val_idx) in zip(
        new_splits, old_splits
    ):
        np.testing.assert_array_equal(new_train_idx, old_train_idx)
        np.testing.assert_array_equal(new_val_idx, old_val_idx)
        # Check if validation set contains both classes if y_train has both classes
        if len(np.unique(y_train)) == 2 and len(y_train[new_val_idx]) > 0:
            assert len(np.unique(y_train[new_val_idx])) == 2, (
                f"Val idx: {new_val_idx}, y_val: {y_train[new_val_idx]}"
            )


@pytest.mark.parametrize(
    "n_samples, n_features, penalty, parallel, return_classifier_flag",
    [
        (60, 10, "l2", False, False),
        (70, 15, "l1", False, True),
        (4, 5, "l2", False, False),  # Test small sample size (fallback to default C)
        (100, 20, "l2", True, True),  # Test parallel execution
    ],
)
def test_find_best_reg_comparison_and_behavior(
    n_samples: int,
    n_features: int,
    penalty: str,
    parallel: bool,
    return_classifier_flag: bool,
):
    """Test find_best_reg by comparing outputs of old and new versions."""
    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    X_test = np.random.rand(n_samples // 2, n_features)
    y_test = np.random.randint(0, 2, n_samples // 2)

    # Ensure y_train and y_test have both classes if n_samples is large enough
    if n_samples >= 2:
        # Ensure at least one of each class for robust testing if possible
        if n_samples >= 2 and len(np.unique(y_train)) < 2:
            y_train[0] = 0
            if n_samples > 1:
                y_train[1] = 1

    if len(y_test) >= 2:
        if len(np.unique(y_test)) < 2:
            y_test[0] = 0
            if len(y_test) > 1:
                y_test[1] = 1

    common_args = {
        "X_train": X_train.copy(),
        "y_train": y_train.copy(),
        "X_test": X_test.copy(),
        "y_test": y_test.copy(),
        "plot": False,
        "penalty": penalty,
        "seed": 42,
        "return_classifier": return_classifier_flag,
        "parallel": parallel,  # Pass parallel flag
        "n_jobs": 2 if parallel else -1,  # Use 2 jobs for parallel test, -1 otherwise
    }

    # Add n_jobs only if parallel is True for old function if it doesn't always accept it
    # However, inspecting the old code, it seems to handle n_jobs=-1 fine even when parallel=False.
    # So, we can keep it simple.

    output_new = new_utils_training.find_best_reg(**common_args)
    output_old = old_utils_training.find_best_reg(**common_args)

    if return_classifier_flag:
        metrics_new, clf_new = output_new
        metrics_old, clf_old = output_old
        assert isinstance(clf_new, LogisticRegression)
        assert isinstance(clf_old, LogisticRegression)
        # Compare some properties of the classifier if desired, e.g., C value
        if hasattr(clf_new, "C_") and hasattr(
            clf_old, "C_"
        ):  # if C_ exists (not always for default)
            if clf_new.C_ is not None and clf_old.C_ is not None:
                np.testing.assert_allclose(clf_new.C_, clf_old.C_, rtol=1e-5)
        elif clf_new.C is not None and clf_old.C is not None:  # C is param
            np.testing.assert_allclose(clf_new.C, clf_old.C, rtol=1e-5)

    else:
        metrics_new = output_new
        metrics_old = output_old

    assert metrics_new.keys() == metrics_old.keys()
    for key in metrics_new.keys():
        if (
            key == "val_auc" and n_samples <= 3
        ):  # val_auc might differ slightly if C is default due to no CV
            np.testing.assert_allclose(
                metrics_new[key], metrics_old[key], rtol=0.1
            )  # Looser tolerance
        else:
            np.testing.assert_allclose(metrics_new[key], metrics_old[key], rtol=1e-5)


@pytest.mark.parametrize(
    "n_samples, n_features, max_pca_comps_param",
    [
        (60, 10, 8),
        (70, 25, 20),
        (20, 5, 4),  # Test smaller dataset
        (
            5,
            3,
            2,
        ),  # Test very small dataset (PCA might use fewer components than requested)
    ],
)
def test_find_best_pcareg_comparison(
    n_samples: int, n_features: int, max_pca_comps_param: int
):
    """Test find_best_pcareg by comparing outputs of old and new versions."""
    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    X_test = np.random.rand(
        max(1, n_samples // 2), n_features
    )  # Ensure X_test is not empty
    y_test = np.random.randint(0, 2, max(1, n_samples // 2))

    # Ensure y_train and y_test have both classes if n_samples is large enough
    if n_samples > 1:
        y_train[0] = 0
        if n_samples > 2:
            y_train[1] = 1
    if len(y_test) > 1:
        y_test[0] = 0
        if len(y_test) > 2:
            y_test[1] = 1
    if n_samples == 1:
        y_train = np.array([0])  # single class if only 1 sample
    if len(y_test) == 1:
        y_test = np.array([0])

    common_args = {
        "X_train": X_train.copy(),
        "y_train": y_train.copy(),
        "X_test": X_test.copy(),
        "y_test": y_test.copy(),
        "plot": False,
        "max_pca_comps": max_pca_comps_param,
    }

    metrics_new = new_utils_training.find_best_pcareg(**common_args)
    metrics_old = old_utils_training.find_best_pcareg(**common_args)

    assert metrics_new.keys() == metrics_old.keys()
    for key in metrics_new.keys():
        is_small_or_constrained_data = (
            n_samples <= 5
            or n_features < max_pca_comps_param
            or X_train.shape[1] < max_pca_comps_param
        )
        if (key == "val_auc" or key == "test_auc") and is_small_or_constrained_data:
            np.testing.assert_allclose(
                metrics_new[key], metrics_old[key], rtol=0.2, atol=0.1
            )
        elif key.startswith("test_"):
            np.testing.assert_allclose(
                metrics_new[key], metrics_old[key], rtol=0.2, atol=0.1
            )
        else:
            np.testing.assert_allclose(metrics_new[key], metrics_old[key], rtol=1e-5)


@pytest.mark.parametrize(
    "n_samples, n_features, cv_folds_param",
    [
        (60, 10, 3),
        (30, 5, 2),  # Test smaller dataset and fewer folds
        # XGBoost RandomizedSearchCV needs at least 2 classes and enough samples for cv_folds.
        # (5, 3, 2) might be too small for reliable CV split if classes are imbalanced by chance
    ],
)
@pytest.mark.xfail(
    reason="Known compatibility issue between scikit-learn and XGBoost versions regarding estimator tags in RandomizedSearchCV"
)  # XFAIL Mark
def test_find_best_xgboost_comparison(
    n_samples: int, n_features: int, cv_folds_param: int
):
    """Test find_best_xgboost by comparing outputs of old and new versions."""
    X_train = np.random.rand(n_samples, n_features)
    y_train_list = ([0] * (n_samples // 2)) + ([1] * (n_samples - (n_samples // 2)))
    np.random.shuffle(y_train_list)
    y_train = np.array(y_train_list)

    X_test = np.random.rand(max(1, n_samples // 2), n_features)
    y_test_list = ([0] * (max(1, n_samples // 2) // 2)) + (
        [1] * (max(1, n_samples // 2) - (max(1, n_samples // 2) // 2))
    )
    np.random.shuffle(y_test_list)
    y_test = np.array(y_test_list)

    if n_samples < 4 or cv_folds_param >= n_samples // 2:
        pytest.skip(
            "Skipping XGBoost test for very small n_samples or too many cv_folds relative to samples"
        )
    if len(np.unique(y_train)) < 2:
        y_train[0] = 0
        y_train[1] = 1  # Force two classes
    if len(np.unique(y_test)) < 2 and len(y_test) > 1:
        y_test[0] = 0
        y_test[1] = 1  # Force two classes
    elif len(y_test) == 1:
        y_test[0] = y_train[0]

    common_args = {
        "X_train": X_train.copy(),
        "y_train": y_train.copy(),
        "X_test": X_test.copy(),
        "y_test": y_test.copy(),
        "plot": False,
        "cv_folds": cv_folds_param,
    }

    metrics_new = new_utils_training.find_best_xgboost(**common_args)
    metrics_old = old_utils_training.find_best_xgboost(**common_args)

    assert metrics_new.keys() == metrics_old.keys()
    for key in metrics_new.keys():
        if key == "val_auc":
            np.testing.assert_allclose(
                metrics_new[key], metrics_old[key], rtol=0.2, atol=0.1
            )
        elif key.startswith("test_"):
            np.testing.assert_allclose(
                metrics_new[key], metrics_old[key], rtol=0.15, atol=0.05
            )
        else:
            np.testing.assert_allclose(metrics_new[key], metrics_old[key], rtol=1e-5)


@pytest.mark.parametrize(
    "n_samples, n_features",
    [
        (60, 10),
        (25, 5),  # Test smaller dataset
        (
            10,
            3,
        ),  # Test very small dataset (KNN might behave differently with few neighbors)
    ],
)
def test_find_best_knn_comparison(n_samples: int, n_features: int):
    """Test find_best_knn by comparing outputs of old and new versions."""
    X_train = np.random.rand(n_samples, n_features)
    # Ensure at least two classes for y_train for StratifiedKFold and roc_auc to work reliably
    if n_samples >= 4:
        y_train_list = ([0] * (n_samples // 2)) + ([1] * (n_samples - (n_samples // 2)))
        np.random.shuffle(y_train_list)
        y_train = np.array(y_train_list)
    elif (
        n_samples > 0
    ):  # if less than 4, make it single class to avoid issues with too few samples for CV
        y_train = np.zeros(n_samples, dtype=int)
    else:
        y_train = np.array([])

    X_test = np.random.rand(max(1, n_samples // 2), n_features)
    y_test = np.random.randint(0, 2, max(1, n_samples // 2))
    if (
        len(y_test) > 0
        and len(np.unique(y_test)) < 2
        and len(y_train) > 0
        and len(np.unique(y_train)) == 2
    ):
        # If y_test is single class but y_train is multi-class, make y_test multi-class for AUC
        if len(y_test) >= 2:
            y_test[0] = 0
            y_test[1] = 1
        else:  # if y_test has only 1 sample, make it align with one of y_train classes
            y_test[0] = y_train[0]

    common_args = {
        "X_train": X_train.copy(),
        "y_train": y_train.copy(),
        "X_test": X_test.copy(),
        "y_test": y_test.copy(),
        "plot": False,
        # classification and binary are True by default in both versions
    }

    # Old version might have issues if y_train is single-class with some CV strategies
    # New version relies on get_splits which filters for valid CV folds
    # We are trying to create data that is valid for both
    if n_samples < 4 or len(np.unique(y_train)) < 2:
        # Skip test for very small or single-class y_train where behavior might diverge
        # or internal CV might fail in old version or lead to vacuous results
        pytest.skip(
            "Skipping KNN test for very small/single-class y_train due to CV instability"
        )

    metrics_new = new_utils_training.find_best_knn(**common_args)
    metrics_old = old_utils_training.find_best_knn(**common_args)

    assert metrics_new.keys() == metrics_old.keys()
    for key in metrics_new.keys():
        # KNN can be sensitive, especially AUC with small N or near-chance performance.
        # F1/accuracy might also vary slightly based on tie-breaking or exact neighbor sets.
        if key == "val_auc" or key == "test_auc":
            np.testing.assert_allclose(
                metrics_new[key], metrics_old[key], rtol=0.1, atol=0.05
            )
        elif key.startswith("test_"):
            np.testing.assert_allclose(
                metrics_new[key], metrics_old[key], rtol=0.1, atol=0.05
            )
        else:
            np.testing.assert_allclose(metrics_new[key], metrics_old[key], rtol=1e-5)


@pytest.mark.parametrize(
    "n_samples, n_features",
    [
        (60, 10),
        (30, 5),  # Test smaller dataset
        # MLP with RandomizedSearchCV can also be sensitive to small N / few iterations.
    ],
)
def test_find_best_mlp_comparison(n_samples: int, n_features: int):
    """Test find_best_mlp by comparing outputs of old and new versions."""
    X_train = np.random.rand(n_samples, n_features)
    y_train_list = ([0] * (n_samples // 2)) + ([1] * (n_samples - (n_samples // 2)))
    np.random.shuffle(y_train_list)
    y_train = np.array(y_train_list)

    X_test = np.random.rand(max(1, n_samples // 2), n_features)
    y_test_list = ([0] * (max(1, n_samples // 2) // 2)) + (
        [1] * (max(1, n_samples // 2) - (max(1, n_samples // 2) // 2))
    )
    np.random.shuffle(y_test_list)
    y_test = np.array(y_test_list)

    # Ensure y_train and y_test have at least two classes for RandomizedSearchCV and AUC
    # MLP has internal CV in RandomizedSearchCV, which needs enough samples per class per split.
    # Default CV in RandomizedSearchCV is 5-fold.
    min_samples_for_cv = 10  # Heuristic, as it depends on class balance
    if n_samples < min_samples_for_cv:
        pytest.skip(
            "Skipping MLP test for very small n_samples due to CV instability in RandomizedSearchCV"
        )
    if len(np.unique(y_train)) < 2:
        y_train[0] = 0
        y_train[1] = 1  # Force two classes
    if len(np.unique(y_test)) < 2 and len(y_test) > 1:
        y_test[0] = 0
        y_test[1] = 1  # Force two classes
    elif len(y_test) == 1:
        y_test[0] = y_train[0]

    common_args = {
        "X_train": X_train.copy(),
        "y_train": y_train.copy(),
        "X_test": X_test.copy(),
        "y_test": y_test.copy(),
        "plot": False,
        # classification and binary are True by default
    }

    metrics_new = new_utils_training.find_best_mlp(**common_args)
    metrics_old = old_utils_training.find_best_mlp(**common_args)

    assert metrics_new.keys() == metrics_old.keys()
    for key in metrics_new.keys():
        # MLP training involves randomness (initialization, and RandomizedSearchCV if used for HP tuning).
        # Results can vary. We check for approximate equality.
        if key == "val_auc":  # val_auc from RandomizedSearchCV best_score_
            np.testing.assert_allclose(
                metrics_new[key], metrics_old[key], rtol=0.2, atol=0.1
            )
        elif key.startswith("test_"):
            np.testing.assert_allclose(
                metrics_new[key], metrics_old[key], rtol=0.15, atol=0.1
            )
        else:
            np.testing.assert_allclose(metrics_new[key], metrics_old[key], rtol=1e-5)
