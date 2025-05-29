import matplotlib.pyplot as plt
import numpy as np
import xgboost
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeavePOut, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False

# TRAINING UTILS


def get_cv(X_train):
    n_samples = X_train.shape[0]
    if n_samples <= 12:
        cv = LeavePOut(2)
    elif n_samples < 128:
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    else:
        val_size = min(
            int(0.2 * n_samples), 100
        )  # 20% of data or max 100 samples for validation
        train_size = n_samples - val_size
        cv = [(list(range(train_size)), list(range(train_size, n_samples)))]
    return cv


def get_splits(cv, X_train, y_train):
    # Generate only splits where validation has at least one of each class
    if hasattr(cv, "split"):
        splits = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            if (
                len(np.unique(y_train[val_idx])) == 2
            ):  # Ensures both classes in the validation set
                splits.append((train_idx, val_idx))
    else:
        splits = cv  # For predefined list-based splits

    return splits


def find_best_reg(
    X_train,
    y_train,
    X_test,
    y_test,
    plot=False,
    n_jobs=-1,
    parallel=False,
    penalty="l2",
    seed=1,
    return_classifier=False,
):
    # Determine cross-validation strategy
    best_C = None
    if (
        X_train.shape[0] > 3
    ):  # cannot reliably to cross val. just going with default parameters
        cv = get_cv(X_train)

        Cs = np.logspace(5, -5, 10)
        avg_scores = []

        def evaluate_fold(C, train_index, val_index):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

            if penalty == "l1":
                model = LogisticRegression(
                    C=C, penalty="l1", solver="saga", random_state=seed, max_iter=1000
                )
            else:
                model = LogisticRegression(C=C, random_state=seed, max_iter=1000)
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            return roc_auc_score(y_fold_val, y_pred_proba)

        for C in Cs:
            splits = get_splits(cv, X_train, y_train)
            # Parallelize the inner loop using joblib
            if parallel:
                fold_scores = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_fold)(C, train_index, val_index)
                    for train_index, val_index in splits
                )
            else:
                fold_scores = [
                    evaluate_fold(C, train_index, val_index)
                    for train_index, val_index in splits
                ]

            avg_scores.append(np.mean(fold_scores))

        # Find the index of the best score (max for classification, min for regression)
        best_C_index = np.argmax(avg_scores)
        best_C = Cs[best_C_index]

    # Train final model with best C
    metrics = {}

    if best_C is not None:
        if penalty == "l1":
            final_model = LogisticRegression(
                C=best_C, penalty="l1", solver="saga", random_state=seed, max_iter=1000
            )
        else:
            final_model = LogisticRegression(C=best_C, random_state=seed, max_iter=1000)
    else:
        if penalty == "l1":
            final_model = LogisticRegression(
                penalty="l1", solver="saga", random_state=seed, max_iter=1000
            )
        else:
            final_model = LogisticRegression(random_state=seed, max_iter=1000)
    # Shuffle X_train and y_train based on seed
    rng = np.random.RandomState(seed)
    shuffle_idx = rng.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)
    metrics["test_f1"] = f1_score(y_test, y_test_pred, average="weighted")
    metrics["test_acc"] = accuracy_score(y_test, y_test_pred)
    # Use predict_proba to get probability estimates for the positive class (class 1)
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    metrics["test_auc"] = roc_auc_score(y_test, y_test_pred_proba)
    if best_C is not None:
        metrics["val_auc"] = np.max(avg_scores)
    else:
        metrics["val_auc"] = roc_auc_score(
            y_train, final_model.predict_proba(X_train)[:, 1]
        )
    if plot:
        plt.semilogx(Cs, avg_scores)
        plt.xlabel("Inverse of Regularization Strength (C)")
        met1_name, met2_name = "auc", "auc"
        plt.ylabel(f"{met1_name} on validation data")
        plt.title(
            f"{'Logistic Regression'} Performance vs Regularization\nBest C = {best_C:.5f}; {met2_name} = {metrics[met2_name]:.2f}"
        )
        plt.show()
    if return_classifier:
        return metrics, final_model
    return metrics


def find_best_pcareg(X_train, y_train, X_test, y_test, plot=False, max_pca_comps=100):
    # Standardize the data
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Determine the range of PCA dimensions to try
    max_components = min(X_train.shape[0], X_train.shape[1], max_pca_comps)
    pca_dimensions = np.unique(
        np.logspace(0, np.log2(max_components), num=10, base=2, dtype=int)
    )

    # Fit PCA for the maximum number of components
    pca = PCA(n_components=max_components)
    X_combined_pca_full = pca.fit_transform(X_combined_scaled)

    best_score = -float("inf")
    best_model = None
    best_n_components = None
    metrics = {}
    if X_combined_pca_full.shape[0] > 3:
        cv = get_cv(X_train)
        scores = []

        for n_components in pca_dimensions:
            X_pca = X_combined_pca_full[:, :n_components]
            fold_scores = []
            splits = get_splits(cv, X_train, y_train)
            for train_index, val_index in splits:
                X_fold_train, X_fold_val = X_pca[train_index], X_pca[val_index]
                y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

                model = LogisticRegression(random_state=42, max_iter=1000)

                model.fit(X_fold_train, y_fold_train)

                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                fold_scores.append(roc_auc_score(y_fold_val, y_pred_proba))

            avg_score = np.mean(fold_scores)
            scores.append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_model = LogisticRegression(random_state=42, max_iter=1000).fit(
                    X_pca, y_train
                )
                best_n_components = n_components
                metrics["val_auc"] = best_score
    else:
        best_n_components = X_combined_pca_full.shape[0]
        best_model = LogisticRegression(random_state=42, max_iter=1000).fit(
            X_combined_pca_full, y_train
        )
        y_train_pred_proba = best_model.predict_proba(X_combined_pca_full)[:, 1]
        metrics["val_auc"] = roc_auc_score(y_train, y_train_pred_proba)

    # Transform test data using PCA
    X_test_pca = pca.transform(X_test_scaled)[:, :best_n_components]

    # Make predictions on test set
    y_test_pred = best_model.predict(X_test_pca)

    metrics["test_f1"] = f1_score(y_test, y_test_pred, average="weighted")
    metrics["test_acc"] = accuracy_score(y_test, y_test_pred)
    # Use predict_proba to get probability estimates for the positive class (class 1)
    y_test_pred_proba = best_model.predict_proba(X_test_pca)[:, 1]
    metrics["test_auc"] = roc_auc_score(y_test, y_test_pred_proba)

    if plot and X_combined_pca_full.shape[0] > 3:
        plt.semilogx(pca_dimensions, scores)
        plt.xlabel("Number of PCA Components")
        plt.xscale("log", base=2)
        met1_name, met2_name = "auc", "auc"
        plt.ylabel(f"{met1_name} on validation data")
        plt.title(
            f"Best PCA dimension: {best_n_components}, {met2_name} = {metrics[met2_name]:.2f}"
        )
        plt.show()

    return metrics


def find_best_knn(
    X_train, y_train, X_test, y_test, classification=True, binary=True, plot=False
):
    # Standardize the data
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Determine the range of k values to try
    max_k = min(20, len(X_train) - 1)
    if max_k < 1:
        # Not enough samples for KNN
        return {
            "test_f1": 0,
            "test_acc": 0,
            "test_auc": 0.5,
            "val_auc": 0.5,  # Or some other default/error indicator
        }

    k_values = list(range(1, max_k + 1, 2))  # Try odd k values up to max_k

    best_score = -float("inf")
    best_model = None
    best_k = None
    metrics = {}
    if X_combined_scaled.shape[0] > 3:
        cv = get_cv(X_train)
        scores = []

        def evaluate_fold(k, train_index, val_index):
            X_fold_train, X_fold_val = (
                X_combined_scaled[train_index],
                X_combined_scaled[val_index],
            )
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            return roc_auc_score(y_fold_val, y_pred_proba)

        for k in k_values:
            splits = get_splits(cv, X_combined_scaled, y_train)
            fold_scores = Parallel(n_jobs=-1)(
                delayed(evaluate_fold)(k, train_index, val_index)
                for train_index, val_index in splits
            )
            avg_score = np.mean(fold_scores)
            scores.append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
                metrics["val_auc"] = best_score
    else:
        best_k = 1
        metrics["val_auc"] = 0.5  # Placeholder

    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_combined_scaled, y_train)
    y_test_pred = best_model.predict(X_test_scaled)

    metrics["test_f1"] = f1_score(y_test, y_test_pred, average="weighted")
    metrics["test_acc"] = accuracy_score(y_test, y_test_pred)
    # Use predict_proba to get probability estimates for the positive class (class 1)
    y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics["test_auc"] = roc_auc_score(y_test, y_test_pred_proba)

    if plot and X_combined_scaled.shape[0] > 3:
        plt.plot(k_values, scores)
        plt.xlabel("Number of Neighbors (k)")
        met1_name, met2_name = "auc", "auc"
        plt.ylabel(f"{met1_name} on validation data")
        plt.title(f"Best K: {best_k}, {met2_name} = {metrics[met2_name]:.2f}")
        plt.show()

    return metrics


def find_best_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    classification=True,
    binary=True,
    plot=False,
    cv_folds=3,
):
    # Check if X_train has less than 3 samples
    if X_train.shape[0] < 3:
        # Not enough samples for XGBoost
        return {
            "test_f1": 0,
            "test_acc": 0,
            "test_auc": 0.5,
            "val_auc": 0.5,  # Or some other default/error indicator
        }

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    xgb_model = xgboost.XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )

    # Use RandomizedSearchCV to find the best hyperparameters
    # Use a smaller number of iterations (n_iter) for faster search if needed
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_grid,
        n_iter=10,  # Number of parameter settings that are sampled
        scoring="roc_auc",
        cv=cv_folds,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train_scaled, y_train)

    best_model = random_search.best_estimator_

    metrics = {}
    metrics["val_auc"] = random_search.best_score_

    # Make predictions on test set
    y_test_pred = best_model.predict(X_test_scaled)
    metrics["test_f1"] = f1_score(y_test, y_test_pred, average="weighted")
    metrics["test_acc"] = accuracy_score(y_test, y_test_pred)
    # Use predict_proba to get probability estimates for the positive class (class 1)
    y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics["test_auc"] = roc_auc_score(y_test, y_test_pred_proba)

    # Optionally, plot performance (if applicable)
    if plot:
        # Plotting for XGBoost might involve feature importance or other specific plots
        pass

    return metrics


def find_best_mlp(
    X_train, y_train, X_test, y_test, classification=True, binary=True, plot=False
):
    # Combine train and validation sets
    X_combined = X_train
    y_combined = y_train

    # Standardize the data
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        "hidden_layer_sizes": [
            (50,),
            (100,),
            (50, 50),
            (100, 50),
            (20, 20, 20),
        ],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
    }

    mlp_model = MLPClassifier(random_state=42, max_iter=300)  # Increased max_iter

    # Use RandomizedSearchCV to find the best hyperparameters
    random_search = RandomizedSearchCV(
        mlp_model,
        param_distributions=param_grid,
        n_iter=10,  # Number of parameter settings that are sampled
        scoring="roc_auc",
        cv=3,  # Using 3-fold CV for MLP due to potentially longer training times
        random_state=42,
        n_jobs=-1,  # Use parallel processing
    )

    random_search.fit(X_combined_scaled, y_combined)

    best_model = random_search.best_estimator_

    metrics = {}
    metrics["val_auc"] = random_search.best_score_

    # Make predictions on test set
    y_test_pred = best_model.predict(X_test_scaled)
    metrics["test_f1"] = f1_score(y_test, y_test_pred, average="weighted")
    metrics["test_acc"] = accuracy_score(y_test, y_test_pred)
    # Use predict_proba to get probability estimates for the positive class (class 1)
    y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics["test_auc"] = roc_auc_score(y_test, y_test_pred_proba)

    # Optionally, plot performance (if applicable)
    if plot:
        # Plotting for MLP might involve learning curves or other specific plots
        pass

    return metrics


# example usage
# if __name__ == "__main__":
#     # Example usage for testing find_best_reg
#     # Need to load a model first to use get_model_activations_for_dataset
#     # from transformer_lens import HookedTransformer
#     # model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
#     # model.cfg.model_name = "gpt2-small" # Ensure model_name is set
#
#     # X_train, y_train, X_test, y_test = get_model_activations_for_dataset(
#     #     model=model,
#     #     dataset_name="10_financial_transactions", # This is a numbered_dataset_tag, ensure function handles it or use Dataset Tag
#     #     layer_idx=9,
#     #     device="cpu",
#     #     # num_train_samples_target=100 # get_model_activations_for_dataset uses this for scarcity
#     # )
#     pass # Commenting out for now as it needs a loaded model.
#
#     # metrics_reg = find_best_reg(X_train, y_train, X_test, y_test, plot=True)
#     # print(f"Regression Metrics: {metrics_reg}")

#     # Example usage for testing find_best_pcareg
#     # metrics_pcareg = find_best_pcareg(X_train, y_train, X_test, y_test, plot=True)
#     # print(f"PCA + Regression Metrics: {metrics_pcareg}")

#     # Example usage for testing find_best_knn
#     # metrics_knn = find_best_knn(X_train, y_train, X_test, y_test, plot=True)
#     # print(f"KNN Metrics: {metrics_knn}")

#     # Example usage for testing find_best_xgboost
#     # metrics_xgboost = find_best_xgboost(X_train, y_train, X_test, y_test, plot=True)
#     # print(f"XGBoost Metrics: {metrics_xgboost}")

#     # Example usage for testing find_best_mlp
#     # metrics_mlp = find_best_mlp(X_train, y_train, X_test, y_test, plot=True)
#     # print(f"MLP Metrics: {metrics_mlp}")
