# %%
import numpy as np
import pandas as pd
import torch
from sae_lens import SAE

from sae_probes.utils_data import get_xy_glue, get_xy_OOD, get_yvals


@torch.no_grad()
def get_sae_features(
    sae: SAE, model_activations: torch.Tensor, device: str | torch.device | None = None
) -> torch.Tensor:
    """
    Computes SAE features from model activations.

    Args:
        sae: The Sparse Autoencoder model.
        model_activations: A tensor of model activations.
        device: The device to run the SAE on. If None, uses sae.device.

    Returns:
        A tensor of SAE features.
    """
    if device is None:
        device = sae.device
    sae.to(device)
    sae.eval()  # Ensure SAE is in evaluation mode

    # Assuming model_activations is already on the correct device or needs to be moved
    # and is of shape (n_examples, d_model_act)
    # The SAE expects input of shape (batch_size, d_in)
    # If model_activations is a large file, consider processing in batches if memory is an issue

    # Ensure activations are on the same device as the SAE
    model_activations = model_activations.to(device)

    # Get SAE features
    # _, features, _, _, _, _ = sae(model_activations) # Old SAE Lens API
    # New SAE Lens API might be different, typically sae.encode(model_activations)
    if hasattr(sae, "encode"):
        features = sae.encode(model_activations)
    elif hasattr(sae, "forward") and not isinstance(
        sae.forward, torch.nn.Module
    ):  # check if it's the actual forward method
        # This is a bit heuristic, might need adjustment based on SAE implementation details
        # Typical SAE forward returns multiple values, features are often the second element
        # Or it could be the direct output if it's just an encoder.
        # Assuming standard SAE Lens forward: H_res, H_forward_pass, H_sae_out, H_sae_reconstructed, mse_loss_reconstruction, L_sparsity
        output = sae(model_activations)
        if isinstance(output, tuple) and len(output) > 1:
            features = output[1]  # Often H_forward_pass for SAE Lens
            if features.shape[-1] != sae.cfg.d_sae:  # Check if it's the actual features
                # Fallback if the second element is not features (e.g. if only one output)
                if hasattr(output[0], "shape") and output[0].shape[-1] == sae.cfg.d_sae:
                    features = output[0]
                else:  # if it's a more complex output. We assume the primary output for single output.
                    # This branch might need more sophisticated handling if SAEs have varied output structures.
                    # For now, we'll assume the first tensor with correct dimension is the feature.
                    found_features = False
                    for item in output:
                        if (
                            isinstance(item, torch.Tensor)
                            and item.shape[-1] == sae.cfg.d_sae
                        ):
                            features = item
                            found_features = True
                            break
                    if not found_features:
                        raise ValueError(
                            "Could not automatically determine SAE features from SAE output. Please ensure your SAE has an 'encode' method or a standard forward pass output."
                        )
        else:  # If SAE forward returns a single tensor, assume it's the features
            features = output
    else:
        raise AttributeError(
            "SAE object must have an 'encode' method or a callable 'forward' method."
        )

    return features.cpu()  # Move to CPU as original functions did


def get_xy_OOD_sae(
    sae: SAE,
    model_acts_train: torch.Tensor,
    model_acts_test: torch.Tensor,
    dataset_name: str,  # Renamed from 'dataset' to avoid confusion
    k: int = 128,
    return_indices: bool = False,
    num_train: int = 1024,
    # device parameter added for SAE processing
    device: str | torch.device | None = None,
):
    """
    Prepares training and testing data (X, y) for Out-of-Distribution experiments
    using SAE features derived from provided model activations.
    """
    _, y_test = get_xy_OOD(dataset_name)  # y_test remains the same
    # y_train is loaded based on original model activations/dataset properties
    # Assuming model_name was used by get_xyvals to fetch correct labels or metadata.
    # If get_xyvals doesn't actually depend on model_name for label generation, it can be removed there too.
    # For now, we keep model_name for get_xyvals if it's strictly for label generation.
    # Let's assume for now we need a model_name for get_xyvals, or it needs refactoring too.
    # We will temporarily hardcode a placeholder or require it if get_xyvals truly needs it.
    # This highlights a dependency that might need further untangling.
    # For the purpose of this refactoring, we will assume 'model_name_for_labels' is a new param if needed by get_xyvals.
    # However, inspecting get_xyvals, it seems it loads model activations itself if not provided.
    # This is problematic. get_xyvals should ideally just return labels if activations are handled elsewhere.
    # For now, we assume y_train can be obtained without model_name here if X_train (model_acts_train) is given.
    # This part needs careful review of utils_data.get_xyvals.
    # For now, let's assume y_train can be fetched with dataset_name only for label purposes.
    # If get_xyvals is about labels for model_acts_train, then model_acts_train already implies the source.

    # Re-evaluating: get_xyvals is likely loading labels based on dataset name, and potentially model name
    # if labels are tied to specific model activations.
    # If we pass model_acts_train, its corresponding y_train must also be passed or loaded consistently.
    # Let's assume y_train needs to be passed alongside model_acts_train, or fetched by a function
    # that *only* fetches labels based on dataset_name, etc.
    # To simplify here, we'll assume get_xyvals can provide y_train without loading X_train itself.

    # If get_xyvals also loads X, we have a problem. Let's assume it can just give y.
    # Check `utils_data.get_xyvals` structure. If it loads activations, it needs to change.
    # For now, to proceed:
    y_train = get_yvals(dataset_name)

    X_train_sae_features = get_sae_features(sae, model_acts_train, device=device)
    X_test_sae_features = get_sae_features(sae, model_acts_test, device=device)

    # Balance and select top k features (logic remains similar to original)
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    # Ensure num_train // 2 does not exceed available samples for any class
    max_pos_samples = len(pos_indices)
    max_neg_samples = len(neg_indices)

    num_pos_to_select = min(num_train // 2, max_pos_samples)
    num_neg_to_select = min(num_train // 2, max_neg_samples)

    pos_selected = np.random.choice(pos_indices, num_pos_to_select, replace=False)
    neg_selected = np.random.choice(neg_indices, num_neg_to_select, replace=False)

    selected_indices = np.concatenate([pos_selected, neg_selected])
    # Shuffle selected_indices before applying to X_train_sae_features and y_train
    np.random.shuffle(selected_indices)

    X_train_balanced = X_train_sae_features[selected_indices]
    y_train_balanced = y_train[selected_indices]

    if X_train_balanced.shape[0] == 0:  # Handle case with no samples after balancing
        # This can happen if a class has 0 samples or too few samples.
        # Return empty tensors or handle as an error, depending on desired behavior.
        # For now, returning empty tensors of appropriate shape.
        empty_features_shape_train = (0, k if k > 0 else X_train_sae_features.shape[1])
        empty_features_shape_test = (
            X_test_sae_features.shape[0],
            k if k > 0 else X_test_sae_features.shape[1],
        )
        if return_indices:
            return (
                torch.empty(empty_features_shape_train),
                torch.empty(0, dtype=y_train.dtype),
                torch.empty(empty_features_shape_test),
                y_test,
                torch.empty(0, dtype=torch.long),
            )
        return (
            torch.empty(empty_features_shape_train),
            torch.empty(0, dtype=y_train.dtype),
            torch.empty(empty_features_shape_test),
            y_test,
        )

    X_train_diff = X_train_balanced[y_train_balanced == 1].mean(
        dim=0
    ) - X_train_balanced[y_train_balanced == 0].mean(dim=0)
    # Handle cases where one class might be empty after balancing, resulting in NaNs for X_train_diff
    if torch.isnan(X_train_diff).any():
        # If diff is NaN (e.g. one class is empty), cannot sort. Fallback: use all features or error.
        # For now, let's use all features if k is not specified, or first k if k is specified.
        # This might not be ideal, but prevents a crash.
        print(
            f"Warning: NaN encountered in X_train_diff for dataset {dataset_name}. This might be due to empty classes after balancing."
        )
        if k > 0 and k <= X_train_balanced.shape[1]:
            top_by_average_diff = torch.arange(k)
        else:  # use all features
            top_by_average_diff = torch.arange(X_train_balanced.shape[1])
    else:
        sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
        top_by_average_diff = sorted_indices[
            : k if k > 0 else X_train_balanced.shape[1]
        ]  # if k=0, use all features

    X_train_filtered = X_train_balanced[:, top_by_average_diff]
    X_test_filtered = X_test_sae_features[:, top_by_average_diff]

    if return_indices:
        return (
            X_train_filtered,
            y_train_balanced,
            X_test_filtered,
            y_test,
            top_by_average_diff,
        )
    return X_train_filtered, y_train_balanced, X_test_filtered, y_test


def get_xy_glue_sae(
    sae: SAE,
    model_acts_train: torch.Tensor,
    model_acts_test: torch.Tensor,  # Assuming GLUE test set also has corresponding model activations
    dataset_name: str,  # e.g., "87_glue_cola"
    toget_y_test: str = "ensemble",  # Parameter for get_xy_glue for y_test
    k: int = 128,
    # device parameter added for SAE processing
    device: str | torch.device | None = None,
    num_train_samples_per_class: int = 512,  # from original hardcoding
):
    """
    Prepares training and testing data (X, y) for GLUE task experiments
    using SAE features derived from provided model activations.
    """
    _, y_test = get_xy_glue(toget=toget_y_test)
    # Similar concern for get_xyvals as in get_xy_OOD_sae
    y_train = get_yvals(dataset_name)

    X_train_sae_features = get_sae_features(sae, model_acts_train, device=device)
    X_test_sae_features = get_sae_features(sae, model_acts_test, device=device)

    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    max_pos_samples = len(pos_indices)
    max_neg_samples = len(neg_indices)

    num_pos_to_select = min(num_train_samples_per_class, max_pos_samples)
    num_neg_to_select = min(num_train_samples_per_class, max_neg_samples)

    pos_selected = np.random.choice(pos_indices, num_pos_to_select, replace=False)
    neg_selected = np.random.choice(neg_indices, num_neg_to_select, replace=False)

    selected_indices = np.concatenate([pos_selected, neg_selected])
    np.random.shuffle(selected_indices)

    X_train_balanced = X_train_sae_features[selected_indices]
    y_train_balanced = y_train[selected_indices]

    if X_train_balanced.shape[0] == 0:  # Handle case with no samples
        empty_features_shape_train = (0, k if k > 0 else X_train_sae_features.shape[1])
        empty_features_shape_test = (
            X_test_sae_features.shape[0],
            k if k > 0 else X_test_sae_features.shape[1],
        )
        return (
            torch.empty(empty_features_shape_train),
            torch.empty(0, dtype=y_train.dtype),
            torch.empty(empty_features_shape_test),
            y_test,
        )

    X_train_diff = X_train_balanced[y_train_balanced == 1].mean(
        dim=0
    ) - X_train_balanced[y_train_balanced == 0].mean(dim=0)

    if torch.isnan(X_train_diff).any():
        print(
            f"Warning: NaN encountered in X_train_diff for GLUE dataset {dataset_name}. This might be due to empty classes after balancing."
        )
        if k > 0 and k <= X_train_balanced.shape[1]:
            top_by_average_diff = torch.arange(k)
        else:  # use all features
            top_by_average_diff = torch.arange(X_train_balanced.shape[1])
    else:
        sorted_indices = torch.argsort(torch.abs(X_train_diff), descending=True)
        if k == 1:  # Special handling from original code
            # Original code used sorted_indices[2:3] - this implies at least 3 features exist.
            # And that we want the 3rd feature if k=1 for some reason. This is odd.
            # Let's assume if k=1, they want the top 1, unless there's a strong reason for index 2.
            # Reverting to original logic for now: top_by_average_diff = sorted_indices[2:3]
            # This needs clarification. If it's a bug, it should be fixed to sorted_indices[0:1] or sorted_indices[:1]
            if len(sorted_indices) >= 3:
                top_by_average_diff = sorted_indices[2:3]
                print(f"Using feature at index 2 for k=1: {top_by_average_diff}")
            elif len(sorted_indices) > 0:  # Fallback if less than 3 features
                top_by_average_diff = sorted_indices[:1]
                print(
                    f"Warning: Less than 3 features available for GLUE k=1 special case. Using top feature: {top_by_average_diff}"
                )
            else:  # No features available
                top_by_average_diff = torch.empty(0, dtype=torch.long)
                print("Warning: No features available for GLUE k=1 special case.")
        else:
            top_by_average_diff = sorted_indices[
                : k if k > 0 else X_train_balanced.shape[1]
            ]

    if (
        top_by_average_diff.numel() == 0 and k > 0
    ):  # If no indices selected but k was requested
        # This can happen if X_train_balanced was empty or top_by_average_diff ended up empty.
        # Return empty tensors for features.
        X_train_filtered = torch.empty(
            (X_train_balanced.shape[0], 0), device=X_train_balanced.device
        )
        X_test_filtered = torch.empty(
            (X_test_sae_features.shape[0], 0), device=X_test_sae_features.device
        )
    elif (
        top_by_average_diff.numel() == 0 and k == 0
    ):  # request all features but no features available
        X_train_filtered = X_train_balanced
        X_test_filtered = X_test_sae_features
    else:
        X_train_filtered = X_train_balanced[:, top_by_average_diff]
        X_test_filtered = X_test_sae_features[:, top_by_average_diff]

    return X_train_filtered, y_train_balanced, X_test_filtered, y_test


def get_grammar_feature_examples(
    sae: SAE,  # Added
    model_acts_train: torch.Tensor,  # Added
    model_acts_test: torch.Tensor,  # Added (for X)
    glue_dataset_name: str = "87_glue_cola",  # Added
    device: str | torch.device | None = None,  # Added
):
    # We need X_test_sae_features corresponding to the original 'X' in the function.
    # The original function called get_xy_glue_sae(k=1) and used its X_test_filtered output as 'X'.
    # So we replicate that here.
    _, _, X_test_sae_k1, _ = get_xy_glue_sae(
        sae=sae,
        model_acts_train=model_acts_train,  # These might be dummy if only X_test_sae_k1 is used from test acts
        model_acts_test=model_acts_test,
        dataset_name=glue_dataset_name,
        k=1,  # As per original usage
        device=device,
    )
    X = X_test_sae_k1  # This is X_test_filtered with k=1

    # Read prompts and get their lengths
    # This path is hardcoded, consider making it a parameter or ensuring it's always available.
    try:
        df = pd.read_csv("results/investigate/87_glue_cola_investigate.csv")
        prompts = df["prompt"].tolist()
    except FileNotFoundError:
        print(
            "Warning: Could not load prompts from results/investigate/87_glue_cola_investigate.csv for get_grammar_feature_examples. Skipping print."
        )
        return

    _, yog = get_xy_glue(toget="original_target")
    _, yens = get_xy_glue(toget="ensemble")

    # Ensure X, yog, yens, and prompts are of compatible lengths.
    # This depends on how get_xy_glue aligns with model_acts_test and the CSV.
    # Assuming the CSV and get_xy_glue outputs align with the examples in model_acts_test.
    min_len = min(len(prompts), X.shape[0], len(yog), len(yens))
    if (
        X.shape[0] != min_len
        or len(yog) != min_len
        or len(yens) != min_len
        or len(prompts) != min_len
    ):
        print(
            f"Warning: Length mismatch in get_grammar_feature_examples. Truncating to smallest common length: {min_len}"
        )
        X = X[:min_len]
        yog = yog[:min_len]
        yens = yens[:min_len]
        prompts = prompts[:min_len]

    if X.shape[0] == 0:
        print(
            "Warning: No data to process in get_grammar_feature_examples (X is empty)."
        )
        return
    if (
        X.shape[1] == 0
    ):  # k=1 but no features selected, e.g. from the k=1 warning above.
        print(
            "Warning: No feature data (X has 0 columns) in get_grammar_feature_examples. Skipping print."
        )
        return

    # Get indices where original target is 1
    # Ensure yog is a tensor for boolean indexing if it's not already
    yog_tensor = torch.tensor(yog) if not isinstance(yog, torch.Tensor) else yog
    valid_indices = torch.where(yog_tensor == 1)[0]

    if len(valid_indices) == 0:
        print(
            "Warning: No examples with original target == 1 in get_grammar_feature_examples. Skipping print."
        )
        return

    X_valid = X[valid_indices]
    if X_valid.shape[0] == 0:  # Should be caught by previous check, but as a safeguard.
        print(
            "Warning: X_valid is empty in get_grammar_feature_examples. Skipping print."
        )
        return

    # Get indices of top 5 highest feature values among valid examples
    # X_valid[:, 0] assumes the k=1 feature is the first (and only) column
    top_5_relative_idx = torch.argsort(X_valid[:, 0], descending=True)[:5]
    top_5_idx = valid_indices[top_5_relative_idx]

    # Print table
    print("\nPrompt | Original | Ensemble | Feature Fired")
    print("-" * 50)
    for (
        idx_val
    ) in top_5_idx:  # Changed loop variable name from idx to idx_val to avoid conflict
        idx = idx_val.item()  # convert tensor to int for indexing lists
        print(
            f"{prompts[idx]:<30} | {yog[idx]:<8} | {yens[idx]:<8} | {X[idx].item():.2f}"
        )


# Removed get_sae_layers and commented out script execution lines
