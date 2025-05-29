# %%
import numpy as np
import torch
from sae_lens import SAE  # type: ignore


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
        output = sae.forward(model_activations)
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
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int = 128,
    return_indices: bool = False,
    num_train: int = 1024,
    device: str | torch.device | None = None,
):
    """
    Prepares training and testing data (X, y) for Out-of-Distribution experiments
    using SAE features derived from provided model activations.
    y_train and y_test are now direct inputs.
    """
    X_train_sae_features = get_sae_features(sae, model_acts_train, device=device)
    X_test_sae_features = get_sae_features(sae, model_acts_test, device=device)

    # Balance and select top k features (logic remains similar to original)
    if len(y_train) == 0:  # Handle empty y_train case early
        # This means model_acts_train was likely also empty or y_train couldn't be formed.
        empty_features_shape_train = (
            0,
            k
            if k > 0
            else X_train_sae_features.shape[1]
            if X_train_sae_features.numel() > 0
            else (0, sae.cfg.d_sae if k == 0 else k),
        )
        # Adjust X_test_sae_features based on k, even if training data is empty
        if k > 0 and X_test_sae_features.shape[1] >= k:
            X_test_filtered = X_test_sae_features[
                :, :k
            ]  # Fallback: take first k if no training data for diff
        elif k == 0:  # use all features
            X_test_filtered = X_test_sae_features
        else:  # k > d_sae or k > available
            X_test_filtered = X_test_sae_features  # Take all available if k is too large for test set features

        empty_indices = torch.empty(0, dtype=torch.long)

        if return_indices:
            return (
                torch.empty(
                    empty_features_shape_train,
                    device=X_train_sae_features.device,
                    dtype=X_train_sae_features.dtype,
                ),
                np.array([], dtype=y_train.dtype),
                X_test_filtered.to(
                    X_test_sae_features.device, dtype=X_test_sae_features.dtype
                ),  # ensure X_test is on original device
                y_test,
                empty_indices,
            )
        return (
            torch.empty(
                empty_features_shape_train,
                device=X_train_sae_features.device,
                dtype=X_train_sae_features.dtype,
            ),
            np.array([], dtype=y_train.dtype),
            X_test_filtered.to(
                X_test_sae_features.device, dtype=X_test_sae_features.dtype
            ),
            y_test,
        )

    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    if (
        len(pos_indices) == 0 or len(neg_indices) == 0
    ):  # If only one class (or no classes) in y_train
        # Cannot balance or compute meaningful diff.
        # Return potentially empty/unfiltered X_train and filtered X_test (first k or all).
        # This behavior might need refinement based on desired outcome for single-class inputs.
        print(
            "Warning: y_train for OOD SAE processing has only one class or is empty. Cannot balance or compute feature differences effectively."
        )

        # Create selected_indices from all available y_train samples if any, up to num_train
        if len(y_train) > 0:
            selected_indices = np.random.choice(
                np.arange(len(y_train)),
                size=min(num_train, len(y_train)),
                replace=False,
            )
            X_train_balanced = X_train_sae_features[selected_indices]
            y_train_balanced = y_train[selected_indices]
        else:  # Should have been caught by len(y_train) == 0 above, but defensive
            X_train_balanced = torch.empty(
                (0, X_train_sae_features.shape[1]),
                device=X_train_sae_features.device,
                dtype=X_train_sae_features.dtype,
            )
            y_train_balanced = np.array([], dtype=y_train.dtype)

        top_by_average_diff = torch.arange(
            min(
                k if k > 0 else X_train_sae_features.shape[1],
                X_train_sae_features.shape[1],
            )
        )

        X_train_filtered = (
            X_train_balanced[:, top_by_average_diff]
            if X_train_balanced.numel() > 0
            else X_train_balanced
        )
        X_test_filtered = (
            X_test_sae_features[:, top_by_average_diff]
            if X_test_sae_features.numel() > 0
            else X_test_sae_features
        )

        if return_indices:
            return (
                X_train_filtered,
                y_train_balanced,
                X_test_filtered,
                y_test,
                top_by_average_diff,
            )
        return X_train_filtered, y_train_balanced, X_test_filtered, y_test

    max_pos_samples = len(pos_indices)
    max_neg_samples = len(neg_indices)

    num_pos_to_select = min(num_train // 2, max_pos_samples)
    num_neg_to_select = min(num_train // 2, max_neg_samples)

    # Recalculate actual num_train based on selections if one class was limiting
    actual_num_train = num_pos_to_select + num_neg_to_select
    if actual_num_train < num_train and (
        num_pos_to_select < num_train // 2 or num_neg_to_select < num_train // 2
    ):
        # Try to fill remaining from the other class if it has capacity
        if num_pos_to_select < num_train // 2:  # neg was full or num_train//2
            num_pos_to_select = min(
                actual_num_train - num_neg_to_select, max_pos_samples
            )
        elif num_neg_to_select < num_train // 2:  # pos was full or num_train//2
            num_neg_to_select = min(
                actual_num_train - num_pos_to_select, max_neg_samples
            )

    # Final check if num_train still not met (e.g. total samples < num_train)
    # This scenario is implicitly handled by min // replace=False logic already.
    # Just ensure selected_indices isn't trying to pick more than available.

    pos_selected = np.random.choice(pos_indices, num_pos_to_select, replace=False)
    neg_selected = np.random.choice(neg_indices, num_neg_to_select, replace=False)

    selected_indices = np.concatenate([pos_selected, neg_selected])
    np.random.shuffle(selected_indices)

    X_train_balanced = X_train_sae_features[selected_indices]
    y_train_balanced = y_train[selected_indices]

    if X_train_balanced.shape[0] == 0:
        empty_features_dim = (
            k
            if k > 0
            else X_train_sae_features.shape[1]
            if X_train_sae_features.numel() > 0
            else sae.cfg.d_sae
        )
        empty_features_shape_train = (0, empty_features_dim)

        if k > 0 and X_test_sae_features.shape[1] >= k:
            X_test_filtered = X_test_sae_features[:, :k]
        elif k == 0:
            X_test_filtered = X_test_sae_features
        else:
            X_test_filtered = X_test_sae_features

        empty_indices = torch.empty(0, dtype=torch.long)

        if return_indices:
            return (
                torch.empty(
                    empty_features_shape_train,
                    device=X_train_sae_features.device,
                    dtype=X_train_sae_features.dtype,
                ),
                np.array([], dtype=y_train.dtype),
                X_test_filtered.to(
                    X_test_sae_features.device, dtype=X_test_sae_features.dtype
                ),
                y_test,
                empty_indices,
            )
        return (
            torch.empty(
                empty_features_shape_train,
                device=X_train_sae_features.device,
                dtype=X_train_sae_features.dtype,
            ),
            np.array([], dtype=y_train.dtype),
            X_test_filtered.to(
                X_test_sae_features.device, dtype=X_test_sae_features.dtype
            ),
            y_test,
        )

    # Calculate mean difference only if both classes are present in the balanced set
    y_train_balanced_unique = np.unique(y_train_balanced)
    if len(y_train_balanced_unique) < 2:
        print(
            f"Warning: y_train_balanced for OOD SAE processing has only one class ({y_train_balanced_unique}) after balancing. Cannot compute feature differences. Using first k features or all."
        )
        top_by_average_diff = torch.arange(
            min(k if k > 0 else X_train_balanced.shape[1], X_train_balanced.shape[1])
        )
    else:
        X_train_diff = X_train_balanced[y_train_balanced == 1].mean(
            dim=0
        ) - X_train_balanced[y_train_balanced == 0].mean(dim=0)
        if torch.isnan(X_train_diff).any():
            print(
                "Warning: NaN encountered in X_train_diff for OOD SAE. This might be due to empty classes after balancing. Using first k or all features."
            )
            top_by_average_diff = torch.arange(
                min(
                    k if k > 0 else X_train_balanced.shape[1], X_train_balanced.shape[1]
                )
            )
        else:
            sorted_indices_diff = torch.argsort(
                torch.abs(X_train_diff), descending=True
            )
            top_by_average_diff = sorted_indices_diff[
                : k if k > 0 else X_train_balanced.shape[1]
            ]

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
    model_acts_test: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int = 128,
    device: str | torch.device | None = None,
    num_train_samples_per_class: int = 512,
):
    """
    Prepares training and testing data (X, y) for GLUE task experiments
    using SAE features derived from provided model activations.
    y_train and y_test are now direct inputs.
    """
    X_train_sae_features = get_sae_features(sae, model_acts_train, device=device)
    X_test_sae_features = get_sae_features(sae, model_acts_test, device=device)

    if len(y_train) == 0:
        empty_features_dim = (
            k
            if k > 0
            else X_train_sae_features.shape[1]
            if X_train_sae_features.numel() > 0
            else sae.cfg.d_sae
        )
        empty_features_shape_train = (0, empty_features_dim)

        if k > 0 and X_test_sae_features.shape[1] >= k:
            X_test_filtered = X_test_sae_features[:, :k]
        elif k == 0:
            X_test_filtered = X_test_sae_features
        else:
            X_test_filtered = X_test_sae_features

        return (
            torch.empty(
                empty_features_shape_train,
                device=X_train_sae_features.device,
                dtype=X_train_sae_features.dtype,
            ),
            np.array([], dtype=y_train.dtype),
            X_test_filtered.to(
                X_test_sae_features.device, dtype=X_test_sae_features.dtype
            ),
            y_test,
        )

    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        print(
            "Warning: y_train for GLUE SAE processing has only one class or is empty. Cannot balance or compute feature differences effectively."
        )
        # Fallback: use all y_train samples up to num_train_samples_per_class * 2 (approx total)
        # And use first k features for X.
        if len(y_train) > 0:
            num_to_select = min(
                num_train_samples_per_class * 2, len(y_train)
            )  # crude upper bound
            selected_indices = np.random.choice(
                np.arange(len(y_train)), size=num_to_select, replace=False
            )
            X_train_balanced = X_train_sae_features[selected_indices]
            y_train_balanced = y_train[selected_indices]
        else:
            X_train_balanced = torch.empty(
                (0, X_train_sae_features.shape[1]),
                device=X_train_sae_features.device,
                dtype=X_train_sae_features.dtype,
            )
            y_train_balanced = np.array([], dtype=y_train.dtype)

        top_k_indices = torch.arange(
            min(k if k > 0 else X_train_balanced.shape[1], X_train_balanced.shape[1])
        )

        X_train_filtered = (
            X_train_balanced[:, top_k_indices]
            if X_train_balanced.numel() > 0
            else X_train_balanced
        )
        X_test_filtered = (
            X_test_sae_features[:, top_k_indices]
            if X_test_sae_features.numel() > 0
            else X_test_sae_features
        )

        return X_train_filtered, y_train_balanced, X_test_filtered, y_test

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

    if (
        X_train_balanced.shape[0] == 0
    ):  # Should be caught by earlier y_train len check, but defensive
        empty_features_dim = (
            k
            if k > 0
            else X_train_sae_features.shape[1]
            if X_train_sae_features.numel() > 0
            else sae.cfg.d_sae
        )
        empty_features_shape_train = (0, empty_features_dim)

        if k > 0 and X_test_sae_features.shape[1] >= k:
            X_test_filtered = X_test_sae_features[:, :k]
        elif k == 0:
            X_test_filtered = X_test_sae_features
        else:
            X_test_filtered = X_test_sae_features

        return (
            torch.empty(
                empty_features_shape_train,
                device=X_train_sae_features.device,
                dtype=X_train_sae_features.dtype,
            ),
            np.array([], dtype=y_train.dtype),
            X_test_filtered.to(
                X_test_sae_features.device, dtype=X_test_sae_features.dtype
            ),
            y_test,
        )

    y_train_balanced_unique = np.unique(y_train_balanced)
    if len(y_train_balanced_unique) < 2:
        print(
            f"Warning: y_train_balanced for GLUE SAE processing has only one class ({y_train_balanced_unique}) after balancing. Cannot compute feature differences. Using first k features or all."
        )
        top_by_average_diff = torch.arange(
            min(k if k > 0 else X_train_balanced.shape[1], X_train_balanced.shape[1])
        )
    else:
        X_train_diff = X_train_balanced[y_train_balanced == 1].mean(
            dim=0
        ) - X_train_balanced[y_train_balanced == 0].mean(dim=0)
        if torch.isnan(X_train_diff).any():
            print(
                "Warning: NaN encountered in X_train_diff for GLUE SAE. Using first k or all features."
            )
            top_by_average_diff = torch.arange(
                min(
                    k if k > 0 else X_train_balanced.shape[1], X_train_balanced.shape[1]
                )
            )
        else:
            sorted_indices_diff = torch.argsort(
                torch.abs(X_train_diff), descending=True
            )
            top_by_average_diff = sorted_indices_diff[
                : k if k > 0 else X_train_balanced.shape[1]
            ]

    X_train_filtered = X_train_balanced[:, top_by_average_diff]
    X_test_filtered = X_test_sae_features[:, top_by_average_diff]

    return X_train_filtered, y_train_balanced, X_test_filtered, y_test


def get_grammar_feature_examples(
    sae: SAE,
    model_acts_train: torch.Tensor,
    model_acts_test: torch.Tensor,
    y_train_for_glue: np.ndarray,
    y_test_for_glue: np.ndarray,
    device: str | torch.device | None = None,
):
    # We need X_test_sae_features corresponding to the original 'X' in the function.
    # The original function called get_xy_glue_sae(k=1) and used its X_test_filtered output as 'X'.
    # So we replicate that here.
    _, _, X_test_filtered_k1, _ = get_xy_glue_sae(
        sae=sae,
        model_acts_train=model_acts_train,
        model_acts_test=model_acts_test,
        y_train=y_train_for_glue,
        y_test=y_test_for_glue,
        k=1,
        device=device,
    )
    X = X_test_filtered_k1  # This is X_test_sae_features[:, top_1_feature_index]

    # The rest of the function from the old utils_sae.py needs to be integrated here.
    # It used 'X', 'model_name', 'layer', 'sae_buffer', 'top_k_latents', 'dataset'
    # This will require further refactoring as this function's purpose was tied to the old data loading.
    # For now, returning X as a placeholder for the top-1 SAE feature activations on the test set.
    # This function likely needs a more significant overhaul or to be deprecated if its
    # original purpose (finding grammar feature examples) is now handled differently.
    print(
        "get_grammar_feature_examples has been partially refactored. Its core logic for finding examples needs review."
    )
    return X.cpu().numpy()  # Assuming X was torch tensor and original returned numpy


# Removed get_sae_layers and commented out script execution lines
