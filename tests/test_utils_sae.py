# from sae_probes import utils_sae as new_utils_sae
# from tests._comparison import utils_sae as old_utils_sae

# We'll need to inspect both files to find comparable functions.


import numpy as np
import pytest
import torch
from sae_lens import SAE  # Ensure SAE is imported for type hints and usage

from sae_probes import utils_sae as new_utils_sae

# No direct old_utils_sae comparison for get_sae_features a it's a new structure


# Basic test for get_sae_features
def test_get_sae_features_basic(gpt2_l4_sae: SAE):
    sae = gpt2_l4_sae
    d_in = sae.cfg.d_in
    d_sae = sae.cfg.d_sae
    num_samples = 10

    # Dummy model activations
    model_activations = torch.randn(num_samples, d_in)

    # 1. Test with SAE already on CPU (default for fixture)
    sae.to("cpu")
    features_cpu_sae = new_utils_sae.get_sae_features(
        sae, model_activations.cpu(), device="cpu"
    )
    assert features_cpu_sae.shape == (num_samples, d_sae)
    assert features_cpu_sae.device.type == "cpu"

    # 2. Test with specified device (if CUDA available, otherwise skip or force CPU)
    target_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(target_device_str)

    sae.to(target_device)  # Move SAE to target device
    features_target_device = new_utils_sae.get_sae_features(
        sae, model_activations.to(target_device), device=target_device_str
    )
    assert features_target_device.shape == (num_samples, d_sae)
    assert features_target_device.device.type == "cpu"  # Function should return on CPU

    # Ensure original SAE device is restored if necessary for other tests (though fixtures handle this)
    sae.to("cpu")

    # 3. Test with device=None (should use sae.device)
    sae.to("cpu")  # Ensure SAE is on CPU
    features_none_device = new_utils_sae.get_sae_features(
        sae, model_activations.cpu(), device=None
    )
    assert features_none_device.shape == (num_samples, d_sae)
    assert features_none_device.device.type == "cpu"

    # Verify content consistency (if possible and simple)
    # Re-run on CPU to compare results if target_device was CUDA
    if target_device_str == "cuda":
        sae.to("cpu")
        features_cpu_again = new_utils_sae.get_sae_features(
            sae, model_activations.cpu(), device="cpu"
        )
        torch.testing.assert_close(
            features_target_device, features_cpu_again, rtol=1e-5, atol=1e-5
        )

    torch.testing.assert_close(
        features_cpu_sae, features_none_device, rtol=1e-5, atol=1e-5
    )


# Mock SAE class for testing different .forward() behaviors
class MockSAEOnlyForward:
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        device: str = "cpu",
        output_style: str = "tuple_H_forward_pass_is_features",
    ):
        self.cfg = lambda: None  # Simple mock for cfg
        self.cfg.d_in = d_in
        self.cfg.d_sae = d_sae
        self.device = torch.device(device)
        self.output_style = output_style
        # Mock parameters (not used for encode logic here but good for completeness)
        self.W_enc = torch.randn(d_in, d_sae, device=self.device)
        self.b_enc = torch.randn(d_sae, device=self.device)

    def to(self, device_val):
        self.device = torch.device(device_val)
        self.W_enc = self.W_enc.to(self.device)
        self.b_enc = self.b_enc.to(self.device)
        return self

    def eval(self):
        pass  # Mock eval

    def forward(self, x: torch.Tensor):
        # Simplified encode logic for mock
        # x_cent = x - x.mean(dim=-1, keepdim=True)
        # features = torch.relu(x_cent @ self.W_enc + self.b_enc)
        # For this mock, just create dummy features of the right shape
        batch_size = x.shape[0]
        features = torch.randn(batch_size, self.cfg.d_sae, device=x.device)

        if self.output_style == "tuple_H_forward_pass_is_features":
            # H_res, H_forward_pass, H_sae_out, H_sae_reconstructed, mse_loss, L_sparsity
            mock_H_res = torch.randn_like(x)
            mock_H_sae_out = torch.randn_like(features)  # or features themselves
            mock_H_sae_reconstructed = torch.randn_like(x)
            mock_mse_loss = torch.randn(1)
            mock_L_sparsity = torch.randn(1)
            return (
                mock_H_res,
                features,
                mock_H_sae_out,
                mock_H_sae_reconstructed,
                mock_mse_loss,
                mock_L_sparsity,
            )
        elif self.output_style == "single_tensor_is_features":
            return features
        elif self.output_style == "tuple_first_is_features":
            other_tensor = torch.randn(batch_size, self.cfg.d_sae // 2, device=x.device)
            return (features, other_tensor)
        elif self.output_style == "tuple_no_correct_dim_feature":
            return (torch.randn(batch_size, 1), torch.randn(batch_size, 2))
        else:
            raise ValueError(f"Unknown output_style: {self.output_style}")


@pytest.mark.parametrize(
    "sae_output_style, expect_error",
    [
        ("tuple_H_forward_pass_is_features", False),
        ("single_tensor_is_features", False),
        ("tuple_first_is_features", False),
        ("tuple_no_correct_dim_feature", True),
    ],
)
def test_get_sae_features_mock_sae_forward_styles(
    sae_output_style: str, expect_error: bool
):
    d_in_mock = 768
    d_sae_mock = d_in_mock * 2
    num_samples = 5
    model_activations = torch.randn(num_samples, d_in_mock)

    mock_sae = MockSAEOnlyForward(d_in_mock, d_sae_mock, output_style=sae_output_style)

    if expect_error:
        with pytest.raises(
            ValueError, match="Could not automatically determine SAE features"
        ):
            new_utils_sae.get_sae_features(mock_sae, model_activations, device="cpu")
    else:
        features = new_utils_sae.get_sae_features(
            mock_sae, model_activations, device="cpu"
        )
        assert features.shape == (num_samples, d_sae_mock)
        assert features.device.type == "cpu"


# Placeholder test
def test_placeholder_sae():
    """Placeholder test for utils_sae."""
    assert True


@pytest.mark.parametrize(
    "num_train_total, k_features, return_indices_flag, y_train_balance, y_test_len, acts_dim, sae_dim",
    [
        (100, 64, False, [0] * 50 + [1] * 50, 20, 768, 768 * 4),  # Balanced, k < d_sae
        (60, 128, True, [0] * 20 + [1] * 40, 10, 768, 768 * 4),  # Unbalanced, k < d_sae
        (50, 0, False, [0] * 25 + [1] * 25, 5, 768, 768 * 4),  # k=0 (all features)
        (4, 2, True, [0] * 2 + [1] * 2, 2, 768, 768 * 4),  # Very small N
        (
            10,
            8,
            False,
            [0] * 10,
            2,
            768,
            768 * 4,
        ),  # Single class in y_train (after potential slicing)
    ],
)
def test_get_xy_OOD_sae_logic(
    gpt2_l4_sae: SAE,
    num_train_total: int,
    k_features: int,
    return_indices_flag: bool,
    y_train_balance: list[int],
    y_test_len: int,
    acts_dim: int,
    sae_dim: int,
):
    sae = gpt2_l4_sae
    # Override SAE dimensions if test parameters differ from gpt2_l4_sae (for more general testing)
    # However, gpt2_l4_sae fixture is fixed. So acts_dim and sae_dim should match it.
    acts_dim = sae.cfg.d_in
    sae_dim = sae.cfg.d_sae

    # Ensure num_train_total is achievable with y_train_balance and sae.cfg.d_in
    # The function samples num_train_total // 2 from each class if possible.
    num_pos_available = sum(y_train_balance)
    num_neg_available = len(y_train_balance) - num_pos_available

    if num_pos_available == 0 or num_neg_available == 0:  # Single class y_train
        actual_num_train_balanced = min(num_train_total, len(y_train_balance))
    else:  # Two classes available
        # Initial selection attempt (half from each class)
        _num_pos_to_select = min(num_train_total // 2, num_pos_available)
        _num_neg_to_select = min(num_train_total // 2, num_neg_available)

        # Check if we can fill more from one class if the other was limiting
        _current_total_selected = _num_pos_to_select + _num_neg_to_select
        if _current_total_selected < num_train_total:
            remaining_needed = num_train_total - _current_total_selected
            # If positive class was not filled to its half and has more samples
            if (
                _num_pos_to_select < num_train_total // 2
                and _num_pos_to_select < num_pos_available
            ):
                can_add_pos = min(
                    remaining_needed, num_pos_available - _num_pos_to_select
                )
                _num_pos_to_select += can_add_pos

            _current_total_selected = (
                _num_pos_to_select + _num_neg_to_select
            )  # Recalculate current total
            remaining_needed = (
                num_train_total - _current_total_selected
            )  # Recalculate remaining needed

            # If negative class was not filled to its half and has more samples (and still need more)
            if (
                remaining_needed > 0
                and _num_neg_to_select < num_train_total // 2
                and _num_neg_to_select < num_neg_available
            ):
                can_add_neg = min(
                    remaining_needed, num_neg_available - _num_neg_to_select
                )
                _num_neg_to_select += can_add_neg

        actual_num_train_balanced = _num_pos_to_select + _num_neg_to_select

    # Dummy model activations
    model_acts_train = torch.randn(len(y_train_balance), acts_dim)
    model_acts_test = torch.randn(y_test_len, acts_dim)

    # Dummy labels (now passed directly)
    mock_y_train = np.array(y_train_balance)
    mock_y_test = np.random.randint(0, 2, y_test_len)

    # REMOVED monkeypatching for get_yvals and get_xy_OOD

    # dataset_name_dummy = "dummy_dataset_for_ood_sae" # No longer needed
    results = new_utils_sae.get_xy_OOD_sae(
        sae=sae,
        model_acts_train=model_acts_train,
        model_acts_test=model_acts_test,
        y_train=mock_y_train,  # PASSED DIRECTLY
        y_test=mock_y_test,  # PASSED DIRECTLY
        # dataset_name=dataset_name_dummy, # REMOVED
        k=k_features,
        return_indices=return_indices_flag,
        num_train=num_train_total,
        device="cpu",
    )

    if return_indices_flag:
        X_train_f, y_train_f, X_test_f, y_test_f, top_indices = results
        assert isinstance(top_indices, torch.Tensor)
        if (
            k_features > 0
            and actual_num_train_balanced > 0
            and len(
                np.unique(
                    mock_y_train[
                        np.random.choice(
                            np.arange(len(mock_y_train)),
                            size=actual_num_train_balanced,
                            replace=False,
                        )
                    ]
                )
            )
            > 1
        ):
            # Only check length if features could be computed and k > 0
            assert len(top_indices) == k_features
        elif k_features == 0:  # All features
            assert len(top_indices) == sae_dim

    else:
        X_train_f, y_train_f, X_test_f, y_test_f = results

    assert isinstance(X_train_f, torch.Tensor)
    assert isinstance(y_train_f, np.ndarray)  # y is numpy array
    assert isinstance(X_test_f, torch.Tensor)
    assert isinstance(y_test_f, np.ndarray)

    assert X_train_f.shape[0] == actual_num_train_balanced
    assert y_train_f.shape[0] == actual_num_train_balanced

    expected_feature_dim = k_features if k_features > 0 else sae_dim
    if actual_num_train_balanced == 0:  # if no samples after balancing
        assert (
            X_train_f.shape[1] == expected_feature_dim
        )  # Should still have k columns or d_sae if k=0
    else:
        assert X_train_f.shape[1] == expected_feature_dim

    assert X_test_f.shape[0] == y_test_len
    assert X_test_f.shape[1] == expected_feature_dim
    np.testing.assert_array_equal(y_test_f, mock_y_test)


@pytest.mark.parametrize(
    "k_features, y_train_balance, y_test_len, acts_dim, sae_dim, num_train_per_class",
    [
        (64, [0] * 600 + [1] * 600, 20, 768, 768 * 4, 512),  # k < d_sae, ample data
        (0, [0] * 600 + [1] * 600, 10, 768, 768 * 4, 512),  # k=0 (all features)
        (
            8,
            [0] * 5 + [1] * 5,
            2,
            768,
            768 * 4,
            4,
        ),  # Very small N, num_train_per_class will be capped by available
        (
            16,
            [0] * 10 + [1] * 0,
            2,
            768,
            768 * 4,
            8,
        ),  # Single class in y_train (no balancing possible)
    ],
)
def test_get_xy_glue_sae_logic(
    gpt2_l4_sae: SAE,
    k_features: int,
    y_train_balance: list[int],
    y_test_len: int,
    acts_dim: int,
    sae_dim: int,
    num_train_per_class: int,
):
    sae = gpt2_l4_sae
    acts_dim = sae.cfg.d_in
    sae_dim = sae.cfg.d_sae

    num_pos_available = sum(y_train_balance)
    num_neg_available = len(y_train_balance) - num_pos_available

    if num_pos_available == 0 or num_neg_available == 0:  # Single class y_train
        actual_num_train_balanced = min(num_train_per_class * 2, len(y_train_balance))
    else:
        num_pos_to_select = min(num_train_per_class, num_pos_available)
        num_neg_to_select = min(num_train_per_class, num_neg_available)
        actual_num_train_balanced = num_pos_to_select + num_neg_to_select

    model_acts_train = torch.randn(len(y_train_balance), acts_dim)
    model_acts_test = torch.randn(y_test_len, acts_dim)
    mock_y_train = np.array(y_train_balance)
    mock_y_test = np.random.randint(0, 2, y_test_len)

    # REMOVED monkeypatching for get_yvals and get_xy_glue

    # dataset_name_dummy = "87_glue_cola"  # No longer needed
    results = new_utils_sae.get_xy_glue_sae(
        sae=sae,
        model_acts_train=model_acts_train,
        model_acts_test=model_acts_test,
        y_train=mock_y_train,  # PASSED DIRECTLY
        y_test=mock_y_test,  # PASSED DIRECTLY
        # dataset_name=dataset_name_dummy, # REMOVED
        k=k_features,
        # toget_y_test="ensemble", # REMOVED (part of y_test now)
        device="cpu",
        num_train_samples_per_class=num_train_per_class,
    )

    X_train_f, y_train_f, X_test_f, y_test_f = results

    assert isinstance(X_train_f, torch.Tensor)
    assert isinstance(y_train_f, np.ndarray)
    assert isinstance(X_test_f, torch.Tensor)
    assert isinstance(y_test_f, np.ndarray)

    assert X_train_f.shape[0] == actual_num_train_balanced
    assert y_train_f.shape[0] == actual_num_train_balanced

    expected_feature_dim = k_features if k_features > 0 else sae_dim
    # Handle k=1 specific logic in old code if comparing, but new one doesn't have that exotic k=1 slice.
    # The new code uses [:k if k>0 else all], which is simpler.
    if actual_num_train_balanced == 0:  # if no samples after balancing
        assert X_train_f.shape[1] == expected_feature_dim
    else:
        assert X_train_f.shape[1] == expected_feature_dim

    assert X_test_f.shape[0] == y_test_len
    assert X_test_f.shape[1] == expected_feature_dim
    np.testing.assert_array_equal(y_test_f, mock_y_test)
