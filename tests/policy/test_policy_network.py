"""
Tests for M7 ATARPolicy network.

All tests run against CPU — no GPU required for shape/architecture checks.
Tests prove the exact MSD-specified architecture, not any model quality.
"""

import pytest
import torch
from atar.policy.policy_network import ATARPolicy


@pytest.fixture(scope="module")
def policy() -> ATARPolicy:
    return ATARPolicy()


def test_policy_forward_batch_shapes(policy: ATARPolicy) -> None:
    """
    ATARPolicy.forward() must produce correctly-shaped outputs for a batch
    of states.  DoD item: 'forward pass produces correctly-shaped outputs
    for a batch of states'.
    """
    B = 4
    states = torch.randn(B, 512)
    logits, values = policy(states)

    assert logits.shape == (B, 7), f"Expected logits (B,7), got {logits.shape}"
    assert values.shape == (B, 1), f"Expected values (B,1), got {values.shape}"


def test_policy_forward_single_state(policy: ATARPolicy) -> None:
    """
    forward() must accept a 1-D (512,) tensor and unsqueeze internally,
    returning (1, 7) and (1, 1).
    """
    state = torch.randn(512)
    logits, values = policy(state)
    assert logits.shape == (1, 7), f"Expected (1,7), got {logits.shape}"
    assert values.shape == (1, 1), f"Expected (1,1), got {values.shape}"


def test_policy_backbone_architecture(policy: ATARPolicy) -> None:
    """
    Verify the exact MSD-specified backbone dimensions:
      Linear(512,256) + LayerNorm(256) + GELU
      Linear(256,128) + LayerNorm(128) + GELU
    """
    layers = list(policy.backbone.children())
    # Should have 6 sub-modules in order
    assert len(layers) == 6, f"Expected 6 backbone layers, got {len(layers)}"

    import torch.nn as nn
    assert isinstance(layers[0], nn.Linear), "Layer 0 should be Linear"
    assert layers[0].in_features == 512
    assert layers[0].out_features == 256

    assert isinstance(layers[1], nn.LayerNorm), "Layer 1 should be LayerNorm"
    assert layers[2].__class__.__name__ == "GELU", "Layer 2 should be GELU"

    assert isinstance(layers[3], nn.Linear), "Layer 3 should be Linear"
    assert layers[3].in_features == 256
    assert layers[3].out_features == 128

    assert isinstance(layers[4], nn.LayerNorm), "Layer 4 should be LayerNorm"
    assert layers[5].__class__.__name__ == "GELU", "Layer 5 should be GELU"


def test_policy_heads_dimensions(policy: ATARPolicy) -> None:
    """Policy head must be Linear(128,7); value head must be Linear(128,1)."""
    import torch.nn as nn
    assert isinstance(policy.policy_head, nn.Linear)
    assert policy.policy_head.in_features == 128
    assert policy.policy_head.out_features == 7

    assert isinstance(policy.value_head, nn.Linear)
    assert policy.value_head.in_features == 128
    assert policy.value_head.out_features == 1


def test_policy_output_dtype(policy: ATARPolicy) -> None:
    """Outputs must be float32 by default."""
    states = torch.randn(2, 512)
    logits, values = policy(states)
    assert logits.dtype == torch.float32
    assert values.dtype == torch.float32


def test_policy_has_parameters(policy: ATARPolicy) -> None:
    """Policy must have trainable parameters."""
    params = list(policy.parameters())
    assert len(params) > 0, "ATARPolicy has no parameters"
    assert all(p.requires_grad for p in params), "All policy params should require grad"
