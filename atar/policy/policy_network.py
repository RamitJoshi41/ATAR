"""
M7 Policy Network — ATARPolicy

Exact architecture from the M7 MSD:

    Shared backbone:
        Linear(512, 256) + LayerNorm(256) + GELU
        Linear(256, 128) + LayerNorm(128) + GELU
    Policy head: Linear(128, 7)  → action logits
    Value head:  Linear(128, 1)  → state value estimate

The optimizer parameter group in PPOTrainer includes BOTH ATARPolicy.parameters()
AND M3's LLMInterface.projection.parameters() — the co-adapted projection design.
"""

import torch
import torch.nn as nn


class ATARPolicy(nn.Module):
    """
    Shared-backbone actor-critic network for ATAR.

    Input:  state tensor, shape (B, 512) or (512,)
    Output: (action_logits, value_estimate)
                action_logits   — (B, 7)  raw logits, no activation applied
                value_estimate  — (B, 1)  scalar value per state
    """

    def __init__(self) -> None:
        super().__init__()

        # Shared backbone — exactly as specified in MSD
        self.backbone = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # Policy head → 7 action logits (Discrete(7) action space)
        self.policy_head = nn.Linear(128, 7)

        # Value head → scalar estimate
        self.value_head = nn.Linear(128, 1)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the shared backbone then both heads.

        Parameters
        ----------
        state : torch.Tensor
            Shape (B, 512) or (512,).  Unsqueezed to (B, 512) internally
            if a 1-D tensor is passed so callers don't need to manage dims.

        Returns
        -------
        action_logits : torch.Tensor
            Shape (B, 7) — raw logits (pass through Categorical for sampling).
        value_estimate : torch.Tensor
            Shape (B, 1) — state value estimate.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)   # (512,) → (1, 512)

        features = self.backbone(state)          # (B, 128)
        action_logits = self.policy_head(features)   # (B, 7)
        value_estimate = self.value_head(features)   # (B, 1)
        return action_logits, value_estimate
