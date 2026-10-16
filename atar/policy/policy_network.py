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

Action masking (ADR-23): CLARIFY is allowed once per episode.  The env answers
a repeated CLARIFY with the same text, so it can never add information; the
2026-10-16 diagnostic showed the policy looping on it until the turn limit.
The mask is read from the state's own tool-history flags, so the collector,
the PPO loss and evaluation all apply the same mask with no extra plumbing.
"""

import torch
import torch.nn as nn

from atar.shared.shared_types import TOOL_HISTORY_RANGE, ActionType

_CLARIFY_USED_INDEX: int = TOOL_HISTORY_RANGE[0] + ActionType.CLARIFY.value


def action_mask(state: torch.Tensor) -> torch.Tensor:
    """(B, 7) bool, True = allowed.  Blocks CLARIFY once it has been used."""
    mask = torch.ones(state.shape[0], len(ActionType), dtype=torch.bool, device=state.device)
    mask[:, ActionType.CLARIFY.value] = state[:, _CLARIFY_USED_INDEX] < 0.5
    return mask


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
            Shape (B, 7) — logits (pass through Categorical for sampling).
            Masked actions (see action_mask) get the dtype's minimum, i.e.
            probability 0.
        value_estimate : torch.Tensor
            Shape (B, 1) — state value estimate.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)   # (512,) → (1, 512)

        features = self.backbone(state)          # (B, 128)
        action_logits = self.policy_head(features)   # (B, 7)
        action_logits = action_logits.masked_fill(
            ~action_mask(state), torch.finfo(action_logits.dtype).min
        )
        value_estimate = self.value_head(features)   # (B, 1)
        return action_logits, value_estimate
