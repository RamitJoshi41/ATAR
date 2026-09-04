# MSD — M7: RL Policy & Training

## Purpose
Train the decision policy using PPO. **This is the module the human
architect wants to own the core algorithm for** — see split below. This is
intentional: it's the highest-value module for interview/resume depth, and
the architect needs to be able to defend every line of the PPO update
without hesitation.

## Split of responsibility
**Agent builds:**
- Network architecture (exact spec below)
- Rollout buffer data structure
- Training loop scaffolding: episode collection loop, checkpointing,
  Weights & Biases logging calls, curriculum phase-switching logic,
  evaluation-run scheduling
- Baseline implementations: Random policy, ReAct-style prompted baseline
  (using M3 directly with a hand-written reasoning+acting prompt loop, no
  RL)
- **`scripts/train.py`** — a standalone CLI entry point (argparse), runnable
  independently of this Antigravity workspace. This script will be copied
  to and executed inside a Kaggle notebook or rented GPU instance, not run
  from here — it must not assume anything about the local dev environment.
- **Checkpoint/resume, treated as a hard requirement, not optional:**
  `--resume-from <path>` must restore model weights, optimizer state,
  curriculum phase, and step count exactly, such that training continues
  as if uninterrupted. Free-tier GPU sessions cap around 12 hours; a
  training run that can't resume risks losing significant progress. Save
  a checkpoint at minimum every N steps (agent picks a sensible N) and
  immediately before any curriculum phase transition.

**Architect implements personally (leave as a stub with a clear docstring):**
- The PPO clipped-surrogate loss computation
- The GAE advantage calculation
- The value loss and entropy bonus terms
- The actual optimizer.step() update logic

Agent: write the stub as
```python
def compute_ppo_loss(rollout_batch: RolloutBatch, policy: ATARPolicy,
                      clip_range: float, value_coef: float, entropy_coef: float
                      ) -> PPOLossOutput:
    """
    TODO(architect): implement clipped surrogate PPO loss.
    Reference: Schulman et al. 2017, https://arxiv.org/abs/1707.06347
    Inputs: rollout_batch contains states, actions, old_log_probs,
    advantages (already GAE-computed upstream), returns.
    Must return: PPOLossOutput(policy_loss, value_loss, entropy, total_loss)
    """
    raise NotImplementedError("Architect implements this — see AGENTS.md")

def compute_gae(rewards: list[float], values: list[float], dones: list[bool],
                 gamma: float, lam: float) -> list[float]:
    """
    TODO(architect): implement Generalized Advantage Estimation.
    Reference: Schulman et al. 2016, https://arxiv.org/abs/1506.02438
    """
    raise NotImplementedError("Architect implements this — see AGENTS.md")
```

## Network architecture (agent implements exactly)
```python
class ATARPolicy(nn.Module):
    # Shared backbone: Linear(512,256) + LayerNorm + GELU
    #                -> Linear(256,128) + LayerNorm + GELU
    # Policy head: Linear(128, 7) -> action logits
    # Value head: Linear(128, 1) -> state value estimate
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...  # returns (action_logits, value_estimate)
```
The optimizer's parameter group must include both `ATARPolicy.parameters()`
and M3's `LLMInterface.projection.parameters()` — this is the "co-adapted
projection layer" design decision, confirm it's actually wired this way.

## Hyperparameters (use exactly, these are the spec's tuned defaults —
architect may retune after first run, log any changes)
| Param | Value |
|---|---|
| Learning rate | 3e-4, cosine decay to 1e-5 |
| Rollout steps | 2048 |
| Batch size | 64 |
| Epochs per update | 10 |
| Discount (gamma) | 0.95 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 → 0.001 (decay over training) |
| Value function coefficient | 0.5 |
| Max gradient norm | 0.5 |

## Curriculum schedule
- Phase 1 (0-30% of training): Tiers 1-2 only
- Phase 2 (30-70%): Tiers 1-4
- Phase 3 (70-100%): all 5 tiers including error-recovery

## Baselines to implement
- **Random**: uniform random action selection each step.
- **ReAct**: prompted reasoning+acting loop using M3's `generate()` directly,
  no policy network, following the standard ReAct prompting pattern
  (Yao et al. 2023). This is the primary comparison baseline for the demo.

## Ablations (build these as alternate training configs, run after the
main model trains successfully — needed for the project's writeup)
- No-curriculum: train on the full 5-tier mix from step 0.
- No-projection-cotraining: freeze M3's projection layer, don't include it
  in the PPO optimizer.

## Definition of Done
- [ ] `ATARPolicy` forward pass produces correctly-shaped outputs for a
  batch of states.
- [ ] Rollout collection loop runs against a real `ATAREnv` for at least
  100 steps without error, storing correctly-typed `EpisodeStep` entries.
- [ ] Checkpointing saves/restores a policy such that its outputs are
  identical before and after a save/load cycle (bitwise or near-bitwise).
- [ ] `--resume-from` tested explicitly: run a short training session,
  interrupt it, resume from the saved checkpoint, and confirm curriculum
  phase + step count match what they would have been without interruption.
- [ ] `scripts/train.py` runs standalone from a fresh shell (not just
  inside this Antigravity session) using the tiny stand-in model from
  M3 — confirms it has no hidden dependency on the local dev environment.
- [ ] W&B logging confirmed working (a real run appears in the dashboard,
  not just that the call doesn't crash).
- [ ] Curriculum phase-switching triggers at the correct step counts —
  test with a mocked short training run.
- [ ] Random and ReAct baselines both run a full evaluation pass against
  the 200 held-out tasks and report real accuracy/tool-call numbers.
- [ ] `compute_ppo_loss` / `compute_gae` stubs are in place with correct
  signatures — NOT implemented by the agent.
- [ ] `pytest tests/policy/ -v` passes for everything except the stubbed
  functions (mark those tests `xfail` until the architect implements them).
