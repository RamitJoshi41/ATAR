# Running ATAR training & evaluation on Kaggle

Copy each block into its own notebook cell. Notebook settings:
**Accelerator: GPU T4 x1**, **Internet: on**, **Persistence: Files only**.

Timing below is an estimate built from the M7 MSD's measured numbers
(~2.7 s per `generate()` + ~0.1 s encode on a T4 with Qwen2.5-7B, i.e.
~2.9 s per env step, because generation is still sequential). Check the first
session's logs and adjust `--stop-after-updates`.

---

## 1. Setup (every session)

```python
!git clone https://github.com/RamitJoshi41/ATAR.git /kaggle/working/ATAR
%cd /kaggle/working/ATAR
!pip install -q -e . "bitsandbytes>=0.42,<0.51"
```

## 2. W&B login (every session)

Add your W&B API key as a Kaggle secret named `WANDB_API_KEY`
(Add-ons → Secrets), then:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

## 3a. Start a NEW training run (first session only)

**Recommended: run a ~2.5 h pilot first** — the same command as below but
with `--stop-after-updates 3`. Before spending the rest of the week's quota,
check the W&B charts in section 4: if the action mix collapses to a single
action or `success_rate` doesn't move, stop and debug. If it looks healthy,
continue the same run with 3b (resume from the pilot's last checkpoint).

Do **not** resume from the old `checkpoint_step_*` files: they were trained
with the GAE/returns/projection/grader bugs fixed on 2026-10-13
(see ARCHITECTURE_DECISIONS.md ADR-15..17).

```python
!python scripts/train.py \
    --model-name Qwen/Qwen2.5-7B-Instruct --device cuda \
    --total-steps 30000 --rollout-steps 1024 \
    --checkpoint-dir /kaggle/working/checkpoints --checkpoint-interval 1 \
    --stop-after-updates 12 \
    --wandb-project atar-training --wandb-mode online
```

Why these numbers:
- `--rollout-steps 1024` (MSD default 2048): one update ≈ 1024 × 2.9 s ≈ 50 min,
  so you get ~12 policy updates per 12 h session instead of ~6. More, smaller
  updates is the better trade when the whole budget is only ~30k steps.
  (Logged in W&B config; the MSD allows retuning if logged.)
- `--checkpoint-interval 1`: a checkpoint after every update, so a session
  cut-off loses at most ~50 min.
- `--stop-after-updates 12`: exits cleanly at ~10 h, before Kaggle's 12 h cap.
- `--total-steps 30000`: ~30 updates ≈ 3 sessions ≈ one week of Kaggle GPU quota.
  Curriculum phases switch at 9,000 and 21,000 steps.

When the cell finishes, make sure `/kaggle/working/checkpoints` survives the
session the same way you carried checkpoints between notebooks before
(saved notebook output or a Kaggle dataset), **and** download the newest
`checkpoint_step_*.pt` as a backup.

## 3b. Resume (every later session)

Add the previous notebook's output as an input (Add Input → Your Work →
Notebooks), find the newest checkpoint, and resume. The W&B run id is read
from the checkpoint, so all sessions land in the **same** W&B run.

```python
!ls -t /kaggle/input/*/checkpoints/ | head
```

```python
!python scripts/train.py \
    --model-name Qwen/Qwen2.5-7B-Instruct --device cuda \
    --total-steps 30000 --rollout-steps 1024 \
    --checkpoint-dir /kaggle/working/checkpoints --checkpoint-interval 1 \
    --stop-after-updates 12 \
    --wandb-project atar-training --wandb-mode online \
    --resume-from /kaggle/input/<PREVIOUS-NOTEBOOK>/checkpoints/checkpoint_step_<N>.pt
```

`--total-steps` and `--rollout-steps` must be the same in every session
(they define the curriculum boundaries and the LR schedule).

## 4. What to watch in W&B

| Chart | Healthy | Problem |
|---|---|---|
| `action_frac/*` | mix shifts toward the tools each tier needs | one action → ~1.0 (collapse; usually ANSWER_DIRECTLY) |
| `success_rate`, `success_rate/tier_k` | rises within a phase; dips when a new phase adds harder tiers | flat at the random-agent level for many updates |
| `mean_episode_reward` | trends up | trends down / stuck very negative (invalid-param penalties) |
| `entropy` | decreases slowly | drops to ~0 in the first few updates |
| `value_loss` | falls then plateaus | grows steadily |
| `curriculum_phase` | 1 → 2 at step 9k, 2 → 3 at 21k | wrong phase after a resume |

## 5. Evaluate a checkpoint (after training, or any time)

Always on the GPU with the **same model used for training**. The policy was
trained on that model's hidden states; the local 0.5B stand-in produces a
completely different input and the result would be meaningless (the script
now refuses a hidden-size mismatch). Use a **new `--out-dir` per checkpoint**.

Quick pipeline check (≈ 20–30 min):

```python
!python scripts/evaluate.py \
    --checkpoint /kaggle/working/checkpoints/checkpoint_step_<N>.pt \
    --agents atar,zeroshot --max-tasks 20 \
    --model-name Qwen/Qwen2.5-7B-Instruct --device cuda \
    --out-dir /kaggle/working/eval_step<N>_smoke
```

Full 200-task comparison (several hours across all four agents; run
`--agents` one or two at a time across sessions if needed — results resume
safely into the same `--out-dir` as long as checkpoint/model/seed match):

```python
!python scripts/evaluate.py \
    --checkpoint /kaggle/working/checkpoints/checkpoint_final.pt \
    --agents atar,random,zeroshot,react \
    --model-name Qwen/Qwen2.5-7B-Instruct --device cuda \
    --out-dir /kaggle/working/eval_final
```

Outputs: `comparison.md` (table with 95% bootstrap CIs, per tier),
`results.json`, `per_task.jsonl`, `failures.jsonl` (ATAR failures, input for M6).
