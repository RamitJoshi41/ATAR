# Running ATAR training & evaluation on Kaggle

Copy each block into its own notebook cell. Notebook settings:
**Accelerator: GPU T4 x1**, **Internet: on**, **Persistence: Files only**.

Timing below uses the throughput measured in the first run's W&B export
(2026-09-28 → 10-01): **4.0–4.9 s per env step** on a T4 with Qwen2.5-7B
(e.g. 8,192 steps in 33,520 s), because generation is still sequential.
The M7 spec's 2.9 s/step estimate was too optimistic. Re-check against the
pilot's runtime and adjust `--stop-after-updates`.

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

**Recommended: run a ~4 h pilot first** — the same command as below but
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
    --total-steps 24576 --rollout-steps 1024 \
    --checkpoint-dir /kaggle/working/checkpoints --checkpoint-interval 1 \
    --stop-after-updates 7 \
    --wandb-project atar-training --wandb-mode online
```

Why these numbers:
- `--rollout-steps 1024` (MSD default 2048): one update ≈ 1024 × 4.0–4.9 s
  ≈ 70–85 min, so a session gets ~7 policy updates instead of ~3–4. More,
  smaller updates is the better trade when the whole budget is ~25k steps
  (the first run got only 12 updates in total). Logged in W&B config; the
  MSD allows retuning if logged.
- `--checkpoint-interval 1`: a checkpoint after every update, so a session
  cut-off loses at most one update (~85 min).
- `--stop-after-updates 7`: exits cleanly at ~8–10 h, before Kaggle's 12 h cap
  (model loading adds ~10 min).
- `--total-steps 24576`: same budget as the first run, but 24 updates instead
  of 12; ≈ 28–34 GPU-hours ≈ 4 sessions, a little over one week of Kaggle's
  ~30 h/week quota. Curriculum phases switch at steps 7,372 and 17,203.

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
    --total-steps 24576 --rollout-steps 1024 \
    --checkpoint-dir /kaggle/working/checkpoints --checkpoint-interval 1 \
    --stop-after-updates 7 \
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
| `curriculum_phase` | 1 → 2 at step ~7.4k, 2 → 3 at ~17.2k (first update boundary after each) | wrong phase after a resume |

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
