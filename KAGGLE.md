# Running ATAR training & evaluation on Kaggle

Copy each block into its own notebook cell. Notebook settings:
**Accelerator: GPU T4 x1**, **Internet: on**, **Persistence: Files only**.

Timing below uses the 2026-10-14 pilot: **3.7 s per env step** on a T4 with
Qwen2.5-7B (3,072 steps in 11,295 s, model loading included). The first run
(pre-fix code) measured 4.0–4.9 s/step, so keep a margin.

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

**Pilot: done 2026-10-14** (3 updates; it found the CLARIFY reward bug, see
ADR-20 — so the real run starts fresh, not from the pilot checkpoint).
For any future pilot: run a ~3 h pilot first — the same command as below but
with `--stop-after-updates 3`. Before spending the rest of the week's quota,
check the W&B charts in section 4: if the action mix collapses to a single
action or `success_rate` doesn't move, stop and debug. If it looks healthy,
continue the same run with 3b (resume from the pilot's last checkpoint).

Use a **new notebook** (or run `!rm -rf /kaggle/working/checkpoints` first)
so the pilot's checkpoints don't mix with the real run's.

Do **not** resume from the old `checkpoint_step_*` files or the pilot's: they were trained
with the GAE/returns/projection/grader bugs fixed on 2026-10-13
(see ARCHITECTURE_DECISIONS.md ADR-15..17).

```python
!python scripts/train.py \
    --model-name Qwen/Qwen2.5-7B-Instruct --device cuda \
    --total-steps 24576 --rollout-steps 1024 \
    --checkpoint-dir /kaggle/working/checkpoints --checkpoint-interval 1 \
    --stop-after-updates 8 \
    --wandb-project atar-training --wandb-mode online
```

Why these numbers:
- `--rollout-steps 1024` (MSD default 2048): one update ≈ 1024 × 3.7 s
  ≈ 63 min, so a session gets ~8 policy updates instead of ~4. More,
  smaller updates is the better trade when the whole budget is ~25k steps
  (the first run got only 12 updates in total). Logged in W&B config; the
  MSD allows retuning if logged.
- `--checkpoint-interval 1`: a checkpoint after every update, so a session
  cut-off loses at most one update (~65 min).
- `--stop-after-updates 8`: exits cleanly at ~8.5–9 h, before Kaggle's 12 h cap
  (model loading adds ~10 min).
- `--total-steps 24576`: same budget as the first run, but 24 updates instead
  of 12; ≈ 25 GPU-hours ≈ 3 sessions, within one week of Kaggle's
  ~30 h/week quota. Curriculum phases switch at steps 7,372 and 17,203.

When the cell finishes, make sure `/kaggle/working/checkpoints` survives the
session the same way you carried checkpoints between notebooks before
(saved notebook output or a Kaggle dataset), **and** download the newest
`checkpoint_step_*.pt` as a backup. Also download
`/kaggle/working/checkpoints/metrics.jsonl` (one line per update — the full
history) and send it for review after each session.

When resuming, first copy the previous `metrics.jsonl` into the new
checkpoint dir so the history stays in one file:

```python
!mkdir -p /kaggle/working/checkpoints && cp /kaggle/input/<PREVIOUS-NOTEBOOK>/checkpoints/metrics.jsonl /kaggle/working/checkpoints/
```

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
    --stop-after-updates 8 \
    --wandb-project atar-training --wandb-mode online \
    --resume-from /kaggle/input/<PREVIOUS-NOTEBOOK>/checkpoints/checkpoint_step_<N>.pt
```

`--total-steps` and `--rollout-steps` must be the same in every session
(they define the curriculum boundaries and the LR schedule).

**Keep every checkpoint** (one per update, ~20 MB each with the projection
and optimizer state). They are the policy "versions" for the learning curve
and the demo's version dropdown.

## 3c. Decide whether to extend (after the planned 24,576 steps)

Evaluate a few checkpoints on the held-out tasks (section 5, e.g. steps
6,144 / 12,288 / 18,432 / final) and look at held-out success vs. step:

- **Still clearly rising at the end** → extend. All extra steps use all 5
  tiers; the LR restarts at 1e-4 and decays to 1e-5 over the extra updates.
- **Flat over the last third** → stop; more steps won't help, and you save
  1–3 weeks of quota.

Extension command (the same command in every session of the extension;
keep `--total-steps 24576` — it's the original plan, not the new target):

```python
!python scripts/train.py \
    --model-name Qwen/Qwen2.5-7B-Instruct --device cuda \
    --total-steps 24576 --rollout-steps 1024 --extend-to 49152 \
    --checkpoint-dir /kaggle/working/checkpoints --checkpoint-interval 1 \
    --stop-after-updates 8 \
    --wandb-project atar-training --wandb-mode online \
    --resume-from /kaggle/input/<PREVIOUS-NOTEBOOK>/checkpoints/<newest checkpoint>.pt
```

The first extension session resumes from `checkpoint_final.pt`; later ones
from the newest `checkpoint_step_*.pt`. The extension ends with
`checkpoint_final_49152.pt` (the planned run's `checkpoint_final.pt` is kept).

## 4. What to watch in W&B

| Chart | Healthy | Problem |
|---|---|---|
| `action_frac/*` | mix shifts toward the tools each tier needs | one action → ~1.0 (collapse; usually ANSWER_DIRECTLY) |
| `step_reward/*`, `invalid_param_rate/*` | explain the action mix: what each action earns, how often the LLM's params for it are rejected | an action avoided because its params are rejected (a prompt/translator problem, not a policy one) |
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

## 6. Demo data (after training: learning curve + replays)

Everything the Streamlit demo shows is computed here, on the GPU, with the
training model. Use a separate session from training if needed — every step
below resumes safely.

1. **Step-0 checkpoint** (the untrained starting point; no download needed):
   ```python
   !python scripts/make_untrained_checkpoint.py --hidden-size 3584 --out /kaggle/working/checkpoints/checkpoint_step_0.pt
   ```
2. **Evaluate the versions you want on the curve** — e.g. step 0, every
   ~6k steps, and the final one — each into its own folder (≈ 45 min each
   for 200 tasks; `--max-tasks 50` ≈ 12 min gives a rougher curve):
   ```python
   for step in [0, 6144, 12288, 18432, 24576]:
       !python scripts/evaluate.py --checkpoint /kaggle/working/checkpoints/checkpoint_step_{step}.pt \
           --agents atar --model-name Qwen/Qwen2.5-7B-Instruct --device cuda --out-dir /kaggle/working/eval_step{step}
   ```
3. **Baselines once** (the same tasks; or reuse section 5's `eval_final`):
   ```python
   !python scripts/evaluate.py --agents react,zeroshot,random \
       --model-name Qwen/Qwen2.5-7B-Instruct --device cuda --out-dir /kaggle/working/eval_baselines
   ```
4. **Record replays** (3 tasks per tier × each version + baselines):
   ```python
   # In a notebook, {...} inside a ! command is Python, so build the list first.
   ckpts = " ".join(f"/kaggle/working/checkpoints/checkpoint_step_{s}.pt" for s in [0, 6144, 12288, 18432, 24576])
   !python scripts/record_replays.py --checkpoints {ckpts} \
       --baselines react,zeroshot,random --tasks-per-tier 3 \
       --model-name Qwen/Qwen2.5-7B-Instruct --device cuda --out /kaggle/working/replays.jsonl
   ```
5. **Build the bundle** (CPU is fine) and download `demo_data/`:
   ```python
   !python scripts/build_demo_bundle.py --model-name Qwen/Qwen2.5-7B-Instruct \
       --eval-dirs /kaggle/working/eval_step* /kaggle/working/eval_baselines \
       --replays /kaggle/working/replays.jsonl --out /kaggle/working/demo_data
   ```
6. Locally: put the folder at `<repo>/demo_data/` and run
   `streamlit run atar/demo/app.py`.
