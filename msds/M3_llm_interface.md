# MSD — M3: LLM Interface

## Purpose
Wrap a frozen, instruction-tuned LLM to provide (a) state encoding for the
RL policy and (b) constrained JSON generation for tool parameters.

## Model
Qwen2.5-7B-Instruct (production), loaded 4-bit quantized via BitsAndBytes +
Accelerate. Weights are frozen (`requires_grad_(False)` on all base model
parameters) — only the projection layer below is trainable, and even that
is trained by M7's PPO optimizer, not by this module.

**Model name must come from `configs/base_config.py`, never hardcoded.**
The architect develops locally on a 4GB GPU and will swap in a tiny
stand-in (e.g. Qwen2.5-0.5B-Instruct) for local testing, then switch to
the real 7B model only when running on a remote training GPU (see
AGENTS.md "Hardware & execution environment"). `LLMInterface` must load
whatever model name the config specifies, with no code path that assumes
7B specifically (e.g. don't hardcode `hidden_dim` — read it from the
loaded model's config).

## Public interface (matches shared/interfaces.md section 4)
```python
class LLMInterface:
    def __init__(self, model_name: str | None = None, device: str = "cuda"):
        # model_name=None reads from configs/base_config.py's default
        ...

    def encode(self, messages: list[dict]) -> torch.Tensor:
        """Forward pass, mean-pool last hidden state, project to (384,).
        The projection Linear(hidden_dim, 384) is exposed as
        `self.projection` — an nn.Module — so M7 can include its
        parameters in the PPO optimizer's parameter group."""
        ...

    def generate(self, messages: list[dict], json_schema: dict) -> dict:
        """Constrained generation matching json_schema, using Outlines
        or an equivalent logits processor. Returns the parsed dict, not
        raw text. Raises `GenerationError` if constrained decoding fails
        after retries (do not silently return an empty dict)."""
        ...
```

## Important implementation notes
- `self.projection` must be a real `nn.Module` (not a numpy operation) so
  it is optimizable by PyTorch. Initialize it with a fixed seed so results
  are reproducible run-to-run before training starts.
- `encode()` returns a (384,) tensor — this fills only the semantic region
  of the 512-d state vector (see shared/interfaces.md section 2). Do NOT
  attempt to produce the full 512-d vector here; that composition happens
  in M5.
- Batch encode() calls where possible — this is the most expensive
  operation in the whole pipeline (see performance note below).
- **Performance note (flag if this becomes a bottleneck):** naive
  per-environment-step encoding will be slow across thousands of PPO
  rollout steps. If conversation history is append-only, investigate
  reusing/caching hidden states for the unchanged prefix rather than
  recomputing from scratch every step. Report actual measured
  encode() latency in your summary so the architect can decide if this
  optimization is needed before M7 starts.

## Error handling
If the model fails to load (OOM, missing weights), raise a clear
`ModelLoadError` naming the failure — do not silently fall back to a
different model.

## Definition of Done
**All items below run against the tiny stand-in model (e.g.
Qwen2.5-0.5B-Instruct) locally — this proves the code path, not the
production model's behavior. Name tests accordingly (`test_encode_smoke`,
not `test_encode`).**
- [ ] `LLMInterface().encode(messages)` returns a `(384,)` tensor for a
  sample conversation, deterministic across repeated calls with the same
  input and same seed.
- [ ] `self.projection.parameters()` is iterable and has `requires_grad=True`;
  all other model parameters have `requires_grad=False` — test this
  explicitly.
- [ ] `generate()` with a simple schema (e.g. `{"query": str, "table": str}`)
  produces valid, schema-conforming JSON on 20/20 trials with varied inputs.
- [ ] Measured and reported: average `encode()` latency per call, and
  average `generate()` latency per call, using the stand-in model on local
  hardware (report separately once re-measured against the real 7B model
  on the training GPU — flag this as a follow-up in README.md, not done
  here).
- [ ] Config-driven model loading tested explicitly: instantiate
  `LLMInterface` with two different `model_name` values, confirm both load
  without code changes.
- [ ] `pytest tests/llm/ -v` passes, report actual count.
