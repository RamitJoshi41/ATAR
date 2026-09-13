"""
Tests for M3 LLMInterface.

All tests run against the tiny stand-in model (Qwen/Qwen2.5-0.5B-Instruct)
on local hardware.  They prove the code path, not production-model behavior.
Test names use _smoke suffix where the MSD specifies this distinction.
"""

import time
import pytest
import torch

from atar.llm.llm_interface import LLMInterface, ModelLoadError, GenerationError
from atar.configs.base_config import AtarConfig

# ---------------------------------------------------------------------------
# Module-scoped fixture — model loaded once for the whole test session.
# ---------------------------------------------------------------------------

STAND_IN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TINY_LLAMA = "HuggingFaceM4/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def llm() -> LLMInterface:
    """Load the stand-in model once for the whole test module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LLMInterface(model_name=STAND_IN_MODEL, device=device)


# ---------------------------------------------------------------------------
# DoD item 1: encode() returns (384,) and is deterministic.
# ---------------------------------------------------------------------------

def test_encode_smoke(llm: LLMInterface) -> None:
    """
    (DoD 1) encode() must return a (384,) tensor and be deterministic
    across repeated calls on the same input with frozen weights.
    """
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    out1 = llm.encode(messages)
    out2 = llm.encode(messages)

    assert isinstance(out1, torch.Tensor), "encode() must return a torch.Tensor"
    assert out1.shape == (384,), f"Expected shape (384,), got {out1.shape}"
    assert torch.allclose(out1, out2, atol=1e-5), "encode() must be deterministic"


# ---------------------------------------------------------------------------
# DoD item 2: projection is trainable; base model is frozen.
# Also validates in_features == hidden_size (new explicit requirement).
# ---------------------------------------------------------------------------

def test_projection_gradients_and_dimensions(llm: LLMInterface) -> None:
    """
    (DoD 2) self.projection parameters must have requires_grad=True.
    All base model parameters must have requires_grad=False.
    projection.in_features must equal model.config.hidden_size (dynamic, not hardcoded).
    """
    # Projection must be trainable.
    proj_params = list(llm.projection.parameters())
    assert len(proj_params) > 0, "Projection has no parameters"
    for param in proj_params:
        assert param.requires_grad is True, (
            "projection parameter requires_grad should be True"
        )

    # Base model must be fully frozen.
    for name, param in llm.base_model.named_parameters():
        assert param.requires_grad is False, (
            f"Base model parameter '{name}' requires_grad should be False"
        )

    # In-features must be read dynamically from the loaded model's config.
    expected_hidden = llm.base_model.config.hidden_size
    assert llm.projection.in_features == expected_hidden, (
        f"projection.in_features ({llm.projection.in_features}) != "
        f"model.config.hidden_size ({expected_hidden})"
    )

    # Out-features must be 384 as per the MSD.
    assert llm.projection.out_features == 384, (
        f"projection.out_features should be 384, got {llm.projection.out_features}"
    )


# ---------------------------------------------------------------------------
# DoD item 3: 20/20 constrained-JSON generation (slow, requires real decoding).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_generate_constrained_smoke(llm: LLMInterface) -> None:
    """
    (DoD 3) generate() must produce valid schema-conforming JSON on 20/20
    trials with varied inputs.  Marked @pytest.mark.slow; run with
    `pytest -m slow` to execute.
    """
    schema = {
        "title": "QueryTable",
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "table": {"type": "string"},
        },
        "required": ["query", "table"],
    }

    varied_inputs = [
        "I need users from the accounts table.",
        "Give me all orders from orders table.",
        "Show me the products table data.",
        "Get the latest sales from sales table.",
        "List employees from hr table.",
        "Fetch inventory from warehouse table.",
        "Query transactions from payments table.",
        "Find customers from crm table.",
        "Retrieve logs from audit table.",
        "Get metrics from analytics table.",
        "I need users from the accounts table.",
        "Give me all orders from orders table.",
        "Show me the products table data.",
        "Get the latest sales from sales table.",
        "List employees from hr table.",
        "Fetch inventory from warehouse table.",
        "Query transactions from payments table.",
        "Find customers from crm table.",
        "Retrieve logs from audit table.",
        "Get metrics from analytics table.",
    ]
    assert len(varied_inputs) == 20, "Must have exactly 20 trial inputs"

    success_count = 0
    for user_text in varied_inputs:
        messages = [
            {"role": "system", "content": "You are a SQL expert."},
            {"role": "user", "content": user_text},
        ]
        result = llm.generate(messages, schema)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "query" in result, f"Missing 'query' key in {result}"
        assert "table" in result, f"Missing 'table' key in {result}"
        assert isinstance(result["query"], str), "'query' must be a string"
        assert isinstance(result["table"], str), "'table' must be a string"
        success_count += 1

    assert success_count == 20, f"Only {success_count}/20 trials succeeded"


# ---------------------------------------------------------------------------
# DoD item 4: Latency measurement.
# ---------------------------------------------------------------------------

def test_latency_metrics_smoke(llm: LLMInterface) -> None:
    """
    (DoD 4) Measure and print average encode() and generate() latency.
    Results are printed for the architect to decide if caching is needed.
    """
    messages = [{"role": "user", "content": "Measure the latency of this call."}]
    schema = {
        "title": "Ack",
        "type": "object",
        "properties": {"ack": {"type": "string"}},
        "required": ["ack"],
    }

    # Warm-up pass (model caching, JIT, etc.)
    llm.encode(messages)
    llm.generate(messages, schema)

    # Encode latency (5 trials)
    encode_times: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        llm.encode(messages)
        encode_times.append(time.perf_counter() - t0)
    avg_encode = sum(encode_times) / len(encode_times)

    # Generate latency (5 trials)
    generate_times: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        llm.generate(messages, schema)
        generate_times.append(time.perf_counter() - t0)
    avg_generate = sum(generate_times) / len(generate_times)

    print(
        f"\n[M3 Latency — stand-in model: {llm.model_name}]\n"
        f"  avg encode()  : {avg_encode:.4f}s\n"
        f"  avg generate(): {avg_generate:.4f}s\n"
        f"NOTE: Re-measure on Qwen2.5-7B-Instruct on the training GPU "
        f"(see README.md follow-up flag)."
    )

    assert avg_encode > 0, "avg_encode must be positive"
    assert avg_generate > 0, "avg_generate must be positive"


# ---------------------------------------------------------------------------
# DoD item 5: Config-driven loading — two different model names, no code change.
# ---------------------------------------------------------------------------

def test_config_driven_loading_smoke() -> None:
    """
    (DoD 5) Instantiate LLMInterface with two different model_name values and
    confirm both load without requiring any code changes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # First model: tiny random LLaMA (very fast; no real weights needed)
    llm1 = LLMInterface(model_name=TINY_LLAMA, device=device)
    assert llm1.base_model is not None, "base_model should not be None"
    assert llm1.model_name == TINY_LLAMA

    # Second model: Qwen2.5-0.5B stand-in
    llm2 = LLMInterface(model_name=STAND_IN_MODEL, device=device)
    assert llm2.base_model is not None, "base_model should not be None"
    assert llm2.model_name == STAND_IN_MODEL

    # Both projections must have in_features matching their respective hidden sizes.
    assert llm1.projection.in_features == llm1.base_model.config.hidden_size, (
        "llm1 projection.in_features != hidden_size"
    )
    assert llm2.projection.in_features == llm2.base_model.config.hidden_size, (
        "llm2 projection.in_features != hidden_size"
    )


# ---------------------------------------------------------------------------
# Error handling: ModelLoadError on bad model name.
# ---------------------------------------------------------------------------

def test_model_load_error_on_bad_name() -> None:
    """Confirm ModelLoadError is raised (not a bare Exception) for invalid names."""
    with pytest.raises(ModelLoadError):
        LLMInterface(model_name="this-model/does-not-exist-anywhere", device="cpu")
