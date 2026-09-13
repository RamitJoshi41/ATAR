"""
M3 LLM Interface — wraps a frozen instruction-tuned LLM for:
  (a) state encoding: mean-pool last hidden state → project to (384,)
  (b) constrained JSON generation via Outlines 1.3.x

Model name is always read from configs/base_config.py (or passed explicitly),
never hardcoded, so the architect can swap between the 0.5B stand-in (local
4 GB GPU) and the 7B production model (remote GPU) by changing a config value.
"""

import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import outlines
import outlines.models
from outlines.models import Transformers

from atar.configs.base_config import load_config


class ModelLoadError(Exception):
    """Raised when the underlying model fails to load."""
    pass


class GenerationError(Exception):
    """Raised when constrained generation fails after retries."""
    pass


class LLMInterface:
    """
    Frozen LLM wrapper providing encode() and generate().

    Parameters
    ----------
    model_name : str | None
        HuggingFace model name/path.  None reads from configs/base_config.py.
    device : str
        'cuda' or 'cpu'.  On CPU the model is loaded in full precision
        (BitsAndBytes requires CUDA).
    """

    def __init__(self, model_name: str | None = None, device: str = "cuda") -> None:
        if model_name is None:
            config = load_config()
            model_name = config.llm_model_name

        self.model_name = model_name
        self.device = device

        try:
            # BitsAndBytes 4-bit quantisation requires CUDA.
            if self.device == "cuda" and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=self.device,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            raise ModelLoadError(f"Failed to load model '{model_name}': {e}") from e

        # Freeze the base model — only self.projection is trainable.
        self.base_model.requires_grad_(False)

        # Read hidden dimension dynamically so no hardcoded value lives here.
        hidden_size: int = self.base_model.config.hidden_size

        # Trainable projection layer; seeded for reproducibility before training.
        torch.manual_seed(42)
        proj_dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.projection = nn.Linear(hidden_size, 384, dtype=proj_dtype).to(
            self.base_model.device
        )
        self.projection.requires_grad_(True)

        # Wrap with Outlines for constrained generation (Outlines 1.3.x API).
        # outlines.models.Transformers(model, tokenizer) is the constructor.
        try:
            self.outlines_model: Transformers = outlines.models.Transformers(
                self.base_model, self.tokenizer
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to build Outlines wrapper for '{model_name}': {e}"
            ) from e

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(self, messages: list[dict]) -> torch.Tensor:
        """
        Forward pass → mean-pool last hidden state → project to (384,).

        Returns
        -------
        torch.Tensor
            Shape (384,), dtype float32.
        """
        prompt: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.base_model.device
        )

        with torch.no_grad():
            outputs = self.base_model(**inputs, output_hidden_states=True)

        # last_hidden_state: (1, seq_len, hidden_size)
        last_hidden = outputs.hidden_states[-1]

        # Mean pool over sequence dimension → (1, hidden_size)
        mean_pooled = last_hidden.mean(dim=1)

        # Cast to projection dtype and apply linear → (1, 384)
        projected = self.projection(mean_pooled.to(self.projection.weight.dtype))

        # Return (384,) on CPU for downstream consistency
        return projected.squeeze(0).to(torch.float32).detach()

    def generate(
        self,
        messages: list[dict],
        json_schema: dict,
        max_new_tokens: int = 256,
    ) -> dict:
        """
        Constrained JSON generation that matches *json_schema*.

        Uses Outlines 1.3.x callable API:
          outlines_model(prompt, output_type=outlines.json_schema(schema), ...)

        Parameters
        ----------
        messages : list[dict]
            Conversation in OpenAI chat format.
        json_schema : dict
            JSON Schema dict the output must conform to.
        max_new_tokens : int
            Maximum tokens to generate (default 256).  The model-agnostic
            default max_length is often only 76 *total* tokens, which is
            too short to close all JSON strings — always override it here.

        Returns
        -------
        dict
            Parsed JSON dict conforming to json_schema.

        Raises
        ------
        GenerationError
            If constrained decoding fails after max_retries attempts.
        """
        prompt: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Build the output type descriptor (outlines 1.3.x).
        output_type = outlines.json_schema(json_schema)

        max_retries = 3
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                result = self.outlines_model(
                    prompt,
                    output_type=output_type,
                    max_new_tokens=max_new_tokens,
                )

                # outlines may return a Pydantic model or a plain dict.
                if hasattr(result, "model_dump"):
                    return result.model_dump()
                if hasattr(result, "dict"):
                    return result.dict()  # type: ignore[attr-defined]
                if isinstance(result, dict):
                    return result
                if isinstance(result, str):
                    return json.loads(result)
                raise GenerationError(
                    f"Unexpected output type from Outlines: {type(result)}"
                )
            except GenerationError:
                raise
            except Exception as e:
                last_exc = e
                # Retry on transient failures.

        raise GenerationError(
            f"Failed to generate constrained JSON after {max_retries} attempts: "
            f"{last_exc}"
        ) from last_exc
