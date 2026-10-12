"""Adapter for MiniCPM-V 4.6 (``MiniCPMV4_6ForConditionalGeneration``): ViT + merger + a hybrid-attention language model.

The job needs four things from a VLM, and this is all it asks of one (a Qwen3.5-VL adapter only has to provide the same four):

* ``preprocess(sample)``      chat-template the image(s) and the question into model inputs (generation prompt included);
* ``generate(inputs, n)``     the model's own greedy answer, as token ids;
* ``answer_logits(inputs, a)`` teacher-forced logits for the answer tokens ``a`` given the image and question -- the last
                              ``len(a) + 1`` positions, the first ``len(a)`` of which predict the answer tokens;
* ``prompt_forward(inputs)``  one forward pass over the prompt, used to calibrate (activations of every layer on real inputs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mqde.models.base import QuantModel

IMAGE_SIZE = (448, 448)


class MiniCPMV:
    family_name = "minicpmv"

    def __init__(self, path: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.path = str(path)
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(self.path, trust_remote_code=True, dtype=dtype)
        self.model.to(self.device).eval()
        self.qmodel = QuantModel(self.model, QuantModel.load_family(self.family_name))
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | ({self.tokenizer.eos_token_id} - {None})

    def preprocess(self, sample: Any) -> dict[str, torch.Tensor]:
        content: list[dict[str, Any]] = []
        for img in sample.images:
            content.append({"type": "image", "image": np.asarray(Image.fromarray(img).convert("RGB").resize(IMAGE_SIZE)).copy()})
        content.append({"type": "text", "text": sample.question})
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt", processor_kwargs={"padding": False})
        return {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    @torch.inference_mode()
    def generate(self, inputs: dict[str, torch.Tensor], max_new_tokens: int) -> list[int]:
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1, do_sample=False)
        ids = out[0][inputs["input_ids"].shape[1]:].tolist()
        cut = next((i for i, t in enumerate(ids) if t in self.eos_ids), len(ids))
        return ids[:cut] or ids[:1]

    def decode(self, ids: list[int]) -> str:
        return str(self.tokenizer.decode(ids, skip_special_tokens=True))

    @torch.inference_mode()
    def answer_logits(self, inputs: dict[str, torch.Tensor], answer: list[int]) -> torch.Tensor:
        n = len(answer)
        ids = torch.cat([inputs["input_ids"], torch.tensor([answer], dtype=inputs["input_ids"].dtype, device=self.device)], dim=1)
        kw = {**inputs, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
        try:
            return self.model(**kw, use_cache=False, return_dict=True, logits_to_keep=n + 1).logits[:, -(n + 1):]
        except TypeError:  # an implementation without logits_to_keep computes every position
            return self.model(**kw, use_cache=False, return_dict=True).logits[:, -(n + 1):]

    @torch.inference_mode()
    def prompt_forward(self, inputs: dict[str, torch.Tensor]) -> None:
        try:
            self.model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
        except TypeError:
            self.model(**inputs, use_cache=False, return_dict=True)
