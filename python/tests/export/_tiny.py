from __future__ import annotations

import torch
import transformers

from mqde.models.hf_llm import HFCausalLM


def tiny(tied: bool = True) -> HFCausalLM:
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=128, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=tied,
    )
    cfg._attn_implementation = "eager"
    return HFCausalLM.from_model(transformers.LlamaForCausalLM(cfg).float())


def calib(n: int = 4, s: int = 32) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 128, (1, s), generator=g) for _ in range(n)]
