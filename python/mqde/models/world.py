"""World-model adapters (DiT denoisers) for the quantization pipeline.

Two backends are provided:

* :class:`TinyDiTWorld` — a self-contained, deterministic DiT with the module layout described by
  ``data/model_families/dit_world.yaml``. It runs on CPU in a second and is what the demo checkpoint
  and the unit tests use. Its weights can be seeded (no checkpoint file) or loaded from
  ``model.safetensors`` / ``model.pt``.
* :class:`DiffusersWorld` — an optional runtime adapter for a Hugging Face ``diffusers`` checkpoint.
  ``diffusers`` is not a declared dependency (adding one is an L3 decision); if it is importable, a
  local text-to-video/image pipeline is wrapped so the same report can be produced. Only prompt
  conditioning is supported; an action-conditioned EWM checkpoint needs a model-specific adapter.

The job needs three calls from either adapter:

``preprocess(sample)``   prompt / action tensors + the real frames as ``x0``
``calibrate(inputs)``    one denoising forward pass on a real calibration clip (activation capture)
``generate(sample, seed, inputs)``  a deterministic sampled clip ``[T, 3, H, W]`` in ``[0, 1]``
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mqde.models.base import QuantModel


def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half))
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)[:, :dim]


def prompt_vector(prompt: str, dim: int) -> torch.Tensor:
    """Stable pseudo-embedding for the built-in model (a real checkpoint uses its text encoder)."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    vec = np.resize(raw, dim) * 2.0 - 1.0
    return torch.from_numpy(vec.astype(np.float32))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm1.linear = nn.Linear(dim, 6 * dim)
        self.attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(dim, dim),
                "k_proj": nn.Linear(dim, dim),
                "v_proj": nn.Linear(dim, dim),
                "out_proj": nn.Linear(dim, dim),
            }
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2.linear = nn.Linear(dim, 6 * dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.ModuleDict({"fc1": nn.Linear(dim, hidden), "fc2": nn.Linear(hidden, dim)})

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.norm1.linear(F.silu(t_emb)).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        b, n, d = h.shape
        # Call the modules (not a bare matmul): after quantization these are FakeQuantLinear and
        # their forward applies activation quantization / the input scale as well as the weight.
        q, k, v = self.attn["q_proj"](h), self.attn["k_proj"](h), self.attn["v_proj"](h)
        q, k, v = (z.view(b, n, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(b, n, d)
        a = self.attn["out_proj"](a)
        x = x + gate1[:, None, :] * a
        h2 = self.norm2(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        h2 = F.silu(self.mlp["fc1"](h2))
        h2 = self.mlp["fc2"](h2)
        return x + gate2[:, None, :] * h2


class TinyDiT(nn.Module):
    """A small DiT with the tensor layout the `dit_world` family expects."""

    def __init__(self, *, image_size: int = 32, patch_size: int = 8, in_channels: int = 3,
                 dim: int = 128, depth: int = 2, heads: int = 4, cond_dim: int = 32,
                 action_dim: int = 128, max_frames: int = 16) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.grid = image_size // patch_size
        self.n_tokens = self.grid * self.grid
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.ModuleDict({"proj": nn.Linear(patch_dim, dim)})
        self.conditioner = nn.ModuleDict(
            {
                "prompt": nn.Linear(cond_dim, dim),
                "time": nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)),
            }
        )
        self.action_encoder = nn.ModuleDict({"proj": nn.Linear(action_dim, dim)})
        self.register_buffer("pos_embed", torch.zeros(1, self.n_tokens, dim), persistent=True)
        self.register_buffer("frame_embed", torch.zeros(1, max_frames, dim), persistent=True)
        self.transformer_blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.final_layer = nn.ModuleDict({"norm": nn.LayerNorm(dim), "linear": nn.Linear(dim, patch_dim)})
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.frame_embed, std=0.02)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        t, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        return self.patch_embed["proj"](patches)

    def _unpatch(self, tokens: torch.Tensor, out_shape: tuple[int, int, int, int]) -> torch.Tensor:
        t, _, h, w = out_shape
        y = self.final_layer["linear"](self.final_layer["norm"](tokens))
        y = y.transpose(1, 2)
        return F.fold(y, output_size=(h, w), kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                actions: torch.Tensor | None = None) -> torch.Tensor:
        """Predict the velocity ``noise - x0`` for a batch of clips ``x`` ``[T, C, H, W]``."""
        t_emb = self.conditioner["time"](_sinusoidal(t.expand(x.shape[0]), self.dim))
        t_emb = t_emb + self.conditioner["prompt"](cond)
        tokens = self._tokens(x) + self.pos_embed
        frame_ids = torch.arange(x.shape[0], device=x.device).clamp_max(self.frame_embed.shape[1] - 1)
        tokens = tokens + self.frame_embed[0, frame_ids][:, None]
        if actions is not None:
            tokens = tokens + self.action_encoder["proj"](actions)[:, None]
        h = tokens
        for block in self.transformer_blocks:
            h = block(h, t_emb)
        return self._unpatch(h, tuple(x.shape))


def _load_config(path: Path) -> dict[str, Any]:
    for name in ("world_config.json", "config.json"):
        p = path / name
        if p.is_file():
            cfg = json.loads(p.read_text())
            if name == "config.json" and isinstance(cfg.get("world_config"), dict):
                return {**cfg, **cfg["world_config"], "config_json": cfg}
            return cfg
    raise FileNotFoundError(f"{path}: no world_config.json or config.json (not a world-model checkpoint)")


def build_tiny(cfg: dict[str, Any]) -> TinyDiT:
    return TinyDiT(
        image_size=int(cfg.get("image_size", 32)),
        patch_size=int(cfg.get("patch_size", 8)),
        in_channels=int(cfg.get("in_channels", 3)),
        dim=int(cfg.get("dim", 128)),
        depth=int(cfg.get("depth", 2)),
        heads=int(cfg.get("heads", 4)),
        cond_dim=int(cfg.get("cond_dim", 32)),
        action_dim=int(cfg.get("action_dim", 128)),
        max_frames=int(cfg.get("frames", 4)) + 4,
    )


def save_tiny_checkpoint(path: str | Path, **overrides: Any) -> Path:
    """Write the deterministic demo checkpoint used by the UI and the smoke test (no training)."""
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {
        "format": "mqde-world-dit",
        "arch": "tiny_dit",
        "model_type": "mqde_dit",
        "image_size": 32,
        "patch_size": 8,
        "in_channels": 3,
        "dim": 128,
        "depth": 2,
        "heads": 4,
        "cond_dim": 32,
        "action_dim": 128,
        "frames": 4,
        "sample_steps": 8,
        "seed": 0,
        "untrained": True,
        "weights": "seeded",
        **overrides,
    }
    torch.manual_seed(int(cfg["seed"]))
    model = build_tiny(cfg)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path / "model.safetensors"))
    (path / "world_config.json").write_text(json.dumps(cfg, indent=2))
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TinyDiT"],
                "model_type": "mqde_dit",
                "num_hidden_layers": cfg["depth"],
                "hidden_size": cfg["dim"],
                "num_attention_heads": cfg["heads"],
                "vision_config": {
                    "patch_size": cfg["patch_size"],
                    "image_size": cfg["image_size"],
                    "num_hidden_layers": cfg["depth"],
                    "hidden_size": cfg["dim"],
                },
                "world_config": cfg,
            },
            indent=2,
        )
    )
    return path


class TinyDiTWorld:
    """Adapter around :class:`TinyDiT`; deterministic and CPU-friendly."""

    family_name = "dit_world"

    def __init__(self, path: str | Path, device: str = "cpu", action_conditioned: bool = False,
                 dtype: torch.dtype = torch.float32, seed: int = 0) -> None:
        self.path = str(path)
        self.device = torch.device(device)
        self.action_conditioned = action_conditioned
        self.cfg = _load_config(Path(path))
        if self.cfg.get("arch", "tiny_dit") != "tiny_dit":
            raise ValueError(f"{path}: arch '{self.cfg.get('arch')}' is not tiny_dit; use DiffusersWorld")
        self.frames = int(self.cfg.get("frames", 4))
        self.image_size = int(self.cfg.get("image_size", 32))
        self.action_dim = int(self.cfg.get("action_dim", 8))
        self.steps = int(self.cfg.get("sample_steps", 8))
        self.family_name = "dit_ewm" if action_conditioned else "dit_world"
        torch.manual_seed(int(self.cfg.get("seed", seed)))
        self.model = build_tiny(self.cfg).to(self.device).eval()
        sf = Path(path) / "model.safetensors"
        pt = Path(path) / "model.pt"
        if sf.is_file():
            from safetensors.torch import load_file

            self.model.load_state_dict(load_file(str(sf), device=str(self.device)))
            self.weights_source = "model.safetensors"
        elif pt.is_file():
            self.model.load_state_dict(torch.load(pt, map_location=self.device, weights_only=True))
            self.weights_source = "model.pt"
        else:
            self.weights_source = "seeded (no weight file)"
        self.qmodel = QuantModel(self.model, QuantModel.load_family(self.family_name))

    def preprocess(self, sample: Any) -> dict[str, Any]:
        frames = torch.from_numpy(np.asarray(sample.frames, dtype=np.float32).transpose(0, 3, 1, 2))
        cond = prompt_vector(sample.prompt, int(self.cfg.get("cond_dim", 32))).to(self.device)
        actions = None
        if self.action_conditioned:
            a = sample.actions
            if a is None:
                a = np.zeros((frames.shape[0], self.action_dim), dtype=np.float32)
            a = np.asarray(a, dtype=np.float32)[: frames.shape[0]]
            if a.shape[1] < self.action_dim:
                a = np.pad(a, ((0, 0), (0, self.action_dim - a.shape[1])))
            actions = torch.from_numpy(a[:, : self.action_dim]).to(self.device)
        return {"cond": cond, "actions": actions, "x0": frames.to(self.device), "prompt": sample.prompt}

    def _forward(self, x: torch.Tensor, t: torch.Tensor, inputs: dict[str, Any]) -> torch.Tensor:
        return self.model(x, t, inputs["cond"].expand(x.shape[0], -1),
                          inputs["actions"] if self.action_conditioned else None)

    @torch.inference_mode()
    def calibrate(self, inputs: dict[str, Any], seed: int = 0) -> None:
        x0 = inputs["x0"] * 2.0 - 1.0
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(seed)
            noise = torch.randn_like(x0)
            t = torch.rand(x0.shape[0], device=x0.device)
        xt = (1.0 - t[:, None, None, None]) * x0 + t[:, None, None, None] * noise
        self._forward(xt, t, inputs)

    @torch.inference_mode()
    def generate(self, sample: Any, seed: int = 0, inputs: dict[str, Any] | None = None,
                 steps: int | None = None) -> torch.Tensor:
        inputs = inputs or self.preprocess(sample)
        n = int(inputs["x0"].shape[0])
        steps = steps or self.steps
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(seed)
            x = torch.randn(n, int(self.cfg.get("in_channels", 3)), self.image_size, self.image_size, device=self.device)
        for i in range(steps):
            t = torch.full((n,), 1.0 - i / steps, device=self.device)
            v = self._forward(x, t, inputs)
            x = x - v / steps
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0).float().cpu()


class DiffusersWorld:
    """Runtime adapter for a local diffusers text-to-video/image checkpoint.

    This is optional and deliberately not a declared dependency. Only prompt conditioning is
    supported; an action-conditioned EWM checkpoint needs a model-specific adapter.
    """

    family_name = "dit_diffusers"

    def __init__(self, path: str | Path, device: str = "cuda:0", action_conditioned: bool = False,
                 dtype: torch.dtype = torch.float16, seed: int = 0) -> None:
        if action_conditioned:
            raise ValueError("DiffusersWorld does not support action-conditioned EWM checkpoints yet")
        try:
            import diffusers  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RuntimeError("diffusers is not installed; use the built-in tiny_dit checkpoint or install diffusers") from e
        from diffusers import DiffusionPipeline

        self.path = str(path)
        self.device = torch.device(device)
        self.pipe = DiffusionPipeline.from_pretrained(self.path, torch_dtype=dtype, local_files_only=True)
        self.pipe.to(self.device)
        transformer = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
        if transformer is None:
            raise ValueError(f"{path}: diffusers pipeline has no transformer/unet to quantize")
        self.model = transformer.eval()
        self.weights_source = "diffusers"
        self.action_dim = 0
        self.family_name = "dit_diffusers"
        self.qmodel = QuantModel(self.model, QuantModel.load_family(self.family_name))
        self.steps = int(getattr(self.pipe, "num_inference_steps", 20) or 20)

    def preprocess(self, sample: Any) -> dict[str, Any]:
        return {"prompt": sample.prompt, "frames_hint": int(sample.frames.shape[0])}

    def _run(self, prompt: str, frames: int, seed: int, steps: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device).manual_seed(seed)
        kwargs: dict[str, Any] = {"prompt": prompt, "generator": gen, "num_inference_steps": steps, "output_type": "np"}
        for key in ("num_frames", "video_length"):
            try:
                return self._frames_to_tensor(self.pipe(**kwargs, **{key: frames}).frames)
            except TypeError:
                continue
        return self._frames_to_tensor(self.pipe(**kwargs).frames)

    @staticmethod
    def _frames_to_tensor(frames: Any) -> torch.Tensor:
        arr = np.asarray(frames)
        if arr.ndim == 6:
            arr = arr[0]
        elif arr.ndim == 4 and arr.shape[-1] == 3:
            arr = arr[None]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"diffusers returned frames with shape {arr.shape}, expected [T, H, W, 3]")
        return torch.from_numpy(arr.astype(np.float32)).permute(0, 3, 1, 2).clamp(0.0, 1.0).cpu()

    @torch.inference_mode()
    def calibrate(self, inputs: dict[str, Any], seed: int = 0) -> None:
        self._run(str(inputs["prompt"]), int(inputs.get("frames_hint", 4)), seed, steps=2)

    @torch.inference_mode()
    def generate(self, sample: Any, seed: int = 0, inputs: dict[str, Any] | None = None,
                 steps: int | None = None) -> torch.Tensor:
        inputs = inputs or self.preprocess(sample)
        return self._run(str(inputs["prompt"]), int(inputs.get("frames_hint", 4)), seed, steps or self.steps)


def load_world_adapter(path: str | Path, device: str = "cpu", action_conditioned: bool = False,
                       seed: int = 0) -> TinyDiTWorld | DiffusersWorld:
    p = Path(path)
    if (p / "model_index.json").is_file():
        return DiffusersWorld(p, device=device, action_conditioned=action_conditioned, seed=seed)
    cfg = _load_config(p)
    if cfg.get("arch", "tiny_dit") == "tiny_dit":
        return TinyDiTWorld(p, device=device, action_conditioned=action_conditioned, seed=seed)
    raise ValueError(f"{path}: unknown world-model arch '{cfg.get('arch')}'")


if __name__ == "__main__":  # tiny demo checkpoint: python -m mqde.models.world /path/to/dir
    import argparse

    ap = argparse.ArgumentParser(description="Create the built-in TinyDiT demo checkpoint")
    ap.add_argument("path", help="output directory")
    args = ap.parse_args()
    print(save_tiny_checkpoint(args.path))
