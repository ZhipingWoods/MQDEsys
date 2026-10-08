[中文](README.md) | **English**

# MQDEsys: Model Quantization & Deployment Evaluation System

Give it a model, a quantization recipe (W8A16 / FP8 / W4A16 / W4A8 / W2A16 / W2A8), evaluation data and target hardware, and the system will:

1. **Estimate** (C++, no weights loaded): model footprint, KV cache / linear-attention state and activations; peak memory for single and multiple concurrent requests, maximum concurrency, a latency floor, and a "recipe x hardware" deployability matrix.
2. **Quantize and evaluate** (Python executor, scheduled onto a GPU pool by C++): build a BF16 baseline, attribute error per modality, run fake-quantization and trace the error layer by layer.
3. **Visualize** (web): live execution and logs, per-layer error heat maps, a worst-layers list and recipe comparison.

The first supported model is the VLA model MiniCPM-RobotManip.

Documentation (the user guide is built from `docs/site/`, in English and Chinese; see `docs/site/README.md`):

| File | Content |
|------|---------|
| [AGENTS.md](AGENTS.md) | Working rules for agents (entry point, in Chinese) |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Environment, build, test, debug, commit (Chinese) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design (Chinese) |
| [docs/STRUCTURE.md](docs/STRUCTURE.md) | What every file is for and who may change it (Chinese) |
| [docs/TASKS.md](docs/TASKS.md) | Task list and acceptance criteria (Chinese) |
| [docs/REFERENCES.md](docs/REFERENCES.md) | Open-source projects used as references; appendix A has the llmfit algorithm details (Chinese) |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Session log and current progress (Chinese) |

## Quick start (development machine with an NVIDIA GPU)

The commands below are the ones from the Chinese README. `<MODEL_DIR>` is a local Hugging Face model directory (for example a MiniCPM-RobotManip checkout) and `<MODEL_PARENT_DIR>` its parent; an environment with the C++ dependencies (conda or Homebrew) is assumed.

```bash
cmake --preset dev && cmake --build --preset dev -j 32 && ctest --preset dev
PYTHONPATH=python python -m pytest python/tests -q

build/dev/bin/mqde probe
build/dev/bin/mqde model inspect <MODEL_DIR>
build/dev/bin/mqde estimate -m <MODEL_DIR> -r bf16,w4a16,fp8 --images 2 --conc 1,8,64

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python -m mqde.pipeline.vla_job examples/jobs/minicpm_vla_libero.yaml

(cd web && npm install && npm run build)
build/dev/bin/mqde serve --repo $PWD --runs $PWD/runs --web $PWD/web/dist \
  --python $(which python) --gpus 2,3 --allow <MODEL_PARENT_DIR> --port 18080
# From another machine: ssh -N -L 18080:127.0.0.1:18080 <dev-server>, then open http://localhost:18080
```

The user guide has a step-by-step LLM walkthrough that does not need a GPU: `docs/site/en/get_started/quickstart.md`.

## Layout

| Path | Content |
|------|---------|
| `cpp/` | C++ core: common, modelspec, quant (the normative numerics), estimator, server; C hardware probe, CLI, pybind11 binding |
| `python/mqde/` | Executor: Torch numerics mirror, model adapters, data, fake-quant engine and tracing, job pipelines |
| `data/` | Recipes, hardware profiles, backend parameters, model architecture templates (data-driven; adding one needs no code change) |
| `web/` | React + ECharts front end |
| `examples/jobs/` | Example job files |
| `docs/` | Architecture, file responsibilities, task list, progress; `docs/site/` is the user documentation site |

The reference libraries in `3rdpart/` are git submodules (on GitHub, clicking a directory jumps to the upstream commit); their sources are not distributed with this repository. `runs/`, `build/`, model weights and `*.gguf` files are large and not tracked.

## License

MIT, see [LICENSE](LICENSE).
