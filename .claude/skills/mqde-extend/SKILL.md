---
name: mqde-extend
description: >
  Add a component to MQDEsys: a quantization algorithm, an export format, a serving backend, a model-family
  template, a recipe, or a KV-cache method. Gives the exact registration points, the data files that must change with
  the code, the tests to write, and which paths need maintainer approval. Use when the user says "add X algorithm",
  "support Y model", "new exporter", "add a backend", "new recipe". Do NOT use to run existing functionality (use
  mqde-quantize-eval or mqde-deploy-benchmark).
argument-hint: algorithm|exporter|backend|family|recipe|kv name
user-invocable: true
allowed-tools: Bash(python:*) Bash(python3:*) Bash(build/*:*) Bash(git:*) Read Edit Write Grep Glob
---

# Extend MQDEsys

Before anything: read `docs/STRUCTURE.md` §1 (change levels). **L0/L1** (new files under `python/`, new data entries,
front-end pages, tests) can be done directly. **L2** (public headers `cpp/include/**`, the compatibility and golden data
directories, build / CI files, AGENTS / ARCHITECTURE / STRUCTURE / REFERENCES docs) needs the maintainer's approval first: write
the proposal, record it in `internal/PROGRESS.md` under "待人工决策", and stop on that part. **L3** (new third-party
dependency, licence, breaking change) needs an ADR in `docs/adr/` and an explicit decision.

Never edit anything under `3rdpart/` (read-only upstream references), never change golden data to make a test pass,
and keep numerics in one place: `mqde._core` (C++) is the specification, Torch code must match it bit for bit.

## Algorithm

1. In `python/mqde/quant/algorithms.py` add `@register("name") def name(w, x, ws, as_, p) -> AlgoResult` (`w`
   float32 weight `[out, in]`, `x` captured activations `[rows, in]` or None, `ws` / `as_` weight and activation
   `QuantSpec`, `p` params). Return the weight to quantize with the recipe spec, or set `final=True` with `scale` /
   `zero` when the output is already on the grid (GPTQ).
2. **Category matters.** `resolve()` accepts at most one *transform* (`awq`, `smoothquant`: they set `in_scale`) and
   one *quantizer* (`rtn`, `gptq`) joined by `+`. A new transform or quantizer must be added to those tuples in
   `resolve()`, otherwise `name+gptq` is rejected. A transform's scales must fold into the preceding op if the model
   should stay exportable (`BlockwiseQuantizer._fold_block`).
3. Add an entry to `data/algorithms.yaml` (`name`, `label`, `kind: quantizer|transform|combo`, `description`,
   `needs_calibration`, `params`). The server validates `quant.algorithm` against this file and the UI lists it.
4. Tests in `python/tests/test_algorithms.py`: registration and `resolve`, output error versus RTN on a synthetic
   layer, grid membership when `final=True`, determinism. Run `PYTHONPATH=python python -m pytest python/tests/test_algorithms.py -q`.

## Exporter

1. New module under `python/mqde/export/`, class deriving `Exporter`, decorated `@register("fmt")`; set `backends`.
   Implement `check_layer()` (return the reason a layer cannot be exported, never raise), `export()` (writes
   `model.safetensors`, `config.json` with `quantization_config`, copies tokenizer files from `source_dir`).
2. Import the module in `python/mqde/export/__init__.py` (that is what registers it).
3. Make it readable by the fake/real check: `real_check.load_real` / `dequant_real` / `use_real_weights` dispatch on
   `quant_method`. A format that cannot be read back does not get the S10 guarantee.
4. Tests: layout (names, shapes, dtypes), dequantized weight equals the fake weight up to scale storage, mixed
   precision, `check()` rejections; `python/tests/export/test_cuda_export.py` pattern for GPU-quantized models. Compare
   against the reference implementation of the format byte for byte when its library is installable
   (`test_pack_compat.py`, `test_marlin_pack.py` show how; they skip when the library is absent).
5. If it needs a C++ packer: `cpp/src/quant/pack.cpp`, declaration in `cpp/include/mqde/quant/pack.h` (**L2**),
   binding in `cpp/bindings/python/core_module.cpp`, wrapper in `python/mqde/core.py`, test in `cpp/tests/unit/pack_test.cpp`.

## Serving backend

1. Subclass `ServerBackend` (`python/mqde/backend/base.py`): build the argument list and environment, implement
   `ready()`, set `diagnoses` as `(log substring, explanation)` pairs taken from **real** failure logs.
   `vllm_server.py` and `llamacpp_server.py` are the two examples.
2. Add the binary lookup (environment variable, then PATH, then `build/<name>/bin`) and make a missing binary a
   `skipped` stage in the job, not a failure.
3. Estimator profile: `data/backends/<name>.yaml` (`id`, `runtime_overhead_bytes`, `allocator_factor`, `eta_bw`,
   `mfu`). Changing what the estimator models (not just these numbers) touches `cpp/include/**` (L2).
4. Tests with the mock server in `python/tests/backend/fake_llama_server.py` (lifecycle, diagnosis from real log
   lines, load test accuracy) plus one real-binary test that skips unless an environment variable points at it.

## Model family

Prefer data over code: `data/model_families/<family>.yaml` with `family`, `modality`, `match.model_type`,
`config_paths`, `components` (name, modality, `prefix`, `block_regex`) and the layers kept in high precision. Check with
`build/dev/bin/mqde model inspect <MODEL_DIR>` that components and block counts come out right. Add the model to a
golden `ModelSpec` only through the L2 process. Do not publish checkpoint names that are not public.

## Recipe

`data/recipes/<name>.yaml` (`name`, `description`, `maturity`, `default_algorithm`, `modalities`, `weight`, `act`,
`kv`); `mqde recipes` must list it and `mqde estimate -r <name>` must run. Add an estimator test case for it.

## KV method

`@register("name")` class in `python/mqde/quant/kv.py` with `quant_k` / `quant_v`; it is selected by `quant.kv.method`.

## Before you finish

Run the local gate (mqde-dev-gate), update `internal/TASKS.md` (status and what is still missing) and add an entry at the
top of the log in `internal/PROGRESS.md`. Say plainly what was verified and what was not.
