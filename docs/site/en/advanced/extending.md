# Extending MQDEsys

Before changing anything read the review levels in `docs/STRUCTURE.md`: new Python files, data entries, UI pages and tests are directly editable; public C++ headers (`cpp/include/**`), build files, golden and compatibility data, and the architecture documents need maintainer approval first; a new third-party dependency needs an ADR. Never edit anything under `3rdpart/`, and never change golden data to make a test pass.

Each extension point below is a registry plus, usually, a data file. After any change run the local gate ([Testing](testing.md)).

## Add a quantization algorithm

1. In `python/mqde/quant/algorithms.py` add a function decorated with `@register("name")` and the signature `(w, x, ws, as_, p) -> AlgoResult`: `w` is the float32 weight `[out, in]`, `x` the captured input activations `[rows, in]` (or `None`), `ws` and `as_` the weight and activation `QuantSpec`, `p` the parameters. Return the weight to hand to the recipe quantizer, or set `final=True` with `scale` and `zero` when the result is already on the grid (as GPTQ does).
2. Mind the category: `resolve()` allows one transform (`awq`, `smoothquant`: they set `in_scale`) and one quantizer (`rtn`, `gptq`) joined by `+`. A new transform or quantizer must be added to those tuples, or `name+gptq` is rejected. A transform's scales must be foldable into the preceding operation if the model should stay exportable.
3. Add an entry to `data/algorithms.yaml` (`name`, `label`, `kind: quantizer|transform|combo`, `description`, `needs_calibration`, `params`); the server validates `quant.algorithm` against it and the UI lists it.
4. Tests in `python/tests/test_algorithms.py`: registration and `resolve`, error against RTN on a synthetic layer, grid membership, determinism.

## Add an export format

1. A new module under `python/mqde/export/` with a class deriving `Exporter`, decorated `@register("fmt")`. Implement `check_layer()` (return the reason a layer cannot be exported; never raise) and `export()` (write `model.safetensors`, `config.json` with `quantization_config`, copy the tokenizer files).
2. Import the module in `python/mqde/export/__init__.py`; that import registers it.
3. Make it readable by the consistency check: `real_check.load_real`, `dequant_real` and `use_real_weights` dispatch on `quant_method`. A format that cannot be read back does not get the S10 guarantee.
4. Tests: layout, dequantized weight equals the fake weight up to scale storage, mixed precision, `check()` rejections; compare byte for byte with the format's reference library when it is installable (see `python/tests/test_pack_compat.py`).
5. A C++ packer needs `cpp/src/quant/pack.cpp`, a header in `cpp/include/mqde/quant/` (approval), a pybind binding and a wrapper in `python/mqde/core.py`.

## Add a serving backend

1. Subclass `ServerBackend` (`python/mqde/backend/base.py`): build the argument list and environment, implement `ready()`, and set `diagnoses` as (log substring, explanation) pairs taken from **real** failure logs. `vllm_server.py` and `llamacpp_server.py` are the examples.
2. Look up the binary through an environment variable, then `PATH`, then `build/<name>/bin`; a missing binary must make the job stage `skipped`, not failed.
3. Add the estimator profile `data/backends/<name>.yaml` (`id`, `runtime_overhead_bytes`, `allocator_factor`, `eta_bw`, `mfu`).
4. Test with the mock server in `python/tests/backend/fake_llama_server.py` (lifecycle, diagnosis from real log lines, timing accuracy) plus one real-binary test that skips unless an environment variable points at it.

## Add a model family

Prefer data over code: `data/model_families/<family>.yaml` with `family`, `modality`, `match.model_type`, `config_paths`, `components` (name, modality, `prefix`, `block_regex`) and `default_keep_fp` (layers kept in high precision). Verify with `mqde model inspect <MODEL_DIR>` that components and block counts come out right. Architectures that Hugging Face transformers loads as causal LMs already work through `generic_decoder` and the `HFCausalLM` adapter; a new Python `QuantModel` subclass is only needed for unusual structures.

## Add a recipe or a KV method

A recipe is `data/recipes/<name>.yaml` (`name`, `description`, `maturity`, `default_algorithm`, `modalities`, `weight`, `act`, `kv`); `mqde recipes` must list it and `mqde estimate -r <name>` must run. A KV method is a class registered with `@register("name")` in `python/mqde/quant/kv.py` providing `quant_k` and `quant_v`; it is selected with `quant.kv.method`.

## Add a hardware profile, an API route or a page

Hardware: `data/hardware_db/<vendor>/<id>.yaml`. REST route: `cpp/src/server/server.cpp` plus the typed client in `web/src/api.ts`. Page: `web/src/pages/<Name>.tsx` and the route and navigation entry in `web/src/main.tsx`. A hardware *plugin* (vendor probe or micro-benchmark through a stable C ABI) is planned, not available.
