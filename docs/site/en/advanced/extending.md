# Extending the system

MQDEsys extensions are built around registries and data files. Adding a capability generally requires one module or one YAML entry, without modifying the main workflow. Run the checks in [Testing and quality gates](testing.md) after changes.

## Adding a quantization algorithm

1. Add a function decorated with `@register("name")` in `python/mqde/quant/algorithms.py`, with the signature `(w, x, ws, as_, p) -> AlgoResult`. `w` is a float32 weight tensor `[out, in]`, `x` is the captured input activation `[rows, in]` (may be `None`), `ws` and `as_` are the weight and activation `QuantSpec`, and `p` holds parameters. Return the weights passed to the recipe quantizer; if the result already lies on the quantization grid (as with GPTQ), set `final=True` and provide `scale` and `zero`.
2. Follow the composition rules: `resolve()` accepts one transform (`awq`, `smoothquant`, which set `in_scale`) joined by `+` to one quantizer (`rtn`, `gptq`, `hqq`). New transforms or quantizers must be added to the corresponding tuples, otherwise `name+gptq` forms are rejected. To keep the model exportable, the transform scaling must fold into the preceding operator.
3. Add an entry to `data/algorithms.yaml` with `name`, `label`, `kind` (`quantizer`, `transform` or `combo`), `description`, `needs_calibration` and `params`. The server validates `quant.algorithm` against this file, and the web UI builds its options from it.
4. Add tests in `python/tests/test_algorithms.py` covering registration and `resolve`, error relative to RTN on a synthetic layer, quantization-grid membership and determinism.

## Adding an export format

1. Create a module under `python/mqde/export/`, subclass `Exporter` and decorate it with `@register("fmt")`. Implement `check_layer()` (return the reason a layer cannot be exported instead of raising) and `export()` (write `model.safetensors`, `config.json` with `quantization_config`, and copy tokenizer files).
2. Import the module in `python/mqde/export/__init__.py` to perform registration.
3. Make the authenticity check able to read the format: `real_check.load_real`, `dequant_real` and `use_real_weights` dispatch on `quant_method`.
4. Add tests covering layout, dequantized weights matching fake-quantized weights within scale storage precision, mixed precision and `check()` rejections; compare byte-for-byte against the reference library when it is available (see `python/tests/test_pack_compat.py`).
5. If C++ packing is required, update `cpp/src/quant/pack.cpp`, the headers under `cpp/include/mqde/quant/`, the pybind bindings and the wrappers in `python/mqde/core.py`.

## Adding a serving backend

1. Subclass `ServerBackend` (`python/mqde/backend/base.py`): define constructor arguments and environment variables, implement `ready()`, and populate `diagnoses` with (log substring, explanation) pairs taken from real failure logs. `vllm_server.py`, `sglang_server.py` and `llamacpp_server.py` are reference implementations.
2. Resolve the binary in the order environment variable, `PATH`, `build/<name>/bin`; when it is missing, the job stage must report `skipped` rather than fail.
3. Add an estimator profile `data/backends/<name>.yaml` with `id`, `runtime_overhead_bytes`, `allocator_factor`, `eta_bw` and `mfu`.
4. Test lifecycle, log-derived diagnostics and timing accuracy with the mock server in `python/tests/backend/fake_llama_server.py`, and add an integration test that runs when an environment variable points to a real binary.

## Adding a model architecture

Prefer a data description in `data/model_families/<family>.yaml`, defining `family`, `modality`, `match.model_type`, `config_paths`, `components` (`name`, `modality`, `prefix`, `block_regex`) and `default_keep_fp` (layers kept in high precision). Validate components and block counts with `mqde model inspect <MODEL_DIR>`.

Any architecture that Hugging Face transformers can load as a causal language model can use the `generic_decoder` and `HFCausalLM` adapters. Architectures with non-standard structure require a new Python `QuantModel` subclass.

## Adding a recipe or KV quantization method

Recipes are defined in `data/recipes/<name>.yaml` with `name`, `description`, `maturity`, `default_algorithm`, `modalities`, `weight`, `act` and `kv`. After adding one, ensure `mqde recipes` lists it and `mqde estimate -r <name>` runs.

KV quantization methods are registered in `python/mqde/quant/kv.py` with `@register("name")` and provide `quant_k` and `quant_v`; select one through `quant.kv.method`.

## Adding a hardware profile, API route or page

- Hardware: `data/hardware_db/<vendor>/<id>.yaml`
- REST routes: `cpp/src/server/server.cpp` and the typed client in `web/src/api.ts`
- Pages: `web/src/pages/<Name>.tsx`, registered in `web/src/main.tsx` with a navigation entry
