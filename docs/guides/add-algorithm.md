# Adding a quantization algorithm

A weight-quantization algorithm is one function registered in `python/mqde/quant/algorithms.py`'s
`ALGO_REGISTRY`, plus one entry in `data/algorithms.yaml` so the server will accept `quant.algorithm` naming
it and the web UI can list it. `scripts/check_algo_registry_sync.py` (T-1302) fails the build if the two
drift apart — run it after step 2 below.

## 0. What an algorithm actually receives and returns

Every registered function has this signature (see `AlgoFn` in `algorithms.py`):

```python
def my_algo(w: torch.Tensor, x: torch.Tensor | None, ws: QuantSpec, as_: QuantSpec,
            p: dict[str, Any]) -> AlgoResult:
    ...
```

- `w`: the layer's FP32 weight, `[out, in]`.
- `x`: a sample of real input activations captured on calibration data, `[rows, in]`, or `None` if the
  algorithm needs no calibration (e.g. `rtn`).
- `ws`, `as_`: the recipe's weight / activation `QuantSpec` (format, bits, granularity, group size).
- `p`: the algorithm's own parameters, read from `data/algorithms.yaml`'s `params:` for this algorithm.

Return an `AlgoResult`:

- `w`: the (possibly transformed) weight to hand to the recipe's quantizer.
- `in_scale`: an optional per-input-channel divisor. If set, the layer computes `Q(X/s) @ Q(W*s)^T`,
  mathematically identical in FP before quantization, which just redistributes quantization difficulty
  between activations and weights (this is what AWQ and SmoothQuant do; `rtn` leaves it `None`).
- `final=True` plus `scale`/`zero`: only for an algorithm that already produced the quantized weight itself
  (GPTQ does this — it needs the quantizer *inside* its column loop, not after). Everything else leaves
  `final=False` and the shared `fake_quant` path in `materialize()` quantizes `w` afterward.

The simplest real one, `rtn` (round-to-nearest, needs no calibration):

```python
@register("rtn")
def rtn(w: torch.Tensor, x: torch.Tensor | None, ws: QuantSpec, as_: QuantSpec, p: dict[str, Any]) -> AlgoResult:
    return AlgoResult(w=w)
```

For a transform that scales before quantizing, read `smoothquant` or `awq` in the same file — both are
short (15-25 lines) and real, not abbreviated for this guide.

## 1. Write and register the function

```python
@register("my_algo")
def my_algo(w, x, ws, as_, p):
    ...
    return AlgoResult(w=..., info={"whatever you want visible in the per-layer trace table"})
```

`register(name)` just does `ALGO_REGISTRY[name] = fn`; the module decorator runs at import time, so
`my_algo` must live inside `python/mqde/quant/algorithms.py` itself (or be imported somewhere that always
runs before `ALGO_REGISTRY` is read — the existing file has no multi-file registration, unlike
`python/mqde/export/`'s exporter registry, which needs `export/__init__.py` to import each exporter module
for its side effect; don't copy that pattern here unless you actually split algorithms across files).

If your algorithm is a **quantizer** (produces the final quantized weight, like `gptq`) rather than a
**transform** (rescales before an existing quantizer, like `awq`/`smoothquant`), also add it to the
`quantizers = [...]` tuple inside `resolve()` so `"my_algo+something"` composition rules apply correctly —
read `resolve()`'s existing logic before changing it; at most one transform and one quantizer can combine.

## 2. Add the `data/algorithms.yaml` entry

```yaml
  - name: my_algo
    label: My Algo
    kind: quantizer   # or transform; combos (e.g. "awq+gptq") are a separate kind, see below
    description: One sentence a user picking from a dropdown would understand.
    needs_calibration: true   # false only if, like rtn, it never reads calibration activations
    params: {some_param: 1.0}   # becomes `p` in your function; omit if you have none
```

`needs_calibration: true` is enforced server-side: `planner::validate_jobspec` (cpp/src/planner/validate.cpp)
rejects a job that picks your algorithm with `calibration.n_samples <= 0`.

Composite names like `awq+gptq` are `kind: combo` entries; they are **not** separate `@register()` keys —
`resolve()` splits the name on `+` and looks up each part in `ALGO_REGISTRY`. You only need a combo entry
if you want it to show up as its own labeled choice in the UI; any `"transform+quantizer"` string already
works without one.

## 3. Check and test

```bash
python scripts/check_algo_registry_sync.py   # catches a name present in only one of the two places
pytest python/tests/quant/test_<your_algorithm>.py -q
```

Write the test the way `test_rtn.py` / existing algorithm tests do: a tiny random layer, compare your
algorithm's quantized output error against a plain RTN baseline (an algorithm that never beats RTN on
*some* input is a sign something is backwards, though a single adversarial case proves little by itself).

## 4. What this guide does not cover

- **GPTQModel / llm-compressor external adapters** (T-0609): wrapping a third-party package as
  `external:<name>` instead of a function in this file is a different, larger task — it adds a new runtime
  dependency, which needs an ADR (`docs/adr/0004-external-algorithm-adapters.md`, Proposed, not yet
  approved) before any code lands, per this project's L3 rule for new dependencies.
- **C++ registration.** `mqde::Registry<Base>` (`cpp/include/mqde/common/registry.h`) exists and is
  documented for "algorithms, exporters, backends," but nothing in this codebase actually registers a
  quantization algorithm through it today — the real registry quantization algorithms go through is the
  Python one described above. Don't assume the C++ template has a production consumer; check before relying
  on it.
