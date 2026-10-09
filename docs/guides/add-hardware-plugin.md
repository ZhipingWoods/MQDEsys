# Adding a hardware plugin

A hardware vendor adds support for a device by writing **one shared library** that exports `mqde_hal_get_plugin_v1()`.
Nothing above the loader changes: the plugin's devices appear in `mqde probe`, in `/api/v1/hardware` and on the web
Hardware page by themselves. The structure follows the one-directory-per-vendor model of FlagTree's `third_party/<vendor>/`
(each backend is a self-contained directory with a small interface; the host discovers it by scanning and a broken backend
never takes the others down).

The ABI is `cpp/include/mqde/hal/mqde_hal.h`. It is plain C99, so a plugin can be written in C, C++, Rust or anything else
that can export a C symbol, and it never links a vendor SDK at build time.

## 1. Start from the template

```bash
cp -r cpp/plugins/template cpp/plugins/<vendor>
```

Edit `template.c` and replace the four marked parts:

| Part | What to do | If your runtime is missing |
|---|---|---|
| 1. device discovery (`device_probe`) | `dlopen` the vendor runtime, list devices, fill `mqde_hal_device` | return 0 devices and success: that is "inactive", not an error |
| 2. capabilities (`supports`) | answer per dtype / op from the hardware's real table, say whether the path is `native` or emulated | return an error with `last_error()` |
| 3. memory (`alloc` / `free`) | call the vendor allocator | optional: leave the pointers `NULL` |
| 4. micro-benchmarks (`profile`) | time a large copy (`"bandwidth"`, GB/s) and a GEMM (`"gemm"`, TFLOPS) on the device | optional: leave `NULL` |

`compile` and `launch` are reserved for later versions and may be `NULL` in v1. Only `device_probe` and `last_error` are
mandatory.

Rules the loader checks (each failure is reported with its reason, see section 3):

- `abi_version` equals `MQDE_HAL_ABI_VERSION` (1) and `struct_size` is at least `sizeof(mqde_hal_plugin_v1)`;
- `name` is unique, lowercase letters, digits, `_` and `-`, at most 63 characters;
- `device_probe` and `last_error` are not `NULL`;
- `device_probe` returns 0, reports `0 <= n <= max`, and never writes more than `max` entries.

## 2. Build it

Register it in `cpp/CMakeLists.txt` next to the template:

```cmake
mqde_add_plugin(<vendor> plugins/<vendor>/<vendor>.c)
```

The module is written to `<build>/bin/plugins/<vendor>.so`, which is where a running `mqde` looks.

## 3. See it, and see why it is not seen

The loader scans, in order: `$MQDE_PLUGIN_DIR` (colon separated), `$MQDE_HOME/plugins`, and `<directory of the running
executable>/plugins`. Files ending in `.so` or `.dylib` are loaded in name order.

```bash
MQDE_TEMPLATE_DEVICES=2 build/dev/bin/mqde probe      # the simulated devices of the template plugin
MQDE_HAL_DISABLE=<vendor>   mqde probe                # skip a plugin
MQDE_HAL_PLUGINS=<vendor>   mqde probe                # load only the named plugins
```

A plugin ends in one of four states: `LOADED` (has devices), `INACTIVE` (loaded, no device found, normal), `REJECTED`
(with the reason, such as `ABI version 2, this host supports 1`, `symbol 'mqde_hal_get_plugin_v1' not found`,
`device_probe failed: <your last_error()>`, `duplicate plugin name`), or `DISABLED` (by one of the two variables above).
Read them from C with `mqde_hal_plugin_count()` and `mqde_hal_plugin_get()`.

A device that the built-in probe already found (same vendor, name and index; for example a CUDA plugin next to NVML) is not
listed twice.

## 4. Test it

Add a test in the style of `cpp/tests/unit/hal_test.cpp`: load your library with `MQDE_PLUGIN_DIR`, check the status, the
devices and a capability answer. Then run the gates:

```bash
ctest --preset dev -R "hal_|probe"
scripts/check_abi.sh            # the ABI layout against cpp/include/mqde/hal/abi_v1.golden
```

## 5. Limits you should know

- A plugin runs inside the host process. The loader rejects plugins that fail its checks and survives error returns, but it
  cannot survive a plugin that crashes or hangs inside `device_probe`. Keep probing short and defensive.
- `check_abi.sh` compares the struct layout of the 64-bit Linux / macOS ABI. It does not inspect a built plugin's symbols.
- Changing the ABI is a breaking change (L3): raise `MQDE_HAL_ABI_VERSION`, regenerate the golden file with
  `scripts/check_abi.sh --update` after approval, and keep loading version 1 plugins for a transition period.
- The hardware profile database (`data/hardware_db/<vendor>/<id>.yaml`: memory, bandwidth, peak TFLOPS per dtype, native
  formats) is separate from the plugin and is what the estimator reads. A plugin's `profile` results are meant to feed it;
  that connection is not built yet.

## 6. The CUDA plugin as a worked example

`cpp/plugins/cuda/cuda.c` is the first real plugin. It loads `libcuda`, `libcudart` and `libcublas` with `dlopen` (so a machine
without NVIDIA software lists no device and no error), lists devices through the driver API, and implements `profile`:

| Request | How it is measured | Measured on an RTX 4090 (GPU 5, cuBLAS 13) |
|---|---|---|
| `bandwidth` | 20 device-to-device copies of a 1 GiB buffer after 3 warm-up copies, CUDA events, read plus write counted | 886 to 919 GB/s, 88 to 91% of the 1008 GB/s datasheet value (the spec asks for 70 to 95%) |
| `gemm` fp32 / fp16 / bf16 / int8 | 4096 cubed `cublasGemmEx`, 20 timed runs after 3 warm-up runs | 62.2 / 170.6 / 155.6 / 169.9 TFLOPS against datasheet 82.6 / 165.2 / 165.2 / 660.6 |

Read these honestly: FP16 is 3% above the datasheet figure (boost clocks), FP32 is 75% of it, and **INT8 is 26%: that is a
property of this `cublasGemmEx` call, not of the hardware**, so treat the INT8 number as a lower bound. FP8 and a proper INT8
number need `cublasLt`, which is not implemented. The built-in NVML probe ignores `CUDA_VISIBLE_DEVICES`
and lists every GPU, while the plugin's ordinals follow the mask. Merging compares vendor, name and index, so a masked run only
avoids duplicates when the plugin's ordinals happen to equal NVML indices that are the same model: on the 4090 host with
`CUDA_VISIBLE_DEVICES=4,5` the plugin's devices 0 and 1 matched NVML entries 0 and 1 by name and index, so 8 rows were shown (not
10), but those are not the same physical GPUs. Treat the merge as a convenience for the unmasked case, not as an identity check.

## 7. Plugins for hardware you do not have, and how to check them

`cpp/plugins/rocm` was written without an AMD GPU. What made that workable, and what it cannot prove:

1. **Verify every constant and signature against a real header, not memory.** FlagTree vendors the HIP header in
   `3rdpart/FlagTree/third_party/amd/backend/include/hip`. Compiling a four-line program that prints the enum values showed that two
   attribute ids written from memory were wrong (CU count is 63, not 16; minor version is 61, not 24). A call whose signature cannot
   be checked this way is not written: the ROCm plugin's GEMM returns a clear "not implemented" error instead, because a wrong
   signature crashes the host process.
2. **Test the plugin's own code against a stand-in runtime.** `cpp/tests/fakes/fake_hip.c` is a small library that exports the same
   symbols; the tests point the plugin at it with `MQDE_ROCM_LIB_DIR`. It uses the real attribute ids and fails on a wrong one, and
   returns different numbers for different ids, so a mistaken id changes the result and a test notices. Scenarios that change the
   stand-in's behaviour run in a child process, because a plugin keeps its resolved runtime for the life of the process.
3. **Mark it as unverified in the plugin version string and in TASKS.md** (`0.1.0-untested`) until it has run on the hardware.

What this does not prove: that the real library returns what the code expects, the capability tables, and anything numeric. Those
are listed in `docs/TASKS.md` per plugin as "needs a machine with ...".

## 8. What is verified for each shipped plugin

| Plugin | Run on real hardware | How the numbers were obtained |
|---|---|---|
| `template` | simulated | none, by design |
| `cpu` | Mac (Apple M5) and the 4090 host (AMD EPYC 7402) | memcpy and an fp32 loop, so the GEMM figure is a floor, not a BLAS number |
| `cuda` | RTX 4090 (GPUs within 4 to 7) | cudart events and cuBLAS; see section 6 |
| `metal` | Apple M5 | a standalone Objective-C program was run first to confirm the API calls and typical numbers, then the plugin; its tests run on the real GPU |
| `rocm` | **no** | stand-in HIP library; attribute ids checked against the vendored HIP header; GEMM not implemented |
| Ascend | **not started** | no CANN header or library is available to check calls against |

Across the four GPUs 4 to 7 of the development host (one GEMM run and three bandwidth repeats each): bandwidth is 886 to 919 GB/s on
GPUs 4, 5 and 7, and **764 to 796 GB/s on GPU 6**, which also reports less memory (22554 MiB against 24080 MiB). Compute is the same
on all four (fp16 168 to 174 TFLOPS). So "an RTX 4090" is not one performance number: a pool of cards can differ, which is a reason
to measure the card a job will actually run on.
