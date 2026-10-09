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
