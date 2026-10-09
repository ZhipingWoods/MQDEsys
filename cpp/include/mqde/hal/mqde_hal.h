/* MQDEsys hardware plugin ABI, version 1 (pure C99, stable).
 *
 * A hardware vendor adds support for a device by building one shared library that exports
 * `mqde_hal_get_plugin_v1()`. The host never links a vendor SDK: the plugin loads its own runtime (with dlopen) and a
 * missing runtime simply means "no devices here". Layout follows the one-directory-per-vendor model of FlagTree's
 * `third_party/<vendor>/`: `cpp/plugins/<vendor>/` holds the plugin, `docs/guides/add-hardware-plugin.md` the steps.
 *
 * Rules every plugin must follow:
 *  - fill the struct below with its address stable for the life of the process; functions the plugin does not offer are
 *    NULL (only `device_probe` and `last_error` are mandatory in v1);
 *  - return 0 on success and a non-zero code on failure, and describe the failure in `last_error()`;
 *  - never write more than `max` entries into `out` and report the real count in `*n`; zero devices is a normal result
 *    (the vendor runtime is absent), not an error;
 *  - be callable from any thread after load, and have no work to do at unload.
 *
 * Changing any struct below or `MQDE_HAL_ABI_VERSION` is a breaking change (L3). `scripts/check_abi.sh` compares the
 * layout with `abi_v1.golden`. */
#ifndef MQDE_HAL_H_
#define MQDE_HAL_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MQDE_HAL_ABI_VERSION 1

#if defined(_WIN32)
#define MQDE_HAL_EXPORT __declspec(dllexport)
#else
#define MQDE_HAL_EXPORT __attribute__((visibility("default")))
#endif

typedef enum { MQDE_HAL_DEV_CPU = 0, MQDE_HAL_DEV_GPU = 1, MQDE_HAL_DEV_NPU = 2, MQDE_HAL_DEV_OTHER = 9 } mqde_hal_dev_kind;

/* One device. `kind`/`index`/strings match mqde_device_info of the built-in probe, so plugin devices are listed next to
 * the built-in ones without any change above the loader. */
typedef struct mqde_hal_device {
  int32_t kind;           /* mqde_hal_dev_kind */
  int32_t index;          /* vendor index, -1 when there is only one */
  char vendor[32];
  char name[128];
  char arch[32];          /* e.g. "sm_89", "gfx1100", "ascend910b" */
  uint64_t mem_total_bytes;
  uint64_t mem_free_bytes;
  uint32_t compute_units; /* SMs / CUs / cores when known, else 0 */
  uint32_t reserved0;
  char driver_version[64];
} mqde_hal_device;

/* "Can this device do X?" -- dtype is a lowercase name such as "fp16", "bf16", "fp8_e4m3", "int8", "int4"; op may be
 * empty to ask about the dtype alone, or a name such as "gemm". */
typedef struct mqde_hal_capability_query {
  int32_t device;
  int32_t reserved0;
  char dtype[16];
  char op[32];
} mqde_hal_capability_query;

typedef struct mqde_hal_capability_result {
  int32_t supported;      /* 1 yes, 0 no */
  int32_t native;         /* 1 when a hardware path exists, 0 when emulated */
  double peak_tflops;     /* 0 when unknown */
  char note[96];
} mqde_hal_capability_result;

/* Micro-benchmark request: kind is "bandwidth" (GB/s of a large copy) or "gemm" (TFLOPS of a dtype GEMM). */
typedef struct mqde_hal_profile_request {
  int32_t device;
  int32_t reserved0;
  char kind[16];
  char dtype[16];
  uint64_t bytes;         /* working set for "bandwidth"; 0 = plugin default */
  int32_t m, n, k;        /* shape for "gemm"; zeros = plugin default */
  int32_t iterations;     /* 0 = plugin default */
} mqde_hal_profile_request;

typedef struct mqde_hal_profile_result {
  double value;           /* GB/s or TFLOPS */
  double seconds;         /* wall time of the timed region */
  char unit[16];
} mqde_hal_profile_result;

/* Optional in v1 (may be NULL): kernel compilation and launch. Opaque on purpose, reserved for later versions. */
typedef struct mqde_hal_compile_request { const char* source; const char* options; } mqde_hal_compile_request;
typedef struct mqde_hal_compile_result { void* handle; char log[256]; } mqde_hal_compile_result;
typedef struct mqde_hal_launch_request { void* handle; int32_t device; int32_t reserved0; } mqde_hal_launch_request;

typedef struct mqde_hal_plugin_v1 {
  uint32_t abi_version;          /* must equal MQDE_HAL_ABI_VERSION */
  uint32_t struct_size;          /* sizeof(mqde_hal_plugin_v1) as the plugin was built */
  const char* name;              /* unique, lowercase, e.g. "cuda", "rocm", "template" */
  const char* vendor;
  const char* plugin_version;
  /* mandatory */
  int (*device_probe)(mqde_hal_device* out, int max, int* n);
  const char* (*last_error)(void);
  /* optional */
  int (*supports)(const mqde_hal_capability_query* q, mqde_hal_capability_result* r);
  int (*alloc)(int device, size_t bytes, void** ptr);
  int (*free)(int device, void* ptr);
  int (*profile)(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res);
  int (*compile)(const mqde_hal_compile_request* req, mqde_hal_compile_result* res);
  int (*launch)(const mqde_hal_launch_request* req);
} mqde_hal_plugin_v1;

/* The single symbol a plugin must export. */
MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void);
typedef const mqde_hal_plugin_v1* (*mqde_hal_get_plugin_v1_fn)(void);

/* ------------------------------------------------------------------ host side (implemented in libmqde_probe) */

typedef enum {
  MQDE_HAL_LOADED = 0,   /* loaded, at least one device */
  MQDE_HAL_INACTIVE = 1, /* loaded, no device (vendor runtime absent): normal */
  MQDE_HAL_REJECTED = 2, /* not usable: see `reason` */
  MQDE_HAL_DISABLED = 3  /* skipped by MQDE_HAL_PLUGINS / MQDE_HAL_DISABLE */
} mqde_hal_status;

typedef struct mqde_hal_plugin_info {
  char path[512];
  char name[64];       /* from the plugin when it got that far, else the file name */
  char vendor[32];
  char version[32];
  int32_t status;      /* mqde_hal_status */
  int32_t devices;
  char reason[256];    /* why REJECTED / DISABLED, empty otherwise */
} mqde_hal_plugin_info;

/* Scans the plugin directories and loads what is usable; safe to call again (it rescans). Directories, in order:
 * $MQDE_PLUGIN_DIR (colon-separated), $MQDE_HOME/plugins, <dir of the running executable>/plugins. A plugin that fails any
 * check is reported as REJECTED and never affects the others. `MQDE_HAL_PLUGINS=a,b` loads only those names;
 * `MQDE_HAL_DISABLE=a,b` skips those. Returns the number of plugins found (any status). */
int mqde_hal_load_all(void);
/* Loads once if nothing has been loaded yet (what mqde_probe_all() calls); later calls keep the current registry. */
void mqde_hal_ensure_loaded(void);
int mqde_hal_plugin_count(void);
int mqde_hal_plugin_get(int i, mqde_hal_plugin_info* out);
/* The loaded plugin named `name`, or NULL. The pointer stays valid until the next mqde_hal_load_all(). */
const mqde_hal_plugin_v1* mqde_hal_plugin_by_name(const char* name);
/* Devices of every loaded plugin, appended after the built-in ones by mqde_probe_all(). Returns how many were written. */
int mqde_hal_devices(mqde_hal_device* out, int max, const char** plugin_names /* optional, parallel to out */);

#ifdef __cplusplus
}
#endif
#endif /* MQDE_HAL_H_ */
