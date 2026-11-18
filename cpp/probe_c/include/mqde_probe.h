/* MQDEsys hardware probe - pure C99 ABI.
 * Vendor libraries (NVML, ROCm-SMI, ...) are loaded with dlopen at runtime: a missing library
 * means "no such device", never a build or runtime failure. */
#ifndef MQDE_PROBE_H_
#define MQDE_PROBE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { MQDE_DEV_CPU = 0, MQDE_DEV_GPU = 1, MQDE_DEV_NPU = 2, MQDE_DEV_OTHER = 9 } mqde_dev_kind;

typedef struct {
  mqde_dev_kind kind;
  int index; /* vendor index (e.g. NVML index), -1 for CPU */
  char vendor[32];
  char name[128];
  char arch[32]; /* "sm_89", "x86_64", "arm64" */
  uint64_t mem_total_bytes;
  uint64_t mem_free_bytes;
  uint32_t compute_units; /* logical CPUs / SMs when known */
  char driver_version[64];
} mqde_device_info;

/* Fills up to max_devices entries; returns 0 on success (even when only the CPU is found). */
int mqde_probe_all(mqde_device_info* out, int max_devices, int* n_found);
const char* mqde_probe_last_error(void);
/* "cpu" | "gpu" | "npu" | "other": the one spelling every consumer (CLI, server, Python, web UI) uses for a
 * device kind. */
const char* mqde_dev_kind_name(mqde_dev_kind kind);

/* FlagTree-style target descriptor, e.g. "cuda:sm_89:32", "rocm:gfx1100:64", "ascend:910b:32".
 * A 2-part form ("ascend:910b") is accepted and gets the backend's conventional warp size. */
typedef struct mqde_target {
  char backend[16]; /* cuda | rocm | metal | ascend | maca | musa | sycl | cpu | other */
  char arch[32];
  int32_t warp_size;
} mqde_target;

/* Parses a target string. Returns 0 on success, non-zero on a malformed target (last_error has the reason). */
int mqde_target_parse(const char* text, mqde_target* out);
/* Derives the target for a probed device (vendor + arch -> backend, conventional warp size). */
int mqde_target_for_device(const mqde_device_info* dev, mqde_target* out);
/* Writes "backend:arch:warp" into buf (always NUL-terminated). Returns 0, or -1 when the buffer is too small. */
int mqde_target_format(const mqde_target* t, char* buf, size_t cap);

#ifdef __cplusplus
}
#endif
#endif /* MQDE_PROBE_H_ */
