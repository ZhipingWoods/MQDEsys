/* MQDEsys hardware probe - pure C99 ABI.
 * Vendor libraries (NVML, ROCm-SMI, ...) are loaded with dlopen at runtime: a missing library
 * means "no such device", never a build or runtime failure. */
#ifndef MQDE_PROBE_H_
#define MQDE_PROBE_H_

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
/* "cpu" | "gpu" | "npu" | "other": the one spelling every consumer (CLI, server, Python, web UI) uses for a device kind. */
const char* mqde_dev_kind_name(mqde_dev_kind kind);

#ifdef __cplusplus
}
#endif
#endif /* MQDE_PROBE_H_ */
