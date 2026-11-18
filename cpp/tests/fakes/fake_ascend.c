/* Stand-in CANN runtime for the Ascend plugin tests. It implements only the ACL symbols the plugin
 * resolves, with the same signatures llama.cpp's ggml-cann backend uses. No hardware required. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail_if(const char* fn) {
  const char* f = getenv("FAKE_ASCEND_FAIL");
  return f && strcmp(f, fn) == 0 ? 1 : 0;
}

int aclInit(const char* config) {
  (void)config;
  return fail_if("aclInit") ? 1 : 0;
}

int aclrtGetDeviceCount(uint32_t* n) {
  if (fail_if("aclrtGetDeviceCount")) return 1;
  const char* s = getenv("FAKE_ASCEND_DEVICES");
  *n = s ? (uint32_t)strtoul(s, NULL, 10) : 2;
  return 0;
}

int aclrtSetDevice(int device) {
  (void)device;
  return fail_if("aclrtSetDevice") ? 1 : 0;
}

int aclrtResetDevice(int device) {
  (void)device;
  return fail_if("aclrtResetDevice") ? 1 : 0;
}

const char* aclrtGetSocName(void) {
  const char* s = getenv("FAKE_ASCEND_SOC");
  return s && *s ? s : "Ascend910B";
}

int aclrtGetMemInfo(int type, size_t* free_b, size_t* total_b) {
  (void)type;
  if (fail_if("aclrtGetMemInfo")) return 1;
  const char* s = getenv("FAKE_ASCEND_MEM_MB");
  const size_t mb = s ? (size_t)strtoull(s, NULL, 10) : 65536;
  *total_b = mb << 20;
  *free_b = (*total_b) * 3 / 4;
  return 0;
}

int aclrtMalloc(void** ptr, size_t bytes, int policy) {
  (void)policy;
  if (fail_if("aclrtMalloc")) return 1;
  *ptr = malloc(bytes ? bytes : 1);
  return *ptr ? 0 : 1;
}

int aclrtFree(void* ptr) {
  if (fail_if("aclrtFree")) return 1;
  free(ptr);
  return 0;
}

int aclrtMemcpy(void* dst, size_t dst_bytes, const void* src, size_t bytes, int kind) {
  (void)kind;
  if (fail_if("aclrtMemcpy")) return 1;
  if (bytes > dst_bytes) return 1;
  memmove(dst, src, bytes);
  return 0;
}

int aclrtSynchronizeDevice(void) { return fail_if("aclrtSynchronizeDevice") ? 1 : 0; }
