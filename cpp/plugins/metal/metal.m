/* Metal HAL plugin (T-1103), macOS only. Devices from MTLCopyAllDevices, bandwidth from a blit copy, GEMM
 * from MetalPerformanceShaders. Everything here was run on real Apple silicon while writing it (an M5: about
 * 120-126 GB/s copy, about 2.96 TFLOPS fp32 at 2048 cubed).
 *
 * Apple GPUs use unified memory: `mem_total_bytes` is the recommended GPU working-set size, not a separate
 * VRAM size, and `mem_free_bytes` is that minus what Metal says is already allocated. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <string.h>

#include "mqde/hal/mqde_hal.h"

static char g_err[256] = "";
static int set_err(const char* m) {
  snprintf(g_err, sizeof g_err, "%s", m);
  return -1;
}
static const char* last_error(void) { return g_err; }

static double now_s(void) {
  static mach_timebase_info_data_t tb;
  if (!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

static NSArray<id<MTLDevice>>* all_devices(void) {
  NSArray<id<MTLDevice>>* d = MTLCopyAllDevices();
  return d ? d : @[];
}

static int device_probe(mqde_hal_device* out, int max, int* n) {
  if (!out || !n || max < 0) return set_err("invalid arguments");
  *n = 0;
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devs = all_devices();
    for (NSUInteger i = 0; i < [devs count] && *n < max; ++i) {
      id<MTLDevice> d = devs[i];
      mqde_hal_device* o = &out[*n];
      memset(o, 0, sizeof *o);
      o->kind = MQDE_HAL_DEV_GPU;
      o->index = (int)i;
      snprintf(o->vendor, sizeof o->vendor, "apple");
      snprintf(o->name, sizeof o->name, "%s", [[d name] UTF8String]);
      snprintf(o->arch, sizeof o->arch, "%s", [d hasUnifiedMemory] ? "apple-gpu" : "metal-gpu");
      const uint64_t ws = (uint64_t)[d recommendedMaxWorkingSetSize];
      const uint64_t used = (uint64_t)[d currentAllocatedSize];
      o->mem_total_bytes = ws;
      o->mem_free_bytes = ws > used ? ws - used : 0;
      o->compute_units = 0; /* Metal does not expose the core count */
      snprintf(o->driver_version, sizeof o->driver_version, "metal");
      ++*n;
    }
  }
  return 0;
}

static int supports(const mqde_hal_capability_query* q, mqde_hal_capability_result* r) {
  if (!q || !r) return set_err("null argument");
  memset(r, 0, sizeof *r);
  const char* t = q->dtype;
  int sup = 0, nat = 0;
  const char* note = "dtype not known to the Metal plugin";
  if (!strcmp(t, "fp32"))
    sup = 1, nat = 1, note = "MPS matrix multiply";
  else if (!strcmp(t, "fp16"))
    sup = 1, nat = 1, note = "MPS matrix multiply";
  else if (!strcmp(t, "bf16"))
    sup = 1, nat = 0, note = "supported by recent Metal, not benchmarked by this plugin";
  else if (!strcmp(t, "int8"))
    sup = 1, nat = 0, note = "no integer matrix path used here";
  else if (!strcmp(t, "int4") || !strcmp(t, "fp8_e4m3") || !strcmp(t, "fp8_e5m2"))
    note = "no hardware path";
  r->supported = sup;
  r->native = nat;
  snprintf(r->note, sizeof r->note, "%s", note);
  return 0;
}

static int bench_bandwidth(id<MTLDevice> dev, const mqde_hal_profile_request* req,
                           mqde_hal_profile_result* res) {
  size_t bytes = req->bytes ? (size_t)req->bytes : (size_t)256 << 20;
  if (bytes < ((size_t)1 << 20)) bytes = (size_t)1 << 20;
  if (bytes > [dev recommendedMaxWorkingSetSize] / 4) bytes = (size_t)[dev recommendedMaxWorkingSetSize] / 4;
  id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLBuffer> a = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  if (!q || !a || !b) return set_err("cannot allocate the Metal buffers for the bandwidth test");
  const int iters = req->iterations > 0 ? req->iterations : 10;
  for (int w = 0; w < 2; ++w) { /* warm-up, not timed */
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
    [e copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:bytes];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }
  const double t0 = now_s();
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
  for (int i = 0; i < iters; ++i)
    [e copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:bytes];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if ([cb status] != MTLCommandBufferStatusCompleted) return set_err("the Metal copy did not complete");
  res->seconds = now_s() - t0;
  res->value = res->seconds > 0 ? 2.0 * (double)bytes * iters / res->seconds / 1e9 : 0;
  snprintf(res->unit, sizeof res->unit, "GB/s");
  return 0;
}

static int bench_gemm(id<MTLDevice> dev, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  MPSDataType dt;
  size_t esz;
  if (!strcmp(req->dtype, "fp32"))
    dt = MPSDataTypeFloat32, esz = 4;
  else if (!strcmp(req->dtype, "fp16"))
    dt = MPSDataTypeFloat16, esz = 2;
  else
    return set_err("gemm dtype not supported by the Metal plugin (fp32, fp16)");
  const NSUInteger m = req->m > 0 ? (NSUInteger)req->m : 2048, n = req->n > 0 ? (NSUInteger)req->n : 2048,
                   k = req->k > 0 ? (NSUInteger)req->k : 2048;
  const int iters = req->iterations > 0 ? req->iterations : 10;
  id<MTLCommandQueue> q = [dev newCommandQueue];
  MPSMatrixDescriptor* da = [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                                  columns:k
                                                                 rowBytes:k * esz
                                                                 dataType:dt];
  MPSMatrixDescriptor* db = [MPSMatrixDescriptor matrixDescriptorWithRows:k
                                                                  columns:n
                                                                 rowBytes:n * esz
                                                                 dataType:dt];
  MPSMatrixDescriptor* dc = [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                                  columns:n
                                                                 rowBytes:n * esz
                                                                 dataType:dt];
  id<MTLBuffer> A = [dev newBufferWithLength:m * k * esz options:MTLResourceStorageModeShared];
  id<MTLBuffer> B = [dev newBufferWithLength:k * n * esz options:MTLResourceStorageModeShared];
  id<MTLBuffer> C = [dev newBufferWithLength:m * n * esz options:MTLResourceStorageModeShared];
  if (!q || !A || !B || !C) return set_err("cannot allocate the Metal buffers for the gemm test");
  MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:A descriptor:da];
  MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:B descriptor:db];
  MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:C descriptor:dc];
  MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                                  transposeLeft:NO
                                                                 transposeRight:NO
                                                                     resultRows:m
                                                                  resultColumns:n
                                                                interiorColumns:k
                                                                          alpha:1
                                                                           beta:0];
  for (int w = 0; w < 2; ++w) { /* warm-up, not timed */
    id<MTLCommandBuffer> cb = [q commandBuffer];
    [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    [cb commit];
    [cb waitUntilCompleted];
  }
  const double t0 = now_s();
  id<MTLCommandBuffer> cb = [q commandBuffer];
  for (int i = 0; i < iters; ++i) [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
  [cb commit];
  [cb waitUntilCompleted];
  if ([cb status] != MTLCommandBufferStatusCompleted)
    return set_err("the Metal matrix multiply did not complete");
  res->seconds = now_s() - t0;
  res->value = res->seconds > 0 ? 2.0 * (double)m * n * k * iters / res->seconds / 1e12 : 0;
  snprintf(res->unit, sizeof res->unit, "TFLOPS");
  return 0;
}

static int profile(int device, const mqde_hal_profile_request* req, mqde_hal_profile_result* res) {
  if (!req || !res) return set_err("null argument");
  memset(res, 0, sizeof *res);
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devs = all_devices();
    if (device < 0 || (NSUInteger)device >= [devs count]) return set_err("no such device");
    if (!strcmp(req->kind, "bandwidth")) return bench_bandwidth(devs[(NSUInteger)device], req, res);
    if (!strcmp(req->kind, "gemm")) return bench_gemm(devs[(NSUInteger)device], req, res);
  }
  return set_err("unknown profile kind (expected 'bandwidth' or 'gemm')");
}

static const mqde_hal_plugin_v1 kPlugin = {
    .abi_version = MQDE_HAL_ABI_VERSION,
    .struct_size = sizeof(mqde_hal_plugin_v1),
    .name = "metal",
    .vendor = "apple",
    .plugin_version = "0.1.0",
    .device_probe = device_probe,
    .last_error = last_error,
    .supports = supports,
    .profile = profile,
};

MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void) { return &kPlugin; }
