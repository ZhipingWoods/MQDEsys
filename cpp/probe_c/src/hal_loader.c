/* HAL plugin loader (T-1101): scan, load, validate, isolate failures. Pure C99 + dlopen, no vendor SDK. */
#define _GNU_SOURCE
#include "mqde/hal/mqde_hal.h"

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#endif

#define MAX_PLUGINS 32

typedef struct {
  mqde_hal_plugin_info info;
  void* handle;                  /* kept open only for accepted plugins */
  const mqde_hal_plugin_v1* api; /* NULL unless LOADED / INACTIVE */
} slot_t;

static slot_t g_slots[MAX_PLUGINS];
static int g_count = 0;
static int g_loaded = 0;
#if !defined(_WIN32)
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
#define LOCK() pthread_mutex_lock(&g_mu)
#define UNLOCK() pthread_mutex_unlock(&g_mu)
#else
#define LOCK() ((void)0)
#define UNLOCK() ((void)0)
#endif

/* Bounded copy: longer input is cut on purpose (names and reasons are for display), always NUL-terminated. */
static void copy(char* dst, size_t cap, const char* src) {
  size_t n = src ? strlen(src) : 0;
  if (n >= cap) n = cap - 1;
  if (n) memcpy(dst, src, n);
  dst[n] = '\0';
}

static void reject(slot_t* s, const char* fmt, const char* a, const char* b) {
  char msg[1200];
  s->info.status = MQDE_HAL_REJECTED;
  const int n = snprintf(msg, sizeof msg, fmt, a ? a : "", b ? b : "");
  /* A long path in the message is cut by snprintf; say so instead of ending mid-word (and the checked return
   * value is what tells -Wformat-truncation, active at -O3, that the truncation is handled). */
  if (n < 0 || (size_t)n >= sizeof msg) memcpy(msg + sizeof msg - 4, "...", 4);
  copy(s->info.reason, sizeof s->info.reason, msg);
}

/* "a,b" list membership, exact match, case-sensitive */
static int in_list(const char* list, const char* name) {
  if (!list || !*list) return 0;
  size_t n = strlen(name);
  const char* p = list;
  while (*p) {
    const char* e = strchr(p, ',');
    size_t len = e ? (size_t)(e - p) : strlen(p);
    if (len == n && strncmp(p, name, n) == 0) return 1;
    if (!e) break;
    p = e + 1;
  }
  return 0;
}

static int valid_name(const char* n) {
  if (!n || !*n || strlen(n) >= sizeof(((mqde_hal_plugin_info*)0)->name)) return 0;
  for (const char* p = n; *p; ++p)
    if (!(islower((unsigned char)*p) || isdigit((unsigned char)*p) || *p == '_' || *p == '-')) return 0;
  return 1;
}

static void unload_all(void) {
  for (int i = 0; i < g_count; ++i)
    if (g_slots[i].handle) dlclose(g_slots[i].handle);
  memset(g_slots, 0, sizeof g_slots);
  g_count = 0;
}

/* Checks `api` and probes it once. Fills the slot status. */
static void validate(slot_t* s, const mqde_hal_plugin_v1* api, const char* only, const char* off) {
  if (api->abi_version != MQDE_HAL_ABI_VERSION) {
    char v[16];
    snprintf(v, sizeof v, "%u", (unsigned)api->abi_version);
    return reject(s, "ABI version %s, this host supports 1", v, NULL);
  }
  if (api->struct_size < sizeof(mqde_hal_plugin_v1)) {
    char v[16];
    snprintf(v, sizeof v, "%u", (unsigned)api->struct_size);
    return reject(s, "struct_size %s is smaller than the v1 struct (built against a different header)", v,
                  NULL);
  }
  if (!valid_name(api->name))
    return reject(
        s, "missing or invalid plugin name (lowercase letters, digits, '_' and '-', at most 63 characters)",
        NULL, NULL);
  copy(s->info.name, sizeof s->info.name, api->name);
  copy(s->info.vendor, sizeof s->info.vendor, api->vendor);
  copy(s->info.version, sizeof s->info.version, api->plugin_version);
  if (!api->device_probe) return reject(s, "mandatory function '%s' is NULL", "device_probe", NULL);
  if (!api->last_error) return reject(s, "mandatory function '%s' is NULL", "last_error", NULL);
  for (int i = 0; i < g_count; ++i)
    if (&g_slots[i] != s && g_slots[i].api && strcmp(g_slots[i].info.name, s->info.name) == 0)
      return reject(s, "duplicate plugin name, already provided by %s", g_slots[i].info.path, NULL);
  if (in_list(off, s->info.name)) {
    s->info.status = MQDE_HAL_DISABLED;
    copy(s->info.reason, sizeof s->info.reason, "listed in MQDE_HAL_DISABLE");
    return;
  }
  if (only && *only && !in_list(only, s->info.name)) {
    s->info.status = MQDE_HAL_DISABLED;
    copy(s->info.reason, sizeof s->info.reason, "not listed in MQDE_HAL_PLUGINS");
    return;
  }
  mqde_hal_device tmp[64];
  int n = -12345;
  int rc = api->device_probe(tmp, 64, &n);
  if (rc != 0) {
    const char* why = api->last_error();
    return reject(s, "device_probe failed: %s", why && *why ? why : "(no message)", NULL);
  }
  if (n < 0 || n > 64) {
    char v[16];
    snprintf(v, sizeof v, "%d", n);
    return reject(s, "device_probe reported %s devices for a capacity of 64", v, NULL);
  }
  s->api = api;
  s->info.devices = n;
  s->info.status = n > 0 ? MQDE_HAL_LOADED : MQDE_HAL_INACTIVE;
}

/* The same file reached through two directory entries (MQDE_PLUGIN_DIR and <exe dir>/plugins often name one
 * directory, and a symlink can alias a file) must be loaded once: a second copy would be reported as a
 * duplicate and look like a failure. */
static int already_seen(const char* path) {
  char real[PATH_MAX];
  if (!realpath(path, real)) return 0;
  for (int i = 0; i < g_count; ++i) {
    char other[PATH_MAX];
    if (realpath(g_slots[i].info.path, other) && strcmp(real, other) == 0) return 1;
  }
  return 0;
}

static void load_file(const char* path, const char* only, const char* off) {
  if (g_count >= MAX_PLUGINS || already_seen(path)) return;
  slot_t* s = &g_slots[g_count];
  memset(s, 0, sizeof *s);
  copy(s->info.path, sizeof s->info.path, path);
  const char* base = strrchr(path, '/');
  copy(s->info.name, sizeof s->info.name, base ? base + 1 : path);
  ++g_count;
  void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    const char* e = dlerror();
    return reject(s, "cannot load: %s", e, NULL);
  }
  mqde_hal_get_plugin_v1_fn fn = (mqde_hal_get_plugin_v1_fn)dlsym(h, "mqde_hal_get_plugin_v1");
  if (!fn) {
    dlclose(h);
    return reject(s, "symbol '%s' not found", "mqde_hal_get_plugin_v1", NULL);
  }
  const mqde_hal_plugin_v1* api = fn();
  if (!api) {
    dlclose(h);
    return reject(s, "%s returned NULL", "mqde_hal_get_plugin_v1", NULL);
  }
  validate(s, api, only, off);
  if (s->api)
    s->handle = h;
  else
    dlclose(h);
}

static int cmp_str(const void* a, const void* b) { return strcmp(*(char* const*)a, *(char* const*)b); }

static void scan_dir(const char* dir, const char* only, const char* off) {
  DIR* d = opendir(dir);
  if (!d) return;
  char* names[256];
  int n = 0;
  struct dirent* e;
  while ((e = readdir(d)) && n < 256) {
    const char* nm = e->d_name;
    size_t len = strlen(nm);
    if (nm[0] == '.') continue;
    int is_lib =
        (len > 3 && strcmp(nm + len - 3, ".so") == 0) || (len > 6 && strcmp(nm + len - 6, ".dylib") == 0);
    if (is_lib) names[n++] = strdup(nm);
  }
  closedir(d);
  qsort(names, (size_t)n, sizeof(char*),
        cmp_str); /* deterministic order: the first of two duplicates always wins */
  for (int i = 0; i < n; ++i) {
    char path[4200];
    snprintf(path, sizeof path, "%.4090s/%.100s", dir, names[i]);
    free(names[i]);
    load_file(path, only, off);
  }
}

static void exe_dir(char* out, size_t cap) {
  out[0] = '\0';
#if defined(__linux__)
  ssize_t n = readlink("/proc/self/exe", out, cap - 1);
  if (n > 0) out[n] = '\0';
#elif defined(__APPLE__)
  uint32_t sz = (uint32_t)cap;
  if (_NSGetExecutablePath(out, &sz) != 0) out[0] = '\0';
#endif
  char* slash = strrchr(out, '/');
  if (slash)
    *slash = '\0';
  else
    out[0] = '\0';
}

int mqde_hal_load_all(void) {
#if defined(_WIN32)
  return 0;
#else
  LOCK();
  unload_all();
  const char* only = getenv("MQDE_HAL_PLUGINS");
  const char* off = getenv("MQDE_HAL_DISABLE");
  char dirs[3][4096];
  int nd = 0;
  const char* env = getenv("MQDE_PLUGIN_DIR");
  if (env && *env) {
    char* copy_env = strdup(env);
    for (char* tok = strtok(copy_env, ":"); tok && nd < 3; tok = strtok(NULL, ":"))
      copy(dirs[nd++], sizeof dirs[0], tok);
    free(copy_env);
  }
  const char* home = getenv("MQDE_HOME");
  if (home && *home && nd < 3) snprintf(dirs[nd++], sizeof dirs[0], "%.4000s/plugins", home);
  if (nd < 3) {
    char ex[2048];
    exe_dir(ex, sizeof ex);
    if (ex[0]) snprintf(dirs[nd++], sizeof dirs[0], "%.2040s/plugins", ex);
  }
  for (int i = 0; i < nd; ++i) scan_dir(dirs[i], only, off);
  g_loaded = 1;
  int n = g_count;
  UNLOCK();
  return n;
#endif
}

void mqde_hal_ensure_loaded(void) {
#if !defined(_WIN32)
  LOCK();
  int need = !g_loaded;
  UNLOCK();
  if (need) mqde_hal_load_all();
#endif
}

int mqde_hal_plugin_count(void) {
  LOCK();
  int n = g_count;
  UNLOCK();
  return n;
}

int mqde_hal_plugin_get(int i, mqde_hal_plugin_info* out) {
  LOCK();
  int ok = out && i >= 0 && i < g_count;
  if (ok) *out = g_slots[i].info;
  UNLOCK();
  return ok ? 0 : -1;
}

const mqde_hal_plugin_v1* mqde_hal_plugin_by_name(const char* name) {
  const mqde_hal_plugin_v1* r = NULL;
  LOCK();
  for (int i = 0; name && i < g_count; ++i)
    if (g_slots[i].api && strcmp(g_slots[i].info.name, name) == 0) r = g_slots[i].api;
  UNLOCK();
  return r;
}

int mqde_hal_devices(mqde_hal_device* out, int max, const char** plugin_names) {
  int w = 0;
  LOCK();
  for (int i = 0; i < g_count && w < max; ++i) {
    const slot_t* s = &g_slots[i];
    if (!s->api || s->info.status != MQDE_HAL_LOADED) continue;
    mqde_hal_device tmp[64];
    int n = -1;
    if (s->api->device_probe(tmp, 64, &n) != 0 || n < 0 || n > 64)
      continue; /* a plugin that breaks later is skipped */
    for (int k = 0; k < n && w < max; ++k) {
      out[w] = tmp[k];
      out[w].vendor[sizeof out[w].vendor - 1] = '\0';
      out[w].name[sizeof out[w].name - 1] = '\0';
      out[w].arch[sizeof out[w].arch - 1] = '\0';
      out[w].driver_version[sizeof out[w].driver_version - 1] = '\0';
      if (plugin_names) plugin_names[w] = s->info.name;
      ++w;
    }
  }
  UNLOCK();
  return w;
}
