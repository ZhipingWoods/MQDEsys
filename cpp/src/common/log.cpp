#include "mqde/common/log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

namespace mqde::log {
namespace {

std::atomic<int> g_level{static_cast<int>(Level::kWarn)};
std::atomic<int> g_format{static_cast<int>(Format::kText)};
std::mutex g_mu;

constexpr const char* kNames[] = {"trace", "debug", "info", "warn", "error", "off"};

Level parse_level(std::string_view s) {
  for (int i = 0; i <= static_cast<int>(Level::kOff); ++i)
    if (s == kNames[i]) return static_cast<Level>(i);
  if (s == "warning") return Level::kWarn;
  return Level::kInfo;
}

}  // namespace

void init(std::string_view level, Format format) {
  g_level = static_cast<int>(parse_level(level));
  g_format = static_cast<int>(format);
}

bool enabled(Level l) { return static_cast<int>(l) >= g_level.load(std::memory_order_relaxed) && l != Level::kOff; }

void write(Level l, std::string_view msg) {
  const auto now = std::chrono::system_clock::now();
  const auto t = std::chrono::system_clock::to_time_t(now);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
  std::tm tm{};
  localtime_r(&t, &tm);
  char ts[32];
  std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", &tm);
  std::string line;
  if (g_format == static_cast<int>(Format::kJson)) {
    line = nlohmann::json{{"ts", fmt::format("{}.{:03d}", ts, ms)}, {"level", kNames[static_cast<int>(l)]}, {"msg", msg}}.dump();
  } else {
    line = fmt::format("[{} {}] {}", ts + 11, kNames[static_cast<int>(l)], msg);
  }
  std::lock_guard lk(g_mu);
  std::fputs(line.c_str(), stderr);
  std::fputc('\n', stderr);
}

}  // namespace mqde::log
