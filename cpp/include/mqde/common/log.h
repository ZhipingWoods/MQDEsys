#pragma once
// Minimal thread-safe logger on fmt. Human-readable to stderr by default, JSON Lines when requested,
// so stdout stays reserved for command output (CLI) or protocol frames (worker).

#include <string_view>

#include <fmt/format.h>

namespace mqde::log {

enum class Level { kTrace = 0, kDebug, kInfo, kWarn, kError, kOff };
enum class Format { kText, kJson };

// Idempotent; safe to call from every entry point. level: trace|debug|info|warn|error|off
void init(std::string_view level = "info", Format format = Format::kText);
bool enabled(Level l);
void write(Level l, std::string_view msg);

template <class... Args>
void logf(Level l, fmt::format_string<Args...> f, Args&&... args) {
  if (enabled(l)) write(l, fmt::format(f, std::forward<Args>(args)...));
}

}  // namespace mqde::log

#define MQDE_LOG_DEBUG(...) ::mqde::log::logf(::mqde::log::Level::kDebug, __VA_ARGS__)
#define MQDE_LOG_INFO(...) ::mqde::log::logf(::mqde::log::Level::kInfo, __VA_ARGS__)
#define MQDE_LOG_WARN(...) ::mqde::log::logf(::mqde::log::Level::kWarn, __VA_ARGS__)
#define MQDE_LOG_ERROR(...) ::mqde::log::logf(::mqde::log::Level::kError, __VA_ARGS__)
