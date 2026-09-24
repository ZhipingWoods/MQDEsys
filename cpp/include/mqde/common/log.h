#pragma once
// Logging facade over spdlog. Human-readable to stderr by default, JSON Lines when requested,
// so stdout stays reserved for command output (CLI) or protocol frames (worker).

#include <string_view>

#include <spdlog/spdlog.h>

namespace mqde::log {

enum class Format { kText, kJson };

// Idempotent; safe to call from every entry point.
void init(std::string_view level = "info", Format format = Format::kText);

}  // namespace mqde::log

#define MQDE_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define MQDE_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define MQDE_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define MQDE_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
