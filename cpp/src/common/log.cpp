#include "mqde/common/log.h"

#include <mutex>
#include <string>

#include <spdlog/sinks/stdout_color_sinks.h>

namespace mqde::log {

void init(std::string_view level, Format format) {
  static std::mutex mu;
  std::lock_guard lock(mu);
  auto logger = spdlog::get("mqde");
  if (!logger) {
    logger = spdlog::stderr_color_mt("mqde");
    spdlog::set_default_logger(logger);
  }
  if (format == Format::kJson) {
    logger->set_pattern(R"({"ts":"%Y-%m-%dT%H:%M:%S.%e","level":"%l","msg":"%v"})");
  } else {
    logger->set_pattern("%^[%H:%M:%S.%e %L]%$ %v");
  }
  logger->set_level(spdlog::level::from_str(std::string(level)));
}

}  // namespace mqde::log
