#pragma once
// Error model shared by every module: module boundaries never throw, they return mqde::Result<T>.

#include <expected>
#include <string>
#include <string_view>
#include <utility>

#include <fmt/format.h>

namespace mqde {

enum class Errc {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kUnsupported,
  kResourceExhausted,
  kIo,
  kParse,
  kTimeout,
  kCancelled,
  kInternal,
  kExternal,
};

std::string_view errc_name(Errc c);

struct Error {
  Errc code = Errc::kInternal;
  std::string message;

  // Prepends context ("while loading config.json: ...") so the final message reads outer -> inner.
  Error with_context(std::string_view ctx) const& {
    return Error{code, fmt::format("{}: {}", ctx, message)};
  }
  std::string to_string() const { return fmt::format("[{}] {}", errc_name(code), message); }
};

template <class T>
using Result = std::expected<T, Error>;

template <class... Args>
std::unexpected<Error> make_error(Errc code, fmt::format_string<Args...> f, Args&&... args) {
  return std::unexpected(Error{code, fmt::format(f, std::forward<Args>(args)...)});
}

inline std::string_view errc_name(Errc c) {
  switch (c) {
    case Errc::kOk: return "ok";
    case Errc::kInvalidArgument: return "invalid_argument";
    case Errc::kNotFound: return "not_found";
    case Errc::kUnsupported: return "unsupported";
    case Errc::kResourceExhausted: return "resource_exhausted";
    case Errc::kIo: return "io";
    case Errc::kParse: return "parse";
    case Errc::kTimeout: return "timeout";
    case Errc::kCancelled: return "cancelled";
    case Errc::kInternal: return "internal";
    case Errc::kExternal: return "external";
  }
  return "unknown";
}

}  // namespace mqde

// Propagate an error from an expression returning Result<T>, binding the value otherwise.
#define MQDE_CONCAT_INNER(a, b) a##b
#define MQDE_CONCAT(a, b) MQDE_CONCAT_INNER(a, b)
#define MQDE_ASSIGN_OR_RETURN(lhs, expr)                     \
  auto MQDE_CONCAT(_mqde_r_, __LINE__) = (expr);             \
  if (!MQDE_CONCAT(_mqde_r_, __LINE__))                      \
    return std::unexpected(MQDE_CONCAT(_mqde_r_, __LINE__).error()); \
  lhs = std::move(*MQDE_CONCAT(_mqde_r_, __LINE__))
