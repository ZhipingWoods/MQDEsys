#pragma once
// Name -> factory registry for plug-in style components (algorithms, exporters, backends).
//
//   // in a header:   using AlgoRegistry = mqde::Registry<Algorithm>;   AlgoRegistry& algos();
//   // in a .cpp:     MQDE_REGISTER(algos(), "rtn", RtnAlgorithm);
//   // anywhere that needs the registration kept when this lives in a static library:
//   //                MQDE_FORCE_LINK(rtn);   (declared next to the registration with MQDE_REGISTER_NAMED(id,
//   ...))
//
// Static registration objects in a static library are dropped by the linker when nothing references their
// translation unit. MQDE_REGISTER_NAMED defines a symbol per registration and MQDE_FORCE_LINK references it,
// which keeps the unit.

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <fmt/ranges.h>

#include "mqde/common/result.h"

namespace mqde {

template <class Base>
class Registry {
 public:
  using base_type = Base;
  using Factory = std::function<std::unique_ptr<Base>()>;

  // A second registration of a name is an error: two components silently sharing a name would make results
  // depend on link order.
  Result<void> add(const std::string& name, Factory f) {
    std::lock_guard lk(mu_);
    if (name.empty()) return make_error(Errc::kInvalidArgument, "registry: empty name");
    if (!f) return make_error(Errc::kInvalidArgument, "registry: '{}' has no factory", name);
    if (!items_.emplace(name, std::move(f)).second)
      return make_error(Errc::kInvalidArgument, "registry: '{}' is already registered", name);
    return {};
  }

  Result<std::unique_ptr<Base>> create(const std::string& name) const {
    std::lock_guard lk(mu_);
    const auto it = items_.find(name);
    if (it == items_.end()) {
      std::vector<std::string> n;
      for (const auto& [k, v] : items_) n.push_back(k);
      return make_error(Errc::kNotFound, "unknown name '{}'; available: {}", name,
                        n.empty() ? std::string("(none)") : fmt::format("{}", fmt::join(n, ", ")));
    }
    return it->second();
  }

  bool contains(const std::string& name) const {
    std::lock_guard lk(mu_);
    return items_.count(name) > 0;
  }
  std::vector<std::string> names() const {
    std::lock_guard lk(mu_);
    std::vector<std::string> n;
    for (const auto& [k, v] : items_) n.push_back(k);
    return n;  // std::map order: sorted
  }
  size_t size() const {
    std::lock_guard lk(mu_);
    return items_.size();
  }

 private:
  mutable std::mutex mu_;
  std::map<std::string, Factory> items_;
};

// Registration helper object: constructing it registers; a failure (duplicate name) is logged to stderr by
// `report`.
template <class Base, class Derived>
struct Registrar {
  Registrar(Registry<Base>& r, const char* name) {
    auto res = r.add(name, [] { return std::make_unique<Derived>(); });
    ok = res.has_value();
    if (!ok) error = res.error().to_string();
  }
  bool ok = true;
  std::string error;
};

}  // namespace mqde

#define MQDE_REGISTER_NAMED(id, registry, name, Type)                                    \
  extern "C" {                                                                           \
  int mqde_force_link_##id = 0;                                                          \
  }                                                                                      \
  static ::mqde::Registrar<std::remove_reference_t<decltype(registry)>::base_type, Type> \
      mqde_registrar_##id {                                                              \
    (registry), name                                                                     \
  }

#define MQDE_REGISTER(registry, name, Type) MQDE_REGISTER_NAMED(Type##_registered, registry, name, Type)

#define MQDE_FORCE_LINK(id)            \
  extern "C" int mqde_force_link_##id; \
  static const int* mqde_keep_##id [[maybe_unused]] = &mqde_force_link_##id
