// netron-style module graph built only from tensor names/shapes (no weights are read).
// Tree: model -> component -> block -> module -> tensor. Each module node gets an op type inferred
// from its name/shape so the UI can color it like netron (Linear / Norm / Attention / Embedding / Conv).
#include <map>

#include "mqde/modelspec/model_spec.h"

namespace mqde::modelspec {
namespace {

std::string op_of(const std::string& module, const TensorInfo& w) {
  auto has = [&](const char* s) { return module.find(s) != std::string::npos; };
  if (has("embed") || has("pos_embed") || has("position_embedding") || has("future_tokens"))
    return "Embedding";
  if (has("norm") || has("layernorm") || has("ln_")) return "Norm";
  if (has("conv") || has("patch_embed") || w.shape.size() >= 4) return "Conv";
  if (has("experts.")) return "Expert";
  if (has("router") || module.ends_with(".gate") || has("shared_expert_gate")) return "Router";
  if (has("attn") || has("q_proj") || has("k_proj") || has("v_proj") || has("o_proj") || has("qkv"))
    return "Attention";
  if (has("mlp") || has("ff") || has("fc")) return "MLP";
  if (w.shape.size() == 2) return "Linear";
  return "Param";
}

struct Node {
  std::string name, path, kind, op;
  int64_t params = 0;
  uint64_t bytes = 0;
  std::map<std::string, Node> children;  // ordered by name; UI re-sorts blocks numerically
  std::vector<size_t> tensors;           // indices into spec.tensors
};

Json emit(const Node& n, const ModelSpec& spec, int depth, int max_depth) {
  Json j = {{"name", n.name}, {"path", n.path}, {"kind", n.kind}, {"params", n.params}, {"bytes", n.bytes}};
  if (!n.op.empty()) j["op"] = n.op;
  if (!n.tensors.empty()) {
    Json ts = Json::array();
    for (size_t i : n.tensors) {
      const auto& t = spec.tensors[i];
      const auto& u = spec.units[i];
      ts.push_back({{"name", t.name.substr(t.name.rfind('.') + 1)},
                    {"full", t.name},
                    {"dtype", t.dtype},
                    {"shape", t.shape},
                    {"logical_shape", u.logical_shape},
                    {"quantizable", u.quantizable},
                    {"aux", u.aux}});
    }
    j["tensors"] = ts;
  }
  if (!n.children.empty()) {
    j["n_children"] = n.children.size();
    if (depth < max_depth) {
      Json cs = Json::array();
      for (const auto& [_, c] : n.children) cs.push_back(emit(c, spec, depth + 1, max_depth));
      j["children"] = cs;
    }
  }
  return j;
}

}  // namespace

Json to_graph_json(const ModelSpec& spec, const std::string& focus, int max_depth) {
  Node root;
  root.name = spec.family;
  root.kind = "model";
  for (size_t i = 0; i < spec.tensors.size(); ++i) {
    const auto& t = spec.tensors[i];
    const auto& u = spec.units[i];
    const int64_t numel = u.aux ? 0 : (u.logical_numel > 0 ? u.logical_numel : t.numel);
    root.params += numel;
    root.bytes += t.nbytes;
    // component -> [block] -> module path (everything but the tensor leaf name)
    Node& comp = root.children[u.component];
    comp.name = u.component;
    comp.path = u.component;
    comp.kind = "component";
    comp.params += numel;
    comp.bytes += t.nbytes;
    Node* parent = &comp;
    if (u.block >= 0) {
      const std::string bname = fmt::format("{:04d}", u.block);  // zero-pad so map order = numeric order
      Node& b = comp.children[bname];
      b.name = fmt::format("block {}", u.block);
      b.path = fmt::format("{}/{}", u.component, u.block);
      b.kind = "block";
      b.params += numel;
      b.bytes += t.nbytes;
      parent = &b;
    }
    const auto dot = t.name.rfind('.');
    std::string module = dot == std::string::npos ? t.name : t.name.substr(0, dot);
    // Short module label: strip everything up to and including the block index.
    std::string label = module;
    if (u.block >= 0) {
      const std::string key = fmt::format(".{}.", u.block);
      if (auto p = module.find(key); p != std::string::npos) label = module.substr(p + key.size());
    } else if (auto p = module.find('.'); p != std::string::npos) {
      label = module.substr(p + 1);
    }
    Node& m = parent->children[label];
    m.name = label;
    m.path = module;
    m.kind = "module";
    if (m.op.empty() || t.name.ends_with(".weight")) m.op = op_of(module, t);
    m.params += numel;
    m.bytes += t.nbytes;
    m.tensors.push_back(i);
  }
  // Focus on a sub-tree ("language", "language/3", ...) for lazy expansion in the UI.
  const Node* start = &root;
  if (!focus.empty()) {
    const auto slash = focus.find('/');
    const std::string comp = focus.substr(0, slash);
    auto c = root.children.find(comp);
    if (c == root.children.end()) return {{"error", "no such component"}};
    start = &c->second;
    if (slash != std::string::npos) {
      auto b = c->second.children.find(fmt::format("{:04d}", std::stoll(focus.substr(slash + 1))));
      if (b == c->second.children.end()) return {{"error", "no such block"}};
      start = &b->second;
    }
  }
  return emit(*start, spec, 0, max_depth);
}

}  // namespace mqde::modelspec
