#include "mqde/hub/hf.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>

#include <httplib.h>

namespace mqde::hub {

namespace fs = std::filesystem;

namespace {

std::string token_of(const HfOptions& o) {
  if (!o.token.empty()) return o.token;
  if (const char* t = std::getenv("HF_TOKEN"); t && *t) return t;
  return {};
}

// Splits "https://host[:port]" into scheme+host and path-less base for httplib.
std::unique_ptr<httplib::Client> make_client(const HfOptions& o, const std::string& base) {
  auto cli = std::make_unique<httplib::Client>(base);
  cli->set_follow_location(true);
  cli->set_connection_timeout(o.timeout_s, 0);
  cli->set_read_timeout(o.timeout_s, 0);
  for (const char* ca : {"/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
                         "/etc/ssl/cert.pem"})
    if (fs::exists(ca)) { cli->set_ca_cert_path(ca); break; }
  // https_proxy=http://host:port (only the host:port part is used).
  for (const char* var : {"https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"}) {
    const char* p = std::getenv(var);
    if (!p || !*p) continue;
    static const std::regex re(R"(^(?:[a-z]+://)?([^:/]+):(\d+))");
    std::cmatch m;
    if (std::regex_search(p, m, re)) cli->set_proxy(m[1].str(), std::stoi(m[2].str()));
    break;
  }
  if (const auto t = token_of(o); !t.empty()) cli->set_bearer_token_auth(t);
  return cli;
}

// Never include the token in an error message: only status and path.
Error http_error(const httplib::Result& r, const std::string& what) {
  if (!r) return Error{Errc::kExternal, fmt::format("{}: network error ({})", what, httplib::to_string(r.error()))};
  const Errc c = r->status == 404 ? Errc::kNotFound : r->status == 401 || r->status == 403 ? Errc::kInvalidArgument
                                                                                          : Errc::kExternal;
  return Error{c, fmt::format("{}: HTTP {}{}", what, r->status,
                              r->status == 401 || r->status == 403 ? " (gated or private repo; set HF_TOKEN)" : "")};
}

Result<Json> get_json(const HfOptions& o, const std::string& path, const std::string& what) {
  auto cli = make_client(o, o.endpoint);
  auto r = cli->Get(path);
  if (!r || r->status != 200) return std::unexpected(http_error(r, what));
  try {
    return Json::parse(r->body);
  } catch (const std::exception& e) {
    return make_error(Errc::kParse, "{}: bad JSON ({})", what, e.what());
  }
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

}  // namespace

Result<Json> list_models(const HfOptions& o, const std::string& query) {
  return get_json(o, "/api/models?" + query, "list models");
}

Result<Json> model_info(const HfOptions& o, const std::string& repo) {
  return get_json(o,
                  "/api/models/" + repo +
                      "?expand[]=safetensors&expand[]=config&expand[]=cardData&expand[]=downloads"
                      "&expand[]=likes&expand[]=pipeline_tag&expand[]=createdAt&expand[]=author",
                  "model info " + repo);
}

Result<Json> fetch_json_file(const HfOptions& o, const std::string& repo, const std::string& file) {
  return get_json(o, "/" + repo + "/resolve/main/" + file, repo + "/" + file);
}

Result<std::vector<RepoFile>> list_files(const HfOptions& o, const std::string& repo) {
  MQDE_ASSIGN_OR_RETURN(Json j, get_json(o, "/api/models/" + repo + "/tree/main?recursive=1", "list files " + repo));
  std::vector<RepoFile> out;
  for (const auto& e : j) {
    if (e.value("type", "") != "file") continue;
    RepoFile f;
    f.path = e.value("path", "");
    // LFS files report their real size under lfs.size; "size" is the pointer size for some repos.
    if (e.contains("lfs") && e["lfs"].is_object()) f.size = e["lfs"].value("size", uint64_t{0});
    if (f.size == 0) f.size = e.value("size", uint64_t{0});
    out.push_back(std::move(f));
  }
  return out;
}

Result<uint64_t> download_file(const HfOptions& o, const std::string& repo, const std::string& file,
                               const fs::path& dest, const std::function<bool(uint64_t, uint64_t)>& progress) {
  std::error_code ec;
  fs::create_directories(dest.parent_path(), ec);
  const fs::path part = dest.string() + ".part";
  std::ofstream out(part, std::ios::binary | std::ios::trunc);
  if (!out) return make_error(Errc::kIo, "cannot write {}", part.string());
  auto cli = make_client(o, o.endpoint);
  cli->set_read_timeout(std::max(o.timeout_s, 120), 0);
  uint64_t done = 0;
  bool aborted = false;
  auto r = cli->Get(
      "/" + repo + "/resolve/main/" + file, httplib::Headers{},
      [&](const char* data, size_t n) {
        out.write(data, static_cast<std::streamsize>(n));
        done += n;
        return static_cast<bool>(out);
      },
      [&](uint64_t cur, uint64_t total) {
        if (progress && !progress(cur, total)) { aborted = true; return false; }
        return true;
      });
  out.close();
  if (aborted) { fs::remove(part, ec); return make_error(Errc::kCancelled, "download of {} cancelled", file); }
  if (!r || r->status != 200) { fs::remove(part, ec); return std::unexpected(http_error(r, repo + "/" + file)); }
  fs::rename(part, dest, ec);
  if (ec) return make_error(Errc::kIo, "rename {} -> {}: {}", part.string(), dest.string(), ec.message());
  return done;
}

// ---- file selection ----

namespace {

// Approximate bits per weight of common GGUF quant tags, used as the quality ordering.
double gguf_bits(const std::string& tag) {
  static const std::map<std::string, double> k = {
      {"f32", 32}, {"f16", 16}, {"bf16", 16}, {"q8_0", 8.5}, {"q6_k", 6.6}, {"q5_k_m", 5.7}, {"q5_k_s", 5.5},
      {"q5_0", 5.5}, {"q4_k_m", 4.8}, {"q4_k_s", 4.6}, {"q4_0", 4.5}, {"iq4_xs", 4.3}, {"q3_k_l", 4.3},
      {"q3_k_m", 3.9}, {"q3_k_s", 3.5}, {"iq3_m", 3.7}, {"iq3_xxs", 3.1}, {"q2_k", 3.0}, {"iq2_m", 2.7},
      {"iq2_xs", 2.3}, {"iq1_m", 1.8}};
  const auto it = k.find(lower(tag));
  return it == k.end() ? 4.0 : it->second;
}

}  // namespace

std::vector<FileGroup> group_files(const std::vector<RepoFile>& files) {
  static const std::regex shard_re(R"(-\d{5}-of-\d{5})");
  static const std::regex quant_re(R"((?:^|[-_.])((?:I?Q\d(?:_[A-Z0-9]+)*)|FP16|F16|BF16|FP32|F32)(?:[-_.]|$))",
                                   std::regex::icase);
  std::map<std::string, FileGroup> gguf;
  FileGroup st{"safetensors", {}, 0, 16};
  bool has_st = false;
  std::vector<RepoFile> side;  // config / tokenizer files that ride along with safetensors
  for (const auto& f : files) {
    const std::string name = fs::path(f.path).filename().string();
    if (name.ends_with(".gguf")) {
      const std::string base = std::regex_replace(name, shard_re, "");
      std::smatch m;
      std::string tag = std::regex_search(base, m, quant_re) ? m[1].str() : base;
      std::transform(tag.begin(), tag.end(), tag.begin(), ::toupper);
      if (tag == "FP16") tag = "F16";
      if (tag == "FP32") tag = "F32";
      auto& g = gguf[tag];
      g.key = tag;
      g.quality = gguf_bits(tag);
      g.files.push_back(f);
      g.bytes += f.size;
    } else if (name.ends_with(".safetensors")) {
      st.files.push_back(f);
      st.bytes += f.size;
      has_st = true;
    } else if (name.ends_with(".json") || name.ends_with(".txt") || name.ends_with(".model") ||
               name.ends_with(".py") || name.ends_with(".jinja") || name == "tokenizer.model") {
      side.push_back(f);
    }
  }
  std::vector<FileGroup> out;
  for (auto& [_, g] : gguf) out.push_back(std::move(g));
  if (has_st) {
    for (const auto& s : side) { st.files.push_back(s); st.bytes += s.size; }
    out.push_back(std::move(st));
  }
  return out;
}

std::optional<FileGroup> choose_group(const std::vector<FileGroup>& groups, uint64_t budget, const std::string& want) {
  if (groups.empty()) return std::nullopt;
  if (!want.empty()) {
    for (const auto& g : groups)
      if (lower(g.key) == lower(want)) return g;
    return std::nullopt;
  }
  const FileGroup* best = nullptr;
  for (const auto& g : groups) {
    if (budget > 0 && g.bytes > budget) continue;
    if (!best || g.quality > best->quality || (g.quality == best->quality && g.bytes < best->bytes)) best = &g;
  }
  if (best) return *best;
  // Nothing fits: llmfit falls back to the smallest file.
  return *std::min_element(groups.begin(), groups.end(), [](const auto& a, const auto& b) { return a.bytes < b.bytes; });
}

std::string url_encode(const std::string& s) {
  std::string out;
  for (unsigned char c : s) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out += static_cast<char>(c);
    else out += fmt::format("%{:02X}", c);
  }
  return out;
}

std::optional<uint64_t> parse_size(const std::string& s) {
  static const std::regex re(R"(^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)i?B?\s*$)", std::regex::icase);
  std::smatch m;
  if (!std::regex_match(s, m, re)) return std::nullopt;
  double v = std::stod(m[1].str());
  switch (std::toupper(m[2].str().empty() ? ' ' : m[2].str()[0])) {
    case 'T': v *= 1024; [[fallthrough]];
    case 'G': v *= 1024; [[fallthrough]];
    case 'M': v *= 1024; [[fallthrough]];
    case 'K': v *= 1024; break;
    default: break;
  }
  return static_cast<uint64_t>(v);
}

}  // namespace mqde::hub
