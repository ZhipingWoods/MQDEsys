#include "mqde/hub/modelscope.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <regex>
#include <thread>

#include <httplib.h>

namespace mqde::hub {

namespace {

namespace fs = std::filesystem;

bool retryable(const httplib::Result& r) { return !r || r->status == 429 || r->status >= 500; }

template <class F>
httplib::Result with_retry(F&& f) {
  httplib::Result r = f();
  for (int i = 0; i < 2 && retryable(r); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500 << i));
    r = f();
  }
  return r;
}

std::string token_of(const ModelscopeOptions& o) {
  if (!o.token.empty()) return o.token;
  for (const char* var : {"MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN"})
    if (const char* t = std::getenv(var); t && *t) return t;
  return {};
}

std::unique_ptr<httplib::Client> make_client(const ModelscopeOptions& o) {
  auto cli = std::make_unique<httplib::Client>(o.endpoint);
  cli->set_follow_location(true);
  cli->set_default_headers(
      {{"User-Agent", "mqde/0.1 (+https://github.com/zhipingwoods/MQDEsys)"}, {"Accept", "*/*"}});
  cli->set_connection_timeout(o.timeout_s, 0);
  cli->set_read_timeout(o.timeout_s, 0);
  for (const char* ca :
       {"/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/cert.pem"})
    if (fs::exists(ca)) {
      cli->set_ca_cert_path(ca);
      break;
    }
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

Error http_error(const httplib::Result& r, const std::string& what) {
  if (!r)
    return Error{Errc::kExternal, fmt::format("{}: network error ({})", what, httplib::to_string(r.error()))};
  const Errc c = r->status == 404                       ? Errc::kNotFound
                 : r->status == 401 || r->status == 403 ? Errc::kInvalidArgument
                                                        : Errc::kExternal;
  return Error{c, fmt::format("{}: HTTP {}", what, r->status)};
}

Result<Json> get_json(const ModelscopeOptions& o, const std::string& path, const std::string& what) {
  auto cli = make_client(o);
  auto r = with_retry([&] { return cli->Get(path); });
  if (!r || r->status != 200) return std::unexpected(http_error(r, what));
  try {
    return Json::parse(r->body);
  } catch (const std::exception& e) {
    return make_error(Errc::kParse, "{}: bad JSON ({})", what, e.what());
  }
}

}  // namespace

ModelscopeOptions default_modelscope_options() {
  ModelscopeOptions o;
  if (const char* e = std::getenv("MODELSCOPE_ENDPOINT"); e && *e) o.endpoint = e;
  while (o.endpoint.ends_with('/')) o.endpoint.pop_back();
  o.token = token_of(o);
  return o;
}

Result<Json> modelscope_search(const ModelscopeOptions& o, const std::string& query, int limit) {
  const int n = std::clamp(limit, 1, 100);
  auto j = get_json(o,
                    "/openapi/v1/models?search=" + url_encode(query) + "&page_size=" + std::to_string(n) +
                        "&sort=downloads",
                    "search ModelScope");
  if (!j) return j;
  if (j->contains("data") && (*j)["data"].is_object()) return (*j)["data"];
  return *j;
}

Result<std::vector<RepoFile>> modelscope_list_files(const ModelscopeOptions& o, const std::string& repo,
                                                    const std::string& revision) {
  MQDE_ASSIGN_OR_RETURN(auto ref, parse_repo_ref(repo));
  std::string rev = revision;
  if (const auto at = repo.rfind('@'); at != std::string::npos) rev = repo.substr(at + 1);
  if (rev.empty()) rev = "master";
  static const std::regex rev_re(R"(^[A-Za-z0-9._/-]+$)");
  if (!std::regex_match(rev, rev_re) || rev.find("..") != std::string::npos)
    return make_error(Errc::kInvalidArgument, "bad ModelScope revision '{}'", rev);

  MQDE_ASSIGN_OR_RETURN(
      Json j, get_json(o,
                       "/api/v1/models/" + ref.repo + "/repo/files?Revision=" + url_encode(rev) +
                           "&Recursive=True",
                       "list ModelScope files " + repo));
  const Json* files = nullptr;
  if (j.contains("Data")) {
    if (j["Data"].is_array()) files = &j["Data"];
    else if (j["Data"].contains("Files") && j["Data"]["Files"].is_array()) files = &j["Data"]["Files"];
  }
  if (!files && j.contains("Files") && j["Files"].is_array()) files = &j["Files"];
  if (!files) return make_error(Errc::kParse, "list ModelScope files {}: response has no file list", repo);

  std::vector<RepoFile> out;
  for (const auto& e : *files) {
    if (e.value("Type", std::string("blob")) == "tree") continue;
    RepoFile f;
    f.path = e.value("Path", e.value("Name", std::string{}));
    if (f.path.empty()) continue;
    f.size = e.value("Size", uint64_t{0});
    out.push_back(std::move(f));
  }
  return out;
}

HfOptions modelscope_hf_options(const ModelscopeOptions& o) {
  HfOptions h;
  h.endpoint = o.endpoint;
  h.repo_path_prefix = "/models";
  h.token = token_of(o);
  h.timeout_s = o.timeout_s;
  return h;
}

}  // namespace mqde::hub
