#include "mqde/fit/benchmarks.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>

namespace mqde::fit {
namespace fs = std::filesystem;
namespace {

std::string lower(std::string_view s) {
  std::string o(s);
  std::transform(o.begin(), o.end(), o.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return o;
}

bool ieq(std::string_view a, std::string_view b) { return lower(a) == lower(b); }

double median_sorted(const std::vector<double>& v) {
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

}  // namespace

bool plausible_tps(double tps) { return std::isfinite(tps) && tps > 0.0 && tps <= 10'000.0; }

std::string canonical_slug(std::string_view name) {
  // Strip trailing separators so "/cache/hub/Model/" and "Model" agree.
  while (!name.empty() && (name.back() == '/' || name.back() == '\\')) name.remove_suffix(1);
  const auto cut = name.find_last_of("/\\");
  if (cut != std::string_view::npos) name.remove_prefix(cut + 1);
  std::string out;
  out.reserve(name.size());
  for (unsigned char c : name)
    if (c != '-' && c != '_' && c != '.') out.push_back(static_cast<char>(std::tolower(c)));
  return out;
}

std::string hardware_slug(std::string_view gpu_name) {
  std::string out;
  bool dash = false;
  for (unsigned char c : gpu_name) {
    if (std::isalnum(c)) {
      out.push_back(static_cast<char>(std::tolower(c)));
      dash = false;
    } else if (!dash && !out.empty()) {
      out.push_back('-');
      dash = true;
    }
  }
  while (!out.empty() && out.back() == '-') out.pop_back();
  return out.empty() ? "cpu-only" : out;
}

Result<std::vector<BenchmarkIndex::Row>> read_submission(const fs::path& file, const HardwareId& hw,
                                                         bool require_source) {
  MQDE_ASSIGN_OR_RETURN(Json j, load_structured_file(file));
  if (!j.is_object()) return make_error(Errc::kParse, "{}: not a JSON object", file.string());
  if (require_source &&
      !(j.contains("source") && j["source"].is_string() && !j["source"].get<std::string>().empty()))
    return make_error(Errc::kInvalidArgument,
                      "{}: community data must carry a non-empty `source` (origin + license)", file.string());
  const Json h = j.value("hardware", Json::object());
  // llmfit hardware_payload_matches: same CPU name AND same GPU name (both absent also matches).
  const std::string cpu = h.value("cpu", "");
  const std::string gpu = h.value("hardwareName", "");
  const bool cpu_ok = !cpu.empty() && ieq(cpu, hw.cpu_name);
  const bool gpu_ok = hw.gpu_name.empty() ? gpu.empty() : (!gpu.empty() && ieq(gpu, hw.gpu_name));
  std::vector<BenchmarkIndex::Row> rows;
  if (!cpu_ok || !gpu_ok) return rows;
  for (const auto& r : j.value("results", Json::array())) {
    if (!r.is_object() || !r.contains("model") || !r["model"].is_string()) continue;
    const double tps = r.value("avgTps", 0.0);
    if (!plausible_tps(tps)) continue;
    BenchmarkIndex::Row row;
    row.slug = canonical_slug(r["model"].get<std::string>());
    row.quant = lower(r.value("quant", ""));
    row.tps = tps;
    if (r.contains("avgTtftMs") && r["avgTtftMs"].is_number()) row.ttft_ms = r["avgTtftMs"].get<double>();
    row.runs = std::max(1, r.value("numRuns", 1));
    rows.push_back(std::move(row));
  }
  return rows;
}

BenchmarkIndex BenchmarkIndex::load(const HardwareId& hw, const fs::path& root_in) {
  BenchmarkIndex idx;
  const fs::path root = root_in.empty() ? data_dir() / "benchmarks" : root_in;
  auto scan = [&](const char* sub, bool require_source, std::vector<Row>& out) {
    std::error_code ec;
    const fs::path dir = root / sub;
    if (!fs::is_directory(dir, ec)) return;
    std::vector<fs::path> files;
    for (auto it = fs::recursive_directory_iterator(dir, ec); !ec && it != fs::recursive_directory_iterator();
         it.increment(ec))
      if (it->is_regular_file() && it->path().extension() == ".json") files.push_back(it->path());
    std::sort(files.begin(), files.end());  // deterministic order
    for (const auto& f : files) {
      auto rows = read_submission(f, hw, require_source);
      if (!rows) {
        ++idx.skipped_;
        continue;
      }
      out.insert(out.end(), rows->begin(), rows->end());
    }
  };
  scan("local", false, idx.local_);
  scan("community", true, idx.community_);
  return idx;
}

BenchmarkIndex BenchmarkIndex::from_rows(std::vector<Row> local, std::vector<Row> community) {
  BenchmarkIndex idx;
  idx.local_ = std::move(local);
  idx.community_ = std::move(community);
  return idx;
}

std::optional<MeasuredTps> BenchmarkIndex::find(const std::vector<Row>& rows, const std::string& slug,
                                                std::string_view quant, MeasuredSource src) const {
  const std::string q = lower(quant);
  std::vector<double> tps, ttft;
  int samples = 0;
  for (const auto& r : rows) {
    if (r.slug != slug) continue;
    if (!q.empty() && !r.quant.empty() && r.quant != q) continue;
    tps.push_back(r.tps);
    if (r.ttft_ms) ttft.push_back(*r.ttft_ms);
    samples += r.runs;
  }
  if (tps.empty()) return std::nullopt;
  std::sort(tps.begin(), tps.end());
  std::sort(ttft.begin(), ttft.end());
  MeasuredTps m;
  m.tok_s = median_sorted(tps);
  if (!ttft.empty()) m.ttft_ms = median_sorted(ttft);
  m.sample_count = samples;
  m.source = src;
  m.hardware_label = src == MeasuredSource::kLocal ? "this machine" : "identical hardware";
  return m;
}

std::optional<MeasuredTps> BenchmarkIndex::lookup(std::string_view model, std::string_view quant) const {
  const std::string slug = canonical_slug(model);
  if (auto m = find(local_, slug, quant, MeasuredSource::kLocal)) return m;
  return find(community_, slug, quant, MeasuredSource::kCommunity);
}

Json to_json(const Submission& s) {
  Json results = Json::array();
  for (const auto& r : s.results) {
    Json e = {{"model", r.model},    {"quant", r.quant},    {"runtime", r.runtime},  {"avgTps", r.avg_tps},
              {"minTps", r.min_tps}, {"maxTps", r.max_tps}, {"numRuns", r.num_runs}, {"unit", r.unit}};
    if (r.avg_ttft_ms) e["avgTtftMs"] = *r.avg_ttft_ms;
    if (r.peak_mem_bytes > 0) e["peakMemBytes"] = r.peak_mem_bytes;
    if (r.context_tokens > 0) e["contextTokens"] = r.context_tokens;
    results.push_back(e);
  }
  const auto now =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch());
  return {{"schemaVersion", 1},
          {"submittedAtUnix", now.count()},
          {"tool", {{"name", "mqde"}, {"version", s.tool_version}}},
          {"hardware",
           {{"cpu", s.hw.cpu_name},
            {"cpuCores", s.cpu_cores},
            {"gpuCount", s.gpu_count},
            {"hardwareName", s.hw.gpu_name},
            {"ramGb", s.ram_gb},
            {"vramGb", s.vram_gb}}},
          {"results", results}};
}

Result<fs::path> write_local_submission(const Submission& s, const fs::path& root_in) {
  if (s.results.empty()) return make_error(Errc::kInvalidArgument, "submission has no results");
  for (const auto& r : s.results)
    if (!plausible_tps(r.avg_tps))
      return make_error(Errc::kInvalidArgument, "implausible avgTps {} for {}", r.avg_tps, r.model);
  const fs::path root = root_in.empty() ? data_dir() / "benchmarks" : root_in;
  const fs::path dir = root / "local" / hardware_slug(s.hw.gpu_name);
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) return make_error(Errc::kIo, "cannot create {}: {}", dir.string(), ec.message());
  const Json j = to_json(s);
  std::mt19937 rng(std::random_device{}());
  const fs::path file = dir / fmt::format("{}-{:08x}.json", j["submittedAtUnix"].get<int64_t>(), rng());
  std::ofstream out(file);
  if (!out) return make_error(Errc::kIo, "cannot write {}", file.string());
  out << j.dump(2) << "\n";
  if (!out) return make_error(Errc::kIo, "short write on {}", file.string());
  return file;
}

}  // namespace mqde::fit
