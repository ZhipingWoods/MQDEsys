#pragma once
// Minimal ModelScope Hub client: model search, repo file listing, and HF-compatible file downloads
// (`/<repo>/resolve/<revision>/<file>`, which modelscope.cn serves). Kept separate from hf.h because
// ModelScope's search and file-tree endpoints differ from Hugging Face's.

#include <cstdint>
#include <string>
#include <vector>

#include "mqde/common/result.h"
#include "mqde/hub/hf.h"

namespace mqde::hub {

struct ModelscopeOptions {
  std::string endpoint = "https://modelscope.cn";
  std::string token;  // empty -> MODELSCOPE_API_TOKEN / MODELSCOPE_TOKEN
  int timeout_s = 30;
};

ModelscopeOptions default_modelscope_options();

// GET /openapi/v1/models?search=<q>&page_size=<n>&sort=downloads -> the `data` object:
// {"models":[{id, display_name, downloads, tasks, params, license, ...}], "total_count": ...}
Result<Json> modelscope_search(const ModelscopeOptions& o, const std::string& query, int limit);

// GET /api/v1/models/<repo>/repo/files?Revision=<rev>&Recursive=True -> files with sizes.
Result<std::vector<RepoFile>> modelscope_list_files(const ModelscopeOptions& o, const std::string& repo,
                                                    const std::string& revision = "master");

// The resolve/file-download endpoints are compatible with Hugging Face's, so the existing resumable
// downloader can be reused with these options.
HfOptions modelscope_hf_options(const ModelscopeOptions& o);

}  // namespace mqde::hub
