// T-0204: repo refs, HF_ENDPOINT, retries and safetensors headers over HTTP Range against a local stub
// (no network): only the header bytes may be transferred, never the tensor payload.
#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <thread>

#include <httplib.h>

#include "mqde/hub/hf.h"

using namespace mqde;
using namespace mqde::hub;

namespace {

std::string safetensors_blob(size_t payload_bytes, std::string* header_out = nullptr) {
  // two tensors: 8x4 BF16 and a large F32 payload
  const uint64_t big = payload_bytes / 4;
  const std::string header = fmt::format(
      R"({{"__metadata__":{{"format":"pt"}},"model.embed_tokens.weight":{{"dtype":"BF16","shape":[8,4],"data_offsets":[0,64]}},)"
      R"("model.layers.0.mlp.down_proj.weight":{{"dtype":"F32","shape":[{},4],"data_offsets":[64,{}]}}}})",
      big / 4, 64 + big * 4);
  std::string out(8, '\0');
  for (int i = 0; i < 8; ++i) out[static_cast<size_t>(i)] = static_cast<char>((header.size() >> (8 * i)) & 0xFF);
  out += header;
  out.append(64 + big * 4, '\0');
  if (header_out) *header_out = header;
  return out;
}

struct Stub {
  httplib::Server svr;
  std::thread th;
  int port = 0;
  std::atomic<int> flaky{0};
  std::atomic<uint64_t> served{0};
  std::string blob;

  Stub() {
    blob = safetensors_blob(4 << 20);
    const std::string cfg = R"({"model_type":"llama","architectures":["LlamaForCausalLM"],"hidden_size":4,
      "intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":8,
      "max_position_embeddings":64,"torch_dtype":"bfloat16"})";
    for (const std::string rev : {"main", "v1"}) {
      svr.Get("/api/models/org/tiny/tree/" + rev, [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(fmt::format(R"([{{"type":"file","path":"config.json","size":10}},
          {{"type":"file","path":"model.safetensors","size":{0},"lfs":{{"size":{0}}}}},
          {{"type":"directory","path":"sub"}}])", blob.size()), "application/json");
      });
      svr.Get("/org/tiny/resolve/" + rev + "/config.json",
              [cfg](const httplib::Request&, httplib::Response& res) { res.set_content(cfg, "application/json"); });
      svr.Get("/org/tiny/resolve/" + rev + "/model.safetensors", [this](const httplib::Request& req, httplib::Response& res) {
        // httplib slices the body to the requested Range (206); count what the client asked for
        ASSERT_TRUE(req.has_header("Range"));
        const auto& r = req.ranges.at(0);
        served += static_cast<uint64_t>(r.second - r.first + 1);
        res.set_content(blob, "application/octet-stream");
      });
    }
    svr.Get("/api/models/org/flaky/tree/main", [this](const httplib::Request&, httplib::Response& res) {
      if (flaky++ < 2) {
        res.status = 503;
        return;
      }
      res.set_content(R"([{"type":"file","path":"a.bin","size":3}])", "application/json");
    });
    port = svr.bind_to_any_port("127.0.0.1");
    th = std::thread([this] { svr.listen_after_bind(); });
    svr.wait_until_ready();
  }
  ~Stub() {
    svr.stop();
    th.join();
  }
  HfOptions opts() const {
    HfOptions o;
    o.endpoint = fmt::format("http://127.0.0.1:{}", port);
    o.timeout_s = 5;
    return o;
  }
};

}  // namespace

TEST(hub_remote, parse_repo_ref) {
  auto a = parse_repo_ref("Qwen/Qwen2.5-7B-Instruct");
  ASSERT_TRUE(a);
  EXPECT_EQ(a->repo, "Qwen/Qwen2.5-7B-Instruct");
  EXPECT_EQ(a->revision, "main");
  auto b = parse_repo_ref("org/m@refs/pr/3");
  ASSERT_TRUE(b);
  EXPECT_EQ(b->revision, "refs/pr/3");
  EXPECT_TRUE(parse_repo_ref("gpt2"));
  for (const char* bad : {"", "org/", "/m", "a/b/c", "org/m@", "org/../m", "org/m@..", "or g/m"})
    EXPECT_FALSE(parse_repo_ref(bad)) << bad;
}

TEST(hub_remote, endpoint_from_env) {
  ::setenv("HF_ENDPOINT", "https://hf-mirror.com/", 1);
  EXPECT_EQ(default_endpoint(), "https://hf-mirror.com");
  EXPECT_EQ(HfOptions{}.endpoint, "https://hf-mirror.com");
  ::unsetenv("HF_ENDPOINT");
  EXPECT_EQ(default_endpoint(), "https://huggingface.co");
}

TEST(hub_remote, header_over_range_without_payload) {
  Stub s;
  std::string header;
  safetensors_blob(4 << 20, &header);
  auto t = remote_safetensors_header(s.opts(), "org/tiny", "model.safetensors");
  ASSERT_TRUE(t) << t.error().message;
  ASSERT_EQ(t->size(), 2u);
  EXPECT_EQ((*t)[1].name, "model.layers.0.mlp.down_proj.weight");
  EXPECT_EQ((*t)[1].shape, (std::vector<int64_t>{262144, 4}));
  EXPECT_EQ(s.served.load(), 8 + header.size());  // 4 MiB payload never transferred
}

TEST(hub_remote, inspect_remote_builds_spec_at_revision) {
  Stub s;
  uint64_t fetched = 0;
  auto spec = inspect_remote(s.opts(), "org/tiny@v1", &fetched);
  ASSERT_TRUE(spec) << spec.error().message;
  EXPECT_EQ(spec->path, "hf://org/tiny@v1");
  EXPECT_EQ(spec->tensors.size(), 2u);
  EXPECT_LT(fetched, 1024u);
  EXPECT_FALSE(inspect_remote(s.opts(), "org/../x"));
}

TEST(hub_remote, retries_transient_errors) {
  Stub s;
  auto f = list_files(s.opts(), "org/flaky");
  ASSERT_TRUE(f) << f.error().message;
  EXPECT_EQ(s.flaky.load(), 3);
  EXPECT_EQ(f->at(0).path, "a.bin");
}
