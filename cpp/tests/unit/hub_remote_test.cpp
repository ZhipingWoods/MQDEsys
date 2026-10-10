// T-0204: repo refs, HF_ENDPOINT, retries and safetensors headers over HTTP Range against a local stub
// (no network): only the header bytes may be transferred, never the tensor payload.
#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <thread>

#include <httplib.h>

#include "mqde/common/hash.h"
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
  for (int i = 0; i < 8; ++i)
    out[static_cast<size_t>(i)] = static_cast<char>((header.size() >> (8 * i)) & 0xFF);
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
          {{"type":"directory","path":"sub"}}])",
                                    blob.size()),
                        "application/json");
      });
      svr.Get("/org/tiny/resolve/" + rev + "/config.json",
              [cfg](const httplib::Request&, httplib::Response& res) {
                res.set_content(cfg, "application/json");
              });
      svr.Get("/org/tiny/resolve/" + rev + "/model.safetensors",
              [this](const httplib::Request& req, httplib::Response& res) {
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

// ---- T-0505: resumable download with checksum verification
// ---------------------------------------------------------

namespace {

std::string pattern(size_t n) {
  std::string s(n, '\0');
  for (size_t i = 0; i < n; ++i) s[i] = static_cast<char>((i * 31 + 7) % 251);
  return s;
}

std::string slurp(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

struct DlStub {
  httplib::Server svr;
  std::thread th;
  int port = 0;
  std::string blob = pattern(3 * 1024 * 1024 + 123);
  std::string etag;                 // sent as X-Linked-ETag when not empty
  bool honor_range = true;          // false: always answer 200 with the full body
  std::atomic<int> drop_after{-1};  // >= 0: send this many bytes then cut the connection (once)
  std::atomic<int> always_drop_after{-1};  // like drop_after, but every request, not just the first
  std::atomic<int> requests{0};
  std::atomic<long long> first_range{-1};
  std::atomic<uint64_t> bytes_sent{0};

  DlStub() {
    svr.Get("/org/m/resolve/main/w.bin", [this](const httplib::Request& req, httplib::Response& res) {
      ++requests;
      if (!honor_range)
        const_cast<httplib::Request&>(req).ranges.clear();  // behave like a server without Range support
      if (!req.ranges.empty()) first_range = static_cast<long long>(req.ranges[0].first);
      if (!etag.empty()) res.set_header("X-Linked-ETag", "\"" + etag + "\"");
      // A 1-byte Range probe (bytes=0-0, used by download_file_chunked's size/Range-support check) only
      // reads headers; it must not consume the one-shot drop_after meant for a real chunk's data transfer.
      const bool is_probe = req.ranges.size() == 1 && req.ranges[0].first == 0 && req.ranges[0].second == 0;
      const int one_shot_cut = is_probe ? -1 : drop_after.exchange(-1);
      const int persistent_cut = always_drop_after.load();
      const int cut = one_shot_cut >= 0           ? one_shot_cut
                     : persistent_cut >= 0 && !is_probe ? persistent_cut
                                                        : -1;
      auto delivered = std::make_shared<size_t>(0);
      // httplib turns this into 200 or 206 and calls the provider with absolute offsets inside the requested
      // range
      res.set_content_provider(blob.size(), "application/octet-stream",
                               [this, cut, delivered](size_t offset, size_t length, httplib::DataSink& sink) {
                                 size_t n = std::min<size_t>(length, 64 * 1024);
                                 if (cut >= 0) {
                                   if (*delivered >= static_cast<size_t>(cut))
                                     return false;  // connection dies mid-body
                                   n = std::min<size_t>(n, static_cast<size_t>(cut) - *delivered);
                                 }
                                 sink.write(blob.data() + offset, n);
                                 *delivered += n;
                                 bytes_sent += n;
                                 return true;
                               });
    });
    port = svr.bind_to_any_port("127.0.0.1");
    th = std::thread([this] { svr.listen_after_bind(); });
    svr.wait_until_ready();
  }
  ~DlStub() {
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

std::filesystem::path tmpdir(const char* name) {
  auto d = std::filesystem::temp_directory_path() / fmt::format("mqde_dl_{}_{}", name, ::getpid());
  std::filesystem::remove_all(d);
  std::filesystem::create_directories(d);
  return d;
}

}  // namespace

TEST(hub_download, sha256_matches_the_standard_test_vectors) {
  EXPECT_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");  // two blocks
  EXPECT_EQ(sha256_hex(std::string(1000000, 'a')),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(hub_download, plain_download_writes_the_file_and_leaves_no_part_file) {
  DlStub s;
  const auto d = tmpdir("plain");
  uint64_t last_done = 0, last_total = 0;
  auto n = download_file(s.opts(), "org/m", "w.bin", d / "sub" / "w.bin", [&](uint64_t cur, uint64_t total) {
    last_done = cur, last_total = total;
    return true;
  });
  ASSERT_TRUE(n) << n.error().message;
  EXPECT_EQ(*n, s.blob.size());
  EXPECT_EQ(slurp(d / "sub" / "w.bin"), s.blob);
  EXPECT_FALSE(std::filesystem::exists(d / "sub" / "w.bin.part"));
  EXPECT_EQ(last_done, s.blob.size());
  EXPECT_EQ(last_total, s.blob.size());
}

TEST(hub_download, an_existing_part_file_is_continued_not_restarted) {
  DlStub s;
  const auto d = tmpdir("resume");
  const size_t have = 1'000'003;
  std::ofstream(d / "w.bin.part", std::ios::binary) << s.blob.substr(0, have);
  uint64_t first_progress = 0;
  auto n = download_file(s.opts(), "org/m", "w.bin", d / "w.bin", [&](uint64_t cur, uint64_t) {
    if (first_progress == 0) first_progress = cur;
    return true;
  });
  ASSERT_TRUE(n) << n.error().message;
  EXPECT_EQ(s.first_range.load(), static_cast<long long>(have))
      << "asked for the bytes after the ones it already had";
  EXPECT_EQ(s.bytes_sent.load(), s.blob.size() - have) << "only the missing part was transferred";
  EXPECT_EQ(slurp(d / "w.bin"), s.blob);
  EXPECT_GT(first_progress, have) << "progress counts the bytes already on disk";
  EXPECT_EQ(*n, s.blob.size());
}

TEST(hub_download, a_connection_dropped_mid_body_keeps_the_partial_file_and_the_next_call_finishes) {
  DlStub s;
  const auto d = tmpdir("drop");
  s.drop_after = 700'000;
  auto first = download_file(s.opts(), "org/m", "w.bin", d / "w.bin");
  EXPECT_FALSE(first) << "the truncated body is an error, never a finished file";
  EXPECT_FALSE(std::filesystem::exists(d / "w.bin"));
  ASSERT_TRUE(std::filesystem::exists(d / "w.bin.part"));
  const auto kept = std::filesystem::file_size(d / "w.bin.part");
  EXPECT_GT(kept, 0u);
  EXPECT_LT(kept, s.blob.size());
  EXPECT_EQ(slurp(d / "w.bin.part"), s.blob.substr(0, kept)) << "what was kept is a correct prefix";
  auto second = download_file(s.opts(), "org/m", "w.bin", d / "w.bin");
  ASSERT_TRUE(second) << second.error().message;
  EXPECT_EQ(s.first_range.load(), static_cast<long long>(kept));
  EXPECT_EQ(slurp(d / "w.bin"), s.blob);
}

TEST(hub_download, a_server_that_ignores_range_makes_it_start_over_with_a_correct_file) {
  DlStub s;
  s.honor_range = false;
  const auto d = tmpdir("norange");
  std::ofstream(d / "w.bin.part", std::ios::binary) << s.blob.substr(0, 4096);
  auto n = download_file(s.opts(), "org/m", "w.bin", d / "w.bin");
  ASSERT_TRUE(n) << n.error().message;
  EXPECT_EQ(slurp(d / "w.bin"), s.blob) << "no duplicated prefix";
  EXPECT_EQ(*n, s.blob.size());
}

TEST(hub_download, checksum_is_verified_when_the_server_sends_one) {
  DlStub s;
  s.etag = sha256_hex(s.blob);
  const auto d = tmpdir("sha_ok");
  ASSERT_TRUE(download_file(s.opts(), "org/m", "w.bin", d / "w.bin"));
  s.etag = std::string(64, 'a');  // wrong
  const auto d2 = tmpdir("sha_bad");
  auto bad = download_file(s.opts(), "org/m", "w.bin", d2 / "w.bin");
  ASSERT_FALSE(bad);
  EXPECT_NE(bad.error().message.find("sha256 mismatch"), std::string::npos) << bad.error().message;
  EXPECT_FALSE(std::filesystem::exists(d2 / "w.bin")) << "a corrupt file is never published";
  EXPECT_FALSE(std::filesystem::exists(d2 / "w.bin.part"))
      << "and the bad partial file is removed so it is not resumed";
}

TEST(hub_download, a_resumed_download_is_verified_over_the_whole_file) {
  DlStub s;
  s.etag = sha256_hex(s.blob);
  const auto d = tmpdir("resume_sha");
  std::ofstream(d / "w.bin.part", std::ios::binary) << s.blob.substr(0, 500'000);
  ASSERT_TRUE(download_file(s.opts(), "org/m", "w.bin", d / "w.bin"));
  // a part file with corrupted content: the resume succeeds in transferring, but the whole-file hash catches
  // it
  const auto d2 = tmpdir("resume_corrupt");
  std::string wrong = s.blob.substr(0, 500'000);
  wrong[1000] ^= 0x55;
  std::ofstream(d2 / "w.bin.part", std::ios::binary) << wrong;
  auto bad = download_file(s.opts(), "org/m", "w.bin", d2 / "w.bin");
  ASSERT_FALSE(bad);
  EXPECT_NE(bad.error().message.find("sha256 mismatch"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(d2 / "w.bin"));
}

TEST(hub_download, cancelling_keeps_the_partial_file_for_a_later_resume) {
  DlStub s;
  const auto d = tmpdir("cancel");
  auto r = download_file(s.opts(), "org/m", "w.bin", d / "w.bin",
                         [](uint64_t cur, uint64_t) { return cur < 300'000; });
  ASSERT_FALSE(r);
  EXPECT_EQ(r.error().code, Errc::kCancelled);
  EXPECT_TRUE(std::filesystem::exists(d / "w.bin.part"));
  EXPECT_FALSE(std::filesystem::exists(d / "w.bin"));
}

// ---- download_file_chunked (T-0505 concurrent chunked download) ----

TEST(hub_download_chunked, reassembles_the_exact_bytes_from_several_concurrent_range_requests) {
  DlStub s;
  const auto d = tmpdir("chunked_ok");
  std::atomic<uint64_t> last_done{0}, last_total{0};
  auto n = download_file_chunked(s.opts(), "org/m", "w.bin", d / "w.bin", 4,
                                 [&](uint64_t cur, uint64_t total) {
                                   last_done = cur;
                                   last_total = total;
                                   return true;
                                 });
  ASSERT_TRUE(n) << n.error().message;
  EXPECT_EQ(*n, s.blob.size());
  EXPECT_EQ(slurp(d / "w.bin"), s.blob) << "every byte must land at the right offset, not just the right count";
  EXPECT_FALSE(std::filesystem::exists(d / "w.bin.part"));
  EXPECT_EQ(last_total.load(), s.blob.size());
  EXPECT_EQ(last_done.load(), s.blob.size()) << "progress must reach the full size, summed across chunks";
  EXPECT_GE(s.requests.load(), 4) << "at least one request per chunk (plus the size probe)";
}

TEST(hub_download_chunked, one_chunk_behaves_like_a_plain_download) {
  DlStub s;
  const auto d = tmpdir("chunked_one");
  auto n = download_file_chunked(s.opts(), "org/m", "w.bin", d / "w.bin", 1);
  ASSERT_TRUE(n) << n.error().message;
  EXPECT_EQ(slurp(d / "w.bin"), s.blob);
}

TEST(hub_download_chunked, falls_back_to_a_single_stream_when_the_server_ignores_range) {
  DlStub s;
  s.honor_range = false;
  const auto d = tmpdir("chunked_fallback");
  auto n = download_file_chunked(s.opts(), "org/m", "w.bin", d / "w.bin", 4);
  ASSERT_TRUE(n) << n.error().message;
  EXPECT_EQ(slurp(d / "w.bin"), s.blob);
}

TEST(hub_download_chunked, checksum_is_verified_over_the_whole_reassembled_file) {
  DlStub s;
  s.etag = sha256_hex(s.blob);
  const auto d = tmpdir("chunked_sha_ok");
  ASSERT_TRUE(download_file_chunked(s.opts(), "org/m", "w.bin", d / "w.bin", 4));
  s.etag = std::string(64, 'a');  // wrong
  const auto d2 = tmpdir("chunked_sha_bad");
  auto bad = download_file_chunked(s.opts(), "org/m", "w.bin", d2 / "w.bin", 4);
  ASSERT_FALSE(bad);
  EXPECT_NE(bad.error().message.find("sha256 mismatch"), std::string::npos) << bad.error().message;
  EXPECT_FALSE(std::filesystem::exists(d2 / "w.bin"));
}

TEST(hub_download_chunked, a_one_shot_connection_drop_on_one_chunk_is_retried_and_still_succeeds) {
  DlStub s;
  s.drop_after = 1000;  // whichever chunk's request happens to arrive first is cut short, once
  const auto d = tmpdir("chunked_drop_once");
  auto r = download_file_chunked(s.opts(), "org/m", "w.bin", d / "w.bin", 4);
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_EQ(slurp(d / "w.bin"), s.blob) << "the cut chunk's own retry must recover it, like a single-stream "
                                       "download recovers from one dropped connection";
}

TEST(hub_download_chunked, a_persistent_range_failure_fails_the_whole_download_and_removes_the_partial_file) {
  DlStub s;
  s.always_drop_after = 1000;  // every request for this file is cut short, not just the first -- no retry helps
  const auto d = tmpdir("chunked_drop_persistent");
  auto r = download_file_chunked(s.opts(), "org/m", "w.bin", d / "w.bin", 4);
  ASSERT_FALSE(r) << "a chunk that can never complete must not be silently reported as success";
  EXPECT_FALSE(std::filesystem::exists(d / "w.bin"));
  EXPECT_FALSE(std::filesystem::exists(d / "w.bin.part")) << "the broken partial file must not be left behind";
}
