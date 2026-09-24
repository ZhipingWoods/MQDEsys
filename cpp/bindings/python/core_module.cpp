// mqde._core - Python binding of the C++ core (numerical quantization semantics, ModelSpec,
// estimator, probe). Arrays are passed via the buffer protocol: C-contiguous float32 inputs are
// read without copies (torch: `t.numpy()` on CPU tensors). JSON-shaped results cross as strings
// and are decoded by the thin wrapper in python/mqde/core.py.

#include <cstring>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mqde/estimator/estimator.h"
#include "mqde/quant/metrics.h"
#include "mqde/quant/ops.h"
#include "mqde_probe.h"

namespace py = pybind11;
using mqde::Json;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

[[noreturn]] void raise(const mqde::Error& e) {
  if (e.code == mqde::Errc::kNotFound) throw py::key_error(e.to_string());
  throw py::value_error(e.to_string());
}

mqde::quant::QuantSpec spec_from(const std::string& spec_json) {
  auto s = mqde::quant::parse_quant_spec(Json::parse(spec_json), "spec");
  if (!s) raise(s.error());
  return *s;
}

std::pair<int64_t, int64_t> as_2d(const FloatArray& x) {
  if (x.ndim() == 2) return {x.shape(0), x.shape(1)};
  if (x.ndim() == 1) return {1, x.shape(0)};
  throw py::value_error("expected a 1-D or 2-D float32 array");
}

py::tuple qparams_tuple(const mqde::quant::QParams& qp, int64_t rows) {
  const int64_t r = qp.per_tensor ? 1 : rows;
  const int64_t g = qp.per_tensor ? 1 : qp.groups_per_row;
  FloatArray scale({r, g});
  std::memcpy(scale.mutable_data(), qp.scale.data(), qp.scale.size() * sizeof(float));
  py::object zero = py::none();
  if (!qp.zero.empty()) {
    FloatArray z({r, g});
    std::memcpy(z.mutable_data(), qp.zero.data(), qp.zero.size() * sizeof(float));
    zero = z;
  }
  return py::make_tuple(scale, zero);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "MQDEsys C++ core";
  m.def("version", [] { return std::string(MQDE_VERSION_INFO); });

  m.def(
      "fake_quant",
      [](const FloatArray& x, const std::string& spec_json) {
        const auto spec = spec_from(spec_json);
        const auto [rows, cols] = as_2d(x);
        FloatArray out({rows, cols});
        mqde::quant::QParams qp;
        {
          py::gil_scoped_release nogil;
          auto r = mqde::quant::fake_quant(x.data(), rows, cols, spec, out.mutable_data(), &qp);
          if (!r) {
            py::gil_scoped_acquire gil;
            raise(r.error());
          }
        }
        py::tuple t = qparams_tuple(qp, rows);
        return py::make_tuple(out, t[0], t[1]);
      },
      py::arg("x"), py::arg("spec_json"), "Quantize-dequantize a [rows, cols] float32 matrix. Returns (out, scale, zero).");

  m.def(
      "quantize_int",
      [](const FloatArray& x, const std::string& spec_json) {
        const auto spec = spec_from(spec_json);
        const auto [rows, cols] = as_2d(x);
        auto qp = mqde::quant::compute_qparams(x.data(), rows, cols, spec);
        if (!qp) raise(qp.error());
        py::array_t<int16_t> codes({rows, cols});
        auto r = mqde::quant::quantize_int(x.data(), rows, cols, spec, *qp, codes.mutable_data());
        if (!r) raise(r.error());
        py::tuple t = qparams_tuple(*qp, rows);
        return py::make_tuple(codes, t[0], t[1]);
      },
      py::arg("x"), py::arg("spec_json"), "Integer codes + qparams. Returns (codes int16, scale, zero).");

  m.def(
      "fp8_round",
      [](const FloatArray& x, const std::string& fmt) {
        const auto f = fmt == "e5m2" ? mqde::quant::NumFormat::kFp8E5M2 : mqde::quant::NumFormat::kFp8E4M3;
        FloatArray out(std::vector<py::ssize_t>(x.shape(), x.shape() + x.ndim()));
        const float* in = x.data();
        float* o = out.mutable_data();
        for (py::ssize_t i = 0; i < x.size(); ++i) o[i] = mqde::quant::fp8_round(in[i], f);
        return out;
      },
      py::arg("x"), py::arg("fmt") = "e4m3");

  m.def(
      "measure",
      [](const FloatArray& ref, const FloatArray& approx) {
        if (ref.size() != approx.size()) throw py::value_error("size mismatch");
        return mqde::quant::to_json(mqde::quant::measure(ref.data(), approx.data(), ref.size())).dump();
      },
      py::arg("ref"), py::arg("approx"));

  m.def(
      "tensor_stats", [](const FloatArray& x) { return mqde::quant::to_json(mqde::quant::tensor_stats(x.data(), x.size())).dump(); },
      py::arg("x"));

  m.def(
      "inspect_model",
      [](const std::string& path, bool tensors) {
        auto s = mqde::modelspec::inspect_local(path);
        if (!s) raise(s.error());
        return mqde::modelspec::to_json(*s, tensors).dump();
      },
      py::arg("path"), py::arg("tensors") = false);

  m.def(
      "estimate",
      [](const std::string& model_path, const std::string& recipe, const std::string& hw, const std::string& backend,
         const std::string& runtime_json) {
        auto spec = mqde::modelspec::inspect_local(model_path);
        if (!spec) raise(spec.error());
        auto r = mqde::estimator::load_recipe(recipe);
        if (!r) raise(r.error());
        auto h = mqde::estimator::load_hardware(hw);
        if (!h) raise(h.error());
        auto b = mqde::estimator::load_backend(backend);
        if (!b) raise(b.error());
        const Json rj = Json::parse(runtime_json);
        mqde::estimator::RuntimeParams rt;
        rt.concurrency = rj.value("concurrency", rt.concurrency);
        rt.images_per_step = rj.value("images_per_step", rt.images_per_step);
        rt.text_tokens = rj.value("text_tokens", rt.text_tokens);
        rt.context_tokens = rj.value("context_tokens", rt.context_tokens);
        rt.image_resolution = rj.value("image_resolution", rt.image_resolution);
        rt.gpu_memory_utilization = rj.value("gpu_memory_utilization", rt.gpu_memory_utilization);
        rt.extra_modalities = rj.value("extra_modalities", rt.extra_modalities);
        auto e = mqde::estimator::estimate(*spec, *r, *h, *b, rt);
        Json j = mqde::estimator::to_json(e);
        j["recipe_detail"] = mqde::estimator::to_json(*r);
        j["curves"] = {
            {"concurrency", mqde::estimator::to_json(mqde::estimator::memory_vs_concurrency(*spec, *r, *h, *b, rt, {1, 2, 4, 8, 16, 32, 64}))},
            {"images", mqde::estimator::to_json(mqde::estimator::memory_vs_images(*spec, *r, *h, *b, rt, {1, 2, 4, 8, 16, 30, 60}))}};
        return j.dump();
      },
      py::arg("model_path"), py::arg("recipe"), py::arg("hardware") = "nvidia/rtx4090",
      py::arg("backend") = "hf_transformers", py::arg("runtime_json") = "{}");

  m.def("load_recipe", [](const std::string& name) {
    auto r = mqde::estimator::load_recipe(name);
    if (!r) raise(r.error());
    return mqde::estimator::to_json(*r).dump();
  });

  m.def("probe", [] {
    mqde_device_info devs[32];
    int n = 0;
    Json arr = Json::array();
    if (mqde_probe_all(devs, 32, &n) == 0)
      for (int i = 0; i < n; ++i)
        arr.push_back({{"kind", devs[i].kind == MQDE_DEV_CPU ? "cpu" : "gpu"},
                       {"index", devs[i].index},
                       {"name", devs[i].name},
                       {"arch", devs[i].arch},
                       {"mem_total_bytes", devs[i].mem_total_bytes},
                       {"mem_free_bytes", devs[i].mem_free_bytes}});
    return arr.dump();
  });
}
