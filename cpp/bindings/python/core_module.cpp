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
#include "mqde/quant/pack.h"
#include "mqde_probe.h"

namespace py = pybind11;
using mqde::Json;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using U8Array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using I32Array = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using U16Array = py::array_t<uint16_t, py::array::c_style | py::array::forcecast>;

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

namespace pk = mqde::quant::pack;

// (codes [out,in] u8, zeros [out,groups] u8, scales [out,groups] f32) -> (qweight i32, qzeros i32, scales
// f16)
py::tuple packed_tuple(const pk::IntPacked& p) {
  I32Array qw({p.qweight_rows, p.qweight_cols});
  std::memcpy(qw.mutable_data(), p.qweight.data(), p.qweight.size() * 4);
  I32Array qz({p.qzeros_rows, p.qzeros_cols});
  std::memcpy(qz.mutable_data(), p.qzeros.data(), p.qzeros.size() * 4);
  // int64_t is long long on macOS but py::ssize_t is long, so both extents need the explicit cast.
  py::array sc(py::dtype("float16"), {static_cast<py::ssize_t>(p.qzeros_rows),
                                      static_cast<py::ssize_t>(p.scales.size() / p.qzeros_rows)});
  std::memcpy(sc.mutable_data(), p.scales.data(), p.scales.size() * 2);
  return py::make_tuple(qw, qz, sc);
}

void check_int_inputs(const U8Array& codes, const U8Array& zeros, const FloatArray& scales) {
  if (codes.ndim() != 2 || zeros.ndim() != 2 || scales.ndim() != 2)
    throw py::value_error("codes/zeros/scales must be 2-D");
  if (zeros.shape(0) != codes.shape(0) || scales.shape(0) != codes.shape(0) ||
      scales.shape(1) != zeros.shape(1))
    throw py::value_error("zeros/scales must be [out, groups] matching codes [out, in]");
}

pk::IntPacked packed_from(const I32Array& qw, const I32Array& qz) {
  if (qw.ndim() != 2 || qz.ndim() != 2) throw py::value_error("qweight/qzeros must be 2-D");
  pk::IntPacked p;
  p.qweight.assign(qw.data(), qw.data() + qw.size());
  p.qzeros.assign(qz.data(), qz.data() + qz.size());
  p.qweight_rows = qw.shape(0);
  p.qweight_cols = qw.shape(1);
  p.qzeros_rows = qz.shape(0);
  p.qzeros_cols = qz.shape(1);
  return p;
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
      py::arg("x"), py::arg("spec_json"),
      "Quantize-dequantize a [rows, cols] float32 matrix. Returns (out, scale, zero).");

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

  // ---- bit packing (T-0405): GPTQ / AWQ checkpoint tensors and GGUF blocks ----
  m.def(
      "pack_int",
      [](const std::string& layout, const U8Array& codes, const U8Array& zeros, const FloatArray& scales,
         int bits, bool v1) {
        check_int_inputs(codes, zeros, scales);
        const int64_t out = codes.shape(0), in = codes.shape(1), gs = in / zeros.shape(1);
        auto p =
            layout == "gptq" ? pk::pack_gptq(codes.data(), zeros.data(), scales.data(), out, in, gs, bits, v1)
            : layout == "awq"
                ? pk::pack_awq(codes.data(), zeros.data(), scales.data(), out, in, gs, bits)
                : std::unexpected(mqde::Error{mqde::Errc::kInvalidArgument, "layout must be gptq | awq"});
        if (!p) raise(p.error());
        return packed_tuple(*p);
      },
      py::arg("layout"), py::arg("codes"), py::arg("zeros"), py::arg("scales"), py::arg("bits"),
      py::arg("v1") = true,
      "Pack unsigned codes into GPTQ/AWQ tensors. Returns (qweight int32, qzeros int32, scales float16 "
      "[groups, out]).");

  m.def(
      "unpack_int",
      [](const std::string& layout, const I32Array& qweight, const I32Array& qzeros, int64_t out, int64_t in,
         int64_t group_size, int bits, bool v1) {
        const auto p = packed_from(qweight, qzeros);
        U8Array codes({out, in});
        U8Array zeros({out, in / group_size});
        auto r =
            layout == "gptq" ? pk::unpack_gptq(p, out, in, group_size, bits, v1, codes.mutable_data(),
                                               zeros.mutable_data())
            : layout == "awq"
                ? pk::unpack_awq(p, out, in, group_size, bits, codes.mutable_data(), zeros.mutable_data())
                : std::unexpected(mqde::Error{mqde::Errc::kInvalidArgument, "layout must be gptq | awq"});
        if (!r) raise(r.error());
        return py::make_tuple(codes, zeros);
      },
      py::arg("layout"), py::arg("qweight"), py::arg("qzeros"), py::arg("out"), py::arg("in_features"),
      py::arg("group_size"), py::arg("bits"), py::arg("v1") = true);

  // ---- Marlin (vLLM gptq_marlin / awq_marlin) ----
  m.def(
      "pack_marlin",
      [](const U8Array& codes, const U8Array& zeros, const FloatArray& scales, int bits, bool with_zeros) {
        check_int_inputs(codes, zeros, scales);
        const int64_t out = codes.shape(0), in = codes.shape(1), groups = zeros.shape(1), gs = in / groups;
        auto p = pk::pack_marlin(codes.data(), zeros.data(), scales.data(), out, in, gs, bits, with_zeros);
        if (!p) raise(p.error());
        I32Array qw({static_cast<py::ssize_t>(p->qweight_rows), static_cast<py::ssize_t>(p->qweight_cols)});
        std::memcpy(qw.mutable_data(), p->qweight.data(), p->qweight.size() * 4);
        py::array sc(py::dtype("float16"), {static_cast<py::ssize_t>(groups), static_cast<py::ssize_t>(out)});
        std::memcpy(sc.mutable_data(), p->scales.data(), p->scales.size() * 2);
        py::object qz = py::none();
        if (with_zeros) {
          I32Array z({static_cast<py::ssize_t>(groups), static_cast<py::ssize_t>(p->qzeros_cols)});
          std::memcpy(z.mutable_data(), p->qzeros.data(), p->qzeros.size() * 4);
          qz = z;
        }
        return py::make_tuple(qw, qz, sc);
      },
      py::arg("codes"), py::arg("zeros"), py::arg("scales"), py::arg("bits"), py::arg("with_zeros"),
      "codes [out, in] / zeros, scales [out, groups] -> Marlin (qweight int32 [in/16, out*16*bits/32], "
      "qzeros int32 [groups, out*bits/32] or None, scales float16 [groups, out]).");

  m.def(
      "unpack_marlin",
      // scales_bits: the fp16 scales viewed as uint16 (numpy .view(np.uint16)), [groups, out]
      [](const I32Array& qweight, const py::object& qzeros, const U16Array& scales_bits, int64_t out,
         int64_t in, int64_t group_size, int bits) {
        if (qweight.ndim() != 2) throw py::value_error("qweight must be 2-D");
        pk::MarlinPacked p;
        p.qweight_rows = qweight.shape(0);
        p.qweight_cols = qweight.shape(1);
        p.qweight.assign(qweight.data(), qweight.data() + qweight.size());
        p.scales.assign(scales_bits.data(), scales_bits.data() + scales_bits.size());
        const bool has_z = !qzeros.is_none();
        if (has_z) {
          const auto z = qzeros.cast<I32Array>();
          if (z.ndim() != 2) throw py::value_error("qzeros must be 2-D");
          p.qzeros_cols = z.shape(1);
          p.qzeros.assign(z.data(), z.data() + z.size());
        }
        const int64_t groups = group_size > 0 ? in / group_size : 0;
        p.groups = groups;
        U8Array codes({static_cast<py::ssize_t>(out), static_cast<py::ssize_t>(in)});
        U8Array zeros({static_cast<py::ssize_t>(out), static_cast<py::ssize_t>(groups)});
        FloatArray scales({static_cast<py::ssize_t>(out), static_cast<py::ssize_t>(groups)});
        auto r = pk::unpack_marlin(p, out, in, group_size, bits, codes.mutable_data(),
                                   has_z ? zeros.mutable_data() : nullptr, scales.mutable_data());
        if (!r) raise(r.error());
        return py::make_tuple(codes, has_z ? py::object(zeros) : py::object(py::none()), scales);
      },
      py::arg("qweight"), py::arg("qzeros"), py::arg("scales_bits"), py::arg("out"), py::arg("in_features"),
      py::arg("group_size"), py::arg("bits"),
      "Marlin tensors -> (codes uint8 [out, in], zeros uint8 [out, groups] or None, scales float32 [out, "
      "groups]).");

  m.def(
      "gguf_quantize",
      [](const FloatArray& x, const std::string& type) {
        auto t = pk::gguf_type_from_name(type);
        if (!t) raise(t.error());
        std::vector<uint8_t> q;
        {
          py::gil_scoped_release nogil;
          auto r = pk::gguf_quantize(*t, x.data(), x.size());
          py::gil_scoped_acquire gil;
          if (!r) raise(r.error());
          q = std::move(*r);
        }
        U8Array out(static_cast<py::ssize_t>(q.size()));
        std::memcpy(out.mutable_data(), q.data(), q.size());
        return out;
      },
      py::arg("x"), py::arg("type"), "GGUF block quantization (bit-identical to ggml quantize_row_*_ref).");

  m.def(
      "gguf_dequantize",
      [](const U8Array& data, const std::string& type, int64_t n) {
        auto t = pk::gguf_type_from_name(type);
        if (!t) raise(t.error());
        const auto tr = pk::gguf_traits(*t);
        if (n % tr.block_elems || data.size() != n / tr.block_elems * tr.block_bytes)
          throw py::value_error("data size does not match n elements of " + std::string(tr.name));
        FloatArray out(n);
        auto r = pk::gguf_dequantize(*t, data.data(), n, out.mutable_data());
        if (!r) raise(r.error());
        return out;
      },
      py::arg("data"), py::arg("type"), py::arg("n"));

  m.def(
      "measure",
      [](const FloatArray& ref, const FloatArray& approx) {
        if (ref.size() != approx.size()) throw py::value_error("size mismatch");
        return mqde::quant::to_json(mqde::quant::measure(ref.data(), approx.data(), ref.size())).dump();
      },
      py::arg("ref"), py::arg("approx"));

  m.def(
      "tensor_stats",
      [](const FloatArray& x) {
        return mqde::quant::to_json(mqde::quant::tensor_stats(x.data(), x.size())).dump();
      },
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
      [](const std::string& model_path, const std::string& recipe, const std::string& hw,
         const std::string& backend, const std::string& runtime_json) {
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
        j["curves"] = {{"concurrency", mqde::estimator::to_json(mqde::estimator::memory_vs_concurrency(
                                           *spec, *r, *h, *b, rt, {1, 2, 4, 8, 16, 32, 64}))},
                       {"images", mqde::estimator::to_json(mqde::estimator::memory_vs_images(
                                      *spec, *r, *h, *b, rt, {1, 2, 4, 8, 16, 30, 60}))}};
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
        arr.push_back({{"kind", mqde_dev_kind_name(devs[i].kind)},
                       {"vendor", devs[i].vendor},
                       {"index", devs[i].index},
                       {"name", devs[i].name},
                       {"arch", devs[i].arch},
                       {"mem_total_bytes", devs[i].mem_total_bytes},
                       {"mem_free_bytes", devs[i].mem_free_bytes}});
    return arr.dump();
  });
}
