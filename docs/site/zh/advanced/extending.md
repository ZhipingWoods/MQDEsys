# 扩展 MQDEsys

动手之前先读 `docs/STRUCTURE.md` 中的评审级别：新增 Python 文件、数据条目、界面页面和测试可以直接修改；公共 C++ 头文件（`cpp/include/**`）、构建文件、golden 与兼容性数据以及架构文档需要先获得维护者批准；新增第三方依赖需要先写 ADR。绝不要修改 `3rdpart/` 下的任何内容，也绝不要为了让测试通过而修改 golden 数据。

下面每个扩展点都是一个注册表，通常再加一个数据文件。任何改动之后都要运行本地门禁（见 [测试](testing.md)）。

## 新增量化算法

1. 在 `python/mqde/quant/algorithms.py` 中添加用 `@register("name")` 装饰的函数，签名为 `(w, x, ws, as_, p) -> AlgoResult`：`w` 是 float32 权重 `[out, in]`，`x` 是捕获的输入激活 `[rows, in]`（或 `None`），`ws` 与 `as_` 是权重和激活的 `QuantSpec`，`p` 是参数。返回要交给配方量化器的权重；若结果已经在量化网格上（如 GPTQ），则设置 `final=True` 并给出 `scale` 与 `zero`。
2. 注意类别：`resolve()` 允许一个变换（`awq`、`smoothquant`：它们设置 `in_scale`）和一个量化器（`rtn`、`gptq`）用 `+` 连接。新的变换或量化器必须加进这些元组，否则 `name+gptq` 会被拒绝。若希望模型保持可导出，变换的缩放必须能折叠进前一个算子。
3. 在 `data/algorithms.yaml` 中添加条目（`name`、`label`、`kind: quantizer|transform|combo`、`description`、`needs_calibration`、`params`）；服务端据此校验 `quant.algorithm`，界面据此列出。
4. 在 `python/tests/test_algorithms.py` 中写测试：注册与 `resolve`、合成层上相对 RTN 的误差、是否落在量化网格上、确定性。

## 新增导出格式

1. 在 `python/mqde/export/` 下新建模块，类继承 `Exporter`，用 `@register("fmt")` 装饰。实现 `check_layer()`（返回该层无法导出的原因，不要抛异常）和 `export()`（写 `model.safetensors`、带 `quantization_config` 的 `config.json`，并复制 tokenizer 文件）。
2. 在 `python/mqde/export/__init__.py` 中 import 该模块；该 import 就是注册动作。
3. 让一致性检查能读回它：`real_check.load_real`、`dequant_real` 和 `use_real_weights` 按 `quant_method` 分派。不能被读回的格式得不到 S10 的保证。
4. 测试：布局、反量化权重在 scale 存储精度内等于伪量化权重、混合精度、`check()` 的拒绝；在格式的参考库可安装时，与它逐字节比对（见 `python/tests/test_pack_compat.py`）。
5. C++ 打包器需要 `cpp/src/quant/pack.cpp`、`cpp/include/mqde/quant/` 中的头文件（需批准）、pybind 绑定以及 `python/mqde/core.py` 里的封装。

## 新增部署后端

1. 继承 `ServerBackend`（`python/mqde/backend/base.py`）：构造参数列表和环境变量，实现 `ready()`，并把 `diagnoses` 设为取自**真实**失败日志的（日志子串，说明）对。`vllm_server.py` 和 `llamacpp_server.py` 是现成的例子。
2. 通过环境变量、`PATH`、`build/<name>/bin` 的顺序查找二进制；找不到时作业阶段必须是 `skipped` 而不是失败。
3. 添加预估器画像 `data/backends/<name>.yaml`（`id`、`runtime_overhead_bytes`、`allocator_factor`、`eta_bw`、`mfu`）。
4. 用 `python/tests/backend/fake_llama_server.py` 里的 mock 服务器测试（生命周期、从真实日志行得到的诊断、计时精度），再加一个只有环境变量指向真实二进制时才运行的测试。

## 新增模型架构

数据优先于代码：`data/model_families/<family>.yaml`，包含 `family`、`modality`、`match.model_type`、`config_paths`、`components`（name、modality、`prefix`、`block_regex`）和 `default_keep_fp`（保持高精度的层）。用 `mqde model inspect <MODEL_DIR>` 检查组件和块数是否正确。Hugging Face transformers 能当作因果语言模型加载的架构，已经可以通过 `generic_decoder` 和 `HFCausalLM` 适配器使用；只有结构特殊时才需要新的 Python `QuantModel` 子类。

## 新增配方或 KV 方法

配方是 `data/recipes/<name>.yaml`（`name`、`description`、`maturity`、`default_algorithm`、`modalities`、`weight`、`act`、`kv`）；`mqde recipes` 必须能列出它，`mqde estimate -r <name>` 必须能运行。KV 方法是 `python/mqde/quant/kv.py` 中用 `@register("name")` 注册的类，提供 `quant_k` 与 `quant_v`；通过 `quant.kv.method` 选用。

## 新增硬件画像、API 路由或页面

硬件：`data/hardware_db/<vendor>/<id>.yaml`。REST 路由：`cpp/src/server/server.cpp`，加上 `web/src/api.ts` 里带类型的客户端。页面：`web/src/pages/<Name>.tsx`，以及 `web/src/main.tsx` 中的路由和导航项。硬件*插件*（通过稳定 C ABI 提供厂商探针或微基准）为计划中，目前不可用。
