# 扩展系统

MQDEsys 的扩展点以注册表与数据文件为核心。新增能力通常只需添加一个模块或一个 YAML 条目，无需修改主流程。修改后应运行[测试与本地门禁](testing.md)中的检查。

## 新增量化算法

1. 在 `python/mqde/quant/algorithms.py` 中新增以 `@register("name")` 装饰的函数，签名为 `(w, x, ws, as_, p) -> AlgoResult`。其中 `w` 为 float32 权重 `[out, in]`，`x` 为捕获的输入激活 `[rows, in]`（可为 `None`），`ws` 与 `as_` 分别是权重和激活的 `QuantSpec`，`p` 为参数。函数返回交给配方量化器的权重；若结果已位于量化网格（如 GPTQ），需设置 `final=True` 并提供 `scale` 与 `zero`。
2. 遵循算法组合规则：`resolve()` 允许一个变换（`awq`、`smoothquant`，设置 `in_scale`）与一个量化器（`rtn`、`gptq`、`hqq`）以 `+` 连接。新增变换或量化器需加入相应元组，否则 `name+gptq` 形式会被拒绝。若需保持模型可导出，变换的缩放必须能折叠进前一个算子。
3. 在 `data/algorithms.yaml` 中添加条目，字段包括 `name`、`label`、`kind`（`quantizer`、`transform` 或 `combo`）、`description`、`needs_calibration` 与 `params`。服务端据此校验 `quant.algorithm`，Web 界面据此生成选项。
4. 在 `python/tests/test_algorithms.py` 中添加测试，覆盖注册与 `resolve`、合成层上相对 RTN 的误差、量化网格约束与确定性。

## 新增导出格式

1. 在 `python/mqde/export/` 下新建模块，继承 `Exporter` 并以 `@register("fmt")` 装饰。实现 `check_layer()`（返回该层无法导出的原因，不抛出异常）与 `export()`（写出 `model.safetensors`、带 `quantization_config` 的 `config.json`，并复制 tokenizer 文件）。
2. 在 `python/mqde/export/__init__.py` 中导入该模块以完成注册。
3. 使真实性检查支持读回该格式：`real_check.load_real`、`dequant_real` 与 `use_real_weights` 按 `quant_method` 分派。
4. 添加测试，覆盖布局、反量化权重在 scale 存储精度内等于伪量化权重、混合精度与 `check()` 的拒绝行为；在参考库可用时进行逐字节比对（参见 `python/tests/test_pack_compat.py`）。
5. 如需在 C++ 中打包，需同步修改 `cpp/src/quant/pack.cpp`、`cpp/include/mqde/quant/` 下的头文件、pybind 绑定与 `python/mqde/core.py` 中的封装。

## 新增部署后端

1. 继承 `ServerBackend`（`python/mqde/backend/base.py`）：定义构造参数与环境变量，实现 `ready()`，并将 `diagnoses` 设置为取自真实失败日志的（日志子串，说明）对。`vllm_server.py`、`sglang_server.py` 与 `llamacpp_server.py` 可作为参考实现。
2. 按环境变量、`PATH`、`build/<name>/bin` 的顺序查找二进制；未找到时作业阶段应标记为 `skipped`，而非失败。
3. 添加预估器画像 `data/backends/<name>.yaml`，字段包括 `id`、`runtime_overhead_bytes`、`allocator_factor`、`eta_bw` 与 `mfu`。
4. 使用 `python/tests/backend/fake_llama_server.py` 中的 mock 服务测试生命周期、日志诊断与计时精度，并补充在环境变量指向真实二进制时运行的集成测试。

## 新增模型架构

优先使用数据描述：在 `data/model_families/<family>.yaml` 中定义 `family`、`modality`、`match.model_type`、`config_paths`、`components`（`name`、`modality`、`prefix`、`block_regex`）与 `default_keep_fp`（保持高精度的层）。使用 `mqde model inspect <MODEL_DIR>` 校验组件与块数。

凡能由 Hugging Face transformers 作为因果语言模型加载的架构，可直接通过 `generic_decoder` 与 `HFCausalLM` 适配器使用；结构特殊的模型需实现新的 Python `QuantModel` 子类。

## 新增配方或 KV 量化方法

配方定义在 `data/recipes/<name>.yaml`，字段包括 `name`、`description`、`maturity`、`default_algorithm`、`modalities`、`weight`、`act` 与 `kv`。新增后应确保 `mqde recipes` 能列出该配方，且 `mqde estimate -r <name>` 可正常运行。

KV 量化方法在 `python/mqde/quant/kv.py` 中以 `@register("name")` 注册，提供 `quant_k` 与 `quant_v`，通过 `quant.kv.method` 选用。

## 新增硬件画像、API 路由或页面

- 硬件：`data/hardware_db/<vendor>/<id>.yaml`
- REST 路由：`cpp/src/server/server.cpp`，以及 `web/src/api.ts` 中的类型化客户端
- 页面：`web/src/pages/<Name>.tsx`，并在 `web/src/main.tsx` 中注册路由与导航项
