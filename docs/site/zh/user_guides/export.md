# 导出格式与一致性校验

伪量化用于评估精度，但不会减小模型体积：权重仍以浮点形式存放于量化网格上。导出器将伪量化模型转换为推理引擎可直接加载的 checkpoint。可在作业 YAML 的 `export.formats` 或命令行 `--export gptq,awq,compressed-tensors` 中启用。每个配方输出至 `<job-dir>/export/<recipe>-<format>/`，包含 `model.safetensors`、`config.json`（含 `quantization_config`）、`quantize_config.json`（GPTQ/AWQ）与 tokenizer 文件。

导出前，每个导出器都会执行 `check()`。无法表示的层会在作业报告中被标记为 `status: unsupported`，并附层名与原因，而不会生成错误的 checkpoint。

## 支持的格式

| 格式（`--export`） | 布局 | 支持的权重 | 兼容引擎 |
|---|---|---|---|
| `gptq` | GPTQ v1：`qweight`、`qzeros`、`scales`、`g_idx`（`desc_act=false`） | INT 2/3/4/8 bit，per-group 或 per-channel，对称或非对称 | vLLM、SGLang、transformers |
| `awq` | AWQ GEMM，`zero_point: true` | INT4，per-group | vLLM、SGLang、transformers |
| `compressed-tensors` | INT4/INT8 使用 `pack-quantized`（`weight_packed`、`weight_scale`、`weight_shape`、打包零点）；FP8 E4M3 使用 `float-quantized` | INT 4/8 仅权重；FP8 per-channel 或 per-tensor，激活为动态 per-token 或静态 per-tensor scale | vLLM |
| GGUF | 先由 llama.cpp 转换器写出 F16，再编码 `blk.*.weight` 张量 | `q8_0`、`q5_0`、`q4_0`、`iq4_nl`、`q4_k`、`q2_k` | llama.cpp |

不受支持的组合包括：compressed-tensors 中带量化激活的整数权重（W4A8、W8A8-int）、per-tensor 整数 scale、per-group FP8 scale，以及 AWQ/SmoothQuant 缩放无法折叠进前一个算子的层。GPTQ 与 AWQ checkpoint 不包含激活量化，引擎以 FP16 计算激活，作业会在备注中说明。

## 平滑系数折叠

AWQ 与 SmoothQuant 为读取同一输入的每组层生成一个缩放系数。导出器要求该系数已折叠进产生该输入的算子（RMSNorm 权重、`up_proj` 的行、`v_proj` 的行）。该变换在浮点下精确，推理时无需额外除法。在分组查询注意力模型中，`o_proj` 不参与平滑。

## 混合精度 checkpoint

逐层位宽覆盖在导出时写入对应格式的配置：GPTQ 使用 GPTQModel 的 `dynamic` 配置，AWQ 使用 `modules_to_not_convert`，compressed-tensors 使用额外的 `config_groups` 与 `ignore`。保持浮点精度的层（`skip: true`、`lm_head`）会如实列出。

## GGUF 细节

llama.cpp 源码目录中的 `convert_hf_to_gguf.py` 写出 F16 文件（通过 `MQDE_LLAMACPP_SRC` 指定目录）。线性层张量随后由 MQDEsys 的打包器量化。在 Qwen2.5-0.5B 上，六种量化类型（每种 290 个张量）的输出与 `llama-quantize` 逐字节一致。行长不能被块大小整除的张量遵循 llama.cpp 的回退规则：Q4_K 回退到 Q5_0，Q2_K 回退到 Q4_0；仍不能整除时保持 F16。embedding、输出头与 norm 保持 F16/F32。GGUF 使用 llama.cpp 自身的数值对原始权重量化，与作业配方和算法无关，因此仅通过 PPL 与各配方结果对照。

## Marlin 打包

`mqde.core.pack_marlin` 与 `unpack_marlin` 实现 4/8 位权重、16 位激活的 Marlin tile 布局，移植自 vLLM v0.30.0，并与 vLLM 参考函数在三种形状下逐字节一致。导出器不直接写 Marlin checkpoint；vLLM 在加载时将 GPTQ/AWQ/compressed-tensors 权重重排为 Marlin 布局。项目提供与 vLLM `gptq_marlin_repack` 的逐字节比对测试。

## 一致性校验

导出完成后，流水线依次执行以下检查：

1. **真实性检查**：使用参考解包器读回 checkpoint，重建每层的整数码与零点，并与伪量化模型比对。整数码与零点必须完全一致，任何差异均视为打包错误；反量化权重仅允许因 scale 以 FP16 存储而产生差异，并以 max-abs 与相对 L2 误差报告。
2. **真实权重复测**：将反量化后的真实权重替换进模型，再次测量 PPL。

以 Qwen2.5-0.5B-Instruct、W4A16 GPTQ 为例：AWQ 与 compressed-tensors checkpoint 均为 448 MiB，全部整数码一致，权重相对 L2 误差为 2.0e-4，真实权重 PPL 为 20.656，伪量化为 20.655（+0.00%）。同一模型的 FP8 compressed-tensors 导出，真实权重 PPL 为 18.333，部署后 PPL 为 18.382（+0.27%）。

```{note}
vLLM 0.30.0 的校验器不接受非对称 4-bit GPTQ（`bits=4, sym=False`），而 `w4a16` 配方会产生该格式。这是 vLLM 的限制，文件结构本身正确。用于 vLLM 时请选择 AWQ 或 compressed-tensors 导出，或使用对称配方。
```
