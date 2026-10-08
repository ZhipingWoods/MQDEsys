# 导出格式与伪量化/真实量化一致性检查

**状态：部分实现。** GPTQ、AWQ、compressed-tensors 和 GGUF 导出可用且已验证；SGLang 与 TensorRT-LLM 目标为计划中。

伪量化可以评估精度，但不会让任何东西变小：权重仍以浮点形式存放在量化网格上。**导出器**把伪量化模型变成推理引擎能加载的 checkpoint。在作业 YAML 的 `export.formats` 或命令行 `--export gptq,awq,compressed-tensors` 中启用。每个配方写出 `<job-dir>/export/<recipe>-<format>/`，里面有 `model.safetensors`、`config.json`（含 `quantization_config`）、`quantize_config.json`（GPTQ/AWQ）和 tokenizer 文件。

写出之前，每个导出器都会运行 `check()`。无法表示的层会被*报告*（作业报告里 `status: unsupported`，附层名与原因），而不是被错误地导出。

## 格式

| 格式（`--export`） | 布局 | 支持的权重 | 可被加载于 |
|---|---|---|---|
| `gptq` | GPTQ v1：`qweight`、`qzeros`、`scales`、`g_idx`（`desc_act=false`） | INT 2、3、4、8 bit，per-group 或 per-channel，对称或非对称 | vLLM、SGLang、transformers（按设计） |
| `awq` | AWQ GEMM，`zero_point: true` | INT4，仅 per-group | vLLM（已验证）、SGLang、transformers（按设计） |
| `compressed-tensors` | INT4/INT8 用 `pack-quantized`（`weight_packed`、`weight_scale`、`weight_shape`、打包的零点）；FP8 E4M3 用 `float-quantized` | INT 4/8 仅权重；FP8 per-channel 或 per-tensor，激活为动态 per-token 或静态 per-tensor scale | vLLM（已验证） |
| GGUF（`--llamacpp` 类型） | 先由 llama.cpp 的转换器写出 F16，再重新编码 `blk.*.weight` 张量 | `q8_0`、`q5_0`、`q4_0`、`iq4_nl`、`q4_k`、`q2_k` | llama.cpp |

目前无法导出：compressed-tensors 中带量化激活的整数权重（W4A8、W8A8-int）、per-tensor 整数 scale、per-group FP8 scale，以及 AWQ/SmoothQuant 缩放无法折叠进前一个算子的层。GPTQ 与 AWQ checkpoint 不包含激活量化；引擎以 fp16 计算激活，作业会在备注中写明。

### 平滑系数会被折叠

AWQ 和 SmoothQuant 为读取同一输入的每组层产生一个缩放。导出器要求它已被折叠进产生该输入的算子（RMSNorm 权重、`up_proj` 的行、`v_proj` 的行），这在浮点下是精确的，推理时也没有额外的除法。在分组查询注意力模型中，`o_proj` 不做平滑。

### checkpoint 中的混合精度

逐层位宽覆盖：GPTQ 写成 GPTQModel 的 `dynamic` 配置，AWQ 写成 `modules_to_not_convert`，compressed-tensors 写成额外的 `config_groups` 加 `ignore`。保持 FP 的层（`skip: true`、`lm_head`）会如实列出。

### GGUF 细节

llama.cpp 源码目录中的 `convert_hf_to_gguf.py` 写出 F16 文件（用 `MQDE_LLAMACPP_SRC` 指定该目录）。线性层张量随后由 MQDEsys 自己的打包器量化；在 Qwen2.5-0.5B 上，六种类型（每种 290 个张量）的输出与 `llama-quantize` 逐字节比对一致。行长不能被块大小整除的张量遵循 llama.cpp 的回退规则：Q4_K 回退到 Q5_0，Q2_K 回退到 Q4_0；仍不能整除则保持 F16。embedding、输出头和 norm 保持 F16/F32。GGUF 用 llama.cpp 自己的数值对原始权重量化，与作业的配方和算法无关，所以只能通过 PPL 与各配方的结果对照。

### Marlin 打包

`mqde.core.pack_marlin` / `unpack_marlin` 实现 4 位和 8 位权重、16 位激活的 Marlin tile 布局，移植自 vLLM v0.30.0，并针对三种形状与 vLLM 的参考函数逐字节比对一致。导出器不写 Marlin checkpoint；vLLM 在加载时自己把 GPTQ/AWQ/compressed-tensors 权重重排成 Marlin（下面的 4 位 checkpoint 它选用了 `MarlinLinearKernel`）。与已安装 vLLM 的 `gptq_marlin_repack` 的逐字节比对测试已写好，但尚未执行。

## 伪量化/真实量化一致性检查（S10）

每次导出之后，流水线运行两个阶段：

1. `real_check`：用参考解包器读回 checkpoint，重建每一层的整数码和零点，与伪量化模型比对。整数码与零点必须**完全相同**（任何差异都是打包 bug）；反量化后的权重只允许因 scale 以 fp16 存储而不同（以 max-abs 与相对 L2 报告）。
2. `eval_real`：把反量化的真实权重换进模型，再测一次 PPL。

在 Qwen2.5-0.5B-Instruct、W4A16 GPTQ 上的结果（界面里的作业 20261008-101629-62da92）：AWQ 与 compressed-tensors checkpoint 各 448 MiB，所有整数码一致，权重相对 L2 为 2.0e-4，真实权重 PPL 20.656，对比伪量化的 20.655（+0.00%）。同一模型的 FP8 compressed-tensors 导出，真实权重 PPL 为 18.333，vLLM 部署后 PPL 为 18.382（+0.27%）。

```{note}
vLLM 0.30.0 的校验器会拒绝非对称 4 位 GPTQ（`bits=4, sym=False`），而 `w4a16` 配方恰好产生这种格式。这是 vLLM 的限制，不是导出 bug，文件结构本身是正确的。给 vLLM 用时请选 AWQ 或 compressed-tensors 导出，或使用对称配方。
```
