# 量化配方与算法

一个作业由**配方**（定义权重、激活与 KV Cache 的数据格式）与**算法**（定义量化值的选取方式）组合而成。两者均由数据驱动：配方位于 `data/recipes/*.yaml`，算法位于 `data/algorithms.yaml`。

## 配方

`mqde recipes` 列出内置配方：

| 配方 | 权重 | 激活 | 默认算法 |
|---|---|---|---|
| `bf16` | 无（基线） | 无 | none |
| `w8a16` | INT8 对称，per-channel，min/max | 16 位 | rtn |
| `fp8` | FP8 E4M3，per-channel | FP8 E4M3，动态 per-token | rtn |
| `w4a16` | INT4 非对称，per-group（128），MSE 范围搜索 | 16 位 | gptq |
| `w4a8` | INT4 对称，per-group（128），MSE | INT8 对称，动态 per-token | smoothquant+gptq |
| `w2a16` | INT2 非对称，group 64，MSE | 16 位 | gptq |
| `w2a8` | INT2 非对称，group 64，MSE | INT8 动态 per-token | gptq |

内置配方的权重位宽为 8、4、2 bit。通过逐层 `mixed_precision` 覆盖，可将整数层的位宽调整为任意整数；GPTQ 导出器支持 2、3、4、8 bit。FP8 配方同时量化权重与激活。FP8 原生 kernel 需要 sm_89 及以上或 MI300 架构。

W4A8、W2A8 与 FP8 的激活默认采用动态 per-token 量化。若需静态 per-tensor 激活 scale，可将 `quant.act_static` 设为 `minmax`、`ema` 或 `percentile`（命令行 `--act-static`）。

## 算法

`GET /api/v1/algorithms` 或 `data/algorithms.yaml` 列出全部算法；服务端拒绝未知名称。

| 名称 | 类别 | 需要校准 | 说明 |
|---|---|---|---|
| `rtn` | quantizer | 否 | 就近取整，使用配方指定的范围搜索（min/max 或 MSE）。 |
| `gptq` | quantizer | 是 | 在捕获的 FP 激活上执行逐层二阶（Hessian）逐列误差补偿（`block_size 128`，`damp 0.01`）。 |
| `hqq` | quantizer | 否 | Half-Quadratic Quantization，对 scale/zero 做无数据近端优化，适用于无校准数据的 2-bit 场景。 |
| `awq` | transform | 是 | 搜索逐输入通道缩放（20 档网格），最小化输出误差，随后执行 RTN。 |
| `smoothquant` | transform | 是 | 将激活离群值迁移到权重（`s = max|X|^a / max|W|^(1-a)`，`alpha 0.5`），主要用于 W4A8。 |
| `awq+gptq`、`smoothquant+gptq`、`awq+hqq` | combo | 是 | 先执行变换，再对缩放后的权重执行对应量化器。 |

组合规则为：至多一个变换（`awq` 或 `smoothquant`）通过 `+` 连接至多一个量化器（`rtn`、`gptq` 或 `hqq`）。`algorithm: recipe`（默认）表示使用配方定义的默认算法。

块按顺序量化，第 *i* 块的校准输入为已量化的第 0..i-1 块的输出。校准数据取自 wikitext-2 train（`calibration.n_samples` 个长度为 `seq_len` 的窗口）。AWQ/SmoothQuant 的缩放由读取同一输入的层共享（q/k/v、gate/up），并折叠进前一个归一化层或投影层，因此模型在浮点下数学等价且可导出（见[导出](export.md)）。在分组查询注意力（GQA）模型中，`o_proj` 的输入宽度与 `v_proj` 的输出不同，因此不参与平滑。

## 混合精度

`quant.mixed_precision.rules` 为有序列表，第一条匹配的规则生效。规则包含 `match` 以及 `bits` 或 `skip: true`：

- `match: "block:3"`、`"block:0-2"`、`"block:-1"` 选择解码块（负数从末尾计数）；
- 其他 `match` 为模块路径上的正则表达式，例如 `"mlp\\.down_proj$"`。

```yaml
quant:
  recipe: [w4a16]
  algorithm: gptq
  mixed_precision:
    rules:
      - {match: "block:0", bits: 8}
      - {match: "mlp\\.down_proj$", bits: 8}
      - {match: "lm_head", skip: true}
```

逐层覆盖会带入导出：GPTQ 写入 GPTQModel 的 `dynamic` 配置，AWQ 写入 `modules_to_not_convert`，compressed-tensors 写入额外的 `config_groups`。

**敏感度分析**（`analysis.sensitivity: block | type`）逐组将层恢复为浮点精度，测量其可挽回的 PPL 退化。报告对各组排序，并在权重显存预算内给出可挽回 50% 退化的规则建议。该建议按各组独立效果相加，结果偏乐观，需通过新作业验证。Web 界面可依据建议直接生成验证作业。

## KV Cache 量化

`quant.kv` 对进入注意力的 key 与 value 执行伪量化，用于评估精度；推理引擎中实际的 KV Cache 数据类型由引擎配置决定。

```yaml
quant:
  kv: {format: int, bits: 8, symmetric: false, method: kivi}
```

`method: kivi` 对 key 做 per-channel、对 value 做 per-token 量化；`method: naive` 对两者使用相同粒度（`granularity: per_token | per_head | per_channel`）。格式支持带 `bits` 的 `int` 或 `fp8_e4m3`（命令行 `--kv int8`、`--kv int4`、`--kv fp8`）。在 Qwen2.5-0.5B 上，INT8 KIVI 对 PPL 的影响约为 -0.01% 至 -0.03%（W4A16 不带 KV 量化为 20.411，带 KV 量化为 20.408）；naive per-token key 量化效果较差，原因是 Qwen2 的 key 中存在离群通道。

## 评估位点

每个配方最多在三个位置报告 PPL：**pretrain**（原模型）、**fake_quant_wo_kv**（量化权重与激活）、**fake_quant**（在上一项基础上加入 KV Cache 量化）。导出后增加 **real**（读回 checkpoint 的权重），部署后增加 **served**（部署后模型）。各评估位点默认使用 wikitext-2 测试集的同一组 40 个 512-token 窗口，因此同一作业内的结果可直接比较；llama.cpp 自带工具的 PPL 使用不同协议，只能在其自身结果内比较。
