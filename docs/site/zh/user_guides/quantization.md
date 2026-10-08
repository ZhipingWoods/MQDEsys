# 量化配方与算法

**状态：已实现**，针对 LLM 作业（伪量化评估加真实导出，见 [导出格式](export.md)）。实验性配方有标注。

一个作业由**配方**（权重、激活和 KV Cache 采用什么格式）和**算法**（如何选出量化后的值）组合而成。两者都由数据驱动：配方在 `data/recipes/*.yaml`，算法在 `data/algorithms.yaml`。

## 配方

`mqde recipes` 会列出内置配方：

| 配方 | 权重 | 激活 | 默认算法 | 成熟度 |
|---|---|---|---|---|
| `bf16` | 无（基线） | 无 | none | production |
| `w8a16` | INT8 对称，per-channel，min/max | 16 位 | rtn | production |
| `fp8` | FP8 E4M3，per-channel | FP8 E4M3，动态 per-token | rtn | production（原生 kernel 需要 sm_89+ / MI300） |
| `w4a16` | INT4 非对称，per-group（128），MSE 范围搜索 | 16 位 | gptq | production |
| `w4a8` | INT4 对称，per-group（128），MSE | INT8 对称，动态 per-token | smoothquant+gptq | production |
| `w2a16` | INT2 非对称，group 64，MSE | 16 位 | gptq | experimental |
| `w2a8` | INT2 非对称，group 64，MSE | INT8 动态 per-token | gptq | experimental |

内置配方的权重位宽是 8、4、2。其他位宽：逐层的 `mixed_precision` 覆盖（见下）可以把整数层改成任意整数位宽，GPTQ 导出器接受 2、3、4、8 bit；W3、W5、W6 的配方文件没有内置，也不在已验证的运行范围内。FP8 配方同时量化权重和激活；目前没有 FP8 per-block 配方（T-0605 为部分完成）。

W4A8、W2A8 和 FP8 的激活默认是动态 per-token 量化。若要静态 per-tensor 激活 scale，把 `quant.act_static` 设为 `minmax`、`ema` 或 `percentile`（命令行 `--act-static`）。

## 算法

`GET /api/v1/algorithms` 或 `data/algorithms.yaml` 会列出它们；服务端会拒绝未知名称。

| 名称 | 类别 | 需要校准 | 作用 |
|---|---|---|---|
| `rtn` | quantizer | 否 | 就近取整，使用配方指定的范围搜索（min/max 或 MSE）。 |
| `gptq` | quantizer | 是 | 在捕获的 FP 激活上，逐层做二阶（Hessian）逐列误差补偿（`block_size 128`，`damp 0.01`）。 |
| `awq` | transform | 是 | 搜索逐输入通道的缩放（网格 20 档）使输出误差最小，然后做 RTN。 |
| `smoothquant` | transform | 是 | 把激活离群值迁移到权重（`s = max|X|^a / max|W|^(1-a)`，`alpha 0.5`）；主要用于 W4A8。 |
| `awq+gptq`、`smoothquant+gptq` | combo | 是 | 先做变换，再对缩放后的权重做 GPTQ。 |

组合规则是*至多一个变换*（`awq` 或 `smoothquant`）用 `+` 连接*至多一个量化器*（`rtn` 或 `gptq`）。`algorithm: recipe`（默认）表示使用上表中各配方的默认算法。

块按顺序量化，第 *i* 块的校准输入是已量化的第 0..i-1 块的输出。校准数据是 wikitext-2 train（`calibration.n_samples` 个长度为 `seq_len` 的窗口）。AWQ/SmoothQuant 的缩放由读取同一输入的层共享（q/k/v、gate/up），并折叠进前一个归一化层或投影层，所以模型在 FP 下数学上不变、可以导出（见 [导出](export.md)）；在分组查询注意力（GQA）模型中，`o_proj` 的输入宽度与 `v_proj` 的输出不同，所以不对它做平滑。

```{note}
这里的 AWQ 和 SmoothQuant 是各自方法的等价变换部分加上 RTN 或 GPTQ。“transformed”评估位点（变换后、量化前的困惑度）尚未实现（T-0603 部分完成），QuaRot 也未实现。GPTQModel 与 llm-compressor 的外部适配器为计划中（T-0609）。
```

## 混合精度

`quant.mixed_precision.rules` 是有序列表，第一条匹配的规则生效。规则含 `match`，以及 `bits` 或 `skip: true`：

- `match: "block:3"`、`"block:0-2"`、`"block:-1"` 选择解码块（负数从末尾数起）；
- 其他 `match` 是在模块路径上搜索的正则表达式，例如 `"mlp\\.down_proj$"`。

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

逐层覆盖会带入导出：GPTQ 写成 GPTQModel 的 `dynamic` 配置，AWQ 写成 `modules_to_not_convert`，compressed-tensors 写成额外的 `config_groups`。

**敏感度分析**（`analysis.sensitivity: block | type`）每次把一组层恢复成 FP，测量它挽回了多少 PPL 退化；报告会给各组排序，并在权重显存预算内建议能挽回 50% 退化的规则。该建议是各组单独效果的相加，偏乐观，需要用新作业验证。在界面里有按钮可以根据建议生成这个新作业。

## KV Cache 量化

`quant.kv` 量化进入注意力的 key 和 value（伪量化，所以测的是精度；引擎里真实的 KV Cache 数据类型是引擎自己的设置）：

```yaml
quant:
  kv: {format: int, bits: 8, symmetric: false, method: kivi}
```

`method: kivi` 对 key 做 per-channel、对 value 做 per-token 量化；`method: naive` 对两者使用同一种粒度（`granularity: per_token | per_head | per_channel`）。格式：带 `bits` 的 `int`，或 `fp8_e4m3`（命令行 `--kv int8`、`--kv int4`、`--kv fp8`）。在 Qwen2.5-0.5B 上，INT8 KIVI 对 PPL 的影响约为 -0.01% 到 -0.03%，即没有可测量的影响（作业 c80eda：W4A16 不带 KV 量化 20.411，带 KV 量化 20.408）；naive 的 per-token key 量化更差，因为 Qwen2 的 key 里有离群通道。

## 读懂评估位点

每个配方最多在三个位置报告 PPL：**pretrain**（原模型）、**fake_quant_wo_kv**（量化权重和激活）、**fake_quant**（再加上 KV Cache 量化）。导出会增加 **real**（读回 checkpoint 的权重），部署会增加 **served**（部署后）PPL。它们默认都使用 wikitext-2 测试集的同样 40 个 512 token 窗口，所以同一作业内的位点可以直接比较；llama.cpp 自带工具的 PPL 使用不同协议，只能在它自己的表内比较。
