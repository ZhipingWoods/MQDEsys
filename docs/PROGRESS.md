# MQDEsys 进度与交接日志（PROGRESS）

> 每个 Agent **开始工作前读**、**结束会话前写**。格式见 [../AGENTS.md](../AGENTS.md) 第 10 节。
> 任务状态以 [TASKS.md](TASKS.md) 总览表为准；本文件记录“过程”和“交接上下文”。
> 文档级别：L0（Agent 自行维护，只追加，不删除他人记录）。

---

## 当前快照（每次更新时覆盖本节）

| 项目 | 内容 |
|------|------|
| 当前阶段 | **VLA 纵向切片可用**：C++ 核心 + 预估 + fake-quant 评估 + 逐层误差追踪 + API 服务 + Web UI |
| 首个目标模型 | MiniCPM-RobotManip（1.5B VLA：MiniCPM-V 4.6 视觉 + Qwen3.5 混合注意力 LM + DiT 动作头） |
| 开发机 | 4090 服务器 `ssh 4090`（内网地址已隐去），仓库 `/cache/wuzhiping/MQDEsys`，**只用 GPU 2、3** |
| 环境 | conda env `mqde`（python 3.11, torch 2.11+cu128, transformers 5.7.0, cmake/gcc12/fmt/yaml-cpp/httplib 来自 conda-forge） |
| 服务 | `mqde serve` 监听 127.0.0.1:18080，GPU 池 2,3；本机访问：`ssh -N -L 18080:127.0.0.1:18080 4090` → http://localhost:18080 |
| 下一步建议 | ① 真实部署路径（T-0405 打包 → T-0705 llama.cpp / T-0701 vLLM）让显存真正下降；② GPTQ/AWQ/SmoothQuant（T-0601~0603）改善 W4A8/W2；③ 敏感层混合精度（T-0608/T-0904，数据已指向视觉前 5 块）；④ T-0005/T-0006 CI |
| 最后更新 | 2026-09-30 · Claude（Opus 5） |

### 快速复现

```bash
ssh 4090
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mqde
cd /cache/wuzhiping/MQDEsys
cmake --preset dev && cmake --build --preset dev -j 32 && ctest --preset dev      # C++: 28 tests
PYTHONPATH=python python -m pytest python/tests -q                                   # Python: 21 tests（含 CUDA 对拍）
build/dev/bin/mqde model inspect /cache/wuzhiping/hub/MiniCPM-RobotManip
build/dev/bin/mqde estimate -m /cache/wuzhiping/hub/MiniCPM-RobotManip -r bf16,w4a16,fp8 --images 2 --conc 1,8,64
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python -m mqde.pipeline.vla_job examples/jobs/minicpm_vla_libero.yaml
# 前端（修改后）：cd web && npm install --registry=https://registry.npmmirror.com && npm run build
build/dev/bin/mqde serve --repo $PWD --runs $PWD/runs --web $PWD/web/dist \
  --python /root/miniconda3/envs/mqde/bin/python --gpus 2,3 --allow /cache/wuzhiping/hub --port 18080
```

外网：服务器直连 HF 超时。Mac 上 Clash 在 7897，开反向隧道 `ssh -f -N -R 127.0.0.1:17897:127.0.0.1:7897 4090`，
服务器上 `export https_proxy=http://127.0.0.1:17897`。npm 走 npmmirror，pip 走 aliyun（已配置）。

---

## 首批评估结果（LIBERO-10，24 个评估帧 / 16 个校准帧，互不重叠，RTX 4090）

作业 `runs/20260924-114936-7125e8`。量化范围：视觉塔 + merger + 语言模型的线性层（292 层）；
embedding/lm_head、GatedDeltaNet 的 in_proj_a/b、DiT 动作头保持原精度。算法：RTN + 配方标定（W4/W2 为 MSE 搜索）。

| 配方 | 动作 rel-L2 | ×噪声下限 | 动作 cos | VL 嵌入 cos | 仅视觉 | 仅语言 | 预估部署峰值 |
|------|-----------:|---------:|--------:|-----------:|------:|------:|-----------:|
| BF16 | — | — | — | — | — | — | 4.02 GiB（实测 3.65 GiB） |
| W8A16 | 0.017 | 0.32 | 0.9994 | 0.999 | 0.008 | 0.015 | 3.01 GiB |
| FP8 | 0.032 | 0.60 | 0.9987 | 0.994 | 0.019 | 0.018 | 3.01 GiB |
| W4A16 | 0.056 | 1.06 | 0.9972 | 0.938 | 0.029 | 0.056 | 2.52 GiB |
| W4A8 | 0.060 | 1.13 | 0.9968 | 0.918 | 0.036 | 0.062 | 2.52 GiB |
| W2A16 | 0.290 | 5.46 | 0.952 | 0.457 | 0.155 | 0.242 | 2.28 GiB |
| W2A8 | 0.293 | 5.52 | 0.951 | 0.455 | 0.157 | 0.244 | 2.28 GiB |

- **噪声下限**：同一 BF16 模型只换流匹配噪声种子，动作 rel-L2 = 0.053。W8A16/FP8 低于它；W4 与它持平；W2 是它的 5 倍以上。
- **误差来源**：W4 以上配方中语言模型贡献更大；逐层看，最差的是视觉塔前 5 块（输出 SQNR ≈ 18 dB，后面的块 ≈ 30 dB）和 `v_proj`、`linear_attn.out_proj`（激活峰度高达 10²–10³）。它们是混合精度的首选候选。
- **局限**：
  1. fake-quant 以 BF16 存储，实测显存不会下降；“预估部署峰值”来自 C++ 预估器。
  2. 与 LIBERO 真值动作的 MAE 不可比：模型输出 80 维，LIBERO 是 7 维，且缺少训练时的动作归一化统计。报告保留了该字段，UI 不据此下结论。
  3. 离线动作误差不等于任务成功率；需要在仿真器闭环（LIBERO sim）中验证。
  4. BF16 延迟 276 ms/步，未安装 flash-linear-attention / causal-conv1d 快速路径。

---

## 与原设计的偏离（需负责人确认，已登记为待决策）

| # | 偏离 | 原因 |
|---|------|------|
| X-1 | HTTP 用 **cpp-httplib**（header-only）而非 Drogon；事件流用 **SSE** 而非 WebSocket | conda-forge 可直接装，零构建成本；SSE 单向推送 + `after_seq` 续传已满足需求。接口保持 `/api/v1`，后续可替换 |
| X-2 | C++ 依赖来自 **conda-forge**（CMakePresets 用 `$CONDA_PREFIX`），尚未接 vcpkg | 服务器已有 conda，避免编译整个依赖树；vcpkg manifest 留待 CI 任务 |
| X-3 | 去掉 spdlog，自写约 50 行的 fmt 日志 | conda 的 spdlog 1.10 捆绑的 fmt 与 fmt 12 冲突 |
| X-4 | 作业 DAG 暂在 Python（`python/mqde/pipeline/vla_job.py`）；C++ 负责调度子进程、GPU 分配、状态、事件流 | 先打通纵向切片；C++ DAG + Worker RPC 仍按 T-0501/T-0503 做 |
| X-5 | C++ 标准从 C++20 升到 **C++23**（`std::expected`） | gcc 12 支持，免去 tl::expected 依赖 |
| X-6 | `quantize_int` 的 codes 用 int16 存储 | 8-bit 非对称码值范围 0..255 超出 int8；打包阶段再压缩 |

---

## 待人工决策（负责人处理后在此标注结论）

| # | 日期 | 问题 | 选项 / 建议 | 结论 |
|---|------|------|------------|------|
| D-1 | 2026-09-23 | ADR-0001：C++ 控制面 + 数值核心 / Python 执行器 | 建议采纳（已按此实现） | 待定 |
| D-2 | 2026-09-23 | 开源许可证 | Apache-2.0（与模型一致） | 待定 |
| D-3 | 2026-09-23 | 前端 UI 组件库 | 目前为原生 CSS + ECharts，无组件库；如需 Ant Design 再加 | 待定 |
| D-4 | 2026-09-23 | GPU 验收环境 | 4090 服务器 GPU 2/3（已用） | 部分解决 |
| D-5 | 2026-09-23 | W4A8 激活用 INT8 还是 FP8 | 目前 INT8 per-token 动态 | 待定 |
| D-6 | 2026-09-24 | 接受上表 X-1~X-6 偏离？ | 建议接受 X-2~X-6；X-1 可在需要双向交互时再上 WebSocket | 待定 |
| D-7 | 2026-09-24 | Git 远端 | 推到 github.com/zhipingwoods/MQDEsys？需要服务器上的推送凭证 | 待定 |
| D-8 | 2026-09-24 | VLA 精度指标 | 需要：动作反归一化统计 / 80→7 维映射，或 LIBERO 仿真闭环成功率 | 待定 |

---

## 发现的问题 / 技术债

| # | 日期 | 描述 | 影响 | 跟踪 |
|---|------|------|------|------|
| B-1 | 2026-09-24 | 服务器 git 全局身份是他人（zhangkaihuo）；仓库内已设 local user=zhipingwu | 不要改全局配置 | — |
| B-2 | 2026-09-24 | 缺 fla/causal-conv1d，Qwen3.5 线性注意力走 torch 慢路径 | 延迟偏高、影响性能测量 | T-0803 前处理 |
| B-3 | 2026-09-24 | 前端 JS bundle 1.2 MB（ECharts 全量） | 首屏慢 | 按需引入 echarts/core |
| B-4 | 2026-09-24 | 预估的激活项为经验公式；BF16 预估 4.02 GiB vs 实测 torch 峰值 3.65 GiB（未含 CUDA context） | 需 S15 回填校准 | T-0905 |
| B-5 | 2026-09-24 | server 的 JobRunner 只在内存记录进程；重启后运行中作业状态靠 events.jsonl 推断 | 重启时运行中作业会“失联” | T-0504 |

---


### 2026-09-28 · T-R20 实测基线库

- **完成**：
  - `cpp/{include,src}/fit/benchmarks.*`：读取 `data/benchmarks/local/`（本机）与 `community/`（外部，必须带 `source`）；**仅相同硬件**匹配（GPU 名 + CPU 名，不区分大小写）；同模型多条取中位数，样本数累加；名字按 llmfit `canonical_slug` 归一化；过滤不合理值（≤0 或 >10000）。
  - `mqde bench <model> --gpu N`：fork/exec `python -m mqde.eval.bench`（不经 shell），写 `local/<hw-slug>/<unix>-<hex>.json`。目前只支持 VLA（单位 steps/s）。
  - `fit` 结果新增 `measured_tps`、`formula_tps`；有实测时 `estimated_tps` 用实测值，置信度 `measured_local` / `measured_community`。前端速度列显示"实测"徽标，详情面板同时显示实测和公式值。
  - **只在评估硬件就是本机、且单卡时才套用实测**；换成其他硬件画像或多卡时保持 estimated。
- **校验**：
  - ctest 56/56（新增 11 个 benchmarks 测试）。
  - 在 GPU 3 上实测 MiniCPM-RobotManip BF16：**4.41 steps/s（226.8 ms/步，20 次）**，峰值显存 3.78 GB，上下文 157 token。`/api/v1/fit`（rtx4090）返回 `measured_local`；`/fit:plan` 中其他画像仍为 estimated；前端截图已核对。
- **定义变更**：原 TASKS 写的"三级硬件回退"在 llmfit 源码中不存在（调研误读），已按源码改为相同硬件匹配，REFERENCES.md A.8 同步重写。
- **重要发现**：公式估算 46.89 steps/s，实测 4.41，**公式高估约 10.6 倍**。原因是 VLA 的一步包含 10 次流匹配去噪 + 视觉编码，而公式只按带宽算一次权重读取。T-R19 的本地校准需要 ≥2 个稠密锚点，目前只有 1 个，所以还不会生效。VLA 速度模型需要单独修正（建议新开任务）。
- **下一步**：T-R19 置信度阶梯与本地校准。

### 2026-09-28 · T-R17 评分校准 + T-R18 MoE 两级速度模型

- **完成**：
  - T-R17：质量分档改为 llmfit 的 30/45/60/75/82/89/95（严格小于，按活跃参数）；家族/时效/任务加分；W2 惩罚 −12；embedding 用途；多核 ×1.1；TTFT/prefill 在算力未知或纯 CPU 时为 null。
  - T-R18：`moe_decomposition()`（Tier-1 输入）；GPU 模式 Tier-1 / Tier-2 + 显存压力惩罚；MoE 卸载模式 DDR + GPU 时间；`ModelFit` 新增 `moe_tier`、`moe_vram_pressure`、`moe_offloaded_gb`。常数在 `data/fit/fit.yaml` 的 `moe:` 段。
- **校验**：ctest 45/45。4090 上 MiniCPM-o：233.5 → 129.8 tok/s（Tier-1，压力 ×0.856，手算一致）；两个稠密模型不变。
- **已知偏差**：
  - llmfit 的 Tier-1 把所有层都按全注意力计参数，MiniCPM-o 实际是线性/全注意力混合层，固定项略偏大。
  - llmfit 把 embedding 和 lm_head 都记入固定项（其源码注释承认对大词表模型偏保守）。两处都与 llmfit 保持一致，未自行修正。
  - 时效加分用文件 mtime，反映的是下载时间，T-R25 接入 HF 发布日期后修正。
- **下一步**：T-R20 实测基线库。

### 2026-09-28 · 文档化 llmfit 对齐方案（无代码改动）

- **完成**：
  - `docs/REFERENCES.md` 追加**附录 A**（A.1–A.18）：llmfit 的全部公式与常数已抄录，
    含质量分档、量化惩罚、用途权重、速度四分支、MoE 两级模型、TTFT、置信度阶梯与本地校准、
    实测数据格式与三级硬件匹配、并发反解、claim 边界、运行时探测、HF 库 schema、
    下载文件选择、API 端点、ModelFit 字段全集、运行时选择、存储估算。
    **后续实现不需要再读 `3rdpart/llmfit` 源码。**
  - `docs/STRUCTURE.md` 重写：原文描述的是设想中的目录树（`api/`、`schemas/`、`scripts/` 等均不存在），
    改为当前真实文件的逐文件职责表 + 级别，并单列「计划中但尚未创建」。
  - `docs/TASKS.md` 附录 R 重写：27 个任务，新增 T-R17–T-R27 十一项，每项含产出、验收标准、校验方法。
    执行顺序：R17 → R18 → R20 → R19 → R21 → R26 → R22 → R23 → R24 → R25 → R27 → GPTQ(R06)。
  - 新增 `docs/DEVELOPMENT.md`：机器分工与 GPU 约束、rsync 同步、构建测试、起服务与隧道、
    截图、CLI、gdb 与交叉验证脚本、提交格式与自查清单。
  - `AGENTS.md` 阅读顺序加入 DEVELOPMENT.md 与 REFERENCES.md；`README.md` 加文档导航表。
- **校验**：仅文档改动，无代码变更；未跑构建。
- **用户已确认的范围**：算法校准 + 核心功能，**额外包含** claim 边界（T-R23）、
  已安装运行时探测（T-R24）、HF 模型库与搜索下载（T-R25）。
  不做：TUI、MCP server、NATS、质量评测 rubric、desktop、i18n。
- **下一步**：T-R17 评分常数校准（改 `data/fit/fit.yaml` 的质量分档为 30/45/60/75/82/89/95、
  W2 惩罚改 −12、新增 embedding 用途与家族/时效加分表；TTFT 无算力时返回 null）。
  注意这会让现有 6 个 fit 测试的部分断言失效，需同步更新并在 PR 说明新旧值。
- **坑**：
  - 我们当前的质量分档（45/55/65/72/80/87/92）是我上次凭印象写的，与 llmfit 实际值
    （30/45/60/75/82/89/95）不一致，且 llmfit 是按**活跃**参数分档。
  - 我们当前会在算力未知时也算出 TTFT 数字，llmfit 返回 null。别把假数字当结论。
  - `data/benchmarks/` 目录从未创建，所以 `measured_tps` 字段虽在但永远无数据，置信度恒为 estimated。

### 2026-09-28 · 参考架构对齐（第一批）
- **完成**：
  - T-R01/R02 llmfit 式 `fit`（C++，6 个单测）+ `/fit`、`/fit:plan` + 部署预评估页。
  - T-R04 netron 式 `models:graph` / `models:tensor` + 结构页。
  - T-R16 深色控制台 UI。
  - modelspec 修复：分片 index 迭代器悬空导致 Qwen3.8 / MiniCPM-o 崩溃；支持 MoE 活跃参数、预量化 checkpoint 的逻辑参数。
- **校验**：
  - ctest 34/34。
  - `mqde fit /cache/wuzhiping/hub --hw nvidia/rtx4090` 对 3 个本地模型给出结论。
  - 前端 build 通过，截图人工核对。
- **下一步**（按顺序）：T-R06→R07→R08（GPTQModel 式量化流程），T-R11→R12→R13（evalscope 式评测），T-R14→R15（FlagTree 式多硬件）。
- **注意**：GPU 2 被他人进程占用约 21 GB，mqde serve 现只用 GPU 3（`--gpus 3`）。
## 会话日志（新记录加在最上面）

### 2026-09-30 · 前端 LLM 化 + AWQ/SmoothQuant 折叠 + 代码门禁（T-0005/0603/0705/0807）
- **本次完成**：
  - 前端：配方对比页重写为 LLM / VLA 双页签（LLM 为 PPL × 量化层平均位宽散点 + 明细表 + llama.cpp 部署表）；LLM 作业页新增 llama.cpp 部署卡片（解码速度柱 + PPL 折线）与敏感度卡片；作业列表显示作业类型与 CPU/GPU 设备；资源预估、模型结构页支持 `hf:org/name`（服务端对目录外模型回退 `inspect_remote`）；新建作业页可勾选 GGUF 部署类型与敏感度分析。
  - `llm_job` 新增 `deploy_llamacpp` 阶段：F16 转换 → 我们的块量化重编码 → `llama-perplexity` + `llama-bench`，结果进 report。
  - AWQ / SmoothQuant 改为**按输入分组搜索一个平滑系数并折叠进产生该输入的算子**（input_layernorm / post_attention_layernorm / up_proj 行 / v_proj 行）：FP 下等价、推理时无额外除法、可直接导出；GQA 模型 v_proj 行数 ≠ o_proj 输入维，o_proj 不折叠（trace 里写明原因）。`restore()` 会逐字节还原被改写的 norm/权重。
  - T-0005 代码门禁：`scripts/check_all.sh`（deps / clang-format / 构建 / ctest / clang-tidy / ruff / mypy / pytest / 前端构建，末尾汇总表，SKIP 与 FAIL 区分）、`scripts/check_deps.py`（按 STRUCTURE.md 的模块依赖方向检查 `#include <mqde/...>`）、`.clang-format`、`.clang-tidy`、`.editorconfig`、`.pre-commit-config.yaml`、`python/pyproject.toml`（ruff + mypy + pytest 配置）、`docs/guides/coding-standards.md`。
- **校验结果**（实测）：
  - `scripts/check_all.sh` 全绿：deps 55 文件 0 违规、clang-format 通过、ctest 98/98、ruff 通过、mypy `31 source files` 无问题、pytest 66 passed、前端构建通过；clang-tidy 为 SKIP（conda clang-tools 23 与系统 clang 14 的 builtin 头不兼容，脚本按“仓库外文件报错=环境问题”判定为 SKIP 而非 FAIL）。
  - 门禁有效性：故意加入反向依赖 + 格式错误 + 无注解函数，分别被 check_deps（`cpp/src/quant/metrics.cpp:2: 'quant' must not depend on 'server'`）、clang-format（`metrics.cpp:84:33`）、mypy（`ppl.py:37 no-untyped-def`）指出并使门禁失败。
  - 为此次门禁做的实质修复：ruff 116 → 0（含 `zip(..., strict=True)` 7 处、percentile 下界 bug 之外的 `E741`/长行/未用 noqa）；mypy 51 → 0（`FakeQuantLinear` 的 buffer 类型声明、`materialize` 返回 Optional zero、GPTQ 组参数断言、进度回调改为返回 None 的具名函数等）；clang-format 一次性格式化 51 个文件（1354+/833-，纯格式，ctest 仍 98/98）；删除 `quant_spec.cpp` 里 `? 8 : 8` 的死分支。
  - 全功能 LLM 作业（API 提交，CPU）：W4A16/W4A8 GPTQ + KV int8 + 按层类型敏感度 + GPTQ/AWQ 导出 + llama.cpp(q8_0,q4_k) 一次跑通；W4A16 PPL 20.411（导出后真实权重 20.412），敏感度最差层类型 `mlp.down_proj`/`up_proj`/`gate_proj`，建议升 8 bit 的是 `self_attn.v_proj`+`mlp.down_proj`；llama.cpp：F16 13.890（27.9 tok/s 解码）、我们的 Q8_0 13.935（41.6）、Q4_K 15.741（54.2）。
- **未完成 / 下一步**：clang-tidy 需要一致的 clang 工具链（装 conda `clangxx` 或系统 clang-19+）才能真正跑；T-0006 CI 可直接调用 `scripts/check_all.sh`。
- **折叠后的 AWQ/SmoothQuant 实测结果**（Qwen2.5-0.5B W4A16，原模型 18.102，对照未折叠：AWQ 20.591、AWQ+GPTQ 20.475）：

  | 配置 | PPL（W+A）| PPL（+KV KIVI）| 折叠层数 | 导出码一致 | 量化耗时 |
  |---|---|---|---|---|---|
  | AWQ W4A16 | 21.405 | 21.402 | 144/168 | ✓ | 1267 s |
  | AWQ+GPTQ W4A16 | 20.764 | 20.761 | 143/168 | ✓ | 1608 s |
  | SmoothQuant+GPTQ W4A16 | 21.296 | 21.308 | 144/168 | ✓ | 218 s |
  | AWQ W4A8 | 23.442 | 23.403 | 136/168 | ✓ | 478 s |
  | AWQ+GPTQ W4A8 | 21.894 | 21.894 | 136/168 | ✓ | 511 s |
  | SmoothQuant+GPTQ W4A8 | 24.088 | 24.062 | 144/168 | ✓ | 218 s |

  折叠后 PPL 略高于 per-layer 独立搜索（AWQ W4A16 +0.81 PPL）：Qwen2 的 key-channel 离群值使得各层的最优缩放系数相差较大，共享一个系数是近似，AWQ+GPTQ 差距缩小到 +0.29 PPL（GPTQ 的 H-inverse 误差补偿同时修正了缩放后的残差）。SmoothQuant W4A16 介于两者之间。所有情形下整数码与零点逐层 0 处不一致（real_check 通过），可以直接用于 GPTQ/AWQ 导出。


### 2026-09-29 · 敏感度分析 + 混合精度建议（T-0902/T-0904）+ Hub 远程检查（T-0204）
- **本次完成**：
  - `mqde/trace/sensitivity.py`：全量化后逐组（按 block 或按层类型）单独恢复 FP，测 PPL 挽回量并排序；`suggest()` 按“挽回量/额外字节”贪心选组提升到 8 bit，直到覆盖目标比例（默认 50%）或超出显存预算，输出 `mixed_precision.rules` 与权重增量。`llm_job --sensitivity block|type`（作业 YAML `analysis.sensitivity`），前端作业页显示敏感度柱状图（建议组高亮）+ “按建议生成新作业”一键克隆提交；新建作业页可选敏感度分析。
  - Hub（T-0204）：`org/name@revision`（`parse_repo_ref`，拒绝 `..`/空段）、`HF_ENDPOINT` 镜像、GET 重试（网络错误/429/5xx，3 次退避）、safetensors 头部 HTTP Range 读取（`remote_safetensors_header`）与 `inspect_remote`；CLI `mqde model inspect|estimate` 接受 `hf:org/name[@rev]`。modelspec 抽出 `parse_safetensors_header` 供本地与远程共用。
- **校验结果**（实测）：
  - `ctest` 98/98（新增 `hub_remote.*` 5 项：本地 httplib 桩，4 MiB payload 场景只传 8+header 字节；revision；重试 3 次后成功）；`pytest python/tests/trace` 4 passed。
  - 在线：`mqde model inspect hf:Qwen/Qwen2.5-7B-Instruct` → 339 个张量、7.616 B 参数，仅取 37.95 KiB（config + 4 个分片头），经代理 60 s；`hf:meta-llama/Llama-3.1-8B` → `HTTP 401 (gated or private repo; set HF_TOKEN)`；`mqde estimate -m hf:Qwen/Qwen2.5-7B-Instruct -r w4a16,fp8` 可直接出预估。
  - Qwen2.5-0.5B W4A16 RTN 按块敏感度（8 窗口 PPL，CPU 16 线程 204 s）：最敏感 block 23 / 16 / 17 / 8 / 0 / 2；建议 6 块升 8 bit（+44.7 MB）。按建议经 API 提交的新作业实测：8 窗口 PPL 24.036 → 22.014（预测 21.748，预测偏乐观：实际挽回 2.02 / 预测 2.29）；40 窗口 PPL 22.064 → **20.271**（优于整模型 GPTQ W4A16 的 20.550）。W4A8 RTN 最敏感 block 2 / 23 / 8。
  - 后台 sweep 最终结果（40 窗口 PPL，W+A / +KV int8 KIVI，原模型 18.102）：GPTQ W4A8 22.098 / 22.109；GPTQ W2A16 2142.6（崩溃）；AWQ+GPTQ W4A16 20.475 / 20.478、W4A8 22.269 / 22.231、W2A16 10390（崩溃）。
  - 静态激活 GPTQ W4A8（per-tensor minmax）：**116.26**（动态 per-token 为 22.10）——Qwen2 激活离群值使单一静态 scale 不可用，静态激活需配合 SmoothQuant 或 percentile。
  - W2A16 GPTQ + 混合精度（block 0、23 为 8 bit，down_proj 4 bit）：2142.6 → **94.38**，仍远差于 W4。
- **未完成 / 下一步**：7B 单卡 < 1 h 的 T-0902 验收待 GPU；T-0904 的 Playwright 用例未写；Hub 离线模式（读 `~/.cache/huggingface/hub`）未做。


### 2026-09-29 · LLM 路径 + 真实导出（T-0508/0510/0604/0607/0608/0405/0701/0702/0705）
- **状态**：见 TASKS.md（T-0508/0604/0607/0608/0702 DONE；T-0405/0510/0701/0705 PARTIAL）
- **本次完成**：
  - `mqde.pipeline.llm_job`：HF CausalLM 的逐块顺序量化（`quant/blockwise.py`，块 i 用已量化 0..i-1 的输出校准）→ wikitext-2 PPL 三位点（pretrain / fake_quant_wo_kv / fake_quant）→ 可选导出 + 一致性检查。server 通过 `model.type: llm` 调度，`deploy.device: cpu` 不占 GPU 池；前端新建作业页/作业页支持 LLM。
  - 静态激活（minmax / ema / percentile，per-tensor）；修复 percentile 下界取成全局最小值的 bug。
  - KV 量化 `quant/kv.py`（naive / KIVI），经 HF attention 接口包装；**修复**：自定义 attention 名必须同时注册到 `ALL_MASK_ATTENTION_FUNCTIONS` 的全局表，否则 eager 路径没有因果 mask（sdpa 恰好推断成 causal 才没暴露）。
  - 混合精度 `MixedPrecisionPolicy`：正则 / `block:i`、`block:lo-hi`、负索引；`skip` 保持 FP。
  - C++ `quant/pack.h`：连续位流（GPTQ 2/3/4/8、AWQ 交错序）、GPTQ v1/v2、AWQ GEMM、GGUF Q8_0/Q4_0/Q4_K/Q2_K（ggml `*_ref` 逐行移植）+ fp16 RNE；pybind 暴露。
  - `mqde.export`：GPTQ / AWQ checkpoint（混合精度写 GPTQModel `dynamic`），`real_check`（读回解包比对整数码）、`use_real_weights`；`export/gguf.py`（llama.cpp 转换器出 F16 → 我们的块量化重编码 blk.* 线性层）；`backend/llamacpp.py`（llama-perplexity / llama-bench）。llama.cpp CPU 工具在仓库外源码树上 out-of-tree 构建到 `build/llamacpp`（未改动源码树与其原有 build 目录）。
- **校验结果**（均为本次实测）：
  - `ctest --test-dir build/dev` → 93/93；`pytest python/tests` → 57 passed（含 `test_pack_compat.py` 14 项、`tests/export/*` 14 项）。
  - GGUF：4 种块格式与 ggml `quantize_row_*_ref`/`dequantize_row_*` **逐字节一致**（ctypes 调 libggml-base，含全零/常数/离群/全正块）；gguf-py 可读且反量化一致；Qwen2.5-0.5B 上与 `llama-quantize` 输出对比，类型相同的 blk.* 张量全部逐字节一致（Q4_0 168/168、Q8_0 168/168、Q4_K 21、Q2_K 120；其余为 llama-quantize 选了不同类型/回退 Q5_0）。
  - AWQ 打包与 GPTQModel `pack_awq` 逐 word 一致；GPTQ 与经典公式一致；3bit 与 10-1-10-1-10 方案一致。
  - Qwen2.5-0.5B-Instruct（CPU，校准 wikitext-2 train 32×512，评估 test 40×512 不重叠窗口，原模型 PPL 18.102）：

    | 配方 | 算法 | PPL（W+A） | PPL（+KV int8 KIVI） | 相对原模型 | 量化耗时 |
    |---|---|---|---|---|---|
    | W4A16 | RTN | 22.064 | 22.064 | +21.9% | 53 s |
    | W4A16 | AWQ | 20.591 | 20.612 | +13.9% | 826 s |
    | W4A16 | GPTQ | 20.550 / 20.387* | 20.548 / 20.392* | +13.5% / +12.6%* | 220 s |
    | W4A8 | RTN | 24.432 | 24.453 | +35.1% | 120 s |
    | W4A8 | AWQ | 23.376 | 23.329 | +28.9% | 848 s |
    | W2A16 | RTN | 13791.8 | 13839.1 | 崩溃 | 38 s |
    | W2A16 | AWQ | 5269.8 | 5290.1 | 崩溃 | 792 s |
    | W8A16 | GPTQ | 18.145 | 18.145 | +0.24% | 180 s |

    \* 同配置两次运行：32 线程 vs 16 线程。GPTQ 的 Cholesky/GEMM 结果随 BLAS 线程数在末位不同，经顺序误差反馈放大（同一线程数下可复现，已实测）；报告现记录 `env.torch_threads`。
  - KV（FP 模型，20 窗口）：INT8 KIVI −0.03%、INT8 naive per-token +3.06%（Qwen2 key 离群通道）、INT4 KIVI +6.3%、FP8 per-token +5.4%（与 torch float8_e4m3fn 参考逐元素一致，属格式本身）→ LLM 作业默认 KIVI。
  - 真实导出（W4A16 GPTQ、W8A16 GPTQ）：GPTQ/AWQ checkpoint 读回后整数码/零点 **0 处不一致**；fp16 scale 使权重 rel-L2 2.0e-4；真实权重 PPL 20.3867 vs 伪量化 20.3866（+0.0005%），W8A16 18.1442 vs 18.1445。W4A16 GPTQ checkpoint 471 MB。
  - llama.cpp CPU（24 线程，ctx 512，40 chunks，llama-perplexity 协议：只计每块后半，数值与上表不可直接比）：F16 13.890（988 MB，tg 18.5 tok/s）；我们的 Q8_0 13.935（653 MB，28.4）；Q4_0 15.843（474 MB，47.0）；Q4_K 15.741（24 个 Q4_K + 144 个 Q4_0 回退，47.0/46.1）；Q2_K 27.047；对照 llama-quantize Q4_K_S 混合 14.570（380 MB，61.5）。
- **未完成 / 下一步**：
  1. vLLM 加载 smoke（GPU 2/3 仍被他人进程占用，且环境无 vLLM）→ T-0701 保持 PARTIAL；compressed-tensors（FP8/W8A8）导出未做。
  2. Marlin 布局；GGUF 的 Q5_0/IQ4_NL 回退（目前回退 Q4_0，896 维模型 Q4_K 实际大多是 Q4_0）；llama-server 部署适配（T-0705）。
  3. AWQ/SmoothQuant 的激活缩放按层独立搜索，导出前需按共享输入分组并折叠进前一 norm/linear（exporter 现会拒绝并说明）。
  4. 后台仍在跑：GPTQ W4A8/W2A16、AWQ+GPTQ 三配方、GPTQ W4A8 静态激活、W2A16 混合精度（`/cache/wuzhiping/llm_sweep*`），结果出来后补表。
- **发现的问题 / 风险**：GPTQ 结果依赖线程数（~1% PPL）；W2 在 0.5B 模型上 RTN/AWQ 均崩溃，需 GPTQ + 混合精度。


### 2026-09-29 · 前端修复 + 可选量化算法（T-0601/0602/0603/0610）
- **状态**：DONE（b45a46c 前端；b2248e4 算法）
- **前端修复（用户反馈）**：
  - 构建曾失败（Catalog.tsx 类型错误），服务一直在发旧 bundle；已修复。
  - 部署预评估页新增“并发能力”整卡：4 个指标 + 两张图（显存堆叠 vs 路数、每路最大上下文 vs 路数）。后端 `/fit:concurrency` 新增 `curve[]`。
  - HF 搜索加“查询”按钮 + 回车（原来只有防抖）。服务端启动需带 `https_proxy=http://127.0.0.1:17897`（反向隧道到 Mac Clash），否则搜索超时。
  - 模型目录页重写：服务端过滤，提供者/能力下拉从全目录生成，点击行跳到 `/fit?source=catalog&search=`。
  - 下拉 option 颜色跟随主题 token。
- **量化算法**：`python/mqde/quant/algorithms.py`（ALGO_REGISTRY）：RTN、GPTQ、AWQ、SmoothQuant、awq+gptq、smoothquant+gptq。
  作业字段 `quant.algorithm`（默认 recipe = 各配方 default_algorithm）；`GET /api/v1/algorithms`；新建作业页有下拉。
- **校验**：pytest 20/20（新增 8 个算法测试），ctest 86/86；截图核对 fit/new/catalog。
  真实模型逐层对比（MiniCPM-RobotManip，LIBERO 4 帧校准，12 层：语言块 0-1 + 视觉块 0，CPU），平均输出 SQNR dB：

  | 配方 | RTN | AWQ | GPTQ | SmoothQuant | AWQ+GPTQ |
  |---|---:|---:|---:|---:|---:|
  | W4A16 | 22.1 | 24.4 | 36.5 | 23.3 | 38.7 |
  | W4A8 | 20.6 | 22.5 | 29.8 | 21.2 | 33.4 |
  | W2A16 | 11.4 | 13.6 | 22.9 | 12.6 | 25.5 |

- **局限 / 下一步**：
  1. GPTQ 是逐层（输入取 FP 模型的激活），不是 GPTQModel 的逐块顺序传播；下一步做 blockwise 顺序版（T-0508）。
  2. 端到端动作误差（LIBERO 全作业）**未在 GPU 上跑**：GPU 2、3 被其他进程占用（各约 21 GB）。空出后运行 `python -m mqde.pipeline.vla_job examples/jobs/minicpm_vla_libero.yaml --algorithm gptq`。
  3. SmoothQuant 的 transformed 评测位点、AWQ scale 吸收到上游 LayerNorm 尚未做（现为 X/s 在线缩放，精度等价但不省算）。
  4. 下一批：T-0604 静态激活、T-0607 KV 量化、T-0608 混合精度、T-0405 位打包 → T-0701 vLLM 导出。

### 2026-09-28 · T-R21 并发反解

- **状态**：DONE（待 commit sha）
- **本次完成**：
  1. `cpp/include/mqde/fit/concurrency.h`：`ConcurrencyEstimate` 结构体（requested_users / max_users / context / weights_gb / kv_per_user_gb / overhead_gb / total_gb / capacity_gb / utilization_pct / fits / high_pressure / max_context_for_users）；`estimate_concurrency()` 声明；`to_json()` 声明。
  2. `cpp/src/fit/concurrency.cpp`：基于 `estimator::estimate()` 的前向推算（单用户提取 KV 代价）；`max_users` 解析式求解；`max_context_for_users` 通过 2 的幂步进 + `estimate()` 正向验证（解决了近似求解导致 forward-check 失败的 bug）。
  3. `cpp/CMakeLists.txt`：新增 `concurrency.cpp` + `concurrency_test.cpp`。
  4. `cpp/src/server/server.cpp`：`POST /api/v1/fit:concurrency` 端点；接受 model/hardware/recipe/users/context。
  5. `web/src/api.ts`：`ConcurrencyEstimate` 类型；`api.concurrency()` 方法。
  6. `web/src/pages/Fit.tsx`：`ConcurrencyPanel` 组件（自含 `useAsync`，嵌入详情面板下方）；展示最大并发、最大上下文、利用率、显存构成；高压橙色徽标。
- **校验结果**：
  - `ctest --preset dev` → 69/69 通过（新增 6 个 concurrency 测试）。
  - Web build（tsc + vite）→ 通过。
  - `POST /api/v1/fit:concurrency`（MiniCPM-RobotManip, BF16, 1 用户, 4096 tok）→ fits=true, max_users=268, max_ctx_for=131072, util=19.3%。
- **技术说明**：`max_context_for_users` 最初用近似公式求解，但 `estimate()` 在较长上下文时激活字节非线性增长，导致正向验证失败。改为对 2 的幂从大到小遍历并 forward-validate，确保结果与 `estimate()` 完全一致。
- **下一步**：T-R26 ModelFit 字段补全与运行时选择（依赖 T-R18/T-R20/T-R24/T-R25，其中 T-R24/T-R25 尚未做，按顺序应先做 T-R22）。实际执行顺序 T-R21 → T-R26 → T-R22 → T-R23 → T-R24 → T-R25 → T-R27。


### 2026-09-28 · T-R19 置信度阶梯与本地校准

- **状态**：DONE（d9f51d8）
- **本次完成**：
  1. `Confidence` 枚举扩展为五级阶梯：`kMeasuredLocal > kMeasuredCommunity > kCalibrated > kEstimated > kUnsupported`。
  2. `EstimateBasis` 结构体加入 `ModelFit`：记录 `method`（formula/calibrated/measured_local/measured_community）、`gpu_bandwidth_gbps`、`ddr_bandwidth_gbps`、`efficiency`、`assumed_context`、`local_calibration`（校准系数，默认 1.0）。
  3. `apply_local_calibration(fits)`：锚点 = 稠密、≥1B、有本地实测；按 `speed_unit` 分组各自校准（tok/s 和 steps/s 不混用）；< 2 个锚点时 no-op；中位数比值截断 [0.05, 3.0]；幂等。
  4. `to_json(ModelFit)` 导出 `estimate_basis` 对象。
  5. `web/src/api.ts`：`ModelFit` 加 `estimate_basis` 类型。
  6. `Fit.tsx`：详情面板展示估算依据（方法、带宽、效率、上下文、校准系数）；置信度徽标加 "本机校准"（accent 色）。
  7. 7 个新测试（`fit_calibration.*`）：< 2 锚点 noop、双锚点校准、截断上下界、幂等性、MoE 排除、跨 unit 隔离、`estimate_basis` 字段非空。
- **校验结果**：
  - `ctest --preset dev` → 63/63 通过（较 T-R20 新增 7 个）。
  - Web build（tsc + vite）→ 通过（修复了 CONF_LABEL 重复键）。
- **VLA 注意**：steps/s 目前只有 1 个本地锚点（4.41 steps/s），< 2 个，校准 no-op。这是正确行为；等 ADR-0001 批准、快路径安装后，重跑 `mqde bench` 增加锚点。
- **下一步**：T-R21 并发反解（`max_users` / `max_context_for`）。


### 2026-09-28 · T-R20 后续 + ADR-0001（VLA 上限标注 + flash-linear-attention 提案）

- **状态**：DONE（8f07c25）
- **本次完成**：
  1. `ModelFit.speed_is_upper_bound`：在 VLA 分支（`formula_tps > 0`）置 true；有实测值时清除。
  2. `fit.cpp` VLA notes：有上限标志时追加一条说明（公式 46.9 vs 实测 4.41 steps/s，eager 模式内核发射开销）。
  3. `to_json(ModelFit)` 导出 `speed_is_upper_bound` 字段。
  4. `web/src/api.ts`：`ModelFit` 类型增加 `speed_is_upper_bound: boolean`。
  5. `Fit.tsx`：速度列加 "≤" 前缀 + 橙色 "上限" 徽标（含 hover tooltip）；详情面板速度条目加内联说明与 `mqde bench` 指引；跨硬件规划表同步 "≤" 前缀。
  6. `styles.css`：新增 `.badge.warn`（warning hue，16% fill）。
  7. `docs/adr/0001-flash-linear-attention.md`：ADR 提案（状态 Proposed），待负责人批准后安装 FLA + causal-conv1d。
  8. `docs/STRUCTURE.md`：§7 docs 表新增 `adr/NNNN-*.md` 行；从"计划中未创建"列表移除 `docs/adr/`。
- **校验结果**：
  - `ctest --preset dev` → 56/56 通过。
  - Web build（tsc + vite）→ 通过，无类型错误。
  - `/api/v1/fit` POST → VLA 模型：`speed_is_upper_bound=false`（实测替换公式，标志清除）；
    `formula_tps=46.89`，`estimated_tps=4.41`，`confidence=measured_local`。
  - 其他稠密模型：`speed_is_upper_bound=false`（正确，从未置 true）。
- **下一步**：T-R19 置信度阶梯与本地校准。
  - ADR-0001 等负责人批准；批准后去 `pyproject.toml`（L3）加 `vla-fast` extra，
    重跑 `mqde bench`，逐模块剖析验证快路径效果。
- **注意**：
  - `speed_is_upper_bound=true` 只在"有公式但无实测"时展示；本机已有 4.41 的实测，
    所以目前在 VLA 卡片上**看不到**橙色上限徽标——这是正确行为。
    换一台没有实测的机器或硬件画像，才会触发橙色标注。
  - 当前 STRUCTURE.md 变更随本次 commit 一起提交（8f07c25 仅含 C++/ADR/Web；
    STRUCTURE/PROGRESS 更新在下一个 commit）。


### 2026-09-24 · Claude（Opus 5.5） · VLA 纵向切片
- **状态**：多任务 DONE / PARTIAL（见 TASKS.md）；git 提交在服务器仓库
- **本次完成**：
  - 服务器环境：conda env `mqde`，未改系统/全局配置；只用 GPU 2/3。
  - C++：common（Result/日志/配置）、modelspec（safetensors 头 + 数据驱动架构模板，识别 MiniCPM-VLA 的视觉/merger/语言/动作头与混合注意力）、quant（INT2–8/FP8 fake 与 real 路径共用 qparams，MSE 标定，误差度量）、estimator（权重+元数据、KV、线性注意力状态、激活、并发上限、屋顶线延迟下界）、C 探针（CPU + dlopen NVML）、CLI、pybind11 `_core`、REST+SSE 服务与 GPU 作业调度。
  - Python：Torch 数值实现与 C++ 逐位一致（CPU+CUDA 测试）；VLA 适配器；LeRobot v3 数据加载（校准/评估互斥划分）；fake-quant 引擎 + 逐层追踪（真实校准激活）；分模态误差归因；作业流水线与事件流。
  - Web：资源预估（显存构成、并发/图像曲线、组件占用、可部署矩阵）、新建作业、执行过程（实时阶段+日志）、误差分析（逐层热力图、块曲线、峰度散点、Top-15）、配方对比、硬件。
  - 数据：LIBERO-10 两个分片下载到 `/cache/wuzhiping/hub/datasets/lerobot_libero_10_image`。
- **校验结果**：`ctest` 28/28 通过；`pytest` 21/21 通过（CPU+CUDA 对拍）；LIBERO 全配方作业成功（约 4 分钟）；通过 API 提交的作业在 GPU 2 上成功；各页面用 headless Chrome 截图检查。
- **未完成 / 下一步**：见“当前快照 · 下一步建议”与 D-6~D-8。
- **需要人工决策**：D-1~D-8。

### 2026-09-23 · 规划 Agent · 系统级规划
- **状态**：DONE（文档）
- **本次完成**：产出 `AGENTS.md`、`docs/ARCHITECTURE.md`、`docs/STRUCTURE.md`、`docs/TASKS.md`、`docs/PROGRESS.md`、`CLAUDE.md`。
