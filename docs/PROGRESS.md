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
| 最后更新 | 2026-09-24 · Claude（Opus 5.5） |

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

## 会话日志（新记录加在最上面）

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
