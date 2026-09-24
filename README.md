# MQDEsys — 模型量化部署评估系统

指定模型、量化配方（W8A16 / FP8 / W4A16 / W4A8 / W2A16 / W2A8）、评估数据和目标硬件后，系统会：

1. **预估**（C++，不加载权重）：计算模型占用、KV Cache / 线性注意力状态、激活，得出单/多并发峰值显存、最大并发、延迟下界，以及“配方 × 硬件”的可部署矩阵。
2. **量化评估**（Python 执行器，由 C++ 调度到 GPU 池）：生成 BF16 基线，逐模态做误差归因，执行 fake-quant 并逐层追踪误差。
3. **可视化**（Web）：展示实时执行过程和日志、逐层误差热力图、最差层列表以及配方对比。

目前首个支持的模型是 VLA 模型 MiniCPM-RobotManip。设计与任务见 `docs/`，Agent 工作规则见 `AGENTS.md`，当前进度见 `docs/PROGRESS.md`。

## 快速开始（4090 开发机）

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mqde
cmake --preset dev && cmake --build --preset dev -j 32 && ctest --preset dev
PYTHONPATH=python python -m pytest python/tests -q

build/dev/bin/mqde probe
build/dev/bin/mqde model inspect /cache/wuzhiping/hub/MiniCPM-RobotManip
build/dev/bin/mqde estimate -m /cache/wuzhiping/hub/MiniCPM-RobotManip -r bf16,w4a16,fp8 --images 2 --conc 1,8,64

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python -m mqde.pipeline.vla_job examples/jobs/minicpm_vla_libero.yaml

(cd web && npm install --registry=https://registry.npmmirror.com && npm run build)
build/dev/bin/mqde serve --repo $PWD --runs $PWD/runs --web $PWD/web/dist \
  --python $CONDA_PREFIX/bin/python --gpus 2,3 --allow /cache/wuzhiping/hub --port 18080
# 本机：ssh -N -L 18080:127.0.0.1:18080 4090，然后打开 http://localhost:18080
```

## 目录

| 路径 | 内容 |
|------|------|
| `cpp/` | C++ 核心：common、modelspec、quant（数值唯一规范）、estimator、server，C 硬件探针，CLI，pybind11 绑定 |
| `python/mqde/` | 执行器：Torch 数值镜像、模型适配、数据、fake-quant 引擎与追踪、作业流水线 |
| `data/` | 配方、硬件画像、后端参数、模型架构模板（数据驱动，新增无需改代码） |
| `web/` | React + ECharts 前端 |
| `examples/jobs/` | 作业示例 |
| `docs/` | 架构、目录职责、任务清单、进度 |
