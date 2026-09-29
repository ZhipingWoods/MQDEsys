# 快速上手（T-0515）

从零到一次完整的“预估 → 量化 → 评估 → 真实导出 → 一致性检查 → 部署评测”。所有命令都在 4090 开发机上验证过（2026-09-29）。

## 0. 环境

```bash
ssh 4090
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mqde
cd /cache/wuzhiping/MQDEsys
cmake --preset dev && cmake --build build/dev -j 32 && ctest --test-dir build/dev      # C++
PYTHONPATH=python python -m pytest python/tests -q                                     # Python
cd web && npm run build && cd ..                                                       # 前端
```

访问 Hugging Face 需要代理：Mac 上开反向隧道 `ssh -f -N -R 127.0.0.1:17897:127.0.0.1:7897 4090`，
服务器上 `export https_proxy=http://127.0.0.1:17897`（镜像站可改用 `export HF_ENDPOINT=https://hf-mirror.com`）。

## 1. 启动服务与界面

```bash
https_proxy=http://127.0.0.1:17897 build/dev/bin/mqde serve --repo $PWD --runs $PWD/runs --web $PWD/web/dist \
  --python /root/miniconda3/envs/mqde/bin/python --gpus 2,3 --allow /cache/wuzhiping/hub --port 18080
# Mac 上：ssh -N -L 18080:127.0.0.1:18080 4090，然后打开 http://localhost:18080
```

服务只监听 127.0.0.1。`--gpus` 只写允许使用的卡；LLM 作业可选 `deploy.device: cpu`，不占 GPU 池（同一时间一个 CPU 作业）。

## 2. 不下载权重的预估

```bash
build/dev/bin/mqde model inspect hf:Qwen/Qwen2.5-7B-Instruct          # 只取 config + safetensors 头（约 38 KiB）
build/dev/bin/mqde estimate -m hf:Qwen/Qwen2.5-7B-Instruct -r w4a16,fp8 --conc 1,8
build/dev/bin/mqde model inspect /cache/wuzhiping/hub/MiniCPM-RobotManip   # 本地目录
```

界面里的“部署预评估”页给出显存构成、KV Cache 与并发曲线；“模型库”页可搜索 HF 并下载。

## 3. 量化评估作业

界面：**新建作业** → 选 VLA 或 LLM → 勾选配方（W4A16 / W4A8 / W2A16 / W8A16 / FP8 …）→ 选算法（RTN / GPTQ / AWQ / SmoothQuant / 组合）→ LLM 可再选静态激活、KV Cache 量化（默认 KIVI）、混合精度、真实导出格式、敏感度分析 → 提交。作业页实时显示阶段、逐层误差热力图和各评估位点结果。

命令行（等价）：

```bash
# LLM：wikitext-2 PPL（pretrain / fake_quant_wo_kv / fake_quant），CPU 即可
CUDA_VISIBLE_DEVICES="" PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model /cache/wuzhiping/hub/Qwen2.5-0.5B-Instruct --recipes w4a16,w8a16 --algorithm gptq \
  --kv int8 --export gptq,awq --sensitivity block --job-dir runs/qwen-demo
# 或用作业文件
PYTHONPATH=python python -m mqde.pipeline.llm_job examples/jobs/qwen25_05b_wikitext.yaml

# VLA（需要 GPU）
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python -m mqde.pipeline.vla_job examples/jobs/minicpm_vla_libero.yaml --algorithm gptq
```

`report.json` 在作业目录下；`env.torch_threads` 会被记录，因为 GPTQ 的结果随 BLAS 线程数在末位不同（W4 约 1% PPL）。

## 4. 真实导出与一致性（S09 / S10）

`--export gptq,awq` 为每个配方写出 GPTQ v1 / AWQ GEMM checkpoint（`export/<recipe>-<fmt>/`，含 tokenizer 与 `quantization_config`，混合精度写入 GPTQModel `dynamic`），随后：

- `real_check`：用参考解包器读回，逐层比对整数码与零点（必须 0 处不一致）；
- `eval_real`：把 fp16 scale 反量化出的真实权重换进模型重测 PPL（Qwen2.5-0.5B W4A16：与伪量化相差 0.0005%）。

AWQ / SmoothQuant 的激活缩放尚未折叠进前一层，这类层会被导出器拒绝并说明原因。

## 5. llama.cpp（GGUF，纯 CPU 闭环）

```bash
# 一次性：在仓库外的 llama.cpp 源码树上 out-of-tree 构建到 build/llamacpp
cmake -S /cache/wuzhiping/llama.cpp-omni -B build/llamacpp -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build build/llamacpp --target llama-perplexity llama-quantize llama-bench -j 48
```

```python
from mqde.export.gguf import convert_f16, requantize
from mqde.backend.llamacpp import perplexity, bench
from mqde.data.text import wikitext_text
open("wiki.test.txt", "w").write(wikitext_text("test"))                                  # 本地 HF 缓存中的 wikitext-2
f16 = convert_f16("/cache/wuzhiping/hub/Qwen2.5-0.5B-Instruct", "out/qwen-f16.gguf")   # llama.cpp 自己的转换器
requantize(f16, "out/qwen-q4_k.gguf", "q4_k")     # blk.* 线性层用与 ggml 逐字节一致的块量化重编码
perplexity("out/qwen-q4_k.gguf", "wiki.test.txt", ctx=512, chunks=40)
bench("out/qwen-q4_k.gguf")                        # pp / tg tokens/s
```

## 6. 敏感度与混合精度

作业开启“敏感度分析”后，每个配方页会显示逐块（或逐层类型）恢复 FP 能挽回的 PPL，并高亮建议升到 8 bit 的组；点击“按建议生成新作业”即以这些规则克隆并提交。建议值是单组效果的相加，偏乐观，以新作业实测为准。

## 常见问题

- HF 搜索/下载超时：服务进程没有带 `https_proxy` 启动，或反向隧道断了。
- `deploy.device=cpu is only supported for model.type=llm`：VLA 作业需要 GPU 池。
- GPU 2/3 被他人进程占用时不要结束它们；LLM 作业改用 CPU。
