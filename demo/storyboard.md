# MQDEsys demo storyboard / 演示分镜
Generated from `scenes.mjs` + measured timeline (`work/timeline-*.json`). Narration lengths come from `say`; ZH total 272.9 s, EN total 260.6 s (about 4.5 min).
Skipped on purpose: model catalog and graph pages (the graph page contains an internal checkpoint name), and the jobs list and FP8 job (cut for length). Not covered: the hardware/fit pages.

## title: Title card (HTML)

Start: ZH 0 s / EN 0 s
- ZH (16.3 s): MQDEsys，模型量化与部署评估系统。从预估、量化、评测、真实导出，到 vLLM 和 llama.cpp 的部署压测，一条链路，一个网页。下面用真实跑完的作业演示。
- EN (14.5 s): MQDEsys, the Model Quantization and Deployment Evaluation System: from estimation, quantization and real export to load tests on vLLM and llama.cpp, in one pipeline and one web interface. The demo uses real, finished jobs.

## estimate: /estimate  resource estimate, deployability matrix

Start: ZH 23 s / EN 22 s
- ZH (13.4 s): 为什么要量化？资源预估页不加载权重，只读配置和权重头。模型是 MiniCPM-RobotManip，1.54 B 参数，硬件是一张 RTX 4090。
- EN (11.9 s): Why quantize? The estimate page does not load weights, it only reads the config and weight headers. The model is MiniCPM-RobotManip, 1.54 B parameters, on one RTX 4090.
- ZH (8.3 s): BF16 峰值显存 4.02 GiB，W4A16 降到 2.52 GiB，可用显存 21.59 GiB。
- EN (8.2 s): BF16 peaks at 4.02 GiB, W4A16 at 2.52 GiB, against 21.59 GiB available.
- ZH (11.3 s): 每个配方都标出可否部署和最大并发：BF16 是 310，W4A16 是 336。W4A8 延迟下界 10.3 ms。
- EN (13.0 s): Each recipe is marked deployable or not, with maximum concurrency: 310 for BF16, 336 for W4A16. W4A8 has a latency lower bound of 10.3 ms.

## job: /jobs/20261008-101629-62da92  stage timeline, PPL

Start: ZH 60 s / EN 58 s
- ZH (10.7 s): 模型是 Qwen2.5-0.5B-Instruct。顶部是阶段时间线，从模型加载、逐块量化到导出和一致性检查，每个阶段都有耗时。
- EN (10.6 s): The model is Qwen2.5-0.5B-Instruct. The top shows the stage timeline, from model load and block-wise quantization to export and the consistency check, each with its duration.
- ZH (9.4 s): 原模型困惑度 18.103，GPTQ 量化 W4A16 后是 20.655，高 14.10%。
- EN (9.9 s): Baseline perplexity is 18.103. After W4A16 with GPTQ it is 20.655, 14.10% higher.

## export: same job: S09/S10 export + fake/real check

Start: ZH 83 s / EN 82 s
- ZH (6.9 s): 导出 AWQ 和 compressed-tensors 两种格式，约 448 MiB，整数码全部一致。
- EN (7.2 s): Export produces AWQ and compressed-tensors checkpoints of about 448 MiB, with all integer codes matching.
- ZH (9.3 s): 用导出的权重重测，困惑度都是 20.656，相对伪量化加 0.00%。这就是伪量化与真实权重的对拍。
- EN (9.4 s): Re-measuring with the exported weights gives 20.656 for both, plus 0.00% against fake quantization. This is the fake versus real check.

## vllm: same job: S11/S12 vLLM card

Start: ZH 103 s / EN 102 s
- ZH (12.2 s): 两个检查点由 vllm serve 直接加载，内核是 MarlinLinearKernel，启动 28 秒，权重显存 0.43 GiB，KV 容量 560k token。
- EN (10.8 s): Both checkpoints load directly in vllm serve with MarlinLinearKernel: 28 seconds to start, 0.43 GiB of weights, 560k tokens of KV capacity.
- ZH (5.2 s): 部署后困惑度 20.658，相对真实权重只加 0.01%。
- EN (5.7 s): Served perplexity is 20.658, only 0.01% above the real weights.
- ZH (15.5 s): 四并发压测，AWQ 吞吐 240 token 每秒，TTFT 52 ms，TPOT 16.2 ms；compressed-tensors 是 225，55 ms 和 16.9 ms。
- EN (15.6 s): Under a four-way load test, AWQ serves 240 tokens per second, with a TTFT of 52 ms and a TPOT of 16.2 ms. compressed-tensors gives 225, 55 ms and 16.9 ms.

## llamacpp: /jobs/20261001-004149-75d020  llama.cpp GGUF card

Start: ZH 140 s / EN 138 s
- ZH (9.4 s): 另一个作业用 llama.cpp 在 CPU 上测 GGUF 类型的 PPL 和速度。
- EN (5.3 s): Another job tests GGUF types with llama.cpp on the CPU: perplexity and speed.
- ZH (15.8 s): F16 是 948.10 MiB，PPL 13.890，解码 26.0 token 每秒。Q8_0 是 628.18 MiB，PPL 13.935，只高 0.33%，解码 40.6。
- EN (18.1 s): F16 is 948.10 MiB, PPL 13.890, decoding 26.0 tokens per second. Q8_0 is 628.18 MiB, PPL 13.935, only 0.33% higher, decoding 40.6.
- ZH (18.3 s): Q4_K 是 487.74 MiB，PPL 14.749，144 层按规则回退。IQ4_NL 是 457.55 MiB，PPL 15.077，高 8.55%，解码 44.8。
- EN (18.7 s): Q4_K is 487.74 MiB, PPL 14.749, with 144 layers falling back by rule. IQ4_NL is 457.55 MiB, PPL 15.077, 8.55% higher, decoding 44.8.

## layers: /jobs/20260930-190720-c80eda  W4A16 error analysis: sensitivity, heat map, worst layers

Start: ZH 190 s / EN 186 s
- ZH (9.7 s): 再看误差定位。敏感度分析把某一类层单独恢复成浮点，看能挽回多少 PPL，mlp.down_proj 挽回最多。
- EN (9.7 s): Now error localization. The sensitivity analysis restores one layer type at a time to floating point and shows how much PPL that recovers. mlp.down_proj recovers the most.
- ZH (13.5 s): 混合精度建议：把 self_attn.v_proj 和 mlp.down_proj 提到 8 bit，预计 PPL 约 20.923，权重增加 51.19 MiB。预计值要以重新运行为准。
- EN (12.4 s): Mixed-precision suggestion: raise self_attn.v_proj and mlp.down_proj to 8 bit, for an expected PPL of about 20.923 and 51.19 MiB more weights. The real result comes from re-running.
- ZH (11.1 s): 逐块误差热力图：横轴是块，纵轴是层类型，颜色是 SQNR，越深越低。前几个块更浅，SQNR 更高。
- EN (10.1 s): The block-wise error heat map: blocks across, layer types down, colored by SQNR in dB, darker meaning lower. The first blocks are lighter, so their SQNR is higher.
- ZH (11.2 s): 误差最大的层是 layers.16.self_attn.v_proj，输出 SQNR 21.80 dB，是混合精度的候选。
- EN (8.5 s): The worst layer is layers.16.self_attn.v_proj, with an output SQNR of 21.80 dB, a mixed-precision candidate.

## compare: /compare  recipe comparison

Start: ZH 239 s / EN 230 s
- ZH (5.2 s): 配方对比页把已完成的作业放在一张图上，看精度与位宽的权衡。
- EN (4.2 s): The compare page puts finished jobs on one chart: accuracy against bit width.
- ZH (16.6 s): FP8 的 PPL 是 18.333，W4A8 是 21.933，GPTQ 的 W4A16 在 20.411 到 20.655 之间。同一模型、同一评估窗口才可直接比较。
- EN (16.0 s): FP8 has a PPL of 18.333, W4A8 has 21.933, and the GPTQ W4A16 jobs range from 20.411 to 20.655. Only jobs with the same model and evaluation window compare directly.

## closing: Closing card (docs folder, repo name MQDEsys)

Start: ZH 262 s / EN 252 s
- ZH (9.6 s): MQDEsys，每一步的数字都能在页面上追溯。完整文档在仓库的 docs 目录里，仓库名是 MQDEsys。
- EN (7.6 s): MQDEsys: every number can be traced on the page. The full documentation is in the docs folder of the MQDEsys repository.
