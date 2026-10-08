// Storyboard as data: each scene has a URL (or card), optional setup, and sentences.
// A sentence = { zh, en, hl } ; hl = list of highlight specs (see record.mjs: locate()).
// Numbers are quoted exactly as displayed on the pages (checked 2026-10-08).
const J1 = '/jobs/20261008-101629-62da92';
const card = (c, extra = {}) => ({ card: c, ...extra });

export const scenes = [
  { id: 'title', card: 'title', sentences: [
    { zh: 'MQDEsys，模型量化与部署评估系统。从预估、量化、评测、真实导出，到 vLLM 和 llama.cpp 的部署压测，一条链路，一个网页。',
      en: 'MQDEsys: estimation, quantization, evaluation, real export, and load tests on vLLM and llama.cpp, in one pipeline and one web page.' },
    { zh: '下面用真实跑完的作业演示。',
      en: 'Everything shown comes from real, finished jobs.' },
  ]},
  { id: 'estimate', url: '/estimate', wait: 'table', sentences: [
    { zh: '为什么要量化？资源预估页不加载权重，只读配置和权重头。模型是 MiniCPM-RobotManip，1.54 B 参数，硬件是一张 RTX 4090。',
      en: 'Why quantize? The estimate page does not load weights, it only reads the config and weight headers. The model is MiniCPM-RobotManip, 1.54 B parameters, on one RTX 4090.',
      hl: [{ text: 'minicpm_vla', up: 1 }, { text: '21.59 GiB', up: 1 }] },
    { zh: 'BF16 峰值显存 4.02 GiB，W4A16 降到 2.52 GiB，可用显存 21.59 GiB。',
      en: 'BF16 peaks at 4.02 GiB, W4A16 at 2.52 GiB, against 21.59 GiB available.',
      hl: [{ css: 'table', rowIdx: 0, col: 3 }, { css: 'table', rowIdx: 3, col: 3 }] },
    { zh: '每个配方都标出可否部署和最大并发：BF16 是 310，W4A16 是 336。W4A8 延迟下界 10.3 ms。',
      en: 'Each recipe is marked deployable or not, with maximum concurrency: 310 for BF16, 336 for W4A16. W4A8 has a latency lower bound of 10.3 ms.',
      hl: [{ css: 'table', rowIdx: 0, col: 5 }, { css: 'table', rowIdx: 3, col: 5 }, { css: 'table', rowIdx: 4, col: 6 }] },
  ]},
  { id: 'job', url: J1, wait: 'text=敏感度分析', sentences: [
    { zh: '模型是 Qwen2.5-0.5B-Instruct。顶部是阶段时间线，从模型加载、逐块量化到导出和一致性检查，每个阶段都有耗时。',
      en: 'The model is Qwen2.5-0.5B-Instruct. The top shows the stage timeline, from model load and block-wise quantization to export and the consistency check, each with its duration.',
      hl: [card('执行过程')] },
    { zh: '原模型困惑度 18.103，GPTQ 量化 W4A16 后是 20.655，高 14.10%。',
      en: 'Baseline perplexity is 18.103. After W4A16 with GPTQ it is 20.655, 14.10% higher.',
      hl: [{ text: '18.103', up: 1 }, { card: '配方汇总', row: 'W4A16', col: 5 }] },
  ]},
  { id: 'export', url: J1, wait: 'text=敏感度分析', sentences: [
    { zh: '导出 AWQ 和 compressed-tensors 两种格式，约 448 MiB，整数码全部一致。',
      en: 'Export produces AWQ and compressed-tensors checkpoints of about 448 MiB, with all integer codes matching.',
      hl: [card('真实低比特导出', { row: 'awq' }), card('真实低比特导出', { row: 'compressed-tensors' })] },
    { zh: '用导出的权重重测，困惑度都是 20.656，相对伪量化加 0.00%。这就是伪量化与真实权重的对拍。',
      en: 'Re-measuring with the exported weights gives 20.656 for both, plus 0.00% against fake quantization. This is the fake versus real check.',
      hl: [card('真实低比特导出', { row: 'awq', col: 6 }), card('真实低比特导出', { row: 'compressed-tensors', col: 6 })] },
  ]},
  { id: 'vllm', url: J1, wait: 'text=敏感度分析', sentences: [
    { zh: '两个检查点由 vllm serve 直接加载，内核是 MarlinLinearKernel，启动 28 秒，权重显存 0.43 GiB，KV 容量 560k token。',
      en: 'Both checkpoints load directly in vllm serve with MarlinLinearKernel: 28 seconds to start, 0.43 GiB of weights, 560k tokens of KV capacity.',
      hl: [card('vLLM 部署', { row: 'awq' })] },
    { zh: '部署后困惑度 20.658，相对真实权重只加 0.01%。',
      en: 'Served perplexity is 20.658, only 0.01% above the real weights.',
      hl: [card('vLLM 部署', { row: 'awq', col: 7 }), card('vLLM 部署', { row: 'awq', col: 8 }), card('vLLM 部署', { row: 'compressed-tensors', col: 7 }), card('vLLM 部署', { row: 'compressed-tensors', col: 8 })] },
    { zh: '四并发压测，AWQ 吞吐 240 token 每秒，TTFT 52 ms，TPOT 16.2 ms；compressed-tensors 是 225，55 ms 和 16.9 ms。',
      en: 'Under a four-way load test, AWQ serves 240 tokens per second, with a TTFT of 52 ms and a TPOT of 16.2 ms. compressed-tensors gives 225, 55 ms and 16.9 ms.',
      hl: [card('vLLM 部署', { row: 'awq', col: 9 }), card('vLLM 部署', { row: 'awq', col: 10 }), card('vLLM 部署', { row: 'awq', col: 11 }), card('vLLM 部署', { row: 'compressed-tensors', col: 9 }), card('vLLM 部署', { row: 'compressed-tensors', col: 10 }), card('vLLM 部署', { row: 'compressed-tensors', col: 11 })] },
  ]},
  { id: 'llamacpp', url: '/jobs/20261001-004149-75d020', wait: 'text=llama.cpp 部署实测', sentences: [
    { zh: '另一个作业用 llama.cpp 在 CPU 上测 GGUF 类型的 PPL 和速度。',
      en: 'Another job tests GGUF types with llama.cpp on the CPU: perplexity and speed.',
      hl: [card('llama.cpp 部署实测')] },
    { zh: 'F16 是 948.10 MiB，PPL 13.890，解码 26.0 token 每秒。Q8_0 是 628.18 MiB，PPL 13.935，只高 0.33%，解码 40.6。',
      en: 'F16 is 948.10 MiB, PPL 13.890, decoding 26.0 tokens per second. Q8_0 is 628.18 MiB, PPL 13.935, only 0.33% higher, decoding 40.6.',
      hl: [card('llama.cpp 部署实测', { row: 'F16' }), card('llama.cpp 部署实测', { row: 'Q8_0' })] },
    { zh: 'Q4_K 是 487.74 MiB，PPL 14.749，144 层按规则回退。IQ4_NL 是 457.55 MiB，PPL 15.077，高 8.55%，解码 44.8。',
      en: 'Q4_K is 487.74 MiB, PPL 14.749, with 144 layers falling back by rule. IQ4_NL is 457.55 MiB, PPL 15.077, 8.55% higher, decoding 44.8.',
      hl: [card('llama.cpp 部署实测', { row: 'Q4_K' }), card('llama.cpp 部署实测', { row: 'IQ4_NL' })] },
  ]},
  { id: 'layers', url: '/jobs/20260930-190720-c80eda', wait: 'text=W4A16 误差分析', click: 'W4A16 误差分析', sentences: [
    { zh: '再看误差定位。这里换了一个带敏感度分析的作业，数字和上一个不同。把某一类层单独恢复成浮点，mlp.down_proj 挽回的 PPL 最多。',
      en: 'Now error localization, on a different job that includes the sensitivity analysis, so its numbers differ from the last one. Restoring mlp.down_proj to floating point recovers the most PPL.',
      hl: [card('敏感度分析')] },
    { zh: '混合精度建议：把 self_attn.v_proj 和 mlp.down_proj 提到 8 bit，预计 PPL 约 20.923，权重增加 51.19 MiB。预计值要以重新运行为准。',
      en: 'Suggestion: raise self_attn.v_proj and mlp.down_proj to 8 bit, for an expected PPL of about 20.923 and 51.19 MiB more weights. The real result comes from re-running.',
      hl: [{ text: /建议把/ }] },
    { zh: '逐块误差热力图：横轴是块，纵轴是层类型，颜色是 SQNR，越深越低。前几个块更浅，SQNR 更高。',
      en: 'The error heat map: blocks across, layer types down, colored by SQNR in dB, darker is lower. The first blocks are lighter, so their SQNR is higher.',
      hl: [card('逐层量化误差热力图')] },
    { zh: '误差最大的层是 layers.16.self_attn.v_proj，输出 SQNR 21.80 dB，是混合精度的候选。',
      en: 'The worst layer is layers.16.self_attn.v_proj, with an output SQNR of 21.80 dB, a mixed-precision candidate.',
      hl: [card('误差最大的 15 层', { rowIdx: 0 })] },
  ]},
  { id: 'compare', url: '/compare', wait: 'table', sentences: [
    { zh: '配方对比页把已完成的作业放在一张图上，看精度与位宽的权衡。',
      en: 'The compare page puts finished jobs on one chart: accuracy against bit width.',
      hl: [{ css: 'svg, canvas', first: true, up: 3 }] },
    { zh: 'FP8 的 PPL 是 18.333，W4A8 是 21.933，GPTQ 的 W4A16 在 20.411 到 20.655 之间。同一模型、同一评估窗口才可直接比较。',
      en: 'FP8 has a PPL of 18.333, W4A8 has 21.933, and the GPTQ W4A16 jobs range from 20.411 to 20.655. Only jobs with the same model and evaluation window compare directly.',
      hl: [{ css: 'table', row: '290e53', col: 5 }, { css: 'table', row: 'W4A8/gptq', col: 5 }, { css: 'table', row: '62da92', col: 5 }] },
  ]},
  { id: 'closing', card: 'closing', sentences: [
    { zh: 'MQDEsys，每一步的数字都能在页面上追溯。完整文档在仓库的 docs 目录里，仓库名是 MQDEsys。',
      en: 'MQDEsys: every number can be traced on the page. The full documentation is in the docs folder of the MQDEsys repository.' },
  ]},
];

export const cards = {
  title: { zh: ['MQDEsys', '模型量化与部署评估系统', '预估 · 量化 · 评测 · 导出 · 部署压测'], en: ['MQDEsys', 'Model Quantization & Deployment Evaluation System', 'Estimate · Quantize · Evaluate · Export · Load-test'] },
  closing: { zh: ['MQDEsys', '每一步的数字，都能在页面上追溯', '文档：仓库 docs/ 目录  ·  仓库：MQDEsys'], en: ['MQDEsys', 'Every number traceable on the page', 'Docs: the docs/ folder  ·  Repository: MQDEsys'] },
};
