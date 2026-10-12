import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, CatalogModel, DownloadStatus, fmtBytes, RuntimeParams } from "../api";
import { DEFAULT_DATASET, DEFAULT_LLM, DEFAULT_MODEL, useAsync, useDebounced } from "../hooks";
import { Meter } from "./Fit";

const LIVE_RUNTIME: RuntimeParams = { concurrency: 1, images_per_step: 2, text_tokens: 42, image_resolution: 448, gpu_memory_utilization: 0.9, extra_modalities: [] };

// T-1301/T-1304: domains with a real quantization pipeline today. Anything else from the registry (vlm,
// world, ewm) is selectable but shown as a blocked notice instead of a form that would just be rejected by
// the server's own validate_jobspec check -- see docs/TASKS.md T-1304 for why those are not wired up yet.
const LLM_LIKE = new Set(["llm"]);

export default function NewJobPage() {
  const nav = useNavigate();
  const domainsQ = useAsync(api.domains, []);
  const recipes = useAsync(api.recipes, []);
  const algos = useAsync(api.algorithms, []);
  const [algorithm, setAlgorithm] = useState("recipe");
  const profiles = useAsync(api.hardwareProfiles, []);
  const cfg = useAsync(api.config, []);
  const [domain, setDomain] = useState("vla");
  const [name, setName] = useState("minicpm-vla-eval");
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [modelMode, setModelMode] = useState<"custom" | "catalog">("custom");
  const [picked, setPicked] = useState<string[]>(["w8a16", "fp8", "w4a16", "w4a8"]);
  const [noQuant, setNoQuant] = useState(false);
  // T-1304/T-1306: "仅量化不评测" (evaluation.skip) -- now a real backend capability (llm_job.py/vla_job.py's
  // quantize_only branches, planner::validate_jobspec rejects combining it with export/vllm/llamacpp/
  // sensitivity/logits_probe/attribution). The job-construction logic below strips those fields outright
  // when this is on, matching the server's own rejection list exactly, so the UI can never produce a job
  // the server would 400.
  const [quantizeOnly, setQuantizeOnly] = useState(false);
  // T-1304: advanced per-job group_size override (quant.group_size, read by llm_job.py/vla_job.py's
  // group_size_override). "" = not overridden, so an untouched job omits the field entirely and matches
  // every picked recipe's own data/recipes/*.yaml group_size exactly.
  const [groupSizeOverride, setGroupSizeOverride] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // LLM (wikitext-2 PPL) options
  const [seqLen, setSeqLen] = useState(512);
  const [windows, setWindows] = useState(40);
  const [actStatic, setActStatic] = useState("");
  const [kv, setKv] = useState("int8");
  const [kvMethod, setKvMethod] = useState("kivi");
  // T-1304/T-1306: quant_device/eval_device split (real backend: schemas/job_spec.schema.json's
  // quant_device/evaluation.device, consumed by llm_job.py and resolved by JobRunner against the GPU
  // pool). Omitted from the submitted job when equal -- an unsplit job looks exactly like it did before
  // this existed (single deploy.device, no new fields).
  const [quantDevice, setQuantDevice] = useState("cpu");
  const [evalDevice, setEvalDevice] = useState("cpu");
  const [keepEnds, setKeepEnds] = useState(false);
  const [downBits, setDownBits] = useState("");
  const [exportFmts, setExportFmts] = useState<string[]>(["gptq"]);
  const [sens, setSens] = useState("");
  const [gguf, setGguf] = useState<string[]>([]);
  // logits probe (KL / Top-1 agreement), vLLM deployment (S11/S12) and lm-eval task accuracy: all LLM-only
  const [probe, setProbe] = useState(false);
  const [probeExact, setProbeExact] = useState(false);
  const [useVllm, setUseVllm] = useState(false);
  const [vllmUtil, setVllmUtil] = useState(0.3);
  const [useLmEval, setUseLmEval] = useState(false);
  const [lmTasks, setLmTasks] = useState("gsm8k");
  const [lmLimit, setLmLimit] = useState(50);
  const [lmShots, setLmShots] = useState(5);
  const [dataset, setDataset] = useState(DEFAULT_DATASET);
  const [nCalib, setNCalib] = useState(16);
  const [nEval, setNEval] = useState(24);
  const [hw, setHw] = useState("nvidia/rtx4090");
  const [attribution, setAttribution] = useState(true);
  const [action, setAction] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // T-1304: which real backend this job deploys to, if any. Drives an auto-suggested export format
  // instead of leaving the GPTQ/AWQ/GGUF chip lists below to be lined up with vLLM/llama.cpp by hand.
  const [deployBackend, setDeployBackend] = useState<"none" | "vllm" | "llamacpp" | "sglang">("none");

  const kind: "vla" | "llm" = LLM_LIKE.has(domain) ? "llm" : "vla";
  const currentDomain = domainsQ.data?.find((d) => d.id === domain);
  const blocked = currentDomain != null && currentDomain.pipeline_status !== "implemented";

  const switchDomain = (d: string) => {
    setDomain(d);
    const llmLike = LLM_LIKE.has(d);
    setModel(llmLike ? DEFAULT_LLM : DEFAULT_MODEL);
    setName(llmLike ? "qwen-wikitext-ppl" : "minicpm-vla-eval");
    setPicked(llmLike ? ["w4a16", "w4a8", "w2a16"] : ["w8a16", "fp8", "w4a16", "w4a8"]);
    setNCalib(llmLike ? 32 : 16);
    setNoQuant(false);
    setQuantizeOnly(false);
    setGroupSizeOverride("");
    setModelMode("custom");
  };

  // "不量化（保持原始精度）": bf16 (data/recipes/bf16.yaml) already means "no quantization" end to end, so
  // this reuses that real recipe rather than inventing a new, unimplemented concept.
  useEffect(() => {
    if (noQuant) setPicked(["bf16"]);
  }, [noQuant]);

  // Picking a real deploy backend suggests (and can override) the export chips below, via the C++
  // planner's own compat-matrix logic (GET /api/v1/export-suggestion, T-1304) -- not a frontend guess.
  // Fetching and applying happen inside the same effect/callback (not a separate useAsync + a second
  // effect keyed on its .data) so there is no window where a stale suggestion from the *previous* backend
  // pairs with the *new* deployBackend value -- that combination briefly set llamacpp's "gguf" suggestion
  // onto a vllm selection in testing before this was merged into one effect.
  const primaryRecipe = picked.find((r) => r !== "bf16") ?? picked[0];
  const [suggestion, setSuggestion] = useState<Awaited<ReturnType<typeof api.exportSuggestion>> | null>(null);
  useEffect(() => {
    if (deployBackend === "none") {
      setUseVllm(false);
      setGguf([]);
      setSuggestion(null);
      return;
    }
    if (deployBackend === "sglang" || kind !== "llm" || !primaryRecipe) {
      setSuggestion(null);
      return; // sglang isn't wired into llm_job.py yet (T-0704); nothing to suggest
    }
    let alive = true;
    api
      .exportSuggestion(primaryRecipe, deployBackend, hw)
      .then((s) => {
        if (!alive) return;
        setSuggestion(s);
        // Turning deployment on does not depend on there being a format to suggest (e.g. a bf16/no-quant
        // recipe legitimately has no export format but can still be served as-is).
        if (deployBackend === "vllm") {
          setUseVllm(true);
          if (s.format && s.verdict !== "unsupported") setExportFmts([s.format]);
        } else if (deployBackend === "llamacpp") {
          setUseVllm(false);
          if (s.format) setGguf((g) => (g.length ? g : ["q4_k"]));
        }
      })
      .catch(() => alive && setSuggestion(null));
    return () => {
      alive = false;
    };
  }, [deployBackend, primaryRecipe, hw, kind]);

  const kvSpec = kv === "" ? null : kv === "fp8" ? { format: "fp8_e4m3", bits: 8 } : { format: "int", bits: Number(kv.slice(3)), symmetric: false };
  const rules = [
    ...(keepEnds ? [{ match: "block:0", bits: 8 }, { match: "block:-1", bits: 8 }] : []),
    ...(downBits === "fp" ? [{ match: "mlp\\.down_proj$", skip: true }] : downBits ? [{ match: "mlp\\.down_proj$", bits: Number(downBits) }] : []),
  ];
  const job =
    kind === "vla"
      ? {
          domain,
          metadata: { name },
          model: { source: "local", path: model, type: "vla", modality: "vla" },
          quant: { recipe: picked, algorithm, ...(action ? { also_quantize: ["action"] } : {}), ...(groupSizeOverride !== "" ? { group_size: Number(groupSizeOverride) } : {}) },
          calibration: { dataset, n_samples: nCalib },
          evaluation: {
            dataset,
            n_samples: nEval,
            concurrency: [1, 4, 16, 64],
            attribution: quantizeOnly ? false : attribution,
            ...(quantizeOnly ? { skip: true } : {}),
          },
          deploy: { hardware: hw },
        }
      : {
          domain,
          metadata: { name },
          model: { source: "local", path: model, type: "llm" },
          quant: {
            recipe: picked,
            algorithm,
            act_static: actStatic || null,
            ...(kvSpec ? { kv: { ...kvSpec, method: kvMethod } } : {}),
            ...(rules.length ? { mixed_precision: { rules } } : {}),
            ...(groupSizeOverride !== "" ? { group_size: Number(groupSizeOverride) } : {}),
          },
          ...(quantDevice !== evalDevice ? { quant_device: quantDevice } : {}),
          calibration: { dataset: "wikitext-2-raw-v1:train", n_samples: nCalib, seq_len: seqLen },
          evaluation: {
            dataset: "wikitext-2-raw-v1:test",
            seq_len: seqLen,
            max_windows: windows,
            ...(quantDevice !== evalDevice ? { device: evalDevice } : {}),
            ...(quantizeOnly ? { skip: true } : {}),
          },
          deploy: {
            hardware: hw,
            device: quantDevice === "gpu" || evalDevice === "gpu" ? "gpu" : "cpu",
            ...(!quantizeOnly && gguf.length ? { llamacpp: { types: gguf } } : {}),
            ...(!quantizeOnly && useVllm
              ? {
                  vllm: {
                    gpu_memory_utilization: vllmUtil,
                    ...(useLmEval
                      ? { lm_eval: { tasks: lmTasks.split(",").map((t) => t.trim()).filter(Boolean), limit: lmLimit, num_fewshot: lmShots } }
                      : {}),
                  },
                }
              : {}),
          },
          ...(!quantizeOnly && exportFmts.length ? { export: { formats: exportFmts } } : {}),
          ...(!quantizeOnly && (sens || probe)
            ? {
                analysis: {
                  ...(sens ? { sensitivity: sens, sensitivity_windows: 8 } : {}),
                  ...(probe ? { logits_probe: probeExact ? { exact: true } : {} } : {}),
                },
              }
            : {}),
        };

  const submit = async () => {
    setErr(null);
    setBusy(true);
    try {
      const { job_id } = await api.submit(job);
      nav(`/jobs/${job_id}`);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // live estimate sidebar: same endpoint as the estimate page, 300 ms after the last change
  const [estBackend, setEstBackend] = useState("hf_transformers");
  const backendList = useAsync(api.backends, []);
  const live = useDebounced({ model, hw, picked, action, estBackend }, 300);
  const liveRt = { ...LIVE_RUNTIME, extra_modalities: live.action && kind === "vla" ? ["action"] : [] };
  const est = useAsync(
    () => (live.picked.length ? api.estimate(live.model, live.hw, live.picked, liveRt, live.estBackend) : Promise.resolve(null)),
    [JSON.stringify(live), kind],
  );

  const toggle = (id: string) => setPicked(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id]);

  return (
    <>
      <h1>新建评估作业</h1>
      <p className="subtitle">
        选领域 → 选模型 → 选量化算法 → 选配方 → 评估数据与设备 → 部署与导出。作业由 C++ 调度到 GPU 池（{cfg.data?.gpus.join(", ") ?? "…"}）上的 Python 执行器，全过程事件实时推送。
      </p>
      {err && <div className="error-box">{err}</div>}
      <div className="split">
      <div>
      <div className="card">
        <h2>① 领域 + 模型</h2>
        <div className="tabs" style={{ marginBottom: 12 }}>
          {(domainsQ.data ?? [{ id: "vla", display_name: "VLA（视觉-语言-动作）" }, { id: "llm", display_name: "LLM（HF CausalLM · wikitext-2 PPL）" }]).map((d) => (
            <div key={d.id} className={`tab ${domain === d.id ? "on" : ""}`} onClick={() => switchDomain(d.id)} data-testid={`domain-tab-${d.id}`}>
              {d.display_name}
            </div>
          ))}
        </div>
        {blocked ? (
          <div className="error-box" data-testid="domain-blocked">
            领域"{currentDomain?.display_name}"目前只在界面上可选，还没有真正的量化流水线（pipeline_status: {currentDomain?.pipeline_status}）。
            提交作业会被服务端拒绝并给出同样的理由。参考方法（尚未实现，仅作记录）：
            {currentDomain?.classic_algorithms.map((a) => a.name).join("、") || "—"}。
          </div>
        ) : (
          <>
            {kind === "llm" && (
              <div className="tabs" style={{ marginBottom: 8, borderBottom: "none" }}>
                <div className={`tab ${modelMode === "custom" ? "on" : ""}`} onClick={() => setModelMode("custom")}>
                  自定义路径
                </div>
                <div className={`tab ${modelMode === "catalog" ? "on" : ""}`} onClick={() => setModelMode("catalog")}>
                  从模型目录选择
                </div>
              </div>
            )}
            {modelMode === "catalog" && kind === "llm" ? (
              <CatalogModelPicker onPick={setModel} currentPath={model} />
            ) : (
              <div className="form-row">
                <div className="field grow">
                  <label>本地模型路径（允许根目录：{cfg.data?.allowed_roots.join(", ")}）</label>
                  <input value={model} onChange={(e) => setModel(e.target.value)} data-testid="model-path" />
                </div>
                <div className="field">
                  <label>作业名称</label>
                  <input value={name} onChange={(e) => setName(e.target.value)} />
                </div>
              </div>
            )}
            <h2 style={{ marginTop: 16 }}>② 量化算法</h2>
            <div className="form-row">
              <div className="field" style={{ minWidth: 260 }}>
                <label>算法（所有勾选配方共用）</label>
                <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)} disabled={noQuant}>
                  <option value="recipe">按配方默认（{recipes.data?.filter((r) => picked.includes(r.id)).map((r) => `${r.name}:${r.default_algorithm}`).join(" · ") || "—"}）</option>
                  {algos.data?.map((a) => (
                    <option key={a.name} value={a.name}>
                      {a.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field grow">
                <label>说明</label>
                <div className="small muted" style={{ minHeight: 32, display: "flex", alignItems: "center" }}>
                  {algorithm === "recipe"
                    ? "每个配方使用 data/recipes/*.yaml 里的 default_algorithm。"
                    : algos.data?.find((a) => a.name === algorithm)?.description}
                  {algos.data?.find((a) => a.name === algorithm)?.needs_calibration && " 需要校准数据。"}
                </div>
              </div>
            </div>
            {currentDomain && currentDomain.classic_algorithms.length > 0 && (
              <p className="note">
                {currentDomain.display_name} 领域经典方法参考：{currentDomain.classic_algorithms.map((a) => a.name).join("、")}
                （上面的下拉框列出的是本系统实际实现并可选的算法，不是这个参考列表本身）。
              </p>
            )}
            {algos.error && <div className="error-box">{algos.error}</div>}
            <h2 style={{ marginTop: 16 }}>③ 量化配方</h2>
            <label className="small" style={{ display: "block", marginBottom: 8 }}>
              <input type="checkbox" checked={noQuant} onChange={(e) => setNoQuant(e.target.checked)} data-testid="opt-no-quant" /> 不量化（保持原始精度，等同于只选 BF16 配方）
            </label>
            <label className="small" style={{ display: "block", marginBottom: 8 }}>
              <input type="checkbox" checked={quantizeOnly} onChange={(e) => setQuantizeOnly(e.target.checked)} data-testid="opt-quantize-only" /> 仅量化不评测（跳过
              {kind === "vla" ? "误差归因与评测" : "PPL 评测、敏感度分析、导出一致性检查与 vLLM/llama.cpp 部署"}，只保留量化本身与层级误差追踪）
            </label>
            <div className="chips" style={{ marginBottom: 8, opacity: noQuant ? 0.5 : 1, pointerEvents: noQuant ? "none" : "auto" }}>
              {recipes.data
                ?.filter((r) => r.weight.format !== "none")
                .map((r) => (
                  <span key={r.id} className={`chip ${picked.includes(r.id) ? "on" : ""}`} onClick={() => toggle(r.id)} title={r.summary}>
                    {r.name}
                    {r.maturity === "experimental" && <span className="exp">实验</span>}
                  </span>
                ))}
            </div>
            {(() => {
              const perGroupPicked = (recipes.data ?? []).filter((r) => picked.includes(r.id) && r.weight.granularity === "per_group");
              return (
                <div className="card" style={{ margin: "0 0 12px", padding: 10 }}>
                  <div style={{ cursor: "pointer", fontWeight: 600 }} onClick={() => setAdvancedOpen(!advancedOpen)} data-testid="advanced-params-toggle">
                    {advancedOpen ? "▾" : "▸"} 高级参数
                  </div>
                  {advancedOpen && (
                    <div className="form-row" style={{ marginTop: 8 }}>
                      <div className="field">
                        <label>Group size 覆盖（覆盖所选配方自身的 per-group 分组大小）</label>
                        <input
                          type="number"
                          min={1}
                          value={groupSizeOverride}
                          placeholder={perGroupPicked[0] ? String(perGroupPicked[0].weight.group_size) : "—"}
                          disabled={perGroupPicked.length === 0}
                          onChange={(e) => setGroupSizeOverride(e.target.value)}
                          data-testid="group-size-override"
                        />
                        <p className="small muted" style={{ marginTop: 4 }}>
                          {perGroupPicked.length > 0
                            ? `对以下配方生效：${perGroupPicked.map((r) => `${r.name}（默认 ${r.weight.group_size}）`).join("、")}；非 per-group 配方不受影响。`
                            : "当前所选配方都不是 per-group 粒度（例如 W8A16/FP8 是 per-channel），这个开关对它们没有意义，已禁用。"}
                        </p>
                      </div>
                      <div className="field grow">
                        <label>旋转量化（QuaRot）</label>
                        <select disabled value="">
                          <option value="">未实现（T-0610），敬请期待</option>
                        </select>
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}
            {kind === "vla" ? (
              <label className="small">
                <input type="checkbox" checked={action} onChange={(e) => setAction(e.target.checked)} /> 同时量化 DiT 动作头（默认保持 FP32）
              </label>
            ) : (
              <div className="form-row">
                <div className="field">
                  <label>激活量化</label>
                  <select value={actStatic} onChange={(e) => setActStatic(e.target.value)} disabled={noQuant}>
                    <option value="">动态 per-token</option>
                    <option value="minmax">静态 per-tensor · minmax</option>
                    <option value="ema">静态 per-tensor · EMA</option>
                    <option value="percentile">静态 per-tensor · 99.99% 分位</option>
                  </select>
                </div>
                <div className="field">
                  <label>KV Cache 量化</label>
                  <select value={kv} onChange={(e) => setKv(e.target.value)} disabled={noQuant}>
                    <option value="">按配方（多数为不量化）</option>
                    <option value="int8">INT8</option>
                    <option value="int4">INT4</option>
                    <option value="int2">INT2</option>
                    <option value="fp8">FP8 E4M3</option>
                  </select>
                </div>
                <div className="field">
                  <label>KV 方法</label>
                  <select value={kvMethod} onChange={(e) => setKvMethod(e.target.value)} disabled={!kv || noQuant}>
                    <option value="kivi">KIVI（K per-channel / V per-token）</option>
                    <option value="naive">naive（K、V 均 per-token）</option>
                  </select>
                </div>
                <div className="field">
                  <label>混合精度：首尾块</label>
                  <select value={keepEnds ? "8" : ""} onChange={(e) => setKeepEnds(e.target.value === "8")} disabled={noQuant}>
                    <option value="">同配方</option>
                    <option value="8">首块 + 末块 8 bit</option>
                  </select>
                </div>
                <div className="field">
                  <label>混合精度：down_proj</label>
                  <select value={downBits} onChange={(e) => setDownBits(e.target.value)} disabled={noQuant}>
                    <option value="">同配方</option>
                    <option value="8">8 bit</option>
                    <option value="4">4 bit</option>
                    <option value="fp">保持 FP（不量化）</option>
                  </select>
                </div>
              </div>
            )}
          </>
        )}
      </div>
      {!blocked && (
      <div className="card">
        <h2>④ 评估数据 + 设备</h2>
        {kind === "vla" ? (
          <>
            <div className="form-row">
              <div className="field grow">
                <label>数据集（LeRobot v3 目录，或 synthetic）</label>
                <input value={dataset} onChange={(e) => setDataset(e.target.value)} data-testid="dataset-path" />
              </div>
              <div className="field">
                <label>校准样本</label>
                <input type="number" value={nCalib} min={1} onChange={(e) => setNCalib(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>评估样本</label>
                <input type="number" value={nEval} min={1} onChange={(e) => setNEval(Number(e.target.value))} />
              </div>
            </div>
            <label className="small">
              <input type="checkbox" checked={attribution && !quantizeOnly} disabled={quantizeOnly} onChange={(e) => setAttribution(e.target.checked)} /> 分阶段误差归因（逐模态单独量化，定位误差来源环节）
              {quantizeOnly && <span className="muted"> （仅量化不评测已关闭，需要未量化基线）</span>}
            </label>
          </>
        ) : (
          <>
            <div className="form-row">
              <div className="field">
                <label>校准样本（wikitext-2 train）</label>
                <input type="number" value={nCalib} min={1} onChange={(e) => setNCalib(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>序列长度</label>
                <input type="number" value={seqLen} min={64} step={64} onChange={(e) => setSeqLen(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>PPL 窗口数（wikitext-2 test）</label>
                <input type="number" value={windows} min={1} onChange={(e) => setWindows(Number(e.target.value))} />
              </div>
            </div>
            <div className="form-row">
              <div className="field">
                <label>敏感度分析（逐组回退 FP，8 窗口 PPL）</label>
                <select value={sens} onChange={(e) => setSens(e.target.value)} disabled={quantizeOnly}>
                  <option value="">不做</option>
                  <option value="block">按块（每个 decoder block）</option>
                  <option value="type">按层类型（q/k/v/o/gate/up/down）</option>
                </select>
              </div>
              <div className="field">
                <label>量化设备</label>
                <select value={quantDevice} onChange={(e) => setQuantDevice(e.target.value)} data-testid="quant-device-select">
                  <option value="cpu">CPU（不占 GPU 池）</option>
                  <option value="gpu">GPU 池（{cfg.data?.gpus.join(", ") ?? "…"}）</option>
                </select>
              </div>
              <div className="field">
                <label>评估设备</label>
                <select value={evalDevice} onChange={(e) => setEvalDevice(e.target.value)} data-testid="eval-device-select">
                  <option value="cpu">CPU（不占 GPU 池）</option>
                  <option value="gpu">GPU 池（{cfg.data?.gpus.join(", ") ?? "…"}）</option>
                </select>
              </div>
            </div>
            <div className="form-row">
              <div className="field">
                <label>
                  <input type="checkbox" checked={probe} disabled={quantizeOnly} onChange={(e) => setProbe(e.target.checked)} data-testid="opt-probe" /> logits 探针（量化前后输出分布的 KL 与 Top-1 一致率）
                </label>
                <label className="small">
                  <input type="checkbox" checked={probeExact} disabled={!probe || quantizeOnly} onChange={(e) => setProbeExact(e.target.checked)} data-testid="opt-probe-exact" /> 精确 KL（保存全部 logits，约 1.2 GiB / 8 窗口；默认只存 top-256，KL 为下界）
                </label>
              </div>
            </div>
            <p className="note">评估位点：pretrain（原模型）→ fake_quant_wo_kv（权重+激活量化）→ fake_quant（再加 KV Cache 量化）。PPL 为不重叠窗口的 token 级困惑度。量化设备与评估设备可以不同（例如量化在 CPU 上跑以保证可复现，评估用 GPU 加速）；两者相同时不影响今天的提交行为。</p>
          </>
        )}
      </div>
      )}
      {!blocked && (
      <div className="card">
        <h2>⑤ 部署与导出</h2>
        <div className="form-row">
          <div className="field">
            <label>目标硬件画像（用于资源预估）</label>
            <select value={hw} onChange={(e) => setHw(e.target.value)} data-testid="hw-select">
              {profiles.data?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
          {kind === "llm" && (
            <div className="field" style={{ minWidth: 220 }}>
              <label>部署后端（自动建议导出格式）</label>
              <select
                value={deployBackend}
                disabled={quantizeOnly}
                onChange={(e) => setDeployBackend(e.target.value as typeof deployBackend)}
                data-testid="deploy-backend"
              >
                <option value="none">不部署（仅评估量化误差）</option>
                <option value="vllm">vLLM</option>
                <option value="llamacpp">llama.cpp（GGUF）</option>
                <option value="sglang">SGLang（尚未接入，T-0704）</option>
              </select>
              {suggestion && (
                <div className="small muted" data-testid="export-suggestion">
                  建议格式：{suggestion.format || "—"}（{suggestion.verdict}
                  {suggestion.reason ? `，${suggestion.reason}` : ""}）
                </div>
              )}
              {quantizeOnly && <div className="small muted">仅量化不评测已开启：导出/部署需要评测得到的 fake-quant 基线，本次跳过。</div>}
            </div>
          )}
          {kind === "llm" && (
            <div className="field" style={{ opacity: quantizeOnly ? 0.5 : 1, pointerEvents: quantizeOnly ? "none" : "auto" }}>
              <label>真实导出（S09/S10 一致性检查）</label>
              <div className="chips">
                {["gptq", "awq"].map((f) => (
                  <span key={f} className={`chip ${exportFmts.includes(f) ? "on" : ""}`} onClick={() => setExportFmts(exportFmts.includes(f) ? exportFmts.filter((x) => x !== f) : [...exportFmts, f])}>
                    {f.toUpperCase()}
                  </span>
                ))}
              </div>
            </div>
          )}
          {kind === "llm" && (
            <div className="field" style={{ opacity: quantizeOnly ? 0.5 : 1, pointerEvents: quantizeOnly ? "none" : "auto" }}>
              <label>llama.cpp 部署实测（GGUF，CPU）</label>
              <div className="chips">
                {["q8_0", "q5_0", "q4_0", "iq4_nl", "q4_k", "q2_k"].map((f) => (
                  <span key={f} className={`chip ${gguf.includes(f) ? "on" : ""}`} onClick={() => setGguf(gguf.includes(f) ? gguf.filter((x) => x !== f) : [...gguf, f])}>
                    {f.toUpperCase()}
                  </span>
                ))}
              </div>
            </div>
          )}
          {kind === "llm" && (
            <div className="field" style={{ minWidth: 320, opacity: quantizeOnly ? 0.5 : 1, pointerEvents: quantizeOnly ? "none" : "auto" }}>
              <label>
                <input type="checkbox" checked={useVllm} disabled={quantizeOnly} onChange={(e) => setUseVllm(e.target.checked)} data-testid="opt-vllm" /> vLLM 部署实测（需要 GPU 与 vLLM；缺失则该阶段跳过）
              </label>
              {useVllm && (
                <>
                  <label className="small">
                    显存利用率 <input type="number" step={0.05} min={0.1} max={0.95} value={vllmUtil} style={{ width: 70 }} onChange={(e) => setVllmUtil(Number(e.target.value))} data-testid="opt-vllm-util" />
                  </label>
                  <label className="small">
                    <input type="checkbox" checked={useLmEval} onChange={(e) => setUseLmEval(e.target.checked)} data-testid="opt-lmeval" /> lm-eval 任务精度（同时评未量化基线）
                  </label>
                  {useLmEval && (
                    <div className="form-row" style={{ margin: 0 }}>
                      <input value={lmTasks} onChange={(e) => setLmTasks(e.target.value)} placeholder="gsm8k,arc_easy" style={{ width: 150 }} aria-label="lm-eval 任务" data-testid="opt-lmeval-tasks" />
                      <input type="number" min={1} value={lmLimit} style={{ width: 70 }} onChange={(e) => setLmLimit(Number(e.target.value))} aria-label="样本数" />
                      <input type="number" min={0} value={lmShots} style={{ width: 60 }} onChange={(e) => setLmShots(Number(e.target.value))} aria-label="few-shot" />
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
        <p className="note">当前执行后端为 PyTorch fake-quant（评估量化误差）；选择一个部署后端后，上面的导出格式/GGUF 类型会按建议自动勾选，仍可手动调整。</p>
      </div>
      )}
      <div className="card">
        <h2>JobSpec 预览</h2>
        <pre className="log" style={{ height: "auto", maxHeight: 240 }}>
          {JSON.stringify(job, null, 2)}
        </pre>
        <button disabled={busy || blocked || picked.length === 0} onClick={submit}>
          {busy ? "提交中…" : blocked ? "该领域尚未支持提交" : "提交作业"}
        </button>
      </div>
      </div>
      <aside className="card" style={{ position: "sticky", top: 12 }} data-testid="live-estimate">
        <h2>实时预估</h2>
        <p className="note">{hw} · 并发 1 · 不加载权重</p>
        <select value={estBackend} onChange={(e) => setEstBackend(e.target.value)} data-testid="live-backend" aria-label="推理后端画像">
          {(backendList.data ?? [{ id: "hf_transformers" }]).map((b) => (
            <option key={b.id} value={b.id}>
              后端：{b.id}
            </option>
          ))}
        </select>
        {est.loading && <p className="note">计算中…</p>}
        {est.error && <div className="error-box" data-testid="live-estimate-error">{est.error}</div>}
        {est.data && (
          <table>
            <thead>
              <tr>
                <th>配方</th>
                <th>权重</th>
                <th>峰值</th>
                <th>可部署</th>
              </tr>
            </thead>
            <tbody>
              {est.data.estimates.map((e) => (
                <tr key={e.recipe_id} data-testid={`live-${e.recipe_id}`}>
                  <td>{e.recipe}</td>
                  <td>{fmtBytes(e.breakdown.weights)}</td>
                  <td>{fmtBytes(e.peak_bytes)}</td>
                  <td>{e.feasible ? "✓" : `✗ ${e.reasons[0] ?? ""}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {est.data && <p className="note">显存容量 {fmtBytes(est.data.estimates[0]?.capacity_bytes)}（利用率 0.9）。由 C++ 预估器计算，与"资源预估"页同源。</p>}
      </aside>
      </div>
    </>
  );
}

// ---- T-1304: catalog-backed model picker, replacing free-text-only entry for LLM-domain jobs ----
// Grouped two levels deep (category -> family_group, T-1303) so same-series different-size variants
// collapse into one dropdown group instead of 200+ flat rows. Picking an entry that is not available
// locally downloads it (reusing the same /api/v1/download machinery Fit.tsx's HubPanel already uses, not
// a new download path) and fills the model path once it lands.
function CatalogModelPicker({ onPick, currentPath }: { onPick: (path: string) => void; currentPath: string }) {
  const catalog = useAsync(() => api.catalog("", "", "", 500), []);
  const [picked, setPicked] = useState("");
  const [dl, setDl] = useState<DownloadStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!dl || (dl.state !== "running" && dl.state !== "queued")) return;
    const t = setInterval(() => {
      api
        .downloadStatus(dl.id)
        .then((s) => {
          setDl(s);
          if (s.state === "succeeded") onPick(s.dir);
        })
        .catch(() => {});
    }, 1000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dl?.id, dl?.state]);

  const groups = useMemo(() => {
    const byKey = new Map<string, CatalogModel[]>();
    for (const m of catalog.data?.models ?? []) {
      const key = `${m.category || "Other"} · ${m.family_group || m.name}`;
      const arr = byKey.get(key);
      if (arr) arr.push(m);
      else byKey.set(key, [m]);
    }
    return [...byKey.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [catalog.data]);

  const start = async () => {
    if (!picked) return;
    setErr(null);
    setBusy(true);
    try {
      const d = await api.download(picked);
      setDl(d);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="form-row">
      <div className="field grow">
        <label>模型目录（{catalog.data?.total ?? 0} 条，按领域 · 系列分组）</label>
        <select
          value={picked}
          onChange={(e) => {
            setPicked(e.target.value);
            setDl(null);
            setErr(null);
          }}
          data-testid="catalog-model-select"
        >
          <option value="">— 选择一个模型 —</option>
          {groups.map(([group, models]) => (
            <optgroup key={group} label={group}>
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} ({m.parameter_count}, {m.quantization})
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        {catalog.error && <div className="error-box">{catalog.error}</div>}
      </div>
      <div className="field">
        <label>&nbsp;</label>
        <button onClick={start} disabled={!picked || busy || dl?.state === "running" || dl?.state === "queued"} data-testid="catalog-download">
          {dl?.state === "succeeded" ? "已下载" : busy ? "启动中…" : "下载并使用"}
        </button>
      </div>
      {err && <div className="error-box">{err}</div>}
      {dl && (
        <div className="field grow">
          <Meter pct={dl.bytes_total ? (100 * dl.bytes_done) / dl.bytes_total : 0} />
          <span className="muted small">
            {fmtBytes(dl.bytes_done)} / {fmtBytes(dl.bytes_total)} · {dl.state}
            {dl.state === "succeeded" && ` · ${dl.dir}`}
            {dl.error && ` · ${dl.error}`}
          </span>
        </div>
      )}
      {!dl && currentPath && (
        <p className="note" style={{ width: "100%" }}>
          当前使用路径：{currentPath}
        </p>
      )}
    </div>
  );
}
