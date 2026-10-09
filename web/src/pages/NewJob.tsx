import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, fmtBytes, RuntimeParams } from "../api";
import { DEFAULT_DATASET, DEFAULT_LLM, DEFAULT_MODEL, useAsync, useDebounced } from "../hooks";

const LIVE_RUNTIME: RuntimeParams = { concurrency: 1, images_per_step: 2, text_tokens: 42, image_resolution: 448, gpu_memory_utilization: 0.9, extra_modalities: [] };

export default function NewJobPage() {
  const nav = useNavigate();
  const recipes = useAsync(api.recipes, []);
  const algos = useAsync(api.algorithms, []);
  const [algorithm, setAlgorithm] = useState("recipe");
  const profiles = useAsync(api.hardwareProfiles, []);
  const cfg = useAsync(api.config, []);
  const [kind, setKind] = useState<"vla" | "llm">("vla");
  const [name, setName] = useState("minicpm-vla-eval");
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [picked, setPicked] = useState<string[]>(["w8a16", "fp8", "w4a16", "w4a8"]);
  // LLM (wikitext-2 PPL) options
  const [seqLen, setSeqLen] = useState(512);
  const [windows, setWindows] = useState(40);
  const [actStatic, setActStatic] = useState("");
  const [kv, setKv] = useState("int8");
  const [kvMethod, setKvMethod] = useState("kivi");
  const [device, setDevice] = useState("cpu");
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

  const switchKind = (k: "vla" | "llm") => {
    setKind(k);
    setModel(k === "llm" ? DEFAULT_LLM : DEFAULT_MODEL);
    setName(k === "llm" ? "qwen-wikitext-ppl" : "minicpm-vla-eval");
    setPicked(k === "llm" ? ["w4a16", "w4a8", "w2a16"] : ["w8a16", "fp8", "w4a16", "w4a8"]);
    setNCalib(k === "llm" ? 32 : 16);
  };

  const kvSpec = kv === "" ? null : kv === "fp8" ? { format: "fp8_e4m3", bits: 8 } : { format: "int", bits: Number(kv.slice(3)), symmetric: false };
  const rules = [
    ...(keepEnds ? [{ match: "block:0", bits: 8 }, { match: "block:-1", bits: 8 }] : []),
    ...(downBits === "fp" ? [{ match: "mlp\\.down_proj$", skip: true }] : downBits ? [{ match: "mlp\\.down_proj$", bits: Number(downBits) }] : []),
  ];
  const job =
    kind === "vla"
      ? {
          metadata: { name },
          model: { source: "local", path: model, type: "vla", modality: "vla" },
          quant: { recipe: picked, algorithm, ...(action ? { also_quantize: ["action"] } : {}) },
          calibration: { dataset, n_samples: nCalib },
          evaluation: { dataset, n_samples: nEval, concurrency: [1, 4, 16, 64], attribution },
          deploy: { hardware: hw },
        }
      : {
          metadata: { name },
          model: { source: "local", path: model, type: "llm" },
          quant: {
            recipe: picked,
            algorithm,
            act_static: actStatic || null,
            ...(kvSpec ? { kv: { ...kvSpec, method: kvMethod } } : {}),
            ...(rules.length ? { mixed_precision: { rules } } : {}),
          },
          calibration: { dataset: "wikitext-2-raw-v1:train", n_samples: nCalib, seq_len: seqLen },
          evaluation: { dataset: "wikitext-2-raw-v1:test", seq_len: seqLen, max_windows: windows },
          deploy: {
            hardware: hw,
            device,
            ...(gguf.length ? { llamacpp: { types: gguf } } : {}),
            ...(useVllm
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
          ...(exportFmts.length ? { export: { formats: exportFmts } } : {}),
          ...(sens || probe
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
        指定模型 → 量化配方 → 评估数据 → 部署硬件。作业由 C++ 调度到 GPU 池（{cfg.data?.gpus.join(", ") ?? "…"}）上的 Python 执行器，全过程事件实时推送。
      </p>
      {err && <div className="error-box">{err}</div>}
      <div className="split">
      <div>
      <div className="card">
        <h2>① 模型</h2>
        <div className="tabs" style={{ marginBottom: 12 }}>
          <div className={`tab ${kind === "vla" ? "on" : ""}`} onClick={() => switchKind("vla")}>
            VLA（MiniCPM-VLA · 动作误差）
          </div>
          <div className={`tab ${kind === "llm" ? "on" : ""}`} onClick={() => switchKind("llm")}>
            LLM（HF CausalLM · wikitext-2 PPL）
          </div>
        </div>
        <div className="form-row">
          <div className="field grow">
            <label>本地模型路径（允许根目录：{cfg.data?.allowed_roots.join(", ")}）</label>
            <input value={model} onChange={(e) => setModel(e.target.value)} />
          </div>
          <div className="field">
            <label>作业名称</label>
            <input value={name} onChange={(e) => setName(e.target.value)} />
          </div>
        </div>
        <h2>② 量化配方</h2>
        <div className="chips" style={{ marginBottom: 8 }}>
          {recipes.data
            ?.filter((r) => r.weight.format !== "none")
            .map((r) => (
              <span key={r.id} className={`chip ${picked.includes(r.id) ? "on" : ""}`} onClick={() => toggle(r.id)} title={r.summary}>
                {r.name}
                {r.maturity === "experimental" && <span className="exp">实验</span>}
              </span>
            ))}
        </div>
        {kind === "vla" ? (
          <label className="small">
            <input type="checkbox" checked={action} onChange={(e) => setAction(e.target.checked)} /> 同时量化 DiT 动作头（默认保持 FP32）
          </label>
        ) : (
          <div className="form-row">
            <div className="field">
              <label>激活量化</label>
              <select value={actStatic} onChange={(e) => setActStatic(e.target.value)}>
                <option value="">动态 per-token</option>
                <option value="minmax">静态 per-tensor · minmax</option>
                <option value="ema">静态 per-tensor · EMA</option>
                <option value="percentile">静态 per-tensor · 99.99% 分位</option>
              </select>
            </div>
            <div className="field">
              <label>KV Cache 量化</label>
              <select value={kv} onChange={(e) => setKv(e.target.value)}>
                <option value="">按配方（多数为不量化）</option>
                <option value="int8">INT8</option>
                <option value="int4">INT4</option>
                <option value="int2">INT2</option>
                <option value="fp8">FP8 E4M3</option>
              </select>
            </div>
            <div className="field">
              <label>KV 方法</label>
              <select value={kvMethod} onChange={(e) => setKvMethod(e.target.value)} disabled={!kv}>
                <option value="kivi">KIVI（K per-channel / V per-token）</option>
                <option value="naive">naive（K、V 均 per-token）</option>
              </select>
            </div>
            <div className="field">
              <label>混合精度：首尾块</label>
              <select value={keepEnds ? "8" : ""} onChange={(e) => setKeepEnds(e.target.value === "8")}>
                <option value="">同配方</option>
                <option value="8">首块 + 末块 8 bit</option>
              </select>
            </div>
            <div className="field">
              <label>混合精度：down_proj</label>
              <select value={downBits} onChange={(e) => setDownBits(e.target.value)}>
                <option value="">同配方</option>
                <option value="8">8 bit</option>
                <option value="4">4 bit</option>
                <option value="fp">保持 FP（不量化）</option>
              </select>
            </div>
          </div>
        )}
        <h2 style={{ marginTop: 16 }}>③ 量化算法</h2>
        <div className="form-row">
          <div className="field" style={{ minWidth: 260 }}>
            <label>算法（所有勾选配方共用）</label>
            <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
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
        {algos.error && <div className="error-box">{algos.error}</div>}
      </div>
      <div className="card">
        <h2>④ 评估数据</h2>
        {kind === "vla" ? (
          <>
            <div className="form-row">
              <div className="field grow">
                <label>数据集（LeRobot v3 目录，或 synthetic）</label>
                <input value={dataset} onChange={(e) => setDataset(e.target.value)} />
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
              <input type="checkbox" checked={attribution} onChange={(e) => setAttribution(e.target.checked)} /> 分阶段误差归因（逐模态单独量化，定位误差来源环节）
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
                <select value={sens} onChange={(e) => setSens(e.target.value)}>
                  <option value="">不做</option>
                  <option value="block">按块（每个 decoder block）</option>
                  <option value="type">按层类型（q/k/v/o/gate/up/down）</option>
                </select>
              </div>
            </div>
            <div className="form-row">
              <div className="field">
                <label>
                  <input type="checkbox" checked={probe} onChange={(e) => setProbe(e.target.checked)} data-testid="opt-probe" /> logits 探针（量化前后输出分布的 KL 与 Top-1 一致率）
                </label>
                <label className="small">
                  <input type="checkbox" checked={probeExact} disabled={!probe} onChange={(e) => setProbeExact(e.target.checked)} data-testid="opt-probe-exact" /> 精确 KL（保存全部 logits，约 1.2 GiB / 8 窗口；默认只存 top-256，KL 为下界）
                </label>
              </div>
            </div>
            <p className="note">评估位点：pretrain（原模型）→ fake_quant_wo_kv（权重+激活量化）→ fake_quant（再加 KV Cache 量化）。PPL 为不重叠窗口的 token 级困惑度。</p>
          </>
        )}
      </div>
      <div className="card">
        <h2>⑤ 部署硬件</h2>
        <div className="form-row">
          <div className="field">
            <label>目标硬件画像（用于资源预估）</label>
            <select value={hw} onChange={(e) => setHw(e.target.value)}>
              {profiles.data?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
          {kind === "llm" && (
            <div className="field">
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
            <div className="field">
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
            <div className="field" style={{ minWidth: 320 }}>
              <label>
                <input type="checkbox" checked={useVllm} onChange={(e) => setUseVllm(e.target.checked)} data-testid="opt-vllm" /> vLLM 部署实测（需要 GPU 与 vLLM；缺失则该阶段跳过）
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
          {kind === "llm" && (
            <div className="field">
              <label>执行设备</label>
              <select value={device} onChange={(e) => setDevice(e.target.value)}>
                <option value="cpu">CPU（不占 GPU 池）</option>
                <option value="gpu">GPU 池（{cfg.data?.gpus.join(", ") ?? "…"}）</option>
              </select>
            </div>
          )}
        </div>
        <p className="note">当前执行后端为 PyTorch fake-quant（评估量化误差）；真实低比特内核部署（vLLM / llama.cpp）在后续里程碑接入。</p>
      </div>
      <div className="card">
        <h2>JobSpec 预览</h2>
        <pre className="log" style={{ height: "auto", maxHeight: 240 }}>
          {JSON.stringify(job, null, 2)}
        </pre>
        <button disabled={busy || picked.length === 0} onClick={submit}>
          {busy ? "提交中…" : "提交作业"}
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
        {est.data && <p className="note">显存容量 {fmtBytes(est.data.estimates[0]?.capacity_bytes)}（利用率 0.9）。由 C++ 预估器计算，与“资源预估”页同源。</p>}
      </aside>
      </div>
    </>
  );
}
