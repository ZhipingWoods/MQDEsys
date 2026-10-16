// Deployment pre-evaluation, modeled on llmfit-web: SystemPanel (what-if hardware), FilterBar,
// ModelTable (fit / mode / quant / mem / speed / score), DetailPanel (score components, memory
// breakdown, notes) and a "plan" view that grades one model on every hardware profile.
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, DownloadStatus, FitLevel, FitQuery, fmt, fmtBytes, HfHit, ModelFit, probeNote } from "../api";
import { barItem, baseOption, Chart, token } from "../charts";
import { useAsync, useDebounced } from "../hooks";

const FIT_LABEL: Record<FitLevel, string> = { perfect: "完美", good: "良好", marginal: "勉强", too_tight: "放不下" };
const FIT_CLASS: Record<FitLevel, string> = { perfect: "good", good: "good", marginal: "warning", too_tight: "critical" };
const MODE_LABEL: Record<string, string> = {
  gpu: "GPU",
  tensor_parallel: "张量并行",
  moe_offload: "MoE 专家卸载",
  cpu_offload: "CPU 卸载",
  cpu_only: "纯 CPU",
};
const USE_LABEL: Record<string, string> = {
  auto: "自动", general: "通用", chat: "对话", coding: "代码", reasoning: "推理", multimodal: "多模态", vla: "VLA 具身",
};
const name = (m: ModelFit) => (m.model.startsWith("hf:") ? m.model.slice(3) : m.model.split("/").pop() ?? m.model);
const UPPER_BOUND_HINT = `屋顶线上限：未计入逐内核发射开销（实测 4.41 vs 公式 46.9 steps/s）。VLA 的一步是数千个小内核，eager 模式下发射开销主导步时。请用 mqde bench 取实测值。`;

export function FitBadge({ level }: { level: FitLevel }) {
  return <span className={`status ${FIT_CLASS[level]}`}>{FIT_LABEL[level]}</span>;
}

function ConcurrencyPanel({ model, hardware, recipe, users, context }: {
  model: string; hardware: string; recipe: string; users: number; context: number;
}) {
  const res = useAsync(
    () => api.concurrency(model, hardware, recipe, users, context),
    [model, hardware, recipe, users, context]
  );
  if (res.loading && !res.data) return <div className="muted small" style={{ padding: "8px 0" }}>并发能力计算中…</div>;
  if (res.error) return <div className="error-box">并发能力计算失败：{res.error}</div>;
  if (!res.data) return null;
  const c = res.data;
  const pct = Math.round(c.utilization_pct);
  const curve = c.curve ?? [];
  return (
    <div className="card">
      <div className="card-head">
        <h2>并发能力（{recipe}，每路 {context.toLocaleString()} tok）</h2>
        <span>
          {c.high_pressure && <span className="badge warn">高压</span>}{" "}
          {!c.fits && <span className="badge critical">当前配置放不下</span>}
        </span>
      </div>
      <div className="tiles">
        <Tile label="最大并发" value={`${c.max_users} 路`} hint={`每路 ${context.toLocaleString()} tokens`} />
        <Tile label={`${users} 路时最大上下文`} value={c.max_context_for_users > 0 ? `${c.max_context_for_users.toLocaleString()}` : "–"} hint="tokens / 路（2 的幂）" />
        <Tile label={`${users} 路显存`} value={`${fmt(c.total_gb, 1)} GiB`} hint={`容量 ${fmt(c.capacity_gb, 1)} GiB · ${pct}%`} />
        <Tile label="每路 KV" value={`${fmt(c.kv_per_user_gb * 1024, 0)} MiB`} hint={`权重 ${fmt(c.weights_gb, 1)} · 运行时 ${fmt(c.overhead_gb, 1)} GiB`} />
      </div>
      <p className="note" data-testid="resource-probe-note">{probeNote(c.resource_probe)}</p>
      {curve.length > 1 && (
        <div className="grid-2">
          <div>
            <div className="lbl" style={{ marginBottom: 4 }}>显存 vs 并发路数</div>
            <Chart
              height={240}
              option={() => {
                const o = baseOption();
                const xs = curve.map((p) => String(p.users));
                const area = (name: string, key: "weights_gb" | "kv_gb" | "overhead_gb", i: number) => ({
                  name, type: "line", stack: "mem", symbol: "none", areaStyle: { opacity: 0.85 },
                  lineStyle: { width: 0 }, color: token(`--series-${i}`), data: curve.map((p) => +p[key].toFixed(2)),
                });
                return {
                  ...o,
                  grid: { left: 48, right: 24, top: 48, bottom: 30 },
                  tooltip: { ...(o.tooltip as object), trigger: "axis" },
                  xAxis: { ...(o.xAxis as object), type: "category", data: xs, name: "路数", nameGap: 6 },
                  yAxis: { ...(o.yAxis as object), type: "value", name: "GiB", nameGap: 10 },
                  series: [
                    area("权重", "weights_gb", 1), area("运行时+激活", "overhead_gb", 4), area("KV cache", "kv_gb", 3),
                    { name: "显存容量", type: "line", symbol: "none", lineStyle: { type: "dashed", color: token("--critical"), width: 1.5 },
                      color: token("--critical"), data: curve.map(() => +c.capacity_gb.toFixed(2)) },
                  ],
                };
              }}
            />
          </div>
          <div>
            <div className="lbl" style={{ marginBottom: 4 }}>每路最大上下文 vs 并发路数</div>
            <Chart
              height={240}
              option={() => {
                const o = baseOption();
                return {
                  ...o,
                  grid: { left: 64, right: 24, top: 48, bottom: 30 },
                  tooltip: { ...(o.tooltip as object), trigger: "axis" },
                  xAxis: { ...(o.xAxis as object), type: "category", data: curve.map((p) => String(p.users)), name: "路数", nameGap: 6 },
                  yAxis: { ...(o.yAxis as object), type: "log", name: "tokens", nameGap: 10, logBase: 2, min: 256 },
                  series: [{ name: "最大上下文", type: "line", step: "end", color: token("--series-7"),
                    data: curve.map((p) => (p.max_context > 0 ? p.max_context : null)) }],
                };
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default function FitPage() {
  const profiles = useAsync(api.hardwareProfiles, []);
  const [q, setQ] = useState<FitQuery>({ hardware: "nvidia/rtx4090", gpu_count: 1, use_case: "auto", concurrency: 1, context: 0, sort: "score" });
  const [params] = useSearchParams();
  const [search, setSearch] = useState(params.get("search") ?? "");
  const [minFit, setMinFit] = useState<"all" | "runnable" | "good">("all");
  const [selected, setSelected] = useState<string | null>(null);
  const [source, setSource] = useState<"local" | "catalog">(params.get("source") === "catalog" ? "catalog" : "local");
  // Catalog search runs on the server (it can hold thousands of entries); local search filters in the page.
  const dq = useDebounced(source === "catalog" ? { ...q, source, search, limit: 60 } : { ...q, source }, 300);
  const res = useAsync(() => api.fit(dq), [JSON.stringify(dq)]);
  const cat = useAsync(() => api.catalog("", "", "", 1), []);
  const rows = useMemo(() => {
    let r = res.data?.models ?? [];
    if (search && source === "local") r = r.filter((m) => m.model.toLowerCase().includes(search.toLowerCase()));
    if (minFit === "runnable") r = r.filter((m) => m.fit_level !== "too_tight");
    if (minFit === "good") r = r.filter((m) => m.fit_level === "perfect" || m.fit_level === "good");
    return r;
  }, [res.data, search, minFit, source]);
  const sel = rows.find((m) => m.model === selected) ?? rows[0];
  useEffect(() => {
    if (!selected && rows[0]) setSelected(rows[0].model);
  }, [rows, selected]);
  const sys = res.data?.system;
  const set = <K extends keyof FitQuery>(k: K, v: FitQuery[K]) => setQ({ ...q, [k]: v });

  return (
    <>
      <div className="page-head">
        <div>
          <h1>部署预评估</h1>
          <p className="subtitle">仿 llmfit：对本地模型库逐一判断「能不能跑、用哪种量化、以什么模式、多快」，并综合打分排序。只读 config 与 safetensors 头，不加载权重。</p>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h2>目标系统</h2>
          {sys && (
            <span className="badge">
              {sys.backend.toUpperCase()} · {sys.native_formats.join(" / ")}
            </span>
          )}
        </div>
        <div className="form-row">
          <div className="field" style={{ minWidth: 230 }}>
            <label>硬件画像</label>
            <select value={q.hardware} onChange={(e) => set("hardware", e.target.value)}>
              {profiles.data?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} · {fmtBytes(p.mem_bytes)}
                </option>
              ))}
            </select>
          </div>
          <Num label="GPU 数量" value={q.gpu_count ?? 1} min={1} onChange={(v) => set("gpu_count", v)} />
          <Num label="系统内存 GiB（0=探测）" value={q.ram_gb ?? 0} min={0} onChange={(v) => set("ram_gb", v || undefined)} />
          <Num label="并发数" value={q.concurrency ?? 1} min={1} onChange={(v) => set("concurrency", v)} />
          <Num label="上下文（0=用途默认）" value={q.context ?? 0} min={0} step={1024} onChange={(v) => set("context", v)} />
          <div className="field">
            <label>用途（决定评分权重）</label>
            <select value={q.use_case} onChange={(e) => set("use_case", e.target.value)}>
              {["auto", ...(res.data?.use_cases ?? [])].map((u) => (
                <option key={u} value={u}>
                  {USE_LABEL[u] ?? u}
                </option>
              ))}
            </select>
          </div>
        </div>
        {sys && (
          <div className="tiles" style={{ marginBottom: 0 }}>
            <Tile label="显存" value={`${fmt(sys.vram_gb * sys.gpu_count, 1)} GiB`} hint={`${sys.gpu_count} × ${fmt(sys.vram_gb, 1)} GiB`} />
            <Tile label="显存带宽" value={`${fmt(sys.mem_bw_gbps, 0)} GB/s`} hint="决定解码速度（带宽受限）" />
            <Tile label="BF16 算力" value={`${fmt(sys.peak_tflops_bf16, 0)} TFLOPS`} hint="决定预填充 / TTFT" />
            <Tile label="系统内存" value={`${fmt(sys.ram_gb, 0)} GiB`} hint={sys.unified_memory ? "统一内存" : "可用于 CPU 卸载"} />
          </div>
        )}
      </div>

      <div className="card flush">
        <div className="form-row" style={{ padding: "12px 14px 0" }}>
          <div className="field">
            <label>模型来源</label>
            <div className="seg">
              <button className={source === "local" ? "on" : ""} onClick={() => setSource("local")}>本地模型库</button>
              <button className={source === "catalog" ? "on" : ""} onClick={() => setSource("catalog")}
                title={cat.data ? `${cat.data.total} 条，更新于 ${cat.data.updated_at}` : "mqde catalog update 生成"}>
                HF 模型库{cat.data ? ` · ${cat.data.total - cat.data.demoted}` : ""}
              </button>
            </div>
          </div>
          <div className="field grow">
            <label>搜索</label>
            <input placeholder={source === "catalog" ? "在 HF 模型库中搜索，如 qwen3、llama…" : "模型名…"} value={search} onChange={(e) => setSearch(e.target.value)} />
          </div>
          <div className="field">
            <label>适配</label>
            <div className="seg">
              {(["all", "runnable", "good"] as const).map((k) => (
                <button key={k} className={minFit === k ? "on" : ""} onClick={() => setMinFit(k)}>
                  {{ all: "全部", runnable: "可运行", good: "良好以上" }[k]}
                </button>
              ))}
            </div>
          </div>
          <div className="field">
            <label>排序</label>
            <div className="seg">
              {(["score", "tps", "params", "mem"] as const).map((k) => (
                <button key={k} className={q.sort === k ? "on" : ""} onClick={() => set("sort", k)}>
                  {{ score: "评分", tps: "速度", params: "参数", mem: "显存" }[k]}
                </button>
              ))}
            </div>
          </div>
        </div>
        {res.error && <div className="error-box" style={{ margin: 14 }}>{res.error}</div>}
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table>
            <thead>
              <tr>
                <th>模型</th>
                <th>类型</th>
                <th className="num">参数</th>
                <th>适配</th>
                <th>运行模式</th>
                <th>最佳量化</th>
                <th className="num">显存需求</th>
                <th>占用</th>
                <th className="num">速度</th>
                <th className="num">评分</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m) => (
                <tr key={m.model} className={`clickable ${sel?.model === m.model ? "selected" : ""}`} onClick={() => setSelected(m.model)}>
                  <td>
                    <b>{name(m)}</b>
                    {m.installed && <span className="badge good" style={{ marginLeft: 6 }} title={m.installed_providers.join(", ")}>已安装</span>}
                    {m.model.startsWith("hf:") && !m.installed && <span className="badge" style={{ marginLeft: 6 }} title="仅元数据，未下载权重">目录</span>}
                  </td>
                  <td>
                    <span className="badge">{m.modality.toUpperCase()}</span> {m.is_moe && <span className="badge accent">MoE</span>}
                  </td>
                  <td className="num">
                    {fmt(m.params_b, 2)} B{m.is_moe && <span className="muted small"> ({fmt(m.active_params_b, 1)} 活跃)</span>}
                  </td>
                  <td>
                    <FitBadge level={m.fit_level} />
                  </td>
                  <td>{MODE_LABEL[m.run_mode] ?? m.run_mode}</td>
                  <td>
                    <span className="badge accent">{m.best_quant}</span>
                  </td>
                  <td className="num">{fmt(m.memory_required_gb, 1)} GiB</td>
                  <td style={{ width: 110 }}>
                    <Meter pct={m.utilization_pct} />
                  </td>
                  <td className="num">
                    {m.speed_is_upper_bound && <span className="muted">≤ </span>}
                    {fmt(m.estimated_tps, 1)} <span className="muted small">{m.speed_unit}</span>
                    {m.measured_tps && <span className="badge good" style={{ marginLeft: 6 }} title={`实测 · ${m.measured_tps.sample_count} 次`}>实测</span>}
                    {m.speed_is_upper_bound && (
                      <span className="badge warn" style={{ marginLeft: 6 }} title={UPPER_BOUND_HINT}>
                        上限
                      </span>
                    )}
                  </td>
                  <td className="num">
                    <b>{fmt(m.score, 1)}</b>
                  </td>
                </tr>
              ))}
              {!res.loading && rows.length === 0 && (
                <tr>
                  <td colSpan={10} className="empty">
                    {source === "catalog" && cat.data?.total === 0 ? "HF 模型库为空：在服务器上运行 mqde catalog update" : "没有符合条件的模型"}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {sel && <Detail m={sel} query={dq} />}

      <HubPanel />
    </>
  );
}

// ---- HF live search + download (llmfit hf-search / download) ----
function HubPanel() {
  const [q, setQ] = useState("");
  const [dq, setDq] = useState("");
  const [nonce, setNonce] = useState(0);
  const runSearch = () => { setDq(q.trim()); setNonce((n) => n + 1); };
  const hits = useAsync(() => (dq.length >= 2 ? api.hfSearch(dq, 20) : Promise.resolve({ models: [] as HfHit[] })), [dq, nonce]);
  const [budget, setBudget] = useState("");
  const [picked, setPicked] = useState<string | null>(null);
  const groups = useAsync(
    () => (picked ? api.download(picked, { list: true, budget: budget || undefined }) : Promise.resolve(null)),
    [picked, budget],
  );
  const [jobs, setJobs] = useState<DownloadStatus[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const active = jobs.some((j) => j.state === "queued" || j.state === "running");
  useEffect(() => {
    if (!active) return;
    const t = setInterval(async () => {
      const next = await Promise.all(jobs.map((j) => (j.state === "queued" || j.state === "running" ? api.downloadStatus(j.id).catch(() => j) : j)));
      setJobs(next);
    }, 1000);
    return () => clearInterval(t);
  }, [active, jobs]);
  const start = async (quant?: string) => {
    if (!picked) return;
    setErr(null);
    try {
      const d = await api.download(picked, { quant, budget: budget || undefined });
      setJobs((js) => [d, ...js]);
    } catch (e) {
      setErr(String(e));
    }
  };
  return (
    <div className="card">
      <div className="card-head">
        <h2>HF 搜索与下载</h2>
        <span className="muted small">实时查询 huggingface.co；下载到服务器 --allow 的第一个目录，完成后出现在本地模型库</span>
      </div>
      <div className="form-row">
        <div className="field grow">
          <label>搜索 Hub</label>
          <input placeholder="如 Qwen3 8B GGUF" value={q} onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runSearch()} />
        </div>
        <div className="field">
          <label>&nbsp;</label>
          <button onClick={runSearch} disabled={q.trim().length < 2 || hits.loading}>
            {hits.loading ? "查询中…" : "查询"}
          </button>
        </div>
        <div className="field">
          <label>下载预算（如 12G，可空）</label>
          <input value={budget} onChange={(e) => setBudget(e.target.value)} style={{ width: 120 }} />
        </div>
      </div>
      {hits.error && <div className="error-box">{hits.error}</div>}
      {err && <div className="error-box">{err}</div>}
      <div className="grid-2">
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>模型</th><th className="num">参数</th><th className="num">下载量</th><th>类型</th></tr>
            </thead>
            <tbody>
              {hits.data?.models.map((h) => (
                <tr key={h.name} className={`clickable ${picked === h.name ? "selected" : ""}`} onClick={() => setPicked(h.name)}>
                  <td><b>{h.name}</b></td>
                  <td className="num">{h.parameters_raw ? `${fmt(h.parameters_raw / 1e9, 2)} B` : "–"}</td>
                  <td className="num">{h.hf_downloads.toLocaleString()}</td>
                  <td className="muted small">{h.pipeline_tag}</td>
                </tr>
              ))}
              {dq.length >= 2 && !hits.loading && hits.data?.models.length === 0 && (
                <tr><td colSpan={4} className="empty">无结果</td></tr>
              )}
            </tbody>
          </table>
        </div>
        <div>
          {!picked && <div className="muted">选择左侧模型查看可下载的权重组</div>}
          {picked && groups.loading && <div className="muted">读取文件列表…</div>}
          {picked && groups.error && <div className="error-box">{groups.error}</div>}
          {picked && groups.data?.groups && (
            <table>
              <thead><tr><th>权重组</th><th className="num">大小</th><th className="num">文件</th><th></th></tr></thead>
              <tbody>
                {groups.data.groups.map((g) => (
                  <tr key={g.group} className={g.group === groups.data?.chosen ? "selected" : ""}>
                    <td><b>{g.group}</b>{g.group === groups.data?.chosen && <span className="badge accent" style={{ marginLeft: 6 }}>推荐</span>}</td>
                    <td className="num">{fmtBytes(g.bytes)}</td>
                    <td className="num">{g.files}</td>
                    <td><button className="secondary" onClick={() => start(g.group)}>下载</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
      {jobs.length > 0 && (
        <div style={{ marginTop: 12 }}>
          {jobs.map((j) => (
            <div key={j.id} className="form-row" style={{ alignItems: "center" }}>
              <div className="grow">
                <b>{j.repo}</b> <span className="badge">{j.group}</span> <span className="muted small">{j.dir}</span>
                <Meter pct={j.bytes_total ? (100 * j.bytes_done) / j.bytes_total : 0} />
                <span className="muted small">
                  {fmtBytes(j.bytes_done)} / {fmtBytes(j.bytes_total)} · {j.files_done}/{j.files_total} 文件 · {STATE_LABEL[j.state]}
                  {j.error && ` · ${j.error}`}
                </span>
              </div>
              {(j.state === "running" || j.state === "queued") && (
                <button className="secondary" onClick={() => api.cancelDownload(j.id)}>取消</button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
const STATE_LABEL: Record<DownloadStatus["state"], string> = { queued: "排队", running: "下载中", succeeded: "完成", failed: "失败", cancelled: "已取消" };

function Detail({ m, query }: { m: ModelFit; query: FitQuery }) {
  const plan = useAsync(() => api.plan(m.model, { use_case: query.use_case, concurrency: query.concurrency, context: query.context }), [m.model, query.use_case, query.concurrency, query.context]);
  const d = m.detail;
  return (
    <div className="split">
      <div>
        <div className="card">
          <div className="card-head">
            <h2>{name(m)}</h2>
            <span>
              <FitBadge level={m.fit_level} /> <span className="badge accent">{m.best_quant}</span>
            </span>
          </div>
          <div className="grid-2">
            <div>
              <h3>评分构成</h3>
              {(["quality", "speed", "fit", "context"] as const).map((k) => (
                <div key={k} style={{ display: "grid", gridTemplateColumns: "56px 1fr 44px", gap: 10, alignItems: "center", margin: "6px 0" }}>
                  <span className="muted small">{{ quality: "质量", speed: "速度", fit: "适配", context: "上下文" }[k]}</span>
                  <Meter pct={m.score_components[k]} plain />
                  <span className="num small">{fmt(m.score_components[k], 0)}</span>
                </div>
              ))}
              <p className="small muted" style={{ marginTop: 10 }}>
                总分 {fmt(m.score, 1)} = 按用途加权（llmfit 权重表）
              </p>
            </div>
            <div>
              <h3>关键指标</h3>
              <dl className="kv">
                <dt>运行模式</dt>
                <dd>{MODE_LABEL[m.run_mode] ?? m.run_mode}</dd>
                <dt>显存需求 / 可用</dt>
                <dd>
                  {fmt(m.memory_required_gb, 2)} / {fmt(m.memory_available_gb, 1)} GiB（{fmt(m.utilization_pct, 0)}%）
                </dd>
                <dt>速度</dt>
                <dd>
                  {m.speed_is_upper_bound && <span className="muted">≤ </span>}
                  {fmt(m.estimated_tps, 1)} {m.speed_unit}
                  {m.speed_is_upper_bound && (
                    <div className="muted small" style={{ marginTop: 4 }}>
                      屋顶线上限，不是预测值：未计入逐内核发射开销。VLA 的一步是数千个小内核（视觉塔 + LM 预填充 + 多次流匹配去噪），
                      eager 模式下发射开销而非带宽决定步时——本机实测 4.41 vs 公式 46.9 steps/s。请用 <code>mqde bench</code> 取实测值。
                    </div>
                  )}
                </dd>
                {m.speed_unit === "tok/s" && (
                  <>
                    <dt>预填充 / TTFT</dt>
                    <dd>
                      {fmt(m.prefill_tps, 0)} tok/s · {fmt(m.ttft_ms, 0)} ms @ {m.context}
                    </dd>
                  </>
                )}
                {m.speed_unit === "steps/s" && (
                  <>
                    <dt>决策步延迟下界</dt>
                    <dd>
                      ≥ {fmt(m.ttft_ms, 1)} ms
                      <span className="muted small"> （与上方速度上限同源）</span>
                    </dd>
                  </>
                )}
                <dt>可用量化</dt>
                <dd>{m.fitting_quants.join(" · ") || "–"}</dd>
                <dt>置信度</dt>
                <dd>
                  <ConfidenceBadge c={m.estimate_confidence} />
                </dd>
                {m.measured_tps && (
                  <>
                    <dt>实测</dt>
                    <dd>
                      {fmt(m.measured_tps.tok_s, 2)} {m.speed_unit} · 中位数，{m.measured_tps.sample_count} 次 · {m.measured_tps.hardware_label}
                    </dd>
                    <dt>公式估算</dt>
                    <dd className="muted">{fmt(m.formula_tps, 2)} {m.speed_unit}</dd>
                  </>
                )}
                {m.estimate_basis && (
                  <>
                    <dt>估算依据</dt>
                    <dd className="muted small">
                      方法：{m.estimate_basis.method}
                      {m.estimate_basis.gpu_bandwidth_gbps > 0 && <span>  · GPU BW {fmt(m.estimate_basis.gpu_bandwidth_gbps, 0)} GB/s</span>}
                      {m.estimate_basis.efficiency > 0 && <span>  · 效率 {fmt(m.estimate_basis.efficiency * 100, 0)}%</span>}
                      {m.estimate_basis.assumed_context > 0 && <span>  · 上下文 {m.estimate_basis.assumed_context} tok</span>}
                      {m.estimate_basis.local_calibration !== 1.0 && (
                        <span>  · 本机校准系数 ×{fmt(m.estimate_basis.local_calibration, 3)}</span>
                      )}
                    </dd>
                  </>
                )}
              </dl>
            </div>
          </div>
        </div>
        <ConcurrencyPanel model={m.model} hardware={query.hardware}
          recipe={(m.best_quant || "bf16").toLowerCase()} users={query.concurrency || 1}
          context={m.context || 4096} />
        {d && (
          <div className="card">
            <h2>显存构成（{m.best_quant}）</h2>
            <Chart
              height={220}
              option={() => {
                const o = baseOption();
                const parts: [string, number][] = [
                  ["权重", d.breakdown.weights],
                  ["量化元数据", d.breakdown.qmeta],
                  ["KV / 状态", d.breakdown.kv_cache],
                  ["激活", d.breakdown.activation],
                  ["运行时", d.breakdown.runtime],
                ];
                return {
                  ...o,
                  grid: { ...(o.grid as object), left: 80, top: 36 },
                  tooltip: { ...(o.tooltip as object), trigger: "item", formatter: (p: { seriesName: string; value: number }) => `${p.seriesName}: ${fmtBytes(p.value * 2 ** 30)}` },
                  xAxis: { ...(o.xAxis as object), type: "value", axisLabel: { color: token("--text-muted"), formatter: "{value} GiB" } },
                  yAxis: { ...(o.yAxis as object), type: "category", data: ["峰值"], splitLine: { show: false } },
                  series: parts.map(([n, v], i) => ({ name: n, type: "bar", stack: "m", barWidth: 26, data: [v / 2 ** 30], itemStyle: barItem(i, false) })),
                };
              }}
            />
            <table style={{ marginTop: 6 }}>
              <thead>
                <tr>
                  <th>组件</th>
                  <th className="num">原始</th>
                  <th className="num">量化后</th>
                  <th>是否量化</th>
                </tr>
              </thead>
              <tbody>
                {d.components.map((c) => (
                  <tr key={c.name}>
                    <td>{c.name}</td>
                    <td className="num">{fmtBytes(c.original_bytes)}</td>
                    <td className="num">{fmtBytes(c.weight_bytes)}</td>
                    <td>{c.quantized ? <span className="badge accent">已量化</span> : <span className="muted small">保持原精度</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      <div>
        <div className="card">
          <h2>跨硬件规划</h2>
          <p className="small muted" style={{ marginTop: -4 }}>同一模型在每个硬件画像上的最佳方案</p>
          <table>
            <thead>
              <tr>
                <th>硬件</th>
                <th>适配</th>
                <th>量化</th>
                <th className="num">速度</th>
              </tr>
            </thead>
            <tbody>
              {plan.data?.rows.map((r) => (
                <tr key={r.hardware} className={r.hardware === query.hardware ? "selected" : ""}>
                  <td className="small">{r.system?.name}</td>
                  <td>
                    <FitBadge level={r.fit_level} />
                  </td>
                  <td>
                    <span className="badge">{r.best_quant}</span>
                  </td>
                  <td className="num small">
                    {r.speed_is_upper_bound && <span className="muted">≤ </span>}
                    {fmt(r.estimated_tps, 1)} {r.speed_unit}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card note">
          <h2>说明</h2>
          <ul style={{ paddingLeft: 18, margin: 0 }}>
            {m.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
            <li>适配等级：需求/可用 ≤60% 完美，≤85% 良好，≤98% 勉强（llmfit 阈值）。</li>
            <li>速度 = 带宽 ÷ 每 token 读取字节 × 0.55 × 运行模式系数；VLA 按每决策步读取字节与 FLOPs 的屋顶线。</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export function Meter({ pct, plain }: { pct: number; plain?: boolean }) {
  const v = Math.max(0, Math.min(100, pct));
  const cls = plain ? "" : pct > 98 ? "bad" : pct > 85 ? "warn" : "";
  // T-D: data-tip, not title= -- a native tooltip's hover timer resets whenever the attribute's value
  // changes, and this one is rewritten every second while a download is in progress, so it used to flicker
  // instead of staying put while the cursor rested on it.
  return (
    <div className={`meter ${cls}`} data-tip={`${fmt(pct, 0)}%`}>
      <i style={{ width: `${v}%` }} />
    </div>
  );
}

function Num(p: { label: string; value: number; min?: number; step?: number; onChange: (v: number) => void }) {
  return (
    <div className="field" style={{ width: 130 }}>
      <label>{p.label}</label>
      <input type="number" value={p.value} min={p.min} step={p.step ?? 1} onChange={(e) => p.onChange(Number(e.target.value))} />
    </div>
  );
}

function Tile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

const CONF_LABEL: Record<string, [string, string]> = {
  measured_local: ["本机实测", "good"],
  measured_community: ["社区实测", "good"],
  calibrated: ["本机校准", "accent"],
  estimated: ["估算（无实测）", ""],
  unsupported: ["不支持", "critical"],
};
function ConfidenceBadge({ c }: { c: string }) {
  const [label, cls] = CONF_LABEL[c] ?? [c, ""];
  return <span className={`badge ${cls}`}>{label}</span>;
}
