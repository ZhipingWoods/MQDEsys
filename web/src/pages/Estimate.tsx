import { useEffect, useMemo, useRef, useState } from "react";
import { api, Estimate, fmt, fmtBytes, gib, probeNote, RuntimeParams } from "../api";
import { barItem, baseOption, Chart, refLine, series, token } from "../charts";
import { DEFAULT_MODEL, useAsync, useDebounced } from "../hooks";
import { domainDefaultModel, familyLabel, useDomain } from "../domain";
import { useSearchParams } from "react-router-dom";
import { Icon, IconName } from "../components/Icon";

const PARTS: { key: keyof Estimate["breakdown"]; label: string }[] = [
  { key: "weights", label: "权重" },
  { key: "qmeta", label: "量化元数据" },
  { key: "kv_cache", label: "KV Cache + 线性注意力状态" },
  { key: "activation", label: "激活" },
  { key: "runtime", label: "运行时开销" },
];

export default function EstimatePage() {
  const { domain } = useDomain();
  const [params] = useSearchParams();
  const [model, setModel] = useState(() => params.get("model") || domainDefaultModel(domain) || DEFAULT_MODEL);
  const firstDomain = useRef(domain);
  useEffect(() => {
    if (firstDomain.current === domain) return; // keep a model the user typed until the domain actually changes
    firstDomain.current = domain;
    const m = domainDefaultModel(domain);
    if (m) setModel(m);
  }, [domain]);
  const [hw, setHw] = useState("nvidia/rtx4090");
  const [backend, setBackend] = useState("hf_transformers");
  const backends = useAsync(api.backends, []);
  const [picked, setPicked] = useState<string[]>(["bf16", "w8a16", "fp8", "w4a16", "w4a8", "w2a16"]);
  const [rt, setRt] = useState<RuntimeParams>({
    concurrency: 1,
    images_per_step: 2,
    text_tokens: 42,
    image_resolution: 448,
    gpu_memory_utilization: 0.9,
    extra_modalities: [],
  });
  const recipes = useAsync(api.recipes, []);
  const profiles = useAsync(api.hardwareProfiles, []);
  const q = useDebounced({ model, hw, picked, rt, backend }, 300);
  const spec = useAsync(() => api.inspect(q.model), [q.model]);
  const est = useAsync(() => api.estimate(q.model, q.hw, q.picked, q.rt, q.backend), [JSON.stringify(q)]);
  const matrix = useAsync(() => api.matrix(q.model, q.rt), [q.model, JSON.stringify(q.rt)]);
  const ests = est.data?.estimates ?? [];
  const cap = ests[0]?.capacity_bytes ?? 0;
  const best = useMemo(() => {
    const feasible = ests.filter((e) => e.feasible);
    // 先排除实验配方，避免推荐 W2A16 这类仅用于研究的方案；都不可行时才退回全部候选。
    const pool = feasible.length ? feasible : ests;
    const production = pool.filter((e) => e.recipe_detail?.maturity === "production");
    const preferred = production.length ? production : pool;
    return [...preferred].sort((a, b) => b.max_concurrency - a.max_concurrency)[0];
  }, [est.data]);

  const setR = (k: keyof RuntimeParams, v: number) => setRt({ ...rt, [k]: v });
  const toggle = (id: string) => setPicked(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id]);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>资源预估</h1>
          <p className="subtitle">不加载权重，直接判断这套模型 × 配方 × 硬件放不放得下。</p>
        </div>
      </div>

      <div className="card">
        <div className="form-row">
          <div className="field grow">
            <label>模型：本地目录，或 hf:org/name（只读 config + safetensors 头，不下载权重）</label>
            <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="hf:Qwen/Qwen2.5-7B-Instruct" />
          </div>
          <div className="field">
            <label>目标硬件</label>
            <select value={hw} onChange={(e) => setHw(e.target.value)}>
              {profiles.data?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({fmtBytes(p.mem_bytes)})
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>推理后端画像</label>
            <select value={backend} onChange={(e) => setBackend(e.target.value)} data-testid="est-backend">
              {(backends.data ?? [{ id: "hf_transformers", activation_model: "per_seq" }]).map((b) => (
                <option key={b.id} value={b.id}>
                  {b.id}（{b.activation_model === "batched" ? "按批计激活" : "按序列计激活"}）
                </option>
              ))}
            </select>
          </div>
          <Num label="并发数" value={rt.concurrency} min={1} onChange={(v) => setR("concurrency", v)} />
          <Num label="每步图像数（视角×帧）" value={rt.images_per_step} min={1} onChange={(v) => setR("images_per_step", v)} />
          <Num label="指令 token" value={rt.text_tokens} min={1} onChange={(v) => setR("text_tokens", v)} />
          <Num label="显存利用率" value={rt.gpu_memory_utilization} step={0.05} min={0.1} onChange={(v) => setR("gpu_memory_utilization", v)} />
          <div className="field">
            <label>动作头</label>
            <label className="small">
              <input
                type="checkbox"
                checked={rt.extra_modalities.includes("action")}
                onChange={(e) => setRt({ ...rt, extra_modalities: e.target.checked ? ["action"] : [] })}
              />{" "}
              也量化 DiT 动作头
            </label>
          </div>
        </div>
        <div className="field">
          <label>量化配方</label>
          <div className="chips">
            {recipes.data?.map((r) => (
              <span key={r.id} className={`chip ${picked.includes(r.id) ? "on" : ""}`} onClick={() => toggle(r.id)} title={r.summary}>
                {r.name}
                {r.maturity === "experimental" && <span className="exp">实验</span>}
              </span>
            ))}
          </div>
        </div>
      </div>

      {est.data && (
        <p className="note" data-testid="resource-probe-note">
          {probeNote(est.data.resource_probe)}
        </p>
      )}

      {(est.error || spec.error) && <div className="error-box">{est.error ?? spec.error}</div>}

      {spec.data && (
        <div className="tiles">
          {best && <Tile label="推荐配方" value={best.recipe} hint={best.feasible ? `最大并发 ${best.max_concurrency} · 峰值 ${fmtBytes(best.peak_bytes)}` : "当前配置都放不下"} />}
          <Tile label="模型" value={familyLabel(String(spec.data.family))} hint={`${spec.data.modality?.toUpperCase()} · ${(spec.data.total_params / 1e9).toFixed(2)} B 参数`} />
          <Tile label="语言模型层" value={`${spec.data.text.n_layers}`} hint={`${spec.data.text.n_full_attention} 全注意力 + ${spec.data.text.n_linear_attention} 线性注意力`} />
          <Tile label="上下文 token" value={`${ests[0]?.context_tokens ?? "–"}`} hint={`${rt.images_per_step} 图 × ${spec.data.vision?.tokens_per_image ?? 0} + ${rt.text_tokens} 文本`} />
          <Tile label="设备可用显存" value={fmtBytes(cap)} hint={est.data?.hardware.name} />
        </div>
      )}

      {ests.length > 0 && (
        <>
          <div className="card">
            <h2>显存构成（并发 {rt.concurrency}）</h2>
            <p className="note">可用显存 {fmtBytes(cap)}；可部署性与峰值占比见下表。</p>
            <Chart height={300} option={() => memoryStack(ests)} />
            <EstimateTable ests={ests} />
          </div>
          <details className="card details-card">
            <summary>详细曲线与组件拆分</summary>
            <div className="grid-2">
              <div>
                <h3>峰值显存 vs 并发数</h3>
                <Chart option={() => curve(ests, "concurrency", cap, "并发数")} />
              </div>
              <div>
                <h3>峰值显存 vs 每步图像数</h3>
                <Chart option={() => curve(ests, "images", cap, "图像数")} />
              </div>
            </div>
            <h3>各组件权重占用（按配方）</h3>
            <Chart height={260} option={() => componentBars(ests)} />
          </details>
        </>
      )}

      {matrix.data && (
        <details className="card details-card">
          <summary>可部署矩阵（配方 × 硬件）</summary>
          <MatrixCard cells={matrix.data.cells} embedded />
        </details>
      )}

      {ests[0] && (
        <details className="card details-card">
          <summary>预估假设与置信度</summary>
          <ul>
            {ests[0].assumptions.map((a) => (
              <li key={a}>{a}</li>
            ))}
            <li>置信度：{ests[0].confidence}（尚无该硬件 × 后端 × 配方的实测回填时为 low）</li>
          </ul>
        </details>
      )}
    </>
  );
}

function Num(p: { label: string; value: number; min?: number; step?: number; onChange: (v: number) => void }) {
  return (
    <div className="field" style={{ width: 120 }}>
      <label>{p.label}</label>
      <input type="number" value={p.value} min={p.min} step={p.step ?? 1} onChange={(e) => p.onChange(Number(e.target.value))} />
    </div>
  );
}

export function Tile({ label, value, hint, icon }: { label: string; value: string; hint?: string; icon?: IconName }) {
  return (
    <div className="tile">
      {icon && (
        <span className="tile-icon" aria-hidden="true">
          <Icon name={icon} size={16} />
        </span>
      )}
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

function EstimateTable({ ests }: { ests: Estimate[] }) {
  return (
    <div className="table-wrap" style={{ marginTop: 10 }}>
      <table>
        <thead>
          <tr>
            <th>配方</th>
            <th className="num">权重</th>
            <th className="num">KV+状态</th>
            <th className="num">峰值</th>
            <th>可部署</th>
            <th className="num">最大并发</th>
            <th className="num">延迟下界</th>
            <th>说明</th>
          </tr>
        </thead>
        <tbody>
          {ests.map((e) => (
            <tr key={e.recipe}>
              <td>
                <b>{e.recipe}</b> <span className="muted small">{e.recipe_detail.summary}</span>
              </td>
              <td className="num">{fmtBytes(e.breakdown.weights + e.breakdown.qmeta)}</td>
              <td className="num">{fmtBytes(e.breakdown.kv_cache)}</td>
              <td className="num">{fmtBytes(e.peak_bytes)}</td>
              <td>{e.feasible ? <span className="status good">可部署</span> : <span className="status critical">超出</span>}</td>
              <td className="num">{e.max_concurrency}</td>
              <td className="num">{fmt(e.perf.latency_floor_ms, 1)} ms</td>
              <td className="small muted" style={{ whiteSpace: "normal", minWidth: 260 }}>
                {e.reasons.join("；") || "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function memoryStack(ests: Estimate[]) {
  const o = baseOption();
  return {
    ...o,
    legend: { ...(o.legend as object), data: PARTS.map((p) => p.label) },
    grid: { ...(o.grid as object), top: 56, right: 30 },
    tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => `${v.toFixed(3)} GiB` },
    xAxis: { ...(o.xAxis as object), type: "category", data: ests.map((e) => e.recipe) },
    yAxis: {
      ...(o.yAxis as object),
      type: "value",
      name: "GiB",
      nameTextStyle: { color: token("--text-muted") },
      max: Math.ceil(Math.max(...ests.map((e) => gib(e.peak_bytes)), 0.5) * 1.25 * 10) / 10,
    },
    series: [
      ...PARTS.map((p, i) => ({
        name: p.label,
        type: "bar",
        stack: "mem",
        barMaxWidth: 52,
        label: i === PARTS.length - 1
          ? {
              show: true,
              position: "top",
              color: token("--text-secondary"),
              fontSize: 11,
              formatter: (d: { dataIndex: number }) => `${gib(ests[d.dataIndex].peak_bytes).toFixed(2)} GiB`,
            }
          : undefined,
        data: ests.map((e) => gib(e.breakdown[p.key])),
        itemStyle: barItem(i, i === PARTS.length - 1),
      })),
    ],
  };
}

function curve(ests: Estimate[], key: "concurrency" | "images", cap: number, xname: string) {
  const o = baseOption();
  return {
    ...o,
    legend: { ...(o.legend as object), data: ests.map((e) => e.recipe) },
    grid: { ...(o.grid as object), top: 56 },
    tooltip: { ...(o.tooltip as object), trigger: "axis", valueFormatter: (v: number) => `${v.toFixed(2)} GiB` },
    xAxis: { ...(o.xAxis as object), type: "log", name: xname, nameLocation: "middle", nameGap: 24, nameTextStyle: { color: token("--text-muted") }, logBase: 2 },
    yAxis: { ...(o.yAxis as object), type: "value", name: "GiB", nameTextStyle: { color: token("--text-muted") } },
    series: ests.map((e, i) => ({
      name: e.recipe,
      type: "line",
      showSymbol: true,
      symbolSize: 8,
      lineStyle: { width: 2, color: series(i) },
      itemStyle: { color: series(i), borderColor: token("--surface-1"), borderWidth: 2 },
      data: e.curves[key].map((p) => [p.x, gib(p.peak_bytes)]),
      markLine: i === 0 ? refLine(gib(cap), "可用显存") : undefined,
    })),
  };
}

function componentBars(ests: Estimate[]) {
  const o = baseOption();
  const comps = ests[0].components.filter((c) => c.name !== "other").map((c) => c.name);
  return {
    ...o,
    legend: { ...(o.legend as object), data: ests.map((e) => e.recipe) },
    grid: { ...(o.grid as object), top: 56 },
    tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => `${(v * 1024).toFixed(0)} MiB` },
    xAxis: { ...(o.xAxis as object), type: "category", data: comps },
    yAxis: { ...(o.yAxis as object), type: "value", name: "GiB", nameTextStyle: { color: token("--text-muted") } },
    series: ests.map((e, i) => ({
      name: e.recipe,
      type: "bar",
      barGap: "15%",
      barMaxWidth: 22,
      itemStyle: { color: series(i), borderRadius: [4, 4, 0, 0], borderColor: token("--surface-1"), borderWidth: 1 },
      data: comps.map((c) => gib(e.components.find((x) => x.name === c)?.weight_bytes ?? 0)),
    })),
  };
}

function MatrixCard({ cells, embedded = false }: { cells: { hardware: string; hardware_name: string; recipe: string; feasible: boolean; max_concurrency: number; latency_floor_ms: number; peak_bytes: number; capacity_bytes: number }[]; embedded?: boolean }) {
  const hws = useMemo(() => [...new Set(cells.map((c) => c.hardware_name))], [cells]);
  const rs = useMemo(() => [...new Set(cells.map((c) => c.recipe))], [cells]);
  const maxC = Math.max(1, ...cells.map((c) => c.max_concurrency));
  // 5-step sequential blue (validated ramp); log scale because capacity spans orders of magnitude
  const step = (c: number) => Math.min(4, Math.floor((Math.log2(c + 1) / Math.log2(maxC + 1)) * 4.999));
  return (
    <div className={embedded ? "" : "card"}>
      {!embedded && <h2>可部署矩阵（配方 × 硬件，数值 = 最大并发）</h2>}
      <div className="table-wrap">
        <table className="matrix">
          <thead>
            <tr>
              <th>硬件</th>
              {rs.map((r) => (
                <th key={r} className="num">
                  {r}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {hws.map((h) => (
              <tr key={h}>
                <td>{h}</td>
                {rs.map((r) => {
                  const c = cells.find((x) => x.hardware_name === h && x.recipe === r);
                  if (!c) return <td key={r} />;
                  const s = c.feasible ? step(c.max_concurrency) : -1;
                  return (
                    <td
                      key={r}
                      className={`num cell ${s >= 3 ? "dark" : ""}`}
                      style={{ background: s < 0 ? "transparent" : `var(--seq-${s})` }}
                      title={`${c.recipe} · ${c.hardware_name}\n峰值 ${fmtBytes(c.peak_bytes)} / 可用 ${fmtBytes(c.capacity_bytes)}\n延迟下界 ${c.latency_floor_ms.toFixed(1)} ms`}
                    >
                      {c.feasible ? c.max_concurrency : <span className="status critical">超出</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">颜色越深 = 可支持的并发越多（对数刻度）；“超出” = 单并发即超出显存。悬停查看峰值 / 可用显存与延迟下界。数值为预估，实测后由反馈环节校准。</p>
    </div>
  );
}
