import { useMemo, useState } from "react";
import { api, Estimate, fmt, fmtBytes, gib, RuntimeParams } from "../api";
import { barItem, baseOption, Chart, refLine, series, token } from "../charts";
import { DEFAULT_MODEL, useAsync, useDebounced } from "../hooks";

const PARTS: { key: keyof Estimate["breakdown"]; label: string }[] = [
  { key: "weights", label: "权重" },
  { key: "qmeta", label: "量化元数据" },
  { key: "kv_cache", label: "KV Cache + 线性注意力状态" },
  { key: "activation", label: "激活" },
  { key: "runtime", label: "运行时开销" },
];

export default function EstimatePage() {
  const [model, setModel] = useState(DEFAULT_MODEL);
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

  const setR = (k: keyof RuntimeParams, v: number) => setRt({ ...rt, [k]: v });
  const toggle = (id: string) => setPicked(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id]);

  return (
    <>
      <h1>资源预估</h1>
      <p className="subtitle">不加载权重：解析 config.json 与 safetensors 头，由 C++ 预估器计算模型占用、KV Cache / 状态、并发上限与延迟下界。</p>

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

      {(est.error || spec.error) && <div className="error-box">{est.error ?? spec.error}</div>}

      {spec.data && (
        <div className="tiles">
          <Tile label="模型" value={String(spec.data.family)} hint={`${spec.data.modality?.toUpperCase()} · ${(spec.data.total_params / 1e9).toFixed(2)} B 参数`} />
          <Tile label="语言模型层" value={`${spec.data.text.n_layers}`} hint={`${spec.data.text.n_full_attention} 全注意力 + ${spec.data.text.n_linear_attention} 线性注意力`} />
          <Tile label="上下文 token" value={`${ests[0]?.context_tokens ?? "–"}`} hint={`${rt.images_per_step} 图 × ${spec.data.vision?.tokens_per_image ?? 0} + ${rt.text_tokens} 文本`} />
          <Tile label="KV / token" value={fmtBytes(ests[0]?.kv_bytes_per_token)} hint={`线性注意力状态 ${fmtBytes(ests[0]?.state_bytes_per_seq)} / 序列`} />
          <Tile label="设备可用显存" value={fmtBytes(cap)} hint={est.data?.hardware.name} />
        </div>
      )}

      {ests.length > 0 && (
        <>
          <div className="card">
            <h2>显存构成 vs 可用容量（并发 {rt.concurrency}）</h2>
            <Chart height={300} option={() => memoryStack(ests, cap)} />
            <EstimateTable ests={ests} />
          </div>
          <div className="grid-2">
            <div className="card">
              <h2>峰值显存 vs 并发数</h2>
              <Chart option={() => curve(ests, "concurrency", cap, "并发数")} />
            </div>
            <div className="card">
              <h2>峰值显存 vs 每步图像数（历史帧）</h2>
              <Chart option={() => curve(ests, "images", cap, "图像数")} />
            </div>
          </div>
          <div className="card">
            <h2>各组件权重占用（按配方）</h2>
            <Chart height={260} option={() => componentBars(ests)} />
          </div>
        </>
      )}

      {matrix.data && <MatrixCard cells={matrix.data.cells} />}

      {ests[0] && (
        <div className="card note">
          <h2>预估假设</h2>
          <ul>
            {ests[0].assumptions.map((a) => (
              <li key={a}>{a}</li>
            ))}
            <li>置信度：{ests[0].confidence}（尚无该硬件 × 后端 × 配方的实测回填时为 low）</li>
          </ul>
        </div>
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

export function Tile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="tile">
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

function memoryStack(ests: Estimate[], cap: number) {
  const o = baseOption();
  return {
    ...o,
    legend: { ...(o.legend as object), data: PARTS.map((p) => p.label) },
    grid: { ...(o.grid as object), top: 56, right: 30 },
    tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => `${v.toFixed(3)} GiB` },
    xAxis: { ...(o.xAxis as object), type: "category", data: ests.map((e) => e.recipe) },
    yAxis: { ...(o.yAxis as object), type: "value", name: "GiB", nameTextStyle: { color: token("--text-muted") }, max: Math.ceil(Math.max(gib(cap) * 1.08, ...ests.map((e) => gib(e.peak_bytes) * 1.1))) },
    series: [
      ...PARTS.map((p, i) => ({
        name: p.label,
        type: "bar",
        stack: "mem",
        barMaxWidth: 44,
        data: ests.map((e) => gib(e.breakdown[p.key])),
        itemStyle: barItem(i, i === PARTS.length - 1),
      })),
      // capacity reference: a dashed, symbol-less line across every category (recessive text ink)
      {
        name: "可用显存",
        type: "line",
        data: ests.map(() => gib(cap)),
        symbol: "none",
        silent: true,
        lineStyle: { color: token("--text-secondary"), type: "dashed", width: 1 },
        endLabel: { show: true, formatter: `可用 ${gib(cap).toFixed(1)} GiB`, color: token("--text-secondary"), fontSize: 11, offset: [-70, -12] },
        tooltip: { show: false },
      },
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

function MatrixCard({ cells }: { cells: { hardware: string; hardware_name: string; recipe: string; feasible: boolean; max_concurrency: number; latency_floor_ms: number; peak_bytes: number; capacity_bytes: number }[] }) {
  const hws = useMemo(() => [...new Set(cells.map((c) => c.hardware_name))], [cells]);
  const rs = useMemo(() => [...new Set(cells.map((c) => c.recipe))], [cells]);
  const maxC = Math.max(1, ...cells.map((c) => c.max_concurrency));
  // 5-step sequential blue (validated ramp); log scale because capacity spans orders of magnitude
  const step = (c: number) => Math.min(4, Math.floor((Math.log2(c + 1) / Math.log2(maxC + 1)) * 4.999));
  return (
    <div className="card">
      <h2>可部署矩阵（配方 × 硬件，数值 = 最大并发）</h2>
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
