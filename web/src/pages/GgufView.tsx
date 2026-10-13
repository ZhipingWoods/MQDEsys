// GGUF file inspector (T-R05): header only -- metadata, the tensor table, and how the quantization types are spread over
// the blocks (a Q4_K_M file keeps some blocks' down_proj at Q6_K, which only shows per block).
import { useEffect, useMemo, useState } from "react";
import { api, fmt, fmtBytes, GgufSummary } from "../api";
import { baseOption, Chart, series, token } from "../charts";
import { useAsync } from "../hooks";
import { Tile } from "./Estimate";

const fmtParams = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n));
const SHOW = 200;

function metaText(v: unknown): string {
  if (v && typeof v === "object" && "array_of" in (v as object)) {
    const a = v as { array_of: string; length: number; head: unknown[] };
    return `${a.array_of}[${a.length.toLocaleString()}]  ${a.head.slice(0, 4).map((x) => JSON.stringify(x)).join(", ")}${a.length > 4 ? " …" : ""}`;
  }
  return typeof v === "string" ? v : JSON.stringify(v);
}

export default function GgufView() {
  const files = useAsync(api.ggufFiles, []);
  const [path, setPath] = useState("");
  const [typed, setTyped] = useState("");
  const [g, setG] = useState<GgufSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [mq, setMq] = useState("");
  const [tq, setTq] = useState("");
  const [tt, setTt] = useState("");
  const [more, setMore] = useState(false);
  useEffect(() => {
    if (!path && files.data?.[0]) setPath(files.data[0].path);
  }, [files.data, path]);
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setErr(null);
    setG(null);
    api.gguf(path).then((x) => alive && setG(x), (e: Error) => alive && setErr(e.message));
    return () => {
      alive = false;
    };
  }, [path]);
  const types = useMemo(() => Object.entries(g?.quant_types ?? {}).sort((a, b) => b[1].bytes - a[1].bytes), [g]);
  const allTypes = useMemo(() => types.map(([t]) => t), [types]);
  const meta = useMemo(() => Object.entries(g?.metadata ?? {}).filter(([k]) => k.toLowerCase().includes(mq.toLowerCase())), [g, mq]);
  const tensors = useMemo(() => (g?.tensors ?? []).filter((t) => t.name.toLowerCase().includes(tq.toLowerCase()) && (!tt || t.type === tt)), [g, tq, tt]);
  const totalBytes = g?.tensor_bytes ?? 0;
  const mix = types.slice(0, 3).map(([t, a]) => `${t} ${((100 * a.bytes) / Math.max(1, totalBytes)).toFixed(0)}%`).join(" · ");
  return (
    <>
      <div className="page-head">
        <div>
          <h1>GGUF 文件</h1>
          <p className="subtitle">只读文件头（元数据 + 张量表），不读权重：看每个块用了什么量化类型、各类型占多少字节，以及完整的张量表。</p>
        </div>
        <div className="field" style={{ minWidth: 360 }}>
          <label>GGUF 文件（允许目录下）</label>
          <select value={files.data?.some((f) => f.path === path) ? path : ""} onChange={(e) => setPath(e.target.value)} data-testid="gguf-select">
            {!files.data?.some((f) => f.path === path) && <option value="">{path || "—"}</option>}
            {files.data?.map((f) => (
              <option key={f.path} value={f.path}>
                {f.dir ? `${f.dir}/` : ""}{f.name} · {fmtBytes(f.bytes)}
              </option>
            ))}
          </select>
        </div>
        <div className="field" style={{ minWidth: 280 }}>
          <label>或输入路径</label>
          <div className="form-row" style={{ margin: 0, flexWrap: "nowrap" }}>
            <input placeholder="/path/to/model.gguf" value={typed} onChange={(e) => setTyped(e.target.value)} onKeyDown={(e) => e.key === "Enter" && typed && setPath(typed)} data-testid="gguf-path" />
            <button className="secondary" disabled={!typed} onClick={() => setPath(typed)} data-testid="gguf-open">
              查看
            </button>
          </div>
        </div>
      </div>
      {files.data?.length === 0 && !path && <p className="note" data-testid="gguf-none">允许目录下没有找到 .gguf 文件；可以在右上角输入路径。</p>}
      {err && <div className="error-box" data-testid="gguf-error">{err}</div>}
      {path && !g && !err && <p className="note">读取中…</p>}
      {g && (
        <>
          <div className="tiles" data-testid="gguf-tiles">
            <Tile label="架构" value={g.architecture || "–"} hint={`${g.name || "（无 general.name）"} · GGUF v${g.version}`} />
            <Tile label="参数量" value={fmtParams(g.n_params)} hint={`${g.n_tensors} 个张量${g.block_count != null ? ` · ${g.block_count} 块` : ""}`} />
            <Tile label="文件大小" value={fmtBytes(g.file_bytes)} hint={`张量 ${fmtBytes(g.tensor_bytes)} · 对齐 ${g.alignment}`} />
            <Tile label="量化类型" value={`${types.length} 种`} hint={mix} />
          </div>
          <div className="card table-wrap">
            <h2>量化类型分布</h2>
            <table data-testid="gguf-types">
              <thead>
                <tr><th>类型</th><th className="num">张量数</th><th className="num">参数量</th><th className="num">大小</th><th className="num">占字节</th><th className="num">bit / 权重</th></tr>
              </thead>
              <tbody>
                {types.map(([t, a]) => (
                  <tr key={t} data-testid={`gguf-type-${t}`}>
                    <td><span className="badge">{t}</span></td>
                    <td className="num">{a.tensors}</td>
                    <td className="num">{fmtParams(a.params)}</td>
                    <td className="num">{fmtBytes(a.bytes)}</td>
                    <td className="num">{((100 * a.bytes) / Math.max(1, totalBytes)).toFixed(1)}%</td>
                    <td className="num">{fmt((8 * a.bytes) / Math.max(1, a.params), 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {g.blocks.length > 0 && (
            <div className="card">
              <h2>每个块的大小与量化类型</h2>
              <Chart
                height={220}
                option={() => {
                  const o = baseOption();
                  return {
                    ...o,
                    grid: { ...(o.grid as object), left: 84 },  // byte labels ("7.63 MiB") are wider than the default margin
                    tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => fmtBytes(v) },
                    xAxis: { ...(o.xAxis as object), type: "category", data: g.blocks.map((b) => b.index), name: "块", nameLocation: "middle", nameGap: 24, nameTextStyle: { color: token("--text-muted") } },
                    yAxis: { ...(o.yAxis as object), type: "value", name: "字节", axisLabel: { formatter: (v: number) => fmtBytes(v) } },
                    series: [{ type: "bar", barMaxWidth: 22, itemStyle: { color: series(0), borderRadius: [3, 3, 0, 0] }, data: g.blocks.map((b) => b.bytes) }],
                  };
                }}
              />
              <table data-testid="gguf-blocks">
                <thead><tr><th className="num">块</th><th className="num">张量</th><th className="num">大小</th><th>类型（张量数）</th></tr></thead>
                <tbody>
                  {g.blocks.slice(0, more ? undefined : 12).map((b) => (
                    <tr key={b.index}>
                      <td className="num">{b.index}</td>
                      <td className="num">{b.tensors}</td>
                      <td className="num">{fmtBytes(b.bytes)}</td>
                      <td>{Object.entries(b.types).map(([t, n]) => <span key={t} className="badge" style={{ marginRight: 4 }}>{t} × {n}</span>)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {g.blocks.length > 12 && <button className="secondary" onClick={() => setMore(!more)} data-testid="gguf-blocks-more">{more ? "收起" : `显示全部 ${g.blocks.length} 个块`}</button>}
              {g.global.length > 0 && <p className="small muted" data-testid="gguf-global">块之外的张量：{g.global.join("、")}</p>}
            </div>
          )}
          <div className="card table-wrap">
            <div className="form-row" style={{ justifyContent: "space-between", alignItems: "flex-end" }}>
              <h2 style={{ margin: 0 }}>张量表（{tensors.length} / {g.n_tensors}）</h2>
              <div className="form-row" style={{ margin: 0 }}>
                <input placeholder="按名称过滤，如 attn_q、ffn_down" value={tq} onChange={(e) => setTq(e.target.value)} data-testid="gguf-tensor-filter" style={{ minWidth: 260 }} />
                <select value={tt} onChange={(e) => setTt(e.target.value)} data-testid="gguf-tensor-type">
                  <option value="">全部类型</option>
                  {allTypes.map((t) => <option key={t}>{t}</option>)}
                </select>
              </div>
            </div>
            <table data-testid="gguf-tensors">
              <thead><tr><th>名称</th><th>类型</th><th>形状</th><th className="num">大小</th><th className="num">偏移</th></tr></thead>
              <tbody>
                {tensors.slice(0, SHOW).map((t) => (
                  <tr key={t.name}>
                    <td className="mono small">{t.name}</td>
                    <td><span className="badge">{t.type}</span></td>
                    <td className="mono small">{t.shape.join(" × ")}</td>
                    <td className="num">{fmtBytes(t.nbytes)}</td>
                    <td className="num mono small">{t.offset.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {tensors.length > SHOW && <p className="small muted">只显示前 {SHOW} 个，用过滤框缩小范围。</p>}
          </div>
          <div className="card table-wrap">
            <div className="form-row" style={{ justifyContent: "space-between", alignItems: "flex-end" }}>
              <h2 style={{ margin: 0 }}>元数据（{meta.length} / {Object.keys(g.metadata).length}）</h2>
              <input placeholder="按键过滤，如 rope、tokenizer" value={mq} onChange={(e) => setMq(e.target.value)} data-testid="gguf-meta-filter" style={{ minWidth: 260 }} />
            </div>
            <table data-testid="gguf-meta">
              <tbody>
                {meta.map(([k, v]) => (
                  <tr key={k}>
                    <td className="mono small" style={{ whiteSpace: "nowrap" }}>{k}</td>
                    <td className="small" style={{ wordBreak: "break-all" }}>{metaText(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}
