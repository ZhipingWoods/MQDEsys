// Model structure viewer, modeled on netron: hierarchy tree (left), block graph with op-colored
// nodes (center), property panel with tensor dtype/shape, value histogram and a quantization
// preview for a chosen recipe (right). Data comes from safetensors headers; values are read on demand.
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import GgufView from "./GgufView";
import { api, fmt, fmtBytes, fmtParams, GraphNode, GraphTensor } from "../api";
import { baseOption, Chart, token } from "../charts";
import { useAsync } from "../hooks";
import { modelInDomain, useDomain } from "../domain";

const OP_VAR: Record<string, string> = {
  Attention: "--op-attention", MLP: "--op-mlp", Linear: "--op-linear", Norm: "--op-norm", Embedding: "--op-embedding",
  Conv: "--op-conv", Expert: "--op-expert", Router: "--op-router", Param: "--op-param",
};

/** 模型结构 = 两种来源：HF 目录（safetensors，按组件/块/模块的 netron 式图）和 GGUF 文件（逐块的量化类型与张量表）。 */
export default function GraphPage() {
  const [params, setParams] = useSearchParams();
  const view = params.get("view") === "gguf" ? "gguf" : "hf";
  return (
    <>
      <div className="tabs" style={{ marginBottom: 12 }}>
        <div className={`tab ${view === "hf" ? "on" : ""}`} onClick={() => setParams({}, { replace: true })} data-testid="graph-tab-hf">
          HF 模型（safetensors）
        </div>
        <div className={`tab ${view === "gguf" ? "on" : ""}`} onClick={() => setParams({ view: "gguf" }, { replace: true })} data-testid="graph-tab-gguf">
          GGUF 文件
        </div>
      </div>
      {view === "gguf" ? <GgufView /> : <HfGraph />}
    </>
  );
}

function HfGraph() {
  const lib = useAsync(api.models, []);
  const recipes = useAsync(api.recipes, []);
  const [params] = useSearchParams();
  const urlModel = useRef(params.get("model") ?? "");
  const [model, setModel] = useState(urlModel.current);
  const [hfInput, setHfInput] = useState("");
  const [focus, setFocus] = useState("");
  const [picked, setPicked] = useState<GraphNode | null>(null);
  const [previewRecipe, setPreviewRecipe] = useState("w4a16");
  const { domain } = useDomain();
  const mine = (lib.data ?? []).filter((m) => modelInDomain(m, domain));
  const others = (lib.data ?? []).filter((m) => !modelInDomain(m, domain));
  // default: the first model of the navigation bar's domain (again whenever the domain changes)
  useEffect(() => {
    if (urlModel.current) {
      urlModel.current = ""; // a ?model= link wins once
      return;
    }
    const first = mine[0] ?? lib.data?.[0];
    if (first && (!model || !mine.some((m) => m.path === model))) setModel(first.path);
  }, [lib.data, domain]); // eslint-disable-line react-hooks/exhaustive-deps
  const root = useAsync(() => (model ? api.graph(model, "", 2) : Promise.resolve(null)), [model]);
  const view = useAsync(() => (model && focus ? api.graph(model, focus, 1) : Promise.resolve(null)), [model, focus]);
  useEffect(() => {
    // default focus: first block of the language model (or of the first component with blocks)
    const comps = root.data?.children ?? [];
    const c = comps.find((x) => x.name === "language" && x.n_children) ?? comps.find((x) => (x.children ?? []).some((b) => b.kind === "block"));
    const b = c?.children?.find((x) => x.kind === "block");
    setFocus(b ? b.path : c ? c.path : "");
    setPicked(null);
  }, [root.data]);
  const meta = lib.data?.find((m) => m.path === model);
  const recipe = recipes.data?.find((r) => r.id === previewRecipe);
  const qbits = recipe ? (recipe.weight.bits ?? 16) : null;
  const fpBytes = meta?.total_bytes ?? 0;
  const qbytes = meta && qbits != null && fpBytes > 0 ? Math.round((fpBytes * qbits) / 16) : null;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>模型结构</h1>
          <p className="subtitle">组件 → 块 → 模块；选中节点查看权重、分布与量化误差。</p>
        </div>
        <div className="field" style={{ minWidth: 320 }}>
          <label>模型</label>
          <select value={model.startsWith("hf:") ? "" : model} onChange={(e) => setModel(e.target.value)}>
            {model.startsWith("hf:") && <option value="">{model}</option>}
            {[["当前领域", mine], ["其他领域", others]].map(([label, ms]) =>
              (ms as typeof mine).length ? (
                <optgroup key={label as string} label={label as string}>
                  {(ms as typeof mine).map((m) => (
                    <option key={m.path} value={m.path}>
                      {m.name} {m.family ? `(${m.family})` : ""}
                    </option>
                  ))}
                </optgroup>
              ) : null,
            )}
          </select>
        </div>
        <div className="field" style={{ minWidth: 280 }}>
          <label>或 Hub 模型（不下载权重）</label>
          <div className="form-row" style={{ margin: 0, flexWrap: "nowrap" }}>
            <input placeholder="Qwen/Qwen2.5-7B-Instruct" value={hfInput} onChange={(e) => setHfInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && hfInput && setModel(`hf:${hfInput.replace(/^hf:/, "")}`)} />
            <button className="secondary" disabled={!hfInput} onClick={() => setModel(`hf:${hfInput.replace(/^hf:/, "")}`)}>
              查看
            </button>
          </div>
        </div>
      </div>
      {meta && (
        <div className="tiles">
          <div className="tile"><div className="label">架构</div><div className="value">{meta.family}</div><div className="hint">{meta.model_type} · {meta.modality?.toUpperCase()}</div></div>
          <div className="tile"><div className="label">逻辑参数</div><div className="value">{fmtParams(meta.logical_params ?? meta.total_params ?? 0)}</div><div className="hint">{meta.active_params && meta.active_params !== meta.logical_params ? `活跃 ${fmtParams(meta.active_params)}` : "稠密"}</div></div>
          <div className="tile"><div className="label">磁盘大小</div><div className="value">{fmtBytes(meta.total_bytes)}</div><div className="hint">{meta.checkpoint_quant ? `已量化 · ${meta.checkpoint_quant.method} ${meta.checkpoint_quant.algo}` : "未量化"}</div></div>
          <div className="tile"><div className="label">组件</div><div className="value">{meta.components?.length ?? 0}</div><div className="hint">{meta.components?.map((c) => c.name).join(" · ")}</div></div>
          <div className="tile"><div className="label">量化预览</div><div className="value">{qbytes == null ? "关闭" : fmtBytes(qbytes)}</div><div className="hint">{recipe ? `${recipe.name} · 约 ${fmtBytes(fpBytes)} 的 ${Math.round((qbits! / 16) * 100)}%（按位宽估算）` : "选择配方后显示"}</div></div>
        </div>
      )}
      <div className="card form-row" style={{ alignItems: "center" }} data-testid="quant-preview-bar">
        <div className="field" style={{ minWidth: 220 }}>
          <label>节点量化预览</label>
          <select value={previewRecipe} onChange={(e) => setPreviewRecipe(e.target.value)} data-testid="quant-preview-recipe">
            <option value="">关闭</option>
            {(recipes.data ?? []).filter((r) => r.weight.format !== "none").map((r) => (
              <option key={r.id} value={r.id}>{r.name} · W{r.weight.bits ?? 16}</option>
            ))}
          </select>
        </div>
        <span className="small muted">节点内的“≈”是按权重位宽估算的量化后字节；右侧属性面板的 SQNR 来自真实张量数值。</span>
      </div>
      {root.error && <div className="error-box">{root.error}</div>}
      <div className="graph-shell">
        <div className="card tree">
          <h3>层级</h3>
          {root.data?.children?.map((c) => (
            <TreeItem key={c.path} node={c} focus={focus} onFocus={setFocus} model={model} />
          ))}
        </div>
        <div className="canvas">
          {view.data ? <BlockGraph node={view.data} picked={picked} onPick={setPicked} qbits={qbits} /> : <div className="empty">选择左侧的块</div>}
        </div>
        <Props model={model} node={picked} />
      </div>
      <div className="legend-inline" style={{ marginTop: 10 }}>
        {Object.entries(OP_VAR).map(([k, v]) => (
          <span key={k} style={{ ["--c" as string]: `var(${v})` }}>
            {k}
          </span>
        ))}
      </div>
    </>
  );
}

function TreeItem({ node, focus, onFocus, model }: { node: GraphNode; focus: string; onFocus: (f: string) => void; model: string }) {
  const [open, setOpen] = useState(node.kind === "component" && node.name === "language");
  const blocks = (node.children ?? []).filter((c) => c.kind === "block");
  const flat = (node.children ?? []).filter((c) => c.kind !== "block");
  return (
    <div>
      <div className={`row ${focus === node.path ? "on" : ""}`} onClick={() => { setOpen(!open); if (!blocks.length) onFocus(node.path); }}>
        <span className="caret">{blocks.length ? (open ? "▾" : "▸") : "•"}</span>
        <b>{node.name}</b>
        <span className="cnt">{fmtParams(node.params)}</span>
      </div>
      {open && (
        <div style={{ paddingLeft: 14 }}>
          {blocks.map((b) => (
            <div key={b.path} className={`row ${focus === b.path ? "on" : ""}`} onClick={() => onFocus(b.path)}>
              <span className="caret" />
              {b.name}
              <span className="cnt">{fmtParams(b.params)}</span>
            </div>
          ))}
          {flat.length > 0 && blocks.length > 0 && (
            <div className={`row ${focus === node.path ? "on" : ""}`} onClick={() => onFocus(node.path)}>
              <span className="caret" />
              <span className="muted">其它 {flat.length} 个模块</span>
            </div>
          )}
        </div>
      )}
      {void model}
    </div>
  );
}

/** Orders a block's modules into forward order: norms/attention first, then MLP/experts; q/k/v in parallel. */
function layout(mods: GraphNode[]): GraphNode[][] {
  const rank = (m: GraphNode) => {
    const n = m.name;
    if (/embed/.test(n)) return 0;
    if (/input_layernorm|norm1|pre_norm|layer_norm1|ln_1/.test(n)) return 1;
    if (/(q_proj|k_proj|v_proj|qkv|in_proj)/.test(n)) return 2;
    if (/(q_norm|k_norm|conv1d|A_log|dt_bias)/.test(n)) return 3;
    if (/^mlp\./.test(n) || /(gate_proj|up_proj|down_proj|fc1|fc2)/.test(n)) {
      if (/(down_proj|fc2|linear_fc2|w2)/.test(n)) return 9;
      if (/experts/.test(n)) return 8;
      return 7;
    }
    if (/(o_proj|out_proj|attn\.proj)$/.test(n)) return 4;
    if (/post_attention_layernorm|norm2|layer_norm2|ln_2/.test(n)) return 5;
    if (/(gate\.|router|shared_expert_gate)/.test(n) || n.endsWith("gate")) return 6;
    if (/(gate_proj|up_proj|fc1|linear_fc1|w1|w3)/.test(n)) return 7;
    if (/experts/.test(n)) return 8;
    if (/(down_proj|fc2|linear_fc2|w2)/.test(n)) return 9;
    return 10;
  };
  const groups = new Map<number, GraphNode[]>();
  for (const m of mods) {
    const r = rank(m);
    groups.set(r, [...(groups.get(r) ?? []), m]);
  }
  // Collapse per-expert modules ("mlp.experts.17.up_proj") into one representative row.
  return [...groups.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, g]) => {
      const experts = g.filter((m) => /experts\.\d+\./.test(m.name));
      if (experts.length > 3) {
        const params = experts.reduce((s, m) => s + m.params, 0);
        const rep: GraphNode = { ...experts[0], name: `experts × ${new Set(experts.map((m) => m.name.match(/experts\.(\d+)\./)?.[1])).size}`, params, tensors: experts[0].tensors };
        return [...g.filter((m) => !experts.includes(m)), rep];
      }
      return g;
    });
}

function BlockGraph({ node, picked, onPick, qbits }: { node: GraphNode; picked: GraphNode | null; onPick: (n: GraphNode) => void; qbits: number | null }) {
  const mods = (node.children ?? []).filter((c) => c.kind === "module");
  const rows = useMemo(() => layout(mods), [node]);
  return (
    <div className="gblock" style={{ width: "fit-content", minWidth: 340, maxWidth: "100%" }}>
      <div className="gblock-title">
        {node.name} · {fmtParams(node.params)} params · {fmtBytes(node.bytes)}
      </div>
      {rows.map((r, i) => (
        <div key={i}>
          {i > 0 && <div className="gedge" />}
          <div className="gpar" style={r.length === 1 ? {} : undefined}>
            {r.map((m) => (
              <GNode key={m.path + m.name} m={m} on={picked?.path === m.path && picked?.name === m.name} wide={r.length === 1} onPick={onPick} qbits={qbits} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function GNode({ m, on, wide, onPick, qbits }: { m: GraphNode; on: boolean; wide: boolean; onPick: (n: GraphNode) => void; qbits: number | null }) {
  const w = m.tensors?.find((t) => t.name === "weight" || t.name === "qweight") ?? m.tensors?.[0];
  const q = m.tensors?.some((t) => t.quantizable);
  const qn = qbits != null && q && w ? Math.round(((w.logical_shape.length ? w.logical_shape : w.shape).reduce((a, b) => a * b, 1) * qbits) / 8) : null;
  return (
    <div className={`gnode ${on ? "on" : ""}`} style={wide ? { width: 280 } : undefined} onClick={() => onPick(m)}>
      <div className="gh" style={{ background: `var(${OP_VAR[m.op ?? "Param"] ?? "--op-param"})` }}>
        <span>{m.op}</span>
        <span style={{ opacity: 0.8, fontWeight: 500 }}>{fmtParams(m.params)}</span>
      </div>
      <div className="gb">
        <div style={{ color: "var(--text-primary)" }}>{m.name}</div>
        {w && (
          <div>
            {w.dtype} [{(w.logical_shape.length ? w.logical_shape : w.shape).join("×")}] {q && <span className="q">· 可量化</span>}
          </div>
        )}
        {qn != null && <div className="small muted">量化后 ≈ {fmtBytes(qn)}</div>}
      </div>
    </div>
  );
}

function Props({ model, node }: { model: string; node: GraphNode | null }) {
  const [tensor, setTensor] = useState<GraphTensor | null>(null);
  const [recipe, setRecipe] = useState("w4a16");
  useEffect(() => {
    setTensor(node?.tensors?.find((t) => t.name === "weight") ?? node?.tensors?.[0] ?? null);
  }, [node]);
  const tv = useAsync(() => (tensor ? api.tensor(model, tensor.full, tensor.quantizable ? recipe : "") : Promise.resolve(null)), [model, tensor?.full, recipe]);
  if (!node) return <div className="card"><h3>属性</h3><p className="muted small">点击中间的节点查看详情</p></div>;
  const s = tv.data?.stats;
  return (
    <div className="card">
      <h3>属性</h3>
      <dl className="kv">
        <dt>算子</dt><dd>{node.op}</dd>
        <dt>路径</dt><dd className="mono" style={{ wordBreak: "break-all" }}>{node.path}</dd>
        <dt>参数</dt><dd>{fmtParams(node.params)}</dd>
        <dt>存储</dt><dd>{fmtBytes(node.bytes)}</dd>
      </dl>
      <h3 style={{ marginTop: 14 }}>张量</h3>
      <table>
        <tbody>
          {node.tensors?.map((t) => (
            <tr key={t.full} className={`clickable ${tensor?.full === t.full ? "selected" : ""}`} onClick={() => setTensor(t)}>
              <td className="mono">{t.name}</td>
              <td className="small">{t.dtype}</td>
              <td className="small mono">[{t.shape.join("×")}]</td>
            </tr>
          ))}
        </tbody>
      </table>
      {tensor && (
        <>
          <h3 style={{ marginTop: 14 }}>数值分布 {tv.data?.sampled && tv.data.sampled < tv.data.numel ? <span className="muted">（抽样 {fmtParams(tv.data.sampled)}）</span> : null}</h3>
          {tv.loading && <p className="muted small">读取中…</p>}
          {tv.data?.unsupported && <p className="muted small">{tv.data.unsupported}</p>}
          {tv.data?.hist && (
            <Chart
              height={150}
              option={() => {
                const o = baseOption();
                const h = tv.data!.hist!;
                const w = (h.max - h.min) / h.counts.length;
                return {
                  ...o,
                  grid: { left: 36, right: 8, top: 8, bottom: 22 },
                  legend: { show: false },
                  tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" } },
                  xAxis: { ...(o.xAxis as object), type: "category", data: h.counts.map((_, i) => (h.min + (i + 0.5) * w).toPrecision(2)), axisLabel: { color: token("--text-muted"), fontSize: 10, interval: 15 } },
                  yAxis: { ...(o.yAxis as object), type: "log", min: 1, axisLabel: { show: false } },
                  series: [{ type: "bar", data: h.counts.map((c) => Math.max(c, 1)), barCategoryGap: "8%", itemStyle: { color: token("--series-1"), borderRadius: [2, 2, 0, 0] } }],
                };
              }}
            />
          )}
          {s && (
            <dl className="kv">
              <dt>min / max</dt><dd>{s.min.toPrecision(4)} / {s.max.toPrecision(4)}</dd>
              <dt>mean ± std</dt><dd>{s.mean.toPrecision(3)} ± {s.std.toPrecision(3)}</dd>
              <dt>峰度</dt><dd>{fmt(s.kurtosis, 2)} {s.kurtosis > 10 && <span className="status warning">重尾</span>}</dd>
              <dt>离群比例（&gt;6σ）</dt><dd>{(s.outlier_ratio * 100).toFixed(4)}%</dd>
            </dl>
          )}
          {tensor.quantizable && (
            <>
              <h3 style={{ marginTop: 14 }}>量化误差预览</h3>
              <div className="seg" style={{ marginBottom: 8 }}>
                {["fp8", "w8a16", "w4a16", "w2a16"].map((r) => (
                  <button key={r} className={recipe === r ? "on" : ""} onClick={() => setRecipe(r)}>
                    {r.toUpperCase()}
                  </button>
                ))}
              </div>
              {tv.data?.quant_preview ? (
                <>
                  <dl className="kv">
                    <dt>SQNR</dt><dd><b>{fmt(tv.data.quant_preview.metrics.sqnr_db, 2)} dB</b></dd>
                    <dt>余弦相似度</dt><dd>{fmt(tv.data.quant_preview.metrics.cosine, 5)}</dd>
                    <dt>相对 L2</dt><dd>{fmt(tv.data.quant_preview.metrics.rel_l2, 4)}</dd>
                    <dt>最大绝对误差</dt><dd>{tv.data.quant_preview.metrics.max_abs.toPrecision(3)}</dd>
                  </dl>
                  {!tv.data.quant_preview.exact && (
                    <p className="muted small" data-testid="quant-preview-approx">
                      基于 {fmtParams(tv.data.quant_preview.rows_sampled)}/{fmtParams(tv.data.quant_preview.rows_total)} 行抽样的近似值（张量太大，没有全读）。
                    </p>
                  )}
                </>
              ) : (
                <p className="muted small">{tv.loading ? "计算中…" : "该张量不支持预览（需 2-D 浮点权重）"}</p>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
