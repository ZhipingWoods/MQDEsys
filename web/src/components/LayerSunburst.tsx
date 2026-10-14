import { useMemo, useState } from "react";
import { LayerTrace } from "../api";
import { baseOption, Chart, token } from "../charts";

/**
 * Where the quantization error sits in the model, as a hierarchy: model -> block -> attention / MLP -> module. Sector size is the
 * parameter count, colour is the error (red = worse). Click a sector to zoom into it, the centre to go back.
 */
type Metric = "out_sqnr_db" | "w_sqnr_db" | "w_rel_l2";
const METRICS: { id: Metric; label: string; higherIsBetter: boolean; unit: string }[] = [
  { id: "out_sqnr_db", label: "输出 SQNR（dB，越高越好）", higherIsBetter: true, unit: "dB" },
  { id: "w_sqnr_db", label: "权重 SQNR（dB，越高越好）", higherIsBetter: true, unit: "dB" },
  { id: "w_rel_l2", label: "权重相对 L2（越低越好）", higherIsBetter: false, unit: "" },
];

export interface SunNode {
  name: string;
  /** full module name for a leaf, otherwise the path of the group */
  path: string;
  params: number;
  /** parameter-weighted mean of the metric over the leaves below (null when none of them has it) */
  value: number | null;
  children?: SunNode[];
}

const valueOf = (l: LayerTrace, m: Metric): number | null => {
  const v = l[m];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
};

/** "model.layers.3.self_attn.q_proj" -> ["block 3", "self_attn", "q_proj"]; anything outside the blocks goes under "其它". */
export function pathOf(name: string, strip = "model."): string[] {
  const n = name.startsWith(strip) ? name.slice(strip.length) : name;
  const m = /^(?:layers|blocks|h)\.(\d+)\.(.+)$/.exec(n);
  if (!m) return ["其它", n];
  const rest = m[2].split(".");
  return [`block ${m[1]}`, ...(rest.length > 1 ? [rest[0], rest.slice(1).join(".")] : rest)];
}

export function buildTree(layers: LayerTrace[], m: Metric): SunNode {
  const root: SunNode = { name: "model", path: "", params: 0, value: null, children: [] };
  for (const l of layers) {
    const parts = pathOf(l.name);
    const params = l.shape[0] * l.shape[1];
    let node = root;
    parts.forEach((p, i) => {
      const leaf = i === parts.length - 1;
      node.children ??= [];
      let c = node.children.find((x) => x.name === p && !!x.children === !leaf);
      if (!c) {
        c = { name: p, path: leaf ? l.name : parts.slice(0, i + 1).join("/"), params: 0, value: null, ...(leaf ? {} : { children: [] }) };
        node.children.push(c);
      }
      node = c;
    });
    node.params = params;
    node.value = valueOf(l, m);
  }
  const fold = (n: SunNode): void => {
    if (!n.children) return;
    n.children.forEach(fold);
    n.params = n.children.reduce((a, c) => a + c.params, 0);
    const w = n.children.filter((c) => c.value !== null);
    const tot = w.reduce((a, c) => a + c.params, 0);
    n.value = tot > 0 ? w.reduce((a, c) => a + (c.value as number) * c.params, 0) / tot : null;
  };
  fold(root);
  return root;
}

/** Red (worst) to green (best) over the range of the leaves, so the colours mean something within this model. */
function colorScale(values: number[], higherIsBetter: boolean): (v: number | null) => string {
  const s = [...values].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))];
  const lo = q(0.05), hi = q(0.95);
  return (v) => {
    if (v === null || !Number.isFinite(v)) return token("--text-muted");
    let t = hi === lo ? 0.5 : (v - lo) / (hi - lo);
    t = Math.max(0, Math.min(1, higherIsBetter ? t : 1 - t));
    const hue = 4 + t * 118; // 4 = red, 122 = green
    return `hsl(${hue.toFixed(0)}, 62%, 46%)`;
  };
}

export function LayerSunburst({ layers }: { layers: LayerTrace[] }) {
  const have = METRICS.filter((m) => layers.some((l) => valueOf(l, m.id) !== null));
  const [metric, setMetric] = useState<Metric>(have[0]?.id ?? "w_sqnr_db");
  const meta = METRICS.find((m) => m.id === metric) ?? METRICS[1];
  const tree = useMemo(() => buildTree(layers, metric), [layers, metric]);
  const color = useMemo(
    () => colorScale(layers.map((l) => valueOf(l, metric)).filter((v): v is number => v !== null), meta.higherIsBetter),
    [layers, metric, meta.higherIsBetter],
  );
  if (!layers.length || !have.length) return null;
  const blocks = (tree.children ?? []).filter((c) => c.name.startsWith("block ")).length;
  const scored = layers.flatMap((l) => (valueOf(l, metric) === null ? [] : [{ name: l.name, v: valueOf(l, metric) as number }]));
  const worstLayer = scored.length ? scored.reduce((a, b) => ((meta.higherIsBetter ? b.v < a.v : b.v > a.v) ? b : a)) : null;
  const toData = (n: SunNode): object => ({
    name: n.name,
    value: Math.max(n.params, 1),
    path: n.path,
    metric: n.value,
    params: n.params,
    itemStyle: { color: color(n.value), borderColor: token("--surface-1"), borderWidth: 1 },
    ...(n.children ? { children: n.children.map(toData) } : {}),
  });
  return (
    <div className="card" data-testid="layer-sunburst">
      <h2>误差在模型里的分布 · 块 → 注意力 / MLP → 模块</h2>
      <div className="chips" style={{ marginBottom: 8 }}>
        {have.map((m) => (
          <span key={m.id} className={`chip ${m.id === metric ? "on" : ""}`} onClick={() => setMetric(m.id)} data-testid={`sunburst-metric-${m.id}`}>
            {m.label}
          </span>
        ))}
      </div>
      <Chart
        height={460}
        option={() => {
          const o = baseOption();
          return {
            ...o,
            grid: undefined,
            legend: undefined,
            xAxis: undefined,
            yAxis: undefined,
            tooltip: {
              ...(o.tooltip as object),
              trigger: "item",
              formatter: (p: { data: { name: string; path: string; metric: number | null; params: number } }) =>
                `${p.data.path || p.data.name}<br/>参数 ${(p.data.params / 1e6).toFixed(2)} M<br/>${meta.label.split("（")[0]}：${p.data.metric === null ? "–" : `${p.data.metric.toFixed(meta.unit ? 2 : 4)}${meta.unit ? ` ${meta.unit}` : ""}`}`,
            },
            series: [
              {
                type: "sunburst",
                radius: ["12%", "96%"],
                sort: null, // keep the blocks in order (0, 1, 2 ...) rather than by size
                nodeClick: "rootToNode",
                data: [toData(tree)],
                label: { rotate: "radial", fontSize: 10, color: "#fff", minAngle: 6 },
                levels: [{}, { r0: "12%", r: "30%", label: { rotate: 0 } }, { r0: "30%", r: "52%" }, { r0: "52%", r: "74%" }, { r0: "74%", r: "96%", label: { fontSize: 9 } }],
              },
            ],
          };
        }}
      />
      <p className="small" data-testid="sunburst-summary">
        {layers.length} 个量化层 · {blocks} 个块
        {worstLayer && (
          <>
            {" "}· 最差的一层：<span className="mono" data-testid="sunburst-worst">{worstLayer.name.replace("model.", "")}</span>（{worstLayer.v.toFixed(meta.unit ? 2 : 4)}
            {meta.unit ? ` ${meta.unit}` : ""}）
          </>
        )}
      </p>
      <p className="note">
        扇区大小是参数量，颜色是{meta.label.split("（")[0]}在本模型里的相对位置（红 = 最差的 5% 一带，绿 = 最好的 5% 一带）。点一个扇区放大，点中心返回。
        父扇区的颜色是它下面各层按参数量加权的平均。
      </p>
    </div>
  );
}
