import { useMemo, useState } from "react";
import { fmt, LayerTrace } from "../api";
import { baseOption, Chart, series, token } from "../charts";

/** Per-layer error views shared by the VLA and LLM job pages: heatmap, per-block SQNR, kurtosis scatter, Top-15. */
export function LayerAnalysis({ layers, strip = "vlm.model." }: { layers: LayerTrace[]; strip?: string }) {
  const r = { layers };
  const comps = useMemo(() => [...new Set(layers.map((l) => l.component))], [layers]);
  const [comp, setComp] = useState(comps[0] ?? "language");
  const [metric, setMetric] = useState<"out_sqnr_db" | "w_sqnr_db">("out_sqnr_db");
  const worst = useMemo(() => [...layers].filter((l) => l.out_sqnr_db != null).sort((a, b) => a.out_sqnr_db! - b.out_sqnr_db!).slice(0, 15), [layers]);
  return (
    <>
      <div className="card">
        <div className="form-row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>逐层量化误差热力图（块 × 层类型，SQNR dB，越低越差）</h2>
          <div className="form-row" style={{ margin: 0 }}>
            <select value={comp} onChange={(e) => setComp(e.target.value)}>
              {comps.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <select value={metric} onChange={(e) => setMetric(e.target.value as typeof metric)}>
              <option value="out_sqnr_db">输出 SQNR（真实校准激活）</option>
              <option value="w_sqnr_db">权重 SQNR</option>
            </select>
          </div>
        </div>
        <LayerHeatmap layers={r.layers.filter((l) => l.component === comp)} metric={metric} />
      </div>
      <div className="grid-2">
        <div className="card">
          <h2>各块平均输出 SQNR（误差沿网络的分布）</h2>
          <Chart
            option={() => {
              const o = baseOption();
              return {
                ...o,
                legend: { ...(o.legend as object), data: comps },
                grid: { ...(o.grid as object), top: 56 },
                tooltip: { ...(o.tooltip as object), trigger: "axis", valueFormatter: (v: number) => `${v?.toFixed(2)} dB` },
                xAxis: { ...(o.xAxis as object), type: "value", name: "block", nameLocation: "middle", nameGap: 24, nameTextStyle: { color: token("--text-muted") } },
                yAxis: { ...(o.yAxis as object), type: "value", name: "dB", scale: true, nameTextStyle: { color: token("--text-muted") } },
                series: comps.map((c, i) => {
                  const by = new Map<number, number[]>();
                  for (const l of r.layers)
                    if (l.component === c && l.block >= 0 && l.out_sqnr_db != null) by.set(l.block, [...(by.get(l.block) ?? []), l.out_sqnr_db]);
                  return {
                    name: c,
                    type: "line",
                    symbolSize: 8,
                    lineStyle: { width: 2, color: series(i) },
                    itemStyle: { color: series(i), borderColor: token("--surface-1"), borderWidth: 2 },
                    data: [...by.entries()].sort((a, b) => a[0] - b[0]).map(([k, v]) => [k, v.reduce((s, x) => s + x, 0) / v.length]),
                  };
                }),
              };
            }}
          />
        </div>
        <div className="card">
          <h2>误差 vs 激活峰度（离群值越重，量化越难）</h2>
          <Chart
            option={() => {
              const o = baseOption();
              const pts = r.layers.filter((l) => l.out_sqnr_db != null && l.act_kurtosis != null);
              return {
                ...o,
                legend: { ...(o.legend as object), data: comps },
                grid: { ...(o.grid as object), top: 56 },
                tooltip: {
                  ...(o.tooltip as object),
                  formatter: (p: { data: [number, number, string] }) => `${p.data[2]}<br/>峰度 ${p.data[0].toFixed(1)} · 输出 SQNR ${p.data[1].toFixed(2)} dB`,
                },
                xAxis: { ...(o.xAxis as object), type: "log", name: "激活峰度", nameLocation: "middle", nameGap: 24, nameTextStyle: { color: token("--text-muted") } },
                yAxis: { ...(o.yAxis as object), type: "value", name: "dB", scale: true, nameTextStyle: { color: token("--text-muted") } },
                series: comps.map((c, i) => ({
                  name: c,
                  type: "scatter",
                  symbolSize: 8,
                  itemStyle: { color: series(i), borderColor: token("--surface-1"), borderWidth: 1, opacity: 0.85 },
                  data: pts.filter((l) => l.component === c).map((l) => [Math.max(l.act_kurtosis!, 0.1), l.out_sqnr_db, l.name]),
                })),
              };
            }}
          />
        </div>
      </div>
      <div className="card table-wrap">
        <h2>误差最大的 15 层（候选：保持高精度 / 混合精度）</h2>
        <table>
          <thead>
            <tr>
              <th>层</th>
              <th>组件</th>
              <th className="num">块</th>
              <th className="num">形状</th>
              <th className="num">输出 SQNR</th>
              <th className="num">输出 cos</th>
              <th className="num">权重 SQNR</th>
              <th className="num">激活 |max|</th>
              <th className="num">激活峰度</th>
              <th className="num">离群比例</th>
            </tr>
          </thead>
          <tbody>
            {worst.map((l) => (
              <tr key={l.name}>
                <td className="mono small">{l.name.replace(strip, "")}</td>
                <td>{l.component}</td>
                <td className="num">{l.block}</td>
                <td className="num">
                  {l.shape[0]}×{l.shape[1]}
                </td>
                <td className="num">{fmt(l.out_sqnr_db, 2)} dB</td>
                <td className="num">{fmt(l.out_cosine, 4)}</td>
                <td className="num">{fmt(l.w_sqnr_db, 2)} dB</td>
                <td className="num">{fmt(l.act_absmax, 1)}</td>
                <td className="num">{fmt(l.act_kurtosis, 1)}</td>
                <td className="num">{l.act_outlier_ratio != null ? `${(l.act_outlier_ratio * 100).toFixed(3)}%` : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function roleOf(name: string): string {
  const parts = name.split(".");
  const tail = parts.slice(-2).join(".");
  return tail.replace(/^(self_attn|mlp|linear_attn|attn1|ff\.net|net)\./, "").replace(/^\d+\./, "");
}

function LayerHeatmap({ layers, metric }: { layers: LayerTrace[]; metric: "out_sqnr_db" | "w_sqnr_db" }) {
  const rows = [...new Set(layers.map((l) => roleOf(l.name)))];
  const blocks = [...new Set(layers.map((l) => l.block))].sort((a, b) => a - b);
  const vals = layers.map((l) => l[metric]).filter((v): v is number => v != null && isFinite(v));
  if (!vals.length) return <p className="muted">无数据</p>;
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const shown = layers.filter((l) => l[metric] != null && isFinite(l[metric] as number));
  return (
    <Chart
      height={70 + rows.length * 26}
      option={() => {
        const o = baseOption();
        return {
          ...o,
          grid: { left: 130, right: 70, top: 10, bottom: 34 },
          tooltip: {
            ...(o.tooltip as object),
            formatter: (p: { data: number[] }) => `${shown[p.data[3]].name}<br/>${metric === "out_sqnr_db" ? "输出" : "权重"} SQNR ${p.data[2].toFixed(2)} dB`,
          },
          xAxis: { ...(o.xAxis as object), type: "category", data: blocks.map((b) => (b < 0 ? "—" : String(b))), name: "block", nameLocation: "middle", nameGap: 22, nameTextStyle: { color: token("--text-muted") } },
          yAxis: { ...(o.yAxis as object), type: "category", data: rows, splitLine: { show: false } },
          visualMap: {
            min: lo,
            max: hi,
            calculable: false,
            orient: "vertical",
            right: 0,
            top: "middle",
            itemHeight: 120,
            text: [`${hi.toFixed(1)} dB`, `${lo.toFixed(1)} dB`],
            textStyle: { color: token("--text-muted"), fontSize: 11 },
            // low SQNR (bad) = darkest step so the worst layers stand out
            inRange: { color: [token("--seq-4"), token("--seq-3"), token("--seq-2"), token("--seq-1"), token("--seq-0")] },
          },
          series: [
            {
              type: "heatmap",
              // numeric rows only (ECharts drops mixed string dimensions): [block, role, value, layer idx]
              data: shown.map((l, i) => [blocks.indexOf(l.block), rows.indexOf(roleOf(l.name)), l[metric] as number, i]),
              itemStyle: { borderColor: token("--surface-1"), borderWidth: 2, borderRadius: 3 },
            },
          ],
        };
      }}
    />
  );
}

