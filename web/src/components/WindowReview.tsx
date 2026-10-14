import { useMemo, useState } from "react";
import { baseOption, Chart, token } from "../charts";
import { signedPct } from "../semantics";

/**
 * Per-window review of a PPL evaluation (the counterpart of evalscope's per-sample review): every evaluation window's own mean
 * loss, unquantized against quantized. A recipe can lose little on average and a lot on a few windows; this is where that shows.
 */
export interface WindowSeries {
  name: string;
  nll: number[];
}
interface Props {
  base: number[];
  recipes: WindowSeries[];
  previews?: string[];
  seqLen: number;
}

const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b);
  return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : 0;
};

/** Histogram of the per-window loss change: bin edges chosen so that zero is an edge, which makes better and worse windows separate bars. */
export function deltaHistogram(d: number[], bins = 14): { lo: number; hi: number; count: number }[] {
  if (!d.length) return [];
  const mn = Math.min(...d, 0), mx = Math.max(...d, 0);
  if (mx === mn) return [{ lo: mn, hi: mx, count: d.length }];
  const neg = mn < 0 ? Math.max(1, Math.round((bins * -mn) / (mx - mn))) : 0;
  const pos = mx > 0 ? Math.max(1, bins - neg) : 0;
  const out: { lo: number; hi: number; count: number }[] = [];
  for (let i = 0; i < neg; i++) out.push({ lo: mn + (i * -mn) / neg, hi: mn + ((i + 1) * -mn) / neg, count: 0 });
  for (let i = 0; i < pos; i++) out.push({ lo: (i * mx) / pos, hi: ((i + 1) * mx) / pos, count: 0 });
  for (const v of d) {
    const k = out.findIndex((b, i) => (v >= b.lo && v < b.hi) || (i === out.length - 1 && v >= b.lo));
    out[k < 0 ? out.length - 1 : k].count++;
  }
  return out;
}

export function WindowReview({ base, recipes, previews, seqLen }: Props) {
  const [which, setWhich] = useState(recipes[0]?.name ?? "");
  const cur = recipes.find((r) => r.name === which) ?? recipes[0];
  const d = useMemo(() => (cur ? cur.nll.map((q, i) => q - base[i]) : []), [cur, base]);
  const worse = d.filter((x) => x > 0).length;
  const order = useMemo(() => d.map((_, i) => i).sort((a, b) => d[b] - d[a]), [d]);
  const top = order.slice(0, 8);
  const hist = useMemo(() => deltaHistogram(d), [d]);
  if (!cur || !base.length || cur.nll.length !== base.length) return null;
  const lo = Math.min(...base, ...cur.nll), hi = Math.max(...base, ...cur.nll);
  const pad = (hi - lo) * 0.04 || 0.1;
  const label = (i: number) => `窗口 ${i}${previews?.[i] ? `：${previews[i]}` : ""}`;
  const ppx = (x: number) => (Math.exp(x) - 1) * 100; // a per-window loss change, as the PPL change it alone would make
  return (
    <div className="card" data-testid="window-review">
      <h2>逐窗口复盘 · 量化损失集中在哪些文本上</h2>
      {recipes.length > 1 && (
        <div className="chips" style={{ marginBottom: 8 }}>
          {recipes.map((r) => (
            <span key={r.name} className={`chip ${r.name === cur.name ? "on" : ""}`} onClick={() => setWhich(r.name)} data-testid={`window-recipe-${r.name}`}>
              {r.name}
            </span>
          ))}
        </div>
      )}
      <div className="tiles" data-testid="window-tiles">
        <div className="tile">
          <div className="label">变差的窗口</div>
          <div className="value" data-testid="window-worse">{worse} / {d.length}</div>
          <div className="hint">量化后损失比原模型高</div>
        </div>
        <div className="tile">
          <div className="label">窗口损失变化的中位数</div>
          <div className="value" data-testid="window-median">{median(d) >= 0 ? "+" : ""}{median(d).toFixed(4)}</div>
          <div className="hint">nats / token（≈ {signedPct(ppx(median(d)))} 的 PPL）</div>
        </div>
        <div className="tile">
          <div className="label">最差的窗口</div>
          <div className="value" data-testid="window-worst">{d[order[0]] >= 0 ? "+" : ""}{d[order[0]].toFixed(4)}</div>
          <div className="hint">窗口 {order[0]} · ≈ {signedPct(ppx(d[order[0]]), 1)}</div>
        </div>
      </div>
      <div className="grid-2">
        <div>
          <h3 className="small">每个窗口：原模型 vs {cur.name}（点在对角线上方 = 量化后更差）</h3>
          <Chart
            height={300}
            option={() => {
              const o = baseOption();
              const pts = (sign: 1 | -1) => cur.nll.map((q, i) => ({ value: [base[i], q, i], d: q - base[i] })).filter((p) => (sign > 0 ? p.d > 0 : p.d <= 0));
              const item = (name: string, color: string, sign: 1 | -1) => ({ name, type: "scatter", symbolSize: 9, itemStyle: { color, opacity: 0.85 }, data: pts(sign) });
              return {
                ...o,
                legend: { ...(o.legend as object), data: ["变差", "没变差"] },
                tooltip: {
                  ...(o.tooltip as object),
                  trigger: "item",
                  formatter: (p: { data: { value: number[]; d: number } }) =>
                    `${label(p.data.value[2])}<br/>原模型 ${p.data.value[0].toFixed(4)} · 量化后 ${p.data.value[1].toFixed(4)}<br/>变化 ${p.data.d >= 0 ? "+" : ""}${p.data.d.toFixed(4)}`,
                },
                xAxis: { ...(o.xAxis as object), type: "value", min: lo - pad, max: hi + pad, name: "原模型 损失 (nats/token)", nameLocation: "middle", nameGap: 28 },
                yAxis: { ...(o.yAxis as object), type: "value", min: lo - pad, max: hi + pad, name: `${cur.name} 损失` },
                series: [
                  {
                    ...item("变差", token("--critical"), 1),
                    markLine: { silent: true, symbol: "none", lineStyle: { color: token("--text-muted"), type: "dashed" }, label: { show: false }, data: [[{ coord: [lo - pad, lo - pad] }, { coord: [hi + pad, hi + pad] }]] },
                  },
                  item("没变差", token("--good"), -1),
                ],
              };
            }}
          />
        </div>
        <div>
          <h3 className="small">窗口损失变化的分布（右侧 = 量化后更差）</h3>
          <Chart
            height={300}
            option={() => {
              const o = baseOption();
              return {
                ...o,
                tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" } },
                xAxis: { ...(o.xAxis as object), type: "category", data: hist.map((b) => ((b.lo + b.hi) / 2).toFixed(3)), name: "损失变化 (nats/token)", nameLocation: "middle", nameGap: 28 },
                yAxis: { ...(o.yAxis as object), type: "value", name: "窗口数", minInterval: 1 },
                series: [{ type: "bar", barMaxWidth: 30, data: hist.map((b) => ({ value: b.count, itemStyle: { color: b.hi <= 0 ? token("--good") : token("--critical") } })) }],
              };
            }}
          />
        </div>
      </div>
      <table data-testid="window-worst-table">
        <thead>
          <tr>
            <th>窗口</th>
            <th>文本（开头）</th>
            <th>原模型</th>
            <th>{cur.name}</th>
            <th>变化</th>
            <th>≈ PPL</th>
          </tr>
        </thead>
        <tbody>
          {top.map((i) => (
            <tr key={i} data-testid={`window-row-${i}`}>
              <td>{i}</td>
              <td title={previews?.[i]} style={{ maxWidth: 420, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {previews?.[i] ?? <span className="muted">（这份报告没有记录窗口文本）</span>}
              </td>
              <td>{base[i].toFixed(4)}</td>
              <td>{cur.nll[i].toFixed(4)}</td>
              <td>{d[i] >= 0 ? "+" : ""}{d[i].toFixed(4)}</td>
              <td>{signedPct(ppx(d[i]), 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        每个窗口是 {seqLen} 个 token，损失是该窗口的平均交叉熵（nats/token）。整体 PPL 是各窗口损失均值的 exp，所以少数窗口的大幅恶化会被平均掉；
        表里是变化最大的 {top.length} 个窗口。
      </p>
    </div>
  );
}
