import { useMemo, useState } from "react";
import { MATRIX_METRICS, MetricSpec, relChange, signedPct, toneOf } from "../semantics";

/**
 * Runs and recipes side by side against a baseline of your choice (the counterpart of evalscope's score matrix): the absolute numbers,
 * or each one's change against the baseline coloured by whether it is better or worse for THAT metric (a lower PPL is better, a higher
 * throughput is better). Only runs of the same model scored under the same protocol are put together: PPL is not comparable otherwise.
 */
export interface MatrixEntry {
  key: string;
  label: string;
  model: string;
  /** how PPL was scored, e.g. "40 窗口 × 512": entries with different protocols are never in one table */
  protocol: string;
  metrics: Partial<Record<string, number>>;
  /** the serving engine the throughput came from: vLLM (eager) and SGLang (CUDA graphs) are not comparable with each other */
  engine?: string;
}

const FP = "__fp__";
const COLOR = { good: "var(--good-text)", bad: "var(--critical)", flat: undefined } as const;

export function groupOf(e: MatrixEntry): string {
  return `${e.model}|${e.protocol}`;
}

const fmtVal = (m: MetricSpec, v: number) => `${v.toFixed(m.digits)}${m.unit ? ` ${m.unit}` : ""}`;

export function ScoreMatrix({ entries }: { entries: MatrixEntry[] }) {
  const groups = useMemo(() => {
    const g = new Map<string, MatrixEntry[]>();
    for (const e of entries) g.set(groupOf(e), [...(g.get(groupOf(e)) ?? []), e]);
    return [...g.entries()].sort((a, b) => b[1].length - a[1].length);
  }, [entries]);
  const [gk, setGk] = useState<string>("");
  const [bk, setBk] = useState<string>(FP);
  const [mode, setMode] = useState<"delta" | "abs">("delta");
  const group = groups.find(([k]) => k === gk) ?? groups[0];
  if (!group) return null;
  const [key, members] = group;
  const cols = members.filter((e) => e.key !== FP);
  const base = members.find((e) => e.key === bk) ?? members.find((e) => e.key === FP) ?? cols[0];
  const rows = MATRIX_METRICS.filter((m) => members.some((e) => e.metrics[m.id] != null));
  const best = (m: MetricSpec): string | null => {
    const have = cols.filter((e) => e.metrics[m.id] != null);
    if (have.length < 2) return null;
    return have.reduce((a, b) => ((m.direction === "lower" ? b.metrics[m.id]! < a.metrics[m.id]! : b.metrics[m.id]! > a.metrics[m.id]!) ? b : a)).key;
  };
  const [model, protocol] = key.split("|");
  const engines = [...new Set(cols.filter((e) => e.metrics.tps != null && e.engine).map((e) => e.engine as string))];
  return (
    <div className="card table-wrap" data-testid="score-matrix">
      <h2>对比矩阵 · 以哪一列为基线</h2>
      <div className="form-row" style={{ alignItems: "end" }}>
        {groups.length > 1 && (
          <div className="field">
            <label>同一个模型、同一种评估协议的放在一起</label>
            <select value={key} onChange={(e) => { setGk(e.target.value); setBk(FP); }} data-testid="matrix-group">
              {groups.map(([k, es]) => (
                <option key={k} value={k}>{k.split("|")[0]} · {k.split("|")[1]}（{es.filter((e) => e.key !== FP).length} 列）</option>
              ))}
            </select>
          </div>
        )}
        <div className="field">
          <label>基线</label>
          <select value={base?.key ?? ""} onChange={(e) => setBk(e.target.value)} data-testid="matrix-baseline">
            {members.map((e) => (
              <option key={e.key} value={e.key}>{e.label}</option>
            ))}
          </select>
        </div>
        <div className="chips">
          {(["delta", "abs"] as const).map((m) => (
            <span key={m} className={`chip ${mode === m ? "on" : ""}`} onClick={() => setMode(m)} data-testid={`matrix-mode-${m}`}>
              {m === "delta" ? "相对基线的变化" : "绝对值"}
            </span>
          ))}
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>指标</th>
            {cols.map((e) => (
              <th key={e.key} data-testid={`matrix-col-${e.key}`}>{e.label}{e.key === base?.key ? "（基线）" : ""}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => {
            const b = base?.metrics[m.id];
            const bestKey = best(m);
            return (
              <tr key={m.id} data-testid={`matrix-row-${m.id}`}>
                <td>
                  {m.label}
                  <span className="muted small"> · {m.direction === "lower" ? "越低越好" : "越高越好"}</span>
                </td>
                {cols.map((e) => {
                  const v = e.metrics[m.id];
                  const isBase = e.key === base?.key;
                  const pct = v != null && b != null && !isBase ? relChange(v, b) : null;
                  const tone = pct == null ? "flat" : toneOf(m.direction, pct);
                  const shown = v == null ? "–" : mode === "abs" || isBase ? fmtVal(m, v) : pct == null ? "–" : signedPct(pct);
                  return (
                    <td
                      key={e.key}
                      data-testid={`cell-${e.key}-${m.id}`}
                      data-tone={mode === "delta" && !isBase ? tone : "none"}
                      data-best={bestKey === e.key ? "1" : "0"}
                      title={v == null ? "这一列没有这个指标" : `${fmtVal(m, v)}${b != null && !isBase ? `；基线 ${fmtVal(m, b)}` : ""}`}
                      style={{ color: mode === "delta" && !isBase ? COLOR[tone] : undefined, fontWeight: bestKey === e.key ? 650 : undefined }}
                    >
                      {shown}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="note">
        {model} · {protocol}。PPL 只在同一个模型、同一种评估协议（窗口数 × 序列长度）下可比，所以别的组合在上面的下拉里另选。颜色按每个指标自己的方向：PPL、体积越低越好，吞吐越高越好；
        加粗是这一行里最好的一列。{base?.key === FP ? "基线是未量化的原模型，只有 PPL 有它的数。" : "基线列本身显示绝对值；基线没有的指标显示“–”。"}
        {engines.length > 1 && (
          <span data-testid="matrix-engine-note"> 吞吐来自不同引擎（{engines.join(" / ")}）：vLLM 阶段用 enforce_eager，SGLang 用 CUDA graph，跨引擎的吞吐不能直接比。</span>
        )}
      </p>
    </div>
  );
}

export { FP as FP_KEY };
