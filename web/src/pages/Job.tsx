import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { api, fmt, fmtBytes, followJob, JobDetail, JobEvent, JobReport, LayerTrace, RecipeReport } from "../api";
import { barItem, baseOption, Chart, series, token } from "../charts";
import { Tile } from "./Estimate";
import { StateBadge } from "./Jobs";

const STAGES: [string, string][] = [
  ["validate", "校验"],
  ["preflight", "资源预估"],
  ["inspect", "模型加载"],
  ["calib_prepare", "数据准备"],
  ["eval_baseline", "BF16 基线"],
  ["attribution", "分阶段归因"],
  ["quantize", "量化+逐层追踪"],
  ["eval_fake", "量化评估"],
  ["report", "报告"],
];

interface StageState {
  status: "pending" | "running" | "succeeded" | "failed";
  seconds?: number;
  count: number;
  msg?: string;
}

export default function JobPage() {
  const { id = "" } = useParams();
  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [live, setLive] = useState(false);

  useEffect(() => {
    let alive = true;
    setEvents([]);
    const refresh = () => api.job(id).then((d) => alive && setDetail(d), (e: Error) => alive && setErr(e.message));
    refresh();
    setLive(true);
    const stop = followJob(
      id,
      (ev) => {
        if (!alive) return;
        setEvents((xs) => [...xs, ev]);
        if (ev.type === "stage_finished") refresh();
      },
      () => {
        setLive(false);
        refresh();
      },
    );
    return () => {
      alive = false;
      stop();
    };
  }, [id]);

  const stages = useMemo(() => {
    const m: Record<string, StageState> = {};
    for (const [k] of STAGES) m[k] = { status: "pending", count: 0 };
    for (const e of events) {
      if (!e.stage || !m[e.stage]) continue;
      if (e.type === "stage_started") m[e.stage] = { ...m[e.stage], status: "running", count: m[e.stage].count + 1, msg: e.msg };
      if (e.type === "stage_finished") m[e.stage] = { ...m[e.stage], status: e.status === "failed" ? "failed" : "succeeded", seconds: (m[e.stage].seconds ?? 0) + (e.seconds ?? 0) };
    }
    return m;
  }, [events]);

  const progress = useMemo(() => {
    const t = [...events].reverse().find((e) => e.type === "trace");
    return t ? { i: (t.index ?? 0) + 1, n: t.total ?? 0, name: t.trace?.name, stage: t.stage } : null;
  }, [events]);

  const rep = detail?.report ?? null;
  const state = events.find((e) => e.type === "job_state" && e.state !== "RUNNING")?.state ?? detail?.state ?? "UNKNOWN";

  return (
    <>
      <h1>
        作业 <span className="mono">{id}</span> <StateBadge state={state} />
      </h1>
      <p className="subtitle">
        {rep?.spec.name ?? ""} · {rep?.spec.model_path ?? ""} · GPU {detail?.gpu ?? "–"} {live && <span className="status running">实时</span>}
        {state === "RUNNING" && (
          <button className="secondary" style={{ marginLeft: 12 }} onClick={() => api.cancel(id)}>
            取消
          </button>
        )}
      </p>
      {err && <div className="error-box">{err}</div>}
      {rep?.error && <div className="error-box">{rep.error}</div>}

      <div className="card">
        <h2>执行过程</h2>
        <div className="pipeline">
          {STAGES.map(([k, label]) => {
            const s = stages[k];
            return (
              <div key={k} className={`step ${s.status}`}>
                <div className="n">{label}</div>
                <div className="t">
                  {s.status === "pending" ? "等待" : s.status === "running" ? `进行中 ${s.msg ?? ""}` : `${fmt(s.seconds, 1)} s${s.count > 1 ? ` · ${s.count} 次` : ""}`}
                </div>
              </div>
            );
          })}
        </div>
        {progress && state === "RUNNING" && (
          <p className="small muted" style={{ marginTop: 8 }}>
            {progress.stage}: 第 {progress.i}/{progress.n} 层 <span className="mono">{progress.name}</span>
          </p>
        )}
        <EventLog events={events} />
      </div>

      {rep && <ReportView rep={rep} />}
    </>
  );
}

function EventLog({ events }: { events: JobEvent[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [events.length]);
  const lines = events.filter((e) => e.type !== "trace" && (e.type as string) !== "layer");
  return (
    <div className="log" ref={ref} style={{ marginTop: 12 }}>
      {lines.map((e) => {
        const t = new Date(e.ts * 1000).toLocaleTimeString();
        const cls = e.type === "metric" ? "lv-metric" : e.status === "failed" || e.state === "FAILED" ? "lv-error" : "";
        const text =
          e.type === "metric"
            ? `${e.name} = ${typeof e.value === "number" ? e.value.toFixed(4) : e.value}`
            : e.type === "stage_finished"
              ? `${e.status} (${fmt(e.seconds, 2)} s)${e.error ? " " + e.error : ""}`
              : (e.msg ?? e.state ?? "");
        return (
          <div key={e.seq} className={cls}>
            {t} {e.type.padEnd(15)} {(e.stage ?? "-").padEnd(14)} {text}
          </div>
        );
      })}
    </div>
  );
}

function ReportView({ rep }: { rep: JobReport }) {
  // the server re-serializes JSON with sorted keys; restore the job's recipe order
  const names = orderedRecipes(rep);
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "overview";
  const setTab = (t: string) => setParams(t === "overview" ? {} : { tab: t }, { replace: true });
  const b = rep.baseline;
  return (
    <>
      {b && (
        <div className="tiles">
          <Tile label="BF16 延迟 / 决策步" value={`${fmt(b.latency_ms_per_step, 0)} ms`} hint="PyTorch eager（无 fla 快速路径）" />
          <Tile label="BF16 峰值显存（实测）" value={fmtBytes(b.peak_mem_bytes)} hint={`上下文 ${b.context_tokens} token`} />
          <Tile label="流匹配噪声下限" value={fmt(b.noise_floor.action_rel_l2, 4)} hint="同模型换噪声种子的动作相对 L2" />
          <Tile label="评估样本" value={`${rep.data?.eval.n ?? "–"}`} hint={`${rep.data?.eval.source.split("/").pop()} · ${rep.data?.eval.views} 视角`} />
        </div>
      )}
      {names.length > 0 && (
        <div className="tabs">
          <div className={`tab ${tab === "overview" ? "on" : ""}`} onClick={() => setTab("overview")}>
            总览
          </div>
          {names.map((n) => (
            <div key={n} className={`tab ${tab === n ? "on" : ""}`} onClick={() => setTab(n)}>
              {n} 误差分析
            </div>
          ))}
        </div>
      )}
      {tab === "overview" && names.length > 0 && <Overview rep={rep} />}
      {tab !== "overview" && rep.recipes?.[tab] && <RecipeView r={rep.recipes[tab]} rep={rep} />}
    </>
  );
}

function orderedRecipes(rep: JobReport): string[] {
  const all = Object.keys(rep.recipes ?? {});
  const order = (rep.spec.recipes ?? []).map((x) => x.toLowerCase());
  return [...all].sort((a, b) => order.indexOf(a.toLowerCase()) - order.indexOf(b.toLowerCase()));
}

function Overview({ rep }: { rep: JobReport }) {
  const rs = orderedRecipes(rep).map((n) => rep.recipes![n]);
  const nf = rep.baseline?.noise_floor.action_rel_l2 ?? 0;
  const est = rep.preflight?.estimates ?? {};
  const estPeak = (name: string) => {
    const k = Object.keys(est).find((x) => x.toLowerCase() === name.toLowerCase());
    return k ? Object.values(est[k])[0]?.peak_bytes : undefined;
  };
  return (
    <>
      <div className="grid-2">
        <div className="card">
          <h2>动作误差（相对 BF16，越低越好）</h2>
          <Chart
            option={() => {
              const o = baseOption();
              return {
                ...o,
                tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => v.toFixed(4) },
                xAxis: { ...(o.xAxis as object), type: "category", data: rs.map((r) => r.recipe) },
                yAxis: { ...(o.yAxis as object), type: "value", name: "action rel-L2", nameTextStyle: { color: token("--text-muted") } },
                series: [
                  {
                    type: "bar",
                    name: "action rel-L2",
                    barMaxWidth: 40,
                    itemStyle: { color: series(0), borderRadius: [4, 4, 0, 0] },
                    data: rs.map((r) => r.eval.action_rel_l2),
                    label: { show: true, position: "top", color: token("--text-secondary"), fontSize: 11, formatter: (p: { value: number }) => p.value.toFixed(3) },
                    markLine: {
                      silent: true,
                      symbol: "none",
                      lineStyle: { color: token("--text-secondary"), type: "dashed", width: 1 },
                      label: { formatter: `噪声下限 ${nf.toFixed(3)}`, color: token("--text-secondary"), position: "insideEndTop", fontSize: 11 },
                      data: [{ yAxis: nf }],
                    },
                  },
                ],
              };
            }}
          />
          <p className="note">虚线 = 流匹配采样自身的随机性（BF16 换噪声种子）。低于虚线的量化误差小于模型本身的采样波动。</p>
        </div>
        <div className="card">
          <h2>分阶段归因：只量化某一模态时的动作误差</h2>
          <Chart
            option={() => {
              const o = baseOption();
              const mods = [...new Set(rs.flatMap((r) => (r.attribution ?? []).map((a) => a.modality)))];
              return {
                ...o,
                legend: { ...(o.legend as object), data: [...mods.map((m) => `仅 ${m}`), "全部"] },
                grid: { ...(o.grid as object), top: 56 },
                tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => v?.toFixed(4) },
                xAxis: { ...(o.xAxis as object), type: "category", data: rs.map((r) => r.recipe) },
                yAxis: { ...(o.yAxis as object), type: "value", name: "action rel-L2", nameTextStyle: { color: token("--text-muted") } },
                series: [
                  ...mods.map((m, i) => ({
                    name: `仅 ${m}`,
                    type: "bar",
                    barMaxWidth: 18,
                    itemStyle: { color: series(i + 1), borderRadius: [4, 4, 0, 0] },
                    data: rs.map((r) => r.attribution?.find((a) => a.modality === m)?.action_rel_l2 ?? null),
                  })),
                  { name: "全部", type: "bar", barMaxWidth: 18, itemStyle: { color: series(0), borderRadius: [4, 4, 0, 0] }, data: rs.map((r) => r.eval.action_rel_l2) },
                ],
              };
            }}
          />
        </div>
      </div>
      <div className="card table-wrap">
        <h2>配方汇总</h2>
        <table>
          <thead>
            <tr>
              <th>配方</th>
              <th className="num">动作 rel-L2</th>
              <th className="num">× 噪声下限</th>
              <th className="num">动作 cos</th>
              <th className="num">VL 嵌入 cos</th>
              <th>最大误差环节</th>
              <th>最差层</th>
              <th className="num">预估峰值显存</th>
            </tr>
          </thead>
          <tbody>
            {rs.map((r) => (
              <tr key={r.recipe}>
                <td>
                  <b>{r.recipe}</b>
                  {r.maturity === "experimental" && <span className="muted small"> 实验</span>}
                </td>
                <td className="num">{fmt(r.eval.action_rel_l2, 4)}</td>
                <td className="num">{fmt(r.summary.action_rel_l2_over_noise_floor, 2)}</td>
                <td className="num">{fmt(r.eval.action_cos, 4)}</td>
                <td className="num">{fmt(r.eval.vl_cos, 4)}</td>
                <td>{r.summary.worst_stage ?? "–"}</td>
                <td className="mono small">{r.summary.worst_layers[0]?.name.replace("vlm.model.", "") ?? "–"}</td>
                <td className="num">{fmtBytes(estPeak(r.recipe))}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">
          fake-quant 以 BF16 存储反量化权重，因此实测显存不会下降；“预估峰值显存”为真实低比特部署的 C++ 预估。
          {rep.data?.eval.has_gt && " 与数据集真值动作的对比未启用：模型 80 维动作空间与 LIBERO 7 维动作的映射和归一化统计未知。"}
        </p>
      </div>
    </>
  );
}

function RecipeView({ r, rep }: { r: RecipeReport; rep: JobReport }) {
  const comps = useMemo(() => [...new Set(r.layers.map((l) => l.component))], [r]);
  const [comp, setComp] = useState(comps[0] ?? "language");
  const [metric, setMetric] = useState<"out_sqnr_db" | "w_sqnr_db">("out_sqnr_db");
  const worst = useMemo(() => [...r.layers].filter((l) => l.out_sqnr_db != null).sort((a, b) => a.out_sqnr_db! - b.out_sqnr_db!).slice(0, 15), [r]);
  void rep;
  return (
    <>
      <div className="tiles">
        <Tile label="动作 rel-L2" value={fmt(r.eval.action_rel_l2, 4)} hint={`${fmt(r.summary.action_rel_l2_over_noise_floor, 2)} × 噪声下限`} />
        <Tile label="动作 cos" value={fmt(r.eval.action_cos, 4)} />
        <Tile label="VL 嵌入 cos" value={fmt(r.eval.vl_cos, 4)} hint={`SQNR ${fmt(r.eval.vl_sqnr_db, 1)} dB`} />
        <Tile label="量化层数" value={`${r.layers.length}`} hint={`W: ${r.weight.format}${r.weight.bits ?? ""} ${r.weight.granularity ?? ""}${r.weight.group_size ? " g" + r.weight.group_size : ""} · A: ${r.act.format}`} />
      </div>
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
                <td className="mono small">{l.name.replace("vlm.model.", "")}</td>
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

export { barItem };
