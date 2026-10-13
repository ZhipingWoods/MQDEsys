import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api, fmt, fmtBytes, followJob, JobDetail, JobEvent, JobReport, LLMJobReport, RecipeReport, VLMJobReport } from "../api";
import { barItem, baseOption, Chart, series, token } from "../charts";
import { Tile } from "./Estimate";
import { LLM_STAGES, LLMReportView } from "./JobLLM";
import { VLM_STAGES, VLMReportView } from "./JobVLM";
import { DERIVE_STEPS } from "../derive";
import { StateBadge } from "./Jobs";
import { LayerAnalysis } from "./LayerAnalysis";

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

  const rep = detail?.report ?? null;
  const job = detail?.job as { model?: { type?: string }; deploy?: { device?: string }; metadata?: { parent_job?: string } } | undefined;
  const isLLM = rep?.kind === "llm" || job?.model?.type === "llm";
  const isVLM = rep?.kind === ("vlm" as string) || job?.model?.type === "vlm";
  const stageList = isLLM ? LLM_STAGES : isVLM ? VLM_STAGES : STAGES;
  const stages = useMemo(() => {
    const m: Record<string, StageState> = {};
    for (const [k] of stageList) m[k] = { status: "pending", count: 0 };
    for (const e of events) {
      if (!e.stage || !m[e.stage]) continue;
      if (e.type === "stage_started") m[e.stage] = { ...m[e.stage], status: "running", count: m[e.stage].count + 1, msg: e.msg };
      if (e.type === "stage_finished") m[e.stage] = { ...m[e.stage], status: e.status === "failed" ? "failed" : "succeeded", seconds: (m[e.stage].seconds ?? 0) + (e.seconds ?? 0) };
    }
    return m;
  }, [events, stageList]);

  const progress = useMemo(() => {
    const t = [...events].reverse().find((e) => e.type === "trace");
    return t ? { i: (t.index ?? 0) + 1, n: t.total ?? 0, name: t.trace?.name, stage: t.stage } : null;
  }, [events]);

  const state = events.find((e) => e.type === "job_state" && e.state !== "RUNNING")?.state ?? detail?.state ?? "UNKNOWN";

  return (
    <>
      <h1>
        作业 <span className="mono">{id}</span> <StateBadge state={state} />
      </h1>
      <p className="subtitle">
        {rep?.spec?.name ?? ""} · {rep?.spec?.model_path ?? ""} · {job?.deploy?.device === "cpu" ? "CPU" : `GPU ${detail?.gpu ?? "–"}`} {live && <span className="status running">实时</span>}
        {state === "RUNNING" && (
          <button className="secondary" style={{ marginLeft: 12 }} onClick={() => api.cancel(id)}>
            取消
          </button>
        )}
        {isLLM && state === "SUCCEEDED" && (
          <span style={{ marginLeft: 12 }}>
            独立报告页（可离线打开）：{" "}
            <a href={`/api/v1/jobs/${id}/report.zh.html`} target="_blank" rel="noreferrer">
              中文
            </a>{" "}
            ·{" "}
            <a href={`/api/v1/jobs/${id}/report.en.html`} target="_blank" rel="noreferrer">
              English
            </a>
          </span>
        )}
      </p>
      {job?.metadata?.parent_job && (
        <p className="small" data-testid="derived-from">
          派生自作业 <Link to={`/jobs/${job.metadata.parent_job}`}>{job.metadata.parent_job}</Link>
        </p>
      )}
      {detail?.job && state !== "RUNNING" && state !== "QUEUED" && (
        <div className="form-row" style={{ margin: "4px 0 12px", alignItems: "center" }} data-testid="derive-actions">
          <span className="small muted">基于此作业新建：</span>
          {DERIVE_STEPS.map(({ step, label }) =>
            step === "deploy" && !isLLM ? (
              <button key={step} className="secondary" disabled title="VLM/VLA 作业目前不做推理框架部署" data-testid={`derive-${step}`}>
                {label}
              </button>
            ) : (
              <Link key={step} to={`/new?from=${id}&step=${step}`} data-testid={`derive-${step}`}>
                <button className="secondary">{label}</button>
              </Link>
            ),
          )}
        </div>
      )}
      {err && <div className="error-box">{err}</div>}
      {rep?.error && <div className="error-box">{rep.error}</div>}

      <div className="card">
        <h2>执行过程</h2>
        <div className="pipeline">
          {stageList.map(([k, label]) => {
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

      {rep && (isLLM ? <LLMReportView rep={rep as unknown as LLMJobReport} job={detail?.job} jobId={id} /> : isVLM ? <VLMReportView rep={rep as unknown as VLMJobReport} job={detail?.job} jobId={id} /> : <ReportView rep={rep} jobId={id} />)}
    </>
  );
}

function EventLog({ events }: { events: JobEvent[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [events.length]);
  const lines = events.filter((e) => e.type !== "trace" && e.type !== "progress" && (e.type as string) !== "layer");
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

function ReportView({ rep, jobId }: { rep: JobReport; jobId?: string }) {
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
      {tab !== "overview" && rep.recipes?.[tab] && jobId && (
        <p className="small muted" data-testid="open-standalone-links">
          在独立页面打开：{" "}
          <Link to={`/analysis/${jobId}?recipe=${tab}`} data-testid="open-analysis-link">
            误差分析
          </Link>
        </p>
      )}
      {tab !== "overview" && rep.recipes?.[tab] && <VLARecipeErrorAnalysis r={rep.recipes[tab]} rep={rep} />}
    </>
  );
}

export function orderedRecipes(rep: JobReport): string[] {
  const all = Object.keys(rep.recipes ?? {});
  const order = (rep.spec?.recipes ?? []).map((x) => x.toLowerCase());
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
          <ul className="note" data-testid="attribution-largest" style={{ margin: "8px 0 0", paddingLeft: 18 }}>
            {rs.map((r) => {
              const big = (r.attribution ?? []).filter((a) => a.action_rel_l2 != null).sort((a, b) => b.action_rel_l2! - a.action_rel_l2!)[0];
              if (!big) return null;
              const all = r.eval.action_rel_l2;
              return (
                <li key={r.recipe}>
                  <b>{r.recipe}</b>：只量化 <b>{big.modality}</b> 的误差最大，rel-L2 {fmt(big.action_rel_l2, 4)}
                  {all ? `，为全部量化时（${fmt(all, 4)}）的 ${((100 * big.action_rel_l2!) / all).toFixed(0)}%` : ""}。
                </li>
              );
            })}
          </ul>
        </div>
      </div>
      <div className="card table-wrap">
        <h2>配方汇总</h2>
        <table>
          <thead>
            <tr>
              <th>配方</th>
              <th>算法</th>
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
                <td>
                  <span className="badge accent">{r.algorithm ?? "rtn"}</span>
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

export function VLARecipeErrorAnalysis({ r, rep }: { r: RecipeReport; rep: JobReport }) {
  void rep;
  return (
    <>
      <div className="tiles">
        <Tile label="动作 rel-L2" value={fmt(r.eval.action_rel_l2, 4)} hint={`${fmt(r.summary.action_rel_l2_over_noise_floor, 2)} × 噪声下限`} />
        <Tile label="动作 cos" value={fmt(r.eval.action_cos, 4)} />
        <Tile label="VL 嵌入 cos" value={fmt(r.eval.vl_cos, 4)} hint={`SQNR ${fmt(r.eval.vl_sqnr_db, 1)} dB`} />
        <Tile label="量化层数" value={`${r.layers.length}`} hint={`W: ${r.weight.format}${r.weight.bits ?? ""} ${r.weight.granularity ?? ""}${r.weight.group_size ? " g" + r.weight.group_size : ""} · A: ${r.act.format}`} />
      </div>
      <LayerAnalysis layers={r.layers} />
    </>
  );
}

export { barItem };
