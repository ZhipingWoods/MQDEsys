import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api, fmt, fmtBytes, followJob, JobDetail, JobEvent, JobReport, LLMJobReport, RecipeReport, VLMJobReport, WorldJobReport, DeployJobReport, EvalJobReport } from "../api";
import { barItem, baseOption, Chart, series, token } from "../charts";
import { Tile } from "./Estimate";
import { LLM_STAGES, LLMReportView } from "./JobLLM";
import { VLM_STAGES, VLMReportView } from "./JobVLM";
import { WORLD_STAGES, WorldReportView } from "./JobWorld";
import { DEPLOY_STAGES, DeployReportView } from "./JobDeploy";
import { EVAL_STAGES, EvalReportView } from "./JobEval";
import { DERIVE_STEPS } from "../derive";
import { StateBadge } from "./Jobs";
import { LayerAnalysis } from "./LayerAnalysis";
import { NativeServeCard } from "../components/NativeServeCard";

const STAGES: [string, string][] = [
  ["validate", "校验"],
  ["preflight", "资源预估"],
  ["inspect", "模型加载"],
  ["calib_prepare", "数据准备"],
  ["eval_baseline", "BF16 基线"],
  ["attribution", "分阶段归因"],
  ["quantize", "量化+逐层追踪"],
  ["eval_fake", "量化评估"],
  ["deploy_native", "原生连续输出服务"],
  ["report", "报告"],
];

interface StageState {
  status: "pending" | "running" | "succeeded" | "failed";
  seconds?: number;
  count: number;
  msg?: string;
}

function Ring({ pct, state }: { pct: number; state: string }) {
  const size = 68, stroke = 7, r = (size - stroke) / 2, circ = 2 * Math.PI * r;
  const color = state === "FAILED" ? "var(--critical)" : state === "CANCELLED" ? "var(--warning)" : state === "RUNNING" || state === "QUEUED" ? "var(--accent)" : "var(--good)";
  const offset = circ * (1 - Math.max(0, Math.min(1, pct / 100)));
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="job-ring" aria-label={`进度 ${Math.round(pct)}%`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--surface-3)" strokeWidth={stroke} />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke} strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round" />
      <text x="50%" y="50%" textAnchor="middle" dominantBaseline="central" fill="var(--text-primary)" fontSize="13" fontWeight="700">{Math.round(pct)}%</text>
    </svg>
  );
}

function shortDuration(s: number | null): string {
  if (s == null) return "–";
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${(s / 3600).toFixed(1)}h`;
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
  const job = detail?.job as { model?: { type?: string; path?: string }; deploy?: { device?: string }; metadata?: { parent_job?: string }; artifact?: unknown } | undefined;
  // a deploy job (B2) has model.type llm too, but serves an existing export: its own report and stages
  const isEval = rep?.kind === ("llm_eval" as string);
  const isDeploy = !isEval && (rep?.kind === ("llm_deploy" as string) || (job?.artifact != null && rep?.kind !== ("llm_eval" as string)));
  const isLLM = !isDeploy && !isEval && (rep?.kind === "llm" || job?.model?.type === "llm");
  const isVLM = rep?.kind === ("vlm" as string) || job?.model?.type === "vlm";
  const isWorld = rep?.kind === ("world" as string) || rep?.kind === ("ewm" as string) || job?.model?.type === "world" || job?.model?.type === "ewm";
  const deployKeys = Object.keys((detail?.job as { deploy?: Record<string, unknown> } | undefined)?.deploy ?? {});
  const reportedStages = new Set((rep?.stages ?? []).map((s) => s.stage));
  const stageList = isEval
    ? EVAL_STAGES.filter(([k]) => reportedStages.size === 0 || reportedStages.has(k))
    : isDeploy ? DEPLOY_STAGES.filter(([k]) => !k.startsWith("deploy_") || deployKeys.includes(k.slice(7))) : isLLM ? LLM_STAGES : isVLM ? VLM_STAGES : isWorld ? WORLD_STAGES : STAGES;
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
  const lastProgress = [...events].reverse().find((e) => e.type === "progress");
  const lastTrace = [...events].reverse().find((e) => e.type === "trace");
  const pct = state === "SUCCEEDED" ? 100
    : lastProgress?.value != null ? Math.max(0, Math.min(100, lastProgress.value * 100))
    : lastTrace?.total ? Math.max(0, Math.min(100, (((lastTrace.index ?? 0) + 1) * 100) / lastTrace.total))
    : state === "FAILED" ? 100 : 0;
  const firstTs = events[0]?.ts;
  const lastTs = events.length ? events[events.length - 1].ts : firstTs;
  const elapsed = firstTs && lastTs ? Math.max(0, Math.round(lastTs - firstTs)) : null;
  const currentStage = [...events].reverse().find((e) => e.type === "stage_started")?.stage ?? detail?.stage ?? null;
  const metrics = events.filter((e) => e.type === "metric").slice(-6).reverse();
  const stageDone = Object.values(stages).filter((s) => s.status === "succeeded").length;
  const stageFailed = Object.values(stages).filter((s) => s.status === "failed").length;
  const kindLabel = isEval ? "评测作业" : isDeploy ? "部署评测" : isLLM ? "LLM 全流程" : isVLM ? "VLM 作业" : isWorld ? "WM 作业" : "VLA 作业";

  if (!detail) {
    return (
      <>
        <h1>作业 <span className="mono">{id}</span></h1>
        {err ? <div className="error-box">{err}</div> : <div className="card job-loading"><span className="status running">读取作业</span><p className="muted small">正在读取 job.yaml、报告和事件流…</p></div>}
      </>
    );
  }

  return (
    <>
      <section className="job-hero card" data-testid="job-hero">
        <div className="job-hero-main">
          <div className="eyebrow">{kindLabel} <span className="eyebrow-dot" /></div>
          <h1>作业 <span className="mono">{id}</span></h1>
          <p className="job-hero-name">{rep?.spec?.name ?? id}</p>
          <p className="job-hero-model mono small">{rep?.spec?.model_path ?? String(job?.model?.path ?? "—")}</p>
          <p className="subtitle job-hero-device">
            {job?.deploy?.device === "cpu" ? "CPU" : detail?.gpu != null && detail.gpu >= 0 ? `GPU ${detail.gpu}` : deviceName(rep) ?? "GPU"}
          </p>
          <div className="job-hero-meta">
            <StateBadge state={state} />
            {live && <span className="status running">实时</span>}
            <span className="badge">{job?.deploy?.device === "cpu" ? "CPU" : detail?.gpu != null && detail.gpu >= 0 ? `GPU ${detail.gpu}` : deviceName(rep) ?? "GPU"}</span>
            <span className="badge">阶段 {stageDone}/{stageList.length}{stageFailed ? ` · 失败 ${stageFailed}` : ""}</span>
            <span className="badge">已运行 {shortDuration(elapsed)}</span>
          </div>
        </div>
        <div className="job-hero-side">
          <Ring pct={pct} state={state} />
          <div className="job-hero-actions">
            {state === "RUNNING" && <button className="secondary" onClick={() => api.cancel(id)}>取消作业</button>}
            {(isLLM || isEval) && state === "SUCCEEDED" && (
              <>
                <a className="btn-link" href={`/api/v1/jobs/${id}/report.zh.html`} target="_blank" rel="noreferrer">中文报告</a>
                <a className="btn-link" href={`/api/v1/jobs/${id}/report.en.html`} target="_blank" rel="noreferrer">English</a>
              </>
            )}
            {detail?.job && (
              <button className="secondary" onClick={() => navigator.clipboard?.writeText(JSON.stringify(detail.job, null, 2))}>
                复制 JobSpec
              </button>
            )}
            <a className="btn-link" href={`/api/v1/jobs/${id}/log`} download={`${id}.log`}>下载日志</a>
          </div>
        </div>
      </section>
      {job?.metadata?.parent_job && (
        <p className="small" data-testid="derived-from">
          派生自作业 <Link to={`/jobs/${job.metadata.parent_job}`}>{job.metadata.parent_job}</Link>
        </p>
      )}
      {detail?.job && state !== "RUNNING" && state !== "QUEUED" && (
        <div className="form-row" style={{ margin: "4px 0 12px", alignItems: "center" }} data-testid="derive-actions">
          <span className="small muted">基于此作业新建：</span>
          {DERIVE_STEPS.map(({ step, label }) =>
            step === "deploy" && !isLLM && !isDeploy ? (
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
        <div className="card-head">
          <div>
            <h2>执行过程</h2>
            <p className="note" style={{ margin: 0 }}>阶段进度来自事件流；指标和日志实时追加。</p>
          </div>
          <span className="badge">{events.length} 个事件</span>
        </div>
        <div className="job-live-grid" data-testid="job-live-grid">
          <div><small>当前阶段</small><b>{currentStage ?? (state === "SUCCEEDED" ? "已完成" : "—")}</b></div>
          <div><small>进度</small><b>{Math.round(pct)}%</b></div>
          <div><small>最近指标</small><b className="mono">{metrics[0]?.name ? `${metrics[0].name} = ${typeof metrics[0].value === "number" ? metrics[0].value.toFixed(4) : metrics[0].value}` : "—"}</b></div>
          <div><small>耗时</small><b>{shortDuration(elapsed)}</b></div>
        </div>
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
        <EventLog events={events} state={state} jobId={id} />
      </div>

      {rep && (isEval ? <EvalReportView rep={rep as unknown as EvalJobReport} /> : isDeploy ? <DeployReportView rep={rep as unknown as DeployJobReport} /> : isLLM ? <LLMReportView rep={rep as unknown as LLMJobReport} job={detail?.job} jobId={id} /> : isVLM ? <VLMReportView rep={rep as unknown as VLMJobReport} job={detail?.job} jobId={id} /> : isWorld ? <WorldReportView rep={rep as unknown as WorldJobReport} job={detail?.job} jobId={id} /> : <ReportView rep={rep} jobId={id} />)}
    </>
  );
}

function EventLog({ events, state, jobId }: { events: JobEvent[]; state: string; jobId: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [filter, setFilter] = useState<"all" | "errors" | "metrics">("all");
  const [follow, setFollow] = useState(true);
  const [copied, setCopied] = useState(false);
  const lines = events.filter((e) => e.type !== "trace" && e.type !== "progress" && (e.type as string) !== "layer"
    && (filter === "all" || (filter === "metrics" ? e.type === "metric" : e.status === "failed" || e.state === "FAILED" || e.level === "error")));
  const textOf = (e: JobEvent) => {
    const t = new Date(e.ts * 1000).toLocaleTimeString();
    const text = e.type === "metric"
      ? `${e.name} = ${typeof e.value === "number" ? e.value.toFixed(4) : e.value}`
      : e.type === "stage_finished"
        ? `${e.status} (${fmt(e.seconds, 2)} s)${e.error ? " " + e.error : ""}`
        : (e.msg ?? e.state ?? "");
    return `${t} ${e.type.padEnd(15)} ${(e.stage ?? "-").padEnd(14)} ${text}`;
  };
  useEffect(() => {
    if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines.length, follow]);
  return (
    <div className="job-log-wrap" data-testid="event-log">
      <div className="job-log-toolbar">
        <div className="seg small-seg">
          {(["all", "metrics", "errors"] as const).map((f) => (
            <button key={f} className={filter === f ? "on" : ""} onClick={() => setFilter(f)} data-testid={`log-filter-${f}`}>
              {{ all: "全部", metrics: "指标", errors: "错误" }[f]}
            </button>
          ))}
        </div>
        <label className="small"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> 跟随最新</label>
        <span className="small muted">{lines.length} 行</span>
        <button className="secondary small" onClick={async () => {
          try { await navigator.clipboard.writeText(lines.map(textOf).join("\n")); setCopied(true); setTimeout(() => setCopied(false), 1200); } catch { /* clipboard may be unavailable */ }
        }}>{copied ? "已复制" : "复制日志"}</button>
        {state !== "RUNNING" && state !== "QUEUED" && <a className="btn-link small" href={`/api/v1/jobs/${jobId}/log`} download={`${jobId}.log`}>下载</a>}
      </div>
      <div className="log job-log" ref={ref}>
        {lines.map((e) => {
          const cls = e.type === "metric" ? "lv-metric" : e.status === "failed" || e.state === "FAILED" ? "lv-error" : "";
          return <div key={e.seq} className={cls}>{textOf(e)}</div>;
        })}
        {lines.length === 0 && <div className="muted">没有符合筛选条件的事件。</div>}
      </div>
    </div>
  );
}

function ReportView({ rep, jobId }: { rep: JobReport; jobId?: string }) {
  // the server re-serializes JSON with sorted keys; restore the job's recipe order
  const names = orderedRecipes(rep);
  const [params, setParams] = useSearchParams();
  const want = params.get("tab") ?? "overview";
  const tab = names.includes(want) ? want : "overview"; // a stale ?tab= (other job's recipe) must not leave the page blank
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
      {rs.map((r) => r.deploy?.native ? <NativeServeCard key={r.recipe} recipe={r.recipe} d={r.deploy.native} /> : null)}
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
        <Tile label="量化层数" value={`${r.layers.length}`} hint={`W: ${r.weight.format}${r.weight.format.startsWith("fp8") ? "" : (r.weight.bits ?? "")} ${r.weight.granularity ?? ""}${r.weight.group_size ? " g" + r.weight.group_size : ""} · A: ${r.act.format}`} />
      </div>
      <LayerAnalysis layers={r.layers} />
    </>
  );
}

export { barItem };

/** The device the report says it ran on (env.device_name for vlm/deploy jobs, the eval/quant device names for llm). */
function deviceName(rep: unknown): string | undefined {
  const env = (rep as { env?: Record<string, unknown> } | null)?.env;
  for (const k of ["device_name", "eval_device_name", "quant_device_name"]) {
    const v = env?.[k];
    if (typeof v === "string" && v && v !== "cpu") return v;
  }
  return undefined;
}
