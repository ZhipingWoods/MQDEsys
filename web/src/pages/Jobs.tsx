import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, JobListItem } from "../api";
import { DOMAIN_SHORT, jobDomain, useDomain } from "../domain";
import { Tile } from "./Estimate";

export function StateBadge({ state }: { state: string }) {
  const cls = state === "SUCCEEDED" ? "good" : state === "FAILED" ? "critical" : state === "RUNNING" || state === "QUEUED" ? "running" : "warning";
  const label: Record<string, string> = { SUCCEEDED: "成功", FAILED: "失败", RUNNING: "运行中", QUEUED: "排队中", CANCELLED: "已取消" };
  return <span className={`status ${cls}`}>{label[state] ?? state}</span>;
}

type Mode = "full" | "quantize" | "eval" | "deploy" | "unknown";

/** Explicit stages are the source of truth; old jobs without them fall back to the report kind. */
function modeOf(j: JobListItem): Mode {
  const s = new Set(j.stages ?? []);
  if (s.has("quantize") && s.has("evaluate")) return "full";
  if (s.has("quantize")) return "quantize";
  if (s.has("evaluate")) return "eval";
  if (j.kind === "llm_deploy") return "deploy";
  if (j.stages?.length) return "unknown";
  return j.kind === "llm" ? "full" : "unknown";
}

const MODE: Record<Mode, { label: string; cls: string }> = {
  full: { label: "全流程", cls: "accent" },
  quantize: { label: "仅量化", cls: "good" },
  eval: { label: "仅评测", cls: "warn" },
  deploy: { label: "部署评测", cls: "critical" },
  unknown: { label: "领域作业", cls: "" },
};
const KIND_LABEL: Record<string, string> = { vlm: "VLM", vla: "VLA", world: "WM", ewm: "EWM", llm: "LLM" };

const DURATION = (j: JobListItem) => (j.started_ts && j.last_ts ? Math.round(j.last_ts - j.started_ts) : null);
const fmtDuration = (s: number | null) => (s == null ? "–" : s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s` : `${(s / 3600).toFixed(1)}h`);

export default function JobsPage() {
  const nav = useNavigate();
  const [jobs, setJobs] = useState<JobListItem[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const { domain } = useDomain();
  const [kind, setKind] = useState(""); // "" = every domain; "domain" = the navigation bar's domain
  const [state, setState] = useState("");
  const [mode, setMode] = useState<"all" | Mode>("all");
  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return (jobs ?? []).filter((j) =>
      (!kind || (kind === "domain" ? jobDomain(j.kind) === domain : (j.kind ?? "vla") === kind)) &&
      (!state || j.state === state) &&
      (mode === "all" || modeOf(j) === mode) &&
      (!needle || [j.job_id, j.name, j.model, ...(j.recipes ?? [])].some((x) => (x ?? "").toLowerCase().includes(needle))));
  }, [jobs, q, kind, state, mode, domain]);
  const counts = useMemo(() => {
    const c = { running: 0, failed: 0, ok: 0 };
    for (const j of jobs ?? []) { if (j.state === "RUNNING" || j.state === "QUEUED") c.running++; else if (j.state === "FAILED") c.failed++; else if (j.state === "SUCCEEDED") c.ok++; }
    return c;
  }, [jobs]);
  useEffect(() => {
    let alive = true;
    const load = () => api.jobs().then((j) => alive && setJobs(j), (e: Error) => alive && setErr(e.message));
    load();
    const t = setInterval(load, 3000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);
  return (
    <>
      <div className="page-head">
        <div>
          <h1>作业与执行过程</h1>
          <p className="subtitle">
            所有作业（含 CLI 运行的）都在 runs/ 下，服务重启不丢失；每 3 秒自动刷新。
          </p>
        </div>
        <div className="page-head-actions">
          <Link to="/new/quantize"><button className="secondary">+ 仅量化</button></Link>
          <Link to="/new/eval"><button className="secondary">+ 仅评测</button></Link>
          <Link to="/new"><button>+ 全流程</button></Link>
        </div>
      </div>
      {err && <div className="error-box">{err}</div>}
      <div className="tiles">
        <Tile icon="activity" label="运行中 / 排队" value={`${counts.running}`} hint={counts.running ? "正在占用 GPU 池" : "当前没有活动作业"} />
        <Tile icon="play" label="成功" value={`${counts.ok}`} hint="已完成并产出报告" />
        <Tile icon="alert" label="失败" value={`${counts.failed}`} hint={counts.failed ? "可打开日志定位阶段" : "没有失败作业"} />
        <Tile icon="library" label="全部作业" value={`${jobs?.length ?? "–"}`} hint={`当前筛选显示 ${shown.length} 个`} />
      </div>
      <div className="card jobs-filter">
        <div className="form-row" style={{ alignItems: "flex-end" }}>
          <div className="field grow">
            <label>搜索（作业号 / 名称 / 模型 / 配方）</label>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="如 minicpm、w4a16、20261013" data-testid="jobs-search" />
          </div>
          <div className="field">
            <label>领域 / 类型</label>
            <select value={kind} onChange={(e) => setKind(e.target.value)} data-testid="jobs-kind">
              <option value="">全部领域</option>
              <option value="domain">当前领域（{DOMAIN_SHORT[domain] ?? domain}）</option>
              <option value="llm">LLM</option>
              <option value="vlm">VLM</option>
              <option value="vla">VLA</option>
              <option value="world">WM</option>
              <option value="ewm">EWM</option>
              <option value="llm_deploy">部署评测</option>
            </select>
          </div>
          <div className="field">
            <label>作业模式</label>
            <select value={mode} onChange={(e) => setMode(e.target.value as typeof mode)} data-testid="jobs-mode">
              <option value="all">全部</option>
              <option value="full">全流程</option>
              <option value="quantize">仅量化</option>
              <option value="eval">仅评测</option>
              <option value="deploy">部署评测</option>
            </select>
          </div>
          <div className="field">
            <label>状态</label>
            <select value={state} onChange={(e) => setState(e.target.value)} data-testid="jobs-state">
              <option value="">全部</option>
              <option value="RUNNING">运行中</option>
              <option value="QUEUED">排队中</option>
              <option value="SUCCEEDED">成功</option>
              <option value="FAILED">失败</option>
              <option value="CANCELLED">已取消</option>
            </select>
          </div>
          <div className="field">
            <label>&nbsp;</label>
            <div className="small muted" style={{ minHeight: 32, display: "flex", alignItems: "center", gap: 10 }} data-testid="jobs-counts">
              <span>{shown.length} / {jobs?.length ?? "–"}</span>
              {counts.running > 0 && <span className="status running">运行中 {counts.running}</span>}
              {counts.failed > 0 && <span className="status critical">失败 {counts.failed}</span>}
              <span className="status good">成功 {counts.ok}</span>
            </div>
          </div>
        </div>
      </div>
      <div className="card table-wrap jobs-table-card">
        <table className="jobs-table">
          <thead>
            <tr>
              <th>作业</th>
              <th>名称 / 模型</th>
              <th>模式</th>
              <th>状态</th>
              <th>阶段 / 进度</th>
              <th>配方</th>
              <th className="num">设备</th>
              <th className="num">耗时</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {shown.map((j) => {
              const m = MODE[modeOf(j)];
              const modeLabel = modeOf(j) === "unknown" ? (KIND_LABEL[j.kind ?? ""] ?? m.label) : m.label;
              const pct = j.progress != null ? Math.max(0, Math.min(100, j.progress * 100)) : j.trace_total ? Math.max(0, Math.min(100, ((j.trace_index ?? 0) + 1) * 100 / j.trace_total)) : null;
              return (
                <tr key={j.job_id} data-testid={`job-row-${j.job_id}`} className="clickable" onClick={() => nav(`/jobs/${j.job_id}`)}>
                  <td>
                    <Link to={`/jobs/${j.job_id}`} className="mono job-id-link" title={j.job_id} onClick={(e) => e.stopPropagation()}>
                      {j.job_id}
                    </Link>
                    {j.parent_job && (
                      <div className="small muted" title={`派生自 ${j.parent_job}`}>
                        ↳ <Link to={`/jobs/${j.parent_job}`} className="mono" onClick={(e) => e.stopPropagation()}>{j.parent_job.slice(-6)}</Link>
                      </div>
                    )}
                  </td>
                  <td className="job-name-cell">
                    <b>{j.name ?? "–"}</b>
                    <div className="small muted" title={j.model}>{j.model ?? "–"}</div>
                  </td>
                  <td><span className={`badge ${m.cls}`}>{modeLabel}</span></td>
                  <td><StateBadge state={j.state} /></td>
                  <td className="job-stage-cell">
                    <div className="small">{j.stage ?? "–"}</div>
                    {(j.state === "RUNNING" || j.state === "QUEUED") && (
                      <div className="job-progress">
                        <span style={{ width: `${pct ?? 8}%` }} />
                      </div>
                    )}
                    {pct != null && <div className="small muted">{Math.round(pct)}%{j.progress_msg ? ` · ${j.progress_msg}` : ""}</div>}
                  </td>
                  <td className="small" title={(j.recipes ?? []).join(", ")}>
                    {(j.recipes ?? []).slice(0, 3).join(", ") || "–"}{(j.recipes?.length ?? 0) > 3 ? ` +${(j.recipes?.length ?? 0) - 3}` : ""}
                  </td>
                  <td className="num" title={j.device_name}>{j.device === "cpu" ? "CPU" : j.gpu != null && j.gpu >= 0 ? `GPU ${j.gpu}` : j.device_name ? shortDevice(j.device_name) : "GPU"}</td>
                  <td className="num">{fmtDuration(DURATION(j))}</td>
                  <td className="job-open">→</td>
                </tr>
              );
            })}
            {jobs && shown.length === 0 && (
              <tr>
                <td colSpan={9} className="empty">{jobs.length === 0 ? "还没有作业" : "没有符合筛选条件的作业"}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}

/** "NVIDIA GeForce RTX 4090" -> "RTX 4090": the column is narrow and every GPU here is NVIDIA. */
const shortDevice = (n: string) => n.replace(/^NVIDIA\s+/, "").replace(/^GeForce\s+/, "");
