import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, JobListItem } from "../api";

export function StateBadge({ state }: { state: string }) {
  const cls = state === "SUCCEEDED" ? "good" : state === "FAILED" ? "critical" : state === "RUNNING" || state === "QUEUED" ? "running" : "warning";
  const label: Record<string, string> = { SUCCEEDED: "成功", FAILED: "失败", RUNNING: "运行中", QUEUED: "排队中", CANCELLED: "已取消" };
  return <span className={`status ${cls}`}>{label[state] ?? state}</span>;
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<JobListItem[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [kind, setKind] = useState("");
  const [state, setState] = useState("");
  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return (jobs ?? []).filter((j) => (!kind || (j.kind ?? "vla") === kind) && (!state || j.state === state) &&
      (!needle || [j.job_id, j.name, j.model, ...(j.recipes ?? [])].some((x) => (x ?? "").toLowerCase().includes(needle))));
  }, [jobs, q, kind, state]);
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
      <h1>作业与执行过程</h1>
      <p className="subtitle">
        所有作业（含 CLI 运行的）都在 runs/ 下，服务重启不丢失。<Link to="/new">新建作业 →</Link>
      </p>
      {err && <div className="error-box">{err}</div>}
      <div className="card">
        <div className="form-row" style={{ alignItems: "flex-end" }}>
          <div className="field grow">
            <label>搜索（作业号 / 名称 / 模型 / 配方）</label>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="如 minicpm、w4a16、20261013" data-testid="jobs-search" />
          </div>
          <div className="field">
            <label>类型</label>
            <select value={kind} onChange={(e) => setKind(e.target.value)} data-testid="jobs-kind">
              <option value="">全部</option>
              <option value="llm">LLM</option>
              <option value="vlm">VLM</option>
              <option value="vla">VLA</option>
              <option value="llm_deploy">部署评测</option>
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
      <div className="card table-wrap">
        <table>
          <thead>
            <tr>
              <th>作业</th>
              <th>名称</th>
              <th>类型</th>
              <th>状态</th>
              <th>当前阶段</th>
              <th>配方</th>
              <th className="num">设备</th>
              <th className="num">耗时</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((j) => (
              <tr key={j.job_id} data-testid={`job-row-${j.job_id}`}>
                <td>
                  <Link to={`/jobs/${j.job_id}`} className="mono">
                    {j.job_id}
                  </Link>
                  {j.parent_job && (
                    <div className="small muted" title={`派生自 ${j.parent_job}`}>
                      ↳ <Link to={`/jobs/${j.parent_job}`} className="mono">{j.parent_job.slice(-6)}</Link>
                    </div>
                  )}
                </td>
                <td title={j.model}>{j.name ?? "–"}</td>
                <td>
                  <span className="badge">{j.kind === "llm" ? "LLM" : j.kind === "vlm" ? "VLM" : j.kind === "llm_deploy" ? "部署评测" : "VLA"}</span>
                </td>
                <td>
                  <StateBadge state={j.state} />
                </td>
                <td>{j.stage ?? "–"}</td>
                <td className="small">{(j.recipes ?? []).join(", ")}</td>
                <td className="num" title={j.device_name}>{j.device === "cpu" ? "CPU" : j.gpu != null && j.gpu >= 0 ? `GPU ${j.gpu}` : j.device_name ? shortDevice(j.device_name) : "GPU"}</td>
                <td className="num">{j.started_ts && j.last_ts ? `${Math.round(j.last_ts - j.started_ts)} s` : "–"}</td>
              </tr>
            ))}
            {jobs && shown.length === 0 && (
              <tr>
                <td colSpan={8} className="empty">{jobs.length === 0 ? "还没有作业" : "没有符合筛选条件的作业"}</td>
              </tr>
            )}
            {jobs?.length === 0 && (
              <tr>
                <td colSpan={8} className="muted">
                  暂无作业
                </td>
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
