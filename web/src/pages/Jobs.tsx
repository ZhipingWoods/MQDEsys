import { useEffect, useState } from "react";
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
            {jobs?.map((j) => (
              <tr key={j.job_id}>
                <td>
                  <Link to={`/jobs/${j.job_id}`} className="mono">
                    {j.job_id}
                  </Link>
                </td>
                <td>{j.name ?? "–"}</td>
                <td>
                  <span className="badge">{j.kind === "llm" ? "LLM" : j.kind === "vlm" ? "VLM" : "VLA"}</span>
                </td>
                <td>
                  <StateBadge state={j.state} />
                </td>
                <td>{j.stage ?? "–"}</td>
                <td className="small">{(j.recipes ?? []).join(", ")}</td>
                <td className="num">{j.device === "cpu" ? "CPU" : j.gpu != null && j.gpu >= 0 ? `GPU ${j.gpu}` : "–"}</td>
                <td className="num">{j.started_ts && j.last_ts ? `${Math.round(j.last_ts - j.started_ts)} s` : "–"}</td>
              </tr>
            ))}
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
