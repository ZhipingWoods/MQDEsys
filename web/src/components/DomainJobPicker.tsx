// Landing view of a page that works on one finished job (误差分析 / 优化建议) when it is opened from the sidebar:
// the current domain's finished jobs, newest first, each linking to this page with that job.
import { Link } from "react-router-dom";
import { api, JobListItem } from "../api";
import { DOMAIN_SHORT, jobDomain, useDomain } from "../domain";
import { useAsync } from "../hooks";

export function DomainJobPicker({ base, kinds, what }: { base: string; kinds?: string[]; what: string }) {
  const { domain } = useDomain();
  const jobs = useAsync(api.jobs, []);
  const rows = (jobs.data ?? [])
    .filter((j: JobListItem) => j.state === "SUCCEEDED" && jobDomain(j.kind) === domain && (!kinds || kinds.includes(j.kind ?? "vla")))
    .sort((a, b) => (b.last_ts ?? 0) - (a.last_ts ?? 0));
  return (
    <div className="card table-wrap" data-testid="job-picker">
      <h2>选择一个 {DOMAIN_SHORT[domain] ?? domain} 作业查看{what}</h2>
      {jobs.error && <div className="error-box">{jobs.error}</div>}
      <table>
        <thead>
          <tr><th>作业</th><th>名称</th><th>配方</th><th className="num">完成于</th></tr>
        </thead>
        <tbody>
          {rows.map((j) => (
            <tr key={j.job_id} data-testid={`pick-${j.job_id}`}>
              <td><Link to={`${base}/${j.job_id}`} className="mono">{j.job_id}</Link></td>
              <td title={j.model}>{j.name ?? "–"}</td>
              <td className="small">{(j.recipes ?? []).join(", ")}</td>
              <td className="num small muted">{j.last_ts ? new Date(j.last_ts * 1000).toLocaleString() : "–"}</td>
            </tr>
          ))}
          {jobs.data && rows.length === 0 && (
            <tr><td colSpan={4} className="empty">当前领域还没有可查看{what}的已完成作业。<Link to="/new">新建作业 →</Link></td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
