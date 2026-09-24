import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmt, fmtBytes, gib, JobListItem, JobReport } from "../api";
import { baseOption, Chart, series, token } from "../charts";

/** Accuracy (action error) vs. estimated deployed memory, across all finished jobs' recipes. */
export default function ComparePage() {
  const [jobs, setJobs] = useState<JobListItem[]>([]);
  const [reports, setReports] = useState<JobReport[]>([]);
  useEffect(() => {
    api.jobs().then(async (js) => {
      const done = js.filter((j) => j.state === "SUCCEEDED");
      setJobs(done);
      const reps = await Promise.all(done.slice(0, 12).map((j) => api.job(j.job_id).then((d) => d.report).catch(() => null)));
      setReports(reps.filter((r): r is JobReport => !!r && !!r.recipes && Object.keys(r.recipes).length > 0));
    });
  }, []);

  const points = reports.flatMap((r) =>
    Object.values(r.recipes ?? {}).map((rr) => {
      const est = r.preflight?.estimates ?? {};
      const k = Object.keys(est).find((x) => x.toLowerCase() === rr.recipe.toLowerCase());
      const peak = k ? Object.values(est[k])[0]?.peak_bytes : undefined;
      return { job: r.job_id, name: r.spec.name, recipe: rr.recipe, err: rr.eval.action_rel_l2, peak, nf: r.baseline?.noise_floor.action_rel_l2 };
    }),
  );
  const recipes = [...new Set(points.map((p) => p.recipe))];

  return (
    <>
      <h1>配方对比</h1>
      <p className="subtitle">精度（动作误差）与部署显存（C++ 预估）的权衡，汇总 {reports.length} 个已完成作业。左下角 = 更省显存且更准。</p>
      <div className="card">
        <Chart
          height={360}
          option={() => {
            const o = baseOption();
            return {
              ...o,
              legend: { ...(o.legend as object), data: recipes },
              grid: { ...(o.grid as object), top: 56, bottom: 44 },
              tooltip: {
                ...(o.tooltip as object),
                formatter: (p: { seriesName?: string; data: [number, number, string, string] }) => `${p.seriesName ?? ""} · ${p.data[3]}<br/>预估峰值 ${p.data[0].toFixed(2)} GiB<br/>动作 rel-L2 ${p.data[1].toFixed(4)}`,
              },
              xAxis: { ...(o.xAxis as object), type: "value", name: "预估部署峰值显存 (GiB)", nameLocation: "middle", nameGap: 28, scale: true, nameTextStyle: { color: token("--text-muted") } },
              yAxis: { ...(o.yAxis as object), type: "log", name: "动作 rel-L2", nameTextStyle: { color: token("--text-muted") } },
              series: recipes.map((rc, i) => ({
                name: rc,
                type: "scatter",
                symbolSize: 12,
                itemStyle: { color: series(i), borderColor: token("--surface-1"), borderWidth: 2 },
                label: { show: true, position: "right", formatter: rc, color: token("--text-secondary"), fontSize: 11 },
                data: points.filter((p) => p.recipe === rc && p.peak).map((p) => [gib(p.peak!), p.err, p.job, p.name]),
              })),
            };
          }}
        />
      </div>
      <div className="card table-wrap">
        <table>
          <thead>
            <tr>
              <th>作业</th>
              <th>配方</th>
              <th className="num">动作 rel-L2</th>
              <th className="num">噪声下限</th>
              <th className="num">预估峰值显存</th>
            </tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.job + p.recipe}>
                <td>
                  <Link to={`/jobs/${p.job}`} className="mono small">
                    {p.job}
                  </Link>{" "}
                  <span className="muted small">{p.name}</span>
                </td>
                <td>{p.recipe}</td>
                <td className="num">{fmt(p.err, 4)}</td>
                <td className="num">{fmt(p.nf, 4)}</td>
                <td className="num">{fmtBytes(p.peak)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {jobs.length === 0 && <p className="muted">暂无已完成作业</p>}
      </div>
    </>
  );
}
