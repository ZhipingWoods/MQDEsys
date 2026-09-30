import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, fmt, fmtBytes, gib, JobListItem, JobReport, llamaSpeed, LLMJobReport } from "../api";
import { baseOption, Chart, series, token } from "../charts";

/** Accuracy vs. cost across finished jobs: VLA (action error vs. deployed memory) and LLM (PPL vs. weight bits). */
export default function ComparePage() {
  const [jobs, setJobs] = useState<JobListItem[]>([]);
  const [reports, setReports] = useState<(JobReport | LLMJobReport)[]>([]);
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "vla" ? "vla" : "llm";
  useEffect(() => {
    api.jobs().then(async (js) => {
      const done = js.filter((j) => j.state === "SUCCEEDED");
      setJobs(done);
      const reps = await Promise.all(done.slice(0, 30).map((j) => api.job(j.job_id).then((d) => d.report).catch(() => null)));
      setReports(reps.filter((r): r is JobReport => !!r && !!r.recipes && Object.keys(r.recipes).length > 0));
    });
  }, []);
  const llm = reports.filter((r): r is LLMJobReport => r.kind === "llm");
  const vla = reports.filter((r): r is JobReport => r.kind !== "llm");
  return (
    <>
      <h1>配方对比</h1>
      <p className="subtitle">汇总已完成作业：精度与成本的权衡。LLM {llm.length} 个作业 · VLA {vla.length} 个作业。</p>
      <div className="tabs">
        <div className={`tab ${tab === "llm" ? "on" : ""}`} onClick={() => setParams({ tab: "llm" }, { replace: true })}>
          LLM（PPL）
        </div>
        <div className={`tab ${tab === "vla" ? "on" : ""}`} onClick={() => setParams({ tab: "vla" }, { replace: true })}>
          VLA（动作误差）
        </div>
      </div>
      {tab === "llm" ? <LLMCompare reps={llm} /> : <VLACompare reps={vla} />}
      {jobs.length === 0 && <p className="muted">暂无已完成作业</p>}
    </>
  );
}

function LLMCompare({ reps }: { reps: LLMJobReport[] }) {
  const pts = useMemo(
    () =>
      reps.flatMap((r) =>
        Object.values(r.recipes ?? {}).map((rr) => {
          let pb = 0,
            p = 0;
          for (const l of rr.layers) {
            const n = l.shape[0] * l.shape[1];
            p += n;
            pb += n * (l.algo_info?.bits ?? rr.weight.bits ?? 16);
          }
          const mp = rr.layers.some((l) => l.algo_info?.bits != null && l.algo_info.bits !== rr.weight.bits);
          return {
            job: r.job_id,
            name: r.spec.name,
            model: r.spec.model_path.split("/").pop() ?? "",
            label: `${rr.recipe}/${rr.algorithm}${mp ? "+MP" : ""}${r.spec.act_static ? "+static" : ""}`,
            recipe: rr.recipe,
            bits: p ? pb / p : 16,
            ppl: rr.eval.fake_quant_wo_kv.ppl,
            pplKv: rr.eval.fake_quant?.ppl,
            base: r.baseline?.ppl,
            real: Object.values(rr.export ?? {}).find((e) => e.real_ppl != null)?.real_ppl,
          };
        }),
      ),
    [reps],
  );
  const recipes = [...new Set(pts.map((p) => p.recipe))];
  const gg = reps.flatMap((r) => Object.values(r.llamacpp?.results ?? {}).map((x) => ({ r, x })));
  return (
    <>
      <div className="card">
        <h2>PPL（W+A，越低越好）vs 量化线性层平均位宽</h2>
        <Chart
          height={380}
          option={() => {
            const o = baseOption();
            return {
              ...o,
              legend: { ...(o.legend as object), data: recipes },
              grid: { ...(o.grid as object), top: 56, bottom: 44 },
              tooltip: {
                ...(o.tooltip as object),
                formatter: (p: { data: [number, number, string, string] }) => `${p.data[2]}<br/>${p.data[3]}<br/>平均 ${p.data[0].toFixed(2)} bit · PPL ${p.data[1].toFixed(3)}`,
              },
              xAxis: { ...(o.xAxis as object), type: "value", name: "平均权重位宽 (bit)", nameLocation: "middle", nameGap: 28, scale: true, nameTextStyle: { color: token("--text-muted") } },
              yAxis: { ...(o.yAxis as object), type: "log", name: "PPL", nameTextStyle: { color: token("--text-muted") } },
              series: recipes.map((rc, i) => ({
                name: rc,
                type: "scatter",
                symbolSize: 11,
                itemStyle: { color: series(i), borderColor: token("--surface-1"), borderWidth: 2 },
                data: pts.filter((p) => p.recipe === rc).map((p) => [p.bits, p.ppl, p.label, `${p.model} · ${p.job}`]),
              })),
            };
          }}
        />
        <p className="note">同一模型、同一评估窗口的作业才可直接比较。W2 崩溃点（PPL 上千）用对数纵轴显示。</p>
      </div>
      <div className="card table-wrap">
        <h2>各作业配方明细</h2>
        <table>
          <thead>
            <tr>
              <th>作业</th>
              <th>模型</th>
              <th>配方 / 算法</th>
              <th className="num">平均位宽</th>
              <th className="num">原模型</th>
              <th className="num">PPL（W+A）</th>
              <th className="num">PPL（+KV）</th>
              <th className="num">真实导出 PPL</th>
            </tr>
          </thead>
          <tbody>
            {pts
              .slice()
              .sort((a, b) => a.ppl - b.ppl)
              .map((p) => (
                <tr key={p.job + p.label}>
                  <td>
                    <Link to={`/jobs/${p.job}`} className="mono small">
                      {p.job}
                    </Link>
                  </td>
                  <td className="small">{p.model}</td>
                  <td>
                    <span className="badge accent">{p.label}</span>
                  </td>
                  <td className="num">{p.bits.toFixed(2)}</td>
                  <td className="num">{fmt(p.base, 3)}</td>
                  <td className="num">{fmt(p.ppl, 3)}</td>
                  <td className="num">{fmt(p.pplKv, 3)}</td>
                  <td className="num">{fmt(p.real, 3)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      {gg.length > 0 && (
        <div className="card table-wrap">
          <h2>llama.cpp 部署实测（GGUF）</h2>
          <table>
            <thead>
              <tr>
                <th>作业</th>
                <th>类型</th>
                <th className="num">文件大小</th>
                <th className="num">PPL</th>
                <th className="num">预填充 tok/s</th>
                <th className="num">解码 tok/s</th>
              </tr>
            </thead>
            <tbody>
              {gg.map(({ r, x }) => (
                <tr key={r.job_id + x.type}>
                  <td>
                    <Link to={`/jobs/${r.job_id}`} className="mono small">
                      {r.job_id}
                    </Link>
                  </td>
                  <td>{x.type.toUpperCase()}</td>
                  <td className="num">{fmtBytes(x.bytes)}</td>
                  <td className="num">{fmt(x.ppl, 3)}</td>
                  <td className="num">{fmt(llamaSpeed(x).pp, 0)}</td>
                  <td className="num">{fmt(llamaSpeed(x).tg, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">llama-perplexity 只统计每个 chunk 的后半段，数值与上表的 PPL 协议不同，只在本表内比较。</p>
        </div>
      )}
    </>
  );
}

function VLACompare({ reps }: { reps: JobReport[] }) {
  const points = reps.flatMap((r) =>
    Object.values(r.recipes ?? {}).map((rr) => {
      const est = r.preflight?.estimates ?? {};
      const k = Object.keys(est).find((x) => x.toLowerCase() === rr.recipe.toLowerCase());
      const peak = k ? Object.values(est[k])[0]?.peak_bytes : undefined;
      return { job: r.job_id, name: r.spec.name, recipe: rr.recipe, err: rr.eval?.action_rel_l2, peak, nf: r.baseline?.noise_floor?.action_rel_l2 };
    }),
  );
  const recipes = [...new Set(points.map((p) => p.recipe))];
  return (
    <>
      <div className="card">
        <h2>动作误差 vs 预估部署峰值显存（左下角 = 更省且更准）</h2>
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
                data: points.filter((p) => p.recipe === rc && p.peak && p.err != null).map((p) => [gib(p.peak!), p.err, p.job, p.name]),
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
      </div>
    </>
  );
}
