import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, DeployJobReport, fmt, fmtBytes, gib, JobListItem, JobReport, llamaSpeed, LLMJobReport, VLMJobReport } from "../api";
import { engineNote, servingRuns } from "../perf";
import { baseOption, Chart, series, token } from "../charts";

/** Accuracy vs. cost across finished jobs: VLA (action error vs. deployed memory) and LLM (PPL vs. weight bits). */
export default function ComparePage() {
  const [jobs, setJobs] = useState<JobListItem[]>([]);
  const [reports, setReports] = useState<(JobReport | LLMJobReport | VLMJobReport | DeployJobReport)[]>([]);
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "vla" ? "vla" : params.get("tab") === "vlm" ? "vlm" : "llm";
  useEffect(() => {
    api.jobs().then(async (js) => {
      const done = js.filter((j) => j.state === "SUCCEEDED");
      setJobs(done);
      const reps = await Promise.all(done.slice(0, 30).map((j) => api.job(j.job_id).then((d) => d.report).catch(() => null)));
      setReports(reps.filter((r): r is JobReport => !!r && (r.kind === "llm_deploy" || (!!r.recipes && Object.keys(r.recipes).length > 0))));
    });
  }, []);
  const llm = reports.filter((r): r is LLMJobReport => r.kind === "llm");
  const vlm = reports.filter((r): r is VLMJobReport => (r.kind as string) === "vlm");
  const dep = reports.filter((r): r is DeployJobReport => r.kind === "llm_deploy");
  const vla = reports.filter((r): r is JobReport => r.kind !== "llm" && (r.kind as string) !== "vlm" && r.kind !== "llm_deploy");
  return (
    <>
      <h1>配方对比</h1>
      <p className="subtitle">汇总已完成作业：精度与成本的权衡。LLM {llm.length} 个作业 · VLM {vlm.length} 个作业 · VLA {vla.length} 个作业。</p>
      <div className="tabs">
        <div className={`tab ${tab === "llm" ? "on" : ""}`} onClick={() => setParams({ tab: "llm" }, { replace: true })}>
          LLM（PPL）
        </div>
        <div className={`tab ${tab === "vlm" ? "on" : ""}`} onClick={() => setParams({ tab: "vlm" }, { replace: true })} data-testid="compare-tab-vlm">
          VLM（答案分布）
        </div>
        <div className={`tab ${tab === "vla" ? "on" : ""}`} onClick={() => setParams({ tab: "vla" }, { replace: true })}>
          VLA（动作误差）
        </div>
      </div>
      {tab === "llm" ? <LLMCompare reps={llm} deploys={dep} /> : tab === "vlm" ? <VLMCompare reps={vlm} /> : <VLACompare reps={vla} />}
      {jobs.length === 0 && <p className="muted">暂无已完成作业</p>}
    </>
  );
}

/** The measured device name; older reports only have the torch device string, which is not a hardware name. */
function hwLabel(r: LLMJobReport): string {
  if (r.env?.eval_device_name) return r.env.eval_device_name;
  const d = r.env?.eval_device || r.spec?.device || "";
  return d === "cpu" ? "CPU" : /^cuda/.test(d) ? "GPU（型号未记录）" : d || "unknown";
}

function VLMCompare({ reps }: { reps: VLMJobReport[] }) {
  const rows = reps.flatMap((r) => Object.values(r.recipes ?? {}).filter((x) => x.summary.kl != null).map((x) => ({ job: r.job_id, name: r.spec?.name, x })));
  if (rows.length === 0) return <p className="muted">暂无带评测结果的 VLM 作业</p>;
  const pct = (v: number | null) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
  return (
    <div className="card">
      <h2>VLM 配方：答案分布的变化（只统计答案位置）</h2>
      <table data-testid="vlm-compare-table">
        <thead>
          <tr>
            <th>作业</th>
            <th>配方</th>
            <th>算法</th>
            <th>KL</th>
            <th>Top-1 一致</th>
            <th>答案完全一致</th>
            <th>答对率</th>
          </tr>
        </thead>
        <tbody>
          {rows
            .sort((a, b) => (a.x.summary.kl ?? 0) - (b.x.summary.kl ?? 0))
            .map(({ job, name, x }) => (
              <tr key={job + x.recipe}>
                <td>
                  <Link to={`/jobs/${job}`}>{name ?? job}</Link>
                </td>
                <td>{x.recipe}</td>
                <td>{x.algorithm}</td>
                <td className="num">{x.summary.kl!.toExponential(2)}</td>
                <td className="num">{pct(x.summary.top1_agreement)}</td>
                <td className="num">{pct(x.summary.generation_exact_match)}</td>
                <td className="num">{pct(x.summary.gt_accuracy)}</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}

function LLMCompare({ reps, deploys }: { reps: LLMJobReport[]; deploys: DeployJobReport[] }) {
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
  // T-1308: same recipe run on more than one real hardware (distinguished by the actually-measured
  // eval_device_name, not the submitted deploy.hardware label) -- throughput (stage wall-clock seconds)
  // should differ by hardware, quantization error (PPL) should not. Hidden entirely when every job in
  // the comparison ran on the same hardware, rather than showing a one-row "comparison" that compares
  // nothing.
  const stageSeconds = (r: LLMJobReport, stage: string) => r.stages?.find((s) => s.stage === stage)?.seconds;
  const hwPts = useMemo(
    () =>
      reps.flatMap((r) =>
        Object.values(r.recipes ?? {}).map((rr) => ({
          job: r.job_id,
          recipe: rr.recipe,
          hw: hwLabel(r),
          quantizeSeconds: stageSeconds(r, "quantize"),
          evalSeconds: stageSeconds(r, "eval_fake_wo_kv"),
          ppl: rr.eval.fake_quant_wo_kv.ppl,
        })),
      ),
    [reps],
  );
  const comparableRecipes = recipes.filter((rc) => new Set(hwPts.filter((p) => p.recipe === rc).map((p) => p.hw)).size > 1);
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
      <PerfCompare reps={[...reps, ...deploys]} />
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
      {comparableRecipes.length > 0 && (
        <div className="card table-wrap" data-testid="hardware-compare">
          <h2>按硬件对比（同一配方，不同硬件）</h2>
          <p className="note">同一配方在不同硬件上的吞吐（量化/评估耗时）与误差（PPL）对比：误差应基本不随硬件变化，耗时应该随硬件明显不同。</p>
          <table>
            <thead>
              <tr>
                <th>配方</th>
                <th>硬件</th>
                <th>作业</th>
                <th className="num">量化耗时</th>
                <th className="num">评估耗时</th>
                <th className="num">PPL（W+A）</th>
              </tr>
            </thead>
            <tbody>
              {comparableRecipes.flatMap((rc) =>
                hwPts
                  .filter((p) => p.recipe === rc)
                  .map((p) => (
                    <tr key={p.job + p.recipe} data-testid={`hw-compare-row-${p.recipe}-${p.hw}`}>
                      <td>
                        <span className="badge accent">{p.recipe}</span>
                      </td>
                      <td className="mono small">{p.hw}</td>
                      <td>
                        <Link to={`/jobs/${p.job}`} className="mono small">
                          {p.job}
                        </Link>
                      </td>
                      <td className="num">{fmt(p.quantizeSeconds, 2)} s</td>
                      <td className="num">{fmt(p.evalSeconds, 2)} s</td>
                      <td className="num">{fmt(p.ppl, 3)}</td>
                    </tr>
                  )),
              )}
            </tbody>
          </table>
        </div>
      )}
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

/** T-R13: concurrency sweeps of every job that has one, side by side (throughput vs concurrency, saturation, tail latency). */
function PerfCompare({ reps }: { reps: (LLMJobReport | DeployJobReport)[] }) {
  const runs = useMemo(() => servingRuns(reps), [reps]);
  if (!runs.length) return null;
  const levels = [...new Set(runs.flatMap((r) => r.sweep.points.map((p) => p.concurrency)))].sort((a, b) => a - b);
  const name = (r: (typeof runs)[number]) => `${r.job.slice(-6)} · ${r.label} · ${r.backend}`;
  const note = engineNote(runs);
  return (
    <div className="card" data-testid="perf-compare">
      <h2>服务吞吐 vs 并发（{runs.length} 条并发扫描）</h2>
      <Chart
        height={340}
        option={() => {
          const o = baseOption();
          return {
            ...o,
            legend: { ...(o.legend as object), data: runs.map(name), type: "scroll" },
            grid: { ...(o.grid as object), top: 56, bottom: 44 },
            xAxis: { ...(o.xAxis as object), type: "category", data: levels.map(String), name: "并发", nameLocation: "middle", nameGap: 28, nameTextStyle: { color: token("--text-muted") } },
            yAxis: { ...(o.yAxis as object), type: "value", name: "输出 tok/s", nameTextStyle: { color: token("--text-muted") } },
            series: runs.map((r, i) => ({
              name: name(r),
              type: "line",
              symbolSize: 7,
              itemStyle: { color: series(i) },
              connectNulls: true,
              data: levels.map((c) => r.sweep.points.find((p) => p.concurrency === c)?.output_tokens_per_s ?? null),
            })),
          };
        }}
      />
      <table data-testid="perf-compare-table">
        <thead>
          <tr><th>作业</th><th>服务的模型</th><th>引擎</th><th>峰值 tok/s</th><th>@并发</th><th>饱和点</th><th>峰值处 TTFT p99</th></tr>
        </thead>
        <tbody>
          {[...runs].sort((a, b) => b.peak - a.peak).map((r) => (
            <tr key={`${r.job}-${r.label}-${r.backend}`} data-testid={`perf-row-${r.job}-${r.backend}`}>
              <td><Link to={`/jobs/${r.job}`}>{r.job.slice(-6)}</Link></td>
              <td>{r.label}</td>
              <td>{r.backend}</td>
              <td>{fmt(r.peak, 0)}</td>
              <td>{r.sweep.peak?.concurrency ?? "—"}</td>
              <td>{r.saturation ?? "未饱和"}</td>
              <td>{r.sweep.peak?.ttft_ms?.p99 != null ? `${fmt(r.sweep.peak.ttft_ms.p99, 0)} ms` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {note && <p className="note" data-testid="perf-engine-note">{note}</p>}
    </div>
  );
}
