import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, fmt, fmtBytes, VLMJobReport, VLMRecipeReport } from "../api";
import { baseOption, Chart, series, token } from "../charts";
import { Tile } from "./Estimate";
import { LayerAnalysis } from "./LayerAnalysis";

export const VLM_STAGES: [string, string][] = [
  ["validate", "校验"],
  ["inspect", "模型加载"],
  ["calib_prepare", "数据准备"],
  ["reference", "原模型答案"],
  ["attribution", "模态归因"],
  ["quantize", "逐层量化"],
  ["eval_quantized", "量化后评测"],
  ["report", "报告"],
];

const MODALITY: Record<string, string> = { vision: "视觉塔 + 投影", language: "语言模型" };
const pct = (v: number | null | undefined, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
const signed = (v: number | null | undefined) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${Math.abs(v) >= 1e5 ? v.toExponential(2) : v.toFixed(3)}%`);

export function orderedVlm(rep: VLMJobReport): VLMRecipeReport[] {
  const order = (rep.spec?.recipes ?? []).map((x) => x.toLowerCase());
  return Object.values(rep.recipes ?? {}).sort((a, b) => order.indexOf(a.recipe.toLowerCase()) - order.indexOf(b.recipe.toLowerCase()));
}

export function VLMReportView({ rep, job, jobId }: { rep: VLMJobReport; job?: Record<string, unknown>; jobId?: string }) {
  const rs = orderedVlm(rep);
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "overview";
  const setTab = (t: string) => setParams(t === "overview" ? {} : { tab: t }, { replace: true });
  const b = rep.baseline;
  const cur = rs.find((r) => r.recipe === tab);
  const quantizeOnly = rep.spec?.quantize_only;
  return (
    <>
      <div className="tiles" data-testid="vlm-tiles">
        {b && <Tile label="原模型答案困惑度" value={fmt(b.answer_ppl, 4)} hint={`${b.n_samples} 题 · ${b.n_answer_tokens} 个答案 token`} />}
        {b && <Tile label="原模型答对率" value={pct(b.gt_accuracy, 0)} hint={b.gt_accuracy == null ? "该数据源没有标准答案" : `${b.gt_samples} 道有标准答案的题`} />}
        <Tile label="数据" value={`${rep.data?.calib.n ?? "–"} 校准 / ${rep.data?.eval.n ?? "–"} 评测`} hint={rep.data?.eval.source.split("/").pop()} />
        <Tile label="模型" value={`${rep.model?.n_linears ?? "–"} 个线性层`} hint={`${rep.model?.family ?? ""} · ${rep.spec?.device ?? ""}`} />
      </div>
      {quantizeOnly && <p className="note">仅量化不评测：没有未量化基线，只给出逐层量化误差。</p>}
      {rs.length > 0 && (
        <div className="tabs">
          <div className={`tab ${tab === "overview" ? "on" : ""}`} onClick={() => setTab("overview")}>
            总览
          </div>
          {rs.map((r) => (
            <div key={r.recipe} className={`tab ${tab === r.recipe ? "on" : ""}`} onClick={() => setTab(r.recipe)}>
              {r.recipe} 误差分析
            </div>
          ))}
        </div>
      )}
      {tab === "overview" && rs.length > 0 && <Overview rep={rep} rs={rs} />}
      {cur && jobId && (
        <p className="small muted" data-testid="open-standalone-links">
          在独立页面打开：{" "}
          <Link to={`/analysis/${jobId}?recipe=${cur.recipe}`} data-testid="open-analysis-link">
            误差分析
          </Link>{" "}
          ·{" "}
          <Link to={`/suggest/${jobId}?recipe=${cur.recipe}`} data-testid="open-suggest-link">
            优化建议
          </Link>
        </p>
      )}
      {cur && <RecipeView r={cur} rep={rep} job={job} />}
    </>
  );
}

function Overview({ rep, rs }: { rep: VLMJobReport; rs: VLMRecipeReport[] }) {
  const hasEval = rs.some((r) => r.summary.kl != null);
  return (
    <>
      {hasEval && (
        <div className="card">
          <h2>量化前后答案的差异（越低越好）</h2>
          <Chart
            option={() => {
              const o = baseOption();
              return {
                ...o,
                grid: { ...(o.grid as object), top: 40 },
                tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => v?.toExponential(3) },
                xAxis: { ...(o.xAxis as object), type: "category", data: rs.map((r) => `${r.recipe}\n${r.algorithm}`) },
                yAxis: { ...(o.yAxis as object), type: "log", name: "KL(原 ‖ 量化)", nameTextStyle: { color: token("--text-muted") } },
                series: [
                  {
                    type: "bar",
                    barMaxWidth: 36,
                    itemStyle: { color: series(0), borderRadius: [4, 4, 0, 0] },
                    label: { show: true, position: "top", color: token("--text-secondary"), fontSize: 11, formatter: (p: { value: number }) => p.value?.toExponential(2) },
                    data: rs.map((r) => r.summary.kl),
                  },
                ],
              };
            }}
            height={260}
          />
        </div>
      )}
      <div className="card">
        <h2>配方对比</h2>
        <table data-testid="vlm-summary-table">
          <thead>
            <tr>
              <th>配方</th>
              <th>算法</th>
              <th>KL</th>
              <th>Top-1 一致</th>
              <th>答案 PPL Δ</th>
              <th>答案完全一致</th>
              <th>答案前缀一致</th>
              <th>答对率</th>
              <th>主要误差来源</th>
            </tr>
          </thead>
          <tbody>
            {rs.map((r) => (
              <tr key={r.recipe}>
                <td>{r.recipe}</td>
                <td>{r.algorithm}</td>
                <td className="num">{r.summary.kl == null ? "–" : r.summary.kl.toExponential(2)}</td>
                <td className="num">{pct(r.summary.top1_agreement)}</td>
                <td className="num">{signed(r.summary.answer_ppl_delta_pct)}</td>
                <td className="num">{pct(r.summary.generation_exact_match)}</td>
                <td className="num">{pct(r.summary.generation_prefix_match)}</td>
                <td className="num">
                  {pct(r.summary.gt_accuracy, 0)}
                  {r.summary.gt_accuracy_delta ? <span className="muted"> ({r.summary.gt_accuracy_delta > 0 ? "+" : ""}{(r.summary.gt_accuracy_delta * 100).toFixed(0)}pp)</span> : null}
                </td>
                <td>{r.summary.worst_stage ? MODALITY[r.summary.worst_stage] ?? r.summary.worst_stage : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">
          KL、Top-1 一致率只在<b>答案</b>位置上比较：以未量化模型自己生成的答案为参照，让量化模型对同样的答案逐 token 打分（图像与问题只是条件）。「答案完全一致」是量化模型自己贪心生成的答案与参照逐 token 相同的比例。
          {rep.data?.eval.has_gt && "答对率用合成场景的标准答案判定（回答中含正确的词，数字可写作英文单词）。"}
        </p>
      </div>
    </>
  );
}

function RecipeView({ r, rep, job }: { r: VLMRecipeReport; rep: VLMJobReport; job?: Record<string, unknown> }) {
  const [showAll, setShowAll] = useState(false);
  const ex = r.eval?.examples ?? [];
  const base = rep.baseline?.examples ?? [];
  return (
    <>
      <VLMSuggestion r={r} job={job} />
      {r.attribution && r.attribution.length > 0 && (
        <div className="card">
          <h2>模态归因：只量化其中一部分时的 KL</h2>
          <table data-testid="vlm-attribution">
            <thead>
              <tr>
                <th>量化范围</th>
                <th>线性层数</th>
                <th>KL</th>
                <th>Top-1 一致</th>
                <th>答案 PPL</th>
              </tr>
            </thead>
            <tbody>
              {r.attribution.map((a) => (
                <tr key={a.modality} className={a.modality === r.summary.worst_stage ? "worst" : ""}>
                  <td>{MODALITY[a.modality] ?? a.modality}</td>
                  <td className="num">{a.units}</td>
                  <td className="num">{a.kl.toExponential(2)}</td>
                  <td className="num">{pct(a.top1_agreement)}</td>
                  <td className="num">{fmt(a.answer_ppl, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {ex.length > 0 && (
        <div className="card">
          <h2>量化后的回答（对照原模型）</h2>
          <table data-testid="vlm-examples">
            <thead>
              <tr>
                <th>问题</th>
                <th>标准答案</th>
                <th>原模型</th>
                <th>量化后</th>
              </tr>
            </thead>
            <tbody>
              {ex.slice(0, showAll ? ex.length : 5).map((e, i) => (
                <tr key={i} className={base[i] && base[i].answer !== e.answer ? "worst" : ""}>
                  <td>{e.question}</td>
                  <td>{e.truth ?? "–"}</td>
                  <td>{base[i]?.answer ?? "–"}</td>
                  <td>{e.answer}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {ex.length > 5 && (
            <button className="secondary" onClick={() => setShowAll(!showAll)}>
              {showAll ? "收起" : `显示全部 ${ex.length} 条`}
            </button>
          )}
        </div>
      )}
      {r.layers.length > 0 && <LayerAnalysis layers={r.layers} strip="model." />}
    </>
  );
}

/** T-R10 动态量化：按逐层误差给出"保持原精度"的层清单，一键带着这份清单重新跑。 */
export function VLMSuggestion({ r, job }: { r: VLMRecipeReport; job?: Record<string, unknown> }) {
  const nav = useNavigate();
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const sug = r.summary.suggestion;
  if (!sug) return <p className="note" data-testid="vlm-suggest-none">这份报告是旧版本生成的，没有逐层跳过建议；重新运行一次即可。</p>;
  const clone = async () => {
    if (!job) return;
    setErr(null);
    setBusy(true);
    try {
      const j = JSON.parse(JSON.stringify(job)) as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
      delete j.output_dir;
      delete j.apiVersion;
      delete j.kind;
      j.metadata = { ...(j.metadata ?? {}), name: `${j.metadata?.name ?? "job"}-dq` };
      j.quant = { ...j.quant, recipe: [r.recipe], mixed_precision: { rules: sug.rules } };
      const { job_id } = await api.submit(j);
      nav(`/jobs/${job_id}`);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="card" data-testid="vlm-suggest">
      <h2>动态量化建议：哪些层保持原精度</h2>
      {sug.layers.length === 0 ? (
        <p className="small" data-testid="vlm-suggest-empty">没有层的输出误差明显偏离同组件的其它层，不需要跳过任何层。</p>
      ) : (
        <>
          <table data-testid="vlm-suggest-table">
            <thead>
              <tr>
                <th>层</th>
                <th>组件</th>
                <th>输出 SQNR (dB)</th>
                <th>该组件中位数 (dB)</th>
              </tr>
            </thead>
            <tbody>
              {sug.layers.map((l) => (
                <tr key={l.name}>
                  <td className="mono">{l.name}</td>
                  <td>{l.component}</td>
                  <td className="num">{l.out_sqnr_db.toFixed(1)}</td>
                  <td className="num">{l.median_db.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small">
            共 {sug.layers.length} 层，保持原精度的额外权重约 <b>{fmtBytes(sug.extra_weight_bytes)}</b>。依据：{sug.basis}。
          </p>
        </>
      )}
      <p className="note">只看量化误差（不需要评测），所以这是线索而不是保证；点按钮按这份清单重新跑一次，在结果页对比 KL 才是证据。</p>
      {err && <div className="error-box">{err}</div>}
      <button disabled={!job || busy || sug.rules.length === 0} onClick={clone} data-testid="vlm-suggest-clone">
        {busy ? "提交中…" : `按建议生成新作业（${r.recipe} + 跳过 ${sug.layers.length} 层）`}
      </button>
    </div>
  );
}
