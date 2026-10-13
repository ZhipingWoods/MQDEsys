// T-1306: "优化建议" (sensitivity -> mixed-precision suggestion -> clone-as-new-job, T-0904) promoted out
// of the job detail page into an independent top-level nav item. LLM only -- VLA has no sensitivity/
// suggestion feature at all, so a VLA job just explains that here instead of showing an empty page.
import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api, JobDetail, LLMJobReport, VLMJobReport } from "../api";
import { ordered, RecipeSuggestion } from "./JobLLM";
import { orderedVlm, VLMSuggestion } from "./JobVLM";

export default function SuggestPage() {
  const { jobId } = useParams();
  const [params, setParams] = useSearchParams();
  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    api.job(jobId).then((d) => alive && setDetail(d), (e: Error) => alive && setErr(e.message));
    return () => {
      alive = false;
    };
  }, [jobId]);

  if (!jobId) {
    return (
      <>
        <h1>优化建议</h1>
        <p className="subtitle">从某个 LLM / VLM 作业的详情页里，每个配方的"误差分析"旁边有一个带上下文的链接会跳到这里。先去挑一个已完成的作业。</p>
        <Link to="/jobs">前往作业与执行 →</Link>
      </>
    );
  }
  if (err) return <div className="error-box">{err}</div>;
  if (!detail) return <p className="note">加载中…</p>;
  const rep = detail.report;
  if (!rep) return <p className="note">该作业还没有报告（尚未完成，或已失败）。</p>;

  const isLLM = rep.kind === "llm" || (detail.job as { model?: { type?: string } } | undefined)?.model?.type === "llm";
  const isVLM = rep.kind === ("vlm" as string) || (detail.job as { model?: { type?: string } } | undefined)?.model?.type === "vlm";
  if (isVLM) {
    const vrep = rep as unknown as VLMJobReport;
    const vs = orderedVlm(vrep);
    const vtab = params.get("recipe") ?? vs[0]?.recipe;
    const vcur = vs.find((r) => r.recipe === vtab);
    return (
      <>
        <h1>优化建议</h1>
        <p className="subtitle">
          作业 <Link to={`/jobs/${jobId}`}>{jobId}</Link> · {vrep.spec?.name ?? ""}
        </p>
        {vs.length > 1 && (
          <div className="tabs">
            {vs.map((r) => (
              <div key={r.recipe} className={`tab ${vtab === r.recipe ? "on" : ""}`} onClick={() => setParams({ recipe: r.recipe }, { replace: true })} data-testid={`suggest-recipe-tab-${r.recipe}`}>
                {r.recipe}
              </div>
            ))}
          </div>
        )}
        {!vcur && <p className="note">该作业没有配方报告。</p>}
        {vcur && <VLMSuggestion r={vcur} job={detail.job} />}
      </>
    );
  }
  if (!isLLM) {
    return (
      <>
        <h1>优化建议</h1>
        <p className="subtitle">
          作业 <Link to={`/jobs/${jobId}`}>{jobId}</Link> 是 VLA 作业——优化建议目前只有 LLM（敏感度分析 → 混合精度）和 VLM（逐层误差 → 跳过清单）实现了，VLA 没有对应的后端能力，这里不假装有。
        </p>
      </>
    );
  }
  const llmRep = rep as unknown as LLMJobReport;
  const rs = ordered(llmRep);
  const tab = params.get("recipe") ?? rs[0]?.recipe;
  const setTab = (t: string) => setParams({ recipe: t }, { replace: true });
  const cur = rs.find((r) => r.recipe === tab);

  return (
    <>
      <h1>优化建议</h1>
      <p className="subtitle">
        作业 <Link to={`/jobs/${jobId}`}>{jobId}</Link> · {llmRep.spec?.name ?? ""}
      </p>
      {rs.length > 1 && (
        <div className="tabs">
          {rs.map((r) => (
            <div key={r.recipe} className={`tab ${tab === r.recipe ? "on" : ""}`} onClick={() => setTab(r.recipe)} data-testid={`suggest-recipe-tab-${r.recipe}`}>
              {r.recipe}
            </div>
          ))}
        </div>
      )}
      {!cur && <p className="note">该作业没有配方报告。</p>}
      {cur && <RecipeSuggestion r={cur} job={detail.job} />}
    </>
  );
}
