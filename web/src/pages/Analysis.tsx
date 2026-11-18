// T-1306: "误差分析" promoted out of the job detail page's own tabs into an independent top-level nav
// item. The job detail page (Job.tsx) still has its own embedded per-recipe tabs unchanged; this page
// renders the same content (LayerAnalysis + tiles) standalone, reached either cold from the sidebar (no
// job picked yet) or with context from a job's "在独立页面打开" link (?recipe=<name> pre-selected).
import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api, JobDetail, LLMJobReport, VLMJobReport, WorldJobReport } from "../api";
import { LayerAnalysis } from "./LayerAnalysis";
import { DomainJobPicker } from "../components/DomainJobPicker";
import { DomainPendingNote } from "../domain";
import { orderedVlm } from "./JobVLM";
import { orderedWorld } from "./JobWorld";
import { ordered, RecipeErrorAnalysis } from "./JobLLM";
import { orderedRecipes, VLARecipeErrorAnalysis } from "./Job";

export default function AnalysisPage() {
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
        <h1>误差分析</h1>
        <p className="subtitle">逐层量化误差热力图、各块平均 SQNR、误差 vs 激活峰度、误差最大的 15 层。作业详情页每个配方旁也有带上下文的链接。</p>
        <DomainPendingNote />
        <DomainJobPicker base="/analysis" what="误差分析" />
      </>
    );
  }
  if (err) return <div className="error-box">{err}</div>;
  if (!detail) return <p className="note">加载中…</p>;
  const rep = detail.report;
  if (!rep) return <p className="note">该作业还没有报告（尚未完成，或已失败）。</p>;

  const isLLM = rep.kind === "llm" || (detail.job as { model?: { type?: string } } | undefined)?.model?.type === "llm";
  const isVLM = rep.kind === ("vlm" as string) || (detail.job as { model?: { type?: string } } | undefined)?.model?.type === "vlm";
  const isWorld = rep.kind === ("world" as string) || rep.kind === ("ewm" as string) || (detail.job as { model?: { type?: string } } | undefined)?.model?.type === "world" || (detail.job as { model?: { type?: string } } | undefined)?.model?.type === "ewm";
  const names = isLLM ? ordered(rep as unknown as LLMJobReport).map((r) => r.recipe) : isVLM ? orderedVlm(rep as unknown as VLMJobReport).map((r) => r.recipe) : isWorld ? orderedWorld(rep as unknown as WorldJobReport).map((r) => r.recipe) : orderedRecipes(rep);
  const tab = params.get("recipe") ?? names[0];
  const setTab = (t: string) => setParams({ recipe: t }, { replace: true });

  return (
    <>
      <h1>误差分析</h1>
      <p className="subtitle">
        作业 <Link to={`/jobs/${jobId}`}>{jobId}</Link> · {rep.spec?.name ?? ""}
      </p>
      {names.length > 1 && (
        <div className="tabs">
          {names.map((n) => (
            <div key={n} className={`tab ${tab === n ? "on" : ""}`} onClick={() => setTab(n)} data-testid={`analysis-recipe-tab-${n}`}>
              {n}
            </div>
          ))}
        </div>
      )}
      {!tab && <p className="note">该作业没有配方报告。</p>}
      {tab && isLLM && (rep as unknown as LLMJobReport).recipes?.[tab] && (
        <RecipeErrorAnalysis r={(rep as unknown as LLMJobReport).recipes![tab]} base={(rep as unknown as LLMJobReport).baseline?.ppl} />
      )}
      {tab && isVLM && (rep as unknown as VLMJobReport).recipes?.[tab] && <LayerAnalysis layers={(rep as unknown as VLMJobReport).recipes![tab].layers} strip="model." />}
      {tab && isWorld && (rep as unknown as WorldJobReport).recipes?.[tab] && <LayerAnalysis layers={(rep as unknown as WorldJobReport).recipes![tab].layers} />}
      {tab && !isLLM && !isVLM && !isWorld && rep.recipes?.[tab] && <VLARecipeErrorAnalysis r={rep.recipes[tab]} rep={rep} />}
    </>
  );
}
