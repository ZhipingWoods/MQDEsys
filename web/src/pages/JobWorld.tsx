import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, fmt, fmtBytes, WorldJobReport, WorldRecipeReport } from "../api";
import { baseOption, Chart, series, token } from "../charts";
import { Tile } from "./Estimate";
import { LayerAnalysis } from "./LayerAnalysis";
import { NativeServeCard } from "../components/NativeServeCard";

export const WORLD_STAGES: [string, string][] = [
  ["validate", "校验"],
  ["inspect", "模型加载"],
  ["calib_prepare", "视频数据准备"],
  ["eval_baseline", "BF16 采样基线"],
  ["attribution", "主干/动作归因"],
  ["quantize", "逐层量化"],
  ["eval_quantized", "量化后生成质量"],
  ["deploy_native", "原生视频服务"],
  ["report", "报告"],
];

const MODALITY: Record<string, string> = {
  vision: "视频块嵌入（保持 FP）",
  world: "去噪主干",
  action: "动作编码器",
};

const num = (v: number | null | undefined, digits = 2) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(digits));

export function orderedWorld(rep: WorldJobReport): WorldRecipeReport[] {
  const order = (rep.spec?.recipes ?? []).map((x) => x.toLowerCase());
  return Object.values(rep.recipes ?? {}).sort((a, b) => order.indexOf(a.recipe.toLowerCase()) - order.indexOf(b.recipe.toLowerCase()));
}

export function WorldReportView({ rep, job, jobId }: { rep: WorldJobReport; job?: Record<string, unknown>; jobId?: string }) {
  const rs = orderedWorld(rep);
  const [params, setParams] = useSearchParams();
  const want = params.get("tab") ?? "overview";
  const tab = rs.some((r) => r.recipe === want) ? want : "overview";
  const setTab = (t: string) => setParams(t === "overview" ? {} : { tab: t }, { replace: true });
  const b = rep.baseline;
  const cur = rs.find((r) => r.recipe === tab);
  const quantizeOnly = rep.spec?.quantize_only;
  const best = rs.filter((r) => r.summary.psnr_vs_ref != null).sort((a, b) => (b.summary.psnr_vs_ref ?? 0) - (a.summary.psnr_vs_ref ?? 0))[0];
  return (
    <>
      <div className="tiles" data-testid="world-tiles">
        {b && <Tile icon="video" label="BF16 对真实未来 PSNR" value={`${num(b.psnr_vs_gt)} dB`} hint={`${b.frames} 帧 · ${b.image_size || "?"}px · FP 采样 ${fmt(b.latency_ms_per_video, 0)} ms/段`} />}
        {b && <Tile icon="activity" label="采样噪声下限" value={`${num(b.noise_floor.psnr)} dB`} hint="同一 FP 模型换随机种子：量化差异应低于此值" />}
        {best && <Tile icon="layers" label={`最佳保真 ${best.recipe}`} value={`${num(best.summary.psnr_vs_ref)} dB`} hint="量化输出 vs BF16 输出（相同种子）" />}
        <Tile icon="library" label="数据" value={`${rep.data?.calib.n ?? "–"} 校准 / ${rep.data?.eval.n ?? "–"} 评测`} hint={`${rep.data?.eval.source.split("/").pop()} · ${rep.data?.eval.frames ?? "–"} 帧${rep.kind === "ewm" ? " · 动作条件" : ""}`} />
      </div>
      {quantizeOnly && <p className="note">仅量化不评测：没有 BF16 参考视频，只给出逐层量化误差。</p>}
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
            逐层误差
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

function Overview({ rep, rs }: { rep: WorldJobReport; rs: WorldRecipeReport[] }) {
  const nf = rep.baseline?.noise_floor.psnr;
  const hasEval = rs.some((r) => r.summary.psnr_vs_ref != null);
  return (
    <>
      {hasEval && (
        <div className="card">
          <h2>量化输出相对 BF16 的保真度（PSNR，越高越好）</h2>
          <Chart
            option={() => {
              const o = baseOption();
              const vals = rs.map((r) => r.summary.psnr_vs_ref ?? 0);
              const lo = Math.max(0, Math.floor(Math.min(...vals, nf ?? Infinity) / 5) * 5 - 5);
              return {
                ...o,
                grid: { ...(o.grid as object), top: 40 },
                tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => `${v?.toFixed(2)} dB` },
                xAxis: { ...(o.xAxis as object), type: "category", data: rs.map((r) => `${r.recipe}\n${r.algorithm}`) },
                yAxis: { ...(o.yAxis as object), type: "value", min: lo, name: "PSNR (dB)", nameTextStyle: { color: token("--text-muted") } },
                series: [
                  {
                    type: "bar",
                    barMaxWidth: 42,
                    itemStyle: { color: series(0), borderRadius: [4, 4, 0, 0] },
                    label: { show: true, position: "top", color: token("--text-secondary"), fontSize: 11, formatter: (p: { value: number }) => `${p.value?.toFixed(1)}` },
                    data: rs.map((r) => r.summary.psnr_vs_ref),
                    markLine: nf
                      ? {
                          silent: true,
                          symbol: "none",
                          lineStyle: { color: token("--warning"), type: "dashed" },
                          label: { color: token("--warning"), formatter: `采样噪声下限 ${nf.toFixed(1)} dB` },
                          data: [{ yAxis: nf }],
                        }
                      : undefined,
                  },
                ],
              };
            }}
            height={280}
          />
          <p className="note">柱越高，量化模型的生成结果越接近 BF16；虚线是同一 BF16 模型换随机种子时的差异。高于虚线说明「量化造成的变化小于模型本身的采样随机性」。</p>
        </div>
      )}
      <div className="card table-wrap">
        <h2>配方对比</h2>
        <table data-testid="world-summary-table">
          <thead>
            <tr>
              <th>配方</th>
              <th>算法</th>
              <th>PSNR vs BF16</th>
              <th>MSE vs BF16</th>
              <th>PSNR vs 真实未来</th>
              <th>相对 BF16 的 PSNR 下降</th>
              <th>低于采样噪声？</th>
              <th>主要误差来源</th>
            </tr>
          </thead>
          <tbody>
            {rs.map((r) => (
              <tr key={r.recipe}>
                <td>{r.recipe}</td>
                <td>{r.algorithm}</td>
                <td className="num">{num(r.summary.psnr_vs_ref)} dB</td>
                <td className="num">{r.summary.mse_vs_ref == null ? "–" : r.summary.mse_vs_ref.toExponential(2)}</td>
                <td className="num">{num(r.summary.psnr_vs_gt)} dB</td>
                <td className="num">{r.summary.psnr_drop_db == null ? "–" : `${r.summary.psnr_drop_db >= 0 ? "+" : ""}${r.summary.psnr_drop_db.toFixed(3)} dB`}</td>
                <td>{r.summary.below_noise_floor == null ? "–" : r.summary.below_noise_floor ? "✓ 是" : "✗ 否"}</td>
                <td>{r.summary.worst_stage ? MODALITY[r.summary.worst_stage] ?? r.summary.worst_stage : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">
          「vs BF16」让两个模型用<b>相同随机种子</b>采样，差异来自量化本身；「vs 真实未来」只有数据源自带真实后续帧时才有（synthetic-video 有）。LPIPS 需要环境里额外安装 <span className="mono">lpips</span>，未安装时报告里明确写 null，而不是悄悄省略。
          {rep.kind === "ewm" && " EWM 样本逐帧带动作向量，视频主干与动作编码器分阶段归因。"}
        </p>
      </div>
      {rs.map((r) => r.deploy?.native ? <NativeServeCard key={r.recipe} recipe={r.recipe} d={r.deploy.native} /> : null)}
    </>
  );
}

function RecipeView({ r, rep, job }: { r: WorldRecipeReport; rep: WorldJobReport; job?: Record<string, unknown> }) {
  const [showAll, setShowAll] = useState(false);
  const per = r.eval?.per_sample ?? [];
  return (
    <>
      <WorldSuggestion r={r} job={job} />
      <div className="tiles">
        <Tile icon="activity" label="PSNR vs BF16" value={`${num(r.summary.psnr_vs_ref)} dB`} hint="越高越接近原模型输出" />
        <Tile icon="video" label="PSNR vs 真实未来" value={`${num(r.summary.psnr_vs_gt)} dB`} hint={`BF16 基线 ${num(rep.baseline?.psnr_vs_gt)} dB`} />
        <Tile icon="layers" label="量化后延迟" value={`${fmt(r.eval?.latency_ms_per_video, 0)} ms/段`} hint={`${fmt(r.eval?.latency_ms_per_frame, 1)} ms/帧`} />
        <Tile icon="database" label="量化后峰值内存" value={fmtBytes(r.eval?.peak_mem_bytes)} hint={r.eval?.peak_mem_note} />
      </div>
      {r.attribution && r.attribution.length > 0 && (
        <div className="card">
          <h2>分阶段归因：只量化其中一部分时的保真度</h2>
          <table data-testid="world-attribution">
            <thead>
              <tr>
                <th>量化范围</th>
                <th>线性层数</th>
                <th>PSNR vs BF16</th>
                <th>MSE vs BF16</th>
                <th>PSNR vs 真实未来</th>
              </tr>
            </thead>
            <tbody>
              {r.attribution.map((a) => (
                <tr key={a.modality} className={a.modality === r.summary.worst_stage ? "worst" : ""}>
                  <td>{MODALITY[a.modality] ?? a.modality}</td>
                  <td className="num">{a.units}</td>
                  <td className="num">{num(a.psnr_vs_ref)} dB</td>
                  <td className="num">{a.mse_vs_ref.toExponential(2)}</td>
                  <td className="num">{num(a.psnr_vs_gt)} dB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {per.length > 0 && (
        <div className="card">
          <h2>逐片段结果</h2>
          <table data-testid="world-samples">
            <thead>
              <tr>
                <th>片段</th>
                <th>PSNR vs BF16</th>
                <th>MSE vs BF16</th>
                <th>PSNR vs 真实未来</th>
              </tr>
            </thead>
            <tbody>
              {per.slice(0, showAll ? per.length : 8).map((p, i) => (
                <tr key={i}>
                  <td>#{p.seed_id ?? i}</td>
                  <td className="num">{num(p.psnr)} dB</td>
                  <td className="num">{p.mse.toExponential(2)}</td>
                  <td className="num">{num(p.psnr_vs_gt)} dB</td>
                </tr>
              ))}
            </tbody>
          </table>
          {per.length > 8 && (
            <button className="secondary" onClick={() => setShowAll(!showAll)}>
              {showAll ? "收起" : `显示全部 ${per.length} 条`}
            </button>
          )}
        </div>
      )}
      {r.layers.length > 0 && <LayerAnalysis layers={r.layers} />}
    </>
  );
}

/** Keep the FP-sensitive layers that the per-layer trace says are unusually noisy. */
function WorldSuggestion({ r, job }: { r: WorldRecipeReport; job?: Record<string, unknown> }) {
  const nav = useNavigate();
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const sug = r.summary.suggestion;
  if (!sug) return <p className="note" data-testid="world-suggest-none">这份报告是旧版本生成的，没有逐层跳过建议；重新运行一次即可。</p>;
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
    <div className="card" data-testid="world-suggest">
      <h2>动态量化建议：哪些层保持原精度</h2>
      {sug.layers.length === 0 ? (
        <p className="small" data-testid="world-suggest-empty">没有层的输出误差明显偏离同组件的其它层，不需要跳过任何层。</p>
      ) : (
        <>
          <table data-testid="world-suggest-table">
            <thead>
              <tr>
                <th>层</th>
                <th>组件</th>
                <th>输出 SQNR (dB)</th>
                <th>该组件中位数 (dB)</th>
              </tr>
            </thead>
            <tbody>
              {sug.layers.map((x) => (
                <tr key={x.name}>
                  <td className="mono small">{x.name}</td>
                  <td>{x.component}</td>
                  <td className="num">{num(x.out_sqnr_db)}</td>
                  <td className="num">{num(x.median_db)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small muted">{sug.basis}。额外保留权重约 {fmtBytes(sug.extra_weight_bytes)}。</p>
          <button disabled={busy || !job} onClick={clone} data-testid="world-suggest-clone">
            {busy ? "提交中…" : "按这份清单重新跑"}
          </button>
        </>
      )}
      {err && <div className="error-box">{err}</div>}
    </div>
  );
}
