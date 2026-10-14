import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, fmt, fmtBytes, LlamaCppDeploy, LlamaServe, llamaSpeed, LLMJobReport, LLMRecipeReport, Sensitivity, LmEval } from "../api";
import { baseOption, Chart, series, token } from "../charts";
import { Tile } from "./Estimate";
import { LayerSunburst } from "../components/LayerSunburst";
import { RunCompleteness, VerdictStrip } from "../components/VerdictStrip";
import { WindowReview } from "../components/WindowReview";
import { LayerAnalysis } from "./LayerAnalysis";
import { SweepCard } from "./SweepCard";

export const LLM_STAGES: [string, string][] = [
  ["validate", "校验"],
  ["inspect", "模型加载"],
  ["calib_prepare", "数据准备"],
  ["eval_baseline", "原模型 PPL"],
  ["quantize", "逐块量化"],
  ["eval_fake_wo_kv", "PPL（W+A）"],
  ["sensitivity", "敏感度分析"],
  ["eval_fake", "PPL（+KV）"],
  ["export", "导出"],
  ["real_check", "真实/伪量化一致性"],
  ["eval_real", "真实权重 PPL"],
  ["deploy_llamacpp", "llama.cpp 部署"],
  ["report", "报告"],
];

const pct = (v: number | null | undefined) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`);
const wdesc = (r: LLMRecipeReport) =>
  `W ${r.weight.format}${r.weight.bits ?? ""}${r.weight.group_size ? " g" + r.weight.group_size : ""} · A ${r.act.format === "none" ? "FP" : `${r.act.format}${r.act.bits ?? ""}`}`;

export function ordered(rep: LLMJobReport): LLMRecipeReport[] {
  // A job that failed before writing its full report (or a stand-in worker in a test) may have no spec / recipes at
  // all; fall back to whatever `recipes` order the report itself has rather than crashing the page.
  const order = (rep.spec?.recipes ?? []).map((x) => x.toLowerCase());
  return Object.values(rep.recipes ?? {}).sort((a, b) => order.indexOf(a.recipe.toLowerCase()) - order.indexOf(b.recipe.toLowerCase()));
}

/** "wikitext-2 train", or for a user-named source: its path and how much of it was read. */
function calibHint(rep: LLMJobReport): string | undefined {
  const c = rep.data?.calib;
  if (!c) return undefined;
  if (c.tokens == null) return c.dataset;
  return `${c.dataset.split("/").filter(Boolean).slice(-2).join("/")} · ${c.files ?? "?"} 个文件 · ${c.tokens} token${c.truncated ? "（已截断）" : ""}`;
}

export function LLMReportView({ rep, job, jobId }: { rep: LLMJobReport; job?: Record<string, unknown>; jobId?: string }) {
  const rs = ordered(rep);
  const [params, setParams] = useSearchParams();
  const want = params.get("tab") ?? "overview";
  const tab = rs.some((r) => r.recipe === want) ? want : "overview"; // a stale ?tab= must not leave the page blank
  const setTab = (t: string) => setParams(t === "overview" ? {} : { tab: t }, { replace: true });
  const b = rep.baseline;
  const cur = rs.find((r) => r.recipe === tab);
  return (
    <>
      {b && (
        <div className="tiles">
          <Tile label="原模型 PPL（pretrain）" value={fmt(b.ppl, 3)} hint={`${rep.data?.eval.dataset ?? "wikitext-2 test"}`} />
          <Tile label="评估 token" value={`${b.tokens}`} hint={`${b.windows} 窗口 × ${b.seq_len}`} />
          <Tile label="校准" value={`${rep.data?.calib.n ?? "–"} × ${rep.data?.calib.seq_len ?? "–"}`} hint={calibHint(rep)} />
          <Tile label="模型" value={`${rep.model?.blocks ?? "–"} 块`} hint={`${rep.model?.linears ?? "–"} 个线性层 · ${rep.spec.device}`} />
        </div>
      )}
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
      {cur && <RecipeView r={cur} base={b?.ppl} job={job} />}
    </>
  );
}

function Overview({ rep, rs }: { rep: LLMJobReport; rs: LLMRecipeReport[] }) {
  const base = rep.baseline?.ppl ?? 0;
  const hasKv = rs.some((r) => r.eval.fake_quant);
  return (
    <>
      <RunCompleteness rep={rep} />
      <VerdictStrip base={base} rs={rs} />
      <div className="card">
        <h2>困惑度（越低越好）· 三个评估位点</h2>
        <Chart
          option={() => {
            const o = baseOption();
            const mk = (name: string, i: number, data: (number | null)[]) => ({
              name,
              type: "bar",
              barMaxWidth: 28,
              itemStyle: { color: series(i), borderRadius: [4, 4, 0, 0] },
              label: { show: true, position: "top", color: token("--text-secondary"), fontSize: 11, formatter: (p: { value: number }) => p.value?.toFixed(2) },
              data,
            });
            return {
              ...o,
              legend: { ...(o.legend as object), data: ["fake_quant_wo_kv（W+A）", ...(hasKv ? ["fake_quant（W+A+KV）"] : [])] },
              grid: { ...(o.grid as object), top: 56 },
              tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: number) => v?.toFixed(3) },
              xAxis: { ...(o.xAxis as object), type: "category", data: rs.map((r) => `${r.recipe}\n${r.algorithm}`) },
              yAxis: { ...(o.yAxis as object), type: "value", name: "PPL", scale: false, nameTextStyle: { color: token("--text-muted") } },
              series: [
                {
                  ...mk("fake_quant_wo_kv（W+A）", 0, rs.map((r) => r.eval.fake_quant_wo_kv.ppl)),
                  markLine: {
                    silent: true,
                    symbol: "none",
                    lineStyle: { color: token("--text-secondary"), type: "dashed", width: 1 },
                    label: { formatter: `pretrain ${base.toFixed(2)}`, color: token("--text-secondary"), position: "insideEndTop", fontSize: 11 },
                    data: [{ yAxis: base }],
                  },
                },
                ...(hasKv ? [mk("fake_quant（W+A+KV）", 1, rs.map((r) => r.eval.fake_quant?.ppl ?? null))] : []),
              ],
            };
          }}
        />
        <p className="note">虚线为原模型（pretrain）PPL。fake_quant_wo_kv 只量化权重与激活；fake_quant 在此基础上再量化 KV Cache，两者之差即 KV 量化带来的损失。</p>
      </div>
      <div className="card table-wrap">
        <h2>配方汇总</h2>
        <table>
          <thead>
            <tr>
              <th>配方</th>
              <th>算法</th>
              <th>量化规格</th>
              <th className="num">PPL（W+A）</th>
              <th className="num">PPL（+KV）</th>
              <th className="num">相对原模型</th>
              <th className="num">KV 增量</th>
              <th>KV</th>
              <th className="num">量化层 / 跳过</th>
              <th className="num">量化耗时</th>
              <th>最差层</th>
            </tr>
          </thead>
          <tbody>
            {rs.map((r) => (
              <tr key={r.recipe}>
                <td>
                  <b>{r.recipe}</b>
                </td>
                <td>
                  <span className="badge accent">{r.algorithm}</span>
                </td>
                <td className="small">{wdesc(r)}</td>
                <td className="num">{fmt(r.eval.fake_quant_wo_kv.ppl, 3)}</td>
                <td className="num">{fmt(r.eval.fake_quant?.ppl, 3)}</td>
                <td className="num">{pct(r.summary.ppl_delta_pct)}</td>
                <td className="num">{pct(r.summary.kv_delta_pct)}</td>
                <td className="small">{r.kv ? `${r.kv.format}${r.kv.format === "int" ? r.kv.bits : ""} · ${r.kv.method}` : "FP"}</td>
                <td className="num">
                  {r.layers.length} / {r.skipped.length}
                </td>
                <td className="num">{fmt(r.quantize_seconds, 1)} s</td>
                <td className="mono small">{r.summary.worst_layers[0]?.name.replace("model.layers.", "L") ?? "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">
          逐块顺序量化：第 i 块用已量化的 0..i-1 块的输出做校准（GPTQModel / LLMC 约定）。fake-quant 以浮点存储反量化权重，数值与真实低比特内核一致但不降低显存。
        </p>
      </div>
      {rep.baseline?.window_nll ? (
        <WindowReview
          base={rep.baseline.window_nll}
          recipes={rs.flatMap((r) => (r.eval.fake_quant_wo_kv.window_nll ? [{ name: r.recipe, nll: r.eval.fake_quant_wo_kv.window_nll }] : []))}
          previews={rep.data?.eval.previews}
          seqLen={rep.baseline.seq_len}
        />
      ) : (
        <p className="note" data-testid="window-review-missing">这份报告没有逐窗口的损失（它是在逐窗口复盘出现之前跑的）；重跑即可得到。</p>
      )}
      {rs.some((r) => r.export) && <ExportCard rs={rs} />}
      {rs.some((r) => r.logits) && <LogitsCard rep={rep} rs={rs} />}
      {rs.some((r) => Object.values(r.export ?? {}).some((e) => e.vllm)) && <ServeCard rs={rs} which="vllm" />}
      {rs.some((r) => Object.values(r.export ?? {}).some((e) => e.sglang)) && <ServeCard rs={rs} which="sglang" />}
      {rs.flatMap((r) => Object.entries(r.export ?? {}).flatMap(([f, e]) => (["vllm", "sglang"] as const).flatMap((k) => (e[k]?.sweep ? [<SweepCard key={`${r.recipe}-${f}-${k}`} title={`${r.recipe} · ${f} · ${k === "vllm" ? "vLLM" : "SGLang"}`} sweep={e[k]!.sweep!} />] : []))))}
      {(rep.baseline_lm_eval || rs.some((r) => Object.values(r.export ?? {}).some((e) => e.vllm?.lm_eval))) && <LmEvalCard rep={rep} rs={rs} />}
      {rep.llamacpp && <LlamaCppCard d={rep.llamacpp} />}
    </>
  );
}

/** "Metal 全部卸载" / "CPU 32 线程" etc. from the run's -ngl setting and llama-bench backend string. */
function deviceLabel(d: LlamaCppDeploy): string {
  const gpu = (d.backend ?? "").split(",").find((b) => b && b !== "CPU" && b !== "BLAS");
  if (d.gpu_layers && d.gpu_layers > 0 && gpu) {
    const name = gpu === "MTL" ? "Metal" : gpu;
    return d.gpu_layers >= 99 ? `${name} 全部层卸载` : `${name} 卸载 ${d.gpu_layers} 层`;
  }
  return `CPU ${d.threads} 线程`;
}

function ServeCells({ s }: { s?: LlamaServe }) {
  if (!s || s.status === "skipped") {
    return (
      <td className="muted small" colSpan={3} title={s?.reason}>
        未运行
      </td>
    );
  }
  if (s.status === "failed" || !s.load) {
    return (
      <td className="small" colSpan={3} title={s.error}>
        启动失败{s.diagnosis ? `：${s.diagnosis}` : ""}
      </td>
    );
  }
  const l = s.load;
  return (
    <>
      <td className="num" title={`${l.ok}/${l.requests} 请求成功`}>
        {fmt(l.output_tokens_per_s, 0)}
        {l.ok < l.requests && <span className="muted small"> ({l.ok}/{l.requests})</span>}
      </td>
      <td className="num">{l.ttft_ms?.p50 !== undefined ? `${fmt(l.ttft_ms.p50, 0)} ms` : "–"}</td>
      <td className="num">{l.tpot_ms?.p50 !== undefined ? `${fmt(l.tpot_ms.p50, 1)} ms` : "–"}</td>
    </>
  );
}

function LlamaCppCard({ d }: { d: LlamaCppDeploy }) {
  const rows = Object.values(d.results);
  const f16 = d.results.f16;
  const hasServe = rows.some((r) => r.serve);
  const conc = rows.find((r) => r.serve?.load)?.serve?.load?.concurrency;
  return (
    <div className="card">
      <h2>llama.cpp 部署实测（{deviceLabel(d)}，GGUF）</h2>
      <div className="grid-2">
        <Chart
          height={260}
          option={() => {
            const o = baseOption();
            return {
              ...o,
              legend: { ...(o.legend as object), data: ["解码 tok/s", "PPL"] },
              grid: { ...(o.grid as object), top: 48 },
              tooltip: { ...(o.tooltip as object), trigger: "axis", axisPointer: { type: "shadow" } },
              xAxis: { ...(o.xAxis as object), type: "category", data: rows.map((r) => r.type.toUpperCase()) },
              yAxis: [
                { ...(o.yAxis as object), type: "value", name: "tok/s", nameTextStyle: { color: token("--text-muted") } },
                { ...(o.yAxis as object), type: "value", name: "PPL", scale: true, splitLine: { show: false }, nameTextStyle: { color: token("--text-muted") } },
              ],
              series: [
                { name: "解码 tok/s", type: "bar", barMaxWidth: 28, itemStyle: { color: series(0), borderRadius: [4, 4, 0, 0] }, data: rows.map((r) => llamaSpeed(r).tg ?? null) },
                { name: "PPL", type: "line", yAxisIndex: 1, symbolSize: 8, lineStyle: { color: series(1), width: 2 }, itemStyle: { color: series(1) }, data: rows.map((r) => r.ppl) },
              ],
            };
          }}
        />
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>类型</th>
                <th className="num">文件</th>
                <th className="num">PPL</th>
                <th className="num">相对 F16</th>
                <th className="num">预填充</th>
                <th className="num">解码</th>
                {hasServe && <th className="num">服务 tok/s</th>}
                {hasServe && <th className="num">TTFT p50</th>}
                {hasServe && <th className="num">TPOT p50</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.type}>
                  <td>
                    <b>{r.type.toUpperCase()}</b>
                    {r.fallback && r.fallback.length > 0 && <span className="muted small"> · {r.fallback.length} 层回退</span>}
                  </td>
                  <td className="num">{fmtBytes(r.bytes)}</td>
                  <td className="num">{fmt(r.ppl, 3)}</td>
                  <td className="num">{f16 && r !== f16 ? pct(100 * (r.ppl / f16.ppl - 1)) : "–"}</td>
                  <td className="num">{fmt(llamaSpeed(r).pp, 0)}</td>
                  <td className="num">{fmt(llamaSpeed(r).tg, 1)}</td>
                  {hasServe && <ServeCells s={r.serve} />}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="note">
        {d.protocol}（与上方 PPL 协议不同，只在本表内比较）。GGUF 用 llama.cpp 自己的块量化（与 ggml 参考实现逐字节一致）量化原始权重，与配方/算法无关；行长不是 256 的倍数的层按 llama-quantize 的规则回退（Q4_K→Q5_0，Q2_K→Q4_0，仍不整除则 F16）。
        {hasServe && ` 服务列为 llama-server 闭环流式压测（${conc ?? "?"} 并发）的输出吞吐与 TTFT / TPOT 中位数。`}
      </p>
    </div>
  );
}

function ExportCard({ rs }: { rs: LLMRecipeReport[] }) {
  const rows = rs.flatMap((r) => Object.entries(r.export ?? {}).map(([f, e]) => ({ r, f, e })));
  return (
    <div className="card table-wrap">
      <h2>真实低比特导出（S09）与伪量化/真实一致性（S10）</h2>
      <table>
        <thead>
          <tr>
            <th>配方</th>
            <th>格式</th>
            <th>状态</th>
            <th className="num">大小</th>
            <th>整数码一致</th>
            <th className="num">权重 rel-L2（fp16 scale）</th>
            <th className="num">真实权重 PPL</th>
            <th className="num">相对伪量化</th>
            <th>目录 / 原因</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ r, f, e }) => (
            <tr key={r.recipe + f}>
              <td>
                <b>{r.recipe}</b>
              </td>
              <td>
                <span className="badge accent">{f}</span>
              </td>
              <td>{e.status === "exported" ? "已导出" : "不支持"}</td>
              <td className="num">{e.bytes ? fmtBytes(e.bytes) : "–"}</td>
              <td>{e.real_check ? (e.real_check.codes_identical ? "✓ 全部一致" : `✗ ${e.real_check.code_mismatch} 处不一致`) : "–"}</td>
              <td className="num">{e.real_check ? e.real_check.w_rel_l2_mean.toExponential(2) : "–"}</td>
              <td className="num">{fmt(e.real_ppl, 3)}</td>
              <td className="num">{pct(e.real_vs_fake_ppl_pct)}</td>
              <td className="mono small">{e.status === "exported" ? e.out_dir?.split("/").slice(-2).join("/") : `${e.problems?.[0]?.reason ?? ""}（${e.n_problems} 层）`}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        检查方式：用参考解包器读回 checkpoint，逐层比较整数码与零点（必须完全一致），再用 fp16 scale 反量化出后端实际使用的权重重新测 PPL。GPTQ/AWQ 格式不含激活量化。AWQ/SmoothQuant 的平滑系数按输入分组共享，并折叠进前一层（norm 或 up_proj/v_proj），因此可以直接导出；GQA 模型的 o_proj 无法折叠，保持不做平滑。
      </p>
    </div>
  );
}

/** KL / Top-1 agreement against the unquantized model. Top-k KL is a lower bound, and the page says so. */
function LogitsCard({ rep, rs }: { rep: LLMJobReport; rs: LLMRecipeReport[] }) {
  const probe = rep.logits_probe;
  return (
    <div className="card table-wrap" data-testid="logits-card">
      <h2>输出分布偏移（logits 探针）</h2>
      <table>
        <thead>
          <tr>
            <th>配方</th>
            <th className="num">KL（nats）</th>
            <th className="num">Top-1 一致率</th>
            <th className="num">原 Top-1 概率下降</th>
            <th className="num">位置数</th>
          </tr>
        </thead>
        <tbody>
          {rs
            .filter((r) => r.logits)
            .map((r) => (
              <tr key={r.recipe} data-testid={`logits-${r.recipe}`}>
                <td>{r.recipe}</td>
                <td className="num">{fmt(r.logits!.kl, 4)}</td>
                <td className="num">{(100 * r.logits!.top1_agreement).toFixed(1)}%</td>
                <td className="num">{fmt(r.logits!.ref_top1_prob_drop, 4)}</td>
                <td className="num">{r.logits!.positions.toLocaleString()}</td>
              </tr>
            ))}
        </tbody>
      </table>
      <p className="note">
        PPL 只看实际下一个 token 的概率；这里比较整个输出分布。参考分布在量化前从原模型捕获
        {probe ? `（${probe.windows} 个窗口 × ${probe.seq_len} token，参考占 ${probe.reference_mib.toFixed(0)} MiB）` : ""}。
        {probe && !probe.exact && ` 只保存 top-${probe.k} logits，所以 KL 是下界，实测约偏低 11%（Qwen2.5-0.5B）；Top-1 一致率不受影响。`}
      </p>
    </div>
  );
}

/** lm-eval accuracy rows: the unquantized baseline first, then each exported checkpoint served by vLLM. */
function LmEvalCard({ rep, rs }: { rep: LLMJobReport; rs: LLMRecipeReport[] }) {
  const sources: { who: string; fmt: string; lm: LmEval }[] = [];
  if (rep.baseline_lm_eval) sources.push({ who: "未量化（基线）", fmt: "–", lm: rep.baseline_lm_eval });
  for (const r of rs) for (const [f, e] of Object.entries(r.export ?? {})) if (e.vllm?.lm_eval) sources.push({ who: r.recipe, fmt: f, lm: e.vllm.lm_eval });
  return (
    <div className="card table-wrap" data-testid="lmeval-card">
      <h2>任务精度（lm-eval，部署后的服务）</h2>
      <table>
        <thead>
          <tr>
            <th>配方</th>
            <th>格式</th>
            <th>任务</th>
            <th>指标</th>
            <th className="num">acc ± se</th>
            <th className="num">样本数</th>
            <th>说明</th>
          </tr>
        </thead>
        <tbody>
          {sources.flatMap(({ who, fmt: f, lm }) =>
            lm.status !== "succeeded" || !lm.tasks
              ? [
                  <tr key={who + f}>
                    <td>{who}</td>
                    <td>{f}</td>
                    <td>–</td>
                    <td className="bad">{lm.status === "skipped" ? "已跳过" : "失败"}</td>
                    <td className="num">–</td>
                    <td className="num">–</td>
                    <td className="small">{lm.reason ?? (lm.error ?? "").split("\n")[0]}</td>
                  </tr>,
                ]
              : Object.entries(lm.tasks).flatMap(([task, t]) =>
                  Object.entries(t.metrics).map(([k, m]) => (
                    <tr key={who + f + task + k} data-testid={`lmeval-${who}-${task}-${k}`}>
                      <td>{who}</td>
                      <td>{f}</td>
                      <td>{t.alias ?? task}</td>
                      <td>
                        {m.metric}（{m.filter}）
                      </td>
                      <td className="num">
                        {m.value.toFixed(4)}
                        {m.stderr != null ? ` ± ${m.stderr.toFixed(4)}` : ""}
                      </td>
                      <td className="num">{t.samples?.effective ?? "–"}</td>
                      <td className="small">
                        lm-eval {lm.version ?? ""} · limit {lm.limit ?? "–"} · {lm.num_fewshot == null ? "任务默认" : `${lm.num_fewshot}-shot`}
                      </td>
                    </tr>
                  )),
                ),
          )}
        </tbody>
      </table>
      <p className="note">样本数少时标准误很大，两个配方的差异在标准误之内就不能说谁更好。</p>
    </div>
  );
}

const SERVE_INFO = {
  vllm: { title: "vLLM", cmd: "vllm serve", logprobs: "vLLM 返回的 prompt 对数概率" },
  sglang: { title: "SGLang", cmd: "python -m sglang.launch_server", logprobs: "SGLang /generate 返回的 input_token_logprobs" },
} as const;

function ServeCard({ rs, which }: { rs: LLMRecipeReport[]; which: "vllm" | "sglang" }) {
  const info = SERVE_INFO[which];
  const rows = rs.flatMap((r) => Object.entries(r.export ?? {}).flatMap(([f, e]) => (e[which] ? [{ r, f, v: e[which]! }] : [])));
  const conc = rows.find((x) => x.v.load)?.v.load?.spec.concurrency;
  return (
    <div className="card table-wrap">
      <h2>{info.title} 部署（S11）与部署后精度（S12）</h2>
      <table>
        <thead>
          <tr>
            <th>配方</th>
            <th>格式</th>
            <th>状态</th>
            <th>量化内核</th>
            <th className="num">启动</th>
            <th className="num">权重显存</th>
            <th className="num">KV 容量</th>
            <th className="num">部署后 PPL</th>
            <th className="num">相对真实权重</th>
            <th className="num">服务 tok/s</th>
            <th className="num">TTFT p50</th>
            <th className="num">TPOT p50</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ r, f, v }) => (
            <tr key={r.recipe + f}>
              <td>
                <b>{r.recipe}</b>
              </td>
              <td>
                <span className="badge accent">{f}</span>
              </td>
              {v.status === "skipped" || v.status === "failed" ? (
                <td colSpan={10} title={v.error ?? v.reason}>
                  {v.status === "skipped" ? `未运行：${v.reason ?? ""}` : `失败：${v.diagnosis ?? v.error?.split("\n")[0] ?? ""}`}
                </td>
              ) : (
                <>
                  <td>{v.status === "succeeded" ? "成功" : "部分成功"}</td>
                  <td className="mono small">{v.engine?.quant_kernel ?? "–"}</td>
                  <td className="num">{v.startup_s == null ? "–" : `${fmt(v.startup_s, 0)} s`}</td>
                  <td className="num">{v.engine?.weights_gib == null ? "–" : `${fmt(v.engine.weights_gib, 2)} GiB`}</td>
                  <td className="num" title={v.engine?.max_concurrency_len ? `${v.engine.max_concurrency_len} token 请求约 ${fmt(v.engine.max_concurrency_at_len, 0)} 并发` : ""}>
                    {v.engine?.kv_cache_tokens == null ? "–" : `${fmt(v.engine.kv_cache_tokens / 1000, 0)}k tok`}
                  </td>
                  <td className="num">{fmt(v.served_ppl, 3)}</td>
                  <td className="num">{pct(v.served_vs_real_ppl_pct)}</td>
                  <td className="num" title={v.load ? `${v.load.ok}/${v.load.spec.requests} 请求成功` : ""}>
                    {fmt(v.load?.output_tokens_per_s, 0)}
                  </td>
                  <td className="num">{v.load?.ttft_ms?.p50 == null ? "–" : `${fmt(v.load.ttft_ms.p50, 0)} ms`}</td>
                  <td className="num">{v.load?.tpot_ms?.p50 == null ? "–" : `${fmt(v.load.tpot_ms.p50, 1)} ms`}</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        每个导出的 checkpoint 由 <span className="mono">{info.cmd}</span> 直接加载。部署后 PPL 与上方的伪量化 / 真实权重 PPL 使用同一评测窗口，
        取自 {info.logprobs}，差异来自后端内核与精度。压测为闭环流式请求{conc ? `（${conc} 并发）` : ""}，TTFT / TPOT 取中位数。
      </p>
    </div>
  );
}

// T-1306: split so /analysis and /suggest (standalone pages, promoted out of the job detail tabs) can
// render just one half each; RecipeView below still composes both for the embedded per-recipe tab.
export function RecipeErrorAnalysis({ r, base }: { r: LLMRecipeReport; base?: number }) {
  return (
    <>
      <div className="tiles">
        <Tile label="PPL（W+A）" value={fmt(r.eval.fake_quant_wo_kv.ppl, 3)} hint={base ? `原模型 ${fmt(base, 3)}` : undefined} />
        <Tile label="PPL（+KV）" value={fmt(r.eval.fake_quant?.ppl, 3)} hint={r.kv ? `KV ${r.kv.format}${r.kv.format === "int" ? r.kv.bits : ""} · ${r.kv.method} · ${pct(r.summary.kv_delta_pct)}` : "KV 未量化"} />
        <Tile label="相对原模型" value={pct(r.summary.ppl_delta_pct)} hint={`算法 ${r.algorithm}`} />
        <Tile label="量化层数" value={`${r.layers.length}`} hint={`${wdesc(r)} · 跳过 ${r.skipped.length}`} />
      </div>
      <LayerSunburst layers={r.layers} />
      <LayerAnalysis layers={r.layers} strip="model." />
    </>
  );
}

export function RecipeSuggestion({ r, job }: { r: LLMRecipeReport; job?: Record<string, unknown> }) {
  if (!r.sensitivity) return <p className="note">该配方未开启敏感度分析（analysis.sensitivity），没有可生成的优化建议。</p>;
  return <SensitivityCard s={r.sensitivity} recipe={r.recipe} job={job} />;
}

function RecipeView({ r, base, job }: { r: LLMRecipeReport; base?: number; job?: Record<string, unknown> }) {
  return (
    <>
      <RecipeErrorAnalysis r={r} base={base} />
      {r.sensitivity && <RecipeSuggestion r={r} job={job} />}
    </>
  );
}

function SensitivityCard({ s, recipe, job }: { s: Sensitivity; recipe: string; job?: Record<string, unknown> }) {
  const nav = useNavigate();
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const sug = s.suggestion;
  const picked = new Set(sug.groups);
  const clone = async () => {
    if (!job) return;
    setErr(null);
    setBusy(true);
    try {
      const j = JSON.parse(JSON.stringify(job)) as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
      delete j.output_dir;
      delete j.apiVersion;
      delete j.kind;
      j.metadata = { ...(j.metadata ?? {}), name: `${j.metadata?.name ?? "job"}-mp` };
      j.quant = { ...j.quant, recipe: [recipe], mixed_precision: { rules: sug.rules } };
      const { job_id } = await api.submit(j);
      nav(`/jobs/${job_id}`);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="card">
      <h2>敏感度分析：单独把某{s.by === "block" ? "个块" : "类层"}恢复为 FP 能挽回多少 PPL（{s.metric}）</h2>
      <Chart
        height={300}
        option={() => {
          const o = baseOption();
          return {
            ...o,
            tooltip: {
              ...(o.tooltip as object),
              trigger: "axis",
              axisPointer: { type: "shadow" },
              formatter: (ps: { dataIndex: number }[]) => {
                const g = s.groups[ps[0].dataIndex];
                return `${g.group}（${g.layers} 层）<br/>恢复 FP 后 PPL ${g.metric.toFixed(3)}<br/>挽回 ${g.recovered.toFixed(3)}`;
              },
            },
            xAxis: { ...(o.xAxis as object), type: "category", data: s.groups.map((g) => g.group.replace("block:", "")), name: s.by === "block" ? "block" : "", nameLocation: "middle", nameGap: 24, nameTextStyle: { color: token("--text-muted") } },
            yAxis: { ...(o.yAxis as object), type: "value", name: "挽回 PPL", nameTextStyle: { color: token("--text-muted") } },
            series: [
              {
                type: "bar",
                barMaxWidth: 24,
                data: s.groups.map((g) => ({ value: g.recovered, itemStyle: { color: picked.has(g.group) ? series(1) : series(0), borderRadius: [3, 3, 0, 0] } })),
              },
            ],
          };
        }}
      />
      <p className="small">
        全部量化 PPL <b>{fmt(s.metric_quantized, 3)}</b> · FP {fmt(s.metric_fp, 3)} · 退化 {fmt(sug.degradation, 3)}。建议把
        <b> {sug.groups.join("、") || "—"} </b>提升到 {sug.bits} bit（高亮），预计 PPL ≈ {fmt(sug.predicted_metric, 3)}，权重增加 {fmtBytes(sug.extra_weight_bytes)}。
      </p>
      <p className="note">预计值是各组单独恢复 FP 时挽回量的简单相加（组间并不独立），实际效果以按建议重新运行的作业为准。</p>
      {err && <div className="error-box">{err}</div>}
      <button disabled={!job || busy || !sug.rules.length} onClick={clone}>
        {busy ? "提交中…" : `按建议生成新作业（${recipe} + 混合精度）`}
      </button>
    </div>
  );
}
