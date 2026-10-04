import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, fmt, fmtBytes, LlamaCppDeploy, LlamaServe, llamaSpeed, LLMJobReport, LLMRecipeReport, Sensitivity } from "../api";
import { baseOption, Chart, series, token } from "../charts";
import { Tile } from "./Estimate";
import { LayerAnalysis } from "./LayerAnalysis";

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

function ordered(rep: LLMJobReport): LLMRecipeReport[] {
  const order = rep.spec.recipes.map((x) => x.toLowerCase());
  return Object.values(rep.recipes ?? {}).sort((a, b) => order.indexOf(a.recipe.toLowerCase()) - order.indexOf(b.recipe.toLowerCase()));
}

export function LLMReportView({ rep, job }: { rep: LLMJobReport; job?: Record<string, unknown> }) {
  const rs = ordered(rep);
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "overview";
  const setTab = (t: string) => setParams(t === "overview" ? {} : { tab: t }, { replace: true });
  const b = rep.baseline;
  const cur = rs.find((r) => r.recipe === tab);
  return (
    <>
      {b && (
        <div className="tiles">
          <Tile label="原模型 PPL（pretrain）" value={fmt(b.ppl, 3)} hint={`${rep.data?.eval.dataset ?? "wikitext-2 test"}`} />
          <Tile label="评估 token" value={`${b.tokens}`} hint={`${b.windows} 窗口 × ${b.seq_len}`} />
          <Tile label="校准" value={`${rep.data?.calib.n ?? "–"} × ${rep.data?.calib.seq_len ?? "–"}`} hint={rep.data?.calib.dataset} />
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
      {cur && <RecipeView r={cur} base={b?.ppl} job={job} />}
    </>
  );
}

function Overview({ rep, rs }: { rep: LLMJobReport; rs: LLMRecipeReport[] }) {
  const base = rep.baseline?.ppl ?? 0;
  const hasKv = rs.some((r) => r.eval.fake_quant);
  return (
    <>
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
      {rs.some((r) => r.export) && <ExportCard rs={rs} />}
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

function RecipeView({ r, base, job }: { r: LLMRecipeReport; base?: number; job?: Record<string, unknown> }) {
  return (
    <>
      <div className="tiles">
        <Tile label="PPL（W+A）" value={fmt(r.eval.fake_quant_wo_kv.ppl, 3)} hint={base ? `原模型 ${fmt(base, 3)}` : undefined} />
        <Tile label="PPL（+KV）" value={fmt(r.eval.fake_quant?.ppl, 3)} hint={r.kv ? `KV ${r.kv.format}${r.kv.format === "int" ? r.kv.bits : ""} · ${r.kv.method} · ${pct(r.summary.kv_delta_pct)}` : "KV 未量化"} />
        <Tile label="相对原模型" value={pct(r.summary.ppl_delta_pct)} hint={`算法 ${r.algorithm}`} />
        <Tile label="量化层数" value={`${r.layers.length}`} hint={`${wdesc(r)} · 跳过 ${r.skipped.length}`} />
      </div>
      {r.sensitivity && <SensitivityCard s={r.sensitivity} recipe={r.recipe} job={job} />}
      <LayerAnalysis layers={r.layers} strip="model." />
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
