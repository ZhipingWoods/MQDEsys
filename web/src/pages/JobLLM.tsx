import { useSearchParams } from "react-router-dom";
import { fmt, fmtBytes, LLMJobReport, LLMRecipeReport } from "../api";
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
  ["eval_fake", "PPL（+KV）"],
  ["export", "导出"],
  ["real_check", "真实/伪量化一致性"],
  ["eval_real", "真实权重 PPL"],
  ["report", "报告"],
];

const pct = (v: number | null | undefined) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`);
const wdesc = (r: LLMRecipeReport) =>
  `W ${r.weight.format}${r.weight.bits ?? ""}${r.weight.group_size ? " g" + r.weight.group_size : ""} · A ${r.act.format === "none" ? "FP" : `${r.act.format}${r.act.bits ?? ""}`}`;

function ordered(rep: LLMJobReport): LLMRecipeReport[] {
  const order = rep.spec.recipes.map((x) => x.toLowerCase());
  return Object.values(rep.recipes ?? {}).sort((a, b) => order.indexOf(a.recipe.toLowerCase()) - order.indexOf(b.recipe.toLowerCase()));
}

export function LLMReportView({ rep }: { rep: LLMJobReport }) {
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
      {cur && <RecipeView r={cur} base={b?.ppl} />}
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
    </>
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
        检查方式：用参考解包器读回 checkpoint，逐层比较整数码与零点（必须完全一致），再用 fp16 scale 反量化出后端实际使用的权重重新测 PPL。GPTQ/AWQ 格式不含激活量化；AWQ/SmoothQuant 的激活缩放需先折叠进前一层才能导出。
      </p>
    </div>
  );
}

function RecipeView({ r, base }: { r: LLMRecipeReport; base?: number }) {
  return (
    <>
      <div className="tiles">
        <Tile label="PPL（W+A）" value={fmt(r.eval.fake_quant_wo_kv.ppl, 3)} hint={base ? `原模型 ${fmt(base, 3)}` : undefined} />
        <Tile label="PPL（+KV）" value={fmt(r.eval.fake_quant?.ppl, 3)} hint={r.kv ? `KV ${r.kv.format}${r.kv.format === "int" ? r.kv.bits : ""} · ${r.kv.method} · ${pct(r.summary.kv_delta_pct)}` : "KV 未量化"} />
        <Tile label="相对原模型" value={pct(r.summary.ppl_delta_pct)} hint={`算法 ${r.algorithm}`} />
        <Tile label="量化层数" value={`${r.layers.length}`} hint={`${wdesc(r)} · 跳过 ${r.skipped.length}`} />
      </div>
      <LayerAnalysis layers={r.layers} strip="model." />
    </>
  );
}
