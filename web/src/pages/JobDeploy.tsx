import { Link } from "react-router-dom";
import { DeployJobReport, fmt, fmtBytes, VllmDeploy } from "../api";
import { Tile } from "./Estimate";
import { SweepCard } from "./SweepCard";

export const DEPLOY_STAGES: [string, string][] = [
  ["validate", "校验产物"],
  ["eval_prepare", "评测数据"],
  ["deploy_vllm", "vLLM 部署"],
  ["deploy_sglang", "SGLang 部署"],
  ["deploy_llamacpp", "llama.cpp 部署"],
  ["report", "报告"],
];

const pct = (v: number | null | undefined) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`);

/** 部署评测作业：复用父作业导出的产物，换推理框架 / 设备 / 评测设置，不重新量化。 */
export function DeployReportView({ rep }: { rep: DeployJobReport }) {
  const a = rep.artifact;
  const d = rep.deploy ?? {};
  const ref = a?.format === "gguf" ? a?.parent_ppl : a?.parent_real_ppl;
  return (
    <>
      <div className="tiles" data-testid="deploy-tiles">
        <Tile
          label="复用的产物"
          value={a ? (a.format === "gguf" ? `GGUF ${a.type}` : `${a.recipe} · ${a.format}`) : "–"}
          hint={a ? `来自作业 ${a.job}${a.bytes ? ` · ${fmtBytes(a.bytes)}` : ""}` : ""}
        />
        <Tile label={a?.format === "gguf" ? "父作业 llama.cpp PPL" : "父作业真实权重 PPL"} value={fmt(ref, 3)} hint={a?.parent_baseline_ppl ? `原模型 ${fmt(a.parent_baseline_ppl, 3)}` : ""} />
        <Tile label="评测" value={`${rep.data?.eval.windows ?? rep.spec.eval_windows} 窗口 × ${rep.data?.eval.seq_len ?? rep.spec.eval_seq_len}`} hint={rep.data?.eval.dataset ?? "wikitext-2 test"} />
        <Tile label="设备" value={rep.env?.device_name ?? rep.spec.device} hint="本作业不重新量化" />
      </div>
      {a && a.ppl_comparable === false && (
        <p className="note" data-testid="ppl-not-comparable" style={{ borderLeft: "3px solid #e0a84a", paddingLeft: 10 }}>
          {a.parent_eval
            ? `本作业评测 ${rep.spec.eval_windows} 个窗口 × ${rep.spec.eval_seq_len}，父作业是 ${a.parent_eval.windows} × ${a.parent_eval.seq_len}：PPL 与上下文长度、所取窗口有关，不同评测设置的 PPL 不能直接比，所以没有给出「相对父作业」的差异（下面的 PPL 本身仍是准的）。要对比，请沿用父作业的序列长度和窗口数。`
            : "父作业的报告没有记录它的评测设置，无法确认 PPL 是否可比，所以没有给出「相对父作业」的差异。"}
        </p>
      )}
      {a && (
        <p className="small" data-testid="deploy-artifact">
          产物 <span className="mono">{a.path}</span>，来自作业 <Link to={`/jobs/${a.job}`}>{a.job}</Link>
          {a.algorithm ? `（${a.algorithm}）` : ""}。
        </p>
      )}
      {(d.vllm || d.sglang) && (
        <div className="card table-wrap">
          <h2>推理框架部署结果</h2>
          <table data-testid="deploy-table">
            <thead>
              <tr>
                <th>推理框架</th>
                <th>状态</th>
                <th className="num">启动</th>
                <th className="num">权重显存</th>
                <th className="num">KV 容量</th>
                <th className="num">部署后 PPL</th>
                <th className="num" title="只在评测设置与父作业相同时给出">相对父作业真实权重</th>
                <th className="num">服务 tok/s</th>
                <th className="num">TTFT p50 / p99</th>
                <th className="num">TPOT p50</th>
              </tr>
            </thead>
            <tbody>
              {(["vllm", "sglang"] as const).filter((k) => d[k]).map((k) => <Row key={k} name={k === "vllm" ? "vLLM" : "SGLang"} v={d[k]!} />)}
            </tbody>
          </table>
          <p className="note">部署后 PPL 由推理框架自己的 prompt logprobs 算出，与父作业的真实权重 PPL 用同一套 wikitext-2 窗口协议；差异来自推理框架的内核与精度。</p>
        </div>
      )}
      {(["vllm", "sglang"] as const).map((k) => d[k]?.sweep && <SweepCard key={k} title={k === "vllm" ? "vLLM" : "SGLang"} sweep={d[k]!.sweep!} />)}
      {d.llamacpp && (
        <div className="card table-wrap">
          <h2>llama.cpp 部署结果</h2>
          <table data-testid="deploy-llamacpp">
            <thead>
              <tr>
                <th>设备</th>
                <th className="num">PPL</th>
                <th className="num">相对父作业</th>
                <th className="num">pp tok/s</th>
                <th className="num">tg tok/s</th>
                <th className="num">服务 tok/s</th>
                <th className="num">TTFT p50</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{d.llamacpp.gpu_layers ? `GPU 卸载 ${d.llamacpp.gpu_layers} 层` : `CPU ${d.llamacpp.threads} 线程`}</td>
                <td className="num">{fmt(d.llamacpp.ppl, 3)}</td>
                <td className="num">{pct(d.llamacpp.ppl_vs_parent_pct)}</td>
                <td className="num">{fmt(Object.entries(d.llamacpp).find(([k]) => k.startsWith("pp") && k.endsWith("per_s"))?.[1] as number, 0)}</td>
                <td className="num">{fmt(Object.entries(d.llamacpp).find(([k]) => k.startsWith("tg") && k.endsWith("per_s"))?.[1] as number, 1)}</td>
                <td className="num">{fmt(d.llamacpp.serve?.load?.output_tokens_per_s, 0)}</td>
                <td className="num">{d.llamacpp.serve?.load?.ttft_ms?.p50 == null ? "–" : `${fmt(d.llamacpp.serve.load.ttft_ms.p50, 0)} ms`}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function Row({ name, v }: { name: string; v: VllmDeploy }) {
  if (v.status === "skipped" || v.status === "failed")
    return (
      <tr>
        <td>{name}</td>
        <td colSpan={9} title={v.error ?? v.reason}>
          {v.status === "skipped" ? `未运行：${v.reason ?? ""}` : `失败：${v.diagnosis ?? v.error?.split("\n")[0] ?? ""}`}
        </td>
      </tr>
    );
  return (
    <tr>
      <td>{name}</td>
      <td>{v.status === "succeeded" ? "成功" : "部分成功"}</td>
      <td className="num">{v.startup_s == null ? "–" : `${fmt(v.startup_s, 0)} s`}</td>
      <td className="num">{v.engine?.weights_gib == null ? "–" : `${fmt(v.engine.weights_gib, 2)} GiB`}</td>
      <td className="num">{v.engine?.kv_cache_tokens == null ? "–" : `${fmt(v.engine.kv_cache_tokens / 1000, 0)}k tok`}</td>
      <td className="num">{fmt(v.served_ppl, 3)}</td>
      <td className="num">{pct(v.served_vs_real_ppl_pct)}</td>
      <td className="num">{fmt(v.load?.output_tokens_per_s, 0)}</td>
      <td className="num">
        {v.load?.ttft_ms?.p50 == null ? "–" : `${fmt(v.load.ttft_ms.p50, 0)} / ${fmt(v.load.ttft_ms.p99, 0)} ms`}
      </td>
      <td className="num">{v.load?.tpot_ms?.p50 == null ? "–" : `${fmt(v.load.tpot_ms.p50, 1)} ms`}</td>
    </tr>
  );
}
