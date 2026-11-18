import { useMemo } from "react";
import { api, EvalJobReport, fmt, fmtBytes, OfflineBenchmarks, VllmDeploy } from "../api";
import { useAsync } from "../hooks";
import { Tile } from "./Estimate";
import { SweepCard } from "./SweepCard";

export const EVAL_STAGES: [string, string][] = [
  ["validate", "校验模型"],
  ["eval_prepare", "评测数据"],
  ["eval_hf", "HF 离线 PPL"],
  ["eval_llamacpp", "llama.cpp PPL"],
  ["deploy_vllm", "vLLM 部署"],
  ["deploy_sglang", "SGLang 部署"],
  ["deploy_llamacpp", "llama.cpp 服务"],
  ["report", "报告"],
];

const pct = (v: number | null | undefined) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`);

function statusText(v: { status?: string; reason?: string; error?: string } | undefined): string {
  if (!v) return "–";
  if (v.status === "skipped") return `跳过：${v.reason ?? "环境不支持"}`;
  if (v.status === "failed") return `失败：${v.error ?? "未知错误"}`;
  return v.status ?? "–";
}

function ServeCard({ name, v }: { name: string; v: VllmDeploy }) {
  return (
    <div className="card table-wrap" data-testid={`eval-serve-${name}`}>
      <h2>{name} 部署评测</h2>
      <table>
        <tbody>
          <tr><th>状态</th><td>{statusText(v)}</td></tr>
          <tr><th>部署后 PPL</th><td className="num">{fmt(v.served_ppl, 3)}</td></tr>
          <tr><th>相对离线 PPL</th><td className="num">{pct(v.served_vs_real_ppl_pct)}</td></tr>
          <tr><th>服务吞吐</th><td className="num">{fmt(v.load?.output_tokens_per_s, 1)} tok/s</td></tr>
          <tr><th>TTFT p50 / p99</th><td className="num">{fmt(v.load?.ttft_ms?.p50, 0)} / {fmt(v.load?.ttft_ms?.p99, 0)} ms</td></tr>
        </tbody>
      </table>
      {v.lm_eval && v.lm_eval.status === "succeeded" && (
        <p className="small" data-testid="eval-lm-eval">
          lm-eval：{Object.entries(v.lm_eval.tasks ?? {}).map(([t, r]) =>
            `${t} ${Object.values(r.metrics).map((m) => m.value.toFixed(4)).join(" / ")}`).join(" · ")}
        </p>
      )}
      {v.evalscope && v.evalscope.status === "succeeded" && (
        <p className="small" data-testid="eval-evalscope">
          EvalScope：{Object.entries(v.evalscope.datasets ?? {}).map(([d, r]) =>
            `${d} ${Object.values(r.metrics).filter((m) => m.value != null).map((m) => m.value!.toFixed(4)).join(" / ")}`).join(" · ")}
        </p>
      )}
      {v.sweep && <SweepCard title={name} sweep={v.sweep} />}
    </div>
  );
}

function BenchmarksCard({ b }: { b: OfflineBenchmarks }) {
  const catalog = useAsync(api.benchmarks, []);
  const meta = useMemo(() => new Map([...(catalog.data?.benchmarks ?? []), ...(catalog.data?.native ?? []), ...(catalog.data?.evalscope ?? [])].map((x) => [x.name, x])), [catalog.data]);
  const rows = Object.entries(b.benchmarks);
  if (!rows.length) return null;
  const done = rows.filter(([, r]) => r.status !== "skipped");
  const totalSamples = done.reduce((n, [, r]) => n + (r.n ?? 0), 0);
  const totalSeconds = done.reduce((n, [, r]) => n + (r.seconds ?? 0), 0);
  return (
    <div className="card" data-testid="eval-benchmarks">
      <div className="card-head">
        <div>
          <h2>数据集评测</h2>
          <p className="note" style={{ margin: 0 }}>{done.length} 个数据集完成 · {totalSamples} 条样本 · {totalSeconds.toFixed(1)} s</p>
        </div>
        <span className="badge">{rows.length} 项</span>
      </div>
      <div className="eval-bench-grid">
        {rows.map(([name, r]) => {
          const m = meta.get(name);
          const skipped = r.status === "skipped";
          return (
            <article key={name} className={`eval-bench-card ${skipped ? "skipped" : "done"}`}>
              <div className="eval-bench-head">
                <div>
                  <b>{m?.label ?? name}</b>
                  <div className="small muted mono">{name}{m?.group ? ` · ${m.group}` : ""}</div>
                </div>
                <span className={`status ${skipped ? "muted" : "good"}`}>{skipped ? "跳过" : "完成"}</span>
              </div>
              {skipped ? (
                <p className="small muted">{r.reason ?? "环境不支持"}</p>
              ) : (
                <>
                  <div className="eval-bench-metrics">
                    {Object.entries(r.metrics ?? {}).map(([k, v]) => (
                      <div key={k}>
                        <small>{k.toUpperCase()}</small>
                        <b className="num">{k === "ppl" ? v.toFixed(3) : `${(v * 100).toFixed(2)}%`}</b>
                      </div>
                    ))}
                    <div><small>样本</small><b className="num">{r.n ?? "–"}</b></div>
                    <div><small>耗时</small><b className="num">{r.seconds == null ? "–" : `${r.seconds.toFixed(1)}s`}</b></div>
                  </div>
                  {r.categories && Object.keys(r.categories).length > 0 && (
                    <div className="eval-bench-cats">
                      {Object.entries(r.categories).map(([c, v]) => (
                        <span key={c} title={`${c}: n=${v.n}`}>{c} <b>{((v.accuracy ?? 0) * 100).toFixed(1)}%</b></span>
                      ))}
                    </div>
                  )}
                </>
              )}
            </article>
          );
        })}
      </div>
      {rows.some(([, r]) => r.categories) && (
        <p className="small muted" style={{ marginBottom: 0 }}>分类准确率按评测集原始 subject 聚合；悬停可看样本数。</p>
      )}
    </div>
  );
}

/** 仅评测作业：直接用已有的量化 checkpoint 跑评测，不重新量化。 */
export function EvalReportView({ rep }: { rep: EvalJobReport }) {
  const c = rep.checkpoint;
  const e = rep.eval ?? {};
  const ppl = e.hf?.ppl ?? e.llamacpp?.ppl;
  const dataset = rep.data?.eval?.dataset ?? rep.spec.eval_dataset;
  const windows = rep.data?.eval?.windows ?? rep.spec.eval_windows;
  const seqLen = rep.data?.eval?.seq_len ?? rep.spec.eval_seq_len;
  return (
    <>
      <div className="tiles" data-testid="eval-tiles">
        <Tile label="已量化模型" value={c ? (c.type ? `GGUF ${c.type}` : `${c.format}`) : "–"}
              hint={c ? `${c.source === "artifact" ? `作业 ${c.job} · ` : ""}${c.bytes ? fmtBytes(c.bytes) : c.path}` : ""} />
        <Tile label="评测 PPL" value={fmt(ppl, 3)} hint={dataset} />
        <Tile label="评测设置" value={`${windows} 窗口 × ${seqLen}`} hint="PPL 与上下文长度和窗口数有关" />
        <Tile label="设备" value={rep.env?.device_name ?? rep.spec.device} hint="本作业不重新量化" />
      </div>
      {c && (
        <p className="small" data-testid="eval-checkpoint">
          模型 <span className="mono">{c.path}</span>（{c.format}）
          {c.source === "artifact" && c.job ? `，复用作业 ${c.job} 的导出产物` : "，外部量化 checkpoint"}。
        </p>
      )}
      {e.hf && e.hf.status !== "succeeded" && (
        <p className="note" data-testid="eval-hf-skipped">
          HF 离线加载未执行：{statusText(e.hf)}。量化 checkpoint 可能需要对应的加载器（AWQ → autoawq，GPTQ →
          gptqmodel）；也可以在部署步骤选择 vLLM / SGLang 并勾选部署，让推理框架直接加载。
        </p>
      )}
      {(e.hf?.status === "succeeded" || e.llamacpp?.status === "succeeded") && (
        <div className="card">
          <h2>离线评测</h2>
          <table data-testid="eval-offline">
            <thead>
              <tr><th>路径</th><th>状态</th><th className="num">PPL</th><th className="num">误差</th><th className="num">窗口 × 上下文</th><th>说明</th></tr>
            </thead>
            <tbody>
              {e.hf && (
                <tr>
                  <td>HF / transformers</td>
                  <td>{statusText(e.hf)}</td>
                  <td className="num">{fmt(e.hf.ppl, 3)}</td>
                  <td className="num">{fmt(e.hf.ppl_err, 3)}</td>
                  <td className="num">{e.hf.windows ?? windows} × {e.hf.seq_len ?? seqLen}</td>
                  <td>{e.hf.dtype ?? ""}</td>
                </tr>
              )}
              {e.llamacpp && (
                <tr>
                  <td>llama.cpp</td>
                  <td>{statusText(e.llamacpp)}</td>
                  <td className="num">{fmt(e.llamacpp.ppl, 3)}</td>
                  <td className="num">{fmt(e.llamacpp.ppl_err, 3)}</td>
                  <td className="num">{e.llamacpp.chunks ?? windows} × {e.llamacpp.ctx ?? seqLen}</td>
                  <td>{e.llamacpp.backend ?? ""}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
      {e.hf?.benchmarks && <BenchmarksCard b={e.hf.benchmarks} />}
      {rep.deploy?.vllm && <ServeCard name="vLLM" v={rep.deploy.vllm} />}
      {rep.deploy?.sglang && <ServeCard name="SGLang" v={rep.deploy.sglang} />}
      {rep.deploy?.llamacpp && (
        <div className="card" data-testid="eval-serve-llamacpp">
          <h2>llama.cpp 服务</h2>
          <p className="small">{statusText(rep.deploy.llamacpp)}</p>
        </div>
      )}
    </>
  );
}
