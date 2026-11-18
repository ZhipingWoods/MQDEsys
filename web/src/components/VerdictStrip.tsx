import { LLMJobReport, LLMRecipeReport } from "../api";
import { lowerIsBetterTone, pplVerdict, signedPct, VerdictStatus } from "../semantics";

const LABEL: Record<VerdictStatus, string> = { pass: "达标", fail: "超出预算", record: "仅记录", none: "没有门槛" };

/** One line per recipe: how much PPL it lost against the unquantized model, and whether that is within what the project accepts. */
export function VerdictStrip({ base, rs }: { base: number; rs: LLMRecipeReport[] }) {
  if (!base || !rs.length) return null;
  const rows = rs.map((r) => {
    const pct = (r.eval.fake_quant_wo_kv.ppl / base - 1) * 100;
    return { r, pct, v: pplVerdict(r.recipe, pct) };
  });
  const gated = rows.filter((x) => x.v.status === "pass" || x.v.status === "fail");
  return (
    <div className="card" data-testid="verdict-strip">
      <h2>结论 · 相对原模型的 PPL 变化与验收门槛</h2>
      <table>
        <thead>
          <tr>
            <th>配方</th>
            <th>算法</th>
            <th>PPL（W+A）</th>
            <th>相对原模型</th>
            <th>门槛（§11.4）</th>
            <th>结论</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ r, pct, v }) => (
            <tr key={r.recipe} data-testid={`verdict-${r.recipe}`} data-status={v.status}>
              <td><b>{r.recipe}</b></td>
              <td>{r.algorithm}</td>
              <td>{r.eval.fake_quant_wo_kv.ppl.toFixed(3)}</td>
              <td>
                <span style={{ color: lowerIsBetterTone(pct) === "good" ? "var(--good-text)" : lowerIsBetterTone(pct) === "bad" ? "var(--critical)" : undefined }} data-testid={`verdict-delta-${r.recipe}`}>
                  {signedPct(pct)}
                </span>
              </td>
              <td>{v.budget === undefined ? "–" : v.budget === null ? "不设门槛" : `≤ ${v.budget}%`}</td>
              <td>
                <span className={`badge ${v.status === "pass" ? "good" : v.status === "fail" ? "warn" : ""}`} data-testid={`verdict-badge-${r.recipe}`}>
                  {v.status === "pass" ? "✓ " : v.status === "fail" ? "✗ " : ""}
                  {LABEL[v.status]}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        参考阈值为：FP8 ≤ 1%，W4A16 ≤ 5%，W4A8 ≤ 8%，W2 只记录。
        {gated.length > 0 && ` 本作业有门槛的 ${gated.length} 个配方中 ${gated.filter((x) => x.v.status === "pass").length} 个达标。`}
        PPL 越低越好，所以变化为负是变好。这是伪量化（fake-quant）评估的结果，真实导出与部署后的数字在下面各卡片里。
      </p>
    </div>
  );
}

/** Stages and deployments that did not finish cleanly: a run with holes in it must not read as a finished one. */
export function RunCompleteness({ rep }: { rep: LLMJobReport }) {
  const stageIssues = (rep.stages ?? []).filter((s) => s.status !== "succeeded");
  const deployIssues: { what: string; status: string; why?: string }[] = [];
  for (const r of Object.values(rep.recipes ?? {}))
    for (const [fmt, ex] of Object.entries(r.export ?? {}))
      for (const b of ["vllm", "sglang"] as const) {
        const d = ex[b];
        if (d && d.status !== "succeeded") deployIssues.push({ what: `${r.recipe} · ${fmt} · ${b}`, status: d.status, why: d.reason ?? d.error });
      }
  if (!stageIssues.length && !deployIssues.length) return null;
  return (
    <div className="card" data-testid="run-incomplete">
      <h2>这次运行有没跑完整的部分</h2>
      <ul className="note">
        {stageIssues.map((s) => (
          <li key={s.stage} data-testid={`incomplete-stage-${s.stage}`}>
            阶段 <b>{s.stage}</b>：{s.status}
            {s.error ? ` — ${s.error}` : ""}
          </li>
        ))}
        {deployIssues.map((d) => (
          <li key={d.what} data-testid="incomplete-deploy">
            部署 <b>{d.what}</b>：{d.status}
            {d.why ? ` — ${d.why}` : ""}
          </li>
        ))}
      </ul>
      <p className="note">下面的数字只覆盖成功的部分，不要把它当作完整结果。</p>
    </div>
  );
}
