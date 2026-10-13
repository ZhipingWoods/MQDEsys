import { ConcurrencySweep, fmt } from "../api";
import { baseOption, Chart, series, token } from "../charts";

const WHY: Record<string, string> = {
  throughput: "再加并发吞吐涨幅不到 10%",
  ttft: "再加并发 TTFT p99 就超出预算",
  "not-reached": "测到的最高并发吞吐仍在增长，还没饱和",
  "single-level": "只测了一个并发档",
};

/** 并发扫描：吞吐、TTFT、TPOT、ITL 随并发的变化，以及吞吐不再增长的位置。 */
export function SweepCard({ sweep, title }: { sweep: ConcurrencySweep | { error: string }; title: string }) {
  if ("error" in sweep)
    return (
      <p className="note" data-testid="sweep-error">
        {title} 并发扫描失败：{sweep.error}
      </p>
    );
  const pts = sweep.points;
  const sat = sweep.saturation;
  const ms = (v?: number) => (v == null ? "–" : `${fmt(v, v < 10 ? 1 : 0)} ms`);
  return (
    <div className="card" data-testid="sweep-card">
      <h2>{title} · 并发扫描</h2>
      <Chart
        height={260}
        option={() => {
          const o = baseOption();
          return {
            ...o,
            legend: { ...(o.legend as object), data: ["吞吐 tok/s", "TTFT p50", "TTFT p99"] },
            tooltip: { ...(o.tooltip as object), trigger: "axis" },
            xAxis: { ...(o.xAxis as object), type: "category", data: pts.map((p) => p.concurrency), name: "并发", nameLocation: "middle", nameGap: 24, nameTextStyle: { color: token("--text-muted") } },
            yAxis: [
              { ...(o.yAxis as object), type: "value", name: "tok/s", nameTextStyle: { color: token("--text-muted") } },
              { ...(o.yAxis as object), type: "value", name: "TTFT ms", splitLine: { show: false }, nameTextStyle: { color: token("--text-muted") } },
            ],
            series: [
              {
                name: "吞吐 tok/s", type: "bar", barMaxWidth: 28, yAxisIndex: 0,
                itemStyle: { color: series(0), borderRadius: [3, 3, 0, 0] },
                data: pts.map((p) => ({ value: p.output_tokens_per_s ?? null, itemStyle: { color: sat && p.concurrency === sat.concurrency ? series(2) : series(0), borderRadius: [3, 3, 0, 0] } })),
              },
              { name: "TTFT p50", type: "line", yAxisIndex: 1, symbol: "circle", itemStyle: { color: series(1) }, data: pts.map((p) => p.ttft_ms?.p50 ?? null) },
              { name: "TTFT p99", type: "line", yAxisIndex: 1, symbol: "circle", lineStyle: { type: "dashed" }, itemStyle: { color: series(3) }, data: pts.map((p) => p.ttft_ms?.p99 ?? null) },
            ],
          };
        }}
      />
      <table data-testid="sweep-table">
        <thead>
          <tr>
            <th className="num">并发</th>
            <th className="num">成功 / 失败</th>
            <th className="num">吞吐 tok/s</th>
            <th className="num">TTFT p50 / p99</th>
            <th className="num">TPOT p50 / p99</th>
            <th className="num">ITL p50 / p99</th>
          </tr>
        </thead>
        <tbody>
          {pts.map((p) => (
            <tr key={p.concurrency} className={sat && p.concurrency === sat.concurrency ? "worst" : ""} data-testid={`sweep-row-${p.concurrency}`}>
              <td className="num">{p.concurrency}</td>
              <td className="num">{p.ok} / {p.failed}</td>
              <td className="num">{p.output_tokens_per_s == null ? "–" : fmt(p.output_tokens_per_s, 0)}</td>
              <td className="num">{ms(p.ttft_ms?.p50)} / {ms(p.ttft_ms?.p99)}</td>
              <td className="num">{ms(p.tpot_ms?.p50)} / {ms(p.tpot_ms?.p99)}</td>
              <td className="num">{ms(p.itl_ms?.p50)} / {ms(p.itl_ms?.p99)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note" data-testid="sweep-verdict">
        {sat
          ? `建议并发 ${sat.concurrency}（高亮行）：${WHY[sat.reason]}${sat.next_gain_pct != null ? `（下一档吞吐 ${sat.next_gain_pct >= 0 ? "+" : ""}${sat.next_gain_pct.toFixed(1)}%）` : ""}。`
          : "没有任何并发档成功完成请求，无法判断饱和点。"}
        {sweep.max_ttft_p99_ms != null && ` TTFT p99 预算 ${sweep.max_ttft_p99_ms} ms。`}
        {sweep.peak && ` 峰值 ${fmt(sweep.peak.output_tokens_per_s, 0)} tok/s @ 并发 ${sweep.peak.concurrency}。`}
      </p>
    </div>
  );
}
