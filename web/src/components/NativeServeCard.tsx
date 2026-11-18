import { NativeServeReport, fmt } from "../api";

/** Native continuous-output serving result for VLA / WM / EWM. */
export function NativeServeCard({ recipe, d }: { recipe: string; d: NativeServeReport }) {
  const st = d.status === "succeeded" ? "good" : d.status === "degraded" ? "warn" : d.status === "skipped" ? "muted" : "critical";
  return (
    <div className="card" data-testid={`native-serve-${recipe}`}>
      <div className="card-head">
        <div>
          <h2>原生连续输出服务 · {recipe}</h2>
          <p className="note" style={{ margin: 0 }}>
            {d.protocol ?? "MQDE native"} · loopback HTTP · 直接服务当前内存中的量化模型
          </p>
        </div>
        <span className={`status ${st}`}>{d.status}</span>
      </div>
      {d.error && <div className="error-box">{d.error}</div>}
      {d.smoke && (
        <div className="tiles">
          <div className="tile"><div className="label">Smoke</div><div className="value">{d.smoke.ok ? "通过" : "失败"}</div><div className="hint">{fmt(d.smoke.latency_ms, 1)} ms</div></div>
          <div className="tile"><div className="label">压测成功</div><div className="value">{d.load ? `${d.load.ok} / ${d.load.requests}` : "–"}</div><div className="hint">并发 {d.load?.concurrency ?? "–"}</div></div>
          <div className="tile"><div className="label">吞吐</div><div className="value">{fmt(d.load?.requests_per_s, 2)}</div><div className="hint">req/s</div></div>
          <div className="tile"><div className="label">延迟 p50 / p95</div><div className="value">{fmt(d.load?.latency_ms.p50, 1)} / {fmt(d.load?.latency_ms.p95, 1)}</div><div className="hint">ms</div></div>
        </div>
      )}
      {d.load?.errors && d.load.errors.length > 0 && (
        <ul className="note warn">{d.load.errors.map((e) => <li key={e}>{e}</li>)}</ul>
      )}
      {d.endpoint && <p className="small muted mono">endpoint: {d.endpoint}</p>}
    </div>
  );
}
