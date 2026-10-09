import { api, fmtBytes } from "../api";
import { useAsync } from "../hooks";

export default function HardwarePage() {
  const hw = useAsync(api.hardware, []);
  const profiles = useAsync(api.hardwareProfiles, []);
  return (
    <>
      <h1>硬件</h1>
      <p className="subtitle">C 语言探针实时探测（NVML 动态加载，无厂商库也可运行），并匹配硬件画像库。</p>
      {hw.error && <div className="error-box">{hw.error}</div>}
      <div className="card table-wrap">
        <h2>本机设备</h2>
        <table>
          <thead>
            <tr>
              <th className="num">#</th>
              <th>类型</th>
              <th>名称</th>
              <th>架构</th>
              <th className="num">总内存</th>
              <th className="num">空闲</th>
              <th>画像</th>
              <th>作业池</th>
            </tr>
          </thead>
          <tbody>
            {hw.data?.devices.map((d) => {
              const pool = hw.data?.job_gpus.pool.find((p) => d.kind === "gpu" && p.gpu === d.index);
              return (
                <tr key={`${d.vendor ?? ""}${d.kind}${d.index}`}>
                  <td className="num">{d.index}</td>
                  <td>{d.kind.toUpperCase()}</td>
                  <td>
                    {d.name}
                    {d.vendor && d.kind !== "cpu" && <span className="muted small"> · {d.vendor}</span>}
                  </td>
                  <td className="mono">{d.arch}</td>
                  <td className="num">{fmtBytes(d.mem_total_bytes)}</td>
                  <td className="num">{fmtBytes(d.mem_free_bytes)}</td>
                  <td className="mono small">{d.profile ?? "–"}</td>
                  <td>{pool ? pool.job ? <span className="status running">{pool.job}</span> : <span className="status good">空闲</span> : <span className="muted">—</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="card table-wrap">
        <h2>硬件画像库（data/hardware_db）</h2>
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>名称</th>
              <th>架构</th>
              <th className="num">显存</th>
              <th className="num">带宽</th>
              <th className="num">BF16</th>
              <th className="num">FP8</th>
              <th className="num">INT8</th>
              <th>原生格式</th>
            </tr>
          </thead>
          <tbody>
            {profiles.data?.map((p) => (
              <tr key={p.id}>
                <td className="mono small">{p.id}</td>
                <td>{p.name}</td>
                <td className="mono">{p.arch}</td>
                <td className="num">{fmtBytes(p.mem_bytes)}</td>
                <td className="num">{p.mem_bw_gbps} GB/s</td>
                <td className="num">{p.peak_tflops.bf16 ?? "–"}</td>
                <td className="num">{p.peak_tflops.fp8 ?? "–"}</td>
                <td className="num">{p.peak_tflops.int8 ?? "–"}</td>
                <td className="small">{p.native_formats.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
