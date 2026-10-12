import { useState } from "react";
import { api, fmtBytes, HalPlugin, HardwareProfile, ProfileResult } from "../api";
import { useAsync } from "../hooks";
import { Tile } from "./Estimate";
import { KindIcon, VendorIcon } from "./hwIcons";

const STATUS_LABEL: Record<string, string> = { loaded: "已加载", inactive: "未发现设备", rejected: "已拒绝", disabled: "已禁用", unknown: "未知" };
const STATUS_CLASS: Record<string, string> = { loaded: "good", inactive: "muted", rejected: "bad", disabled: "muted", unknown: "muted" };
const DTYPES = ["fp32", "fp16", "bf16", "int8"];

/** One plugin row with its micro-benchmark buttons: runs bandwidth or GEMM on a chosen device and shows the measured number. */
function PluginRow({ p }: { p: HalPlugin }) {
  const [device, setDevice] = useState(0);
  const [dtype, setDtype] = useState("fp32");
  const [busy, setBusy] = useState<string | null>(null);
  const [results, setResults] = useState<ProfileResult[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const run = async (kind: "bandwidth" | "gemm") => {
    setBusy(kind);
    setErr(null);
    try {
      const r = await api.profileDevice(p.name, device, kind, dtype);
      setResults((x) => [r, ...x].slice(0, 6));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(null);
    }
  };
  const canRun = p.status === "loaded" && p.has_profile;
  return (
    <tr data-testid={`plugin-${p.name}`}>
      <td>
        <VendorIcon vendor={p.vendor || p.name} />
      </td>
      <td className="mono">{p.name}</td>
      <td>{p.vendor}</td>
      <td className="mono small">{p.version}</td>
      <td>
        <span className={`status ${STATUS_CLASS[p.status]}`} data-testid={`plugin-status-${p.name}`}>{STATUS_LABEL[p.status]}</span>
        {p.reason && <div className="small muted">{p.reason}</div>}
      </td>
      <td className="num">{p.devices}</td>
      <td>
        {canRun ? (
          <div className="form-row" style={{ margin: 0, gap: 6 }}>
            <input type="number" min={0} value={device} style={{ width: 56 }} onChange={(e) => setDevice(Number(e.target.value))} aria-label="设备序号" />
            <select value={dtype} onChange={(e) => setDtype(e.target.value)} aria-label="数据类型">
              {DTYPES.map((d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
            <button disabled={busy !== null} onClick={() => run("bandwidth")} data-testid={`bench-bw-${p.name}`}>
              {busy === "bandwidth" ? "测试中…" : "测带宽"}
            </button>
            <button disabled={busy !== null} onClick={() => run("gemm")} data-testid={`bench-gemm-${p.name}`}>
              {busy === "gemm" ? "测试中…" : "测 GEMM"}
            </button>
          </div>
        ) : (
          <span className="muted small">{p.status === "loaded" ? "该插件没有微基准" : "—"}</span>
        )}
        {err && <div className="error-box" data-testid={`bench-err-${p.name}`}>{err}</div>}
        {results.length > 0 && (
          <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 16 }} data-testid={`bench-results-${p.name}`}>
            {results.map((r, i) => (
              <li key={i}>
                设备 {r.device} · {r.kind}{r.kind === "gemm" ? ` ${r.dtype}` : ""}：<b>{r.value.toFixed(r.value < 10 ? 3 : 1)} {r.unit}</b>（{r.seconds.toFixed(3)} s）
              </li>
            ))}
          </ul>
        )}
      </td>
    </tr>
  );
}

const TFLOPS_ORDER = ["fp32", "bf16", "fp16", "fp8", "int8", "int4"];

/** One hardware_db row; clicking it opens a detail view with the full peak_tflops breakdown, every
 * native format and the vendor datasheet this entry's numbers came from (T-1307). */
function ProfileRow({ p }: { p: HardwareProfile }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr onClick={() => setOpen(!open)} style={{ cursor: "pointer" }} data-testid={`profile-row-${p.id}`}>
        <td>
          <VendorIcon vendor={p.vendor} />
        </td>
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
      {open && (
        <tr data-testid={`profile-detail-${p.id}`}>
          <td colSpan={10}>
            <div className="card" style={{ margin: "4px 0" }}>
              <h3 style={{ marginTop: 0 }}>
                {p.name} <span className="muted small mono">{p.id}</span>
              </h3>
              <div className="tiles">
                <Tile label="显存容量" value={fmtBytes(p.mem_bytes)} />
                <Tile label="显存带宽" value={`${p.mem_bw_gbps} GB/s`} />
                <Tile label="架构" value={p.arch || "–"} />
                <Tile label="厂商" value={p.vendor || "–"} />
              </div>
              <table>
                <thead>
                  <tr>
                    {TFLOPS_ORDER.filter((k) => p.peak_tflops[k] != null).map((k) => (
                      <th key={k} className="num">
                        {k.toUpperCase()}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    {TFLOPS_ORDER.filter((k) => p.peak_tflops[k] != null).map((k) => (
                      <td key={k} className="num">
                        {p.peak_tflops[k]} TFLOPS
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
              <p className="small">原生格式：{p.native_formats.join("、") || "–"}</p>
              {p.source && (
                <p className="small muted">
                  数据来源：{" "}
                  <a href={p.source} target="_blank" rel="noreferrer" data-testid={`profile-source-${p.id}`}>
                    {p.source}
                  </a>
                </p>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function HardwarePage() {
  const hw = useAsync(api.hardware, []);
  const profiles = useAsync(api.hardwareProfiles, []);
  const plugins = useAsync(api.hardwarePlugins, []);
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
              <th />
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
                  <td>
                    <KindIcon kind={d.kind} />
                  </td>
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
      <div className="card table-wrap" data-testid="plugins-card">
        <h2>硬件插件（HAL）</h2>
        <p className="note">
          每个厂商一个插件（cuda、rocm、metal、cpu……），放在插件目录里即被发现；坏插件只会显示“已拒绝”和原因，不影响其他插件。
          微基准实测显存带宽和 GEMM 吞吐，一次只跑一个，以免互相干扰。
        </p>
        {plugins.error && <div className="error-box">{plugins.error}</div>}
        <table>
          <thead>
            <tr>
              <th />
              <th>插件</th>
              <th>厂商</th>
              <th>版本</th>
              <th>状态</th>
              <th className="num">设备数</th>
              <th>微基准</th>
            </tr>
          </thead>
          <tbody>
            {plugins.data?.map((p) => (
              <PluginRow key={p.name} p={p} />
            ))}
          </tbody>
        </table>
        {plugins.data && plugins.data.length === 0 && <p className="muted">没有找到插件（检查 MQDE_PLUGIN_DIR 或可执行文件旁的 plugins/ 目录）。</p>}
      </div>
      <div className="card table-wrap">
        <h2>硬件画像库（data/hardware_db）</h2>
        <p className="note">点击一行查看该硬件更完整的单卡信息。</p>
        <table>
          <thead>
            <tr>
              <th />
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
              <ProfileRow key={p.id} p={p} />
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
