import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, Device, fmtBytes, HalPlugin, HardwareProfile, ProfileResult } from "../api";
import { HardwarePicker, hardwareCategory } from "../components/HardwarePicker";
import { Chart, baseOption, series, token } from "../charts";
import { useAsync } from "../hooks";
import BackendEnvsCard from "./BackendEnvs";
import { Tile } from "./Estimate";
import { KindIcon, VendorIcon } from "./hwIcons";

const STATUS_LABEL: Record<string, string> = { loaded: "已加载", inactive: "未发现设备", rejected: "已拒绝", disabled: "已禁用", unknown: "未知" };
const STATUS_CLASS: Record<string, string> = { loaded: "good", inactive: "muted", rejected: "bad", disabled: "muted", unknown: "muted" };
const DTYPES = ["fp32", "fp16", "bf16", "fp8", "int8", "int4"];
const DTYPE_LABEL: Record<string, string> = { fp32: "FP32", fp16: "FP16", bf16: "BF16", fp8: "FP8", int8: "INT8", int4: "INT4" };

/** One HAL plugin as a card. The probe devices themselves are shown above; this block is the runtime
 *  that found them, plus an optional micro-benchmark. */
function PluginCard({ p }: { p: HalPlugin }) {
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
    <article className={`hal-card ${p.status}`} data-testid={`plugin-${p.name}`}>
      <div className="hal-head">
        <VendorIcon vendor={p.vendor || p.name} />
        <div className="hal-title">
          <b>{p.name}</b>
          <small>{p.vendor} · v{p.version}</small>
        </div>
        <span className={`status ${STATUS_CLASS[p.status]}`} data-testid={`plugin-status-${p.name}`}>{STATUS_LABEL[p.status]}</span>
      </div>
      <div className="hal-meta">
        <span>{p.devices} 个设备</span>
        {p.reason && <span className="small muted">{p.reason}</span>}
      </div>
      {canRun ? (
        <div className="hal-bench">
          <div className="form-row" style={{ margin: 0, gap: 6 }}>
            <input type="number" min={0} value={device} style={{ width: 54 }} onChange={(e) => setDevice(Number(e.target.value))} aria-label="设备序号" />
            <select value={dtype} onChange={(e) => setDtype(e.target.value)} aria-label="数据类型">
              {DTYPES.map((d) => <option key={d}>{d}</option>)}
            </select>
            <button className="secondary" disabled={busy !== null} onClick={() => run("bandwidth")} data-testid={`bench-bw-${p.name}`}>
              {busy === "bandwidth" ? "测试中…" : "测带宽"}
            </button>
            <button className="secondary" disabled={busy !== null} onClick={() => run("gemm")} data-testid={`bench-gemm-${p.name}`}>
              {busy === "gemm" ? "测试中…" : "测 GEMM"}
            </button>
          </div>
          {err && <div className="error-box" data-testid={`bench-err-${p.name}`}>{err}</div>}
          {results.length > 0 && (
            <ul className="small hal-results" data-testid={`bench-results-${p.name}`}>
              {results.map((r, i) => (
                <li key={i}>
                  设备 {r.device} · {r.kind}{r.kind === "gemm" ? ` ${r.dtype}` : ""}：<b>{r.value.toFixed(r.value < 10 ? 3 : 1)} {r.unit}</b>（{r.seconds.toFixed(3)} s）
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <span className="muted small">{p.status === "loaded" ? "该插件没有微基准" : "插件未启用"}</span>
      )}
    </article>
  );
}

const TFLOPS_ORDER = ["fp32", "bf16", "fp16", "fp8", "int8", "int4"];

/** One hardware_db row; clicking it opens the full peak_tflops breakdown, every native format and
 *  the vendor datasheet this entry's numbers came from (T-1307). */
function ProfileRow({ p }: { p: HardwareProfile }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr onClick={() => setOpen(!open)} className={`profile-row ${hardwareCategory(p)}`} data-testid={`profile-row-${p.id}`}>
        <td><VendorIcon vendor={p.vendor} /></td>
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
              <h3 style={{ marginTop: 0 }}>{p.name} <span className="muted small mono">{p.id}</span></h3>
              <div className="tiles">
                <Tile label="显存容量" value={fmtBytes(p.mem_bytes)} />
                <Tile label="显存带宽" value={`${p.mem_bw_gbps} GB/s`} />
                <Tile label="架构" value={p.arch || "–"} />
                <Tile label="厂商" value={p.vendor || "–"} />
              </div>
              <table>
                <thead>
                  <tr>
                    {TFLOPS_ORDER.filter((k) => p.peak_tflops[k] != null).map((k) => <th key={k} className="num">{k.toUpperCase()}</th>)}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    {TFLOPS_ORDER.filter((k) => p.peak_tflops[k] != null).map((k) => <td key={k} className="num">{p.peak_tflops[k]} TFLOPS</td>)}
                  </tr>
                </tbody>
              </table>
              <p className="small">原生格式：{p.native_formats.join("、") || "–"}</p>
              {p.source && (
                <p className="small muted">
                  数据来源： <a href={p.source} target="_blank" rel="noreferrer" data-testid={`profile-source-${p.id}`}>{p.source}</a>
                </p>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

const DEVICE_PLUGIN: Record<string, string> = { nvidia: "cuda", amd: "rocm", apple: "metal", huawei: "ascend", ascend: "ascend", cpu: "cpu" };

/** Probe results as cards: what the machine actually reports, not a datasheet. */
function ProbeDevices({ devices, pool, profiles, onSelect }: {
  devices: Device[];
  pool: { gpu: number; job: string | null }[];
  profiles: HardwareProfile[];
  onSelect: (id: string) => void;
}) {
  const byId = useMemo(() => new Map(profiles.map((p) => [p.id, p])), [profiles]);
  return (
    <div className="card" data-testid="probe-devices">
      <div className="card-head">
        <div>
          <h2>目标设备探针</h2>
          <p className="note" style={{ margin: 0 }}>来自 NVML / HAL / 系统探针的实时信息：架构、target、内存、驱动与作业池占用。</p>
        </div>
        <span className="badge">{devices.length} 个设备</span>
      </div>
      <div className="probe-grid">
        {devices.map((d) => {
          const profile = d.profile ? byId.get(d.profile) : undefined;
          const poolItem = d.kind === "gpu" ? pool.find((p) => p.gpu === d.index) : undefined;
          const used = d.mem_total_bytes > 0 ? 100 * (1 - d.mem_free_bytes / d.mem_total_bytes) : 0;
          const plugin = DEVICE_PLUGIN[(d.vendor || "").toLowerCase()] ?? (d.kind === "cpu" ? "cpu" : "");
          return (
            <article className={`probe-card ${d.kind}`} data-testid={`probe-device-${d.kind}-${d.index}`}>
              <div className="probe-head">
                <KindIcon kind={d.kind} />
                <div className="probe-title">
                  <b>{d.name}</b>
                  <small>{d.vendor || "—"} · {d.kind.toUpperCase()}{d.index >= 0 ? ` #${d.index}` : ""}</small>
                </div>
                {poolItem && <span className={`status ${poolItem.job ? "running" : "good"}`}>{poolItem.job ? "作业占用" : "空闲"}</span>}
              </div>
              <div className="probe-facts">
                <span><small>架构</small><b className="mono">{d.arch || "—"}</b></span>
                <span><small>Target</small><b className="mono">{d.target || "—"}</b></span>
                <span><small>计算单元</small><b className="num">{d.compute_units || "—"}</b></span>
                <span><small>驱动</small><b className="mono">{d.driver_version || "—"}</b></span>
              </div>
              <div className="probe-memory">
                <div className="probe-memory-head">
                  <span>内存</span>
                  <span>{fmtBytes(d.mem_free_bytes)} 空闲 / {fmtBytes(d.mem_total_bytes)}</span>
                </div>
                <div className="probe-bar"><i style={{ width: `${Math.min(100, Math.max(0, used))}%` }} /></div>
                <span className="small muted">已用 {used.toFixed(0)}%{plugin ? ` · 探针插件 ${plugin}` : ""}</span>
              </div>
              <div className="probe-foot">
                {profile ? <span className="badge accent">画像 · {profile.name}</span> : <span className="muted small">没有匹配的硬件画像</span>}
                {profile && <button className="secondary small" onClick={() => onSelect(profile.id)}>选为部署目标</button>}
              </div>
            </article>
          );
        })}
        {devices.length === 0 && <p className="muted small">探针没有返回任何设备。</p>}
      </div>
    </div>
  );
}

function fmtTflops(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k`;
  if (v >= 100) return String(Math.round(v));
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

/** Multi-hardware compute capability as lines (log scale) plus an exact-value table. */
function CapabilityChart({ profiles }: { profiles: HardwareProfile[] }) {
  const preferred = ["nvidia/rtx4090", "nvidia/jetson-agx-orin-64g", "nvidia/jetson-agx-thor", "nvidia/dgx-spark-gb10", "nvidia/h100-sxm"];
  const [chosen, setChosen] = useState<string[] | null>(null);
  const [showValues, setShowValues] = useState(true);
  useEffect(() => {
    if (chosen !== null || profiles.length === 0) return;
    const ids = preferred.filter((id) => profiles.some((p) => p.id === id));
    setChosen(ids.length ? ids : profiles.slice(0, 4).map((p) => p.id));
  }, [profiles, chosen]); // eslint-disable-line react-hooks/exhaustive-deps
  const selected = (chosen ?? []).map((id) => profiles.find((p) => p.id === id)).filter((p): p is HardwareProfile => !!p);
  const dtypes = DTYPES.filter((dt) => selected.some((p) => p.peak_tflops[dt] != null));
  const toggle = (id: string) => setChosen((cur) => {
    const ids = cur ?? [];
    return ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id];
  });
  return (
    <div className="card" data-testid="capability-matrix">
      <div className="card-head">
        <div>
          <h2>算力能力</h2>
          <p className="note" style={{ margin: 0 }}>各精度峰值 TFLOPS；纵轴为对数刻度，点/线标签与下方表格都可看到具体数值。</p>
        </div>
        <label className="small" style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input type="checkbox" checked={showValues} onChange={(e) => setShowValues(e.target.checked)} data-testid="capability-show-values" /> 图上显示数值
        </label>
      </div>
      <div className="chips" style={{ marginBottom: 10 }} data-testid="capability-picker">
        {profiles.filter((p) => Object.keys(p.peak_tflops).length > 0).map((p) => (
          <span key={p.id} className={`chip ${selected.some((x) => x.id === p.id) ? "on" : ""}`} onClick={() => toggle(p.id)} data-testid={`capability-pick-${p.id}`}>
            {p.name}
          </span>
        ))}
      </div>
      {selected.length === 0 || dtypes.length === 0 ? (
        <p className="muted small">选择至少一个带算力数据的硬件。</p>
      ) : (
        <>
          <div data-testid="capability-chart">
            <Chart
              height={330}
              option={() => {
                const o = baseOption();
                return {
                  ...o,
                  grid: { left: 58, right: 24, top: 44, bottom: 34 },
                  legend: { ...(o.legend as object), type: "scroll", top: 0, left: 0 },
                  tooltip: {
                    ...(o.tooltip as object),
                    trigger: "axis",
                    valueFormatter: (v: number | null) => (v == null ? "–" : `${v} TFLOPS`),
                  },
                  xAxis: { ...(o.xAxis as object), type: "category", data: dtypes.map((d) => DTYPE_LABEL[d] ?? d), boundaryGap: false },
                  yAxis: {
                    ...(o.yAxis as object),
                    type: "log",
                    logBase: 10,
                    name: "TFLOPS",
                    nameGap: 12,
                    axisLabel: { color: token("--text-muted"), formatter: (v: number) => fmtTflops(v) },
                  },
                  series: selected.map((p, i) => ({
                    name: p.name,
                    type: "line",
                    smooth: 0.18,
                    connectNulls: true,
                    symbol: "circle",
                    symbolSize: 6,
                    color: series(i),
                    itemStyle: { color: series(i) },
                    lineStyle: { width: 2 },
                    emphasis: { focus: "series" },
                    label: {
                      show: showValues && selected.length <= 6,
                      position: "top",
                      fontSize: 9,
                      color: token("--text-secondary"),
                      formatter: (x: { value?: number | null }) => (x.value == null ? "" : fmtTflops(x.value)),
                    },
                    data: dtypes.map((dt) => p.peak_tflops[dt] ?? null),
                  })),
                };
              }}
            />
          </div>
          <div className="table-wrap" style={{ marginTop: 8 }} data-testid="capability-values">
            <table className="matrix">
              <thead>
                <tr>
                  <th>硬件</th>
                  {dtypes.map((d) => <th key={d} className="num">{DTYPE_LABEL[d] ?? d}</th>)}
                </tr>
              </thead>
              <tbody>
                {selected.map((p, i) => (
                  <tr key={p.id}>
                    <td><span className="legend-dot" style={{ background: series(i) }} />{p.name}</td>
                    {dtypes.map((d) => <td key={d} className="num small">{p.peak_tflops[d] != null ? `${p.peak_tflops[d]} TFLOPS` : "—"}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function ProfileGroups({ profiles }: { profiles: HardwareProfile[] }) {
  const groups = useMemo(() => {
    const order = ["edge", "workstation", "datacenter"] as const;
    const labels = { edge: "边缘设备", workstation: "桌面工作站", datacenter: "数据中心" };
    return order
      .map((cat) => ({ cat, label: labels[cat], items: profiles.filter((p) => hardwareCategory(p) === cat) }))
      .filter((g) => g.items.length > 0);
  }, [profiles]);
  return (
    <div className="card" data-testid="profile-groups">
      <div className="card-head">
        <div>
          <h2>全部硬件画像</h2>
          <p className="note" style={{ margin: 0 }}>按设备形态归纳；悬停高亮，点击一行展开完整规格与厂商数据来源。</p>
        </div>
        <span className="badge">{profiles.length} 条</span>
      </div>
      {groups.map((g) => (
        <section key={g.cat} className={`profile-group ${g.cat}`} data-testid={`profile-group-${g.cat}`}>
          <div className="profile-group-head">
            <span className="profile-group-dot" />
            <b>{g.label}</b>
            <span className="muted small">{g.items.length} 个画像</span>
          </div>
          <div className="table-wrap">
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
                {g.items.map((p) => <ProfileRow key={p.id} p={p} />)}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}

export default function HardwarePage() {
  const hw = useAsync(api.hardware, []);
  const profiles = useAsync(api.hardwareProfiles, []);
  const plugins = useAsync(api.hardwarePlugins, []);
  const [selected, setSelected] = useState("");
  const userPicked = useRef(false);
  // Start from the machine's own GPU profile when there is one; otherwise the first catalog entry.
  // The two API calls race, so re-apply the default when the device probe arrives -- unless the user
  // has already clicked a card.
  useEffect(() => {
    if (userPicked.current || !profiles.data?.length) return;
    const local = hw.data?.devices.find((d) => d.kind === "gpu" && d.profile)?.profile;
    setSelected(local && profiles.data.some((p) => p.id === local) ? local : profiles.data[0].id);
  }, [profiles.data, hw.data]);
  const choose = (id: string) => {
    userPicked.current = true;
    setSelected(id);
  };
  const chosen = profiles.data?.find((p) => p.id === selected);
  const q = encodeURIComponent(selected);
  return (
    <>
      <h1>硬件与后端</h1>
      <p className="subtitle">先看推理后端与本机探针，再选一台部署目标直接新建作业。所有数字来自实时探针或硬件画像，并标明来源。</p>
      {hw.error && <div className="error-box">{hw.error}</div>}

      <BackendEnvsCard />

      <div className="card" data-testid="hw-target-card">
        <div className="card-head">
          <div>
            <h2>选择部署硬件</h2>
            <p className="note" style={{ margin: 0 }}>Jetson Orin / Thor、DGX Spark 与数据中心 GPU；点卡片选中即可带到新建作业。</p>
          </div>
          {chosen && <span className="badge accent">已选 · {chosen.name}</span>}
        </div>
        <HardwarePicker value={selected} onChange={choose} profiles={profiles.data ?? []} detected={hw.data?.devices ?? []} />
        {chosen && (
          <div className="hw-selected-bar" data-testid="hw-selected-bar">
            <div>
              <span className="muted small">部署目标</span>
              <div><b>{chosen.name}</b><span className="muted small"> · {fmtBytes(chosen.mem_bytes)} · {chosen.mem_bw_gbps} GB/s · {chosen.arch}</span></div>
              <span className="small muted">原生格式：{chosen.native_formats.join(" / ") || "—"}</span>
            </div>
            <div className="form-row" style={{ margin: 0 }}>
              <Link to={`/new?hw=${q}`}><button data-testid="hw-use-full">用它新建全流程作业</button></Link>
              <Link to={`/new/quantize?hw=${q}`}><button className="secondary" data-testid="hw-use-quantize">新建量化作业</button></Link>
              <Link to={`/new/eval?hw=${q}`}><button className="secondary" data-testid="hw-use-eval">新建评测作业</button></Link>
            </div>
          </div>
        )}
      </div>

      <ProbeDevices devices={hw.data?.devices ?? []} pool={hw.data?.job_gpus.pool ?? []} profiles={profiles.data ?? []} onSelect={choose} />

      <CapabilityChart profiles={profiles.data ?? []} />

      <div className="card" data-testid="plugins-card">
        <div className="card-head">
          <div>
            <h2>探针运行时（HAL 插件）</h2>
            <p className="note" style={{ margin: 0 }}>每个厂商一个插件；加载失败的插件只显示原因，不影响其他设备。微基准实测带宽与 GEMM 吞吐。</p>
          </div>
        </div>
        {plugins.error && <div className="error-box">{plugins.error}</div>}
        <div className="hal-grid">
          {plugins.data?.map((p) => <PluginCard key={p.name} p={p} />)}
        </div>
        {plugins.data && plugins.data.length === 0 && <p className="muted small">没有找到插件（检查 MQDE_PLUGIN_DIR 或可执行文件旁的 plugins/ 目录）。</p>}
      </div>

      <ProfileGroups profiles={profiles.data ?? []} />
    </>
  );
}
