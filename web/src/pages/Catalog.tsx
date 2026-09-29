// HF model catalog (data/models/catalog.json, built by `mqde catalog update`). Filters run on the
// server; picking a row opens the deployment pre-evaluation for that catalog entry (no weights needed).
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, fmt } from "../api";
import { useAsync, useDebounced } from "../hooks";

const CAP_LABEL: Record<string, string> = { text: "文本", vision: "视觉", moe: "MoE", audio: "音频" };

export default function CatalogPage() {
  const [search, setSearch] = useState("");
  const [provider, setProvider] = useState("");
  const [capability, setCapability] = useState("");
  const [demoted, setDemoted] = useState(false);
  const dq = useDebounced({ search, provider, capability, demoted }, 250);
  const all = useAsync(() => api.catalog("", "", "", 5000, true), []);
  const res = useAsync(() => api.catalog(dq.search, dq.provider, dq.capability, 500, dq.demoted), [JSON.stringify(dq)]);
  const nav = useNavigate();
  // Dropdown options come from the whole catalog, not from the filtered page.
  const providers = useMemo(() => {
    const n = new Map<string, number>();
    for (const m of all.data?.models ?? []) n.set(m.provider, (n.get(m.provider) ?? 0) + 1);
    return [...n.entries()].sort((a, b) => b[1] - a[1]);
  }, [all.data]);
  const caps = useMemo(() => [...new Set((all.data?.models ?? []).flatMap((m) => m.capabilities))].sort(), [all.data]);
  const rows = res.data?.models ?? [];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>模型目录</h1>
          <p className="subtitle">
            HF 模型库（{all.data?.total ?? "–"} 条，{all.data?.demoted ?? 0} 条降级；更新于 {all.data?.updated_at || "–"}）。
            服务器上运行 <code>mqde catalog update --limit N</code> 刷新。点击一行进入部署预评估。
          </p>
        </div>
      </div>
      <div className="card">
        <div className="form-row">
          <div className="field grow">
            <label>搜索</label>
            <input placeholder="名称子串，如 qwen3、llama" value={search} onChange={(e) => setSearch(e.target.value)} />
          </div>
          <div className="field" style={{ minWidth: 200 }}>
            <label>提供者</label>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="">全部（{providers.length}）</option>
              {providers.map(([p, n]) => (
                <option key={p} value={p}>
                  {p} · {n}
                </option>
              ))}
            </select>
          </div>
          <div className="field" style={{ minWidth: 140 }}>
            <label>能力</label>
            <select value={capability} onChange={(e) => setCapability(e.target.value)}>
              <option value="">全部</option>
              {caps.map((c) => (
                <option key={c} value={c}>
                  {CAP_LABEL[c] ?? c}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>&nbsp;</label>
            <label className="small" style={{ display: "flex", gap: 6, alignItems: "center", minHeight: 32 }}>
              <input type="checkbox" checked={demoted} onChange={(e) => setDemoted(e.target.checked)} /> 包含降级条目
            </label>
          </div>
        </div>
        {(res.error || all.error) && <div className="error-box">{res.error || all.error}</div>}
        <div className="muted small" style={{ marginBottom: 8 }}>
          {res.loading ? "加载中…" : `${rows.length} 条`}
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>模型</th>
                <th>架构</th>
                <th className="num">参数</th>
                <th>存储精度</th>
                <th className="num">上下文</th>
                <th>能力</th>
                <th>许可证</th>
                <th className="num">下载量</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m) => (
                <tr key={m.name} className="clickable" style={{ opacity: m.demoted ? 0.55 : 1 }}
                  onClick={() => nav(`/fit?source=catalog&search=${encodeURIComponent(m.name)}`)}>
                  <td>
                    <b>{m.name}</b>
                  </td>
                  <td className="small muted">{m.architecture || "–"}</td>
                  <td className="num">{m.parameter_count || "–"}</td>
                  <td>
                    <span className="badge">{m.quantization}</span>
                  </td>
                  <td className="num">{m.context_length ? m.context_length.toLocaleString() : "–"}</td>
                  <td>
                    {m.capabilities.map((c) => (
                      <span key={c} className="badge" style={{ marginRight: 4 }}>
                        {CAP_LABEL[c] ?? c}
                      </span>
                    ))}
                  </td>
                  <td className="small">{m.license || "–"}</td>
                  <td className="num">{m.hf_downloads >= 1e6 ? `${fmt(m.hf_downloads / 1e6, 1)}M` : m.hf_downloads >= 1e3 ? `${fmt(m.hf_downloads / 1e3, 1)}k` : m.hf_downloads}</td>
                  <td className="small muted">{m.demote_reason}</td>
                </tr>
              ))}
              {!res.loading && rows.length === 0 && (
                <tr>
                  <td colSpan={9} className="empty">
                    {all.data?.total === 0 ? "目录为空：在服务器上运行 mqde catalog update" : "没有符合条件的模型"}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
