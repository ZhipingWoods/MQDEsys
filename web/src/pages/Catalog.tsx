import { useEffect, useState } from "react";
import api from "../api";
import type { CatalogModel } from "../api";

export default function CatalogPage() {
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [search, setSearch] = useState("");
  const [provider, setProvider] = useState("");
  const [capability, setCapability] = useState("");
  const [includeDemoted, setIncludeDemoted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [meta, setMeta] = useState<{ updated_at: string; total: number; demoted: number } | null>(null);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.catalog(search, provider, capability, 200, includeDemoted);
      setModels(res.models);
      setMeta({ updated_at: res.updated_at, total: res.total, demoted: res.demoted });
    } catch (e: any) {
      setError(e.message || "加载失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleSearch = () => {
    load();
  };

  const providers = Array.from(new Set(models.map(m => m.provider))).filter(Boolean).sort();
  const capabilities = Array.from(new Set(models.flatMap(m => m.capabilities))).filter(Boolean).sort();

  return (
    <div className="page">
      <header className="page-header">
        <h1>模型目录</h1>
        <p className="page-subtitle">
          从 HuggingFace 模型库中搜索和浏览可用模型
        </p>
      </header>

      <div className="card" style={{ marginBottom: "1rem" }}>
        <div style={{ display: "grid", gap: "0.75rem" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "0.75rem" }}>
            <input
              type="text"
              placeholder="搜索模型名称、作者或架构..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              style={{ width: "100%" }}
            />
            <button onClick={handleSearch} disabled={loading}>
              {loading ? "搜索中..." : "搜索"}
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem" }}>
            <label>
              <span style={{ fontSize: "0.875rem", opacity: 0.7 }}>提供者</span>
              <select value={provider} onChange={(e) => setProvider(e.target.value)}>
                <option value="">全部</option>
                {providers.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>

            <label>
              <span style={{ fontSize: "0.875rem", opacity: 0.7 }}>能力</span>
              <select value={capability} onChange={(e) => setCapability(e.target.value)}>
                <option value="">全部</option>
                {capabilities.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>

            <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "1.25rem" }}>
              <input
                type="checkbox"
                checked={includeDemoted}
                onChange={(e) => setIncludeDemoted(e.target.checked)}
              />
              <span style={{ fontSize: "0.875rem" }}>包含降级模型</span>
            </label>
          </div>
        </div>
      </div>

      {error && (
        <div className="card" style={{ background: "var(--error-bg)", borderColor: "var(--error)", marginBottom: "1rem" }}>
          <p style={{ color: "var(--error)", margin: 0 }}>{error}</p>
        </div>
      )}

      {meta && (
        <div style={{ fontSize: "0.875rem", opacity: 0.7, marginBottom: "0.75rem" }}>
          找到 {meta.total} 个模型
          {meta.demoted > 0 && !includeDemoted && ` (${meta.demoted} 个已降级)`}
          {meta.updated_at && ` · 目录更新于 ${new Date(meta.updated_at).toLocaleString("zh-CN")}`}
        </div>
      )}

      <div className="card">
        {loading && models.length === 0 ? (
          <p style={{ textAlign: "center", opacity: 0.5, padding: "2rem" }}>加载中...</p>
        ) : models.length === 0 ? (
          <p style={{ textAlign: "center", opacity: 0.5, padding: "2rem" }}>
            {search || provider || capability ? "未找到匹配的模型" : "目录为空，请先运行 mqde catalog update"}
          </p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: "0.75rem 0.5rem", fontWeight: 600 }}>模型</th>
                  <th style={{ textAlign: "left", padding: "0.75rem 0.5rem", fontWeight: 600 }}>架构</th>
                  <th style={{ textAlign: "right", padding: "0.75rem 0.5rem", fontWeight: 600 }}>参数</th>
                  <th style={{ textAlign: "left", padding: "0.75rem 0.5rem", fontWeight: 600 }}>提供者</th>
                  <th style={{ textAlign: "left", padding: "0.75rem 0.5rem", fontWeight: 600 }}>能力</th>
                  <th style={{ textAlign: "right", padding: "0.75rem 0.5rem", fontWeight: 600 }}>下载</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, idx) => (
                  <tr
                    key={idx}
                    style={{
                      borderBottom: idx < models.length - 1 ? "1px solid var(--border-subtle)" : undefined,
                      opacity: m.demoted ? 0.5 : 1,
                    }}
                  >
                    <td style={{ padding: "0.75rem 0.5rem" }}>
                      <div style={{ fontWeight: 500 }}>{m.name}</div>
                      {m.ollama_name && (
                        <div style={{ fontSize: "0.8125rem", opacity: 0.6, marginTop: "0.125rem" }}>
                          {m.ollama_name}
                        </div>
                      )}
                    </td>
                    <td style={{ padding: "0.75rem 0.5rem", fontSize: "0.875rem" }}>
                      {m.architecture || "—"}
                    </td>
                    <td style={{ padding: "0.75rem 0.5rem", fontSize: "0.875rem", textAlign: "right" }}>
                      {m.param_b ? `${m.param_b.toFixed(1)}B` : "—"}
                    </td>
                    <td style={{ padding: "0.75rem 0.5rem", fontSize: "0.875rem" }}>
                      <span style={{
                        padding: "0.125rem 0.5rem",
                        borderRadius: "999px",
                        background: "var(--accent-subtle)",
                        color: "var(--accent)",
                        fontSize: "0.8125rem",
                      }}>
                        {m.provider}
                      </span>
                    </td>
                    <td style={{ padding: "0.75rem 0.5rem", fontSize: "0.8125rem" }}>
                      {m.capabilities.length > 0 ? (
                        <div style={{ display: "flex", gap: "0.25rem", flexWrap: "wrap" }}>
                          {m.capabilities.map((cap, i) => (
                            <span
                              key={i}
                              style={{
                                padding: "0.125rem 0.375rem",
                                borderRadius: "4px",
                                background: "var(--surface-2)",
                                opacity: 0.8,
                              }}
                            >
                              {cap}
                            </span>
                          ))}
                        </div>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td style={{ padding: "0.75rem 0.5rem", fontSize: "0.875rem", textAlign: "right", opacity: 0.7 }}>
                      {m.downloads_30d != null ? (
                        m.downloads_30d >= 1000
                          ? `${(m.downloads_30d / 1000).toFixed(1)}k`
                          : m.downloads_30d.toString()
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
