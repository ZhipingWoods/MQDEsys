// 模型库 (framework: 模型量化 → 模型选择): models already on this server, the HF catalog (data/models/catalog.json,
// refreshed daily, two-level grouping), HF search & download, and ModelScope (not integrated yet -- said so, not hidden).
import { Fragment, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, CatalogModel, fmt, fmtBytes, fmtParams, LibraryModel } from "../api";
import { DOMAIN_SHORT, modelInDomain, useDomain } from "../domain";
import { useAsync, useDebounced } from "../hooks";
import { HubPanel } from "./Fit";

const CAP_LABEL: Record<string, string> = { text: "文本", vision: "视觉", moe: "MoE", audio: "音频" };

function HfCatalog() {
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
  // T-1303's two-level grouping, so 200 rows are not one flat 7000-pixel table: category -> family_group. A
  // search / filter flattens it (the user is looking for something specific); otherwise categories start closed.
  const filtering = !!(dq.search || dq.provider || dq.capability);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [allOpen, setAllOpen] = useState(false);
  const groups = useMemo(() => {
    const m = new Map<string, Map<string, CatalogModel[]>>();
    for (const r of rows) {
      const c = r.category || "其他", f = r.family_group || r.provider || "其他";
      if (!m.has(c)) m.set(c, new Map());
      const fm = m.get(c)!;
      if (!fm.has(f)) fm.set(f, []);
      fm.get(f)!.push(r);
    }
    const size = (fm: Map<string, CatalogModel[]>) => [...fm.values()].reduce((a, b) => a + b.length, 0);
    return [...m.entries()].sort((a, b) => size(b[1]) - size(a[1])).map(([c, fm]) => ({ category: c, n: size(fm), families: [...fm.entries()].sort((a, b) => b[1].length - a[1].length) }));
  }, [rows]);
  const isOpen = (c: string) => filtering || allOpen || open.has(c);
  const toggle = (c: string) => setOpen((o) => { const n = new Set(o); if (n.has(c)) n.delete(c); else n.add(c); return n; });

  return (
    <>
      <div className="page-head">
        <div>
          <h2 style={{ margin: 0 }}>HF 模型库 · 后台每日增量同步</h2>
          <p className="subtitle">
            HF 模型库（{all.data?.total ?? "–"} 条，{all.data?.demoted ?? 0} 条降级；更新于 {all.data?.updated_at || "–"}）。
            {all.data && all.data.refresh_interval_s > 0
              ? `服务自动每 ${Math.round(all.data.refresh_interval_s / 3600)} 小时刷新（上次${all.data.last_refresh_ok ? "成功" : "失败"}${all.data.last_refresh_at ? "：" + all.data.last_refresh_at : ""}）；`
              : "本服务未开启自动刷新，"}
            也可在服务器上运行 <code>mqde catalog update --limit N</code>。点击一行进入部署预评估。
          </p>
        </div>
      </div>
      <div className="card">
        <div className="form-row">
          <div className="field grow">
            <label>搜索</label>
            <input placeholder="模糊搜索，如 qwen25、llama3（按字符顺序匹配，不用连着打全）" value={search} onChange={(e) => setSearch(e.target.value)} />
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
        <div className="form-row" style={{ marginBottom: 8, alignItems: "center" }}>
          <span className="muted small" data-testid="catalog-count">{res.loading ? "加载中…" : `${rows.length} 条 · ${groups.length} 个类别`}</span>
          {!filtering && (
            <button className="secondary" onClick={() => { setAllOpen(!allOpen); setOpen(new Set()); }} data-testid="catalog-toggle-all">
              {allOpen ? "全部收起" : "全部展开"}
            </button>
          )}
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
              {groups.map((g) => (
                <Fragment key={g.category}>
                  <tr className="clickable group-head" onClick={() => toggle(g.category)} data-testid={`catalog-group-${g.category}`}>
                    <td colSpan={9}>
                      <span className="muted">{isOpen(g.category) ? "▾" : "▸"}</span> <b>{g.category}</b> <span className="muted small">· {g.n} 个模型 · {g.families.length} 个系列{isOpen(g.category) ? "" : `：${g.families.slice(0, 6).map(([f, xs]) => `${f} (${xs.length})`).join("、")}${g.families.length > 6 ? " …" : ""}`}</span>
                    </td>
                  </tr>
                  {isOpen(g.category) &&
                    g.families.map(([f, xs]) => (
                      <Fragment key={f}>
                        {g.families.length > 1 && (
                          <tr className="group-sub">
                            <td colSpan={9} className="small">
                              {f} <span className="muted">· {xs.length}</span>
                            </td>
                          </tr>
                        )}
                        {xs.map((m) => (
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
                      </Fragment>
                    ))}
                </Fragment>
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

const TABS: [string, string][] = [["local", "本地模型"], ["catalog", "HF 模型库"], ["hf", "HF 搜索与下载"], ["modelscope", "ModelScope"]];

export default function ModelLibraryPage() {
  const [params, setParams] = useSearchParams();
  const tab = TABS.some(([t]) => t === params.get("tab")) ? (params.get("tab") as string) : "local";
  return (
    <>
      <h1>模型库</h1>
      <p className="subtitle">本地模型加载、HuggingFace 模型库（每日增量同步）、在线搜索下载。本地模型按导航栏的领域筛选。</p>
      <div className="tabs" data-testid="library-tabs">
        {TABS.map(([t, label]) => (
          <div key={t} className={`tab ${tab === t ? "on" : ""}`} onClick={() => setParams({ tab: t }, { replace: true })} data-testid={`library-tab-${t}`}>
            {label}
            {t === "modelscope" && <span className="badge" style={{ marginLeft: 6 }}>未接入</span>}
          </div>
        ))}
      </div>
      {tab === "local" ? <LocalModels /> : tab === "catalog" ? <HfCatalog /> : tab === "hf" ? <HubPanel /> : <ModelScopeNote />}
    </>
  );
}

function ModelScopeNote() {
  return (
    <div className="card" data-testid="modelscope-note">
      <h2>ModelScope 下载</h2>
      <p className="note">
        还没有接入：服务端的下载器（hub::download_file）目前只认 HuggingFace 的 resolve 接口（含 HF_ENDPOINT 镜像）。ModelScope 的
        文件列表与下载接口不同，需要单独的适配（任务 T-0204）。国内网络下载 HF 模型可以先设置 HF_ENDPOINT 指向镜像，在「HF 搜索与下载」里使用。
      </p>
    </div>
  );
}

// one colour per component modality, shared by every card so the bars are comparable at a glance
const MOD_COLOR: Record<string, string> = { language: "--series-1", vision: "--series-2", action: "--series-3", audio: "--series-4", tts: "--series-5", mtp: "--series-6" };

function LocalModels() {
  const { domain } = useDomain();
  const lib = useAsync(api.models, []);
  const [all, setAll] = useState(false);
  const rows = (lib.data ?? []).filter((m) => all || modelInDomain(m, domain));
  return (
    <>
      <div className="form-row" style={{ alignItems: "center", marginBottom: 10 }}>
        <span className="small muted" data-testid="local-count">
          {all ? "全部领域" : `${DOMAIN_SHORT[domain] ?? domain} 领域`}：{rows.length} 个模型（服务器允许目录下的 HF 格式检查点，含 config.json + safetensors）
        </span>
        <label className="small" style={{ marginLeft: "auto" }}>
          <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} data-testid="local-all" /> 显示全部领域
        </label>
      </div>
      {lib.error && <div className="error-box">{lib.error}</div>}
      <div className="model-grid" data-testid="local-models">
        {rows.map((m) => <LocalModelCard key={m.path} m={m} />)}
        {lib.data && rows.length === 0 && (
          <div className="card empty">
            当前领域还没有本地模型。去「HF 搜索与下载」下载一个，或把检查点放到服务器的允许目录下。
          </div>
        )}
      </div>
    </>
  );
}

function LocalModelCard({ m }: { m: LibraryModel }) {
  const comps = m.components ?? [];
  const total = comps.reduce((a, c) => a + c.bytes, 0) || 1;
  const q = encodeURIComponent(m.path);
  return (
    <div className="card model-card" data-testid={`local-model-${m.name}`}>
      <div className="model-card-head">
        <div style={{ minWidth: 0 }}>
          <div className="model-card-name" title={m.path}>{m.name}</div>
          <div className="small muted mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.path}</div>
        </div>
        <span className="badge accent">{(m.modality ?? "?").toUpperCase()}</span>
      </div>
      {m.error ? (
        <div className="note warn">{m.error}</div>
      ) : (
        <>
          <div className="model-card-stats">
            <div><span className="muted small">参数</span><b>{fmtParams(m.logical_params ?? m.total_params ?? 0)}</b></div>
            <div><span className="muted small">权重</span><b>{m.total_bytes ? fmtBytes(m.total_bytes) : "–"}</b></div>
            <div><span className="muted small">结构</span><b className="mono small">{m.family ?? m.model_type ?? "–"}</b></div>
          </div>
          {comps.length > 0 && (
            <>
              <div className="stackbar" aria-label="各组件权重占比">
                {comps.map((c, i) => (
                  <span key={i} style={{ width: `${(100 * c.bytes) / total}%`, background: `var(${MOD_COLOR[c.modality] ?? "--series-7"})` }}
                    data-tip={`${c.name}（${c.modality}）${fmtBytes(c.bytes)} · ${c.n_blocks} 块`} />
                ))}
              </div>
              <div className="legend-row small muted">
                {[...new Set(comps.map((c) => c.modality))].map((mod) => (
                  <span key={mod}><i style={{ background: `var(${MOD_COLOR[mod] ?? "--series-7"})` }} />{mod}</span>
                ))}
              </div>
            </>
          )}
          {m.checkpoint_quant && (
            <div className="small" style={{ marginTop: 6 }}>
              <span className="badge warn">已量化检查点</span> {m.checkpoint_quant.method} · {m.checkpoint_quant.bits} bit · g{m.checkpoint_quant.group_size}
            </div>
          )}
        </>
      )}
      <div className="model-card-actions">
        <Link to={`/new?model=${q}`}>新建作业</Link>
        <Link to={`/graph?model=${q}`}>模型结构</Link>
        <Link to={`/estimate?model=${q}`}>资源预估</Link>
      </div>
    </div>
  );
}
