import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";
import "./styles.css";
import FitPage from "./pages/Fit";
import GraphPage from "./pages/Graph";
import EstimatePage from "./pages/Estimate";
import NewJobPage from "./pages/NewJob";
import JobsPage from "./pages/Jobs";
import JobPage from "./pages/Job";
import HardwarePage from "./pages/Hardware";
import ComparePage from "./pages/Compare";
import CatalogPage from "./pages/Catalog";
import AnalysisPage from "./pages/Analysis";
import SuggestPage from "./pages/Suggest";

// T-1306: "误差分析"/"优化建议" used to live only as tabs buried inside a job's own detail page; promoted
// to their own top-level nav items (standalone /analysis, /suggest routes) so the site map is flat.
const NAV: { group: string; items: [string, string, string][] }[] = [
  { group: "评估前", items: [["/fit", "◎", "部署预评估"], ["/graph", "◇", "模型结构"], ["/estimate", "▦", "资源预估"]] },
  { group: "量化与评测", items: [["/new", "＋", "新建作业"], ["/jobs", "▶", "作业与执行"], ["/analysis", "△", "误差分析"], ["/suggest", "☆", "优化建议"], ["/compare", "⇄", "配方对比"]] },
  { group: "部署", items: [["/hardware", "▣", "硬件与后端"], ["/catalog", "◈", "模型目录"]] },
];

function ThemeToggle() {
  const [theme, setTheme] = useState<string>(() => {
    try {
      return localStorage.getItem("mqde-theme") ?? "dark";
    } catch {
      return "dark";
    }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("mqde-theme", theme);
    } catch {
      /* storage unavailable: theme still applies for this session */
    }
  }, [theme]);
  return (
    <button className="ghost" style={{ minHeight: 26, padding: "2px 8px" }} onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="切换主题">
      {theme === "dark" ? "☾ 深色" : "☀ 浅色"}
    </button>
  );
}

function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">MQ</div>
          <div>
            <div className="brand-name">MQDEsys</div>
            <div className="brand-sub">模型量化部署评估</div>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="nav-group">{g.group}</div>
              {g.items.map(([to, ico, label]) => (
                <NavLink key={to} to={to}>
                  <span className="ico">{ico}</span>
                  {label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span>v0.2</span>
          <ThemeToggle />
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/fit" replace />} />
          <Route path="/fit" element={<FitPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/estimate" element={<EstimatePage />} />
          <Route path="/new" element={<NewJobPage />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/jobs/:id" element={<JobPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/analysis/:jobId" element={<AnalysisPage />} />
          <Route path="/suggest" element={<SuggestPage />} />
          <Route path="/suggest/:jobId" element={<SuggestPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/hardware" element={<HardwarePage />} />
          <Route path="/catalog" element={<CatalogPage />} />
        </Routes>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
