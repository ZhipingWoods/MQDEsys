import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import "./styles.css";
import HomePage from "./pages/Home";
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
  { group: "总览", items: [["/", "⌂", "总览"]] },
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

// Collapsed = an icon-only rail; the choice is remembered per browser. "[" toggles it from anywhere but a text field.
function useSidebarCollapsed(): [boolean, () => void] {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem("mqde-sidebar") === "collapsed";
    } catch {
      return false;
    }
  });
  const toggle = () => setCollapsed((c) => !c);
  useEffect(() => {
    try {
      localStorage.setItem("mqde-sidebar", collapsed ? "collapsed" : "expanded");
    } catch {
      /* storage unavailable: the choice holds for this session */
    }
  }, [collapsed]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (e.key !== "[" || e.metaKey || e.ctrlKey || e.altKey) return;
      if (t && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName))) return;
      setCollapsed((c) => !c);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return [collapsed, toggle];
}

function App() {
  const [collapsed, toggle] = useSidebarCollapsed();
  return (
    <div className={`layout ${collapsed ? "collapsed" : ""}`}>
      <aside className="sidebar" data-testid="sidebar" data-collapsed={collapsed ? "1" : "0"}>
        <div className="brand">
          <div className="brand-mark">MQ</div>
          <div className="brand-text">
            <div className="brand-name">MQDEsys</div>
            <div className="brand-sub">模型量化部署评估</div>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="nav-group">{g.group}</div>
              {g.items.map(([to, ico, label]) => (
                <NavLink key={to} to={to} end={to === "/"} aria-label={label} data-tip={collapsed ? label : undefined}>
                  <span className="ico">{ico}</span>
                  <span className="nav-label">{label}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          {!collapsed && <span>v0.2</span>}
          {!collapsed && <ThemeToggle />}
          <button
            className="ghost sidebar-toggle"
            onClick={toggle}
            aria-label={collapsed ? "展开导航栏" : "收起导航栏"}
            aria-expanded={!collapsed}
            data-tip={collapsed ? "展开导航栏 [" : "收起导航栏 ["}
            data-testid="sidebar-toggle"
          >
            {collapsed ? "»" : "«"}
          </button>
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<HomePage />} />
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
