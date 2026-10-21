import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import "./styles.css";
import { DomainProvider, DomainSwitcher } from "./domain";
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

// Grouped as the framework (MQEsys框架.md) lays the system out: 模型量化 → 任务作业 → bench评估 → 模型导出 → 硬件部署 →
// 误差评估 → 资源预估, with 总览 first as the landing page. The domain switch above the groups scopes every page.
const NAV: { group: string; items: [string, string, string][] }[] = [
  { group: "总览", items: [["/", "⌂", "总览"]] },
  { group: "模型量化", items: [["/catalog", "◈", "模型库"]] },
  { group: "任务作业", items: [["/new", "＋", "新建作业"], ["/jobs", "▶", "作业与执行"]] },
  { group: "bench 评估", items: [["/compare", "⇄", "评测结果对比"]] },
  { group: "硬件部署", items: [["/hardware", "▣", "硬件与后端"]] },
  { group: "误差评估", items: [["/graph", "◇", "模型结构"], ["/analysis", "△", "误差分析"], ["/suggest", "☆", "优化建议"]] },
  { group: "资源预估", items: [["/estimate", "▦", "资源预估"], ["/fit", "◎", "部署预评估"]] },
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
        <DomainSwitcher collapsed={collapsed} />
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
      <DomainProvider>
        <App />
      </DomainProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
