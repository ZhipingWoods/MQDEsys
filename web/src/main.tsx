import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes, useLocation } from "react-router-dom";
import "./styles.css";
import { DomainProvider, DomainSwitcher } from "./domain";
import { Icon, IconName } from "./components/Icon";
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

// Decision order, not implementation order: size the model and check the hardware first, then pick a model,
// run a job, and read the evaluation. The resource group therefore sits directly under 总览.
const NAV: { group: string; items: { to: string; icon: IconName; label: string }[] }[] = [
  { group: "总览", items: [{ to: "/", icon: "home", label: "总览" }] },
  {
    group: "资源与适配",
    items: [
      { to: "/estimate", icon: "gauge", label: "资源预估" },
      { to: "/fit", icon: "target", label: "部署预评估" },
      { to: "/hardware", icon: "hardware", label: "硬件与后端" },
    ],
  },
  {
    group: "模型",
    items: [
      { to: "/catalog", icon: "library", label: "模型库" },
      { to: "/graph", icon: "graph", label: "模型结构" },
    ],
  },
  {
    group: "作业",
    items: [
      { to: "/new", icon: "plus", label: "新建全流程作业" },
      { to: "/new/quantize", icon: "layers", label: "新建量化作业" },
      { to: "/new/eval", icon: "target", label: "新建评测作业" },
      { to: "/jobs", icon: "play", label: "作业与执行" },
    ],
  },
  {
    group: "评估",
    items: [
      { to: "/compare", icon: "compare", label: "评测结果对比" },
      { to: "/analysis", icon: "alert", label: "误差分析" },
      { to: "/suggest", icon: "sparkles", label: "优化建议" },
    ],
  },
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
    <button
      className="icon-button theme-toggle"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      aria-label={theme === "dark" ? "切换浅色主题" : "切换深色主题"}
      data-tip={theme === "dark" ? "切换浅色主题" : "切换深色主题"}
    >
      <Icon name={theme === "dark" ? "sun" : "moon"} />
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

function AppTopbar() {
  const { pathname } = useLocation();
  const flat = NAV.flatMap((g) => g.items.map((item) => ({ ...item, group: g.group })));
  const current =
    [...flat]
      .sort((a, b) => b.to.length - a.to.length)
      .find((item) => (item.to === "/" ? pathname === "/" : pathname === item.to || pathname.startsWith(`${item.to}/`))) ?? flat[0];
  return (
    <header className="topbar">
      <div className="crumbs">
        <span className="crumb-root">
          <Icon name="activity" size={15} /> MQDE Control
        </span>
        <Icon name="chevron" size={13} className="crumb-sep" />
        <span className="crumb-group">{current.group}</span>
        <Icon name="chevron" size={13} className="crumb-sep" />
        <strong>{current.label}</strong>
      </div>
      <div className="topbar-actions">
        <span className="workspace-pill">
          <i /> 本地工作台
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
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
            <div className="brand-sub">Quantization operations</div>
          </div>
        </div>
        <DomainSwitcher collapsed={collapsed} />
        <nav className="nav">
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="nav-group">{g.group}</div>
              {g.items.map((item) => (
                <NavLink key={item.to} to={item.to} end={item.to === "/"} aria-label={item.label} data-tip={collapsed ? item.label : undefined}>
                  <span className="ico">
                    <Icon name={item.icon} />
                  </span>
                  <span className="nav-label">{item.label}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          {!collapsed && (
            <span className="build-tag">
              <i /> v0.2 · local
            </span>
          )}
          <button
            className="ghost icon-button sidebar-toggle"
            onClick={toggle}
            aria-label={collapsed ? "展开导航栏" : "收起导航栏"}
            aria-expanded={!collapsed}
            data-tip={collapsed ? "展开导航栏 [" : "收起导航栏 ["}
            data-testid="sidebar-toggle"
          >
            <Icon name="panel" size={16} />
          </button>
        </div>
      </aside>
      <main className="main">
        <AppTopbar />
        <div className="page-body">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/fit" element={<FitPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/estimate" element={<EstimatePage />} />
          <Route path="/new" element={<NewJobPage />} />
          <Route path="/new/quantize" element={<NewJobPage />} />
          <Route path="/new/eval" element={<NewJobPage />} />
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
        </div>
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
