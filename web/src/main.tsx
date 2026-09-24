import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";
import "./styles.css";
import EstimatePage from "./pages/Estimate";
import NewJobPage from "./pages/NewJob";
import JobsPage from "./pages/Jobs";
import JobPage from "./pages/Job";
import HardwarePage from "./pages/Hardware";
import ComparePage from "./pages/Compare";

function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">MQDEsys</div>
        <div className="brand-sub">模型量化部署评估系统</div>
        <nav className="nav">
          <NavLink to="/estimate">资源预估</NavLink>
          <NavLink to="/new">新建评估作业</NavLink>
          <NavLink to="/jobs">作业与执行过程</NavLink>
          <NavLink to="/compare">配方对比</NavLink>
          <NavLink to="/hardware">硬件</NavLink>
        </nav>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/estimate" replace />} />
          <Route path="/estimate" element={<EstimatePage />} />
          <Route path="/new" element={<NewJobPage />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/jobs/:id" element={<JobPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/hardware" element={<HardwarePage />} />
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
