// 总览：一眼看到 GPU 池、正在跑的作业、最近结果和推理后端环境，再从这里进到各功能页。
import { Link } from "react-router-dom";
import { api, fmtBytes, JobListItem } from "../api";
import { useAsync } from "../hooks";
import { StateBadge } from "./Jobs";
import { Tile } from "./Estimate";

const KIND: Record<string, string> = { llm: "LLM", vlm: "VLM", vla: "VLA", llm_deploy: "部署评测" };
const short = (n: string) => n.replace(/^NVIDIA\s+/, "").replace(/^GeForce\s+/, "");

export default function HomePage() {
  const hw = useAsync(api.hardware, []);
  const jobs = useAsync(api.jobs, []);
  const envs = useAsync(() => api.backendEnvs(false), []);
  const list: JobListItem[] = jobs.data ?? [];
  const running = list.filter((j) => j.state === "RUNNING" || j.state === "QUEUED");
  const dayAgo = Date.now() / 1000 - 86400;
  const recent = list.filter((j) => (j.last_ts ?? 0) > dayAgo);
  const pool = hw.data?.job_gpus.pool ?? [];
  const gpus = (hw.data?.devices ?? []).filter((d) => d.kind === "gpu");
  const poolDevs = pool.map((p) => ({ ...p, dev: gpus.find((d) => d.index === p.gpu) }));
  const free = pool.filter((p) => !p.job).length;
  const installed = (envs.data?.backends ?? []).filter((b) => b.installed);
  return (
    <>
      <div className="page-head">
        <div>
          <h1>总览</h1>
          <p className="subtitle">模型量化 → 评测 → 部署，全部作业与资源一页看完。</p>
        </div>
        <div className="form-row" style={{ margin: 0 }}>
          <Link to="/new"><button data-testid="home-new-job">＋ 新建评估作业</button></Link>
          <Link to="/fit"><button className="secondary">部署预评估</button></Link>
        </div>
      </div>
      <div className="tiles" data-testid="home-tiles">
        <Tile label="GPU 池" value={pool.length ? `${free} / ${pool.length} 空闲` : "–"} hint={gpus[0] ? short(gpus[0].name) : "未探测到 GPU"} />
        <Tile label="运行中 / 排队" value={`${running.length}`} hint={running[0] ? `${running[0].name ?? running[0].job_id} · ${running[0].stage ?? ""}` : "空闲"} />
        <Tile label="24 小时内" value={`${recent.filter((j) => j.state === "SUCCEEDED").length} 成功 / ${recent.filter((j) => j.state === "FAILED").length} 失败`} hint={`共 ${list.length} 个作业`} />
        <Tile label="推理后端" value={envs.data ? `${installed.length} / ${envs.data.backends.length} 可用` : "–"} hint={installed.map((b) => b.id).join(" · ") || "未安装"} />
      </div>
      <div className="split">
        <div>
          <div className="card table-wrap">
            <h2>最近作业</h2>
            <table data-testid="home-jobs">
              <thead>
                <tr>
                  <th>作业</th>
                  <th>类型</th>
                  <th>状态</th>
                  <th>阶段</th>
                  <th>配方</th>
                  <th className="num">耗时</th>
                </tr>
              </thead>
              <tbody>
                {list.slice(0, 8).map((j) => (
                  <tr key={j.job_id}>
                    <td>
                      <Link to={`/jobs/${j.job_id}`}>{j.name ?? j.job_id}</Link>
                      <div className="small muted mono">{j.job_id}{j.parent_job ? ` · 派生自 ${j.parent_job.slice(-6)}` : ""}</div>
                    </td>
                    <td><span className="badge">{KIND[j.kind ?? "vla"] ?? j.kind}</span></td>
                    <td><StateBadge state={j.state} /></td>
                    <td className="small">{j.stage ?? "–"}</td>
                    <td className="small">{(j.recipes ?? []).join(", ")}</td>
                    <td className="num">{j.started_ts && j.last_ts ? `${Math.round(j.last_ts - j.started_ts)} s` : "–"}</td>
                  </tr>
                ))}
                {jobs.data && list.length === 0 && (
                  <tr><td colSpan={6} className="empty">还没有作业，<Link to="/new">新建一个</Link></td></tr>
                )}
              </tbody>
            </table>
            {list.length > 8 && <p className="small"><Link to="/jobs">全部 {list.length} 个作业 →</Link></p>}
            {jobs.error && <div className="error-box">{jobs.error}</div>}
          </div>
        </div>
        <div>
          <div className="card">
            <h2>GPU 池</h2>
            {hw.error && <div className="error-box">{hw.error}</div>}
            {pool.length === 0 && hw.data && <p className="muted small">服务没有 GPU 池（--gpus）</p>}
            {poolDevs.map((p) => (
              <div key={p.gpu} className="form-row" style={{ alignItems: "center", margin: "6px 0" }} data-testid={`home-gpu-${p.gpu}`}>
                <span className={`status ${p.job ? "running" : "good"}`}>{p.job ? "占用" : "空闲"}</span>
                <span className="mono">GPU {p.gpu}</span>
                <span className="small">{p.dev ? short(p.dev.name) : ""}</span>
                <span className="small muted">{p.dev ? `${fmtBytes(p.dev.mem_free_bytes)} 空闲 / ${fmtBytes(p.dev.mem_total_bytes)}` : ""}</span>
                {p.job && <Link to={`/jobs/${p.job}`} className="mono small">{p.job}</Link>}
              </div>
            ))}
            {hw.data && hw.data.job_gpus.queued > 0 && <p className="small muted">排队 {hw.data.job_gpus.queued} 个</p>}
            <p className="small"><Link to="/hardware">硬件与后端 →</Link></p>
          </div>
          <div className="card">
            <h2>推理后端环境</h2>
            {(envs.data?.backends ?? []).map((b) => (
              <div key={b.id} className="form-row" style={{ alignItems: "center", margin: "6px 0" }}>
                <b style={{ width: 80 }}>{{ vllm: "vLLM", sglang: "SGLang", llamacpp: "llama.cpp" }[b.id] ?? b.id}</b>
                {b.installed ? <span className="mono small">{b.version ?? "已安装"}</span> : <span className="muted small">未安装</span>}
                <span className="muted small">{b.source === "env" ? "环境变量指定" : b.source_build ? "源码构建" : b.source === "managed" ? "托管环境" : ""}</span>
              </div>
            ))}
            {envs.error && <div className="error-box">{envs.error}</div>}
            <p className="small"><Link to="/hardware">安装 / 更新 →</Link></p>
          </div>
        </div>
      </div>
    </>
  );
}
