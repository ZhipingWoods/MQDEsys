// "推理后端环境": installed version of vLLM / SGLang / llama.cpp, the newest one available, and an install / update
// button that names the version it would install (scripts/setup_backend.py runs behind /api/v1/backend-envs).
import { useCallback, useEffect, useRef, useState } from "react";
import { api, BackendEnv, BackendId, BackendInstall } from "../api";

const NAMES: Record<BackendId, string> = { vllm: "vLLM", sglang: "SGLang", llamacpp: "llama.cpp" };
const SOURCE: Record<string, string> = { env: "环境变量指定", managed: "托管环境", path: "系统 PATH", legacy: "本地构建" };

/** `0.31.0` -> `v0.31.0`; llama.cpp build tags (`b11565`) are shown as they are. */
export const fmtVersion = (v: string) => (/^b\d+$/.test(v) ? v : `v${v}`);

export function actionFor(e: BackendEnv): { label: string; disabled: boolean; version?: string; hint?: string } {
  // An MQDE_*_BIN variable outranks the managed environment: an install would succeed and then never be used.
  if (e.source === "env")
    return { label: "由环境变量指定", disabled: true, hint: "该后端由 MQDE_VLLM_BIN / MQDE_SGLANG_PYTHON / MQDE_LLAMA_SERVER_BIN 指定，优先级高于托管环境，在这里更新不会生效。取消该环境变量后重启服务即可使用本页的安装与更新。" };
  if (e.source_build)
    return { label: `重新构建（源码 ${fmtVersion(e.version ?? "")}）`, disabled: false,
      hint: "当前 llama.cpp 是从 3rdpart/llama.cpp 源码构建的（官方 CUDA 发行包在本机的 glibc 上不能运行）。这里不会用发行包覆盖它；点击按子模块当前版本重新构建。" };
  if (e.check_error) return { label: "无法检查更新", disabled: true };
  if (!e.latest) return { label: e.installed ? "检查更新中…" : "安装", disabled: !e.installed ? false : true };
  if (!e.installed) return { label: `安装 ${fmtVersion(e.latest)}`, disabled: false, version: e.latest };
  if (e.update_available) return { label: `更新到 ${fmtVersion(e.latest)}`, disabled: false, version: e.latest };
  return { label: `已是最新 ${fmtVersion(e.version ?? e.latest)}`, disabled: true };
}

export default function BackendEnvsCard() {
  const [rows, setRows] = useState<BackendEnv[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [install, setInstall] = useState<BackendInstall | null>(null);
  const alive = useRef(true);
  useEffect(() => () => void (alive.current = false), []);

  const load = useCallback(async (check: boolean) => {
    if (check) setChecking(true);
    try {
      const r = await api.backendEnvs(check);
      if (!alive.current) return;
      setErr(null);
      setRows((old) => r.backends.map((b) => (check ? b : { ...b, ...pickLatest(old?.find((o) => o.id === b.id)) })));
      if (r.installing) setInstall((cur) => cur ?? r.installing);
    } catch (e) {
      if (alive.current) setErr((e as Error).message);
    } finally {
      if (alive.current && check) setChecking(false);
    }
  }, []);

  // installed versions first (fast, local), then the newest versions (network) without blocking the table
  useEffect(() => {
    load(false).then(() => load(true));
  }, [load]);

  // follow a running install
  useEffect(() => {
    if (!install || install.state !== "running") return;
    const t = setInterval(() => {
      api
        .backendInstall(install.id)
        .then((s) => {
          if (!alive.current) return;
          setInstall(s);
          if (s.state !== "running") load(false).then(() => load(true));
        })
        .catch(() => {});
    }, 1500);
    return () => clearInterval(t);
  }, [install, load]);

  const start = async (e: BackendEnv, version?: string) => {
    const what = `${NAMES[e.id]}${version ? " " + fmtVersion(version) : ""}`;
    if (!window.confirm(`${e.installed ? "更新" : "安装"} ${what}？\n会下载依赖（vLLM / SGLang 可达数 GB）并修改 ${e.envs_dir} 下的环境，期间不要关闭服务。`)) return;
    try {
      const { install_id } = await api.installBackend(e.id, version);
      setErr(null);
      setInstall(await api.backendInstall(install_id));
    } catch (ex) {
      setErr((ex as Error).message);
    }
  };

  const busy = install?.state === "running";
  return (
    <div className="card" data-testid="backend-envs-card">
      <div className="card-head">
        <div>
          <h2>推理后端环境</h2>
          <p className="note" style={{ margin: 0 }}>vLLM / SGLang / llama.cpp 各自独立安装；更新后对之后提交的作业立即生效。</p>
        </div>
        <button className="secondary" disabled={checking || busy} onClick={() => load(true)} data-testid="backend-env-check">
          {checking ? "检查中…" : "检查更新"}
        </button>
      </div>
      {err && <div className="error-box" data-testid="backend-env-error">{err}</div>}
      <div className="backend-env-grid">
        {rows?.map((e) => {
          const a = actionFor(e);
          return (
            <article key={e.id} className={`backend-env-card ${e.installed ? "installed" : "missing"} ${e.id}`} data-testid={`backend-env-${e.id}`}>
              <div className="backend-env-head">
                <span className={`backend-env-logo ${e.id}`} aria-hidden="true">{e.id === "vllm" ? "V" : e.id === "sglang" ? "S" : "L"}</span>
                <div className="backend-env-title">
                  <b>{NAMES[e.id]}</b>
                  <small>{e.source_build ? "源码构建（3rdpart/llama.cpp）" : e.source ? SOURCE[e.source] : "未安装"}</small>
                </div>
                <span className={`status ${e.installed ? "good" : "muted"}`}>{e.installed ? "可用" : "未安装"}</span>
              </div>
              <div className="backend-env-versions">
                <div>
                  <span>当前版本</span>
                  <b className="mono" data-testid={`backend-env-version-${e.id}`} title={e.path ?? ""}>
                    {e.installed ? (e.version ? fmtVersion(e.version) : "未知") : "—"}
                  </b>
                </div>
                <div>
                  <span>最新版本</span>
                  <b className="mono" data-testid={`backend-env-latest-${e.id}`} title={e.source_build ? "源码构建的版本随子模块走，不跟官方发行包比较" : e.check_error ?? ""}>
                    {e.source_build ? "随子模块" : e.latest ? fmtVersion(e.latest) : e.check_error ? "无法获取" : "…"}
                  </b>
                </div>
              </div>
              <button className={e.installed ? "secondary" : ""} disabled={a.disabled || busy} title={a.hint} onClick={() => start(e, a.version)} data-testid={`backend-env-action-${e.id}`}>
                {a.label}
              </button>
            </article>
          );
        })}
        {!rows && <div className="muted small">读取后端环境中…</div>}
      </div>
      {install && (
        <div style={{ marginTop: 12 }} data-testid="backend-install">
          <p className="small">
            <b>
              {NAMES[install.backend]} {install.version ? fmtVersion(install.version) : ""}
            </b>{" "}
            <span className={`status ${install.state === "running" ? "running" : install.state === "succeeded" ? "good" : "bad"}`} data-testid="backend-install-state">
              {install.state === "running" ? "安装中" : install.state === "succeeded" ? "完成" : `失败（退出码 ${install.exit_code}）`}
            </span>
          </p>
          <pre className="log" style={{ height: "auto", maxHeight: 220 }} data-testid="backend-install-log">
            {install.log_tail.slice(-14).join("\n") || "…"}
          </pre>
        </div>
      )}
    </div>
  );
}

// the newest-version fields survive a refresh of the installed versions (which does not look them up again)
function pickLatest(o?: BackendEnv): Partial<BackendEnv> {
  return o ? { latest: o.latest, update_available: o.update_available, check_error: o.check_error } : {};
}
