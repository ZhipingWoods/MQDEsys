import { ReactNode } from "react";
import { Link } from "react-router-dom";
import { BackendEnv, BackendId } from "../api";

/** One card per inference backend, with what it serves, whether it is installed here, and a switch. */
interface Info {
  name: string;
  serves: string;
  where: string;
  measures: string;
}
const INFO: Record<BackendId, Info> = {
  llamacpp: {
    name: "llama.cpp",
    serves: "GGUF（Q8_0 / Q5_0 / Q4_0 / IQ4_NL / Q4_K / Q2_K）",
    where: "CPU，或 Metal / CUDA 卸载",
    measures: "llama-perplexity 的 PPL、pp / tg 速度，llama-server 并发压测",
  },
  vllm: {
    name: "vLLM",
    serves: "AWQ · GPTQ · compressed-tensors（W4A16 / FP8）",
    where: "GPU",
    measures: "服务端 PPL、TTFT / TPOT / 吞吐、并发扫描、lm-eval 任务精度",
  },
  sglang: {
    name: "SGLang",
    serves: "AWQ · GPTQ · compressed-tensors（W4A16 / FP8）",
    where: "GPU（独立的 SGLang 环境）",
    measures: "服务端 PPL、TTFT / TPOT / 吞吐、并发扫描",
  },
};
export const BACKEND_ORDER: BackendId[] = ["llamacpp", "vllm", "sglang"];

interface Props {
  env: BackendEnv[] | null; // null while loading (or when the server could not say)
  sel: Record<BackendId, boolean>;
  onToggle: (id: BackendId, on: boolean) => void;
  /** which backends can be chosen at all (a GGUF artifact can only go to llama.cpp, an HF one only to vLLM / SGLang) */
  allowed?: BackendId[];
  /** "single": choosing one deselects the others (a reused artifact is served by exactly one backend) */
  single?: boolean;
  disabled?: boolean;
  disabledNote?: string;
  /** a backend's own options, shown inside its card while it is switched on */
  options?: Partial<Record<BackendId, ReactNode>>;
}

export function BackendCards({ env, sel, onToggle, allowed = BACKEND_ORDER, single = false, disabled = false, disabledNote, options = {} }: Props) {
  const nothing = !BACKEND_ORDER.some((b) => sel[b]);
  return (
    <div data-testid="backend-cards">
      <div className="backend-grid">
        {BACKEND_ORDER.map((id) => {
          const info = INFO[id];
          const e = env?.find((x) => x.id === id);
          const can = allowed.includes(id);
          const on = sel[id] && can;
          const missing = e != null && !e.installed;
          return (
            <div
              key={id}
              className={`backend-card ${on ? "on" : ""} ${can && !disabled ? "" : "off"}`}
              data-testid={`backend-card-${id}`}
              data-on={on ? "1" : "0"}
              data-installed={e == null ? "?" : e.installed ? "1" : "0"}
            >
              <label className="backend-head">
                <input
                  type={single ? "radio" : "checkbox"}
                  name={single ? "backend" : undefined}
                  checked={on}
                  disabled={!can || disabled}
                  onChange={(ev) => onToggle(id, ev.target.checked)}
                  data-testid={`backend-toggle-${id}`}
                />
                <span className="backend-name">{info.name}</span>
                <span className={`backend-badge ${e == null ? "" : e.installed ? "ok" : "warn"}`} data-testid={`backend-status-${id}`}>
                  {e == null ? "状态未知" : e.installed ? `已安装${e.version ? ` ${e.version}` : ""}` : "未安装"}
                </span>
              </label>
              <dl className="backend-facts">
                <dt>服务</dt>
                <dd>{info.serves}</dd>
                <dt>运行在</dt>
                <dd>{info.where}</dd>
                <dt>测量</dt>
                <dd>{info.measures}</dd>
              </dl>
              {!can && <p className="note muted">{single ? "这个产物的格式不能用这个后端服务。" : "当前设置下不可选。"}</p>}
              {on && missing && (
                <p className="note warn" data-testid={`backend-missing-${id}`}>
                  这台服务器上没有 {info.name}：该阶段会被跳过。<Link to="/hardware">到「硬件与后端」安装</Link>
                </p>
              )}
              {on && options[id] && <div className="backend-options">{options[id]}</div>}
            </div>
          );
        })}
      </div>
      {nothing && !disabled && (
        <p className="note muted" data-testid="backend-none">
          还没有选部署后端：只评估量化误差，不做真实部署。勾选一个或几个来真实部署并对比。
        </p>
      )}
      {disabled && disabledNote && <p className="note muted">{disabledNote}</p>}
    </div>
  );
}
