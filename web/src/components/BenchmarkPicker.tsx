import { useEffect, useMemo, useState } from "react";
import { api, BenchmarkCatalog, BenchmarkMeta } from "../api";
import { useAsync } from "../hooks";

type Runner = "offline" | "evalscope";

interface Props {
  runner: Runner;
  value: string[];
  onChange: (v: string[]) => void;
  /** Optional per-run sample cap; hidden when omitted. */
  limit?: number;
  onLimit?: (v: number) => void;
  /** Show the named suites from eval_suites.yaml (offline runner only). */
  suites?: boolean;
  title?: string;
  testid?: string;
}

const GROUP_ORDER = ["快速", "基础语言建模", "冒烟检查", "知识与推理", "知识", "推理", "数学", "代码", "中文", "长上下文", "指令遵循", "问答", "真实性", "多模态", "Agent", "工具调用", "对话竞技", "信息抽取", "医学", "其他"];

function runnerItems(catalog: BenchmarkCatalog | null, runner: Runner): BenchmarkMeta[] {
  if (!catalog) return [];
  return runner === "offline" ? [...catalog.benchmarks, ...(catalog.native ?? [])] : catalog.evalscope;
}

function info(meta: BenchmarkMeta | undefined) {
  if (!meta) return null;
  const sample = meta.sample_count != null ? `${meta.sample_count} 条` : meta.runner === "evalscope" ? "由 EvalScope 下载" : "—";
  return (
    <div className="benchmark-info" data-testid="benchmark-info">
      <div className="benchmark-info-head">
        <div>
          <b>{meta.label}</b>
          <span className="mono muted small"> {meta.name}</span>
        </div>
        <span className={`badge ${meta.offline ? "good" : "accent"}`}>
          {meta.runner === "native" ? "原生 · 需 datasets" : meta.offline ? "离线可跑" : "需 EvalScope + 服务"}
        </span>
      </div>
      {meta.description && <p className="small muted">{meta.description}</p>}
      <div className="benchmark-facts">
        <span><small>类别</small><b>{meta.group ?? "其他"}</b></span>
        <span><small>指标</small><b className="mono">{(meta.metrics ?? []).join(" / ") || "—"}</b></span>
        <span><small>样本</small><b>{sample}</b></span>
        <span><small>语言</small><b>{(meta.language ?? []).join(" / ") || "—"}</b></span>
        <span><small>Runner</small><b className="mono">{meta.runner ?? "offline"}</b></span>
      </div>
      <div className="benchmark-tags">
        {(meta.tags ?? []).map((t) => <span key={t} className="badge">{t}</span>)}
        {meta.docs && <a className="small" href={meta.docs} target="_blank" rel="noreferrer">数据集文档 ↗</a>}
      </div>
    </div>
  );
}

/** Grouped, metadata-driven evaluation dataset chooser.
 *
 * The visual model mirrors EvalScope: pick a suite/category, pick one dataset, read its task type,
 * metrics and sample size, then add it to the run.  Multi-dataset runs stay as a compact list instead
 * of a wall of chips. */
export function BenchmarkPicker({ runner, value, onChange, limit, onLimit, suites, title, testid = "benchmark" }: Props) {
  const q = useAsync(api.benchmarks, []);
  const items = runnerItems(q.data, runner);
  const [group, setGroup] = useState("全部");
  const [needle, setNeedle] = useState("");
  const [choice, setChoice] = useState("");
  const [suite, setSuite] = useState("");

  const groups = useMemo(() => {
    const present = [...new Set(items.map((x) => x.group ?? "其他"))];
    return GROUP_ORDER.filter((g) => present.includes(g)).concat(present.filter((g) => !GROUP_ORDER.includes(g)));
  }, [items]);

  const shown = useMemo(() => {
    const n = needle.trim().toLowerCase();
    return items
      .filter((x) => group === "全部" || (x.group ?? "其他") === group)
      .filter((x) => !n || `${x.name} ${x.label} ${(x.tags ?? []).join(" ")}`.toLowerCase().includes(n));
  }, [items, group, needle]);

  useEffect(() => {
    if (!choice || !shown.some((x) => x.name === choice)) setChoice(shown[0]?.name ?? "");
  }, [shown, choice]);
  useEffect(() => {
    if (runner !== "offline" || !suites || !q.data || value.length) return;
    const quick = q.data.suites?.quick;
    if (quick?.length) {
      setSuite("quick");
      onChange(quick);
    }
  }, [q.data, runner, suites]); // eslint-disable-line react-hooks/exhaustive-deps

  const current = items.find((x) => x.name === choice);
  const add = () => {
    if (choice && !value.includes(choice)) onChange([...value, choice]);
  };
  const applySuite = (name: string) => {
    setSuite(name);
    if (name && q.data?.suites?.[name]) onChange(q.data.suites[name]);
    else if (!name) onChange([]);
  };

  return (
    <div className="benchmark-picker" data-testid={testid}>
      <div className="benchmark-picker-head">
        <div>
          <b>{title ?? (runner === "offline" ? "离线评测数据集" : "EvalScope 数据集")}</b>
          <p className="small muted">
            {runner === "offline"
              ? "直接加载 checkpoint 评分，不需要部署；适合量化前后快速对比。"
              : "需要 EvalScope CLI 和已部署的模型服务；数据集由 EvalScope 自动下载。"}
          </p>
        </div>
        {limit != null && onLimit && (
          <label className="field" style={{ margin: 0, width: 110 }}>
            <span className="lbl">样本上限</span>
            <input type="number" min={1} value={limit} onChange={(e) => onLimit(Math.max(1, Number(e.target.value)))} data-testid={`${testid}-limit`} />
          </label>
        )}
      </div>

      {suites && runner === "offline" && (
        <div className="benchmark-row">
          <label className="field grow">
            <span className="lbl">评测套件</span>
            <select value={suite} onChange={(e) => applySuite(e.target.value)} data-testid={`${testid}-suite`}>
              <option value="">自定义</option>
              {Object.entries(q.data?.suites ?? {}).map(([name, list]) => (
                <option key={name} value={name}>{name === "quick" ? "快速检查" : name === "standard" ? "标准评测" : name}（{list.join(" + ")}）</option>
              ))}
            </select>
          </label>
          <label className="field grow">
            <span className="lbl">数据集类别</span>
            <select value={group} onChange={(e) => setGroup(e.target.value)} data-testid={`${testid}-group`}>
              <option value="全部">全部类别</option>
              {groups.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </label>
        </div>
      )}

      <div className="benchmark-row">
        <label className="field grow">
          <span className="lbl">数据集</span>
          <select value={choice} onChange={(e) => setChoice(e.target.value)} data-testid={`${testid}-select`}>
            {shown.map((x) => (
              <option key={x.name} value={x.name}>
                {x.label} · {x.name}{x.sample_count != null ? ` · ${x.sample_count} 条` : ""}
              </option>
            ))}
          </select>
        </label>
        <div className="field" style={{ justifyContent: "flex-end" }}>
          <button className="secondary" onClick={add} disabled={!choice || value.includes(choice)} data-testid={`${testid}-add`}>
            + 加入
          </button>
        </div>
      </div>

      {runner === "evalscope" && items.length > 0 && (
        <input className="benchmark-search" value={needle} onChange={(e) => setNeedle(e.target.value)}
          placeholder="筛选数据集（名称、标签）…" data-testid={`${testid}-search`} />
      )}

      {q.error && <div className="error-box">{q.error}</div>}
      {!q.loading && items.length === 0 && <p className="muted small">服务器没有返回这个 runner 的评测目录。</p>}

      {value.length > 0 && (
        <div className="benchmark-selected" data-testid={`${testid}-selected`}>
          {value.map((name) => {
            const m = items.find((x) => x.name === name);
            return (
              <span key={name} className="benchmark-pill">
                {m?.label ?? name}
                <button aria-label={`移除 ${name}`} onClick={() => onChange(value.filter((x) => x !== name))}>×</button>
              </span>
            );
          })}
        </div>
      )}
      {info(current)}
    </div>
  );
}
