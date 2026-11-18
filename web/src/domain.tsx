// Global problem domain (LLM / VLM / VLA / World Model / Embodied WM). The framework puts the domain switch in the
// navigation bar: every page that shows models, jobs or results follows it, so it lives in a context, not in one page.
import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from "react";
import { api, Domain, LibraryModel } from "./api";
import { DEFAULT_EWM, DEFAULT_LLM, DEFAULT_MODEL, DEFAULT_VLM, DEFAULT_WORLD } from "./hooks";

export const DOMAIN_SHORT: Record<string, string> = { llm: "LLM", vlm: "VLM", vla: "VLA", world: "WM", ewm: "EWM" };
// 解析族的展示名。族 id 来自 data/model_families/*.yaml 的 `family`，这里只是给它一个人能看懂的名字。
const FAMILY_LABEL: Record<string, string> = {
  decoder_llm: "Decoder LLM",
  generic_decoder: "通用 Decoder",
  qwen_vl: "Qwen-VL",
  llava: "LLaVA",
  internvl: "InternVL",
  glm4v: "GLM-4V",
  llama4: "Llama 4",
  idefics3: "Idefics3",
  minicpmo: "MiniCPM-o",
};
export const familyLabel = (f: string): string => FAMILY_LABEL[f] ?? f;
const FAMILY_COVERAGE: Record<string, string> = {
  decoder_llm: "Llama / Qwen2-3 / Mistral / Mixtral / Gemma / Phi / DeepSeek / GLM4 / InternLM2 / Falcon / GPT-2 / OPT 等",
  generic_decoder: "没有专门解析器时使用的标准 decoder 布局",
  qwen_vl: "Qwen2-VL / Qwen2.5-VL / Qwen3-VL",
  llama4: "Llama 4 Scout / Maverick",
};
export const familyCoverage = (f: string): string => FAMILY_COVERAGE[f] ?? f;
// the registry is data (data/domains/*.yaml); this is only what the switch shows before the server answers
const FALLBACK: Domain[] = ["llm", "vlm", "vla", "world", "ewm"].map((id) => ({
  id, display_name: DOMAIN_SHORT[id], model_families: [], applicable_recipes: [], classic_algorithms: [], default_eval_dataset: "",
  pipeline_status: "implemented", deploy_backends: [], deploy_note: "",
}));

/** The domain a job belongs to: a deploy job serves an LLM export; reports older than the domain field are VLA. */
export const jobDomain = (kind?: string): string =>
  kind === "llm" || kind === "llm_deploy" ? "llm" : kind === "world" || kind === "ewm" || kind === "vlm" || kind === "vla" ? kind : "vla";
/** Does a model in the local library belong to a domain (an omni model is a vision-language model as far as the
 * pipelines go)? */
export const modelInDomain = (m: Pick<LibraryModel, "modality">, d: string): boolean => m.modality === d || (d === "vlm" && m.modality === "omni");
/** The model a page starts from in a domain ("" = none ships with the server for it). */
export const domainDefaultModel = (d: string): string =>
  ({ llm: DEFAULT_LLM, vlm: DEFAULT_VLM, vla: DEFAULT_MODEL, world: DEFAULT_WORLD, ewm: DEFAULT_EWM })[d] ?? "";

interface Ctx {
  domain: string;
  setDomain: (d: string) => void;
  domains: Domain[];
  current: Domain | undefined;
}
const DomainCtx = createContext<Ctx>({ domain: "llm", setDomain: () => {}, domains: FALLBACK, current: FALLBACK[0] });

export function DomainProvider({ children }: { children: ReactNode }) {
  const [domain, setDomainState] = useState<string>(() => {
    try {
      return localStorage.getItem("mqde-domain") ?? "llm";
    } catch {
      return "llm";
    }
  });
  const [domains, setDomains] = useState<Domain[]>(FALLBACK);
  useEffect(() => {
    api.domains().then((d) => d.length && setDomains(d), () => {});
  }, []);
  const value = useMemo<Ctx>(
    () => ({
      domain,
      setDomain: (d: string) => {
        setDomainState(d);
        try {
          localStorage.setItem("mqde-domain", d);
        } catch {
          /* storage unavailable: holds for this session */
        }
      },
      domains,
      current: domains.find((x) => x.id === domain),
    }),
    [domain, domains],
  );
  return <DomainCtx.Provider value={value}>{children}</DomainCtx.Provider>;
}

export const useDomain = () => useContext(DomainCtx);

/** The switch itself (sidebar): one chip per domain; a domain without a pipeline is marked, not hidden. */
export function DomainSwitcher({ collapsed }: { collapsed: boolean }) {
  const { domain, setDomain, domains } = useDomain();
  return (
    <div className={`domain-switch ${collapsed ? "vertical" : ""}`} role="radiogroup" aria-label="领域" data-testid="domain-switch">
      {!collapsed && <div className="nav-group" style={{ padding: "0 4px 6px" }}>领域</div>}
      <div className="domain-chips">
        {domains.map((d) => {
          const pending = d.pipeline_status !== "implemented";
          return (
            <button
              key={d.id}
              role="radio"
              aria-checked={domain === d.id}
              className={`domain-chip ${domain === d.id ? "on" : ""} ${pending ? "pending" : ""}`}
              onClick={() => setDomain(d.id)}
              data-testid={`nav-domain-${d.id}`}
              data-tip={`${d.display_name}${pending ? "（尚无量化流水线）" : ""}`}
            >
              {DOMAIN_SHORT[d.id] ?? d.id.toUpperCase()}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** A one-line notice for pages viewed in a domain that is unavailable. */
export function DomainPendingNote() {
  const { current } = useDomain();
  if (!current || current.pipeline_status === "implemented") return null;
  return (
    <div className="note warn" data-testid="domain-pending" style={{ marginBottom: 12 }}>
      {current.display_name}：当前暂不支持提交作业，你仍可以浏览相关模型、硬件和评测信息。
    </div>
  );
}
