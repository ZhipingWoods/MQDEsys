import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { deriveFromJob, DeriveStep, DERIVE_STEPS, Rule } from "../derive";
import { Advice, AdviceObjective, api, CatalogModel, DownloadStatus, fmtBytes, probeNote, RuntimeParams , BackendId } from "../api";
import { BackendCards } from "../components/BackendCards";
import { BenchmarkPicker } from "../components/BenchmarkPicker";
import { DatasetPicker } from "../components/DatasetPicker";
import { HardwarePicker } from "../components/HardwarePicker";
import { DEFAULT_DATASET, DEFAULT_EWM, DEFAULT_LLM, DEFAULT_MODEL, DEFAULT_VLM, DEFAULT_WORLD, useAsync, useDebounced } from "../hooks";
import { Meter } from "./Fit";
import { useDomain } from "../domain";

const LIVE_RUNTIME: RuntimeParams = { concurrency: 1, images_per_step: 2, text_tokens: 42, image_resolution: 448, gpu_memory_utilization: 0.9, extra_modalities: [] };

// Every registered domain has a real pipeline now (T-1304/T-1005). A domain with
// pipeline_status != implemented still shows the blocked notice instead of a form.
const LLM_LIKE = new Set(["llm"]);
type Kind = "llm" | "vlm" | "vla" | "world" | "ewm";
// LLM 作业的三种入口：全流程、只量化（可导出权重）、只评测已有量化模型。其它领域仍是原来的全流程。
type JobMode = "full" | "quantize" | "eval";
// what the wizard's form looks like for a domain: LLM (PPL), VLM (image+question -> answer),
// VLA (actions), world (video clip) or EWM (action-conditioned video clip).
const kindOf = (d: string): Kind =>
  LLM_LIKE.has(d) ? "llm" : d === "vlm" ? "vlm" : d === "world" ? "world" : d === "ewm" ? "ewm" : "vla";

const PASTE_MAX = 2_000_000; // characters; the job reads at most 50 million, and a form is not the place for more than this

/** Pasted calibration text -> samples: a blank line separates them; each is trimmed and the empty ones are dropped. */
export function splitSamples(text: string): string[] {
  return text.split(/\n\s*\n/).map((t) => t.trim()).filter((t) => t !== "");
}

export default function NewJobPage() {
  const nav = useNavigate();
  const { pathname } = useLocation();
  // Dedicated entry points (/new/quantize, /new/eval) preset the wizard mode without changing the
  // existing /new?mode=all single-page form used by older e2e tests.
  const routeMode: JobMode = pathname.replace(/\/+$/, "").endsWith("/new/eval")
    ? "eval"
    : pathname.replace(/\/+$/, "").endsWith("/new/quantize")
      ? "quantize"
      : "full";
  const domainsQ = useAsync(api.domains, []);
  const recipes = useAsync(api.recipes, []);
  const algos = useAsync(api.algorithms, []);
  const [algorithm, setAlgorithm] = useState("recipe");
  const profiles = useAsync(api.hardwareProfiles, []);
  const localHardware = useAsync(api.hardware, []);
  const cfg = useAsync(api.config, []);
  // the domain is the global one from the navigation bar (domain.tsx); the wizard's own step-1 tabs set the same value
  const { domain, setDomain } = useDomain();
  const [name, setName] = useState("qwen-wikitext-ppl");
  const [model, setModel] = useState(DEFAULT_LLM);
  const [modelMode, setModelMode] = useState<"custom" | "catalog">("custom");
  const [picked, setPicked] = useState<string[]>(["w4a16", "w4a8", "w2a16"]);
  const [noQuant, setNoQuant] = useState(false);
  // T-1304/T-1306: "仅量化不评测" (evaluation.skip) -- now a real backend capability (llm_job.py/vla_job.py's
  // quantize_only branches, planner::validate_jobspec rejects combining it with export/vllm/llamacpp/
  // sensitivity/logits_probe/attribution). The job-construction logic below strips those fields outright
  // when this is on, matching the server's own rejection list exactly, so the UI can never produce a job
  // the server would 400.
  const [quantizeOnly, setQuantizeOnly] = useState(false);
  const [jobMode, setJobMode] = useState<JobMode>(routeMode);
  // 仅评测：直接加载已有的量化 checkpoint（本地目录或 GGUF 文件），不重新量化。
  const [quantizedPath, setQuantizedPath] = useState("");
  const [quantFormat, setQuantFormat] = useState("auto");
  // T-1304: advanced per-job group_size override (quant.group_size, read by llm_job.py/vla_job.py's
  // group_size_override). "" = not overridden, so an untouched job omits the field entirely and matches
  // every picked recipe's own data/recipes/*.yaml group_size exactly.
  const [groupSizeOverride, setGroupSizeOverride] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // LLM (wikitext-2 PPL) options
  const [seqLen, setSeqLen] = useState(512);
  const [windows, setWindows] = useState(40);
  const [actStatic, setActStatic] = useState("");
  const [kv, setKv] = useState("int8");
  const [kvMethod, setKvMethod] = useState("kivi");
  // T-1304/T-1306: quant_device/eval_device split (real backend: schemas/job_spec.schema.json's
  // quant_device/evaluation.device, consumed by llm_job.py and resolved by JobRunner against the GPU
  // pool). Omitted from the submitted job when equal -- an unsplit job looks exactly like it did before
  // this existed (single deploy.device, no new fields).
  const [quantDevice, setQuantDevice] = useState("cpu");
  const [evalDevice, setEvalDevice] = useState("cpu");
  const [keepEnds, setKeepEnds] = useState(false);
  const [downBits, setDownBits] = useState("");
  const [exportFmts, setExportFmts] = useState<string[]>(["gptq"]);
  const [sens, setSens] = useState("");
  const [gguf, setGguf] = useState<string[]>([]);
  // logits probe (KL / Top-1 agreement), vLLM deployment (S11/S12) and lm-eval task accuracy: all LLM-only
  const [probe, setProbe] = useState(false);
  const [probeExact, setProbeExact] = useState(false);
  const [useVllm, setUseVllm] = useState(false);
  const [vllmUtil, setVllmUtil] = useState(0.3);
  // T-0704: SGLang is a second real serving backend, independent of vLLM (both can be on to compare them)
  const [useSglang, setUseSglang] = useState(false);
  const [sglangMem, setSglangMem] = useState(0.3);
  // T-R12: concurrency sweep of the served model (deploy.vllm|sglang.sweep)
  const [useSweep, setUseSweep] = useState(false);
  const [sweepLevels, setSweepLevels] = useState("1,2,4,8,16");
  const [sweepBudget, setSweepBudget] = useState("");
  const levelsOk = /^\s*\d+(\s*,\s*\d+)*\s*$/.test(sweepLevels) && new Set(sweepLevels.split(",").map((x) => Number(x))).size === sweepLevels.split(",").length && sweepLevels.split(",").every((x) => Number(x) >= 1);
  const sweepCfg = useSweep && levelsOk ? { sweep: { levels: sweepLevels.split(",").map((x) => Number(x)), requests_per_level: 16, ...(sweepBudget !== "" && Number(sweepBudget) > 0 ? { max_ttft_p99_ms: Number(sweepBudget) } : {}) } } : {};
  const [useLmEval, setUseLmEval] = useState(false);
  const [lmTasks, setLmTasks] = useState("gsm8k");
  const [lmLimit, setLmLimit] = useState(50);
  const [lmShots, setLmShots] = useState(5);
  // VLA / WM / EWM output actions or video, not tokens. Their native loopback server is enabled
  // independently of the three token-serving backends.
  const [useNative, setUseNative] = useState(false);
  const [nativeConc, setNativeConc] = useState(2);
  const [nativeReq, setNativeReq] = useState(4);
  const lmEvalCfg = useLmEval
    ? { lm_eval: { tasks: lmTasks.split(",").map((t) => t.trim()).filter(Boolean), limit: lmLimit, num_fewshot: lmShots } }
    : {};
  // EvalScope benchmarks (external CLI, optional): dataset ids are passed straight to `evalscope eval`.
  const [useEvalScope, setUseEvalScope] = useState(false);
  const [evalScopeDatasets, setEvalScopeDatasets] = useState<string[]>(["gsm8k", "mmlu_pro"]);
  const [evalScopeLimit, setEvalScopeLimit] = useState(100);
  const [evalScopeBatch, setEvalScopeBatch] = useState(8);
  // Offline benchmarks (eval_job.py): PPL / MCQ / QA scored directly by transformers, no serving backend.
  const [offlineBench, setOfflineBench] = useState<string[]>(["wikitext2_ppl"]);
  const [offlineBenchLimit, setOfflineBenchLimit] = useState(50);
  const [dataset, setDataset] = useState(DEFAULT_DATASET); // VLA / VLM: where calibration samples come from
  const [evalDataset, setEvalDataset] = useState(DEFAULT_DATASET); // VLA / VLM: where evaluation samples come from (may differ)
  const [calibText, setCalibText] = useState("wikitext-2-raw-v1:train"); // LLM: calibration text (built-in, or a server path)
  // LLM: evaluation text (built-in wikitext-2 test by default -- keeps PPL comparable across jobs -- or a server path)
  const [evalTextDataset, setEvalTextDataset] = useState("wikitext-2-raw-v1:test");
  // a typed path the server cannot read blocks submitting; each picker reports whether its value is fine
  const [calibTextOk, setCalibTextOk] = useState(true);
  const [evalTextOk, setEvalTextOk] = useState(true);
  // LLM: text pasted straight into the form instead of a dataset (calibration.custom_samples); a blank line separates samples
  const [usePaste, setUsePaste] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const pasteSamples = splitSamples(pasteText);
  const [calibVisionOk, setCalibVisionOk] = useState(true);
  const [evalVisionOk, setEvalVisionOk] = useState(true);
  const [calibWorldOk, setCalibWorldOk] = useState(true);
  const [evalWorldOk, setEvalWorldOk] = useState(true);
  const [maxNewTokens, setMaxNewTokens] = useState(24); // VLM: longest answer generated per question
  const [frames, setFrames] = useState(4); // world / EWM: frames per clip
  const [quantizeAction, setQuantizeAction] = useState(false); // EWM: also quantize the action encoder
  const [nCalib, setNCalib] = useState(32);
  const [nEval, setNEval] = useState(24);
  const [hw, setHw] = useState("nvidia/rtx4090");
  const [attribution, setAttribution] = useState(true);
  const [action, setAction] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // 作业派生 (/new?from=<job>&step=eval|device|deploy|recipe|all): every field comes from that job's job.yaml
  const [params, setParams] = useSearchParams();
  const fromJob = params.get("from");
  // A real step-by-step wizard (one step on screen, ①…⑥ nav on top) is the default; ?mode=all shows every step on one
  // page, which is also how the older e2e files drive the form.
  const singlePage = params.get("mode") === "all";
  const [step, setStep] = useState<number>(() => ({ eval: 4, device: 4, deploy: 5, recipe: 3 } as Record<string, number>)[params.get("step") ?? ""] ?? 1);
  const show = (n: number) => singlePage || step === n;
  // /new/eval and /new/quantize are real navigation entries. Keep the mode in sync when the
  // user switches between them without unmounting the wizard.
  useEffect(() => {
    setJobMode(routeMode);
  }, [routeMode]);
  // ?hw=<profile-id> comes from a hardware card ("选为部署目标" -> 新建作业). Apply it once the
  // profile list is available; the user can still pick a different card afterwards.
  const appliedHwQuery = useRef<string | null>(null);
  useEffect(() => {
    const h = params.get("hw");
    if (!h || h === appliedHwQuery.current) return;
    if (!profiles.data?.some((p) => p.id === h)) return;
    appliedHwQuery.current = h;
    setHw(h);
  }, [params, profiles.data]);
  const fromStep = (params.get("step") ?? "all") as DeriveStep;
  const [parentJob, setParentJob] = useState<string | null>(null);
  const [inheritedRules, setInheritedRules] = useState<Rule[]>([]);
  const [deriveErr, setDeriveErr] = useState<string | null>(null);
  // B2: serve one of the parent's exported checkpoints instead of quantizing again (mqde.pipeline.deploy_job)
  const [artifactOpts, setArtifactOpts] = useState<ArtifactOpt[]>([]);
  const [artifactKey, setArtifactKey] = useState("");
  const [parentEval, setParentEval] = useState<{ windows: number; seq_len: number } | null>(null); // how the parent scored PPL
  const artifact = artifactOpts.find((a) => a.key === artifactKey)?.artifact ?? null;
  const isGguf = artifact?.format === "gguf";
  const keepParentDeploy = useRef(false); // the backend effect below must not overwrite the parent's own export/deploy choices
  // T-1304: which real backend this job deploys to, if any. Drives an auto-suggested export format
  // instead of leaving the GPTQ/AWQ/GGUF chip lists below to be lined up with vLLM/llama.cpp by hand.
  const [deployBackend, setDeployBackend] = useState<"none" | "vllm" | "llamacpp" | "sglang">("none");

  const backendEnv = useAsync(() => api.backendEnvs().then((r) => r.backends), []);
  const kind = kindOf(domain);
  const nativeCfg = useNative && (kind === "vla" || kind === "world" || kind === "ewm")
    ? { native: { concurrency: nativeConc, requests: nativeReq } }
    : {};
  // LLM 的阶段模式（其它领域沿用原来的"仅量化"开关）。
  const evalOnly = kind === "llm" && jobMode === "eval";
  const llmQuantOnly = kind === "llm" && jobMode === "quantize";
  // LLM 的"仅量化"直接由模式决定，不依赖 quantizeOnly 的异步同步；非 LLM 仍用原来的开关。
  const noEval = kind === "llm" ? llmQuantOnly : quantizeOnly;
  const evalGguf = evalOnly && (quantFormat === "gguf" || /\.gguf$/i.test(quantizedPath));
  useEffect(() => {
    if (kind === "llm") setQuantizeOnly(jobMode === "quantize");
  }, [jobMode, kind]);
  // only the pickers this kind shows count; a deploy job (B2) has no calibration data at all
  const dataOk = artifact
    ? true
    : kind === "llm"
      ? (evalOnly ? evalTextOk : (usePaste ? pasteSamples.length > 0 : calibTextOk) && (noEval || evalTextOk))
      : kind === "world" || kind === "ewm"
        ? calibWorldOk && evalWorldOk
        : calibVisionOk && evalVisionOk;
  const currentDomain = domainsQ.data?.find((d) => d.id === domain);
  const blocked = currentDomain != null && currentDomain.pipeline_status !== "implemented";

  const switchDomain = (d: string) => setDomain(d);
  // The form's initial values are LLM's. Whenever the domain differs from the one the form was last set up for
  // (opened in another domain, switched from the sidebar, or switched with the wizard's own domain tabs) reset
  // it to that domain's defaults. A derived job sets `appliedDomain` before changing the domain, same reason.
  // A layout effect, not an effect: the reset must land before the first paint, so a value the user types
  // immediately after clicking a domain tab can never be overwritten by a later default reset.
  const appliedDomain = useRef("llm");
  useLayoutEffect(() => {
    if (appliedDomain.current === domain) return;
    appliedDomain.current = domain;
    resetForDomain(domain);
  }, [domain]); // eslint-disable-line react-hooks/exhaustive-deps
  // ?model=<path> (a 模型库 card): start from that model; declared after the domain reset so it wins on first render
  useEffect(() => {
    const m = params.get("model");
    if (m && !params.get("from")) {
      setModel(m);
      setModelMode("custom");
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const resetForDomain = (d: string) => {
    const k = kindOf(d);
    setModel({ llm: DEFAULT_LLM, vlm: DEFAULT_VLM, vla: DEFAULT_MODEL, world: DEFAULT_WORLD, ewm: DEFAULT_EWM }[k]);
    setName({ llm: "qwen-wikitext-ppl", vlm: "minicpmv-vqa-eval", vla: "minicpm-vla-eval", world: "tiny-dit-world-eval", ewm: "tiny-ewm-eval" }[k]);
    setPicked(k === "llm" ? ["w4a16", "w4a8", "w2a16"] : k === "vlm" ? ["w8a16", "w4a16"] : k === "world" || k === "ewm" ? ["w8a16", "w4a16"] : ["w8a16", "fp8", "w4a16", "w4a8"]);
    setNCalib(k === "llm" ? 32 : k === "world" || k === "ewm" ? 8 : 16);
    setNEval(k === "vlm" ? 12 : k === "world" || k === "ewm" ? 8 : 24);
    setMaxNewTokens(24);
    setFrames(4);
    setQuantizeAction(false);
    setDataset(k === "vlm" ? "synthetic" : k === "world" || k === "ewm" ? "synthetic-video" : DEFAULT_DATASET);
    setEvalDataset(k === "vlm" ? "synthetic" : k === "world" || k === "ewm" ? "synthetic-video" : DEFAULT_DATASET);
    setCalibText("wikitext-2-raw-v1:train");
    setNoQuant(false);
    setJobMode(routeMode);
    setQuantizedPath("");
    setQuantFormat("auto");
    setQuantizeOnly(false);
    setGroupSizeOverride("");
    setUseNative(false);
    setModelMode("custom");
  };

  // "不量化（保持原始精度）": bf16 (data/recipes/bf16.yaml) already means "no quantization" end to end, so
  // this reuses that real recipe rather than inventing a new, unimplemented concept.
  // Unticking restores what was picked before ticking (it used to leave the job stuck on bf16-only).
  const pickedBeforeNoQuant = useRef<string[] | null>(null);
  useEffect(() => {
    if (noQuant) {
      setPicked((p) => {
        pickedBeforeNoQuant.current = p;
        return ["bf16"];
      });
    } else if (pickedBeforeNoQuant.current) {
      setPicked(pickedBeforeNoQuant.current);
      pickedBeforeNoQuant.current = null;
    }
  }, [noQuant]);

  // Picking a real deploy backend suggests (and can override) the export chips below, via the C++
  // planner's own compat-matrix logic (GET /api/v1/export-suggestion, T-1304) -- not a frontend guess.
  // Fetching and applying happen inside the same effect/callback (not a separate useAsync + a second
  // effect keyed on its .data) so there is no window where a stale suggestion from the *previous* backend
  // pairs with the *new* deployBackend value -- that combination briefly set llamacpp's "gguf" suggestion
  // onto a vllm selection in testing before this was merged into one effect.
  const primaryRecipe = picked.find((r) => r !== "bf16") ?? picked[0];
  const evalScopeCfg = useEvalScope && evalScopeDatasets.length
    ? { evalscope: { datasets: evalScopeDatasets, limit: evalScopeLimit, eval_batch_size: evalScopeBatch } }
    : {};
  const offlineBenchCfg = offlineBench.length
    ? { benchmarks: offlineBench, limit: offlineBenchLimit }
    : {};
  const pickedKey = picked.filter((r) => r !== "bf16").join(","); // a recipe added or removed can change which export format works
  const [suggestion, setSuggestion] = useState<Awaited<ReturnType<typeof api.exportSuggestion>> | null>(null);
  // where the server lets jobs read (for asking the planner about a job whose model path is not typed yet); null on an older server
  const [allowedRoots, setAllowedRoots] = useState<string[] | null>(null);
  useEffect(() => {
    api.datasets().then((d) => setAllowedRoots(d.allowed_roots), () => setAllowedRoots(null));
  }, []);
  const [servingNotes, setServingNotes] = useState<{ recipe: string; backend: string; reason: string }[]>([]);
  // A: what depends on the PRIMARY backend (the one last turned on): the suggestion shown on the card, llama.cpp's default GGUF type,
  // and the reset when every backend is off. Which formats to export is not decided here (see B): it must not depend on click order.
  useEffect(() => {
    if (deployBackend === "none") {
      setUseVllm(false);
      setUseSglang(false);
      setGguf([]);
      setSuggestion(null);
      return;
    }
    if (kind !== "llm" || !primaryRecipe || evalOnly) {
      setSuggestion(null);
      return;
    }
    let alive = true;
    api
      .exportSuggestion(primaryRecipe, deployBackend, hw)
      .then((s) => {
        if (!alive) return;
        setSuggestion(s);
        if (deployBackend === "llamacpp" && s.format) setGguf((g) => (g.length ? g : ["q4_k"]));
      })
      .catch(() => alive && setSuggestion(null));
    return () => {
      alive = false;
    };
  }, [deployBackend, primaryRecipe, hw, kind, evalOnly]);

  // B: what depends on the SET of vLLM / SGLang backends that are on and the recipes picked, however they were clicked. The export
  // format must be one the planner accepts for EVERY recipe (AWQ cannot export a 2-bit one), and a backend that cannot serve a
  // picked recipe is named with the server's own reason (the job still exports it; that stage will not work).
  const hfKey = `${useVllm ? "v" : ""}${useSglang ? "s" : ""}`;
  useEffect(() => {
    if (kind !== "llm" || artifact || evalOnly || !primaryRecipe || !hfKey) {
      setServingNotes([]);
      return;
    }
    let alive = true;
    const asked = picked.filter((r) => r !== "bf16");
    if (!asked.length) asked.push(primaryRecipe);
    const backends = [useVllm && "vllm", useSglang && "sglang"].filter(Boolean) as ("vllm" | "sglang")[];
    Promise.all(backends.flatMap((b) => asked.map((r) => api.exportSuggestion(r, b, hw).then((x) => ({ b, r, x })))))
      .then(async (rows) => {
        if (!alive) return;
        setServingNotes(rows.filter((y) => y.x.verdict === "unsupported").map((y) => ({ recipe: y.r, backend: y.b, reason: y.x.reason })));
        if (keepParentDeploy.current) {
          keepParentDeploy.current = false; // a derived job keeps the export formats it had, once
          return;
        }
        // each recipe's own suggestion first, then the usual formats; keep the first the planner's own dry run does not refuse on
        // /export/formats. "The planner spoke" = it validated the job (JSON-pointer fields like /export/formats); the path sandbox's
        // 403 and a network error say nothing about the formats.
        const candidates = [...new Set([...rows.filter((y) => y.x.verdict !== "unsupported").map((y) => y.x.format), ...rows.map((y) => y.x.format), "gptq", "awq", "compressed-tensors"].filter(Boolean))];
        const spoke = (r: Awaited<ReturnType<typeof api.dryRun>>) => r.ok || r.problems.some((p) => p.field.startsWith("/"));
        const base = job as unknown as Record<string, unknown> & { model: Record<string, unknown> };
        const ask = async (c: string) => {
          let r = await api.dryRun({ ...base, export: { formats: [c] } });
          // the user may not have typed a model path yet (or typed one the server may not read): ask again with a path it does allow
          if (!spoke(r) && allowedRoots?.length) r = await api.dryRun({ ...base, model: { ...base.model, path: allowedRoots[0] }, export: { formats: [c] } });
          return spoke(r) ? r : null;
        };
        let format = candidates[0];
        for (const c of candidates) {
          const r = await ask(c);
          if (!alive) return;
          if (r === null) {
            format = "gptq"; // the planner could not look: GPTQ is the one format that takes every weight width
            break;
          }
          if (r.ok || !r.problems.some((p) => p.field === "/export/formats")) {
            format = c;
            break;
          }
        }
        if (alive && format) setExportFmts([format]);
      })
      .catch(() => alive && setServingNotes([]));
    return () => {
      alive = false;
    };
  }, [hfKey, pickedKey, hw, kind, !!artifact, evalOnly]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!fromJob) return;
    let alive = true;
    api
      .job(fromJob)
      .then((det) => {
        if (!alive) return;
        if (!det.job) throw new Error(`作业 ${fromJob} 没有 job.yaml，无法派生`);
        const d = deriveFromJob(det.job, fromStep, {
          domain, name, model, picked, algorithm, groupSizeOverride, action, actStatic, kv, kvMethod, keepEnds, downBits,
          inheritedRules: [], nCalib, seqLen, dataset, evalDataset, calibText, calibPaste: pasteText, evalTextDataset, nEval, windows, maxNewTokens, frames, quantizeAction, attribution, quantizeOnly, quantDevice,
          evalDevice, hw, deployBackend, useVllm, vllmUtil, useSglang, sglangMem, useSweep, sweepLevels, sweepBudget, useLmEval, lmTasks, lmLimit, lmShots, gguf,
          exportFmts, sens, probe, probeExact,
        });
        keepParentDeploy.current = d.useVllm || d.useSglang; // only a vLLM / SGLang parent has export formats of its own to keep
        appliedDomain.current = d.domain; setDomain(d.domain); setName(d.name); setModel(d.model); setModelMode("custom");
        setNoQuant(d.picked.length === 1 && d.picked[0] === "bf16"); setPicked(d.picked);
        setAlgorithm(d.algorithm); setGroupSizeOverride(d.groupSizeOverride); setAdvancedOpen(d.groupSizeOverride !== "");
        setAction(d.action); setActStatic(d.actStatic); setKv(d.kv); setKvMethod(d.kvMethod); setKeepEnds(d.keepEnds);
        setDownBits(d.downBits); setInheritedRules(d.inheritedRules); setNCalib(d.nCalib); setSeqLen(d.seqLen);
        setDataset(d.dataset); setEvalDataset(d.evalDataset); setCalibText(d.calibText); setUsePaste(d.calibPaste !== ""); setPasteText(d.calibPaste); setEvalTextDataset(d.evalTextDataset); setNEval(d.nEval); setWindows(d.windows); setMaxNewTokens(d.maxNewTokens);
        setFrames(d.frames); setQuantizeAction(d.quantizeAction);
        setAttribution(d.attribution); setQuantizeOnly(d.quantizeOnly); setJobMode(d.quantizeOnly ? "quantize" : "full"); setQuantDevice(d.quantDevice); setEvalDevice(d.evalDevice);
        const derivedModel = (det.job as { model?: { quantized_path?: string; quant_format?: string } }).model;
        setQuantizedPath(derivedModel?.quantized_path ?? "");
        setQuantFormat(derivedModel?.quant_format ?? "auto");
        setHw(d.hw); setDeployBackend(d.deployBackend); setUseVllm(d.useVllm); setVllmUtil(d.vllmUtil); setUseSglang(d.useSglang);
        setSglangMem(d.sglangMem); setUseSweep(d.useSweep); setSweepLevels(d.sweepLevels); setSweepBudget(d.sweepBudget); setUseLmEval(d.useLmEval); setLmTasks(d.lmTasks); setLmLimit(d.lmLimit); setLmShots(d.lmShots);
        setGguf(d.gguf); setExportFmts(d.exportFmts); setSens(d.sens); setProbe(d.probe); setProbeExact(d.probeExact);
        setParentJob(fromJob);
        const own = (det.job as { artifact?: Artifact }).artifact;
        const source = own?.job ?? fromJob;
        (source === fromJob ? Promise.resolve(det) : api.job(source))
          .then((src) => {
            if (!alive) return;
            const opts = artifactOptions(source, src.report as unknown as Record<string, any>); // eslint-disable-line @typescript-eslint/no-explicit-any
            setArtifactOpts(opts);
            const pe = (src.report as unknown as { data?: { eval?: { windows?: number; seq_len?: number } } } | null)?.data?.eval;
            setParentEval(pe?.windows != null && pe?.seq_len != null ? { windows: pe.windows, seq_len: pe.seq_len } : null);
            const mine = own && opts.find((o) => sameArtifact(o.artifact, own));
            if (mine) setArtifactKey(mine.key);
            // re-quantizing is only needed to change the quantization or the fake-quant evaluation; serving elsewhere is not
            else if (opts.length && (fromStep === "deploy" || fromStep === "device")) setArtifactKey(opts[0].key);
          })
          .catch(() => alive && setArtifactOpts([]));
        const target = { eval: "step-eval", device: "step-eval", deploy: "step-deploy", recipe: "step-recipe", all: null }[fromStep];
        if (target) setTimeout(() => document.getElementById(target)?.scrollIntoView({ block: "start" }), 50);
      })
      .catch((e: Error) => alive && setDeriveErr(e.message));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fromJob, fromStep]);

  useEffect(() => {
    if (!artifact) return;
    if (artifact.format === "gguf") setDeployBackend("llamacpp");
    else setDeployBackend((b) => (b === "vllm" || b === "sglang" ? b : "vllm"));
  }, [artifactKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // The backends this job deploys to. A reused artifact (B2) goes to exactly one; a quantizing job can go to several.
  // Only what this domain can deploy to counts: a vLLM tick left over from the LLM tab is not a VLM choice.
  const domainBackends: BackendId[] = currentDomain?.deploy_backends ?? [];
  const chosenBackends: BackendId[] = (artifact
    ? deployBackend === "none" ? [] : [deployBackend]
    : ([useVllm && "vllm", useSglang && "sglang", gguf.length > 0 && "llamacpp"].filter(Boolean) as BackendId[])
  ).filter((b) => domainBackends.includes(b));
  const toggleBackend = (id: BackendId, on: boolean) => {
    if (artifact) {
      if (on) setDeployBackend(id);
      return;
    }
    const next: Record<BackendId, boolean> = { vllm: useVllm, sglang: useSglang, llamacpp: gguf.length > 0 };
    next[id] = on;
    if (id === "vllm") setUseVllm(on);
    if (id === "sglang") setUseSglang(on);
    if (id === "llamacpp") setGguf(on ? (gguf.length ? gguf : ["q4_k"]) : []);
    // the one the export suggestion follows: the one just turned on, else whichever is still on
    setDeployBackend(on ? id : (["vllm", "sglang", "llamacpp"] as const).find((b) => next[b]) ?? "none");
  };

  const exportChips = (
    <div>
      <div className="small muted">真实导出格式（vLLM / SGLang 共用；自动执行一致性校验）</div>
      <div className="chips">
        {["gptq", "awq"].map((f) => (
          <span key={f} className={`chip ${exportFmts.includes(f) ? "on" : ""}`} onClick={() => setExportFmts(exportFmts.includes(f) ? exportFmts.filter((x) => x !== f) : [...exportFmts, f])} data-testid={`export-chip-${f}`}>
            {f.toUpperCase()}
          </span>
        ))}
      </div>
    </div>
  );

  const offlineBenchChips = (
    <div data-testid="offline-benchmarks">
      <BenchmarkPicker
        runner="offline"
        value={offlineBench}
        onChange={setOfflineBench}
        limit={offlineBenchLimit}
        onLimit={setOfflineBenchLimit}
        suites
        title="离线评测数据集"
        testid="benchmark"
      />
    </div>
  );
  const lmEvalControls = (
    <>
      <label className="small">
        <input type="checkbox" checked={useLmEval} onChange={(e) => setUseLmEval(e.target.checked)} data-testid="opt-lmeval" /> lm-eval 任务精度（同时评未量化基线）
      </label>
      {useLmEval && (
        <div className="form-row" style={{ margin: 0 }}>
          <input value={lmTasks} onChange={(e) => setLmTasks(e.target.value)} placeholder="gsm8k,arc_easy" style={{ width: 150 }} aria-label="lm-eval 任务" data-testid="opt-lmeval-tasks" />
          <input type="number" min={1} value={lmLimit} style={{ width: 70 }} onChange={(e) => setLmLimit(Number(e.target.value))} aria-label="样本数" />
          <input type="number" min={0} value={lmShots} style={{ width: 60 }} onChange={(e) => setLmShots(Number(e.target.value))} aria-label="few-shot" />
        </div>
      )}
    </>
  );
  const evalScopeChips = (
    <div>
      <label className="small">
        <input type="checkbox" checked={useEvalScope} onChange={(e) => setUseEvalScope(e.target.checked)} data-testid="opt-evalscope" />{" "}
        EvalScope benchmark（外部 CLI，未安装则跳过）
      </label>
      {useEvalScope && (
        <>
          <BenchmarkPicker
            runner="evalscope"
            value={evalScopeDatasets}
            onChange={setEvalScopeDatasets}
            limit={evalScopeLimit}
            onLimit={setEvalScopeLimit}
            title="EvalScope 数据集"
            testid="evalscope-benchmark"
          />
          <div className="form-row" style={{ margin: 0 }}>
            <label className="field" style={{ width: 110 }}>
              <span className="lbl">批大小</span>
              <input type="number" min={1} value={evalScopeBatch} onChange={(e) => setEvalScopeBatch(Number(e.target.value))} aria-label="EvalScope 批大小" />
            </label>
          </div>
          {evalScopeDatasets.length === 0 && <div className="note warn">至少选择一个数据集，否则不会运行 EvalScope。</div>}
        </>
      )}
    </div>
  );

  const kvSpec = kv === "" ? null : kv === "fp8" ? { format: "fp8_e4m3", bits: 8 } : { format: "int", bits: Number(kv.slice(3)), symmetric: false };
  const rules: Rule[] = inheritedRules.length ? inheritedRules : [
    ...(keepEnds ? [{ match: "block:0", bits: 8 }, { match: "block:-1", bits: 8 }] : []),
    ...(downBits === "fp" ? [{ match: "mlp\\.down_proj$", skip: true }] : downBits ? [{ match: "mlp\\.down_proj$", bits: Number(downBits) }] : []),
  ];
  const deployJob = artifact && {
    domain,
    metadata: { name, ...(parentJob ? { parent_job: parentJob } : {}) },
    model: { source: "local", path: model, type: "llm" },
    artifact,
    // 只评测选中的产物；勾了后端才额外部署。
    stages: chosenBackends.length ? ["evaluate", "deploy"] : ["evaluate"],
    evaluation: { dataset: "wikitext-2-raw-v1:test", seq_len: seqLen, max_windows: windows, ...offlineBenchCfg },
    deploy: {
      hardware: hw,
      device: isGguf ? evalDevice : "gpu",
      ...(isGguf
        ? { llamacpp: { gpu_layers: evalDevice === "cpu" ? 0 : 99, serve: { concurrency: 4, requests: 16, prompt_len: 256, n_predict: 64 } } }
        : deployBackend === "sglang"
          ? { sglang: { mem_fraction_static: sglangMem, ...sweepCfg, ...evalScopeCfg, ...lmEvalCfg } }
          : {
              vllm: {
                gpu_memory_utilization: vllmUtil,
                ...sweepCfg,
                ...evalScopeCfg,
                ...lmEvalCfg,
              },
            }),
    },
  };
  // 显式阶段：仅量化不导出也能结束；量化+导出会落盘可复用的 checkpoint；仅评测不量化。
  const llmStages: string[] = artifact
    ? chosenBackends.length ? ["evaluate", "deploy"] : ["evaluate"]
    : evalOnly
      ? chosenBackends.length ? ["evaluate", "deploy"] : ["evaluate"]
      : llmQuantOnly
        ? exportFmts.length ? ["quantize", "export"] : ["quantize"]
        : ["quantize", "evaluate", ...(exportFmts.length ? ["export"] : []),
           ...(chosenBackends.length && exportFmts.length ? ["deploy"] : [])];
  const job =
    kind === "llm" && evalOnly
      ? {
          domain,
          metadata: { name, ...(parentJob ? { parent_job: parentJob } : {}) },
          model: {
            source: "local", path: model, type: "llm",
            quantized_path: quantizedPath,
            ...(quantFormat !== "auto" ? { quant_format: quantFormat } : {}),
          },
          stages: llmStages,
          evaluation: {
            dataset: evalTextDataset,
            seq_len: seqLen,
            max_windows: windows,
            ...(evalDevice !== "cpu" ? { device: evalDevice } : {}),
            ...(evalGguf ? {} : offlineBenchCfg),
          },
          deploy: {
            hardware: hw,
            device: chosenBackends.length ? "gpu" : evalDevice,
            ...(useSglang ? { sglang: { mem_fraction_static: sglangMem, ...sweepCfg, ...evalScopeCfg, ...lmEvalCfg } } : {}),
            ...(useVllm
              ? {
                  vllm: {
                    gpu_memory_utilization: vllmUtil,
                    ...sweepCfg,
                    ...evalScopeCfg,
                    ...lmEvalCfg,
                  },
                }
              : {}),
            ...(gguf.length ? { llamacpp: { gpu_layers: evalDevice === "cpu" ? 0 : 99 } } : {}),
          },
        }
      : kind === "llm" && deployJob
      ? deployJob
      : kind === "vlm"
      ? {
          domain,
          metadata: { name, ...(parentJob ? { parent_job: parentJob } : {}) },
          model: { source: "local", path: model, type: "vlm" },
          quant: {
            recipe: picked,
            algorithm,
            ...(groupSizeOverride !== "" ? { group_size: Number(groupSizeOverride) } : {}),
            ...(inheritedRules.length ? { mixed_precision: { rules: inheritedRules } } : {}),
          },
          calibration: { dataset, n_samples: nCalib },
          evaluation: {
            dataset: evalDataset,
            n_samples: nEval,
            max_new_tokens: maxNewTokens,
            attribution: quantizeOnly ? false : attribution,
            ...(quantizeOnly ? { skip: true } : {}),
          },
          deploy: { hardware: hw },
        }
      : kind === "vla"
      ? {
          domain,
          metadata: { name, ...(parentJob ? { parent_job: parentJob } : {}) },
          model: { source: "local", path: model, type: "vla", modality: "vla" },
          quant: { recipe: picked, algorithm, ...(action ? { also_quantize: ["action"] } : {}), ...(groupSizeOverride !== "" ? { group_size: Number(groupSizeOverride) } : {}) },
          calibration: { dataset, n_samples: nCalib },
          evaluation: {
            dataset: evalDataset,
            n_samples: nEval,
            concurrency: [1, 4, 16, 64],
            attribution: quantizeOnly ? false : attribution,
            ...(quantizeOnly ? { skip: true } : {}),
          },
          deploy: { hardware: hw, ...nativeCfg },
        }
      : kind === "world" || kind === "ewm"
      ? {
          domain,
          metadata: { name, ...(parentJob ? { parent_job: parentJob } : {}) },
          model: { source: "local", path: model, type: kind },
          quant: {
            recipe: picked,
            algorithm,
            ...(kind === "ewm" && quantizeAction ? { also_quantize: ["action"] } : {}),
            ...(groupSizeOverride !== "" ? { group_size: Number(groupSizeOverride) } : {}),
            ...(inheritedRules.length ? { mixed_precision: { rules: inheritedRules } } : {}),
          },
          calibration: { dataset, n_samples: nCalib, frames },
          evaluation: {
            dataset: evalDataset,
            n_samples: nEval,
            frames,
            action_conditioned: kind === "ewm",
            attribution: quantizeOnly ? false : attribution,
            ...(quantizeOnly ? { skip: true } : {}),
          },
          deploy: { hardware: hw, device: "gpu", ...nativeCfg },
        }
      : {
          domain,
          metadata: { name, ...(parentJob ? { parent_job: parentJob } : {}) },
          model: { source: "local", path: model, type: "llm" },
          quant: {
            recipe: picked,
            algorithm,
            act_static: actStatic || null,
            ...(kvSpec ? { kv: { ...kvSpec, method: kvMethod } } : {}),
            ...(rules.length ? { mixed_precision: { rules } } : {}),
            ...(groupSizeOverride !== "" ? { group_size: Number(groupSizeOverride) } : {}),
          },
          ...(quantDevice !== evalDevice ? { quant_device: quantDevice } : {}),
          calibration: { dataset: calibText, n_samples: nCalib, seq_len: seqLen, ...(usePaste && pasteSamples.length ? { custom_samples: pasteSamples } : {}) },
          ...(noEval ? {} : {
            evaluation: {
              dataset: evalTextDataset,
              seq_len: seqLen,
              max_windows: windows,
              ...(quantDevice !== evalDevice ? { device: evalDevice } : {}),
            },
          }),
          stages: llmStages,
          deploy: {
            hardware: hw,
            device: quantDevice === "gpu" || evalDevice === "gpu" ? "gpu" : "cpu",
            ...(!noEval && gguf.length ? { llamacpp: { types: gguf } } : {}),
            ...(!noEval && useSglang ? { sglang: { mem_fraction_static: sglangMem, ...sweepCfg, ...evalScopeCfg, ...lmEvalCfg } } : {}),
            ...(!noEval && useVllm
              ? {
                  vllm: {
                    gpu_memory_utilization: vllmUtil,
                    ...sweepCfg,
                    ...evalScopeCfg,
                    ...lmEvalCfg,
                  },
                }
              : {}),
          },
          ...((!noEval || llmQuantOnly) && exportFmts.length ? { export: { formats: exportFmts } } : {}),
          ...(!noEval && (sens || probe)
            ? {
                analysis: {
                  ...(sens ? { sensitivity: sens, sensitivity_windows: 8 } : {}),
                  ...(probe ? { logits_probe: probeExact ? { exact: true } : {} } : {}),
                },
              }
            : {}),
        };

  const submit = async () => {
    setErr(null);
    setBusy(true);
    try {
      const { job_id } = await api.submit(job);
      nav(`/jobs/${job_id}`);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // live estimate sidebar: same endpoint as the estimate page, 300 ms after the last change
  const [estBackend, setEstBackend] = useState("hf_transformers");
  const backendList = useAsync(api.backends, []);
  const live = useDebounced({ model, hw, picked, action, estBackend }, 300);
  const liveRt = { ...LIVE_RUNTIME, extra_modalities: live.action && kind === "vla" ? ["action"] : [] };
  const est = useAsync(
    () =>
      live.picked.length && kind !== "world" && kind !== "ewm"
        ? api.estimate(live.model, live.hw, live.picked, liveRt, live.estBackend)
        : Promise.resolve(null),
    [JSON.stringify(live), kind],
  );

  const toggle = (id: string) => setPicked(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id]);
  const pageTitle = kind === "llm"
    ? jobMode === "eval" ? "新建评测作业" : jobMode === "quantize" ? "新建量化作业" : "新建全流程作业"
    : "新建作业";
  const pageSubtitle = kind === "llm"
    ? jobMode === "eval"
      ? "填已量化模型路径和评测数据集即可；可选部署后端做真实压测。GPU 池 " + (cfg.data?.gpus.join(", ") ?? "…") + "。"
      : jobMode === "quantize"
        ? "只量化并导出可复用权重，不跑评估、不部署。GPU 池 " + (cfg.data?.gpus.join(", ") ?? "…") + "。"
        : "选模型 → 配方 → 数据 → 部署；右侧实时预估。GPU 池 " + (cfg.data?.gpus.join(", ") ?? "…") + "。"
    : "选模型、配方、数据与部署硬件；右侧实时预估。";

  // Dedicated, low-noise layouts for the two "single-stage" entry points. The full six-step wizard
  // remains for the full pipeline (and for ?mode=all, which older automation still uses).
  if (kind === "llm" && (evalOnly || llmQuantOnly)) {
    const recipeOptions = recipes.data?.filter((r) => r.weight.format !== "none") ?? [];
    const primary = picked[0] ?? "";
    const setPrimary = (id: string) => setPicked(id ? [id, ...picked.filter((x) => x !== id)] : picked);
    const addRecipe = (id: string) => {
      if (!id || picked.includes(id)) return;
      setNoQuant(false);
      setPicked([...picked, id]);
    };
    const modeChips = (
      <div className="chips" data-testid="job-mode">
        {([
          ["full", "全流程", "量化 → 评估 →（可选）导出 / 部署"],
          ["quantize", "仅量化", "只量化并导出可复用权重"],
          ["eval", "仅评测", "直接评测已有的量化 checkpoint"],
        ] as const).map(([m, label, hint]) => (
          <span key={m} className={`chip ${jobMode === m ? "on" : ""}`} onClick={() => setJobMode(m)} title={hint} data-testid={`job-mode-${m}`}>
            {label}
          </span>
        ))}
      </div>
    );
    return (
      <>
        <div className="page-head">
          <div>
            <h1>{evalOnly ? "新建评测作业" : "新建量化作业"}</h1>
            <p className="subtitle">{pageSubtitle}</p>
          </div>
          {modeChips}
        </div>
        {err && <div className="error-box">{err}</div>}

        {evalOnly ? (
          <>
            <div className="card focused-card" data-testid="focused-eval-object">
              <div className="card-head">
                <div>
                  <h2>评测对象</h2>
                  <p className="note" style={{ margin: 0 }}>只读取这个已量化 checkpoint，不重新量化。</p>
                </div>
                <span className="badge accent">evaluate only</span>
              </div>
              <div className="form-row">
                <label className="field grow">
                  <span className="lbl">已量化模型路径</span>
                  <input value={quantizedPath} onChange={(e) => setQuantizedPath(e.target.value)} placeholder="/path/to/quantized-checkpoint 或 model.gguf" data-testid="quantized-path" />
                </label>
                <label className="field" style={{ width: 190 }}>
                  <span className="lbl">量化格式</span>
                  <select value={quantFormat} onChange={(e) => setQuantFormat(e.target.value)} data-testid="quant-format">
                    <option value="auto">自动识别</option>
                    <option value="compressed-tensors">compressed-tensors</option>
                    <option value="awq">AWQ</option>
                    <option value="gptq">GPTQ</option>
                    <option value="gguf">GGUF（llama.cpp）</option>
                  </select>
                </label>
                <label className="field" style={{ width: 220 }}>
                  <span className="lbl">作业名称</span>
                  <input value={name} onChange={(e) => setName(e.target.value)} />
                </label>
              </div>
              <details className="focused-advanced" data-testid="eval-advanced">
                <summary>基座模型与 tokenizer（可选）</summary>
                <div className="form-row">
                  <label className="field grow">
                    <span className="lbl">基座模型路径</span>
                    <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="checkpoint 自带 tokenizer 时可留空" data-testid="model-path" />
                  </label>
                </div>
                <p className="small muted" data-testid="eval-only-note">
                  量化 checkpoint 自带 tokenizer 时不需要基座模型；本机 transformers 加载不了时，可在下方启用 vLLM / SGLang 加载。
                </p>
              </details>
            </div>

            <div className="card focused-card" data-testid="focused-eval-data">
              <div className="card-head">
                <div>
                  <h2>评测数据集</h2>
                  <p className="note" style={{ margin: 0 }}>先选套件或类别，再选数据集；下方会显示指标、样本量和运行方式。</p>
                </div>
              </div>
              {offlineBenchChips}
              <details className="focused-advanced">
                <summary>PPL 语料与运行参数</summary>
                <div className="form-row">
                  <DatasetPicker
                    kind="text"
                    role="evaluation"
                    label="PPL 语料（默认 wikitext-2 test）"
                    value={evalTextDataset}
                    onChange={setEvalTextDataset}
                    onValid={setEvalTextOk}
                    testid="eval-dataset"
                  />
                  <label className="field" style={{ width: 140 }}>
                    <span className="lbl">序列长度</span>
                    <input type="number" value={seqLen} min={64} step={64} onChange={(e) => setSeqLen(Number(e.target.value))} />
                  </label>
                  <label className="field" style={{ width: 130 }}>
                    <span className="lbl">PPL 窗口数</span>
                    <input type="number" value={windows} min={1} onChange={(e) => setWindows(Number(e.target.value))} data-testid="eval-windows" />
                  </label>
                  <label className="field" style={{ width: 160 }}>
                    <span className="lbl">运行设备</span>
                    <select value={evalDevice} onChange={(e) => setEvalDevice(e.target.value)} data-testid="eval-device">
                      <option value="cpu">CPU</option>
                      <option value="gpu">GPU 池</option>
                    </select>
                  </label>
                </div>
                {evalGguf ? (
                  <p className="small muted" data-testid="eval-gguf-note">GGUF 由 llama.cpp 的 llama-perplexity 直接计算。</p>
                ) : null}
              </details>
            </div>

            <details className="card focused-card focused-collapse" data-testid="focused-eval-deploy">
              <summary>
                <b>可选：部署与压测</b>
                <span className="muted small">不选后端就只做离线评测；展开后可起服务、测吞吐并运行 EvalScope。</span>
              </summary>
              <div className="focused-collapse-body">
              <div className="form-row">
                <label className="field" style={{ width: 220 }}>
                  <span className="lbl">推理后端</span>
                  <select
                    value={useSglang ? "sglang" : useVllm ? "vllm" : gguf.length ? "llamacpp" : ""}
                    onChange={(e) => {
                      const v = e.target.value as "" | "vllm" | "sglang" | "llamacpp";
                      toggleBackend("vllm", v === "vllm");
                      toggleBackend("sglang", v === "sglang");
                      if (v !== "llamacpp") setGguf([]);
                      else setGguf(["q4_k"]);
                    }}
                    data-testid="eval-deploy-backend"
                  >
                    <option value="">不部署</option>
                    {!evalGguf && <option value="vllm">vLLM</option>}
                    {!evalGguf && <option value="sglang">SGLang</option>}
                    {evalGguf && <option value="llamacpp">llama.cpp</option>}
                  </select>
                </label>
                <label className="field" style={{ width: 300 }}>
                  <span className="lbl">目标硬件</span>
                  <select value={hw} onChange={(e) => setHw(e.target.value)} data-testid="hw-select">
                    {profiles.data?.map((p) => <option key={p.id} value={p.id}>{p.name}（{p.id}）</option>)}
                  </select>
                </label>
                {(useVllm || useSglang) && (
                  <label className="field" style={{ width: 170 }}>
                    <span className="lbl">显存占比</span>
                    <input type="number" step={0.05} min={0.1} max={0.95} value={useSglang ? sglangMem : vllmUtil}
                      onChange={(e) => (useSglang ? setSglangMem(Number(e.target.value)) : setVllmUtil(Number(e.target.value)))} />
                  </label>
                )}
              </div>
              {(useVllm || useSglang) && (
                <details className="focused-advanced" open>
                  <summary>部署评测选项</summary>
                  <label className="small" style={{ display: "block", marginBottom: 8 }}>
                    <input type="checkbox" checked={useEvalScope} onChange={(e) => setUseEvalScope(e.target.checked)} data-testid="opt-evalscope" /> 同时运行 EvalScope 数据集
                  </label>
                  {useEvalScope && evalScopeChips}
                  {(useVllm || useSglang) && (
                    <label className="small" style={{ display: "block", marginTop: 8 }}>
                      <input type="checkbox" checked={useLmEval} onChange={(e) => setUseLmEval(e.target.checked)} data-testid="opt-lmeval" /> lm-eval 任务精度
                    </label>
                  )}
                </details>
              )}
              </div>
            </details>
          </>
        ) : (
          <>
            <div className="card focused-card" data-testid="focused-quant-model">
              <div className="card-head">
                <div>
                  <h2>模型</h2>
                  <p className="note" style={{ margin: 0 }}>量化完成后直接写出一份可复用的 checkpoint。</p>
                </div>
                <span className="badge accent">quantize + export</span>
              </div>
              <div className="form-row">
                <label className="field grow">
                  <span className="lbl">本地模型路径</span>
                  <input value={model} onChange={(e) => setModel(e.target.value)} data-testid="model-path" />
                </label>
                <label className="field" style={{ width: 220 }}>
                  <span className="lbl">作业名称</span>
                  <input value={name} onChange={(e) => setName(e.target.value)} />
                </label>
              </div>
            </div>

            <div className="card focused-card" data-testid="focused-quant-recipe">
              <div className="card-head">
                <div>
                  <h2>量化设置</h2>
                  <p className="note" style={{ margin: 0 }}>先选主配方。需要对比多个配方时，再追加到右侧列表。</p>
                </div>
              </div>
              <div className="form-row">
                <label className="field grow">
                  <span className="lbl">主配方</span>
                  <select value={primary} onChange={(e) => setPrimary(e.target.value)} data-testid="recipe-primary">
                    {recipeOptions.map((r) => (
                      <option key={r.id} value={r.id}>{r.name} · {r.summary}</option>
                    ))}
                  </select>
                </label>
                <label className="field grow">
                  <span className="lbl">追加对比配方</span>
                  <select value="" onChange={(e) => addRecipe(e.target.value)} data-testid="recipe-add">
                    <option value="">选择后加入…</option>
                    {recipeOptions.filter((r) => !picked.includes(r.id)).map((r) => (
                      <option key={r.id} value={r.id}>{r.name} · {r.summary}</option>
                    ))}
                  </select>
                </label>
                <label className="field grow">
                  <span className="lbl">量化算法</span>
                  <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)} disabled={noQuant}>
                    <option value="recipe">按配方默认</option>
                    {algos.data?.map((a) => <option key={a.name} value={a.name}>{a.label}</option>)}
                  </select>
                </label>
              </div>
              <div className="benchmark-selected" data-testid="recipe-selected">
                {picked.map((id) => {
                  const r = recipeOptions.find((x) => x.id === id);
                  return (
                    <span key={id} className="benchmark-pill">
                      {r?.name ?? id.toUpperCase()}
                      {picked.length > 1 && <button aria-label={`移除 ${id}`} onClick={() => setPicked(picked.filter((x) => x !== id))}>×</button>}
                    </span>
                  );
                })}
              </div>
              <details className="focused-advanced">
                <summary>高级：量化设备、group size、混合精度</summary>
                <div className="form-row">
                  <label className="field" style={{ width: 220 }}>
                    <span className="lbl">量化设备</span>
                    <select value={quantDevice} onChange={(e) => setQuantDevice(e.target.value)} data-testid="quant-device-select">
                      <option value="cpu">CPU（不占 GPU 池）</option>
                      <option value="gpu">GPU 池（{cfg.data?.gpus.join(", ") ?? "…"}）</option>
                    </select>
                  </label>
                  <label className="field" style={{ width: 170 }}>
                    <span className="lbl">Group size 覆盖</span>
                    <input type="number" min={1} value={groupSizeOverride} placeholder="按配方" onChange={(e) => setGroupSizeOverride(e.target.value)} data-testid="group-size-override" />
                  </label>
                  <label className="field" style={{ width: 220 }}>
                    <span className="lbl">首尾块混合精度</span>
                    <select value={keepEnds ? "8" : ""} onChange={(e) => setKeepEnds(e.target.value === "8")}>
                      <option value="">同配方</option>
                      <option value="8">首块 + 末块 8 bit</option>
                    </select>
                  </label>
                  <label className="field" style={{ width: 220 }}>
                    <span className="lbl">down_proj</span>
                    <select value={downBits} onChange={(e) => setDownBits(e.target.value)}>
                      <option value="">同配方</option>
                      <option value="8">8 bit</option>
                      <option value="4">4 bit</option>
                      <option value="fp">保持 FP</option>
                    </select>
                  </label>
                </div>
              </details>
            </div>

            <div className="card focused-card" data-testid="focused-quant-calib">
              <div className="card-head">
                <div>
                  <h2>校准数据</h2>
                  <p className="note" style={{ margin: 0 }}>GPTQ / AWQ 等方法需要少量校准文本；内置 WikiText-2 可直接使用。</p>
                </div>
              </div>
              <div className="form-row">
                <DatasetPicker
                  kind="text"
                  role="calibration"
                  label="校准数据"
                  value={calibText}
                  onChange={setCalibText}
                  onValid={setCalibTextOk}
                  testid="calib-dataset"
                />
                <label className="field" style={{ width: 160 }}>
                  <span className="lbl">校准样本</span>
                  <input type="number" value={nCalib} min={1} onChange={(e) => setNCalib(Number(e.target.value))} data-testid="calib-n" />
                </label>
                <label className="field" style={{ width: 150 }}>
                  <span className="lbl">序列长度</span>
                  <input type="number" value={seqLen} min={64} step={64} onChange={(e) => setSeqLen(Number(e.target.value))} />
                </label>
              </div>
            </div>

            <div className="card focused-card" data-testid="quantize-export">
              <div className="card-head">
                <div>
                  <h2>导出</h2>
                  <p className="note" style={{ margin: 0 }}>权重写到 runs/&lt;job&gt;/export/&lt;recipe&gt;-&lt;format&gt;/，之后可被“仅评测”直接加载。</p>
                </div>
              </div>
              <div className="form-row">
                <label className="field" style={{ width: 220 }}>
                  <span className="lbl">导出格式</span>
                  <select
                    value={exportFmts[0] ?? ""}
                    onChange={(e) => setExportFmts(e.target.value ? [e.target.value] : [])}
                    data-testid="export-format"
                  >
                    <option value="">不导出（只量化）</option>
                    <option value="gptq">GPTQ</option>
                    <option value="awq">AWQ</option>
                    <option value="compressed-tensors">compressed-tensors</option>
                  </select>
                </label>
                <label className="field" style={{ width: 320 }}>
                  <span className="lbl">目标硬件（用于导出适配建议）</span>
                  <select value={hw} onChange={(e) => setHw(e.target.value)} data-testid="hw-select">
                    {profiles.data?.map((p) => <option key={p.id} value={p.id}>{p.name}（{p.id}）</option>)}
                  </select>
                </label>
                <label className="field grow">
                  <span className="lbl">导出后用途</span>
                  <div className="small muted" style={{ display: "flex", alignItems: "center", minHeight: 32 }}>
                    可直接在「新建评测作业」中加载，不必重新量化。
                  </div>
                </label>
              </div>
            </div>
          </>
        )}

        <div className="card submit-bar" data-testid="focused-submit">
          <div>
            <b>{evalOnly ? "评测作业" : "量化作业"}</b>
            <p className="small muted" style={{ margin: 0 }}>
              {evalOnly
                ? `${offlineBench.length} 个离线数据集${chosenBackends.length ? ` · 部署 ${chosenBackends.join(" + ")}` : " · 不部署"}`
                : `${picked.map((x) => x.toUpperCase()).join(" + ")} · ${exportFmts.join(" / ") || "不导出"}`}
            </p>
          </div>
          <div className="focused-spec">
            <span className="small muted">JobSpec</span>
            <pre className="log" style={{ maxHeight: 110, height: "auto" }}>{JSON.stringify(job, null, 2)}</pre>
          </div>
          <button disabled={busy || blocked || (evalOnly ? !quantizedPath : picked.length === 0) || (useSweep && !levelsOk) || !dataOk} onClick={submit}>
            {busy ? "提交中…" : "提交作业"}
          </button>
        </div>
      </>
    );
  }

  return (
    <>
      <h1>{pageTitle}</h1>
      <p className="subtitle">{pageSubtitle}</p>
      {err && <div className="error-box">{err}</div>}
      {deriveErr && <div className="error-box">{deriveErr}</div>}
      {parentJob && (
        <div className="note" data-testid="derive-banner" style={{ marginBottom: 12 }}>
          派生自作业 <Link to={`/jobs/${parentJob}`}>{parentJob}</Link>（{DERIVE_STEPS.find((x) => x.step === fromStep)?.label}）：已沿用它的全部参数，只改你要改的部分即可。
          {artifact
            ? "本次复用父作业导出的产物，不重新量化：只部署、评测。"
            : "量化会重新运行——量化后的权重没有落盘，但同样的配方、数据和随机种子得到同样的量化结果。"}
          {artifactOpts.length > 0 && (
            <div className="form-row" style={{ marginTop: 8 }}>
              <div className="field grow">
                <label>复用父作业的导出产物（不重新量化）</label>
                <select value={artifactKey} onChange={(e) => setArtifactKey(e.target.value)} data-testid="artifact-select">
                  <option value="">不复用：按下面的设置重新量化</option>
                  {artifactOpts.map((o) => (
                    <option key={o.key} value={o.key}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>
      )}
      <div className="split newjob-layout">
      <div>
      <StepNav
        step={step}
        setStep={setStep}
        singlePage={singlePage}
        setSinglePage={(v) => setParams((p) => { const n = new URLSearchParams(p); if (v) n.set("mode", "all"); else n.delete("mode"); return n; }, { replace: true })}
        blocked={blocked}
        skipQuant={!!artifact || evalOnly}
        skipReason={artifact ? "沿用父作业" : "无需量化"}
        canLeave={{ 1: (evalOnly || !!model) && (!evalOnly || !!quantizedPath), 3: evalOnly || picked.length > 0 }}
        summary={{
          1: `${domain.toUpperCase()} · ${(evalOnly && quantizedPath ? quantizedPath : model).split("/").pop() || "—"}`,
          2: evalOnly ? "已量化" : algorithm === "recipe" ? "按配方默认" : algorithm,
          3: evalOnly ? "无需量化" : artifact ? "沿用父作业产物" : picked.map((x) => x.toUpperCase()).join("、") || "—",
          4: kind === "llm" ? (artifact ? `${windows} 窗口 × ${seqLen}` : `${windows} 窗口 × ${seqLen} · 量化 ${quantDevice.toUpperCase()} / 评估 ${evalDevice.toUpperCase()}`) : `${dataset.split("/").pop()}${evalDataset !== dataset ? ` → ${evalDataset.split("/").pop()}` : ""} · ${nEval} 样本${kind === "world" || kind === "ewm" ? ` × ${frames} 帧` : ""}`,
          5: `${profiles.data?.find((p) => p.id === hw)?.name ?? hw}${chosenBackends.length ? ` · ${chosenBackends.join(" + ")}` : ""}`,
        }}
      />
      <div className="card" style={{ display: show(1) || show(2) || show(3) ? undefined : "none" }}>
        <h2 style={{ display: show(1) ? undefined : "none" }}>① 领域 + 模型</h2>
        <div style={{ display: show(1) ? undefined : "none" }}>
        <div className="tabs" style={{ marginBottom: 12 }}>
          {(domainsQ.data ?? [{ id: "llm", display_name: "LLM（HF CausalLM · wikitext-2 PPL）" }, { id: "vla", display_name: "VLA（视觉-语言-动作）" }]).map((d) => (
            <div key={d.id} className={`tab ${domain === d.id ? "on" : ""}`} onClick={() => switchDomain(d.id)} data-testid={`domain-tab-${d.id}`}>
              {d.display_name}
            </div>
          ))}
        </div>
        {kind === "llm" && !artifact && (
          <div className="chips" style={{ marginBottom: 12 }} data-testid="job-mode">
            {([
              ["full", "全流程", "量化 → 评估 →（可选）导出 / 部署"],
              ["quantize", "仅量化", "只量化；勾选导出格式即可把权重落盘复用"],
              ["eval", "仅评测", "直接加载已有的量化 checkpoint，不重新量化"],
            ] as const).map(([m, label, hint]) => (
              <span
                key={m}
                className={`chip ${jobMode === m ? "on" : ""}`}
                onClick={() => setJobMode(m)}
                title={hint}
                data-testid={`job-mode-${m}`}
              >
                {label}
              </span>
            ))}
          </div>
        )}
        </div>
        {blocked ? (
          <div className="error-box" data-testid="domain-blocked">
            领域"{currentDomain?.display_name}"目前只在界面上可选，还没有真正的量化流水线（pipeline_status: {currentDomain?.pipeline_status}）。
            提交作业会被服务端拒绝并给出同样的理由。参考方法（尚未实现，仅作记录）：
            {currentDomain?.classic_algorithms.map((a) => a.name).join("、") || "—"}。
          </div>
        ) : (
          <>
            <div style={{ display: show(1) ? undefined : "none" }}>
            {(kind === "llm" || kind === "vlm") && (
              <div className="tabs" style={{ marginBottom: 8, borderBottom: "none" }}>
                <div className={`tab ${modelMode === "custom" ? "on" : ""}`} onClick={() => setModelMode("custom")}>
                  自定义路径
                </div>
                <div className={`tab ${modelMode === "catalog" ? "on" : ""}`} onClick={() => setModelMode("catalog")}>
                  从模型目录选择
                </div>
              </div>
            )}
            {modelMode === "catalog" && (kind === "llm" || kind === "vlm") ? (
              <CatalogModelPicker onPick={setModel} currentPath={model} />
            ) : (
              <div className="form-row">
                <div className="field grow">
                  <label>{evalOnly ? "基座模型路径（可选，仅作分词器 / 架构兜底；允许根目录：" : "本地模型路径（允许根目录："}{cfg.data?.allowed_roots.join(", ")}）</label>
                  <input value={model} onChange={(e) => setModel(e.target.value)} data-testid="model-path" />
                </div>
                <div className="field">
                  <label>作业名称</label>
                  <input value={name} onChange={(e) => setName(e.target.value)} />
                </div>
              </div>
            )}
            {evalOnly && (
              <>
                <div className="form-row">
                  <div className="field grow">
                    <label>已量化模型路径（AWQ / GPTQ / compressed-tensors 目录，或 .gguf 文件）</label>
                    <input
                      value={quantizedPath}
                      onChange={(e) => setQuantizedPath(e.target.value)}
                      placeholder="/path/to/quantized-checkpoint"
                      data-testid="quantized-path"
                    />
                  </div>
                  <div className="field">
                    <label>量化格式</label>
                    <select value={quantFormat} onChange={(e) => setQuantFormat(e.target.value)} data-testid="quant-format">
                      <option value="auto">自动识别</option>
                      <option value="compressed-tensors">compressed-tensors</option>
                      <option value="awq">AWQ</option>
                      <option value="gptq">GPTQ</option>
                      <option value="gguf">GGUF（llama.cpp）</option>
                    </select>
                  </div>
                </div>
                <p className="small muted" data-testid="eval-only-note">
                  上面的“基座模型路径”可以留空：checkpoint 自带 tokenizer 时只用它；评测直接读这个已量化 checkpoint，不重新量化。
                  本机 transformers 加载不了时，可在第⑤步选 vLLM / SGLang 让它加载。
                </p>
              </>
            )}
            </div>
            <div style={{ display: artifact || evalOnly ? "none" : undefined }} data-testid="quant-steps">
            <div style={{ display: show(1) ? undefined : "none" }}>
            {kind !== "world" && kind !== "ewm" && <AdvicePanel
              model={model}
              kind={kind}
              hw={hw}
              profiles={profiles.data ?? []}
              onAdopt={(a) => {
                const id = recipes.data?.find((r) => r.name === a.recipe)?.id;
                if (id) {
                  setNoQuant(false);
                  setPicked([id]);
                }
                setAlgorithm(a.algorithm || "recipe");
                setHw(a.hardware);
                if (kind === "llm") (a.backend === "vllm" ? toggleBackend("vllm", true) : setDeployBackend("none")); // the same path the cards use: flags, format and warnings all follow
              }}
            />}
            </div>
            <div style={{ display: show(2) ? undefined : "none" }}>
            <h2 style={{ marginTop: 16 }}>② 量化算法</h2>
            <div className="form-row">
              <div className="field" style={{ minWidth: 260 }}>
                <label>算法（所有勾选配方共用）</label>
                <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)} disabled={noQuant}>
                  <option value="recipe">按配方默认（{recipes.data?.filter((r) => picked.includes(r.id)).map((r) => `${r.name}:${r.default_algorithm}`).join(" · ") || "—"}）</option>
                  {algos.data?.map((a) => (
                    <option key={a.name} value={a.name}>
                      {a.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field grow">
                <label>说明</label>
                <div className="small muted" style={{ minHeight: 32, display: "flex", alignItems: "center" }}>
                  {algorithm === "recipe"
                    ? "每个配方使用 data/recipes/*.yaml 里的 default_algorithm。"
                    : algos.data?.find((a) => a.name === algorithm)?.description}
                  {algos.data?.find((a) => a.name === algorithm)?.needs_calibration && " 需要校准数据。"}
                </div>
              </div>
            </div>
            {currentDomain && currentDomain.classic_algorithms.length > 0 && (
              <p className="note">
                {currentDomain.display_name} 领域经典方法参考：{currentDomain.classic_algorithms.map((a) => a.name).join("、")}
                （上面的下拉框列出的是本系统实际实现并可选的算法，不是这个参考列表本身）。
              </p>
            )}
            {algos.error && <div className="error-box">{algos.error}</div>}
            </div>
            <div style={{ display: show(3) ? undefined : "none" }}>
            <h2 style={{ marginTop: 16 }} id="step-recipe" className={parentJob && fromStep === "recipe" ? "derive-focus" : ""}>③ 量化配方</h2>
            {inheritedRules.length > 0 && (
              <div className="small" style={{ marginBottom: 8 }} data-testid="inherited-rules">
                沿用父作业的 {inheritedRules.length} 条混合精度规则（{inheritedRules.filter((r) => r.skip).length} 条保持原精度，例如动态量化建议的跳过清单）：
                <span className="mono"> {inheritedRules.slice(0, 3).map((r) => r.match).join("，")}{inheritedRules.length > 3 ? " …" : ""}</span>{" "}
                <button className="secondary" onClick={() => setInheritedRules([])} data-testid="drop-inherited-rules">
                  去掉这些规则
                </button>
              </div>
            )}
            <label className="small" style={{ display: "block", marginBottom: 8 }}>
              <input type="checkbox" checked={noQuant} onChange={(e) => setNoQuant(e.target.checked)} data-testid="opt-no-quant" /> 不量化（保持原始精度，等同于只选 BF16 配方）
            </label>
            {kind !== "llm" && (
              <label className="small" style={{ display: "block", marginBottom: 8 }}>
                <input type="checkbox" checked={quantizeOnly} onChange={(e) => setQuantizeOnly(e.target.checked)} data-testid="opt-quantize-only" /> 仅量化不评测（跳过
                误差归因与评测，只保留量化本身与层级误差追踪）
              </label>
            )}
            <div className="chips" style={{ marginBottom: 8, opacity: noQuant ? 0.5 : 1, pointerEvents: noQuant ? "none" : "auto" }}>
              {recipes.data
                ?.filter((r) => r.weight.format !== "none")
                .map((r) => (
                  <span key={r.id} className={`chip ${picked.includes(r.id) ? "on" : ""}`} onClick={() => toggle(r.id)} title={r.summary} data-testid={`recipe-${r.id}`}>
                    {r.name}
                    {r.maturity === "experimental" && <span className="exp">实验</span>}
                  </span>
                ))}
            </div>
            {(() => {
              const perGroupPicked = (recipes.data ?? []).filter((r) => picked.includes(r.id) && r.weight.granularity === "per_group");
              return (
                <div className="card" style={{ margin: "0 0 12px", padding: 10 }}>
                  <div style={{ cursor: "pointer", fontWeight: 600 }} onClick={() => setAdvancedOpen(!advancedOpen)} data-testid="advanced-params-toggle">
                    {advancedOpen ? "▾" : "▸"} 高级参数
                  </div>
                  {advancedOpen && (
                    <div className="form-row" style={{ marginTop: 8 }}>
                      <div className="field">
                        <label>Group size 覆盖（覆盖所选配方自身的 per-group 分组大小）</label>
                        <input
                          type="number"
                          min={1}
                          value={groupSizeOverride}
                          placeholder={perGroupPicked[0] ? String(perGroupPicked[0].weight.group_size) : "—"}
                          disabled={perGroupPicked.length === 0}
                          onChange={(e) => setGroupSizeOverride(e.target.value)}
                          data-testid="group-size-override"
                        />
                        <p className="small muted" style={{ marginTop: 4 }}>
                          {perGroupPicked.length > 0
                            ? `对以下配方生效：${perGroupPicked.map((r) => `${r.name}（默认 ${r.weight.group_size}）`).join("、")}；非 per-group 配方不受影响。`
                            : "当前所选配方都不是 per-group 粒度（例如 W8A16/FP8 是 per-channel），这个开关对它们没有意义，已禁用。"}
                        </p>
                      </div>
                      <div className="field grow">
                        <label>旋转量化（QuaRot）</label>
                        <select disabled value="">
                          <option value="">未实现（T-0610），敬请期待</option>
                        </select>
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}
            {kind === "vla" ? (
              <label className="small">
                <input type="checkbox" checked={action} onChange={(e) => setAction(e.target.checked)} /> 同时量化 DiT 动作头（默认保持 FP32）
              </label>
            ) : kind === "vlm" ? (
              <p className="note" data-testid="vlm-note">
                VLM 分别量化视觉塔（含视觉融合/投影）与语言模型；lm_head 与 embedding 保持原精度。图像经视觉塔后与问题一起送入语言模型，评测只看<b>答案</b>位置的分布。
              </p>
            ) : kind === "world" || kind === "ewm" ? (
              <p className="note" data-testid="world-note">
                DiT 只量化去噪主干（注意力 + MLP）；patch embedding、时间步条件、adaLN 调制和输出投影保持原精度。
                {kind === "ewm" ? " 动作编码器默认保持原精度，可在第④步单独勾选量化；报告会把误差归因到视频主干/动作路径。" : " 评测以未量化模型在相同随机种子下生成的视频为参照，报告 PSNR/MSE（LPIPS 仅在环境里装了 lpips 时计算）。"}
              </p>
            ) : (
              <div className="form-row">
                <div className="field">
                  <label>激活量化</label>
                  <select value={actStatic} onChange={(e) => setActStatic(e.target.value)} disabled={noQuant}>
                    <option value="">动态 per-token</option>
                    <option value="minmax">静态 per-tensor · minmax</option>
                    <option value="ema">静态 per-tensor · EMA</option>
                    <option value="percentile">静态 per-tensor · 99.99% 分位</option>
                  </select>
                </div>
                <div className="field">
                  <label>KV Cache 量化</label>
                  <select value={kv} onChange={(e) => setKv(e.target.value)} disabled={noQuant}>
                    <option value="">按配方（多数为不量化）</option>
                    <option value="int8">INT8</option>
                    <option value="int4">INT4</option>
                    <option value="int2">INT2</option>
                    <option value="fp8">FP8 E4M3</option>
                  </select>
                </div>
                <div className="field">
                  <label>KV 方法</label>
                  <select value={kvMethod} onChange={(e) => setKvMethod(e.target.value)} disabled={!kv || noQuant}>
                    <option value="kivi">KIVI（K per-channel / V per-token）</option>
                    <option value="naive">naive（K、V 均 per-token）</option>
                  </select>
                </div>
                <div className="field">
                  <label>混合精度：首尾块</label>
                  <select value={keepEnds ? "8" : ""} onChange={(e) => setKeepEnds(e.target.value === "8")} disabled={noQuant || inheritedRules.length > 0} title={inheritedRules.length ? "正在沿用父作业的混合精度规则；去掉后才能用这里的选项" : undefined}>
                    <option value="">同配方</option>
                    <option value="8">首块 + 末块 8 bit</option>
                  </select>
                </div>
                <div className="field">
                  <label>混合精度：down_proj</label>
                  <select value={downBits} onChange={(e) => setDownBits(e.target.value)} disabled={noQuant || inheritedRules.length > 0}>
                    <option value="">同配方</option>
                    <option value="8">8 bit</option>
                    <option value="4">4 bit</option>
                    <option value="fp">保持 FP（不量化）</option>
                  </select>
                </div>
              </div>
            )}
            </div>
            </div>
            {artifact && (
              <p className="note" data-testid="artifact-quant-note">
                ②③ 量化算法与配方：产物 {artifact.format === "gguf" ? `GGUF ${artifact.type}` : `${artifact.recipe} · ${artifact.format}`} 在父作业里已经量化好，这次不再量化。
              </p>
            )}
          </>
        )}
      </div>
      {!blocked && (
      <div className={`card ${parentJob && (fromStep === "eval" || fromStep === "device") ? "derive-focus" : ""}`} id="step-eval" style={{ display: show(4) ? undefined : "none" }}>
        <h2>④ 评估数据 + 设备</h2>
        {parentJob && fromStep === "device" && kind !== "llm" && (
          <p className="note" data-testid="derive-device-note">
            {kind.toUpperCase()} 作业只在 GPU 池上运行（由调度器分配空闲卡），这里能换的是⑤里的目标硬件画像（用于资源预估）。
          </p>
        )}
        {kind === "llm" && evalOnly ? (
          <>
            <div className="form-row" data-testid="eval-only-data">
              <DatasetPicker
                kind="text"
                role="evaluation"
                label="评估数据集（已量化模型直接在这些文本上算 PPL）"
                value={evalTextDataset}
                onChange={setEvalTextDataset}
                onValid={setEvalTextOk}
                testid="eval-dataset"
              />
              <div className="field">
                <label>序列长度</label>
                <input type="number" value={seqLen} min={64} step={64} onChange={(e) => setSeqLen(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>PPL 窗口数</label>
                <input type="number" value={windows} min={1} onChange={(e) => setWindows(Number(e.target.value))} data-testid="eval-windows" />
              </div>
              <div className="field">
                <label>运行设备</label>
                <select value={evalDevice} onChange={(e) => setEvalDevice(e.target.value)} data-testid="eval-device">
                  <option value="cpu">CPU</option>
                  <option value="gpu">GPU 池</option>
                </select>
              </div>
            </div>
            {!evalGguf && offlineBenchChips}
            {evalGguf && (
              <p className="small muted" data-testid="eval-gguf-note">
                GGUF 由 llama.cpp 的 <span className="mono">llama-perplexity</span> 直接计算；选择 llama.cpp 后端还会额外起
                llama-server 做冒烟与压测。
              </p>
            )}
          </>
        ) : kind === "llm" && artifact ? (
          <>
          {parentEval && (windows !== parentEval.windows || seqLen !== parentEval.seq_len) && (
            <p className="note" data-testid="artifact-eval-differs" style={{ borderLeft: "3px solid #e0a84a", paddingLeft: 10 }}>
              评测设置（{windows} 窗口 × {seqLen}）与父作业（{parentEval.windows} × {parentEval.seq_len}）不同：部署后 PPL 仍会测，但不会给出「相对父作业」的差异，因为不同设置的 PPL 不可比。
            </p>
          )}
          <div className="form-row" data-testid="artifact-eval">
            <div className="field">
              <label>序列长度（wikitext-2 test）</label>
              <input type="number" value={seqLen} min={64} step={64} onChange={(e) => setSeqLen(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>PPL 窗口数</label>
              <input type="number" value={windows} min={1} onChange={(e) => setWindows(Number(e.target.value))} data-testid="artifact-windows" />
            </div>
            <div className="field">
              <label>运行设备</label>
              {isGguf ? (
                <select value={evalDevice} onChange={(e) => setEvalDevice(e.target.value)} data-testid="artifact-device">
                  <option value="cpu">CPU（llama.cpp 全部在 CPU 上）</option>
                  <option value="gpu">GPU 池（全部层卸载到 GPU）</option>
                </select>
              ) : (
                <div className="small muted" style={{ minHeight: 32, display: "flex", alignItems: "center" }}>GPU 池（vLLM / SGLang 需要 GPU）</div>
              )}
            </div>
          </div>
          {!isGguf && offlineBenchChips}
          </>
        ) : kind !== "llm" ? (
          <>
            <div className="form-row">
              <DatasetPicker
                kind={kind === "world" || kind === "ewm" ? "world" : "lerobot"}
                role="calibration"
                label={kind === "vlm" ? "校准数据（synthetic：程序绘制的图形场景；或 LeRobot v3 目录：真实图像）" : kind === "world" || kind === "ewm" ? "校准数据（synthetic-video：自带真实未来帧；或 .npz 视频片段目录）" : "校准数据（synthetic，或 LeRobot v3 目录）"}
                value={dataset}
                onChange={setDataset}
                onValid={kind === "world" || kind === "ewm" ? setCalibWorldOk : setCalibVisionOk}
                defaultValue={fromJob ? undefined : kind === "world" || kind === "ewm" ? { path: "synthetic-video", fallback: "synthetic-video" } : { path: DEFAULT_DATASET, fallback: "synthetic" }}
                testid="dataset-path"
              />
              <DatasetPicker
                kind={kind === "world" || kind === "ewm" ? "world" : "lerobot"}
                role="evaluation"
                label={kind === "vlm" ? "评估数据（答案已知的 synthetic 可算准确率；真实图像以未量化模型自己的回答为参照）" : kind === "world" || kind === "ewm" ? "评估数据（与 FP 输出比较 PSNR；synthetic-video 还能对真实未来帧打分）" : "评估数据（可与校准数据不同；相同来源时自动切成互不重叠的两份）"}
                value={evalDataset}
                onChange={setEvalDataset}
                onValid={kind === "world" || kind === "ewm" ? setEvalWorldOk : setEvalVisionOk}
                defaultValue={fromJob ? undefined : kind === "world" || kind === "ewm" ? { path: "synthetic-video", fallback: "synthetic-video" } : { path: DEFAULT_DATASET, fallback: "synthetic" }}
                testid="eval-dataset-path"
              />
              <div className="field">
                <label>校准样本</label>
                <input type="number" value={nCalib} min={1} onChange={(e) => setNCalib(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>评估样本</label>
                <input type="number" value={nEval} min={1} onChange={(e) => setNEval(Number(e.target.value))} />
              </div>
              {kind === "vlm" && (
                <div className="field">
                  <label>最长回答（token）</label>
                  <input type="number" value={maxNewTokens} min={1} max={256} onChange={(e) => setMaxNewTokens(Number(e.target.value))} data-testid="vlm-max-new-tokens" />
                </div>
              )}
              {(kind === "world" || kind === "ewm") && (
                <div className="field">
                  <label>视频帧数</label>
                  <input type="number" value={frames} min={2} max={32} onChange={(e) => setFrames(Number(e.target.value))} data-testid="world-frames" />
                </div>
              )}
            </div>
            <label className="small">
              <input type="checkbox" checked={attribution && !quantizeOnly} disabled={quantizeOnly} onChange={(e) => setAttribution(e.target.checked)} /> 分阶段误差归因（逐模态单独量化，定位误差来源环节）
              {quantizeOnly && <span className="muted"> （仅量化不评测已关闭，需要未量化基线）</span>}
            </label>
            {kind === "ewm" && (
              <label className="small" style={{ display: "block", marginTop: 6 }}>
                <input type="checkbox" checked={quantizeAction} onChange={(e) => setQuantizeAction(e.target.checked)} data-testid="ewm-quantize-action" /> 同时量化动作编码器（默认保持原精度；归因会单列动作路径）
              </label>
            )}
          </>
        ) : (
          <>
            <div className="form-row">
              <DatasetPicker kind="text" role="calibration" label={usePaste ? "校准数据（下面粘贴的文本优先，这里暂不用）" : "校准数据"} value={calibText} onChange={setCalibText} onValid={setCalibTextOk} testid="calib-dataset" />
              <div className="field">
                <label>校准样本（窗口数）</label>
                <input type="number" value={nCalib} min={1} onChange={(e) => setNCalib(Number(e.target.value))} data-testid="calib-n" />
              </div>
              <div className="field">
                <label>序列长度</label>
                <input type="number" value={seqLen} min={64} step={64} onChange={(e) => setSeqLen(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>PPL 窗口数</label>
                <input type="number" value={windows} min={1} onChange={(e) => setWindows(Number(e.target.value))} disabled={noEval} />
              </div>
            </div>
            {!noEval && (
              <div className="form-row">
                <DatasetPicker
                  kind="text"
                  role="evaluation"
                  label="评估数据（默认 wikitext-2 test：跟别的作业比 PPL 才有意义；换成别的来源，对比页只会跟同来源的作业放一起）"
                  value={evalTextDataset}
                  onChange={setEvalTextDataset}
                  onValid={setEvalTextOk}
                  testid="eval-dataset"
                />
              </div>
            )}
            <div data-testid="calib-paste-section" style={{ marginBottom: 8 }}>
              <label className="small">
                <input type="checkbox" checked={usePaste} onChange={(e) => setUsePaste(e.target.checked)} data-testid="calib-paste-toggle" /> 直接粘贴校准文本（代替上面的数据集）
              </label>
              {usePaste && (
                <>
                  <textarea
                    value={pasteText}
                    maxLength={PASTE_MAX}
                    rows={8}
                    style={{ width: "100%", fontFamily: "inherit" }}
                    placeholder={"把要校准的文字贴在这里。空一行分成一段（一个样本）；窗口随机取自拼接后的全文。"}
                    onChange={(e) => setPasteText(e.target.value)}
                    data-testid="calib-paste"
                  />
                  <p className={`note ${pasteSamples.length ? "" : "warn"}`} data-testid="calib-paste-status">
                    {pasteSamples.length ? `✓ ${pasteSamples.length} 段 · ${pasteText.length} 字符` : "填一点文字：全是空白不能校准。"}
                    {pasteText.length >= PASTE_MAX && "（已到上限，多出的不会收）"}
                  </p>
                  {pasteSamples.length > 0 && pasteSamples.join("\n\n").length / 3 < seqLen + 2 && (
                    <p className="note warn" data-testid="calib-paste-short">
                      可能太短：一个窗口要 {seqLen + 2} 个 token，约 {(seqLen + 2) * 3} 个字符起（英文）。太短的话作业在数据准备阶段失败，并说明有多少 token。
                    </p>
                  )}
                </>
              )}
            </div>
            <div className="form-row">
              <div className="field">
                <label>敏感度分析（逐组回退 FP，8 窗口 PPL）</label>
                <select value={sens} onChange={(e) => setSens(e.target.value)} disabled={noEval}>
                  <option value="">不做</option>
                  <option value="block">按块（每个 decoder block）</option>
                  <option value="type">按层类型（q/k/v/o/gate/up/down）</option>
                </select>
              </div>
              <div className="field">
                <label>量化设备</label>
                <select value={quantDevice} onChange={(e) => setQuantDevice(e.target.value)} data-testid="quant-device-select">
                  <option value="cpu">CPU（不占 GPU 池）</option>
                  <option value="gpu">GPU 池（{cfg.data?.gpus.join(", ") ?? "…"}）</option>
                </select>
              </div>
              <div className="field">
                <label>评估设备</label>
                <select value={evalDevice} onChange={(e) => setEvalDevice(e.target.value)} data-testid="eval-device-select">
                  <option value="cpu">CPU（不占 GPU 池）</option>
                  <option value="gpu">GPU 池（{cfg.data?.gpus.join(", ") ?? "…"}）</option>
                </select>
              </div>
            </div>
            <div className="form-row">
              <div className="field">
                <label>
                  <input type="checkbox" checked={probe} disabled={noEval} onChange={(e) => setProbe(e.target.checked)} data-testid="opt-probe" /> logits 探针（量化前后输出分布的 KL 与 Top-1 一致率）
                </label>
                <label className="small">
                  <input type="checkbox" checked={probeExact} disabled={!probe || noEval} onChange={(e) => setProbeExact(e.target.checked)} data-testid="opt-probe-exact" /> 精确 KL（保存全部 logits，约 1.2 GiB / 8 窗口；默认只存 top-256，KL 为下界）
                </label>
              </div>
            </div>
            <p className="note">PPL = 不重叠窗口的 token 级困惑度；量化与评估设备可以不同。</p>
          </>
        )}
      </div>
      )}
      {!blocked && (
      <div className={`card ${parentJob && fromStep === "deploy" ? "derive-focus" : ""}`} id="step-deploy" style={{ display: show(5) ? undefined : "none" }}>
        <h2>⑤ {llmQuantOnly ? "导出量化结果" : evalOnly ? "可选：推理框架部署" : "部署与导出"}</h2>
        <div style={{ marginBottom: 14 }}>
          <label className="lbl">目标部署硬件 · 点卡片即可选中（用于资源预估与部署适配）</label>
          <HardwarePicker
            value={hw}
            onChange={setHw}
            profiles={profiles.data ?? []}
            detected={localHardware.data?.devices ?? []}
            compact
          />
          <div className="form-row" style={{ marginTop: 8, marginBottom: 0 }}>
            <div className="field" style={{ minWidth: 280, maxWidth: 420 }}>
              <label>精确 ID（自动化 / 自定义画像）</label>
              <select value={hw} onChange={(e) => setHw(e.target.value)} data-testid="hw-select">
                {profiles.data?.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}（{p.id}）
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
        {kind === "llm" && llmQuantOnly && (
          <div className="card" style={{ margin: "8px 0 12px", padding: 12 }} data-testid="quantize-export">
            {exportChips}
            <p className="small muted" style={{ marginBottom: 0 }}>
              量化结果会写到 <span className="mono">runs/&lt;job&gt;/export/&lt;recipe&gt;-&lt;format&gt;/</span>，
              之后可以在“仅评测”模式里直接加载，不用再量化一次。不选格式则只跑量化，权重不落盘。
            </p>
          </div>
        )}
        {(kind === "vla" || kind === "world" || kind === "ewm") && (
          <div className="card" style={{ margin: "8px 0 12px", padding: 14 }} data-testid="native-serve-card">
            <div className="card-head" style={{ marginBottom: 8 }}>
              <div>
                <h2>原生连续输出服务</h2>
                <p className="note" style={{ margin: 0 }}>
                  {kind === "vla" ? "VLA 输出动作向量：/v1/actions" : `${kind.toUpperCase()} 输出视频帧：/v1/video`} · 直接服务当前内存中的量化模型
                </p>
              </div>
              <span className={`status ${useNative ? "good" : "muted"}`}>{useNative ? "已启用" : "未启用"}</span>
            </div>
            <label className="small" style={{ display: "block", marginBottom: 8 }}>
              <input type="checkbox" checked={useNative} onChange={(e) => setUseNative(e.target.checked)} data-testid="opt-native-serve" /> 在本作业中启动原生理服务并做 smoke / 并发压测
            </label>
            {useNative && (
              <div className="form-row" style={{ margin: 0 }}>
                <div className="field" style={{ width: 150 }}>
                  <label>并发</label>
                  <input type="number" min={1} value={nativeConc} onChange={(e) => setNativeConc(Math.max(1, Number(e.target.value)))} data-testid="native-concurrency" />
                </div>
                <div className="field" style={{ width: 150 }}>
                  <label>请求数</label>
                  <input type="number" min={1} value={nativeReq} onChange={(e) => setNativeReq(Math.max(1, Number(e.target.value)))} data-testid="native-requests" />
                </div>
                <div className="field grow">
                  <label>&nbsp;</label>
                  <div className="small muted" style={{ minHeight: 32, display: "flex", alignItems: "center" }}>
                    作业结束后服务自动停止；结果写入每个配方的 deploy.native。
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        {currentDomain && !blocked && !llmQuantOnly && (
          <div data-testid="backend-section" style={{ marginTop: 10, display: (kind === "vla" || kind === "world" || kind === "ewm") ? "none" : undefined }}>
            <label className="small">部署后端（真实部署并实测；可以同时选几个来对比）</label>
            <BackendCards
              env={backendEnv.data}
              sel={{
                vllm: artifact ? deployBackend === "vllm" : useVllm,
                sglang: artifact ? deployBackend === "sglang" : useSglang,
                llamacpp: artifact ? deployBackend === "llamacpp" : gguf.length > 0,
              }}
              onToggle={toggleBackend}
              single={!!artifact}
              allowed={artifact ? (isGguf ? ["llamacpp"] : ["vllm", "sglang"])
                : evalOnly ? (evalGguf ? ["llamacpp"] : ["vllm", "sglang"])
                : domainBackends}
              disabled={quantizeOnly || domainBackends.length === 0}
              disabledNote={domainBackends.length === 0 ? `${kind.toUpperCase()} 暂不能部署到推理后端：${currentDomain.deploy_note || "该领域没有声明可用后端。"}` : "仅量化不评测已开启：导出/部署需要评测得到的 fake-quant 基线，本次跳过。"}
              unavailableNote={artifact ? undefined : `${kind.toUpperCase()} 领域不支持这个后端。`}
              options={{
                llamacpp: artifact || evalOnly ? undefined : (
                  <div>
                    <div className="small muted">GGUF 类型（llama.cpp 自己量化，数值与上面的配方无关）</div>
                    <div className="chips">
                      {["q8_0", "q5_0", "q4_0", "iq4_nl", "q4_k", "q2_k"].map((f) => (
                        <span key={f} className={`chip ${gguf.includes(f) ? "on" : ""}`} onClick={() => setGguf(gguf.includes(f) ? gguf.filter((x) => x !== f) : [...gguf, f])}>
                          {f.toUpperCase()}
                        </span>
                      ))}
                    </div>
                  </div>
                ),
                vllm: (
                  <>
                    {!artifact && !evalOnly && exportChips}
                    <label className="small">
                      显存利用率 <input type="number" step={0.05} min={0.1} max={0.95} value={vllmUtil} style={{ width: 70 }} onChange={(e) => setVllmUtil(Number(e.target.value))} data-testid="opt-vllm-util" />
                    </label>
                    {lmEvalControls}
                    {evalScopeChips}
                  </>
                ),
                sglang: (
                  <>
                    {!artifact && !evalOnly && exportChips}
                    <label className="small">
                      静态显存占比 <input type="number" step={0.05} min={0.1} max={0.95} value={sglangMem} style={{ width: 70 }} onChange={(e) => setSglangMem(Number(e.target.value))} data-testid="opt-sglang-mem" />
                    </label>
                    {!useVllm && lmEvalControls}
                    {evalScopeChips}
                  </>
                ),
              }}
            />
            {servingNotes.length > 0 && (
              <ul className="note warn" data-testid="serve-warnings">
                {servingNotes.map((n) => (
                  <li key={n.recipe}>
                    <b>{n.recipe.toUpperCase()}</b> 在 {n.backend === "sglang" ? "SGLang" : "vLLM"} 上服务不了：{n.reason}。作业仍会导出它，这一步会失败或被跳过；不需要的话在第③步把它去掉。
                  </li>
                ))}
              </ul>
            )}
            {suggestion && (
              <div className="small muted" data-testid="export-suggestion">
                {deployBackend === "llamacpp" ? "llama.cpp" : deployBackend === "sglang" ? "SGLang" : "vLLM"} 的建议格式：{suggestion.format || "—"}（{suggestion.verdict}
                {suggestion.reason ? `，${suggestion.reason}` : ""}）
              </div>
            )}
          </div>
        )}
        {kind === "llm" && (artifact ? !isGguf : useVllm || useSglang) && (
          <div className="form-row" data-testid="sweep-options" style={{ marginTop: 8 }}>
            <div className="field" style={{ minWidth: 340 }}>
              <label>
                <input type="checkbox" checked={useSweep} onChange={(e) => setUseSweep(e.target.checked)} data-testid="opt-sweep" /> 并发扫描（吞吐 / TTFT / TPOT / ITL 随并发变化，找出饱和点）
              </label>
              {useSweep && (
                <div className="small" style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 4 }}>
                  并发档 <input value={sweepLevels} onChange={(e) => setSweepLevels(e.target.value)} style={{ width: 150 }} placeholder="1,2,4,8,16" data-testid="opt-sweep-levels" aria-invalid={!levelsOk} />
                  TTFT p99 预算 <input type="number" min={0} value={sweepBudget} onChange={(e) => setSweepBudget(e.target.value)} style={{ width: 80 }} placeholder="不限" data-testid="opt-sweep-budget" /> ms
                  {!levelsOk && <span className="warn" style={{ color: "#e0a84a" }}>请填用逗号分隔、互不相同的正整数</span>}
                </div>
              )}
            </div>
          </div>
        )}
        {domainBackends.length > 0 && !evalOnly && !llmQuantOnly && (
          <p className="note">勾选部署后端后，导出格式会按兼容矩阵自动建议，仍可手动调整。</p>
        )}
      </div>
      )}
      <div className="card" style={{ display: show(6) ? undefined : "none" }} data-testid="step-confirm">
        <h2>⑥ 确认提交 · JobSpec 预览</h2>
        {!singlePage && <StepSummary rows={[
          ["领域 / 模型", `${domain.toUpperCase()} · ${model}${evalOnly && quantizedPath ? ` → ${quantizedPath}` : ""}`],
          ["算法 / 配方", evalOnly ? `已量化 · ${quantFormat === "auto" ? "自动识别" : quantFormat}` : artifact ? `沿用父作业产物 ${artifact.format === "gguf" ? `GGUF ${artifact.type}` : `${artifact.recipe} · ${artifact.format}`}` : `${algorithm === "recipe" ? "按配方默认" : algorithm} · ${picked.map((x) => x.toUpperCase()).join("、")}`],
          ["校准数据", evalOnly ? "—" : kind === "llm" ? `${usePaste && pasteSamples.length ? `粘贴的文本 ${pasteSamples.length} 段` : calibText === "wikitext-2-raw-v1:train" ? "wikitext-2 train" : calibText} · ${nCalib} 段 × ${seqLen}` : `${dataset} · ${nCalib} 样本${kind === "world" || kind === "ewm" ? ` × ${frames} 帧` : ""}`],
          ["评估", kind === "llm" ? (evalOnly ? `${evalTextDataset} · ${windows} 窗口 × ${seqLen}` : noEval ? "仅量化不评测" : `${artifact ? "wikitext-2 test" : evalTextDataset === "wikitext-2-raw-v1:test" ? "wikitext-2 test" : evalTextDataset} · ${windows} 窗口 × ${seqLen}`) : `${evalDataset} · ${nEval} 样本${kind === "vlm" ? ` · 最长回答 ${maxNewTokens}` : kind === "world" || kind === "ewm" ? ` × ${frames} 帧${kind === "ewm" ? " · 动作条件" : ""}` : ""}`],
          ["设备", kind === "llm" ? (evalOnly ? evalDevice.toUpperCase() : artifact ? (isGguf ? evalDevice.toUpperCase() : "GPU 池") : `量化 ${quantDevice.toUpperCase()} · 评估 ${evalDevice.toUpperCase()}`) : "GPU 池"],
          ["部署", useNative ? "原生连续输出服务" : !noEval && chosenBackends.length ? `${chosenBackends.join(" + ")}${!evalOnly && exportFmts.length ? ` · 导出 ${exportFmts.join("/")}` : ""}${gguf.length ? ` · GGUF ${gguf.join("/")}` : ""}` : "不部署"],
        ]} />}
        <pre className="log" style={{ height: "auto", maxHeight: 240 }}>
          {JSON.stringify(job, null, 2)}
        </pre>
        <button disabled={busy || blocked || (!evalOnly && picked.length === 0) || (evalOnly && !quantizedPath) || (useSweep && !levelsOk) || !dataOk} onClick={submit} title={useSweep && !levelsOk ? "并发档格式不对" : !dataOk ? "校准 / 评估数据的路径还不能用（见第④步）" : evalOnly && !quantizedPath ? "请填写已量化模型路径" : undefined}>
          {busy ? "提交中…" : blocked ? "该领域尚未支持提交" : "提交作业"}
        </button>
      </div>
      {!singlePage && (
        <div className="form-row" style={{ justifyContent: "space-between", marginTop: 4 }} data-testid="step-buttons">
          <button className="secondary" disabled={step <= 1} onClick={() => setStep(prevStep(step, !!artifact || evalOnly))} data-testid="step-prev">
            ← 上一步
          </button>
          <span className="small muted">第 {step} / 6 步</span>
          {step < 6 ? (
            <button disabled={blocked || (step === 1 && ((!model && !evalOnly) || (evalOnly && !quantizedPath))) || (step === 3 && !evalOnly && picked.length === 0)} onClick={() => setStep(nextStep(step, !!artifact || evalOnly))} data-testid="step-next">
              下一步 →
            </button>
          ) : (
            <span />
          )}
        </div>
      )}
      </div>
      <aside className="card" style={{ position: "sticky", top: 12 }} data-testid="live-estimate">
        {kind === "world" || kind === "ewm" ? (
          <>
            <h2>生成作业概览</h2>
            <p className="note" data-testid="world-estimate">
              WM / EWM 的产物是未来视频帧，不走 vLLM / SGLang / llama.cpp 的文本推理路径，因此没有 KV Cache 与并发预估。
            </p>
            <table>
              <tbody>
                <tr>
                  <th>模型</th>
                  <td className="mono small">{model.split("/").pop() || "—"}</td>
                </tr>
                <tr>
                  <th>片段</th>
                  <td>{nEval} 段 × {frames} 帧</td>
                </tr>
                <tr>
                  <th>采样步数</th>
                  <td>8</td>
                </tr>
                <tr>
                  <th>配方</th>
                  <td>{picked.map((x) => x.toUpperCase()).join(" · ") || "—"}</td>
                </tr>
                <tr>
                  <th>评测口径</th>
                  <td>BF16 同种子参考</td>
                </tr>
                {kind === "ewm" && (
                  <tr>
                    <th>动作编码器</th>
                    <td>{quantizeAction ? "随配方量化" : "保持原精度"}</td>
                  </tr>
                )}
              </tbody>
            </table>
            <p className="note">
              报告会给出 <b>PSNR / MSE</b>（量化 vs BF16）、同模型换种子的采样噪声下限，以及逐模态误差归因。
            </p>
          </>
        ) : (
          <>
            <h2>实时预估</h2>
            <p className="note">{hw} · 并发 1 · 不加载权重</p>
            <select value={estBackend} onChange={(e) => setEstBackend(e.target.value)} data-testid="live-backend" aria-label="推理后端画像">
              {(backendList.data ?? [{ id: "hf_transformers" }]).map((b) => (
                <option key={b.id} value={b.id}>
                  后端：{b.id}
                </option>
              ))}
            </select>
            {est.loading && <p className="note">计算中…</p>}
            {est.error && <div className="error-box" data-testid="live-estimate-error">{est.error}</div>}
            {est.data && (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>配方</th>
                      <th className="num">权重</th>
                      <th className="num">峰值</th>
                      <th>可部署</th>
                    </tr>
                  </thead>
                  <tbody>
                    {est.data.estimates.map((e) => (
                      <tr key={e.recipe_id} data-testid={`live-${e.recipe_id}`}>
                        <td>{e.recipe}</td>
                        <td className="num">{fmtBytes(e.breakdown.weights)}</td>
                        <td className="num">{fmtBytes(e.peak_bytes)}</td>
                        <td>
                          {e.feasible ? (
                            <span className="status good">✓</span>
                          ) : (
                            <span className="status critical" data-tip={e.reasons[0] ?? "超出显存"}>✗</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {est.data && (
              <p className="note" data-testid="resource-probe-note">
                可用 {fmtBytes(est.data.estimates[0]?.capacity_bytes)} · 利用率 0.9 · 实时预估。{probeNote(est.data.resource_probe)}
              </p>
            )}
          </>
        )}
      </aside>
      </div>
    </>
  );
}

// ---- T-1304: catalog-backed model picker, replacing free-text-only entry for LLM-domain jobs ----
// Grouped two levels deep (category -> family_group, T-1303) so same-series different-size variants
// collapse into one dropdown group instead of 200+ flat rows. Picking an entry that is not available
// locally downloads it (reusing the same /api/v1/download machinery Fit.tsx's HubPanel already uses, not
// a new download path) and fills the model path once it lands.
function CatalogModelPicker({ onPick, currentPath }: { onPick: (path: string) => void; currentPath: string }) {
  const catalog = useAsync(() => api.catalog("", "", "", 500), []);
  const [picked, setPicked] = useState("");
  const [dl, setDl] = useState<DownloadStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!dl || (dl.state !== "running" && dl.state !== "queued")) return;
    const t = setInterval(() => {
      api
        .downloadStatus(dl.id)
        .then((s) => {
          setDl(s);
          if (s.state === "succeeded") onPick(s.dir);
        })
        .catch(() => {});
    }, 1000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dl?.id, dl?.state]);

  const groups = useMemo(() => {
    const byKey = new Map<string, CatalogModel[]>();
    for (const m of catalog.data?.models ?? []) {
      const key = `${m.category || "Other"} · ${m.family_group || m.name}`;
      const arr = byKey.get(key);
      if (arr) arr.push(m);
      else byKey.set(key, [m]);
    }
    return [...byKey.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [catalog.data]);

  const start = async () => {
    if (!picked) return;
    setErr(null);
    setBusy(true);
    try {
      const d = await api.download(picked);
      setDl(d);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="form-row">
      <div className="field grow">
        <label>模型目录（{catalog.data?.total ?? 0} 条，按领域 · 系列分组）</label>
        <select
          value={picked}
          onChange={(e) => {
            setPicked(e.target.value);
            setDl(null);
            setErr(null);
          }}
          data-testid="catalog-model-select"
        >
          <option value="">— 选择一个模型 —</option>
          {groups.map(([group, models]) => (
            <optgroup key={group} label={group}>
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} ({m.parameter_count}, {m.quantization})
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        {catalog.error && <div className="error-box">{catalog.error}</div>}
      </div>
      <div className="field">
        <label>&nbsp;</label>
        <button onClick={start} disabled={!picked || busy || dl?.state === "running" || dl?.state === "queued"} data-testid="catalog-download">
          {dl?.state === "succeeded" ? "已下载" : busy ? "启动中…" : "下载并使用"}
        </button>
      </div>
      {err && <div className="error-box">{err}</div>}
      {dl && (
        <div className="field grow">
          <Meter pct={dl.bytes_total ? (100 * dl.bytes_done) / dl.bytes_total : 0} />
          <span className="muted small">
            {fmtBytes(dl.bytes_done)} / {fmtBytes(dl.bytes_total)} · {dl.state}
            {dl.state === "succeeded" && ` · ${dl.dir}`}
            {dl.error && ` · ${dl.error}`}
          </span>
        </div>
      )}
      {!dl && currentPath && (
        <p className="note" style={{ width: "100%" }}>
          当前使用路径：{currentPath}
        </p>
      )}
    </div>
  );
}

const RISK: Record<Advice["risk"], string> = { low: "低", medium: "中", high: "高" };
const METRIC: Record<string, (v: number) => string> = {
  ppl_delta_pct: (v) => `PPL ${v >= 0 ? "+" : ""}${v.toFixed(2)}%`,
  top1_agreement: (v) => `答案 Top-1 一致 ${(v * 100).toFixed(1)}%`,
  action_rel_l2_over_noise_floor: (v) => `动作误差 ${v.toFixed(2)}× 噪声下限`,
};

/** 部署建议：先看目标设备能装下什么、精度风险多大，再决定怎么量化、用哪个推理框架。 */
function AdvicePanel({ model, kind, hw, profiles, onAdopt }: {
  model: string; kind: "llm" | "vlm" | "vla"; hw: string; profiles: { id: string; name: string }[]; onAdopt: (a: Advice) => void;
}) {
  const devices = useAsync(api.hardware, []);
  const backendList = useAsync(api.backends, []);
  const detected = devices.data?.devices.find((d) => d.kind === "gpu" && (d as { profile?: string }).profile) as ({ name: string; profile?: string } | undefined);
  const [target, setTarget] = useState<string | null>(null);
  const [objective, setObjective] = useState<AdviceObjective>("balanced");
  const [backends, setBackends] = useState<string[]>(["hf_transformers", "vllm"]);
  const [conc, setConc] = useState(8);
  const [ctx, setCtx] = useState(4096);
  const [res, setRes] = useState<Awaited<ReturnType<typeof api.recommend>> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const tgt = target ?? detected?.profile ?? hw;
  const run = async () => {
    setErr(null);
    setBusy(true);
    try {
      const rt = kind === "llm" ? { concurrency: conc, context_tokens: ctx } : { concurrency: conc, images_per_step: 2, text_tokens: 42 };
      setRes(await api.recommend(model, [tgt], backends, objective, rt));
    } catch (e) {
      setErr((e as Error).message);
      setRes(null);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="card" style={{ margin: "12px 0", padding: 12 }} data-testid="advice-card">
      <h2 style={{ marginTop: 0 }}>部署建议（按目标设备推荐量化方式与推理框架）</h2>
      <div className="form-row">
        <div className="field">
          <label>目标设备{detected ? `（本机检测到 ${detected.name}）` : ""}</label>
          <select value={tgt} onChange={(e) => setTarget(e.target.value)} data-testid="advice-hw">
            {profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>目标</label>
          <select value={objective} onChange={(e) => setObjective(e.target.value as AdviceObjective)} data-testid="advice-objective">
            <option value="balanced">均衡（排除高风险配方后比并发）</option>
            <option value="accuracy">精度优先</option>
            <option value="throughput">吞吐优先（只看能装下多少并发）</option>
          </select>
        </div>
        <div className="field">
          <label>并发</label>
          <input type="number" min={1} value={conc} onChange={(e) => setConc(Number(e.target.value))} data-testid="advice-conc" />
        </div>
        {kind === "llm" && (
          <div className="field">
            <label>上下文（token）</label>
            <input type="number" min={128} step={512} value={ctx} onChange={(e) => setCtx(Number(e.target.value))} />
          </div>
        )}
        <div className="field">
          <label>候选推理框架</label>
          <div className="chips">
            {(backendList.data ?? []).map((b) => (
              <span key={b.id} className={`chip ${backends.includes(b.id) ? "on" : ""}`} data-testid={`advice-backend-${b.id}`}
                onClick={() => setBackends(backends.includes(b.id) ? backends.filter((x) => x !== b.id) : [...backends, b.id])}>
                {b.id}
              </span>
            ))}
          </div>
        </div>
        <div className="field" style={{ justifyContent: "flex-end" }}>
          <button onClick={run} disabled={busy || !model || backends.length === 0} data-testid="advice-run">
            {busy ? "计算中…" : "生成建议"}
          </button>
        </div>
      </div>
      <p className="small muted">
        SGLang 与 llama.cpp 还没有估算画像，暂不参与推荐。{kind !== "llm" && "VLM/VLA 作业目前只做伪量化评测、不做推理框架部署；这里的推理框架只用于显存与并发估算。"}
      </p>
      {err && <div className="error-box">{err}</div>}
      {res && (
        <>
          <table data-testid="advice-table">
            <thead>
              <tr>
                <th>#</th>
                <th>配方</th>
                <th>算法</th>
                <th>推理框架</th>
                <th>导出格式</th>
                <th>峰值 / 容量</th>
                <th>最大并发</th>
                <th>精度风险</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {res.top.map((a, i) => (
                <tr key={`${a.recipe}-${a.backend}-${a.hardware}`} data-testid={`advice-row-${i}`}>
                  <td>{i + 1}</td>
                  <td>
                    {a.recipe}
                    {a.maturity === "experimental" && <span className="exp">实验</span>}
                  </td>
                  <td>{a.algorithm}</td>
                  <td>
                    {a.backend}
                    {a.deploy_verdict !== "native" && <span className="muted">（{a.deploy_verdict}）</span>}
                  </td>
                  <td>{a.export_format || "—"}</td>
                  <td className="num">
                    {fmtBytes(a.peak_bytes)} / {fmtBytes(a.capacity_bytes)}
                  </td>
                  <td className="num">{a.max_concurrency}</td>
                  <td data-testid={`advice-risk-${i}`}>
                    <b>{RISK[a.risk]}</b>{" "}
                    {a.risk_source === "measured" && a.measured ? (
                      <span className="small">
                        实测：{(METRIC[a.measured.metric] ?? ((v: number) => String(v)))(a.measured.value)}（{a.measured.algorithm}，
                        <a href={`/jobs/${a.measured.job_id}`}>作业</a>）
                      </span>
                    ) : (
                      <span className="small muted">经验值（按位宽，未实测）</span>
                    )}
                  </td>
                  <td>
                    <button className="secondary" onClick={() => onAdopt(a)} data-testid={`advice-adopt-${i}`}>
                      采用
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">
            显存与并发来自估算器；精度风险优先用这个模型已完成作业的实测结果（共 {res.measured_jobs} 个配方有实测），否则按位宽给经验值（≥8 bit 低，4 bit 中，≤3 bit 或 4 bit 激活为高）。「采用」只是填好配方、算法、硬件{kind === "llm" ? "与部署后端" : ""}，提交前仍可修改；{res.excluded} 个组合因装不下被排除。
          </p>
        </>
      )}
    </div>
  );
}

type Artifact = { job: string; recipe?: string; format: string; type?: string };
type ArtifactOpt = { key: string; label: string; artifact: Artifact };
const sameArtifact = (a: Artifact, b: Artifact) => a.job === b.job && a.format === b.format && (a.recipe ?? "") === (b.recipe ?? "") && (a.type ?? "") === (b.type ?? "");

/** The checkpoints a finished LLM job left that another job can serve again: HF exports per recipe, and its GGUF files. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function artifactOptions(job: string, rep: Record<string, any> | null | undefined): ArtifactOpt[] {
  if (!rep || rep.kind !== "llm") return [];
  const out: ArtifactOpt[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  for (const r of Object.values<any>(rep.recipes ?? {}))
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    for (const [fmt_, e] of Object.entries<any>(r.export ?? {}))
      if (e?.status === "exported")
        out.push({ key: `${r.recipe}/${fmt_}`, label: `${r.recipe} · ${fmt_}${e.real_ppl != null ? `（真实权重 PPL ${Number(e.real_ppl).toFixed(3)}）` : ""}`, artifact: { job, recipe: r.recipe, format: fmt_ } });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  for (const [ty, g] of Object.entries<any>(rep.llamacpp?.results ?? {}))
    out.push({ key: `gguf/${ty}`, label: `GGUF ${ty}${g?.ppl != null ? `（llama.cpp PPL ${Number(g.ppl).toFixed(3)}）` : ""}`, artifact: { job, format: "gguf", type: ty } });
  return out;
}

const STEP_TITLES = ["领域 + 模型", "量化算法", "量化配方", "评估数据 + 设备", "部署与导出", "确认提交"];
// reusing a parent's export, or evaluating an existing checkpoint, skips ②③ (nothing is quantized)
const nextStep = (n: number, skipQuant: boolean) => (skipQuant && n === 1 ? 4 : Math.min(6, n + 1));
const prevStep = (n: number, skipQuant: boolean) => (skipQuant && n === 4 ? 1 : Math.max(1, n - 1));

function StepNav({ step, setStep, singlePage, setSinglePage, blocked, skipQuant, skipReason, canLeave, summary }: {
  step: number; setStep: (n: number) => void; singlePage: boolean; setSinglePage: (v: boolean) => void; blocked: boolean;
  skipQuant: boolean; skipReason: string;
  canLeave: Record<number, boolean>; summary: Record<number, string>;
}) {
  return (
    <div className="stepper card" data-testid="stepper">
      <ol>
        {STEP_TITLES.map((t, i) => {
          const n = i + 1;
          const skipped = skipQuant && (n === 2 || n === 3);
          const state = singlePage ? "all" : n === step ? "current" : n < step ? "done" : "todo";
          const reachable = !blocked || n === 1;
          return (
            <li key={n} className={`${state} ${skipped ? "skipped" : ""}`} onClick={() => reachable && !skipped && setStep(n)} data-testid={`step-tab-${n}`} aria-current={state === "current" ? "step" : undefined}>
              <span className="n">{state === "done" ? "✓" : n}</span>
              <span className="t">
                <b>{t}</b>
                {!singlePage && (skipped ? <small>{skipReason}</small> : summary[n] && n !== step && n < 6 ? <small title={summary[n]}>{summary[n]}</small> : null)}
                {!singlePage && n === step && n < 6 && canLeave[n] === false && <small className="warn">请先完成本步</small>}
              </span>
            </li>
          );
        })}
      </ol>
      <button className="secondary small" onClick={() => setSinglePage(!singlePage)} data-testid="toggle-single-page">
        {singlePage ? "分步显示" : "单页显示全部"}
      </button>
    </div>
  );
}

function StepSummary({ rows }: { rows: [string, string][] }) {
  return (
    <table className="kv" data-testid="step-summary">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <th>{k}</th>
            <td>{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
