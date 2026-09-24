import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { DEFAULT_DATASET, DEFAULT_MODEL, useAsync } from "../hooks";

export default function NewJobPage() {
  const nav = useNavigate();
  const recipes = useAsync(api.recipes, []);
  const profiles = useAsync(api.hardwareProfiles, []);
  const cfg = useAsync(api.config, []);
  const [name, setName] = useState("minicpm-vla-eval");
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [picked, setPicked] = useState<string[]>(["w8a16", "fp8", "w4a16", "w4a8"]);
  const [dataset, setDataset] = useState(DEFAULT_DATASET);
  const [nCalib, setNCalib] = useState(16);
  const [nEval, setNEval] = useState(24);
  const [hw, setHw] = useState("nvidia/rtx4090");
  const [attribution, setAttribution] = useState(true);
  const [action, setAction] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const job = {
    metadata: { name },
    model: { source: "local", path: model, modality: "vla" },
    quant: { recipe: picked, ...(action ? { also_quantize: ["action"] } : {}) },
    calibration: { dataset, n_samples: nCalib },
    evaluation: { dataset, n_samples: nEval, concurrency: [1, 4, 16, 64], attribution },
    deploy: { hardware: hw },
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

  const toggle = (id: string) => setPicked(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id]);

  return (
    <>
      <h1>新建评估作业</h1>
      <p className="subtitle">
        指定模型 → 量化配方 → 评估数据 → 部署硬件。作业由 C++ 调度到 GPU 池（{cfg.data?.gpus.join(", ") ?? "…"}）上的 Python 执行器，全过程事件实时推送。
      </p>
      {err && <div className="error-box">{err}</div>}
      <div className="card">
        <h2>① 模型</h2>
        <div className="form-row">
          <div className="field grow">
            <label>本地模型路径（允许根目录：{cfg.data?.allowed_roots.join(", ")}）</label>
            <input value={model} onChange={(e) => setModel(e.target.value)} />
          </div>
          <div className="field">
            <label>作业名称</label>
            <input value={name} onChange={(e) => setName(e.target.value)} />
          </div>
        </div>
        <h2>② 量化配方</h2>
        <div className="chips" style={{ marginBottom: 8 }}>
          {recipes.data
            ?.filter((r) => r.weight.format !== "none")
            .map((r) => (
              <span key={r.id} className={`chip ${picked.includes(r.id) ? "on" : ""}`} onClick={() => toggle(r.id)} title={r.summary}>
                {r.name}
                {r.maturity === "experimental" && <span className="exp">实验</span>}
              </span>
            ))}
        </div>
        <label className="small">
          <input type="checkbox" checked={action} onChange={(e) => setAction(e.target.checked)} /> 同时量化 DiT 动作头（默认保持 FP32）
        </label>
      </div>
      <div className="card">
        <h2>③ 评估数据</h2>
        <div className="form-row">
          <div className="field grow">
            <label>数据集（LeRobot v3 目录，或 synthetic）</label>
            <input value={dataset} onChange={(e) => setDataset(e.target.value)} />
          </div>
          <div className="field">
            <label>校准样本</label>
            <input type="number" value={nCalib} min={1} onChange={(e) => setNCalib(Number(e.target.value))} />
          </div>
          <div className="field">
            <label>评估样本</label>
            <input type="number" value={nEval} min={1} onChange={(e) => setNEval(Number(e.target.value))} />
          </div>
        </div>
        <label className="small">
          <input type="checkbox" checked={attribution} onChange={(e) => setAttribution(e.target.checked)} /> 分阶段误差归因（逐模态单独量化，定位误差来源环节）
        </label>
      </div>
      <div className="card">
        <h2>④ 部署硬件</h2>
        <div className="form-row">
          <div className="field">
            <label>目标硬件画像（用于资源预估）</label>
            <select value={hw} onChange={(e) => setHw(e.target.value)}>
              {profiles.data?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        </div>
        <p className="note">当前执行后端为 PyTorch fake-quant（评估量化误差）；真实低比特内核部署（vLLM / llama.cpp）在后续里程碑接入。</p>
      </div>
      <div className="card">
        <h2>JobSpec 预览</h2>
        <pre className="log" style={{ height: "auto", maxHeight: 240 }}>
          {JSON.stringify(job, null, 2)}
        </pre>
        <button disabled={busy || picked.length === 0} onClick={submit}>
          {busy ? "提交中…" : "提交作业"}
        </button>
      </div>
    </>
  );
}
