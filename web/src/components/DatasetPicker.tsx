import { useEffect, useState } from "react";
import { api, BuiltinDataset, DatasetInfo, fmtBytes } from "../api";
import { useDebounced } from "../hooks";

/** What kind of data the job reads: free text (LLM calibration) or a LeRobot v3 directory (VLA / VLM images). */
export type DatasetKind = "text" | "lerobot";

interface Props {
  /** "text": wikitext-2 or a local text file / directory. "lerobot": synthetic or a LeRobot v3 directory. */
  kind: DatasetKind;
  role: "calibration" | "evaluation";
  value: string;
  onChange: (v: string) => void;
  /** Tells the form whether it may be submitted: false while a typed path is not usable. */
  onValid?: (ok: boolean) => void;
  testid: string;
  label: string;
  /**
   * A default that only exists on some servers (the dev server's LeRobot path). While the user has not touched the field and this
   * default turns out to be unreadable here, use `fallback` instead and say so; a path the user typed, or one that came from another
   * job, is never swapped silently.
   */
  defaultValue?: { path: string; fallback: string };
}

const CUSTOM = "__custom__";

/** What an older server (before /api/v1/datasets) can be assumed to offer: it cannot be asked, and it does not check paths either. */
const STATIC_BUILTINS: BuiltinDataset[] = [
  { id: "wikitext-2-raw-v1:train", label: "wikitext-2（train 切分）", kind: "text", roles: ["calibration"], available: true },
  { id: "synthetic", label: "synthetic（程序生成的图文样本）", kind: "vision", roles: ["calibration", "evaluation"], available: true },
];

const describe = (i: DatasetInfo): string => {
  if (i.builtin) return "内置数据集";
  const fmts = Object.entries(i.formats ?? {})
    .map(([k, v]) => `${k}×${v}`)
    .join(" ");
  const n = `${i.files}${i.files_capped ? "+" : ""}`;
  return i.kind === "dir"
    ? `目录 · ${n} 个${fmts ? "可读" : ""}文件 · ${fmtBytes(i.bytes)}${fmts ? ` · ${fmts}` : ""}`
    : `文件 · ${fmtBytes(i.bytes)}${fmts ? ` · ${fmts}` : ""}`;
};

/** Pick a built-in dataset or give a path on the server; a typed path is checked against the server as it is typed. */
export function DatasetPicker({ kind, role, value, onChange, onValid, testid, label, defaultValue }: Props) {
  const [list, setList] = useState<{ builtin: BuiltinDataset[]; roots: string[]; legacy?: boolean } | null>(null);
  useEffect(() => {
    let alive = true;
    api.datasets().then(
      (d) => alive && setList({ builtin: d.builtin, roots: d.allowed_roots }),
      () => alive && setList({ builtin: STATIC_BUILTINS, roots: [], legacy: true }), // an older server: no way to ask, so no way to check
    );
    return () => {
      alive = false;
    };
  }, []);
  const options = (list?.builtin ?? []).filter((b) => b.roles.includes(role) && (kind === "text" ? b.kind === "text" : b.kind === "vision"));
  const [custom, setCustom] = useState(false);
  const [touched, setTouched] = useState(false); // the user chose something: the default is no longer ours to replace
  const [fellBack, setFellBack] = useState<string | null>(null); // the default we replaced, for the note
  // a value that is not a built-in is a path (it may arrive late: derived from another job, or the list loads after it)
  useEffect(() => {
    if (list && value !== "" && !options.some((b) => b.id === value)) setCustom(true);
  }, [value, list]); // eslint-disable-line react-hooks/exhaustive-deps
  const selected = custom ? CUSTOM : value;

  const typed = useDebounced(custom ? value.trim() : "", 350);
  const [info, setInfo] = useState<{ path: string; res?: DatasetInfo; err?: string } | null>(null);
  useEffect(() => {
    if (!typed || list?.legacy) {
      setInfo(null);
      return;
    }
    let alive = true;
    api.inspectDataset(typed, kind).then(
      (res) => alive && setInfo({ path: typed, res }),
      (e: Error) => alive && setInfo({ path: typed, err: e.message }),
    );
    return () => {
      alive = false;
    };
  }, [typed, kind, list?.legacy]);

  useEffect(() => {
    if (!defaultValue || touched || legacy || value !== defaultValue.path || info?.path !== value) return;
    if (info.err || (info.res && !info.res.usable)) {
      setFellBack(defaultValue.path);
      setCustom(false);
      onChange(defaultValue.fallback);
    }
  }, [info]); // eslint-disable-line react-hooks/exhaustive-deps

  const legacy = !!list?.legacy;
  const pending = custom && !legacy && (value.trim() === "" || info?.path !== value.trim());
  // an older server cannot say whether a path is readable: the path is accepted (it still has to be non-empty) and the
  // submit-time check, if the server has one, decides
  const ok = !custom || (legacy ? value.trim() !== "" : !pending && !!info?.res?.usable);
  useEffect(() => {
    onValid?.(ok);
  }, [ok]); // eslint-disable-line react-hooks/exhaustive-deps

  const chosen = options.find((b) => b.id === value);
  const roots = list?.roots ?? [];
  return (
    <div className="field grow" data-testid={`${testid}-picker`}>
      <label>{label}</label>
      <select
        value={selected}
        data-testid={`${testid}-select`}
        onChange={(e) => {
          setTouched(true);
          setFellBack(null);
          if (e.target.value === CUSTOM) {
            setCustom(true);
            onChange("");
          } else {
            setCustom(false);
            onChange(e.target.value);
          }
        }}
      >
        {options.map((b) => (
          <option key={b.id} value={b.id}>
            {b.label}
            {b.available ? "" : "（本机没有）"}
          </option>
        ))}
        <option value={CUSTOM}>{kind === "text" ? "本地文件或目录…" : "LeRobot v3 目录…"}</option>
      </select>
      {fellBack && !custom && (
        <p className="note" data-testid={`${testid}-fellback`}>
          默认的数据 {fellBack} 在这台服务器上用不了，已改用 {value}。要用别的，在上面选。
        </p>
      )}
      {!custom && chosen && !chosen.available && (
        <p className="note warn" data-testid={`${testid}-unavailable`}>
          服务器上找不到这个数据集：{chosen.reason}。可以改用本地文件或目录。
        </p>
      )}
      {custom && (
        <>
          <input
            value={value}
            placeholder={kind === "text" ? "服务器上的路径：.txt .md .jsonl .json .parquet 文件，或装有它们的目录" : "服务器上的 LeRobot v3 目录"}
            onChange={(e) => {
              setTouched(true);
              onChange(e.target.value);
            }}
            data-testid={testid}
          />
          <p className={`note ${pending || ok ? "" : "warn"}`} data-testid={`${testid}-status`}>
            {value.trim() === ""
              ? "填一个路径。"
              : legacy
                ? "这台服务器的版本较旧，不能预先检查路径；提交时才会知道能不能读。"
                : pending
                ? "检查中…"
                : info?.err
                  ? /outside allowed roots|403/i.test(info.err)
                    ? `路径不在服务器允许的目录里${roots.length ? `（${roots.join("、")}）` : ""}。`
                    : info.err
                  : info?.res?.usable
                    ? `✓ ${describe(info.res)}`
                    : `✗ ${info?.res?.reason ?? "不可用"}`}
          </p>
          {kind === "text" && role === "calibration" && (
            <p className="note muted">目录会递归读取，按路径排序，最多读前 5000 万字符；窗口随机取自拼接后的全文。.jsonl / .json / .parquet 取 text 字段。</p>
          )}
          {roots.length > 0 && <p className="note muted">服务器允许的目录：{roots.join("、")}</p>}
        </>
      )}
    </div>
  );
}
