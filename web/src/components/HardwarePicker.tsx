import { useMemo, useState } from "react";
import { Device, fmtBytes, HardwareProfile } from "../api";
import { VendorIcon } from "../pages/hwIcons";

type Category = "all" | "edge" | "workstation" | "datacenter";

const CATEGORIES: { id: Category; label: string }[] = [
  { id: "all", label: "全部" },
  { id: "edge", label: "边缘设备" },
  { id: "workstation", label: "桌面工作站" },
  { id: "datacenter", label: "数据中心" },
];

/** The hardware_db has no "form factor" field, so classify by well-known product families.
 *  This is display-only: every profile remains selectable in every category. */
export function hardwareCategory(p: HardwareProfile): Exclude<Category, "all"> {
  const s = `${p.id} ${p.name}`.toLowerCase();
  if (/jetson|orin|thor|agx|nano|nx\b/.test(s)) return "edge";
  if (/dgx|spark|workstation|rtx|quadro/.test(s)) return "workstation";
  return "datacenter";
}

const CATEGORY_LABEL: Record<Exclude<Category, "all">, string> = {
  edge: "边缘",
  workstation: "工作站",
  datacenter: "数据中心",
};

function peak(p: HardwareProfile, key: string): string | null {
  const v = p.peak_tflops[key];
  return v == null ? null : `${v} TFLOPS`;
}

/** Include common spellings users type instead of the catalog name (e.g. "dspark" for DGX Spark). */
function searchText(p: HardwareProfile): string {
  const base = `${p.name} ${p.id} ${p.vendor} ${p.arch}`.toLowerCase();
  return /spark/.test(base) ? `${base} dspark` : base;
}

export interface HardwarePickerProps {
  value: string;
  onChange: (id: string) => void;
  profiles: HardwareProfile[];
  /** Devices probed on this machine. A profile matching one of them gets a "本机可运行" badge. */
  detected?: Device[];
  compact?: boolean;
  /** Hide search/filter when the picker sits in a narrow form column. */
  simple?: boolean;
}

/** Card-based deployment target selector. One click selects a hardware profile; the same id is what
 *  JobSpec.deploy.hardware receives, so no manual id typing or vendor knowledge is needed. */
export function HardwarePicker({ value, onChange, profiles, detected = [], compact = false, simple = false }: HardwarePickerProps) {
  const [category, setCategory] = useState<Category>("all");
  const [query, setQuery] = useState("");

  const present = useMemo(() => new Set(detected.map((d) => d.profile).filter(Boolean)), [detected]);
  const counts = useMemo(() => {
    const c: Record<Category, number> = { all: profiles.length, edge: 0, workstation: 0, datacenter: 0 };
    for (const p of profiles) c[hardwareCategory(p)] += 1;
    return c;
  }, [profiles]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    const order: Record<Exclude<Category, "all">, number> = { edge: 0, workstation: 1, datacenter: 2 };
    return profiles
      .filter((p) => category === "all" || hardwareCategory(p) === category)
      .filter((p) => !q || searchText(p).includes(q))
      .sort((a, b) => {
        const pa = present.has(a.id) ? 0 : 1;
        const pb = present.has(b.id) ? 0 : 1;
        if (pa !== pb) return pa - pb;
        const ca = order[hardwareCategory(a)];
        const cb = order[hardwareCategory(b)];
        if (ca !== cb) return ca - cb;
        return a.name.localeCompare(b.name, "en");
      });
  }, [profiles, category, query, present]);

  if (profiles.length === 0) {
    return <p className="muted small">正在读取硬件画像…</p>;
  }

  return (
    <div className={`hw-picker${compact ? " compact" : ""}`} data-testid="hardware-picker">
      {!simple && (
        <div className="hw-picker-tools">
          <div className="chips" data-testid="hw-category">
            {CATEGORIES.map((c) => (
              <span
                key={c.id}
                className={`chip ${category === c.id ? "on" : ""}`}
                onClick={() => setCategory(c.id)}
                data-testid={`hw-filter-${c.id}`}
              >
                {c.label} <span className="exp">{counts[c.id]}</span>
              </span>
            ))}
          </div>
          <input
            className="hw-search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索型号，如 Orin / Thor / Spark"
            aria-label="搜索硬件"
            data-testid="hw-search"
          />
        </div>
      )}
      <div className="hw-cards" data-testid="hw-cards">
        {shown.map((p) => {
          const selected = p.id === value;
          const local = present.has(p.id);
          const cat = hardwareCategory(p);
          return (
            <button
              type="button"
              key={p.id}
              className={`hw-card${selected ? " on" : ""}`}
              onClick={() => onChange(p.id)}
              aria-pressed={selected}
              data-testid={`hw-card-${p.id}`}
            >
              <span className="hw-card-top">
                <VendorIcon vendor={p.vendor || p.id} />
                <span className="hw-card-title">
                  <b>{p.name}</b>
                  <small className="mono">{p.id}</small>
                </span>
                {selected ? <span className="hw-card-check" aria-hidden="true">✓</span> : null}
              </span>
              <span className="hw-card-tags">
                <span className={`badge${cat === "edge" ? " warn" : cat === "workstation" ? " accent" : ""}`}>{CATEGORY_LABEL[cat]}</span>
                {local ? <span className="badge good">本机可运行</span> : null}
              </span>
              <span className="hw-card-stats">
                <span>
                  <small>{cat === "edge" || /dgx|spark/.test(p.id) ? "统一内存" : "显存"}</small>
                  <b>{fmtBytes(p.mem_bytes)}</b>
                </span>
                <span>
                  <small>带宽</small>
                  <b>{p.mem_bw_gbps} GB/s</b>
                </span>
                <span>
                  <small>BF16</small>
                  <b>{peak(p, "bf16") ?? "—"}</b>
                </span>
              </span>
              <span className="hw-card-footer">
                <span className="hw-formats">{p.native_formats.slice(0, 4).join(" · ") || p.arch}</span>
                <span className="hw-card-cta" data-testid={`hw-card-select-${p.id}`}>{selected ? "已选为部署目标" : "选为部署目标 →"}</span>
              </span>
            </button>
          );
        })}
        {shown.length === 0 && <p className="muted small" data-testid="hw-empty">没有匹配的硬件，换一个关键词或分类。</p>}
      </div>
    </div>
  );
}
