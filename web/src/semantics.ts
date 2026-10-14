/** What a metric's number means, in one place: which direction is better, and what the project accepts. */

export type VerdictStatus = "pass" | "fail" | "record" | "none";

export interface PplVerdict {
  status: VerdictStatus;
  /** PPL change against the unquantized model, in percent (positive = worse: lower PPL is better) */
  pct: number;
  /** the budget for this recipe in percent, or null when it is recorded only; undefined when no rule names this recipe */
  budget: number | null | undefined;
}

/**
 * docs/ARCHITECTURE.md section 11.4: how much wikitext-2 PPL a recipe may lose against the unquantized model.
 * `null` = recorded only (experimental, no gate). A recipe the table does not name has no rule at all.
 */
export function pplBudget(recipe: string): number | null | undefined {
  const r = recipe.toLowerCase();
  if (r === "fp8" || r.startsWith("fp8")) return 1;
  if (r === "w4a16") return 5;
  if (r === "w4a8") return 8;
  if (r.startsWith("w2")) return null;
  return undefined;
}

export function pplVerdict(recipe: string, pct: number): PplVerdict {
  const budget = pplBudget(recipe);
  if (budget === undefined) return { status: "none", pct, budget };
  if (budget === null) return { status: "record", pct, budget };
  return { status: pct <= budget ? "pass" : "fail", pct, budget };
}

export type Direction = "lower" | "higher";

/** A change of `pct` percent in a metric: is it better or worse? Direction-aware; a change under 0.05% is flat. */
export function toneOf(direction: Direction, pct: number): "good" | "bad" | "flat" {
  if (Math.abs(pct) < 0.05) return "flat";
  return (direction === "lower" ? pct < 0 : pct > 0) ? "good" : "bad";
}

/** A change in a lower-is-better metric: negative is an improvement. */
export const lowerIsBetterTone = (pct: number) => toneOf("lower", pct);

/** What a number in the cross-run score matrix means: which way is better, and how to print it. */
export interface MetricSpec {
  id: string;
  label: string;
  direction: Direction;
  unit: string;
  digits: number;
}
export const MATRIX_METRICS: MetricSpec[] = [
  { id: "ppl", label: "PPL（W+A，伪量化）", direction: "lower", unit: "", digits: 3 },
  { id: "ppl_kv", label: "PPL（+KV Cache）", direction: "lower", unit: "", digits: 3 },
  { id: "real_ppl", label: "PPL（真实低比特权重）", direction: "lower", unit: "", digits: 3 },
  { id: "served_ppl", label: "PPL（部署后，服务端）", direction: "lower", unit: "", digits: 3 },
  { id: "tps", label: "服务吞吐", direction: "higher", unit: "tok/s", digits: 0 },
  { id: "size_mib", label: "导出体积", direction: "lower", unit: "MiB", digits: 0 },
];

/** Percent change of x against a baseline b; null when there is nothing to compare with. */
export function relChange(x: number, b: number): number | null {
  if (!Number.isFinite(x) || !Number.isFinite(b) || b === 0) return null;
  return (x / b - 1) * 100;
}

export const signedPct = (pct: number, digits = 2): string => `${pct > 0 ? "+" : ""}${pct.toFixed(digits)}%`;
