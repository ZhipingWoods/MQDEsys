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

/** A change in a lower-is-better metric: negative is an improvement. */
export function lowerIsBetterTone(pct: number): "good" | "bad" | "flat" {
  if (Math.abs(pct) < 0.05) return "flat";
  return pct < 0 ? "good" : "bad";
}

export const signedPct = (pct: number, digits = 2): string => `${pct > 0 ? "+" : ""}${pct.toFixed(digits)}%`;
