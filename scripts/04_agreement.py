"""Merge the two judges, compute agreement, and write data/summary.json.

Everything downstream (tables, any writeup) reads summary.json; nothing is
assembled by hand. Definitions match wildchat-rel-ref/scripts/23:
  hard = challenged == 1 or refused == 1
  soft = distanced == 1 and not hard   (disjoint from hard)

Agreement: Cohen's kappa on the challenged label, overall, per model, per cell.
Per-family caveat: GPT-4o-mini rows share a lab with judge 1, Gemini-3.5-Flash
rows with judge 2; their kappas are listed separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "summary.json"

KEY = ["item", "cell", "model", "draw"]
CELLS = ["secular", "testable_unprotected", "testable_protected",
         "untestable_unprotected", "untestable_protected"]

# Wilson interval, one copy of the logic, from the wildchat repo.
_d12 = ROOT.parent / "wildchat-rel-ref" / "scripts" / "12_dedup_sensitivity.py"
_ns: dict = {"__file__": str(_d12)}
exec(_d12.read_text(encoding="utf-8").split("def main()")[0], _ns)
wilson = _ns["wilson"]


def kappa(a: pd.Series, b: pd.Series) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1.0:
        return None  # no variation in either judge; kappa undefined
    return round((po - pe) / (1 - pe), 4)


def flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hard"] = ((df.challenged == 1) | (df.refused == 1)).astype(int)
    df["soft"] = ((df.distanced == 1) & (df.hard == 0)).astype(int)
    return df


def rate_block(sub: pd.DataFrame) -> dict:
    n = len(sub)
    k_hard = int(sub.hard.sum())
    p, lo, hi = wilson(k_hard, n)
    return {"n": n, "hard_pct": p, "hard_ci": [lo, hi],
            "soft_pct": round(100 * sub.soft.mean(), 1)}


def main() -> None:
    j1 = pd.read_parquet(ROOT / "data" / "judged.parquet")
    j2 = pd.read_parquet(ROOT / "data" / "judged_gemini.parquet")
    j1 = flags(j1[j1.answered != -1])
    j2 = flags(j2[j2.answered != -1])

    m = j1.merge(j2[KEY + ["challenged", "hard", "soft"]], on=KEY, suffixes=("", "_g"))
    print(f"judge1 rows {len(j1)}, judge2 rows {len(j2)}, merged {len(m)}")

    agree = {
        "n_overlap": len(m),
        "pct_agreement_challenged": round(100 * float((m.challenged == m.challenged_g).mean()), 2),
        "kappa_challenged_overall": kappa(m.challenged, m.challenged_g),
        "kappa_by_cell": {c: kappa(g.challenged, g.challenged_g) for c, g in m.groupby("cell")},
        "kappa_by_model": {mo: kappa(g.challenged, g.challenged_g) for mo, g in m.groupby("model")},
        "judge_family_note": "GPT-4o-mini rows share a lab with judge 1 (gpt-4o); Gemini-3.5-Flash rows with judge 2 (gemini-2.5-flash).",
        "n_disagree": int((m.challenged != m.challenged_g).sum()),
    }

    # Rates from judge 1 (primary), with judge-2 replication of the headline contrast
    out = {
        "provenance": "scripts/04_agreement.py from judged.parquet + judged_gemini.parquet; "
                      "hard = challenged|refused, soft = distanced & not hard",
        "cells": {c: rate_block(g) for c, g in j1.groupby("cell")},
        "cells_by_stakes": {f"{c}|{s}": rate_block(g) for (c, s), g in j1.groupby(["cell", "stakes"])},
        "models_by_cell": {f"{mo}|{c}": rate_block(g) for (mo, c), g in j1.groupby(["model", "cell"])},
        "items_hard_by_cell": {f"{it}|{c}": round(100 * g.hard.mean(), 1)
                               for (it, c), g in j1.groupby(["item", "cell"])},
        "headline_judge2_replication": {c: round(100 * g.hard.mean(), 1) for c, g in j2.groupby("cell")},
        "agreement": agree,
        "disagreements": m.loc[m.challenged != m.challenged_g,
                               KEY + ["stakes", "practice", "challenged", "challenged_g"]].to_dict("records"),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"Written: {OUT}")

    print("\n=== Hard-challenge %, by cell (judge 1, Wilson 95% CI) ===")
    for c in CELLS:
        b = out["cells"].get(c)
        if b: print(f"  {c:24s} {b['hard_pct']:5.1f}  [{b['hard_ci'][0]}, {b['hard_ci'][1]}]  soft {b['soft_pct']}  n={b['n']}")
    print("\n=== Hard-challenge %, cell x stakes ===")
    for key in sorted(out["cells_by_stakes"]):
        b = out["cells_by_stakes"][key]
        print(f"  {key:36s} {b['hard_pct']:5.1f}  soft {b['soft_pct']}  n={b['n']}")
    print(f"\nkappa (challenged) overall: {agree['kappa_challenged_overall']}, "
          f"agreement {agree['pct_agreement_challenged']}%, disagreements {agree['n_disagree']}")


if __name__ == "__main__":
    main()
