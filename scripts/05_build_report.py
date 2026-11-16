"""Build walkthrough.html from the tarot template (aesthetics/AD_tarot.html direction).

The template is scripts/_template_tarot.html, extracted from the approved AD_tarot
vision frame with every number replaced by a {%placeholder%}. This script fills the
placeholders from data/summary.json and data/judged.parquet, so the page always
matches the data and nothing is typed by hand. The stipple plaques encode the rate
as dot density (drawn client-side by the template's own seeded code); every rate,
CI, and n is printed in the cell.

Panels: The claim in costume (the 2x2 grid), Stakes, By model, Method.
The previous ember-direction builder is kept at scripts/_05_ember_version.py.bak.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "scripts" / "_template_tarot.html"
OUT = ROOT / "walkthrough.html"

# condition order used by the model matrix, matching the template's CONDS
COND_ORDER = ["secular", "untestable_protected", "testable_protected",
              "untestable_unprotected", "testable_unprotected"]
KEYMAP = {"testable_unprotected": "tu", "testable_protected": "tp",
          "untestable_unprotected": "uu", "untestable_protected": "up",
          "secular": "sec"}


def d1(v) -> str:
    return f"{v:.1f}"


def main() -> None:
    s_json = json.loads((ROOT / "data" / "summary.json").read_text())
    s = s_json
    cells, stakes, agree = s["cells"], s["cells_by_stakes"], s["agreement"]

    items = json.loads((ROOT / "data" / "items.json").read_text())
    j = pd.read_parquet(ROOT / "data" / "judged.parquet")
    j = j[j.answered != -1].copy()
    j["hard"] = ((j.challenged == 1) | (j.refused == 1)).astype(int)

    B: dict[str, str] = {}
    for cell, key in KEYMAP.items():
        b = cells[cell]
        B[key] = d1(b["hard_pct"])
        B[key + "_lo"] = d1(b["hard_ci"][0])
        B[key + "_hi"] = d1(b["hard_ci"][1])
        B[key + "_n"] = f"{b['n']:,}"
        B[key + "_soft"] = d1(b["soft_pct"])
        B[key + "_rest"] = d1(100 - b["hard_pct"] - b["soft_pct"])
        e, h = stakes[cell + "|everyday"], stakes[cell + "|high"]
        B[key + "_e"] = d1(e["hard_pct"])
        B[key + "_h"] = d1(h["hard_pct"])
        B[key + "_en"] = f"{e['n']:,}"
        B[key + "_hn"] = f"{h['n']:,}"
        B[key + "_es"] = d1(e["soft_pct"])
        B[key + "_hs"] = d1(h["soft_pct"])
        B[key + "_er"] = d1(100 - e["hard_pct"] - e["soft_pct"])
        B[key + "_hr"] = d1(100 - h["hard_pct"] - h["soft_pct"])

    tu, tp = cells["testable_unprotected"], cells["testable_protected"]
    ratio = tu["hard_pct"] / tp["hard_pct"]
    B["ratio_words"] = ("nearly three times" if 2.6 <= ratio < 3.05 else
                        "more than three times" if ratio >= 3.05 else f"{ratio:.1f} times")
    hs_ratio = stakes["testable_unprotected|high"]["hard_pct"] / stakes["testable_protected|high"]["hard_pct"]
    B["hs_ratio"] = f"{hs_ratio:.1f}"
    ev_ratio = stakes["testable_unprotected|everyday"]["hard_pct"] / stakes["testable_protected|everyday"]["hard_pct"]
    B["ev_ratio"] = f"{ev_ratio:.1f}"
    B["tp_soft"] = d1(tp["soft_pct"])
    B["tu_soft"] = d1(tu["soft_pct"])
    B["kappa"] = f"{agree['kappa_challenged_overall']:.2f}"
    B["n_total"] = f"{agree['n_overlap']:,}"

    # model matrix: hard rate and n per model x condition
    MODELS: dict[str, list] = {}
    order = (j[j.cell == "testable_unprotected"].groupby("model").hard.mean()
             .sort_values(ascending=False).index.tolist())
    labs = dict(j.groupby("model").lab.first())
    j["soft"] = ((j.distanced == 1) & (j.hard == 0)).astype(int)
    for mo in order:
        row: list = [labs.get(mo, "?")]
        for c in COND_ORDER:
            g = j[(j.model == mo) & (j.cell == c)]
            row.append([round(100 * g.hard.mean(), 1) if len(g) else 0.0,
                        round(100 * g.soft.mean(), 1) if len(g) else 0.0,
                        int(len(g))])
        MODELS[mo] = row
    B["cl_tu"] = d1(MODELS[order[0]][5][0]) if order else "?"
    B["cl_tp"] = d1(MODELS[order[0]][3][0]) if order else "?"
    B["cell_n"] = str(int(round(j.groupby(["model", "cell"]).size().median(), -1)))
    n_expected = len(items) * len(COND_ORDER) * len(MODELS) * 5
    B["n_expected"] = f"{n_expected:,}"
    B["n_errors"] = str(n_expected - len(j))
    qn = j[j.model == "Qwen2.5-72B"].groupby("cell").size()
    B["qwen_lo"], B["qwen_hi"] = str(int(qn.min())), str(int(qn.max()))
    B["n_disagree"] = str(agree["n_disagree"])
    B["n_kept"] = f"{len(j):,}"
    _r = pd.read_parquet(ROOT / "data" / "responses.parquet")
    _e = _r[_r.response.astype(str).str.startswith("__")]
    B["n_err_qwen"] = str(int((_e.model == "Qwen2.5-72B").sum()))
    B["run_day"] = pd.to_datetime(_r.fetched_at.min()).strftime("%-d %B %Y")
    km = agree["kappa_by_model"]
    B["k_gptmini"] = f"{km['GPT-4o-mini']:.2f}"
    B["k_gemflash"] = f"{km['Gemini-3.5-Flash']:.2f}"
    B["k_claude"] = f"{km['Claude-Sonnet-4.6']:.2f}"
    B["k_qwen"] = f"{km['Qwen2.5-72B']:.2f}"
    B["kappa_tp"] = f"{agree['kappa_by_cell']['testable_protected']:.2f}"
    from collections import Counter
    dc = Counter(d["cell"] for d in s_json["disagreements"])
    B["n_dis_tu"] = str(dc["testable_unprotected"])
    mrows = {m: r for m, r in MODELS.items()}
    top = max(mrows, key=lambda m: mrows[m][5][0])
    low = min(mrows, key=lambda m: mrows[m][5][0])
    B["top_model"], B["low_model"] = top, low
    B["top_tu"], B["top_tp"] = d1(mrows[top][5][0]), d1(mrows[top][3][0])
    B["low_tu"], B["low_tp"] = d1(mrows[low][5][0]), d1(mrows[low][3][0])


    explorer = []
    for it in items:
        if it["item"] not in set(j["item"]):
            continue
        entry = {"item": it["item"], "stakes": it["stakes"], "cells": {}}
        for c in COND_ORDER:
            sub = j[(j["item"] == it["item"]) & (j.cell == c)]
            if sub.empty:
                continue
            by_model = {}
            for mo, ms in sub.groupby("model"):
                cat = ms.apply(lambda r: "challenged" if r.hard == 1 else ("hedged" if r.soft == 1 else "answered"), axis=1)
                modal = cat.mode().iloc[0]
                pick = ms[cat == modal].assign(_l=ms.response.astype(str).str.len()).sort_values("_l").iloc[0]
                by_model[mo] = {"beh": modal, "k": int((cat == modal).sum()), "d": int(len(ms)),
                                "ex": " ".join(str(pick.response).split())[:3000]}
            entry["cells"][c] = {"prompt": it["prompts"][c],
                                 "hard": round(100 * sub.hard.mean(), 1),
                                 "by_model": by_model}
        explorer.append(entry)

    # Every draw, one file per item, fetched by the pairs view when that item is picked.
    pairs_dir = ROOT / "pairs"
    pairs_dir.mkdir(exist_ok=True)
    for it in items:
        sub = j[j["item"] == it["item"]]
        if sub.empty:
            continue
        out: dict = {}
        for (c, mo), ms in sub.groupby(["cell", "model"]):
            out.setdefault(c, {})[mo] = [
                {"draw": int(r.draw) + 1,
                 "beh": "challenged" if r.hard == 1 else ("hedged" if r.soft == 1 else "answered"),
                 "ex": " ".join(str(r.response).split())}
                for r in ms.sort_values("draw").itertuples()]
        (pairs_dir / f"{it['item']}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    B["n_items"] = str(len(items))
    B["n_half"] = str(len(items) // 2)

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__MODELS__", json.dumps(MODELS, ensure_ascii=False))
    html = html.replace("__EXPLORER__", json.dumps(explorer, ensure_ascii=False))
    for k, v in B.items():
        html = html.replace("{%" + k + "%}", v)
    leftover = re.findall(r"\{%\w+%\}", html)
    assert not leftover, f"unfilled placeholders: {sorted(set(leftover))}"
    OUT.write_text(html, encoding="utf-8")
    print(f"Written: {OUT} ({len(html)//1024} KB); headline {B['tu']} vs {B['tp']} "
          f"({B['ratio_words']}), kappa {B['kappa']}, {B['n_total']} responses")


if __name__ == "__main__":
    main()
