"""
LENGTH-STANDARDIZED PMI-vs-ROUGE WIN RATES

Both judges used in this project have a large, and OPPOSITE, length preference.
Measured on wikihow over 8 checkpoints (decided pairs only):

    PMI 10+ words SHORTER than ROUGE   GPT 36.8%   Prometheus 60.1%
    PMI 10+ words LONGER  than ROUGE   GPT 64.3%   Prometheus 40.6%

That matters because the PMI/ROUGE length ratio is not constant across
checkpoints -- on wikihow it drifts from 0.955 at 1M to 0.886 at 8M -- so part
of the checkpoint-to-checkpoint movement in the raw win rate is length, not
quality. On wikihow the raw numbers UNDERSTATE PMI's improvement, because PMI's
summaries get relatively shorter just as the pro-length judge is scoring them.

WHAT THIS SCRIPT DOES
---------------------
Direct standardization. Pairs are binned by (PMI words - ROUGE words); every
checkpoint's win rate is then recomputed against one common bin mix (the mix
pooled over all checkpoints of that dataset), so every checkpoint is scored as
if it faced the same distribution of length differences.

Why not just compare equal-length pairs: that works but throws away ~92% of the
data. On wikihow it leaves ~450 pairs per checkpoint and a 1M->8M CI of +/-6.6,
too wide to conclude anything. Standardization keeps all pairs and gave
+8.26 +/-2.23 for the same comparison.

Bins with fewer than MIN_BIN samples in a checkpoint are dropped from that
checkpoint's estimate and the weights renormalized, so a sparse tail cannot
swing a number.

NOTE the CIs treat checkpoints as independent. They are not quite -- every
checkpoint judges the same documents -- so the interval is mildly conservative.

USAGE
-----
    python analyse_length_standardized_winrates.py                  # all datasets
    python analyse_length_standardized_winrates.py --datasets xsum
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS = [f"{i}M" for i in range(1, 9)]
RESULT_FOLDER = {"cnn": "cnn_result_files", "xsum": "xsum_result_files",
                 "wikihow": "wikihow_result_files"}

# (low, high, label) on (PMI words - ROUGE words), inclusive.
BINS = [(-10**9, -10, "PMI 10+ shorter"), (-9, -5, "PMI 5-9 shorter"),
        (-4, -2, "PMI 2-4 shorter"), (-1, 1, "within 1 word"),
        (2, 4, "PMI 2-4 longer"), (5, 9, "PMI 5-9 longer"),
        (10, 10**9, "PMI 10+ longer")]
MIN_BIN = 30

# Judges to look for. Each is (label, filename template).
JUDGES = [
    ("GPT (step 7, dual-order consensus)",
     "{ds}_{ck}_llm_judge_gpt_vs_reference_summaries__step7_bothorders.json"),
    ("Prometheus (step 5)",
     "{ds}_{ck}_llm_judge_vs_reference_summaries__step5.json"),
]


def bin_index(delta):
    for j, (lo, hi, _) in enumerate(BINS):
        if lo <= delta <= hi:
            return j
    raise AssertionError("bins must cover the line")


def load_pairs(path):
    """[(length_delta, pmi_won)] over DECIDED pairs only."""
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for row in rows:
        if row["llm_judge_winner"] not in ("pmi", "rouge"):
            continue
        delta = len(row["pmi_summary"].split()) - len(row["rouge_summary"].split())
        out.append((delta, row["llm_judge_winner"] == "pmi"))
    return out


def standardize(pairs, weights):
    """(standardized PMI rate, its variance) against the given bin weights."""
    per = defaultdict(lambda: [0, 0])
    for delta, won in pairs:
        cell = per[bin_index(delta)]
        cell[0] += won
        cell[1] += 1

    usable = [j for j in range(len(BINS)) if per[j][1] >= MIN_BIN]
    total_weight = sum(weights[j] for j in usable)
    if not usable or total_weight == 0:
        return None, None

    rate = variance = 0.0
    for j in usable:
        p = per[j][0] / per[j][1]
        w = weights[j] / total_weight
        rate += w * p
        variance += w * w * p * (1 - p) / per[j][1]
    return rate, variance


def raw_rate(pairs):
    wins = sum(1 for _, won in pairs if won)
    p = wins / len(pairs)
    return p, p * (1 - p) / len(pairs)


def difference(r1, v1, r2, v2):
    return (r2 - r1) * 100, 1.96 * math.sqrt(v1 + v2) * 100


def ols_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(r * r for r in resid) / (n - 2) / sxx)
    return slope, 2.447 * se  # t(6, 0.975)


def collect_judge(dataset, result_dir, template):
    """Per-checkpoint raw and standardized rates for one judge, or None."""

    found = {}
    for ck in CHECKPOINTS:
        path = result_dir / template.format(ds=dataset, ck=ck)
        if path.exists():
            found[ck] = load_pairs(path)
    if not found:
        return None

    pooled = defaultdict(int)
    for pairs in found.values():
        for delta, _ in pairs:
            pooled[bin_index(delta)] += 1
    weights = [pooled[j] for j in range(len(BINS))]

    bias = []
    for j, (_, _, label) in enumerate(BINS):
        n = pooled[j]
        if n:
            hits = sum(1 for pairs in found.values()
                       for delta, won in pairs if bin_index(delta) == j and won)
            bias.append((label, hits / n, n))

    rows = {}
    for ck, pairs in found.items():
        pr, pv = raw_rate(pairs)
        sr, sv = standardize(pairs, weights)
        with open(result_dir / template.format(ds=dataset, ck=ck), "r", encoding="utf-8") as f:
            all_rows = json.load(f)
        ties = sum(1 for r in all_rows if r["llm_judge_winner"] == "tie")
        rows[ck] = {"raw": (pr, pv), "std": (sr, sv), "n": len(pairs),
                    "ties": ties, "total": len(all_rows)}

    return {"bias": bias, "rows": rows, "pooled": pooled, "found": found}


def analyse_dataset(dataset):
    result_dir = SCRIPT_DIR / RESULT_FOLDER[dataset]

    judges = []
    for label, template in JUDGES:
        data = collect_judge(dataset, result_dir, template)
        judges.append((label, template, data))

    lines = [f"LENGTH-STANDARDIZED PMI WIN RATES -- {dataset}", "=" * 78]

    # ---- headline: standardized PMI vs ROUGE per checkpoint, both judges ----
    live = [(lab, d) for lab, _, d in judges if d]
    if live:
        lines += ["",
                  "STANDARDIZED PMI vs ROUGE PER CHECKPOINT",
                  "-" * 88,
                  "  Every checkpoint scored against one common mix of PMI-minus-ROUGE",
                  "  summary lengths, so length drift between checkpoints cannot move these.",
                  "",
                  "  PMI and ROUGE are shares of the DECIDED pairs; the tie% column is the",
                  "  share of ALL pairs and sits outside that split. Ties are kept out of the",
                  "  win rate on purpose -- it answers 'when the judge has a preference, how",
                  "  often is it for PMI', which is comparable between judges even though",
                  "  their tie rates are not. A 3-way split would penalise whichever judge",
                  "  simply ties more, and these two differ by ~10 points.",
                  "",
                  "  The two judges' ties are also not the same kind of thing:",
                  "    Prometheus  -- the judge itself answered TIE;",
                  "    GPT         -- almost all are pairs whose verdict FLIPPED when the",
                  "                   candidates were swapped (1M: 1755 of 1792), i.e. our",
                  "                   own bookkeeping for 'no position-independent opinion'.",
                  "  Read GPT's tie% as an instability rate, not as the judge calling it even.",
                  "",
                  "  ROUGE is exactly 100 - PMI: one measurement shown from both sides, NOT",
                  "  independent confirmation. The CI is printed on PMI only for that reason.",
                  ""]
        header = f"  {'ckpt':6s}"
        for lab, _ in live:
            header += f"{lab.split(' (')[0]:>38s}"
        lines.append(header)
        sub = f"  {'':6s}" + "".join(f"{'PMI':>14s}{'ROUGE':>13s}{'tie%':>11s}" for _ in live)
        lines.append(sub)
        for ck in CHECKPOINTS:
            row = f"  {ck:6s}"
            for _, d in live:
                if ck in d["rows"]:
                    r, v = d["rows"][ck]["std"]
                    cell = d["rows"][ck]
                    row += (f"{r * 100:9.2f} +/-{1.96 * math.sqrt(v) * 100:.2f}"
                            f"{(1 - r) * 100:13.2f}"
                            f"{cell['ties'] / cell['total'] * 100:11.2f}")
                else:
                    row += f"{'--':>14s}{'--':>13s}{'--':>11s}"
            lines.append(row)

    # ---- per-judge detail ---------------------------------------------------
    standardized_series = {}
    for judge_label, _, data in judges:
        if data is None:
            lines += ["", f"{judge_label}: no result files found", "-" * 78]
            continue

        lines += ["", judge_label, "-" * 78,
                  "  LENGTH BIAS OF THIS JUDGE (all checkpoints pooled)"]
        for label, rate, n in data["bias"]:
            lines.append(f"    {label:16s}: PMI wins {rate * 100:6.2f}%  (n={n})")

        lines += ["", f"  {'ckpt':6s}{'raw':>18s}{'length-standardized':>26s}{'n':>9s}"]
        raws, stds = [], []
        for ck in CHECKPOINTS:
            if ck not in data["rows"]:
                lines.append(f"  {ck:6s}{'(not run)':>18s}")
                continue
            pr, pv = data["rows"][ck]["raw"]
            sr, sv = data["rows"][ck]["std"]
            raws.append((ck, pr, pv))
            stds.append((ck, sr, sv))
            lines.append(
                f"  {ck:6s}{pr * 100:12.2f} +/-{1.96 * math.sqrt(pv) * 100:.2f}"
                f"{sr * 100:20.2f} +/-{1.96 * math.sqrt(sv) * 100:.2f}"
                f"{data['rows'][ck]['n']:9d}")

        if len(stds) >= 2:
            (_, r1, v1), (_, r2, v2) = stds[0], stds[-1]
            d, ci = difference(r1, v1, r2, v2)
            (_, q1, w1), (_, q2, w2) = raws[0], raws[-1]
            dr, cir = difference(q1, w1, q2, w2)
            lines += ["",
                      f"  {stds[0][0]} -> {stds[-1][0]} raw          : {dr:+.2f} +/-{cir:.2f} pts"
                      f"   {'SIGNIFICANT' if abs(dr) > cir else 'not significant'}",
                      f"  {stds[0][0]} -> {stds[-1][0]} standardized : {d:+.2f} +/-{ci:.2f} pts"
                      f"   {'SIGNIFICANT' if abs(d) > ci else 'not significant'}"]
        if len(stds) >= 3:
            xs = [int(ck[:-1]) for ck, _, _ in stds]
            b, bci = ols_slope(xs, [r * 100 for _, r, _ in stds])
            lines.append(f"  standardized slope       : {b:+.3f} +/-{bci:.3f} pts per 1M steps"
                         f"   {'SIGNIFICANT' if abs(b) > bci else 'not significant'}")
        standardized_series[judge_label] = {ck: r for ck, r, _ in stds}

    # ---- length-difference mix per checkpoint -------------------------------
    ref = next((t for lab, t, d in judges if d), None)
    if ref:
        lines += ["", "LENGTH-DIFFERENCE MIX PER CHECKPOINT (% of decided pairs)",
                  "-" * 78,
                  "  " + f"{'ckpt':6s}" + "".join(f"{lab[:13]:>15s}" for _, _, lab in BINS)
                  + f"{'PMI/ROUGE':>12s}"]
        for ck in CHECKPOINTS:
            path = result_dir / ref.format(ds=dataset, ck=ck)
            if not path.exists():
                continue
            pairs = load_pairs(path)
            counts = defaultdict(int)
            for delta, _ in pairs:
                counts[bin_index(delta)] += 1
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            lp = sum(len(r["pmi_summary"].split()) for r in rows) / len(rows)
            lr = sum(len(r["rouge_summary"].split()) for r in rows) / len(rows)
            lines.append("  " + f"{ck:6s}"
                         + "".join(f"{counts[j] / len(pairs) * 100:14.1f}%" for j in range(len(BINS)))
                         + f"{lp / lr:12.3f}")
        lines.append("  (the last column is the mean word-count ratio; drift here is exactly")
        lines.append("   what the standardization above corrects for)")

    # ---- do the judges agree once length is removed? -----------------------
    if len(standardized_series) == 2:
        (la, sa), (lb, sb) = standardized_series.items()
        common = [ck for ck in CHECKPOINTS if ck in sa and ck in sb]
        if len(common) >= 3:
            u = [sa[ck] for ck in common]
            v = [sb[ck] for ck in common]
            mu, mv = sum(u) / len(u), sum(v) / len(v)
            num = sum((a - mu) * (b - mv) for a, b in zip(u, v))
            den = math.sqrt(sum((a - mu) ** 2 for a in u) * sum((b - mv) ** 2 for b in v))
            lines += ["", "JUDGE AGREEMENT ACROSS CHECKPOINTS (standardized)", "-" * 78,
                      f"  r = {num / den:+.3f} over {len(common)} checkpoints",
                      "  (on wikihow this rises from -0.08 raw to +0.50 standardized: the two",
                      "   judges look uncorrelated only because their length biases point in",
                      "   OPPOSITE directions and each scrambles the ordering differently)"]

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="all",
                        help=f"comma separated subset of {', '.join(RESULT_FOLDER)} (default: all)")
    args = parser.parse_args()

    selected = (list(RESULT_FOLDER) if args.datasets.strip().lower() == "all"
                else [d.strip().lower() for d in args.datasets.split(",") if d.strip()])
    for d in selected:
        if d not in RESULT_FOLDER:
            raise SystemExit(f"Unknown dataset '{d}'. Valid: {', '.join(RESULT_FOLDER)}, all")

    for dataset in selected:
        report = analyse_dataset(dataset)
        out = (SCRIPT_DIR / RESULT_FOLDER[dataset]
               / f"{dataset}_length_standardized_winrates_vs_ref_summaries.log")
        with open(out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(report)
        print(f"\n-> written to {out}\n")
