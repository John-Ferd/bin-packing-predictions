"""Compare online bin packing algorithms as frequency predictions get worse.

Experiment 1 (prediction quality): the prediction is a mix of the true size
frequencies and a wrong distribution of small items, with mixing weight delta
from 0 (perfect) to 1 (entirely wrong). Error eta = L1 distance.

Experiment 2 (learned predictions): frequencies estimated from the first s
items of an independent sample, for growing s.

Reported number: bins used / lower bound ceil(total size / k), averaged over
seeds. This is an upper bound on each algorithm's ratio to the optimum.
Writes artifacts/results.json, robustness.png and learned.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from binpacking import (
    best_fit,
    first_fit,
    hybrid,
    l1_error,
    lower_bound,
    predicted_frequencies,
    profile_packing,
    true_frequencies,
)
from binpacking.instances import sample_items, weibull_dist

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--n", type=int, default=100_000)
ap.add_argument("--m", type=int, default=5000, help="profile size")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.n, args.m, args.seeds = 5000, 1000, 1
out = Path(__file__).resolve().parents[1] / "artifacts"
if args.quick:
    out = out / "quick"
out.mkdir(parents=True, exist_ok=True)

K = 100
TRUE = weibull_dist(K, shape=3.0, scale=45.0, seed=0)
WRONG = weibull_dist(K, shape=3.0, scale=20.0, seed=1)
deltas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lams = [0.25, 0.5, 0.75]
sample_sizes = [10, 30, 100, 300, 1000, 10_000]

rows, learned = [], []
for seed in range(args.seeds):
    items = sample_items(TRUE, args.n, seed=100 + seed)
    f = true_frequencies(items, K)
    lb = lower_bound(items, K)
    ff, bf = first_fit(items, K) / lb, best_fit(items, K) / lb
    for d in deltas:
        pred = predicted_frequencies(f, WRONG, d)
        r = {"seed": seed, "delta": d, "eta": l1_error(pred, f), "first_fit": ff, "best_fit": bf,
             "profile_packing": profile_packing(items, K, pred, args.m) / lb}
        for lam in lams:
            r[f"hybrid_{lam}"] = hybrid(items, K, pred, lam, args.m) / lb
        rows.append(r)
    for s in sample_sizes:
        pred = true_frequencies(sample_items(TRUE, s, seed=900 + seed), K)
        learned.append({"seed": seed, "sample_size": s, "eta": l1_error(pred, f), "first_fit": ff,
                        "profile_packing": profile_packing(items, K, pred, args.m) / lb,
                        "hybrid_0.5": hybrid(items, K, pred, 0.5, args.m) / lb})
    print(f"seed {seed} done", flush=True)


def average(table, key):
    out_rows = []
    for v in sorted({r[key] for r in table}):
        group = [r for r in table if r[key] == v]
        out_rows.append({c: float(np.mean([g[c] for g in group])) for c in group[0] if c != "seed"})
    return out_rows


res = {"n_items": args.n, "capacity": K, "profile_size_m": args.m, "seeds": args.seeds,
       "true_distribution": "round(45 * Weibull(shape 3)), clipped to 1..100",
       "wrong_distribution": "round(20 * Weibull(shape 3)), clipped to 1..100",
       "prediction_quality": average(rows, "delta"), "learned_predictions": average(learned, "sample_size")}
(out / "results.json").write_text(json.dumps(res, indent=2))

cols = ["first_fit", "best_fit", "profile_packing"] + [f"hybrid_{lam}" for lam in lams]
print("\ndelta   eta  " + "".join(c.rjust(16) for c in cols))
for r in res["prediction_quality"]:
    print(f"{r['delta']:5.1f} {r['eta']:5.2f}  " + "".join(f"{r[c]:16.4f}" for c in cols))
print("\nsample  eta   first_fit  profile_packing  hybrid_0.5")
for r in res["learned_predictions"]:
    print(f"{int(r['sample_size']):6d} {r['eta']:5.2f}  {r['first_fit']:9.4f}  {r['profile_packing']:15.4f}"
          f"  {r['hybrid_0.5']:10.4f}")

labels = {"first_fit": "First Fit", "best_fit": "Best Fit", "profile_packing": "ProfilePacking",
          "hybrid_0.25": "Hybrid(0.25)", "hybrid_0.5": "Hybrid(0.5)", "hybrid_0.75": "Hybrid(0.75)"}
fig, ax = plt.subplots(figsize=(6.5, 4))
eta = [r["eta"] for r in res["prediction_quality"]]
for c in cols:
    style = "--" if c in ("first_fit", "best_fit") else "o-"
    ax.plot(eta, [r[c] for r in res["prediction_quality"]], style, ms=4, label=labels[c])
ax.set_xlabel("prediction error  eta = L1 distance of frequencies")
ax.set_ylabel("bins used / lower bound")
ax.set_title("Trusting predictions: gain when right, loss when wrong")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "robustness.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.8))
L = res["learned_predictions"]
x = [r["sample_size"] for r in L]
for c in ("first_fit", "profile_packing", "hybrid_0.5"):
    ax.plot(x, [r[c] for r in L], "--" if c == "first_fit" else "o-", label=labels[c])
ax.set_xscale("log")
ax.set_xlabel("items used to learn the frequencies")
ax.set_ylabel("bins used / lower bound")
ax.set_title("Predictions learned from a sample")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "learned.png", dpi=130)
plt.close(fig)
