"""Online bin packing on real VM arrivals from the Azure packing trace.

Instances: each day of week 2 (days 7..13) of VM requests that can run on a
given machine type, packed in their real arrival order. Two machine types,
chosen by a fixed rule before looking at any packing result:
    21  the machine type that can host the most requests (typical traffic)
    28  among machine types with at least 1M requests, the one with the
        largest average VM size (VMs are big relative to the machine)

Frequency predictions for day d:
    week1     size frequencies of all of week 1 (days 0..6): what you would
              actually have, including any drift between the weeks
    prev_day  size frequencies of day d - 1
    oracle    the true frequencies of day d itself (not available in
              practice; the best case for the prediction-based algorithms)

Score: bins used / ceil(total size / 100), an upper bound on the ratio to
the optimum. Writes artifacts/azure_results.json and artifacts/azure.png.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from binpacking import best_fit, first_fit, hybrid, l1_error, lower_bound, next_fit, profile_packing, true_frequencies
from binpacking.azure import load_azure_items
from binpacking.env import environment

K = 100
MACHINES = {21: "most requests (typical traffic)", 28: "largest average VM size"}
DAYS = range(7, 14)
LAMS = (0.25, 0.5, 0.75)


def _run(task):
    name, items, pred, m = task
    if name == "next_fit":
        return next_fit(items, K)
    if name == "first_fit":
        return first_fit(items, K)
    if name == "best_fit":
        return best_fit(items, K)
    if name == "profile_packing":
        return profile_packing(items, K, pred, m)
    return hybrid(items, K, pred, float(name.split("_")[1]), m)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, default=5000, help="profile size")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts"
    out.mkdir(exist_ok=True)

    tasks, keys, meta = [], [], {}
    for mach in MACHINES:
        sizes, day = load_azure_items(mach, root / "data", K)
        dint = np.floor(day).astype(int)
        week1 = true_frequencies(sizes[dint < 7], K)
        meta[mach] = {"requests_14_days": int(len(sizes)), "mean_size_pct": round(float(sizes.mean()), 2),
                      "share_items_over_25pct": round(float((sizes > 25).mean()), 4),
                      "distinct_sizes": int(len(np.unique(sizes))), "days": {}}
        for d in DAYS:
            items = sizes[dint == d]
            f = true_frequencies(items, K)
            preds = {"week1": week1, "prev_day": true_frequencies(sizes[dint == d - 1], K), "oracle": f}
            meta[mach]["days"][d] = {"n_items": int(len(items)), "lower_bound": lower_bound(items, K),
                                     "eta": {p: round(l1_error(v, f), 4) for p, v in preds.items()}}
            for name in ("next_fit", "first_fit", "best_fit"):
                tasks.append((name, items, None, args.m))
                keys.append((mach, d, name, None))
            for p, v in preds.items():
                for name in ["profile_packing"] + [f"hybrid_{lam}" for lam in LAMS]:
                    tasks.append((name, items, v, args.m))
                    keys.append((mach, d, name, p))

    with ProcessPoolExecutor(args.workers) as ex:
        bins = list(ex.map(_run, tasks))

    ratio = {}
    for (mach, d, name, p), b in zip(keys, bins):
        label = name if p is None else f"{name}[{p}]"
        ratio.setdefault(mach, {}).setdefault(label, {})[d] = b / meta[mach]["days"][d]["lower_bound"]
    summary = {m: {lab: {"mean": round(float(np.mean(list(v.values()))), 4),
                         "max": round(float(np.max(list(v.values()))), 4),
                         "per_day": {d: round(x, 4) for d, x in v.items()}} for lab, v in r.items()}
               for m, r in ratio.items()}
    res = {"data": "Azure Trace for Packing 2020, zone A (Microsoft, CC-BY); Hadary et al., OSDI 2020",
           "size": "ceil(100 * max(cpu, memory) fraction of the machine); bin capacity 100",
           "instances": "each day of week 2 (days 7-13), real arrival order",
           "profile_size_m": args.m,
           "machines": {m: {"why": MACHINES[m], **meta[m], "ratio_to_lower_bound": summary[m]} for m in MACHINES}}
    res["environment"] = environment()
    (out / "azure_results.json").write_text(json.dumps(res, indent=2))

    for m in MACHINES:
        etas = {p: np.mean([meta[m]["days"][d]["eta"][p] for d in DAYS]) for p in ("week1", "prev_day", "oracle")}
        print(f"\nmachine {m} ({MACHINES[m]}): {meta[m]['requests_14_days']:,} requests, mean size "
              f"{meta[m]['mean_size_pct']}%, mean eta week1 {etas['week1']:.3f} prev_day {etas['prev_day']:.3f}")
        for lab, s in sorted(summary[m].items(), key=lambda t: t[1]["mean"]):
            print(f"  {lab:32s} mean {s['mean']:.4f}  worst day {s['max']:.4f}")

    # plot: mean ratio per algorithm, by prediction source, for each machine
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    algos = ["profile_packing"] + [f"hybrid_{lam}" for lam in LAMS]
    colors = {"week1": "tab:blue", "prev_day": "tab:orange", "oracle": "tab:green"}
    for ax, m in zip(axes, MACHINES):
        x = np.arange(len(algos))
        for i, p in enumerate(colors):
            ax.bar(x + (i - 1) * 0.27, [summary[m][f"{a}[{p}]"]["mean"] for a in algos], 0.27,
                   color=colors[p], label=f"prediction: {p}")
        for name, ls in (("first_fit", "-"), ("best_fit", "--")):
            ax.axhline(summary[m][name]["mean"], color="k", ls=ls, lw=1, label=name)
        ax.set_xticks(x, ["Profile", "Hybrid .25", "Hybrid .5", "Hybrid .75"])
        lo = min(min(v["mean"] for v in summary[m].values()), 1.0)
        ax.set_ylim(lo - 0.002, max(summary[m][f"{a}[{p}]"]["mean"] for a in algos for p in colors) + 0.004)
        ax.set_title(f"machine {m}: {MACHINES[m]}")
        ax.set_ylabel("bins / lower bound (mean of 7 days)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "azure.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
