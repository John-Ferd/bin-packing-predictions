# bin-packing-predictions

[![CI](https://github.com/John-Ferd/bin-packing-predictions/actions/workflows/ci.yml/badge.svg)](https://github.com/John-Ferd/bin-packing-predictions/actions/workflows/ci.yml) ![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

Online bin packing, where items arrive one at a time and each must be put in a bin immediately and for good, with and without **predictions** of how often each item size will appear. The repo implements the classic algorithms (Next Fit, First Fit, Best Fit) and the prediction-based *ProfilePacking* and *Hybrid(λ)* of Angelopoulos, Kamali & Shadkami (IJCAI 2022), then measures the core trade-off in learning-augmented algorithms: how much you gain when predictions are good, and how much you lose when they are bad. It does this first on controlled synthetic inputs, then on **over two million real virtual-machine arrivals from Microsoft Azure**, where packing VMs onto as few servers as possible is exactly this problem.

## In one paragraph

Classic online algorithms such as First Fit have to work on every possible input, so they are cautious; on typical inputs First Fit uses about 4% more bins than the lower bound. If you know roughly how often each item size will appear, you can plan ahead: ProfilePacking packs an imaginary "profile" of items with those frequencies offline, then fills its slots as real items arrive. With accurate predictions it came within **0.9%** of the lower bound on 100,000 items. When the prediction left out the large sizes entirely, it fell to **23%** above the bound. Hybrid(λ) sends only a λ fraction of each size to ProfilePacking, which caps how bad things can get, at the price of a smaller gain.

On **real Azure VM traffic** the picture changes: First Fit is already at or within 1.4% of the lower bound, so predictions have little to gain, and frequencies learned from the previous week can be badly wrong because the mix of large VMs shifts. ProfilePacking with last week's frequencies used up to 2.6x the lower bound on one machine type, while Hybrid(0.25) stayed within 4% on average, which is the robustness argument for Hybrid made concrete.

## Results

The setup: 100,000 items with integer sizes from round(45·Weibull(3)), bins of capacity 100, profile size m = 5,000, averaged over 3 random sequences (`make experiments`). The score is **bins used ÷ ⌈total size / 100⌉**. Since no packing can beat that lower bound, 1.00 would be perfect.

**1. Prediction quality.** The prediction mixes the true size frequencies with a wrong distribution of small items, with weight δ on the wrong one. The error η is the L1 distance between predicted and true frequencies.

| δ | η | First Fit | Best Fit | ProfilePacking | Hybrid(0.25) | Hybrid(0.5) | Hybrid(0.75) |
|---|---|---|---|---|---|---|---|
| 0.0 (perfect) | 0.00 | 1.041 | 1.039 | **1.009** | 1.033 | 1.024 | 1.017 |
| 0.5 | 0.72 | 1.041 | 1.039 | **1.009** | 1.033 | 1.025 | 1.017 |
| 0.9 | 1.30 | 1.041 | 1.039 | **1.011** | 1.033 | 1.024 | 1.016 |
| 1.0 (all wrong) | 1.44 | 1.041 | **1.039** | 1.229 | 1.085 | 1.126 | 1.171 |

<p align="center"><img src="artifacts/robustness.png" width="60%"></p>

- **Consistency:** with good predictions, ProfilePacking uses about 3% fewer bins than First Fit (1.009 vs 1.041).
- **Not robust:** at δ = 1 the prediction gives zero frequency to every size above 53, which is 29% of the real items. ProfilePacking then does far worse than First Fit (1.229 vs 1.041).
- **λ is the dial between the two.** Smaller λ gains less when predictions are right and loses less when they are wrong. Hybrid(0.25) is never worse than 1.085.
- **What surprised me:** the L1 error η barely matters until it crosses that threshold. ProfilePacking stayed within about 1% of the bound up to η = 1.30, as long as every size that really occurs had *some* predicted frequency. What hurts is predicting zero for sizes that do appear, not getting the frequencies slightly wrong. A single number like η hides that distinction.

**2. Predictions learned from a sample.** The predicted frequencies are the size counts from the first s items of an independent sample:

| sample size s | η | First Fit | ProfilePacking | Hybrid(0.5) |
|---|---|---|---|---|
| 10 | 1.65 | 1.041 | 1.078 | 1.048 |
| 100 | 0.62 | 1.041 | 1.191 | 1.071 |
| 1,000 | 0.21 | 1.041 | 1.061 | **1.024** |
| 10,000 | 0.06 | 1.041 | **1.015** | 1.023 |

<p align="center"><img src="artifacts/learned.png" width="55%"></p>

- **Too few samples hurt pure ProfilePacking.** With s = 100 there are about 70 sizes that occur regularly but only 100 observations, so many sizes get a frequency of 0 or 1/100. The profile is a poor plan, and the algorithm keeps opening new profile groups.
- **Small samples can look better than medium ones.** With s = 10 most sizes are missing from the prediction, so most items go straight to the First Fit fallback, which is why s = 10 looks better than s = 100.
- Hybrid(0.5) beats First Fit once the sample has about 1,000 items. Pure ProfilePacking needs about 10,000.

**3. Real data: Azure VM placement** (`make azure`, about 1 minute after a one-time 53 MB download).

The [Azure Trace for Packing 2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md) (Microsoft, CC-BY; Hadary et al., *Protean*, OSDI 2020) lists 14 days of VM requests in one Azure zone, and how much of each machine type's CPU and memory every VM type needs. Packing arriving VMs onto as few machines as possible is online bin packing:
- **Item size:** the VM's dominant resource, max(CPU share, memory share), rounded *up* to a whole percent of the machine. Capacity is 100. Rounding up means any packing of the rounded sizes also fits the real VMs.
- **Instances:** each day of week 2 (days 7 to 13), in real arrival order: 7 instances per machine type.
- **Two machine types, chosen by a rule fixed before running anything:** the one that can host the most requests (machine 21: 2.1M VMs in week 2, average size 9.3%, 24 distinct sizes) and, among types with at least 1M requests, the one with the largest average VM size (machine 28: 0.6M VMs, average 20.1%, 28% of VMs above a quarter of a machine).
- **Predictions:**
  - `week1`: size frequencies from all of week 1. This is what you would actually have in practice, including drift between the weeks.
  - `prev_day`: frequencies from the day before.
  - `oracle`: the day's own true frequencies, which are unknowable in practice and give the best case for prediction.

Mean of bins / lower bound over the 7 days (worst day in brackets):

| algorithm | machine 21 (typical) | machine 28 (large VMs) |
|---|---|---|
| Next Fit | 1.074 | 1.246 |
| **First Fit** | **1.000** (1.000) | **1.014** (1.015) |
| Best Fit | 1.000 | 1.014 |
| ProfilePacking, oracle | 1.008 | 1.022 |
| ProfilePacking, week1 | 1.020 (1.038) | 1.868 (**2.615**) |
| ProfilePacking, prev_day | 1.027 (1.085) | 1.240 (1.651) |
| Hybrid(0.25), week1 | 1.003 (1.007) | 1.037 (1.099) |
| Hybrid(0.25), prev_day | 1.001 (1.003) | 1.016 (1.023) |
| Hybrid(0.5), week1 | 1.007 (1.016) | 1.208 (1.522) |

<p align="center"><img src="artifacts/azure.png" width="85%"></p>

- **First Fit is already (near) optimal on real traffic.** On machine 21 it meets the lower bound exactly on every day: most VMs are small, and they fill whatever gaps the big ones leave. With no gap to close, predictions can only cost bins, and every prediction-based variant, even with oracle frequencies, is slightly worse than First Fit.
- **Last week's frequencies can be badly wrong.** On machine 28, 35% of week-1 VMs used at least a third of a machine, against 17% on day 10. ProfilePacking reserves template space for large VMs that never arrive, so its bins run mostly empty: 1.87x the bound on average and 2.6x on the worst day. Setting only the large sizes' frequencies to their true values brings that day from 1.99 to 1.05, which confirms the cause. On the synthetic data the damage came from *under*-predicting sizes; here it comes from *over*-predicting large ones.
- **Hybrid(λ) with small λ is the safe choice.** Hybrid(0.25) stayed within 4% of the bound on average, and within 10% on the worst day, even with the drifted week-1 prediction. That is the paper's robustness argument showing up on real data.
- **Fresher predictions are better.** The previous day's frequencies were about 2.5x closer (in L1 error) to the true ones than last week's, and every prediction-based algorithm except ProfilePacking on machine 21 did better with them.

## Architecture

```mermaid
flowchart LR
  P["predicted size frequencies"] --> PR["profile<br/>packed offline with<br/>First Fit Decreasing"]
  W["synthetic Weibull items<br/>or real Azure VM arrivals"] --> I["items arrive one at a time"]
  I --> H{"Hybrid(λ)<br/>route each item"}
  H -->|"λ share of each size"| PP["ProfilePacking<br/>fill profile slots"]
  H -->|"the rest"| FF["First Fit"]
  PR --> PP
  PP --> B["bins used vs lower bound"]
  FF --> B
```

## How the algorithms work

**Classic online algorithms** (`algorithms.py`). Bins are a NumPy array of remaining capacities.

| algorithm | rule | known asymptotic worst-case ratio |
|---|---|---|
| Next Fit | keep one open bin; if the item doesn't fit, start a new bin | 2 |
| First Fit | lowest-numbered bin with room | 1.7 |
| Best Fit | fullest bin with room | 1.7 |

**ProfilePacking:**
1. Build a profile of m imaginary items: ⌈f′ₓ·m⌉ items of each size x, where f′ is the predicted frequency.
2. Pack the profile offline with First Fit Decreasing. Each resulting bin becomes a *template* of size-labelled slots.
3. When a real item of size x arrives, put it in a free x-slot, preferring a bin that is already in use. If no x-slot is free anywhere, open a new copy of all the templates (a new "profile group").
4. A bin only counts once it holds a real item. Sizes that aren't in the profile at all are packed separately with First Fit.

**Hybrid(λ):** an item that fits a free slot in a bin ProfilePacking is already using always goes there, since it costs nothing extra. Otherwise it goes to ProfilePacking only while fewer than a λ fraction of the items of its size have been sent there, and to First Fit if not. λ = 0 is exactly First Fit and λ = 1 is exactly ProfilePacking, and the tests check both endpoints.

**A small example where predictions help** (tested): 100 items of size 4 arrive, then 100 of size 6, with capacity 10.
- First Fit puts two 4s in each bin, then needs a new bin for every 6: **150 bins**.
- ProfilePacking knows sizes 4 and 6 are equally common, so its template is a bin with one 4-slot and one 6-slot. Every 4 gets a bin with a 6-slot reserved next to it: **100 bins**, which is optimal.

## Tests

`make test` runs 14 tests:
- Next Fit, First Fit and Best Fit on hand-checked examples, including a case where First Fit and Best Fit differ.
- The 4/6 example above: First Fit 150 vs ProfilePacking 100.
- Hybrid(0) = First Fit and Hybrid(1) = ProfilePacking.
- Sizes missing from the prediction fall back to First Fit.
- No algorithm ever beats the lower bound.
- First Fit Decreasing respects capacity.
- The error and lower-bound helpers are correct.
- Azure VM sizes are the dominant resource rounded up, and the trace loader keeps only the 14-day window and the chosen machine type, in arrival order (checked on a tiny fake trace, so no download is needed).

## How to run

```bash
pip install -e ".[dev]"     # or: make setup
make test
make experiments            # ~20 minutes: synthetic tables above, artifacts/results.json, robustness.png, learned.png
make azure                  # real Azure VM trace: one-time 53 MB download into data/, then ~1 minute
                            # -> artifacts/azure_results.json, azure.png
```

## Limitations

- Synthetic experiments use one item-size family (Weibull) and one way of corrupting predictions.
- The Azure instances are simplified: two resources are reduced to one size, and VMs never leave (classic online bin packing has no departures). Real VM placement is multi-dimensional and dynamic.
- Scores are relative to the simple lower bound ⌈total size / capacity⌉, so they overstate each algorithm's true ratio to the optimum.
- The profile is packed with First Fit Decreasing rather than an exact solver. The paper's experiments do the same.

## Reference

S. Angelopoulos, S. Kamali, K. Shadkami. *Online Bin Packing with Predictions.* IJCAI 2022; extended version in JAIR 2023. [arXiv:2102.03311](https://arxiv.org/abs/2102.03311). This repo is an independent, simplified reimplementation for learning purposes.

O. Hadary et al. *Protean: VM Allocation Service at Scale.* OSDI 2020. Data: [Azure Public Dataset, Azure Trace for Packing 2020](https://github.com/Azure/AzurePublicDataset), CC-BY.
