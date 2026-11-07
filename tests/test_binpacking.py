import numpy as np
import pytest

from binpacking import best_fit, first_fit, hybrid, l1_error, lower_bound, next_fit, profile_packing, true_frequencies
from binpacking.algorithms import first_fit_decreasing
from binpacking.instances import sample_items, weibull_dist


def test_next_fit_only_looks_at_the_current_bin():
    # [4] -> 7 does not fit, open [7] -> 3 fits, [7,3] -> 6 does not fit, open [6]
    assert next_fit([4, 7, 3, 6], 10) == 3


def test_first_fit_vs_best_fit():
    # Best Fit puts the 3 into the bin with 3 left (the 7), so the 6 fits with the 4.
    # First Fit puts the 3 into the first bin (the 4), and the 6 needs a new bin.
    assert first_fit([4, 7, 3, 6], 10) == 3
    assert best_fit([4, 7, 3, 6], 10) == 2


def test_first_fit_bad_order_vs_profile_packing():
    # 100 items of size 4, then 100 of size 6. First Fit pairs the 4s (two per
    # bin) and then needs a new bin per 6: 50 + 100 = 150 bins. The optimum is
    # 100 bins of {4, 6}. With the right frequencies ProfilePacking reserves a
    # 6-slot next to every 4 and finds the optimum.
    k = 10
    items = [4] * 100 + [6] * 100
    pred = true_frequencies(items, k)
    assert first_fit(items, k) == 150
    assert profile_packing(items, k, pred, m=20) == 100


def test_hybrid_endpoints_match_first_fit_and_profile_packing():
    k = 100
    items = sample_items(weibull_dist(k), 3000, seed=3)
    pred = true_frequencies(sample_items(weibull_dist(k), 500, seed=4), k)
    assert hybrid(items, k, pred, 0.0, m=500) == first_fit(items, k)
    assert hybrid(items, k, pred, 1.0, m=500) == profile_packing(items, k, pred, m=500)


def test_unpredicted_sizes_fall_back_to_first_fit():
    k = 10
    pred = np.zeros(k + 1)
    pred[6] = 1.0                                     # prediction says: only size 6
    items = [6] * 10 + [3] * 10
    assert profile_packing(items, k, pred, m=10) == 10 + first_fit([3] * 10, k)


@pytest.mark.parametrize("algo", ["next_fit", "first_fit", "best_fit", "profile_packing", "hybrid"])
def test_never_better_than_the_lower_bound(algo):
    k = 100
    items = sample_items(weibull_dist(k), 2000, seed=5)
    pred = true_frequencies(items, k)
    fn = {"next_fit": lambda: next_fit(items, k), "first_fit": lambda: first_fit(items, k),
          "best_fit": lambda: best_fit(items, k), "profile_packing": lambda: profile_packing(items, k, pred, 500),
          "hybrid": lambda: hybrid(items, k, pred, 0.5, 500)}[algo]
    assert fn() >= lower_bound(items, k)


def test_first_fit_decreasing_respects_capacity():
    bins = first_fit_decreasing([7, 5, 5, 4, 3, 2, 2], 10)
    assert all(sum(b) <= 10 for b in bins)
    assert sorted(x for b in bins for x in b) == [2, 2, 3, 4, 5, 5, 7]


def test_l1_error_and_lower_bound():
    assert l1_error(np.array([0, 0.5, 0.5]), np.array([0, 1.0, 0.0])) == 1.0
    assert lower_bound([6, 5, 4], 10) == 2
