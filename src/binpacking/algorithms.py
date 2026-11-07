"""
Online bin packing algorithms. Items have integer sizes in 1..k and every bin
has capacity k. Items arrive one at a time and must be placed immediately and
permanently. Each function returns the number of bins used.

Classic algorithms (no predictions)
    next_fit   keep one open bin; if the item does not fit, close it and open a new one.
    first_fit  put the item in the lowest-numbered bin it fits in.
    best_fit   put the item in the fullest bin it fits in.

With predictions (Angelopoulos, Kamali & Shadkami, "Online Bin Packing with
Predictions", IJCAI 2022)
    profile_packing  trust the predicted size frequencies: pack an imagined
                     "profile" of m items offline, then fill its placeholders
                     with real items as they arrive.
    hybrid           send a lambda fraction of each size to profile_packing and
                     the rest to First Fit, trading trust for safety.

Bins are stored as a NumPy array of remaining capacities, so finding the first
or best bin that fits is one vectorized scan per item.
"""
from __future__ import annotations

import math

import numpy as np


class _Bins:
    """Remaining capacity of every opened bin, with First Fit / Best Fit lookups."""

    def __init__(self, k: int, size_hint: int = 1024):
        self.k = k
        self.rem = np.empty(size_hint, dtype=np.int64)
        self.n = 0

    def open(self) -> int:
        if self.n == len(self.rem):
            self.rem = np.concatenate([self.rem, np.empty_like(self.rem)])
        self.rem[self.n] = self.k
        self.n += 1
        return self.n - 1

    def first_fit(self, x: int) -> int:
        fits = np.flatnonzero(self.rem[: self.n] >= x)
        return int(fits[0]) if len(fits) else self.open()

    def best_fit(self, x: int) -> int:
        r = self.rem[: self.n]
        slack = np.where(r >= x, r - x, self.k + 1)
        if self.n == 0 or slack.min() > self.k:
            return self.open()
        return int(np.argmin(slack))

    def place(self, i: int, x: int):
        self.rem[i] -= x


# ----------------------------------------------------------------------------- classic
def next_fit(items, k: int) -> int:
    bins, room = 0, 0
    for x in items:
        if x > room:
            bins, room = bins + 1, k
        room -= x
    return bins


def first_fit(items, k: int) -> int:
    b = _Bins(k)
    for x in items:
        b.place(b.first_fit(x), x)
    return b.n


def best_fit(items, k: int) -> int:
    b = _Bins(k)
    for x in items:
        b.place(b.best_fit(x), x)
    return b.n


# ----------------------------------------------------------------------------- with predictions
def first_fit_decreasing(sizes, k: int) -> list[list[int]]:
    """Offline packing used for the profile: sort big to small, then First Fit."""
    bins: list[list[int]] = []
    room: list[int] = []
    for x in sorted(sizes, reverse=True):
        for j, r in enumerate(room):
            if r >= x:
                bins[j].append(x)
                room[j] -= x
                break
        else:
            bins.append([x])
            room.append(k - x)
    return bins


class _ProfilePacker:
    """
    ProfilePacking, step by step.

    1. Build the profile: for every size x, ceil(f'_x * m) imaginary items of
       size x, where f' are the predicted frequencies.
    2. Pack the profile offline (First Fit Decreasing). Each resulting bin is a
       template made of size-labelled placeholders.
    3. When an item of size x arrives, put it in a free size-x placeholder,
       preferring a bin that already holds a real item. If no size-x
       placeholder is free anywhere, open a fresh copy of all the templates (a
       new "profile group") and use one of its placeholders.
    4. A bin only counts once a real item is in it, so unused placeholders
       cost nothing.
    Items whose size never appears in the profile have no placeholder; they are
    packed separately with First Fit.

    Bookkeeping: for each size x we keep two stacks of bin ids, one entry per
    free size-x placeholder: ``in_used`` (bins that already hold an item) and
    ``in_unused`` (bins that are still empty). When an empty bin receives its
    first item, all its placeholders are pushed onto the ``in_used`` stacks and
    its old ``in_unused`` entries are skipped when popped (lazy deletion).
    """

    def __init__(self, pred_freq: np.ndarray, k: int, m: int):
        profile = [x for x in range(1, k + 1) for _ in range(math.ceil(pred_freq[x] * m))]
        self.templates = first_fit_decreasing(profile, k)
        self.sizes_in_profile = set(profile)
        self.in_used: dict[int, list[int]] = {}
        self.in_unused: dict[int, list[int]] = {}
        self.bin_template: list[int] = []            # template index of each opened bin
        self.bin_used: list[bool] = []
        self.fallback = _Bins(k)

    def has_free_placeholder_in_used_bin(self, x: int) -> bool:
        return bool(self.in_used.get(x))

    def _new_group(self):
        for t, template in enumerate(self.templates):
            j = len(self.bin_used)
            self.bin_template.append(t)
            self.bin_used.append(False)
            for x in template:
                self.in_unused.setdefault(x, []).append(j)

    def _pop_unused(self, x: int):
        stack = self.in_unused.get(x, [])
        while stack:
            j = stack.pop()
            if not self.bin_used[j]:
                return j
        return None

    def add(self, x: int):
        if x not in self.sizes_in_profile:                  # no placeholder will ever exist
            self.fallback.place(self.fallback.first_fit(x), x)
            return
        if self.in_used.get(x):                              # 3a. free slot in a bin already in use
            self.in_used[x].pop()
            return
        j = self._pop_unused(x)
        if j is None:                                        # 3b. no free size-x slot anywhere
            self._new_group()
            j = self._pop_unused(x)
        self.bin_used[j] = True                              # bin j now counts
        for y in self.templates[self.bin_template[j]]:
            self.in_used.setdefault(y, []).append(j)
        self.in_used[x].remove(j)                            # the slot this item just took

    def bins(self) -> int:
        return sum(self.bin_used) + self.fallback.n


def profile_packing(items, k: int, pred_freq: np.ndarray, m: int = 1000) -> int:
    p = _ProfilePacker(pred_freq, k, m)
    for x in items:
        p.add(int(x))
    return p.bins()


def hybrid(items, k: int, pred_freq: np.ndarray, lam: float, m: int = 1000) -> int:
    """Hybrid(lambda): lambda = 1 is ProfilePacking, lambda = 0 is First Fit.

    An item that fits a free placeholder in a bin ProfilePacking already uses
    goes there (it costs nothing extra). Otherwise it goes to ProfilePacking
    only if fewer than a lambda fraction of the items of its size seen so far
    were sent there; else it goes to First Fit.
    """
    p = _ProfilePacker(pred_freq, k, m)
    ff = _Bins(k)
    seen: dict[int, int] = {}
    to_pp: dict[int, int] = {}
    for x in items:
        x = int(x)
        seen[x] = seen.get(x, 0) + 1
        if p.has_free_placeholder_in_used_bin(x) or to_pp.get(x, 0) < lam * seen[x]:
            p.add(x)
            to_pp[x] = to_pp.get(x, 0) + 1
        else:
            ff.place(ff.first_fit(x), x)
    return p.bins() + ff.n
