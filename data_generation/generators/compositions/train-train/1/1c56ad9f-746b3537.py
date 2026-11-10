"""Run-clock lattices: color-run distance controls a held four-step wave.

Only NumPy and the standard library are used.  Generation renders a latent
scaffold; solve reconstructs its clock from the supplied input alone.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _phase(distance):
    return (0, -1, 0, 1)[distance % 4]


def _runs(values):
    """Return labels, starts and exclusive ends of maximal adjacent runs."""
    values = np.asarray(values)
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    return tuple(int(values[s]) for s in starts), starts, ends


def _analyze(value):
    """Parse either orientation into a canonical horizontal-band lattice."""
    a = _grid(value)
    bg = int(a[0, 0])
    mask = a != bg
    rr, cc = np.where(mask)
    _require(len(rr) > 0, "empty input")
    top, bottom, left, right = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    row_uniform = all(len(np.unique(a[r, mask[r]])) == 1 for r in range(top, bottom + 1))
    col_uniform = all(len(np.unique(a[mask[:, c], c])) == 1 for c in range(left, right + 1))
    _require(row_uniform != col_uniform, "color-changing axis must be unambiguous")
    axis = 0 if row_uniform else 1
    g = a if axis == 0 else a.T
    m = g != bg
    rr, cc = np.where(m)
    t, b, l, r = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    _require(l >= 1 and r < g.shape[1] - 1, "one-cell transverse clearance required")
    _require(b - t + 1 >= 4 and r - l + 1 >= 3, "lattice too small")
    core = m[t:b + 1, l:r + 1]
    _require(np.all(core.any(axis=1)), "empty occupied slice")
    _require(np.all(core[0]) and np.all(core[-1]), "end crossbars must span the lattice")
    rails = np.all(core[1:-1], axis=0)
    _require(rails[0] and rails[-1], "continuous outer supports required")
    starts = np.flatnonzero(rails & ~np.r_[False, rails[:-1]])
    ends = np.flatnonzero(rails & ~np.r_[rails[1:], False]) + 1
    supports = tuple((int(x), int(y)) for x, y in zip(starts, ends))
    _require(2 <= len(supports) <= 6, "need 2..6 separated supports")
    _require(sum(np.array_equal(row, rails) for row in core[1:-1]) >= 2,
             "need two bare support slices")
    bars, partial = [], []
    for i, row in enumerate(core[1:-1], start=1):
        _require(np.all(row[rails]), "interrupted support")
        fills = []
        for j in range(len(supports) - 1):
            segment = row[supports[j][1]:supports[j + 1][0]]
            _require(len(segment) and (np.all(segment) or not np.any(segment)), "broken crossbar")
            fills.append(bool(np.all(segment)))
        if any(fills):
            bars.append(t + i)
            if not all(fills):
                partial.append(t + i)
    labels = np.asarray([int(g[row, l]) for row in range(t, b + 1)], dtype=np.int8)
    colors, run_starts, run_ends = _runs(labels)
    _require(4 <= len(colors) <= 30, "need at least four color runs")
    rank = np.empty(len(labels), dtype=int)
    for i, (s, e) in enumerate(zip(run_starts, run_ends)):
        rank[s:e] = len(colors) - 1 - i
    return dict(g=g, mask=m, bg=bg, axis=axis, top=t, bottom=b, left=l, right=r,
                labels=labels, colors=colors, starts=run_starts, ends=run_ends,
                widths=tuple(int(e - s) for s, e in zip(run_starts, run_ends)),
                rank=rank, supports=supports, bars=tuple(bars), partial=tuple(partial),
                height=b - t + 1, width=r - l + 1,
                end_padding=g.shape[0] - b - 1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer orientation, color runs and anchor entirely from the input."""
    info = _analyze(input_grid)
    g, bg = info["g"], info["bg"]
    out = np.full_like(g, bg)
    for i, row in enumerate(range(info["top"], info["bottom"] + 1)):
        columns = np.flatnonzero(g[row] != bg)
        out[row, columns + _phase(int(info["rank"][i]))] = g[row, columns]
    return out.copy() if info["axis"] == 0 else out.T.copy()


def _support_mask(rng, count):
    widths = [int(rng.integers(1, 4)) for _ in range(count)]
    gaps = [1] * (count - 1)
    budget = 28 - sum(widths) - sum(gaps)
    for i in rng.permutation(count - 1):
        extra = int(rng.integers(0, min(5, budget) + 1))
        gaps[int(i)] += extra
        budget -= extra
    supports, cursor = [], 0
    for i, w in enumerate(widths):
        supports.append((cursor, cursor + w))
        cursor += w + (gaps[i] if i < count - 1 else 0)
    return supports, cursor


def _sequence(rng, count, bg, recurrence):
    maximum = min(9, count - int(recurrence))
    distinct = int(rng.integers(2, maximum + 1))
    palette = [int(v) for v in rng.choice([c for c in range(10) if c != bg],
                                        size=distinct, replace=False)]
    seq = [int(v) for v in rng.permutation(palette)]
    while len(seq) < count:
        choices = [(p, c) for p in range(len(seq) + 1) for c in palette
                   if (p == 0 or seq[p - 1] != c) and (p == len(seq) or seq[p] != c)]
        p, c = choices[int(rng.integers(len(choices)))]
        seq.insert(p, c)
    return seq


def _thicknesses(rng, count, regime, demo):
    if demo:
        values = rng.integers(1, 4, size=count)
        values[0], values[-1] = rng.integers(2, 4, size=2)
        values[int(rng.integers(1, count - 1))] = 1
        return values
    if regime == "dwell":
        if count == 4 and rng.random() < 0.25:
            return np.full(count, 6, dtype=int)
        values = rng.integers(1, 4, size=count)
        held = int(rng.integers(count))
        values[held] = int(rng.integers(6, 11))
        while values.sum() > 26:
            eligible = [i for i in range(count) if i != held and values[i] > 1]
            values[int(rng.choice(eligible))] -= 1
        return values
    limit = 3 if regime == "train" else 4
    mode = int(rng.integers(4))
    if mode == 0:
        values = np.ones(count, dtype=int)
    elif mode == 1:
        values = np.full(count, int(rng.integers(2, min(limit, 26 // count) + 1)), dtype=int)
    elif mode == 2:
        values = np.ones(count, dtype=int)
        values[int(rng.integers(count))] = int(rng.integers(2, limit + 1))
    else:
        values = rng.integers(1, limit + 1, size=count)
    while values.sum() > 26:
        eligible = np.flatnonzero(values > 1)
        values[int(rng.choice(eligible))] -= 1
    return values


def _make_pair(rng, axis, regime="train", demo=0, count=None):
    if count is None:
        count = int(rng.integers(9, 13) if regime == "long" else rng.integers(4, 9))
    bg = 0 if rng.random() < 0.5 else int(rng.integers(1, 10))
    colors = _sequence(rng, count, bg, recurrence=(demo == 1))
    widths = _thicknesses(rng, count, regime, bool(demo))
    h = int(sum(widths))
    support_count = 2 if demo == 1 else int(rng.integers(3 if demo == 2 else 2, 7))
    supports, w = _support_mask(rng, support_count)
    mask = np.zeros((h, w), dtype=bool)
    for start, end in supports:
        mask[:, start:end] = True
    mask[0] = mask[-1] = True
    hollow = demo == 1 or (demo == 0 and rng.random() < 0.25)
    if not hollow:
        density = float(rng.uniform(0.12, 0.45))
        rows = [i for i in range(1, h - 1) if rng.random() < density]
        if demo == 2 and not rows:
            rows = [int(rng.integers(1, h - 1))]
        while len(rows) > h - 4:
            rows.pop(int(rng.integers(len(rows))))
        for index, row in enumerate(rows):
            if demo == 2 and index == 0:
                first = int(rng.integers(0, support_count - 1))
                last = first + 1  # necessarily partial with >=3 supports
            elif rng.random() < 0.5:
                first, last = 0, support_count - 1
            else:
                first = int(rng.integers(0, support_count - 1))
                last = int(rng.integers(first + 1, support_count))
            mask[row, supports[first][0]:supports[last][1]] = True
    remaining = 30 - h
    if demo == 1:
        bottom_margin = 0
    elif demo == 2:
        bottom_margin = int(rng.integers(1, min(3, remaining) + 1))
    else:
        bottom_margin = int(rng.integers(0, min(6, remaining) + 1))
    top_margin = int(rng.integers(0, min(6, remaining - bottom_margin) + 1))
    transverse = 30 - w - 2
    left_margin = 1 + int(rng.integers(0, transverse + 1))
    right_margin = 1 + int(rng.integers(0, transverse - left_margin + 2))
    shape = (top_margin + h + bottom_margin, left_margin + w + right_margin)
    inp = np.full(shape, bg, dtype=np.int8)
    out = np.full(shape, bg, dtype=np.int8)
    cursor = 0
    # Render the intended output directly from sampled bands and scaffold rows;
    # the public solver does not receive any of these construction parameters.
    for i, (color, thickness) in enumerate(zip(colors, widths)):
        shift = _phase(count - 1 - i)
        for relative_row in range(cursor, cursor + int(thickness)):
            cols = np.flatnonzero(mask[relative_row]) + left_margin
            inp[top_margin + relative_row, cols] = color
            out[top_margin + relative_row, cols + shift] = color
        cursor += int(thickness)
    if axis == 1:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


_SHORTCUTS = ("no_wave", "pixel_clock", "top_clock", "unique_colors", "canvas_clock",
              "whole_slice_runs", "reverse_sign", "left_only", "bar_reset")


def _shortcut_offsets(info, name):
    ranks = info["rank"].copy()
    n = len(ranks)
    if name == "no_wave":
        return [0] * n
    if name == "pixel_clock":
        ranks = np.arange(n - 1, -1, -1)
    elif name == "top_clock":
        ranks = len(info["colors"]) - 1 - ranks
    elif name == "unique_colors":
        seen = {}
        for i in range(n - 1, -1, -1):
            color = int(info["labels"][i])
            if color not in seen:
                seen[color] = len(seen)
            ranks[i] = seen[color]
    elif name == "canvas_clock":
        ranks += int(info["end_padding"] > 0)
    elif name == "whole_slice_runs":
        step = 0
        for i in range(n - 1, -1, -1):
            row = info["top"] + i
            if i < n - 1 and not np.array_equal(info["g"][row], info["g"][row + 1]):
                step += 1
            ranks[i] = step
    elif name == "bar_reset":
        step = 0
        for i in range(n - 1, -1, -1):
            if i < n - 1 and info["labels"][i] != info["labels"][i + 1]:
                step += 1
            if info["top"] + i in info["bars"]:
                step = 0
            ranks[i] = step
    offsets = [_phase(int(d)) for d in ranks]
    if name == "reverse_sign":
        offsets = [-x for x in offsets]
    elif name == "left_only":
        offsets = [-abs(x) for x in offsets]
    return offsets


def _training_evidence(infos):
    correct = [[_phase(int(d)) for d in x["rank"]] for x in infos]
    return all(any(_shortcut_offsets(x, name) != offsets for x, offsets in zip(infos, correct))
               for name in _SHORTCUTS)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.choice(3, p=(0.72, 0.14, 0.14)))
    axes = [0, 1] if mode == 0 else [mode - 1] * 2
    rng.shuffle(axes)
    shift = "long" if rng.integers(2) else "dwell"
    # Two complementary demonstrations suffice even at the smallest legal count.
    for _ in range(96):
        train = [_make_pair(rng, axes[0], demo=1, count=int(rng.choice((6, 8)))),
                 _make_pair(rng, axes[1], demo=2)]
        infos = [_analyze(p["input"]) for p in train]
        if _training_evidence(infos) and _key(train[0]["input"]) != _key(train[1]["input"]):
            break
    else:
        raise RuntimeError("could not establish contrasting training evidence")
    seen = {_key(p["input"]) for p in train}
    def add(destination, regime):
        for _ in range(128):
            pair = _make_pair(rng, int(rng.choice(axes)), regime=regime)
            key = _key(pair["input"])
            if key not in seen:
                destination.append(pair)
                seen.add(key)
                return
        raise RuntimeError("could not sample a distinct lattice")
    for _ in range(int(num_train) - 2):
        add(train, "train")
    rng.shuffle(train)
    test = []
    for _ in range(int(num_test)):
        add(test, shift)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check geometry, held phases, conservation, evidence and the actual shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    seen, infos, learned = set(), {"train": [], "test": []}, {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "canvas size changed")
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(a)
            infos[split].append(info)
            g, out, bg = info["g"], b if info["axis"] == 0 else b.T, info["bg"]
            _require(np.array_equal(np.bincount(a.ravel(), minlength=10),
                                   np.bincount(b.ravel(), minlength=10)), "color counts changed")
            _require(np.all(out[:info["top"]] == bg) and np.all(out[info["bottom"] + 1:] == bg),
                     "foreground appeared outside occupied slices")
            for i, row in enumerate(range(info["top"], info["bottom"] + 1)):
                before, after = np.flatnonzero(g[row] != bg), np.flatnonzero(out[row] != bg)
                _require(len(before) == len(after), "slice lost cells")
                delta = after - before
                _require(np.all(delta == delta[0]), "slice is not rigid")
                _require(np.array_equal(g[row, before], out[row, after]), "slice recolored")
                residue = int(info["rank"][i]) % 4
                expected = -1 if residue == 1 else (1 if residue == 3 else 0)
                _require(int(delta[0]) == expected, "wrong run-clock phase")
                if split == "train":
                    _require(residue not in learned or learned[residue] == int(delta[0]), "inconsistent clock")
                    learned[residue] = int(delta[0])
                else:
                    _require(learned.get(residue) == int(delta[0]), "test uses an undemonstrated phase")
            _require(not np.array_equal(a, b), "identity pair")
    train_info, test_info = infos["train"], infos["test"]
    _require(set(learned) == {0, 1, 2, 3}, "training must expose all four clock phases")
    _require({x["axis"] for x in test_info} <= {x["axis"] for x in train_info}, "unshown orientation")
    _require(any(len(set(x["colors"])) < len(x["colors"]) for x in train_info), "recurrence not demonstrated")
    _require(any(len(set(x["widths"])) > 1 for x in train_info), "unequal duration not demonstrated")
    _require(any(x["partial"] for x in train_info), "partial bars not demonstrated")
    _require(any(not x["bars"] for x in train_info), "bare frame not demonstrated")
    _require(any(x["end_padding"] == 0 for x in train_info) and any(x["end_padding"] > 0 for x in train_info),
             "anchor padding lacks contrast")
    _require(_training_evidence(train_info), "a named simpler alternative still fits all training")
    # Generation regime constraints are distinct from the solver's broader domain.
    _require(all(len(x["colors"]) <= 8 and max(x["widths"]) <= 3 for x in train_info), "training regime violated")
    _require(all(len(x["colors"]) >= 9 for x in test_info)
             or all(max(x["widths"]) >= 6 for x in test_info), "missing clock-length/dwell-time shift")


def self_test():
    """Bounded 42-seed semantic/diversity audit plus legal count boundaries."""
    puzzles, shapes, structures = set(), set(), set()
    axes, support_counts, run_counts, durations, palettes = set(), set(), set(), set(), set()
    long_tests = dwell_tests = recurrence_tests = 0
    zero_foreground = False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))
        puzzles.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _analyze(a)
                _require(np.array_equal(solve(a.copy()), pair["output"]), "solve/render disagreement")
                shapes.add(a.shape)
                roles = {}
                structures.add((a.shape, tuple(roles.setdefault(int(v), len(roles)) for v in a.flat)))
                axes.add(info["axis"])
                support_counts.add(len(info["supports"]))
                run_counts.add(len(info["colors"]))
                durations.update(info["widths"])
                palettes.add((info["bg"], frozenset(info["colors"])))
                zero_foreground |= info["bg"] != 0 and 0 in info["colors"]
                if split == "test":
                    long_tests += int(len(info["colors"]) >= 9)
                    dwell_tests += int(max(info["widths"]) >= 6)
                    recurrence_tests += int(len(set(info["colors"])) < len(info["colors"]))
    _require(len(puzzles) == 42 and len(shapes) >= 40 and len(structures) >= 150, "insufficient structural diversity")
    _require(axes == {0, 1} and support_counts == {2, 3, 4, 5, 6}, "orientation/support diversity missing")
    _require(set(range(4, 13)) <= run_counts and {1, 2, 3, 6} <= durations, "clock scale diversity missing")
    _require(long_tests > 0 and dwell_tests > 0 and recurrence_tests > 0, "test regimes missing")
    _require(len(palettes) >= 50 and zero_foreground, "local color roles insufficiently varied")
    # All legal explicit counts, including minimum demonstrations and four tests.
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(8675309, ntrain, ntest)
            _require((len(p["train"]), len(p["test"])) == (ntrain, ntest), "counts ignored")
    first, second = generate_puzzle(2026, 5, 3), generate_puzzle(2026, 5, 3)
    _require(all(np.array_equal(p[k], q[k]) for s in ("train", "test")
                 for p, q in zip(first[s], second[s]) for k in ("input", "output")), "nondeterministic seed")
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Direct metamorphic check: dilating a color run holds its phase; transposing
    # the entire problem transposes the answer. This does not call generation.
    pair = generate_puzzle(0)["train"][0]
    a = pair["input"]
    _require(np.array_equal(solve(a.T), solve(a).T), "transpose equivariance broken")
    info = _analyze(a)
    row = info["top"] + int(info["starts"][1])
    if info["g"].shape[0] < 30:
        thick = np.insert(info["g"], row, info["g"][row], axis=0)
        expected = np.insert(solve(info["g"]), row, solve(info["g"])[row], axis=0)
        _require(np.array_equal(solve(thick), expected), "run dilation changed other phases")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "input_shapes": len(shapes),
            "color_normalized_inputs": len(structures), "long_tests": long_tests,
            "dwell_tests": dwell_tests}


if __name__ == "__main__":
    print(self_test())
