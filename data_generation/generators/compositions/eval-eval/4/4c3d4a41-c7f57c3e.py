"""A comb press moves pin-mounted flags through a state-exchanging gate.

The reader uses pixels alone. The constructor uses direct destination formulas;
the solver instead translates the actual pin pixels and replaces crossed flags.
"""

from __future__ import annotations

from collections import Counter
import json
import numpy as np


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _signature(mapping):
    return tuple(sorted((y, x, int(c)) for (y, x), c in mapping.items()))


def _expand(grid, mapping, top, left, scale):
    for (y, x), color in mapping.items():
        yy, xx = top + y * scale, left + x * scale
        grid[yy:yy + scale, xx:xx + scale] = color


def _draw_frame(grid, f):
    t, b, l, r = (f[k] for k in ("top", "base", "left", "right"))
    grid[t, l:r + 1] = f["frame"]
    grid[t:b + 1, l] = grid[t:b + 1, r] = f["frame"]
    grid[f["gate"], l] = grid[f["gate"], r] = f["marker"]


def _draw_pin(grid, f, pin, anchor, template):
    s, left = pin["scale"], pin["bar_left"]
    q = left + f["stem"] * s
    grid[f["top"] + 1:anchor, q:q + s] = f["carrier"]
    shift = pin["anchor"] - anchor
    prefix = pin["prefix"][shift:]
    for index, color in enumerate(prefix):
        grid[f["top"] + 1 + index, q:q + s] = color
    grid[anchor:anchor + s, left:left + f["width"] * s] = f["carrier"]
    _expand(grid, template, anchor, left, s)


def _construct(f, output=False):
    """Direct construction from sampled features, not used by solve()."""
    a = np.full(f["shape"], f["background"], dtype=int)
    _draw_frame(a, f)
    base = f["base"] if output else f["key_row"]
    a[base, f["left"]:f["right"] + 1] = f["frame"]
    for p in f["pins"]:
        s, h = p["scale"], p["height"]
        q = p["bar_left"] + f["stem"] * s
        a[base - h:base, q:q + s] = f["frame"]
        final = f["base"] - h - s
        crossed = p["anchor"] + s - 1 >= f["gate"] > final + s - 1
        state = p["state"] ^ int(output and crossed)
        _draw_pin(a, f, p, final if output else p["anchor"], f["templates"][state])
    return a


def _runs(row, color):
    indices = np.flatnonzero(row == color)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    return [(int(g[0]), int(g[-1]) + 1) for g in np.split(indices, breaks)]


def _read_canonical(a, bg, frame, top, key_row, left, right):
    """Read an upright open-bottom chamber and the separate comb below it."""
    assert key_row - top >= 8 and right - left >= 8
    assert np.all(a[top, left:right + 1] == frame)
    assert np.all(a[key_row, left:right + 1] == frame)
    y = top + 1
    while y < key_row and a[y, left] != bg and a[y, right] != bg:
        y += 1
    base = y - 1
    assert top + 5 <= base < key_row - 1
    walls = a[top:base + 1, [left, right]]
    assert np.array_equal(walls[:, 0], walls[:, 1])
    special = np.flatnonzero(walls[:, 0] != frame)
    assert len(special) == 1 and 1 < special[0] < base - top
    gate = top + int(special[0])
    marker = int(a[gate, left])
    assert marker not in (bg, frame)
    assert np.all(a[base + 1:key_row, [left, right]] == bg)
    tooth_runs = _runs(a[key_row - 1, left + 1:right], frame)
    assert 2 <= len(tooth_runs) <= 6
    pins, templates = [], []
    carrier = None
    width = stem = None
    reserved = []
    for start, stop in tooth_runs:
        q, s = left + 1 + start, stop - start
        assert 1 <= s <= 3
        row = key_row - 1
        while row >= 0 and np.all(a[row, q:q + s] == frame):
            row -= 1
        h = key_row - 1 - row
        assert h >= 1 and row > base
        occupied = np.flatnonzero(a[top + 1:base, q] != bg)
        assert len(occupied) and np.array_equal(occupied, np.arange(len(occupied)))
        bottom = top + int(occupied[-1]) + 1
        c = int(a[bottom, q])
        assert c not in (bg, frame, marker)
        assert carrier is None or c == carrier
        carrier = c
        anchor = bottom - s + 1
        runs = [(l, r) for l, r in _runs(a[bottom], c) if l <= q < r]
        assert len(runs) == 1
        bar_left, bar_stop = runs[0]
        assert (bar_stop - bar_left) % s == 0 and (q - bar_left) % s == 0
        w, j = (bar_stop - bar_left) // s, (q - bar_left) // s
        assert 3 <= w <= 5 and 1 <= j < w - 1
        assert width is None or (w, j) == (width, stem)
        width, stem = w, j
        final = base - h - s
        d = anchor - final
        assert 0 <= d <= h, "pin must touch or overlap its destination tooth"
        assert final - 3 * s > top, "the complete flag must survive roof clipping"
        assert anchor + s <= base
        prefix = []
        for row in range(top + 1, anchor - 3 * s):
            block = a[row, q:q + s]
            assert np.all(block == block[0]) and int(block[0]) not in (bg, frame, marker)
            prefix.append(int(block[0]))
        # Every potentially occupied cell of either flag state is reserved.
        l, r = bar_left - s, bar_stop + s
        assert left < l and r <= right
        assert all(r <= l0 - 1 or r0 <= l - 1 for l0, r0 in reserved)
        reserved.append((l, r))
        template = {}
        for dy in range(-3, 0):
            for dx in range(-1, w + 1):
                block = a[anchor + dy * s:anchor + (dy + 1) * s,
                          bar_left + dx * s:bar_left + (dx + 1) * s]
                assert block.shape == (s, s) and np.all(block == block[0, 0])
                value = int(block[0, 0])
                if dx == j:
                    assert value == c
                else:
                    assert value not in (frame, marker, c)
                    if value != bg:
                        template[dy, dx] = value
        assert template
        templates.append(template)
        pins.append(dict(scale=s, height=h, anchor=anchor, bar_left=bar_left,
                         prefix=tuple(prefix)))
    signatures = sorted(set(map(_signature, templates)))
    assert len(signatures) == 2, "exactly two visible flag states are needed"
    minimum = min(p["scale"] for p in pins)
    assert {_signature(t) for p, t in zip(pins, templates) if p["scale"] == minimum} == set(signatures)
    for p, template in zip(pins, templates):
        p["state"] = signatures.index(_signature(template))
    f = dict(shape=a.shape, background=bg, frame=frame, carrier=carrier,
             marker=marker, top=top, base=base, key_row=key_row,
             left=left, right=right, gate=gate, width=width, stem=stem,
             templates=[{(y, x): c for y, x, c in sig} for sig in signatures], pins=pins)
    # This check also excludes extra cells, broken stems and nonrectangular teeth.
    assert np.array_equal(a, _construct(f))
    return f


def _read_input(grid):
    a = _grid(grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert counts.max() * 2 > a.size, "background must be a strict majority"
    candidates = []
    for rotation in range(4):
        upright = np.rot90(a, rotation)
        long_runs = {}
        for y, row in enumerate(upright):
            breaks = [0, *(np.flatnonzero(row[1:] != row[:-1]) + 1).tolist(), len(row)]
            for l, r in zip(breaks, breaks[1:]):
                color = int(row[l])
                if color != bg and r - l >= 9:
                    long_runs.setdefault((color, l, r - 1), []).append(y)
        for (frame, left, right), rows in long_runs.items():
            for top in rows:
                for key_row in rows:
                    if key_row <= top:
                        continue
                    try:
                        f = _read_canonical(upright, bg, frame, top, key_row, left, right)
                    except (AssertionError, IndexError):
                        continue
                    f["rotation"] = rotation
                    candidates.append(f)
    assert len(candidates) == 1, "scene must have one roof/key interpretation"
    return candidates[0]


def _simulate(grid, f, exchange=True, movement=True, crossing_mode="foot"):
    """Move literal input pixels; optionally ablate either inherited mechanism."""
    a = np.rot90(_grid(grid), f["rotation"])
    out = a.copy()
    bg, roof, base = f["background"], f["top"], f["base"]
    out[base + 1:, :] = bg
    # Gather before clearing. Pin regions are pairwise separated.
    for p in f["pins"]:
        s, l = p["scale"], p["bar_left"]
        q = l + f["stem"] * s
        final = base - p["height"] - s
        shift = p["anchor"] - final if movement else 0
        crossed = p["anchor"] + s - 1 >= f["gate"] > p["anchor"] + s - 1 - shift
        if crossing_mode == "moving":
            crossed = shift > 0
        elif crossing_mode == "top":
            crossed = p["anchor"] >= f["gate"] > p["anchor"] - shift
        elif crossing_mode == "arrival_inclusive":
            crossed = p["anchor"] + s - 1 > f["gate"] >= p["anchor"] + s - 1 - shift
        region = a[roof + 1:base, l - s:l + (f["width"] + 1) * s]
        out[roof + 1:base, l - s:l + (f["width"] + 1) * s] = bg
        for yy, xx in np.argwhere(region != bg):
            dest_y = roof + 1 + int(yy) - shift
            if dest_y > roof:
                out[dest_y, l - s + int(xx)] = region[yy, xx]
        # Remove the old state at its translated anchor before writing its
        # replacement. The ordered prefix and scaffold are outside this mask.
        for dy in range(-3, 0):
            for dx in range(-1, f["width"] + 1):
                if dx != f["stem"]:
                    yy, xx = p["anchor"] - shift + dy * s, l + dx * s
                    out[yy:yy + s, xx:xx + s] = bg
        template = f["templates"][p["state"] ^ int(exchange and crossed)]
        _expand(out, template, p["anchor"] - shift, l, s)
        out[base - p["height"]:base, q:q + s] = f["frame"]
    out[base, f["left"]:f["right"] + 1] = f["frame"]
    return np.rot90(out, -f["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the current input."""
    a = _grid(input_grid)
    return _simulate(a, _read_input(a))


def _design(rng, width, stem, change):
    available = [(y, x) for y in range(-3, 0)
                 for x in range(-1, width + 1) if x != stem]
    # At least one exterior cell makes scaffold registration visibly relevant.
    exterior = [(y, x) for y, x in available if x in (-1, width)]
    common = [(y, x) for y, x in available if 0 <= x < width]
    rng.shuffle(common)
    stable = common[:int(rng.integers(0, 3))]
    rest = [p for p in available if p not in stable]
    rng.shuffle(rest)
    count = int(rng.integers(2, min(7, len(rest)) + 1))
    support0 = set(rest[:count])
    support0.add(exterior[int(rng.integers(len(exterior)))])
    if change == "recolor":
        support1 = set(support0)
    else:
        support1 = set(rest[count:count + int(rng.integers(1, 6))])
        support1.add(exterior[int(rng.integers(len(exterior)))])
        if support1 == support0:
            support1.symmetric_difference_update({next(p for p in available if p not in stable)})
        assert support1
    # Logical labels: background/frame/marker/carrier occupy 0..3.
    maps = [{p: 4 for p in stable}, {p: 4 for p in stable}]
    for state, support in enumerate((support0, support1)):
        for i, point in enumerate(sorted(support)):
            if change == "relocate":
                maps[state][point] = 5
            else:
                maps[state][point] = 5 + state
                if change == "combined" and i % 3 == 2:
                    maps[state][point] = 7 + state
    # In combined mode, guarantee both changed support and common-cell recolor.
    if change == "combined":
        point = next(p for p in common if p not in stable)
        maps[0][point], maps[1][point] = 5, 6
        if set(maps[0]) == set(maps[1]):
            point = next(p for p in available if p not in maps[0])
            maps[1][point] = 8
    return maps


def _sample_pair(rng, change, regime, test, witness=None):
    # Geometry is sampled before colors. Fitting limits have physical meanings.
    if test and regime == "scale":
        scales = [1, 1, 3]
        width = 3
    elif witness == 0:
        scales = [1, 1, 2]
        width = int(rng.integers(3, 5))
    elif witness == 1:
        scales = [1, 1, 2] if rng.random() < .65 else [1, 1, 1, 1]
        width = int(rng.integers(3, 5))
    else:
        width = int(rng.integers(3, 6))
        feasible = []
        for n in range(2, 5):
            for extra in (0, 1):
                ss = [1] * n
                if extra and n >= 3:
                    ss[-1] = 2
                if (width + 2) * sum(ss) + n + 1 <= 30:
                    feasible.append(ss)
        scales = list(feasible[int(rng.integers(len(feasible)))])
    n, largest = len(scales), max(scales)
    stem = int(rng.integers(1, width - 1))
    design = _design(rng, width, stem, change)
    palette = [int(c) for c in rng.permutation(10)]
    templates = [{p: palette[c] for p, c in m.items()} for m in design]
    span = int(rng.integers(4 * largest + 5, min(20, 4 * largest + 10)))
    cap = min(7, span - 4 * largest - 1, 26 - span)
    assert cap >= 2
    top = int(rng.integers(0, 2))
    base = top + span
    target = 0 if witness == 0 else (2 if witness == 1 else int(rng.integers(n)))
    other = 2 if witness == 0 else (target + 1) % n
    stationary = None if witness == 0 else ((target + 2) % n if n >= 3 else None)
    for _ in range(1000):
        heights = rng.integers(1, cap + 1, n).tolist()
        if test and regime == "flat":
            heights = [int(rng.integers(2, cap + 1))] * n
        elif regime == "flat" and not test and len(set(heights)) == 1:
            continue
        shifts = [int(rng.integers(h + 1)) for h in heights]
        shifts[target] = int(rng.integers(1, heights[target] + 1))
        shifts[other] = int(rng.integers(1, heights[other] + 1))
        if witness == 0:
            shifts[1] = int(rng.integers(1, heights[1] + 1))
        if stationary is not None:
            shifts[stationary] = 0
        finals = [base - h - 1 for h in heights]
        bottoms = [r + d for r, d in zip(finals, shifts)]
        if witness == 0:
            gate = bottoms[target]  # departure exactly on the line counts
        elif witness == 1:
            gate = finals[other]  # arriving exactly on it does not count
        else:
            gate = int(rng.integers(finals[target] + 1, bottoms[target] + 1))
        crosses = [old >= gate > new for old, new in zip(bottoms, finals)]
        if (top + 2 <= gate < base and crosses[target] and not crosses[other]
                and (witness != 0 or crosses[1])):
            break
    else:
        raise AssertionError("bounded contact/gate sampling exhausted")
    key_row = base + max(heights) + int(rng.integers(2, 4))
    assert key_row < 30
    budget = 30 - ((width + 2) * sum(scales) + n + 1)
    extras = [0] * (n + 2)
    for index in rng.permutation(n + 2):
        extras[index] = int(rng.integers(0, min(2, budget) + 1))
        budget -= extras[index]
    left = extras[0]
    cursor = left + 1 + extras[1]
    pins = []
    states = rng.integers(0, 2, n).tolist()
    # Two unit-scale references always expose the complete local vocabulary.
    if target < 2:
        states[target] = int(witness == 1) if witness is not None else int(rng.integers(2))
        states[1 - target] = 1 - states[target]
    else:
        states[0], states[1] = 0, 1
        if witness is not None:
            states[target] = int(witness == 1)
    for i, (s, h, r, state) in enumerate(zip(scales, heights, bottoms, states)):
        prefix_length = r - s + 1 - 3 * s - top - 1
        prefix = [palette[3]] * prefix_length
        accent_colors = sorted({c for t in templates for c in t.values()})
        for row in range(prefix_length):
            if rng.random() < .22:
                prefix[row] = int(rng.choice(accent_colors))
        if witness == 0 and i == 0:
            # A colored prefix cell disappears at the roof, while a different
            # colored cell becomes the first surviving row, preserving order.
            prefix[0] = accent_colors[0]
            prefix[shifts[i]] = accent_colors[-1]
        pins.append(dict(scale=s, height=h, anchor=r - s + 1,
                         bar_left=cursor + s, state=state, prefix=tuple(prefix)))
        cursor += (width + 2) * s
        if i < n - 1:
            cursor += 1 + extras[i + 2]
    right = cursor
    shape = (key_row + 1 + int(rng.integers(0, min(2, 29 - key_row) + 1)),
             right + 1 + extras[-1])
    assert max(shape) <= 30
    f = dict(shape=shape, background=palette[0], frame=palette[1],
             marker=palette[2], carrier=palette[3], top=top, base=base,
             key_row=key_row, left=left, right=right, gate=gate,
             width=width, stem=stem, templates=templates, pins=pins)
    inp, out = _construct(f), _construct(f, True)
    assert np.count_nonzero(inp == palette[0]) * 2 > inp.size
    reflection, rotation = bool(rng.integers(2)), int(rng.integers(4))
    if reflection:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {"input": np.rot90(inp, rotation).copy(),
            "output": np.rot90(out, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ns = _count(num_test, 1, 4, "num_test")
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("composition", "scale", "flat")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            if regime == "composition":
                change = ("recolor", "relocate")[index % 2] if split == "train" else "combined"
            else:
                change = "combined" if index == 0 else str(rng.choice(("recolor", "relocate", "combined")))
            for _ in range(64):
                pair = _sample_pair(rng, change, regime, split == "test",
                                    index if split == "train" and index < 2 else None)
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise AssertionError("distinct input sampling exhausted")
    return puzzle


def _features(f):
    shifts, crosses, states = [], [], []
    for p in f["pins"]:
        final = f["base"] - p["height"] - p["scale"]
        shifts.append(p["anchor"] - final)
        crosses.append(p["anchor"] + p["scale"] - 1 >= f["gate"] > final + p["scale"] - 1)
        states.append(p["state"])
    a, b = f["templates"]
    return dict(shifts=shifts, crosses=crosses, states=states,
                recolor=any(a[p] != b[p] for p in set(a) & set(b)),
                relocate=set(a) != set(b))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, training, test_features = set(), [], []
    both_concepts = {"A": False, "B": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            f = _read_input(inp)
            assert np.array_equal(out, _simulate(inp, f)), "incorrect motion or flag exchange"
            assert not np.array_equal(inp, out)
            features = _features(f)
            assert any(features["crosses"]) and not all(features["crosses"])
            assert any(d > 0 for d in features["shifts"])
            # The complete foot always meets its tooth, including unchanged pins.
            canonical = np.rot90(out, f["rotation"])
            for p, d in zip(f["pins"], features["shifts"]):
                bottom = f["base"] - p["height"] - 1
                l, s = p["bar_left"], p["scale"]
                assert np.all(canonical[bottom, l:l + f["width"] * s] == f["carrier"])
                q = l + f["stem"] * s
                assert np.all(canonical[bottom + 1:f["base"], q:q + s] == f["frame"])
                kept = np.array(p["prefix"][d:], dtype=int)
                actual = canonical[f["top"] + 1:bottom - s + 1 - 3 * s, q:q + s]
                assert actual.shape == (len(kept), s)
                assert np.array_equal(actual, np.repeat(kept[:, None], s, axis=1)), "prefix order/clipping changed"
            if split == "train":
                training.append((f, features))
                both_concepts["A"] |= not np.array_equal(out, _simulate(inp, f, movement=False))
                both_concepts["B"] |= not np.array_equal(out, _simulate(inp, f, exchange=False))
            else:
                test_features.append(features)
    assert all(both_concepts.values()), "demonstrations must require both parents"
    assert any(0 in x["shifts"] for _, x in training)
    assert any(any(d > 0 and not c for d, c in zip(x["shifts"], x["crosses"])) for _, x in training)
    assert any(max(p["scale"] for p in f["pins"]) >= 2 for f, _ in training)
    assert any({p["state"] for p, c in zip(f["pins"], z["crosses"]) if c} == {0, 1}
               for f, z in training), "show both directions of exchange in one local vocabulary"
    assert any(any(any(c != f["carrier"] for c in p["prefix"][:d])
                       and any(c != f["carrier"] for c in p["prefix"][d:])
                       for p, d in zip(f["pins"], z["shifts"]))
               for f, z in training), "show colored prefix clipping and survival"
    assert any(x["recolor"] for _, x in training) or not any(x["recolor"] for x in test_features)
    assert any(x["relocate"] for _, x in training) or not any(x["relocate"] for x in test_features)
    # Which state crosses is not a fixed literal color or a one-way mapping.
    # Templates differ between examples; states are ordered locally by pixels.
    assert len(training) >= 2


def self_test():
    shapes, structures, designs, scales, counts, widths, profiles = (set() for _ in range(7))
    backgrounds, frames, markers, carriers, rotations = (set() for _ in range(5))
    flags = Counter()
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p[r]) for s in puzzle for p in puzzle[s] for r in ("input", "output")))
        local_designs = set()
        for split in puzzle:
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _read_input(inp)
                z = _features(f)
                assert np.array_equal(solve(inp.copy()), out)
                assert np.array_equal(np.rot90(solve(np.rot90(inp)), -1), out)
                shapes.add(inp.shape)
                roles = {}
                structures.add((inp.shape, tuple(roles.setdefault(int(v), len(roles)) for v in inp.flat)))
                designs.add((f["width"], f["stem"], tuple(_signature(t) for t in f["templates"])))
                local_designs.add(tuple(_signature(t) for t in f["templates"]))
                scales.update(p["scale"] for p in f["pins"])
                counts.add(len(f["pins"]))
                widths.add(f["width"])
                profiles.add(tuple(p["height"] for p in f["pins"]))
                backgrounds.add(f["background"])
                frames.add(f["frame"])
                markers.add(f["marker"])
                carriers.add(f["carrier"])
                rotations.add(f["rotation"])
                flags["stationary"] += sum(d == 0 for d in z["shifts"])
                flags["moving"] += sum(d > 0 for d in z["shifts"])
                flags["crossed"] += sum(z["crosses"])
                flags["moving_without_crossing"] += sum(d > 0 and not c for d, c in zip(z["shifts"], z["crosses"]))
                flags["recolor"] += z["recolor"]
                flags["relocate"] += z["relocate"]
                flags["flat"] += len({p["height"] for p in f["pins"]}) == 1
                for p, d in zip(f["pins"], z["shifts"]):
                    flags["colored_prefix_clipped"] += any(c != f["carrier"] for c in p["prefix"][:d])
                    flags["colored_prefix_survives"] += any(c != f["carrier"] for c in p["prefix"][d:])
                flags["A_ablation"] += not np.array_equal(out, _simulate(inp, f, movement=False))
                flags["B_ablation"] += not np.array_equal(out, _simulate(inp, f, exchange=False))
                for shortcut in ("moving", "top", "arrival_inclusive"):
                    flags["shortcut_" + shortcut] += not np.array_equal(out, _simulate(inp, f, crossing_mode=shortcut))
                for p, cross in zip(f["pins"], z["crosses"]):
                    final_bottom = f["base"] - p["height"] - 1
                    initial_bottom = p["anchor"] + p["scale"] - 1
                    flags["departure_equality"] += cross and initial_bottom == f["gate"]
                    flags["arrival_equality"] += not cross and final_bottom == f["gate"] and initial_bottom > f["gate"]
        assert len(local_designs) >= 3
    assert len(fingerprints) == 42 and len(structures) >= 190
    assert len(shapes) >= 50 and len(designs) >= 150 and len(profiles) >= 60
    assert scales == {1, 2, 3} and counts == {2, 3, 4} and widths == {3, 4, 5}
    assert all(len(s) == 10 for s in (backgrounds, frames, markers, carriers))
    assert rotations == {0, 1, 2, 3}
    for name in ("stationary", "moving", "crossed", "moving_without_crossing", "recolor", "relocate", "flat",
                 "A_ablation", "B_ablation", "shortcut_moving", "shortcut_top", "shortcut_arrival_inclusive",
                 "colored_prefix_clipped", "colored_prefix_survives", "departure_equality", "arrival_equality"):
        assert flags[name] > 0, name
    # Exercise all count combinations, and deterministic full-puzzle reproduction.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(900 + 10 * nt + ns, nt, ns)
            validate_generated_puzzle(p)
    a, b = generate_puzzle(71, 10, 4), generate_puzzle(71, 10, 4)
    assert all(np.array_equal(x[r], y[r]) for s in a for x, y in zip(a[s], b[s]) for r in x)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A changed output and a broken wall must be rejected.
    bad = {s: [{r: g.copy() for r, g in p.items()} for p in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    sample = a["train"][0]["input"]
    f = _read_input(sample)
    broken = np.rot90(sample, f["rotation"]).copy()
    broken[f["top"], f["left"] + 1] = f["background"]
    try:
        solve(np.rot90(broken, -f["rotation"]))
    except AssertionError:
        pass
    else:
        raise AssertionError("broken chamber accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "color_normalized_structures": len(structures), "designs": len(designs),
            "profiles": len(profiles), "counts": sorted(counts), "scales": sorted(scales),
            "widths": sorted(widths), "features": dict(flags)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
