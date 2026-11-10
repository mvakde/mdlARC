"""Ray-response jigsaw: emerging beam signatures assemble loose optical panels.

Only NumPy and the standard library are used. Generation and all tests are
in-memory; importing this module creates no files and performs no experiments.
See task_spec.md for the complete input grammar and transformation.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and a.dtype != np.dtype(bool), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _runs(row, barrier):
    result = []
    c = 0
    while c < len(row):
        if row[c] != barrier:
            c += 1
            continue
        lo = c
        while c + 1 < len(row) and row[c + 1] == barrier:
            c += 1
        result.append((lo, c))
        c += 1
    return result


def _advance(row, active, barrier):
    runs = _runs(row, barrier)
    hits = [(lo, hi) for lo, hi in runs if any(lo <= c <= hi for c in active)]
    following = {c for c in active if row[c] != barrier}
    for lo, hi in hits:
        following.update(c for c in (lo - 1, hi + 1) if 0 <= c < len(row))
    return hits, following


def _component_spans(panel, barrier):
    """Used only by a deliberately incorrect connected-object alternative."""
    remaining = set(map(tuple, np.argwhere(panel == barrier)))
    spans = {}
    while remaining:
        start = remaining.pop()
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        lo, hi = min(c for _, c in cells), max(c for _, c in cells)
        spans.update({p: (lo, hi) for p in cells})
    return spans


def _illuminate(panel, background, guide, barrier, signature, shortcut=None):
    """Immutable obstacles, local transverse runs, set union at mergers.

    Markers occupy the first two layers. The final layer is initially blank.
    No saved execution traces are produced. Counterfactual modes are tests only.
    """
    h, w = panel.shape
    out = panel.copy()
    active = set(signature)
    stats = Counter()
    spans = _component_spans(panel, barrier) if shortcut == "component" else None
    for r in range(2, h):
        runs = _runs(panel[r], barrier)
        hits = [(lo, hi) for lo, hi in runs if any(lo <= c <= hi for c in active)]
        stats["unreached"] += len(runs) - len(hits)
        stats["simultaneous"] += len(hits) > 1
        stats["collisions"] += len(hits)
        stats["singleton"] += sum(lo == hi for lo, hi in hits)
        stats["multi_hit"] += sum(sum(lo <= c <= hi for c in active) > 1 for lo, hi in hits)
        stats["split"] += sum(lo > 0 and hi < w - 1 for lo, hi in hits)
        stats["clipped"] += sum(lo == 0 or hi == w - 1 for lo, hi in hits)
        if shortcut == "ignore_barriers":
            following, selected = active.copy(), []
        else:
            following = {c for c in active if panel[r, c] != barrier}
            selected = hits
            if shortcut == "all_bars":
                selected = runs
            elif shortcut == "one_per_layer":
                selected = hits[:1]
            elif shortcut == "ignore_singletons":
                selected = [(lo, hi) for lo, hi in hits if lo != hi]
            elif shortcut == "component":
                selected = sorted({spans[(r, lo)] for lo, hi in hits})
        for c in following:
            if panel[r, c] == background:
                out[r, c] = guide
        for lo, hi in selected:
            for c in range(max(0, lo - 1), min(w, hi + 2)):
                if panel[r - 1, c] == background:
                    out[r - 1, c] = guide
            for c in (lo - 1, hi + 1):
                if 0 <= c < w:
                    stats["merged"] += c in following
                    following.add(c)
                    if panel[r, c] == background:
                        out[r, c] = guide
        active = following
    return out, tuple(sorted(active)), stats


def _pack(panels, order, layout, divider):
    nr, nc = layout
    h, w = panels[0].shape
    result = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, i in enumerate(order):
        r, c = divmod(slot, nc)
        result[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = panels[i]
    return result


def _partition(a):
    candidates = []
    h, w = a.shape
    for color in np.unique(a):
        rows = list(map(int, np.flatnonzero(np.all(a == color, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(a == color, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        heights = [b - x - 1 for x, b in zip(rr, rr[1:])]
        widths = [b - x - 1 for x, b in zip(cc, cc[1:])]
        if min(heights + widths) < 4 or len(set(heights)) != 1 or len(set(widths)) != 1:
            continue
        panels = [a[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 3 <= len(panels) <= 6 or any(np.any(p == color) for p in panels):
            continue
        candidates.append((panels, int(color), (len(heights), len(widths))))
    _require(len(candidates) == 1, "one unambiguous equal-panel divider partition required")
    return candidates[0]


def _orders(inlets, outlets, source):
    """Find up to two complete chains; repeated signatures need global search."""
    answers = []

    def visit(order, remaining):
        if len(answers) == 2:
            return
        outgoing = outlets[order[-1]]
        if not remaining:
            if not outgoing:
                answers.append(tuple(order))
            return
        if not outgoing:
            return
        for i in remaining:
            if inlets[i] == outgoing:
                visit(order + [i], tuple(j for j in remaining if j != i))

    visit([source], tuple(i for i in range(len(inlets)) if i != source))
    return answers


def _decode(a):
    panels, divider, layout = _partition(a)
    candidates = []
    for rotation in range(4):
        pp = [np.rot90(p, -rotation).copy() for p in panels]
        h, w = pp[0].shape
        if h < 4 or w < 7:
            continue
        bg = int(pp[0][-1, 0])
        if any(not np.all(p[-1] == bg) for p in pp):
            continue
        prefix_colors, inlets = [], []
        valid = True
        for p in pp:
            sig = tuple(map(int, np.flatnonzero(p[0] != bg)))
            colors = np.unique(p[:2][p[:2] != bg])
            if not sig or len(sig) == w or len(colors) != 1 or not np.array_equal(p[0], p[1]):
                valid = False
                break
            prefix_colors.append(int(colors[0]))
            inlets.append(sig)
        if not valid:
            continue
        freq = Counter(prefix_colors)
        if len(freq) != 2 or sorted(freq.values()) != [1, len(pp) - 1]:
            continue
        guide = next(c for c in freq if freq[c] == 1)
        marker = next(c for c in freq if freq[c] != 1)
        remaining_colors = set(map(int, np.unique(np.concatenate(pp)))) - {bg, guide, marker}
        if len(remaining_colors) != 1:
            continue
        barrier = remaining_colors.pop()
        if any(np.any((p[2:] != bg) & (p[2:] != barrier)) or not np.any(p == barrier) for p in pp):
            continue
        source = prefix_colors.index(guide)
        filled, outlets, metrics = [], [], []
        for p, sig in zip(pp, inlets):
            out, state, stats = _illuminate(p, bg, guide, barrier, sig)
            filled.append(out)
            outlets.append(state)
            metrics.append(stats)
        orders = _orders(inlets, outlets, source)
        if len(orders) != 1:
            continue
        if len([s for s in outlets if not s]) != 1:
            continue
        if len(pp) * (h + 1) - 1 > 30:
            continue
        candidates.append(dict(panels=pp, filled=filled, inlets=inlets, outlets=outlets,
                               source=source, order=orders[0], stats=metrics,
                               bg=bg, guide=guide, barrier=barrier, marker=marker,
                               divider=divider, rotation=rotation, layout=layout))
    _require(len(candidates) == 1, "roles, direction and complete response chain must be unique")
    return candidates[0]


def _answer(decoded, panels=None, order=None):
    panels = decoded["filled"] if panels is None else panels
    order = decoded["order"] if order is None else order
    canonical = _pack(panels, order, (len(panels), 1), decoded["divider"])
    return np.ascontiguousarray(np.rot90(canonical, decoded["rotation"]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the input alone: no RNG, examples, file reads, caches or latent plans."""
    return _answer(_decode(_grid(input_grid)))


def _blank_panel(h, w, inlet, source):
    p = np.zeros((h, w), dtype=np.int8)
    p[:2, list(inlet)] = 1 if source else 3
    return p


def _add_inert(rng, row, active):
    free = [c for c in range(len(row)) if c not in active and row[c] == 0
            and (c == 0 or row[c - 1] == 0) and (c == len(row) - 1 or row[c + 1] == 0)]
    if not free:
        return
    pivot = int(rng.choice(free))
    lo = hi = pivot
    while lo > 0 and lo - 1 in free and rng.random() < .55:
        lo -= 1
    while hi + 1 < len(row) and hi + 1 in free and rng.random() < .55:
        hi += 1
    row[lo:hi + 1] = 2


def _random_panel(rng, h, w, inlet, source=False, stable=False):
    p = _blank_panel(h, w, inlet, source)
    active = set(inlet)
    for r in range(2, h - 1):
        if not stable and rng.random() < .78:
            pivots = list(map(int, rng.permutation(sorted(active))))
            for pivot in pivots[:int(rng.integers(1, min(3, len(pivots)) + 1))]:
                for _ in range(12):
                    length = int(rng.integers(1, w))
                    lo = int(rng.integers(max(0, pivot - length + 1), min(pivot, w - length) + 1))
                    hi = lo + length - 1
                    if not np.any(p[r, max(0, lo - 1):min(w, hi + 2)]):
                        p[r, lo:hi + 1] = 2
                        break
        if rng.random() < .5:
            _add_inert(rng, p[r], active)
        _, active = _advance(p[r], active, 2)
    if not np.any(p == 2):
        _add_inert(rng, p[2], set(inlet))
    return p


def _source_witness(rng, h, w, kind):
    if kind == "branching":
        a = int(rng.integers(4, w - 6))
        b = int(rng.integers(a + 4, w - 2))
        p = _blank_panel(h, w, (a, b), True)
        r = int(rng.integers(2, h - 2))
        p[r, a] = 2
        lo = b - int(rng.integers(2))
        p[r, lo:lo + 2] = 2
        p[r, 0] = 2  # Same-sized inert counterpart to the struck singleton.
        p[r + 1, a - 2:a + 1] = 2  # Connected elbow: use the local run, not the whole object.
        return p, (a, b)
    if kind == "clipped":
        edge = int(rng.integers(2))
        hi = int(rng.integers(1, w - 2))
        pivot = int(rng.integers(0, hi + 1))
        p = _blank_panel(h, w, (pivot,), True)
        r = int(rng.integers(2, h - 1))
        p[r, :hi + 1] = 2
        if edge:
            p = np.fliplr(p).copy()
            pivot = w - 1 - pivot
        return p, (pivot,)
    count = int(rng.integers(1, min(4, w - 1)))
    inlet = tuple(sorted(map(int, rng.choice(w, count, replace=False))))
    return _random_panel(rng, h, w, inlet, source=True), inlet


def _next_panel(rng, h, w, inlet, seen_states, stable, required_size=None):
    for _ in range(96):
        p = _random_panel(rng, h, w, inlet, stable=stable)
        filled, state, stats = _illuminate(p, 0, 1, 2, inlet)
        if stable:
            acceptable = state == inlet
        else:
            acceptable = (state and state not in seen_states and len(state) <= min(6, w - 2)
                          and (required_size is None or len(state) == required_size))
        if acceptable and np.any(p == 2):
            return p, filled, state
    # Bounded constructive fallback: enumerate single reached intervals.
    if not stable:
        options = [(lo, hi) for lo in range(w) for hi in range(lo, w)
                   if hi - lo + 1 < w and any(lo <= c <= hi for c in inlet)]
        for i in rng.permutation(len(options)):
            lo, hi = options[int(i)]
            p = _blank_panel(h, w, inlet, False)
            p[int(rng.integers(2, h - 1)), lo:hi + 1] = 2
            filled, state, stats = _illuminate(p, 0, 1, 2, inlet)
            if (state and state not in seen_states and len(state) <= min(6, w - 2)
                    and (required_size is None or len(state) == required_size)):
                return p, filled, state
    return None


def _sink(rng, h, w, inlet):
    p = _random_panel(rng, h, w, inlet)
    p[int(rng.integers(2, h - 1)), :] = 2
    return p


_SHORTCUTS = ("ignore_barriers", "all_bars", "one_per_layer", "ignore_singletons", "component")


def _shortcut_differs(d, shortcut):
    return any(not np.array_equal(_illuminate(p, d["bg"], d["guide"], d["barrier"], sig, shortcut)[0], out)
               for p, sig, out in zip(d["panels"], d["inlets"], d["filled"]))


def _make_pair(rng, kind, repeated, direction):
    for _ in range(64):
        if kind == "branching":
            n, h, w = 3, int(rng.integers(6, 10)), int(rng.integers(12, 19))
        elif kind == "clipped":
            n, h, w = 4, int(rng.integers(5, 7)), int(rng.integers(7, 15))
        else:
            n = int(rng.choice((5, 6) if kind == "test" else (3, 4)))
            h, w = int(rng.integers(4, 31 // n)), int(rng.integers(7, 19))
        source, inlet = _source_witness(rng, h, w, kind)
        filled, state, _stats = _illuminate(source, 0, 1, 2, inlet)
        if not state or state == inlet or len(state) > min(6, w - 2):
            continue
        panels, outputs = [source], [filled]
        states = {inlet, state}
        loop_index = int(rng.integers(1, n - 1)) if repeated else -1
        failed = False
        for index in range(1, n - 1):
            item = _next_panel(rng, h, w, state, states, index == loop_index,
                               required_size=1 if kind == "clipped" else None)
            if item is None:
                failed = True
                break
            p, out, state = item
            panels.append(p)
            outputs.append(out)
            states.add(state)
        if failed:
            continue
        p = _sink(rng, h, w, state)
        out, final_state, _stats = _illuminate(p, 0, 1, 2, state)
        if final_state:
            continue
        panels.append(p)
        outputs.append(out)
        layouts = [(r, n // r) for r in range(1, n + 1) if n % r == 0
                   and r * (h + 1) - 1 <= 30 and (n // r) * (w + 1) - 1 <= 30]
        layout = layouts[int(rng.integers(len(layouts)))]
        order = list(map(int, rng.permutation(n)))
        if order == list(range(n)) and layout == (n, 1):
            order.reverse()
        # Both minimal teaching witnesses change actual order, not only layout.
        if kind in ("branching", "clipped") and order == list(range(n)):
            order[0], order[-1] = order[-1], order[0]
        palette = np.asarray(rng.choice(10, 5, replace=False), dtype=np.int8)
        a = np.ascontiguousarray(np.rot90(palette[_pack(panels, order, layout, 4)], direction))
        b = np.ascontiguousarray(np.rot90(palette[_pack(outputs, list(range(n)), (n, 1), 4)], direction))
        try:
            d = _decode(a)
        except ValueError:
            continue
        if not np.array_equal(_answer(d), b):
            continue
        if kind == "branching" and not all(_shortcut_differs(d, s) for s in _SHORTCUTS):
            continue
        return {"input": a, "output": b}
    raise RuntimeError("bounded ray-response construction exhausted")


def _counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 training, 1..4 test, and num_train > num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # These settings change distributions only; the visible rule never changes.
    repeat_mode = bool(rng.random() < .65)
    fixed_direction = int(rng.integers(4))
    mixed_direction = bool(rng.random() < .5)
    puzzle, seen = {"train": [], "test": []}, set()
    plans = [("train", "branching", False), ("train", "clipped", repeat_mode)]
    plans += [("train", "general", repeat_mode and bool(rng.random() < .5)) for _ in range(num_train - 2)]
    plans += [("test", "test", repeat_mode and bool(rng.random() < .7)) for _ in range(num_test)]
    for split, kind, repeated in plans:
        for _ in range(32):
            direction = int(rng.integers(4)) if mixed_direction else fixed_direction
            pair = _make_pair(rng, kind, repeated, direction)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("bounded unique-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Reconstruct semantics and teaching evidence from returned grids only."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, decoded = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate inputs")
            seen.add(_key(a))
            d = _decode(a)
            _require(np.array_equal(_answer(d), b), "wrong illuminated assembly")
            # Check conservation and allowed edits separately from whole output equality.
            canonical_out = np.rot90(b, -d["rotation"])
            h, w = d["panels"][0].shape
            for slot, i in enumerate(d["order"]):
                actual = canonical_out[slot * (h + 1):slot * (h + 1) + h]
                original = d["panels"][i]
                _require(actual.shape == original.shape, "panel shape changed")
                _require(bool(np.all(actual[original != d["bg"]] == original[original != d["bg"]])), "original panel pixel changed")
                _require(bool(np.all(actual[original != actual] == d["guide"])), "only guide may be added")
                if slot + 1 < len(d["order"]):
                    _require(bool(np.all(canonical_out[slot * (h + 1) + h] == d["divider"])), "divider changed")
                    _require(d["outlets"][i] == d["inlets"][d["order"][slot + 1]], "full signature mismatch")
            _require(not np.array_equal(a, b), "unchanged pair")
            decoded[split].append(d)
    train, test = decoded["train"], decoded["test"]
    _require({len(d["panels"]) for d in train} == {3, 4}, "training must expose short chain lengths")
    _require(all(len(d["panels"]) in (5, 6) for d in test), "test must hold out longer response chains")
    for shortcut in _SHORTCUTS:
        _require(any(_shortcut_differs(d, shortcut) for d in train), "training fails to refute " + shortcut)
    _require(any(not _orders(d["inlets"], d["inlets"], d["source"]) for d in train), "A omission does not change chain feasibility")
    # A B-omission baseline illuminates every panel in its original slot.
    _require(any(not np.array_equal(
        _pack([np.rot90(p, d["rotation"]) for p in d["filled"]],
              list(range(len(d["panels"]))), d["layout"], d["divider"]),
        _answer(d)) for d in train), "B omission is unobservable")
    _require(any(len(_orders([tuple(range(len(s))) for s in d["inlets"]],
                            [tuple(range(len(s))) for s in d["outlets"]], d["source"])) == 2
                 for d in train), "cardinality alone still identifies every teaching chain")
    _require(any(any(a != b and len(a) == len(b) and b for a, b in zip(d["inlets"], d["outlets"])) for d in train), "missing equal-cardinality state change")
    _require(any(any(s["simultaneous"] for s in d["stats"]) for d in train), "missing simultaneous-collision demonstration")
    _require(any(any(s["multi_hit"] for s in d["stats"]) for d in train), "missing many-rays/one-run demonstration")
    _require(any(any(s["unreached"] for s in d["stats"]) for d in train), "missing inert obstacles")
    def has_loop(d):
        return any(a == b for a, b in zip(d["inlets"], d["outlets"]))
    _require(not any(has_loop(d) for d in test) or any(has_loop(d) for d in train), "repeated state not taught")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 sequential seeds, bounded microexamples, invariance and mutation checks.

    Returns aggregate diagnostics, never writes payloads or execution traces.
    """
    p = _blank_panel(6, 9, (4,), True)
    p[2, 3:6] = p[3, 2:7] = 2
    out, state, stats = _illuminate(p, 0, 1, 2, (4,))
    assert state == (1, 7)
    assert np.all(out[1, 2:7] == 1) and np.all(out[4:, [1, 7]] == 1)
    assert np.all(out[p == 2] == 2) and stats["multi_hit"] == 1
    p = _blank_panel(4, 9, (2, 6), True)
    p[2, [2, 6]] = 2
    out, state, stats = _illuminate(p, 0, 1, 2, (2, 6))
    assert state == (1, 3, 5, 7) and stats["simultaneous"] == 1
    assert np.all(out[3, [1, 3, 5, 7]] == 1)
    # Two geometrically distinct loops with identical signatures are ambiguous.
    assert len(_orders([(1,), (2,), (2,), (2,)], [(2,), (2,), (2,), ()], 0)) == 2
    assert _orders([(1,), (2,), (2,), (3,)], [(2,), (2,), (3,), ()], 0) == [(0, 1, 2, 3)]
    shapes, geometries, directions, layouts, counts, hashes = set(), set(), set(), set(), set(), set()
    role_colors = [set() for _ in range(5)]
    feature_counts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_puzzle_digest(puzzle))
        if seed in (0, 17, 41):
            assert _puzzle_digest(generate_puzzle(seed)) == _puzzle_digest(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"].copy()
                d = _decode(a)
                assert np.array_equal(solve(a), pair["output"]) and np.array_equal(a, pair["input"])
                directions.add(d["rotation"])
                layouts.add(d["layout"])
                counts.add(len(d["panels"]))
                shapes.add(a.shape)
                for s in d["stats"]:
                    feature_counts.update(s)
                for i, name in enumerate(("bg", "guide", "barrier", "marker", "divider")):
                    role_colors[i].add(d[name])
                for p in d["panels"]:
                    normalized = np.zeros_like(p)
                    normalized[p == d["barrier"]] = 2
                    normalized[(p == d["guide"]) | (p == d["marker"])] = 1
                    geometries.add(_key(normalized))
                if seed == 0 and split == "test":
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(a, k)), np.rot90(pair["output"], k))
                    color_map = np.arange(9, -1, -1, dtype=np.int8)
                    assert np.array_equal(solve(color_map[a]), color_map[pair["output"]])
                    assert np.array_equal(solve(np.fliplr(a)), np.fliplr(pair["output"]))
    assert len(hashes) == 42 and directions == set(range(4)) and counts == {3, 4, 5, 6}
    assert len(shapes) >= 30 and len(geometries) >= 250
    assert (2, 2) in layouts and ((2, 3) in layouts or (3, 2) in layouts)
    assert all(colors == set(range(10)) for colors in role_colors)
    assert all(feature_counts[name] > 0 for name in ("split", "clipped", "simultaneous", "multi_hit", "merged", "unreached", "singleton"))
    # All legal explicit count combinations; no retained duplicate puzzle payloads.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(7109, nt, nq)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0, 2, 1)
    damaged["train"][0]["output"] = damaged["train"][0]["output"].copy()
    damaged["train"][0]["output"][0, 0] = (int(damaged["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "input_shapes": len(shapes),
            "normalized_panel_geometries": len(geometries), "directions": sorted(directions),
            "panel_counts": sorted(counts), "input_layouts": sorted(layouts),
            "local_feature_counts": dict(sorted(feature_counts.items())),
            "independent_semantic_check": False}
