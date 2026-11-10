"""Locally unstretch panels, then assemble their unique geometric port chain.

Only standard-library modules and NumPy are used. No parent generator is imported.
All solving conventions are visible in the input. Generation retains no state.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 992831869, 415292641, 964790885, 649174163, 473600517,
              538753641, 728865384, 358692243, 510172824, 329429630, 39601764,
              399490771, 51242434, 795519804, 605115004, 587542550, 734978353,
              915939495, 350968054, 704620047, 24371380, 63140520, 670323977,
              257982423, 499061558, 686314576, 58351372, 91682430, 821034680,
              291153117, 910880149, 698315669, 681417975, 632543025, 526909442,
              153042256, 417856939, 740064557, 824340764, 126051716, 494543888)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _run_indices(a):
    """Compare BOTH sets of complete vectors in the original panel."""
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return np.flatnonzero(rows), np.flatnonzero(cols)


def _collapse(a):
    rows, cols = _run_indices(a)
    return a[np.ix_(rows, cols)].copy()


def _run_lengths(a):
    rows, cols = _run_indices(a)
    return np.diff(np.r_[rows, a.shape[0]]), np.diff(np.r_[cols, a.shape[1]])


def _partition(a):
    """Read the exclusive full-line separator, common corners, and raw tiles."""
    candidates = []
    for color in np.unique(a):
        rr = np.flatnonzero(np.all(a == color, axis=1)).tolist()
        cc = np.flatnonzero(np.all(a == color, axis=0)).tolist()
        if not rr and not cc:
            continue
        rows, cols = [-1] + rr + [a.shape[0]], [-1] + cc + [a.shape[1]]
        heights = [y - x - 1 for x, y in zip(rows, rows[1:])]
        widths = [y - x - 1 for x, y in zip(cols, cols[1:])]
        if min(heights + widths) < 1 or len(set(heights)) != 1 or len(set(widths)) != 1:
            continue
        n = len(heights) * len(widths)
        if n < 2:
            continue
        panels = [a[r + 1:rows[i + 1], c + 1:cols[j + 1]].copy()
                  for i, r in enumerate(rows[:-1]) for j, c in enumerate(cols[:-1])]
        if any(np.any(p == color) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels
                   for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        candidates.append({"raw": panels, "divider": int(color),
                           "background": next(iter(corners)),
                           "layout": (len(heights), len(widths))})
    _require(len(candidates) == 1, "need one unambiguous full-line panel partition")
    return candidates[0]


def _signatures(p, bg, axis):
    q = p if axis == "H" else p.T
    return (tuple(map(int, np.flatnonzero(q[:, 0] != bg))),
            tuple(map(int, np.flatnonzero(q[:, -1] != bg))))


def _chain_orders(signatures):
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(order, remaining):
        if len(answers) >= 2:
            return
        if not remaining:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        outgoing = signatures[order[-1]][1]
        if not outgoing:
            return
        for i in remaining:
            if signatures[i][0] == outgoing:
                visit(order + [i], tuple(j for j in remaining if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _decode(a):
    info = _partition(a)
    panels = [_collapse(p) for p in info["raw"]]
    _require(len({p.shape for p in panels}) == 1, "normalized panels must have equal dimensions")
    bg, answers = info["background"], []
    for axis in ("H", "V"):
        oriented = [p if axis == "H" else p.T for p in panels]
        if any(np.any(p[0] != bg) or np.any(p[-1] != bg) for p in oriented):
            continue
        signatures = [_signatures(p, bg, axis) for p in panels]
        orders = _chain_orders(signatures)
        if orders:
            _require(len(orders) == 1, "ambiguous complete port chain")
            answers.append((axis, signatures, orders[0]))
    _require(len(answers) == 1, "need exactly one directed opposing-port chain")
    axis, signatures, order = answers[0]
    info.update(panels=panels, axis=axis, signatures=signatures, order=order)
    return info


def _assemble(panels, order, layout, divider):
    nr, nc = layout
    h, w = panels[0].shape
    a = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, i in enumerate(order):
        r, c = divmod(slot, nc)
        a[r * (h + 1):r * (h + 1) + h,
          c * (w + 1):c * (w + 1) + w] = panels[i]
    return a


def _answer(info):
    n = len(info["panels"])
    layout = (1, n) if info["axis"] == "H" else (n, 1)
    a = _assemble(info["panels"], info["order"], layout, info["divider"])
    _require(all(d <= 30 for d in a.shape), "assembled chain exceeds 30 cells")
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Normalize each visible panel, then use its resulting ports; input only."""
    return _answer(_decode(_grid(input_grid)))


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    stack, seen = [next(iter(points))], set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        stack.extend(q for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                     if q in points and q not in seen)
    return seen == points


def _pool(h):
    return [(r,) for r in range(1, h - 1)] + list(combinations(range(1, h - 1), 2))


def _links(rng, h, n, witness):
    pool = _pool(h)
    if witness == 0:
        return [tuple([int(r)]) for r in rng.choice(np.arange(1, h - 1), 2, replace=False)]
    if witness == 1:
        # Two equally large but different signatures, one contiguous and one split.
        adjacent = _choose(rng, [(r, r + 1) for r in range(1, h - 2)])
        separated = _choose(rng, [s for s in pool if len(s) == 2 and s[1] - s[0] > 1])
        result = [adjacent, separated]
        rng.shuffle(result)
        return result
    for _ in range(32):
        links = [_choose(rng, pool) for _ in range(n - 1)]
        signatures = list(zip([()] + links, links + [()]))
        if len(_chain_orders(signatures)) == 1:
            return links
    indices = rng.choice(len(pool), n - 1, replace=False)
    return [pool[int(i)] for i in indices]


def _draw_path(body, start, end, rng):
    r, c = start
    body.add((r, c))
    while (r, c) != end:
        choices = []
        if r != end[0]:
            choices.append((r + (1 if r < end[0] else -1), c))
        if c != end[1]:
            choices.append((r, c + (1 if c < end[1] else -1)))
        r, c = _choose(rng, choices)
        body.add((r, c))


def _make_panel(rng, h, w, incoming, outgoing, bg, colors, force_weave=False):
    """Orthogonal line lattices, bent/branched paths and color-bearing bodies.

    Canonical tiles are irreducible, so independent expansion can be inverted.
    The lattice fallback is constructive for all sampled odd sides >=5.
    """
    style = 2 if force_weave else int(rng.integers(3))
    attempts = 48 if style == 1 else 1
    for _ in range(attempts + 1):
        p = np.full((h, w), bg, dtype=np.int8)
        if style == 0 or (style == 1 and _ == attempts):
            hc = [_choose(rng, colors) for r in range(1, h - 1, 2)]
            vc = [_choose(rng, colors) for c in range(1, w - 1, 2)]
            for j, c in enumerate(range(1, w - 1, 2)):
                p[1:-1, c] = vc[j]
            for i, r in enumerate(range(1, h - 1, 2)):
                p[r, 1:-1] = hc[i]
                for j, c in enumerate(range(1, w - 1, 2)):
                    p[r, c] = _choose(rng, (hc[i], vc[j]))
        elif style == 2:
            if len(colors) == 1:
                # A monochrome weave uses geometry to distinguish adjacent vectors.
                for r in range(1, h - 1):
                    p[r, 1:-1] = colors[0]
                    if r % 2 == 0:
                        p[r, 2:-1:2] = bg
            else:
                for r in range(1, h - 1):
                    for c in range(1, w - 1):
                        p[r, c] = colors[(r + c) % len(colors)]
        else:
            roots = [(r, 1) for r in incoming] + [(r, w - 2) for r in outgoing]
            waypoints = roots + [(int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
                                 for _ in range(int(rng.integers(2, 7)))]
            rng.shuffle(waypoints)
            body = set()
            for x, y in zip(waypoints, waypoints[1:]):
                _draw_path(body, x, y, rng)
            for r, c in sorted(body):
                p[r, c] = _choose(rng, colors)
        for r in incoming:
            p[r, 0] = colors[0]
            p[r, 1] = colors[0]
        for r in outgoing:
            p[r, -1] = colors[-1]
            p[r, -2] = colors[-1]
        if _collapse(p).shape == (h, w) and _connected(p != bg):
            return p
    raise RuntimeError("constructive canonical panel failed")


def _counts(seed, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if value is not None:
            _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
                     name + " must be an integer")
            _require(low <= value <= high, name + " out of range")
    # None is a supported convenience; ordinary defaults are exactly 4 and 1.
    if num_train is None:
        minimum = 2 if num_test is None else max(2, int(num_test) + 1)
        num_train = minimum + seed % (11 - minimum)
    if num_test is None:
        num_test = 1 + seed % min(4, int(num_train) - 1)
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _configuration(rng, n, witness):
    options = []
    for h in (5, 7, 9, 11):
        for w in (5, 7, 9, 11, 13):
            if n * (w + 1) - 1 > 30:
                continue
            for nr in range(1, n + 1):
                if n % nr:
                    continue
                nc = n // nr
                cap_h, cap_w = 31 // nr - 1, 31 // nc - 1
                # Always stretch the transverse direction to make raw port positions unreliable.
                min_h = h + 2
                min_w = w + (1 if n != 5 else 0)
                if min_h <= cap_h and min_w <= cap_w:
                    options.append((h, w, (nr, nc), cap_h, cap_w))
    _require(bool(options), "no feasible stretched panel geometry")
    h, w, layout, cap_h, cap_w = _choose(rng, options)
    # Local repeat counts, never absolute distance, define the solution.
    raw_h = int(rng.integers(h + 2, min(cap_h, h + (7 if witness is not None else 12)) + 1))
    raw_w = int(rng.integers(w + (n != 5), min(cap_w, w + (5 if witness is not None else 9)) + 1))
    return h, w, layout, raw_h, raw_w


def _repeat_counts(rng, length, total):
    result = np.ones(length, dtype=int)
    extra = total - length
    if extra:
        if rng.random() < .45:
            result[int(rng.integers(length))] += extra
        else:
            result += rng.multinomial(extra, np.ones(length) / length)
    return result


def _make_pair(rng, n, axis, witness, bg, divider, color_mode):
    h, w, layout, raw_h, raw_w = _configuration(rng, n, witness)
    palette = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, divider)])))
    if color_mode == "shared":
        colors = [[palette[0]]] * n
    elif color_mode == "distinct":
        colors = [[palette[i]] for i in range(n)]
    elif color_mode == "partial":
        k = int(rng.integers(2, n)) if n > 2 else 1
        labels = list(range(k)) + list(map(int, rng.integers(k, size=n - k)))
        rng.shuffle(labels)
        colors = [[palette[i]] for i in labels]
    else:
        colors = [list(map(int, rng.choice(palette, size=int(rng.integers(2, 5)), replace=False)))
                  for _ in range(n)]
    links = _links(rng, h, n, witness)
    panels = [_make_panel(rng, h, w, () if i == 0 else links[i - 1],
                          () if i == n - 1 else links[i], bg, colors[i],
                          force_weave=(witness == 1 and i == 1)) for i in range(n)]
    # Construct an explicit dependency witness: the correct normalized joins fail
    # if full raw boundary occupancy is matched before panel-local contraction.
    raw = None
    for _ in range(32):
        raw = [np.repeat(np.repeat(p, _repeat_counts(rng, h, raw_h), axis=0),
                         _repeat_counts(rng, w, raw_w), axis=1) for p in panels]
        signatures = [_signatures(p, bg, "H") for p in raw]
        if any(signatures[i][1] != signatures[i + 1][0] for i in range(n - 1)):
            break
    else:
        # Both totals remain equal: move extra transverse copies from the first
        # row to the last in the second panel. Every nonempty interior port shifts.
        rr0 = np.ones(h, dtype=int)
        rr0[0] += raw_h - h
        rr1 = np.ones(h, dtype=int)
        rr1[-1] += raw_h - h
        for i, rr in ((0, rr0), (1, rr1)):
            raw[i] = np.repeat(np.repeat(panels[i], rr, axis=0),
                               _repeat_counts(rng, w, raw_w), axis=1)
    order = list(map(int, rng.permutation(n)))
    if order == list(range(n)):
        order = order[1:] + order[:1]
    # Occasionally simple reversals/end swaps accompany arbitrary permutations.
    if axis == "V":
        raw = [p.T.copy() for p in raw]
        panels = [p.T.copy() for p in panels]
        layout = layout[::-1]
    inp = _assemble(raw, order, layout, divider)
    out = _assemble(panels, list(range(n)), (1, n) if axis == "H" else (n, 1), divider)
    return {"input": inp, "output": out}


def _fresh_recolor(pair, seen):
    """Bounded uniqueness fallback: 40 role bijections for at most 14 inputs."""
    for shift in range(10):
        for multiplier in (1, 3, 7, 9):
            lut = np.array([(multiplier * c + shift) % 10 for c in range(10)], dtype=np.int8)
            q = {k: lut[v] for k, v in pair.items()}
            if _key(q["input"]) not in seen:
                return q
    raise RuntimeError("unexpected exhaustion of distinct role assignments")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    first_axis = _choose(rng, ("H", "V"))
    # Two anchors suffice even at the minimum legal training count. Shuffle their
    # locations, while keeping their different pedagogical requirements explicit.
    plans = [(3, first_axis, 0, "shared"),
             (3, "V" if first_axis == "H" else "H", 1, "multi")]
    for _ in range(num_train - 2):
        plans.append((int(rng.integers(2, 4)), _choose(rng, ("H", "V")), None,
                      _choose(rng, ("shared", "distinct", "partial", "multi"))))
    rng.shuffle(plans)
    for _ in range(num_test):
        plans.append((int(rng.choice((4, 5), p=(.72, .28))), _choose(rng, ("H", "V")), None,
                      _choose(rng, ("shared", "distinct", "partial", "multi"))))
    previous_bg = previous_divider = None
    for index, (n, axis, witness, color_mode) in enumerate(plans):
        bg = _choose(rng, [c for c in range(10) if c != previous_bg])
        divider = _choose(rng, [c for c in range(10) if c not in (bg, previous_divider)])
        pair = _make_pair(rng, n, axis, witness, bg, divider, color_mode)
        if _key(pair["input"]) in seen:
            pair = _fresh_recolor(pair, seen)
        seen.add(_key(pair["input"]))
        split = "train" if index < num_train else "test"
        puzzle[split].append(pair)
        info = _partition(pair["input"])
        previous_bg, previous_divider = info["background"], info["divider"]
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_separated_repeat(p, axis):
    vectors = p if axis == 0 else p.T
    return len({row.tobytes() for row in vectors}) < len(vectors)


def _mask_collapse(p, bg):
    rows, cols = _run_indices((p != bg).astype(np.int8))
    return p[np.ix_(rows, cols)]


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    source, target = _grid(pair["input"]), _grid(pair["output"])
    info = _decode(source)
    _require(np.array_equal(_answer(info), target), "output differs from normalized port assembly")
    bg, axis, order = info["background"], info["axis"], info["order"]
    raw, panels = info["raw"], info["panels"]
    # Reconstruct EVERY input cell by the measured runs: no crossing repainting,
    # dropping a nonadjacent repeat, or unsupported denoising is permitted.
    shrinks, mask_sensitive = [], False
    for p, q in zip(raw, panels):
        rr, cc = _run_lengths(p)
        restored = np.repeat(np.repeat(q, rr, axis=0), cc, axis=1)
        _require(np.array_equal(restored, p), "run reconstruction lost literal values")
        _require(_connected(q != bg), "motif must remain connected")
        shrinks.append((p.shape[0] > q.shape[0], p.shape[1] > q.shape[1]))
        _require(any(shrinks[-1]), "each panel must contain an actual repeated run")
        mask_sensitive |= _mask_collapse(q, bg).shape != q.shape
    out_info = _partition(target)
    _require(out_info["layout"] == ((1, len(panels)) if axis == "H" else (len(panels), 1)),
             "output must be a straight strip")
    _require(out_info["divider"] == info["divider"], "preserve divider")
    _require(Counter(_key(p) for p in panels) == Counter(_key(p) for p in out_info["raw"]),
             "copy each normalized panel exactly once")
    _require(order != tuple(range(len(panels))), "demonstrate actual ordering")
    local_layout = _assemble(panels, list(range(len(panels))), info["layout"], info["divider"])
    _require(not np.array_equal(local_layout, target), "compression without ordering must change answer")
    raw_sigs = [_signatures(p, bg, axis) for p in raw]
    bad_raw = sum(raw_sigs[a][1] != raw_sigs[b][0] for a, b in zip(order, order[1:]))
    _require(bad_raw > 0, "raw positions must fail at a required normalized join")
    separated_axes = tuple(any(_has_separated_repeat(p, axis) for p in panels) for axis in (0, 1))
    # Stronger than a raw mismatch: absent/wrong assembly when A consumes raw ports.
    raw_orders = _chain_orders(raw_sigs)
    return {"n": len(panels), "axis": axis, "layout": info["layout"],
            "raw_shape": raw[0].shape, "panel_shape": panels[0].shape,
            "background": bg, "divider": info["divider"], "order": order,
            "mask_sensitive": bool(mask_sensitive), "shrinks": tuple(shrinks),
            "separated_axes": separated_axes,
            "local_vs_global": not np.array_equal(local_layout, _collapse(source)),
            "raw_bad_joins": bad_raw, "raw_orders": tuple(raw_orders),
            "port_sizes": tuple(sorted({len(s) for a, b in info["signatures"] for s in (a, b) if s})),
            "signature_kinds": tuple(sorted({"single" if len(s) == 1 else
                                             "contiguous" if s[-1] - s[0] + 1 == len(s) else "separated"
                                             for a, b in info["signatures"] for s in (a, b) if s})),
            "repeated_links": len({info["signatures"][i][1] for i in order[:-1]}) < len(panels) - 1}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(0, len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            record = _analyze_pair(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            records[split].append(record)
    train = records["train"]
    _require({r["axis"] for r in train} == {"H", "V"}, "teach both chain axes")
    _require(any(r["mask_sensitive"] for r in train), "teach literal-vector rather than mask equality")
    _require(any(all(x and y for x, y in r["shrinks"]) for r in train), "teach two-axis contraction")
    _require(any(r["local_vs_global"] for r in train), "teach panel-local rather than canvas contraction")
    _require(all(any(r["separated_axes"][axis] for r in train) for axis in (0, 1)),
             "teach preservation of separated equal rows and columns")
    _require({"single", "contiguous", "separated"} <= {k for r in train for k in r["signature_kinds"]},
             "teach all generated port kinds")
    _require(min(r["n"] for r in records["test"]) > max(r["n"] for r in train), "longer-chain test shift")
    return None


def _puzzle_digest(puzzle):
    serial = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
              for s in ("train", "test")}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """42 seeds, streamed in memory; no files, traces, subprocesses or logging."""
    # A literal crossing/adjacent-run witness. Separated equal rows/columns survive.
    q = np.array([[0, 0, 0, 0, 0], [0, 2, 3, 2, 0], [0, 4, 0, 4, 0],
                  [0, 2, 3, 2, 0], [0, 0, 0, 0, 0]], dtype=np.int8)
    p = np.repeat(np.repeat(q, [2, 1, 3, 1, 2], axis=0), [1, 2, 1, 4, 1], axis=1)
    assert np.array_equal(_collapse(p), q)
    assert np.array_equal(_collapse(_collapse(p)), q)
    assert np.array_equal(_collapse(p.T), q.T)
    # Full vectors are essential: equal masks with unequal colors do not contract.
    colored = np.array([[0, 0, 0], [0, 2, 0], [0, 3, 0], [0, 0, 0]], dtype=np.int8)
    assert _collapse(colored).shape == colored.shape
    # Directed terminals determine ends; distinguishable same-signature loops tie.
    a, b = (1,), (2,)
    assert _chain_orders([((), a), (a, b), (b, ())]) == [(0, 1, 2)]
    assert len(_chain_orders([((), a), (a, a), (a, a), (a, ())])) == 2
    # A hand-built compatible member outside the sampling sizes/counts: seven
    # narrow panels, different colors on opposing ports, and independent runs.
    links = [(1,), (2,), (3,), (4,), (5,), (1, 3)]
    narrow, stretched = [], []
    for i in range(7):
        tile = np.zeros((7, 3), dtype=np.int8)
        tile[1:-1, 1] = [2, 3, 2, 3, 2]
        for r in (() if i == 0 else links[i - 1]):
            tile[r, 0] = 4
        for r in (() if i == 6 else links[i]):
            tile[r, -1] = 5
        narrow.append(tile)
        rr = np.ones(7, dtype=int)
        rr[i] += 2
        stretched.append(np.repeat(tile, rr, axis=0))
    hand_input = _assemble(stretched, [4, 1, 6, 0, 5, 3, 2], (1, 7), 9)
    hand_output = _assemble(narrow, list(range(7)), (1, 7), 9)
    assert np.array_equal(solve(hand_input), hand_output)
    digests, input_shapes, output_shapes, panel_shapes, layouts = set(), set(), set(), set(), set()
    counts, backgrounds, dividers, axes, structures = set(), set(), set(), set(), set()
    raw_bad = mask_sensitive = repeated = one_axis = 0
    pair_total = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                r = _analyze_pair(pair)
                source, target = pair["input"], pair["output"]
                snapshot = source.copy()
                assert np.array_equal(solve(source), target)
                assert np.array_equal(snapshot, source), "solver mutated input"
                info = _decode(source)
                # Invariance under 90-degree coordinate exchange, without rotating
                # individual panels or making their directions a hidden convention.
                assert np.array_equal(solve(source.T), target.T)
                input_shapes.add(source.shape)
                output_shapes.add(target.shape)
                panel_shapes.add(r["panel_shape"])
                layouts.add(r["layout"])
                counts.add(r["n"])
                backgrounds.add(r["background"])
                dividers.add(r["divider"])
                axes.add(r["axis"])
                raw_bad += r["raw_bad_joins"] > 0
                mask_sensitive += r["mask_sensitive"]
                repeated += r["repeated_links"]
                one_axis += any(x != y for x, y in r["shrinks"])
                for panel in info["panels"]:
                    structures.add(_key((panel != r["background"]).astype(np.int8)))
                pair_total += 1
    assert len(digests) == 42 and counts == {2, 3, 4, 5}
    assert len(input_shapes) >= 40 and len(output_shapes) >= 15 and len(panel_shapes) >= 12
    assert len(layouts) >= 5 and len(structures) >= 100
    assert axes == {"H", "V"} and len(backgrounds) == len(dividers) == 10
    assert raw_bad == pair_total and mask_sensitive >= 42 and repeated >= 1 and one_axis >= 1
    assert _puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0))
    # All 30 legal explicit count combinations, convenience counts and strict errors.
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(73, train, test)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None), (None, None)):
        generate_puzzle(73, train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, False)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Mutations must fail even when all ordinary schema checks still pass.
    puzzle = generate_puzzle(9, 2, 1)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("changed normalized content accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": pair_total, "input_shapes": len(input_shapes),
            "output_shapes": len(output_shapes), "panel_shapes": len(panel_shapes),
            "layouts": len(layouts), "motif_masks": len(structures), "raw_port_failures": raw_bad,
            "mask_sensitive_pairs": mask_sensitive, "repeated_link_pairs": repeated,
            "one_axis_pairs": one_axis}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
