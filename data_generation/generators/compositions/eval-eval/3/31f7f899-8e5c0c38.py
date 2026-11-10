"""Folded ranks: sort reflected agreements on the quotient of a carrier.

All inference is input-only. Construction paints known templates; validation
checks conservation and order without calling the constructor or solve.
Nothing is written by this module, including self_test.
"""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _span(template):
    return template[-1] - template[0] + 1


def _analyze(value):
    """Read the anchored fold, its orbits, and cellwise reflected agreements."""
    a = _grid(value)
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > a.size / 2, "background must be a strict majority"
    rows = np.flatnonzero(np.all(a != bg, axis=1))
    cols = np.flatnonzero(np.all(a != bg, axis=0))
    assert len(rows) + len(cols) == 1, "one full carrier required"
    vertical = bool(len(cols))
    g = a.T if vertical else a
    r = int(cols[0] if vertical else rows[0])
    line_counts = np.bincount(g[r].astype(int), minlength=10)
    carrier = int(line_counts.argmax())
    assert np.count_nonzero(line_counts == line_counts[carrier]) == 1
    assert carrier != bg and line_counts[carrier] >= 3
    assert np.count_nonzero(g == carrier) == line_counts[carrier]
    slots = []
    x = 0
    explained = np.full(g.shape, bg, dtype=np.int8)
    explained[r] = g[r]
    while x < g.shape[1]:
        end = x + 1
        while end < g.shape[1] and g[r, end] == g[r, x]:
            end += 1
        color = int(g[r, x])
        if color != carrier:
            assert color != bg
            strip = g[:, x:end]
            assert np.all((strip == color) | (strip == bg))
            explained[:, x:end] = strip
            slots.append((x, end - x, color))
        x = end
    assert np.array_equal(explained, g), "pixels must belong to their port strips"
    assert 4 <= len(slots) <= 11
    assert all(b[0] > a[0] + a[1] for a, b in zip(slots, slots[1:]))
    axis2 = slots[0][0] + slots[-1][0] + slots[-1][1] - 1
    lookup = {s: i for i, s in enumerate(slots)}
    used, orbits = set(), []
    for i, (x, w, color) in enumerate(slots):
        if i in used:
            continue
        mate = (axis2 - (x + w - 1), w, color)
        assert mate in lookup, "anchors must be exactly mirrored in color and width"
        j = lookup[mate]
        members = (i,) if i == j else (i, j)
        used.update(members)
        left_mask = g[:, x:x + w] == color
        mx = slots[j][0]
        agreement = left_mask & (g[:, mx:mx + w] == color)[:, ::-1]
        occupied = np.any(agreement, axis=1)
        assert np.all(np.all(agreement, axis=1) == occupied), "agreement must be full rungs"
        template = tuple(int(t) - r for t in np.flatnonzero(occupied))
        assert 0 in template
        raw = tuple(int(t) - r for t in np.flatnonzero(
            np.any(left_mask, axis=1) | np.any(g[:, mx:mx + w] == color, axis=1)))
        support = len(template) * w * len(members)
        actual = sum(np.count_nonzero(g[:, slots[k][0]:slots[k][0] + w] == color)
                     for k in members)
        orbits.append({"members": members, "width": w, "color": color,
                       "distance2": abs(2 * x + w - 1 - axis2),
                       "template": template, "raw": raw, "noise": int(actual - support)})
    orbits.sort(key=lambda o: o["distance2"])
    assert len({o["color"] for o in orbits}) == len(orbits)
    assert 2 <= sum(len(o["members"]) == 2 for o in orbits) <= 5
    assert sum(len(o["members"]) == 1 for o in orbits) <= 1
    ties = {}
    for o in orbits:
        t = o["template"]
        assert ties.setdefault(_span(t), t) == t, "different-template length ties forbidden"
    return {"grid": g, "background": bg, "carrier": carrier, "row": r,
            "vertical": vertical, "axis2": axis2, "slots": slots, "orbits": orbits}


def _render(analysis, templates):
    g = analysis["grid"]
    out = np.full(g.shape, analysis["background"], dtype=np.int8)
    out[analysis["row"]] = g[analysis["row"]]
    for orbit, template in zip(analysis["orbits"], templates):
        for index in orbit["members"]:
            x, w, color = analysis["slots"][index]
            for offset in template:
                assert 0 <= analysis["row"] + offset < g.shape[0]
                out[analysis["row"] + offset, x:x + w] = color
    return out.T.copy() if analysis["vertical"] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the fold from the intact anchors; rank agreements, once per orbit."""
    a = _analyze(input_grid)
    return _render(a, sorted((o["template"] for o in a["orbits"]), key=_span))


def _counts(num_train, num_test):
    for name, n, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= n <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _random_template(rng, length, sparse=None):
    lo = -int(rng.integers(length))
    hi = lo + length - 1
    probability = float(rng.uniform(.18, .8)) if sparse is None else (.22 if sparse else 1.0)
    offsets = {lo, hi, 0}
    offsets.update(d for d in range(lo, hi + 1) if rng.random() < probability)
    return tuple(sorted(offsets))


def _templates(rng, count, witness):
    if witness == 1:
        # A central medium template, a dense wide short bar, and a sparse narrow
        # long bar expose orbit multiplicity, span vs count/area, and offsets.
        lengths = [1, int(rng.integers(3, 5)), int(rng.integers(6, 8)), int(rng.integers(9, 12))]
        templates = [(0,), tuple(range(1 - lengths[1], 1)), tuple(range(lengths[2])),
                     (1 - lengths[3], 2 - lengths[3], 0)]
        permutation = [(1, 2, 3, 0), (1, 3, 0, 2)][int(rng.integers(2))]
        return [templates[i] for i in permutation]
    if witness == 2:
        short, long = int(rng.integers(4, 7)), int(rng.integers(8, 13))
        return [_random_template(rng, long, True), tuple(range(short))]
    lengths = [int(x) for x in rng.choice(np.arange(1, 13), count, replace=False)]
    templates = [_random_template(rng, n) for n in lengths]
    if rng.random() < .28:
        dst, src = [int(x) for x in rng.choice(count, 2, replace=False)]
        templates[dst] = templates[src]
    mode = int(rng.integers(5))
    if mode == 0:
        templates.sort(key=_span)
    elif mode == 1:
        templates.sort(key=_span, reverse=True)
    return templates


def _layout(rng, paired, center_width, locked_widths, central_gap=None):
    widths = [int(x) for x in rng.integers(1, 4, size=paired)]
    widths = [locked_widths.get(i, w) for i, w in enumerate(widths)]
    gaps = [int(x) for x in rng.integers(1, 3, size=paired - 1)]
    middle = (center_width + 2 * int(rng.integers(1, 3)) if center_width else
              int(central_gap or rng.integers(1, 5)))
    while 2 * sum(widths) + 2 * sum(gaps) + middle > 28:
        choices = [i for i, w in enumerate(widths) if w > locked_widths.get(i, 1)]
        if choices:
            widths[max(choices, key=lambda i: widths[i])] -= 1
        else:
            choices = [i for i, gap in enumerate(gaps) if gap > 1]
            assert choices
            gaps[choices[0]] -= 1
    region = 2 * sum(widths) + 2 * sum(gaps) + middle
    accent_cells = 2 * sum(widths) + center_width
    exposure = max(3, 2 * max(widths) + 1, center_width + 1)
    minimum = max(region, accent_cells + exposure)
    assert minimum <= 30
    width = int(rng.integers(minimum, 31))
    left_margin = int(rng.integers(width - region + 1))
    axis2 = 2 * left_margin + region - 1
    orbits = []
    if center_width:
        orbits.append([((axis2 - center_width + 1) // 2, center_width)])
    distance = middle
    for i, w in enumerate(widths):
        left = (axis2 - distance + 1) // 2 - w
        right = axis2 - (left + w - 1)
        orbits.append([(left, w), (right, w)])
        if i < paired - 1:
            distance += 2 * (w + gaps[i])
    return width, axis2, orbits


def _construct_pair(rng, paired, center_width, vertical, witness=0, central_gap=None):
    k = paired + bool(center_width)
    templates = _templates(rng, k, witness)
    locked = {}
    if witness == 1:
        ranked = sorted(templates, key=_span)
        # Both selected templates are paired, not the central orbit.
        locked[templates.index(ranked[2]) - 1] = 2
        locked[templates.index(ranked[3]) - 1] = 1
    width, axis2, geometry = _layout(rng, paired, center_width, locked, central_gap)
    palette = [int(c) for c in rng.choice(10, k + 2, replace=False)]
    bg, carrier, colors = palette[0], palette[1], palette[2:]
    lo = min(t[0] for t in templates) - 1
    hi = max(t[-1] for t in templates) + 1
    noise = [[] for _ in geometry]
    clean_pair = not witness and rng.random() < .15
    for i, (ports, template) in enumerate(zip(geometry, templates)):
        if len(ports) == 1 or clean_pair:
            continue
        absent = [d for d in range(lo, hi + 1) if d not in template]
        rng.shuffle(absent)
        amount = int(rng.integers(0, min(5, len(absent)) + 1))
        if witness:
            amount = min(3, len(absent))
        chosen = absent[:amount]
        if witness and template == min(templates, key=_span):
            # Opposite unmatched extremities make the raw union's rank wrong.
            chosen = [lo, hi] + [d for d in chosen if d not in (lo, hi)]
        w = ports[0][1]
        for j, offset in enumerate(chosen):
            side = j % 2 if witness else int(rng.integers(2))
            # Partial-width and complete-rung noise are both cellwise deleted.
            bits = int(rng.integers(1, 1 << w))
            noise[i].append((side, offset, bits))
    used_offsets = [d for t in templates for d in t]
    used_offsets += [d for ns in noise for _, d, _ in ns]
    minoff, maxoff = min(used_offsets), max(used_offsets)
    extent = maxoff - minoff + 1
    height = int(rng.integers(extent + 2, 31))
    row = int(rng.integers(1 - minoff, height - maxoff - 1))

    def paint(h, r):
        inp = np.full((h, width), bg, dtype=np.int8)
        out = inp.copy()
        inp[r] = out[r] = carrier
        for i, (ports, template, target) in enumerate(zip(geometry, templates, sorted(templates, key=_span))):
            for x, w in ports:
                for d in template:
                    inp[r + d, x:x + w] = colors[i]
                for d in target:
                    out[r + d, x:x + w] = colors[i]
            for side, offset, bits in noise[i]:
                x, w = ports[side]
                for col in range(w):
                    if bits & (1 << col):
                        inp[r + offset, x + col] = colors[i]
        return inp, out

    inp, out = paint(height, row)
    if min(np.count_nonzero(inp == bg), np.count_nonzero(out == bg)) <= inp.size / 2:
        # A bounded density correction changes only background margins.
        height = 30
        row = int(rng.integers(1 - minoff, height - maxoff - 1))
        inp, out = paint(height, row)
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _recolor_unique(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # At most 13 prior inputs: 90 role assignments give a finite guarantee.
    a = _analyze(pair["input"])
    old = [a["background"], a["carrier"]] + [o["color"] for o in a["orbits"]]
    for bg in range(10):
        for carrier in range(10):
            if bg == carrier:
                continue
            new = [bg, carrier] + [c for c in range(10) if c not in (bg, carrier)]
            table = np.arange(10, dtype=np.int8)
            for x, y in zip(old, new):
                table[x] = y
            candidate = {role: table[grid] for role, grid in pair.items()}
            if _key(candidate["input"]) not in seen:
                return candidate
    raise AssertionError("finite recoloring space exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    orientation = int(rng.integers(3))  # horizontal, vertical, or visibly mixed
    center_width = int(rng.integers(1, 4))
    first_parity = (center_width - 1) % 2
    opposite_gap = int(rng.choice([2, 4] if first_parity == 0 else [1, 3]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = i + 1 if split == "train" and i < 2 else 0
            paired = 3 if witness == 1 else 2 if witness == 2 else int(
                rng.integers(2, 4) if split == "train" else rng.integers(4, 6))
            central = center_width if witness == 1 else 0 if witness == 2 else int(rng.choice([0, 0, 1, 2, 3]))
            vertical = bool(orientation == 1 or (orientation == 2 and (
                i % 2 if witness else rng.integers(2))))
            pair = _construct_pair(rng, paired, central, vertical, witness,
                                   opposite_gap if witness == 2 else None)
            pair = _recolor_unique(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(analysis):
    os = analysis["orbits"]
    ts = [o["template"] for o in os]
    ranked = sorted(ts, key=_span)
    raw = [o["raw"] for o in os]
    return {"shape": tuple(analysis["grid"].shape), "vertical": analysis["vertical"],
            "background": analysis["background"], "carrier": analysis["carrier"],
            "parity": analysis["axis2"] % 2,
            "paired": sum(len(o["members"]) == 2 for o in os),
            "center": len(os[0]["members"]) == 1,
            "widths": tuple(o["width"] for o in os),
            "templates": tuple(ts), "lengths": tuple(_span(t) for t in ts),
            "noise": sum(o["noise"] for o in os),
            "sort_changes": ts != ranked,
            "raw_disagrees": sorted(raw, key=_span) != ranked,
            "raw_rank_changes": sorted(range(len(os)), key=lambda i: _span(raw[i])) !=
                                sorted(range(len(os)), key=lambda i: _span(ts[i])),
            "area_disagrees": [o["template"] for o in sorted(os, key=lambda o: _span(o["template"]) * o["width"])] != ranked,
            "count_disagrees": sorted(ts, key=len) != ranked,
            "reverse_disagrees": ts[::-1] != ranked,
            "holes": any(len(t) < _span(t) for t in ts),
            "ties": len(set(map(_span, ts))) < len(ts),
            "offcenter_moved": any(t != dst and t[0] + t[-1] != 0 for t, dst in zip(ts, ranked)),
            "off_canvas_center": analysis["axis2"] != analysis["grid"].shape[1] - 1}


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    a, b = _analyze(inp), _analyze(out)
    for field in ("background", "carrier", "row", "vertical", "axis2", "slots"):
        assert a[field] == b[field], "port geometry, colors and carrier must stay fixed"
    incoming = Counter(o["template"] for o in a["orbits"])
    outgoing = Counter(o["template"] for o in b["orbits"])
    assert incoming == outgoing, "conserve templates once per reflection orbit"
    lengths = [_span(o["template"]) for o in b["orbits"]]
    assert all(x <= y for x, y in zip(lengths, lengths[1:])), "rank from fold outward"
    assert all(o["noise"] == 0 for o in b["orbits"]), "output must be fully mirror-supported"
    return _features(a)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, fs = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            f = _check_pair(pair)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen, "inputs must be distinct"
            seen.add(key)
            fs[split].append(f)
    train = fs["train"]
    assert sum(f["sort_changes"] and f["noise"] > 0 for f in train) >= 2
    for field in ("area_disagrees", "count_disagrees", "reverse_disagrees", "holes",
                  "raw_disagrees", "raw_rank_changes", "offcenter_moved"):
        assert any(f[field] for f in train), f"missing training distinction: {field}"
    assert {f["center"] for f in train} == {False, True}
    assert {f["parity"] for f in train} == {0, 1}
    assert len({f["paired"] for f in train}) >= 2
    assert max(f["paired"] for f in train) < min(f["paired"] for f in fs["test"])
    assert {f["vertical"] for f in fs["test"]} <= {f["vertical"] for f in train}


def self_test():
    """42 streaming seeds; no saved grids, logs, traces, or accumulated puzzles."""
    feature_sets = {name: set() for name in ("shape", "background", "carrier", "vertical", "parity", "paired", "center")}
    widths, signatures, evidence = set(), set(), Counter()
    rejected = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                before = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, before), "solve must not mutate its input"
                a = _analyze(inp)
                f = _features(a)
                digest.update(bytes(inp.shape) + inp.tobytes())
                for name, values in feature_sets.items():
                    values.add(f[name])
                widths.update(f["widths"])
                for name in ("noise", "holes", "ties", "raw_rank_changes", "offcenter_moved", "off_canvas_center"):
                    evidence[name] += bool(f[name])
                evidence["clean"] += f["noise"] == 0
                evidence["identity"] += np.array_equal(pair["input"], pair["output"])
                os = a["orbits"]
                ts = [o["template"] for o in os]
                ranked = sorted(ts, key=_span)
                variants = {
                    "omit_A": ts,
                    "omit_B": sorted((o["raw"] for o in os), key=_span),
                    "area": [o["template"] for o in sorted(os, key=lambda o: _span(o["template"]) * o["width"])],
                    "count": sorted(ts, key=len),
                    "reverse": ts[::-1],
                    "fill_holes": [tuple(range(t[0], t[-1] + 1)) for t in ranked],
                    "recenter": [tuple(d - (t[0] + t[-1]) // 2 for d in t) for t in ranked],
                }
                for name, templates in variants.items():
                    if name in rejected or templates == ranked or any(0 not in t for t in templates):
                        continue
                    if any(a["row"] + t[0] < 0 or a["row"] + t[-1] >= a["grid"].shape[0] for t in templates):
                        continue
                    bad = {"input": inp, "output": _render(a, templates)}
                    try:
                        _check_pair(bad)
                    except AssertionError:
                        rejected.add(name)
                    else:
                        raise AssertionError(f"accepted wrong {name} output")
        signatures.add(digest.digest())
        if seed < 2:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in ("train", "test")
                       for p, q in zip(puzzle[s], again[s]) for k in ("input", "output"))
    assert len(signatures) == 42
    assert len(feature_sets["shape"]) >= 30
    assert feature_sets["background"] == feature_sets["carrier"] == set(range(10))
    assert feature_sets["vertical"] == feature_sets["parity"] == feature_sets["center"] == {0, 1}
    assert feature_sets["paired"] == {2, 3, 4, 5} and widths == {1, 2, 3}
    assert all(evidence[k] for k in evidence)
    assert rejected == {"omit_A", "omit_B", "area", "count", "reverse", "fill_holes", "recenter"}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(4242, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A hand-specified compatible member outside sampler ranges: width four,
    # extent twenty, asymmetric canvas margins, central-to-pair transfer.
    fixture = np.zeros((30, 30), dtype=np.int8)
    expected = fixture.copy()
    fixture[12] = expected[12] = 9
    ports = [(1, 4, 1), (7, 2, 2), (10, 1, 3), (12, 2, 2), (16, 4, 1)]
    incoming = {1: (-1, 0, 1), 2: (0, 3), 3: (-10, -7, 0, 9)}
    outgoing = {1: (-10, -7, 0, 9), 2: (0, 3), 3: (-1, 0, 1)}
    for x, w, color in ports:
        for d in incoming[color]:
            fixture[12 + d, x:x + w] = color
        for d in outgoing[color]:
            expected[12 + d, x:x + w] = color
    fixture[22, 1:3] = 1
    fixture[7, 12:14] = 2
    assert np.array_equal(solve(fixture), expected)
    assert np.array_equal(solve(fixture.T), expected.T)
    _check_pair({"input": fixture, "output": expected})
    return {"seeds": 42, "pairs": 210, "shapes": len(feature_sets["shape"]),
            "additional_fixtures": 1, "mutation_types_rejected": sorted(rejected), "features": dict(evidence)}
