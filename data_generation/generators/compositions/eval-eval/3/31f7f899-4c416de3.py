"""Elastic lanterns: sort spans and repair ornaments at the resized corners.

No filesystem access, parent imports, cached puzzles, or generation state is
used by solve. All randomness belongs to generate_puzzle's local NumPy RNG.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
HOST_SEEDS = (0,151409183,468744785,860181662,788122576,577200404,
              688765775,748286732,979300767,688336373,30690677,847050392,
              6754570,338027103,803602059,804845610,458923305,457443395,
              268866564,13471656,34622101,730571356,824124020,145599377,
              276952645,860629448,650599455,12497145,565817005,692473827,
              878611608,625978271,206924187,543206339,604916731,90902098,
              115320011,650938469,938610448,839337774,11413374,387422690)


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _corners(record, row, span):
    x, width = record["x"], record["width"]
    top, bottom = row + span[0], row + span[1]
    return ((top, x), (top, x + width - 1),
            (bottom, x), (bottom, x + width - 1))


def _base(scene, spans):
    a = np.full(scene["shape"], scene["background"], dtype=np.int8)
    row = scene["row"]
    a[row, :] = scene["carrier"]
    for record, (lo, hi) in zip(scene["records"], spans):
        x, w, color = record["x"], record["width"], record["color"]
        top, bottom = row + lo, row + hi
        a[top, x:x + w] = a[bottom, x:x + w] = color
        a[top:bottom + 1, x] = a[top:bottom + 1, x + w - 1] = color
        a[row, x:x + w] = color
    return a


def _paint(scene, spans, complete=True, rotate=False):
    a = _base(scene, spans)
    for record, span in zip(scene["records"], spans):
        corners = _corners(record, scene["row"], span)
        for q, (color, clue) in record["marks"].items():
            y, x = corners[q]
            dy, dx = DIRS[q]
            points = scene["stencil"] if complete else clue
            # Explicit competing convention, used only for countermodel checks.
            if rotate and q in (1, 2):
                points = {(v, u) for u, v in points}
            for u, v in points:
                yy, xx = y + dy * u, x + dx * v
                assert 0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]
                a[yy, xx] = color
    return a.T.copy() if scene["vertical"] else a


def _read(input_grid):
    """Recover slots, frame intervals, local clues and the intact stencil."""
    original = _grid(input_grid)
    counts = np.bincount(original.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[bg]) == 1
    rows = np.flatnonzero(np.all(original != bg, axis=1))
    cols = np.flatnonzero(np.all(original != bg, axis=0))
    assert len(rows) + len(cols) == 1, "unique carrier required"
    vertical = bool(len(cols))
    a = original.T if vertical else original
    row = int(cols[0] if vertical else rows[0])
    carrier = int(a[row, 0])
    assert carrier == a[row, -1] and carrier != bg
    line = a[row]
    records = []
    x = 0
    while x < a.shape[1]:
        end = x + 1
        while end < a.shape[1] and line[end] == line[x]:
            end += 1
        color = int(line[x])
        if color != carrier:
            width = end - x
            assert width >= 5 and color != bg
            # The central column is beyond both corner-stencil footprints.
            hits = np.flatnonzero(a[:, x + width // 2] == color)
            assert len(hits) == 3 and hits[1] == row
            top, bottom = int(hits[0]), int(hits[-1])
            assert row - top >= 3 and bottom - row >= 3
            records.append({"x": x, "width": width, "color": color,
                            "span": (top - row, bottom - row), "marks": {}})
        x = end
    assert 2 <= len(records) <= 3
    frame_colors = {r["color"] for r in records}
    scene = {"background": bg, "carrier": carrier, "row": row,
             "shape": a.shape, "vertical": vertical, "records": records}
    spans = [r["span"] for r in records]
    base = _base(scene, spans)
    anchors = [(i, q, y, x) for i, r in enumerate(records)
               for q, (y, x) in enumerate(_corners(r, row, r["span"]))]
    grouped = {}
    for yy, xx in np.argwhere(~np.isin(a, [bg, carrier, *frame_colors])):
        y, x = int(yy), int(xx)
        distances = [(y - cy) ** 2 + (x - cx) ** 2 for _, _, cy, cx in anchors]
        best = min(distances)
        assert distances.count(best) == 1, "ambiguous corner association"
        i, q, cy, cx = anchors[distances.index(best)]
        assert max(abs(y - cy), abs(x - cx)) <= 2
        dy, dx = DIRS[q]
        grouped.setdefault((i, q), []).append((int(a[y, x]), (dy * (y - cy), dx * (x - cx))))
    clues = []
    for (i, q), cells in sorted(grouped.items()):
        colors = {c for c, _ in cells}
        assert len(colors) == 1
        clue = frozenset(p for _, p in cells)
        records[i]["marks"][q] = (colors.pop(), clue)
        clues.append(clue)
    assert len(clues) >= 2
    maximal = {p for p in clues if all(other <= p for other in clues)}
    assert len(maximal) == 1, "no intact common stencil"
    stencil = maximal.pop()
    assert 3 <= len(stencil) <= 10 and _connected(stencil)
    radius = max(max(abs(u), abs(v)) for u, v in stencil)
    assert radius in (1, 2)
    scene["stencil"] = stencil
    scene["radius"] = radius
    intervals = {}
    occupied = set()
    for i, record in enumerate(records):
        lo, hi = record["span"]
        assert -lo >= radius + 2 and hi >= radius + 2
        assert record["width"] >= 2 * radius + 3
        assert record["x"] >= radius + 1
        assert record["x"] + record["width"] <= a.shape[1] - radius - 1
        length = hi - lo + 1
        assert intervals.setdefault(length, (lo, hi)) == (lo, hi), "unequal tied intervals"
        if i:
            prev = records[i - 1]
            assert record["x"] - prev["x"] - prev["width"] >= 2 * radius + 1
        for q, (color, clue) in record["marks"].items():
            cy, cx = _corners(record, row, record["span"])[q]
            dy, dx = DIRS[q]
            points = {(cy + dy * u, cx + dx * v) for u, v in stencil}
            assert all(0 <= y < a.shape[0] and 0 <= x < a.shape[1] for y, x in points)
            assert not occupied & points
            occupied |= points
            for u, v in clue:
                base[cy + dy * u, cx + dx * v] = color
    assert np.array_equal(a, base), "unexplained frame, background or accent cells"
    return scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _read(input_grid)
    remaining = [r["span"] for r in scene["records"]]
    assigned = []
    while remaining:
        span = min(remaining, key=lambda p: p[1] - p[0])
        remaining.remove(span)
        assigned.append(span)
    return _paint(scene, assigned)


def _connected(points):
    seen = {min(points)}
    todo = list(seen)
    while todo:
        u, v = todo.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                p = (u + du, v + dv)
                if p in points and p not in seen:
                    seen.add(p)
                    todo.append(p)
    return len(seen) == len(points)


def _stencil(rng, radius, asymmetric=False):
    domain = {(u, v) for u in range(-radius, radius + 1)
              for v in range(-radius, radius + 1)}
    for _ in range(80):
        size = int(rng.integers(3, min(10, len(domain)) + 1))
        points = {sorted(domain)[int(rng.integers(len(domain)))]}
        while len(points) < size:
            frontier = sorted(p for p in domain - points
                              if any(max(abs(p[0] - u), abs(p[1] - v)) == 1 for u, v in points))
            points.add(frontier[int(rng.integers(len(frontier)))])
        if (max(max(abs(u), abs(v)) for u, v in points) == radius and
                (not asymmetric or points != {(v, u) for u, v in points})):
            return frozenset(points)
    # Bounded fallback, still a connected asymmetric observable stencil.
    return frozenset({(-radius, 0), (-radius + 1, 0), (-radius + 1, 1)})


def _sample_pair(rng, vertical, test, witness, roles, allow_large_stencil):
    radius = 1 if witness == 1 else (2 if allow_large_stencil and rng.random() < .42 else 1)
    n = 3 if witness == 1 else 2 if witness == 2 or radius == 2 else int(rng.integers(2, 4))
    minimum = 2 * radius + 5
    if witness == 1:
        short = int(rng.integers(7, 12))
        lengths = [short, short + int(rng.integers(1, 3))]
        lengths.append(lengths[-1] + int(rng.integers(1, 3)))
        widths_by_length = {lengths[0]: 8, lengths[1]: 5, lengths[2]: 5}
        # A three-cycle is neither an increasing list nor its reversal.
        lengths = [lengths[1], lengths[2], lengths[0]]
        widths = [widths_by_length[k] for k in lengths]
    else:
        unique = sorted(int(v) for v in rng.choice(np.arange(minimum, 22), n, replace=False))
        if n == 3 and not witness and rng.random() < .3:
            unique[int(rng.integers(1, 3))] = unique[0]
        lengths = list(unique)
        rng.shuffle(lengths)
        if witness == 2:
            lengths = sorted(lengths, reverse=True)
        widths = [2 * radius + 3] * n
        extra = 30 - (sum(widths) + (n - 1) * (2 * radius + 1) + 2 * (radius + 1))
        for i in rng.permutation(n):
            add = int(rng.integers(0, min(4, extra) + 1))
            widths[int(i)] += add
            extra -= add
    spans_by_length = {}
    for k in sorted(set(lengths)):
        above = int(rng.integers(radius + 2, k - radius - 2))
        spans_by_length[k] = (-above, k - above - 1)
    if witness:
        # Demonstrate a moved off-center interval, not length plus recentering.
        k = max(lengths)
        if witness == 1:
            spans_by_length[k] = (-(radius + 2), k - radius - 3)
        else:
            spans_by_length[k] = (-(k - radius - 3), radius + 2)
    spans = [spans_by_length[k] for k in lengths]
    margin = radius + 1
    min_y, max_y = min(p[0] for p in spans), max(p[1] for p in spans)
    height = max_y - min_y + 1 + 2 * margin
    assert height <= 30
    height += int(rng.integers(0, min(3, 30 - height) + 1))
    row = int(rng.integers(margin - min_y, height - margin - max_y))
    gaps = [margin] + [2 * radius + 1] * (n - 1) + [margin]
    minimum_width = sum(widths) + sum(gaps)
    assert minimum_width <= 30
    width = int(rng.integers(minimum_width, 31))
    allocation = rng.multinomial(width - minimum_width, [1 / (n + 1)] * (n + 1))
    gaps = [g + int(v) for g, v in zip(gaps, allocation)]
    bg, carrier = roles if roles is not None else map(int, rng.choice(10, 2, replace=False))
    pool = [c for c in range(10) if c not in (bg, carrier)]
    rng.shuffle(pool)
    frame_palette = pool[:int(rng.integers(1, n + 1))]
    accent_pool = [c for c in pool if c not in frame_palette]
    accent_palette = accent_pool[:int(rng.integers(1, min(4, len(accent_pool)) + 1))]
    pattern = _stencil(rng, radius, asymmetric=bool(witness))
    records, x = [], gaps[0]
    for i in range(n):
        record = {"x": x, "width": widths[i], "color": int(rng.choice(frame_palette)),
                  "span": spans[i], "marks": {}}
        records.append(record)
        x += widths[i] + gaps[i + 1]
    ref = (0, 0) if witness else (int(rng.integers(n)), int(rng.integers(4)))
    partial = (0, 1) if witness else ((ref[0] + 1) % n, int(rng.integers(4)))
    for i, record in enumerate(records):
        chosen = {int(q) for q in rng.permutation(4)[:int(rng.integers(5))]}
        if i == ref[0]:
            chosen.add(ref[1])
        if i == partial[0]:
            chosen.add(partial[1])
        for q in sorted(chosen):
            full = (i, q) == ref or ((i, q) != partial and rng.random() < .18)
            count = len(pattern) if full else 1 if test else int(rng.integers(2, len(pattern)))
            cells = sorted(pattern)
            indices = rng.choice(len(cells), count, replace=False)
            clue = frozenset(cells[int(j)] for j in indices)
            record["marks"][q] = (int(rng.choice(accent_palette)), clue)
    scene = {"background": bg, "carrier": carrier, "shape": (height, width), "row": row,
             "vertical": vertical, "records": records, "stencil": pattern}
    inp = _paint(scene, spans, complete=False)
    assigned = sorted(spans, key=lambda p: p[1] - p[0])
    out = _paint(scene, assigned)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))  # H only, V only, or demonstrated mixed axes.
    axes = [False, True] if mode == 2 else [bool(mode)] * 2
    large_stencil = bool(rng.integers(2))
    roles = tuple(map(int, rng.choice(10, 2, replace=False))) if rng.random() < .35 else None
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = i + 1 if split == "train" and i < 2 else 0
            vertical = axes[i] if witness else axes[int(rng.integers(2))]
            for _ in range(64):
                try:
                    pair = _sample_pair(rng, vertical, split == "test", witness, roles, large_stencil)
                    key = _key(pair["input"])
                    assert key not in seen
                    _read(pair["input"])
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError("bounded construction budget exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    scene = _read(pair["input"])
    records = scene["records"]
    spans = [r["span"] for r in records]
    target = sorted(spans, key=lambda p: p[1] - p[0])
    widths = [r["width"] for r in records]
    areas = [spans[i] for i in sorted(range(len(records)),
             key=lambda i: widths[i] * (spans[i][1] - spans[i][0] + 1))]
    completion_only = _paint(scene, spans)
    sort_only = _paint(scene, target, complete=False)
    rotated = _paint(scene, target, rotate=True)
    return {"vertical": scene["vertical"], "shape": pair["input"].shape,
            "count": len(records), "widths": widths,
            "lengths": [hi - lo + 1 for lo, hi in spans], "spans": spans,
            "radius": scene["radius"], "stencil": tuple(sorted(scene["stencil"])),
            "marked": sum(len(r["marks"]) for r in records),
            "unmarked_slots": sum(not r["marks"] for r in records),
            "frame_colors": len({r["color"] for r in records}),
            "accents": [c for r in records for c, _ in r["marks"].values()],
            "partial_sizes": [len(p) for r in records for _, p in r["marks"].values()
                              if p != scene["stencil"]],
            "area_disagrees": areas != target, "reversal_disagrees": spans[::-1] != target,
            "moved": spans != target,
            "completion_only_wrong": not np.array_equal(completion_only, pair["output"]),
            "sort_only_wrong": not np.array_equal(sort_only, pair["output"]),
            "rotation_wrong": not np.array_equal(rotated, pair["output"])}


def validate_generated_puzzle(puzzle):
    """Relational invariants recovered from both grids; no construction metadata."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a, b = _read(inp), _read(out)
            for role in ("background", "carrier", "row", "vertical", "shape", "stencil"):
                assert a[role] == b[role]
            assert len(a["records"]) == len(b["records"])
            remaining = [r["span"] for r in a["records"]]
            previous = 0
            for src, dst in zip(a["records"], b["records"]):
                assert all(src[k] == dst[k] for k in ("x", "width", "color"))
                assert set(src["marks"]) == set(dst["marks"])
                span = dst["span"]
                length = span[1] - span[0] + 1
                assert length >= previous and span in remaining
                previous = length
                remaining.remove(span)
                for q, (color, clue) in src["marks"].items():
                    assert clue <= a["stencil"]
                    assert dst["marks"][q] == (color, a["stencil"])
            assert not remaining
            assert np.array_equal(inp[a["row"], :] if not a["vertical"] else inp[:, a["row"]],
                                  out[b["row"], :] if not b["vertical"] else out[:, b["row"]])
            f = _features({"input": inp, "output": out})
            assert f["partial_sizes"] and f["sort_only_wrong"]
            if split == "test":
                assert set(f["partial_sizes"]) == {1}
            else:
                assert min(f["partial_sizes"]) >= 2
            features[split].append(f)
    train = features["train"]
    assert {f["count"] for f in train} == {2, 3}
    assert sum(f["completion_only_wrong"] for f in train) >= 2
    assert any(f["area_disagrees"] and f["moved"] and f["reversal_disagrees"] for f in train)
    assert any(f["rotation_wrong"] for f in train)
    taught_axes = {f["vertical"] for f in train if f["moved"]}
    assert all(f["vertical"] in taught_axes for f in features["test"])


def self_test():
    """Stream 42 seeds; retain only compact feature sets and SHA256 digests."""
    signatures, shapes, stencils, spans, widths, orientations, sizes = [set() for _ in range(7)]
    counts, frames, marked, partials = set(), set(), set(), set()
    radii, repeats, ties, unmarked = set(), False, False, False
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                f = _features(pair)
                shapes.add(inp.shape)
                stencils.add(f["stencil"])
                spans.update(f["spans"])
                widths.update(f["widths"])
                orientations.add(f["vertical"])
                sizes.add(len(f["stencil"]))
                counts.add(f["count"])
                frames.add(f["frame_colors"])
                marked.add(f["marked"])
                partials.update(f["partial_sizes"])
                radii.add(f["radius"])
                repeats |= len(set(f["accents"])) < len(f["accents"])
                ties |= len(set(f["lengths"])) < f["count"]
                unmarked |= bool(f["unmarked_slots"])
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 35 and len(stencils) >= 90
    assert len(spans) >= 40 and len(widths) >= 5 and len(sizes) >= 6
    assert orientations == {False, True} and counts == {2, 3} and radii == {1, 2}
    assert repeats and ties and unmarked and len(marked) >= 6 and len(frames) >= 2
    assert 1 in partials and max(partials) >= 5
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(123 + nt * 5 + nq, nt, nq))
    for nt, nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(4,None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject unrelated edits and two concrete parent-ablation outputs.
    p = generate_puzzle(0, 2, 1)
    scene = _read(p["train"][0]["input"])
    original = p["train"][0]["output"].copy()
    alternatives = [_paint(scene, [r["span"] for r in scene["records"]]),
                    _paint(scene, sorted([r["span"] for r in scene["records"]],
                                         key=lambda z: z[1] - z[0]), complete=False)]
    mutation = original.copy()
    mutation[0, 0] = (int(mutation[0, 0]) + 1) % 10
    alternatives.append(mutation)
    for alternative in alternatives:
        assert not np.array_equal(alternative, original)
        p["train"][0]["output"] = alternative
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("invalid completion accepted")
    return {"seeds":42,"pairs":210,"shapes":len(shapes),"stencils":len(stencils),
            "spans":len(spans),"widths":sorted(widths),"stencil_sizes":sorted(sizes),
            "slot_counts":sorted(counts),"radii":sorted(radii),"repeated_accents":repeats,
            "tied_lengths":ties,"unmarked_slots":unmarked}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
