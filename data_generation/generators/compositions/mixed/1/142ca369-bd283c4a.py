"""Trace color-changing diagonal rays, then frequency-pack the whole canvas.

Only NumPy and the standard library are used. No parent module or saved puzzle
is imported. Route details are bounded in-memory data, never saved traces.
"""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {elbow, (r - dr, c), (r, c - dc)}


def _mirror_cells(midpoint, axis, length):
    r, c = midpoint
    return {(r, c + t) if axis == 0 else (r + t, c)
            for t in range(-(length // 2), length // 2 + 1)}


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] * 2 <= a.size:
        raise ValueError("The input background must occupy a strict majority")
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    sources, mirrors = [], []
    while remaining:
        start = min(remaining)
        shade = int(a[start])
        remaining.remove(start)
        todo, cells = [start], set()
        while todo:
            r, c = todo.pop()
            cells.add((r, c))
            for dr, dc in NEIGHBORS:
                p = (r + dr, c + dc)
                if p in remaining and a[p] == shade:
                    remaining.remove(p)
                    todo.append(p)
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if len(cells) == 3 and r1 - r0 == c1 - c0 == 1:
            missing = next(iter({(r, c) for r in rows for c in cols} - cells))
            elbow = (r0 + r1 - missing[0], c0 + c1 - missing[1])
            direction = (elbow[0] - missing[0], elbow[1] - missing[1])
            sources.append(dict(cells=cells, color=shade, elbow=elbow, direction=direction))
        elif (len(cells) % 2 == 1 and (len(rows) == 1 or len(cols) == 1)
              and max(r1 - r0, c1 - c0) + 1 == len(cells)):
            axis = None if len(cells) == 1 else (0 if len(rows) == 1 else 1)
            mirrors.append(dict(cells=cells, color=shade, axis=axis,
                                midpoint=((r0 + r1) // 2, (c0 + c1) // 2)))
        else:
            raise ValueError("Ambiguous object: expected an L or an odd straight bar")
    if not sources:
        raise ValueError("At least one L source is required")
    return bg, sources, mirrors


def _contacts(mirrors):
    # A directed contact key is (empty cell, incoming diagonal direction).
    lookup = {}
    for index, mirror in enumerate(mirrors):
        mid = mirror["midpoint"]
        for axis in ((0, 1) if mirror["axis"] is None else (mirror["axis"],)):
            for normal in (-1, 1):
                p = list(mid)
                p[axis] -= normal
                for d in DIRECTIONS:
                    if d[axis] == normal:
                        lookup.setdefault((tuple(p), d), []).append((index, axis))
    return lookup


def _trace(a, variant=None):
    """Trace on immutable input objects. Variants are validation ablations only."""
    bg, sources, mirrors = _parse(a)
    lookup = _contacts(mirrors)
    answer = a.copy()
    painted, used, routes = {}, set(), []
    strict = variant is None
    for source in sources:
        d, color = source["direction"], source["color"]
        if variant == "reverse_launch":
            d = (-d[0], -d[1])
        p = (source["elbow"][0] + d[0], source["elbow"][1] + d[1])
        visited, path, events = set(), [], []
        while _inside(p, a.shape):
            state = (p, d)
            if state in visited:
                raise ValueError("Ray loop")
            visited.add(state)
            if a[p] != bg and strict:
                raise ValueError("Ray strikes an object rather than its midpoint face")
            hits = [] if variant == "straight" else lookup.get((p, d), [])
            if len(hits) > 1:
                raise ValueError("Competing mirror contacts")
            cell_color = color
            if hits:
                index, axis = hits[0]
                used.add(index)
                old_color = color
                if variant != "no_recolor":
                    color = mirrors[index]["color"]
                cell_color = old_color if variant == "late_contact" else color
                events.append((index, axis, p, old_color, color))
                d = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
            if a[p] == bg:
                if strict and p in painted and painted[p] != cell_color:
                    raise ValueError("Conflicting colors on a shared ray cell")
                painted[p] = cell_color
                answer[p] = cell_color
                path.append((p, cell_color, d))
            p = (p[0] + d[0], p[1] + d[1])
        routes.append(dict(source=source, path=path, events=events))
    if strict and used != set(range(len(mirrors))):
        raise ValueError("Unused mirror")
    return answer, dict(background=bg, sources=sources, mirrors=mirrors,
                        routes=routes, painted=painted)


def _hist(a):
    return {int(c): int(n) for c, n in zip(*np.unique(a, return_counts=True))}


def _pack(a, require_distinct=True):
    counts = _hist(a)
    if require_distinct and len(set(counts.values())) != len(counts):
        raise ValueError("Post-trace frequencies must be distinct; ties are outside this family")
    # Secondary key only makes validation ablations deterministic when they tie.
    ranked = sorted(counts, key=lambda c: (-counts[c], c))
    stream = np.repeat(ranked, [counts[c] for c in ranked]).astype(np.int8)
    h, w = a.shape
    return stream.reshape(w, h)[:, ::-1].T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all geometry and colors from input, then repack its traced census."""
    a = _grid(input_grid)
    traced, _ = _trace(a)
    return _pack(traced)


def _place(a, cells, color, bg):
    for r, c in cells:
        if not _inside((r, c), a.shape) or a[r, c] != bg:
            return None
        for dr, dc in NEIGHBORS:
            q = (r + dr, c + dc)
            if q not in cells and _inside(q, a.shape) and a[q] == color:
                return None
    result = a.copy()
    for p in cells:
        result[p] = color
    return result


def _add_source(a, rng, palette, required_direction=None):
    bg = int(np.argmax(np.bincount(a.ravel(), minlength=10)))
    h, w = a.shape
    for _ in range(90):
        d = required_direction or DIRECTIONS[int(rng.integers(4))]
        r, c = int(rng.integers(h)), int(rng.integers(w))
        placement = rng.random()
        if placement < .18:
            r = 1 if d[0] == 1 else h - 2
            c = 1 if d[1] == 1 else w - 2
        elif placement < .48:
            if rng.integers(2):
                r = 1 if d[0] == 1 else h - 2
            else:
                c = 1 if d[1] == 1 else w - 2
        elbow = (r, c)
        if not _inside((r + 2 * d[0], c + 2 * d[1]), a.shape):
            continue
        shade = int(rng.choice(palette))
        trial = _place(a, _source_cells(elbow, d), shade, bg)
        if trial is None:
            continue
        try:
            _, info = _trace(trial)
        except ValueError:
            continue
        if any(s["elbow"] == elbow for s in info["sources"]):
            return trial, elbow
    return None


def _extend(a, elbow, rng, palette, axis=None, length=None, changed=False):
    _, info = _trace(a)
    route = next(r for r in info["routes"] if r["source"]["elbow"] == elbow)
    anchor = route["events"][-1][2] if route["events"] else elbow
    d = route["path"][-1][2]
    incoming = route["path"][-1][1]
    options = []
    for distance in range(1, 30):
        p = (anchor[0] + distance * d[0], anchor[1] + distance * d[1])
        if not _inside(p, a.shape):
            break
        for component in ((0, 1) if axis is None else (axis,)):
            options.append((p, component))
    rng.shuffle(options)
    old_events = [(info["mirrors"][i]["midpoint"], axis0, p)
                  for i, axis0, p, _, _ in route["events"]]
    for p, component in options[:50]:
        mid = list(p)
        mid[component] += d[component]
        n = int(rng.choice((1, 3, 5, 7, 9))) if length is None else length
        colors = [c for c in palette if not changed or c != incoming]
        if not colors:
            return None
        shade = int(rng.choice(colors))
        trial = _place(a, _mirror_cells(tuple(mid), component, n), shade, info["background"])
        if trial is None:
            continue
        try:
            _, after = _trace(trial)
        except ValueError:
            continue
        new_route = next(r for r in after["routes"] if r["source"]["elbow"] == elbow)
        new_events = [(after["mirrors"][i]["midpoint"], axis0, q)
                      for i, axis0, q, _, _ in new_route["events"]]
        if (new_events == old_events + [(tuple(mid), component, p)]
                and len(new_route["path"]) >= 4
                and new_route["path"][-1][0] != p):
            return trial
    return None


def _features(a):
    traced, info = _trace(a)
    events = [e for r in info["routes"] for e in r["events"]]
    hist = _hist(traced)
    ordered = sorted(hist, key=lambda c: (-hist[c], c))
    boundaries = np.cumsum([hist[c] for c in ordered])[:-1]
    return dict(info=info, traced=traced, hist=hist,
                axes={e[1] for e in events},
                bar_axes={e[1] for e in events if len(info["mirrors"][e[0]]["cells"]) > 1},
                directions={s["direction"] for s in info["sources"]},
                changed=any(e[3] != e[4] for e in events),
                partial=any(1 < int(b) % a.shape[0] < a.shape[0] for b in boundaries))


def _distinguishes(a, traced, variant):
    try:
        alternative, _ = _trace(a, variant)
    except ValueError:
        return False
    return _hist(alternative) != _hist(traced)


def _rank_reversal(a, traced):
    before, after = _hist(a), _hist(traced)
    return any((before[x] - before[y]) * (after[x] - after[y]) < 0
               for x in after for y in after)


def _candidate(rng, kind, teaching=None):
    h, w = int(rng.integers(9, 31)), int(rng.integers(9, 31))
    if teaching:
        h, w = int(rng.integers(13, 29)), int(rng.integers(13, 29))
    bg = int(rng.integers(10))
    available = np.array([c for c in range(10) if c != bg])
    rng.shuffle(available)
    k = int(rng.integers(2, 6)) if teaching or kind == "test" else int(rng.integers(1, 7))
    palette = available[:k].tolist()
    a = np.full((h, w), bg, dtype=np.int8)
    first = _add_source(a, rng, palette, teaching["direction"] if teaching else None)
    if first is None:
        return None
    a, elbow = first
    count = (int(rng.integers(1, 4)) if teaching else
             int(rng.choice((0, 1, 2, 3, 4), p=(.22, .28, .27, .16, .07))))
    if kind == "test":
        count = int(rng.integers(2, 5))
    if teaching and teaching["point"]:
        count = max(2, count)
    for i in range(count):
        axis = teaching["axis"] if teaching and i == 0 else None
        length = int(rng.choice((3, 5, 7))) if teaching and i == 0 else None
        if teaching and teaching["point"] and i == 1:
            length = 1
        trial = _extend(a, elbow, rng, palette, axis, length,
                        changed=bool(teaching and i == 0))
        if trial is None:
            if teaching or (kind == "test" and i < 2):
                return None
            break
        a = trial
    desired_sources = (teaching["sources"] if teaching else
                       int(rng.integers(2, 7)) if kind == "test" else
                       int(rng.choice((1, 2, 3, 4, 5, 6), p=(.27, .26, .21, .14, .08, .04))))
    for _ in range(desired_sources - 1):
        added = _add_source(a, rng, palette)
        if added is None:
            return None
        a, new_elbow = added
        bounces = 0 if teaching else int(rng.choice((0, 1, 2), p=(.60, .30, .10)))
        for _ in range(bounces):
            trial = _extend(a, new_elbow, rng, palette)
            if trial is None:
                break
            a = trial
    try:
        f = _features(a)
        out = _pack(f["traced"])
    except ValueError:
        return None
    if (not np.any(f["traced"] != a) or np.array_equal(f["traced"], out)
            or np.array_equal(_pack(a, False), out)):
        return None
    if teaching:
        if (teaching["axis"] not in f["bar_axes"] or not f["changed"] or not f["partial"]
                or (teaching["rank"] and not _rank_reversal(a, f["traced"]))
                or (teaching["point"] and not any(len(m["cells"]) == 1 for m in f["info"]["mirrors"]))
                or not all(_distinguishes(a, f["traced"], v)
                           for v in ("no_recolor", "late_contact", "straight", "reverse_launch"))):
            return None
    return {"input": a, "output": out}


def _signature(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _checked_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _checked_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    first_axis = int(rng.integers(2))
    d0 = DIRECTIONS[int(rng.integers(4))]
    d1 = DIRECTIONS[(DIRECTIONS.index(d0) + int(rng.integers(1, 4))) % 4]
    # The two mandatory demonstrations teach complementary axes/directions;
    # all other scene properties, including palettes and sizes, are local.
    teachings = [dict(axis=first_axis, direction=d0, point=False, sources=1, rank=True),
                 dict(axis=1 - first_axis, direction=d1, point=True, rank=False,
                      sources=int(rng.integers(2, 4)))]
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(ntrain + ntest):
        split = "train" if index < ntrain else "test"
        teaching = teachings[index] if index < 2 else None
        for _ in range(240):
            pair = _candidate(rng, split, teaching)
            if pair is not None and _signature(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("Bounded scene construction exhausted; no rule-changing fallback")
        seen.add(_signature(pair["input"]))
        puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_inventory(a, traced, info):
    # Check a census from unique ray cells, including shared cells only once.
    expected = Counter(_hist(a))
    expected[info["background"]] -= len(info["painted"])
    expected.update(info["painted"].values())
    assert {c: n for c, n in expected.items() if n} == _hist(traced)
    foreground = a != info["background"]
    assert np.array_equal(traced[foreground], a[foreground])
    assert np.count_nonzero(traced != a) == len(info["painted"])
    for route in info["routes"]:
        assert route["path"], "Every emitted ray has a visible continuation"
        p, _, d = route["path"][-1]
        assert not _inside((p[0] + d[0], p[1] + d[1]), a.shape), "Premature termination"


def _raster_alternatives(traced):
    """The seven other ordinary row/column directions, for teaching checks."""
    counts = _hist(traced)
    order = sorted(counts, key=lambda c: -counts[c])
    stream = [c for c in order for _ in range(counts[c])]
    h, w = traced.shape
    for major in (0, 1):  # 0 is row-major; 1 is column-major.
        for rev_major in (False, True):
            for rev_minor in (False, True):
                if (major, rev_major, rev_minor) == (1, False, True):
                    continue
                rows = range(h - 1, -1, -1) if (rev_major if major == 0 else rev_minor) else range(h)
                cols = range(w - 1, -1, -1) if (rev_minor if major == 0 else rev_major) else range(w)
                positions = ((r, c) for r in rows for c in cols) if major == 0 else (
                    (r, c) for c in cols for r in rows)
                result = np.empty_like(traced)
                for p, shade in zip(positions, stream):
                    result[p] = shade
                yield (major, rev_major, rev_minor), result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _checked_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_features = set(), []
    possible_rasters = None
    witnesses = {v: False for v in ("no_recolor", "late_contact", "straight", "reverse_launch")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            key = _signature(a)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            f = _features(a)
            traced, info = f["traced"], f["info"]
            _check_inventory(a, traced, info)
            assert len(set(f["hist"].values())) == len(f["hist"]), "Tied post-ray counts"
            assert _hist(out) == f["hist"], "Packed inventory differs from traced inventory"
            # A separate direct stream check avoids relying on pack's reshape.
            stream = [int(out[r, c]) for c in range(out.shape[1])
                      for r in range(out.shape[0] - 1, -1, -1)]
            order = sorted(f["hist"], key=lambda c: -f["hist"][c])
            expected = [c for c in order for _ in range(f["hist"][c])]
            assert stream == expected, "Wrong frequency order or traversal"
            assert not np.array_equal(_pack(a, False), out), "A contributes no inventory change"
            assert not np.array_equal(traced, out), "B contributes no rearrangement"
            assert not np.array_equal(a, out)
            if split == "train":
                train_features.append(f)
                matching = {key for key, alternative in _raster_alternatives(traced)
                            if np.array_equal(alternative, out)}
                possible_rasters = matching if possible_rasters is None else possible_rasters & matching
                for v in witnesses:
                    witnesses[v] |= _distinguishes(a, traced, v)
    assert set.union(*(f["bar_axes"] for f in train_features)) == {0, 1}
    assert len(set.union(*(f["directions"] for f in train_features))) >= 2
    assert any(len(m["cells"]) == 1 for f in train_features for m in f["info"]["mirrors"])
    assert any(f["changed"] for f in train_features)
    assert any(_rank_reversal(np.asarray(p["input"]), f["traced"])
               for p, f in zip(puzzle["train"], train_features)), "Training hides changed frequency ranks"
    assert not possible_rasters, "Training leaves an alternative basic raster direction possible"
    assert all(witnesses.values()), "Training leaves a targeted ray shortcut indistinguishable"
    assert any(f["partial"] for f in train_features), "Training hides the vertical raster direction"


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def _regressions():
    # Hand-counted 9x11 case: L=3, source segment=2, mirror=3,
    # mirror-colored contact/tail=5, background=86. Contact belongs to color 2.
    a = np.zeros((9, 11), dtype=np.int8)
    for p in ((0, 1), (1, 0), (1, 1)):
        a[p] = 1
    a[5, 3:6] = 2
    traced, info = _trace(a)
    assert _hist(traced) == {0: 86, 1: 5, 2: 8}
    assert traced[4, 4] == 2 and traced[3, 5] == 2
    assert traced[8, 8] == 0
    expected = np.zeros_like(a)
    for j in range(86, 94):
        expected[8 - j % 9, j // 9] = 2
    for j in range(94, 99):
        expected[8 - j % 9, j // 9] = 1
    assert np.array_equal(solve(a), expected)
    # Histories rotate with the scene; the raster always uses the world axes.
    for k in range(4):
        rotated = np.rot90(a, k).copy()
        assert _hist(_trace(rotated)[0]) == _hist(traced)
        assert np.array_equal(solve(rotated), _pack(np.rot90(traced, k)))
    # Different-colored touching Ls are separate objects, not one mixed object.
    b = np.zeros((12, 14), dtype=np.int8)
    for p in _source_cells((1, 1), (1, 1)):
        b[p] = 1
    for p in _source_cells((1, 3), (1, 1)):
        b[p] = 2
    assert len(_parse(b)[1]) == 2
    _trace(b)
    # Shared same-color ray cells count once; incompatible shared colors reject.
    c = np.zeros((11, 11), dtype=np.int8)
    for p in _source_cells((1, 1), (1, 1)):
        c[p] = 1
    for p in _source_cells((1, 9), (1, -1)):
        c[p] = 1
    ct, ci = _trace(c)
    _check_inventory(c, ct, ci)
    assert sum(len(r["path"]) for r in ci["routes"]) > len(ci["painted"])
    for p in _source_cells((1, 9), (1, -1)):
        c[p] = 2
    try:
        _trace(c)
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting ray overlap accepted")
    try:
        _pack(np.array([[0, 1], [1, 0]], dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("Tied counts accepted")


def self_test():
    """42 seeds, streamed metrics, no files and at most two retained puzzles."""
    _regressions()
    shapes, backgrounds, source_counts, lengths, directions, structures = set(), set(), set(), set(), set(), set()
    signatures, axes, bounce_counts, palette_sizes = set(), set(), set(), set()
    repeated = unchanged = rank_changes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                f = _features(a)
                info = f["info"]
                digest.update(bytes(a.shape) + a.tobytes())
                shapes.add(a.shape)
                backgrounds.add(info["background"])
                source_counts.add(len(info["sources"]))
                lengths.update(len(m["cells"]) for m in info["mirrors"])
                directions.update(f["directions"])
                axes.update(f["axes"])
                palette_sizes.add(len(f["hist"]))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, hashlib.sha256(normalized).digest()))
                original_hist = _hist(a)
                colors = list(f["hist"])
                rank_changes += any((original_hist[x] - original_hist[y]) * (f["hist"][x] - f["hist"][y]) < 0
                                    for x in colors for y in colors)
                for route in info["routes"]:
                    events = route["events"]
                    bounce_counts.add(len(events))
                    repeated += any(x[1] == y[1] for x, y in zip(events, events[1:]))
                    unchanged += any(e[3] == e[4] for e in events)
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(structures) >= 150
    assert len(shapes) >= 60 and backgrounds == set(range(10))
    assert {1, 2, 3, 4}.issubset(source_counts)
    assert {1, 3, 5, 7, 9}.issubset(lengths)
    assert directions == set(DIRECTIONS) and axes == {0, 1}
    assert {0, 1, 2, 3}.issubset(bounce_counts)
    assert len(palette_sizes) >= 4 and repeated and unchanged and rank_changes >= 20
    assert _equal(generate_puzzle(999, 2, 1), generate_puzzle(999, 2, 1))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9000 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(44, 2, 1)
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
