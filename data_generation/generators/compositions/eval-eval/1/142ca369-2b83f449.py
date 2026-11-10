"""Reflected bead queues: invisible optical paths, conserved terminal queues.

Only NumPy and the standard library are used. Generation plans are discarded;
the public solver reconstructs every route, count and color from its input.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


_DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(g, bg):
    remaining = set(map(tuple, np.argwhere(g != bg)))
    result = []
    while remaining:
        p = min(remaining)
        shade = int(g[p])
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _CARDINALS:
                q = r + dr, c + dc
                if q in remaining and g[q] == shade:
                    remaining.remove(q)
                    stack.append(q)
        result.append((shade, tuple(sorted(cells))))
    return result


def _parse(g, require_beads=True):
    values, counts = np.unique(g, return_counts=True)
    assert int(counts.max()) > g.size / 2, "uniform background needs a majority"
    bg = int(values[int(np.argmax(counts))])
    sources, mirrors, dots, dot_colors = [], [], set(), set()
    structural_colors, occupied = set(), set()
    for shade, cells in _components(g, bg):
        if len(cells) == 1:
            dots.add(cells[0])
            dot_colors.add(shade)
            continue
        rows, cols = {p[0] for p in cells}, {p[1] for p in cells}
        if len(cells) == 3 and len(rows) == len(cols) == 2:
            missing = next(iter({(r, c) for r in rows for c in cols} - set(cells)))
            elbow = min(rows) + max(rows) - missing[0], min(cols) + max(cols) - missing[1]
            direction = elbow[0] - missing[0], elbow[1] - missing[1]
            assert direction in _DIAGONALS
            sources.append(dict(cells=cells, color=shade, elbow=elbow, direction=direction))
        else:
            assert len(cells) >= 3 and len(cells) % 2 == 1
            assert len(rows) == 1 or len(cols) == 1, "mirror must be an odd straight bar"
            assert max(max(rows) - min(rows), max(cols) - min(cols)) + 1 == len(cells)
            axis = 0 if len(rows) == 1 else 1
            midpoint = (min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2
            mirrors.append(dict(cells=cells, color=shade, midpoint=midpoint, axis=axis))
        structural_colors.add(shade)
        occupied.update(cells)
    assert sources
    assert len(dot_colors) <= 1 and not dot_colors & structural_colors
    if require_beads:
        assert dots and len(dot_colors) == 1, "isolated beads share one separate color"
    return dict(background=bg, bead_color=next(iter(dot_colors), None), beads=dots,
                sources=sources, mirrors=mirrors, occupied=occupied)


def _trace(g, require_beads=True):
    """Decode directed, cell-disjoint optical tracks, including their color field."""
    info = _parse(g, require_beads)
    routes, used, owner = [], set(), {}
    for i, source in enumerate(info["sources"]):
        dr, dc = source["direction"]
        r, c = source["elbow"]
        p, shade = (r + dr, c + dc), source["color"]
        cells, colors, contacts, local = [], [], [], set()
        while _inside(p, g.shape):
            assert p not in info["occupied"], "ray strikes an object instead of a midpoint face"
            assert p not in local, "a track cannot revisit a cell"
            assert p not in owner, "tracks must not share a cell"
            local.add(p)
            owner[p] = i
            hits = []
            for j, mirror in enumerate(info["mirrors"]):
                mr, mc = mirror["midpoint"]
                a = mirror["axis"]
                if ((a == 0 and p[1] == mc and mr - p[0] == dr)
                        or (a == 1 and p[0] == mr and mc - p[1] == dc)):
                    hits.append((j, a))
            assert len(hits) <= 1, "ambiguous midpoint contact"
            if hits:
                j, axis = hits[0]
                new = info["mirrors"][j]["color"]
                contacts.append((j, axis, len(cells), shade, new))
                shade = new
                used.add(j)
                if axis == 0:
                    dr = -dr
                else:
                    dc = -dc
            cells.append(p)
            colors.append(shade)
            p = p[0] + dr, p[1] + dc
        assert cells, "source must launch into the canvas"
        beads = set(cells) & info["beads"]
        routes.append(dict(cells=cells, colors=colors, contacts=contacts,
                           beads=beads, direction=(dr, dc)))
    assert used == set(range(len(info["mirrors"]))), "all mirrors participate"
    assert info["beads"] <= set(owner), "each bead must lie on exactly one track"
    info.update(routes=routes, owner=owner)
    return info


def _paint(g, info):
    out = g.copy()
    for p in info["beads"]:
        out[p] = info["background"]
    for route in info["routes"]:
        k = len(route["beads"])
        if k:
            for p, shade in zip(route["cells"][-k:], route["colors"][-k:]):
                out[p] = shade
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pack each transparent bead population at its own reflected ray's exit."""
    g = _grid(input_grid)
    return _paint(g, _trace(g))


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc)}


def _put(g, cells, shade, bg):
    for r, c in cells:
        if not _inside((r, c), g.shape) or g[r, c] != bg:
            return None
        for dr, dc in _CARDINALS:
            q = r + dr, c + dc
            if q not in cells and _inside(q, g.shape) and g[q] == shade:
                return None
    result = g.copy()
    for p in cells:
        result[p] = shade
    return result


def _launch(g, rng, palette, bg):
    for _ in range(160):
        direction = _DIAGONALS[int(rng.integers(4))]
        elbow = int(rng.integers(g.shape[0])), int(rng.integers(g.shape[1]))
        far = elbow[0] + 5 * direction[0], elbow[1] + 5 * direction[1]
        if not _inside(far, g.shape):
            continue
        candidate = _put(g, _source_cells(elbow, direction), int(rng.choice(palette)), bg)
        if candidate is None:
            continue
        try:
            _trace(candidate, False)
        except AssertionError:
            continue
        return candidate, elbow
    return None


def _route_at(info, elbow):
    return next(route for source, route in zip(info["sources"], info["routes"])
                if source["elbow"] == elbow)


def _extend(g, elbow, rng, palette, bg, axis=None, changed=False, short_tail=False):
    before = _trace(g, False)
    route = _route_at(before, elbow)
    start = route["contacts"][-1][2] + 1 if route["contacts"] else 0
    tail = route["cells"][start:]
    dr, dc = route["direction"]
    options = [(p, a, n) for p in tail[1:] for a in ([axis] if axis is not None else (0, 1))
               for n in (3, 5, 7)]
    rng.shuffle(options)
    old_contacts = [(before["mirrors"][j]["midpoint"], a, route["cells"][t])
                    for j, a, t, _, _ in route["contacts"]]
    for contact, a, length in options[:120]:
        mid = contact[0] + (dr if a == 0 else 0), contact[1] + (dc if a == 1 else 0)
        radius = length // 2
        cells = {(mid[0] + (v if a == 1 else 0), mid[1] + (v if a == 0 else 0))
                 for v in range(-radius, radius + 1)}
        available = [x for x in palette if not changed or x != route["colors"][-1]]
        shade = int(rng.choice(available))
        candidate = _put(g, cells, shade, bg)
        if candidate is None:
            continue
        try:
            after = _trace(candidate, False)
        except AssertionError:
            continue
        new = _route_at(after, elbow)
        events = [(after["mirrors"][j]["midpoint"], ax, new["cells"][t])
                  for j, ax, t, _, _ in new["contacts"]]
        if events != old_contacts + [(mid, a, contact)]:
            continue
        if short_tail and (not 3 <= len(new["cells"]) - new["contacts"][-1][2] <= 9
                           or len(new["cells"]) < 9):
            continue
        # An appended mirror may not silently reroute an already built queue.
        if any(_route_at(after, s["elbow"])["cells"] != other["cells"]
               for s, other in zip(before["sources"], before["routes"]) if s["elbow"] != elbow):
            continue
        return candidate
    return None


def _choose_positions(rng, routes, counts, primary, teaching):
    """Sample isolated markers, with evidence before and beyond a taught bend."""
    for _ in range(24):
        chosen, ok = set(), True
        for i, (route, k) in enumerate(zip(routes, counts)):
            if not k:
                continue
            cells = route["cells"]
            mandatory = []
            if i == primary and teaching:
                first, last = route["contacts"][0][2], route["contacts"][-1][2]
                if first > 0:
                    mandatory.append(cells[int(rng.integers(first))])
                if last + 1 < len(cells):
                    mandatory.append(cells[int(rng.integers(last + 1, len(cells)))])
            pool = list(cells)
            rng.shuffle(pool)
            local = []
            for p in mandatory + pool:
                if p in chosen or p in local:
                    continue
                if any((p[0] + dr, p[1] + dc) in chosen for dr, dc in _CARDINALS):
                    continue
                chosen.add(p)
                local.append(p)
                if len(local) == k:
                    break
            if len(local) != k:
                ok = False
                break
            if not set(mandatory) <= set(local):
                ok = False
                break
            if i == primary and teaching and set(local) == set(cells[-k:]):
                ok = False
                break
        if ok:
            return chosen
    return None


def _make_pair(rng, kind, axis=None):
    """Bounded geometric construction; never call a parent generator."""
    teaching = kind in ("teach_one", "teach_two")
    for _ in range(100):
        bg, bead = map(int, rng.choice(10, 2, replace=False))
        available = [x for x in range(10) if x not in (bg, bead)]
        palette = list(map(int, rng.choice(available, int(rng.integers(2, 7)), replace=False)))
        h, w = map(int, rng.integers(12 if teaching else 10, 31, size=2))
        base = np.full((h, w), bg, dtype=np.int8)
        count = 1 if kind == "teach_one" else 2 if kind == "teach_two" else int(rng.integers(1, 5))
        primary_elbow = None
        failed = False
        for i in range(count):
            launched = _launch(base, rng, palette, bg)
            if launched is None:
                failed = True
                break
            base, elbow = launched
            if i == 0:
                primary_elbow = elbow
                bounces = 2 if kind == "teach_one" else 1 if kind == "teach_two" else int(rng.integers(3, 5)) if kind == "test" else int(rng.integers(3))
            else:
                bounces = 0 if kind == "teach_two" else int(rng.integers(3))
            for j in range(bounces):
                final_teaching = teaching and i == 0 and j == bounces - 1
                extended = _extend(base, elbow, rng, palette, bg,
                                   axis=axis if final_teaching else None,
                                   changed=final_teaching, short_tail=final_teaching)
                if extended is None:
                    failed = True
                    break
                base = extended
            if failed:
                break
        if failed:
            continue
        info = _trace(base, False)
        primary = next(i for i, s in enumerate(info["sources"]) if s["elbow"] == primary_elbow)
        observed = [len(r["contacts"]) for r in info["routes"]]
        if (kind != "test" and max(observed) > 2) or (kind == "test" and not 3 <= observed[primary] <= 4):
            continue
        counts = []
        for i, route in enumerate(info["routes"]):
            limit = min(12, len(route["cells"]) - 2)
            if limit < 1:
                failed = True
                break
            if i == primary and teaching:
                minimum = max(3, len(route["cells"]) - route["contacts"][-1][2] + 1)
                if minimum > limit:
                    failed = True
                    break
                k = int(rng.integers(minimum, limit + 1))
            elif kind == "teach_two":
                k = 1
            else:
                k = int(rng.integers(0, limit + 1))
                if i == primary:
                    k = max(1, k)
            counts.append(k)
        if failed:
            continue
        dots = _choose_positions(rng, info["routes"], counts, primary, teaching)
        if dots is None:
            continue
        inp = base.copy()
        for p in dots:
            inp[p] = bead
        final = _trace(inp)
        return {"input": inp, "output": _paint(inp, final)}
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    # Both mirror axes are taught explicitly; flow direction is always visible
    # in each L, so no puzzle-level gravity parameter is learned.
    first_axis = int(rng.integers(2))
    for split, n in (("train", nt), ("test", nq)):
        for i in range(n):
            kind = ("teach_one" if i == 0 else "teach_two" if i == 1 else "train") if split == "train" else "test"
            axis = first_axis if i == 0 else 1 - first_axis
            for _ in range(12):
                pair = _make_pair(rng, kind, axis)
                signature = _key(pair["input"])
                if kind == "teach_two":
                    demos = puzzle["train"] + [pair]
                    dirs = set().union(*(_features(p["input"])["directions"] for p in demos))
                    if len(dirs) < 2:
                        continue
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    info = _trace(inp)
    visible_axes, counts, wrapped = set(), [], 0
    moved = straight_differs = False
    for source, route in zip(info["sources"], info["routes"]):
        k, n = len(route["beads"]), len(route["cells"])
        counts.append(k)
        moved |= route["beads"] != set(route["cells"][-k:] if k else [])
        if k:
            dr, dc = source["direction"]
            r, c = source["elbow"]
            straight = []
            while _inside((r + dr, c + dc), inp.shape):
                r, c = r + dr, c + dc
                straight.append((r, c))
            # Grant an ablation the correct counts; removing reflection must
            # still give a different set of queue positions in training.
            straight_differs |= set(straight[-k:]) != set(route["cells"][-k:])
        for _, axis, t, old, new in route["contacts"]:
            if k and n - k < t and old != new:
                visible_axes.add(axis)
                wrapped += 1
    return dict(shape=inp.shape, background=info["background"], bead=info["bead_color"],
                sources=len(info["sources"]), directions={s["direction"] for s in info["sources"]},
                bounces=tuple(len(r["contacts"]) for r in info["routes"]),
                axes=visible_axes, counts=tuple(counts), wrapped=wrapped, moved=moved,
                straight_differs=straight_differs,
                lengths=tuple(len(m["cells"]) for m in info["mirrors"]),
                palette=tuple(map(int, np.unique(inp))))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, teaching = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _trace(inp)
            bg, bead = info["background"], info["bead_color"]
            assert np.array_equal(out, solve(inp.copy()))
            assert not np.any(out == bead)
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))) - {bead}
            for p in info["occupied"]:
                assert out[p] == inp[p], "sources and mirrors are immutable"
            queue_cells = {tuple(p) for p in np.argwhere(out != bg)} - info["occupied"]
            assert len(queue_cells) == len(info["beads"]), "particle count is conserved"
            expected = set()
            for route in info["routes"]:
                k = len(route["beads"])
                suffix = route["cells"][-k:] if k else []
                assert queue_cells & set(route["cells"]) == set(suffix)
                for p, shade in zip(suffix, route["colors"][-k:] if k else []):
                    assert out[p] == shade, "destination illumination determines color"
                expected.update(suffix)
            assert queue_cells == expected
            if split == "train":
                teaching.append(_features(inp))
    assert set().union(*(f["axes"] for f in teaching)) == {0, 1}
    assert any(f["moved"] for f in teaching)
    assert any(f["straight_differs"] for f in teaching)
    assert len(set().union(*(f["directions"] for f in teaching))) >= 2
    assert any(1 in f["counts"] and max(f["counts"]) >= 3 for f in teaching)
    assert any(max(f["bounces"]) >= 2 for f in teaching)


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def _regressions():
    # Explicit expected coordinates: seven beads fill the last seven cells,
    # including one incoming-color cell and the newly recolored contact cell.
    inp = np.zeros((12, 13), dtype=np.int8)
    inp[0, 2] = inp[1, 1] = inp[1, 2] = 1
    inp[7, 4:11] = 2
    markers = ((2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (5, 8), (4, 9))
    for p in markers:
        inp[p] = 8
    expected = inp.copy()
    expected[expected == 8] = 0
    expected[5, 6] = 1
    for p in ((6, 7), (5, 8), (4, 9), (3, 10), (2, 11), (1, 12)):
        expected[p] = 2
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(expected, rotation))
        assert np.array_equal(solve(np.fliplr(np.rot90(inp, rotation))), np.fliplr(np.rot90(expected, rotation)))
    # A second spatial arrangement with the same count must have the same result.
    redistributed = inp.copy()
    redistributed[redistributed == 8] = 0
    for p in ((2, 3), (3, 4), (4, 5), (5, 8), (3, 10), (2, 11), (1, 12)):
        redistributed[p] = 8
    assert np.array_equal(solve(redistributed), expected)
    # Counterfactual B omission: coloring particles in place fails explicitly.
    in_place = inp.copy()
    for p in markers:
        in_place[p] = 2 if p in ((6, 7), (5, 8), (4, 9)) else 1
    assert not np.array_equal(in_place, expected)
    # Counterfactual A omission: the unreflected route exits at (11,12),
    # whereas the correct queue exits at (1,12).
    assert expected[1, 12] == 2 and expected[11, 12] == 0


def self_test():
    """42 seeds, streamed; no sample corpus, trace files or unbounded retries."""
    _regressions()
    signatures, shapes, structures, backgrounds, beads = set(), set(), set(), set(), set()
    source_counts, bounce_counts, counts, lengths, directions = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                f = _features(inp)
                shapes.add(inp.shape)
                backgrounds.add(f["background"])
                beads.add(f["bead"])
                source_counts.add(f["sources"])
                bounce_counts.update(f["bounces"])
                counts.update(f["counts"])
                lengths.update(f["lengths"])
                directions.update(f["directions"])
                mapping = {f["background"]: 0}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add((inp.shape, normalized))
                assert np.array_equal(solve(inp.copy()), pair["output"])
        assert max(len(r["contacts"]) for p in puzzle["train"] for r in _trace(p["input"])["routes"]) <= 2
        assert all(max(len(r["contacts"]) for r in _trace(p["input"])["routes"]) >= 3 for p in puzzle["test"])
    assert len(signatures) == 42 and len(structures) >= 180 and len(shapes) >= 70
    assert backgrounds == beads == set(range(10))
    assert source_counts == {1, 2, 3, 4}
    assert bounce_counts >= {0, 1, 2, 3, 4} and counts >= {0, 1, 2, 3, 4, 5, 6}
    assert lengths == {3, 5, 7} and directions == set(_DIAGONALS)
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    for nt, nq in ((2, 1), (10, 4), (10, 1), (6, 3)):
        p = generate_puzzle(9700 + nt + nq, nt, nq)
        assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(33)
    info = _trace(bad["test"][0]["input"])
    route = next(r for r in info["routes"] if r["beads"])
    bad["test"][0]["output"][route["cells"][-1]] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing packed bead accepted")
    return {"seeds": 42, "puzzles": len(signatures), "shapes": len(shapes),
            "normalized_structures": len(structures), "source_counts": sorted(source_counts),
            "reflection_counts": sorted(bounce_counts), "bead_counts_per_track": sorted(counts),
            "mirror_lengths": sorted(lengths)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
