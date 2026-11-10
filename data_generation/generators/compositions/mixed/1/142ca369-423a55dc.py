"""Colored ray union, erased apparatus, and one global bottom-anchored shear.

No parent modules, stored examples, or generation metadata enter solve().
Experiments retain only small aggregate reports; no execution traces are saved.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np


_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _decode(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] * 2 <= a.size:
        raise ValueError("background must occupy a strict majority")
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    sources, mirrors = [], []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color, cells, stack = int(a[start]), {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _STEPS:
                q = r + dr, c + dc
                if q in remaining and a[q] == color:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        r0, c0 = map(int, np.min(list(cells), axis=0))
        r1, c1 = map(int, np.max(list(cells), axis=0))
        if len(cells) == 3 and r1 - r0 == c1 - c0 == 1:
            missing = next(iter({(r, c) for r in (r0, r1)
                                 for c in (c0, c1)} - cells))
            elbow = r0 + r1 - missing[0], c0 + c1 - missing[1]
            sources.append({"cells": cells, "color": color, "elbow": elbow,
                            "direction": (elbow[0] - missing[0], elbow[1] - missing[1])})
        elif len(cells) % 2 == 1 and (r0 == r1 or c0 == c1):
            if len(cells) != max(r1 - r0, c1 - c0) + 1:
                raise ValueError("noncontiguous reflector")
            axes = (0, 1) if len(cells) == 1 else ((0,) if r0 == r1 else (1,))
            mirrors.append({"cells": cells, "color": color, "axes": axes,
                            "mid": ((r0 + r1) // 2, (c0 + c1) // 2)})
        else:
            raise ValueError("foreground component is neither an L nor an odd bar")
    if not sources:
        raise ValueError("a source is required")
    # A directed contact is one cell outside a midpoint, approached toward it.
    events = {}
    for i, m in enumerate(mirrors):
        for axis in m["axes"]:
            for normal in (-1, 1):
                p = list(m["mid"])
                p[axis] -= normal
                for tangent in (-1, 1):
                    direction = (normal, tangent) if axis == 0 else (tangent, normal)
                    key = tuple(p), direction
                    events.setdefault(key, []).append((i, axis))
    return bg, sources, mirrors, events


def _trace(value):
    """Trace only new ray pixels; source elbows and all apparatus are excluded."""
    a = _grid(value)
    bg, sources, mirrors, events = _decode(a)
    pixels, routes, used = {}, [], set()
    for source in sources:
        dr, dc = source["direction"]
        r, c = source["elbow"]
        r, c = r + dr, c + dc
        shade, seen, path, contacts = source["color"], set(), [], []
        while 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
            state = (r, c, dr, dc)
            if state in seen:
                raise ValueError("cyclic ray")
            seen.add(state)
            if a[r, c] != bg:
                raise ValueError("ray strikes apparatus instead of a midpoint face")
            hits = events.get(((r, c), (dr, dc)), [])
            if len(hits) > 1:
                raise ValueError("ambiguous contact")
            if hits:
                index, axis = hits[0]
                shade = mirrors[index]["color"]
                used.add(index)
                contacts.append((index, axis, (r, c), shade))
                if axis == 0:
                    dr = -dr
                else:
                    dc = -dc
            p = r, c
            if p in pixels and pixels[p] != shade:
                raise ValueError("incompatible colors at a shared ray cell")
            pixels[p] = shade
            path.append((p, shade))
            r, c = r + dr, c + dc
        if len(path) < 2:
            raise ValueError("each source needs at least two trajectory cells")
        routes.append({"path": path, "contacts": contacts})
    if used != set(range(len(mirrors))):
        raise ValueError("unused reflector")
    return a, bg, sources, mirrors, pixels, routes


def _paint(shape, bg, pixels, anchor=None, rate=1, direction=-1, boundary="clip"):
    out = np.full(shape, bg, dtype=np.int8)
    if not pixels:
        return out
    anchor = max(r for r, c in pixels) if anchor is None else anchor
    for (r, c), shade in pixels.items():
        target = c + direction * rate * (anchor - r)
        if boundary == "wrap":
            target %= shape[1]
        elif boundary == "clamp":
            target = min(shape[1] - 1, max(0, target))
        if 0 <= target < shape[1]:
            out[r, target] = shade
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase apparatus; shear the colored trajectory union about its global bottom.

    Each ray starts one diagonal cell beyond its L elbow. Reflector contact
    cells already carry the reflector color. Source and reflector cells never
    enter the union or determine its anchor; disconnected rays share one anchor.
    No puzzle, seed, generated route plan, or training label is consulted.
    """
    a, bg, _, _, pixels, _ = _trace(input_grid)
    return _paint(a.shape, bg, pixels)


def _source_cells(p, d):
    r, c = p
    dr, dc = d
    return {(r, c), (r - dr, c), (r, c - dc)}


def _mirror_cells(p, axis, length):
    return {(p[0] + (k if axis == 1 else 0),
             p[1] + (k if axis == 0 else 0))
            for k in range(-(length // 2), length // 2 + 1)}


def _place(a, cells, shade, bg):
    for r, c in cells:
        if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) or a[r, c] != bg:
            return None
        for dr, dc in _STEPS:
            q = r + dr, c + dc
            if (q not in cells and 0 <= q[0] < a.shape[0]
                    and 0 <= q[1] < a.shape[1] and a[q] == shade):
                return None
    b = a.copy()
    for p in cells:
        b[p] = shade
    return b


def _extend(a, elbow, rng, palette, attempts=36):
    _, bg, sources, mirrors, _, routes = _trace(a)
    i = next(i for i, s in enumerate(sources) if s["elbow"] == elbow)
    route, source = routes[i], sources[i]
    dr, dc = source["direction"]
    for _, axis, _, _ in route["contacts"]:
        dr, dc = (-dr, dc) if axis == 0 else (dr, -dc)
    origin = route["contacts"][-1][2] if route["contacts"] else elbow
    length_to_edge = min(a.shape[0] - 1 - origin[0] if dr == 1 else origin[0],
                         a.shape[1] - 1 - origin[1] if dc == 1 else origin[1])
    if length_to_edge < 2:
        return None
    old = [(mirrors[j]["mid"], ax, p) for j, ax, p, col in route["contacts"]]
    for _ in range(attempts):
        distance = int(rng.integers(1, length_to_edge + 1))
        point = origin[0] + distance * dr, origin[1] + distance * dc
        axis = int(rng.integers(2))
        mid = point[0] + (dr if axis == 0 else 0), point[1] + (dc if axis == 1 else 0)
        length = int(rng.choice((1, 3, 3, 5, 7)))
        incoming = route["contacts"][-1][3] if route["contacts"] else source["color"]
        shade = incoming if rng.random() < 0.25 else int(rng.choice(palette))
        b = _place(a, _mirror_cells(mid, axis, length), shade, bg)
        if b is None:
            continue
        try:
            _, _, ss, mm, _, rr = _trace(b)
        except ValueError:
            continue
        new_route = rr[next(j for j, s in enumerate(ss) if s["elbow"] == elbow)]
        new = [(mm[j]["mid"], ax, p) for j, ax, p, col in new_route["contacts"]]
        if new == old + [(mid, axis, point)]:
            return b
    return None


def _add_source(a, rng, palette, attempts=70):
    bg = int(np.argmax(np.bincount(a.ravel(), minlength=10)))
    h, w = a.shape
    for _ in range(attempts):
        dr, dc = _DIRS[int(rng.integers(4))]
        if rng.random() < 0.38:
            r = 1 if dr == 1 else h - 2
            c = int(rng.integers(1, w - 1))
            if rng.random() < 0.5:
                c = 1 if dc == 1 else w - 2
                r = int(rng.integers(1, h - 1))
        else:
            r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        b = _place(a, _source_cells((r, c), (dr, dc)), int(rng.choice(palette)), bg)
        if b is None:
            continue
        try:
            _trace(b)
        except ValueError:
            continue
        return b, (r, c)
    return None


def _random_input(rng, bg, palette):
    h, w = int(rng.integers(8, 31)), int(rng.integers(8, 31))
    a = np.full((h, w), bg, dtype=np.int8)
    desired = int(rng.choice((1, 2, 3, 4, 5), p=(.25, .30, .23, .15, .07)))
    for _ in range(desired):
        added = _add_source(a, rng, palette)
        if added is None:
            break
        a, elbow = added
        for _ in range(int(rng.choice((0, 1, 2, 3, 4), p=(.30, .32, .23, .11, .04)))):
            b = _extend(a, elbow, rng, palette)
            if b is None:
                break
            a = b
    return a


def _teaching_input(rng, clipped, bg, source_color, other):
    """Two variable geometry constructions guarantee teaching at count 2/1.

    Their roles are checked after optional flips; no index changes the rule.
    """
    if not clipped:
        r, distance = int(rng.integers(1, 6)), int(rng.integers(2, 7))
        c = int(rng.integers(max(2, distance - 1), 12))
        h = int(rng.integers(max(12, r + distance + 3), 28))
        w = int(rng.integers(max(16, c + distance + 4), 31))
        objects = [(_source_cells((r, c), (1, 1)), source_color),
                   (_mirror_cells((r + distance + 1, c + distance), 0,
                                  int(rng.choice((3, 5)))), source_color)]
    else:
        h = int(rng.integers(18, 25))
        r = int(rng.integers(7, 11))
        distance = int(rng.integers(2, 5))
        u = int(rng.integers(2, min(6, 2 * (r - distance - 1))))
        c = h - 1 - r + u
        w = int(rng.integers(max(22, c + distance + 3), 31))
        lower = int(rng.integers(r + 3, h - 2))
        lower_c = int(rng.integers(2, 6))
        objects = [(_source_cells((r, c), (-1, 1)), source_color),
                   (_mirror_cells((r - distance, c + distance + 1), 1,
                                  int(rng.choice((3, 5)))), other),
                   (_source_cells((lower, lower_c), (1, 1)),
                    int(rng.choice((source_color, other))))]
    a = np.full((h, w), bg, dtype=np.int8)
    for cells, shade in objects:
        a = _place(a, cells, shade, bg)
        if a is None:
            raise ValueError("teaching geometry did not fit")
    # The shear direction does not rotate. Accept a scene flip only if it
    # continues to expose this demonstration's required observable evidence.
    options = [(bool(v), bool(x)) for v in (0, 1) for x in (0, 1)]
    rng.shuffle(options)
    for v, x in options:
        b = a[::-1] if v else a
        b = b[:, ::-1].copy() if x else b.copy()
        f = _facts(b)
        if clipped:
            if f["lost"] and f["global_differs"] and f["contact_visible"]:
                return b
        elif not f["lost"] and f["canvas_differs"] and f["contact_visible"]:
            return b
    raise ValueError("no teaching orientation retained its evidence")


def _facts(value):
    a, bg, sources, mirrors, pixels, routes = _trace(value)
    bottom = max(r for r, c in pixels)
    top = min(r for r, c in pixels)
    answer = _paint(a.shape, bg, pixels)
    independent = np.full(a.shape, bg, dtype=np.int8)
    for route in routes:
        independent_route = dict(route["path"])
        part = _paint(a.shape, bg, independent_route)
        independent[part != bg] = part[part != bg]
    contacts = [ct for route in routes for ct in route["contacts"]]
    emitted = _paint(a.shape, bg, pixels, rate=0)
    lost = sum(c - (bottom - r) < 0 for r, c in pixels)
    visible_contacts = [ct for ct in contacts if ct[2][1] - (bottom - ct[2][0]) >= 0]
    changed = unchanged = False
    for source, route in zip(sources, routes):
        color = source["color"]
        for ct in route["contacts"]:
            if ct in visible_contacts:
                changed |= color != ct[3]
                unchanged |= color == ct[3]
            color = ct[3]
    rows, cols = np.nonzero(a != bg)
    return {"bg": bg, "shape": a.shape, "sources": len(sources),
            "mirror_lengths": tuple(len(m["cells"]) for m in mirrors),
            "directions": {s["direction"] for s in sources},
            "bar_axes": {ct[1] for ct in visible_contacts if len(mirrors[ct[0]]["cells"]) > 1},
            "bounces": tuple(len(route["contacts"]) for route in routes),
            "bottom": bottom, "top": top, "pixels": len(pixels), "lost": lost,
            "retained": int(np.count_nonzero(answer != bg)),
            "contact_visible": bool(visible_contacts), "changed": changed, "unchanged": unchanged,
            "global_differs": not np.array_equal(answer, independent),
            "canvas_differs": not np.array_equal(answer, _paint(a.shape, bg, pixels, anchor=a.shape[0] - 1)),
            "input_differs": not np.array_equal(answer, _paint(a.shape, bg, pixels, anchor=int(rows.max()))),
            "unsheared_differs": not np.array_equal(answer, emitted),
            "answer": answer}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Sample distinct scenes with two compulsory, shuffled teaching witnesses.

    The witnesses expose retained/clipped shear, both reflection axes, visible
    changed/unchanged contact colors, and global versus per-ray anchoring.
    Tests use a source-count/reflection-depth combination absent from training.
    Construction retries are bounded; this module writes no files.
    """
    n = _count(num_train, "num_train", 2, 10)
    m = _count(num_test, "num_test", 1, 4)
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    bg, source_color, other = palette[:3]
    variable_bg = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    # Only the first two construction roles are compulsory; their display order
    # is shuffled. Other examples are composed from geometry, not parent pairs.
    for clipped in (False, True):
        for _ in range(64):
            try:
                a = _teaching_input(rng, clipped, bg, source_color, other)
                f = _facts(a)
            except ValueError:
                continue
            if f["retained"] >= 4 and f["unsheared_differs"] and _key(a) not in seen:
                break
        else:
            raise RuntimeError("teaching construction exhausted its bounded retries")
        seen.add(_key(a))
        puzzle["train"].append({"input": a, "output": f["answer"]})
    training_signatures = {_complexity(p["input"]) for p in puzzle["train"]}
    for split, count in (("train", n), ("test", m)):
        while len(puzzle[split]) < count:
            for attempt in range(96):
                local_bg = int(rng.integers(10)) if variable_bg else bg
                available = [c for c in range(10) if c != local_bg]
                rng.shuffle(available)
                local_palette = available[:int(rng.integers(1, 7))]
                a = _random_input(rng, local_bg, local_palette)
                try:
                    f = _facts(a)
                except ValueError:
                    continue
                signature = _complexity(a)
                if (f["retained"] < 4 or not f["unsheared_differs"] or _key(a) in seen
                        or (split == "test" and signature in training_signatures)):
                    continue
                break
            else:
                raise RuntimeError("distinct scene construction exhausted bounded retries")
            seen.add(_key(a))
            puzzle[split].append({"input": a, "output": f["answer"]})
            if split == "train":
                training_signatures.add(signature)
    rng.shuffle(puzzle["train"])
    return puzzle


def _complexity(a):
    _, _, ss, mm, _, routes = _trace(a)
    # Tests recombine a source count / reflection-depth multiset not shown in
    # training, rather than merely using a new palette or a larger rectangle.
    return len(ss), tuple(sorted(len(r["contacts"]) for r in routes))


def _alternative_outputs(a):
    a, bg, sources, mirrors, pixels, routes = _trace(a)
    b = max(r for r, c in pixels)
    apparatus = {tuple(map(int, p)): int(a[tuple(p)]) for p in np.argwhere(a != bg)}
    completed = dict(apparatus)
    completed.update(pixels)
    per_ray = np.full(a.shape, bg, dtype=np.int8)
    no_recolor, straight = {}, {}
    for source, route in zip(sources, routes):
        part = _paint(a.shape, bg, dict(route["path"]))
        per_ray[part != bg] = part[part != bg]
        no_recolor.update({p: source["color"] for p, shade in route["path"]})
        r, c = source["elbow"]
        dr, dc = source["direction"]
        r, c = r + dr, c + dc
        while 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
            straight[r, c] = source["color"]
            r, c = r + dr, c + dc
    fixed = a.copy()
    sheared = _paint(a.shape, bg, pixels)
    fixed[sheared != bg] = sheared[sheared != bg]
    return {
        "no_rays": _paint(a.shape, bg, apparatus),
        "no_shear": _paint(a.shape, bg, pixels, rate=0),
        "canvas_anchor": _paint(a.shape, bg, pixels, anchor=a.shape[0] - 1),
        "input_anchor": _paint(a.shape, bg, pixels, anchor=max(r for r, c in apparatus)),
        "top_anchor": _paint(a.shape, bg, pixels, anchor=min(r for r, c in pixels)),
        "right_shear": _paint(a.shape, bg, pixels, direction=1),
        "double_rate": _paint(a.shape, bg, pixels, rate=2),
        "wrap": _paint(a.shape, bg, pixels, boundary="wrap"),
        "clamp": _paint(a.shape, bg, pixels, boundary="clamp"),
        "include_apparatus": _paint(a.shape, bg, completed),
        "preserve_apparatus": fixed,
        "per_ray_anchor": per_ray,
        "ignore_reflectors": _paint(a.shape, bg, straight),
        "ignore_recoloring": _paint(a.shape, bg, no_recolor),
        "rigid_translation": _paint(a.shape, bg, {(r, c - (b - min(x for x, y in pixels))): v
                                                   for (r, c), v in pixels.items()}, rate=0),
    }


def validate_generated_puzzle(puzzle):
    """Check grids, rays, exact row mapping, teaching witnesses, and test novelty.

    This is an author-owned consistency check, not an independent solver or a
    proof that the demonstrations identify the rule among all possible rules.
    """
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _count(n, "num_train", 2, 10)
    _count(m, "num_test", 1, 4)
    assert n > m
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            f = _facts(a)
            facts[split].append(f)
            assert np.array_equal(out, solve(a))
            assert f["retained"] >= 4 and f["unsheared_differs"]
            # Explicit per-cell conservation, row preservation, and clipping.
            _, bg, _, _, pixels, _ = _trace(a)
            target_cells = {(r, c - (f["bottom"] - r)): col for (r, c), col in pixels.items()
                            if c >= f["bottom"] - r}
            actual_cells = {tuple(map(int, p)): int(out[tuple(p)]) for p in np.argwhere(out != bg)}
            assert target_cells == actual_cells
            assert len(pixels) == len(target_cells) + f["lost"]
            assert all(out[r, c] == color for (r, c), color in pixels.items() if r == f["bottom"])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(a)))
    train = facts["train"]
    assert any(f["lost"] for f in train) and any(not f["lost"] for f in train)
    assert any(f["global_differs"] for f in train)
    assert any(f["canvas_differs"] for f in train) and any(f["input_differs"] for f in train)
    assert set().union(*(f["bar_axes"] for f in train)) == {0, 1}
    assert any(f["changed"] for f in train) and any(f["unchanged"] for f in train)
    assert len(set().union(*(f["directions"] for f in train))) >= 2
    # The named simple alternatives must each fail at least one demonstration.
    alternatives = set(_alternative_outputs(_grid(puzzle["train"][0]["input"])))
    for pair in puzzle["train"]:
        wrong = _alternative_outputs(_grid(pair["input"]))
        alternatives = {name for name in alternatives if np.array_equal(wrong[name], pair["output"])}
    assert not alternatives, f"training permits alternatives: {sorted(alternatives)}"
    signatures = {_complexity(p["input"]) for p in puzzle["train"]}
    assert all(_complexity(p["input"]) not in signatures for p in puzzle["test"])


def _digest(puzzle):
    serial = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
              for s in ("train", "test")}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def _rule_regressions():
    # A hand-enumerated reflected ray: apparatus bottom=5, trajectory bottom=4,
    # canvas bottom=8. Contact (4,6) already has mirror color 2.
    a = np.zeros((9, 12), dtype=np.int8)
    for p in ((1, 3), (0, 3), (1, 2)):
        a[p] = 1
    a[5, 5:8] = 2
    expected = np.zeros_like(a)
    for r, c, shade in ((2, 2, 1), (3, 4, 1), (4, 6, 2),
                         (3, 6, 2), (2, 6, 2), (1, 6, 2), (0, 6, 2)):
        expected[r, c] = shade
    assert np.array_equal(solve(a), expected)
    # Two upward rays have different individual bottoms. Their global bottom
    # is row 6 although the input extends to row 8 and the canvas to row 10.
    a = np.zeros((11, 16), dtype=np.int8)
    for p in _source_cells((4, 8), (-1, 1)):
        a[p] = 3
    for p in _source_cells((7, 3), (-1, -1)):
        a[p] = 4
    expected = np.zeros_like(a)
    for r in range(4):
        expected[r, 6] = 3
    expected[6, 2], expected[5, 0] = 4, 4
    assert np.array_equal(solve(a), expected)
    # Color-aware decomposition: two differently colored Ls may share an edge.
    a = np.zeros((10, 12), dtype=np.int8)
    for p in _source_cells((1, 1), (1, 1)):
        a[p] = 1
    for p in _source_cells((1, 3), (1, 1)):
        a[p] = 2
    assert len(_trace(a)[2]) == 2
    # Shared cells with the same color are a union, not a double count.
    overlap = np.zeros((11, 11), dtype=np.int8)
    for p in _source_cells((1, 1), (1, 1)) | _source_cells((1, 9), (1, -1)):
        overlap[p] = 1
    _, _, _, _, pixels, routes = _trace(overlap)
    assert len(pixels) < sum(len(r["path"]) for r in routes)
    assert np.array_equal(solve(overlap), _paint(overlap.shape, 0, pixels))
    # Ambiguous shared pixels, invalid shapes, and unused mirrors are rejected.
    bad = np.zeros((11, 11), dtype=np.int8)
    for p in _source_cells((1, 1), (1, 1)):
        bad[p] = 1
    for p in _source_cells((1, 9), (1, -1)):
        bad[p] = 2
    unused = overlap.copy()
    unused[0, 5] = 7
    shape = overlap.copy()
    shape[0, 0] = shape[0, 1] = 1
    ambiguous = np.zeros((8, 8), dtype=np.int8)
    for p in _source_cells((1, 1), (1, 1)):
        ambiguous[p] = 1
    ambiguous[4, 3], ambiguous[3, 4] = 2, 3
    for invalid in (bad, unused, shape, ambiguous):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid ray scene accepted")


def self_test():
    """Bounded 42-seed diversity/invariant sweep, with no per-pair disk writes."""
    _rule_regressions()
    shapes, backgrounds, directions, sources, bounces, lengths = set(), set(), set(), set(), set(), set()
    structures, digests, clipping = set(), set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _digest(puzzle)
        assert digest == _digest(generate_puzzle(seed))
        digests.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                f = _facts(a)
                shapes.add(a.shape)
                backgrounds.add(f["bg"])
                directions.update(f["directions"])
                sources.add(f["sources"])
                bounces.update(f["bounces"])
                lengths.update(f["mirror_lengths"])
                clipping.add(bool(f["lost"]))
                structures.add((a.shape, (a != f["bg"]).tobytes()))
                pairs += 1
    assert len(digests) == 42 and len(shapes) >= 50 and len(structures) >= 160
    assert backgrounds == set(range(10)) and directions == set(_DIRS)
    assert sources >= {1, 2, 3, 4} and bounces >= {0, 1, 2}
    assert lengths >= {1, 3, 5} and clipping == {False, True}
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            puzzle = generate_puzzle(901, n, m)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (n, m)
            validate_generated_puzzle(puzzle)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    bad = generate_puzzle(10)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    p = tuple(np.argwhere(bad["test"][0]["output"] != _facts(bad["test"][0]["input"])["bg"])[0])
    bad["test"][0]["output"][p] = (int(bad["test"][0]["output"][p]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(digests),
            "shapes": len(shapes), "binary_structures": len(structures),
            "source_counts": sorted(sources), "reflection_counts": sorted(bounces),
            "bar_lengths": sorted(lengths), "backgrounds": sorted(backgrounds),
            "all_legal_counts": True, "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
