"""L-ray exclusion field: paint every empty 3x3 footprint after tracing.

No filesystem access, stored puzzles, or generator state is used by solve().
Generation and checks retain only one puzzle and bounded candidate scenes.
"""
from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np


HIGHLIGHT = 1
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (0, 254644009, 710915739, 772992814, 920725906, 993803211,
              403314491, 954237412, 807903690, 77893976, 15037740, 332524445,
              229293008, 786648311, 10503537, 943754743, 167271440, 55771520,
              956469167, 486384173, 833764706, 632439348, 926073949, 479729442,
              152189350, 305918597, 228943133, 174740640, 372474234, 295794476,
              433136032, 649273570, 795595957, 258048039, 418869130, 931954581,
              657196876, 260067532, 564955340, 504550998, 369048148, 740905468)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=True)


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] * 2 <= a.size or bg == HIGHLIGHT:
        raise ValueError("background must be a strict majority, distinct from 1")
    return bg


def _parse(a):
    """Color-aware orthogonal objects; diagonally touching objects stay separate."""
    bg = _background(a)
    if np.any(a == HIGHLIGHT):
        raise ValueError("highlight 1 is output-only")
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    sources, mirrors = [], []
    while remaining:
        origin = min(remaining)
        remaining.remove(origin)
        color = int(a[origin])
        todo, cells = [origin], set()
        while todo:
            r, c = todo.pop()
            cells.add((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and a[p] == color:
                    remaining.remove(p)
                    todo.append(p)
        rows, cols = {r for r, _ in cells}, {c for _, c in cells}
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if len(cells) == 3 and r1-r0 == c1-c0 == 1:
            missing = next(iter({(r, c) for r in rows for c in cols} - cells))
            elbow = (r0+r1-missing[0], c0+c1-missing[1])
            direction = (elbow[0]-missing[0], elbow[1]-missing[1])
            sources.append({"cells": cells, "elbow": elbow,
                            "direction": direction, "color": color})
        elif (len(cells) % 2 == 1 and (len(rows) == 1 or len(cols) == 1)
              and max(r1-r0, c1-c0)+1 == len(cells)):
            axis = None if len(cells) == 1 else (0 if len(rows) == 1 else 1)
            mirrors.append({"cells": cells, "mid": ((r0+r1)//2, (c0+c1)//2),
                            "axis": axis, "color": color})
        else:
            raise ValueError("foreground must decompose into Ls and odd straight mirrors")
    if not sources:
        raise ValueError("at least one L source is required")
    return bg, sources, mirrors


def _trace(a):
    """Trace simultaneously from input objects, never from newly painted cells."""
    bg, sources, mirrors = _parse(a)
    h, w = a.shape
    events = {}
    for index, mirror in enumerate(mirrors):
        for axis in ((0, 1) if mirror["axis"] is None else (mirror["axis"],)):
            for sign in (-1, 1):
                p = list(mirror["mid"])
                p[axis] -= sign
                p = tuple(p)
                for other in (-1, 1):
                    direction = (sign, other) if axis == 0 else (other, sign)
                    events.setdefault((p, direction), []).append((index, axis))
    field, painted, routes, used = a.copy(), {}, [], set()
    for source in sources:
        d, shade = source["direction"], source["color"]
        p = (source["elbow"][0]+d[0], source["elbow"][1]+d[1])
        states, path, contacts = set(), [], []
        while 0 <= p[0] < h and 0 <= p[1] < w:
            state = (p, d)
            if state in states:
                raise ValueError("ray loop")
            states.add(state)
            if a[p] != bg:
                raise ValueError("ray hits an occupied cell rather than a mirror face")
            hits = events.get(state, ())
            if len(hits) > 1:
                raise ValueError("ambiguous simultaneous mirror contact")
            if hits:
                index, axis = hits[0]
                shade = mirrors[index]["color"]
                d = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
                contacts.append((index, axis, p, shade, d))
                used.add(index)
            if p in painted and painted[p] != shade:
                raise ValueError("rays disagree on a shared cell's color")
            painted[p] = shade
            field[p] = shade
            path.append(p)
            p = (p[0]+d[0], p[1]+d[1])
        routes.append({"path": path, "contacts": contacts, "exit": p,
                       "final_direction": d, "final_color": shade})
    if used != set(range(len(mirrors))):
        raise ValueError("every mirror must be reached")
    return field, {"background": bg, "sources": sources, "mirrors": mirrors,
                   "routes": routes}


def _windows(empty, size=3):
    """All fitting windows, evaluated simultaneously; no clipping or repair."""
    h, w = empty.shape
    if min(h, w) < size:
        return np.zeros((0, 0), dtype=bool), np.zeros_like(empty)
    counts = np.zeros((h-size+1, w-size+1), dtype=np.int16)
    for dr in range(size):
        for dc in range(size):
            counts += empty[dr:h-size+1+dr, dc:w-size+1+dc]
    starts = counts == size*size
    union = np.zeros_like(empty)
    for dr in range(size):
        for dc in range(size):
            union[dr:h-size+1+dr, dc:w-size+1+dc] |= starts
    return starts, union


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    field, info = _trace(a)
    _, union = _windows(field == info["background"])
    answer = field.copy()
    answer[union] = HIGHLIGHT
    return answer


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r-dr, c), (r, c-dc)}


def _mirror_cells(mid, axis, length):
    return {(mid[0]+(offset if axis == 1 else 0),
             mid[1]+(offset if axis == 0 else 0))
            for offset in range(-(length//2), length//2+1)}


def _place(a, cells, color, bg):
    h, w = a.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or a[r, c] != bg:
            return None
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if p not in cells and 0 <= p[0] < h and 0 <= p[1] < w and a[p] == color:
                return None
    trial = a.copy()
    for p in cells:
        trial[p] = color
    return trial


def _add_source(a, rng, bg, palette, directions, required=None):
    h, w = a.shape
    for attempt in range(90):
        d = required if required is not None else directions[int(rng.integers(len(directions)))]
        # Edge, corner, and interior positions all occur in either split.
        r, c = int(rng.integers(h)), int(rng.integers(w))
        placement = int(rng.integers(6))
        if placement in (0, 1):
            r = 1 if d[0] == 1 else h-2
        if placement in (0, 2):
            c = 1 if d[1] == 1 else w-2
        end = (r+3*d[0], c+3*d[1])
        if not (0 <= end[0] < h and 0 <= end[1] < w):
            continue
        color = int(rng.choice(palette))
        trial = _place(a, _source_cells((r, c), d), color, bg)
        if trial is None:
            continue
        try:
            _, info = _trace(trial)
        except ValueError:
            continue
        route = next(route for src, route in zip(info["sources"], info["routes"])
                     if src["elbow"] == (r, c))
        if len(route["path"]) >= 3:
            return trial, (r, c)
    return None


def _append_mirror(a, elbow, rng, bg, palette, axis=None, color=None, length=None):
    _, before = _trace(a)
    route = next(route for src, route in zip(before["sources"], before["routes"])
                 if src["elbow"] == elbow)
    anchor = route["contacts"][-1][2] if route["contacts"] else elbow
    d = route["final_direction"]
    options = []
    for distance in range(1, 30):
        contact = (anchor[0]+distance*d[0], anchor[1]+distance*d[1])
        if not (0 <= contact[0] < a.shape[0] and 0 <= contact[1] < a.shape[1]):
            break
        for ax in ((axis,) if axis is not None else (0, 1)):
            for n in ((length,) if length is not None else (1, 3, 5, 7)):
                options.append((contact, ax, n))
    rng.shuffle(options)
    for contact, ax, n in options[:72]:
        mid = (contact[0]+(d[0] if ax == 0 else 0),
               contact[1]+(d[1] if ax == 1 else 0))
        outgoing = (-d[0], d[1]) if ax == 0 else (d[0], -d[1])
        after = (contact[0]+2*outgoing[0], contact[1]+2*outgoing[1])
        if not (0 <= after[0] < a.shape[0] and 0 <= after[1] < a.shape[1]):
            continue
        shade = color
        if shade is None:
            shade = route["final_color"] if rng.random() < .35 else int(rng.choice(palette))
        trial = _place(a, _mirror_cells(mid, ax, n), shade, bg)
        if trial is None:
            continue
        try:
            _, info = _trace(trial)
        except ValueError:
            continue
        new_route = next(rt for src, rt in zip(info["sources"], info["routes"])
                         if src["elbow"] == elbow)
        old_hits = [(before["mirrors"][i]["mid"], ax0, p)
                    for i, ax0, p, _, _ in route["contacts"]]
        new_hits = [(info["mirrors"][i]["mid"], ax0, p)
                    for i, ax0, p, _, _ in new_route["contacts"]]
        if new_hits == old_hits + [(mid, ax, contact)]:
            return trial
    return None


def _features(a):
    field, info = _trace(a)
    bg = info["background"]
    starts, union = _windows(field == bg)
    _, raw_union = _windows(a == bg)
    _, small_union = _windows(field == bg, size=2)
    _, large_union = _windows(field == bg, size=4)
    rays = (a == bg) & (field != bg)
    residual = (field == bg) & ~union
    count = int(starts.sum())
    # A's stronger necessity check excludes the ray cells themselves.
    lost_nonray = raw_union & (field == bg) & ~union
    overlap = False
    for dr in range(3):
        for dc in range(-2, 3):
            if dr == 0 and dc <= 0:
                continue
            r0, r1 = 0, starts.shape[0]-dr
            c0, c1 = max(0, -dc), min(starts.shape[1], starts.shape[1]-dc)
            if r1 > 0 and c1 > c0 and np.any(starts[r0:r1, c0:c1] & starts[dr:r1+dr, c0+dc:c1+dc]):
                overlap = True
    info.update({"field": field, "union": union, "starts": starts,
                 "rays": rays, "residual": residual, "window_count": count,
                 "union_area": int(union.sum()), "overlap": overlap,
                 "lost_nonray": int(lost_nonray.sum()),
                 "wrong_2x2": bool(np.any(small_union != union)),
                 "wrong_4x4": bool(np.any(large_union != union))})
    return info


def _useful(info):
    return (info["window_count"] >= 2 and info["union_area"] > 9
            and info["overlap"] and info["lost_nonray"] >= 2
            and info["wrong_2x2"] and int(info["rays"].sum()) >= 3)


def _make_pair(rng, bg, palette, directions, primary, teaching=None):
    """One shared geometric constructor, with bounded evidence-based selection."""
    for attempt in range(80):
        h, w = int(rng.integers(10, 31)), int(rng.integers(10, 31))
        # Larger minimum only for a required bar contact, not for test status.
        if teaching is not None:
            h, w = max(h, 14), max(w, 14)
        a = np.full((h, w), bg, dtype=np.int8)
        added = _add_source(a, rng, bg, palette, directions, primary)
        if added is None:
            continue
        a, elbow = added
        if teaching is not None:
            _, info = _trace(a)
            src_color = info["sources"][0]["color"]
            shade = src_color if teaching["unchanged"] else int(rng.choice([c for c in palette if c != src_color]))
            trial = _append_mirror(a, elbow, rng, bg, palette, teaching["axis"], shade,
                                   int(rng.choice((3, 5, 7))))
            if trial is None:
                continue
            a = trial
            if teaching["unchanged"]:
                # A point mirror can occur in any test. Show its face rule in
                # the minimum two-example training set rather than assuming it.
                trial = _append_mirror(a, elbow, rng, bg, palette, length=1)
                if trial is None:
                    continue
                a = trial
            desired = int(rng.integers(0, 3))
        else:
            desired = int(rng.choice((0, 1, 2, 3, 4), p=(.22, .29, .25, .17, .07)))
        for _ in range(desired):
            trial = _append_mirror(a, elbow, rng, bg, palette)
            if trial is None:
                break
            a = trial
        # The second demonstration contains simultaneous sources. Other pairs
        # choose their count independently of split; counts are never targets.
        target = (1 if teaching["unchanged"] else int(rng.integers(2, 5))) if teaching else int(rng.integers(1, 7))
        for _ in range(target-1):
            added = _add_source(a, rng, bg, palette, directions)
            if added is None:
                break
            a, other_elbow = added
            for _ in range(int(rng.choice((0, 1, 2), p=(.55, .30, .15)))):
                trial = _append_mirror(a, other_elbow, rng, bg, palette)
                if trial is None:
                    break
                a = trial
        try:
            info = _features(a)
        except ValueError:
            continue
        if teaching and not teaching["unchanged"] and len(info["sources"]) < 2:
            continue
        if not _useful(info):
            continue
        if teaching and teaching["unchanged"] and info["union_area"] % 9 == 0:
            continue
        if teaching and not teaching["unchanged"] and not info["wrong_4x4"]:
            continue
        if teaching:
            transitions = _transitions(info)
            relevant = [t for t in transitions if t[2] == teaching["axis"] and t[3] > 1]
            if not any((a0 == b) == teaching["unchanged"] for a0, b, _, _ in relevant):
                continue
        out = info["field"].copy()
        out[info["union"]] = HIGHLIGHT
        return {"input": a, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def _transitions(info):
    result = []
    for src, route in zip(info["sources"], info["routes"]):
        color = src["color"]
        for i, axis, _, shade, _ in route["contacts"]:
            result.append((color, shade, axis, len(info["mirrors"][i]["cells"])))
            color = shade
    return result


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nv = _integer(num_test, "num_test", 1, 4)
    if nt <= nv:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    bg = int(rng.choice([c for c in range(10) if c != HIGHLIGHT]))
    palette = [c for c in range(10) if c not in (bg, HIGHLIGHT)]
    rng.shuffle(palette)
    palette = palette[:int(rng.integers(2, 9))]
    # Adjacent directions expose a changed coordinate of the L convention.
    d = DIRECTIONS[int(rng.integers(4))]
    second = (-d[0], d[1]) if rng.integers(2) else (d[0], -d[1])
    train_dirs = (d, second)
    novel_dirs = tuple(x for x in DIRECTIONS if x not in train_dirs)
    axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nv)):
        for index in range(count):
            teaching = ({"axis": axis if index == 0 else 1-axis,
                         "unchanged": index == 0}
                        if split == "train" and index < 2 else None)
            directions = train_dirs if split == "train" else DIRECTIONS
            primary = (train_dirs[index % 2] if split == "train"
                       else novel_dirs[index % 2])
            for _ in range(12):
                pair = _make_pair(rng, bg, palette, directions, primary, teaching)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded unique-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == out.shape
    info = _features(a)
    bg, field, union = info["background"], info["field"], info["union"]
    assert _useful(info), "pair lacks observable evidence of both mechanisms"
    # Literal invariant checks as well as a whole-grid prediction comparison.
    assert np.array_equal(out[a != bg], a[a != bg]), "original objects changed"
    assert np.array_equal(out[info["rays"]], field[info["rays"]]), "ray color changed"
    assert np.all(out[union] == HIGHLIGHT)
    assert np.array_equal(out[~union], field[~union]), "outside footprint changed"
    assert np.array_equal(out == HIGHLIGHT, union)
    assert np.all(out[info["residual"]] == bg), "narrow gaps must stay blank"
    assert np.array_equal(out, solve(a)), "solve mismatch"
    # Reconstruct B by a direct scalar scan, separate from vector window logic.
    check = np.zeros_like(union)
    for r in range(a.shape[0]-2):
        for c in range(a.shape[1]-2):
            if all(field[r+dr, c+dc] == bg for dr in range(3) for dc in range(3)):
                check[r:r+3, c:c+3] = True
    assert np.array_equal(check, union)
    assert len({tuple(p) for p in np.argwhere(info["starts"])}) == info["window_count"]
    return a, info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
    seen, backgrounds, train_info, test_info = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a, info = _check_pair(pair)
            key = _key(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            backgrounds.add(info["background"])
            (train_info if split == "train" else test_info).append(info)
    assert len(backgrounds) == 1
    transitions = [t for info in train_info for t in _transitions(info)]
    assert {axis for _, _, axis, n in transitions if n > 1} == {0, 1}
    assert any(a == b for a, b, _, _ in transitions)
    assert any(a != b for a, b, _, _ in transitions)
    assert any(n == 1 for _, _, _, n in transitions), "demonstrate point reflection"
    assert any(len(info["sources"]) == 1 for info in train_info)
    assert any(len(info["sources"]) >= 2 for info in train_info)
    train_dirs = {s["direction"] for info in train_info for s in info["sources"]}
    assert len(train_dirs) == 2
    for info in test_info:
        assert any(s["direction"] not in train_dirs for s in info["sources"])
    assert any(info["union_area"] % 9 for info in train_info), "show overlapping, non-tiled area"
    assert any(info["wrong_4x4"] for info in train_info), "distinguish 3x3 from 4x4"


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _regressions():
    # Hand-specified launch and contact cells, independent of scene generation.
    a = np.zeros((12, 13), dtype=np.int8)
    a[0, 2] = a[1, 1] = a[1, 2] = 4
    a[7, 4:11] = 6
    field, _ = _trace(a)
    expected = a.copy()
    for p in ((2, 3), (3, 4), (4, 5), (5, 6)):
        expected[p] = 4
    for p in ((6, 7), (5, 8), (4, 9), (3, 10), (2, 11), (1, 12)):
        expected[p] = 6
    assert np.array_equal(field, expected)
    for k in range(4):
        for flip in (False, True):
            x = np.rot90(a, k)
            y = np.rot90(expected, k)
            if flip:
                x, y = x[:, ::-1], y[:, ::-1]
            assert np.array_equal(_trace(x)[0], y)
            assert np.array_equal(solve(x), np.rot90(solve(a), k)[:, ::-1] if flip
                                  else np.rot90(solve(a), k))
    # Overlap is union of whole footprints: two starts, twelve cells.
    empty = np.ones((3, 4), dtype=bool)
    starts, union = _windows(empty)
    assert int(starts.sum()) == 2 and int(union.sum()) == 12
    empty[1, 1] = False
    assert not _windows(empty)[1].any()
    # No clipped 3x3s at boundaries, and no 2x3/3x2 substitutes.
    for shape in ((2, 8), (8, 2)):
        assert not _windows(np.ones(shape, dtype=bool))[1].any()
    # Different-colored Ls can share an edge without merging their identities.
    touching = np.zeros((12, 12), dtype=np.int8)
    for p in ((0, 1), (1, 0), (1, 1)):
        touching[p] = 4
    for p in ((0, 3), (1, 2), (1, 3)):
        touching[p] = 6
    expected = touching.copy()
    for r in range(2, 12):
        expected[r, r] = 4
        if r+2 < 12:
            expected[r, r+2] = 6
    assert np.array_equal(_trace(touching)[0], expected)
    # Equal-color shared cells are order independent; conflicts are illegal.
    for size, second_color in ((11, 4), (12, 6), (11, 6)):
        crossing = np.zeros((size, size), dtype=np.int8)
        for p in _source_cells((1, 1), (1, 1)):
            crossing[p] = 4
        for p in _source_cells((1, size-2), (1, -1)):
            crossing[p] = second_color
        if size == 11 and second_color == 6:
            try:
                _trace(crossing)
            except ValueError:
                pass
            else:
                raise AssertionError("conflicting ray colors accepted")
        else:
            _trace(crossing)
    # Valid but not sampled: ray completion can leave no qualifying square.
    tiny = np.zeros((3, 3), dtype=np.int8)
    tiny[0, 1] = tiny[1, 0] = tiny[1, 1] = 4
    expected = tiny.copy()
    expected[2, 2] = 4
    assert np.array_equal(solve(tiny), expected)


def self_test():
    """42 supplied seeds, one puzzle resident at a time; no files or logs."""
    shapes, structures, backgrounds, source_counts = set(), set(), set(), set()
    directions, lengths, bounce_counts, union_areas, windows, hashes = set(), set(), set(), set(), set(), set()
    repeated_axes = overlaps = point_mirrors = multicolor = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _features(a)
                shapes.add(a.shape)
                backgrounds.add(info["background"])
                source_counts.add(len(info["sources"]))
                directions.update(s["direction"] for s in info["sources"])
                lengths.update(len(m["cells"]) for m in info["mirrors"])
                union_areas.add(info["union_area"])
                windows.add(info["window_count"])
                roles = {info["background"]: 0}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, normalized))
                multicolor += len(np.unique(a)) > 2
                painted = []
                for route in info["routes"]:
                    bounce_counts.add(len(route["contacts"]))
                    axes = [hit[1] for hit in route["contacts"]]
                    repeated_axes += any(x == y for x, y in zip(axes, axes[1:]))
                    painted.extend(route["path"])
                overlaps += len(painted) > len(set(painted))
                point_mirrors += any(len(m["cells"]) == 1 for m in info["mirrors"])
    assert len(hashes) == 42
    assert len(shapes) >= 50 and len(structures) >= 180
    assert len(backgrounds) >= 7 and len(source_counts) >= 4
    assert directions == set(DIRECTIONS) and lengths >= {1, 3, 5, 7}
    assert bounce_counts >= {0, 1, 2, 3} and repeated_axes and point_mirrors and multicolor
    assert len(union_areas) >= 60 and len(windows) >= 60
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    for nt in range(2, 11):
        for nv in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(123, nt, nv)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == nv
    for nt, nv in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A wrong highlight and an erased ray must both be rejected by validation.
    sample = generate_puzzle(0)
    info = _features(sample["test"][0]["input"])
    for mask in (info["union"], info["rays"]):
        p = tuple(map(int, np.argwhere(mask)[0]))
        bad = {"train": sample["train"], "test": [dict(sample["test"][0])]}
        bad["test"][0]["output"] = sample["test"][0]["output"].copy()
        bad["test"][0]["output"][p] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("incorrect output accepted")
    _regressions()
