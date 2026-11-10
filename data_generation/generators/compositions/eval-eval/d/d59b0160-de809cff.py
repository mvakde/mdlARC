"""Cooperative fault stamps: query a connected union of hole neighborhoods.

Only NumPy and the standard library are used. Generation has bounded retries,
keeps one puzzle at a time, and performs no file I/O or execution tracing.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _corner(a, corner):
    if corner & 1:
        a = a[:, ::-1]
    if corner & 2:
        a = a[::-1, :]
    return a


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        points = [p]
        for r, c in points:
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    points.append(q)
        result.append(points)
    return result


def _footprint(shape, centers):
    h, w = shape
    return {(r, c) for y, x in centers
            for r in range(max(0, y-1), min(h, y+2))
            for c in range(max(0, x-1), min(w, x+2))}


def _groups(holes):
    remaining = set(holes)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group = [start]
        for r, c in group:
            near = sorted(q for q in remaining
                          if max(abs(q[0]-r), abs(q[1]-c)) <= 2)
            remaining.difference_update(near)
            group.extend(near)
        groups.append(tuple(sorted(group)))
    return groups


def _selector(grid):
    found = []
    if min(grid.shape) < 8:
        raise ValueError("Canvas too small for the selector and plates")
    for corner in range(4):
        a = _corner(grid, corner)
        frame = int(a[3, 3])
        if not (np.all(a[3, :4] == frame) and np.all(a[:4, 3] == frame)):
            continue
        colors, counts = np.unique(a[:3, :3], return_counts=True)
        padding = colors[counts == 6]
        if len(padding) != 1:
            continue
        bg = int(padding[0])
        query = frozenset(map(int, colors[colors != bg]))
        if len(query) == 3 and frame not in query and frame != bg and np.sum(grid == frame) == 7:
            found.append((corner, bg, frame, query))
    if len(found) != 1:
        raise ValueError("Expected one corner L with three distinct query labels")
    return found[0]


def _analyze(input_grid):
    """Recover plates, their clean fills, holes, labels and virtual records."""
    grid = _grid(input_grid)
    corner, bg, frame, query = _selector(grid)
    a = _corner(grid, corner)
    mask = a != bg
    mask = mask.copy()
    mask[:4, :4] = False
    plates, speckles = [], []
    for points in _components(mask):
        if len(points) <= 2:
            speckles.extend(points)
            continue
        rows, cols = zip(*points)
        box = (min(rows), max(rows)+1, min(cols), max(cols)+1)
        r0, r1, c0, c1 = box
        assert min(r1-r0, c1-c0) >= 6
        assert r0 >= 5 or c0 >= 5, "A background gap protects the selector"
        values, counts = np.unique(a[r0:r1, c0:c1], return_counts=True)
        order = np.argsort(counts)
        fill = int(values[order[-1]])
        area = (r1-r0)*(c1-c0)
        assert counts[order[-1]] >= .60*area
        assert fill not in query and fill not in (bg, frame)
        assert all(a[p] == fill for p in ((r0, c0), (r0, c1-1), (r1-1, c0), (r1-1, c1-1)))
        plates.append({"box": tuple(map(int, box)), "fill": fill, "area": area})
    assert 2 <= len(plates) <= 4
    fills = frozenset(p["fill"] for p in plates)
    assert len(fills) == 2
    owner = np.full(a.shape, -1, dtype=np.int8)
    base = np.full(a.shape, bg, dtype=np.int8)
    base[:4, :4] = a[:4, :4]
    holes, tokens, wrong = [], {}, []
    for i, plate in enumerate(plates):
        r0, r1, c0, c1 = plate["box"]
        assert np.all(owner[r0:r1, c0:c1] == -1)
        owner[r0:r1, c0:c1] = i
        base[r0:r1, c0:c1] = plate["fill"]
        for r in range(r0, r1):
            for c in range(c0, c1):
                color = int(a[r, c])
                if color == bg:
                    holes.append((r, c))
                elif color not in fills:
                    assert color != frame
                    tokens[r, c] = color
                    base[r, c] = color
                elif color != plate["fill"]:
                    wrong.append((r, c))
    assert holes
    assert all(a[p] in fills and owner[p] == -1 for p in speckles)
    groups = []
    for centers in _groups(holes):
        intended = {plates[int(owner[p])]["fill"] for p in centers}
        assert len(intended) == 1, "A connected fault group must have one intended fill"
        footprint = _footprint(a.shape, centers)
        assert all(r >= 4 or c >= 4 for r, c in footprint)
        labels = Counter(tokens[p] for p in sorted(footprint) if p in tokens)
        groups.append({"centers": centers, "footprint": footprint, "labels": labels,
                       "fill": next(iter(intended)), "active": query <= labels.keys()})
    return {"grid": a, "corner": corner, "bg": bg, "frame": frame, "query": query,
            "fills": fills, "plates": plates, "owner": owner, "base": base,
            "holes": tuple(holes), "tokens": tokens, "wrong": wrong,
            "speckles": speckles, "groups": groups}


def _stamp(base, groups, fills, highlight):
    out = base.copy()
    centers = set()
    for group in groups:
        if group["active"]:
            other = next(c for c in fills if c != group["fill"])
            for p in group["footprint"]:
                out[p] = other
            centers.update(group["centers"])
    for p in centers:
        out[p] = highlight
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The selector itself supplies the highlight; no demonstrations are read."""
    info = _analyze(input_grid)
    out = _stamp(info["base"], info["groups"], info["fills"], info["frame"])
    return _corner(out, info["corner"]).copy()


def _partition(rng, h, w, count):
    leaves = [(5, h, 0, w)]
    while len(leaves) < count:
        choices = []
        for i, (r0, r1, c0, c1) in enumerate(leaves):
            for gap in (1, 2, 3):
                for cut in range(r0+6, r1-gap-5):
                    choices.append((i, (r0, cut, c0, c1), (cut+gap, r1, c0, c1)))
                for cut in range(c0+6, c1-gap-5):
                    choices.append((i, (r0, r1, c0, cut), (r0, r1, cut+gap, c1)))
        if not choices:
            return None
        i, a, b = choices[int(rng.integers(len(choices)))]
        leaves.pop(i)
        leaves.extend((a, b))
    result = []
    for r0, r1, c0, c1 in leaves:
        dh = int(rng.integers(min(3, r1-r0-5)))
        dw = int(rng.integers(min(3, c1-c0-5)))
        top = int(rng.integers(dh+1))
        left = int(rng.integers(dw+1))
        result.append((r0+top, r1-dh+top, c0+left, c1-dw+left))
    return result


def _pattern(rng, n, kind):
    if kind == "cooperative2":
        points = [(0, 0), (0, 2)]
    elif kind == "bend3":
        points = [(0, 0), (0, 1), (1, 2)]
    else:
        points = [(0, 0)]
        for _ in range(80):
            if len(points) == n:
                break
            anchor = points[int(rng.integers(len(points)))]
            dr, dc = map(int, rng.integers(-2, 3, size=2))
            p = (anchor[0]+dr, anchor[1]+dc)
            if p not in points and max(abs(p[0]), abs(p[1])) <= 5:
                points.append(p)
        if len(points) != n:
            return None
    if rng.integers(2):
        points = [(c, r) for r, c in points]
    sy, sx = rng.choice((-1, 1), size=2)
    return [(int(r*sy), int(c*sx)) for r, c in points]


def _token_assignment(rng, available, centers, query, extras, kind, missing):
    available = list(sorted(available))
    rng.shuffle(available)
    q = list(query)
    if kind == "collective":
        coverage = {p: {i for i, c in enumerate(centers)
                        if max(abs(p[0]-c[0]), abs(p[1]-c[1])) <= 1} for p in available}
        pair = next(((a, b) for a in available for b in available
                     if a != b and coverage[a].isdisjoint(coverage[b])), None)
        if pair is None or len(available) < 3:
            return None
        a, b = pair
        middle = next(p for p in available if p not in pair)
        return {a: q[0], middle: q[1], b: q[2]}
    if kind == "exact":
        marks = q
    elif kind == "extras":
        marks = q + [extras[0], q[0], extras[0]]
    elif kind == "hard":
        kept = [c for i, c in enumerate(q) if i != missing]
        marks = kept + extras[:2] + [kept[0], extras[0]]
    elif kind == "empty":
        marks = []
    elif kind == "positive":
        marks = q + list(rng.permutation(extras)[:int(rng.integers(4))])
        if rng.random() < .5:
            marks.append(int(rng.choice(marks)))
    else:
        marks = list(rng.permutation(q)[:int(rng.integers(3))])
        marks += list(rng.permutation(extras)[:int(rng.integers(4))])
        if marks and rng.random() < .6:
            marks += [int(rng.choice(marks))]*int(rng.integers(1, 3))
    if len(available) < len(marks):
        return None
    return {p: int(c) for p, c in zip(available, rng.permutation(marks))}


def _specs(rng, package, fills):
    a, b = fills
    if package == "identity":
        return [("collective", 2, "cooperative2", a, 0), ("exact", 1, "free", b, 0)] + [
            ("hard", 1, "free", None, j) for j in range(3)] + [("empty", 3, "bend3", None, 0)]
    if package == "interaction":
        return [("collective", 3, "bend3", a, 0), ("extras", 3, "bend3", b, 0),
                ("hard", 3, "bend3", None, int(rng.integers(3))), ("empty", 1, "free", None, 0)]
    specs = [("collective", int(rng.integers(3, 6)) if package == "test" else int(rng.integers(2, 5)),
              "free", a, 0), ("hard", int(rng.integers(1, 4)), "free", b, int(rng.integers(3)))]
    for _ in range(int(rng.integers(4))):
        specs.append((str(rng.choice(("positive", "negative", "empty"))),
                      int(rng.integers(1, 6)), "free", None, 0))
    return specs


def _make_pair(rng, colors, query, package, orientation):
    bg, frame, f0, f1 = colors[:4]
    fills = (f0, f1)
    extras = [c for c in colors[4:] if c not in query]
    specs = _specs(rng, package, fills)
    for _ in range(100):
        low = 24 if package == "identity" else 20 if package == "interaction" else 16
        h, w = map(int, rng.integers(low, 31, size=2))
        boxes = _partition(rng, h, w, int(rng.integers(2, 5)))
        if boxes is None:
            continue
        rng.shuffle(boxes)
        scene = np.full((h, w), bg, dtype=np.int8)
        corners = set()
        for i, (r0, r1, c0, c1) in enumerate(boxes):
            fill = fills[i % 2] if i < 2 else int(rng.choice(fills))
            scene[r0:r1, c0:c1] = fill
            corners.update(((r0, c0), (r0, c1-1), (r1-1, c0), (r1-1, c1-1)))
        scene[3, :4] = frame
        scene[:4, 3] = frame
        for pos, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
            scene[int(pos)//3, int(pos)%3] = color
        plate_mask = np.isin(scene, fills)
        anchors = {f: [tuple(map(int, p)) for p in np.argwhere(scene == f)] for f in fills}
        holes, tokens, groups, decoys = set(), {}, [], set()
        for kind, n, pattern_kind, requested_fill, missing in specs:
            placed = False
            for _ in range(350):
                fill = requested_fill if requested_fill is not None else int(rng.choice(fills))
                anchor = anchors[fill][int(rng.integers(len(anchors[fill])))]
                pattern = _pattern(rng, n, pattern_kind)
                if pattern is None:
                    continue
                centers = tuple((anchor[0]+r, anchor[1]+c) for r, c in pattern)
                if any(not (0 <= r < h and 0 <= c < w) or scene[r, c] != fill
                       or (r, c) in corners for r, c in centers):
                    continue
                if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 2 for p in centers for q in holes):
                    continue
                footprint = _footprint(scene.shape, centers)
                if footprint & decoys:
                    continue
                available = {p for p in footprint if plate_mask[p]} - set(centers) - holes - corners
                assignment = _token_assignment(rng, available, centers, query, extras, kind, missing)
                if assignment is None:
                    continue
                outside = None
                if package == "interaction" and kind == "hard":
                    rows, cols = zip(*footprint)
                    occupied = set().union(*(g["footprint"] for g in groups))
                    candidates = [(r, c) for r in range(min(rows), max(rows)+1)
                                  for c in range(min(cols), max(cols)+1)
                                  if plate_mask[r, c] and (r, c) not in footprint | occupied | corners]
                    if not candidates:
                        continue
                    outside = candidates[int(rng.integers(len(candidates)))]
                # A future hole cannot erase an already assigned label: its
                # footprint is disjoint from every existing fault footprint.
                holes.update(centers)
                tokens.update(assignment)
                if outside is not None:
                    tokens[outside] = int(query[missing])
                    decoys.add(outside)
                groups.append({"centers": centers, "footprint": footprint,
                               "fill": fill, "active": set(query) <= set(assignment.values())})
                placed = True
                break
            if not placed:
                break
        if len(groups) != len(specs):
            continue
        occupied = set().union(*(g["footprint"] for g in groups))
        spare = sorted(set(map(tuple, np.argwhere(plate_mask))) - occupied - corners - tokens.keys())
        rng.shuffle(spare)
        for p in spare[:int(rng.integers(7))]:
            tokens[p] = int(rng.choice(colors[4:]))
        base = scene.copy()
        for p, color in tokens.items():
            base[p] = color
        inp = base.copy()
        for p in holes:
            inp[p] = bg
        # Wrong-fill errors never become centers; small isolated exterior
        # speckles do not create new plates or enter any queried footprint.
        for r0, r1, c0, c1 in boxes:
            eligible = [(r, c) for r in range(r0, r1) for c in range(c0, c1)
                        if (r, c) not in holes | tokens.keys() | corners]
            rng.shuffle(eligible)
            for p in eligible[:int(rng.integers(3))]:
                inp[p] = f1 if scene[p] == f0 else f0
        exterior = [tuple(map(int, p)) for p in np.argwhere(scene == bg)
                    if all(scene[q] == bg for q in _footprint(scene.shape, [tuple(map(int, p))]))]
        rng.shuffle(exterior)
        used = []
        speckle_count = int(rng.integers(5))
        for p in exterior:
            if len(used) >= speckle_count:
                break
            if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for q in used):
                continue
            # Keep the entire selector padding intact, even when bg is zero.
            if p[0] < 5 and p[1] < 5:
                continue
            inp[p] = int(rng.choice(fills))
            used.append(p)
        out = _stamp(base, groups, fills, frame)
        try:
            info = _analyze(inp)
            if not np.array_equal(info["base"], base):
                continue
            if {frozenset(g["centers"]) for g in info["groups"]} != {frozenset(g["centers"]) for g in groups}:
                continue
            if not np.array_equal(solve(inp), out):
                continue
        except (AssertionError, ValueError):
            continue
        def orient(a):
            a = a.T if orientation & 4 else a
            return _corner(a, orientation & 3).copy()
        return {"input": orient(inp), "output": orient(out)}
    raise RuntimeError("Bounded placement exhausted")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    change_palette = bool(rng.integers(2))
    orientations = list(map(int, rng.choice(8, 2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if change_palette:
                colors = list(map(int, rng.permutation(10)))
            query = list(map(int, rng.choice(colors[4:], 3, replace=False)))
            package = ("identity" if i == 0 else "interaction" if i == 1 else "free") if split == "train" else "test"
            orientation = orientations[i % 2] if split == "train" else int(rng.choice(orientations))
            for _ in range(5):
                pair = _make_pair(rng, colors, query, package, orientation)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Distinct-input retries exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _individual_complete(info, group):
    return any(info["query"] <= {info["tokens"][p] for p in _footprint(info["grid"].shape, [center])
                               if p in info["tokens"]} for center in group["centers"])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _analyze(inp)
            assert np.array_equal(solve(inp), out), "Output differs from input-derived rule"
            active = [g for g in info["groups"] if g["active"]]
            inactive = [g for g in info["groups"] if not g["active"]]
            assert active and inactive
            assert any(not _individual_complete(info, g) and len(g["centers"]) >= 2 for g in active)
            canonical_out = _corner(out, info["corner"])
            expected_centers = {p for g in active for p in g["centers"]}
            actual = set(map(tuple, np.argwhere(canonical_out == info["frame"])))
            actual -= set(map(tuple, np.argwhere(info["grid"] == info["frame"])))
            assert actual == expected_centers
            assert np.array_equal(canonical_out[:4, :4], info["grid"][:4, :4])
            if split == "train":
                training.append(info)
    groups = [(i, g) for i in training for g in i["groups"]]
    assert any(g["active"] and sum(g["labels"].values()) == 3 and len(g["centers"]) == 1 for i, g in groups)
    assert any(g["active"] and set(g["labels"]) - i["query"] for i, g in groups)
    assert any(not g["active"] and len(g["labels"]) >= 4 and sum(g["labels"].values()) >= 6 for i, g in groups)
    assert any(not g["labels"] for i, g in groups)
    # A missing query color in a bounding-box corner is deliberately excluded
    # from the nonrectangular union of neighborhoods.
    def box_labels(i, g):
        rows, cols = zip(*g["footprint"])
        return {color for (r, c), color in i["tokens"].items()
                if min(rows) <= r <= max(rows) and min(cols) <= c <= max(cols)}
    assert any(not g["active"] and i["query"] <= box_labels(i, g) for i, g in groups)
    assert any(all(any(not g["active"] and set(g["labels"]) & i["query"] == i["query"] - {q}
                       for g in i["groups"]) for q in i["query"]) for i in training)
    assert any({g["fill"] for g in i["groups"] if g["active"]} == i["fills"] for i in training)
    assert any(any(g["active"] and any(max(abs(a[0]-b[0]), abs(a[1]-b[1])) == 1
                                      for j, a in enumerate(g["centers"]) for b in g["centers"][:j])
                   for g in i["groups"]) for i in training)
    # The plate-level query shortcut must disagree within a single plate.
    assert any(any(len({g["active"] for g in i["groups"]
                        if any(i["owner"][p] == plate for p in g["centers"])}) == 2
                   for plate in range(len(i["plates"]))) for i in training)


def _counterfactuals(info):
    a = info["grid"]
    original = solve(a)
    # A's color identities matter: change only the selector to select a former
    # hard negative; its unchanged holes now become highlighted.
    hard = next((g for g in info["groups"] if not g["active"] and len(g["labels"]) >= 3), None)
    if hard is not None:
        replacement = sorted(hard["labels"])[:3]
        changed = a.copy()
        for p, color in zip(map(tuple, np.argwhere(a[:3, :3] != info["bg"])), replacement):
            changed[p] = color
        predicted = solve(changed)
        assert all(predicted[p] == info["frame"] for p in hard["centers"])
        assert any(original[p] != predicted[p] for p in hard["centers"])
    # Moving query pixels never changes their spatial meaning.
    moved = a.copy()
    moved[:3, :3] = np.roll(moved[:3, :3].copy(), 1, axis=0)
    expected = original.copy()
    expected[:3, :3] = moved[:3, :3]
    assert np.array_equal(solve(moved), expected)
    # Without B's grouping, collective positives are missed. Without its
    # complementary painting, both fill directions give wrong ring cells.
    assert any(g["active"] and not _individual_complete(info, g) for g in info["groups"])
    assert not np.array_equal(info["base"], original)
    # A direct per-cell stencil check uses distance, rather than _stamp.
    active = [g for g in info["groups"] if g["active"]]
    all_centers = {p for g in active for p in g["centers"]}
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            p = (r, c)
            if p in all_centers:
                expected_color = info["frame"]
            else:
                owners = [g for g in active if any(max(abs(r-y), abs(c-x)) <= 1 for y, x in g["centers"])]
                expected_color = next(f for f in info["fills"] if f != owners[0]["fill"]) if owners else info["base"][p]
            assert original[p] == expected_color


def self_test():
    """42 streaming seeds, concrete ablations, symmetry and mutation probes."""
    shapes, layouts, normalized, group_sizes, plate_counts, signatures = set(), set(), set(), set(), set(), set()
    roles, labels, features = set(), set(), Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _analyze(inp)
                digest.update(bytes(inp.shape))
                digest.update(inp.tobytes())
                digest.update(pair["output"].tobytes())
                shapes.add(inp.shape)
                layouts.add((info["grid"].shape, tuple(p["box"] for p in info["plates"])))
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                normalized.add((inp.shape, hashlib.sha256(norm).digest()))
                plate_counts.add(len(info["plates"]))
                roles.add((info["bg"], info["frame"], tuple(sorted(info["fills"]))))
                labels.add(tuple(sorted(info["query"])))
                for g in info["groups"]:
                    group_sizes.add(len(g["centers"]))
                    features["collective"] += g["active"] and not _individual_complete(info, g)
                    features["clipped"] += any(r in (0, info["grid"].shape[0]-1) or c in (0, info["grid"].shape[1]-1)
                                               for r, c in g["centers"])
                    features["cross_boundary"] += any(info["owner"][p] < 0 for p in g["footprint"])
                features["wrong"] += bool(info["wrong"])
                features["speckles"] += bool(info["speckles"])
                if split == "train":
                    _counterfactuals(info)
        signatures.add(digest.digest())
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s]) for k in p)
        first = puzzle["test"][0]
        assert np.array_equal(solve(first["input"].T), first["output"].T)
        assert np.array_equal(solve(first["input"][::-1]), first["output"][::-1])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(layouts) >= 150 and len(normalized) >= 200
    assert plate_counts == {2, 3, 4} and group_sizes >= {1, 2, 3, 4, 5}
    assert len(roles) >= 35 and len(labels) >= 40
    assert all(features[k] > 0 for k in ("clipped", "cross_boundary", "wrong", "speckles", "collective"))
    # A same-fill group may bridge separate plates across a background gap.
    # Three labels are distributed between its two distance-two centers.
    bridge = np.zeros((23, 22), dtype=np.int8)
    bridge[3, :4] = 9
    bridge[:4, 3] = 9
    bridge[0, 0], bridge[1, 1], bridge[2, 2] = 3, 4, 5
    bridge[5:13, :7] = 1
    bridge[5:13, 8:15] = 1
    bridge[15:23, 1:17] = 2
    bridge[8, 6] = bridge[8, 8] = 0
    bridge[7, 5], bridge[7, 6], bridge[9, 9] = 3, 4, 5
    bi = _analyze(bridge)
    assert len(bi["groups"]) == 1 and bi["groups"][0]["active"]
    assert not _individual_complete(bi, bi["groups"][0])
    stamped = solve(bridge.tolist())
    assert stamped[8, 6] == stamped[8, 8] == 9 and stamped[8, 7] == 2
    disconnected = bridge.copy()
    disconnected[8, 8] = 1
    assert solve(disconnected)[8, 6] == 1
    incompatible = bridge.copy()
    incompatible[5:13, 8:15][incompatible[5:13, 8:15] == 1] = 2
    try:
        solve(incompatible)
    except AssertionError:
        pass
    else:
        raise AssertionError("Mixed-fill group accepted without a conflict convention")
    permutation = np.array([7, 4, 0, 9, 1, 8, 3, 2, 6, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[bridge]), permutation[stamped])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(719, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(17, 2, 1)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Mutated answer accepted")
    return {"seeds": 42, "puzzles": len(signatures), "shapes": len(shapes),
            "layouts": len(layouts), "normalized_inputs": len(normalized),
            "group_sizes": sorted(group_sizes), "plate_counts": sorted(plate_counts),
            "role_assignments": len(roles), "query_sets": len(labels), "features": dict(features)}
