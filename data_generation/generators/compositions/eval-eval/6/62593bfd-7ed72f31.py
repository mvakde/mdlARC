"""Corroborated symmetry: match proposed wholes, then materialize supported ink.

Only Python's standard library and NumPy are used. No parent code or evidence is
loaded. Geometry, ownership, and class membership are reconstructed from input.
"""
from collections import Counter
import json

import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 993250583, 142994016, 495006386, 972483583, 18216921,
              758756037, 472014570, 786757014, 512708268, 103216079,
              448614442, 575384575, 9170509, 827933098, 422921222,
              668783419, 538875210, 526691832, 816665191, 150003765,
              342584598, 450696214, 701197862, 807504457, 253366144,
              119360775, 897201593, 527771499, 520919310, 639847764,
              268470857, 914483458, 670189505, 144581743, 737270048,
              419553492, 370443410, 337087727, 995200238, 512340341,
              387406709)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    directions = DIR8 if diagonal else ((1, 0), (-1, 0), (0, 1), (0, -1))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _normal(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _canonical(cells):
    cells, keys = set(cells), []
    for _ in range(4):
        keys.append(_normal(cells))
        cells = {(-c, r) for r, c in cells}
    return min(keys)


def _mirror_key(cells):
    return _canonical({(r, -c) for r, c in cells})


def _kind(marker):
    if len(marker) == 1:
        return "P"
    rows, cols = {r for r, c in marker}, {c for r, c in marker}
    if len(rows) == 1 and max(cols) - min(cols) + 1 == len(marker):
        return "H"
    if len(cols) == 1 and max(rows) - min(rows) + 1 == len(marker):
        return "V"
    return None


def _map(source, marker):
    r0, c0 = min(marker)
    kind = _kind(marker)
    if kind == "H":
        return {(2 * r0 - r, c) for r, c in source}
    if kind == "V":
        return {(r, 2 * c0 - c) for r, c in source}
    if kind == "P":
        return {(2 * r0 - r, 2 * c0 - c) for r, c in source}
    raise ValueError("Invalid symmetry marker")


def _eligible(source, marker, grid, background):
    if min(max(abs(r - a), abs(c - b))
           for r, c in source for a, b in marker) != 1:
        return None
    kind = _kind(marker)
    ar, ac = min(marker)
    if kind == "H" and not (all(r < ar for r, c in source)
                              or all(r > ar for r, c in source)):
        return None
    if kind == "V" and not (all(c < ac for r, c in source)
                              or all(c > ac for r, c in source)):
        return None
    if kind == "P":
        # Centers on or beyond the bbox boundary; not buried inside a payload.
        r0, r1 = min(r for r, c in source), max(r for r, c in source)
        c0, c1 = min(c for r, c in source), max(c for r, c in source)
        if r0 < ar < r1 and c0 < ac < c1:
            return None
    dest = _map(source, marker)
    h, w = grid.shape
    if any(not (0 <= r < h and 0 <= c < w) or grid[r, c] != background
           for r, c in dest):
        return None
    return dest


def _read(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[counts.argmax()])
    if counts.max() * 2 <= grid.size:
        raise ValueError("The background must occupy more than half the input")
    by_color = {int(c): _components(grid == c) for c in colors if c != background}
    candidates = [c for c, comps in by_color.items()
                  if all(_kind(comp) is not None for comp in comps)]
    if len(candidates) != 1:
        raise ValueError("One color must uniquely consist of straight/point markers")
    marker_color = candidates[0]
    markers = by_color[marker_color]
    payloads = [(c, cells) for c, comps in by_color.items() if c != marker_color
                for cells in comps]
    if not payloads or any(_kind(cells) is not None for c, cells in payloads):
        raise ValueError("Each payload must span at least two rows and columns")
    used, groups, copies = set(), [], set()
    for marker in markers:
        choices = []
        for i, (color, source) in enumerate(payloads):
            dest = _eligible(source, marker, grid, background)
            if dest is not None:
                choices.append((i, color, source, dest))
        if len(choices) != 1:
            raise ValueError("Every marker needs exactly one eligible adjacent payload")
        i, color, source, dest = choices[0]
        if i in used or copies & dest:
            raise ValueError("Ownership or proposed copies conflict")
        used.add(i)
        copies.update(dest)
        groups.append({"marker": marker, "source": source, "copy": dest,
                       "kind": _kind(marker), "color": color,
                       "key": _canonical(source | dest),
                       "raw_key": _canonical(source)})
    if len(used) != len(payloads):
        raise ValueError("Every payload needs one marker")
    # Disjoint one-cell halos isolate local ownership, even for unused proposals.
    occupied = set()
    for g in groups:
        footprint = g["marker"] | g["source"] | g["copy"]
        halo = {(r + dr, c + dc) for r, c in occupied
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        if footprint & halo:
            raise ValueError("Different complete groups must not touch")
        occupied.update(footprint)
    frequencies = Counter(g["key"] for g in groups)
    for g in groups:
        g["multiplicity"] = frequencies[g["key"]]
    return grid, background, marker_color, groups


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution; neither training pairs nor generation state are read."""
    grid, _, _, groups = _read(input_grid)
    result = grid.copy()
    for g in groups:
        if g["multiplicity"] >= 2:
            for q in g["copy"]:
                result[q] = g["color"]
    return result


def _mask_cells(rng, limit, chiral=False, symmetric=False, diagonal=False):
    for _ in range(180):
        h, w = int(rng.integers(2, limit + 1)), int(rng.integers(2, limit + 1))
        family = int(rng.integers(5))
        if symmetric:
            w = 2 * int(rng.integers(1, (limit + 1) // 2)) + 1
            mid = w // 2
            cells = {(r, mid) for r in range(h)}
            for r in range(h):
                reach = int(rng.integers(mid + 1))
                cells.update((r, c) for c in range(mid - reach, mid + reach + 1))
            cells.update((h - 1, c) for c in range(w))
        elif family == 0:
            cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
        elif family == 1:
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h - 1) or c in (0, w - 1)}
            if h >= 3 and w >= 3 and rng.random() < .5:
                cells.add((1, 1))
        elif family == 2:
            cells = {(r, min(w - 1, r)) for r in range(h)}
            cells |= {(h - 1, c) for c in range(w)}
        elif family == 3:
            cells = {(r, c) for r in range(h) for c in range(w)}
            a, b = int(rng.integers(1, h)), int(rng.integers(1, w))
            cells -= {(r, c) for r in range(a) for c in range(b)}
        else:
            cells = {(int(rng.integers(h)), int(rng.integers(w)))}
            target = int(rng.integers(3, h * w + 1))
            for _ in range(target - 1):
                frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in DIR8
                                   if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
                if not frontier:
                    break
                cells.add(frontier[int(rng.integers(len(frontier)))])
        cells = set(_normal(cells))
        if len({r for r, c in cells}) < 2 or len({c for r, c in cells}) < 2:
            continue
        if diagonal:
            hh, ww = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
            arr = np.zeros((hh, ww), bool)
            for q in cells:
                arr[q] = True
            if len(_components(arr, diagonal=False)) == 1:
                continue
        if symmetric:
            ww = max(c for r, c in cells) + 1
            if cells != {(r, ww - 1 - c) for r, c in cells}:
                continue
            if _normal(cells) == _normal({(-r, c) for r, c in cells}):
                continue
        elif chiral and _canonical(cells) == _mirror_key(cells):
            continue
        else:
            for _ in range(int(rng.integers(4))):
                cells = set(_normal({(-c, r) for r, c in cells}))
        return cells
    if symmetric:
        return {(0, 1), (1, 0), (1, 1), (1, 2)}
    if diagonal:
        return {(0, 0), (1, 1), (1, 2)}
    return {(0, 0), (1, 0), (2, 0), (2, 1)}


def _normalize_group(group):
    all_cells = group["marker"] | group["source"] | group["copy"]
    r0, c0 = min(r for r, c in all_cells), min(c for r, c in all_cells)
    return {k: {(r - r0, c - c0) for r, c in group[k]}
            for k in ("marker", "source", "copy")}


def _line_marker(group, rng, length=None):
    kind = _kind(group["marker"])
    ar, ac = min(group["marker"])
    contact = sorted({c if kind == "H" else r for r, c in group["source"]
                      if abs(r - ar if kind == "H" else c - ac) == 1})
    t = contact[int(rng.integers(len(contact)))]
    length = int(rng.integers(2, 6)) if length is None else length
    start = t - int(rng.integers(length))
    group["marker"] = ({(ar, c) for c in range(start, start + length)} if kind == "H"
                       else {(r, ac) for r in range(start, start + length)})
    return group


def _line_group(cells, rng):
    source = {(r + 1, c) for r, c in cells}
    group = {"marker": {(0, 0), (0, 1)}, "source": source,
             "copy": {(-r, c) for r, c in source}}
    return _line_marker(group, rng)


def _point_options(cells):
    h, w = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
    result = []
    for ar in range(-1, h + 1):
        for ac in range(-1, w + 1):
            if (ar, ac) in cells or (0 < ar < h - 1 and 0 < ac < w - 1):
                continue
            if min(max(abs(r - ar), abs(c - ac)) for r, c in cells) != 1:
                continue
            copy = {(2 * ar - r, 2 * ac - c) for r, c in cells}
            if copy & cells:
                continue
            result.append({"marker": {(ar, ac)}, "source": set(cells), "copy": copy})
    return result


def _clone(group, rng, turn=None, swap=None, marker_length=None):
    g = {k: set(v) for k, v in group.items()}
    if (bool(rng.integers(2)) if swap is None else swap):
        g["source"], g["copy"] = g["copy"], g["source"]
    for _ in range(int(rng.integers(4)) if turn is None else turn):
        g = {k: {(-c, r) for r, c in v} for k, v in g.items()}
    if _kind(g["marker"]) != "P":
        _line_marker(g, rng, marker_length)
    return _normalize_group(g)


def _class(rng, limit, count, flavor, bridge=False):
    cells = _mask_cells(rng, limit, chiral=bridge, symmetric=flavor == "X",
                        diagonal=bridge)
    if flavor == "P":
        opts = _point_options(cells)
        base = opts[int(rng.integers(len(opts)))]
    else:
        base = _line_group(cells, rng)
    groups = []
    offset = int(rng.integers(4))
    for i in range(count):
        g = base
        if flavor == "X" and i == count - 1:
            # A horizontally symmetric half admits the same whole via a center.
            center = max(c for r, c in cells) // 2
            g = {"marker": {(0, center)}, "source": base["source"], "copy": base["copy"]}
        groups.append(_clone(g, rng, turn=(offset + i) % 4 if bridge else None,
                             swap=bool(i % 2) if bridge else None,
                             marker_length=(2 if i == 0 else 4) if bridge else None))
    return groups


def _false_friends(rng, limit):
    # Same visible shape, distinct mirrored whole proposals: two shortcuts at once.
    for attempt in range(120):
        cells = (_mask_cells(rng, limit) if attempt < 119
                 else {(0, 0), (0, 1), (1, 0)})
        opts = _point_options(cells)
        keys = [_canonical(g["source"] | g["copy"]) for g in opts]
        mirrors = [_mirror_key(g["source"] | g["copy"]) for g in opts]
        choices = [(i, j) for i in range(len(opts)) for j in range(i + 1, len(opts))
                   if keys[i] != keys[j] and mirrors[i] == keys[j]]
        if choices:
            i, j = choices[int(rng.integers(len(choices)))]
            return [_clone(opts[i], rng), _clone(opts[j], rng)]
    raise AssertionError("The finite L-shape false-friend fallback must work")


def _build_groups(rng, limit, plan):
    groups, forbidden = [], set()
    for flavor, count in plan:
        for _ in range(100):
            if flavor == "F":
                new = _false_friends(rng, limit)
            else:
                f = str(rng.choice(["L", "P", "X"], p=[.45, .35, .20])) if flavor == "R" else flavor
                if flavor == "C":
                    new = _class(rng, limit, count, "L", bridge=True)
                else:
                    new = _class(rng, limit, count, f)
            keys = {_canonical(g["source"] | g["copy"]) for g in new}
            if keys & forbidden:
                continue
            forbidden.update(keys)
            groups.extend(new)
            break
        else:
            raise AssertionError("Could not sample distinct whole-shape classes")
    return groups


def _random_plan(rng, test=False):
    if test:
        mode = int(rng.integers(3))
        if mode == 0:
            return [("R", 4), ("R", int(rng.integers(1, 3)))]
        if mode == 1:
            return [("R", 2), ("R", int(rng.integers(2, 4))), ("R", 1)]
        return [("X", int(rng.integers(2, 4))), ("F", 2)]
    mode = int(rng.integers(5))
    if mode == 0:
        return [("R", 1)] * int(rng.integers(2, 7))
    if mode == 1:
        return [("R", int(rng.integers(2, 5)))] * int(rng.integers(1, 3))
    if mode == 2:
        return [("R", int(rng.integers(2, 4))), ("F", 2)]
    return [("R", int(rng.integers(2, 5)))] + [("R", 1)] * int(rng.integers(1, 4))


def _pack(groups, rng, height, width, style):
    blocked = np.zeros((height, width), dtype=bool)
    placed = []
    order = sorted(range(len(groups)), key=lambda i: -len(
        groups[i]["source"] | groups[i]["copy"] | groups[i]["marker"]))
    anchor = int(rng.integers(height)), int(rng.integers(width))
    for i in order:
        g = groups[i]
        footprint = sorted(g["source"] | g["copy"] | g["marker"])
        gh, gw = max(r for r, c in footprint) + 1, max(c for r, c in footprint) + 1
        coords = [(r, c) for r in range(height - gh + 1) for c in range(width - gw + 1)]
        rng.shuffle(coords)
        if style == "edge":
            coords.sort(key=lambda p: min(p[0], p[1], height - gh - p[0], width - gw - p[1]))
        elif style == "cluster":
            coords.sort(key=lambda p: abs(p[0] + gh / 2 - anchor[0]) + abs(p[1] + gw / 2 - anchor[1]))
        rr, cc = np.array(footprint).T
        found = next(((r, c) for r, c in coords if not blocked[rr + r, cc + c].any()), None)
        if found is None:
            return None
        r0, c0 = found
        moved = {k: {(r + r0, c + c0) for r, c in v} for k, v in g.items()}
        placed.append((i, moved))
        for r, c in footprint:
            r, c = r + r0, c + c0
            blocked[max(0, r - 1):min(height, r + 2), max(0, c - 1):min(width, c + 2)] = True
    return [g for _, g in sorted(placed)]


def _pair(rng, plan, parameters, teaching=False, compact=False):
    style = str(rng.choice(["free", "edge", "cluster"]))
    placed = None
    for attempt in range(13):
        limit = 3 if compact or attempt >= 5 else int(rng.integers(3, 6))
        groups = _build_groups(rng, limit, plan)
        if attempt == 12 or compact:
            height = width = 30
            placed = []
            # A bounded construction fallback, never the regular scene layout.
            for i, g in enumerate(groups):
                footprint = g["source"] | g["copy"] | g["marker"]
                assert max(r for r, c in footprint) < 9 and max(c for r, c in footprint) < 9
                placed.append({k: {(r + 10 * (i // 3), c + 10 * (i % 3)) for r, c in v}
                               for k, v in g.items()})
        else:
            low = 12 if attempt < 3 else 23
            height, width = int(rng.integers(low, 31)), int(rng.integers(low, 31))
            placed = _pack(groups, rng, height, width, style)
        if placed is not None:
            break
    background = parameters["background"]
    if background is None:
        background = int(rng.choice([c for c in range(10) if c != parameters["marker"]]))
    palette = [int(c) for c in rng.permutation(10) if c not in (background, parameters["marker"])]
    ncolors = int(rng.integers(1, min(8, len(placed)) + 1))
    colors = [palette[int(rng.integers(ncolors))] for _ in placed]
    if teaching:
        # First class has unlike colors; a unique proposal reuses one of them.
        colors[0], colors[1], colors[-1] = palette[0], palette[1], palette[0]
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    frequencies = Counter(_canonical(g["source"] | g["copy"]) for g in placed)
    for color, g in zip(colors, placed):
        for q in g["marker"]:
            inp[q] = out[q] = parameters["marker"]
        for q in g["source"]:
            inp[q] = out[q] = color
        if frequencies[_canonical(g["source"] | g["copy"])] > 1:
            for q in g["copy"]:
                out[q] = color
    # Dominance is needed only on input; output is allowed to be comparatively dense.
    if np.count_nonzero(inp == background) * 2 <= inp.size:
        raise AssertionError("Packing did not leave a dominant background")
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    nt, ns = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    order = [int(c) for c in rng.permutation(10)]
    parameters = {"marker": order[0], "background": order[1] if rng.random() < .25 else None}
    plans = [_random_plan(rng) for _ in range(nt)]
    a, b = [int(i) for i in rng.choice(nt, 2, replace=False)]
    plans[a], plans[b] = [("C", 2), ("F", 2)], [("X", 3), ("L", 1)]
    puzzle, seen, known = {"train": [], "test": []}, set(), set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            plan = plans[i] if split == "train" else _random_plan(rng, test=True)
            for _ in range(50):
                pair = _pair(rng, plan, parameters, teaching=split == "train" and i in (a, b))
                key = _fingerprint(pair["input"])
                _, _, _, groups = _read(pair["input"])
                wholes = {g["key"] for g in groups}
                if key in seen or (split == "test" and wholes & known):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    known.update(wholes)
                break
            else:
                raise AssertionError("Distinct novel scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(grid, groups, mode):
    keys = []
    for g in groups:
        keys.append(g["raw_key"] if mode == "raw" else
                    _normal(g["source"] | g["copy"]) if mode == "identity" else
                    _canonical(g["source"] | g["copy"] | g["marker"]) if mode == "marker" else
                    min(g["key"], _mirror_key(g["source"] | g["copy"])) if mode == "reflection" else
                    len(g["source"]) if mode == "area" else
                    g["color"] if mode == "color" else g["key"])
    freq = Counter(keys)
    out = grid.copy()
    for g, key in zip(groups, keys):
        active = (True if mode == "all" else freq[key] == 2 if mode == "exactly_two"
                  else freq[key] >= 2)
        if active:
            dest = g["copy"]
            if mode == "translation":
                sr, sc = min(r for r, c in g["source"]), min(c for r, c in g["source"])
                dr, dc = min(r for r, c in dest), min(c for r, c in dest)
                dest = {(r - sr + dr, c - sc + dc) for r, c in g["source"]}
            for q in dest:
                out[q] = g["color"]
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, markers, trained, tested, known, novel = set(), set(), set(), set(), set(), set()
    ruled_out = set()
    witnesses = {"latent_bridge": False, "false_friend": False, "cross_kind": False,
                 "point_turn": False, "line_flip": False, "triplet": False,
                 "copied_diagonal": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "Duplicate inputs"
            seen.add(key)
            _, bg, marker, groups = _read(inp)
            markers.add(marker)
            assert np.array_equal(inp[inp != bg], out[inp != bg]), "Original cells changed"
            assert np.array_equal(solve(inp), out), "Wrong corroborated completion"
            added = {tuple(q) for q in np.argwhere(inp != out).tolist()}
            expected_cells = set().union(*(g["copy"] for g in groups if g["multiplicity"] > 1))
            assert added == expected_cells, "Wrong missing-ink locations"
            for g in groups:
                active = g["multiplicity"] > 1
                assert all(out[q] == (g["color"] if active else bg) for q in g["copy"])
            if split == "test":
                tested.update(g["kind"] for g in groups)
                novel.update(g["key"] for g in groups)
                continue
            trained.update(g["kind"] for g in groups)
            known.update(g["key"] for g in groups)
            for mode in ("raw", "all", "reflection", "area", "color", "exactly_two",
                         "translation", "identity", "marker"):
                if not np.array_equal(_alternative(inp, groups, mode), out):
                    ruled_out.add(mode)
            for i, g in enumerate(groups):
                witnesses["triplet"] |= g["multiplicity"] == 3
                local = _normal(g["source"])
                mask = np.zeros((max(r for r, c in local) + 1,
                                 max(c for r, c in local) + 1), bool)
                for q in local:
                    mask[q] = True
                witnesses["copied_diagonal"] |= (g["multiplicity"] > 1
                                                   and len(_components(mask, False)) > 1)
                translated = {(r - min(a for a, b in g["source"]) + min(a for a, b in g["copy"]),
                               c - min(b for a, b in g["source"]) + min(b for a, b in g["copy"]))
                              for r, c in g["source"]}
                if g["multiplicity"] > 1 and translated != g["copy"]:
                    witnesses["point_turn" if g["kind"] == "P" else "line_flip"] = True
                for h in groups[i + 1:]:
                    witnesses["latent_bridge"] |= g["key"] == h["key"] and g["raw_key"] != h["raw_key"]
                    witnesses["false_friend"] |= (g["raw_key"] == h["raw_key"] and g["key"] != h["key"]
                                                   and g["multiplicity"] == h["multiplicity"] == 1)
                    witnesses["cross_kind"] |= g["key"] == h["key"] and ((g["kind"] == "P") != (h["kind"] == "P"))
    assert len(markers) == 1, "Marker color must be consistent inside one puzzle"
    assert tested <= trained
    assert not known & novel, "Test proposed wholes must be novel"
    assert all(witnesses.values()), witnesses
    assert ruled_out == {"raw", "all", "reflection", "area", "color", "exactly_two",
                         "translation", "identity", "marker"}, ruled_out


def _features(pair):
    grid, bg, marker, groups = _read(pair["input"])
    silhouettes = set()
    holes = diagonal = False
    for g in groups:
        local = _normal(g["source"])
        silhouettes.add(local)
        arr = np.zeros((max(r for r, c in local) + 1, max(c for r, c in local) + 1), bool)
        for q in local:
            arr[q] = True
        diagonal |= len(_components(arr, diagonal=False)) > 1
        holes |= len(_components(~np.pad(arr, 1), diagonal=False)) > 1
    return {"shape": grid.shape, "background": bg, "marker": marker, "count": len(groups),
            "profile": tuple(sorted(Counter(g["key"] for g in groups).values())),
            "kinds": tuple(sorted({g["kind"] for g in groups})), "silhouettes": silhouettes,
            "wholes": {g["key"] for g in groups}, "diagonal": diagonal, "hole": holes,
            "palette": tuple(sorted({g["color"] for g in groups})),
            "marker_lengths": {len(g["marker"]) for g in groups},
            "edge": bool(np.any(grid[0] != bg) or np.any(grid[-1] != bg)
                         or np.any(grid[:, 0] != bg) or np.any(grid[:, -1] != bg))}


def _puzzle_fingerprint(puzzle):
    return tuple(_fingerprint(p[k]) for s in ("train", "test") for p in puzzle[s]
                 for k in ("input", "output"))


def self_test():
    """Bounded structural, semantic-witness, mutation, and 42-seed diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    assert len({_puzzle_fingerprint(p) for p in puzzles}) == 42
    assert _puzzle_fingerprint(puzzles[0]) == _puzzle_fingerprint(generate_puzzle(0))
    fs = [_features(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    assert len({f["shape"] for f in fs}) >= 40
    assert len({f["profile"] for f in fs}) >= 12
    assert len({f["count"] for f in fs}) >= 5
    assert len(set().union(*(f["silhouettes"] for f in fs))) >= 90
    assert len(set().union(*(f["wholes"] for f in fs))) >= 100
    assert {f["marker"] for f in fs} == set(range(10))
    assert {f["background"] for f in fs} == set(range(10))
    assert any(f["diagonal"] for f in fs) and any(f["hole"] for f in fs)
    assert sum(f["edge"] for f in fs) >= 10
    assert {1, 2, 3, 4, 5} <= set().union(*(f["marker_lengths"] for f in fs))
    assert any(4 in f["profile"] for f in fs)
    assert any(f["profile"] == (1,) * f["count"] for f in fs)
    assert any(all(n > 1 for n in f["profile"]) for f in fs)
    assert len({f["palette"] for f in fs}) >= 20
    # Exercise the bounded fallback and compare coordinate mappings with masks.
    rng = np.random.default_rng(73021)
    fallback = _pair(rng, [("C", 2), ("F", 2)], {"marker": 9, "background": 0}, teaching=True, compact=True)
    assert np.array_equal(solve(fallback["input"]), fallback["output"])
    for puzzle in puzzles[:3]:
        for s in ("train", "test"):
            for pair in puzzle[s]:
                inp, bg, _, groups = _read(pair["input"])
                expected = inp.copy()
                for g in groups:
                    if g["multiplicity"] == 1:
                        continue
                    r0, r1 = min(r for r, c in g["source"]), max(r for r, c in g["source"])
                    c0, c1 = min(c for r, c in g["source"]), max(c for r, c in g["source"])
                    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
                    for r, c in g["source"]:
                        mask[r - r0, c - c0] = True
                    ar, ac = min(g["marker"])
                    if g["kind"] == "H":
                        mask, r0 = np.flipud(mask), 2 * ar - r1
                    elif g["kind"] == "V":
                        mask, c0 = np.fliplr(mask), 2 * ac - c1
                    else:
                        mask, r0, c0 = np.rot90(mask, 2), 2 * ar - r1, 2 * ac - c1
                    for r, c in np.argwhere(mask):
                        assert inp[r0 + r, c0 + c] == bg
                        expected[r0 + r, c0 + c] = g["color"]
                assert np.array_equal(expected, pair["output"])
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(6200 + nt + ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s: [{k: np.array(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return {"seeds_checked": 42, "pairs_checked": len(fs),
            "grid_shapes": len({f["shape"] for f in fs}),
            "class_profiles": len({f["profile"] for f in fs}),
            "source_silhouettes": len(set().union(*(f["silhouettes"] for f in fs))),
            "proposed_wholes": len(set().union(*(f["wholes"] for f in fs))),
            "diagonal_scenes": sum(f["diagonal"] for f in fs),
            "hole_scenes": sum(f["hole"] for f in fs)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
