"""Complete conditional diagonals, then contract identical consecutive vectors.

The secondary ink may already occupy perforated bands. It is not restricted to
isolated dots. Only its original cells on finite primary spans activate rays.
No parent module, stored puzzle, seed, or generation state is used by solve.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


HOST_SEEDS = (0, 307396099, 578445715, 896000771, 372653439, 53261561,
              531596940, 639944270, 715607940, 350899563, 969716984, 60629479,
              187426525, 661309059, 318065238, 537182009, 673881814, 808540681,
              836746679, 658690145, 674201275, 539808210, 780545874, 552966715,
              540590558, 113143950, 712809844, 682486506, 527105387, 343673459,
              462516997, 379856118, 173451113, 451378375, 767205963, 451312067,
              254067926, 351951888, 846869067, 488507110, 167682744, 488992475)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _span(a, b):
    # Endpoint pairs are sorted by row, so row steps are always positive.
    slope = 1 if b[1] > a[1] else -1
    return tuple((a[0] + k, a[1] + slope * k) for k in range(b[0] - a[0] + 1))


def _line(point, slope, shape):
    r, c = point
    return tuple((i, c + slope * (i - r)) for i in range(shape[0])
                 if 0 <= c + slope * (i - r) < shape[1])


def _edges(points):
    points = sorted(points)
    degrees = Counter()
    edges = []
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            if 0 < abs(a[0] - b[0]) == abs(a[1] - b[1]):
                edges.append((a, b))
                degrees[a] += 1
                degrees[b] += 1
    return edges, degrees


def _decode(a):
    # Four equal input corners explicitly identify the canvas, even when ink
    # occupies a majority of cells. Frequency is never a semantic convention.
    bg = int(a[0, 0])
    if any(int(a[r, c]) != bg for r, c in
           ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("Input corners must identify one background color")
    colors = set(map(int, np.unique(a))) - {bg}
    if len(colors) != 2:
        raise ValueError("Expected background, primary points, and secondary ink")
    candidates = []
    for color in sorted(colors):
        points = [tuple(map(int, p)) for p in np.argwhere(a == color)]
        edges, degrees = _edges(points)
        if edges and max(degrees.values()) == 1 and all(
                b[0] - x[0] >= 2 for x, b in edges):
            candidates.append((color, points, edges))
    if len(candidates) != 1:
        raise ValueError("Primary points must have unique diagonal partners")
    primary, points, edges = candidates[0]
    secondary = next(iter(colors - {primary}))
    return bg, primary, secondary, points, edges


def _complete(a):
    bg, primary, secondary, points, edges = _decode(a)
    spans = [_span(x, y) for x, y in edges]
    union = set().union(*(set(s) for s in spans))
    secondary_cells = {tuple(map(int, p)) for p in np.argwhere(a == secondary)}
    active = sorted(secondary_cells & union)
    lines = []
    for p in active:
        owners = [s for s in spans if p in s]
        if len(owners) != 1:
            raise ValueError("A secondary launch needs exactly one primary span")
        s = owners[0]
        slope = 1 if s[-1][1] > s[0][1] else -1
        line = _line(p, -slope, a.shape)
        if set(line) & union != {p} or set(line) & set(points):
            raise ValueError("Conflicting primary and secondary constructions")
        lines.append(line)
    b = a.copy()
    for s in spans:
        for p in s:
            b[p] = primary
    for line in lines:
        for p in line:
            b[p] = secondary
    endpoints = {p for edge in edges for p in edge}
    return b, {"background": bg, "primary": primary, "secondary": secondary,
               "spans": spans, "active": active, "lines": lines,
               "primary_dots": set(points), "unpaired": set(points) - endpoints,
               "inactive": secondary_cells - set(active)}


def _keep(a, axis):
    vectors = a if axis == 0 else a.T
    return np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)]


def _contract(a):
    # Removal of duplicate rows cannot erase a column distinction, and vice
    # versa. One pass per axis is sufficient and the two passes commute.
    b = a[_keep(a, 0)]
    return b[:, _keep(b, 1)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles and geometry from this input alone; draw, then contract."""
    completed, _ = _complete(_grid(input_grid))
    return _contract(completed)


def _bands(rng, eligible, count):
    """Sample separated bands from any safe consecutive coordinates."""
    eligible = set(eligible)
    for _ in range(16):
        free = eligible.copy()
        result = []
        for _ in range(count):
            candidates = [(start, length) for start in sorted(free)
                          for length in range(2, 6)
                          if all(i in free for i in range(start, start + length))]
            if not candidates:
                break
            start, length = candidates[int(rng.integers(len(candidates)))]
            result.append(tuple(range(start, start + length)))
            free.difference_update(range(start - 1, start + length + 1))
        if len(result) == count:
            return sorted(result)
    return None


def _place(rng, shape, counts, slopes, length_band):
    h, w = shape
    # The core is sampled in a generous interior; bands later use any safe
    # row/column, including gaps between components when these are available.
    ch = int(rng.integers(max(6, length_band[0]), min(h - 8, 14) + 1))
    cw = int(rng.integers(max(6, length_band[0]), min(w - 8, 14) + 1))
    top = int(rng.integers(3, h - ch - 2))
    left = int(rng.integers(3, w - cw - 2))
    spans, active, endpoints = [], [], []
    occupied, rays = set(), set()
    for count, slope in zip(counts, slopes):
        for _ in range(96):
            low = max(3, count + 3, length_band[0])
            high = min(ch, cw, length_band[1])
            if low > high:
                return None
            length = int(rng.integers(low, high + 1))
            r = top + int(rng.integers(ch - length + 1))
            c = left + int(rng.integers(cw - length + 1))
            if slope == -1:
                c += length - 1
            a, b = (r, c), (r + length - 1, c + slope * (length - 1))
            if any(abs(p[0] - q[0]) == abs(p[1] - q[1])
                   for p in (a, b) for q in endpoints):
                continue
            s = _span(a, b)
            cells = set(s)
            if cells & rays or cells & occupied:
                continue
            possible = []
            for p in s[1:-1]:
                line = set(_line(p, -slope, shape))
                if line & (occupied | cells) == {p}:
                    possible.append((p, line))
            if len(possible) < count:
                continue
            chosen = [possible[int(i)] for i in rng.permutation(len(possible))[:count]]
            spans.append(s)
            endpoints.extend((a, b))
            occupied.update(cells)
            for p, line in chosen:
                active.append(p)
                rays.update(line)
            break
        else:
            return None
    return spans, active, occupied, rays, endpoints


def _new_equal(a, b, axis):
    """Adjacent distinct input vectors that become equal after drawing."""
    return set(map(int, np.flatnonzero(_keep(a, axis)[1:] & ~_keep(b, axis)[1:])))


def _separated(a, axis, color):
    vectors = a if axis == 0 else a.T
    positions = [i for i, v in enumerate(vectors) if np.all(v == color)]
    return bool(positions and any(y > x + 1 for x, y in zip(positions, positions[1:])))


def _attempt(rng, primary, secondary, bg, config):
    h, w = (int(rng.integers(config["size"][0], config["size"][1] + 1))
            for _ in range(2))
    plan = _place(rng, (h, w), config["counts"], config["slopes"], config["length"])
    if plan is None:
        return None
    spans, active, occupied, rays, endpoints = plan
    row_ok = [i for i in range(1, h - 1) if not any(r == i for r, c in occupied)
              and any(r == i for r, c in rays)]
    col_ok = [j for j in range(1, w - 1) if not any(c == j for r, c in occupied)
              and any(c == j for r, c in rays)]
    row_bands = _bands(rng, row_ok, config["bands"][0])
    col_bands = _bands(rng, col_ok, config["bands"][1])
    if row_bands is None or col_bands is None:
        return None
    row_set = set().union(*(set(b) for b in row_bands))
    col_set = set().union(*(set(b) for b in col_bands))
    a = np.full((h, w), bg, dtype=np.int8)
    a[list(row_set), :] = secondary
    a[:, list(col_set)] = secondary
    for p in rays:
        if p[0] in row_set or p[1] in col_set:
            a[p] = bg
    for p in endpoints:
        a[p] = primary
    for p in active:
        a[p] = secondary

    # An inert dot on a primary continuation (strictly beyond its endpoints)
    # distinguishes finite spans from infinite primary-line activation.
    beyond = []
    for s in spans:
        slope = 1 if s[-1][1] > s[0][1] else -1
        for p in _line(s[0], slope, (h, w)):
            if (p not in occupied and p not in rays and p[0] not in row_set
                    and p[1] not in col_set and 0 < p[0] < h - 1
                    and 0 < p[1] < w - 1):
                beyond.append(p)
    if not beyond:
        return None
    inert = beyond[int(rng.integers(len(beyond)))]
    a[inert] = secondary

    # A short nonactivated diagonal scar in a band stays open. Keep one clean
    # band of each orientation, so the central dependency is never lost.
    scar = []
    if config["scar"]:
        axes = [axis for axis, bands in enumerate((row_bands, col_bands)) if len(bands) > 1]
        target_axis = int(rng.choice(axes))
        target = row_bands[-1] if target_axis == 0 else col_bands[-1]
        if (target_axis == 1 and len(col_bands) < 2):
            return None
        slope = int(rng.choice((-1, 1)))
        candidates = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                points = _line((r, c), slope, (h, w))
                cut = [p for p in points if p[target_axis] in target]
                if (len(cut) >= 2 and all(p not in rays and p not in occupied
                                         and p != inert for p in cut)):
                    candidates.append(cut)
        if not candidates:
            return None
        scar = candidates[int(rng.integers(len(candidates)))]
        for p in scar:
            a[p] = bg

    # Optional unpaired primary points outside all constructions/bands.
    for _ in range(config["dots"]):
        candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                      if a[r, c] == bg and (r, c) not in occupied | rays
                      and r not in row_set and c not in col_set
                      and all(abs(r - x) != abs(c - y) for x, y in endpoints)]
        if not candidates:
            return None
        p = candidates[int(rng.integers(len(candidates)))]
        a[p] = primary
        endpoints.append(p)

    # Construct the label from the independently sampled plan, not solve(a).
    b = a.copy()
    for s in spans:
        for p in s:
            b[p] = primary
    for p in rays:
        b[p] = secondary
    if not _new_equal(a, b, 0) or not _new_equal(a, b, 1):
        return None
    # For a two-band witness, at least one orientation retains separated equal
    # vectors. The scar may prevent this on its own orientation.
    if config["separated"] and not any(_separated(b, axis, secondary) for axis in (0, 1)):
        return None
    try:
        observed, info = _complete(a)
    except ValueError:
        return None
    if not np.array_equal(observed, b) or set(info["active"]) != set(active):
        return None
    if scar and not all(b[p] == bg for p in scar):
        return None
    return {"input": a, "output": _contract(b)}


def _counts(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _config(rng, kind, shift):
    # Two informative witnesses, shuffled later. These are evidence requests,
    # not coordinates or a bank of completed layouts.
    if kind == "a":
        counts, slopes = [2], [1]
        bands, scar, dots = (2, 2), True, 1
    elif kind == "b":
        counts, slopes = [1, 0], [-1, int(rng.choice((-1, 1)))]
        bands, scar, dots = (2, 2), False, 0
    elif kind == "test" and shift == "rays":
        counts, slopes = [3], [int(rng.choice((-1, 1)))]
        bands, scar, dots = (2, 2), bool(rng.integers(2)), int(rng.integers(3))
    elif kind == "test" and shift == "segments":
        counts = [int(rng.integers(1, 3)), 0, 1]
        slopes = [int(rng.choice((-1, 1))) for _ in counts]
        bands, scar, dots = (2, 2), False, int(rng.integers(2))
    else:
        count = int(rng.integers(1, 3))
        counts = [int(rng.integers(0, 3)) for _ in range(count)]
        if not any(counts):
            counts[0] = 1
        slopes = [int(rng.choice((-1, 1))) for _ in counts]
        bands = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
        scar = max(bands) > 1 and bool(rng.integers(2))
        dots = int(rng.integers(3))
    if kind == "test" and shift == "length":
        counts = [2, 0]
        slopes = [int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))]
        bands, scar = (2, 2), True
    return {"counts": counts, "slopes": slopes, "bands": bands,
            "scar": scar, "dots": dots, "separated": kind in ("a", "b"),
            "length": (8, 12) if kind == "test" and shift == "length" else (3, 7),
            "size": (22, 30) if kind == "test" else (18, 30)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _counts(num_train, "num_train", 2, 10)
    num_test = _counts(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    primary, secondary = palette[:2]
    varying_bg = bool(rng.integers(2))
    shift = str(rng.choice(("rays", "segments", "length")))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind = ("a" if i == 0 else "b" if i == 1 else "extra") if split == "train" else "test"
            config = _config(rng, kind, shift)
            bg = int(rng.choice(palette[2:])) if varying_bg else palette[2]
            for _ in range(320):
                pair = _attempt(rng, primary, secondary, bg, config)
                if pair is not None and _key(pair["input"]) not in used:
                    used.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded geometry construction exhausted")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a, out = _grid(pair["input"]), _grid(pair["output"])
    b, info = _complete(a)
    primary, secondary = info["primary"], info["secondary"]
    union = set().union(*(set(s) for s in info["spans"]))
    rays = set().union(*(set(line) for line in info["lines"])) if info["lines"] else set()
    assert np.array_equal(_contract(b), out)
    assert not np.array_equal(a, b), "Diagonal completion must change cells"
    assert _new_equal(a, b, 0) and _new_equal(a, b, 1), "Drawing must CREATE repeats on both axes"
    assert out.shape[0] < a.shape[0] and out.shape[1] < a.shape[1]
    assert not np.array_equal(_contract(a), out), "Omitting parent B must change the answer"
    assert not np.array_equal(b, out), "Omitting parent A must change the answer"
    primary_only = a.copy()
    for p in union:
        if a[p] != secondary:
            primary_only[p] = primary
    assert _new_equal(primary_only, b, 0) and _new_equal(primary_only, b, 1)
    assert not np.array_equal(_contract(primary_only), out), "Secondary activation must matter"
    assert all(b[p] == primary for p in union - set(info["active"]))
    assert all(b[p] == secondary for p in rays)
    assert all(b[p] == a[p] for p in np.ndindex(a.shape) if p not in union | rays)
    assert all(b[p] == a[p] for p in info["primary_dots"] | info["inactive"])
    # Literal expansion by post-drawing runs must recover the entire completed
    # canvas, including crossings. This is stronger than size checks.
    row_starts = np.flatnonzero(_keep(b, 0))
    col_starts = np.flatnonzero(_keep(b, 1))
    row_runs = np.diff(np.r_[row_starts, b.shape[0]])
    col_runs = np.diff(np.r_[col_starts, b.shape[1]])
    assert np.array_equal(np.repeat(np.repeat(out, row_runs, 0), col_runs, 1), b)
    assert np.all(_keep(out, 0)) and np.all(_keep(out, 1))
    # The two contraction-axis orders agree, including crossing values.
    assert _contract(b.T).T.shape == out.shape
    assert np.array_equal(_contract(b.T).T, out)
    slopes = [1 if s[-1][1] > s[0][1] else -1 for s in info["spans"]]
    triggers = [sum(p in info["active"] for p in s) for s in info["spans"]]
    inactive_beyond = any(p not in rays and abs(p[0] - s[0][0]) == abs(p[1] - s[0][1])
                         and abs(p[0] - s[-1][0]) == abs(p[1] - s[-1][1])
                         for p in info["inactive"] for s in info["spans"])
    scar_axes = set()
    filled_scars = b.copy()
    for axis in (0, 1):
        vectors = b if axis == 0 else b.T
        for i in range(len(vectors) - 1):
            x, y = vectors[i:i + 2]
            if (np.count_nonzero(x == secondary) == len(x) - 1
                    and np.count_nonzero(y == secondary) == len(y) - 1
                    and np.count_nonzero(x == info["background"]) == 1
                    and np.count_nonzero(y == info["background"]) == 1
                    and not np.array_equal(x, y)):
                scar_axes.add(axis)
                target = filled_scars if axis == 0 else filled_scars.T
                for j in (i, i + 1):
                    target[j, vectors[j] == info["background"]] = secondary
    scar_changes = not np.array_equal(_contract(filled_scars), out)
    # Separately test the tempting, generally invalid interchange of drawing
    # and contraction, using the compressed input's visible geometry.
    try:
        early, _ = _complete(_contract(a))
        premature_differs = not np.array_equal(_contract(early), out)
    except ValueError:
        premature_differs = True
    return {"shape": a.shape, "out_shape": out.shape, "roles": (primary, secondary),
            "bg": info["background"], "segments": len(slopes), "slopes": tuple(slopes),
            "active_slopes": {s for s, t in zip(slopes, triggers) if t},
            "triggers": tuple(triggers), "lengths": tuple(map(len, info["spans"])),
            "new_rows": len(_new_equal(a, b, 0)), "new_cols": len(_new_equal(a, b, 1)),
            "unpaired": len(info["unpaired"]), "beyond": inactive_beyond,
            "separated": any(_separated(b, axis, secondary) for axis in (0, 1)),
            "scar_axes": scar_axes, "scar_changes": scar_changes,
            "premature_differs": premature_differs,
            "row_runs": tuple(map(int, row_runs)), "col_runs": tuple(map(int, col_runs))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used, roles, train_features = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a = _grid(pair["input"])
            assert _key(a) not in used
            used.add(_key(a))
            f = _features(pair)
            roles.add(f["roles"])
            if split == "train":
                train_features.append(f)
    assert len(roles) == 1
    assert set().union(*(f["active_slopes"] for f in train_features)) == {-1, 1}
    assert any(max(f["triggers"]) >= 2 for f in train_features)
    assert any(0 in f["triggers"] for f in train_features)
    assert any(f["unpaired"] for f in train_features)
    assert all(f["beyond"] for f in train_features)
    assert any(f["separated"] for f in train_features)
    assert any(f["scar_axes"] and f["scar_changes"] for f in train_features)
    assert any(f["premature_differs"] for f in train_features)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                a = _grid(pair[key])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded in-memory checks: 42 seeds, explicit counts, and mutations."""
    shapes, out_shapes, roles, signatures, layouts = set(), set(), set(), set(), set()
    all_counts, lengths, triggers, backgrounds = set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(f["shape"])
                out_shapes.add(f["out_shape"])
                roles.add(f["roles"])
                all_counts.add(f["segments"])
                lengths.update(f["lengths"])
                triggers.update(f["triggers"])
                backgrounds.add(f["bg"])
                layouts.add((f["shape"], f["slopes"], f["lengths"], f["triggers"],
                             f["row_runs"], f["col_runs"]))
    assert len(signatures) == 42 and len(shapes) >= 40 and len(out_shapes) >= 30
    assert len(roles) >= 20 and len(backgrounds) >= 8 and len(layouts) >= 180
    assert {1, 2, 3} <= all_counts and {0, 1, 2, 3} <= triggers
    assert min(lengths) == 3 and max(lengths) >= 10
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(81, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    first = generate_puzzle(0)
    assert _fingerprint(first) == _fingerprint(generate_puzzle(0))
    for train, test in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2),
                        (2, 4), (True, 1), (3.0, 1), (3, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal example counts accepted")
    # A tiny explicit dependency witness has no adjacent equal vectors until
    # the activated ray repairs two consecutive secondary rows and columns.
    witness = np.zeros((11, 11), dtype=np.int8)
    witness[1, 1] = witness[5, 5] = 1
    witness[3, 3] = 2
    witness[6:8, :] = 2
    witness[:, 6:8] = 2
    ray = _line((3, 3), -1, witness.shape)
    for p in ray:
        if p[0] in (6, 7) or p[1] in (6, 7):
            witness[p] = 0
    completed, _ = _complete(witness)
    assert _new_equal(witness, completed, 0) and _new_equal(witness, completed, 1)
    assert not np.array_equal(_contract(witness), solve(witness))
    # Equality is local, not global, and the literal crossing cell survives.
    toy = np.array([[2, 2, 0, 2], [2, 2, 0, 2], [1, 1, 2, 1],
                    [2, 2, 0, 2], [2, 2, 0, 2]], dtype=np.int8)
    assert np.array_equal(_contract(toy), [[2, 0, 2], [1, 2, 1], [2, 0, 2]])
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]]
           for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output was accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds; no artifacts written)")
