"""Registered contact circuits. Only NumPy and the standard library are used.

0 = outside, 9 = temporary registration bars, 1 = terminals, 2 = contacts.
One color in 3..8 supplies the sheet material. No files or RNG enter solve().
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


def _components(cells, diagonal=True):
    remaining, result = set(cells), []
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or not (r and c))]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _shift(cells, dr, dc=0):
    return {(r + dr, c + dc) for r, c in cells}


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert dr > 0 and abs(dr) == abs(dc)
    slope = 1 if dc > 0 else -1
    return tuple((a[0] + k, a[1] + slope * k) for k in range(dr + 1))


def _ray(domain, point, slope):
    """Both half-rays stop at the first missing material cell, including holes."""
    cells = {point}
    for direction in (-1, 1):
        r, c = point
        while (r + direction, c + direction * slope) in domain:
            r, c = r + direction, c + direction * slope
            cells.add((r, c))
    return cells


def _line_in_box(domain, point, slope):
    t, b, l, right = _bbox(domain)
    return {(r, point[1] + slope * (r - point[0])) for r in range(t, b + 1)
            if l <= point[1] + slope * (r - point[0]) <= right}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _native_edges(grid):
    points = [tuple(map(int, p)) for p in np.argwhere(np.asarray(grid) == 1)]
    return sum(abs(p[0] - q[0]) == abs(p[1] - q[1])
               for i, p in enumerate(points) for q in points[i + 1:])


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a


def _decode(input_grid):
    """Recover ownership, axis, backing sides and registration from input pixels."""
    grid = _grid(input_grid)
    colors = set(map(int, np.unique(grid)))
    material = colors & set(range(3, 9))
    assert len(material) == 1 and {0, 1, 9} <= colors
    material = next(iter(material))
    colored = {tuple(map(int, p)) for p in np.argwhere(grid != 0)}
    groups = _components(colored)
    bars = _components({tuple(map(int, p)) for p in np.argwhere(grid == 9)}, False)
    assert len(groups) == len(bars) == 2
    assert len(bars[0]) == len(bars[1]) >= 2
    length = len(bars[0])
    axes = [axis for axis in (0, 1) if all(
        len({p[axis] for p in bar}) == 1 and
        max(p[1 - axis] for p in bar) - min(p[1 - axis] for p in bar) + 1 == length
        for bar in bars)]
    assert len(axes) == 1
    axis = axes[0]
    records = []
    for group in groups:
        own = [bar for bar in bars if bar <= group]
        assert len(own) == 1
        bar = own[0]
        normal = min(p[axis] for p in bar)
        tangent = min(p[1 - axis] for p in bar)
        ink = {(p[axis] - normal, p[1 - axis] - tangent): int(grid[p])
               for p in group - bar}
        negative = any((-1, c) in ink for c in range(length))
        positive = any((1, c) in ink for c in range(length))
        assert negative != positive, "bar must have a unique backing side"
        records.append((negative, ink))
    assert records[0][0] != records[1][0]
    records.sort(key=lambda item: not item[0])
    anchor = records[0][1]
    mover = {(r - 1, c): color for (r, c), color in records[1][1].items()}
    return anchor, mover, material, axis, length


def _circuit(anchor, mover, material, *, activation="contact", ray_mode="stop", bridges=True, primary_mode="finite"):
    """Render in registration coordinates; optional modes are explicit ablations."""
    domain, overlap = set(anchor) | set(mover), set(anchor) & set(mover)
    special = {}
    owners = {}
    for owner, layer in enumerate((anchor, mover)):
        for p, color in layer.items():
            if color in (1, 2):
                assert p not in special, "colored dots may not coincide across sheets"
                special[p] = color
                owners[p] = owner
    primary = {p for p, color in special.items() if color == 1}
    secondary = {p for p, color in special.items() if color == 2}
    points = sorted(primary)
    edges = [(a, b) for i, a in enumerate(points) for b in points[i + 1:]
             if a[0] != b[0] and abs(a[0] - b[0]) == abs(a[1] - b[1])]
    degrees = Counter(p for edge in edges for p in edge)
    assert edges and max(degrees.values()) == 1, "terminal mates must be unique"
    segments = [_segment(a, b) for a, b in edges]
    assert all(len(s) >= 3 and set(s) <= domain for s in segments)
    slopes = [1 if s[-1][1] > s[0][1] else -1 for s in segments]
    primary_union = set().union(*map(set, segments))
    # A contact at a crossing of two primary slopes would have no unique normal.
    memberships = {p: [i for i, s in enumerate(segments) if p in s] for p in secondary}
    assert all(len(indices) <= 1 for indices in memberships.values())
    active = {}
    for p in sorted(secondary):
        indices = memberships[p]
        if activation == "infinite":
            indices = [i for i, s in enumerate(segments)
                       if p[1] - s[0][1] == slopes[i] * (p[0] - s[0][0])]
            assert len(indices) <= 1
        if indices and (activation == "all_on_segment" or p in overlap):
            active[p] = indices[0]
    rays = {}
    for p, index in active.items():
        slope = -slopes[index]
        if ray_mode == "stop":
            rays[p] = _ray(domain, p, slope)
        elif ray_mode == "jump":
            rays[p] = _line_in_box(domain, p, slope) & domain
        else:
            assert ray_mode == "box"
            rays[p] = _line_in_box(domain, p, slope)
    t, b, l, right = _bbox(domain)
    output = np.zeros((b - t + 1, right - l + 1), dtype=np.int8)
    for r, c in domain:
        output[r - t, c - l] = material
    if bridges:
        paint = primary_union if primary_mode == "finite" else set().union(
            *(_ray(domain, s[0], slope) for s, slope in zip(segments, slopes)))
        for r, c in paint:
            output[r - t, c - l] = 1
    for p, color in special.items():
        output[p[0] - t, p[1] - l] = color
    if bridges:
        for line in rays.values():
            for r, c in line:
                output[r - t, c - l] = 2
    on = {p for p, indices in memberships.items() if indices}
    infinite = {p for p in secondary if any(
        p[1] - s[0][1] == slope * (p[0] - s[0][0])
        for s, slope in zip(segments, slopes))}
    info = {"domain": domain, "overlap": overlap, "special": special,
            "owners": owners, "primary": primary, "secondary": secondary,
            "segments": segments, "slopes": slopes, "primary_union": primary_union,
            "active": active, "rays": rays, "on": on,
            "unbacked_on": on - overlap, "backed_beyond": (infinite - on) & overlap,
            "backed_off": (secondary - infinite) & overlap,
            "unpaired": primary - set(degrees),
            "gap": any((_line_in_box(domain, p, -slopes[i]) & domain) - _ray(domain, p, -slopes[i])
                       for p, i in active.items())}
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only registration, contact testing, finite bridges, and stopped rays."""
    a, b, material, axis, _ = _decode(input_grid)
    output, _ = _circuit(a, b, material)
    return output.T.copy() if axis else output


def _measure(input_grid, output_grid=None):
    a, b, material, axis, length = _decode(input_grid)
    output, info = _circuit(a, b, material)
    expected = output.T.copy() if axis else output
    if output_grid is not None:
        assert np.array_equal(expected, output_grid), "incorrect contact circuit"
    # Geometric checks go beyond simply repeating solve().
    t, bottom, left, right = _bbox(info["domain"])
    observed_domain = {(int(r) + t, int(c) + left) for r, c in np.argwhere(output != 0)}
    assert observed_domain == info["domain"], "material union must be preserved exactly"
    assert len(info["domain"]) == len(a) + len(b) - len(info["overlap"])
    assert output.shape == (bottom - t + 1, right - left + 1)
    assert all(np.any(output[edge] != 0) for edge in (0, -1))
    assert np.any(output[:, 0] != 0) and np.any(output[:, -1] != 0)
    for p, index in info["active"].items():
        assert p in a and p in b and p in info["segments"][index][1:-1]
        line = info["rays"][p]
        assert line & info["primary_union"] == {p}, "avoid two-color crossings"
        assert not line & info["primary"]
        slope = -info["slopes"][index]
        for sign in (-1, 1):
            q = p
            while (q[0] + sign, q[1] + sign * slope) in line:
                q = q[0] + sign, q[1] + sign * slope
            assert (q[0] + sign, q[1] + sign * slope) not in info["domain"]
    for p, color in info["special"].items():
        assert output[p[0] - t, p[1] - left] == color, "original dots must survive"
    guide_positions = {(r, c) for r in (-1, 0) for c in range(length)}
    info.update({"material": material, "axis": axis, "bar_length": length,
                 "covered_guides": len(guide_positions & info["domain"]),
                 "uncovered_guides": len(guide_positions - info["domain"]),
                 "cross_sheet": sum(info["owners"][s[0]] != info["owners"][s[-1]] for s in info["segments"]),
                 "without_B_diff": not np.array_equal(output, _circuit(a, b, material, bridges=False)[0]),
                 "backing_diff": not np.array_equal(output, _circuit(a, b, material, activation="all_on_segment")[0]),
                 "finite_diff": not np.array_equal(output, _circuit(a, b, material, activation="infinite")[0]),
                 "primary_diff": not np.array_equal(output, _circuit(a, b, material, primary_mode="extend")[0]),
                 "jump_diff": not np.array_equal(output, _circuit(a, b, material, ray_mode="jump")[0]),
                 "box_diff": not np.array_equal(output, _circuit(a, b, material, ray_mode="box")[0])})
    info["native_edges"] = _native_edges(input_grid)
    info["contact_toggle_diff"] = False
    # A concrete counterfactual preserves the flattened substrate and all dots,
    # but adds ordinary backing to one open contact away from registration ink.
    protected = {(r, c) for r in (-2, -1, 0, 1) for c in range(length)}
    for p in sorted(info["unbacked_on"] - protected):
        missing = b if p in a else a
        if not any((p[0] + dr, p[1] + dc) in missing
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc):
            continue
        changed = dict(missing)
        changed[p] = material
        alternate_a, alternate_b = (a, changed) if p in a else (changed, b)
        alternate, alternate_info = _circuit(alternate_a, alternate_b, material)
        assert alternate_info["domain"] == info["domain"]
        if not np.array_equal(output, alternate):
            info["contact_toggle_diff"] = True
            break
    return info


def _masks(rng, h, w, length):
    seam = int(rng.integers(3, h - 2))
    offset = int(rng.integers(w - length + 1))
    a, b = set(), set()
    # Independent column profiles produce asymmetric lobes, steps and long arms.
    for c in range(w):
        top = int(rng.integers(0, min(3, seam - 1)))
        bottom = int(rng.integers(seam - 1, h))
        a.update((r, c) for r in range(top, bottom + 1))
        top = int(rng.integers(0, seam + 1))
        bottom = int(rng.integers(max(seam + 1, h - 3), h))
        b.update((r, c) for r in range(top, bottom + 1))
    bars = [{(seam, c) for c in range(offset, offset + length)},
            {(seam - 1, c) for c in range(offset, offset + length)}]
    protected = {(r, c) for r in range(seam - 2, seam + 2) for c in range(offset, offset + length)}
    # Some holes belong to one sheet; others survive their union. No repair ink.
    for layer in (a, b):
        for p in sorted(layer - protected):
            if 0 < p[0] < h - 1 and 0 < p[1] < w - 1 and rng.random() < .08:
                layer.remove(p)
    for _ in range(int(rng.integers(0, 5))):
        p = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        if p not in protected:
            a.discard(p)
            b.discard(p)
    for c in range(offset, offset + length):
        a.discard((seam, c))
        a.discard((seam + 1, c))
        a.add((seam - 1, c))
        b.discard((seam - 1, c))
        b.discard((seam - 2, c))
        b.add((seam, c))
    if any(len(_components(layer | bar)) != 1 for layer, bar in zip((a, b), bars)):
        return None
    return a, b, bars


def _candidate_segments(a, b):
    domain = a | b
    result = []
    bottom = _bbox(domain)[1]
    for p in sorted(domain):
        for slope in (-1, 1):
            for steps in range(2, min(11, bottom - p[0]) + 1):
                q = p[0] + steps, p[1] + slope * steps
                if q not in domain:
                    continue
                if not ((p in a and q in b) or (p in b and q in a)):
                    continue
                segment = _segment(p, q)
                if set(segment) <= domain:
                    result.append((segment, slope))
    return result


def _make_ink(rng, a, b, segments, slopes, kind, material, bars):
    domain, overlap = a | b, a & b
    primary_union = set().union(*map(set, segments))
    endpoints = {p for s in segments for p in (s[0], s[-1])}
    pins = set()
    active = set()
    all_rays = set()
    for index, (s, slope) in enumerate(zip(segments, slopes)):
        eligible = [p for p in s[1:-1] if p in overlap and
                    _ray(domain, p, -slope) & primary_union == {p} and
                    len(_ray(domain, p, -slope)) >= 3]
        if index == 0 and kind in ("contact", "boundary"):
            if len(eligible) < (2 if kind == "boundary" else 1):
                return None
            if kind == "boundary":
                gap = [p for p in eligible if (_line_in_box(domain, p, -slope) & domain) - _ray(domain, p, -slope)]
                if not gap:
                    return None
                first = gap[int(rng.integers(len(gap)))]
                chosen = [first, eligible[int(rng.choice([i for i, p in enumerate(eligible) if p != first]))]]
            else:
                chosen = [eligible[int(rng.integers(len(eligible)))]]
        else:
            number = int(rng.integers(0, min(3, len(eligible)) + 1))
            chosen = [eligible[int(i)] for i in rng.permutation(len(eligible))[:number]]
        pins.update(chosen)
        active.update(chosen)
        for p in chosen:
            all_rays.update(_ray(domain, p, -slope))
    # Keep unbacked contacts visible on bridges; they do not launch rays.
    unbacked = [p for p in sorted(primary_union - overlap - endpoints)
                if sum(p in s for s in segments) == 1 and p not in all_rays]
    if kind == "contact":
        protected = bars[0] | bars[1] | _shift(bars[0], 1) | _shift(bars[1], -1)
        unbacked = [p for p in unbacked if p not in protected and any(
            (p[0] + dr, p[1] + dc) in (b if p in a else a)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
            and len(_ray(domain, p, -slopes[0])) >= 2]
        if not unbacked:
            return None
        pins.add(unbacked[int(rng.integers(len(unbacked)))])
        s, slope = segments[0], slopes[0]
        beyond = [p for p in sorted(overlap - primary_union - all_rays - endpoints)
                  if p[1] - s[0][1] == slope * (p[0] - s[0][0]) and
                  len(_ray(domain, p, -slope)) >= 2]
        if not beyond:
            return None
        pins.add(beyond[int(rng.integers(len(beyond)))])
    elif unbacked and rng.random() < .45:
        pins.add(unbacked[int(rng.integers(len(unbacked)))])
    off = [p for p in sorted(domain - primary_union - endpoints - all_rays - pins)
           if not any(p[1] - s[0][1] == slope * (p[0] - s[0][0]) for s, slope in zip(segments, slopes))]
    if kind == "boundary":
        backed = [p for p in off if p in overlap]
        if not backed:
            return None
        pins.add(backed[int(rng.integers(len(backed)))])
    else:
        for i in rng.permutation(len(off))[:int(rng.integers(3))]:
            pins.add(off[int(i)])
    ink = [{p: material for p in a}, {p: material for p in b}]
    for s in segments:
        owners = [owner for owner in (0, 1) if s[0] in (a, b)[owner] and s[-1] in (b, a)[owner]]
        owner = owners[int(rng.integers(len(owners)))]
        ink[owner][s[0]] = 1
        ink[1 - owner][s[-1]] = 1
    for p in sorted(pins):
        owners = [i for i, layer in enumerate((a, b)) if p in layer]
        ink[owners[int(rng.integers(len(owners)))]][p] = 2
    # Optional terminals with no diagonal mate distinguish pairing from filling
    # every terminal's entire diagonal. They are not compulsory noise.
    if kind == "regular":
        for i in rng.permutation(len(off)):
            if rng.random() < .65:
                break
            p = off[int(i)]
            if p in pins or any(abs(p[0] - q[0]) == abs(p[1] - q[1]) for q in endpoints):
                continue
            owner = int(rng.choice([j for j, layer in enumerate((a, b)) if p in layer]))
            ink[owner][p] = 1
            endpoints.add(p)
            break
    return ink


def _stage(rng, ink, bars, mode):
    groups = []
    for layer, bar in zip(ink, bars):
        t, b, l, right = _bbox(set(layer) | bar)
        cells = {(r - t, c - l): color for (r, c), color in layer.items()}
        cells.update({(r - t, c - l): 9 for r, c in bar})
        groups.append((cells, b - t + 1, right - l + 1))
    if rng.integers(2):
        groups.reverse()
    x, h1, w1 = groups[0]
    y, h2, w2 = groups[1]
    gap = int(rng.integers(1, min(4, 30 - max(h1 + h2, w1 + w2)) + 1))
    if mode == "horizontal":
        offset = int(rng.integers(-2, 3)), w1 + gap
    elif mode == "vertical":
        offset = h1 + gap, int(rng.integers(-2, 3))
    else:
        offset = h1 + gap, w1 + gap
    y = {(r + offset[0], c + offset[1]): color for (r, c), color in y.items()}
    assert not set(x) & set(y)
    cells = x | y
    t, b, l, right = _bbox(cells)
    h, w = b - t + 1, right - l + 1
    if max(h, w) > 30:
        return None
    pads = []
    for remaining in (30 - h, 30 - w):
        first = int(rng.integers(0, min(3, remaining) + 1))
        last = int(rng.integers(0, min(3, remaining - first) + 1))
        pads.append((first, last))
    grid = np.zeros((h + sum(pads[0]), w + sum(pads[1])), dtype=np.int8)
    for (r, c), color in cells.items():
        grid[r - t + pads[0][0], c - l + pads[1][0]] = color
    return grid


def _make_pair(rng, profile, used):
    for _ in range(160):
        h = int(rng.integers(*profile["height"]))
        w = int(rng.integers(8, 14))
        length = int(rng.choice(profile["lengths"]))
        geometry = _masks(rng, h, w, length)
        if geometry is None:
            continue
        a, b, bars = geometry
        choices = _candidate_segments(a, b)
        if not choices:
            continue
        segments, slopes, endpoints = [], [], []
        # Select free geometry, never a stored puzzle or parent construction.
        for i in rng.permutation(len(choices)):
            s, slope = choices[int(i)]
            if not segments:
                if profile.get("slope") is not None and slope != profile["slope"]:
                    continue
                interior = set(s[1:-1])
                if profile["kind"] == "contact" and (len(s) < 5 or not interior & (a & b) or not interior - (a & b)):
                    continue
                if profile["kind"] == "contact":
                    domain, overlap = a | b, a & b
                    protected = bars[0] | bars[1] | _shift(bars[0], 1) | _shift(bars[1], -1)
                    open_pins = [p for p in interior - overlap - protected if
                                 len(_ray(domain, p, -slope)) >= 2 and any(
                                     (p[0] + dr, p[1] + dc) in (b if p in a else a)
                                     for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)]
                    beyond = [p for p in overlap - set(s) if
                              p[1] - s[0][1] == slope * (p[0] - s[0][0]) and
                              len(_ray(domain, p, -slope)) >= 2]
                    closed_pins = [p for p in interior & overlap if len(_ray(domain, p, -slope)) >= 3]
                    if not (open_pins and closed_pins and beyond and len(_ray(domain, s[0], slope) - set(s)) >= 2):
                        continue
                if profile["kind"] == "boundary" and len(interior & (a & b)) < 2:
                    continue
            if any(abs(p[0] - q[0]) == abs(p[1] - q[1]) for p in (s[0], s[-1]) for q in endpoints):
                continue
            segments.append(s)
            slopes.append(slope)
            endpoints.extend((s[0], s[-1]))
            if len(segments) == profile["count"]:
                break
        if len(segments) != profile["count"]:
            continue
        ink = _make_ink(rng, a, b, segments, slopes, profile["kind"], profile["material"], bars)
        if ink is None:
            continue
        # Registration coordinates: anchor bar is row zero; mover bar is -1.
        seam, _, offset, _ = _bbox(bars[0])
        anchor = {(r - seam, c - offset): color for (r, c), color in ink[0].items()}
        mover = {(r - seam, c - offset): color for (r, c), color in ink[1].items()}
        try:
            expected, info = _circuit(anchor, mover, profile["material"])
        except AssertionError:
            continue
        # Both signs are explicitly taught, independent of assembly orientation.
        if profile.get("slope") is not None and profile["slope"] not in {info["slopes"][i] for i in info["active"].values()}:
            continue
        grid = _stage(rng, ink, bars, profile["stage"])
        if grid is None:
            continue
        k = profile["rotation"]
        pair = {"input": np.rot90(grid, k).copy(), "output": np.rot90(expected, k).copy()}
        if _key(pair["input"]) in used:
            continue
        try:
            measured = _measure(pair["input"], pair["output"])
        except AssertionError:
            continue
        if not measured["without_B_diff"]:
            continue
        if profile["kind"] == "contact" and not (measured["backing_diff"] and measured["finite_diff"] and measured["primary_diff"] and measured["contact_toggle_diff"]):
            continue
        if profile["kind"] == "contact" and measured["native_edges"]:
            continue
        if profile["kind"] == "boundary" and not (measured["gap"] and measured["jump_diff"] and measured["backed_off"]):
            continue
        return pair
    raise RuntimeError("bounded contact-circuit construction exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain = _count("num_train", num_train, 2, 10)
    ntest = _count("num_test", num_test, 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    rotation = int(rng.integers(4))
    material = int(rng.integers(3, 9))
    changing_material = bool(rng.integers(2))
    regime = str(rng.choice(("bar_interpolation", "more_bridges", "smaller_query", "placement")))
    base_stage = str(rng.choice(("horizontal", "vertical", "diagonal")))
    witness_slots = list(map(int, rng.choice(ntrain, 2, replace=False)))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            kind, slope = "regular", None
            if split == "train" and index in witness_slots:
                witness = witness_slots.index(index)
                kind, slope = ("contact", -1) if witness == 0 else ("boundary", 1)
            number = 1 if kind == "contact" else int(rng.choice((1, 1, 2))) if kind == "boundary" else int(rng.integers(1, 4))
            if regime == "more_bridges":
                number = min(number, 2) if split == "train" else 3
            lengths = (2, 3, 4, 5)
            if regime == "bar_interpolation":
                lengths = (2, 4) if split == "train" else (3,)
            height = (8, 14)
            if regime == "smaller_query":
                height = (11, 14) if split == "train" else (8, 11)
            stage = str(rng.choice(("horizontal", "vertical", "diagonal")))
            if regime == "placement":
                stage = base_stage if split == "train" else {"horizontal": "vertical", "vertical": "diagonal", "diagonal": "horizontal"}[base_stage]
            profile = {"kind": kind, "slope": slope, "count": number, "height": height,
                       "lengths": lengths, "stage": stage,
                       "material": int(rng.integers(3, 9)) if changing_material else material,
                       "rotation": (rotation + 2 * int(rng.integers(2))) % 4}
            pair = _make_pair(rng, profile, used)
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in records:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert 9 not in out and not np.array_equal(inp, out)
            info = _measure(inp, out)
            assert info["cross_sheet"] == len(info["segments"])
            assert info["covered_guides"] == 2 * info["bar_length"]
            assert info["without_B_diff"]
            records[split].append(info)
    training = records["train"]
    assert set().union(*({x["slopes"][i] for i in x["active"].values()} for x in training)) == {-1, 1}
    assert any(x["unbacked_on"] and x["backed_beyond"] and x["backing_diff"] and x["finite_diff"] for x in training)
    assert any(x["native_edges"] == 0 and x["active"] for x in training)
    assert any(x["primary_diff"] for x in training)
    assert any(x["contact_toggle_diff"] for x in training)
    assert any(len(x["active"]) >= 2 and x["backed_off"] and x["gap"] and x["jump_diff"] for x in training)
    return True


def self_test():
    """42 streamed seeds; no grids, traces, or per-seed files are retained."""
    fingerprints, shapes, structures, materials = set(), set(), set(), set()
    lengths, counts, overlaps, ray_lengths, flags = set(), set(), set(), set(), Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                original = np.asarray(pair["input"]).copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solve mutated its input"
                info = _measure(pair["input"], pair["output"])
                shapes.add(_key(pair["input"])[0])
                norm = pair["input"].copy()
                norm[norm == info["material"]] = 3
                structures.add(hashlib.sha256(norm.tobytes()).hexdigest())
                materials.add(info["material"])
                lengths.add(info["bar_length"])
                counts.add(len(info["segments"]))
                overlaps.add(len(info["overlap"]))
                ray_lengths.update(map(len, info["rays"].values()))
                for name in ("gap", "backing_diff", "finite_diff", "primary_diff", "contact_toggle_diff", "jump_diff", "box_diff"):
                    flags[name] += bool(info[name])
                flags["no_active"] += not info["active"]
                flags["unpaired"] += bool(info["unpaired"])
    assert len(fingerprints) == 42 and len(structures) >= 190 and len(shapes) >= 70
    assert materials == set(range(3, 9)) and lengths == {2, 3, 4, 5} and counts == {1, 2, 3}
    assert len(overlaps) >= 30 and len(ray_lengths) >= 7
    assert all(flags[name] >= 10 for name in ("gap", "backing_diff", "finite_diff", "primary_diff", "contact_toggle_diff", "jump_diff", "box_diff", "no_active"))
    assert flags["unpaired"] >= 1
    # Reproducibility and all legal explicit count combinations, streamed.
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(1000 + train * 10 + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    assert _fingerprint(generate_puzzle(42)) == _fingerprint(generate_puzzle(42))
    for train, test in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # The validator must reject incorrect labels, including plausible shortcuts.
    puzzle = generate_puzzle(0, 2, 1)
    for mode in ("backing", "finite", "primary", "jump", "no_B"):
        for pair in puzzle["train"]:
            a, b, material, axis, _ = _decode(pair["input"])
            options = {"backing": {"activation": "all_on_segment"}, "finite": {"activation": "infinite"},
                       "primary": {"primary_mode": "extend"},
                       "jump": {"ray_mode": "jump"}, "no_B": {"bridges": False}}[mode]
            wrong = _circuit(a, b, material, **options)[0]
            wrong = wrong.T.copy() if axis else wrong
            if not np.array_equal(wrong, pair["output"]):
                try:
                    _measure(pair["input"], wrong)
                except AssertionError:
                    break
                raise AssertionError("semantic mutant accepted")
        else:
            raise AssertionError("missing negative witness")
    return {"seeds": 42, "pairs": 210, "distinct_structures": len(structures),
            "input_shapes": len(shapes), "materials": sorted(materials),
            "bar_lengths": sorted(lengths), "segment_counts": sorted(counts), "features": dict(flags)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
