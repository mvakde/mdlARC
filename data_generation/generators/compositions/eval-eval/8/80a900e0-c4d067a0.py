"""Endpoint-gated blueprint stamping. Only NumPy and the standard library.

Generation draws labels from logical plan coordinates. The public solver instead
recovers the scene and walks the cap's physical endpoint rays. Neither imports
parents nor reads examples, files, RNG state, or split information.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np


# Rotating the complete scene also rotates the miniature, so transfer remains
# in the miniature's visible row/column coordinates.
_DIRECTIONS = ((1, 1), (-1, 1), (-1, -1), (1, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _diamond(center, radius):
    r, c = center
    return [(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius]


def _cap(center, radius, direction):
    r, c = center
    vr, vc = direction
    return [(r + vr * (k + 1), c + vc * (radius - k + 1))
            for k in range(radius + 1)]


def _inside(point, shape):
    return all(0 <= x < n for x, n in zip(point, shape))


def _patch(center, side):
    r, c = center
    t = side // 2
    return (slice(r - t, r + t + 1), slice(c - t, c + t + 1))


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for rr, cc in np.argwhere(mask):
        start = (int(rr), int(cc))
        if seen[start]:
            continue
        stack, cells = [start], []
        seen[start] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if _inside(point, mask.shape) and mask[point] and not seen[point]:
                    seen[point] = True
                    stack.append(point)
        result.append(cells)
    return result


def _matches(plan, sequence, direction):
    vr, vc = direction
    result = []
    for r, c in np.ndindex(plan.shape):
        end = (r + (len(sequence) - 1) * vr, c + (len(sequence) - 1) * vc)
        if _inside(end, plan.shape):
            if tuple(int(plan[r + k * vr, c + k * vc])
                     for k in range(len(sequence))) == tuple(sequence):
                result.append((r, c))
    return result


def _on_ray(point, endpoint, direction):
    # A point must be strictly beyond the cap, never on or behind it.
    u = (point[0] - endpoint[0]) * direction[0]
    v = (point[1] - endpoint[1]) * direction[1]
    return u == v and u > 0


def _recover(inp):
    """Input-only segmentation, diagonal sequence matching and lattice recovery."""
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > inp.size // 2, "background must occupy a strict majority"
    h, w = inp.shape
    candidates = []
    # Radius is the measured lattice pitch. Complete outlines are required.
    for radius in (4, 5, 6):
        if min(h, w) < 2 * radius + 1:
            continue
        shape = (h - 2 * radius, w - 2 * radius)
        offsets = _diamond((0, 0), radius)
        dr, dc = offsets[0]
        colors = inp[radius + dr:radius + dr + shape[0],
                     radius + dc:radius + dc + shape[1]]
        mask = colors != bg
        for dr, dc in offsets[1:]:
            mask &= inp[radius + dr:radius + dr + shape[0],
                        radius + dc:radius + dc + shape[1]] == colors
        for rr, cc in np.argwhere(mask):
            center = (int(rr) + radius, int(cc) + radius)
            for direction in _DIRECTIONS:
                cap = _cap(center, radius, direction)
                if not all(_inside(p, inp.shape) for p in cap):
                    continue
                cap_color = int(inp[cap[0]])
                if cap_color not in (bg, int(colors[rr, cc])) and all(
                        inp[p] == cap_color for p in cap):
                    candidates.append((center, radius, int(colors[rr, cc]),
                                       direction, cap_color, cap))
    assert len(candidates) == 1, "one complete diamond with one active cap is required"
    center, radius, structural, direction, marker, cap = candidates[0]
    object_mask = np.zeros(inp.shape, dtype=bool)
    for p in _diamond(center, radius) + cap:
        object_mask[p] = True
    # Structural colors are reserved locally, preventing unrelated pixels from
    # masquerading as part of the projector. They can change between examples.
    assert np.all(((inp == structural) | (inp == marker)) == object_mask)
    rest = (inp != bg) & ~object_mask
    swatches, squares = [], []
    for cells in _components(rest):
        if len(cells) == 1:
            swatches.append(cells[0])
            continue
        points = np.asarray(cells)
        r, c = (int(x) for x in points.min(axis=0))
        er, ec = (int(x) + 1 for x in points.max(axis=0))
        side = er - r
        assert side in (3, 5) and ec - c == side and len(cells) == side * side
        color = int(inp[r, c])
        assert np.all(inp[r:er, c:ec] == color), "calibration blocks must be solid"
        squares.append(((r + side // 2, c + side // 2), side, color))
    assert len(swatches) >= 4 and 2 <= len(squares) <= 4
    points = np.asarray(swatches)
    mr, mc = (int(x) for x in points.min(axis=0))
    er, ec = (int(x) + 1 for x in points.max(axis=0))
    assert np.all((points - (mr, mc)) % 2 == 0)
    assert all(np.any(np.diff(np.unique(points[:, d])) == 2) for d in (0, 1))
    plan = inp[mr:er:2, mc:ec:2].copy()
    assert all(3 <= n <= 7 for n in plan.shape)
    mini_mask = np.zeros(inp.shape, dtype=bool)
    mini_mask[mr:er:2, mc:ec:2] = True
    assert np.count_nonzero(plan != bg) == len(swatches)
    assert np.all(inp[mr:er, mc:ec][~mini_mask[mr:er, mc:ec]] == bg)
    endpoints = (cap[0], cap[-1])
    rails = [j for j, e in enumerate(endpoints)
             if all(_on_ray(s[0], e, direction) for s in squares)]
    assert len(rails) == 1, "all clues must occupy one endpoint ray"
    squares.sort(key=lambda s: s[0][0] * direction[0])
    side = squares[0][1]
    assert all(s[1] == side for s in squares)
    deltas = np.diff(np.asarray([s[0] for s in squares]), axis=0)
    pitch = int(deltas[0, 0] * direction[0])
    assert pitch > side and np.all(deltas == np.asarray(direction) * pitch)
    assert pitch == radius, "outline radius and measured block pitch must agree"
    endpoint = endpoints[rails[0]]
    assert (squares[0][0][0] - endpoint[0]) % pitch == 0, "endpoints are lattice centers"
    sequence = tuple(s[2] for s in squares)
    matches = _matches(plan, sequence, direction)
    assert len(matches) == 1, "the complete ordered clue must have one diagonal match"
    match = matches[0]
    origin = tuple(squares[0][0][d] - pitch * match[d] for d in (0, 1))
    half = side // 2
    assert all(half <= origin[d] and
               origin[d] + (plan.shape[d] - 1) * pitch + half < inp.shape[d]
               for d in (0, 1)), "the full virtual lattice must fit"
    selected = set()
    for site in np.ndindex(plan.shape):
        point = tuple(origin[d] + site[d] * pitch for d in (0, 1))
        if any(_on_ray(point, e, direction) for e in endpoints):
            selected.add(site)
    seed_sites = {tuple(match[d] + k * direction[d] for d in (0, 1))
                  for k in range(len(sequence))}
    assert seed_sites <= selected
    expected_input = np.full(inp.shape, bg, dtype=np.int8)
    expected_input[object_mask] = inp[object_mask]
    expected_input[mr:er:2, mc:ec:2] = plan
    for site in selected:
        point = tuple(origin[d] + site[d] * pitch for d in (0, 1))
        patch = _patch(point, side)
        assert not np.any(object_mask[patch])
        # Reserve even blank selected sites, so holes are visibly empty.
        assert not (point[0] + half >= mr and point[0] - half < er
                    and point[1] + half >= mc and point[1] - half < ec)
        if site in seed_sites:
            expected_input[patch] = plan[site]
    assert np.array_equal(inp, expected_input), "unexplained foreground or invalid clue layout"
    return dict(background=bg, center=center, radius=radius, direction=direction,
                structural=structural, marker=marker, cap=cap, endpoints=endpoints,
                object_mask=object_mask, miniature=(mr, mc, er, ec), plan=plan,
                side=side, pitch=pitch, sequence=sequence, match=match, origin=origin,
                selected=selected, seed_sites=seed_sites, rail=rails[0])


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    a = _recover(inp)
    out = inp.copy()
    # Walk physical rays, then ask whether a visited pixel is a lattice center.
    # This differs from the logical diagonal mask used to generate labels.
    for endpoint in a["endpoints"]:
        r, c = endpoint
        vr, vc = a["direction"]
        r, c = r + vr, c + vc
        while _inside((r, c), inp.shape):
            dr, dc = r - a["origin"][0], c - a["origin"][1]
            if dr % a["pitch"] == 0 and dc % a["pitch"] == 0:
                site = (dr // a["pitch"], dc // a["pitch"])
                if _inside(site, a["plan"].shape) and a["plan"][site] != a["background"]:
                    out[_patch((r, c), a["side"])] = a["plan"][site]
            r, c = r + vr, c + vc
    return out


def _logical_tracks(h, w, a, b):
    tracks = [[], []]
    for k in range(1, 16):
        for j, site in enumerate(((a + k, b + 1 + k), (a + 1 + k, b + k))):
            if _inside(site, (h, w)):
                tracks[j].append(site)
    return tracks


def _grow(mask):
    padded = np.pad(mask, 1)
    result = np.zeros(mask.shape, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            result |= padded[dr:dr + mask.shape[0], dc:dc + mask.shape[1]]
    return result


@lru_cache(maxsize=None)
def _geometry_options(pitch, side):
    """Finite feasible geometries, unrelated to solver state or example labels."""
    center = (pitch + 1, pitch + 1)
    objects = _diamond(center, pitch) + _cap(center, pitch, (1, 1))
    half = side // 2
    options = []
    for h in range(3, 8):
        for w in range(3, 8):
            for a in range(-2, 2):
                for b in range(-2, 2):
                    origin = (pitch + 2 - a * pitch, pitch + 2 - b * pitch)
                    if any(origin[d] < half or origin[d] + (n - 1) * pitch + half >= 30
                           for d, n in enumerate((h, w))):
                        continue
                    tracks = _logical_tracks(h, w, a, b)
                    if min(map(len, tracks)) < 2:
                        continue
                    selected = set(tracks[0] + tracks[1])
                    occupied = np.zeros((30, 30), dtype=bool)
                    for point in objects:
                        occupied[point] = True
                    for site in selected:
                        point = tuple(origin[d] + site[d] * pitch for d in (0, 1))
                        occupied[_patch(point, side)] = True
                    probes = []
                    for k in range(1, 16):
                        site = (a + k, b + k)
                        if _inside(site, (h, w)):
                            point = tuple(origin[d] + site[d] * pitch for d in (0, 1))
                            if not np.any(occupied[_patch(point, side)]):
                                probes.append(site)
                    if not probes:
                        continue
                    probe = probes[0]
                    point = tuple(origin[d] + probe[d] * pitch for d in (0, 1))
                    occupied[_patch(point, side)] = True
                    grown = _grow(occupied)
                    mh, mw = 2 * h - 1, 2 * w - 1
                    integral = np.pad(grown.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
                    sums = (integral[mh:, mw:] - integral[:-mh, mw:]
                            - integral[mh:, :-mw] + integral[:-mh, :-mw])
                    placements = tuple(tuple(map(int, p)) for p in np.argwhere(sums == 0))
                    if placements:
                        options.append(dict(shape=(h, w), origin=origin, center=center,
                                            tracks=tracks, selected=selected, probe=probe,
                                            mini_positions=placements))
    assert options, (pitch, side)
    return tuple(options)


def _plan(rng, geometry, seed_sites, palette, bg, force_hole, teaching, attempts=80,
          required_active=()):
    shape = geometry["shape"]
    ordered_seed = list(seed_sites)
    available = sorted(geometry["selected"] - set(ordered_seed))
    hole_choices = [p for p in available if p not in required_active]
    hole = hole_choices[int(rng.integers(len(hole_choices)))] if force_hole else None
    probe = geometry["probe"]

    def acceptable(plan):
        if any(np.all(edge == bg) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
            return False
        points = np.argwhere(plan != bg)
        if not all(np.any(np.diff(np.unique(points[:, d])) == 1) for d in (0, 1)):
            return False
        if any(plan[p] == bg for p in ordered_seed) or plan[probe] == bg:
            return False
        if any(plan[p] == bg for p in required_active):
            return False
        seq = tuple(int(plan[p]) for p in ordered_seed)
        if _matches(plan, seq, (1, 1)) != [ordered_seed[0]]:
            return False
        # Repeated first colors ensure that recognizing just one swatch cannot
        # generally replace the ordered match.
        if np.count_nonzero(plan == seq[0]) < 2:
            return False
        additions = {int(plan[p]) for p in available} - {bg}
        return bool(additions) and (not teaching or len(additions) >= 2)

    for _ in range(attempts):
        plan = rng.choice(palette, size=shape).astype(np.int8)
        if rng.random() < .3:
            axis = int(rng.integers(2))
            i, j = map(int, rng.choice(shape[axis], 2, replace=False))
            if axis == 0:
                plan[i] = plan[j]
            else:
                plan[:, i] = plan[:, j]
        holes = rng.random(shape) < float(rng.choice([0., .08, .18, .3]))
        plan[holes] = bg
        for p in ordered_seed:
            plan[p] = int(rng.choice(palette))
        for p in required_active:
            plan[p] = int(rng.choice(palette))
        plan[probe] = int(rng.choice(palette))
        if hole is not None:
            plan[hole] = bg
        if acceptable(plan):
            return plan
    # Constructive fallback: only one STARTABLE location has color palette[1].
    # Its repeat on the final row cannot begin an occupied length >=2 diagonal.
    plan = np.full(shape, palette[0], dtype=np.int8)
    plan[ordered_seed[0]] = palette[1]
    repeats = [(shape[0] - 1, c) for c in range(shape[1])
               if (shape[0] - 1, c) not in geometry["selected"] | {probe}]
    assert repeats
    plan[repeats[0]] = palette[1]
    if hole is not None:
        plan[hole] = bg
    additions = [p for p in available if p != hole]
    plan[additions[0]] = palette[2]
    # Ensure two colors in additions even if one happened to be the repeat.
    if teaching and len({int(plan[p]) for p in additions}) < 2:
        plan[additions[-1]] = palette[0]
    assert acceptable(plan), "constructive plan fallback must retain teaching evidence"
    return plan


def _make_pair(rng, rotation, kind, desired_rail):
    if kind == "small_witness":
        pitch, side = 4, 3
        length = int(rng.choice([2, 3]))
    elif kind == "large_witness":
        pitch, side, length = 6, 5, 2
    elif kind == "interior_test":
        pitch, side, length = 4, 3, 2
    elif kind == "late_test":
        pitch, side, length = int(rng.choice([4, 5])), 3, 2
    else:
        pitch = int(rng.choice([4, 5, 6]))
        side = 5 if pitch == 6 and rng.random() < .5 else 3
        length = int(rng.choice([2, 3, 4], p=[.6, .3, .1])) if pitch == 4 else (
            int(rng.choice([2, 3])) if pitch == 5 else 2)
    # A quarter turn swaps the geometric cap endpoints.
    rail = desired_rail ^ (rotation % 2)
    options = []
    for g in _geometry_options(pitch, side):
        n = len(g["tracks"][rail])
        total = len(g["selected"])
        if kind == "small_witness" and (n < length + 1 or total - length < 3):
            continue
        if kind == "interior_test" and n < 4:
            continue
        if kind == "late_test" and n < 3:
            continue
        if n >= length and total - length >= 2:
            options.append(g)
    assert options, (kind, pitch, side, length, rail)
    g = options[int(rng.integers(len(options)))]
    track = g["tracks"][rail]
    if kind == "interior_test":
        start = int(rng.integers(1, len(track) - length))
    elif kind == "late_test":
        start = int(rng.integers(1, len(track) - length + 1))
    elif kind in ("small_witness", "large_witness"):
        start = 0
    else:
        start = int(rng.integers(len(track) - length + 1))
    seed_sites = track[start:start + length]
    colors = list(map(int, rng.permutation(10)))
    bg, structural, marker = colors[:3]
    palette = colors[3:3 + int(rng.integers(3, 7))]
    teaching = kind in ("small_witness", "large_witness")
    required_active = ([track[start + length]] if kind == "small_witness" else
                       [track[start - 1], track[start + length]] if kind == "interior_test" else
                       [track[start - 1]] if kind == "late_test" else [])
    plan = _plan(rng, g, seed_sites, palette, bg,
                 force_hole=kind == "small_witness", teaching=teaching,
                 required_active=required_active)
    mini = g["mini_positions"][int(rng.integers(len(g["mini_positions"])))]
    half = side // 2
    required = [max(g["center"][d] + pitch + 2,
                    g["origin"][d] + (g["shape"][d] - 1) * pitch + half + 1,
                    mini[d] + 2 * g["shape"][d] - 1) for d in (0, 1)]
    shape = tuple(30 if rng.random() < .2 else
                  int(rng.integers(n, min(30, n + 3) + 1)) for n in required)
    # Translate the entire scene within any unused top/left margin.
    offset = tuple(int(rng.integers(shape[d] - required[d] + 1)) for d in (0, 1))
    center = tuple(g["center"][d] + offset[d] for d in (0, 1))
    origin = tuple(g["origin"][d] + offset[d] for d in (0, 1))
    mini = tuple(mini[d] + offset[d] for d in (0, 1))
    inp = np.full(shape, bg, dtype=np.int8)
    for point in _diamond(center, pitch):
        inp[point] = structural
    for point in _cap(center, pitch, (1, 1)):
        inp[point] = marker
    inp[mini[0]:mini[0] + 2 * plan.shape[0]:2,
        mini[1]:mini[1] + 2 * plan.shape[1]:2] = plan
    for site in seed_sites:
        point = tuple(origin[d] + site[d] * pitch for d in (0, 1))
        inp[_patch(point, side)] = plan[site]
    out = inp.copy()
    # Ground truth uses the sampled logical tracks; it never calls solve.
    for site in g["selected"]:
        if plan[site] != bg:
            point = tuple(origin[d] + site[d] * pitch for d in (0, 1))
            out[_patch(point, side)] = plan[site]
    return {"input": np.rot90(inp, rotation).copy(),
            "output": np.rot90(out, rotation).copy()}


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    bank_size = int(rng.integers(1, min(4, nt) + 1))
    rotations = list(map(int, rng.choice(4, bank_size, replace=False)))
    training_rotations = [rotations[i % bank_size] for i in range(nt)]
    rng.shuffle(training_rotations)
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            if split == "train":
                rotation = training_rotations[index]
                kind = ("small_witness" if index == 0 else "large_witness" if index == 1
                        else "ordinary")
                rail = index % 2
            else:
                rotation = int(rng.choice(rotations))
                kind = "interior_test" if rng.random() < .65 else "late_test"
                rail = int(rng.integers(2))
            for _ in range(96):
                pair = _make_pair(rng, rotation, kind, rail)
                key = _key(pair["input"])
                # A demonstration must make the ordered sequence necessary,
                # even after using all visible geometric constraints. Tests
                # must likewise defeat an alignment rule that ignores colors.
                a = _recover(pair["input"])
                alternatives = _alignment_ablation(pair["input"], a)
                informative = (alternatives["first_color_wrong"] > 0 if kind == "small_witness"
                               else alternatives["wrong"] > 0 if split == "test" else True)
                if key not in seen and informative:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def _features(inp, a=None):
    a = _recover(inp) if a is None else a
    bg, side, p = a["background"], a["side"], a["pitch"]
    added = a["selected"] - a["seed_sites"]
    active_added = [s for s in added if a["plan"][s] != bg]
    probes = []
    # Middle cap pixels are potential whole-cap-ray origins. A occupied plan
    # site on such a ray, with an empty patch, distinguishes endpoint gating.
    for site in np.ndindex(a["plan"].shape):
        if site in a["selected"] or a["plan"][site] == bg:
            continue
        point = tuple(a["origin"][d] + site[d] * p for d in (0, 1))
        if any(_on_ray(point, e, a["direction"]) for e in a["cap"][1:-1]):
            if np.all(inp[_patch(point, side)] == bg):
                probes.append(site)
    track = sorted([s for s in a["selected"] if _on_ray(
        tuple(a["origin"][d] + s[d] * p for d in (0, 1)),
        a["endpoints"][a["rail"]], a["direction"])],
        key=lambda s: s[0] * a["direction"][0])
    first = track.index(a["match"])
    first_only = _matches(a["plan"], a["sequence"][:1], a["direction"])
    return dict(shape=inp.shape, plan_shape=a["plan"].shape,
                background=bg, pitch=p, side=side, direction=a["direction"],
                rail=a["rail"], length=len(a["sequence"]), match=a["match"],
                origin=a["origin"], miniature=a["miniature"][:2],
                selected=len(a["selected"]), selected_holes=sum(a["plan"][s] == bg for s in a["selected"]),
                holes=int(np.count_nonzero(a["plan"] == bg)),
                added_colors=len({int(a["plan"][s]) for s in active_added}),
                additions=len(active_added), interior_probes=len(probes),
                prefix=first == 0, missing_before=first > 0,
                missing_after=first + len(a["sequence"]) < len(track),
                active_before=any(a["plan"][s] != bg for s in track[:first]),
                active_after=any(a["plan"][s] != bg for s in track[first + len(a["sequence"]):]),
                first_color_matches=len(first_only),
                equal_seed=len(set(a["sequence"])) == 1,
                layout=(inp.shape, a["plan"].shape, a["center"], a["origin"],
                        a["miniature"], p, side, a["direction"], a["match"], len(a["sequence"])))


def _alignment_ablation(inp, a=None):
    """Counterexamples to color-free and first-color-only lattice alignment.

    Enumerate geometrically admissible placements, keeping the supplied blocks
    unchanged even when their omitted color-match constraint would reject the
    placement. This is a diagnostic alternative rule, not a second solver.
    """
    a = _recover(inp) if a is None else a
    expected = inp.copy()
    for site in a["selected"]:
        if a["plan"][site] != a["background"]:
            point = tuple(a["origin"][d] + site[d] * a["pitch"] for d in (0, 1))
            expected[_patch(point, a["side"])] = a["plan"][site]
    seed_centers = {tuple(a["origin"][d] + site[d] * a["pitch"] for d in (0, 1))
                    for site in a["seed_sites"]}
    first = tuple(a["origin"][d] + a["match"][d] * a["pitch"] for d in (0, 1))
    half = a["side"] // 2
    mr, mc, er, ec = a["miniature"]
    count = wrong = first_wrong = 0
    for match in np.ndindex(a["plan"].shape):
        end = tuple(match[d] + (len(a["sequence"]) - 1) * a["direction"][d]
                    for d in (0, 1))
        if not _inside(end, a["plan"].shape):
            continue
        origin = tuple(first[d] - match[d] * a["pitch"] for d in (0, 1))
        if any(origin[d] < half or
               origin[d] + (a["plan"].shape[d] - 1) * a["pitch"] + half >= inp.shape[d]
               for d in (0, 1)):
            continue
        candidate = inp.copy()
        valid = True
        for site in np.ndindex(a["plan"].shape):
            point = tuple(origin[d] + site[d] * a["pitch"] for d in (0, 1))
            if not any(_on_ray(point, e, a["direction"]) for e in a["endpoints"]):
                continue
            patch = _patch(point, a["side"])
            if (np.any(a["object_mask"][patch]) or
                    (point[0] + half >= mr and point[0] - half < er
                     and point[1] + half >= mc and point[1] - half < ec)):
                valid = False
                break
            if point not in seed_centers and a["plan"][site] != a["background"]:
                candidate[patch] = a["plan"][site]
        if valid:
            count += 1
            if not np.array_equal(candidate, expected):
                wrong += 1
                first_wrong += int(a["plan"][match] == a["sequence"][0])
    return {"geometric_placements": count, "wrong": wrong, "first_color_wrong": first_wrong}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    ablations = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            a = _recover(inp)
            expected = inp.copy()
            for site in a["selected"]:
                if a["plan"][site] != a["background"]:
                    point = tuple(a["origin"][d] + site[d] * a["pitch"] for d in (0, 1))
                    expected[_patch(point, a["side"])] = a["plan"][site]
            assert np.array_equal(out, expected), "incorrect gate, transfer, hole, or preservation"
            assert np.array_equal(out, solve(inp)), "physical walking disagrees with selected-site rendering"
            assert np.array_equal(out[inp != a["background"]], inp[inp != a["background"]])
            mr, mc, er, ec = a["miniature"]
            assert np.array_equal(out[mr:er, mc:ec], inp[mr:er, mc:ec])
            for site in a["selected"]:
                if a["plan"][site] == a["background"]:
                    point = tuple(a["origin"][d] + site[d] * a["pitch"] for d in (0, 1))
                    assert np.all(out[_patch(point, a["side"])] == a["background"])
            features[split].append(_features(inp, a))
            ablations[split].append(_alignment_ablation(inp, a))
    trains, tests = features["train"], features["test"]
    assert {f["side"] for f in trains} >= {3, 5}, "teach measured block size"
    assert {f["rail"] for f in trains} == {0, 1}, "teach either endpoint as the calibration rail"
    assert any(f["selected_holes"] > 0 for f in trains), "teach empty selected sites"
    assert any(f["interior_probes"] > 0 for f in trains), "teach endpoint-only projection"
    assert sum(f["added_colors"] >= 2 for f in trains) >= 2, "teach literal swatch colors"
    assert all(f["first_color_matches"] > 1 for f in trains + tests)
    assert {f["direction"] for f in tests} <= {f["direction"] for f in trains}
    assert all(f["missing_before"] for f in tests), "test clues occur later on a ray"
    assert all(f["active_before"] for f in tests), "test requires a stamp before the clue"
    assert any(f["prefix"] and f["active_after"] for f in trains), "teach sampling beyond the clue"
    assert any(f["first_color_wrong"] > 0 for f in ablations["train"]), "teach full sequence matching"
    assert all(f["wrong"] > 0 for f in ablations["test"]), "test alignment must depend on colors"


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for s in ("train", "test") for pair in puzzle[s])


def self_test() -> dict:
    """Bounded 42-seed family audit, count checks, and deliberate corruptions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    rows = {s: [_features(_grid(pair["input"])) for p in puzzles for pair in p[s]]
            for s in ("train", "test")}
    all_rows = rows["train"] + rows["test"]
    assert {f["direction"] for f in all_rows} == set(_DIRECTIONS)
    assert {f["pitch"] for f in all_rows} == {4, 5, 6}
    assert {f["side"] for f in all_rows} == {3, 5}
    assert {f["rail"] for f in rows["test"]} == {0, 1}
    assert len({f["plan_shape"] for f in all_rows}) >= 10
    assert len({f["shape"] for f in all_rows}) >= 12
    assert len({f["match"] for f in rows["test"]}) >= 10
    assert len({f["origin"] for f in rows["test"]}) >= 10
    assert len({f["layout"] for f in all_rows}) >= 150
    assert {f["background"] for f in all_rows} == set(range(10))
    assert any(f["holes"] == 0 for f in all_rows) and any(f["holes"] >= 3 for f in all_rows)
    assert any(f["selected_holes"] > 0 for f in rows["test"])
    assert any(f["missing_before"] and f["missing_after"] for f in rows["test"])
    assert {f["length"] for f in all_rows} >= {2, 3, 4}
    # Directly exercise the constructive fallback, including its unique match
    # and the two-color/selected-hole teaching obligations.
    for force_hole, pitch, side in ((True, 4, 3), (False, 6, 5)):
        geometry = next(g for g in _geometry_options(pitch, side)
                        if len(g["tracks"][0]) >= (3 if force_hole else 2)
                        and len(g["selected"]) >= (5 if force_hole else 4))
        seeds = geometry["tracks"][0][:2]
        plan = _plan(np.random.default_rng(999), geometry, seeds,
                     [1, 2, 3], 0, force_hole, True, attempts=0)
        assert _matches(plan, tuple(plan[s] for s in seeds), (1, 1)) == [seeds[0]]
        added = geometry["selected"] - set(seeds)
        assert len({int(plan[s]) for s in added} - {0}) >= 2
        assert not force_hole or any(plan[s] == 0 for s in added)
    # Every legal explicit count; boundary conditions must not change the rule.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(4200 + nt * 10 + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Test both missing content and illicit whole-cap illumination.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    pair = bad["test"][0]
    cell = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][cell] = pair["input"][cell]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a missing stamped pixel")
    pair = next(p for p in puzzles[0]["train"] if _features(p["input"])["interior_probes"])
    a = _recover(pair["input"])
    for site in np.ndindex(a["plan"].shape):
        point = tuple(a["origin"][d] + site[d] * a["pitch"] for d in (0, 1))
        if (site not in a["selected"] and a["plan"][site] != a["background"]
                and any(_on_ray(point, e, a["direction"]) for e in a["cap"][1:-1])
                and np.all(pair["input"][_patch(point, a["side"])] == a["background"])):
            damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
                       for s in ("train", "test")}
            index = next(i for i, p in enumerate(puzzles[0]["train"]) if p is pair)
            damaged["train"][index]["output"][_patch(point, a["side"])] = a["plan"][site]
            try:
                validate_generated_puzzle(damaged)
            except AssertionError:
                break
            raise AssertionError("validator accepted an interior-lane square")
    else:
        raise AssertionError("missing endpoint-only negative-test witness")
    return {"seeds": 42, "pairs": len(all_rows),
            "plan_shapes": len({f["plan_shape"] for f in all_rows}),
            "canvas_shapes": len({f["shape"] for f in all_rows}),
            "geometry_layouts": len({f["layout"] for f in all_rows}),
            "test_matches": len({f["match"] for f in rows["test"]}),
            "directions": len({f["direction"] for f in all_rows}),
            "tests_with_before_and_after": sum(f["missing_before"] and f["missing_after"]
                                               for f in rows["test"])}


if __name__ == "__main__":
    print(self_test())
