"""Symmetry-addressed corner exchange. Standard library + NumPy; no file I/O.

A marker pairs *anchors*, not world-oriented ornament pixels. Each partner's
L arms define its own coordinate system. Union the two observed local masks
and restore that union at both anchors, retaining each corner's own color.
"""
from __future__ import annotations

import hashlib
from numbers import Integral
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _connected8(points):
    if not points:
        return False
    reached = {min(points)}
    todo = list(reached)
    while todo:
        r, c = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r+dr, c+dc)
                if q in points and q not in reached:
                    reached.add(q)
                    todo.append(q)
    return len(reached) == len(points)


def _support(anchor, direction, lengths):
    r, c = anchor
    sr, sc = direction
    ar, ac = lengths
    return ({(r+sr*k, c) for k in range(ar+1)}
            | {(r, c+sc*k) for k in range(ac+1)})


def _window(anchor, direction):
    r, c = anchor
    sr, sc = direction
    return {(r+sr*u, c+sc*v) for u in range(1, 4) for v in range(1, 4)}


def _world(corner, mask):
    r, c = corner["anchor"]
    sr, sc = corner["direction"]
    return {(r+sr*u, c+sc*v) for u, v in mask}


def _read_supports(points):
    corners = []
    for component in _components(points):
        rows, cols = zip(*component)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        lengths = (r1-r0, c1-c0)
        if not all(4 <= n <= 6 for n in lengths):
            raise ValueError("A corner must have two arms of length 4..6")
        candidates = []
        for r, sr in ((r0, 1), (r1, -1)):
            for c, sc in ((c0, 1), (c1, -1)):
                if _support((r, c), (sr, sc), lengths) == component:
                    candidates.append({"anchor": (r, c), "direction": (sr, sc),
                                       "lengths": lengths, "support": component})
        if len(candidates) != 1:
            raise ValueError("Support component is not one unambiguous L")
        corners.extend(candidates)
    if len(corners) not in (2, 4, 6):
        raise ValueError("Expected one to three pairs of corner carriers")
    return corners


def _kind(marker):
    if len(marker) == 1:
        return "P"
    rows, cols = zip(*marker)
    if len(marker) <= 5:
        if min(rows) == max(rows) and max(cols)-min(cols)+1 == len(marker):
            return "H"
        if min(cols) == max(cols) and max(rows)-min(rows)+1 == len(marker):
            return "V"
    raise ValueError("Marker must be a point or a straight segment of length 2..5")


def _symmetric(point, marker, kind):
    r, c = point
    a, b = min(marker)
    if kind == "H":
        return 2*a-r, c
    if kind == "V":
        return r, 2*b-c
    return 2*a-r, 2*b-c


def _marker_pairs(marker, kind, corners):
    pairs = []
    rows, cols = zip(*marker)
    for i, first in enumerate(corners):
        r, c = first["anchor"]
        # A finite line covers the anchors' projection; its endpoints are not
        # stencil rulers. Points must connect diagonally separated anchors.
        if kind == "H" and not (r != rows[0] and min(cols) <= c <= max(cols)):
            continue
        if kind == "V" and not (c != cols[0] and min(rows) <= r <= max(rows)):
            continue
        if kind == "P" and (r == rows[0] or c == cols[0]):
            continue
        partner = _symmetric((r, c), marker, kind)
        for j in range(i+1, len(corners)):
            if corners[j]["anchor"] == partner:
                pairs.append((i, j))
    return pairs


def _parse_with_frame(a, background, frame):
    supports = {tuple(map(int, p)) for p in np.argwhere(a == frame)}
    corners = _read_supports(supports)
    owners = {}
    for index, corner in enumerate(corners):
        for p in _window(corner["anchor"], corner["direction"]):
            if not (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]):
                raise ValueError("Corner window leaves canvas")
            if p in owners or p in supports:
                raise ValueError("Corner windows or supports overlap")
            owners[p] = index
    foreground = {tuple(map(int, p)) for p in np.argwhere(a != background)}
    outside = foreground - supports - set(owners)
    marker_colors = {int(a[p]) for p in outside}
    if len(marker_colors) != 1:
        raise ValueError("Expected one marker color outside corner windows")
    marker_color = marker_colors.pop()
    if marker_color in (background, frame):
        raise ValueError("Structural roles coincide")
    if any(int(a[p]) == marker_color for p in owners):
        raise ValueError("Marker enters a corner's ornament window")
    for index, corner in enumerate(corners):
        cells = {p for p, owner in owners.items() if owner == index and a[p] != background}
        colors = {int(a[p]) for p in cells}
        if len(colors) != 1:
            raise ValueError("Each corner needs one nonempty monochrome fragment")
        corner["color"] = colors.pop()
        r, c = corner["anchor"]
        sr, sc = corner["direction"]
        corner["clue"] = frozenset(((x-r)*sr, (y-c)*sc) for x, y in cells)
    groups, used = [], set()
    for marker in _components(outside):
        kind = _kind(marker)
        candidates = _marker_pairs(marker, kind, corners)
        if len(candidates) != 1:
            raise ValueError("Marker has missing or ambiguous anchor correspondence")
        i, j = candidates[0]
        if i in used or j in used:
            raise ValueError("A corner has more than one partner")
        used.update((i, j))
        mask = corners[i]["clue"] | corners[j]["clue"]
        if not (3 <= len(mask) <= 9 and _connected8(mask)):
            raise ValueError("Union must be a connected 3..9-cell local stencil")
        groups.append({"kind": kind, "marker": marker, "indices": (i, j), "mask": mask})
    if len(used) != len(corners):
        raise ValueError("Unpaired corner")
    return {"background": background, "frame": frame, "marker_color": marker_color,
            "corners": corners, "groups": groups}


def _read_input(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= a.size//2:
        raise ValueError("Background must occupy more than half the input")
    interpretations = []
    for frame in range(10):
        if frame == background or counts[frame] < 18:
            continue
        try:
            interpretations.append(_parse_with_frame(a, background, frame))
        except ValueError:
            pass
    if len(interpretations) != 1:
        raise ValueError("Missing or ambiguous corner/marker interpretation")
    return a, interpretations[0]


def _complete(a, info):
    out = a.copy()
    for group in info["groups"]:
        for index in group["indices"]:
            corner = info["corners"][index]
            for p in _world(corner, group["mask"]):
                out[p] = corner["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, partners, and local unions entirely from this input."""
    a, info = _read_input(input_grid)
    return _complete(a, info)


def _halo(points):
    return {(r+dr, c+dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _prototype(rng, kind, mismatch=False, compact=False, adjacent=False):
    for _ in range(80):
        lengths = [(int(rng.integers(4, 7)), int(rng.integers(4, 7))) for _ in range(2)]
        directions = [tuple(map(int, rng.choice((-1, 1), 2))) for _ in range(2)]
        if compact:
            lengths = [(4, 4), (4, 4)]
            directions = {"H": [(1, 1), (-1, 1)],
                          "V": [(1, 1), (1, -1)],
                          "P": [(1, 1), (-1, -1)]}[kind]
        if mismatch:
            sr, sc = directions[0]
            expected = {"H": (-sr, sc), "V": (sr, -sc), "P": (-sr, -sc)}[kind]
            flip = 1 if compact and kind == "H" else 0
            if not compact:
                flip = int(rng.integers(2))
            directions[1] = tuple(-v if k == flip else v for k, v in enumerate(expected))
        if adjacent:
            sr, sc = directions[0]
            directions[1] = (sr, -sc)
            if compact and kind == "H":
                directions = [(1, 1), (-1, 1)]
            elif compact and kind == "V":
                directions = [(1, 1), (1, -1)]
        if kind == "H":
            d = 6 if compact else int(rng.integers(5, 9))
            anchors = [(-d, 0), (d, 0)]
        elif kind == "V":
            d = 6 if compact else int(rng.integers(5, 9))
            anchors = [(0, -d), (0, d)]
        else:
            dr, dc = (5, 5) if compact else map(int, rng.integers(5, 9, 2))
            # Both diagonals occur; local directions remain independent.
            sign = 1 if compact else int(rng.choice((-1, 1)))
            anchors = [(-dr, -sign*dc), (dr, sign*dc)]
        if kind == "P":
            marker = {(0, 0)}
        else:
            n = 2 if compact else int(rng.integers(2, 6))
            start = 0 if compact else -int(rng.integers(n))
            marker = {(0, k) if kind == "H" else (k, 0) for k in range(start, start+n)}
        carriers, footprints = [], []
        for anchor, direction, arms in zip(anchors, directions, lengths):
            support = _support(anchor, direction, arms)
            window = _window(anchor, direction)
            carriers.append({"anchor": anchor, "direction": direction,
                             "lengths": arms, "support": support})
            footprints.append(support | window)
        parts = footprints + [marker]
        if any(_halo(parts[i]) & parts[j] for i in range(3) for j in range(i)):
            continue
        footprint = set().union(*parts)
        min_r, min_c = min(r for r, _ in footprint), min(c for _, c in footprint)
        def move(points):
            return {(r-min_r, c-min_c) for r, c in points}
        for corner in carriers:
            r, c = corner["anchor"]
            corner["anchor"] = (r-min_r, c-min_c)
            corner["support"] = move(corner["support"])
        footprint = move(footprint)
        height = max(r for r, _ in footprint)+1
        width = max(c for _, c in footprint)+1
        if max(height, width) <= 30:
            return {"kind": kind, "corners": carriers, "marker": move(marker),
                    "footprint": footprint, "shape": (height, width)}
    raise RuntimeError("Could not construct separated corner carriers")


def _pack(rng, prototypes, shape, deterministic=False):
    h, w = shape
    blocked, placed = set(), []
    order = sorted(range(len(prototypes)), key=lambda i: -np.prod(prototypes[i]["shape"]))
    for index in order:
        proto = prototypes[index]
        ph, pw = proto["shape"]
        origins = [(r, c) for r in range(h-ph+1) for c in range(w-pw+1)]
        if not deterministic:
            rng.shuffle(origins)
        for r, c in origins:
            points = {(r+x, c+y) for x, y in proto["footprint"]}
            if not points & blocked:
                break
        else:
            return None
        placed.append((index, r, c))
        blocked |= _halo(points)
    return placed


def _stencil(rng, forbidden, asymmetric=False):
    for _ in range(80):
        h, w = map(int, rng.integers(2, 4, 2))
        origin_r = int(rng.integers(1, 5-h))
        origin_c = int(rng.integers(1, 5-w))
        domain = {(r, c) for r in range(origin_r, origin_r+h)
                  for c in range(origin_c, origin_c+w)}
        choices = sorted(domain)
        points = {choices[int(rng.integers(len(choices)))]}
        size = int(rng.integers(3, len(domain)+1))
        while len(points) < size:
            frontier = sorted((domain & _halo(points))-points)
            points.add(frontier[int(rng.integers(len(frontier)))])
        mask = frozenset(points)
        if mask not in forbidden and (not asymmetric or mask != frozenset((v, u) for u, v in mask)):
            return mask
    # There are far more legal masks than the three forbidden per scene.
    for omit in range(9):
        mask = frozenset((r, c) for r in range(1, 4) for c in range(1, 4)
                         if (r-1)*3+c-1 != omit)
        if mask not in forbidden and (not asymmetric or mask != frozenset((v, u) for u, v in mask)):
            return mask
    raise RuntimeError("Stencil fallback exhausted")


def _fragments(rng, mask, mode):
    cells = sorted(mask)
    rng.shuffle(cells)
    if mode == "reference":
        count = int(rng.integers(1, len(cells)))
        first, second = set(cells), set(cells[:count])
    elif mode == "overlap":
        overlap = int(rng.integers(1, len(cells)-1))
        cut = int(rng.integers(1, len(cells)-overlap))
        common, rest = set(cells[:overlap]), cells[overlap:]
        first, second = common | set(rest[:cut]), common | set(rest[cut:])
    elif mode == "disjoint":
        cut = int(rng.integers(1, len(cells)))
        first, second = set(cells[:cut]), set(cells[cut:])
    else:
        raise ValueError("Unknown fragment mode")
    return (second, first) if rng.random() < .5 else (first, second)


def _make_pair(rng, kinds, roles, palette, mode, force_mismatch=False):
    for attempt in range(18):
        compact = attempt >= 14
        prototypes = [_prototype(rng, kind, force_mismatch and i == 0, compact,
                                 adjacent=force_mismatch and i == 1)
                      for i, kind in enumerate(kinds)]
        if compact:
            shape = (30, 30)
        else:
            low = 17 if len(kinds) == 1 else 24
            shape = tuple(map(int, rng.integers(low, 31, 2)))
        placed = _pack(rng, prototypes, shape, deterministic=attempt == 17)
        if placed is None:
            continue
        background, frame, marker = roles
        inp = np.full(shape, background, dtype=np.int8)
        out = inp.copy()
        seen_masks = set()
        for index, dr, dc in placed:
            proto = prototypes[index]
            mask = _stencil(rng, seen_masks, asymmetric=force_mismatch and index == 1)
            seen_masks.add(mask)
            clues = _fragments(rng, mask, mode)
            colors = (list(map(int, rng.choice(palette, 2, replace=False)))
                      if force_mismatch and index == 0
                      else [int(rng.choice(palette)) for _ in range(2)])
            for p in proto["marker"]:
                inp[p[0]+dr, p[1]+dc] = out[p[0]+dr, p[1]+dc] = marker
            for corner, clue, color in zip(proto["corners"], clues, colors):
                for r, c in corner["support"]:
                    inp[r+dr, c+dc] = out[r+dr, c+dc] = frame
                for r, c in _world(corner, mask):
                    out[r+dr, c+dc] = color
                for r, c in _world(corner, clue):
                    inp[r+dr, c+dc] = color
        try:
            _, info = _read_input(inp)
        except ValueError:
            continue
        if not np.array_equal(_complete(inp, info), out):
            continue
        if force_mismatch:
            witnesses = _witnesses(info)
            if not (witnesses["local_needed"] and witnesses["rotation_refuted"]):
                continue
        return {"input": inp, "output": out}
    raise RuntimeError("Bounded packing/readability attempts exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """One seed fixes all geometry, fragments, palette roles and pair order."""
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    order = list(map(int, rng.permutation(10)))
    roles = tuple(order[:3])
    pool = order[3:3+int(rng.integers(2, 5))]
    steady = bool(rng.random() < .4)
    kinds = list(map(str, rng.permutation(["H", "V", "P"])))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if split == "train" and index == 0:
                plan, mode = kinds[:2], "reference"
            elif split == "train" and index == 1:
                plan = [kinds[2]] + ([str(rng.choice(kinds))] if rng.random() < .65 else [])
                mode = "overlap"
            else:
                n = int(rng.integers(1 if split == "train" else 2, 4))
                plan = list(map(str, rng.choice(kinds, n)))
                mode = "disjoint" if split == "test" else str(rng.choice(["reference", "overlap"]))
            pair_roles = roles if steady else tuple(map(int, rng.choice(10, 3, replace=False)))
            available = [c for c in pool if c not in pair_roles]
            first_example = split == "train" and index == 0
            if first_example and len(available) < 2:
                available += [c for c in order if c not in pair_roles and c not in available][:2-len(available)]
            if not available:
                available = [c for c in range(10) if c not in pair_roles]
            palette = list(map(int, rng.choice(available, int(rng.integers(2 if first_example else 1,
                                                                         len(available)+1)), replace=False)))
            for _ in range(12):
                pair = _make_pair(rng, plan, pair_roles, palette, mode,
                                  force_mismatch=split == "train" and index == 0)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("Distinct-input retry budget exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _witnesses(info):
    local_needed = False
    rotation_refuted = False
    masks = {group["mask"] for group in info["groups"]}
    for group in info["groups"]:
        i, j = group["indices"]
        for source, target in ((i, j), (j, i)):
            first, second = info["corners"][source], info["corners"][target]
            if np.prod(first["direction"]) != np.prod(second["direction"]):
                rotated_union = second["clue"] | frozenset((v, u) for u, v in first["clue"])
                rotation_refuted |= rotated_union != group["mask"]
            needed = _world(second, group["mask"] - second["clue"])
            physical = {_symmetric(p, group["marker"], group["kind"])
                        for p in _world(first, first["clue"])}
            local_needed |= bool(needed - physical)
    return {"local_needed": local_needed, "pairing_needed": len(masks) > 1,
            "rotation_refuted": rotation_refuted}


def validate_generated_puzzle(puzzle):
    """Validate arrays plus observable ownership, unions, witnesses and shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, taught = set(), set()
    evidence = {"local_needed": False, "pairing_needed": False, "distributed": False,
                "reference": False, "rotation_refuted": False, "local_color": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = (inp.shape, inp.astype(np.int8).tobytes())
            assert key not in seen
            seen.add(key)
            _, info = _read_input(inp)
            assert np.array_equal(_complete(inp, info), out), "Input-derived union disagrees"
            changed = inp != out
            assert changed.any() and np.all(inp[changed] == info["background"])
            assert np.array_equal(out[inp != info["background"]], inp[inp != info["background"]])
            expected_changes = 0
            for group in info["groups"]:
                i, j = group["indices"]
                a, b = (info["corners"][k]["clue"] for k in (i, j))
                mask = group["mask"]
                expected_changes += 2*len(mask)-len(a)-len(b)
                assert a | b == mask
                if split == "train":
                    assert a & b, "Training retains a shared local clue"
                    taught.add(group["kind"])
                    evidence["reference"] |= a == mask or b == mask
                    evidence["distributed"] |= a != mask and b != mask
                    evidence["local_color"] |= (info["corners"][i]["color"] != info["corners"][j]["color"]
                                                 and (a != mask or b != mask))
                else:
                    assert not a & b, "Test uses disjoint complementary fragments"
                    assert group["kind"] in taught
            assert int(changed.sum()) == expected_changes
            if split == "train":
                for name, flag in _witnesses(info).items():
                    evidence[name] |= flag
    assert taught == {"H", "V", "P"}
    assert all(evidence.values()), "Demonstrations fail to expose a required mechanism"


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                a = np.asarray(pair[key], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _regression_tests():
    # Explicit coordinate fixture, not drawn with generation/transport helpers.
    # P pairs (1,1) with (11,11), but their Ls are not physical half-turns.
    inp = np.zeros((17, 17), dtype=np.int8)
    inp[1, 1:6] = inp[1:6, 1] = 5
    inp[11, 11:16] = inp[7:12, 11] = 5
    inp[6, 6] = 8
    for p in ((2, 2), (2, 3), (3, 2)):
        inp[p] = 2
    for p in ((9, 12), (8, 12)):
        inp[p] = 3
    expected = inp.copy()
    expected[4, 2] = 2
    expected[10, 12] = expected[10, 13] = 3
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(solve(expected), expected)
    # Same intended union, now with no shared local clue and no intact copy.
    disjoint = inp.copy()
    disjoint[3, 2] = 0
    assert np.array_equal(solve(disjoint), expected)
    ambiguous = inp.copy()
    ambiguous[6, 8] = 8  # A second, unpaired point cannot be silently ignored.
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("Unpaired marker was ignored")


def self_test():
    """42 seeds, streamed; no saved grids, caches, traces or unbounded retries."""
    _regression_tests()
    stats = {"shapes": set(), "stencils": set(), "directions": set(), "counts": set(),
             "roles": set(), "structures": set(), "digests": set(), "arm_lengths": set()}
    totals = {"pairs": 0, "groups": 0, "local_witness_pairs": 0, "pairing_witness_pairs": 0,
              "rotation_witness_pairs": 0,
              "reused_color_pairs": 0, "both_incomplete_groups": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        stats["digests"].add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                before = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(before, inp), "Solver mutated its observation"
                _, info = _read_input(inp)
                stats["shapes"].add(inp.shape)
                stats["roles"].add((info["background"], info["frame"], info["marker_color"]))
                stats["counts"].add(len(info["groups"]))
                for corner in info["corners"]:
                    stats["directions"].add(corner["direction"])
                    stats["arm_lengths"].update(corner["lengths"])
                colors = [c["color"] for c in info["corners"]]
                totals["reused_color_pairs"] += len(set(colors)) < len(colors)
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                stats["structures"].add((inp.shape, normalized))
                totals["pairs"] += 1
                totals["groups"] += len(info["groups"])
                witness = _witnesses(info)
                totals["local_witness_pairs"] += witness["local_needed"]
                totals["pairing_witness_pairs"] += witness["pairing_needed"]
                totals["rotation_witness_pairs"] += witness["rotation_refuted"]
                for group in info["groups"]:
                    stats["stencils"].add(tuple(sorted(group["mask"])))
                    totals["both_incomplete_groups"] += all(
                        info["corners"][k]["clue"] != group["mask"] for k in group["indices"])
    assert len(stats["digests"]) == 42
    assert len(stats["shapes"]) >= 30 and len(stats["stencils"]) >= 50
    assert len(stats["structures"]) == totals["pairs"]
    assert stats["counts"] == {1, 2, 3}
    assert len(stats["directions"]) == 4 and stats["arm_lengths"] == {4, 5, 6}
    assert len(stats["roles"]) >= 30
    assert totals["local_witness_pairs"] >= 42 and totals["pairing_witness_pairs"] >= 42
    assert totals["rotation_witness_pairs"] >= 42
    assert totals["both_incomplete_groups"] >= 84
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    # Coordinate transport commutes with every global square symmetry and
    # every palette permutation; neither fixed positions nor literal colors solve it.
    example = generate_puzzle(0)
    permutation = np.random.default_rng(123).permutation(10)
    for pair in example["train"] + example["test"]:
        for turns in range(4):
            for flip in (False, True):
                inp = np.rot90(pair["input"], turns)
                expected = np.rot90(pair["output"], turns)
                if flip:
                    inp, expected = np.fliplr(inp), np.fliplr(expected)
                assert np.array_equal(solve(permutation[inp]), permutation[expected])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1701+nt*10+nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # A non-completion edit must be caught by the family validator.
    bad = generate_puzzle(9, 2, 1)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Unrelated output edit accepted")
    return {"seeds": 42, **totals, **{key: len(value) for key, value in stats.items()}}
