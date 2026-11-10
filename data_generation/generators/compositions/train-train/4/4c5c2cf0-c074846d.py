"""Mirror docks: reflected endpoint permissions for clockwise hinged arms.

Only NumPy and the Python standard library are used.  solve() reads no generator
state, seed, examples, or files.  Invalid/ambiguous scenes raise ValueError.
"""

from itertools import permutations
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
X = ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (0, 275964627, 780720261, 703237123, 435593102, 516624278,
              739121163, 724883519, 958944737, 879700234, 247642196,
              571762478, 283254281, 880489251, 546130038, 138972709,
              722377995, 241335425, 248471057, 570331086, 998343562,
              383112172, 86983898, 911966063, 8845003, 128201643,
              325599029, 936078494, 385829326, 455709157, 870577162,
              592380765, 62859163, 303664518, 611787397, 250547822,
              187471936, 354547680, 665094517, 953997072, 492048514,
              815012331)


def _int(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside allowed range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a rectangular integer ARC grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _cells(a, color):
    return {tuple(map(int, p)) for p in np.argwhere(a == color)}


def _inside(points, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in points)


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _ray(pivot, direction, length):
    dr, dc = DIRS[direction]
    r, c = pivot
    return tuple((r + k * dr, c + k * dc) for k in range(1, length + 1))


def _arm(pivot, direction, length):
    source = _ray(pivot, direction, length)
    dest = _ray(pivot, (direction + 1) % 4, length)
    return {"pivot": pivot, "direction": direction, "length": length,
            "source": source, "dest": dest, "tip": dest[-1]}


def _orbit(docks, center):
    a, b = center
    return {(rr, cc) for r, c in docks
            for rr in (r, 2 * a - r) for cc in (c, 2 * b - c)}


def _rotations(docks, center):
    a, b = center
    offsets = {(r - a, c - b) for r, c in docks}
    result = set()
    for _ in range(4):
        result.update((a + r, b + c) for r, c in offsets)
        offsets = {(c, -r) for r, c in offsets}
    return result


def _box_orbit(docks, center):
    """The deliberately incorrect filled-bounding-box alternative."""
    rs, cs = [r for r, c in docks], [c for r, c in docks]
    box = {(r, c) for r in range(min(rs), max(rs) + 1)
           for c in range(min(cs), max(cs) + 1)}
    return _orbit(box, center)


def _quadrant(point, center):
    r, c = point
    a, b = center
    if r == a or c == b:
        return None
    return (-1 if r < a else 1, -1 if c < b else 1)


def _x_candidates(a):
    result = []
    for color in map(int, np.unique(a)):
        points = _cells(a, color)
        if len(points) != 5:
            continue
        for r, c in sorted(points):
            if points == {(r + dr, c + dc) for dr, dc in X}:
                result.append((color, (r, c)))
                break
    return result


def _parse(input_grid):
    """Resolve roles by exact geometry, including the family's collision rules."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 5 or int(counts.max()) * 2 <= a.size:
        raise ValueError("five roles and a strictly dominant background required")
    bg = int(colors[int(np.argmax(counts))])
    candidates = _x_candidates(a)
    if len(candidates) != 1:
        raise ValueError("X marker must be unique")
    marker, center = candidates[0]
    if marker == bg:
        raise ValueError("background cannot be the X")
    marker_cells = _cells(a, marker)
    rest = [int(c) for c in colors if c not in (bg, marker)]
    interpretations = []
    for dock_color, hinge_color, arm_color in permutations(rest):
        docks = _cells(a, dock_color)
        quads = {_quadrant(p, center) for p in docks}
        if (len(quads) != 1 or None in quads
                or any(abs(r - center[0]) < 2 or abs(c - center[1]) < 2
                       for r, c in docks)):
            continue
        targets = _orbit(docks, center)
        if not _inside(targets, a.shape):
            continue
        hinges = _cells(a, hinge_color)
        arm_cells = _cells(a, arm_color)
        if not 2 <= len(hinges) <= 6:
            continue
        arms, used = [], set()
        valid = True
        for pivot in sorted(hinges):
            attached = []
            for d in range(4):
                length = 0
                while _ray(pivot, d, length + 1)[-1] in arm_cells:
                    length += 1
                if length:
                    attached.append((d, length))
            if len(attached) != 1 or not 2 <= attached[0][1] <= 7:
                valid = False
                break
            d, length = attached[0]
            arm = _arm(pivot, d, length)
            if used & set(arm["source"]):
                valid = False
                break
            used.update(arm["source"])
            arms.append(arm)
        if not valid or used != arm_cells:
            continue
        static = docks | marker_cells
        previous_sources, previous_dest = set(), set()
        for arm in arms:
            source = {arm["pivot"], *arm["source"]}
            dest = set(arm["dest"])
            if (source & targets or _halo(source) & (static | previous_sources)
                    or not _inside(dest, a.shape)
                    or any(int(a[p]) != bg for p in dest)
                    or dest & previous_dest):
                valid = False
                break
            previous_sources.update(source)
            previous_dest.update(dest)
            arm["moves"] = arm["tip"] in targets
        if valid:
            interpretations.append({"grid": a, "background": bg,
                                    "marker": marker, "center": center,
                                    "dock_color": dock_color, "hinge_color": hinge_color,
                                    "arm_color": arm_color, "docks": docks,
                                    "targets": targets, "quadrant": next(iter(quads)),
                                    "arms": arms})
    if len(interpretations) != 1:
        raise ValueError("dock, hinge and arm roles must have one interpretation")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Transform the input; no generation context is consulted."""
    info = _parse(input_grid)
    out = info["grid"].copy()
    for arm in info["arms"]:
        if arm["moves"]:
            for p in arm["source"]:
                out[p] = info["dock_color"]
            for p in arm["dest"]:
                out[p] = info["arm_color"]
    return out


def _motif(rng):
    """Unanchored masks: maxima need not occur at the same corner."""
    rows, cols = int(rng.integers(1, 7)), int(rng.integers(1, 7))
    low_r = int(rng.integers(2, 13 - rows))
    low_c = int(rng.integers(2, 13 - cols))
    universe = [(r, c) for r in range(rows) for c in range(cols)]
    mode = int(rng.integers(6))
    if mode < 3:
        density = (0.22, 0.55, 0.88)[mode]
        cells = {p for p in universe if rng.random() < density}
    elif mode == 3:
        cells = {universe[int(rng.integers(len(universe)))]}
        count = int(rng.integers(1, len(universe) + 1))
        while len(cells) < count:
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr, dc in DIRS} & set(universe) - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    elif mode == 4:
        if rng.integers(2):
            row = int(rng.integers(rows))
            cells = {(row, c) for c in range(cols)}
        else:
            col = int(rng.integers(cols))
            cells = {(r, col) for r in range(rows)}
    else:
        cells = {universe[int(rng.integers(len(universe)))]}
    if not cells:
        cells = {universe[int(rng.integers(len(universe)))]}
    return {(r + low_r, c + low_c) for r, c in cells}


def _candidate(rng, kind, info, lengths, existing):
    """Construct one arm backwards from its hypothetical endpoint."""
    center, docks, targets = info["center"], info["docks"], info["targets"]
    static, shape = info["static"], info["shape"]
    sr, sc = info["quadrant"]
    if kind in ("H", "V", "B"):
        desired = {"H": (-sr, sc), "V": (sr, -sc), "B": (-sr, -sc)}[kind]
        tips = sorted(p for p in targets if _quadrant(p, center) == desired)
    elif kind == "match":
        tips = sorted(targets - docks)
    elif kind == "rot":
        tips = sorted(_rotations(docks, center) - targets)
    elif kind == "hole":
        tips = sorted(_box_orbit(docks, center) - targets)
    elif kind == "cross":
        tips = sorted(targets)
    else:
        near = {(r + dr, c + dc) for r, c in targets for dr, dc in DIRS}
        tips = sorted(near - targets)
    if not tips:
        return None
    occupied = set().union(*({a["pivot"], *a["source"]} for a in existing)) if existing else set()
    future = set().union(*(set(a["dest"]) for a in existing)) if existing else set()
    # Chebyshev adjacency is symmetric. Expand the fixed obstacles once,
    # instead of allocating a halo around every rejected candidate arm.
    blocked_source = _halo(static | occupied) | targets | future
    blocked_dest = static | occupied | future
    for _ in range(1100):
        length = int(rng.choice(lengths))
        d = int(rng.integers(4))
        vr, vc = DIRS[(d + 1) % 4]
        tip = tips[int(rng.integers(len(tips)))]
        if kind == "cross":
            # Put this dock strictly inside the candidate destination.
            k = int(rng.integers(1, length))
            tip = (tip[0] + (length - k) * vr, tip[1] + (length - k) * vc)
        pivot = (tip[0] - length * vr, tip[1] - length * vc)
        dr, dc = DIRS[d]
        source_end = (pivot[0] + length * dr, pivot[1] + length * dc)
        if not _inside((pivot, source_end, tip), shape):
            continue
        arm = _arm(pivot, d, length)
        source, dest = {pivot, *arm["source"]}, set(arm["dest"])
        if source & blocked_source or dest & blocked_dest:
            continue
        moves = tip in targets
        if moves != (kind in ("H", "V", "B", "match")):
            continue
        arm["moves"] = moves
        return arm
    return None


def _make_pair(rng, quadrant, colors, kinds, lengths, varied_lengths=False):
    for _ in range(160):
        motif = _motif(rng)
        reach_r, reach_c = max(r for r, c in motif), max(c for r, c in motif)
        minimum = 19 if len(kinds) >= 5 else 15
        h = int(rng.integers(max(minimum, 2 * reach_r + 1), 31))
        w = int(rng.integers(max(minimum, 2 * reach_c + 1), 31))
        center = (int(rng.integers(reach_r, h - reach_r)),
                  int(rng.integers(reach_c, w - reach_c)))
        sr, sc = quadrant
        docks = {(center[0] + sr * r, center[1] + sc * c) for r, c in motif}
        marker_cells = {(center[0] + r, center[1] + c) for r, c in X}
        info = {"center": center, "docks": docks, "targets": _orbit(docks, center),
                "static": docks | marker_cells, "shape": (h, w), "quadrant": quadrant}
        if "hole" in kinds and not (_box_orbit(docks, center) - info["targets"]):
            continue
        if "rot" in kinds and not (_rotations(docks, center) - info["targets"]):
            continue
        arms = []
        teaching_lengths = list(map(int, rng.choice(lengths, 2, replace=False))) if varied_lengths else []
        for index, kind in enumerate(kinds):
            if varied_lengths and index < 2:
                allowed = [teaching_lengths[index]]
            elif varied_lengths and index == 2:
                # A success and the overshoot share length in the same scene.
                allowed = [teaching_lengths[0]]
            else:
                allowed = lengths
            arm = _candidate(rng, kind, info, allowed, arms)
            if arm is None:
                break
            arms.append(arm)
        if len(arms) != len(kinds):
            continue
        bg, marker, dock_color, hinge_color, arm_color = colors
        inp = np.full((h, w), bg, dtype=np.int8)
        for p in marker_cells:
            inp[p] = marker
        for p in docks:
            inp[p] = dock_color
        out = inp.copy()
        for arm in arms:
            inp[arm["pivot"]] = out[arm["pivot"]] = hinge_color
            for p in arm["source"]:
                inp[p] = arm_color
                out[p] = dock_color if arm["moves"] else arm_color
            if arm["moves"]:
                for p in arm["dest"]:
                    out[p] = arm_color
        try:
            parsed = _parse(inp)
        except ValueError:
            continue
        if sum(a["moves"] for a in parsed["arms"]) != sum(a["moves"] for a in arms):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _int("seed", seed, 0)
    nt = _int("num_train", num_train, 2, 10)
    ns = _int("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # quadrant, population, or arm reach
    palette_mode = int(rng.integers(3))
    base_colors = tuple(map(int, rng.permutation(10)[:5]))
    held_index = int(rng.integers(4))
    train_quads = [q for i, q in enumerate(SIGNS) if i != held_index]
    rng.shuffle(train_quads)
    train = []
    test = []
    seen = set()
    for index in range(nt + ns):
        is_test = index >= nt
        if regime == 0:
            quadrant = SIGNS[held_index] if is_test else train_quads[index % 3]
        else:
            quadrant = SIGNS[int(rng.integers(4))]
        lengths = list(range(5, 8)) if regime == 2 and is_test else list(range(2, 5 if regime == 2 else 8))
        if not is_test and index == 0:
            kinds = ["H", "V", "cross"]
        elif not is_test and index == 1:
            kinds = ["B", "rot", "hole"]
        else:
            kinds = ["match", ("cross", "rot", "miss")[int(rng.integers(3))]]
        low = 5 if regime == 1 and is_test else len(kinds)
        high = 4 if regime == 1 and not is_test else 6
        count = int(rng.integers(low, high + 1))
        for _ in range(count - len(kinds)):
            kinds.append("match" if rng.random() < 0.55 else "miss")
        # The mandatory witnesses remain first while constrained objects are packed.
        if palette_mode == 0:
            colors = base_colors
        elif palette_mode == 1:
            colors = (base_colors[0],) + tuple(map(int, rng.permutation(
                [c for c in range(10) if c != base_colors[0]])[:4]))
        else:
            colors = tuple(map(int, rng.permutation(10)[:5]))
        for _ in range(8):
            pair = _make_pair(rng, quadrant, colors, kinds, lengths,
                              varied_lengths=not is_test and index == 0)
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("unable to generate distinct input")
        seen.add(key)
        (test if is_test else train).append(pair)
    rng.shuffle(train)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        raise ValueError("each pair must contain exactly input/output")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _parse(inp)
    assert inp.shape == out.shape
    assert np.array_equal(solve(inp), out)
    active = [a for a in info["arms"] if a["moves"]]
    inactive = [a for a in info["arms"] if not a["moves"]]
    assert active and inactive
    moved_length = sum(a["length"] for a in active)
    assert np.count_nonzero(out == info["dock_color"]) == len(info["docks"]) + moved_length
    assert np.count_nonzero(inp != out) == 2 * moved_length
    assert np.count_nonzero(out == info["arm_color"]) == np.count_nonzero(inp == info["arm_color"])
    # The orbit is consulted, never painted; no unused target changes.
    assert all(out[p] == inp[p] for p in info["targets"]
               if not any(p in a["dest"] for a in active))
    assert all(a["tip"] not in info["docks"] for a in info["arms"])
    info["active"] = active
    info["inactive"] = inactive
    return info


def _witnesses(info):
    sr, sc = info["quadrant"]
    codes = {(-sr, sc): "H", (sr, -sc): "V", (-sr, -sc): "B"}
    found = {codes[_quadrant(a["tip"], info["center"])] for a in info["active"]}
    rotated = _rotations(info["docks"], info["center"])
    if any(set(a["dest"][:-1]) & info["targets"] for a in info["inactive"]):
        found.add("cross")
    if any(a["tip"] in rotated for a in info["inactive"]):
        found.add("rot")
    if any(a["tip"] in _box_orbit(info["docks"], info["center"])
           for a in info["inactive"]):
        found.add("hole")
    return found


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must contain exactly train/test")
    if any(not isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    if not (2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns):
        raise ValueError("invalid pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(info["grid"])
            assert key not in seen
            seen.add(key)
            analyses[split].append(info)
    train, test = analyses["train"], analyses["test"]
    assert {"H", "V", "B", "cross", "rot", "hole"} <= set().union(*map(_witnesses, train))
    assert any({a["length"] for a in t["active"]} &
               {a["length"] for a in t["inactive"]} for t in train)
    lengths = {a["length"] for t in train for a in t["arms"]}
    assert len(lengths) >= 2
    train_quads = {t["quadrant"] for t in train}
    held_out = all(t["quadrant"] not in train_quads for t in test)
    population = max(len(t["arms"]) for t in train) <= 4 and min(len(t["arms"]) for t in test) >= 5
    reach = max(lengths) <= 4 and min(a["length"] for t in test for a in t["arms"]) >= 5
    assert held_out or population or reach


def _fingerprint(puzzle):
    return tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded coverage across the 42 host seeds, plus API and geometry checks."""
    shapes, palettes, layouts, counts, directions, lengths, quadrants = (set() for _ in range(7))
    motifs, successes, shifts, fingerprints = set(), set(), set(), set()
    rectangles = off_center = boundaries = small_motifs = large_motifs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        # Whole-scene quarter turns preserve clockwise handedness. Palette
        # permutations preserve every semantic role without changing geometry.
        example = puzzle["train"][0]
        for k in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(example["input"], k)),
                                  np.rot90(example["output"], k))
        recolor = np.array([9, 2, 7, 0, 6, 4, 1, 8, 5, 3], dtype=np.int8)
        assert np.array_equal(solve(recolor[example["input"]]), recolor[example["output"]])
        infos = {s: [_check_pair(p) for p in puzzle[s]] for s in ("train", "test")}
        train_quads = {i["quadrant"] for i in infos["train"]}
        if all(i["quadrant"] not in train_quads for i in infos["test"]):
            shifts.add("quadrant")
        if max(len(i["arms"]) for i in infos["train"]) < min(len(i["arms"]) for i in infos["test"]):
            shifts.add("population")
        if max(a["length"] for i in infos["train"] for a in i["arms"]) < min(
                a["length"] for i in infos["test"] for a in i["arms"]):
            shifts.add("reach")
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"])
                h, w = before.shape
                shapes.add((h, w))
                palettes.add(tuple(info[k] for k in ("background", "marker", "dock_color", "hinge_color", "arm_color")))
                counts.add(len(info["arms"]))
                successes.add(len(info["active"]))
                quadrants.add(info["quadrant"])
                ar, ac = info["center"]
                motif = tuple(sorted((abs(r-ar), abs(c-ac)) for r, c in info["docks"]))
                motifs.add(motif)
                layouts.add((before.shape, info["center"], info["quadrant"], motif,
                             tuple((a["pivot"], a["direction"], a["length"], a["moves"]) for a in info["arms"])))
                rectangles += h != w
                off_center += (2 * ar != h - 1 or 2 * ac != w - 1)
                small_motifs += len(motif) <= 5
                large_motifs += len(motif) > 15
                for a in info["arms"]:
                    directions.add(a["direction"])
                    lengths.add(a["length"])
                    boundaries += any(r in (0, h-1) or c in (0, w-1)
                                      for r, c in (*a["source"], *a["dest"], a["pivot"]))
    assert len(fingerprints) == 42
    assert len(shapes) >= 45 and len(layouts) >= 180 and len(motifs) >= 100
    assert len(palettes) >= 80 and counts == {2, 3, 4, 5, 6}
    assert directions == {0, 1, 2, 3} and lengths == set(range(2, 8))
    assert len(quadrants) == 4 and len(successes) >= 4
    assert shifts == {"quadrant", "population", "reach"}
    assert min(rectangles, off_center, boundaries, small_motifs, large_motifs) > 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(101 + nt * 10 + ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(11, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    assert _fingerprint(generate_puzzle(777, 5, 3)) == _fingerprint(generate_puzzle(777, 5, 3))
    # Hand geometry: reflections of (2,5) differ from quarter rotations.
    center, docks = (10, 10), {(12, 15)}
    target = _orbit(docks, center)
    assert (12, 5) in target and (15, 8) not in target
    assert (15, 8) in _rotations(docks, center)
    overshoot = _arm((12, 3), 0, 4)  # north arm, clockwise destination to the east
    assert (12, 5) in overshoot["dest"][:-1] and overshoot["tip"] not in target
    return {"seeds": 42, "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "layouts": len(layouts), "motifs": len(motifs),
            "palettes": len(palettes), "arm_counts": sorted(counts),
            "arm_lengths": sorted(lengths), "successful_arm_counts": sorted(successes),
            "shift_modes": sorted(shifts), "rectangles": rectangles,
            "off_center": off_center, "boundary_contacts": boundaries,
            "small_motifs": small_motifs, "large_motifs": large_motifs}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
