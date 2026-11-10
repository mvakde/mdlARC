"""Resonant defect islands: input-only symmetry recovery and arrow addressing.

No parent module, saved puzzle, seed, or construction state is used by solve.
Only Python's standard library and NumPy are required.
"""
from itertools import combinations
import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1),
              (-1, -1), (-1, 1), (1, 1), (1, -1))
CARDINAL = DIRECTIONS[:4]
DIAGONAL = DIRECTIONS[4:]
DISTANCE = 5


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int64, copy=True)


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _orbit(p, shape):
    r, c = p
    h, w = shape
    return frozenset(((r, c), (h-1-r, c), (r, w-1-c), (h-1-r, w-1-c)))


def _components(points, diagonal=True):
    """Connected sets in a deterministic order; holes and arrows use eight neighbors."""
    left = set(points)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            near = {(r+dr, c+dc) for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if diagonal or abs(dr)+abs(dc) == 1} & left
            left -= near
            group |= near
            stack.extend(sorted(near))
        result.append(group)
    return result


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return (tip, direction, frozenset((r+y, c+x) for y, x in offsets))


def _endpoint(arrow):
    (r, c), (dr, dc), _ = arrow
    return r+DISTANCE*dr, c+DISTANCE*dc


def _separated(arrow, arrows):
    occupied = {p for obj in arrows for p in obj[2]}
    return all((r+dy, c+dx) not in occupied for r, c in arrow[2]
               for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def _halo(points, radius, shape):
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        mask[max(0, r-radius):min(shape[0], r+radius+1),
             max(0, c-radius):min(shape[1], c+radius+1)] = True
    return mask


def _decode(inp):
    colors, counts = np.unique(inp[inp != 0], return_counts=True)
    if len(colors) != 2 or counts[0] == counts[1] or not (inp == 0).any():
        raise ValueError("Expected field 0 and two distinguishable object colors")
    solid = int(colors[int(np.argmax(counts))])
    signal = int(next(c for c in colors if int(c) != solid))
    arrow_points = set(map(tuple, np.argwhere(inp == signal).tolist()))
    arrows = []
    for group in _components(arrow_points):
        if len(group) != 3:
            raise ValueError("Signal objects must be separated three-cell arrowheads")
        matches = [_arrow(p, d) for p in sorted(group) for d in DIRECTIONS
                   if _arrow(p, d)[2] == group]
        if len(matches) != 1:
            raise ValueError("Arrow direction is ambiguous")
        arrow = matches[0]
        r, c = _endpoint(arrow)
        if not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]):
            raise ValueError("Arrow endpoint is outside the canvas")
        arrows.append(arrow)
    if not arrows or np.count_nonzero(inp == solid) <= len(arrow_points):
        raise ValueError("Material must be the more abundant nonzero color")

    base = np.zeros(inp.shape, dtype=bool)
    seen = set()
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            p = (r, c)
            if p in seen:
                continue
            orbit = _orbit(p, inp.shape)
            seen |= orbit
            n = sum(int(inp[q] == solid) for q in orbit)
            if len(orbit) == 4:
                if n == 2:
                    raise ValueError("Reflection orbit has no majority")
                material = n >= 3
            else:
                if n not in (0, len(orbit)):
                    raise ValueError("Axis orbit must be uncorrupted")
                material = n == len(orbit)
            for q in orbit:
                base[q] = material
    if any(base[p] for p in arrow_points):
        raise ValueError("Arrowheads must lie in intended field")
    holes = set(map(tuple, np.argwhere(base & (inp == 0)).tolist()))
    extras = set(map(tuple, np.argwhere(~base & (inp == solid)).tolist()))
    return solid, signal, base, holes, extras, arrows


def _events(holes, arrows):
    counts = {}
    for obj in arrows:
        p = _endpoint(obj)
        counts[p] = counts.get(p, 0) + 1
    groups = _components(holes)
    levels = [min(2, max((counts.get(p, 0) for p in g), default=0))
              for g in groups]
    return counts, groups, levels


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles and scaffold, then broadcast each simultaneous hole hit."""
    inp = _grid(input_grid)
    solid, signal, base, holes, extras, arrows = _decode(inp)
    _, groups, levels = _events(holes, arrows)
    out = inp.copy()
    for p in extras:
        out[p] = 0
    paint = np.zeros(inp.shape, dtype=bool)
    for group, level in zip(groups, levels):
        if level:
            paint |= _halo(group, level, inp.shape)
    for p in holes:
        paint[p] = False
    # Arrowheads are persistent instructions, even inside a halo.
    paint[inp == signal] = False
    out[paint] = signal
    return out


def _symmetric(a):
    return a | a[::-1, :] | a[:, ::-1] | a[::-1, ::-1]


def _rectangular_scaffold(base):
    for group in _components(set(map(tuple, np.argwhere(base).tolist())), diagonal=False):
        rs, cs = zip(*group)
        if len(group) != (max(rs)-min(rs)+1)*(max(cs)-min(cs)+1):
            return False
    return True


def _make_base(rng, shape, style, edge_patch=False):
    """Freely sized rectangle compositions, followed by two reflections."""
    h, w = shape
    hh, ww = (h+1)//2, (w+1)//2
    a = np.zeros(shape, dtype=bool)
    count = 1 if style == 0 else int(rng.integers(2, 5))
    for _ in range(count):
        rh = int(rng.integers(3, min(7, hh)+1))
        rw = int(rng.integers(3, min(7, ww)+1))
        y = int(rng.integers(0, hh-rh+1))
        x = int(rng.integers(0, ww-rw+1))
        if style == 2 and rh >= 5 and rw >= 5:
            # L arms and frames introduce corners, cavities and narrow rims.
            thickness = int(rng.integers(2, 4))
            a[y:y+rh, x:x+thickness] = True
            a[y:y+thickness, x:x+rw] = True
            if rng.random() < .5:
                a[y+rh-thickness:y+rh, x:x+rw] = True
                a[y:y+rh, x+rw-thickness:x+rw] = True
        else:
            a[y:y+rh, x:x+rw] = True
    if edge_patch:
        # Room for two independent approach directions and clipped radius two.
        x = int(rng.integers(3, max(4, ww-3)))
        a[:3, x:x+4] = True
    return _symmetric(a)


def _candidate_arrow(endpoint, direction, base, arrows):
    r, c = endpoint
    dr, dc = direction
    obj = _arrow((r-DISTANCE*dr, c-DISTANCE*dc), direction)
    if any(not (0 <= y < base.shape[0] and 0 <= x < base.shape[1])
           or base[y, x] for y, x in obj[2]):
        return None
    return obj if _separated(obj, arrows) else None


def _available_holes(base, used):
    orbits = {_orbit(p, base.shape) for p in used}
    return {tuple(p) for p in np.argwhere(base).tolist()
            if len(_orbit(tuple(p), base.shape)) == 4
            and _orbit(tuple(p), base.shape) not in orbits
            and all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) > 1 for q in used)}


def _grow_group(rng, start, available, size, shape, diagonal_only=False):
    group = {start}
    orbits = {_orbit(start, shape)}
    while len(group) < size:
        frontier = sorted({(r+dy, c+dx) for r, c in group
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                          & (available-group))
        frontier = [p for p in frontier if _orbit(p, shape) not in orbits]
        if diagonal_only:
            frontier = [p for p in frontier if (sum(p)-sum(start)) % 2 == 0]
        if not frontier:
            return None
        p = frontier[int(rng.integers(len(frontier)))]
        group.add(p)
        orbits.add(_orbit(p, shape))
    return group


def _target_group(rng, base, used, arrows, pool, size, kind, clipped=False):
    available = _available_holes(base, used)
    starts = sorted(p for p in available if not clipped
                    or min(p[0], p[1], base.shape[0]-1-p[0],
                           base.shape[1]-1-p[1]) <= 1)
    if not starts:
        return None
    for _ in range(140):
        start = starts[int(rng.integers(len(starts)))]
        group = _grow_group(rng, start, available, size, base.shape,
                            diagonal_only=(kind == "distinct"))
        if group is None:
            continue
        endpoints = [start] if clipped else sorted(group)
        rng.shuffle(endpoints)
        options = []
        for p in endpoints:
            for d in pool:
                obj = _candidate_arrow(p, d, base, arrows)
                if obj is not None:
                    options.append(obj)
        if kind == "single" and options:
            return group, [options[int(rng.integers(len(options)))]]
        choices = []
        for first, second in combinations(options, 2):
            same = _endpoint(first) == _endpoint(second)
            if same != (kind == "collision") or not _separated(second, arrows+[first]):
                continue
            choices.append((first, second))
        if choices:
            chosen = list(choices[int(rng.integers(len(choices)))])
            if kind == "collision" and rng.random() < .3:
                third = [a for a in options if _endpoint(a) == _endpoint(chosen[0])
                         and _separated(a, arrows+chosen)]
                if third:
                    chosen.append(third[int(rng.integers(len(third)))])
            return group, chosen
    return None


def _idle_group(rng, base, used, size, near=None):
    available = _available_holes(base, used)
    if near is not None:
        available = {p for p in available
                     if min(max(abs(p[0]-q[0]), abs(p[1]-q[1])) for q in near) == 2}
    if not available:
        return None
    starts = sorted(available)
    for _ in range(40):
        start = starts[int(rng.integers(len(starts)))]
        g = _grow_group(rng, start, available, size, base.shape)
        if g is not None:
            return g
    return None


def _miss_arrow(rng, base, arrows, pool, forbidden):
    candidates = [tuple(p) for p in np.argwhere(~base & ~forbidden).tolist()
                  if len(_orbit(tuple(p), base.shape)) == 4]
    rng.shuffle(candidates)
    for p in candidates:
        dirs = list(pool)
        rng.shuffle(dirs)
        for d in dirs:
            obj = _candidate_arrow(p, d, base, arrows)
            if obj is not None and p not in {q for a in arrows+[obj] for q in a[2]}:
                return p, obj
    return None


def _render_label(base, holes, extras, arrows, solid, signal):
    """Construction label from explicit masks; does not recover input geometry."""
    inp = np.where(base, solid, 0).astype(np.int64)
    for p in holes:
        inp[p] = 0
    for p in extras:
        inp[p] = solid
    for obj in arrows:
        for p in obj[2]:
            inp[p] = signal
    out = np.where(base, solid, 0).astype(np.int64)
    # Count signals directly from construction vectors, before painting anything.
    ends = [_endpoint(a) for a in arrows]
    for group in _components(holes):
        strength = min(2, max((ends.count(p) for p in group), default=0))
        if strength:
            for r, c in group:
                out[max(0, r-strength):r+strength+1,
                    max(0, c-strength):c+strength+1] = signal
    for p in holes:
        out[p] = 0
    for obj in arrows:
        for p in obj[2]:
            out[p] = signal
    return {"input": inp, "output": out}


def _make_pair(rng, solid, signal, pool, teaching, style, large_groups,
               small_groups=False, scaffold_shift=None):
    for attempt in range(240):
        low = 22 if teaching else 16
        shape = tuple(int(rng.integers(low, 31)) for _ in range(2))
        base = _make_base(rng, shape, style, edge_patch=(teaching == "collision"))
        if not 40 <= int(base.sum()) <= .64*base.size:
            continue
        if scaffold_shift is not None and _rectangular_scaffold(base) != (scaffold_shift == "simple"):
            continue
        holes, arrows = set(), []
        if teaching == "distinct":
            plans = [(3, "distinct", False)]
        elif teaching == "collision":
            plans = [(int(rng.integers(2, 4)), "collision", True)]
        else:
            plans = [(int(rng.integers(4, 7) if large_groups
                          else rng.integers(1, 4 if small_groups else 5)),
                      str(rng.choice(("single", "collision"))), False)
                     for _ in range(int(rng.integers(1, 4)))]
        failed = False
        for size, kind, clipped in plans:
            target = _target_group(rng, base, holes, arrows, pool, size, kind, clipped)
            if target is None:
                failed = True
                break
            group, additions = target
            holes |= group
            arrows += additions
        if failed:
            continue
        # The second anchor protects an unaddressed hole inside a wide halo.
        if teaching or rng.random() < .65:
            idle = _idle_group(rng, base, holes, 1 if teaching == "collision"
                               else int(rng.integers(1, 4)),
                               near=holes if teaching == "collision" else None)
            if idle is None:
                continue
            holes |= idle
        if not teaching and rng.random() < .4:
            extra_idle = _idle_group(rng, base, holes, int(rng.integers(1, 4)))
            if extra_idle:
                holes |= extra_idle
        _, groups, levels = _events(holes, arrows)
        active_halo = np.zeros(shape, dtype=bool)
        for group, level in zip(groups, levels):
            if level:
                active_halo |= _halo(group, level, shape)
        extras = set()
        if teaching or rng.random() < .65:
            miss = _miss_arrow(rng, base, arrows, pool, active_halo)
            if miss is None:
                continue
            miss_point, miss_object = miss
            arrows.append(miss_object)
            # A shot into an extra speck is erased and never generates a halo.
            extras.add(miss_point)
        occupied = {p for obj in arrows for p in obj[2]}
        used_orbits = {_orbit(p, shape) for p in holes | extras}
        available = [tuple(p) for p in np.argwhere(~base).tolist()
                     if tuple(p) not in occupied and len(_orbit(tuple(p), shape)) == 4
                     and _orbit(tuple(p), shape) not in used_orbits]
        rng.shuffle(available)
        # Deliberately expose erasure/halo precedence as well as far erasure.
        if teaching == "collision":
            near = [p for p in available if active_halo[p]]
            if not near:
                continue
            p = near[0]
            extras.add(p)
            used_orbits.add(_orbit(p, shape))
        budget = int(rng.integers(1 if teaching else 0, 7))
        for p in available:
            if len(extras) >= budget:
                break
            orbit = _orbit(p, shape)
            if orbit not in used_orbits:
                extras.add(p)
                used_orbits.add(orbit)
        if int(base.sum())-len(holes)+len(extras) <= 3*len(arrows):
            continue
        pair = _render_label(base, holes, extras, arrows, solid, signal)
        # Construction and input interpretation must agree on the actual pixels.
        try:
            _, _, recovered, decoded_holes, decoded_extras, _ = _decode(pair["input"])
        except ValueError:
            continue
        if (not np.array_equal(base, recovered) or decoded_holes != holes
                or decoded_extras != extras):
            continue
        if not np.array_equal(solve(pair["input"]), pair["output"]):
            raise AssertionError("Construction and solver disagree")
        return pair
    raise RuntimeError("Could not construct a readable resonant defect scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(np.arange(1, 10))]
    signal, solid = palette[:2]
    vary_solid = bool(rng.integers(2))
    regime = int(rng.integers(3))
    if regime == 0:
        train_pool, test_pool = ((CARDINAL, DIAGONAL) if rng.integers(2)
                                 else (DIAGONAL, CARDINAL))
    else:
        train_pool = test_pool = DIRECTIONS
    puzzle, seen = {"train": [], "test": []}, set()
    # These logical demonstrations are shuffled, not tied to a displayed index.
    lessons = ["distinct", "collision"] + [None]*(num_train-2)
    rng.shuffle(lessons)
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            teaching = lessons[i] if split == "train" else None
            color = (palette[1+(i+(num_train if split == "test" else 0)) % 8]
                     if vary_solid else solid)
            style = (0 if split == "train" else 2) if regime == 2 else int(rng.integers(3))
            for _ in range(12):
                pair = _make_pair(rng, color, signal,
                                  train_pool if split == "train" else test_pool,
                                  teaching, style, large_groups=(regime == 1 and split == "test"),
                                  small_groups=(regime == 1 and split == "train"),
                                  scaffold_shift=("simple" if split == "train" else "composite")
                                  if regime == 2 else None)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    solid, signal, base, holes, extras, arrows = _decode(inp)
    counts, groups, levels = _events(holes, arrows)
    wide, ordinary = np.zeros(inp.shape, bool), np.zeros(inp.shape, bool)
    for g, level in zip(groups, levels):
        if level == 1:
            ordinary |= _halo(g, 1, inp.shape)
        elif level == 2:
            wide |= _halo(g, 2, inp.shape)
    clipped = any(level and any(min(p[0], p[1], inp.shape[0]-1-p[0],
                                        inp.shape[1]-1-p[1]) < level for p in g)
                  for g, level in zip(groups, levels))
    normal_distinct = any(level == 1 and sum(p in counts for p in g) >= 2
                          for g, level in zip(groups, levels))
    propagated = any(level and any(p not in counts for p in g)
                     for g, level in zip(groups, levels))
    diagonal_propagation = any(
        level and any(p not in counts and not any(abs(p[0]-q[0])+abs(p[1]-q[1]) == 1
                                                  for q in g) for p in g)
        for g, level in zip(groups, levels))
    return {"shape": inp.shape, "solid": solid, "signal": signal,
            "base": base, "holes": holes, "extras": extras, "arrows": arrows,
            "rectangular_scaffold": _rectangular_scaffold(base),
            "groups": groups, "levels": levels, "counts": counts,
            "normal_distinct": normal_distinct, "propagated": propagated,
            "diagonal_propagation": diagonal_propagation,
            "clipped": clipped, "wide": wide, "ordinary": ordinary,
            "unaddressed": any(level == 0 for level in levels),
            "miss_extra": any(p in extras for p in counts),
            "near_extra": any((wide | ordinary)[p] for p in extras),
            "far_extra": any(not (wide | ordinary)[p] for p in extras),
            "protected_idle": any(level == 0 and any(wide[p] for p in g)
                                   for g, level in zip(groups, levels))}


def _without_arrows(inp):
    """B-like shortcut: outline all holes at radius one, ignoring addressing."""
    _, signal, _, holes, extras, _ = _decode(inp)
    out = inp.copy()
    for p in extras:
        out[p] = 0
    paint = _halo(holes, 1, inp.shape)
    for p in holes:
        paint[p] = False
    out[paint & (inp != signal)] = signal
    return out


def _without_scaffold(inp):
    """A-like shortcut: paint just arrow endpoints and never classify defects."""
    _, signal, _, _, _, arrows = _decode(inp)
    out = inp.copy()
    for obj in arrows:
        out[_endpoint(obj)] = signal
    return out


def _shifts(train, test):
    td = {a[1] for f in train for a in f["arrows"]}
    qd = {a[1] for f in test for a in f["arrows"]}
    training_sizes = [len(g) for f in train for g, level in zip(f["groups"], f["levels"]) if level]
    test_sizes = [len(g) for f in test for g, level in zip(f["groups"], f["levels"]) if level]
    return {"directions": td.isdisjoint(qd),
            "group_size": max(training_sizes) <= 3 and min(test_sizes) >= 4,
            "scaffold": all(f["rectangular_scaffold"] for f in train)
                        and all(not f["rectangular_scaffold"] for f in test)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, signal_colors, training, testing = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _features(inp)
            signal_colors.add(f["signal"])
            base, holes, extras = f["base"], f["holes"], f["extras"]
            assert np.array_equal(base, base[::-1]) and np.array_equal(base, base[:, ::-1])
            assert holes and any(f["levels"])
            corruptions = holes | extras
            assert all(len(_orbit(p, inp.shape)) == 4 for p in corruptions)
            assert len({_orbit(p, inp.shape) for p in corruptions}) == len(corruptions)
            assert all(out[p] == 0 for p in holes), "All hole centers must survive"
            assert np.all(out[inp == f["signal"]] == f["signal"]), "Preserve arrowheads"
            paint = (f["wide"] | f["ordinary"]).copy()
            for p in holes:
                paint[p] = False
            untouched = ~paint
            for p in extras:
                untouched[p] = False
                assert out[p] == (f["signal"] if paint[p] else 0)
            assert np.array_equal(inp[untouched], out[untouched])
            assert np.array_equal(solve(inp), out) and not np.array_equal(inp, out)
            if split == "train":
                training.append((inp, out, f))
            else:
                testing.append(f)
    assert len(signal_colors) == 1, "Signal color is puzzle-wide"
    assert any(f["normal_distinct"] for _, _, f in training), "Teach per-coordinate collisions"
    assert any(2 in f["levels"] for _, _, f in training), "Teach wide halos"
    for feature in ("propagated", "diagonal_propagation", "clipped", "unaddressed", "miss_extra",
                    "near_extra", "far_extra", "protected_idle"):
        assert any(f[feature] for _, _, f in training), f"Missing teaching evidence: {feature}"
    assert any(not np.array_equal(_without_arrows(inp), out) for inp, out, _ in training)
    assert any(not np.array_equal(_without_scaffold(inp), out) for inp, out, _ in training)
    assert any(_shifts([f for _, _, f in training], testing).values()), "Missing realized train/test shift"


def _hand_regressions():
    """Two small cases with manually enumerated output regions, not renderer labels."""
    base = np.zeros((19, 23), bool)
    base[:4, 8:11] = True
    base = _symmetric(base)
    holes = {(1, 8), (1, 9), (2, 8), (3, 10)}
    for collision in (False, True):
        inp = np.where(base, 4, 0).astype(np.int64)
        for p in holes:
            inp[p] = 0
        inp[10, 1] = 4
        # Two distinct endpoints in the first case; one shared endpoint in the second.
        motifs = [_arrow((1, 3), (0, 1)),
                  _arrow((6, 8 if collision else 9), (-1, 0)),
                  _arrow((10, 6), (0, -1))]
        for obj in motifs:
            for p in obj[2]:
                inp[p] = 7
        expected = inp.copy()
        expected[10, 1] = 0
        if collision:
            expected[:4, 6:12] = 7
            expected[4, 6:11] = 7
        else:
            expected[:3, 7:11] = 7
            expected[3, 7:10] = 7
        for p in holes:
            expected[p] = 0
        for k in range(4):
            for flip in (False, True):
                x = np.rot90(np.fliplr(inp) if flip else inp, k).copy()
                y = np.rot90(np.fliplr(expected) if flip else expected, k).copy()
                assert np.array_equal(solve(x), y)
        assert expected[3, 7] == 7, "An unhit member broadcasts its halo"
        assert expected[3, 10] == 0, "Unaddressed centers survive another group's halo"
        assert expected[4, 8] == (7 if collision else 0)
        assert expected[10, 1] == 0, "A speck hit is not a missing-cell hit"
        assert not np.array_equal(_without_arrows(inp), expected)
        assert not np.array_equal(_without_scaffold(inp), expected)


def self_test():
    """Bounded semantic, mutation, symmetry, count and 42-seed diversity checks."""
    _hand_regressions()
    signatures, shapes, structures, palettes = set(), set(), set(), set()
    directions, group_sizes, arrow_counts, hole_counts, extra_counts = set(), set(), set(), set(), set()
    shift_counts = {"directions": 0, "group_size": 0, "scaffold": 0}
    clipped = propagated = triple = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        observed_shifts = _shifts([_features(p["input"]) for p in puzzle["train"]],
                                 [_features(p["input"]) for p in puzzle["test"]])
        for key, value in observed_shifts.items():
            shift_counts[key] += int(value)
        signatures.add(tuple(_key(p[k]) for s in ("train", "test")
                             for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _features(inp)
                shapes.add(inp.shape)
                structures.add(_key(f["base"].astype(np.uint8)))
                palettes.add((f["solid"], f["signal"]))
                directions.update(a[1] for a in f["arrows"])
                group_sizes.update(map(len, f["groups"]))
                arrow_counts.add(len(f["arrows"]))
                hole_counts.add(len(f["holes"]))
                extra_counts.add(len(f["extras"]))
                clipped += int(f["clipped"])
                propagated += int(f["propagated"])
                triple += int(max(f["counts"].values()) >= 3)
                # Rotate/reflect both axes, motifs, targets and clipped halos.
                for k in range(4):
                    x = np.rot90(inp, k).T.copy()
                    y = np.rot90(out, k).T.copy()
                    assert np.array_equal(solve(x), y)
        if seed < 2:
            replay = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], replay[s]) for k in p)
    assert len(signatures) == 42 and len(shapes) >= 50 and len(structures) >= 100
    assert len(palettes) >= 25 and directions == set(DIRECTIONS)
    assert max(group_sizes) >= 6 and len(group_sizes) >= 5
    assert len(arrow_counts) >= 5 and len(hole_counts) >= 8 and len(extra_counts) >= 5
    assert clipped >= 42 and propagated >= 42 and triple >= 3
    assert all(n >= 4 for n in shift_counts.values())
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            q = generate_puzzle(731+tr*10+te, tr, te)
            assert len(q["train"]) == tr and len(q["test"]) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # A label mutation must be detected, not merely accepted as a valid grid.
    q = generate_puzzle(52, 2, 1)
    broken = {s: [{k: a.copy() for k, a in p.items()} for p in q[s]] for s in q}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Incorrect output label accepted")
    # Explicitly reject a tied orbit instead of choosing an arbitrary completion.
    tied = q["train"][0]["input"].copy()
    _, _, b, _, _, _ = _decode(tied)
    for p in map(tuple, np.argwhere(b).tolist()):
        orb = sorted(_orbit(p, b.shape))
        if len(orb) == 4:
            for v in orb[:2]:
                tied[v] = 0
            for v in orb[2:]:
                tied[v] = _decode(q["train"][0]["input"])[0]
            break
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("Tied reflection orbit accepted")
    return {"seeds_checked": 42, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "clean_scaffolds": len(structures),
            "role_palettes": len(palettes), "directions": len(directions),
            "group_sizes": sorted(group_sizes), "arrow_counts": sorted(arrow_counts),
            "hole_counts": sorted(hole_counts), "extra_counts": sorted(extra_counts),
            "clipped_pairs": clipped, "propagating_pairs": propagated,
            "triple_collision_pairs": triple, "train_test_shifts": shift_counts,
            "independent_semantic_review": False}
