"""Breaches merge rooms: recover masonry, then trace the dye's accessible rim.

The fixed three-pixel scale is a family convention, not a seed-dependent choice.
Only NumPy and the standard library are needed. No parent generator is imported.
"""
from collections import deque
import json

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BLOCK = 3


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(int, copy=True)


def _flood(allowed, start):
    reached = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    queue = deque([start])
    h, w = allowed.shape
    while queue:
        y, x = queue.popleft()
        for dy, dx in N4:
            v, u = y + dy, x + dx
            if 0 <= v < h and 0 <= u < w and allowed[v, u] and not reached[v, u]:
                reached[v, u] = True
                queue.append((v, u))
    return reached


def _components(mask):
    remaining = mask.copy()
    result = []
    while remaining.any():
        start = tuple(int(v) for v in np.argwhere(remaining)[0])
        component = _flood(remaining, start)
        result.append(component)
        remaining[component] = False
    return result


def _near(mask, diagonal=True):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    result = np.zeros_like(mask)
    offsets = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)
               if (y or x) and (diagonal or abs(y) + abs(x) == 1)]
    for dy, dx in offsets:
        result |= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return result


def _recover(inp):
    """All roles and the clean map are determined by this input alone."""
    if any(n % BLOCK for n in inp.shape):
        raise ValueError("the masonry map has whole 3x3 tiles")
    values, counts = np.unique(inp, return_counts=True)
    if len(values) != 3 or 0 not in values:
        raise ValueError("input uses floor 0, a repeated wall color, and one dye seed")
    singletons = [int(v) for v, c in zip(values, counts) if v != 0 and c == 1]
    if len(singletons) != 1:
        raise ValueError("one nonzero singleton identifies the dye seed")
    dye = singletons[0]
    wall = int(next(v for v in values if v not in (0, dye)))
    h, w = inp.shape
    observed = inp == wall
    totals = observed.reshape(h // 3, 3, w // 3, 3).sum(axis=(1, 3))
    coarse = totals >= 5
    base = np.repeat(np.repeat(coarse, 3, axis=0), 3, axis=1)
    seed = tuple(int(v) for v in np.argwhere(inp == dye)[0])
    if base[seed] or not base.any() or base.all():
        raise ValueError("seed must lie in intended floor, and both map roles must exist")
    holes = base & (inp == 0)
    extras = ~base & observed
    return base, holes, extras, seed, wall, dye


def _effects(base, holes, seed):
    """The reachability and boundary are computed before deleting remote walls."""
    reached = _flood(~base | holes, seed)
    padded_complement = np.pad(~reached, 1, constant_values=True)
    outside = _flood(padded_complement, (0, 0))
    rim = _near(outside)[1:-1, 1:-1] & reached & ~holes
    active = reached & holes
    halo = _near(active) & ~holes
    kept = np.zeros_like(base)
    for component in _components(base):
        if np.any(component & _near(reached, diagonal=False)):
            kept |= component
    return reached, outside[1:-1, 1:-1], rim, active, halo, kept


def _paint(base, holes, seed, wall, dye):
    reached, outside, rim, active, halo, kept = _effects(base, holes, seed)
    out = np.where(kept, wall, 0).astype(int)
    out[rim | halo] = dye
    out[holes] = 0  # All missing cells, including inactive ones, remain holes.
    out[seed] = dye
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    base, holes, extras, seed, wall, dye = _recover(inp)
    return _paint(base, holes, seed, wall, dye)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _counts(num_train, num_test):
    for name, count, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer)) or not lo <= count <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _partition(rng, h, w, count, minimum=2, style="branch", thick=False):
    """Coarse guillotine masonry; no panel, header, or separate subtask."""
    a = np.zeros((h, w), dtype=bool)
    rooms = [(0, h, 0, w)]
    if style == "cross":
        ty = int(rng.integers(1, 3))
        tx = 2 if thick else int(rng.integers(1, 3))
        y = int(rng.integers(minimum, h - ty - minimum + 1))
        x = int(rng.integers(minimum, w - tx - minimum + 1))
        a[y:y + ty, :] = True
        a[:, x:x + tx] = True
        return a
    preferred = int(rng.integers(2))
    while len(rooms) < count:
        options = []
        for i, (y, v, x, u) in enumerate(rooms):
            for axis, extent in ((0, v - y), (1, u - x)):
                for thickness in (1, 2):
                    if extent >= 2 * minimum + thickness:
                        if style != "parallel" or axis == preferred:
                            options.append((i, axis, thickness))
        if not options:
            break
        i, axis, thickness = options[int(rng.integers(len(options)))]
        y, v, x, u = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(y + minimum, v - thickness - minimum + 1))
            a[cut:cut + thickness, x:u] = True
            rooms.extend([(y, cut, x, u), (cut + thickness, v, x, u)])
        else:
            cut = int(rng.integers(x + minimum, u - thickness - minimum + 1))
            a[y:v, cut:cut + thickness] = True
            rooms.extend([(y, v, x, cut), (y, v, cut + thickness, u)])
    return a


def _island(rng, a, room):
    """Insert a visibly legitimate object with a one-tile floor clearance."""
    permitted = room & ~_near(~room)  # Canvas edge clearance is checked below.
    permitted[[0, -1], :] = False
    permitted[:, [0, -1]] = False
    shapes = [np.ones((1, 1), bool), np.ones((1, 2), bool),
              np.ones((2, 1), bool), np.array([[1, 0], [1, 1]], bool),
              np.ones((2, 2), bool)]
    rng.shuffle(shapes)
    for shape in shapes:
        sh, sw = shape.shape
        starts = [(y, x) for y in range(a.shape[0] - sh + 1)
                  for x in range(a.shape[1] - sw + 1)
                  if np.all(permitted[y:y + sh, x:x + sw][shape])]
        if starts:
            y, x = starts[int(rng.integers(len(starts)))]
            a[y:y + sh, x:x + sw] |= shape
            return True
    return False


def _corridors(base, labels):
    """Candidate straight cracks cross 3- or 6-pixel masonry segments."""
    candidates = []
    for transposed in (False, True):
        b = base.T if transposed else base
        lab = labels.T if transposed else labels
        h, w = b.shape
        for y in range(1, h, 3):
            x = 1
            while x < w - 1:
                if not b[y, x] or b[y, x - 1]:
                    x += 1
                    continue
                stop = x
                while stop < w and b[y, stop]:
                    stop += 1
                if stop < w and stop - x in (3, 6):
                    left, right = int(lab[y, x - 1]), int(lab[y, stop])
                    if left >= 0 and right >= 0 and left != right:
                        points = [(q, y) if transposed else (y, q)
                                  for q in range(x, stop)]
                        candidates.append((left, right, points))
                x = stop + 1
    return candidates


def _can_damage(holes, points):
    trial = holes.copy()
    for p in points:
        if trial[p]:
            return False
        trial[p] = True
    h, w = trial.shape
    loads = trial.reshape(h // 3, 3, w // 3, 3).sum(axis=(1, 3))
    return bool(np.all(loads <= 4))


def _put(holes, points):
    if not _can_damage(holes, points):
        return False
    for p in points:
        holes[p] = True
    return True


def _scene(rng, mode, palette, regime, missing_corners=None):
    anchor = mode in ("merge", "sealed")
    h = int(rng.integers(8 if mode == "merge" else 7 if anchor else 4, 11))
    w = int(rng.integers(8 if mode == "merge" else 7 if anchor else 4, 11))
    minimum = 3 if anchor else 2
    if mode == "merge":
        style, count = "cross", 4
    elif mode == "sealed":
        style, count = "branch", 3
    elif mode == "test":
        style = "branch" if regime == 0 else str(rng.choice(["cross", "branch", "parallel"]))
        h, w = max(h, 7), max(w, 7)
        count = int(rng.integers(4, 7))
    else:
        style = str(rng.choice(["cross", "branch", "parallel"]))
        count = int(rng.integers(2, 6))
        if style == "cross":
            h, w = max(h, 6), max(w, 6)
    coarse = _partition(rng, h, w, count, minimum, style, mode == "merge")
    rooms = _components(~coarse)
    if len(rooms) < (3 if anchor else 2):
        return None
    initial_index = int(rng.integers(len(rooms)))
    # All room labels here belong to the intended map, not to observed gaps.
    labels_coarse = np.full(coarse.shape, -1, int)
    for i, room in enumerate(rooms):
        labels_coarse[room] = i
    labels = np.repeat(np.repeat(labels_coarse, 3, axis=0), 3, axis=1)
    base0 = np.repeat(np.repeat(coarse, 3, axis=0), 3, axis=1)
    candidates0 = _corridors(base0, labels)
    rng.shuffle(candidates0)
    reached_rooms = {initial_index}
    full = []
    wanted = 2 if mode == "merge" else 0 if mode == "sealed" else int(rng.integers(0, 4))
    if mode == "test":
        wanted = min(len(rooms) - 1, int(rng.integers(2, 5)))
    for _ in range(wanted):
        options = [c for c in candidates0 if (c[0] in reached_rooms) != (c[1] in reached_rooms)]
        if not options:
            return None
        corridor = options[int(rng.integers(len(options)))]
        full.append(corridor)
        reached_rooms.update(corridor[:2])
    # Anchors carry both island preservation and far-side object erasure.
    if anchor:
        remote = [i for i in range(len(rooms)) if i not in reached_rooms]
        if not remote:
            return None
        if not _island(rng, coarse, rooms[initial_index]):
            return None
        if not _island(rng, coarse, rooms[remote[int(rng.integers(len(remote)))]]):
            return None
    else:
        for _ in range(int(rng.integers(0, 5))):
            room = rooms[int(rng.integers(len(rooms)))] & ~coarse
            _island(rng, coarse, room)
        # Edge intrusions alter external contours without changing the rule.
        for _ in range(int(rng.integers(0, 3))):
            edge = int(rng.integers(4))
            trial = coarse.copy()
            if edge < 2:
                x = int(rng.integers(w))
                trial[0 if edge == 0 else -1, x] = True
            else:
                y = int(rng.integers(h))
                trial[y, 0 if edge == 2 else -1] = True
            # Do not change the partition or consume an opening's endpoints.
            if len(_components(~trial)) == len(rooms):
                coarse = trial
    base = np.repeat(np.repeat(coarse, 3, axis=0), 3, axis=1)
    # Recheck planned openings against the final visible objects.
    candidates = _corridors(base, labels)
    valid_paths = {tuple(c[2]) for c in candidates}
    if any(tuple(c[2]) not in valid_paths for c in full):
        return None
    holes = np.zeros_like(base)
    for _, _, points in full:
        if not _put(holes, points):
            return None
    options = [c for c in candidates if (c[0] in reached_rooms or c[1] in reached_rooms)
               and tuple(c[2]) not in {tuple(d[2]) for d in full}]
    rng.shuffle(options)
    partial_count = 1 if anchor else int(rng.integers(0, 4))
    put_partials = 0
    for left, right, points in options:
        if put_partials >= partial_count:
            break
        if left not in reached_rooms:
            points = points[::-1]
        depth = int(rng.integers(1, len(points)))
        if _put(holes, points[:depth]):
            put_partials += 1
    if anchor and not put_partials:
        return None
    # A core hole is an enclosed pocket: it stays empty but cannot receive dye.
    interior = base & ~_near(~base, diagonal=False) & ~_near(holes) & ~holes
    locations = np.argwhere(interior)
    rng.shuffle(locations)
    isolated_count = 1 if anchor else int(rng.integers(0, 5))
    placed = 0
    for y, x in locations:
        if placed >= isolated_count:
            break
        if _put(holes, [(int(y), int(x))]):
            placed += 1
    if anchor and not placed:
        return None
    if not holes.any():
        return None
    positions = np.argwhere((labels == initial_index) & ~base)
    if not len(positions):
        return None
    seed = tuple(int(v) for v in positions[int(rng.integers(len(positions)))])
    wall, dye = palette
    inp = np.where(base, wall, 0).astype(int)
    inp[holes] = 0
    extras = np.zeros_like(base)
    floor_positions = np.argwhere(~base)
    rng.shuffle(floor_positions)
    extra_count = int(rng.integers(1 if anchor else 0, 10))
    for y, x in floor_positions:
        p = int(y), int(x)
        if p == seed:
            continue
        if extras.sum() >= extra_count:
            break
        if _can_damage(extras, [p]):
            extras[p] = True
    inp[extras] = wall
    inp[seed] = dye
    out = _paint(base, holes, seed, wall, dye)
    # Reconstruction is exact, including the intended identities of islands.
    recovered = _recover(inp)
    if not np.array_equal(recovered[0], base) or not np.array_equal(recovered[1], holes):
        return None
    reached, outside, rim, active, halo, kept = _effects(base, holes, seed)
    if anchor:
        if mode == "sealed" and int(reached.sum()) >= max(int(c.sum()) for c in _components(~base)):
            return None  # A smaller marker-selected room refutes largest-region selection.
        if not (active.any() and (holes & ~active).any() and (base & ~kept).any()):
            return None
        islands = base & ~outside
        if not islands.any() or not np.any(_near(islands) & ~base & ~rim & ~halo):
            return None
        # Ensure defect halos add evidence beyond the ordinary exterior contour.
        if not np.any(halo & base & kept):
            return None
    k = int(rng.integers(4))
    if regime == 1 and not anchor:
        k = 1 if mode == "test" else 0
    reflect = bool(rng.integers(2))
    def transform(a):
        a = np.rot90(a, k)
        return np.fliplr(a).copy() if reflect else a.copy()
    inp, out, reached = transform(inp), transform(out), transform(reached)
    corners = {i for i, v in enumerate((reached[0, 0], reached[0, -1],
                                       reached[-1, 0], reached[-1, -1])) if v}
    if missing_corners is not None and corners & missing_corners:
        return None
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(int(c) for c in rng.choice(np.arange(1, 10), 2, replace=False))
    palette_mode = int(rng.integers(3))
    regime = int(rng.integers(3))
    puzzle = {"train": [], "test": []}
    seen, first_corners = set(), None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            mode = ("merge" if i == 0 else "sealed" if i == 1 else "random") if split == "train" else "test"
            if palette_mode == 0:
                colors = palette
            elif palette_mode == 1:
                colors = (int(rng.choice([c for c in range(1, 10) if c != palette[1]])), palette[1])
            else:
                colors = tuple(int(c) for c in rng.choice(np.arange(1, 10), 2, replace=False))
            for _ in range(300):
                pair = _scene(rng, mode, colors, regime,
                              first_corners if mode == "sealed" else None)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded scene sampling failed")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if mode == "merge":
                b, holes, _, start, _, _ = _recover(pair["input"])
                reached = _flood(~b | holes, start)
                first_corners = {j for j, v in enumerate((reached[0, 0], reached[0, -1],
                                                         reached[-1, 0], reached[-1, -1])) if v}
        rng.shuffle(puzzle[split])
    return puzzle


def _features(inp):
    b, holes, extras, start, wall, dye = _recover(inp)
    reached, outside, rim, active, halo, kept = _effects(b, holes, start)
    rooms = _components(~b)
    touched = [c for c in rooms if np.any(c & reached)]
    islands = [c for c in _components(b) if not np.any(c & outside)]
    corners = {i for i, v in enumerate((reached[0, 0], reached[0, -1],
                                       reached[-1, 0], reached[-1, -1])) if v}
    return {"shape": inp.shape, "rooms": len(rooms), "merged": len(touched),
            "holes": int(holes.sum()), "active": int(active.sum()),
            "inactive": int((holes & ~active).sum()), "extras": int(extras.sum()),
            "islands": len(islands), "erased": int((b & ~kept).sum()),
            "halo_on_wall": int((halo & b & kept).sum()), "corners": corners,
            "unoutlined": bool(np.any(_near(b & ~outside) & ~b & ~rim & ~halo)),
            "diagonal_rim": int((rim & ~_near(np.pad(~reached, 1, constant_values=True), False)[1:-1, 1:-1]).sum()),
            "wall": wall, "dye": dye, "base": b}


def _ablations(inp):
    """Concrete shortcut predictions for evidence checks, never for solving."""
    base, holes, extras, seed, wall, dye = _recover(inp)
    # Omit A: treat every observed wall pixel as genuine; no defect recognition.
    raw_walls = inp == wall
    without_a = _paint(raw_walls, np.zeros_like(base), seed, wall, dye)
    # Omit B: erase specks and halo every defect, without seed/exterior selection.
    without_b = np.where(base, wall, 0).astype(int)
    without_b[_near(holes) & ~holes] = dye
    without_b[holes] = 0
    without_b[seed] = dye
    return without_a, without_b


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            b, holes, extras, start, wall, dye = _recover(inp)
            reached, outside, rim, active, halo, kept = _effects(b, holes, start)
            assert np.array_equal(solve(inp), out), "input-only solution mismatch"
            assert np.all(out[holes] == 0), "missing cells must remain empty"
            assert out[start] == dye
            assert np.all(out[(~kept) & ~(rim | halo) & ~holes & (inp != dye)] == 0)
            assert np.all(out[(rim | halo) & ~holes & (inp != dye)] == dye)
            assert np.all(out[kept & ~(rim | halo) & ~holes] == wall)
            assert not np.array_equal(inp, out)
            if split == "train":
                training.append(_features(inp))
    # Observable evidence requirements, not tests of hidden sampling style IDs.
    assert any(f["merged"] >= 3 for f in training), "teach propagation across more than one breach"
    assert any(f["merged"] == 1 for f in training), "teach that incomplete cracks do not merge rooms"
    assert any(f["extras"] > 0 for f in training), "teach speck removal"
    assert any(f["active"] and f["inactive"] and f["halo_on_wall"] for f in training)
    assert any(f["unoutlined"] and f["erased"] for f in training), "teach island/exterior distinction"
    assert not set.intersection(*(f["corners"] for f in training)), "fixed corner selection shortcut"
    for which in (0, 1):
        assert any(not np.array_equal(_ablations(_grid(p["input"]))[which], p["output"])
                   for p in puzzle["train"]), "each inherited mechanism must change an answer"


def self_test():
    """Bounded semantic regressions, equivariances, and 42-seed measurements."""
    # Hand-checkable fixture: a 3-pixel wall, enclosed island, dye left of wall.
    b = np.zeros((15, 18), bool)
    b[:, 9:12] = True
    b[6:9, 3:6] = True
    holes = np.zeros_like(b)
    holes[7, 4] = True  # Isolated island pocket, not reached.
    holes[4, 9:11] = True  # An incomplete crack, reached from the left.
    inp = np.where(b, 1, 0)
    inp[holes] = 0
    inp[10, 2] = 2
    inp[11, 7] = 1  # Extra speck, removed even inside selected region.
    out = solve(inp)
    assert out[11, 7] == 0 and out[7, 4] == 0
    assert out[6, 4] == 1 and out[5, 4] == 0  # Inactive defect and island get no halo/rim.
    assert out[4, 9] == out[4, 10] == 0
    assert out[3, 9] == 2 and out[4, 11] == 2  # Active halo crosses onto masonry.
    assert out[0, 17] == 0  # Unreached right compartment.
    opened = inp.copy()
    opened[4, 11] = 0
    opened_out = solve(opened)
    assert opened_out[0, 17] == 2 and opened_out[4, 11] == 0
    # Extra specks do not change connectivity; seed movement within a region
    # changes only old/new seed cells. Across the intact wall it changes the rim.
    clean_speck = inp.copy()
    clean_speck[11, 7] = 0
    assert np.array_equal(solve(clean_speck), out)
    moved = inp.copy()
    moved[10, 2], moved[10, 6] = 0, 2
    expected = out.copy()
    expected[10, 2], expected[10, 6] = 0, 2
    assert np.array_equal(solve(moved), expected)
    across = inp.copy()
    across[10, 2], across[10, 15] = 0, 2
    assert solve(across)[0, 17] == 2 and solve(across)[0, 0] == 0
    # Simultaneous changes of colors, rotations, and reflections remain valid.
    for k in range(4):
        for reflect in (False, True):
            def t(a):
                v = np.rot90(a, k)
                return np.fliplr(v).copy() if reflect else v.copy()
            assert np.array_equal(solve(t(inp)), t(out))
    renamed = np.where(inp == 1, 8, np.where(inp == 2, 7, 0))
    assert np.array_equal(solve(renamed), np.where(out == 1, 8, np.where(out == 2, 7, 0)))
    before = inp.copy()
    solve(inp)
    assert np.array_equal(before, inp), "solve must not mutate input"
    shapes, structures, palettes, room_counts, merged_counts = set(), set(), set(), set(), set()
    hole_counts, island_counts, signatures = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        fs = [_features(p["input"]) for s in puzzle for p in puzzle[s]]
        assert len({f["merged"] for f in fs}) >= 2
        assert len({f["holes"] for f in fs}) >= 2
        for f in fs:
            shapes.add(f["shape"])
            structures.add(_key(f["base"]))
            palettes.add((f["wall"], f["dye"]))
            room_counts.add(f["rooms"])
            merged_counts.add(f["merged"])
            hole_counts.add(f["holes"])
            island_counts.add(f["islands"])
        if seed < 2:
            q = generate_puzzle(seed)
            assert all(np.array_equal(p[k], r[k]) for s in puzzle
                       for p, r in zip(puzzle[s], q[s]) for k in p)
    assert len(signatures) == 42 and len(structures) >= 100
    assert len(shapes) >= 20 and len(palettes) >= 20
    assert len(room_counts) >= 3 and len(merged_counts) >= 3
    assert len(hole_counts) >= 10 and len(island_counts) >= 3
    for a in range(2, 11):
        for c in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(73, a, c))
    for a, c in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2),
                 (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, c)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    bad["test"][0]["output"][0, 0] = (bad["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "scaffolds": len(structures), "palettes": len(palettes),
            "room_counts": sorted(room_counts), "merged_counts": sorted(merged_counts),
            "hole_counts": sorted(hole_counts), "island_counts": sorted(island_counts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
