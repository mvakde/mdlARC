"""Periodic switch fabric: a motif programs routing, then beads settle.

Only NumPy and the standard library are used. The solver reads all roles,
orientation, gate states and bead counts from the query input itself.
"""
from itertools import permutations
from functools import lru_cache

import numpy as np


BACKGROUND = 0
FRAME = 8
ROLE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
ORDERS = tuple((transpose, rows, cols)
               for transpose in (False, True)
               for rows in permutations(range(3))
               for cols in permutations(range(3)))
DIRECT = (False, (1, 2, 0), (1, 2, 0))


def _full_box(mask):
    return (mask.shape == (3, 3) and 2 <= int(mask.sum()) <= 8
            and mask[0].any() and mask[2].any()
            and mask[:, 0].any() and mask[:, 2].any())


MASKS = tuple(m for code in range(1, 511)
              if _full_box(m := np.array([(code >> i) & 1 for i in range(9)],
                                         dtype=bool).reshape(3, 3)))


def _tile(mask, order=DIRECT):
    transpose, rows, cols = order
    source = mask.T if transpose else mask
    return source[np.ix_(rows, cols)]


@lru_cache(maxsize=512)
def _order_matches(mask_bytes):
    """Cache mask symmetries for sampling/validation, never for solving."""
    mask = np.frombuffer(mask_bytes, dtype=bool).reshape(3, 3)
    return frozenset(order for order in ORDERS
                     if np.array_equal(_tile(mask, order), _tile(mask)))


def _matching_orders(masks):
    matching = set(ORDERS)
    for mask in masks:
        matching.intersection_update(_order_matches(mask.tobytes()))
    return [order for order in ORDERS if order in matching]


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return array.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _box(coords):
    assert len(coords)
    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    return tuple(map(int, (low[0], high[0], low[1], high[1])))


def _frame_box(grid):
    box = _box(np.argwhere(grid == FRAME))
    r0, r1, c0, c1 = box
    assert 5 <= r1 - r0 <= 9 and 5 <= c1 - c0 <= 9
    expected = np.zeros(grid.shape, dtype=bool)
    expected[r0, c0:c1] = expected[r1 - 1, c0:c1] = True
    expected[r0:r1, c0] = expected[r0:r1, c1 - 1] = True
    assert np.array_equal(grid == FRAME, expected), "one rectangular frame required"
    return box


def _sites(core, gate):
    h, w = core.shape
    assert h % 2 == 1 and 3 <= (h - 1) // 2 <= 8
    centers = []
    for row in range(1, h, 2):
        xs = np.flatnonzero(core[row] == gate).tolist()
        starts = [x for x in xs if x - 1 not in xs]
        cs = []
        for start in starts:
            assert 1 <= start and start + 3 < w
            assert all(x in xs for x in range(start, start + 3))
            assert start + 3 not in xs
            cs.append(start + 1)
        assert len(xs) == 3 * len(cs) and 3 <= len(cs) <= 5
        centers.append(cs)
    count = len(centers[0])
    pitch = centers[0][1] - centers[0][0]
    assert 4 <= pitch <= 6
    stagger = centers[1][0] - centers[0][0]
    assert abs(stagger) == 2
    for i, cs in enumerate(centers):
        assert len(cs) == count
        assert cs == [centers[0][0] + (i % 2) * stagger + pitch * j
                      for j in range(count)]
    return centers, pitch, stagger


def _trace(mask):
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    neighbors = {p: [q for q in ((p[0] - 1, p[1]), (p[0], p[1] - 1),
                                (p[0], p[1] + 1), (p[0] + 1, p[1])) if q in cells]
                 for p in cells}
    assert all(len(ns) in (1, 2) for ns in neighbors.values()), "branches/isolates"
    pending = set(cells)
    paths = []
    for endpoint in sorted(p for p, ns in neighbors.items() if len(ns) == 1):
        if endpoint not in pending:
            continue
        path, previous, current = [], None, endpoint
        while True:
            assert current in pending
            pending.remove(current)
            path.append(current)
            choices = [q for q in neighbors[current] if q != previous]
            if not choices:
                break
            assert len(choices) == 1
            previous, current = current, choices[0]
        assert len(path) >= 2
        steps = {b[0] - a[0] for a, b in zip(path, path[1:]) if b[0] != a[0]}
        assert len(steps) <= 1, "nonmonotone route would require a new motion rule"
        paths.append(path)
    assert not pending, "closed loops"
    return paths


def _network(core, roles, mask, order=DIRECT):
    track, bead, gate, accent = roles
    assert set(map(int, np.unique(core))) <= {BACKGROUND, track, bead, gate}
    assert np.all(np.isin(core[::2], (BACKGROUND, track, bead)))
    assert np.all(np.isin(core[1::2], (BACKGROUND, gate)))
    assert np.all(np.isin(core[::2, [0, core.shape[1] - 1]], (track, bead)))
    centers, pitch, stagger = _sites(core, gate)
    pattern = _tile(mask, order)
    active, inactive = [], []
    structural = core.copy()
    structural[structural == bead] = track
    writes = {}
    for i, cs in enumerate(centers):
        for j, c in enumerate(cs):
            r = 2 * i + 1
            if not pattern[i % 3, j % 3]:
                inactive.append((r, c))
                continue
            active.append((r, c))
            assert np.all(np.isin(core[r - 1:r + 2:2, c - 1:c + 2], (track, bead)))
            assert core[r - 1, c] == core[r + 1, c] == track
            for y in range(r - 1, r + 2):
                for x in range(c - 1, c + 2):
                    value = accent if x == c else track
                    assert (y, x) not in writes or writes[y, x] == value
                    writes[y, x] = value
    for cell, value in writes.items():
        structural[cell] = value
    paths = _trace(structural == track)
    beads = {tuple(map(int, p)) for p in np.argwhere(core == bead)}
    endpoints = {p for path in paths for p in (path[0], path[-1])}
    assert beads <= endpoints
    for r, c in endpoints:
        assert r % 2 == 0
        assert (c in (0, core.shape[1] - 1)
                or core[r, c - 1] == BACKGROUND or core[r, c + 1] == BACKGROUND)
    breaks = {tuple(map(int, p)) for p in np.argwhere(core[::2] == BACKGROUND)}
    breaks = {(2 * r, c) for r, c in breaks}
    for r, c in breaks:
        assert 2 <= c <= core.shape[1] - 3 and (r, c) not in writes
        assert all(abs(c - x) >= 3 for y, x in breaks if y == r and x != c)
    return dict(structural=structural, paths=paths, beads=beads, active=active,
                inactive=inactive, centers=centers, pitch=pitch, stagger=stagger,
                breaks=breaks, writes=writes)


def _read(input_grid):
    original = _grid(input_grid)
    possibilities = []
    for rotation in range(4):
        grid = np.rot90(original, rotation).copy()
        try:
            panel = _frame_box(grid)
            r0, r1, c0, c1 = panel
            inside = grid[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
            accents = set(map(int, np.unique(inside))) - {BACKGROUND}
            assert len(accents) == 1
            accent = accents.pop()
            assert accent in ROLE_COLORS
            mr0, mr1, mc0, mc1 = _box(np.argwhere(inside == accent))
            mask = inside[mr0:mr1, mc0:mc1] == accent
            assert _full_box(mask)
            outside = grid.copy()
            outside[r0:r1, c0:c1] = BACKGROUND
            box = _box(np.argwhere(outside != BACKGROUND))
            top, bottom, left, right = box
            assert top > r1 - 1, "panel must be above the network in reading orientation"
            assert np.all(outside[:top] == BACKGROUND)
            core = outside[top:bottom, left:right]
            gate_colors = set(map(int, np.unique(core[1::2]))) - {BACKGROUND}
            assert len(gate_colors) == 1
            gate = gate_colors.pop()
            rest = set(map(int, np.unique(core[::2]))) - {BACKGROUND}
            assert len(rest) == 2
            candidates = []
            for track in sorted(rest):
                bead = next(iter(rest - {track}))
                roles = (track, bead, gate, accent)
                assert len(set(roles)) == 4 and set(roles) <= set(ROLE_COLORS)
                try:
                    info = _network(core, roles, mask)
                    candidates.append((roles, info))
                except AssertionError:
                    pass
            assert len(candidates) == 1, "track and bead must be geometrically distinguishable"
            roles, info = candidates[0]
            possibilities.append(dict(grid=grid, rotation=rotation, panel=panel,
                                      box=box, core=core, mask=mask, roles=roles, **info))
        except AssertionError:
            continue
    if len(possibilities) != 1:
        raise ValueError("input is not an unambiguous periodic switch fabric")
    return possibilities[0]


def _settle(info, bead):
    answer = info["structural"].copy()
    target = set()
    single = collisions = levels = long_moves = settled = 0
    for path in info["paths"]:
        old = info["beads"] & {path[0], path[-1]}
        if path[0][0] == path[-1][0]:
            chosen = old
            levels += bool(old)
        else:
            inward = path if path[0][0] > path[-1][0] else path[::-1]
            chosen = set(inward[:len(old)])
            collisions += len(old) == 2
            single += len(old) == 1 and chosen != old
            long_moves += (bool(old) and chosen != old
                           and abs(path[0][0] - path[-1][0]) >= 4)
        settled += bool(old) and chosen == old
        target.update(chosen)
    for cell in target:
        answer[cell] = bead
    return answer, dict(target=target, single=single, collisions=collisions,
                        levels=levels, long_moves=long_moves, settled=settled)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only this grid. The panel fixes reading axes and gravity."""
    info = _read(input_grid)
    core, _ = _settle(info, info["roles"][1])
    answer = info["grid"].copy()
    top, bottom, left, right = info["box"]
    answer[top:bottom, left:right] = core
    return np.rot90(answer, -info["rotation"]).copy()


def _single_gate_ablation(info):
    """Keep A's operation but replace B's program by one fixed active site.

    Used only on the one-period teaching fabric. The earliest activated gate
    is retained, so its neighborhood is known to be clear of track breaks.
    This checks bead destinations as well as the obviously different bars.
    """
    assert len(info["centers"]) == len(info["centers"][0]) == 3
    r, c = min(info["active"])
    i, j = r // 2, info["centers"][r // 2].index(c)
    constant = np.zeros((3, 3), dtype=bool)
    constant[(i + 1) % 3, (j + 1) % 3] = True
    alternate = _network(info["core"], info["roles"], constant)
    return _settle(alternate, info["roles"][1])[1]["target"]


def _contrast_pair(pair, rng):
    """Find a valid different program on exactly the same track/bead scene."""
    info = _read(pair["input"])
    original_targets = _settle(info, info["roles"][1])[1]["target"]
    for index in rng.permutation(len(MASKS)):
        mask = MASKS[int(index)]
        if np.array_equal(mask, info["mask"]):
            continue
        try:
            alternate = _network(info["core"], info["roles"], mask)
            targets = _settle(alternate, info["roles"][1])[1]["target"]
        except AssertionError:
            continue
        if targets == original_targets:
            continue
        grid = info["grid"].copy()
        r0, r1, c0, c1 = info["panel"]
        inside = grid[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
        inside[:] = BACKGROUND
        mr = int(rng.integers(0, inside.shape[0] - 2))
        mc = int(rng.integers(0, inside.shape[1] - 2))
        inside[mr:mr + 3, mc:mc + 3][mask] = info["roles"][3]
        inp = np.rot90(grid, -info["rotation"]).copy()
        return {"input": inp, "output": solve(inp)}, mask
    return None


def _active_centers(mask, centers):
    tile = _tile(mask)
    return [(2 * i + 1, c) for i, cs in enumerate(centers) for j, c in enumerate(cs)
            if tile[i % 3, j % 3]]


def _break_options(base, active, breaks):
    return [(r, c) for r in range(0, base.shape[0], 2)
            for c in range(2, base.shape[1] - 2)
            if all(abs(c - x) >= 2 for y, x in active if abs(y - r) == 1)
            and all(abs(c - x) >= 3 for y, x in breaks if y == r)]


def _make_pair(rng, mask, separators, slots, roles, rotation,
               *, single=False, double=False, level=False, long=False):
    track, bead, gate, accent = roles
    pitch = int(rng.choice([p for p in range(4, 7) if 3 + p * (slots - 1) + 5 <= 28]))
    start = int(rng.integers(2, 4))
    parity = int(rng.integers(2))
    centers = [[start + 2 * ((i + parity) % 2) + pitch * j for j in range(slots)]
               for i in range(separators)]
    width = max(max(cs) for cs in centers) + 3 + int(rng.integers(0, 3))
    assert width <= 30
    base = np.zeros((2 * separators + 1, width), dtype=np.int8)
    base[::2] = track
    for i, cs in enumerate(centers):
        for c in cs:
            base[2 * i + 1, c - 1:c + 2] = gate
    active = _active_centers(mask, centers)
    breaks = []
    for r in range(0, base.shape[0], 2):
        ports = sorted((c, y) for y, c in active if abs(y - r) == 1)
        for (a, side_a), (b, side_b) in zip(ports, ports[1:]):
            if side_a == side_b:
                breaks.append((r, int(rng.integers(a + 2, b - 1))))
    for cell in breaks:
        base[cell] = BACKGROUND

    def good(info):
        return not long or any(abs(p[0][0] - p[-1][0]) >= 4 for p in info["paths"])

    info = _network(base, roles, mask)
    if not good(info):
        return None
    # Extra cuts are visible data, accepted only when the requested teaching
    # route survives. They never touch an activated gate's neighborhood.
    for _ in range(int(rng.integers(0, 5))):
        options = _break_options(base, active, breaks)
        if not options:
            break
        cell = options[int(rng.integers(len(options)))]
        trial = base.copy()
        trial[cell] = BACKGROUND
        trial_info = _network(trial, roles, mask)
        if good(trial_info):
            base, info = trial, trial_info
            breaks.append(cell)
    if level and not any(p[0][0] == p[-1][0] for p in info["paths"]):
        for index in rng.permutation(len(options := _break_options(base, active, breaks))):
            cell = options[int(index)]
            trial = base.copy()
            trial[cell] = BACKGROUND
            trial_info = _network(trial, roles, mask)
            if good(trial_info) and any(p[0][0] == p[-1][0] for p in trial_info["paths"]):
                base, info = trial, trial_info
                breaks.append(cell)
                break
        else:
            return None
    paths = info["paths"]
    nonlevel = [i for i, p in enumerate(paths) if p[0][0] != p[-1][0]]
    if not nonlevel:
        return None
    probability = float(rng.choice((0.12, 0.35, 0.65, 0.9)))
    chosen = {cell for p in paths for cell in (p[0], p[-1]) if rng.random() < probability}
    reserved = set()
    if single or long:
        eligible = [i for i in nonlevel if not long or abs(paths[i][0][0] - paths[i][-1][0]) >= 4]
        i = int(rng.choice(eligible))
        p = paths[i]
        chosen.difference_update((p[0], p[-1]))
        chosen.add(min(p[0], p[-1]))
        reserved.add(i)
    if double:
        eligible = [i for i in nonlevel if i not in reserved]
        if not eligible:
            return None
        p = paths[int(rng.choice(eligible))]
        chosen.update((p[0], p[-1]))
    if level:
        p = paths[int(rng.choice([i for i, p in enumerate(paths) if p[0][0] == p[-1][0]]))]
        chosen.add(p[0] if rng.random() < .5 else p[-1])
    if not chosen:
        p = paths[int(rng.choice(nonlevel))]
        chosen.add(min(p[0], p[-1]))
    for cell in chosen:
        base[cell] = bead
    # A movable motif in a framed carrier field. Its position has no phase
    # significance; the bounding box is the instruction, not its offset.
    ph, pw = int(rng.integers(5, 9)), int(rng.integers(5, 10))
    panel = np.zeros((ph, pw), dtype=np.int8)
    panel[0] = panel[-1] = FRAME
    panel[:, 0] = panel[:, -1] = FRAME
    mr, mc = int(rng.integers(1, ph - 3)), int(rng.integers(1, pw - 3))
    panel[mr:mr + 3, mc:mc + 3][mask] = accent
    top_pad, bottom_pad = int(rng.integers(0, 2)), int(rng.integers(0, 2))
    gap = int(rng.integers(1, 3))
    left_pad = int(rng.integers(0, min(2, 30 - width) + 1))
    right_pad = int(rng.integers(0, min(2, 30 - width - left_pad) + 1))
    h, w = top_pad + ph + gap + base.shape[0] + bottom_pad, width + left_pad + right_pad
    assert h <= 30 and w <= 30
    grid = np.zeros((h, w), dtype=np.int8)
    pc = int(rng.integers(0, w - pw + 1))
    grid[top_pad:top_pad + ph, pc:pc + pw] = panel
    ntop = top_pad + ph + gap
    grid[ntop:ntop + base.shape[0], left_pad:left_pad + width] = base
    grid = np.rot90(grid, rotation).copy()
    return {"input": grid, "output": solve(grid)}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(ROLE_COLORS, 4, replace=False)))
    rotation = int(rng.integers(4))
    puzzle, seen, masks = {"train": [], "test": []}, set(), []
    contrast = None
    # First two demonstrations identify the phase convention among all 72
    # row/column permutations, with or without transpose.
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            if split == "train" and index == 2:
                pair, mask = contrast
                puzzle[split].append(pair)
                seen.add(_key(pair["input"]))
                masks.append(mask)
                continue
            if split == "train" and index == 0:
                eligible = [m for m in MASKS if 3 <= m.sum() <= 7 and len(_matching_orders([m])) < 12]
            elif split == "train" and index == 1:
                eligible = [m for m in MASKS if m.sum() != masks[0].sum()
                            and bool(m[1, 1]) != bool(masks[0][1, 1])
                            and (contrast is None or not np.array_equal(m, contrast[1]))
                            and _matching_orders([masks[0], m]) == [DIRECT]]
            else:
                eligible = [m for m in MASKS if not any(np.array_equal(m, old) for old in masks)]
            for _ in range(100):
                mask = eligible[int(rng.integers(len(eligible)))]
                separators = (3 if index == 0 else int(rng.integers(4, 6))) if split == "train" else int(rng.integers(6, 9))
                slots = 3 if split == "train" and index == 0 else int(rng.integers(3, 6))
                pair = _make_pair(rng, mask, separators, slots, roles, rotation,
                                  single=split == "train" and index == 0,
                                  double=split == "train" and index == 1,
                                  level=split == "train" and index == 0,
                                  long=split == "train" and index == 0)
                if pair is not None and _key(pair["input"]) not in seen:
                    if split == "train" and index == 0 and ntrain >= 3:
                        contrast = _contrast_pair(pair, rng)
                        if contrast is None:
                            continue
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    masks.append(mask)
                    break
            else:
                raise RuntimeError("bounded sampling could not construct a teaching example")
    # Teaching order is not a hidden index convention.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles_seen, rotations, train_masks, separator_counts = set(), set(), set(), [], {"train": [], "test": []}
    program_masks = set()
    train_scenes = []
    coverage = dict(single=0, collisions=0, levels=0, long_moves=0)
    program_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _read(inp)
            assert info["mask"].tobytes() not in program_masks
            program_masks.add(info["mask"].tobytes())
            track, bead, gate, accent = info["roles"]
            roles_seen.add(info["roles"])
            rotations.add(info["rotation"])
            separator_counts[split].append(len(info["centers"]))
            assert info["active"] and info["inactive"]
            expected_core, details = _settle(info, bead)
            canonical_out = np.rot90(out, info["rotation"])
            r0, r1, c0, c1 = info["box"]
            output_core = canonical_out[r0:r1, c0:c1]
            assert np.array_equal(output_core, expected_core)
            untouched = np.ones(info["grid"].shape, dtype=bool)
            untouched[r0:r1, c0:c1] = False
            assert np.array_equal(canonical_out[untouched], info["grid"][untouched])
            assert len(info["beads"]) == int(np.count_nonzero(output_core == bead))
            # Check conservation and endpoint packing independently of the
            # final whole-grid comparison. Level components retain allocation.
            for path in info["paths"]:
                before = info["beads"] & set(path)
                after = {p for p in path if output_core[p] == bead}
                assert len(before) == len(after)
                if path[0][0] == path[-1][0]:
                    assert before == after
                else:
                    inward = path if path[0][0] > path[-1][0] else path[::-1]
                    assert after == set(inward[:len(before)])
            for r, c in info["active"]:
                assert np.all(output_core[r - 1:r + 2, c] == accent)
                assert output_core[r, c - 1] in (track, bead) and output_core[r, c + 1] in (track, bead)
            for r, c in info["inactive"]:
                assert np.all(output_core[r, c - 1:c + 2] == gate)
            assert all(output_core[p] == BACKGROUND for p in info["breaks"])
            assert np.array_equal(out, solve(inp))
            if split == "train":
                train_masks.append(info["mask"])
                train_scenes.append((_key(info["core"]), info["mask"].tobytes(), details["target"]))
                for feature in coverage:
                    coverage[feature] += details[feature]
                if len(info["centers"]) == len(info["centers"][0]) == 3 and details["long_moves"]:
                    program_witness |= _single_gate_ablation(info) != details["target"]
    assert len(roles_seen) == len(rotations) == 1
    assert _matching_orders(train_masks) == [DIRECT]
    assert len({int(m.sum()) for m in train_masks}) >= 2
    assert {bool(m[1, 1]) for m in train_masks} == {False, True}
    assert all(coverage.values()), "training must expose motion, collisions, level retention and multi-gate travel"
    assert program_witness, "the motif must change bead destinations, not just bar colors"
    if len(puzzle["train"]) >= 3:
        assert any(a[0] == b[0] and a[1] != b[1] and a[2] != b[2]
                   for i, a in enumerate(train_scenes) for b in train_scenes[i + 1:]), "controlled program contrast missing"
    assert max(separator_counts["train"]) < min(separator_counts["test"])
    assert all(3 <= s <= 5 for s in separator_counts["train"])
    assert all(6 <= s <= 8 for s in separator_counts["test"])


def self_test():
    """Bounded generator-owned checks and structural diversity over 42 seeds."""
    metrics = dict(seeds=42, pairs=0)
    motifs, shapes, layouts, palettes, directions, pitches, staggers = (set() for _ in range(7))
    active_counts, break_counts, bead_counts, targets = (set() for _ in range(4))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read(pair["input"])
                metrics["pairs"] += 1
                motifs.add(info["mask"].tobytes())
                shapes.add(pair["input"].shape)
                palettes.add(info["roles"])
                directions.add(info["rotation"])
                pitches.add(info["pitch"])
                staggers.add(info["stagger"])
                active_counts.add(len(info["active"]))
                break_counts.add(len(info["breaks"]))
                bead_counts.add(len(info["beads"]))
                normalized = np.zeros(info["core"].shape, dtype=np.int8)
                for code, color in enumerate(info["roles"][:3], 1):
                    normalized[info["core"] == color] = code
                layouts.add((normalized.shape, normalized.tobytes(), info["mask"].tobytes()))
                targets.add(_key(pair["output"]))
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solver mutated query"
                # A omitted: leave beads at their original terminals after
                # doing the local gate writes. Any demonstrated motion fails.
                if split == "train":
                    _, details = _settle(info, info["roles"][1])
                    if details["single"]:
                        assert details["target"] != info["beads"]
        again = generate_puzzle(seed)
        assert all(_key(a[k]) == _key(b[k]) for s in puzzle for a, b in zip(puzzle[s], again[s])
                   for k in ("input", "output"))
    assert len(motifs) >= 70 and len(layouts) >= 180 and len(shapes) >= 40
    assert len(palettes) >= 30 and directions == set(range(4))
    assert pitches == {4, 5, 6} and staggers == {-2, 2}
    assert len(active_counts) >= 18 and len(break_counts) >= 10 and len(bead_counts) >= 15
    assert len(targets) >= 180
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = generate_puzzle(0)
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    metrics.update(motifs=len(motifs), shapes=len(shapes), normalized_layouts=len(layouts),
                   palettes=len(palettes), directions=sorted(directions), pitches=sorted(pitches),
                   active_counts=sorted(active_counts), break_counts=sorted(break_counts),
                   bead_counts=sorted(bead_counts), outputs=len(targets),
                   verification="generator-owned invariants; no independent semantic oracle")
    return metrics
