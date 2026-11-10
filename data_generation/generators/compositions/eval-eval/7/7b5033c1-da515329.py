"""A folded thread controls the color and spacing of a finite pinwheel loom.

Only standard-library modules and NumPy are used. The solver parses pixels;
neither the random seed nor construction metadata enters it.
"""
from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _cw(vector):
    return vector[1], -vector[0]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _runs(sequence):
    result = []
    for color in sequence:
        color = int(color)
        if result and result[-1][0] == color:
            result[-1] = color, result[-1][1] + 1
        else:
            result.append((color, 1))
    return result


def _trace(cells, grid):
    neighbors = {p: [(p[0]+dy, p[1]+dx) for dy, dx in DIRS
                     if (p[0]+dy, p[1]+dx) in cells] for p in cells}
    endpoints = [p for p in cells if len(neighbors[p]) == 1]
    if (not 6 <= len(cells) <= 20 or len(endpoints) != 2
            or any(len(ns) not in (1, 2) for ns in neighbors.values())
            or endpoints[0][0] == endpoints[1][0]):
        raise ValueError("thread must be a simple chain with unequal endpoint rows")
    current = min(endpoints, key=lambda p: p[0])
    previous = None
    path = []
    while current is not None:
        if current in path:
            raise ValueError("cyclic thread")
        path.append(current)
        options = [p for p in neighbors[current] if p != previous]
        previous, current = current, options[0] if options else None
    if len(path) != len(cells):
        raise ValueError("disconnected thread")
    sequence = [int(grid[p]) for p in path]
    if not 2 <= len(set(sequence)) <= 4 or not 2 <= len(_runs(sequence)) <= 6:
        raise ValueError("thread must have 2..4 colors and 2..6 runs")
    return path, sequence


def _parse(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if counts.max() * 2 <= a.size:
        raise ValueError("background must be a strict majority")
    interpretations = []
    for marker in colors:
        marker = int(marker)
        if marker == bg:
            continue
        coordinates = np.argwhere(a == marker)
        if len(coordinates) % 4:
            continue
        length = len(coordinates) // 4
        if length not in (1, 2, 3):
            continue
        mean = coordinates.mean(axis=0)
        cy, cx = map(int, mean)
        if not np.array_equal(mean, [cy, cx]) or a[cy, cx] != bg:
            continue
        cross = {(cy+d*dy, cx+d*dx) for dy, dx in DIRS
                 for d in range(1, length+1)}
        if cross != {tuple(map(int, p)) for p in coordinates}:
            continue
        cells = {tuple(map(int, p)) for p in np.argwhere((a != bg) & (a != marker))}
        if any((y+dy, x+dx) in cross for y, x in cells for dy, dx in DIRS):
            continue
        try:
            path, sequence = _trace(cells, a)
        except ValueError:
            continue
        interpretations.append({"grid": a, "background": bg, "marker": marker,
                                "length": length, "center": (cy, cx),
                                "cross": cross, "path": path, "sequence": sequence})
    if len(interpretations) != 1:
        raise ValueError("input needs exactly one hollow-cross/thread interpretation")
    return interpretations[0]


def _weave(sequence, shape, center, length, bg, marker, *,
           fixed_gap=None, straight=False, first_exit=False):
    """Grow on the plane; clipping affects painting only.

    Optional variants are named ablation diagnostics, never sampling modes.
    """
    h, w = shape
    cy, cx = center
    output = np.full(shape, bg, dtype=np.int8)
    plane, seed, stage_cells = set(), set(), []
    radius = max(2, length)
    tips, headings, dead = [], [], []

    def inside(p):
        return 0 <= p[0] < h and 0 <= p[1] < w

    for dy, dx in DIRS:
        cells = {(cy+d*dy, cx+d*dx) for d in range(1, radius+1)}
        seed.update(cells)
        plane.update(cells)
        for p in cells:
            if inside(p):
                output[p] = marker
        tips.append((cy+radius*dy, cx+radius*dx))
        headings.append((dy, dx) if straight else _cw((dy, dx)))
        dead.append(first_exit and any(not inside(p) for p in cells))
    for color, count in _runs(sequence):
        target_radius = radius + (count+1 if fixed_gap is None else fixed_gap)
        new_cells = set()
        for arm in range(4):
            y, x = tips[arm]
            dy, dx = headings[arm]
            target = (cy+dy*target_radius, x) if dy else (y, cx+dx*target_radius)
            steps = abs(target[0]-y) + abs(target[1]-x)
            for _ in range(steps):
                y, x = y+dy, x+dx
                p = (y, x)
                if p in plane:
                    raise AssertionError("loom tracks collided")
                plane.add(p)
                new_cells.add(p)
                if first_exit and not inside(p):
                    dead[arm] = True
                if inside(p) and not dead[arm]:
                    output[p] = color
            tips[arm] = (y, x)
            if not straight:
                headings[arm] = _cw(headings[arm])
        stage_cells.append(new_cells)
        radius = target_radius
    return output, {"plane": plane, "seed": seed, "stages": stage_cells,
                    "visible": [len([p for p in stage if inside(p)]) for stage in stage_cells],
                    "radius": radius}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _weave(info["sequence"], info["grid"].shape, info["center"],
                  info["length"], info["background"], info["marker"])[0]


def _render(info, sequence=None, **options):
    return _weave(info["sequence"] if sequence is None else sequence,
                  info["grid"].shape, info["center"], info["length"],
                  info["background"], info["marker"], **options)[0]


def _sweeps(path):
    columns = sorted({x for _, x in path})
    result = []
    for order in (columns, columns[::-1]):
        for last_up in (False, True):
            result.append([p for x in order for p in sorted(
                (p for p in path if p[1] == x), reverse=last_up and x == order[-1])])
    return result


def _above(path):
    return min(y for y, _ in path) < path[0][0]


def _sweep_failure(info, answer):
    a = info["grid"]
    return all(not np.array_equal(_render(info, [int(a[p]) for p in order]), answer)
               for order in _sweeps(info["path"]))


def _counts_sequence(sequence):
    counts = Counter(sequence)
    return [c for c in dict.fromkeys(sequence) for _ in range(counts[c])]


def _path_geometry(path):
    y0, x0 = path[0]
    return tuple((y-y0, x-x0) for y, x in path)


def _random_path(rng, n, allow_above, force_above=False):
    # New vertices must touch exactly the previous tip, making an induced path.
    for _ in range(120):
        path, occupied, previous = [(0, 0)], {(0, 0)}, None
        persistence = float(rng.uniform(.6, 5.0))
        for _ in range(n-1):
            candidates, weights = [], []
            for d in DIRS:
                p = (path[-1][0]+d[0], path[-1][1]+d[1])
                if p in occupied or sum((p[0]+dy, p[1]+dx) in occupied
                                        for dy, dx in DIRS) != 1:
                    continue
                candidates.append((p, d))
                weights.append(persistence if d == previous else 1.)
            if not candidates:
                break
            weights = np.asarray(weights)
            p, previous = candidates[int(rng.choice(len(candidates), p=weights/weights.sum()))]
            occupied.add(p)
            path.append(p)
        if len(path) != n or path[0][0] == path[-1][0]:
            continue
        if path[0][0] > path[-1][0]:
            path.reverse()
        if (_above(path) and not allow_above) or (force_above and not _above(path)):
            continue
        return path
    return _fallback_path(n, force_above)


def _fallback_path(n, above=False):
    if above:
        moves = [(-1, 0)] + [(0, 1)]*2 + [(1, 0)]*(n-5) + [(0, -1)]
    elif n >= 7:
        moves = [(0, 1)]*2 + [(1, 0)]*2 + [(0, -1)]*(n-6) + [(1, 0)]
    else:
        moves = [(1, 0)]*(n-1)
    path = [(0, 0)]
    for dy, dx in moves:
        path.append((path[-1][0]+dy, path[-1][1]+dx))
    return path


def _place(rng, path, sequence, shape, center, length, bg, marker, output, used):
    ymin, xmin = min(y for y, _ in path), min(x for _, x in path)
    path = [(y-ymin, x-xmin) for y, x in path]
    bh, bw = max(y for y, _ in path)+1, max(x for _, x in path)+1
    h, w = shape
    if bh > h or bw > w:
        return None
    cy, cx = center
    cross = {(cy+d*dy, cx+d*dx) for dy, dx in DIRS for d in range(1, length+1)}
    forbidden = cross | {(cy, cx)} | {(y+dy, x+dx) for y, x in cross for dy, dx in DIRS}
    positions = [(y, x) for y in range(h-bh+1) for x in range(w-bw+1)]
    for index in rng.permutation(len(positions)):
        top, left = positions[int(index)]
        placed = [(y+top, x+left) for y, x in path]
        if any(p in forbidden for p in placed):
            continue
        if sum(output[p] == bg for p in placed) < 2:
            continue
        grid = np.full(shape, bg, dtype=np.int8)
        for p in cross:
            grid[p] = marker
        for p, color in zip(placed, sequence):
            grid[p] = color
        if _key(grid) in used:
            continue
        try:
            info = _parse(grid)
        except ValueError:
            continue
        return grid, info
    return None


def _make_input(rng, sequence, shape, center, length, bg, marker, used,
                allow_above, *, witness=False, force_above=False, avoid_geometry=None):
    output, details = _weave(sequence, shape, center, length, bg, marker)
    if min(details["visible"]) < 2:
        return None
    for attempt in range(64):
        if attempt < 60:
            path = _random_path(rng, len(sequence), allow_above, force_above)
        else:
            path = _fallback_path(len(sequence), force_above)
            if attempt % 2:
                path = [(y, -x) for y, x in path]
        if avoid_geometry is not None and _path_geometry(path) == avoid_geometry:
            continue
        if witness:
            colors = dict(zip(path, sequence))
            if any([colors[p] for p in order] == sequence for order in _sweeps(path)):
                continue
        placed = _place(rng, path, sequence, shape, center, length, bg, marker, output, used)
        if placed is None:
            continue
        grid, info = placed
        if witness and not _sweep_failure(info, output):
            continue
        used.add(_key(grid))
        return {"input": grid, "output": output}, info
    return None


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _sample_sequence(rng, mode, is_test, palette):
    m = int(rng.integers(5, 7)) if is_test and mode == "stages" else int(rng.integers(2, 5))
    if is_test and mode != "stages":
        m = max(3, m)
    if is_test and mode == "stages":
        sizes = [int(x) for x in rng.integers(1, 3, size=m)]
        sizes[-1] = int(rng.integers(1, 5))
        limit = [2]*(m-1) + [4]
    else:
        cap = 3 if is_test and mode == "boundary" else 4
        sizes = [int(x) for x in rng.integers(1, cap+1, size=m)]
        limit = [cap]*m
        if is_test and mode == "stride":
            index = int(rng.integers(m))
            sizes[index] = int(rng.integers(5, 7))
            limit[index] = 6
    minimum = 8 if is_test else 6
    while sum(sizes) < minimum:
        choices = [i for i in range(m) if sizes[i] < limit[i]]
        i = int(rng.choice(choices))
        sizes[i] += 1
    k = min(len(palette), m)
    colors = list(rng.permutation(palette)[:k])
    while len(colors) < m:
        colors.append(int(rng.choice([c for c in palette if c != colors[-1]])))
    return [int(c) for c, n in zip(colors, sizes) for _ in range(n)]


def _canvas(rng, length, mode, is_test, fallback=False):
    if is_test and mode == "boundary":
        short = 16 if fallback else int(rng.integers(14, 19))
        long = 30 if fallback else int(rng.integers(25, 31))
        shape = (short, long) if fallback or rng.random() < .5 else (long, short)
    else:
        low = 24 if is_test else 18
        shape = (30, 30) if fallback else tuple(map(int, rng.integers(low, 31, size=2)))
    if fallback:
        return shape, (length, length)
    if is_test and mode == "boundary":
        center = [int(rng.integers(length, side-length)) for side in shape]
        axis = int(rng.integers(2))
        center[axis] = int(rng.choice([length, shape[axis]-1-length]))
    else:
        center = [int(rng.integers(length, side-length)) for side in shape]
        if rng.random() < .4:
            center = [(side-1)//2 for side in shape]
    return shape, tuple(center)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(["stages", "stride", "boundary"]))
    shared = bool(rng.random() < .3)
    palette = list(map(int, rng.permutation(10)))
    bg, marker, a, b = palette[:4]
    length = int(rng.integers(1, 4))
    radius = max(2, length)
    shape = tuple(map(int, rng.integers(24, 31, size=2)))
    corner = int(rng.integers(4))
    margin = int(rng.integers(radius+1, radius+3))
    center = (margin if corner < 2 else shape[0]-1-margin,
              margin if corner % 2 == 0 else shape[1]-1-margin)
    p, q = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    sequence = [a]*p + [b]*q + [a]*4
    puzzle, used, infos = {"train": [], "test": []}, set(), []
    teach_above = bool(rng.random() < .4)
    for i, seq in enumerate((sequence, sequence[::-1])):
        made = _make_input(rng, seq, shape, center, length, bg, marker, used,
                           teach_above, witness=True, force_above=teach_above and i == 0,
                           avoid_geometry=_path_geometry(infos[0]["path"]) if i else None)
        if made is None:
            raise AssertionError("controlled-contrast construction exhausted")
        pair, info = made
        puzzle["train"].append(pair)
        infos.append(info)
    allow_above = any(_above(info["path"]) for info in infos)
    for split, count in (("train", ntrain-2), ("test", ntest)):
        is_test = split == "test"
        for index in range(count):
            for attempt in range(66):
                if shared:
                    pair_bg, pair_marker = bg, marker
                    thread_pool = palette[2:2+int(rng.integers(2, 5))]
                else:
                    local = list(map(int, rng.permutation(10)))
                    pair_bg, pair_marker = local[:2]
                    thread_pool = local[2:2+int(rng.integers(2, 5))]
                length = int(rng.choice(sorted({i["length"] for i in infos}))) if is_test else int(rng.integers(1, 4))
                seq = _sample_sequence(rng, mode, is_test, thread_pool)
                shape, center = _canvas(rng, length, mode, is_test, fallback=attempt >= 64)
                made = _make_input(rng, seq, shape, center, length, pair_bg, pair_marker, used,
                                   allow_above, witness=is_test and index == 0)
                if made is not None:
                    pair, info = made
                    puzzle[split].append(pair)
                    if not is_test:
                        infos.append(info)
                    break
            else:
                raise AssertionError("bounded example construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    infos = {"train": [], "test": []}
    used = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in used
            used.add(_key(a))
            info = _parse(a)
            expected, details = _weave(info["sequence"], a.shape, info["center"],
                                       info["length"], info["background"], info["marker"])
            assert np.array_equal(b, expected), "incorrect woven output"
            assert np.array_equal(solve(a), b)
            assert b[info["center"]] == info["background"]
            assert all(b[p] == info["marker"] for p in info["cross"])
            assert min(details["visible"]) >= 2, "an entire growth stage was hidden"
            assert sum(b[p] == info["background"] for p in info["path"]) >= 2
            assert set(map(int, np.unique(b))) <= set(map(int, np.unique(a)))
            # The plane tracks, not merely cropped components, must be induced.
            assert all(sum((y+dy, x+dx) in details["plane"] for dy, dx in DIRS) <= 2
                       for y, x in details["plane"])
            # Every transition's radius increment is n+1: no color-count merging.
            assert details["radius"] == max(2, info["length"]) + len(info["sequence"]) + len(_runs(info["sequence"]))
            infos[split].append(info)
    first, second = infos["train"][:2]
    a, b = puzzle["train"][0]["output"], puzzle["train"][1]["output"]
    assert Counter(first["sequence"]) == Counter(second["sequence"])
    assert second["sequence"] == first["sequence"][::-1]
    assert (first["grid"].shape, first["center"], first["length"], first["background"], first["marker"]) == (
        second["grid"].shape, second["center"], second["length"], second["background"], second["marker"])
    assert _path_geometry(first["path"]) != _path_geometry(second["path"])
    assert not np.array_equal(a, b), "same-histogram contrast must change the answer"
    for info, answer in ((first, a), (second, b)):
        assert _sweep_failure(info, answer), "column sweeps must be distinguished"
        assert not np.array_equal(_render(info, info["sequence"][::-1]), answer)
        assert not np.array_equal(_render(info, _counts_sequence(info["sequence"])), answer)
        assert not np.array_equal(_render(info, straight=True), answer)
        assert not np.array_equal(_render(info, fixed_gap=2), answer)
    assert any(not np.array_equal(_render(i, first_exit=True), p["output"])
               for i, p in zip(infos["train"][:2], puzzle["train"][:2])), "missing re-entry witness"
    assert _sweep_failure(infos["test"][0], puzzle["test"][0]["output"])
    if any(_above(i["path"]) for i in infos["test"]):
        assert any(_above(i["path"]) for i in infos["train"])
    assert {i["length"] for i in infos["test"]} <= {i["length"] for i in infos["train"]}
    # Detect the promised shift from arrays, without consulting an RNG mode.
    tr_runs = [_runs(i["sequence"]) for i in infos["train"]]
    te_runs = [_runs(i["sequence"]) for i in infos["test"]]
    stages = min(map(len, te_runs)) > max(map(len, tr_runs))
    stride = all(max(n for _, n in rs) > max(n for rr in tr_runs for _, n in rr) for rs in te_runs)
    edge = all(min(i["center"][0]-i["length"], i["center"][1]-i["length"],
                   i["grid"].shape[0]-1-i["center"][0]-i["length"],
                   i["grid"].shape[1]-1-i["center"][1]-i["length"]) == 0
               and max(i["grid"].shape)/min(i["grid"].shape) >= 25/18 for i in infos["test"])
    assert stages or stride or edge, "missing structural train/test shift"


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded diagnostics and structural variation across exactly 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, geometry, run_patterns, lengths, stages = set(), set(), set(), set(), set()
    bg_roles, marker_roles, overlaps, reentries = set(), set(), 0, 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                shapes.add(info["grid"].shape)
                geometry.add(_path_geometry(info["path"]))
                rs = _runs(info["sequence"])
                run_patterns.add(tuple(n for _, n in rs))
                lengths.add(len(info["sequence"]))
                stages.add(len(rs))
                bg_roles.add(info["background"])
                marker_roles.add(info["marker"])
                overlaps += any(pair["output"][p] != info["background"] for p in info["path"])
                reentries += not np.array_equal(_render(info, first_exit=True), pair["output"])
    assert len({_signature(p) for p in puzzles}) == 42
    assert len(shapes) >= 45 and len(geometry) >= 100 and len(run_patterns) >= 70
    assert len(lengths) >= 10 and stages == set(range(2, 7))
    assert bg_roles == marker_roles == set(range(10))
    assert overlaps >= 30 and reentries >= 42
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    # Counts: every legal explicit combination, plus non-integer rejection.
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            p = generate_puzzle(859479285, ntrain, ntest)
            assert len(p["train"]) == ntrain and len(p["test"]) == ntest
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, False)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Palette-equivariance: every role is discovered, never tied to a literal.
    permutation = np.array([7, 0, 9, 2, 5, 1, 8, 4, 6, 3], dtype=np.int8)
    pair = puzzles[0]["train"][0]
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    info = _parse(pair["input"])
    # Geometric crop check includes motion outside the smaller canvas.
    cy, cx = info["center"]
    big, _ = _weave(info["sequence"], (90, 90), (45, 45), info["length"],
                    info["background"], info["marker"])
    h, w = info["grid"].shape
    assert np.array_equal(big[45-cy:45-cy+h, 45-cx:45-cx+w], pair["output"])
    # Label corruption must fail, even when schema and palette remain legal.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    target = bad["test"][0]["output"]
    target[0, 0] = (int(target[0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    # Equal-row endpoints, branching, cycles, and unrelated foreground fail.
    for cells in ([(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)],
                  [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (4, 2)],
                  [(1, 1), (1, 2), (2, 1), (2, 2), (4, 4), (5, 4)]):
        grid = np.zeros((8, 8), dtype=np.int8)
        for index, p in enumerate(cells):
            grid[p] = 1 + index % 2
        try:
            _trace(set(cells), grid)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed thread accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "ordered_path_geometries": len(geometry), "run_length_patterns": len(run_patterns),
            "thread_lengths": sorted(lengths), "stage_counts": sorted(stages),
            "source_winding_overlap_pairs": overlaps, "reentry_pairs": reentries}


if __name__ == "__main__":
    print(self_test())
