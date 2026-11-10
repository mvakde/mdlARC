"""Threaded cargo exchange: shape queues ordered by a contracted cord graph.

Only NumPy and the standard library are used. Construction metadata is never
read by solve; output labels are built from the separately constructed route.
"""
from __future__ import annotations

import copy
import itertools
import json

import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SIZES = (3, 5, 7, 9)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _tile(s, color, bg, kind):
    a = np.full((s, s), color, dtype=np.int8)
    if kind == "H":
        a[1:-1, 1:-1] = bg
    return a


def _components(grid, bg):
    """Same-color components separate repeated-color beads from the cord."""
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != bg)}
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = int(grid[start])
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRECTIONS:
                q = (r + dr, c + dc)
                if q in remaining and grid[q] == color:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        yield color, cells


def _analyze(grid):
    """Recover square instances, contract them, then trace actual input pixels."""
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert int(counts.max()) * 2 > grid.size, "background must be a strict majority"
    beads, cord_cells, cord_colors = [], set(), set()
    for color, cells in _components(grid, bg):
        r = min(p[0] for p in cells)
        c = min(p[1] for p in cells)
        s = max(p[0] for p in cells) - r + 1
        w = max(p[1] for p in cells) - c + 1
        bead = None
        if s == w and s in _SIZES:
            patch = grid[r:r+s, c:c+s]
            for kind in ("H", "S"):
                if np.array_equal(patch, _tile(s, color, bg, kind)):
                    bead = dict(top=r, left=c, size=s, color=color, kind=kind)
                    break
        if bead is None:
            cord_cells.update(cells)
            cord_colors.add(color)
        else:
            beads.append(bead)
    beads.sort(key=lambda b: (b["top"], b["left"]))
    assert len(beads) >= 2 and {b["kind"] for b in beads} == {"H", "S"}
    assert len({b["size"] for b in beads}) == 1, "one size per input"
    assert len(cord_colors) == 1 and cord_cells, "one cord color"
    cord = next(iter(cord_colors))
    assert cord not in {b["color"] for b in beads}, "cord color is reserved locally"
    s = beads[0]["size"]
    owner = {}
    for i, b in enumerate(beads):
        r, c = b["top"], b["left"]
        for rr in range(r, r+s):
            for cc in range(c, c+s):
                assert (rr, cc) not in owner
                owner[rr, cc] = i
        for other in beads[:i]:
            assert (abs(r-other["top"]) > s or abs(c-other["left"]) > s), "beads need a gap"
    assert not (set(owner) & cord_cells)
    graph = {("b", i): set() for i in range(len(beads))}
    graph.update({("c", r, c): set() for r, c in cord_cells})
    ports = {i: set() for i in range(len(beads))}
    for r, c in cord_cells:
        node = ("c", r, c)
        for dr, dc in _DIRECTIONS:
            q = (r+dr, c+dc)
            if q in cord_cells:
                graph[node].add(("c", *q))
            elif q in owner:
                i = owner[q]
                b = beads[i]
                br, bc, half = b["top"], b["left"], s//2
                legal_ports = {(br-1, bc+half), (br+s, bc+half),
                               (br+half, bc-1), (br+half, bc+s)}
                assert (r, c) in legal_ports, "cord must meet a side midpoint"
                graph[node].add(("b", i))
                graph["b", i].add(node)
                ports[i].add((r, c))
    assert all(len(p) == 2 for p in ports.values()), "two ports per bead"
    endpoints = [v for v, ns in graph.items() if len(ns) == 1]
    assert len(endpoints) == 2 and all(v[0] == "c" for v in endpoints), "two loose cord ends"
    assert all(len(ns) in (1, 2) for ns in graph.values()), "no branch or cycle"
    assert endpoints[0][1] != endpoints[1][1], "endpoint rows cannot tie"
    start = min(endpoints, key=lambda v: v[1])
    route, seen, previous, current = [], set(), None, start
    while True:
        assert current not in seen
        seen.add(current)
        route.append(current)
        choices = graph[current] - ({previous} if previous is not None else set())
        if not choices:
            break
        assert len(choices) == 1
        previous, current = current, next(iter(choices))
    assert len(seen) == len(graph), "every bead and cord pixel belongs to this chain"
    ordered = [beads[v[1]] for v in route if v[0] == "b"]
    assert len(ordered) == len(beads)
    return dict(background=bg, cord=cord, size=s, beads=ordered,
                cord_cells=cord_cells, route=route, endpoint_rows=sorted(v[1] for v in endpoints),
                above=min(r for r, _ in cord_cells) < start[1])


def _rank_pairs(beads):
    return list(zip([b for b in beads if b["kind"] == "H"],
                    [b for b in beads if b["kind"] == "S"]))


def _exchange(grid, beads, bg, pairs=None):
    """Move each donor's original interior, preserving both rims and the cord."""
    out = grid.copy()
    if pairs is None:
        pairs = _rank_pairs(beads)
    for h, s in pairs:
        size = h["size"]
        r, c = h["top"], h["left"]
        out[r+1:r+size-1, c+1:c+size-1] = s["color"]
        r, c = s["top"], s["left"]
        out[r+1:r+size-1, c+1:c+size-1] = bg
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    try:
        a = _analyze(grid)
    except AssertionError as error:
        raise ValueError(f"not a threaded cargo input: {error}") from error
    return _exchange(grid, a["beads"], a["background"])


def _alternatives(grid, a):
    """Concrete shortcuts, checked on their answers rather than geometry alone."""
    beads, bg = a["beads"], a["background"]
    answers = {"lower_endpoint": _exchange(grid, beads[::-1], bg)}
    for name, key in (("vertical", lambda b: (b["top"], b["left"])),
                      ("horizontal", lambda b: (b["left"], b["top"]))):
        for reverse in (False, True):
            order = sorted(beads, key=key, reverse=reverse)
            answers[name + ("_reverse" if reverse else "")] = _exchange(grid, order, bg)
    rows = [[None, None]]
    for b in beads:
        col = int(b["kind"] == "S")
        if rows[-1][col] is not None:
            rows.append([None, None])
        rows[-1][col] = b
    pairs = [tuple(row) for row in rows if all(b is not None for b in row)]
    answers["greedy_rows"] = _exchange(grid, beads, bg, pairs)
    remaining = [b for b in beads if b["kind"] == "S"]
    pairs = []
    for h in (b for b in beads if b["kind"] == "H"):
        if remaining:
            s = min(remaining, key=lambda b: (abs(b["top"]-h["top"])+abs(b["left"]-h["left"]),
                                             b["top"], b["left"]))
            pairs.append((h, s))
            remaining.remove(s)
    answers["nearest_donor"] = _exchange(grid, beads, bg, pairs)
    answers["ignore_shapes"] = grid.copy()
    return answers


def _walk(rng, length, height, width):
    """Bounded randomized DFS on macro sites; unused neighboring edges stay blank."""
    start = (int(rng.integers(height)), int(rng.integers(width)))
    path, used = [start], {start}
    budget = 1200
    persistence = float(rng.choice((0.3, 1.0, 4.0)))

    def grow():
        nonlocal budget
        if len(path) == length:
            return True
        budget -= 1
        if budget <= 0:
            return False
        directions = list(_DIRECTIONS)
        rng.shuffle(directions)
        if len(path) > 1 and rng.random() < persistence/(1+persistence):
            prev = (path[-1][0]-path[-2][0], path[-1][1]-path[-2][1])
            directions.remove(prev)
            directions.insert(0, prev)
        for dr, dc in directions:
            q = (path[-1][0]+dr, path[-1][1]+dc)
            if not (0 <= q[0] < height and 0 <= q[1] < width) or q in used:
                continue
            path.append(q)
            used.add(q)
            if grow():
                return True
            used.remove(q)
            path.pop()
        return False

    return path if grow() else None


def _macro_path(rng, size, length, above=False, min_turns=0):
    maximum = (30-size)//(size+2) + 1
    shapes = [(h, w) for h in range(2, maximum+1) for w in range(2, maximum+1)
              if h*w >= length]
    for _ in range(40):
        h, w = shapes[int(rng.integers(len(shapes)))]
        path = _walk(rng, length, h, w)
        if path is None or path[0][0] == path[-1][0]:
            continue
        if path[0][0] > path[-1][0]:
            path.reverse()
        if above and min(r for r, _ in path) == path[0][0]:
            continue
        if len(_turn_sites(path)) < min_turns:
            continue
        return path
    if above:
        # Caller supplies the guaranteed five-bead witness if random search fails.
        return None
    cols = min(4, maximum, max(2, length//2))
    return [(r, c if r % 2 == 0 else cols-1-c)
            for r in range((length+cols-1)//cols) for c in range(cols)][:length]


def _turn_sites(path):
    return [i for i in range(1, len(path)-1)
            if (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
            != (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])]


def _centers(rng, count, size):
    gaps = [size+2] * (count-1)
    budget = 30-size-sum(gaps)
    for i in rng.permutation(len(gaps)):
        extra = int(rng.integers(min(3, budget)+1))
        gaps[int(i)] += extra
        budget -= extra
    return np.cumsum([size//2] + gaps).tolist()


def _render(rng, path, indices, kinds, colors, size, bg, cord, reserve=False):
    min_r, min_c = min(r for r, _ in path), min(c for _, c in path)
    path = [(r-min_r, c-min_c) for r, c in path]
    rows = _centers(rng, max(r for r, _ in path)+1, size)
    cols = _centers(rng, max(c for _, c in path)+1, size)
    sites = [(rows[r], cols[c]) for r, c in path]
    grid = np.full((rows[-1]+size//2+1, cols[-1]+size//2+1), bg, dtype=np.int8)
    for (r, c), (rr, cc) in zip(sites, sites[1:]):
        grid[min(r, rr):max(r, rr)+1, min(c, cc):max(c, cc)+1] = cord
    beads = []
    for index, kind, color in zip(indices, kinds, colors):
        r, c = sites[index]
        r, c = r-size//2, c-size//2
        grid[r:r+size, c:c+size] = _tile(size, color, bg, kind)
        beads.append(dict(top=r, left=c, size=size, kind=kind, color=color))
    output = _exchange(grid, beads, bg)
    # Crop to actual content before placing: complete beads or loose ends may touch edges.
    points = np.argwhere(grid != bg)
    lo, hi = points.min(axis=0), points.max(axis=0)+1
    grid, output = grid[lo[0]:hi[0], lo[1]:hi[1]], output[lo[0]:hi[0], lo[1]:hi[1]]
    bh, bw = grid.shape
    if rng.random() < 0.65:
        h, w = min(30, bh+int(rng.integers(5))), min(30, bw+int(rng.integers(5)))
    else:
        h, w = int(rng.integers(bh, 31)), int(rng.integers(bw, 31))
    needed = 2*(int(np.count_nonzero(grid != bg)) + ((size-2)**2 if reserve else 0))
    for _ in range(60):
        if h*w > needed:
            break
        if (h <= w and h < 30) or w == 30:
            h += 1
        else:
            w += 1
    assert h <= 30 and w <= 30 and h*w > needed
    top, left = int(rng.integers(h-bh+1)), int(rng.integers(w-bw+1))
    pair = {}
    for name, source in (("input", grid), ("output", output)):
        canvas = np.full((h, w), bg, dtype=np.int8)
        canvas[top:top+bh, left:left+bw] = source
        pair[name] = canvas
    return pair


def _roles(rng, fixed, demonstration, teaching):
    if fixed is not None:
        return fixed
    choices = [c for c in range(10) if not teaching or c != demonstration]
    bg, cord = rng.choice(choices, 2, replace=False)
    return int(bg), int(cord)


def _make_pair(rng, size, count, free, roles, sign=None, demonstration=None,
               forced_kind=None, witness=False, repeated=False, free_bends=0,
               order_evidence=False):
    bg, cord = roles
    path = _macro_path(rng, size, count+free+2, above=witness, min_turns=free_bends)
    if path is None:
        return None
    turns = _turn_sites(path)
    assert len(turns) >= free_bends
    reserved = set(map(int, rng.choice(turns, free_bends, replace=False)))
    eligible = [i for i in range(1, len(path)-1) if i not in reserved]
    indices = sorted(map(int, rng.choice(eligible, count, replace=False)))
    if sign is None:
        hollow = int(rng.integers(2, count-1)) if order_evidence else int(rng.integers(1, count))
    else:
        choices = [h for h in range(2, count-1) if (count-2*h)*sign > 0]
        hollow = int(rng.choice(choices))
    kinds = list(rng.permutation(["H"]*hollow + ["S"]*(count-hollow)))
    available = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, cord)])))
    if witness or order_evidence:
        colors = available[:min(count, 8)]
        colors += list(map(int, rng.choice(available, max(0, count-8))))
    elif rng.random() < 0.4 and count <= 8:
        colors = available[:count]
    else:
        pool = available[:int(rng.integers(2, min(6, count)+1))]
        colors = list(map(int, rng.choice(pool, count)))
    if forced_kind is not None:
        index = kinds.index(forced_kind)
        if demonstration in colors:
            other = colors.index(demonstration)
            colors[other], colors[index] = colors[index], colors[other]
        else:
            colors[index] = demonstration
        if repeated:
            # Same literal color in both geometries in a single demonstration.
            other = kinds.index("S" if forced_kind == "H" else "H")
            colors[other] = demonstration
    return _render(rng, path, indices, kinds, colors, size, bg, cord, reserve=witness)


def _fallback_witness(rng, size, roles, sign, demonstration):
    # This path has an interior above the upper loose end, and its route order
    # disagrees visibly with both spatial sorts, greedy packing and proximity.
    path = [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (2, 1)]
    kinds = list("SSSHH" if sign > 0 else "HHHSS")
    bg, cord = roles
    colors = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, cord, demonstration)])))[:5]
    colors[kinds.index("H")] = demonstration
    return _render(rng, path, list(range(1, 6)), kinds, colors, size, bg, cord, reserve=True)


def _fallback_test(rng, size, roles, shift):
    bg, cord = roles
    colors = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, cord)])))
    if shift == "cargo":
        path = [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (1, 1),
                (2, 1), (2, 2), (3, 2), (3, 1)]
        kinds, indices = list("SSSHHSSS"), list(range(1, 9))
    elif shift == "routing":
        path = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3),
                (2, 2), (2, 1), (2, 0), (3, 0), (3, 1), (3, 2)]
        kinds, indices = list("SSSHH"), [1, 2, 5, 6, 7]
    else:
        path = [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (2, 1)]
        kinds, indices = list("SSSHH"), list(range(1, 6))
    return _render(rng, path, indices, kinds, colors[:len(kinds)], size, bg, cord)


def _unique(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    a = _analyze(pair["input"])
    available = [c for c in range(10) if c not in (a["background"], a["cord"])]
    # 64 distinct recolorings exceed the maximum 13 preceding inputs.
    for colors in itertools.product(available, repeat=2):
        grid = pair["input"].copy()
        for b, color in zip(a["beads"], colors):
            r, c, s = b["top"], b["left"], b["size"]
            grid[r:r+s, c:c+s] = _tile(s, color, a["background"], b["kind"])
        if _key(grid) not in seen:
            return dict(input=grid, output=solve(grid))
    raise AssertionError("uniqueness fallback exhausted")


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("cargo", "routing", "scale")))
    demonstration = int(rng.integers(10))
    available = [c for c in range(10) if c != demonstration]
    fixed = tuple(map(int, rng.choice(available, 2, replace=False))) if rng.random() < 0.35 else None
    common_size = int(rng.choice((3, 5)))
    first_sign = int(rng.choice((-1, 1)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            teaching = split == "train" and index < 2
            witness = split == "train" and index == 0
            test_evidence = split == "test" and index == 0
            if shift == "scale":
                size = (3 if index % 2 == 0 else 5) if split == "train" else (
                    7 if test_evidence else int(rng.choice((7, 9))))
            else:
                size = common_size
            maximum = ((30-size)//(size+2)+1)**2
            if teaching:
                count, free = int(rng.choice((5, 7))), int(rng.integers(3))
                sign = first_sign if index == 0 else -first_sign
            elif split == "test" and shift == "cargo":
                count, free, sign = int(rng.integers(8, min(14, maximum-2)+1)), int(rng.integers(3)), None
            elif split == "test" and shift == "routing":
                count, free, sign = int(rng.choice((5, 7))), int(rng.integers(3, 7)), None
            elif test_evidence:
                count, free, sign = int(rng.choice((5, 7))), int(rng.integers(3)), None
            else:
                count = int(rng.integers(2, min(7, maximum-2)+1))
                free, sign = int(rng.integers(3)), None
            free = min(free, maximum-count-2)
            roles = _roles(rng, fixed, demonstration, teaching)
            for _ in range(24 if witness or test_evidence else 1):
                pair = _make_pair(rng, size, count, free, roles, sign,
                                  demonstration, "H" if witness else ("S" if teaching else None),
                                  witness, repeated=teaching and not witness,
                                  free_bends=3 if split == "test" and shift == "routing" else 0,
                                  order_evidence=test_evidence)
                if pair is None:
                    continue
                if witness or test_evidence:
                    a = _analyze(pair["input"])
                    if not all(not np.array_equal(pair["output"], alt)
                               for alt in _alternatives(pair["input"], a).values()):
                        continue
                break
            else:
                pair = (_fallback_witness(rng, size, roles, first_sign, demonstration) if witness
                        else _fallback_test(rng, size, roles, shift))
            pair = _unique(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            expected = _exchange(inp, a["beads"], a["background"])
            assert np.array_equal(expected, out), "wrong cargo destination or vacated source"
            assert out.shape == inp.shape and not np.array_equal(out, inp)
            assert np.array_equal(np.bincount(inp.ravel(), minlength=10),
                                  np.bincount(out.ravel(), minlength=10)), "material must be conserved"
            interiors = np.zeros(inp.shape, dtype=bool)
            for b in a["beads"]:
                r, c, s = b["top"], b["left"], b["size"]
                interiors[r+1:r+s-1, c+1:c+s-1] = True
            assert np.array_equal(inp[~interiors], out[~interiors]), "cord, rims and canvas stay fixed"
            pairs = _rank_pairs(a["beads"])
            assert np.count_nonzero(inp != out) == 2*len(pairs)*(a["size"]-2)**2
            for h, s in pairs:
                for b, color in ((h, s["color"]), (s, a["background"])):
                    r, c, size = b["top"], b["left"], b["size"]
                    assert np.all(out[r+1:r+size-1, c+1:c+size-1] == color)
            if split == "train":
                training.append((inp, out, a))
            else:
                testing.append((inp, out, a))
    imbalances = [sum(b["kind"] == "S" for b in a["beads"])
                  - sum(b["kind"] == "H" for b in a["beads"]) for _, _, a in training]
    assert min(imbalances) < 0 < max(imbalances), "teach unmatched tails of either class"
    for name in _alternatives(training[0][0], training[0][2]):
        for split, examples in (("train", training), ("test", testing)):
            assert any(not np.array_equal(out, _alternatives(inp, a)[name])
                       for inp, out, a in examples), f"missing {split} {name} counterexample"
    assert any(a["above"] for _, _, a in training), "teach upper endpoint, not uppermost pixel"
    assert any({b["kind"] for b in a["beads"] if b["color"] == color} == {"H", "S"}
               for _, _, a in training for color in range(10)), "shape is not a color dictionary"


def _cord_turns(a):
    """Count turns on bare cord, away from contracted squares."""
    cord_turns = 0
    for before, v, after in zip(a["route"], a["route"][1:], a["route"][2:]):
        if all(p[0] == "c" for p in (before, v, after)):
            cord_turns += (v[1]-before[1], v[2]-before[2]) != (after[1]-v[1], after[2]-v[2])
    return cord_turns


def _features(pair):
    a = _analyze(_grid(pair["input"]))
    beads = a["beads"]
    h = sum(b["kind"] == "H" for b in beads)
    bead_iter = iter(beads)
    points = []
    for v in a["route"]:
        if v[0] == "c":
            points.append(v[1:])
        else:
            b = next(bead_iter)
            points.append((b["top"]+a["size"]//2, b["left"]+a["size"]//2))
    steps = [(int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))) for a, b in zip(points, points[1:])]
    turns = sum(a != b for a, b in zip(steps, steps[1:]))
    return dict(size=a["size"], background=a["background"], cord=a["cord"],
                count=len(beads), hollow=h, solid=len(beads)-h, above=a["above"],
                cord_length=len(a["cord_cells"]), cord_turns=_cord_turns(a), turns=turns,
                repeat=len({b["color"] for b in beads}) < len(beads),
                shape=tuple(pair["input"].shape))


def self_test():
    """Bounded semantic, malformed-input, interface and 42-seed diversity checks."""
    puzzles = [generate_puzzle(seed, num_test=1+seed % 3) for seed in range(42)]
    features, signatures, shapes, geometry = [], set(), set(), set()
    shifts = dict(cargo=0, routing=0, scale=0)
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[k]) for split in ("train", "test") for p in puzzle[split] for k in ("input", "output"))
        signatures.add(signature)
        tr, te = [[_features(pair) for pair in puzzle[split]] for split in ("train", "test")]
        if all(f["count"] > max(t["count"] for t in tr) for f in te):
            shifts["cargo"] += 1
        elif all(f["cord_turns"] > max(t["cord_turns"] for t in tr) for f in te):
            shifts["routing"] += 1
        elif all(f["size"] not in {t["size"] for t in tr} for f in te):
            shifts["scale"] += 1
        else:
            raise AssertionError("no visible train/test shift")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                f = _features(pair)
                features.append(f)
                shapes.add(f["shape"])
                # Normalize translation and ignore colors, but keep bead/cord distinction.
                a = _analyze(pair["input"])
                coords = [(b["top"], b["left"]) for b in a["beads"]]
                origin = coords[0]
                geometry.add((a["size"], tuple((r-origin[0], c-origin[1]) for r, c in coords),
                              tuple(sorted((r-origin[0], c-origin[1]) for r, c in a["cord_cells"]))))
        # Toggling one hollow changes a different bead's destination: A is causal.
        inp = puzzle["train"][0]["input"].copy()
        a = _analyze(inp)
        hollow = [b for b in a["beads"] if b["kind"] == "H"]
        b = hollow[0]
        r, c, s = b["top"], b["left"], b["size"]
        inp[r+1:r+s-1, c+1:c+s-1] = b["color"]
        changed = solve(inp)
        outside = np.ones(inp.shape, dtype=bool)
        outside[r:r+s, c:c+s] = False
        assert np.any(changed[outside] != puzzle["train"][0]["output"][outside])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(geometry) >= 150
    assert {f["size"] for f in features} == set(_SIZES)
    assert {f["background"] for f in features} == set(range(10))
    assert {f["cord"] for f in features} == set(range(10))
    assert len({f["count"] for f in features}) >= 9
    assert len({(f["hollow"], f["solid"]) for f in features}) >= 25
    assert len({f["cord_turns"] for f in features}) >= 4
    assert len({f["turns"] for f in features}) >= 6
    assert all(n >= 5 for n in shifts.values())
    assert any(f["count"] == 2 for f in features)
    assert any(f["count"] > 9 for f in features)
    assert sum(f["repeat"] for f in features) >= 42
    assert any(f["hollow"] == f["solid"] for f in features)
    assert _key(puzzles[0]["test"][0]["input"]) == _key(generate_puzzle(0)["test"][0]["input"])
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            validate_generated_puzzle(generate_puzzle(17, train, test))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = copy.deepcopy(puzzles[0])
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # A stray cord cell, a genuine internal cut, and a branch must be rejected.
    base = puzzles[0]["train"][0]["input"]
    a = _analyze(base)
    for mode in ("stray", "cut", "branch"):
        bad = base.copy()
        if mode == "cut":
            # Removing a loose end merely shortens a valid cord. Cut strictly
            # inside the route, keeping both original endpoints instead.
            v = next(v for v in a["route"][2:-2] if v[0] == "c")
            bad[v[1], v[2]] = a["background"]
        else:
            end_cells = {(a["route"][i][1], a["route"][i][2]) for i in (0, -1)}
            for r, c in np.argwhere(base == a["background"]):
                neighbors = [(int(r)+dr, int(c)+dc) for dr, dc in _DIRECTIONS]
                cord_neighbors = {p for p in neighbors if p in a["cord_cells"]}
                touches_bead = any(b["top"]-1 <= r <= b["top"]+b["size"]
                                   and b["left"]-1 <= c <= b["left"]+b["size"] for b in a["beads"])
                if not touches_bead and ((mode == "stray" and not cord_neighbors)
                                        or (mode == "branch" and len(cord_neighbors) == 1
                                            and not cord_neighbors & end_cells)):
                    bad[r, c] = a["cord"]
                    try:
                        _analyze(bad)
                    except AssertionError:
                        break
                    bad[r, c] = a["background"]
            else:
                raise AssertionError("could not construct malformed-input control")
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed cord accepted")
    for path in ([(0, 0), (1, 0), (1, 1), (0, 1)],
                 [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]):
        bad = _render(np.random.default_rng(7), path, [1, 2], ["H", "S"],
                      [1, 2], 3, 0, 3)["input"]
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("tied endpoints or a closed cord accepted")
    # A legal route that cannot come from the macro spacing convention: a
    # turning cord column is only two cells beyond a bead's right edge.
    free_form = np.zeros((14, 12), dtype=np.int8)
    free_form[1:4, 1:4] = _tile(3, 1, 0, "H")
    free_form[7:10, 6:9] = _tile(3, 2, 0, "S")
    free_form[0, 2] = 3
    free_form[2, 4:11] = 3
    free_form[3:9, 10] = 3
    free_form[8, 9] = 3
    free_form[10:13, 7] = 3
    expected = free_form.copy()
    expected[2, 2], expected[8, 7] = 2, 0
    assert np.array_equal(solve(free_form), expected)
    for size in (3, 5, 7):
        for sign in (-1, 1):
            fixture = _fallback_witness(np.random.default_rng(size), size, (0, 9), sign, 1)
            a = _analyze(fixture["input"])
            assert np.array_equal(solve(fixture["input"]), fixture["output"])
            assert all(not np.array_equal(alt, fixture["output"])
                       for alt in _alternatives(fixture["input"], a).values())
    for shift, size in (("cargo", 3), ("cargo", 5), ("routing", 3), ("routing", 5), ("scale", 7)):
        fixture = _fallback_test(np.random.default_rng(size), size, (0, 9), shift)
        a = _analyze(fixture["input"])
        assert all(not np.array_equal(alt, fixture["output"])
                   for alt in _alternatives(fixture["input"], a).values())
        assert np.array_equal(solve(fixture["input"]), fixture["output"])
        if shift == "routing":
            assert _cord_turns(a) >= 3
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(signatures),
                input_shapes=len(shapes), route_geometries=len(geometry),
                sizes=sorted({f["size"] for f in features}),
                bead_counts=sorted({f["count"] for f in features}),
                class_count_pairs=len({(f["hollow"], f["solid"]) for f in features}),
                cord_turn_counts=sorted({f["cord_turns"] for f in features}),
                total_turn_counts=sorted({f["turns"] for f in features}),
                visible_shift_puzzles=shifts)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
