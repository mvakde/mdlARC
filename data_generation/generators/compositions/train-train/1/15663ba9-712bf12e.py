"""Corner sluices: local drifting paths can squeeze through reentrant elbows.

Only standard-library modules and NumPy are used.  No parent generator or stored
example is imported.  All geometry used by solve is recovered from the input.
"""

from collections import deque
import numpy as np


HOST_SEEDS = (0, 66381122, 208471402, 686529656, 413391873, 922610114,
              87864500, 984008892, 566462442, 803543864, 211073333,
              424410216, 708963894, 352795428, 630610193, 104615493,
              336709081, 524667141, 398612901, 624080117, 193545812,
              355114545, 797860983, 509318595, 549931846, 752747569,
              246612318, 69033326, 473802568, 505434134, 849820444,
              511547884, 314189331, 251563848, 722389297, 609622941,
              509585992, 798108843, 515416542, 38012928, 488903147,
              47211255)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _neighbors(cell, cells):
    r, c = cell
    return sorted((r+dr, c+dc) for dr, dc in DIRS
                  if (r+dr, c+dc) in cells)


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r+dr, c+dc)
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _contour(cells):
    """Trace one simple raster cycle; label 0 straight, 1 convex, 2 concave."""
    if len(cells) < 8 or any(len(_neighbors(p, cells)) != 2 for p in cells):
        raise ValueError("Contours must be simple, one-cell-wide closed cycles")
    start = min(cells)
    path, seen = [start], {start}
    previous, current = start, _neighbors(start, cells)[0]
    while current != start:
        if current in seen:
            raise ValueError("Repeated contour vertex")
        path.append(current)
        seen.add(current)
        following = [q for q in _neighbors(current, cells) if q != previous]
        previous, current = current, following[0]
    if seen != cells:
        raise ValueError("Incomplete contour trace")
    area = sum(c*path[(i+1) % len(path)][0]
               - path[(i+1) % len(path)][1]*r
               for i, (r, c) in enumerate(path))
    if not area:
        raise ValueError("Empty polygon")
    classes = {}
    for i, (r, c) in enumerate(path):
        pr, pc = path[i-1]
        nr, nc = path[(i+1) % len(path)]
        cross = (c-pc)*(nr-r) - (r-pr)*(nc-c)
        classes[(r, c)] = 0 if not cross else 1 if cross*area > 0 else 2
    if (sum(k == 1 for k in classes.values())
            - sum(k == 2 for k in classes.values()) != 4):
        raise ValueError("Invalid orthogonal turn balance")
    return {"cells": cells, "path": path, "classes": classes}


def _analyze(input_grid):
    original = _grid(input_grid)
    colors, counts = np.unique(original, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(max(counts)) <= original.size // 2:
        raise ValueError("Background must occupy more than half the grid")
    # Only sources touch the boundary, on one edge, never at a corner.
    edges = (original[-1, :], original[:, -1], original[0, :], original[:, 0])
    occupied = [i for i, edge in enumerate(edges) if np.any(edge != bg)]
    if len(occupied) != 1:
        raise ValueError("Sources must identify exactly one boundary edge")
    edge = occupied[0]
    rotation = (0, 3, 2, 1)[edge]
    a = np.ascontiguousarray(np.rot90(original, rotation))
    source_colors = set(map(int, a[-1])) - {bg}
    if len(source_colors) != 1:
        raise ValueError("One source color is required")
    ink = source_colors.pop()
    sources = [(a.shape[0]-1, int(c)) for c in np.flatnonzero(a[-1] == ink)]
    if (not 1 <= len(sources) <= 4 or a[-1, 0] != bg or a[-1, -1] != bg
            or int(np.count_nonzero(a == ink)) != len(sources)):
        raise ValueError("Source cells occur only on the source edge")
    if any(y[1]-x[1] < 2 for x, y in zip(sources, sources[1:])):
        raise ValueError("Sources must have a blank cell between them")
    infos = [_contour(cells) for cells in _components((a != bg) & (a != ink))]
    if not infos:
        raise ValueError("At least one chamber is required")
    owner, kinds = {}, {}
    for i, info in enumerate(infos):
        if len({int(a[p]) for p in info["cells"]}) != 1:
            raise ValueError("Each contour must initially be monochrome")
        for p in info["cells"]:
            owner[p] = i
            kinds[p] = info["classes"][p]
    for (r, c), i in owner.items():
        if not (0 < r < a.shape[0]-1 and 0 < c < a.shape[1]-1):
            raise ValueError("Contours must be strictly inside the boundary")
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r+dr, c+dc) in owner and owner[(r+dr, c+dc)] != i:
                    raise ValueError("Different contours need a blank-cell gap")
    return {"grid": a, "background": bg, "ink": ink, "sources": sources,
            "infos": infos, "owner": owner, "kinds": kinds,
            "rotation": rotation}


def _walk(analysis, mode="sluices"):
    """Use the immutable input as terrain, so paths can merge without ordering."""
    a = analysis["grid"]
    bg, ink = analysis["background"], analysis["ink"]
    owner, kinds = analysis["owner"], analysis["kinds"]
    h, w = a.shape
    out, routes = a.copy(), []
    for start in analysis["sources"]:
        r, c = start
        cells, gates, moves = [start], [], []
        while r > 0:
            up = (r-1, c)
            right = (r, c+1)
            diagonal = (r-1, c+1)
            choices = ((right, "R"), (up, "U")) if mode == "right_first" else (
                (up, "U"), (right, "R"))
            if mode == "vertical":
                choices = choices[:1]
            destination = None
            for q, move in choices:
                if q[1] < w and a[q] == bg:
                    destination, letter = q, move
                    break
            if destination is None and mode != "sealed" and c+1 < w:
                allowed = (1, 2) if mode == "all_bends" else (2,)
                if (kinds.get(diagonal) in allowed and up in owner
                        and right in owner
                        and owner[up] == owner[right] == owner[diagonal]):
                    destination, letter = diagonal, "D"
                    gates.append(diagonal)
            if destination is None:
                break
            r, c = destination
            out[r, c] = ink
            cells.append(destination)
            moves.append(letter)
            if len(cells) > h+w:
                raise AssertionError("A monotone trajectory cannot cycle")
        routes.append({"cells": cells, "gates": gates, "moves": "".join(moves),
                       "ending": "edge" if r == 0 else "blocked"})
    return out, routes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the source edge and loop geometry, then apply the fixed rule."""
    analysis = _analyze(input_grid)
    result, _ = _walk(analysis)
    return np.ascontiguousarray(np.rot90(result, -analysis["rotation"]))


def _raster(vertices):
    cells = []
    for i, (r, c) in enumerate(vertices):
        er, ec = vertices[(i+1) % len(vertices)]
        dr, dc = er-r, ec-c
        if (dr == 0) == (dc == 0):
            raise ValueError("Polygon edges must be nonempty and axial")
        length = abs(dr)+abs(dc)
        for t in range(length):
            cells.append((r+t*int(np.sign(dr)), c+t*int(np.sign(dc))))
    if len(set(cells)) != len(cells):
        raise ValueError("Polygon intersects itself")
    _contour(set(cells))
    return set(cells)


def _edit(vertices, rng):
    """Local geometric edits, not a catalog of whole-puzzle layouts."""
    i = int(rng.integers(len(vertices)))
    if rng.random() < .50:
        # Cut or extend a shoulder; this adds one reentrant turn.
        p, q, s = vertices[i-1], vertices[i], vertices[(i+1) % len(vertices)]
        la, lb = abs(p[0]-q[0])+abs(p[1]-q[1]), abs(s[0]-q[0])+abs(s[1]-q[1])
        if min(la, lb) < 2:
            return None
        x = int(rng.integers(1, min(4, la)))
        y = int(rng.integers(1, min(4, lb)))
        u = ((p[0]-q[0])//la, (p[1]-q[1])//la)
        v = ((s[0]-q[0])//lb, (s[1]-q[1])//lb)
        a = (q[0]+x*u[0], q[1]+x*u[1])
        b = (a[0]+y*v[0], a[1]+y*v[1])
        c = (q[0]+y*v[0], q[1]+y*v[1])
        return vertices[:i]+[a, b, c]+vertices[i+1:]
    p, q = vertices[i], vertices[(i+1) % len(vertices)]
    length = abs(p[0]-q[0])+abs(p[1]-q[1])
    if length < 3:
        return None
    x = int(rng.integers(1, length-1))
    y = int(rng.integers(x+1, length))
    depth = int(rng.integers(1, 4)) * int(rng.choice((-1, 1)))
    dr, dc = (q[0]-p[0])//length, (q[1]-p[1])//length
    nr, nc = dc*depth, -dr*depth
    added = [(p[0]+x*dr, p[1]+x*dc),
             (p[0]+x*dr+nr, p[1]+x*dc+nc),
             (p[0]+y*dr+nr, p[1]+y*dc+nc),
             (p[0]+y*dr, p[1]+y*dc)]
    return vertices[:i+1]+added+vertices[i+1:]


def _base(rng, depth, source_count, merge_demo=False):
    # Build a nested entry corridor. Each shell has its own bounded interior.
    h = int(rng.integers(12+5*(depth-1), min(31, 21+5*(depth-1))))
    w = int(rng.integers(15+3*(depth-1), 31))
    top, left = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    bottom, right = h-3, w-2
    end = max(left+5, w//2-depth)
    vertices = []
    previous_end = None
    for layer in range(depth):
        if layer:
            top += 2
            left += 2
            right -= 2
            bottom = gate_row-2
            end = previous_end+int(rng.integers(1, 3))
        # Reserve enough interior height for the remaining shells.
        max_depth = min(4, bottom-top-3-5*(depth-layer-1))
        if max_depth < 2 or end > right-2:
            raise ValueError("Sampled nested chamber is too small")
        notch_depth = int(rng.integers(2, max_depth+1))
        gate_row = bottom-notch_depth
        max_start = end-3 if previous_end is None else min(end-2, previous_end-1)
        if max_start < left+2:
            raise ValueError("Notch has no accessible mouth")
        start = int(rng.integers(left+2, max_start+1))
        vertices.append([(top, left), (top, right), (bottom, right),
                         (bottom, end), (gate_row, end), (gate_row, start),
                         (bottom, start), (bottom, left)])
        if layer == 0:
            outer_start, outer_end = start, end
        previous_end = end
    # At least one source rises inside the mouth; others can arrive from its left
    # or stay outside to the right. They are not assigned artificial route bands.
    primary = outer_end-2
    columns = [primary, outer_start-1] if merge_demo else [primary]
    possible = [c for c in range(1, w-1) if abs(c-primary) >= 2]
    rng.shuffle(possible)
    for c in possible:
        if len(columns) == source_count:
            break
        if all(abs(c-d) >= 2 for d in columns):
            columns.append(c)
    if len(columns) != source_count:
        raise ValueError("Cannot space sources")
    return h, w, vertices, sorted(columns)


def _draw(h, w, polygons, columns, palette):
    bg, ink, colors = palette
    a = np.full((h, w), bg, dtype=np.int8)
    occupied = set()
    for i, vertices in enumerate(polygons):
        cells = _raster(vertices)
        if any(not (0 < r < h-1 and 0 < c < w-1) for r, c in cells):
            raise ValueError("Contour outside the margin")
        if any((r+dr, c+dc) in occupied for r, c in cells
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            raise ValueError("Contours touch")
        occupied.update(cells)
        for p in cells:
            a[p] = colors[i % len(colors)]
    a[-1, columns] = ink
    return a


def _has_merge(routes):
    return any(set(t["cells"][1:]) & set(s["cells"][1:])
               for i, t in enumerate(routes) for s in routes[i+1:])


def _witness(analysis, required_depth, merge_demo=False):
    out, routes = _walk(analysis)
    if max(len(t["gates"]) for t in routes) != required_depth:
        return False
    if merge_demo and not _has_merge(routes):
        return False
    used = set().union(*(set(t["gates"]) for t in routes))
    if not any(k == 2 and p not in used for p, k in analysis["kinds"].items()):
        return False
    # Preserve evidence against treating every corner as a door or sealing all
    # walls, and ensure up priority and a return to up after a rightward detour.
    if not any("RU" in t["moves"] or "RDU" in t["moves"] for t in routes):
        return False
    if not any("R" in t["moves"] for t in routes):
        return False
    return all(not np.array_equal(out, _walk(analysis, mode)[0])
               for mode in ("sealed", "all_bends", "right_first", "vertical"))


def _fallback_pair(rng, depth, source_count, turn):
    """A spacious witness of exactly the same rule if random packing exhausts."""
    polygons = []
    for i in range(depth):
        top, left, right = 1+2*i, 1+2*i, 28-2*i
        bottom, end, start = 27-6*i, 10+2*i, 3+2*i
        gate = bottom-2
        polygons.append([(top, left), (top, right), (bottom, right),
                         (bottom, end), (gate, end), (gate, start),
                         (bottom, start), (bottom, left)])
    columns = [8] if source_count == 1 else [2, 8, 22, 26][:source_count]
    colors = [int(x) for x in rng.permutation(10)]
    a = _draw(30, 30, polygons, columns, (colors[0], colors[1], colors[2:5]))
    analysis = _analyze(a)
    assert _witness(analysis, depth, source_count > 1)
    out, _ = _walk(analysis)
    return {"input": np.ascontiguousarray(np.rot90(a, turn)),
            "output": np.ascontiguousarray(np.rot90(out, turn))}


def _make_pair(rng, depth, source_count, turn, extra, merge_demo=False):
    # All retries are bounded. Base construction alone already exhibits the
    # mechanism; accepted mutations add independent contour structure.
    for _ in range(180):
        try:
            h, w, polygons, columns = _base(rng, depth, source_count, merge_demo)
            order = [int(x) for x in rng.permutation(10)]
            palette = (order[0], order[1], order[2:2+int(rng.integers(1, 4))])
            a = _draw(h, w, polygons, columns, palette)
            analysis = _analyze(a)
            if not _witness(analysis, depth, merge_demo):
                continue
        except ValueError:
            continue
        # Mutate arbitrary edges/shoulders of independently chosen shells.
        for _ in range(int(rng.integers(12, 31))):
            index = int(rng.integers(len(polygons)))
            proposal = _edit(polygons[index], rng)
            if proposal is None:
                continue
            trial = polygons[:]
            trial[index] = proposal
            try:
                candidate = _draw(h, w, trial, columns, palette)
                candidate_analysis = _analyze(candidate)
                if _witness(candidate_analysis, depth, merge_demo):
                    polygons, a, analysis = trial, candidate, candidate_analysis
            except ValueError:
                pass
        # Optional small chambers fit in remaining actual free space, including
        # inside an existing shell. They need not be reached by a path.
        if extra:
            for _ in range(35):
                sh, sw = int(rng.integers(3, 7)), int(rng.integers(3, 7))
                if sh+2 >= h or sw+2 >= w:
                    continue
                r = int(rng.integers(1, h-sh))
                c = int(rng.integers(1, w-sw))
                small = [(r, c), (r, c+sw-1), (r+sh-1, c+sw-1), (r+sh-1, c)]
                trial = polygons+[small]
                try:
                    candidate = _draw(h, w, trial, columns, palette)
                    candidate_analysis = _analyze(candidate)
                    if _witness(candidate_analysis, depth, merge_demo):
                        polygons, a, analysis = trial, candidate, candidate_analysis
                        break
                except ValueError:
                    pass
        out, _ = _walk(analysis)
        return {"input": np.ascontiguousarray(np.rot90(a, turn)),
                "output": np.ascontiguousarray(np.rot90(out, turn))}
    return _fallback_pair(rng, depth, source_count, turn)


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    orientation = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            depth = (1 if index == 0 else 2 if index == 1 else
                     int(rng.integers(1, 3))) if split == "train" else 3
            source_count = (2 if index == 0 else 3) if split == "train" and index < 2 else int(rng.integers(1, 5))
            turn = ((orientation+index) % 4 if split == "train" and index < 2
                    else int(rng.integers(4)))
            for _ in range(12):
                pair = _make_pair(rng, depth, source_count, turn, bool(rng.integers(2)),
                                  merge_demo=split == "train" and index == 0)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not obtain distinct inputs")
    return puzzle


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _point_inside(path, row, col):
    crossings = sum(c1 == c2 and min(r1, r2) < row < max(r1, r2) and c1 > col
                    for (r1, c1), (r2, c2) in zip(path, path[1:]+path[:1]))
    return crossings % 2 == 1


def _check_contour_independently(info):
    # Ray crossings in four quadrants check the local interior angle, without
    # relying on the signed-turn formula used by the production classifier.
    for (r, c), kind in info["classes"].items():
        quadrants = sum(_point_inside(info["path"], r+dr, c+dc)
                        for dr in (-.25, .25) for dc in (-.25, .25))
        assert quadrants == (2 if kind == 0 else 1 if kind == 1 else 3)
    rows, cols = zip(*info["cells"])
    h, w = max(rows)-min(rows)+3, max(cols)-min(cols)+3
    wall = {(r-min(rows)+1, c-min(cols)+1) for r, c in info["cells"]}
    outside, queue = {(0, 0)}, deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            q = r+dr, c+dc
            if 0 <= q[0] < h and 0 <= q[1] < w and q not in wall and q not in outside:
                outside.add(q)
                queue.append(q)
    assert len(wall)+len(outside) < h*w, "A chamber must enclose raster cells"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_depths, test_depths, source_counts, orientations = set(), [], [], set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            analysis = _analyze(a)
            expected, routes = _walk(analysis)
            normalized_b = np.rot90(b, analysis["rotation"])
            assert np.array_equal(normalized_b, expected)
            assert np.array_equal(solve(a), b)
            for info in analysis["infos"]:
                _check_contour_independently(info)
            terrain = analysis["grid"]
            changed = terrain != normalized_b
            assert changed.any() and np.all(normalized_b[changed] == analysis["ink"])
            visited = set().union(*(set(t["cells"][1:]) for t in routes))
            assert visited == set(map(tuple, np.argwhere(changed)))
            for p in visited:
                assert terrain[p] == analysis["background"] or analysis["kinds"].get(p) == 2
            for p, kind in analysis["kinds"].items():
                if kind in (0, 1):
                    assert normalized_b[p] == terrain[p]
            for t in routes:
                gate_owners = [analysis["owner"][p] for p in t["gates"]]
                assert len(set(gate_owners)) == len(gate_owners)
                for outer, inner_gate in zip(gate_owners, t["gates"][1:]):
                    assert _point_inside(analysis["infos"][outer]["path"],
                                         inner_gate[0]+.25, inner_gate[1]+.25)
                for p, q, letter in zip(t["cells"], t["cells"][1:], t["moves"]):
                    assert (q[0]-p[0], q[1]-p[1]) == {"U": (-1, 0), "R": (0, 1), "D": (-1, 1)}[letter]
                    if letter == "R":
                        assert terrain[p[0]-1, p[1]] != analysis["background"]
                    if letter == "D":
                        assert analysis["kinds"][q] == 2
                        assert terrain[p[0]-1, p[1]] != analysis["background"]
                        assert terrain[p[0], p[1]+1] != analysis["background"]
                end = t["cells"][-1]
                if end[0] > 0:
                    assert terrain[end[0]-1, end[1]] != analysis["background"]
                    assert end[1]+1 == terrain.shape[1] or terrain[end[0], end[1]+1] != analysis["background"]
                    assert analysis["kinds"].get((end[0]-1, end[1]+1)) != 2
            depth = max(len(t["gates"]) for t in routes)
            assert _witness(analysis, depth)
            (train_depths if split == "train" else test_depths).append(depth)
            if split == "train":
                if not train_depths[:-1]:
                    assert _has_merge(routes)
                source_counts.add(len(routes))
                orientations.add(analysis["rotation"])
    assert train_depths[0] == 1 and train_depths[1] == 2
    assert max(train_depths) == 2 and set(test_depths) == {3}
    assert len(source_counts) >= 2 and len(orientations) >= 2


def _puzzle_key(puzzle):
    return tuple(_key(p[k]) for split in ("train", "test")
                 for p in puzzle[split] for k in ("input", "output"))


def self_test():
    """Bounded 42-seed semantic, metamorphic, count and diversity checks."""
    shapes, structures, palettes, concaves, counts, turns, signatures = (set() for _ in range(7))
    merged = untouched_concave = adjacent_bends = extra_chambers = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                out, routes = _walk(a)
                shapes.add(pair["input"].shape)
                canon = np.zeros(a["grid"].shape, dtype=np.int8)
                canon[(a["grid"] != a["background"]) & (a["grid"] != a["ink"])] = 1
                canon[a["grid"] == a["ink"]] = 2
                structures.add(_key(canon))
                palettes.add((a["background"], a["ink"], tuple(sorted(set(map(int, a["grid"].flat)) - {a["background"], a["ink"]}))))
                counts.add(len(routes))
                turns.add(a["rotation"])
                gate_set = set().union(*(set(t["gates"]) for t in routes))
                untouched_concave += sum(k == 2 and p not in gate_set for p, k in a["kinds"].items())
                merged += any(set(t["cells"][1:]) & set(s["cells"][1:]) for i, t in enumerate(routes) for s in routes[i+1:])
                extra_chambers += len(a["infos"]) > max(len(t["gates"]) for t in routes)
                for info in a["infos"]:
                    concaves.add(sum(k == 2 for k in info["classes"].values()))
                    pattern = [info["classes"][p] for p in info["path"]]
                    adjacent_bends += any(pattern[i] and pattern[(i+1) % len(pattern)] for i in range(len(pattern)))
                # Equivariance under all quarter turns and arbitrary palette relabeling.
                rotated = np.rot90(pair["input"], 1)
                assert np.array_equal(solve(rotated), np.rot90(pair["output"], 1))
                permutation = np.random.default_rng(seed+123).permutation(10)
                assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert len(signatures) == 42 and len(shapes) >= 50 and len(structures) >= 180
    assert len(palettes) >= 80 and len(concaves) >= 5
    assert counts == {1, 2, 3, 4} and turns == {0, 1, 2, 3}
    assert merged and untouched_concave and adjacent_bends and extra_chambers
    assert _puzzle_key(generate_puzzle(19)) == _puzzle_key(generate_puzzle(19))
    for depth in (1, 2, 3):
        for source_count in (1, 2, 3, 4):
            pair = _fallback_pair(np.random.default_rng(depth*10+source_count),
                                  depth, source_count, depth % 4)
            assert np.array_equal(solve(pair["input"]), pair["output"])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(71+nt+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
            validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal example counts were accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "grid_shapes": len(shapes), "structures": len(structures),
            "palettes": len(palettes), "concave_counts": sorted(concaves),
            "source_counts": sorted(counts), "orientations": sorted(turns),
            "pairs_with_merged_routes": merged, "pairs_with_extra_chambers": extra_chambers,
            "untouched_concave_bends": untouched_concave, "adjacent_bend_objects": adjacent_bends}
