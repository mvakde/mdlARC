"""Ray-written contour keys. No parent imports, files, or generation state in solve."""
import hashlib
import json

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 106625253, 392952803, 133933960, 142857320, 917297006,
              661257683, 778112156, 544577654, 926960468, 187142595,
              873489959, 886261862, 292376144, 363326596, 677723948,
              42820546, 861081949, 704578377, 171582816, 318280564,
              936802469, 13388591, 329136278, 802661798, 978427539,
              17368095, 676023795, 247089466, 391742429, 853739777,
              210390453, 612260171, 605070506, 241986870, 312529163,
              610940102, 879448235, 48347744, 328350572, 159755279, 88139068)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _inside(mask, p):
    r, c = p
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _components(mask):
    # Python integer offsets avoid NumPy scalar arithmetic at every BFS edge.
    # Scan and neighbor order match the original row-major/D4 traversal.
    h, w = mask.shape
    remaining = mask.ravel().tolist()
    result = []
    for start in range(h * w):
        if not remaining[start]:
            continue
        queue = [start]
        remaining[start] = False
        for pos in queue:
            r, c = divmod(pos, w)
            for q, valid in ((pos - w, r > 0), (pos + w, r + 1 < h),
                             (pos - 1, c > 0), (pos + 1, c + 1 < w)):
                if valid and remaining[q]:
                    remaining[q] = False
                    queue.append(q)
        component = np.zeros(mask.shape, dtype=bool)
        component.ravel()[queue] = True
        result.append(component)
    return result


def _depth(mask, diagonal=True):
    """Iterative erosion: markers belong to the chamber, never form holes."""
    current = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int8)
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    level = 0
    while np.any(current):
        padded[1:-1, 1:-1] = current
        inner = current.copy()
        for dr, dc in D8 if diagonal else D4:
            inner &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~inner] = level
        current = inner
        level += 1
    return depth


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return ((r, c), (r - dr, c), (r, c - dc))


def _parse(grid):
    wall = int(grid[0, 0])
    assert (np.all(grid[0] == wall) and np.all(grid[-1] == wall)
            and np.all(grid[:, 0] == wall) and np.all(grid[:, -1] == wall)), "closed wall frame"
    open_mask = grid != wall
    assert np.any(open_mask)
    colors, counts = np.unique(grid[open_mask], return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert counts.max() > np.count_nonzero(open_mask) / 2, "unique majority chamber background"
    regions = []
    for mask in _components(open_mask):
        sources, mirrors = [], []
        for color in np.unique(grid[mask]):
            if color == bg:
                continue
            for obj in _components(mask & (grid == color)):
                cells = tuple(map(tuple, np.argwhere(obj)))
                rr, cc = np.nonzero(obj)
                r0, r1, c0, c1 = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
                if len(cells) == 3 and r1 - r0 == c1 - c0 == 1:
                    missing = next((r, c) for r in (r0, r1) for c in (c0, c1) if not obj[r, c])
                    elbow = (r0 + r1 - missing[0], c0 + c1 - missing[1])
                    sources.append({"cells": cells, "color": int(color), "elbow": elbow,
                                    "direction": (elbow[0] - missing[0], elbow[1] - missing[1]),
                                    "missing": missing})
                else:
                    assert len(cells) % 2 == 1 and (r0 == r1 or c0 == c1), "only Ls and odd bars"
                    assert len(cells) == max(r1 - r0, c1 - c0) + 1
                    mirrors.append({"cells": cells, "color": int(color),
                                    "mid": ((r0 + r1) // 2, (c0 + c1) // 2),
                                    "axis": None if len(cells) == 1 else (0 if r0 == r1 else 1)})
        assert len(sources) <= 1, "one writer per chamber"
        assert sources or not mirrors, "unmarked chambers have no mirrors"
        regions.append({"mask": mask, "depth": _depth(mask),
                        "source": sources[0] if sources else None, "mirrors": mirrors})
    assert any(r["source"] is not None for r in regions), "at least one writer"
    return bg, wall, regions


def _contact(mirror, p, direction):
    for axis in (0, 1):
        if mirror["axis"] not in (axis, None):
            continue
        other = 1 - axis
        if p[other] == mirror["mid"][other] and mirror["mid"][axis] - p[axis] == direction[axis]:
            return axis
    return None


def _trace(region, *, turn=True, recolor=True, strict=True, last=False):
    source = region["source"]
    mask, depth = region["mask"], region["depth"]
    assert depth[source["missing"]] == 0 and depth[source["elbow"]] == 1, "inward corner writer"
    assert all(depth[p] == (1 if p == source["elbow"] else 0) for p in source["cells"])
    direction, color = source["direction"], source["color"]
    mr, mc = source["missing"]
    assert not _inside(mask, (mr - direction[0], mc)) and not _inside(mask, (mr, mc - direction[1]))
    p = source["elbow"]
    samples = {0: color, 1: color}
    occupied = {p for obj in [source] + region["mirrors"] for p in obj["cells"]}
    path, contacts, seen = [], [], set()
    while True:
        p = (p[0] + direction[0], p[1] + direction[1])
        if not _inside(mask, p):
            break
        state = (p, direction)
        assert state not in seen, "nonterminating writer"
        seen.add(state)
        if strict:
            assert p not in occupied, "ray must reach faces, not objects"
        hits = [(i, _contact(m, p, direction)) for i, m in enumerate(region["mirrors"])]
        hits = [(i, axis) for i, axis in hits if axis is not None]
        assert len(hits) <= 1, "ambiguous contact"
        if hits:
            i, axis = hits[0]
            if recolor:
                color = region["mirrors"][i]["color"]
            contacts.append((i, axis, p, color))
            if turn:
                direction = tuple(-v if j == axis else v for j, v in enumerate(direction))
        d = int(depth[p])
        if last or d not in samples:
            samples[d] = color
        path.append((p, color))
    assert set(samples) == set(range(max(samples) + 1)), "depths are visited without gaps"
    if strict:
        assert {i for i, _, _, _ in contacts} == set(range(len(region["mirrors"]))), "unused mirror"
    return {"period": tuple(samples[i] for i in range(len(samples))), "path": path, "contacts": contacts}


def _paint_region(out, grid, bg, region, route, *, cycle=True, depths=None):
    depth = region["depth"] if depths is None else depths
    writable = region["mask"] & (grid == bg)
    period = np.asarray(route["period"], dtype=np.int8)
    if not cycle:
        writable &= depth < len(period)
    out[writable] = period[depth[writable] % len(period)]


def _evaluate(input_grid):
    grid = _grid(input_grid)
    bg, wall, regions = _parse(grid)
    out = grid.copy()
    for region in regions:
        if region["source"]:
            region["route"] = _trace(region)
            _paint_region(out, grid, bg, region, region["route"])
    return out, {"grid": grid, "bg": bg, "wall": wall, "regions": regions}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode walls, objects, routes and keys entirely from the supplied grid."""
    return _evaluate(input_grid)[0]


def _alternative(info, mode):
    grid, bg = info["grid"], info["bg"]
    out = grid.copy()
    for region in info["regions"]:
        if not region["source"]:
            continue
        route = region["route"]
        if mode == "thin":
            for p, color in route["path"]:
                out[p] = color
            continue
        depths = None
        if mode in ("no_turn", "no_recolor", "last"):
            route = _trace(region, turn=mode != "no_turn", recolor=mode != "no_recolor",
                           strict=mode != "no_turn", last=mode == "last")
        if mode in ("box", "manhattan"):
            if mode == "manhattan":
                depths = _depth(region["mask"], diagonal=False)
            else:
                rr, cc = np.nonzero(region["mask"])
                rows, cols = np.indices(grid.shape)
                depths = np.minimum.reduce((rows - rr.min(), rr.max() - rows,
                                            cols - cc.min(), cc.max() - cols))
        _paint_region(out, grid, bg, region, route, cycle=mode != "no_cycle", depths=depths)
    return out


def _corners(region):
    mask, depth = region["mask"], region["depth"]
    choices = []
    for r, c in map(tuple, np.argwhere(depth == 0)):
        for dr in (-1, 1):
            for dc in (-1, 1):
                elbow = (r + dr, c + dc)
                if (_inside(mask, elbow) and depth[elbow] == 1
                        and not _inside(mask, (r - dr, c))
                        and not _inside(mask, (r, c - dc))
                        and all(_inside(mask, p) and depth[p] in (0, 1)
                                for p in _source_cells(elbow, (dr, dc)))):
                    choices.append((elbow, (dr, dc)))
    return choices


def _place(grid, cells, color, bg, mask):
    for r, c in cells:
        if not _inside(mask, (r, c)) or grid[r, c] != bg:
            return None
        for dr, dc in D4:
            q = (r + dr, c + dc)
            if q not in cells and _inside(mask, q) and grid[q] == color:
                return None
    trial = grid.copy()
    for p in cells:
        trial[p] = color
    return trial


def _mirror_cells(mid, axis, length):
    radius = length // 2
    return tuple((mid[0], mid[1] + k) if axis == 0 else (mid[0] + k, mid[1])
                 for k in range(-radius, radius + 1))


def _extend(grid, mask, rng, palette, first_axis=None, first_bar=False, shallow=False, point=False):
    _, info = _evaluate(grid)
    region = next(r for r in info["regions"] if r["source"] and np.array_equal(r["mask"], mask))
    route = region["route"]
    n = len(route["contacts"])
    tail_start = next((k + 1 for k, (p, _) in reversed(list(enumerate(route["path"])))
                       if route["contacts"] and p == route["contacts"][-1][2]), 0)
    tail = route["path"][tail_start:]
    direction = region["source"]["direction"]
    for _, axis, _, _ in route["contacts"]:
        direction = tuple(-v if j == axis else v for j, v in enumerate(direction))
    axes = [first_axis] if first_axis is not None and n == 0 else [0, 1]
    lengths = [1] if point else ([3, 5] if first_bar and n == 0 else [1, 3, 5])
    options = [(p, axis, length) for p, _ in tail for axis in axes
               for length in lengths
               if not shallow or 2 <= region["depth"][p] <= (region["depth"].max() - 1) // 2]
    rng.shuffle(options)
    for p, axis, length in options[:120]:
        mid = tuple(p[j] + (direction[j] if j == axis else 0) for j in (0, 1))
        shade = int(rng.choice(palette))
        if n == 0:
            shade = int(rng.choice([c for c in palette if c != region["source"]["color"]]))
        cells = _mirror_cells(mid, axis, length)
        trial = _place(grid, cells, shade, info["bg"], mask)
        if trial is None:
            continue
        try:
            _, after = _evaluate(trial)
        except AssertionError:
            continue
        new = next(r for r in after["regions"] if r["source"] and np.array_equal(r["mask"], mask))
        old_events = [(region["mirrors"][i]["mid"], a, q) for i, a, q, _ in route["contacts"]]
        new_events = [(new["mirrors"][i]["mid"], a, q) for i, a, q, _ in new["route"]["contacts"]]
        if new_events != old_events + [(mid, axis, p)] or len(set(new["route"]["period"])) < 2:
            continue
        before_bands, after_bands = grid.copy(), grid.copy()
        _paint_region(before_bands, grid, info["bg"], region, route)
        _paint_region(after_bands, grid, info["bg"], new, new["route"])
        common = mask & (grid == info["bg"]) & (trial == info["bg"])
        if np.any(before_bands[common] != after_bands[common]):
            return trial
    return None


def _walls(rng, shape, target, sculpt):
    h, w = shape
    walls = np.zeros(shape, dtype=bool)
    walls[[0, -1], :] = True
    walls[:, [0, -1]] = True
    rooms = [(1, h - 2, 1, w - 2)]
    while len(rooms) < target:
        options = [(i, axis) for i, (a, b, c, d) in enumerate(rooms)
                   for axis, size in ((0, b - a + 1), (1, d - c + 1)) if size >= 15]
        if not options:
            break
        i, axis = options[int(rng.integers(len(options)))]
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(a + 6, b - 5))
            walls[cut, c - 1:d + 2] = True
            rooms.extend(((a, cut - 1, c, d), (cut + 1, b, c, d)))
        else:
            cut = int(rng.integers(c + 6, d - 5))
            walls[a - 1:b + 2, cut] = True
            rooms.extend(((a, b, c, cut - 1), (a, b, cut + 1, d)))
    # U-shaped walls enclose a side pocket and create a concavity in its host.
    for _ in range(sculpt):
        a, b, c, d = rooms[int(rng.integers(len(rooms)))]
        transposed = bool(rng.integers(2))
        work = walls.T if transposed else walls
        if transposed:
            a, b, c, d = c, d, a, b
        if b - a < 9 or d - c < 8:
            continue
        top = int(rng.integers(a + 2, b - 4))
        bottom = int(rng.integers(top + 4, min(b - 1, top + 9) + 1))
        side = int(rng.choice([-1, 1]))
        edge = c - 1 if side == 1 else d + 1
        far = edge + side * int(rng.integers(3, min(6, d - c - 3)))
        lo, hi = sorted((edge, far))
        block = work[top - 1:bottom + 2, lo:hi + 1]
        if np.any(block[:, 1:] if side == 1 else block[:, :-1]):
            continue
        work[top, lo:hi + 1] = True
        work[bottom, lo:hi + 1] = True
        work[top:bottom + 1, far] = True
    # Shift a straight divider interval; joins stay orthogonal and one cell wide.
    for _ in range(int(rng.integers(0, 4))):
        work = walls.T if rng.integers(2) else walls
        for _attempt in range(20):
            r, c = map(int, np.argwhere(work)[int(rng.integers(np.count_nonzero(work)))])
            end = r + int(rng.integers(4, 9))
            shift = int(rng.choice([-2, -1, 1, 2]))
            lo, hi = sorted((c, c + shift))
            if r < 2 or end >= work.shape[0] - 2 or lo < 2 or hi >= work.shape[1] - 2:
                continue
            block = work[r - 1:end + 2, lo - 1:hi + 2]
            expected = np.zeros_like(block)
            expected[:, c - lo + 1] = True
            if not np.array_equal(block, expected):
                continue
            work[r + 1:end, c] = False
            work[r, lo:hi + 1] = work[end, lo:hi + 1] = True
            work[r:end + 1, c + shift] = True
            break
    return walls


def _witnesses(info, out):
    return {m: not np.array_equal(out, _alternative(info, m))
            for m in ("thin", "no_turn", "no_recolor", "no_cycle", "last", "box", "manhattan")}


def _essential_mirrors(info, out):
    """Deleting a mirror must change writable bands, not just delete its ink."""
    for region in info["regions"]:
        if not region["source"]:
            continue
        for i in range(len(region["mirrors"])):
            hypothetical = dict(region, mirrors=region["mirrors"][:i] + region["mirrors"][i + 1:])
            try:
                route = _trace(hypothetical, strict=False)
            except AssertionError:
                return False
            alternative = out.copy()
            _paint_region(alternative, info["grid"], info["bg"], region, route)
            if np.array_equal(alternative, out):
                return False
    return True


def _make_pair(rng, role_colors, demo, test, first_axis, seen):
    bg, wall = role_colors
    palette = [c for c in range(10) if c not in (bg, wall)]
    for attempt in range(100):
        low = 22 if demo is not None else (18 if test else 13)
        shape = tuple(map(int, rng.integers(low, 31, size=2)))
        target = (1 if demo == 0 else 2) if demo is not None else int(rng.integers(1, 5 if test else 4))
        sculpt = (0 if demo == 0 else 1) if demo is not None else int(rng.integers(0, 3))
        walls = _walls(rng, shape, target, sculpt)
        if np.any(walls[:-1, :-1] & walls[1:, :-1] & walls[:-1, 1:] & walls[1:, 1:]):
            continue
        grid = np.full(shape, bg, dtype=np.int8)
        grid[walls] = wall
        regions = [{"mask": m, "depth": _depth(m)} for m in _components(~walls)]
        eligible = [r for r in regions if r["depth"].max() >= 3 and _corners(r)]
        if not eligible:
            continue
        rng.shuffle(eligible)
        if test:
            # Three-contact test routes need space; random order breaks depth ties.
            eligible.sort(key=lambda r: int(r["depth"].max()), reverse=True)
        active = 1 if demo == 0 else int(rng.integers(1, min(len(eligible), 3 if test else 2) + 1))
        if demo == 1:
            active = min(active, len(regions) - 1)
        good = True
        for j, region in enumerate(eligible[:active]):
            corners = _corners(region)
            elbow, direction = corners[int(rng.integers(len(corners)))]
            color = int(rng.choice(palette))
            for p in _source_cells(elbow, direction):
                grid[p] = color
            if test and j == 0:
                desired = 3
            elif demo is not None and j == 0:
                desired = 1 if demo == 0 else 2
            else:
                desired = int(rng.integers(1, 4 if test else 3))
            for k in range(desired):
                trial = _extend(grid, region["mask"], rng, palette,
                                first_axis=first_axis if demo is not None and j == 0 else None,
                                first_bar=demo is not None and j == 0,
                                shallow=demo == 0 and k == 0,
                                point=demo == 1 and j == 0 and k == 1)
                if trial is None:
                    good = False
                    break
                grid = trial
            if not good:
                break
        if not good or _key(grid) in seen:
            continue
        try:
            out, info = _evaluate(grid)
            witnesses = _witnesses(info, out)
            if not _essential_mirrors(info, out):
                continue
            if not all(witnesses[m] for m in ("thin", "no_turn", "no_recolor", "last")):
                continue
            if demo == 0 and not witnesses["no_cycle"]:
                continue
            if demo == 0 and not any(r["source"] and r["depth"].max() >= 2 * len(r["route"]["period"]) - 1
                                     for r in info["regions"]):
                continue
            if demo == 1 and not (witnesses["box"] and witnesses["manhattan"]):
                continue
        except AssertionError:
            continue
        return {"input": grid, "output": out}
    raise RuntimeError(f"bounded scene construction exhausted: demo={demo}, test={test}")


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _counts(num_train, num_test):
    for v, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError("illegal training/test count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    backgrounds = list(map(int, rng.permutation(10)))
    first_axis = int(rng.integers(2))
    for i in range(num_train + num_test):
        split = "train" if i < num_train else "test"
        bg = backgrounds[i % 10]
        wall = int(rng.choice([c for c in range(10) if c != bg]))
        demo = i if i < 2 else None
        pair = _make_pair(rng, (bg, wall), demo, split == "test",
                          first_axis if i == 0 else 1 - first_axis, seen)
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, axes = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            predicted, info = _evaluate(inp)
            assert np.array_equal(out, predicted), "wrong contour memory"
            assert not np.array_equal(inp, out)
            assert np.array_equal(inp[inp != info["bg"]], out[inp != info["bg"]]), "markers changed"
            assert set(np.unique(out)) <= set(np.unique(inp))
            assert _essential_mirrors(info, out), "a mirror has no effect on the bands"
            for r in info["regions"]:
                if r["source"]:
                    assert 1 <= len(r["route"]["contacts"]) <= (2 if split == "train" else 3)
                    assert len(set(r["route"]["period"])) >= 2
            walls = inp == info["wall"]
            assert len(_components(walls)) == 1
            assert not np.any(walls[:-1, :-1] & walls[1:, :-1] & walls[:-1, 1:] & walls[1:, 1:])
            witness = _witnesses(info, out)
            assert witness["thin"] and witness["no_turn"] and witness["no_recolor"] and witness["last"]
            if split == "train":
                training.append((info, witness))
                for r in info["regions"]:
                    if r["source"]:
                        axes.update(axis for i, axis, _, _ in r["route"]["contacts"] if len(r["mirrors"][i]["cells"]) > 1)
            else:
                assert any(r["source"] and len(r["route"]["contacts"]) == 3 for r in info["regions"]), "three-reflection test shift"
    assert axes == {0, 1}, "training must establish both mirror faces"
    assert any(r["source"] is None for a, _ in training for r in a["regions"]), "unmarked chamber demonstration"
    assert any(len(m["cells"]) == 1 for a, _ in training for r in a["regions"] for m in r["mirrors"]), "point mirror demonstration"
    assert any(w["no_cycle"] for _, w in training), "training must show actual key repetition"
    assert any(r["source"] and r["depth"].max() >= 2 * len(r["route"]["period"]) - 1
               for a, _ in training for r in a["regions"]), "two complete key cycles"
    assert any(w["box"] and w["manhattan"] for _, w in training), "colored concavity teaching witness"
    assert any(len(r["route"]["contacts"]) >= 2 for a, _ in training for r in a["regions"] if r["source"])


def self_test():
    """Bounded 42-seed audit; stores digests and aggregate features, not puzzles."""
    shapes, geometries, palettes, directions, periods, bounces, counts, axes = (set() for _ in range(8))
    signatures = set()
    split_bounces = {s: set() for s in ("train", "test")}
    split_writers = {s: set() for s in ("train", "test")}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp), out)
                _, a = _evaluate(inp)
                shapes.add(inp.shape)
                geometries.add(hashlib.sha256(bytes(inp.shape) + (inp == a["wall"]).tobytes()).hexdigest())
                palettes.add((a["bg"], a["wall"]))
                counts.add((len(a["regions"]), sum(r["source"] is not None for r in a["regions"])))
                split_writers[split].add(sum(r["source"] is not None for r in a["regions"]))
                for region in a["regions"]:
                    if region["source"]:
                        directions.add(region["source"]["direction"])
                        periods.add(len(region["route"]["period"]))
                        bounces.add(len(region["route"]["contacts"]))
                        split_bounces[split].add(len(region["route"]["contacts"]))
                        axes.update(axis for _, axis, _, _ in region["route"]["contacts"])
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 60 and len(geometries) >= 100
    assert len(palettes) >= 40 and len(counts) >= 5 and len(periods) >= 4
    assert directions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert axes == {0, 1} and {1, 2, 3} <= bounces
    a, b = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    pair = bad["test"][0]
    p = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][p] = pair["input"][p]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing band pixel accepted")
    _direct_regressions()
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "wall_geometries": len(geometries),
            "role_assignments": len(palettes), "region_writer_counts": sorted(counts),
            "period_lengths": sorted(periods), "reflection_counts": sorted(bounces),
            "reflection_counts_by_split": {s: sorted(v) for s, v in split_bounces.items()},
            "writer_counts_by_split": {s: sorted(v) for s, v in split_writers.items()}}


def _direct_regressions():
    """Small explicit expectations for contact timing and first-visit memory."""
    grid = np.zeros((18, 20), dtype=np.int8)
    grid[[0, -1], :] = 9
    grid[:, [0, -1]] = 9
    grid[1, 2] = grid[2, 1] = grid[2, 2] = 1
    grid[5, 3:6] = 2
    rows, cols = np.indices(grid.shape)
    depth = np.minimum.reduce((rows - 1, 16 - rows, cols - 1, 18 - cols))
    for chain, period in ((False, (1, 1, 1, 2)), (True, (1, 1, 1, 2, 3, 3, 3))):
        inp = grid.copy()
        if chain:
            inp[1, 5:8] = 3
        expected = inp.copy()
        mask = inp == 0
        expected[mask] = np.asarray(period)[depth[mask] % len(period)]
        for rotation in range(4):
            for flip in (False, True):
                x, y = np.rot90(inp, rotation), np.rot90(expected, rotation)
                if flip:
                    x, y = x[:, ::-1], y[:, ::-1]
                assert np.array_equal(solve(x), y), "explicit optical-key regression"


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
