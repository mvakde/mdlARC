"""Hidden inlay: complete occluded loops, then exact-cover their odd interiors.

All solving is input-derived. No parent module, file, seed or scene descriptor is
consulted by solve. Generation and tests retain no execution traces or payloads.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib

import numpy as np


HOST_SEEDS = (0,619793892,317391403,784582767,130211384,151555287,1763484,
              896119759,541849839,611721291,979817485,922984330,424608046,
              352424002,705917865,21450114,775559348,712299043,39596425,
              538829344,872730263,946034946,435069647,692189036,534721802,
              931845809,918452905,296112599,386415053,824537601,980148850,
              271545685,136982283,987358620,209582610,534851717,61521939,
              760436343,725293217,199087787,705818001,236859736)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _rim(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)} |
            {(b, x) for x in range(l, r + 1)} |
            {(y, l) for y in range(t + 1, b)} |
            {(y, r) for y in range(t + 1, b)})


def _inside(box):
    t, b, l, r = box
    return _area((t + 1, b - 1, l + 1, r - 1))


def _neighbors(p):
    y, x = p
    return ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))


def _components(points):
    remaining, groups = set(points), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, group = [first], {first}
        while stack:
            for q in _neighbors(stack.pop()):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    group.add(q)
        groups.append(group)
    return groups


def _halo(points):
    return {q for y, x in points for q in
            ((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))}


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _board(main, boxes):
    """The crop's odd-depth cells, excluding every complete rectangular rim."""
    crop = _inside(main)
    depth = Counter(p for box in boxes for p in _inside(box) & crop)
    boundaries = set().union(*(_rim(box) for box in boxes))
    return {p for p, d in depth.items() if d % 2 and p not in boundaries}


def _find_frame(grid, bg):
    candidates = []
    for color in sorted(set(map(int, grid.flat)) - {bg}):
        for group in _components(map(tuple, np.argwhere(grid == color))):
            box = _bbox(group)
            t, b, l, r = box
            if b - t < 3 or r - l < 3 or group != _rim(box):
                continue
            if not (0 < t < b < grid.shape[0] - 1 and 0 < l < r < grid.shape[1] - 1):
                continue
            interior = grid[t + 1:b, l + 1:r]
            fill = int(interior[0, 0])
            if fill != color and np.all(interior == fill):
                candidates.append(((b - t - 1) * (r - l - 1), box, color, fill))
    candidates.sort(reverse=True)
    assert candidates and (len(candidates) == 1 or candidates[0][0] > candidates[1][0]), "frame ambiguity"
    _, main, frame, fill = candidates[0]
    return main, frame, fill


def _read_input(value):
    grid = _grid(value)
    bg = int(grid[0, 0])
    assert all(int(grid[p]) == bg for p in
               ((0, -1), (-1, 0), (-1, -1))), "background corner witnesses"
    main, frame, fill = _find_frame(grid, bg)
    masked, crop = _area(main), _inside(main)
    outside = set(map(tuple, np.argwhere(grid != bg))) - masked
    parts = _components(outside)
    candidates = []
    for size in (1, 2):
        for ids in combinations(range(len(parts)), size):
            visible = set().union(*(parts[i] for i in ids))
            box = _bbox(visible)
            t, b, l, r = box
            if min(b - t, r - l) < 3:
                continue
            full = _rim(box)
            hidden = full & crop
            if not hidden or full - masked != visible:
                continue
            phase = {}
            for p in sorted(visible):
                parity, color = sum(p) % 2, int(grid[p])
                if parity in phase and phase[parity] != color:
                    break
                phase[parity] = color
            else:
                if set(phase) == {0, 1}:
                    candidates.append((frozenset(ids), box, phase))
    # Geometric loop grouping must be unambiguous before color/tile placement.
    used = set()
    loops = []
    for ids, box, phase in candidates:
        assert not (used & ids), "multiple rectangular explanations"
        used.update(ids)
        loops.append((box, phase))
    assert 1 <= len(loops) <= 3, "one to three recoverable loops required"
    complete_rims = set()
    for box, phase in loops:
        rim = _rim(box)
        assert not (rim & _halo(complete_rims)), "rims need a blank gap"
        complete_rims.update(rim)
        assert not ({bg, frame, fill} & set(phase.values()))
    boxes = [box for box, _ in loops]
    assert max(sum(p in _inside(box) for box in boxes) for p in crop) <= 2
    pieces = []
    loop_colors = {c for _, phase in loops for c in phase.values()}
    for i, group in enumerate(parts):
        if i in used:
            continue
        colors = {int(grid[p]) for p in group}
        assert len(colors) == 1, "loose piece must be monochrome"
        color = next(iter(colors))
        assert color not in {bg, frame, fill} | loop_colors
        cells = np.array(sorted(group), dtype=int)
        pieces.append((cells, color))
    assert 2 <= len(pieces) <= 6
    target = _board(main, boxes)
    assert target and sum(len(cells) for cells, _ in pieces) == len(target), "tile material must equal hidden board"
    t, b, l, r = main
    base = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    for box, phase in loops:
        for y, x in _rim(box) & crop:
            base[y - t - 1, x - l - 1] = phase[(y + x) % 2]
    return {"grid": grid, "background": bg, "frame": frame, "fill": fill,
            "main": main, "loops": loops, "pieces": pieces, "target": target,
            "base": base}


def _pack(scene):
    """Enumerate ALL translations, then use-once exact cover; stop at 2 colors.

    No restriction to the construction partition, full-width layers, sockets or
    piece order is made. Identical same-color objects are interchangeable.
    """
    target = sorted(scene["target"])
    index = {p: i for i, p in enumerate(target)}
    pieces = scene["pieces"]
    options, by_piece = [], [[] for _ in pieces]
    by_cell = [[] for _ in target]
    for i, (cells, color) in enumerate(pieces):
        shape = cells - cells.min(axis=0)
        # Anchoring one occupied pixel at each target cell covers every shift.
        for anchor in target:
            dest = shape + np.array(anchor) - shape[0]
            coords = tuple(map(tuple, dest))
            if not all(p in index for p in coords):
                continue
            bits = sum(1 << index[p] for p in coords)
            j = len(options)
            options.append((i, bits, coords))
            by_piece[i].append(j)
            for p in coords:
                by_cell[index[p]].append(j)
    full = (1 << len(target)) - 1
    all_pieces = (1 << len(pieces)) - 1
    solutions = {}
    nodes = 0

    def visit(occupied, used, chosen):
        nonlocal nodes
        nodes += 1
        if len(solutions) >= 2:
            return
        if occupied == full:
            if used != all_pieces:
                return
            out = scene["base"].copy()
            t, _, l, _ = scene["main"]
            for j in chosen:
                i, _, coords = options[j]
                for y, x in coords:
                    out[y - t - 1, x - l - 1] = pieces[i][1]
            solutions.setdefault(out.tobytes(), (out, tuple(options[j] for j in chosen)))
            return

        def available(j):
            i, bits, _ = options[j]
            return not (used & (1 << i) or bits & occupied)

        choices = None
        # Both cell and object columns are constraints in the exact-cover matrix.
        for k in range(len(target)):
            if not (occupied >> k) & 1:
                valid = [j for j in by_cell[k] if available(j)]
                if not valid:
                    return
                if choices is None or len(valid) < len(choices):
                    choices = valid
                    if len(choices) == 1:
                        break
        for i in range(len(pieces)):
            if not (used >> i) & 1:
                valid = [j for j in by_piece[i] if available(j)]
                if not valid:
                    return
                if len(valid) < len(choices):
                    choices = valid
        # Do not eliminate identical IDs here: a selected cell may be covered
        # by a later ID while an earlier identical one must occupy another site.
        for j in choices:
            i, bits, _ = options[j]
            visit(occupied | bits, used | (1 << i), chosen + [j])

    visit(0, 0, [])
    return list(solutions.values()), {"candidates": len(options), "nodes": nodes,
                                     "piece_options": tuple(map(len, by_piece))}


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _read_input(input_grid)
    solutions, _ = _pack(scene)
    assert len(solutions) == 1, "hidden board must have one distinct colored assembly"
    return solutions[0][0].copy()


def _geometry(rng, kind):
    h, w = int(rng.integers(23, 31)), int(rng.integers(23, 31))
    t = int(rng.integers(7, min(11, h - 16) + 1))
    l = int(rng.integers(7, min(11, w - 16) + 1))
    if kind in ("disjoint", "mixed"):
        b = min(h - 3, t + int(rng.integers(13, 17)))
        r = min(w - 3, l + int(rng.integers(13, 17)))
    else:
        b = min(h - 3, t + int(rng.integers(10, 15)))
        r = min(w - 3, l + int(rng.integers(10, 15)))
    main = (t, b, l, r)
    if kind == "through":
        boxes = [(t + int(rng.integers(2, 4)), b - int(rng.integers(2, 4)),
                  int(rng.integers(1, 4)), w - int(rng.integers(2, 5)))]
    else:
        depth = int(rng.integers(5, 9)) if kind == "single" else 7
        outer = (int(rng.integers(1, 4)), t + depth,
                 int(rng.integers(1, 4)), l + depth + int(rng.integers(0, 2)))
        if kind == "disjoint":
            outer = (outer[0], t + 5, outer[2], l + int(rng.integers(4, 6)))
        boxes = [outer]
        if kind in ("nested", "mixed"):
            a, z, c, d = outer
            boxes.append((a + int(rng.integers(2, 4)), z - int(rng.integers(2, 4)),
                          c + int(rng.integers(2, 4)), d - int(rng.integers(2, 4))))
        if kind in ("disjoint", "mixed"):
            start = 8 if kind == "disjoint" else 10
            boxes.append((t + start, h - 2, l + start + int(rng.integers(0, 2)), w - 2))
    return (h, w), main, boxes


def _partition(rng, target, count):
    """Multi-source growth makes irregular connected complementary polyominoes."""
    components = _components(target)
    if len(components) > count or len(target) < count * 4:
        return None
    seeds = [sorted(group)[int(rng.integers(len(group)))] for group in components]
    pool = sorted(target - set(seeds))
    for j in rng.choice(len(pool), count - len(seeds), replace=False):
        seeds.append(pool[int(j)])
    groups = [{p} for p in seeds]
    labels = {p: i for i, p in enumerate(seeds)}
    frontier = [(i, q) for i, p in enumerate(seeds) for q in _neighbors(p)
                if q in target and q not in labels]
    while frontier:
        k = int(rng.integers(len(frontier)))
        i, p = frontier[k]
        frontier[k] = frontier[-1]
        frontier.pop()
        if p in labels:
            continue
        labels[p] = i
        groups[i].add(p)
        frontier.extend((i, q) for q in _neighbors(p) if q in target and q not in labels)
    if set(labels) != target or any(len(group) < 4 for group in groups):
        return None
    # Do not teach a collection of unrelated bars or rectangular blocks.
    if not any(len(group) < len(_area(_bbox(group))) for group in groups):
        return None
    if any(min(_bbox(group)[1] - _bbox(group)[0],
               _bbox(group)[3] - _bbox(group)[2]) >= 3 and
           group == _rim(_bbox(group)) for group in groups):
        return None
    return groups


def _scatter(rng, base, groups, colors, bg):
    h, w = base.shape
    result = base.copy()
    forbidden = _halo(map(tuple, np.argwhere(base != bg)))
    # The entire window is opaque even when its fill matches the background.
    main, _, _ = _find_frame(base, bg)
    forbidden.update(_halo(_area(main)))
    order = sorted(range(len(groups)), key=lambda i: -len(groups[i]))
    for i in order:
        cells = np.array(sorted(groups[i]), dtype=int)
        shape = cells - cells.min(axis=0)
        hh, ww = shape.max(axis=0) + 1
        choices = []
        for y in range(1, h - int(hh)):
            for x in range(1, w - int(ww)):
                coords = shape + (y, x)
                if not any(tuple(p) in forbidden for p in coords):
                    choices.append((y, x))
        if not choices:
            return None
        coords = shape + choices[int(rng.integers(len(choices)))]
        result[coords[:, 0], coords[:, 1]] = colors[i]
        forbidden.update(_halo(map(tuple, coords)))
    return result


def _make_pair(rng, kind, style, palette, fill_matches):
    shape, main, boxes = _geometry(rng, kind)
    bg, frame, distinct_fill = palette[:3]
    fill = bg if fill_matches else distinct_fill
    loop_palette, tile_palette = palette[3:5], palette[5:]
    target = _board(main, boxes)
    minimum = 2 if kind in ("single", "through") else 3
    maximum = min(6, len(target) // 4)
    if maximum < minimum:
        return None
    count = int(rng.integers(minimum, maximum + 1))
    # Fewer pieces make the first demonstration easier to inspect.
    if style == 0:
        count = 2 if kind == "single" else 3
    for _ in range(60):
        groups = _partition(rng, target, count)
        if groups is not None:
            break
    else:
        return None
    tile_colors = [tile_palette[i] for i in range(min(count, 2))]
    tile_colors += [int(rng.choice(tile_palette[:min(count, 4)])) for _ in range(count - 2)]
    under = np.full(shape, bg, dtype=np.int8)
    for j, box in enumerate(boxes):
        alternating = style != 0 and (j == 0 or bool(rng.integers(2)))
        first = int(rng.choice(loop_palette))
        second = next(c for c in loop_palette if c != first) if alternating else first
        phase = (first, second) if bool(rng.integers(2)) else (second, first)
        for p in _rim(box):
            under[p] = phase[sum(p) % 2]
    t, b, l, r = main
    output = under[t + 1:b, l + 1:r].copy()
    output[output == bg] = fill
    for group, color in zip(groups, tile_colors):
        for y, x in group:
            output[y - t - 1, x - l - 1] = color
    base = under.copy()
    base[t + 1:b, l + 1:r] = fill
    for p in _rim(main):
        base[p] = frame
    inp = _scatter(rng, base, groups, tile_colors, bg)
    if inp is None:
        return None
    turns, flip = int(rng.integers(4)), bool(rng.integers(2))
    inp, output = np.rot90(inp, turns), np.rot90(output, turns)
    if flip:
        inp, output = inp[:, ::-1], output[:, ::-1]
    inp, output = inp.copy(), output.copy()
    try:
        scene = _read_input(inp)
        solutions, _ = _pack(scene)
        if len(solutions) != 1 or not np.array_equal(solutions[0][0], output):
            return None
        if style == 0:
            # A source-position ordering rule must already fail in the easy demo.
            destinations = {i: min(coords) for i, _, coords in solutions[0][1]}
            if sorted(destinations, key=destinations.get) == list(range(len(scene["pieces"]))):
                return None
    except AssertionError:
        return None
    return {"input": inp, "output": output}


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    stable = bool(rng.random() < .25)
    shared = list(map(int, rng.permutation(10)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "test":
                kind = "mixed"
            elif i == 0:
                kind = "single"
            elif i == 1:
                kind = "nested"
            elif i == 2:
                kind = "disjoint"
            else:
                kind = str(rng.choice(("single", "nested", "disjoint", "through")))
            style = 0 if split == "train" and i == 0 else 1
            matches = style == 0 or (i > 1 and bool(rng.integers(2)))
            palette = shared if stable else list(map(int, rng.permutation(10)))
            for _ in range(240):
                pair = _make_pair(rng, kind, style, palette, matches)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError(f"bounded construction exhausted for {kind}")
    return puzzle


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    before = inp.copy()
    scene = _read_input(inp)
    solutions, stats = _pack(scene)
    assert len(solutions) == 1
    predicted, placements = solutions[0]
    assert np.array_equal(predicted, out), "incorrect inlay"
    assert np.array_equal(inp, before), "solver changed input"
    assert inp.shape != out.shape
    assert all(len(cells) >= 4 for cells, _ in scene["pieces"])
    assert len({color for _, color in scene["pieces"]}) >= 2
    assert any(len(cells) < len(_area(_bbox(map(tuple, cells))))
               for cells, _ in scene["pieces"]), "no complementary irregular tile"
    t, b, l, r = scene["main"]
    assert out.shape == (b - t - 1, r - l - 1)
    occupied, used = set(), set()
    for i, _, coords in placements:
        assert i not in used
        used.add(i)
        source, color = scene["pieces"][i]
        dest = np.array(coords, dtype=int)
        assert set(map(tuple, source - source.min(axis=0))) == set(map(tuple, dest - dest.min(axis=0)))
        assert not (occupied & set(coords))
        occupied.update(coords)
        assert np.any(source.min(axis=0) != dest.min(axis=0))
        assert all(out[y - t - 1, x - l - 1] == color for y, x in coords)
    assert used == set(range(len(scene["pieces"]))) and occupied == scene["target"]
    tile_colors = {color for _, color in scene["pieces"]}
    for color in tile_colors:
        assert np.count_nonzero(inp == color) == np.count_nonzero(out == color)
    # Both inherited mechanisms affect every example, not just some seeds.
    assert not np.array_equal(scene["base"], out), "B-only shortcut unexpectedly succeeds"
    assert len(scene["target"]) < out.size, "whole-window packing shortcut not distinguished"
    scene["stats"] = stats
    scene["placements"] = placements
    return scene


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, scenes = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            scene = _validate_pair(pair)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen
            seen.add(key)
            scenes[split].append(scene)
    train = scenes["train"]
    assert len(train[0]["loops"]) == 1 and train[0]["fill"] == train[0]["background"]
    destination_order = sorted(train[0]["placements"], key=lambda p: min(p[2]))
    assert [p[0] for p in destination_order] != list(range(len(train[0]["pieces"]))), "source order shortcut not distinguished"
    assert len(train[1]["loops"]) == 2 and train[1]["fill"] != train[1]["background"]
    assert any(len(set(phase.values())) == 2 for _, phase in train[1]["loops"])
    boxes = [box for box, _ in train[1]["loops"]]
    protected = _inside(boxes[0]) & _inside(boxes[1]) & _inside(train[1]["main"])
    assert protected and not (protected & train[1]["target"]), "nested hole not taught"
    assert all(len(scene["loops"]) == 3 for scene in scenes["test"])


def self_test():
    """Stream 42 seeds; keep only signatures and small descriptive counters."""
    fixture = {"target": {(1, 1), (1, 2), (2, 1), (2, 2)},
               "main": (0, 3, 0, 3), "base": np.zeros((2, 2), dtype=np.int8),
               "pieces": [(np.array([[5, 5], [5, 6]]), 2),
                          (np.array([[7, 1], [7, 2]]), 3)]}
    assert len(_pack(fixture)[0]) == 2  # Two genuinely different colored answers.
    fixture["pieces"][1] = (fixture["pieces"][1][0], 2)
    assert len(_pack(fixture)[0]) == 1  # Identical-color identity permutations.
    fixture["target"] = {(1, x) for x in range(1, 5)}
    fixture["main"], fixture["base"] = (0, 2, 0, 5), np.zeros((1, 4), dtype=np.int8)
    fixture["pieces"][0] = (np.array([[5, 5], [6, 5]]), 2)
    assert not _pack(fixture)[0]  # Rotating this domino would falsely solve it.
    structures, shapes, boards, tile_sizes, counts, backgrounds = set(), set(), set(), set(), set(), set()
    puzzles = set()
    repeated = through = local_choice = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            digest.update(repr(inp.shape).encode() + inp.tobytes() + out.tobytes())
            scene = _read_input(inp)
            roles, normalized = {}, []
            for v in inp.flat:
                normalized.append(roles.setdefault(int(v), len(roles)))
            structures.add(hashlib.sha256(bytes(normalized)).digest())
            shapes.add(inp.shape)
            boards.add(out.shape)
            backgrounds.add(scene["background"])
            tile_sizes.update(len(cells) for cells, _ in scene["pieces"])
            counts.add(len(scene["pieces"]))
            colors = [c for _, c in scene["pieces"]]
            repeated += len(set(colors)) < len(colors)
            through += any(len(_components(_rim(box) - _area(scene["main"]))) == 2
                           for box, _ in scene["loops"])
            _, stats = _pack(scene)
            local_choice += any(n > 1 for n in stats["piece_options"])
        puzzles.add(digest.digest())
    assert len(puzzles) == 42 and len(structures) >= 200
    assert len(shapes) >= 12 and len(boards) >= 10
    assert len(tile_sizes) >= 15 and len(counts) >= 4
    assert len(backgrounds) == 10 and repeated >= 42 and through >= 1 and local_choice >= 42
    first, again = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], again[s])
               for k in ("input", "output"))
    inp, out = first["train"][0]["input"], first["train"][0]["output"]
    for turns in range(4):
        for flip in (False, True):
            a, b = np.rot90(inp, turns), np.rot90(out, turns)
            if flip:
                a, b = a[:, ::-1], b[:, ::-1]
            assert np.array_equal(solve(a), b)
    recolor = np.arange(9, -1, -1)
    assert np.array_equal(solve(recolor[inp]), recolor[out])
    for counts_pair in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts_pair)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
