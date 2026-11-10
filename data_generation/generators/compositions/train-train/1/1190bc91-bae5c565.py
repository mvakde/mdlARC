"""Baffled loom: aligned template growth inside a ray-defined chamber.

Only NumPy and the standard library are used. All solving conventions are
visible geometry; generation metadata never enters solve().
"""
from collections import deque
import json
import numpy as np


_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(grid):
    a = np.asarray(grid)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grids must contain integer ARC colors")
    return a.astype(np.int8, copy=True)


def _components(mask):
    h, w = mask.shape
    labels = np.full(mask.shape, -1, dtype=np.int16)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        label = len(groups)
        labels[r, c] = label
        cells, queue = [], deque([(r, c)])
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for dy, dx in _DIRECTIONS:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and labels[yy, xx] < 0:
                    labels[yy, xx] = label
                    queue.append((yy, xx))
        groups.append(cells)
    return labels, groups


def _scaffold(shape, spine, background, backward_count=2):
    out = np.full(shape, background, dtype=np.int8)
    dr = spine[1][0] - spine[0][0]
    dc = spine[1][1] - spine[0][1]
    for r, c, color in spine:
        out[r, c] = color
    for i, (r, c, color) in enumerate(spine):
        for sign in ((1, -1) if i < backward_count else (1,)):
            for side in (-1, 1):
                dy, dx = sign * dr - side * dc, sign * dc + side * dr
                y, x = r + dy, c + dx
                while 0 <= y < shape[0] and 0 <= x < shape[1]:
                    _require(out[y, x] in (background, color), "conflicting rays")
                    out[y, x] = color
                    y, x = y + dy, x + dx
    return out


def _coord(origin, direction, d, j):
    dr, dc = direction
    return origin[0] + d * dr - j * dc, origin[1] + d * dc + j * dr


def _inside(shape, cell):
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _trace(passable, origin, direction, radius):
    """Advance one row, then spread at most one cell sideways in that row.

    The intermediate forward cell must be open. Consequently every move is
    made of orthogonal edges: a blocked forward cell cannot be jumped over.
    """
    reached = np.zeros(passable.shape, dtype=bool)
    fronts = []
    previous = {0} if passable[origin] else set()
    for d in range(max(passable.shape) + 1):
        if not _inside(passable.shape, _coord(origin, direction, d, 0)):
            break
        def free(j):
            cell = _coord(origin, direction, d, j)
            return _inside(passable.shape, cell) and passable[cell]
        if d:
            projected = {j for j in previous if free(j)}
            current = set(projected)
            for j in projected:
                for k in (j - 1, j + 1):
                    if -radius <= k <= radius and free(k):
                        current.add(k)
            previous = current
        fronts.append(frozenset(previous))
        for j in previous:
            reached[_coord(origin, direction, d, j)] = True
    return reached, fronts


def _ideal(passable, origin, direction, radius):
    """Static clipping alternative, deliberately without propagation history."""
    mask = np.zeros(passable.shape, dtype=bool)
    for d in range(max(passable.shape) + 1):
        for j in range(-min(d, radius), min(d, radius) + 1):
            cell = _coord(origin, direction, d, j)
            if _inside(passable.shape, cell) and passable[cell]:
                mask[cell] = True
    return mask


def _diagonal_alternative(passable, origin, direction, radius):
    """Diagnostic shortcut: diagonal destinations need no forward bridge."""
    mask = np.zeros(passable.shape, dtype=bool)
    previous = {0}
    for d in range(max(passable.shape) + 1):
        if not _inside(passable.shape, _coord(origin, direction, d, 0)):
            break
        if d:
            current = set()
            for j in range(-radius, radius + 1):
                cell = _coord(origin, direction, d, j)
                if (_inside(passable.shape, cell) and passable[cell]
                        and any(k in previous for k in (j - 1, j, j + 1))):
                    current.add(j)
            previous = current
        for j in previous:
            mask[_coord(origin, direction, d, j)] = True
    return mask


def _t_description(cells, grid):
    cells = set(cells)
    candidates = []
    for dr, dc in _DIRECTIONS:
        for center in cells:
            if (len(cells) - 3) % 2:
                continue
            radius = (len(cells) - 3) // 2
            if not 2 <= radius <= 6:
                continue
            bar = [_coord(center, (dr, dc), 0, j) for j in range(-radius, radius + 1)]
            stem = [_coord(center, (dr, dc), d, 0) for d in (1, 2)]
            if set(bar + stem) != cells:
                continue
            color = int(grid[center])
            if any(grid[p] != color for p in stem):
                continue
            candidates.append({"bar_center": center, "origin": stem[0],
                               "direction": (dr, dc), "radius": radius,
                               "template": [int(grid[p]) for p in bar],
                               "seed_color": color, "source_cells": cells})
    return candidates


def _analyze(input_grid):
    grid = _array(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    _require(counts[background] > grid.size // 2, "background must occupy over half of input")
    _, groups = _components(grid != background)
    spines, seeds, blocks = [], [], []
    for cells in groups:
        rows, cols = {r for r, _ in cells}, {c for _, c in cells}
        colors = {int(grid[p]) for p in cells}
        if (3 <= len(cells) <= 5 and len(colors) == len(cells)
                and (len(rows) == 1 or len(cols) == 1)):
            ordered = sorted(cells)
            if len(rows) == 1:
                ordered.sort(key=lambda p: p[1])
                edge0, edge1 = ordered[0][1] == 0, ordered[-1][1] == grid.shape[1] - 1
                contiguous = [p[1] for p in ordered] == list(range(ordered[0][1], ordered[-1][1] + 1))
                cross_inside = 0 < ordered[0][0] < grid.shape[0] - 1
            else:
                edge0, edge1 = ordered[0][0] == 0, ordered[-1][0] == grid.shape[0] - 1
                contiguous = [p[0] for p in ordered] == list(range(ordered[0][0], ordered[-1][0] + 1))
                cross_inside = 0 < ordered[0][1] < grid.shape[1] - 1
            if edge0 != edge1 and contiguous and cross_inside:
                if edge0:
                    ordered.reverse()
                spines.append([(r, c, int(grid[r, c])) for r, c in ordered])
                continue
        t = _t_description(cells, grid)
        if len(t) == 1:
            seeds.extend(t)
            continue
        if (len(colors) == 1 and len(rows) >= 2 and len(cols) >= 2
                and len(rows) * len(cols) == len(cells)):
            blocks.append(cells)
            continue
        raise ValueError("unrecognized or touching input objects")
    _require(len(spines) == 1 and len(seeds) == 1, "need one spine and one T seed")
    _require(len(blocks) <= 3, "too many baffles")
    spine, seed = spines[0], seeds[0]
    spine_colors = {c for _, _, c in spine}
    source_colors = set(seed["template"])
    block_colors = {int(grid[cells[0]]) for cells in blocks}
    _require(len(block_colors) <= 1 and not (source_colors & spine_colors), "color roles overlap")
    _require(not (block_colors & (source_colors | spine_colors)), "baffle color is reserved")
    scaffold = _scaffold(grid.shape, spine, background)
    baffles = np.zeros(grid.shape, dtype=bool)
    for cells in blocks:
        for p in cells:
            _require(scaffold[p] == background, "baffle covers a generated ray")
            baffles[p] = True
    passable = (scaffold == background) & ~baffles
    labels, regions = _components(passable)
    label = int(labels[seed["origin"]])
    _require(label >= 0 and all(labels[p] == label for p in seed["source_cells"]),
             "seed must fit wholly inside one chamber")
    reached, fronts = _trace(passable, seed["origin"], seed["direction"], seed["radius"])
    _require(np.all(labels[reached] == label), "growth escaped its 4-connected chamber")
    return {"grid": grid, "background": background, "spine": spine, "seed": seed,
            "blocks": blocks, "baffles": baffles, "scaffold": scaffold,
            "passable": passable, "labels": labels, "regions": regions,
            "label": label, "reached": reached, "fronts": fronts}


def _render(info):
    grid, seed = info["grid"], info["seed"]
    out = info["scaffold"].copy()
    out[info["baffles"]] = grid[info["baffles"]]
    for d, front in enumerate(info["fronts"]):
        for j in front:
            out[_coord(seed["origin"], seed["direction"], d, j)] = seed["template"][j + seed["radius"]]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the supplied grid; never consult generation or examples."""
    return _render(_analyze(input_grid))


def _features(info):
    s = info["seed"]
    ideal = _ideal(info["labels"] == info["label"], s["origin"], s["direction"], s["radius"])
    without_a = ~info["baffles"].copy()
    for r, c, _ in info["spine"]:
        without_a[r, c] = False
    no_a, _ = _trace(without_a, s["origin"], s["direction"], s["radius"])
    shadow = ideal & ~info["reached"]
    diagonal = _diagonal_alternative(info["labels"] == info["label"],
                                     s["origin"], s["direction"], s["radius"])
    recovery = 0
    for r, c in np.argwhere(shadow):
        dr, dc = s["direction"]
        for k in range(1, 5):
            p = (int(r) + k * dr, int(c) + k * dc)
            if _inside(info["grid"].shape, p) and info["reached"][p]:
                recovery += 1
                break
    out = _render(info)
    payload_witness = np.count_nonzero(info["reached"] & (out != s["seed_color"]))
    full_fronts = sum(len(f) == 2 * s["radius"] + 1 for f in info["fronts"])
    return {"a_effect": int(np.count_nonzero(no_a & ~info["reached"] &
                                            (info["scaffold"] == info["background"]))),
            "shadow": int(np.count_nonzero(shadow)), "recovery": recovery,
            "corner_effect": int(np.count_nonzero(diagonal & ~info["reached"])),
            "payload_witness": int(payload_witness), "full_fronts": full_fronts,
            "painted": int(np.count_nonzero(info["reached"]))}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _baffle_effects(info):
    """Count downstream open cells enabled by removing each baffle alone."""
    effects = []
    s = info["seed"]
    for cells in info["blocks"]:
        opened = info["passable"].copy()
        for p in cells:
            opened[p] = True
        alternate, _ = _trace(opened, s["origin"], s["direction"], s["radius"])
        effects.append(int(np.count_nonzero(alternate & ~info["reached"] & ~info["baffles"])))
    return effects


def _spine_positions(shape, direction, length, cross):
    h, w = shape
    dr, dc = direction
    if dr == 1:
        return [(h - length + i, cross) for i in range(length)]
    if dr == -1:
        return [(length - 1 - i, cross) for i in range(length)]
    if dc == 1:
        return [(cross, w - length + i) for i in range(length)]
    return [(cross, length - 1 - i) for i in range(length)]


def _template(rng, radius, choices, seed_color, style):
    n = 2 * radius + 1
    if style == "mirror":
        left = [int(x) for x in rng.choice(choices, radius)]
        values = left + [seed_color] + left[::-1]
    elif style == "runs":
        values = [int(choices[min(len(choices) - 1, j * len(choices) // n)]) for j in range(n)]
    elif style == "alternating":
        values = [int(choices[j % min(2, len(choices))]) for j in range(n)]
    elif style == "plain":
        values = [int(choices[0])] * n
    else:
        values = [int(x) for x in rng.choice(choices, n)]
    values[radius] = seed_color
    if style == "anchor":
        values[radius - 1], values[radius + 1] = int(choices[0]), int(choices[1])
        values[0], values[-1] = int(choices[1]), seed_color
    return values


def _moat_free(cells, occupied):
    h, w = occupied.shape
    for r, c in cells:
        if occupied[r, c]:
            return False
        for dr, dc in _DIRECTIONS:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and occupied[y, x]:
                return False
    return True


def _make_pair(rng, baffle_count, teaching, rotation):
    """Canonical seed faces down; rotate the complete scene on return."""
    for attempt in range(1600):
        if baffle_count >= 2:
            h, w = int(rng.integers(23, 31)), int(rng.integers(19, 31))
            radius = int(rng.integers(4, 7))
        else:
            h, w = int(rng.integers(15, 29)), int(rng.integers(13, 29))
            radius = int(rng.integers(3 if baffle_count else 2, 6))
        shape = (h, w)
        palette = [int(x) for x in rng.permutation(10)]
        length = int(rng.integers(3, 6))
        bg, wall, seed_color = palette[:3]
        spine_colors = palette[3:3 + length]
        choices = palette[3 + length:] + [seed_color]
        direction = _DIRECTIONS[int(rng.integers(4))]
        cross_limit = w if direction[0] else h
        cross = int(rng.integers(1, cross_limit - 1))
        positions = _spine_positions(shape, direction, length, cross)
        spine = [(r, c, color) for (r, c), color in zip(positions, spine_colors)]
        scaffold = _scaffold(shape, spine, bg)
        labels, regions = _components(scaffold == bg)
        bar_row = int(rng.integers(1, min(7, h - 10)))
        center = int(rng.integers(radius + 1, w - radius - 1))
        origin = (bar_row + 1, center)
        source_cells = [(bar_row, center + j) for j in range(-radius, radius + 1)]
        source_cells += [(bar_row + 1, center), (bar_row + 2, center)]
        label = int(labels[origin])
        if label < 0 or any(labels[p] != label for p in source_cells):
            continue
        inp = np.full(shape, bg, dtype=np.int8)
        for r, c, color in spine:
            inp[r, c] = color
        if not _moat_free(source_cells, inp != bg):
            continue
        style = "anchor" if teaching else str(rng.choice(
            ("shuffled", "mirror", "runs", "alternating", "plain")))
        template = _template(rng, radius, choices, seed_color, style)
        inp[bar_row, center - radius:center + radius + 1] = template
        inp[bar_row + 1:bar_row + 3, center] = seed_color
        base_reached, base_fronts = _trace(scaffold == bg, origin, (1, 0), radius)
        if np.count_nonzero(base_reached) < 28:
            continue
        if teaching and sum(len(f) == 2 * radius + 1 for f in base_fronts) < 2:
            continue
        # Each baffle is a visible rectangle in the seed's chamber. Some may
        # be off center; all intersect potential growth, with no object joins.
        placed = 0
        for _ in range(110):
            if placed == baffle_count:
                break
            bh = int(rng.integers(2, 4))
            bw = int(rng.integers(3, min(6, 2 * radius)))
            top = int(rng.integers(origin[0] + 3, h - bh - 2))
            left = int(rng.integers(center - radius, center + radius - bw + 2))
            cells = [(r, c) for r in range(top, top + bh) for c in range(left, left + bw)]
            if any(labels[p] != label for p in cells):
                continue
            if not _moat_free(cells, inp != bg):
                continue
            if sum(base_reached[p] for p in cells) < max(3, len(cells) // 2):
                continue
            for p in cells:
                inp[p] = wall
            placed += 1
        if placed != baffle_count:
            continue
        transformed = np.rot90(inp, rotation).copy()
        info = _analyze(transformed)
        f = _features(info)
        if f["a_effect"] < 4 or f["painted"] < 24 or f["payload_witness"] < (6 if teaching else 1):
            continue
        if teaching and baffle_count == 0 and f["full_fronts"] < 2:
            continue
        if baffle_count and (f["shadow"] < 2 or f["recovery"] < 1):
            continue
        if teaching and baffle_count and f["corner_effect"] < 1:
            continue
        # Every baffle must affect travel, rather than just occupy a displayed
        # rectangle: removing it must enable at least one later open cell.
        if not all(effect > 0 for effect in _baffle_effects(info)):
            continue
        return {"input": transformed, "output": _render(info)}
    raise RuntimeError("bounded scene construction exhausted")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # Two small teaching cases establish growth, saturation, shadows and
    # recovery. Tests compose multiple baffles at a larger scale.
    first_rotation = int(rng.integers(4))
    rotations = [first_rotation, (first_rotation + int(rng.choice((1, 3)))) % 4]
    rotations += [int(rng.integers(4)) for _ in range(nt + ne - 2)]
    baffles = [0, 1] + [int(rng.choice((0, 1, 2), p=(0.3, 0.5, 0.2))) for _ in range(nt - 2)]
    baffles += [int(rng.choice((2, 3), p=(0.7, 0.3))) for _ in range(ne)]
    training_maximum = max(baffles[:nt])
    baffles[nt:] = [max(n, training_maximum + 1) for n in baffles[nt:]]
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt + ne):
        for _ in range(20):
            pair = _make_pair(rng, baffles[i], i < 2, rotations[i])
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train" if i < nt else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not generate a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned values, family geometry, witnesses, and the curriculum."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _array(pair["input"]), _array(pair["output"])
            _require(inp.shape == out.shape, "output shape changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(inp)
            _require(np.array_equal(out, _render(info)), "incorrect propagation or color alignment")
            _require(not np.array_equal(inp, out), "identity pair")
            seed = info["seed"]
            _require(all(out[_coord(seed["bar_center"], seed["direction"], 0, j)] == info["background"]
                         for j in range(-seed["radius"], seed["radius"] + 1)), "source bar was not erased")
            _require(np.all(out[info["scaffold"] != info["background"]] ==
                               info["scaffold"][info["scaffold"] != info["background"]]), "ray colors changed")
            _require(np.all(out[info["baffles"]] == inp[info["baffles"]]), "baffles changed")
            _require(np.all(info["labels"][info["reached"]] == info["label"]), "chamber leak")
            f = _features(info)
            _require(f["a_effect"] >= 4 and f["payload_witness"] >= 1 and f["painted"] >= 24,
                     "one inherited mechanism lacks a visible effect")
            if info["blocks"]:
                _require(f["shadow"] >= 2 and f["recovery"] >= 1, "no shadow/recovery witness")
                _require(all(n > 0 for n in _baffle_effects(info)), "ineffective baffle")
            records[split].append((info, f))
    first, second = records["train"][:2]
    _require(not first[0]["blocks"] and len(second[0]["blocks"]) == 1, "teaching schedule missing")
    _require(first[1]["full_fronts"] >= 2, "saturation not demonstrated")
    _require(second[1]["corner_effect"] >= 1, "diagonal-jump shortcut not distinguished")
    for info, f in (first, second):
        s = info["seed"]
        _require(s["template"] != s["template"][::-1] and len(set(s["template"])) >= 3,
                 "aligned asymmetric template witness missing")
        _require(f["payload_witness"] >= 6, "payload copying underexposed")
    _require(first[0]["seed"]["direction"][0] != second[0]["seed"]["direction"][0],
             "training must show both growth axes")
    _require(all(len(info["blocks"]) >= 2 for info, _ in records["test"]), "compound test missing")
    _require(min(len(info["blocks"]) for info, _ in records["test"]) >
             max(len(info["blocks"]) for info, _ in records["train"]), "obstruction-count shift missing")
    return None


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed structural/semantic checks, with no files or network."""
    shapes, backgrounds, growth_directions, spine_directions = set(), set(), set(), set()
    radii, counts, templates, structures = set(), set(), set(), set()
    relative, witness_totals, puzzles = set(), [0, 0, 0], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _analyze(pair["input"])
                f = _features(info)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                backgrounds.add(info["background"])
                s = info["seed"]
                growth_directions.add(s["direction"])
                sd = (info["spine"][1][0] - info["spine"][0][0],
                      info["spine"][1][1] - info["spine"][0][1])
                spine_directions.add(sd)
                relative.add((sd[0] * s["direction"][0] + sd[1] * s["direction"][1],
                              sd[0] * s["direction"][1] - sd[1] * s["direction"][0]))
                radii.add(s["radius"])
                counts.add(len(info["blocks"]))
                templates.add(tuple(s["template"]))
                # Palette-normalized input geometry, not just color variation.
                structures.add((pair["input"].shape,
                                (pair["input"] != info["background"]).tobytes()))
                witness_totals[0] += f["a_effect"]
                witness_totals[1] += f["shadow"]
                witness_totals[2] += f["recovery"]
                # Rotation and a full role permutation must commute with solve.
                rotated = np.rot90(pair["input"])
                assert np.array_equal(solve(rotated), np.rot90(pair["output"]))
        if seed in (0, 21, 41):
            assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed))
            perm = np.random.default_rng(seed + 1000).permutation(10)
            for pair in p["train"] + p["test"]:
                assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
        pair = p["test"][0]
        assert np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"]))
    assert len(puzzles) == 42 and len(shapes) >= 60 and len(structures) >= 170
    assert backgrounds == set(range(10)) and growth_directions == set(_DIRECTIONS)
    assert spine_directions == set(_DIRECTIONS) and len(relative) == 4
    assert radii == {2, 3, 4, 5, 6} and counts == {0, 1, 2, 3}
    assert len(templates) >= 140 and all(n > 100 for n in witness_totals)
    # A valid semantic family member can have a terminally blocked front and
    # a monochrome template, even though teaching examples exclude these.
    terminal = np.zeros((30, 24), dtype=np.int8)
    terminal[27:30, 1] = [2, 3, 4]
    terminal[2, 6:11] = 7
    terminal[3:5, 8] = 7
    terminal[8:10, 6:11] = 1
    info = _analyze(terminal)
    assert info["fronts"][4] == frozenset(range(-2, 3))
    assert all(not front for front in info["fronts"][5:])
    assert np.array_equal(solve(terminal.tolist()), _render(info))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(4200 + 10 * nt + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for counts_bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                       (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_shapes": len(shapes),
            "foreground_geometries": len(structures), "template_sequences": len(templates),
            "radii": sorted(radii), "baffle_counts": sorted(counts),
            "a_effect_cells": witness_totals[0], "shadow_cells": witness_totals[1],
            "recovered_columns": witness_totals[2]}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
