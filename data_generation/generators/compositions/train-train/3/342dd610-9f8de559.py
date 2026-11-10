"""Escort vectors: a leader's first panel exit translates its color group.

Only NumPy and the Python standard library are required. No parent code or data
is imported. Labels are rendered from construction geometry; solve parses only
the supplied input. Run this module directly for the bounded self-test.
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _add(p, v):
    return p[0] + v[0], p[1] + v[1]


def _inside(p, bounds, strict=False):
    t, b, l, r = bounds
    if strict:
        return t < p[0] < b and l < p[1] < r
    return t <= p[0] <= b and l <= p[1] <= r


def _apart(p, points):
    return all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in points)


def _components(points):
    unseen = set(points)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for d in DIRECTIONS:
                q = _add(p, d)
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    queue.append(q)
        result.append(component)
    return result


def _ray(tip, direction, bounds):
    """Return interior ray cells, first exterior cell, and tip-to-exit distance."""
    p = _add(tip, direction)
    cells = []
    while _inside(p, bounds):
        cells.append(p)
        p = _add(p, direction)
    return cells, p, len(cells) + 1


def _analyze(input_grid):
    """Recover every role and command from the input, with no generation state."""
    grid = _grid(input_grid)
    h, w = grid.shape
    background = int(grid[0, 0])
    _require(bool(np.all(grid[0, :] == background) and np.all(grid[-1, :] == background)
                  and np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)),
             "outer border must be uniform exterior")
    positions = np.argwhere(grid != background)
    _require(len(positions) > 0, "missing panel")
    top, left = map(int, positions.min(axis=0))
    bottom, right = map(int, positions.max(axis=0))
    bounds = top, bottom, left, right
    _require(0 < top < bottom < h - 1 and 0 < left < right < w - 1, "panel must be inset")
    panel = grid[top:bottom + 1, left:right + 1]
    fill = int(panel[0, 0])
    _require(fill != background, "missing panel fill")
    _require(bool(np.all(panel[0, :] == fill) and np.all(panel[-1, :] == fill)
                  and np.all(panel[:, 0] == fill) and np.all(panel[:, -1] == fill)),
             "input panel border must be intact")
    _require(not np.any(panel == background), "exterior color cannot occur in panel")
    palette = set(map(int, np.unique(panel))) - {fill}
    components = {color: _components({tuple(map(int, p)) for p in np.argwhere(grid == color)})
                  for color in sorted(palette)}
    shaft_candidates = [color for color, comps in components.items() if any(len(c) > 1 for c in comps)]
    _require(len(shaft_candidates) == 1, "shaft role must be structurally unique")
    shaft = shaft_candidates[0]
    colors = sorted(palette - {shaft})
    _require(1 <= len(colors) <= 4, "need one to four color groups")
    sources = {color: {tuple(map(int, p)) for p in np.argwhere(grid == color)} for color in colors}
    tokens = set().union(*sources.values())
    _require(all(_apart(p, tokens - {p}) for p in tokens), "singleton sources touch")
    _require(all(_inside(p, bounds, strict=True) for p in tokens), "token on panel boundary")
    shaft_cells = {tuple(map(int, p)) for p in np.argwhere(grid == shaft)}
    groups = {}
    leaders = set()
    for comp in components[shaft]:
        length = len(comp)
        _require(2 <= length <= 5, "shaft length must be two to five")
        possibilities = []
        for p in sorted(comp):
            for direction in DIRECTIONS:
                tip = _add(p, direction)
                if tip not in tokens:
                    continue
                expected = {(tip[0] - j * direction[0], tip[1] - j * direction[1])
                            for j in range(1, length + 1)}
                if expected == comp:
                    possibilities.append((tip, direction))
        _require(len(possibilities) == 1, "shaft must have one unique terminal tip")
        tip, direction = possibilities[0]
        color = int(grid[tip])
        _require(color not in groups, "multiple leaders of one color")
        ray, exit_cell, distance = _ray(tip, direction, bounds)
        _require(distance >= 2, "need fill ahead of the tip")
        _require(all(int(grid[p]) == fill for p in ray), "forward ray is not clear")
        _require(0 <= exit_cell[0] < h and 0 <= exit_cell[1] < w, "exit outside grid")
        vector = distance * direction[0], distance * direction[1]
        groups[color] = {"tip": tip, "direction": direction, "length": length,
                         "distance": distance, "exit": exit_cell, "vector": vector,
                         "sources": sources[color], "ray": ray, "shaft_cells": comp}
        leaders.add(tip)
    _require(set(groups) == set(colors), "each token color needs exactly one pointer")
    for tip in leaders:
        color = int(grid[tip])
        other_shafts = shaft_cells - groups[color]["shaft_cells"]
        _require(_apart(tip, other_shafts), "leader touches another shaft")
    _require(all(_apart(p, shaft_cells) for p in tokens - leaders), "follower touches shaft")
    destinations = set()
    for color, group in groups.items():
        _require(len(group["sources"]) >= 2, "each group needs a follower")
        followers_inside = 0
        for source in group["sources"]:
            dest = _add(source, group["vector"])
            _require(0 <= dest[0] < h and 0 <= dest[1] < w, "translated token outside grid")
            _require(dest not in shaft_cells, "destination overwrites a stationary shaft")
            _require(dest not in destinations, "two tokens claim one destination")
            destinations.add(dest)
            followers_inside += source != group["tip"] and _inside(dest, bounds, strict=True)
        _require(followers_inside > 0, "group must demonstrate a follower staying inside")
    return {"background": background, "fill": fill, "shaft": shaft, "bounds": bounds,
            "groups": groups, "tokens": tokens, "shaft_cells": shaft_cells,
            "destinations": destinations}


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    meta = _analyze(grid)
    result = grid.copy()
    # Clear every source before drawing any destination, including vacated sources.
    for source in meta["tokens"]:
        result[source] = meta["fill"]
    for color, group in meta["groups"].items():
        for source in group["sources"]:
            result[_add(source, group["vector"])] = color
    return result


def _sample_pair(rng, roles, active, directions, lengths, populations, forbidden_distances):
    """Bounded geometric construction; this never calls solve to make labels."""
    background, fill, shaft = roles
    count = len(active)
    for _ in range(180):
        ph, pw = map(int, rng.integers(12 + count, 25, size=2))
        t = int(rng.integers(1, min(5, 29 - ph) + 1))
        bmargin = int(rng.integers(1, min(5, 30 - ph - t) + 1))
        l = int(rng.integers(1, min(5, 29 - pw) + 1))
        rmargin = int(rng.integers(1, min(5, 30 - pw - l) + 1))
        shape = ph + t + bmargin, pw + l + rmargin
        bounds = t, t + ph - 1, l, l + pw - 1
        inner = [(r, c) for r in range(t + 1, t + ph - 1) for c in range(l + 1, l + pw - 1)]
        shaft_cells, tip_cells, rays, exits = set(), set(), set(), set()
        groups = {}
        failed = False
        for color, direction, length in zip(active, directions, lengths):
            candidates = []
            dr, dc = direction
            for tip in inner:
                cells = {(tip[0] - step * dr, tip[1] - step * dc) for step in range(1, length + 1)}
                if not all(_inside(p, bounds, strict=True) for p in cells):
                    continue
                ray, exit_cell, distance = _ray(tip, direction, bounds)
                if distance in forbidden_distances.get(color, ()):
                    continue
                if not ray or exit_cell in exits:
                    continue
                markers = cells | {tip}
                if markers & rays or set(ray) & (shaft_cells | tip_cells):
                    continue
                if not all(_apart(p, shaft_cells | tip_cells) for p in markers):
                    continue
                candidates.append((tip, cells, ray, exit_cell, distance))
            if not candidates:
                failed = True
                break
            tip, cells, ray, exit_cell, distance = _pick(rng, candidates)
            shaft_cells |= cells
            tip_cells.add(tip)
            rays.update(ray)
            exits.add(exit_cell)
            groups[color] = {"tip": tip, "direction": direction, "length": length,
                             "distance": distance, "vector": (distance * dr, distance * dc),
                             "sources": [tip]}
        if failed:
            continue
        sources = set(tip_cells)
        destinations = set(exits)
        for color, population in zip(active, populations):
            group = groups[color]
            for follower_index in range(population - 1):
                candidates = []
                for source in inner:
                    if source in rays or not _apart(source, sources | shaft_cells):
                        continue
                    dest = _add(source, group["vector"])
                    if not (0 <= dest[0] < shape[0] and 0 <= dest[1] < shape[1]):
                        continue
                    if dest in destinations or dest in shaft_cells:
                        continue
                    if follower_index == 0 and not _inside(dest, bounds, strict=True):
                        continue
                    candidates.append((source, dest))
                if not candidates:
                    failed = True
                    break
                source, dest = _pick(rng, candidates)
                sources.add(source)
                destinations.add(dest)
                group["sources"].append(source)
            if failed:
                break
        if failed:
            continue
        static = np.full(shape, background, dtype=np.int8)
        static[t:t + ph, l:l + pw] = fill
        for p in shaft_cells:
            static[p] = shaft
        inp, out = static.copy(), static.copy()
        for color, group in groups.items():
            for source in group["sources"]:
                inp[source] = color
                out[_add(source, group["vector"])] = color
        return {"input": inp, "output": out}, groups
    raise RuntimeError("bounded geometric construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    role_count = int(rng.integers(2, 5))
    palette = list(map(int, rng.choice(10, size=role_count + 3, replace=False)))
    roles, colors = palette[:3], palette[3:]
    held_out = _pick(rng, DIRECTIONS)
    allowed = [d for d in DIRECTIONS if d != held_out]
    first_cardinal = _pick(rng, [d for d in CARDINAL if d != held_out])
    second_diagonal = _pick(rng, [d for d in DIAGONAL if d != held_out])
    teaching_length = int(rng.integers(2, 6))
    seen = set()
    first_distance = None
    puzzle = {"train": [], "test": []}
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            if split == "train" and index == 0:
                subset = colors[:2]
                directions = [first_cardinal, second_diagonal]
            elif (split == "train" and index == 1) or (split == "test" and index == 0):
                subset = colors[:]
                directions = [_pick(rng, allowed if split == "train" else DIRECTIONS) for _ in subset]
                directions[0] = second_diagonal if split == "train" else held_out
            else:
                size = int(rng.integers(1, role_count + 1))
                subset = list(map(int, rng.choice(colors, size=size, replace=False)))
                directions = [_pick(rng, allowed if split == "train" else DIRECTIONS) for _ in subset]
            populations = list(map(int, rng.integers(2, 4, size=len(subset)) if split == "train"
                                   else rng.integers(4, 7, size=len(subset))))
            lengths = list(map(int, rng.integers(2, 6, size=len(subset))))
            forbidden = {}
            if split == "train" and index < 2:
                lengths[0] = teaching_length
                forbidden[colors[0]] = {teaching_length} if index == 0 else {first_distance}
            for _ in range(20):
                pair, groups = _sample_pair(rng, roles, subset, directions, lengths,
                                            populations, forbidden)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                first_distance = groups[colors[0]]["distance"]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "illegal counts")
    metadata = {"train": [], "test": []}
    seen, roles = set(), None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "grid shape changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            meta = _analyze(inp)
            metadata[split].append(meta)
            pair_roles = meta["background"], meta["fill"], meta["shaft"]
            if roles is None:
                roles = pair_roles
            _require(pair_roles == roles, "stationary palette roles changed")
            expected = inp.copy()
            for p in meta["tokens"]:
                expected[p] = meta["fill"]
            for color, group in meta["groups"].items():
                destinations = {_add(p, group["vector"]) for p in group["sources"]}
                actual_positions = {tuple(map(int, p)) for p in np.argwhere(out == color)}
                _require(actual_positions == destinations, "color group is not a rigid translation")
                _require(len(actual_positions) == len(group["sources"]), "token count changed")
                _require(out[group["exit"]] == color, "leader does not reach first exterior cell")
                for p in destinations:
                    expected[p] = color
                population = len(group["sources"])
                _require((2 <= population <= 3) if split == "train" else (4 <= population <= 6),
                         "wrong population shift")
            _require(np.array_equal(expected, out), "stationary scene or translated tokens changed")
            _require(np.array_equal(solve(inp), out), "solve disagrees")
            _require(not np.array_equal(inp, out), "identity pair")
            # Ablation A: move leaders only. Followers must materially change the answer.
            leaders_only = inp.copy()
            for group in meta["groups"].values():
                leaders_only[group["tip"]] = meta["fill"]
            for color, group in meta["groups"].items():
                leaders_only[group["exit"]] = color
            _require(not np.array_equal(leaders_only, out), "A contribution is unnecessary")
            # Ablation B: one unit step cannot reach any leader's first exterior target.
            _require(all(_add(g["tip"], g["direction"]) != g["exit"] for g in meta["groups"].values()),
                     "B contribution is unnecessary")
    train_colors = set().union(*(set(m["groups"]) for m in metadata["train"]))
    test_colors = set().union(*(set(m["groups"]) for m in metadata["test"]))
    _require(2 <= len(train_colors) <= 4 and test_colors <= train_colors, "untaught color role")
    first, second = metadata["train"][:2]
    first_directions = {g["direction"] for g in first["groups"].values()}
    _require(any(d in CARDINAL for d in first_directions) and any(d in DIAGONAL for d in first_directions),
             "first demonstration needs cardinal and diagonal groups")
    common = set(first["groups"]) & set(second["groups"])
    _require(any(first["groups"][c]["direction"] != second["groups"][c]["direction"]
                 and first["groups"][c]["distance"] != second["groups"][c]["distance"]
                 and first["groups"][c]["length"] == second["groups"][c]["length"] for c in common),
             "training does not distinguish current geometry from fixed color/shaft commands")
    train_dirs = {g["direction"] for m in metadata["train"] for g in m["groups"].values()}
    test_dirs = {g["direction"] for m in metadata["test"] for g in m["groups"].values()}
    _require(bool(test_dirs - train_dirs), "no held-out direction shift")
    return None


def _puzzle_key(puzzle):
    return tuple((split, _key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _vacated_source_example():
    """A right-facing leader pulls one follower into its own old tip cell."""
    a = np.zeros((18, 20), dtype=np.int8)
    a[1:17, 1:17] = 7
    a[8, 9:11] = 5
    # Tip (8,11) exits at (8,17), vector (0,6). Follower (8,5) -> tip.
    for p in ((8, 11), (8, 5), (5, 4)):
        a[p] = 2
    out = np.zeros_like(a)
    out[1:17, 1:17] = 7
    out[8, 9:11] = 5
    for p in ((8, 17), (8, 11), (5, 10)):
        out[p] = 2
    return a, out


def self_test():
    """Bounded family invariants and observed structural variation over 42 seeds."""
    corpus = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in corpus}) == 42, "seed diversity failed")
    shapes, panel_shapes, palettes, layouts = set(), set(), set(), set()
    directions = {"train": set(), "test": set()}
    counts, lengths, distances, group_counts, exit_sides = set(), set(), set(), set(), set()
    source_overlap = 0
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                meta = _analyze(inp)
                shapes.add(inp.shape)
                top, bottom, left, right = meta["bounds"]
                panel_shapes.add((bottom - top + 1, right - left + 1))
                palettes.add((meta["background"], meta["fill"], meta["shaft"], tuple(sorted(meta["groups"]))))
                group_counts.add(len(meta["groups"]))
                source_overlap += bool(meta["tokens"] & meta["destinations"])
                for color, g in meta["groups"].items():
                    directions[split].add(g["direction"])
                    lengths.add(g["length"])
                    distances.add(g["distance"])
                    counts.add(len(g["sources"]))
                    er, ec = g["exit"]
                    side = (int(er < top) - int(er > bottom), int(ec < left) - int(ec > right))
                    exit_sides.add(side)
                    layouts.add((inp.shape, meta["bounds"], g["tip"], g["direction"],
                                 g["length"], tuple(sorted(g["sources"]))))
    _require(directions["train"] == set(DIRECTIONS) and directions["test"] == set(DIRECTIONS), "direction diversity")
    _require(lengths == {2, 3, 4, 5}, "shaft length diversity")
    _require(counts == {2, 3, 4, 5, 6}, "population diversity")
    _require(group_counts == {1, 2, 3, 4}, "group-count diversity")
    _require(len(shapes) >= 35 and len(panel_shapes) >= 30 and len(palettes) >= 40, "geometry/palette diversity")
    _require(len(layouts) >= 300 and min(distances) == 2 and max(distances) >= 10, "constellation/ray diversity")
    _require(any(a and b for a, b in exit_sides), "no diagonal corner exit")
    _require(source_overlap > 0, "no moves into vacated sources")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(10000 + 10 * nt + ne, nt, ne)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == ne, "count request ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_puzzle_key(generate_puzzle(0)) == _puzzle_key(corpus[0]), "seed determinism failed")
    _require(_puzzle_key(generate_puzzle(771, 10, 4)) == _puzzle_key(generate_puzzle(771, 10, 4)),
             "explicit count determinism failed")
    inp, out = _vacated_source_example()
    _require(np.array_equal(solve(inp), out), "simultaneous movement into vacated source failed")
    # Validate a nested-list copy, and reject a plausible wrong answer with intact schema.
    converted = {s: [{k: np.asarray(v).tolist() for k, v in pair.items()} for pair in corpus[0][s]]
                 for s in ("train", "test")}
    validate_generated_puzzle(converted)
    broken = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in corpus[0][s]]
              for s in ("train", "test")}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an unmoved constellation")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in corpus for s in ("train", "test")),
            "grid_shapes": len(shapes), "panel_shapes": len(panel_shapes), "palettes": len(palettes),
            "constellation_layouts": len(layouts), "group_counts": sorted(group_counts),
            "shaft_lengths": sorted(lengths), "populations": sorted(counts),
            "exit_distances": sorted(distances), "exit_categories": len(exit_sides),
            "source_overlap_pairs": source_overlap, "all_legal_count_combinations": True}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
