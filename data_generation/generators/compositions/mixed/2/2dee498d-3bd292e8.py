"""Recover a column-canonical square, then color its divider-crossing parity.

Only standard-library modules and NumPy are used. Generation and checks do no
I/O, retain no corpus, and use bounded retries. Colors 3/5 are fixed even/odd
fills; the two input colors may be any distinct members of the other eight.
"""
from __future__ import annotations

import hashlib
import itertools
from collections import deque

import numpy as np


EVEN, ODD = 3, 5
INPUT_COLORS = (0, 1, 2, 4, 6, 7, 8, 9)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _need(isinstance(value, (int, np.integer))
          and not isinstance(value, (bool, np.bool_)), f"{name} must be an integer")
    value = int(value)
    _need(value >= low and (high is None or value <= high), f"illegal {name}")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    _need(a > b, "num_train must exceed num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
          "grids must be nonempty, two-dimensional, and at most 30 by 30")
    _need(np.issubdtype(a.dtype, np.integer), "grid colors must be integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _column_keys(a):
    return [tuple(int(x) for x in a[:, c]) for c in range(a.shape[1])]


def _deduplicate(a):
    seen, keep = set(), []
    for c, key in enumerate(_column_keys(a)):
        if key not in seen:
            seen.add(key)
            keep.append(c)
    return a[:, keep].copy()


def _components(mask):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    groups = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] >= 0:
                continue
            label = len(groups)
            labels[r, c] = label
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in N4:
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and mask[yy, xx]
                            and labels[yy, xx] < 0):
                        labels[yy, xx] = label
                        stack.append((yy, xx))
            groups.append(cells)
    return labels, groups


def _analyze_walls(wall):
    """Check actual raster geometry and derive the dual tree, without labels."""
    h, w = wall.shape
    _need(bool(np.any(wall)) and not wall[-1, 0], "missing wall or blocked anchor")
    owners, walls = _components(wall)
    paths, turn_counts, boundary_counts, endpoint_pairs = [], [], [], []
    for owner, cells in enumerate(walls):
        cell_set = set(cells)
        endpoints = []
        for r, c in cells:
            degree = sum((r + dy, c + dx) in cell_set for dy, dx in N4)
            _need(degree in (1, 2), "divider branches, touches itself, or is cyclic")
            if degree == 1:
                endpoints.append((r, c))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = r + dy, c + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        _need(owners[yy, xx] in (-1, owner),
                              "distinct dividers touch, even diagonally")
        _need(len(endpoints) == 2, "divider must be a simple open path")
        _need(all(r in (0, h - 1) or c in (0, w - 1) for r, c in endpoints),
              "divider endpoints must lie on the outer boundary")
        previous, current = None, endpoints[0]
        ordered = [current]
        while current != endpoints[1]:
            following = [(current[0] + dy, current[1] + dx) for dy, dx in N4
                         if (current[0] + dy, current[1] + dx) in cell_set
                         and (current[0] + dy, current[1] + dx) != previous]
            _need(len(following) == 1, "ambiguous divider traversal")
            previous, current = current, following[0]
            ordered.append(current)
        _need(len(ordered) == len(cells), "divider contains an untraced part")
        directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(ordered, ordered[1:])]
        paths.append(ordered)
        endpoint_pairs.append(endpoints)
        turn_counts.append(sum(a != b for a, b in zip(directions, directions[1:])))
        boundary_counts.append(sum(r in (0, h - 1) or c in (0, w - 1) for r, c in cells))

    labels, regions = _components(~wall)
    _need(len(regions) == len(walls) + 1, "each divider must add exactly one region")
    edges, adjacency = [], [set() for _ in regions]
    for cells in walls:
        neighbors = set()
        for r, c in cells:
            for dy, dx in N4:
                yy, xx = r + dy, c + dx
                if 0 <= yy < h and 0 <= xx < w and labels[yy, xx] >= 0:
                    neighbors.add(int(labels[yy, xx]))
        _need(len(neighbors) == 2, "a divider must separate exactly two nonempty regions")
        a, b = sorted(neighbors)
        _need(b not in adjacency[a], "multiple divider edges between the same regions")
        edges.append((a, b))
        adjacency[a].add(b)
        adjacency[b].add(a)
    root = int(labels[-1, 0])
    depths = [-1] * len(regions)
    depths[root] = 0
    queue = deque([root])
    while queue:
        a = queue.popleft()
        for b in sorted(adjacency[a]):
            if depths[b] < 0:
                depths[b] = depths[a] + 1
                queue.append(b)
            else:
                _need((depths[a] - depths[b]) % 2 == 1, "inconsistent crossing parity")
    _need(min(depths) >= 0 and len(edges) == len(regions) - 1,
          "region graph must be a connected tree")
    return {"labels": labels, "regions": regions, "paths": paths, "edges": edges,
            "depths": depths, "root": root, "turns": turn_counts,
            "boundary_counts": boundary_counts, "endpoints": endpoint_pairs,
            "max_degree": max(map(len, adjacency))}


def _read_input(input_grid):
    source = _grid(input_grid)
    colors = set(int(c) for c in np.unique(source))
    _need(len(colors) == 2 and not colors.intersection((EVEN, ODD)),
          "input needs two colors distinct from fixed fill colors 3 and 5")
    canonical = _deduplicate(source)
    _need(canonical.shape[0] == canonical.shape[1], "deduplication must recover a square")
    field = int(canonical[-1, 0])
    divider = next(iter(colors - {field}))
    info = _analyze_walls(canonical == divider)
    return source, canonical, field, divider, info


def _paint(canonical, divider, info):
    result = np.full(canonical.shape, divider, dtype=np.int8)
    for region, depth in enumerate(info["depths"]):
        result[info["labels"] == region] = EVEN if depth % 2 == 0 else ODD
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """An input-only solver: exact column identity precedes all segmentation."""
    _, canonical, _, divider, info = _read_input(input_grid)
    return _paint(canonical, divider, info)


def _parts(rng, total, count):
    if count == 1:
        return [total]
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), count - 1, replace=False))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _corner_path(rng, n, corner, rows, cols, diagonal=False):
    steps = cols if diagonal else int(rng.integers(1, min(rows, cols) + 1))
    horizontal = _parts(rng, cols, steps)
    vertical = _parts(rng, rows, steps)
    r, c = rows, 0
    path = [(r, c)]
    for dx, dy in zip(horizontal, vertical):
        for _ in range(dx):
            c += 1
            path.append((r, c))
        for _ in range(dy):
            r -= 1
            path.append((r, c))
    if corner in ("TR", "BR"):
        path = [(r, n - 1 - c) for r, c in path]
    if corner in ("BL", "BR"):
        path = [(n - 1 - r, c) for r, c in path]
    return path


def _mask(n, paths):
    result = np.zeros((n, n), dtype=bool)
    for path in paths:
        for r, c in path:
            result[r, c] = True
    return result


def _extend_boundary(rng, n, paths):
    """Try a terminal border stretch; actual topology decides validity later."""
    index = int(rng.integers(len(paths)))
    path = paths[index]
    end = int(rng.integers(2))
    r, c = path[0] if end == 0 else path[-1]
    vertical = c in (0, n - 1)
    coordinate = r if vertical else c
    direction = (-1, 1)[int(rng.integers(2))]
    room = coordinate if direction < 0 else n - 1 - coordinate
    if room:
        distance = int(rng.integers(1, room + 1))
        extension = [(r + direction * d, c) if vertical else (r, c + direction * d)
                     for d in range(1, distance + 1)]
        paths[index] = extension[::-1] + path if end == 0 else path + extension


def _corner_mask(rng, n, anchor_demo=False):
    # The two long diagonals encode distinct columns without extra ID marks.
    # Their wall row ranges are disjoint, and together they cover every column.
    a = int(rng.integers(2, n - 3))
    b = n - 2 - a
    paths = [_corner_path(rng, n, "TL", a, a, True),
             _corner_path(rng, n, "BR", b, b, True)]
    if anchor_demo or rng.random() < .65:
        paths.append(_corner_path(rng, n, "TR", int(rng.integers(1, a)),
                                  int(rng.integers(1, b))))
    if not anchor_demo:
        if rng.random() < .4:
            paths.append(_corner_path(rng, n, "BL", int(rng.integers(1, b)),
                                      int(rng.integers(1, a))))
        for corner, reach in (("TL", a), ("BR", b)):
            for inner in range(reach - 4, 0, -4):
                if rng.random() < .6:
                    paths.append(_corner_path(rng, n, corner, inner, inner, True))
        if rng.random() < .3:
            _extend_boundary(rng, n, paths)
    wall = _mask(n, paths)
    # Reflect across the anti-diagonal, which leaves the reference corner fixed.
    if rng.random() < .5:
        wall = wall.T[::-1, ::-1].copy()
    return wall


def _gray_states(bands):
    states = [()]
    for band in reversed(bands):
        following = []
        for i, value in enumerate(band):
            suffixes = states if i % 2 == 0 else states[::-1]
            following.extend((value,) + suffix for suffix in suffixes)
        states = following
    return states


def _crossing_from_states(n, states):
    count = len(states[0])
    paths = [[(states[0][i], 0)] for i in range(count)]
    c = 0
    for j in range(1, len(states)):
        c += 1
        for i in range(count):
            old, new = states[j - 1][i], states[j][i]
            paths[i].append((old, c))
            step = 1 if new > old else -1
            paths[i].extend((r, c) for r in range(old + step, new + step, step))
        # Even widths omit one spacer. The first three states must then be
        # coordinatewise monotone, preventing a wall from doubling back.
        if not (n % 2 == 0 and j == 1):
            c += 1
            for i in range(count):
                paths[i].append((states[j][i], c))
    _need(c == n - 1, "incorrect crossing width")
    return _mask(n, paths)


def _crossing_mask(rng, n, count):
    slack = n - 3 * count - 1
    _need(slack >= 0, "not enough space for crossing bands")
    sizes = 2 + rng.multinomial(slack, [1 / count] * count)
    bands, start = [], 1
    for size in sizes:
        bands.append(list(range(start, start + int(size))))
        start += int(size) + 1
    states = list(itertools.product(*bands))
    required = n // 2 + 1
    _need(len(states) >= required, "not enough distinct column states")
    for attempt in range(97):
        if attempt == 96:
            chosen = _gray_states(bands)[:required]
        else:
            chosen = [states[int(i)] for i in rng.choice(len(states), required, replace=False)]
        if n % 2 == 0:
            monotone = all((a <= b <= c) or (a >= b >= c)
                           for a, b, c in zip(*chosen[:3]))
            if not monotone:
                continue
        wall = _crossing_from_states(n, chosen)
        if len(set(_column_keys(wall))) == n:
            # Flips change geometry without moving the compression axis.
            if rng.random() < .5:
                wall = wall[::-1, :]
            if rng.random() < .5:
                wall = wall[:, ::-1]
            return wall.copy()
    raise RuntimeError("bounded crossing construction failed")


def _canonical_mask(rng, n, style):
    for _ in range(96):
        if style == "anchor":
            wall = _corner_mask(rng, n, True)
        elif style in ("even", "one"):
            wall = _crossing_mask(rng, n, 2 if style == "even" else 1)
        elif rng.random() < .5:
            wall = _corner_mask(rng, n)
        else:
            wall = _crossing_mask(rng, n, int(rng.integers(1, (n - 1) // 3 + 1)))
        if len(set(_column_keys(wall))) != n:
            continue
        try:
            info = _analyze_walls(wall)
        except ValueError:
            continue
        if style == "anchor":
            if not all(not wall[r, c] and info["depths"][info["labels"][r, c]] % 2 == 1
                       for r, c in ((0, 0), (0, n - 1), (n - 1, n - 1))):
                continue
        return wall, info
    raise RuntimeError("bounded canonical geometry sampling failed")


def _expand(rng, canonical, early_style, tail_style=None):
    n = canonical.shape[1]
    repeats = [2] * n
    for _ in range(int(rng.integers(0, min(6, 30 - 2 * n) + 1))):
        repeats[int(rng.integers(n))] += 1
    order = list(range(n))
    if early_style == "adjacent":
        chosen = int(rng.integers(n - 1))
        order.insert(chosen + 1, chosen)
    elif early_style == "separated":
        chosen = int(rng.integers(n - 2))
        order.insert(int(rng.integers(chosen + 2, n)), chosen)
    else:
        count = int(rng.integers(3, min(6, n - 2) + 1))
        for chosen in rng.choice(n - 2, count, replace=False):
            chosen = int(chosen)
            first, last_new = order.index(chosen), order.index(n - 1)
            order.insert(int(rng.integers(first + 2, last_new + 1)), chosen)
    remaining = [i for i in range(n) for _ in range(repeats[i] - order.count(i))]
    style = tail_style or ("shuffle", "forward", "reverse")[int(rng.integers(3))]
    if style == "shuffle":
        rng.shuffle(remaining)
    elif style == "reverse":
        remaining.sort(reverse=True)
    else:
        remaining.sort()
    order.extend(remaining)
    return canonical[:, order].copy()


def _new_pair(rng, n, style, early, seen):
    for _ in range(64):
        wall, info = _canonical_mask(rng, n, style)
        field, divider = (int(c) for c in rng.choice(INPUT_COLORS, 2, replace=False))
        canonical = np.where(wall, divider, field).astype(np.int8)
        source = _expand(rng, canonical, early, "reverse" if style == "anchor" else None)
        if _key(source) in seen:
            continue
        if style == "anchor":
            try:
                _analyze_walls(source == divider)
            except ValueError:
                pass
            else:
                # Require a visible causal-order witness, not only a width change.
                continue
        seen.add(_key(source))
        return {"input": source, "output": _paint(canonical, divider, info)}
    raise RuntimeError("bounded distinct-pair sampling failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    if regime == 0:
        train_sizes = [int(rng.integers(7, 12)) for _ in range(num_train)]
        test_sizes = [int(rng.integers(12, 16)) for _ in range(num_test)]
    elif regime == 1:
        train_sizes = [int(rng.integers(10, 16)) for _ in range(num_train)]
        test_sizes = [int(rng.integers(7, 10)) for _ in range(num_test)]
    else:
        train_sizes = [int(rng.integers(7, 14)) for _ in range(num_train)]
        test_sizes = [int(rng.choice(train_sizes)) for _ in range(num_test)]
    seen, train, test = set(), [], []
    for i, n in enumerate(train_sizes):
        style = "anchor" if i == 0 else ("even" if i == 1 else ("one" if i == 2 else "mixed"))
        early = "separated" if i % 2 == 0 else "adjacent"
        train.append(_new_pair(rng, n, style, early, seen))
    for n in test_sizes:
        test.append(_new_pair(rng, n, "mixed", "many", seen))
    rng.shuffle(train)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _order_evidence(source, canonical):
    keys = _column_keys(source)
    unique = _column_keys(canonical)
    ids = [unique.index(key) for key in keys]
    last_new = max(ids.index(i) for i in range(len(unique)))
    early = [j for j in range(last_new) if ids[j] in ids[:j]]
    last_order = sorted(range(len(unique)), key=lambda i: len(ids) - 1 - ids[::-1].index(i))
    return {"early_count": len(early),
            "early_separated": any(ids[j] != ids[j - 1] for j in early),
            "last_order_differs": last_order != list(range(len(unique))),
            "multiplicities": [ids.count(i) for i in range(len(unique))]}


def validate_generated_puzzle(puzzle):
    """Validate semantics, training counterexamples, split shift, and schema.

    Labels are checked by constant component colors, the anchor, and every
    divider edge, rather than just by calling solve and comparing its return.
    """
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    corner_phases = [set(), set(), set()]
    witness = {"separated": False, "last": False, "raw_invalid": False, "return": False,
               "histogram": False, "rows": False, "sorting": False}
    train_early, test_early = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            source, canonical, _, divider, info = _read_input(pair["input"])
            target = _grid(pair["output"])
            n = source.shape[0]
            _need(7 <= n <= 15 and target.shape == (n, n), "invalid canonical/output dimensions")
            _need(source.shape[1] > n and _key(source) not in seen, "duplicate or unexpanded input")
            seen.add(_key(source))
            order = _order_evidence(source, canonical)
            _need(min(order["multiplicities"]) >= 2, "each retained column must occur at least twice")
            _need(np.array_equal(target == divider, canonical == divider), "divider geometry changed")
            fills = []
            for cells in info["regions"]:
                colors = {int(target[r, c]) for r, c in cells}
                _need(len(colors) == 1 and colors <= {EVEN, ODD}, "region must have one parity fill")
                fills.append(next(iter(colors)))
            _need(fills[info["root"]] == EVEN, "lower-left phase is wrong")
            _need(all(fills[a] != fills[b] for a, b in info["edges"]), "crossing fails to toggle fill")
            if split == "train":
                train_early.append(order["early_count"])
                witness["separated"] |= order["early_separated"]
                witness["last"] |= order["last_order_differs"]
                witness["return"] |= max(info["depths"]) >= 2
                wall_counts = [int(np.count_nonzero(canonical[:, c] == divider)) for c in range(n)]
                witness["histogram"] |= len(set(wall_counts)) < n
                witness["rows"] |= len({tuple(row) for row in canonical.tolist()}) < n
                keys = _column_keys(canonical)
                witness["sorting"] |= keys != sorted(keys)
                for i, (r, c) in enumerate(((0, 0), (0, n - 1), (n - 1, n - 1))):
                    label = int(info["labels"][r, c])
                    if label >= 0:
                        corner_phases[i].add(info["depths"][label] % 2)
                try:
                    _analyze_walls(source == divider)
                except ValueError:
                    witness["raw_invalid"] = True
            else:
                test_early.append(order["early_count"])
    _need(all(witness.values()), f"missing training counterexample: {witness}")
    _need(all(phases == {0, 1} for phases in corner_phases),
          "training does not distinguish every other corner with either phase")
    _need(set(train_early) == {1} and min(test_early) >= 3,
          "missing established one-repeat to distributed-repeat shift")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """A bounded 42-seed diversity audit, legal counts, and negative checks."""
    fingerprints, shapes, normalized = set(), set(), set()
    sizes, counts, turns, degrees, multiplicities, palettes, shifts = (set() for _ in range(7))
    pairs, raw_invalid, boundary_extended, near_matches = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _fingerprint(puzzle)
        _need(signature == _fingerprint(generate_puzzle(seed)), "non-reproducible seed")
        fingerprints.add(signature)
        training_sizes = [np.asarray(p["input"]).shape[0] for p in puzzle["train"]]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, canonical, field, divider, info = _read_input(pair["input"])
                n = canonical.shape[0]
                target = np.asarray(pair["output"])
                before = source.copy()
                _need(np.array_equal(solve(source), target), "input-only solve disagrees")
                _need(np.array_equal(source, before), "solver changed input")
                shapes.add(source.shape)
                normalized.add((canonical.shape, (canonical == divider).tobytes()))
                sizes.add(n)
                counts.add(len(info["paths"]))
                turns.update(info["turns"])
                degrees.add(info["max_degree"])
                palettes.add((field, divider))
                multiplicities.update(_order_evidence(source, canonical)["multiplicities"])
                boundary_extended += any(c > 2 for c in info["boundary_counts"])
                keys = _column_keys(canonical)
                near_matches += any(sum(a != b for a, b in zip(x, y)) == 1
                                    for i, x in enumerate(keys) for y in keys[i + 1:])
                try:
                    _analyze_walls(source == divider)
                except ValueError:
                    raw_invalid += 1
                if split == "test":
                    shifts.add("larger" if n > max(training_sizes) else
                               ("smaller" if n < min(training_sizes) else "overlap"))
                pairs += 1
    _need(len(fingerprints) == 42 and len(normalized) >= 100, "insufficient structural diversity")
    _need(len(shapes) >= 25 and sizes == set(range(7, 16)), "insufficient size coverage")
    _need({1, 2, 3, 4} <= counts and max(degrees) >= 3, "missing divider/graph variation")
    _need(len(turns) >= 5 and max(turns) >= 8, "insufficient path variation")
    _need({2, 3, 4} <= multiplicities and len(palettes) >= 30, "insufficient redundancy/palette variation")
    _need(shifts == {"larger", "smaller", "overlap"}, "missing size-shift regimes")
    _need(boundary_extended > 0 and near_matches > 0, "missing boundary/near-match cases")
    legal_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            _need(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
            legal_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None), (None, 1), (4, "1")):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][-1, 0] = ODD
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted the wrong anchor")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "canonical_wall_structures": len(normalized), "input_shapes": len(shapes),
            "square_sizes": sorted(sizes), "divider_counts": sorted(counts),
            "turn_counts": sorted(turns), "region_max_degrees": sorted(degrees),
            "multiplicities": sorted(multiplicities), "input_role_pairs": len(palettes),
            "size_shift_regimes": sorted(shifts), "raw_topology_invalid_pairs": raw_invalid,
            "boundary_extended_pairs": boundary_extended, "near_match_pairs": near_matches,
            "legal_count_calls": legal_calls, "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
