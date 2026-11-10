"""Connected shapes are recolored exactly when they have a mirror axis."""

import numpy as np


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _check(value, message):
    if not value:
        raise AssertionError(message)


def _normal(cells):
    cells = set(cells)
    low_r = min(r for r, _ in cells)
    low_c = min(c for _, c in cells)
    return frozenset((r - low_r, c - low_c) for r, c in cells)


def _orient(cells, code):
    result = []
    for r, c in cells:
        if code >= 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    return _normal(result)


def _free_key(cells):
    return min(tuple(sorted(_orient(cells, k))) for k in range(8))


def _axes(cells):
    shape = _normal(cells)
    h = max(r for r, _ in shape)
    w = max(c for _, c in shape)
    answer = []
    if shape == frozenset((h - r, c) for r, c in shape):
        answer.append("horizontal")
    if shape == frozenset((r, w - c) for r, c in shape):
        answer.append("vertical")
    if h == w:
        if shape == frozenset((c, r) for r, c in shape):
            answer.append("diagonal")
        if shape == frozenset((w - c, h - r) for r, c in shape):
            answer.append("antidiagonal")
    return tuple(answer)


def _half_turn(cells):
    shape = _normal(cells)
    h = max(r for r, _ in shape)
    w = max(c for _, c in shape)
    return shape == frozenset((h - r, w - c) for r, c in shape)


def _components(mask):
    height, width = mask.shape
    visited = set()
    answer = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or (r, c) in visited:
                continue
            stack = [(r, c)]
            visited.add((r, c))
            cells = set()
            while stack:
                rr, cc = stack.pop()
                cells.add((rr, cc))
                for dr, dc in DIRS:
                    q = (rr + dr, cc + dc)
                    if (0 <= q[0] < height and 0 <= q[1] < width
                            and mask[q] and q not in visited):
                        visited.add(q)
                        stack.append(q)
            answer.append(frozenset(cells))
    return answer


def _distance(cell, shape):
    r, c = cell
    return min(abs(r - rr) + abs(c - cc) for rr, cc in shape)


def _roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    _check(len(colors) >= 3, "background, neutral, and a label are required")
    order = np.argsort(-counts)
    _check(counts[order[0]] > counts[order[1]], "background is not dominant")
    if len(order) > 2:
        _check(counts[order[1]] > counts[order[2]],
               "neutral color is not distinguishable from labels")
    return int(colors[order[0]]), int(colors[order[1]])


def _analyze(grid):
    background, neutral = _roles(grid)
    shapes = _components(grid == neutral)
    _check(bool(shapes), "a candidate is required")
    _check(all(len(s) >= 2 for s in shapes), "candidates have at least two cells")
    labels = []
    for r, c in np.argwhere((grid != background) & (grid != neutral)):
        labels.append((int(r), int(c), int(grid[r, c])))
    _check(bool(labels), "a label is required")
    owners = {}
    decoys = []
    for r, c, color in labels:
        distances = [_distance((r, c), shape) for shape in shapes]
        nearest = min(distances)
        possible = [i for i, distance in enumerate(distances)
                    if distance == nearest and distance <= 2]
        _check(len(possible) <= 1, "label can select multiple shapes")
        if not possible:
            _check(all(_distance((r, c), shape) >= 4 for shape in shapes),
                   "decoy is too close to a shape")
            decoys.append((r, c, color))
        else:
            i = possible[0]
            _check(i not in owners, "candidate has multiple labels")
            owners[i] = (r, c, color, _distance((r, c), shapes[i]))
    _check(len(owners) == len(shapes), "every candidate needs one label")
    return background, neutral, shapes, owners, decoys


def _solve(grid):
    background, neutral, shapes, owners, decoys = _analyze(grid)
    output = grid.copy()
    for r, c, _ in decoys:
        output[r, c] = background
    for i, shape in enumerate(shapes):
        r, c, color, _ = owners[i]
        output[r, c] = background
        if _axes(shape):
            for rr, cc in shape:
                output[rr, cc] = color
    return output.astype(np.int8, copy=False)


def _make_catalog():
    previous = {frozenset({(0, 0)})}
    result = {"positive": {}, "diagonal": {}, "rotational": {},
              "asymmetric": {}}
    for size in range(2, 9):
        grown = set()
        for shape in previous:
            for r, c in shape:
                for dr, dc in DIRS:
                    cell = (r + dr, c + dc)
                    if cell not in shape:
                        grown.add(frozenset(_free_key(set(shape) | {cell})))
        previous = grown
        for name in result:
            result[name][size] = []
        for shape in sorted(grown, key=lambda s: tuple(sorted(s))):
            axes = _axes(shape)
            if axes:
                result["positive"][size].append(shape)
                if not any(a in axes for a in ("horizontal", "vertical")):
                    result["diagonal"][size].append(shape)
            elif _half_turn(shape):
                result["rotational"][size].append(shape)
            else:
                result["asymmetric"][size].append(shape)
    return result


CATALOG = _make_catalog()


def _shape(rng, kind):
    if kind == "bar":
        length = int(rng.integers(2, 7))
        cells = frozenset((0, c) for c in range(length))
    else:
        choices = [(size, values) for size, values in CATALOG[kind].items()
                   if values]
        size, values = choices[int(rng.integers(len(choices)))]
        cells = values[int(rng.integers(len(values)))]
    return _orient(cells, int(rng.integers(8)))


def _motif(rng, kind, distance, contact):
    cells = _shape(rng, kind)
    options = []
    min_r = min(r for r, _ in cells) - 2
    max_r = max(r for r, _ in cells) + 2
    min_c = min(c for _, c in cells) - 2
    max_c = max(c for _, c in cells) + 2
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if (r, c) in cells or _distance((r, c), cells) != distance:
                continue
            if distance == 2 and not any(
                    (r - 2 * dr, c - 2 * dc) in cells for dr, dc in DIRS):
                continue
            if distance == 1 and contact:
                touched = [(r + dr, c + dc) for dr, dc in DIRS
                           if (r + dr, c + dc) in cells]
                degrees = [sum((q[0] + dr, q[1] + dc) in cells
                               for dr, dc in DIRS) for q in touched]
                if contact == "endpoint" and not any(d == 1 for d in degrees):
                    continue
                if contact == "interior" and not any(d >= 2 for d in degrees):
                    continue
            options.append((r, c))
    _check(bool(options), "shape lacks requested label placement")
    label = options[int(rng.integers(len(options)))]
    all_cells = set(cells) | {label}
    low_r = min(r for r, _ in all_cells)
    low_c = min(c for _, c in all_cells)
    return {
        "cells": frozenset((r - low_r, c - low_c) for r, c in cells),
        "label": (label[0] - low_r, label[1] - low_c),
        "height": max(r for r, _ in all_cells) - low_r + 1,
        "width": max(c for _, c in all_cells) - low_c + 1,
    }


def _label_colors(rng, accents, count, repeat):
    colors = [int(x) for x in rng.choice(accents, size=count, replace=False)]
    if count >= 2 and repeat:
        colors[1] = colors[0]
    return colors


def _fits(cells, label, placed):
    for other_cells, other_label in placed:
        if min(_distance(cell, other_cells) for cell in cells) <= 1:
            return False
        if _distance(label, other_cells) <= 2:
            return False
        if _distance(other_label, cells) <= 2:
            return False
        if abs(label[0] - other_label[0]) + abs(label[1] - other_label[1]) <= 1:
            return False
    return True


def _render_pair(rng, motifs, colors, decoy_count, roles):
    background, neutral = roles
    occupied_area = sum((m["height"] + 1) * (m["width"] + 1)
                        for m in motifs) + 7 * decoy_count
    density = float(rng.uniform(1.2, 2.9))
    ratio = float(np.exp(rng.uniform(-0.95, 0.95)))
    height = max(max(m["height"] for m in motifs),
                 int(np.ceil(np.sqrt(occupied_area * density * ratio))))
    width = max(max(m["width"] for m in motifs),
                int(np.ceil(occupied_area * density / height)))
    if width > 30 or height > 30:
        height = min(height, 30)
        width = min(width, 30)
    for growth in range(12):
        h = min(30, height + growth)
        w = min(30, width + growth)
        if any(m["height"] > h or m["width"] > w for m in motifs):
            continue
        for _ in range(45):
            placed = []
            success = True
            for motif in motifs:
                found = False
                for _ in range(160):
                    top = int(rng.integers(h - motif["height"] + 1))
                    left = int(rng.integers(w - motif["width"] + 1))
                    cells = frozenset((top + r, left + c)
                                      for r, c in motif["cells"])
                    label = (top + motif["label"][0],
                             left + motif["label"][1])
                    if _fits(cells, label, placed):
                        placed.append((cells, label))
                        found = True
                        break
                if not found:
                    success = False
                    break
            if not success:
                continue
            grid = np.full((h, w), background, dtype=np.int8)
            all_cells = set()
            labels = []
            for (cells, label), color in zip(placed, colors):
                for r, c in cells:
                    grid[r, c] = neutral
                grid[label] = color
                all_cells.update(cells)
                labels.append(label)
            decoy_places = [(r, c) for r in range(h) for c in range(w)
                            if grid[r, c] == background
                            and (not all_cells or _distance((r, c), all_cells) >= 4)
                            and all(abs(r - a) + abs(c - b) >= 2
                                    for a, b in labels)]
            if len(decoy_places) < decoy_count:
                continue
            if decoy_count:
                order = rng.permutation(len(decoy_places))
                chosen = []
                for index in order:
                    position = decoy_places[int(index)]
                    if all(abs(position[0] - r) + abs(position[1] - c) >= 2
                           for r, c in chosen):
                        chosen.append(position)
                        if len(chosen) == decoy_count:
                            break
                if len(chosen) < decoy_count:
                    continue
                for (r, c), color in zip(chosen, colors[len(motifs):]):
                    grid[r, c] = color
                    labels.append((r, c))
            try:
                _roles(grid)
                output = _solve(grid)
            except AssertionError:
                continue
            return grid, output
    raise ValueError("could not arrange distinct labeled candidates")


def _specifications(rng, split, index):
    if split == "train" and index == 0:
        count = int(rng.integers(1, 4))
        return [("bar", 1, "endpoint")] + [
            ("positive", 1, "interior" if i == 0 else None)
            for i in range(count - 1)]
    if split == "train" and index == 1:
        result = [("diagonal", 2, None), ("rotational", 2, None)]
        if int(rng.integers(2)):
            result.append(("positive", 1, "interior"))
        return result
    cap = 4 if split == "train" else 5
    count = int(rng.integers(1, cap + 1))
    positive_count = int(rng.integers(1, count + 1))
    kinds = ["positive"] * positive_count + [
        "rotational" if int(rng.integers(2)) else "asymmetric"
        for _ in range(count - positive_count)]
    rng.shuffle(kinds)
    return [(kind, 1 if int(rng.integers(3)) else 2,
             None if int(rng.integers(3)) else
             ("endpoint" if int(rng.integers(2)) else "interior"))
            for kind in kinds]


def _counts(rng, seed, num_train, num_test):
    for value, name, low, high in (
        (num_train, "num_train", 2, 10),
        (num_test, "num_test", 1, 4),
    ):
        if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(name + " is outside its legal range")
    if num_train is None and num_test is None:
        tests = 1 + seed % 4
        trains = max(2, tests + 1) + (seed // 4) % (
            11 - max(2, tests + 1))
    elif num_train is None:
        tests = int(num_test)
        trains = int(rng.integers(max(2, tests + 1), 11))
    elif num_test is None:
        trains = int(num_train)
        tests = int(rng.integers(1, min(4, trains - 1) + 1))
    else:
        trains, tests = int(num_train), int(num_test)
    if trains <= tests:
        raise ValueError("num_train must exceed num_test")
    return trains, tests


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_pairs():
    rows = (
        (("76637", "77747", "77767", "77666", "77767"),
         ("73377", "77777", "77747", "77444", "77747")),
        (("7776", "3766", "7767", "7777", "1666", "7767"),
         ("7776", "7766", "7767", "7777", "7111", "7717")),
        (("4767775", "6667777", "7767177", "7777667", "7777667",
          "6687777", "7677777"),
         ("7747777", "4447777", "7747777", "7777117", "7777117",
          "8877777", "7877777")),
    )
    return tuple((np.array([[int(c) for c in row] for row in a],
                           dtype=np.int8),
                  np.array([[int(c) for c in row] for row in b],
                           dtype=np.int8)) for a, b in rows)


SOURCE_PAIRS = _source_pairs()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, int(seed), num_train, num_test)
    role_palette = [int(x) for x in rng.permutation(10)]
    fixed_roles = tuple(role_palette[:2])
    variable_roles = bool(int(rng.integers(3)) == 0)
    allow_decoys = bool(int(rng.integers(3)) != 0)
    allow_repeats = bool(int(rng.integers(2)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(16):
                if variable_roles and (split != "train" or index != 0):
                    colors = [int(x) for x in rng.permutation(10)]
                    roles = tuple(colors[:2])
                    if split == "train" and index == 1 and roles == fixed_roles:
                        roles = (roles[1], roles[0])
                else:
                    roles = fixed_roles
                accents = [c for c in range(10) if c not in roles]
                specs = _specifications(rng, split, index)
                decoy_count = (1 if allow_decoys else 0) if (
                    split == "train" and index == 1) else (
                    int(rng.integers(3)) if allow_decoys and
                    (split == "test" or index >= 2) else 0)
                colors = _label_colors(
                    rng, accents, len(specs) + decoy_count,
                    allow_repeats and (split == "train" and index == 1
                                       or int(rng.integers(3)) == 0),
                )
                try:
                    motifs = [_motif(rng, *spec) for spec in specs]
                    inp, out = _render_pair(
                        rng, motifs, colors, decoy_count, roles)
                except (AssertionError, ValueError):
                    continue
                if _key(inp) not in seen and not any(
                        np.array_equal(inp, a) and np.array_equal(out, b)
                        for a, b in SOURCE_PAIRS):
                    seen.add(_key(inp))
                    puzzle[split].append({"input": inp, "output": out})
                    break
            else:
                raise ValueError("could not generate a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle must have train and test")
    _check(isinstance(puzzle["train"], list) and
           isinstance(puzzle["test"], list), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne,
           "illegal train/test counts")
    inputs, pairs = set(), set()
    details = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair needs input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "grid must be a two-dimensional array")
                _check(1 <= grid.shape[0] <= 30 and
                       1 <= grid.shape[1] <= 30,
                       "grid dimensions outside 1..30")
                _check(np.issubdtype(grid.dtype, np.integer) and
                       bool(np.all((grid >= 0) & (grid <= 9))),
                       "grid colors must be ARC integers")
            _check(inp.shape == out.shape, "shape must be preserved")
            key = (_key(inp), _key(out))
            _check(key[0] not in inputs and key not in pairs,
                   "duplicate pair or input")
            inputs.add(key[0])
            pairs.add(key)
            _check(not any(np.array_equal(inp, a) and np.array_equal(out, b)
                           for a, b in SOURCE_PAIRS), "source pair copy")
            _check(np.array_equal(_solve(inp), out),
                   "output does not follow complete-shape reflection")
            _check(not np.array_equal(inp, out), "pair has no change")
            details[split].append(_analyze(inp))
    train_roles = {(d[0], d[1]) for d in details["train"]}
    test_roles = {(d[0], d[1]) for d in details["test"]}
    if len(train_roles) == 1:
        _check(test_roles == train_roles,
               "test changes color roles without training evidence")
    train_objects = [(shape, d[3][i]) for d in details["train"]
                     for i, shape in enumerate(d[2])]
    _check(any(_axes(shape) for shape, _ in train_objects),
           "training needs a positive")
    _check(any(not _axes(shape) and _half_turn(shape)
               for shape, _ in train_objects),
           "training needs a rotational-only negative")
    _check(any(_axes(shape) and label[3] == 2
               for shape, label in train_objects),
           "training needs a separated positive label")
    _check(any(label[3] == 1 and any(
        sum((r + dr, c + dc) in shape for dr, dc in DIRS) == 1
        for r, c in shape if abs(r - label[0]) + abs(c - label[1]) == 1)
        for shape, label in train_objects),
        "training needs an endpoint label")
    if any(d[4] for d in details["test"]):
        _check(any(d[4] for d in details["train"]),
               "training must demonstrate unused labels")
    if any(len({label[2] for label in d[3].values()}) <
           len(d[3]) for d in details["test"]):
        _check(any(len({label[2] for label in d[3].values()}) <
                   len(d[3]) for d in details["train"]),
               "training must demonstrate repeated label colors")


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and
        np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    for source_input, source_output in SOURCE_PAIRS:
        _check(np.array_equal(_solve(source_input), source_output),
               "source regression failed")
        before = set(map(tuple, np.argwhere(source_input == 6)))
        after = set(map(tuple, np.argwhere(source_output != 7)))
        if source_input.shape == (5, 5):
            _check(before <= after, "source candidate tip was erased")
    counts_train, counts_test = set(), set()
    grid_sizes, free_shapes, object_counts, role_sets = set(), set(), set(), set()
    pair_keys, layout_keys = set(), set()
    saw = {"single": False, "all_positive": False,
           "several_negative": False, "diagonal": False,
           "decoy": False, "repeat": False, "compact": False,
           "spacious": False, "variable_roles": False}
    measured = {"count_varying_puzzles": 0, "dimension_varying_puzzles": 0,
                "role_varying_puzzles": 0, "direct_and_gap_pairs": 0,
                "interior_contact_pairs": 0, "rotational_negative_pairs": 0,
                "asymmetric_negative_pairs": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        roles = set()
        counts_in_puzzle = set()
        dimensions_in_puzzle = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                _check(np.array_equal(_solve(inp), out),
                       "independent input reconstruction failed")
                background, neutral, shapes, owners, decoys = _analyze(inp)
                roles.add((background, neutral))
                counts_in_puzzle.add(len(shapes))
                dimensions_in_puzzle.add(inp.shape)
                role_sets.add((background, neutral))
                pair_keys.add(_key(inp))
                grid_sizes.add(inp.shape)
                object_counts.add(len(shapes))
                negatives = sum(not _axes(s) for s in shapes)
                saw["single"] |= len(shapes) == 1
                saw["all_positive"] |= negatives == 0 and len(shapes) >= 2
                saw["several_negative"] |= negatives >= 2
                saw["diagonal"] |= any(
                    _axes(s) and not any(a in _axes(s)
                    for a in ("horizontal", "vertical")) for s in shapes)
                saw["decoy"] |= bool(decoys)
                saw["repeat"] |= len({v[2] for v in owners.values()}) < len(owners)
                saw["compact"] |= inp.size <= 100
                saw["spacious"] |= inp.size >= 250
                distances = [owner[3] for owner in owners.values()]
                measured["direct_and_gap_pairs"] += (
                    1 in distances and 2 in distances)
                measured["rotational_negative_pairs"] += any(
                    not _axes(s) and _half_turn(s) for s in shapes)
                measured["asymmetric_negative_pairs"] += any(
                    not _axes(s) and not _half_turn(s) for s in shapes)
                measured["interior_contact_pairs"] += any(
                    owner[3] == 1 and any(
                        sum((r + dr, c + dc) in s for dr, dc in DIRS) >= 2
                        for r, c in s
                        if abs(r - owner[0]) + abs(c - owner[1]) == 1)
                    for i, s in enumerate(shapes)
                    for owner in (owners[i],))
                for shape in shapes:
                    free_shapes.add(_free_key(shape))
                layout_keys.add((inp.shape, tuple(sorted(
                    (min(r for r, _ in s), min(c for _, c in s),
                     max(r for r, _ in s), max(c for _, c in s))
                    for s in shapes))))
        saw["variable_roles"] |= len(roles) > 1
        measured["count_varying_puzzles"] += len(counts_in_puzzle) > 1
        measured["dimension_varying_puzzles"] += len(dimensions_in_puzzle) > 1
        measured["role_varying_puzzles"] += len(roles) > 1
    _check(len(counts_train) >= 4 and len(counts_test) >= 3,
           "automatic count diversity is too small")
    _check(len(pair_keys) >= 200 and len(grid_sizes) >= 30 and
           len(free_shapes) >= 40 and len(layout_keys) >= 180 and
           len(role_sets) >= 15 and len(object_counts) >= 4,
           "structural diversity is too small")
    _check(all(saw.values()), "review coverage is incomplete")
    _check(measured["count_varying_puzzles"] >= 30 and
           measured["dimension_varying_puzzles"] >= 40 and
           measured["role_varying_puzzles"] >= 8 and
           measured["direct_and_gap_pairs"] >= 40 and
           measured["interior_contact_pairs"] >= 30 and
           measured["rotational_negative_pairs"] >= 30 and
           measured["asymmetric_negative_pairs"] >= 10,
           "review-requested measured variation is too small")
    for seed, nt, ne in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
                         (1100, None, 1), (1101, None, 4),
                         (1102, 2, None), (1103, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        _check((nt is None or len(puzzle["train"]) == nt) and
               (ne is None or len(puzzle["test"]) == ne),
               "requested count was ignored")
        validate_generated_puzzle(puzzle)
    a = generate_puzzle(2027, 7, 3)
    _check(_same_puzzle(a, generate_puzzle(2027, 7, 3)),
           "generation is not deterministic")
    _check(not _same_puzzle(a, generate_puzzle(2028, 7, 3)),
           "different seeds gave the same puzzle")
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count did not raise ValueError")
