"""Four inward-facing colored objects compete by cell area."""

from itertools import permutations

import numpy as np


ROLES = ("top", "right", "bottom", "left")
SOURCE_INPUTS = {
    "".join(s.split()) for s in (
        """8888888888888888 8888888888888888 8888884448888888 8888884448888888
        2288884448888888 8228884448888888 8822884448888888 8882288488988888
        8888228889888888 8882288788988888 8882888788888888 8888887878888888
        8888888888888888 8888888888888888 8888888888888888 8888888888888888""",
        """8888888888888888 8888888888888888 8888888888888888 8888888888888888
        8888888488888888 8888884448888888 8886884448878888 8886688488778888
        8886668887778888 8886688188778888 8888881188878888 8888888188888888
        8888888888888888 8888888888888888 8888888888888888 8888888888888888""",
        """8888888888888888 8888888888888888 8888888888888888 8883333333338888
        8888333333388888 8818833333888888 8811888388888888 8811188888558888
        8811118885558888 8811188088558888 8818888088888888 8888808080888888
        8888808080888888 8888808080888888 8888800000888888 8888888888888888""",
        """8888888888888888 8888888888888888 8888899999888888 8889999999988888
        8888899999882228 8888888988882888 8886688888222888 8666668882222888
        8886688888222888 8888888188882888 8888111111182228 8888888188888888
        8888888888888888 8888888888888888 8888888888888888 8888888888888888""",
    )
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count_arg(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count_arg("num_train", num_train, 2, 10)
    test = _count_arg("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _connected8(cells):
    todo = [next(iter(cells))]
    seen = set(todo)
    while todo:
        d, x = todo.pop()
        for dd in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (d + dd, x + dx)
                if q in cells and q not in seen:
                    seen.add(q)
                    todo.append(q)
    return len(seen) == len(cells)


def _separated(groups):
    owner = {point: i for i, group in enumerate(groups) for point in group}
    if len(owner) != sum(map(len, groups)):
        return False
    for (r, c), i in owner.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                other = owner.get((r+dr, c+dc))
                if other is not None and other != i:
                    return False
    return True


def _shape(rng):
    """Return a tip-first, eight-connected silhouette in radial coordinates."""
    style = ("solid", "bent", "irregular", "hollow", "pronged", "diagonal")[
        int(rng.integers(0, 6))
    ]
    length = int(rng.integers(3, 9))
    cells = {(0, 0)}
    drift = 0
    for depth in range(1, length):
        if style == "diagonal":
            drift += int(rng.choice((-1, 1))) if depth == 1 else int(rng.choice((0, 1))) * (1 if drift >= 0 else -1)
            drift = max(-3, min(3, drift))
            width = int(rng.integers(1, 3))
        else:
            if style in ("bent", "irregular", "hollow", "pronged"):
                drift = max(-2, min(2, drift + int(rng.integers(-1, 2))))
            if style == "solid":
                width = int(rng.choice((1, 3) if depth == 1 else (1, 3, 5, 7)))
            else:
                width = int(rng.integers(2, min(7, 2 * depth + 3) + 1))
        left = drift - (width - 1) // 2
        left = max(-4, min(4 - width + 1, left))
        offsets = list(range(left, left + width))
        if style == "hollow" and depth >= 2 and width >= 4:
            offsets.remove(offsets[len(offsets) // 2])
        elif style == "pronged" and depth >= length - 2 and width >= 4:
            offsets = [x for x in offsets if x not in (0,)]
        elif style == "irregular" and depth >= 2 and width >= 4 and rng.random() < 0.45:
            offsets.pop(int(rng.integers(1, len(offsets) - 1)))
        cells.update((depth, x) for x in offsets)
    if not _connected8(cells):
        return None
    rear = [x for d, x in cells if d == length - 1]
    if not any(abs(x) <= 2 for x in rear):
        return None
    span = max(x for _, x in cells) - min(x for _, x in cells) + 1
    return {
        "cells": frozenset(cells), "area": len(cells), "length": length,
        "span": span, "box": length * span, "style": style,
    }


def _pool(rng):
    shapes = []
    for _ in range(170):
        shape = _shape(rng)
        if shape is not None and 5 <= shape["area"] <= 30:
            shapes.append(shape)
    return shapes


def _choose_shapes(rng, pool, winner, regime):
    choices = [s for s in pool if 11 <= s["area"] <= 27]
    if not choices:
        return None
    result = {winner: choices[int(rng.integers(len(choices)))]}
    large = result[winner]
    challenge = rng.random() < (0.53 if regime == "target_test" else 0.35)
    close = rng.random() < (0.55 if regime == "target_test" else 0.38)
    for i, role in enumerate(r for r in ROLES if r != winner):
        available = [s for s in pool if s["area"] < large["area"]]
        if challenge and i == 0:
            special = [s for s in available if s["box"] > large["box"]]
            if special:
                available = special
        if close and i == 1:
            special = [s for s in available if large["area"] - s["area"] <= 2]
            if special:
                available = special
        if not available:
            return None
        result[role] = available[int(rng.integers(len(available)))]
    return result


def _absolute_cells(role, shape, gap):
    if role == "top":
        return {(-gap - 1 - d, x) for d, x in shape["cells"]}
    if role == "bottom":
        return {(gap + 1 + d, x) for d, x in shape["cells"]}
    if role == "left":
        return {(x, -gap - 1 - d) for d, x in shape["cells"]}
    return {(x, gap + 1 + d) for d, x in shape["cells"]}


def _rear_bar(role, cells):
    if role == "top":
        row = min(r for r, _ in cells) - 1
        return {(row, c) for c in (-1, 0, 1)}
    if role == "bottom":
        row = max(r for r, _ in cells) + 1
        return {(row, c) for c in (-1, 0, 1)}
    if role == "left":
        col = min(c for _, c in cells) - 1
        return {(r, col) for r in (-1, 0, 1)}
    col = max(c for _, c in cells) + 1
    return {(r, col) for r in (-1, 0, 1)}


def _make_pair(rng, winner, regime):
    for _ in range(18):
        pool = _pool(rng)
        shapes = _choose_shapes(rng, pool, winner, regime)
        if shapes is None:
            continue
        for _ in range(35):
            gaps = {role: int(rng.integers(1, 5)) for role in ROLES}
            cells = {role: _absolute_cells(role, shapes[role], gaps[role]) for role in ROLES}
            if not _separated(list(cells.values())):
                continue
            bar = _rear_bar(winner, cells[winner])
            if bar & set().union(*cells.values()):
                continue
            all_cells = set().union(*cells.values(), bar)
            min_r = min(r for r, _ in all_cells)
            max_r = max(r for r, _ in all_cells)
            min_c = min(c for _, c in all_cells)
            max_c = max(c for _, c in all_cells)
            margins = [int(rng.integers(0, 5)) for _ in range(4)]
            h = max_r - min_r + 1 + margins[0] + margins[1]
            w = max_c - min_c + 1 + margins[2] + margins[3]
            if not (12 <= h <= 30 and 12 <= w <= 30):
                continue
            palette = [int(x) for x in rng.choice(10, size=5, replace=False)]
            background = palette[0]
            colors = {role: palette[i + 1] for i, role in enumerate(ROLES)}
            dr, dc = margins[0] - min_r, margins[2] - min_c
            inp = np.full((h, w), background, dtype=np.int8)
            for role in ROLES:
                for r, c in cells[role]:
                    inp[r + dr, c + dc] = colors[role]
            out = inp.copy()
            for role in ROLES:
                if role != winner:
                    if role == "top":
                        tip = (-gaps[role] - 1, 0)
                    elif role == "bottom":
                        tip = (gaps[role] + 1, 0)
                    elif role == "left":
                        tip = (0, -gaps[role] - 1)
                    else:
                        tip = (0, gaps[role] + 1)
                    out[tip[0] + dr, tip[1] + dc] = background
            for r, c in bar:
                out[r + dr, c + dc] = colors[winner]
            if _is_source(inp):
                continue
            try:
                analysis = _analyze(inp)
            except AssertionError:
                continue
            if analysis["winner"] != winner or not np.array_equal(_solve(inp, analysis), out):
                continue
            return {"input": inp, "output": out}
    raise ValueError("could not construct a valid four-object pair")


def _is_source(grid):
    return grid.shape == (16, 16) and "".join(str(int(x)) for x in grid.flat) in SOURCE_INPUTS


def _key(grid):
    return grid.shape, grid.tobytes()


def _analyze(grid):
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 5, "need a background and four colors")
    background = int(values[int(np.argmax(counts))])
    colors = [int(x) for x in values if x != background]
    objects = {c: {tuple(map(int, rc)) for rc in np.argwhere(grid == c)} for c in colors}
    _require(all(_connected8(points) for points in objects.values()), "object is not eight-connected")
    _require(_separated(list(objects.values())), "different objects touch")
    solutions = []
    for assignment in permutations(colors):
        by_role = dict(zip(ROLES, assignment))
        tips = {}
        for role in ROLES:
            points = objects[by_role[role]]
            if role == "top":
                extreme = max(r for r, _ in points)
                candidates = [p for p in points if p[0] == extreme]
            elif role == "bottom":
                extreme = min(r for r, _ in points)
                candidates = [p for p in points if p[0] == extreme]
            elif role == "left":
                extreme = max(c for _, c in points)
                candidates = [p for p in points if p[1] == extreme]
            else:
                extreme = min(c for _, c in points)
                candidates = [p for p in points if p[1] == extreme]
            if len(candidates) != 1:
                break
            tips[role] = candidates[0]
        if len(tips) != 4:
            continue
        if tips["top"][1] != tips["bottom"][1] or tips["left"][0] != tips["right"][0]:
            continue
        center = tips["left"][0], tips["top"][1]
        cr, cc = center
        if not (tips["top"][0] < cr < tips["bottom"][0] and tips["left"][1] < cc < tips["right"][1]):
            continue
        if not (all(r < cr for r, _ in objects[by_role["top"]])
                and all(r > cr for r, _ in objects[by_role["bottom"]])
                and all(c < cc for _, c in objects[by_role["left"]])
                and all(c > cc for _, c in objects[by_role["right"]])):
            continue
        solutions.append((by_role, tips, center))
    _require(len(solutions) == 1, "cardinal roles or inward tips are ambiguous")
    by_role, tips, center = solutions[0]
    areas = {role: len(objects[by_role[role]]) for role in ROLES}
    maximum = max(areas.values())
    winners = [role for role in ROLES if areas[role] == maximum]
    _require(len(winners) == 1, "the largest object must be unique")
    winner = winners[0]
    cr, cc = center
    points = objects[by_role[winner]]
    if winner == "top":
        rear = min(r for r, _ in points) - 1
        bar = {(rear, cc + delta) for delta in (-1, 0, 1)}
    elif winner == "bottom":
        rear = max(r for r, _ in points) + 1
        bar = {(rear, cc + delta) for delta in (-1, 0, 1)}
    elif winner == "left":
        rear = min(c for _, c in points) - 1
        bar = {(cr + delta, rear) for delta in (-1, 0, 1)}
    else:
        rear = max(c for _, c in points) + 1
        bar = {(cr + delta, rear) for delta in (-1, 0, 1)}
    h, w = grid.shape
    _require(all(0 <= r < h and 0 <= c < w and int(grid[r, c]) == background for r, c in bar), "rear strip is blocked")
    _require(any(max(abs(r-u), abs(c-v)) == 1 for r, c in bar for u, v in points),
             "rear strip must touch the winner, possibly diagonally")
    return {"background": background, "colors": by_role, "objects": objects,
            "tips": tips, "center": center, "areas": areas, "winner": winner, "bar": bar}


def _solve(grid, analysis):
    out = grid.copy()
    for role in ROLES:
        if role != analysis["winner"]:
            out[analysis["tips"][role]] = analysis["background"]
    for cell in analysis["bar"]:
        out[cell] = analysis["colors"][analysis["winner"]]
    return out


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    # The transformation has no hidden color or geometry convention. Two
    # perpendicular training winners demonstrate rotation even at count 2.
    vertical = ("top", "bottom")[int(rng.integers(2))]
    horizontal = ("left", "right")[int(rng.integers(2))]
    train_winners = [vertical, horizontal] + [ROLES[int(rng.integers(4))] for _ in range(train_count - 2)]
    rng.shuffle(train_winners)
    test_winners = [ROLES[int(rng.integers(4))] for _ in range(test_count)]
    familiar = sorted(set(train_winners))
    unseen = [role for role in ROLES if role not in familiar]
    if unseen and rng.random() < 0.65:
        test_winners[0] = unseen[int(rng.integers(len(unseen)))]
        if test_count > 1:
            test_winners[1] = familiar[int(rng.integers(len(familiar)))]
    else:
        test_winners[0] = familiar[int(rng.integers(len(familiar)))]
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, winners in (("train", train_winners), ("test", test_winners)):
        regime = "source_train" if section == "train" else "target_test"
        for winner in winners:
            for _ in range(12):
                pair = _make_pair(rng, winner, regime)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise ValueError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer) and bool(np.all((0 <= grid) & (grid <= 9))), "grid colors")
            _require(inp.shape == out.shape, "grid size changed")
            ki, ko = _key(inp), _key(out)
            _require(ki not in inputs and (ki, ko) not in pairs, "duplicate pair")
            inputs.add(ki)
            pairs.add((ki, ko))
            _require(not _is_source(inp), "source pair copied")
            analysis = _analyze(inp)
            _require(np.array_equal(_solve(inp, analysis), out), "incorrect six-cell transfer")
            _require(int(np.count_nonzero(inp != out)) == 6, "six cells must change")
            for role in ROLES:
                color = analysis["colors"][role]
                delta = int(np.count_nonzero(out == color) - np.count_nonzero(inp == color))
                _require(delta == (3 if role == analysis["winner"] else -1), "area transfer")
            analyses[section].append(analysis)
    train_winners = {a["winner"] for a in analyses["train"]}
    _require(any(x in train_winners for x in ("top", "bottom")) and
             any(x in train_winners for x in ("left", "right")), "training must show both axes")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def _four_components(points):
    remaining = set(points)
    count = 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            r, c = pending.pop()
            for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
    return count


def _review_features(grid, analysis):
    cr, cc = analysis["center"]
    tips = analysis["tips"]
    gaps = (cr-tips["top"][0]-1, tips["right"][1]-cc-1,
            tips["bottom"][0]-cr-1, cc-tips["left"][1]-1)
    shapes = {}
    for role in ROLES:
        points = analysis["objects"][analysis["colors"][role]]
        tip = tips[role]
        if role == "top":
            radial = {(tip[0]-r, c-cc) for r, c in points}
        elif role == "bottom":
            radial = {(r-tip[0], c-cc) for r, c in points}
        elif role == "left":
            radial = {(tip[1]-c, r-cr) for r, c in points}
        else:
            radial = {(c-tip[1], r-cr) for r, c in points}
        length = 1 + max(d for d, _ in radial)
        slices = [sorted(x for d, x in radial if d == depth) for depth in range(length)]
        width = 1 + max(x for _, x in radial) - min(x for _, x in radial)
        shapes[role] = {
            "length": length, "width": width, "box": length * width,
            "split": any(s[-1] - s[0] + 1 > len(s) for s in slices),
            "regular": all(s == list(range(-s[-1], s[-1]+1)) for s in slices),
            "diagonal": _four_components(points) > 1,
        }
    winner = analysis["winner"]
    rivals = [shapes[role] for role in ROLES if role != winner]
    winner_shape = shapes[winner]
    areas = sorted(analysis["areas"].values(), reverse=True)
    edge_loser = any(
        r in (0, grid.shape[0]-1) or c in (0, grid.shape[1]-1)
        for role in ROLES if role != winner
        for r, c in analysis["objects"][analysis["colors"][role]]
    )
    winner_points = analysis["objects"][analysis["colors"][winner]]
    diagonal_bar = not any(abs(r-u)+abs(c-v) == 1
                           for r, c in analysis["bar"] for u, v in winner_points)
    return {
        "gaps": gaps,
        "close": areas[0] - areas[1] <= 2,
        "box_counter": any(s["box"] > winner_shape["box"] for s in rivals),
        "length_counter": any(s["length"] > winner_shape["length"] for s in rivals),
        "width_counter": any(s["width"] > winner_shape["width"] for s in rivals),
        "split": any(s["split"] for s in shapes.values()),
        "regular": any(s["regular"] for s in shapes.values()),
        "diagonal": any(s["diagonal"] for s in shapes.values()),
        "diagonal_bar": diagonal_bar,
        "edge_loser": edge_loser,
    }


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity")
    winners = set()
    sizes = set()
    palettes = set()
    gaps = set()
    review_counts = {name: 0 for name in (
        "close", "box_counter", "length_counter", "width_counter",
        "split", "regular", "diagonal", "diagonal_bar", "edge_loser")}
    puzzles_with_color_changes = 0
    puzzles_with_gap_changes = 0
    novel_test_sides = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_analyses = []
        puzzle_gaps = set()
        train_sides = set()
        test_sides = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                a = _analyze(pair["input"])
                f = _review_features(pair["input"], a)
                for name in review_counts:
                    review_counts[name] += int(f[name])
                puzzle_analyses.append(a)
                puzzle_gaps.add(f["gaps"])
                (train_sides if section == "train" else test_sides).add(a["winner"])
                winners.add(a["winner"])
                sizes.add(pair["input"].shape)
                palettes.add((a["background"], *(a["colors"][r] for r in ROLES)))
                gaps.add(f["gaps"])
        puzzles_with_color_changes += int(
            len({a["background"] for a in puzzle_analyses}) > 1 and
            all(len({a["colors"][role] for a in puzzle_analyses}) > 1 for role in ROLES))
        puzzles_with_gap_changes += int(len(puzzle_gaps) > 1)
        novel_test_sides += int(bool(test_sides - train_sides))
    _require(winners == set(ROLES), "winner rotation diversity")
    _require(len(sizes) >= 25 and len(palettes) >= 100 and len(gaps) >= 50, "structural diversity")
    _require(puzzles_with_color_changes == 42 and puzzles_with_gap_changes == 42,
             "pair-level color and gap variation")
    _require(novel_test_sides >= 5, "test orientation changes")
    for name, minimum in (("close", 40), ("box_counter", 25),
                          ("length_counter", 25), ("width_counter", 20),
                          ("split", 50), ("regular", 50), ("diagonal", 50),
                          ("diagonal_bar", 1), ("edge_loser", 20)):
        _require(review_counts[name] >= minimum, f"review diversity: {name}")
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(20000 + i, *counts)
        _require((len(p["train"]), len(p["test"])) == counts, "explicit counts")
    for i, kwargs in enumerate(({"num_test": 1}, {"num_test": 4}, {"num_train": 2}, {"num_train": 10})):
        validate_generated_puzzle(generate_puzzle(21000 + i, **kwargs))
    for i, kwargs in enumerate(({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                                {"num_test": 5}, {"num_train": 2, "num_test": 2},
                                {"num_train": 3, "num_test": 4})):
        try:
            generate_puzzle(22000 + i, **kwargs)
        except ValueError:
            continue
        raise AssertionError("invalid pair count accepted")
    _require(_puzzle_key(generate_puzzle(12345, 7, 3)) == _puzzle_key(generate_puzzle(12345, 7, 3)), "explicit determinism")
    _require(_puzzle_key(generate_puzzle(54321)) == _puzzle_key(generate_puzzle(54321)), "automatic determinism")
