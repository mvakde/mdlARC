"""Minimal bilateral repair with two occluding, colored transparencies.

Only NumPy and the standard library are used. The solver reads the frames,
ink colors, and masks from its argument; generation metadata is never needed.
"""

from collections import deque
from itertools import combinations

import numpy as np


BACKGROUND = 0
FRAME = 5
INKS = (1, 2, 3, 4, 6, 7, 8, 9)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _array(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _frames(grid):
    """Read isolated rectangular gray rings, without assuming an array layout."""
    gray = grid == FRAME
    seen = np.zeros(grid.shape, dtype=bool)
    boxes = []
    for r, c in zip(*np.where(gray)):
        if seen[r, c]:
            continue
        queue = deque([(int(r), int(c))])
        seen[r, c] = True
        cells = []
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1]
                        and gray[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        top = min(y for y, _ in cells)
        bottom = max(y for y, _ in cells) + 1
        left = min(x for _, x in cells)
        right = max(x for _, x in cells) + 1
        block = grid[top:bottom, left:right]
        options = []
        for thickness in (1, 2):
            if min(block.shape) <= 2 * thickness:
                continue
            ring = np.ones(block.shape, dtype=bool)
            ring[thickness:-thickness, thickness:-thickness] = False
            if np.array_equal(block == FRAME, ring):
                options.append(thickness)
        if len(options) != 1:
            raise ValueError("a gray component is not a thin or double rectangular frame")
        thickness = options[0]
        interior = block[thickness:-thickness, thickness:-thickness].copy()
        ink = set(int(x) for x in np.unique(interior)) - {BACKGROUND}
        if len(ink) != 1 or FRAME in ink:
            raise ValueError("every framed interior must have one non-gray ink")
        boxes.append({"position": (top, left), "bounds": (top, bottom, left, right),
                      "thickness": thickness, "grid": interior,
                      "mask": interior != BACKGROUND, "color": next(iter(ink))})
    return sorted(boxes, key=lambda b: b["position"])


def _read(input_grid):
    grid = _array(input_grid)
    cards = _frames(grid)
    if not 4 <= len(cards) <= 6:
        raise ValueError("expected one target and three to five stamps")
    targets = [card for card in cards if card["thickness"] == 2]
    if len(targets) != 1:
        raise ValueError("exactly one double frame must identify the target")
    target = targets[0]
    stamps = [card for card in cards if card["thickness"] == 1]
    if len({card["color"] for card in cards}) != len(cards):
        raise ValueError("all card inks must be distinct")
    if target["grid"].shape[1] % 2:
        raise ValueError("the target must have even interior width")
    occupied_boxes = np.zeros(grid.shape, dtype=bool)
    for card in cards:
        if card["grid"].shape != target["grid"].shape:
            raise ValueError("all framed interiors must have the same dimensions")
        t, b, l, r = card["bounds"]
        if np.any(occupied_boxes[t:b, l:r]):
            raise ValueError("frames cannot overlap or contain one another")
        occupied_boxes[t:b, l:r] = True
    if np.any(grid[~occupied_boxes] != BACKGROUND):
        raise ValueError("unframed space must be blank")
    return grid, cards, target, stamps


def _pairs(target_mask, stamp_masks):
    """Strict exact cover in the presence of an already-inked base layer."""
    closure = target_mask | np.fliplr(target_mask)
    debt = closure & ~target_mask
    candidates = []
    for a, b in combinations(range(len(stamp_masks)), 2):
        p, q = stamp_masks[a], stamp_masks[b]
        if (np.array_equal(target_mask | p | q, closure)
                and np.any(debt & p & ~q) and np.any(debt & q & ~p)):
            candidates.append((a, b))
    return candidates


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the repaired target interior, deriving every role from the input."""
    _, _, target, stamps = _read(input_grid)
    candidates = _pairs(target["mask"], [s["mask"] for s in stamps])
    if len(candidates) != 1:
        raise ValueError("the minimal repair must have exactly one nonredundant stamp pair")
    output = target["grid"].copy()
    # Stamps are in top-left reading order. Existing ink occludes both layers;
    # the earlier selected stamp occludes the later one on shared new pixels.
    for index in candidates[0]:
        stamp = stamps[index]
        output[stamp["mask"] & (output == BACKGROUND)] = stamp["color"]
    return output


def _ordered_orbits(rng, height, half, style):
    """A shape-biased order of horizontal reflection orbits, not a fixed stencil."""
    coordinates = [(r, c) for r in range(height) for c in range(half)]
    if style == "scatter":
        rng.shuffle(coordinates)
    elif style == "bars":
        row_order = rng.permutation(height)
        row_rank = {int(r): i for i, r in enumerate(row_order)}
        reverse = bool(rng.integers(2))
        coordinates.sort(key=lambda p: (row_rank[p[0]], -p[1] if reverse else p[1]))
    elif style == "stems":
        col_order = rng.permutation(half)
        col_rank = {int(c): i for i, c in enumerate(col_order)}
        reverse = bool(rng.integers(2))
        coordinates.sort(key=lambda p: (col_rank[p[1]], -p[0] if reverse else p[0]))
    else:
        row, col = int(rng.integers(height)), int(rng.integers(half))
        scores = {p: abs(p[0] - row) + abs(p[1] - col) + float(rng.random()) * 1.8
                  for p in coordinates}
        coordinates.sort(key=lambda p: scores[p])
    return coordinates


def _make_target(rng, height, width, debt_count, mixed_sides):
    half = width // 2
    styles = ("bars", "stems", "patch", "scatter")
    for _ in range(80):
        order = _ordered_orbits(rng, height, half, styles[int(rng.integers(4))])
        matched_count = int(rng.integers(1, height * half - debt_count))
        target = np.zeros((height, width), dtype=bool)
        sides = np.full(debt_count, int(rng.integers(2)), dtype=int)
        if mixed_sides:
            mode = int(rng.integers(3))
            if mode == 0:
                sides = rng.integers(2, size=debt_count)
            elif mode == 1:
                sides = np.array([r % 2 for r, _ in order[:debt_count]])
            else:
                cut = int(rng.integers(1, height))
                sides = np.array([int(r >= cut) for r, _ in order[:debt_count]])
            if np.all(sides == sides[0]):
                sides[0], sides[-1] = 0, 1
        for (r, c), side in zip(order[:debt_count], sides):
            target[r, c if side == 0 else width - 1 - c] = True
        for r, c in order[debt_count:debt_count + matched_count]:
            target[r, c] = target[r, width - 1 - c] = True
        closure = target | np.fliplr(target)
        # Keep an observable vertical-vs-horizontal-axis witness in each pair.
        if not np.array_equal(closure, target | np.flipud(target)):
            return target
    raise RuntimeError("could not construct an axis-distinguishing target")


def _stamp_masks(rng, target, stamp_count, shared):
    closure = target | np.fliplr(target)
    debt_indices = rng.permutation(np.flatnonzero(closure & ~target))
    amount = len(debt_indices)
    overlap_count = int(rng.integers(1, max(2, amount - 1))) if shared else 0
    # At least one missing cell belongs only to each selected stamp.
    overlap_count = min(overlap_count, amount - 2)
    a_count = int(rng.integers(1, amount - overlap_count))
    a_only = debt_indices[:a_count]
    b_only = debt_indices[a_count:amount - overlap_count]
    both = debt_indices[amount - overlap_count:] if overlap_count else []
    p, q = np.zeros_like(target), np.zeros_like(target)
    p.flat[a_only] = True
    q.flat[b_only] = True
    p.flat[both] = q.flat[both] = True
    # Transparent stamps may contain substantial ink already hidden by the base.
    base_indices = np.flatnonzero(target)
    for index, stamp in enumerate((p, q)):
        minimum = 2 if index == 0 else 1
        n = int(rng.integers(minimum, max(minimum + 1, len(base_indices) // 2 + 1)))
        stamp.flat[rng.choice(base_indices, size=n, replace=False)] = True
    masks = [p, q]
    if stamp_count >= 4:
        # A competing pair yields a *larger* symmetric completion. Each surplus
        # endpoint alone is asymmetric; together they form an extra mirror pair.
        empty_left = np.argwhere(~closure[:, :target.shape[1] // 2])
        r, c = empty_left[int(rng.integers(len(empty_left)))]
        # Erase occluded ink from one decoy. It is then smaller than a correct
        # stamp despite producing surplus, defeating selection by total area.
        d, e = p & ~target, q.copy()
        d[r, c] = True
        e[r, target.shape[1] - 1 - c] = True
        masks.extend([d, e])
    if stamp_count in (3, 5):
        # This stamp has no surplus, but cannot complete the figure with either
        # true operand. Its abundant occluded ink makes raw area a bad heuristic.
        near = closure.copy()
        near.flat[int(rng.choice(a_only))] = False
        near.flat[int(rng.choice(b_only))] = False
        masks.append(near)
    assert len(masks) == stamp_count
    assert _pairs(target, masks) == [(0, 1)]
    return masks


def _gaps(rng, count, total):
    """Positive gaps with a fixed total, leaving cards isolated."""
    values = np.ones(count, dtype=int)
    for _ in range(total - count):
        values[int(rng.integers(count))] += 1
    return values


def _layout(rng, shapes):
    order = list(int(x) for x in rng.permutation(len(shapes)))
    choices = []
    for columns in (1, 2, 3):
        rows = [order[i:i + columns] for i in range(0, len(order), columns)]
        heights = [max(shapes[i][0] for i in row) for row in rows]
        widths = [sum(shapes[i][1] for i in row) + len(row) + 1 for row in rows]
        if sum(heights) + len(rows) + 1 <= 30 and max(widths) <= 30:
            choices.append((rows, heights, widths))
    if not choices:
        raise RuntimeError("no card packing fits the ARC canvas")
    rows, heights, widths = choices[int(rng.integers(len(choices)))]
    minimum_height = sum(heights) + len(rows) + 1
    height = min(30, minimum_height + int(rng.integers(4)))
    width = min(30, max(widths) + int(rng.integers(4)))
    vertical_gaps = _gaps(rng, len(rows) + 1, height - sum(heights))
    positions = [None] * len(shapes)
    y = int(vertical_gaps[0])
    for index, (row, shelf_height) in enumerate(zip(rows, heights)):
        gaps = _gaps(rng, len(row) + 1, width - sum(shapes[i][1] for i in row))
        x = int(gaps[0])
        for j, card in enumerate(row):
            h, w = shapes[card]
            positions[card] = (y + int(rng.integers(shelf_height - h + 1)), x)
            x += w + int(gaps[j + 1])
        y += shelf_height + int(vertical_gaps[index + 1])
    return positions, (height, width)


def _make_pair(rng, stamp_count, dimensions, shared, mixed_sides,
               priority_sign=None, forbidden_target_rank=None):
    height, width = dimensions
    capacity = height * (width // 2)
    lower = 5 if stamp_count == 5 else 3
    debt_count = int(rng.integers(lower, min(capacity - 2, 12) + 1))
    target = _make_target(rng, height, width, debt_count, mixed_sides)
    stamps = _stamp_masks(rng, target, stamp_count, shared)
    masks = [target] + stamps
    shapes = [(height + 4, width + 4)] + [(height + 2, width + 2)] * stamp_count
    for _ in range(40):
        positions, shape = _layout(rng, shapes)
        target_rank = sorted(range(len(masks)), key=lambda i: positions[i]).index(0)
        if target_rank != forbidden_target_rank:
            break
    else:
        raise RuntimeError("could not vary the target's reading-order position")
    colors = [int(x) for x in rng.choice(INKS, size=len(masks), replace=False)]
    chosen_order = sorted((1, 2), key=lambda i: positions[i])
    if priority_sign is not None:
        # Opposite numeric orders in the first two demonstrations disambiguate
        # spatial precedence from choosing the smaller/larger color number.
        first, second = chosen_order
        low, high = sorted((colors[first], colors[second]))
        colors[first], colors[second] = ((low, high) if priority_sign < 0 else (high, low))
    grid = np.zeros(shape, dtype=np.int8)
    for i, (mask, color, (top, left)) in enumerate(zip(masks, colors, positions)):
        thickness = 2 if i == 0 else 1
        h, w = shapes[i]
        grid[top:top + h, left:left + w] = FRAME
        grid[top + thickness:top + h - thickness,
             left + thickness:left + w - thickness] = np.where(mask, color, BACKGROUND)
    # Construct labels from the sampled masks/positions; do not call solve.
    output = np.where(target, colors[0], BACKGROUND).astype(np.int8)
    for i in chosen_order:
        output[masks[i] & (output == BACKGROUND)] = colors[i]
    return {"input": grid, "output": output}, target_rank


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_target_rank = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "test":
                stamp_count = 5
                dimensions = (int(rng.integers(4, 7)), int(rng.choice((4, 6, 8))))
                shared, mixed, sign = bool(rng.integers(2)), True, None
            elif index == 0:
                stamp_count = 4
                dimensions = (int(rng.integers(3, 6)), int(rng.choice((4, 6))))
                shared, mixed, sign = True, False, -1
            elif index == 1:
                stamp_count = 3
                dimensions = (int(rng.integers(4, 7)), int(rng.choice((6, 8))))
                shared, mixed, sign = True, True, 1
            else:
                stamp_count = int(rng.integers(3, 5))
                dimensions = (int(rng.integers(3, 7)), int(rng.choice((4, 6, 8))))
                shared, mixed, sign = bool(rng.integers(2)), bool(rng.integers(2)), None
            for _ in range(40):
                pair, rank = _make_pair(
                    rng, stamp_count, dimensions, shared, mixed, sign,
                    first_target_rank if split == "train" and index == 1 else None)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                first_target_rank = rank
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, cards, target, stamps = _read(pair["input"])
    output = _array(pair["output"])
    base = target["mask"]
    height, width = base.shape
    assert 3 <= height <= 6 and width in (4, 6, 8)
    closure = base | np.fliplr(base)
    debt = closure & ~base
    assert np.any(base & np.fliplr(base)) and np.any(debt) and np.any(~closure)
    assert not np.array_equal(closure, base | np.flipud(base))
    masks = [stamp["mask"] for stamp in stamps]
    pairs = _pairs(base, masks)
    assert len(pairs) == 1
    a, b = pairs[0]
    p, q = masks[a], masks[b]
    assert np.any(p & base) and np.any(q & base)
    assert output.shape == base.shape
    # Per-pixel oracle uses missing mirror partners and visible selected colors.
    expected = target["grid"].copy()
    for r in range(height):
        for c in range(width):
            if not base[r, c] and base[r, width - 1 - c]:
                owners = [i for i in (a, b) if masks[i][r, c]]
                assert owners
                expected[r, c] = stamps[min(owners)]["color"]
    assert np.array_equal(output, expected)
    assert np.array_equal(output != BACKGROUND, closure)
    assert np.array_equal(output[base], target["grid"][base])
    assert np.array_equal(output[~closure], target["grid"][~closure])
    assert np.array_equal(solve(grid), output)
    # Dropping the union/ownership mechanism and simply reflecting base ink
    # necessarily changes each selected stamp's exclusive missing cells.
    monochrome_repair = np.where(closure, target["color"], BACKGROUND)
    assert not np.array_equal(output, monochrome_repair)
    # Ignoring occlusion and matching the donor-only union to the hole mask fails.
    assert not any(np.array_equal(masks[i] | masks[j], debt)
                   for i, j in combinations(range(len(stamps)), 2))
    larger_symmetric = []
    for i, j in combinations(range(len(stamps)), 2):
        candidate = base | masks[i] | masks[j]
        if np.array_equal(candidate, np.fliplr(candidate)) and np.any(candidate & ~closure):
            larger_symmetric.append((i, j))
    eligible_distractors = [i for i, m in enumerate(masks)
                            if i not in (a, b) and not np.any(m & ~closure)]
    shared = debt & p & q
    rank = next(i for i, card in enumerate(cards) if card is target)
    return {"shape": grid.shape, "interior_shape": base.shape, "target_rank": rank,
            "stamp_count": len(stamps), "debt": int(np.sum(debt)),
            "shared": int(np.sum(shared)), "exclusive": (int(np.sum(debt & p & ~q)),
                                                           int(np.sum(debt & q & ~p))),
            "priority_sign": int(np.sign(stamps[a]["color"] - stamps[b]["color"])),
            "larger_symmetric": bool(larger_symmetric),
            "eligible_distractors": len(eligible_distractors),
            "large_near_miss": any(np.sum(masks[i]) > max(np.sum(p), np.sum(q))
                                   for i in eligible_distractors),
            "area_shortcut_fails": set(sorted(range(len(masks)),
                                                key=lambda i: (int(np.sum(masks[i])), i))[:2])
                                   != {a, b},
            "mixed_debt_sides": bool(np.any(debt[:, :width // 2]) and np.any(debt[:, width // 2:])),
            "chosen_positions": (stamps[a]["position"], stamps[b]["position"]),
            "colors": tuple(card["color"] for card in cards)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            analyses[split].append(info)
    first, second = analyses["train"][:2]
    assert first["stamp_count"] == 4 and first["larger_symmetric"] and first["area_shortcut_fails"]
    assert second["stamp_count"] == 3 and second["large_near_miss"]
    assert first["shared"] > 0 and second["shared"] > 0
    assert (first["priority_sign"], second["priority_sign"]) == (-1, 1)
    assert first["target_rank"] != second["target_rank"]
    assert second["mixed_debt_sides"]
    assert all(info["stamp_count"] in (3, 4) for info in analyses["train"])
    assert all(info["stamp_count"] == 5 and info["larger_symmetric"]
               and info["eligible_distractors"] == 1 and info["mixed_debt_sides"]
               for info in analyses["test"])
    return True


def _equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[role], y[role]) for x, y in zip(a[split], b[split])
        for role in ("input", "output")) for split in ("train", "test"))


def self_test():
    """Bounded 42-seed diversity checks, count boundaries, and failure probes."""
    shapes, interiors, structures, palettes, ranks, debts, contributions = (set() for _ in range(7))
    shared_count = disjoint_count = larger_count = near_count = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(pair["input"]) for split in ("train", "test") for pair in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                shapes.add(info["shape"])
                interiors.add(info["interior_shape"])
                palettes.add(info["colors"])
                ranks.add(info["target_rank"])
                debts.add(info["debt"])
                contributions.add(info["exclusive"])
                shared_count += info["shared"] > 0
                disjoint_count += info["shared"] == 0
                larger_count += info["larger_symmetric"]
                near_count += info["eligible_distractors"] > 0
                grid = np.asarray(pair["input"])
                role_map = {0: 0, 5: 1}
                normalized = tuple(role_map.setdefault(int(x), len(role_map)) for x in grid.flat)
                structures.add((grid.shape, normalized))
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(shapes) >= 30 and len(interiors) >= 10
    assert len(palettes) >= 100 and len(ranks) >= 5
    assert len(debts) >= 7 and len(contributions) >= 15
    assert disjoint_count >= 10 and shared_count >= 84
    assert larger_count >= 84 and near_count >= 84
    assert _equal(generate_puzzle(0), generate_puzzle(0))
    assert not _equal(generate_puzzle(0), generate_puzzle(1))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(1234 + 10 * train + test, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (None, 1), (4, None), (True, 1), (4, 1.5), ("4", 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    for seed in (-1, True, 2.5, "0"):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed was accepted")
    # Validation must catch an answer that has the right silhouette but wrong
    # paint ownership, a frequent superficially plausible alternative rule.
    damaged = generate_puzzle(7)
    pair = damaged["train"][0]
    _, _, target, _ = _read(pair["input"])
    closure = target["mask"] | np.fliplr(target["mask"])
    pair["output"] = np.where(closure, target["color"], BACKGROUND).astype(np.int8)
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("ownership corruption was not detected")
    return {"seeds": 42, "pairs": 210, "distinct_inputs": len(structures),
            "canvas_shapes": len(shapes), "interior_shapes": len(interiors),
            "ink_orders": len(palettes), "target_ranks": sorted(ranks),
            "debt_counts": sorted(debts), "exclusive_count_pairs": len(contributions),
            "shared_pairs": shared_count, "disjoint_pairs": disjoint_count,
            "larger_symmetric_decoys": larger_count, "in_bounds_decoys": near_count}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
