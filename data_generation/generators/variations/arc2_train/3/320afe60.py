"""Solved variations of the open/closed contour sorting puzzle."""

import hashlib

import numpy as np


_SOURCE_INPUT_DIGESTS = {
    "b85d0d52df1d00a65800cbe255fd2872d6946f7b24e02a4139d81b9e3ab34fca",
    "5db7fc83f9816a7bf7e02218add56cac26ef8ccb2995d92ad318e27cee5911ec",
    "042c2afc8a780d367400e3cd43c289ec3f60bce8df894468bdf50f60d4cab089",
    "dcf7005ed1574671a8b2d0d62a48d3bb963fad8716f6f2ab297c890d9373c1ee",
    "280893139d5cb395c2108dcf53039f22cfd4ee17b1ebba45de3f65eff2a5071d",
}


def _validate_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, seed, num_train, num_test):
    a = _validate_count("num_train", num_train, 2, 10)
    b = _validate_count("num_test", num_test, 1, 4)
    if a is not None and b is not None:
        if a <= b:
            raise ValueError("num_train must exceed num_test")
        return a, b
    if a is None and b is None:
        b = 1 + seed % 4
        a = max(2, b + 1) + (seed // 4) % (11 - max(2, b + 1))
    elif a is None:
        a = int(rng.integers(b + 1, 11))
    else:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    return a, b


def _connected(mask):
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    seen = {tuple(cells[0])}
    stack = [tuple(cells[0])]
    height, width = mask.shape
    while stack:
        row, col = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _has_pocket(mask, diagonal):
    if bool(np.all(mask)):
        return False
    height, width = mask.shape
    empty = np.ones((height + 2, width + 2), dtype=np.bool_)
    empty[1:-1, 1:-1] = ~mask
    visited = np.zeros_like(empty)
    visited[0, 0] = True
    stack = [(0, 0)]
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    while stack:
        row, col = stack.pop()
        for dr, dc in steps:
            nr, nc = row + dr, col + dc
            if (0 <= nr < height + 2 and 0 <= nc < width + 2
                    and empty[nr, nc] and not visited[nr, nc]):
                visited[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(empty[1:-1, 1:-1] & ~visited[1:-1, 1:-1]))


def _closed(mask):
    return bool(np.all(mask)) or _has_pocket(mask, True)


def _tight(mask):
    rows, cols = np.where(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _ring(height, width):
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def _stepped(rng, height, width):
    side = int(rng.integers(0, 4))
    mask = np.zeros((height, width), dtype=np.bool_)
    if side == 0:
        mask[1:, :] = _ring(height - 1, width)
        col = int(rng.integers(1, width - 1))
        mask[0, col] = True
    elif side == 2:
        mask[:-1, :] = _ring(height - 1, width)
        col = int(rng.integers(1, width - 1))
        mask[-1, col] = True
    elif side == 3:
        mask[:, 1:] = _ring(height, width - 1)
        row = int(rng.integers(1, height - 1))
        mask[row, 0] = True
    else:
        mask[:, :-1] = _ring(height, width - 1)
        row = int(rng.integers(1, height - 1))
        mask[row, -1] = True
    return mask


def _make_closed(rng, height, width, style):
    if style == "solid":
        return np.ones((height, width), dtype=np.bool_)
    if style == "stepped":
        mask = _stepped(rng, height, width)
    elif style == "thick":
        mask = np.ones((height, width), dtype=np.bool_)
        top = int(rng.integers(1, min(3, height - 1)))
        left = int(rng.integers(1, min(3, width - 1)))
        bottom = int(rng.integers(top + 1, height))
        right = int(rng.integers(left + 1, width))
        mask[top:bottom, left:right] = False
    else:
        mask = _ring(height, width)
    if style == "divided" and min(height, width) >= 5:
        if rng.random() < 0.5:
            mask[:, int(rng.integers(2, width - 2))] = True
        else:
            mask[int(rng.integers(2, height - 2)), :] = True
    if style in ("ring", "stepped", "thick") and height >= 5 and width >= 5:
        for _ in range(int(rng.integers(0, 4))):
            row = int(rng.integers(1, height - 1))
            col = int(rng.integers(1, width - 1))
            candidate = mask.copy()
            candidate[row, col] = True
            if _has_pocket(candidate, True) and _connected(candidate):
                mask = candidate
    assert _connected(mask) and _closed(mask) and _has_pocket(mask, False) == _has_pocket(mask, True)
    return mask


def _puncture(rng, base, side, max_gap=3):
    """Open every pocket through one passage, keeping foreground connected."""
    pockets = np.argwhere(~base)
    pockets = np.asarray([(r, c) for r, c in pockets
                          if 0 < r < base.shape[0] - 1 and 0 < c < base.shape[1] - 1], dtype=int)
    for index in rng.permutation(len(pockets)):
        row, col = map(int, pockets[int(index)])
        candidate = base.copy()
        if side == 0:
            length = int(rng.integers(1, min(max_gap, base.shape[1] - col - 1) + 1))
            candidate[:row + 1, col:col + length] = False
        elif side == 1:
            length = int(rng.integers(1, min(max_gap, base.shape[0] - row - 1) + 1))
            candidate[row:row + length, col:] = False
        elif side == 2:
            length = int(rng.integers(1, min(max_gap, base.shape[1] - col - 1) + 1))
            candidate[row:, col:col + length] = False
        else:
            length = int(rng.integers(1, min(max_gap, base.shape[0] - row - 1) + 1))
            candidate[row:row + length, :col + 1] = False
        if (candidate.any() and _connected(candidate) and not _has_pocket(candidate, False)
                and not _has_pocket(candidate, True) and _tight(candidate).shape == base.shape):
            return candidate
    return None


def _make_open(rng, height, width, side, stepped=False):
    if height == 2:
        mask = np.ones((2, width), dtype=np.bool_)
        mask[1, 1:-1] = False
        return mask if side in (2, 3) else np.flipud(mask)
    if width == 2:
        mask = np.ones((height, 2), dtype=np.bool_)
        mask[1:-1, 1] = False
        return mask if side in (1, 2) else np.fliplr(mask)
    for _ in range(12):
        base = _stepped(rng, height, width) if stepped and min(height, width) >= 4 else _ring(height, width)
        mask = _puncture(rng, base, side)
        if mask is None:
            continue
        if height >= 5 and width >= 5 and rng.random() < 0.6:
            for _ in range(5):
                candidate = mask.copy()
                row = int(rng.integers(1, height - 1))
                col = int(rng.integers(1, width - 1))
                candidate[row, col] = True
                if (_connected(candidate) and not _has_pocket(candidate, False)
                        and not _has_pocket(candidate, True)):
                    mask = candidate
                    break
        return mask
    base = _ring(height, width)
    mask = _puncture(rng, base, side)
    assert mask is not None
    return mask


def _choose_specs(rng, regime, index, gap_phase):
    if regime == "source_train" and index == 0:
        specs = [dict(kind="solid"), dict(kind="twin_closed"),
                 dict(kind="twin_open", side=gap_phase)]
        extra = int(rng.integers(0, 3))
    elif regime == "source_train" and index == 1:
        specs = [dict(kind="ring"), dict(kind="open", side=(gap_phase + 1) % 4),
                 dict(kind="open", side=(gap_phase + 2) % 4)]
        extra = int(rng.integers(0, 3))
    else:
        specs = []
        extra = int(rng.integers(1, 7))
    for _ in range(extra):
        kind = ("solid", "ring", "stepped", "thick", "divided", "open", "open_step")[
            int(rng.integers(0, 7))]
        specs.append(dict(kind=kind, side=int(rng.integers(0, 4))))
    rng.shuffle(specs)
    return specs


def _dimensions(rng, spec, max_height, regime):
    kind = spec["kind"]
    if kind.startswith("twin"):
        return None
    if kind == "solid":
        if rng.random() < 0.28:
            return (2, int(rng.choice((2, 3, 4))))
        low = 2
        high = min(7, max_height)
    elif kind in ("stepped", "open_step"):
        low = 4
        high = min(9, max_height)
    elif kind == "open" and rng.random() < 0.16:
        length = int(rng.integers(3, 7))
        return (2, length) if spec["side"] in (0, 2) else (length, 2)
    else:
        low = 3
        high = min(9, max_height)
    height = int(rng.integers(low, max(low, high) + 1))
    width_low = 4 if kind in ("stepped", "open_step") else (3 if kind != "solid" else 2)
    width_high = 10 if regime == "target_test" else 9
    width = int(rng.integers(width_low, width_high + 1))
    return height, width


def _allocate_rows(rng, specs, canvas_height):
    total = sum(spec["mask"].shape[0] for spec in specs) + len(specs) - 1
    assert total <= canvas_height
    gaps = [0] * (len(specs) + 1)
    for _ in range(canvas_height - total):
        gaps[int(rng.integers(0, len(gaps)))] += 1
    ordered = list(specs)
    rng.shuffle(ordered)
    row = gaps[0]
    for index, spec in enumerate(ordered):
        spec["row"] = row
        row += spec["mask"].shape[0]
        if index + 1 < len(ordered):
            row += 1 + gaps[index + 1]


def _generate_pair(rng, regime, index, global_index, conventions):
    specs = _choose_specs(rng, regime, index, conventions["gap_phase"])
    closed_kinds = ("solid", "ring", "stepped", "thick", "divided", "twin_closed")
    closed_specs = [s for s in specs if s["kind"] in closed_kinds]
    open_specs = [s for s in specs if s["kind"] not in closed_kinds]
    max_group = max(len(closed_specs), len(open_specs))
    max_height = min(9, (30 - max_group + 1) // max_group)
    twin_height = int(rng.integers(5, min(8, max_height) + 1)) if any(s["kind"].startswith("twin") for s in specs) else 0
    twin_width = int(rng.integers(5, 9)) if twin_height else 0
    twin_base = _stepped(rng, twin_height, twin_width) if twin_height else None
    for spec in specs:
        kind = spec["kind"]
        if kind == "twin_closed":
            mask = twin_base
        elif kind == "twin_open":
            mask = _puncture(rng, twin_base, spec["side"], max_gap=1)
            assert mask is not None
        else:
            height, width = _dimensions(rng, spec, max_height, regime)
            if kind in ("solid", "ring", "stepped", "thick", "divided"):
                mask = _make_closed(rng, height, width, kind)
            else:
                mask = _make_open(rng, height, width, spec["side"], kind == "open_step")
        spec["mask"] = mask
        spec["closed"] = _closed(mask)
        assert spec["closed"] == (kind in closed_kinds)
    need_closed = sum(s["mask"].shape[0] for s in closed_specs) + max(0, len(closed_specs) - 1)
    need_open = sum(s["mask"].shape[0] for s in open_specs) + max(0, len(open_specs) - 1)
    need_bands = sum(s["mask"].shape[0] for s in specs) + len(specs) - 1
    bands = len(specs) > 1 and need_bands <= 30 and rng.random() < 0.28
    min_height = max(11, need_bands if bands else max(need_closed, need_open))
    canvas_height = int(rng.integers(min_height, 31))
    if bands:
        _allocate_rows(rng, specs, canvas_height)
    else:
        if closed_specs:
            _allocate_rows(rng, closed_specs, canvas_height)
        if open_specs:
            _allocate_rows(rng, open_specs, canvas_height)
    max_width = max(s["mask"].shape[1] for s in specs)
    min_width = max(15 if regime == "source_train" else 18, 2 * max_width + 3)
    canvas_width = int(rng.integers(min_width, 31))
    palette = conventions["backgrounds"]
    background = palette[global_index % len(palette)]
    available_inks = [c for c in range(10) if c not in (background, conventions["closed_color"], conventions["open_color"])]
    rng.shuffle(available_inks)
    ink_count = 1 if regime == "source_train" and index == 0 else (2 if regime == "source_train" and index == 1 else int(rng.integers(1, 4)))
    inks = available_inks[:ink_count]
    input_grid = np.full((canvas_height, canvas_width), background, dtype=np.int8)
    output_grid = np.full_like(input_grid, background)
    layout = 1 if regime == "source_train" and index == 0 else (0 if regime == "source_train" and index == 1 else int(rng.integers(0, 2)))
    for spec in specs:
        mask = spec["mask"]
        height, width = mask.shape
        closed = spec["closed"]
        left_half = closed == (layout == 0)
        if left_half:
            lo, hi = 0, canvas_width // 2 - width - 1
        else:
            lo, hi = canvas_width // 2 + 1, canvas_width - width
        assert lo <= hi
        if regime == "source_train" and index == 0:
            col = int(rng.integers(lo, hi + 1))
        elif ((closed and left_half) or (not closed and not left_half)) and rng.random() < 0.25:
            col = 0 if closed else canvas_width - width
        else:
            col = int(rng.integers(lo, hi + 1))
        row = spec["row"]
        assert bool(np.all(input_grid[row:row + height, col:col + width][mask] == background))
        ink = inks[0] if ink_count == 1 else inks[int(rng.integers(0, ink_count))]
        input_grid[row:row + height, col:col + width][mask] = ink
        destination = 0 if closed else canvas_width - width
        assert bool(np.all(output_grid[row:row + height, destination:destination + width][mask] == background))
        color = conventions["closed_color"] if closed else conventions["open_color"]
        output_grid[row:row + height, destination:destination + width][mask] = color
    rotation = conventions["rotation"]
    if rotation:
        input_grid = np.rot90(input_grid, rotation).copy()
        output_grid = np.rot90(output_grid, rotation).copy()
    if conventions["mirror"]:
        input_grid = np.fliplr(input_grid).copy()
        output_grid = np.fliplr(output_grid).copy()
    return {"input": input_grid, "output": output_grid}


def _signature(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _source_digest(grid):
    return hashlib.sha256(str(grid.shape).encode() + grid.astype(np.int8, copy=False).tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    colors = [int(x) for x in rng.permutation(10)]
    conventions = {
        "closed_color": colors[0], "open_color": colors[1],
        "backgrounds": colors[2:], "rotation": int(rng.integers(0, 4)),
        "mirror": bool(rng.integers(0, 2)), "gap_phase": int(rng.integers(0, 4)),
    }
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, regime, count in (("train", "source_train", train_count), ("test", "target_test", test_count)):
        for index in range(count):
            for _ in range(32):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _generate_pair(child_rng, regime, index, len(seen), conventions)
                signature = _signature(pair["input"])
                if signature not in seen and _source_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    height, width = mask.shape
    visited = np.zeros_like(mask)
    records = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            cells = [(row, col)]
            visited[row, col] = True
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        cells.append((nr, nc))
            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells)
            shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.bool_)
            for r, c in cells:
                shape[r - r0, c - c0] = True
            records.append({"row": r0, "col": c0, "mask": shape, "closed": _closed(shape), "cells": cells})
    return records


def _analyze_pair(pair, direction):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    height, width = inp.shape
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) >= 2
    background = int(values[int(np.argmax(counts))])
    assert int(counts.max()) > inp.size // 2
    records = _components(inp != background)
    assert 1 <= len(records) <= 6
    labels = np.full(inp.shape, -1, dtype=np.int16)
    target = np.full(inp.shape, -1, dtype=np.int8)
    for number, record in enumerate(records):
        cells = record["cells"]
        inks = {int(inp[r, c]) for r, c in cells}
        assert len(inks) == 1
        mask = record["mask"]
        assert mask.shape[0] >= 2 and mask.shape[1] >= 2
        assert _has_pocket(mask, False) == _has_pocket(mask, True)
        row, col = record["row"], record["col"]
        h, w = mask.shape
        for r, c in cells:
            labels[r, c] = number
        if direction == 0:
            dest_row, dest_col = row, 0 if record["closed"] else width - w
        elif direction == 1:
            dest_row, dest_col = row, width - w if record["closed"] else 0
        elif direction == 2:
            dest_row, dest_col = 0 if record["closed"] else height - h, col
        else:
            dest_row, dest_col = height - h if record["closed"] else 0, col
        block = target[dest_row:dest_row + h, dest_col:dest_col + w]
        assert bool(np.all(block[mask] == -1))
        block[mask] = int(record["closed"])
        record["movement"] = abs(dest_row - row) + abs(dest_col - col)
    for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
        for r in range(height):
            for c in range(width):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and labels[r, c] >= 0 and labels[nr, nc] >= 0:
                    assert labels[r, c] == labels[nr, nc]
    assert bool(np.all(out[target == -1] == background))
    colors = []
    for category in (1, 0):
        cells = out[target == category]
        if len(cells):
            unique = np.unique(cells)
            assert len(unique) == 1 and int(unique[0]) != background
            colors.append(int(unique[0]))
        else:
            colors.append(None)
    assert colors[0] is None or colors[1] is None or colors[0] != colors[1]
    return {"background": background, "closed_color": colors[0], "open_color": colors[1], "records": records}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            inp_sig = _signature(pair["input"])
            pair_sig = (inp_sig, _signature(pair["output"]))
            assert inp_sig not in seen_inputs and pair_sig not in seen_pairs
            assert _source_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS
            seen_inputs.add(inp_sig)
            seen_pairs.add(pair_sig)
    possible = []
    for direction in range(4):
        try:
            analyses = [_analyze_pair(pair, direction) for pair in puzzle["train"]]
            closed_colors = {a["closed_color"] for a in analyses if a["closed_color"] is not None}
            open_colors = {a["open_color"] for a in analyses if a["open_color"] is not None}
            assert len(closed_colors) == len(open_colors) == 1
            closed_color = next(iter(closed_colors))
            open_color = next(iter(open_colors))
            assert closed_color != open_color
            possible.append((direction, closed_color, open_color, analyses))
        except AssertionError:
            continue
    assert len(possible) == 1
    direction, closed_color, open_color, train_analyses = possible[0]
    test_analyses = [_analyze_pair(pair, direction) for pair in puzzle["test"]]
    for analysis in train_analyses + test_analyses:
        assert analysis["background"] not in (closed_color, open_color)
        assert analysis["closed_color"] in (None, closed_color)
        assert analysis["open_color"] in (None, open_color)
    train_records = [r for a in train_analyses for r in a["records"]]
    assert any(r["closed"] and bool(np.all(r["mask"])) for r in train_records)
    assert any(r["closed"] and not bool(np.all(r["mask"])) for r in train_records)
    assert any(r["closed"] and _border_break(r["mask"]) for r in train_records)
    assert sum(not r["closed"] for r in train_records) >= 2
    unambiguous_gaps = {
        next(iter(sides)) for r in train_records if not r["closed"]
        for sides in (_visible_gap_sides(r["mask"]),) if len(sides) == 1
    }
    assert len(unambiguous_gaps) >= 2
    assert any(r["closed"] and r["movement"] > 0 for r in train_records)
    assert any(not r["closed"] and r["movement"] > 0 for r in train_records)
    assert len({a["background"] for a in train_analyses}) >= 2
    assert _near_match(train_records)


def _border_break(mask):
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return False
    return bool(np.any(~mask[0, 1:-1]) or np.any(~mask[-1, 1:-1])
                or np.any(~mask[1:-1, 0]) or np.any(~mask[1:-1, -1]))


def _visible_gap_sides(mask):
    sides = set()
    if mask.shape[1] >= 3:
        if bool(np.any(~mask[0, 1:-1])):
            sides.add("top")
        if bool(np.any(~mask[-1, 1:-1])):
            sides.add("bottom")
    if mask.shape[0] >= 3:
        if bool(np.any(~mask[1:-1, 0])):
            sides.add("left")
        if bool(np.any(~mask[1:-1, -1])):
            sides.add("right")
    return sides


def _near_match(records):
    for closed in records:
        if not closed["closed"] or not _border_break(closed["mask"]):
            continue
        for opened in records:
            if opened["closed"] or opened["mask"].shape != closed["mask"].shape:
                continue
            if bool(np.all(opened["mask"] <= closed["mask"])) and 1 <= int(np.sum(closed["mask"] & ~opened["mask"])) <= 3:
                return True
    return False


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    first = generate_puzzle(918273, 7, 3)
    assert _puzzles_equal(first, generate_puzzle(918273, 7, 3))
    for seed, a, b in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                       (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        if a is not None:
            assert len(puzzle["train"]) == a
        if b is not None:
            assert len(puzzle["test"]) == b
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 4}):
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    train_counts, test_counts, sizes, motifs, layouts = set(), set(), set(), set(), set()
    seen_puzzles, orientations, small_shapes, gap_sides = set(), set(), set(), set()
    stepped_closed = thick_closed = zero = overlap = bands = single_class = 0
    multi_ink = combined = source_palette = long_gaps = short_gaps = 0
    matched_sizes = novel_sizes = within_variation = 0
    challenge_sizes = set()
    for seed in range(4200, 4242):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        sig = tuple((_signature(p["input"]), _signature(p["output"]))
                    for split in ("train", "test") for p in puzzle[split])
        seen_puzzles.add(sig)
        compatible = []
        for direction in range(4):
            try:
                for pair in puzzle["train"]:
                    _analyze_pair(pair, direction)
                compatible.append(direction)
            except AssertionError:
                pass
        assert len(compatible) == 1
        direction = compatible[0]
        orientations.add(direction)
        puzzle_sizes, puzzle_counts, puzzle_backgrounds, puzzle_inks = set(), set(), set(), set()
        train_shapes, test_shapes = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                sizes.add(inp.shape)
                analysis = _analyze_pair(pair, direction)
                bg, records = analysis["background"], analysis["records"]
                puzzle_sizes.add(inp.shape)
                puzzle_counts.add(len(records))
                puzzle_backgrounds.add(bg)
                inks = {int(inp[r["cells"][0]]) for r in records}
                puzzle_inks.update(inks)
                multi_ink += len(inks) > 1
                source_palette += set(int(c) for c in np.unique(inp)) == {1, 4}
                layouts.add((inp.shape, tuple(sorted((r["row"], r["col"], r["mask"].shape) for r in records))))
                kinds = {r["closed"] for r in records}
                single_class += len(kinds) == 1
                pair_overlap = False
                for i, r in enumerate(records):
                    mask = r["mask"]
                    motifs.add((mask.shape, mask.tobytes()))
                    (train_shapes if split == "train" else test_shapes).add(mask.shape)
                    small_shapes.add(mask.shape)
                    if split == "test" and mask.shape in ((6, 6), (4, 9), (9, 4)):
                        challenge_sizes.add(mask.shape)
                    stepped_closed += r["closed"] and _border_break(mask)
                    thick_closed += (r["closed"] and not bool(np.all(mask)) and
                                     any(bool(np.all(mask[u:u + 2, v:v + 2]))
                                         for u in range(mask.shape[0] - 1)
                                         for v in range(mask.shape[1] - 1)))
                    zero += r["movement"] == 0
                    if not r["closed"]:
                        sides = _visible_gap_sides(mask)
                        if len(sides) == 1:
                            side = next(iter(sides))
                            gap_sides.add(side)
                            if side == "top":
                                length = int(np.sum(~mask[0, 1:-1]))
                            elif side == "bottom":
                                length = int(np.sum(~mask[-1, 1:-1]))
                            elif side == "left":
                                length = int(np.sum(~mask[1:-1, 0]))
                            else:
                                length = int(np.sum(~mask[1:-1, -1]))
                            short_gaps += length == 1
                            long_gaps += length >= 2
                    for other in records[i + 1:]:
                        if (r["closed"] != other["closed"] and
                                not (r["row"] + mask.shape[0] <= other["row"] or
                                     other["row"] + other["mask"].shape[0] <= r["row"])):
                            pair_overlap = True
                overlap += pair_overlap
                bands += not pair_overlap
                combined += pair_overlap and len(inks) > 1
        matched_sizes += bool(train_shapes & test_shapes)
        novel_sizes += bool(test_shapes - train_shapes)
        within_variation += (len(puzzle_sizes) > 1 and len(puzzle_counts) > 1 and
                             len(puzzle_backgrounds) > 1 and len(puzzle_inks) > 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(seen_puzzles) == 42 and len(sizes) >= 60
    assert len(motifs) >= 100 and len(layouts) >= 150
    assert stepped_closed >= 40 and thick_closed >= 30
    assert (2, 2) in small_shapes and (2, 3) in small_shapes | {(h, w) for w, h in small_shapes}
    assert (2, 4) in small_shapes | {(h, w) for w, h in small_shapes}
    assert (6, 6) in challenge_sizes and bool({(4, 9), (9, 4)} & challenge_sizes)
    assert gap_sides == {"top", "right", "bottom", "left"}
    assert short_gaps >= 20 and long_gaps >= 20
    assert overlap >= 20 and bands >= 20 and single_class >= 5 and zero >= 20
    assert multi_ink >= 20 and combined >= 10 and source_palette >= 1
    assert matched_sizes >= 10 and novel_sizes >= 10 and within_variation == 42
    assert orientations == {0, 1, 2, 3}
