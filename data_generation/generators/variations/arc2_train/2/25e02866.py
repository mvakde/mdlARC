import hashlib

import numpy as np


# Hashes of the three original input grids (uint8 cells in row-major order).
_SOURCE_INPUTS = {
    ((14, 14), "98da429acd7ac7980ea454dbef955f74fd6f89bb391a89d02357a832bd3e8176"),
    ((16, 16), "5f8d04cdd1bdcc8f3d22483bd9cb27200310c94cb8fdb16c1b62eb049db2edcc"),
    ((18, 18), "4fa87d47ffc69b70c86983abf993f7a6aac8bf5e6351055adc2104e2bc6becf9"),
}


def _choose_counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"illegal {name}")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
    elif num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return int(num_train), int(num_test)


def _specifications(rng, train_count, test_count):
    # The first two demonstrations always differ in panel count. Four
    # schedules show a progression, a shared size, and single-factor shifts.
    schedule = int(rng.integers(4))
    if schedule == 0:
        train = [(2, 4), (3, 5)]
        first_test = (4, 6)
    elif schedule == 1:
        side = int(rng.integers(4, 8))
        train = [(2, side), (4, side)]
        first_test = (int(rng.choice((2, 4))), side)
    elif schedule == 2:
        sides = rng.choice(np.arange(4, 8), size=2, replace=False)
        train = [(2, int(sides[0])), (4, int(sides[1]))]
        first_test = (3, int(sides[0]))
    else:
        sides = rng.choice(np.arange(4, 8), size=2, replace=False)
        train = [(2, int(sides[0])), (4, int(sides[1]))]
        first_test = (4, int(sides[0]))

    for _ in range(train_count - 2):
        train.append((int(rng.integers(2, 5)), int(rng.integers(4, 8))))
    test = [first_test]
    for _ in range(test_count - 1):
        test.append((int(rng.integers(2, 5)), int(rng.integers(4, 8))))
    return train, test


def _disjoint(a, b, side):
    ar, ac = a
    br, bc = b
    return (
        ar + side < br or br + side < ar
        or ac + side < bc or bc + side < ac
    )


def _place_panels(rng, count, side):
    styles = ["scatter"]
    if count * side + count + 1 <= 30:
        styles.extend(("horizontal", "vertical", "down_diagonal", "up_diagonal"))
    style = styles[int(rng.integers(len(styles)))]

    if style == "scatter":
        # Use a roomy temporary field, then crop it to independently sampled
        # margins. The rectangles, not just their accents, must be separated.
        for _ in range(30):
            height = int(rng.integers(2 * side + 6, 31))
            width = int(rng.integers(2 * side + 6, 31))
            positions = []
            for _ in range(count):
                for _ in range(350):
                    candidate = (
                        int(rng.integers(1, height - side)),
                        int(rng.integers(1, width - side)),
                    )
                    if all(_disjoint(candidate, old, side) for old in positions):
                        positions.append(candidate)
                        break
                else:
                    break
            if len(positions) == count:
                break
        else:
            # A 2 by 2 packing always fits, including four 7 by 7 panels.
            positions = [
                ((i // 2) * (side + 2), (i % 2) * (side + 2))
                for i in range(count)
            ]
    else:
        max_gap = (28 - count * side) // (count - 1)
        gap = int(rng.integers(1, min(max_gap, 6) + 1))
        stride = side + gap
        if style == "horizontal":
            positions = [(int(rng.integers(0, 4)), i * stride) for i in range(count)]
        elif style == "vertical":
            positions = [(i * stride, int(rng.integers(0, 4))) for i in range(count)]
        elif style == "down_diagonal":
            positions = [(i * stride, i * stride) for i in range(count)]
        else:
            positions = [(i * stride, (count - 1 - i) * stride) for i in range(count)]

    min_row = min(row for row, _ in positions)
    min_col = min(col for _, col in positions)
    span_h = max(row + side for row, _ in positions) - min_row
    span_w = max(col + side for _, col in positions) - min_col
    top = int(rng.integers(1, min(4, 30 - span_h)))
    left = int(rng.integers(1, min(4, 30 - span_w)))
    bottom = int(rng.integers(1, min(4, 31 - span_h - top)))
    right = int(rng.integers(1, min(4, 31 - span_w - left)))
    shifted = [(r - min_row + top, c - min_col + left) for r, c in positions]
    return shifted, span_h + top + bottom, span_w + left + right, style


def _mask(rng, side, desired, occupied):
    free = {(r, c) for r in range(side) for c in range(side)} - occupied
    assert len(free) >= desired
    if desired == 1:
        edge = sorted(
            cell for cell in free
            if 0 in cell or side - 1 in cell
        )
        choices = edge if edge and rng.random() < 0.45 else sorted(free)
        return {choices[int(rng.integers(len(choices)))]}

    kind = int(rng.choice(3, p=(0.28, 0.57, 0.15)))
    if kind == 0:
        bars = []
        for row in range(side):
            for col in range(side):
                for dr, dc in ((1, 0), (0, 1)):
                    cells = {(row + i * dr, col + i * dc) for i in range(desired)}
                    if all(0 <= r < side and 0 <= c < side for r, c in cells) and cells <= free:
                        bars.append(cells)
        if bars:
            return bars[int(rng.integers(len(bars)))]

    edge = sorted(cell for cell in free if 0 in cell or side - 1 in cell)
    starts = edge if edge and rng.random() < 0.45 else sorted(free)
    start = starts[int(rng.integers(len(starts)))]
    chosen = {start}
    free.remove(start)
    while len(chosen) < desired:
        frontier = sorted({
            (r + dr, c + dc)
            for r, c in chosen
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if (r + dr, c + dc) in free
        })
        choices = frontier if frontier and (kind != 2 or rng.random() < 0.4) else sorted(free)
        cell = choices[int(rng.integers(len(choices)))]
        chosen.add(cell)
        free.remove(cell)
    return chosen


def _generate_pair(rng, count, side, shared_roles=None):
    if shared_roles is None:
        palette = [int(x) for x in rng.choice(10, size=count + 2, replace=False)]
        background, carrier = palette[:2]
        accents = palette[2:]
    else:
        background, carrier = shared_roles
        accents = [
            int(x) for x in rng.choice(
                [c for c in range(10) if c not in shared_roles],
                size=count, replace=False
            )
        ]
    positions, height, width, _ = _place_panels(rng, count, side)
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((side, side), carrier, dtype=np.int8)
    occupied = set()
    remaining_budget = int(side * side * 0.57)
    for index, ((row, col), accent) in enumerate(zip(positions, accents)):
        remaining = count - index - 1
        maximum = min(side * side // 3, remaining_budget - remaining)
        weights = (0.20, 0.25, 0.20, 0.13, 0.09, 0.05,
                   0.035, 0.025, 0.02)
        desired = int(rng.choice(np.arange(1, 10), p=weights))
        desired = min(desired, maximum)
        mask = _mask(rng, side, desired, occupied)
        occupied.update(mask)
        remaining_budget -= len(mask)
        block = np.full((side, side), carrier, dtype=np.int8)
        for local_row, local_col in mask:
            block[local_row, local_col] = accent
            out[local_row, local_col] = accent
        inp[row:row + side, col:col + side] = block
    return {"input": inp, "output": out}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    train_specs, test_specs = _specifications(rng, train_count, test_count)
    shared_roles = None
    if rng.random() < 0.27:
        roles = rng.choice(10, size=2, replace=False)
        shared_roles = (int(roles[0]), int(roles[1]))

    seen = set()
    def make_pair(spec):
        for _ in range(80):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _generate_pair(pair_rng, *spec, shared_roles)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen and not _is_source_input(pair["input"]):
                seen.add(key)
                return pair
        raise RuntimeError("could not generate a new pair")

    return {
        "train": [make_pair(spec) for spec in train_specs],
        "test": [make_pair(spec) for spec in test_specs],
    }


def _is_source_input(inp):
    return (
        inp.shape,
        hashlib.sha256(np.asarray(inp, dtype=np.uint8).tobytes()).hexdigest(),
    ) in _SOURCE_INPUTS


def _extract_panels(inp):
    background = int(inp[0, 0])
    assert np.all(inp[0] == background)
    assert np.all(inp[-1] == background)
    assert np.all(inp[:, 0] == background)
    assert np.all(inp[:, -1] == background)
    active = inp != background
    seen = np.zeros(inp.shape, dtype=bool)
    panels = []
    for sr, sc in np.argwhere(active):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = row + dr, col + dc
                if (0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]
                        and active[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        row0 = min(r for r, _ in cells)
        col0 = min(c for _, c in cells)
        row1 = max(r for r, _ in cells) + 1
        col1 = max(c for _, c in cells) + 1
        side = row1 - row0
        assert side == col1 - col0
        assert len(cells) == side * side
        assert 4 <= side <= 7
        block = inp[row0:row1, col0:col1]
        values, counts = np.unique(block, return_counts=True)
        assert len(values) == 2
        index = int(np.argmax(counts))
        carrier = int(values[index])
        accent = int(values[1 - index])
        assert int(counts[index]) > side * side // 2
        panels.append((row0, col0, side, carrier, accent, block == accent))
    assert np.array_equal(seen, active)
    return panels


def _check_pair(pair):
    inp, out = pair["input"], pair["output"]
    panels = _extract_panels(inp)
    assert 2 <= len(panels) <= 4
    sides = {p[2] for p in panels}
    carriers = {p[3] for p in panels}
    accents = {p[4] for p in panels}
    assert len(sides) == len(carriers) == 1
    assert len(accents) == len(panels)
    assert int(inp[0, 0]) not in carriers | accents
    side = panels[0][2]
    assert out.shape == (side, side)
    assert all(_disjoint(a[:2], b[:2], side)
               for i, a in enumerate(panels) for b in panels[i + 1:])
    predicted = np.full((side, side), panels[0][3], dtype=np.int8)
    occupied = np.zeros((side, side), dtype=bool)
    for _, _, _, _, color, mask in panels:
        assert 1 <= int(mask.sum()) <= side * side // 3
        assert not np.any(occupied & mask)
        occupied |= mask
        predicted[mask] = color
    assert int((~occupied).sum()) >= side * side * 0.4
    assert np.array_equal(predicted, out)
    assert set(int(v) for v in np.unique(out)) == carriers | accents
    return panels


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4
    assert train_count > test_count
    seen_inputs = set()
    seen_pairs = set()
    train_panel_counts = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            ikey = (inp.shape, inp.tobytes())
            pkey = (ikey, out.shape, out.tobytes())
            assert ikey not in seen_inputs and pkey not in seen_pairs
            assert not _is_source_input(inp)
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            panels = _check_pair(pair)
            if split == "train":
                train_panel_counts.add(len(panels))
    assert 2 in train_panel_counts
    assert len(train_panel_counts) >= 2


def _signature(puzzle):
    return tuple(
        (g.shape, g.tobytes())
        for split in ("train", "test")
        for pair in puzzle[split]
        for g in (pair["input"], pair["output"])
    )


def _mask_connected(mask):
    cells = set(zip(*np.nonzero(mask)))
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        row, col = stack.pop()
        for neighbor in ((row + 1, col), (row - 1, col),
                         (row, col + 1), (row, col - 1)):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(cells)


def _layout_kind(panels):
    side = panels[0][2]
    rows = [p[0] for p in panels]
    cols = [p[1] for p in panels]
    if max(rows) - min(rows) < side:
        return "horizontal"
    if max(cols) - min(cols) < side:
        return "vertical"
    ordered_cols = [p[1] for p in sorted(panels, key=lambda p: p[0])]
    if all(a < b for a, b in zip(ordered_cols, ordered_cols[1:])):
        return "down_diagonal"
    if all(a > b for a, b in zip(ordered_cols, ordered_cols[1:])):
        return "up_diagonal"
    return "scattered"


def self_test() -> None:
    train_counts, test_counts = set(), set()
    signatures, shapes, structures = set(), set(), set()
    panel_counts, sides, pairs_of_factors = set(), set(), set()
    singleton = 0
    total_masks = 0
    four_train = two_or_three_test = 0
    within_count = within_side = 0
    role_schedules = set()
    layout_kinds = set()
    one_cell_gaps = wide_gaps = 0
    connected_masks = disconnected_masks = dense_masks = edge_masks = 0
    bars = corner_masks = 0
    three_without_northwest = 0
    no_common_accent = 0
    first_test_relations = set()
    progressions = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_signature(puzzle))
        factor_set = set()
        roles = set()
        pair_accents = []
        train_factors = []
        first_test_factor = None
        for split in ("train", "test"):
            for pair in puzzle[split]:
                panels = _check_pair(pair)
                count, side = len(panels), panels[0][2]
                factor_set.add((count, side))
                if split == "train":
                    train_factors.append((count, side))
                elif first_test_factor is None:
                    first_test_factor = (count, side)
                panel_counts.add(count)
                sides.add(side)
                pairs_of_factors.add((count, side))
                layout_kinds.add(_layout_kind(panels))
                shapes.add(pair["input"].shape)
                structures.add((split, count, side, tuple(sorted(
                    (p[0], p[1]) for p in panels
                ))))
                roles.add((int(pair["input"][0, 0]), panels[0][3]))
                pair_accents.append({p[4] for p in panels})
                total_masks += count
                singleton += sum(int(p[5].sum() == 1) for p in panels)
                for p in panels:
                    cells = list(zip(*np.nonzero(p[5])))
                    bars += len(cells) >= 2 and (
                        len({r for r, _ in cells}) == 1
                        or len({c for _, c in cells}) == 1
                    )
                    corner_masks += any(
                        r in (0, side - 1) and c in (0, side - 1)
                        for r, c in cells
                    )
                    edge_masks += any(
                        r in (0, side - 1) or c in (0, side - 1)
                        for r, c in cells
                    )
                    dense_masks += len(cells) >= 7
                    if _mask_connected(p[5]):
                        connected_masks += 1
                    else:
                        disconnected_masks += 1
                for i, a in enumerate(panels):
                    for b in panels[i + 1:]:
                        gap = max(
                            max(0, abs(a[0] - b[0]) - side),
                            max(0, abs(a[1] - b[1]) - side),
                        )
                        one_cell_gaps += gap == 1
                        wide_gaps += gap >= 5
                if count == 3 and not any(
                    p[0] == min(q[0] for q in panels)
                    and p[1] == min(q[1] for q in panels)
                    for p in panels
                ):
                    three_without_northwest += 1
                if split == "train" and count == 4:
                    four_train += 1
                if split == "test" and count in (2, 3):
                    two_or_three_test += 1
        within_count += len({a for a, _ in factor_set}) > 1
        within_side += len({b for _, b in factor_set}) > 1
        role_schedules.add(len(roles))
        no_common_accent += not set.intersection(*pair_accents)
        progressions += (
            train_factors[:2] == [(2, 4), (3, 5)]
            and first_test_factor == (4, 6)
        )
        if first_test_factor in train_factors:
            first_test_relations.add("both")
        elif any(first_test_factor[0] == n for n, _ in train_factors):
            first_test_relations.add("count")
        elif any(first_test_factor[1] == s for _, s in train_factors):
            first_test_relations.add("side")
        else:
            first_test_relations.add("neither")
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 20
    assert len(structures) >= 100 and len(pairs_of_factors) >= 9
    assert panel_counts == {2, 3, 4} and sides == {4, 5, 6, 7}
    assert singleton >= total_masks // 10
    assert four_train >= 10 and two_or_three_test >= 10
    assert within_count == 42 and within_side >= 20
    assert 1 in role_schedules and max(role_schedules) >= 3
    assert layout_kinds == {
        "horizontal", "vertical", "down_diagonal",
        "up_diagonal", "scattered",
    }
    assert one_cell_gaps >= 20 and wide_gaps >= 20
    assert connected_masks >= total_masks // 2
    assert disconnected_masks >= 10 and dense_masks >= 10
    assert edge_masks >= total_masks // 3
    assert bars >= 20 and corner_masks >= 20
    assert three_without_northwest >= 5
    assert no_common_accent >= 20
    assert progressions >= 4
    assert first_test_relations >= {"both", "count", "side"}

    for index, counts in enumerate((
        (2, 1), (10, 1), (10, 4), (None, 1),
        (None, 4), (2, None), (10, None),
    )):
        puzzle = generate_puzzle(1000 + index, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 3), (3, 4)):
        try:
            generate_puzzle(5, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
    a = generate_puzzle(999, 7, 3)
    b = generate_puzzle(999, 7, 3)
    validate_generated_puzzle(a)
    validate_generated_puzzle(b)
    assert _signature(a) == _signature(b)
