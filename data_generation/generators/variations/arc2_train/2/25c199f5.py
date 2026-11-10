import numpy as np


_ORDERS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)
_REVERSE_ORDER = (2, 1, 0)

_SOURCE_PAIR_ROWS = (
    (
        (
            "77777677777677777",
            "77177677777677777",
            "71117677577677777",
            "77777677777671117",
            "77777677777677777",
        ),
        (
            "77777",
            "71117",
            "77577",
            "77177",
            "71117",
        ),
    ),
    (
        (
            "11111677777677777",
            "17771677557677777",
            "11111677777677777",
            "77777677777677777",
            "77777677777671177",
        ),
        (
            "71177",
            "77557",
            "11111",
            "17771",
            "11111",
        ),
    ),
    (
        (
            "77777677777677777",
            "77777675555677777",
            "77777677777671117",
            "11111677777677777",
            "77777677777677777",
        ),
        (
            "77777",
            "77777",
            "71117",
            "75555",
            "11111",
        ),
    ),
    (
        (
            "77777677777677777",
            "75557677777677757",
            "75757677117677777",
            "75557677777677777",
            "77777677777677777",
        ),
        (
            "77757",
            "77117",
            "75557",
            "75757",
            "75557",
        ),
    ),
    (
        (
            "77777677777677777",
            "77777677177677777",
            "75557671117677577",
            "77777677177677777",
            "77777677777677777",
        ),
        (
            "77577",
            "77177",
            "71117",
            "77177",
            "75557",
        ),
    ),
    (
        (
            "77777677777677777",
            "77777677777677777",
            "77777677557677777",
            "71177677557677717",
            "71177677777677777",
        ),
        (
            "77717",
            "77557",
            "77557",
            "71177",
            "71177",
        ),
    ),
    (
        (
            "77777677777677777",
            "77777677777677777",
            "75777677777677777",
            "55577677777677777",
            "75777671777655555",
        ),
        (
            "55555",
            "71777",
            "75777",
            "55577",
            "75777",
        ),
    ),
)


def _rows_to_grid(rows):
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


def _is_source_pair(input_grid, output_grid):
    for input_rows, output_rows in _SOURCE_PAIR_ROWS:
        source_input = _rows_to_grid(input_rows)
        source_output = _rows_to_grid(output_rows)
        if (
            input_grid.shape == source_input.shape
            and output_grid.shape == source_output.shape
            and np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def _validated_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _validated_count(num_train, "num_train", 2, 10)
    test = _validated_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _signature(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _connected(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    todo = [tuple(points[0])]
    seen = {todo[0]}
    while todo:
        row, col = todo.pop()
        for neighbor in ((row - 1, col), (row + 1, col),
                         (row, col - 1), (row, col + 1)):
            rr, cc = neighbor
            if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                    and mask[rr, cc] and neighbor not in seen):
                seen.add(neighbor)
                todo.append(neighbor)
    return len(seen) == len(points)


def _sample_mask(rng, height, width, family=None):
    """Build one 4-connected motif, with both boundary rows occupied."""
    if family is None:
        names = ["runs", "walk", "branch", "bend", "frame", "cross", "blob"]
        if height == 1:
            names = ["runs", "runs", "runs", "blob"]
        elif height == 2 or width < 3:
            names = ["runs", "walk", "branch", "bend", "blob"]
        family = names[int(rng.integers(len(names)))]
    mask = np.zeros((height, width), dtype=bool)
    if family == "frame" and height >= 3 and width >= 3:
        span = int(rng.integers(3, width + 1))
        left = int(rng.integers(width - span + 1))
        right = left + span - 1
        mask[0, left:right + 1] = True
        mask[-1, left:right + 1] = True
        mask[:, left] = True
        mask[:, right] = True
    elif family == "cross" and height >= 3:
        center = int(rng.integers(width))
        mask[:, center] = True
        row = int(rng.integers(1, height - 1))
        left = int(rng.integers(center + 1))
        right = int(rng.integers(center, width))
        mask[row, left:right + 1] = True
    elif family == "bend" and height >= 2:
        col = int(rng.integers(width))
        mask[:, col] = True
        row = 0 if bool(rng.integers(2)) else height - 1
        edge = 0 if bool(rng.integers(2)) else width - 1
        mask[row, min(col, edge):max(col, edge) + 1] = True
        for _ in range(int(rng.integers(0, 3))):
            rr = int(rng.integers(height))
            if col + 1 < width:
                mask[rr, col + 1] = True
    elif family == "walk" and height >= 2:
        pos = int(rng.integers(width))
        mask[0, pos] = True
        for row in range(1, height):
            step = int(rng.integers(-1, 2))
            nxt = max(0, min(width - 1, pos + step))
            mask[row, min(pos, nxt):max(pos, nxt) + 1] = True
            if bool(rng.integers(2)):
                side = int(rng.integers(-1, 2))
                mask[row, max(0, min(width - 1, nxt + side))] = True
            pos = nxt
    elif family == "branch" and height >= 2:
        pos = int(rng.integers(width))
        for row in range(height):
            if row:
                pos = max(0, min(width - 1, pos + int(rng.integers(-1, 2))))
            mask[row, pos] = True
            reach = int(rng.integers(1, max(2, width // 2 + 2)))
            if bool(rng.integers(2)):
                mask[row, pos:min(width, pos + reach)] = True
            else:
                mask[row, max(0, pos - reach + 1):pos + 1] = True
            if row and not np.any(mask[row] & mask[row - 1]):
                prior = np.flatnonzero(mask[row - 1])
                near = int(prior[np.argmin(abs(prior - pos))])
                mask[row, min(pos, near):max(pos, near) + 1] = True
    elif family == "blob":
        pos = int(rng.integers(width))
        for row in range(height):
            if row:
                pos = max(0, min(width - 1, pos + int(rng.integers(-1, 2))))
            mask[row, pos] = True
            if row and not np.any(mask[row] & mask[row - 1]):
                prev = np.flatnonzero(mask[row - 1])
                near = int(prev[np.argmin(abs(prev - pos))])
                mask[row, min(pos, near):max(pos, near) + 1] = True
        additions = int(rng.integers(1, max(3, height * width // 2 + 1)))
        for _ in range(additions):
            frontier = []
            for row, col in np.argwhere(mask):
                for rr, cc in ((row - 1, col), (row + 1, col),
                               (row, col - 1), (row, col + 1)):
                    if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                        frontier.append((rr, cc))
            if not frontier:
                break
            mask[frontier[int(rng.integers(len(frontier)))]] = True
    else:  # runs, including one-row bars
        if height == 1:
            span = int(rng.integers(1, width + 1))
            left = int(rng.integers(width - span + 1))
            mask[0, left:left + span] = True
        else:
            anchor = int(rng.integers(width))
            for row in range(height):
                left = int(rng.integers(anchor + 1))
                right = int(rng.integers(anchor, width))
                mask[row, left:right + 1] = True
    assert mask[0].any() and mask[-1].any() and _connected(mask)
    return mask


def _make_band(rng, height, width, background, payloads,
               family=None, multicolor=False, witness=False):
    if witness:
        mask = np.zeros((height, width), dtype=bool)
        col = int(rng.integers(1, width - 2))
        mask[0, col] = True
        mask[1, col:col + 2] = True
    else:
        mask = _sample_mask(rng, height, width, family)
    band = np.full((height, width), background, dtype=np.int8)
    primary = int(payloads[int(rng.integers(len(payloads)))])
    band[mask] = primary
    if multicolor and len(payloads) >= 2 and int(mask.sum()) >= 2:
        secondary = int(payloads[int(rng.integers(len(payloads)))])
        if secondary == primary:
            secondary = int(payloads[(payloads.index(primary) + 1) % len(payloads)])
        points = np.argwhere(mask)
        rng.shuffle(points)
        count = int(rng.integers(1, len(points)))
        for row, col in points[:count]:
            band[row, col] = secondary
    return band


def _sample_layout(rng, regime, index, matched_lengths=None):
    if regime == "source_train" and index == 0:
        count = 3
        width = int(rng.integers(4, 9))
        height = int(rng.integers(6, 10))
        lengths = tuple(int(x) for x in rng.permutation((2, 1, 1)))
        return count, width, height, lengths
    if matched_lengths is not None:
        lengths = tuple(int(x) for x in rng.permutation(matched_lengths))
        count = len(lengths)
        total = sum(lengths)
        max_width = min(14, (31 - count) // count)
        width = int(rng.integers(2, max_width + 1))
        if float(rng.random()) < 0.28:
            padding = 0
        else:
            padding = int(rng.integers(1, min(12, 30 - total) + 1)) if total < 30 else 0
        return count, width, total + padding, lengths
    if regime == "source_train" and index == 1:
        count = (2, 4, 5, 6)[int(rng.integers(4))]
    else:
        choices = (2, 3, 4, 5, 6) if regime == "source_train" else (2, 3, 4, 4, 5, 5, 6)
        count = choices[int(rng.integers(len(choices)))]
    size_mode = float(rng.random())
    if size_mode < 0.54:
        height = int(rng.integers(max(4, count), 13))
    elif size_mode < 0.84:
        height = int(rng.integers(max(13, count), 22))
    else:
        height = int(rng.integers(max(22, count), 31))
    max_width = min(14, (31 - count) // count)
    width = int(rng.integers(2, max_width + 1))
    available = height - count
    draw = float(rng.random())
    if draw < 0.24 or available == 0:
        padding = 0
    elif draw < 0.67:
        padding = int(rng.integers(1, min(5, available) + 1))
    else:
        padding = int(rng.integers(1, available + 1))
    extra = available - padding
    concentration = (0.65, 1.8, 5.0)[int(rng.integers(3))]
    weights = rng.dirichlet(np.full(count, concentration))
    lengths = tuple(int(x) + 1 for x in rng.multinomial(extra, weights))
    return count, width, height, lengths


def _sample_starts(rng, height, lengths, witness=False):
    margins = [height - length for length in lengths]
    if witness:
        starts = [int(rng.integers(m + 1)) for m in margins]
        if all(s == m for s, m in zip(starts, margins)):
            starts[0] = 0
        return starts
    mode = ("scattered", "scattered", "top", "bottom", "aligned", "cluster")[
        int(rng.integers(6))]
    if mode == "top":
        return [0 for _ in lengths]
    if mode == "bottom":
        return margins
    if mode == "aligned":
        shared = int(rng.integers(min(margins) + 1))
        return [shared for _ in lengths]
    if mode == "cluster":
        target = int(rng.integers(height))
        return [max(0, min(m, target + int(rng.integers(-2, 3)))) for m in margins]
    return [int(rng.integers(m + 1)) for m in margins]


def _sample_colors(rng, count, witness=False, avoid=None):
    for _ in range(20):
        order = [int(x) for x in rng.permutation(10)]
        if avoid is None or (order[0] != avoid[0] and order[1] != avoid[1]):
            break
    background, divider = order[:2]
    if witness:
        payloads = tuple(order[2:5])
        assignments = list(payloads)
        rng.shuffle(assignments)
    else:
        palette_size = int(rng.integers(1, 5))
        payloads = tuple(order[2:2 + palette_size])
        assignments = [payloads[int(rng.integers(len(payloads)))] for _ in range(count)]
    return background, divider, payloads, assignments


def _generate_pair(rng, regime, index, rotation, avoid_roles=None,
                   matched_lengths=None):
    count, width, height, lengths = _sample_layout(rng, regime, index,
                                                   matched_lengths)
    witness = regime == "source_train" and index == 0
    bg, divider, payloads, assignments = _sample_colors(
        rng, count, witness, avoid_roles)
    starts = _sample_starts(rng, height, lengths, witness)
    bands = []
    witness_panel = lengths.index(2) if witness else -1
    for panel, length in enumerate(lengths):
        if witness:
            color_set = (assignments[panel],)
        else:
            color_set = (assignments[panel],) + tuple(
                c for c in payloads if c != assignments[panel])
        multi = (not witness and float(rng.random()) <
                 (0.40 if regime == "source_train" else 0.55))
        band = _make_band(rng, length, width, bg, color_set,
                          multicolor=multi, witness=(panel == witness_panel))
        bands.append(band)
    inp = np.full((height, count * width + count - 1), bg, dtype=np.int8)
    for panel in range(count):
        left = panel * (width + 1)
        if panel < count - 1:
            inp[:, left + width] = divider
        top = starts[panel]
        inp[top:top + lengths[panel], left:left + width] = bands[panel]
    out = np.full((height, width), bg, dtype=np.int8)
    stack = np.concatenate(bands[::-1], axis=0)
    out[height - len(stack):] = stack
    if rotation:
        inp = np.rot90(inp, k=-1).copy()
        out = np.rot90(out, k=-1).copy()
    return {"input": inp, "output": out}


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    if inp.shape[0] == out.shape[0] and inp.shape[1] > out.shape[1]:
        rotation = 0
        source = inp
        result = out
    elif inp.shape[1] == out.shape[1] and inp.shape[0] > out.shape[0]:
        rotation = 1
        source = np.rot90(inp, k=1)
        result = np.rot90(out, k=1)
    else:
        raise AssertionError("input and output dimensions do not define panels")
    height, full_width = source.shape
    width = result.shape[1]
    assert result.shape[0] == height
    assert 2 <= width <= 14 and 2 <= height <= 30
    assert (full_width + 1) % (width + 1) == 0
    count = (full_width + 1) // (width + 1)
    assert 2 <= count <= 6 and full_width <= 30
    divider = int(source[0, width])
    panels = []
    for panel in range(count):
        left = panel * (width + 1)
        panels.append(source[:, left:left + width])
        if panel < count - 1:
            assert np.all(source[:, left + width] == divider)
    joined = np.concatenate(panels, axis=1)
    assert not np.any(joined == divider)
    values, frequencies = np.unique(joined, return_counts=True)
    bg = int(values[np.argmax(frequencies)])
    assert bg != divider
    assert frequencies.max() > joined.size // 2
    bands, starts = [], []
    for panel in panels:
        active = np.flatnonzero(np.any(panel != bg, axis=1))
        assert active.size > 0
        top, bottom = int(active[0]), int(active[-1]) + 1
        band = panel[top:bottom].copy()
        assert np.all(np.any(band != bg, axis=1))
        assert _connected(band != bg)
        bands.append(band)
        starts.append(top)
    total = sum(len(band) for band in bands)
    assert total <= height
    expected = np.full((height, width), bg, dtype=np.int8)
    expected[height - total:] = np.concatenate(bands[::-1], axis=0)
    assert np.array_equal(result, expected)
    assert not np.any(result == divider)
    return {"rotation": rotation, "count": count, "width": width,
            "height": height, "background": bg, "divider": divider,
            "bands": bands, "starts": tuple(starts),
            "lengths": tuple(len(b) for b in bands), "padding": height - total}


def _witness_is_unambiguous(pair, info):
    # Compare plausible order, internal row reversal, and top alignment rules.
    from itertools import permutations
    inp, out = pair["input"], pair["output"]
    if info["rotation"]:
        out = np.rot90(out, k=1)
    bands = info["bands"]
    bg = info["background"]
    height, width = info["height"], info["width"]
    total = sum(len(b) for b in bands)
    assert info["count"] == 3 and info["padding"] > 0
    for order in permutations(range(3)):
        if order == (2, 1, 0):
            continue
        alternative = np.full((height, width), bg, dtype=np.int8)
        alternative[height - total:] = np.concatenate([bands[k] for k in order])
        assert not np.array_equal(alternative, out)
    reversed_rows = np.full((height, width), bg, dtype=np.int8)
    reversed_rows[height - total:] = np.concatenate(
        [band[::-1] for band in bands[::-1]])
    assert not np.array_equal(reversed_rows, out)
    top_aligned = np.full((height, width), bg, dtype=np.int8)
    top_aligned[:total] = np.concatenate(bands[::-1])
    assert not np.array_equal(top_aligned, out)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            signature = _signature(pair["input"])
            assert signature not in seen
            seen.add(signature)
            assert not _is_source_pair(pair["input"], pair["output"])
            infos[split].append(_inspect_pair(pair))
    orientations = {info["rotation"] for group in infos.values() for info in group}
    assert len(orientations) == 1
    _witness_is_unambiguous(train[0], infos["train"][0])
    assert infos["train"][1]["count"] != 3
    assert (infos["train"][0]["background"] != infos["train"][1]["background"]
            and infos["train"][0]["divider"] != infos["train"][1]["divider"])
    # The same learned rule must reconstruct every output, including tests.
    assert all(info["count"] >= 2 for info in infos["test"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    rotation = int(rng.integers(2))  # One convention for the complete puzzle.
    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            matched_lengths = None
            if split == "test" and float(rng.random()) < 0.32:
                example = puzzle["train"][int(rng.integers(train_count))]
                matched_lengths = _inspect_pair(example)["lengths"]
            for _ in range(32):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                avoid = first_roles if split == "train" and index == 1 else None
                pair = _generate_pair(child, "source_train" if split == "train" else "target_test",
                                      index, rotation, avoid, matched_lengths)
                signature = _signature(pair["input"])
                if signature not in seen and not _is_source_pair(pair["input"], pair["output"]):
                    break
            else:
                raise RuntimeError("could not make a distinct pair")
            seen.add(signature)
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                info = _inspect_pair(pair)
                first_roles = (info["background"], info["divider"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a["input"], b["input"]) and
        np.array_equal(a["output"], b["output"])
        for a, b in zip(first[s], second[s])) for s in ("train", "test"))


def _has_hole(band, background):
    for row in band:
        occupied = np.flatnonzero(row != background)
        if len(occupied) >= 2 and np.any(
                row[occupied[0]:occupied[-1] + 1] == background):
            return True
    return False


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(12345, 8, 4), generate_puzzle(12345, 8, 4))
    assert not _puzzles_equal(generate_puzzle(101), generate_puzzle(102))
    for seed, train, test in ((2001, 2, 1), (2002, 10, 1), (2003, 10, 4),
                              (3001, None, 1), (3002, None, 4),
                              (3003, 2, None), (3004, 10, None)):
        p = generate_puzzle(seed, train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        validate_generated_puzzle(p)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(4001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
    records = [_inspect_pair(pair) for p in puzzles for split in ("train", "test")
               for pair in p[split]]
    train_records = [_inspect_pair(pair) for p in puzzles for pair in p["train"]]
    test_records = [_inspect_pair(pair) for p in puzzles for pair in p["test"]]
    assert len({(r["rotation"], r["height"], r["width"], r["count"])
                for r in records}) >= 70
    assert {2, 3, 4, 5, 6}.issubset({r["count"] for r in records})
    assert max(r["height"] for r in records) >= 25
    assert min(r["width"] for r in records) == 2
    assert {0, 1}.issubset({r["rotation"] for r in records})
    assert any(r["padding"] for r in test_records)
    assert any(r["padding"] == 0 for r in test_records)
    assert any(all(s == r["height"] - h for s, h in zip(r["starts"], r["lengths"]))
               for r in train_records)
    assert any(all(s == 0 for s in r["starts"]) for r in test_records)
    assert any(any(np.all(row != r["background"]) for row in b)
               for r in records for b in r["bands"])
    assert any(len(np.unique(b[b != r["background"]])) >= 2
               for r in train_records for b in r["bands"])
    assert any(len(np.unique(b[b != r["background"]])) >= 2
               for r in test_records for b in r["bands"])
    assert len({tuple(r["lengths"]) for r in records}) >= 40
    assert len({(r["background"], r["divider"]) for r in records}) >= 35
    assert any(len(set(int(v) for b in r["bands"] for v in b.flat
                       if v != r["background"])) == 1 for r in records)
    assert any(len(b) >= 2 and not np.any(np.all(b != r["background"], axis=0))
               for r in records for b in r["bands"])
    for group in (train_records, test_records):
        assert any(_has_hole(b, r["background"]) for r in group for b in r["bands"])
        assert any(all(s == 0 for s in r["starts"]) for r in group)
        assert any(all(s == r["height"] - h for s, h in zip(r["starts"], r["lengths"]))
                   for r in group)
        assert any(len(b) >= 2 and not np.any(np.all(b != r["background"], axis=0))
                   for r in group for b in r["bands"])
        assert any(len({int(x) for b in r["bands"] for x in b.flat
                        if x != r["background"]}) == 1 for r in group)
    assert all(len({_inspect_pair(pair)["background"] for split in ("train", "test")
                    for pair in puzzle[split]}) >= 2 for puzzle in puzzles)
    assert all(len({_inspect_pair(pair)["count"] for split in ("train", "test")
                    for pair in puzzle[split]}) >= 2 for puzzle in puzzles)
