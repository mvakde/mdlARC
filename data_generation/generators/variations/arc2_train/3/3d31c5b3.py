"""Transparent composition of four aligned, monochrome panels."""
from itertools import permutations
import numpy as np

SOURCE_ORDER = (0, 1, 3, 2)
AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 1), (5, 3), (6, 2),
               (7, 4), (8, 3), (9, 4), (10, 1), (10, 2), (10, 3), (10, 4))
SOURCE_INPUTS = frozenset({
    ("555500", "055055", "055555", "444044", "000440", "444040", "202200", "220200", "222020", "008088", "888000", "080080"),
    ("550555", "050505", "000550", "044040", "000004", "040404", "222000", "022020", "222020", "808888", "008888", "000800"),
    ("505000", "005005", "505050", "000404", "000400", "400404", "002002", "220220", "220002", "880888", "888880", "880000"),
    ("500000", "050005", "005550", "440444", "000440", "404400", "202202", "220220", "000002", "888808", "000880", "000888"),
    ("055550", "550555", "550500", "004000", "404440", "400000", "202202", "200002", "000200", "080800", "080000", "080800"),
    ("050550", "050555", "550555", "400044", "000440", "404004", "022220", "222020", "020200", "800808", "800080", "800800"),
    ("505005", "050005", "555000", "000404", "000000", "400400", "202022", "220222", "222222", "008800", "080088", "000800"),
})


def _count(value, low, high, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count(num_train, 2, 10, "num_train")
    test = _count(num_test, 1, 4, "num_test")
    if train is None and test is None:
        train, test = AUTO_COUNTS[seed % len(AUTO_COUNTS)]
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _shape(rng, orientation, panel_count):
    if orientation == "vertical":
        return int(rng.integers(2, 31 // panel_count + 1)), int(rng.choice([x for x in range(3, 31) if x % panel_count]))
    return int(rng.choice([x for x in range(3, 31) if x % panel_count])), int(rng.integers(2, 31 // panel_count + 1))


def _motif(rng, h, w):
    mode = int(rng.integers(8))
    mask = np.zeros((h, w), bool)
    if mode in (0, 7):
        mask = rng.random((h, w)) < float(rng.uniform(0.12, 0.82))
        if mode == 7:
            mask[int(rng.integers(h)):, int(rng.integers(w)):] = False
    elif mode in (1, 3):
        r0, r1 = sorted(int(x) for x in rng.choice(h + 1, 2, replace=False))
        c0, c1 = sorted(int(x) for x in rng.choice(w + 1, 2, replace=False))
        if mode == 1:
            mask[r0:r1, c0:c1] = True
        else:
            mask[r0:r1, c0] = mask[r0:r1, c1 - 1] = True
            mask[r0, c0:c1] = mask[r1 - 1, c0:c1] = True
    elif mode == 2:
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.5:
                mask[int(rng.integers(h)), :] = True
            else:
                mask[:, int(rng.integers(w))] = True
    elif mode == 4:
        th, tw = min(h, int(rng.integers(2, 4))), min(w, int(rng.integers(2, 4)))
        tile = rng.random((th, tw)) < float(rng.uniform(0.25, 0.7))
        mask = np.tile(tile, (int(np.ceil(h / th)), int(np.ceil(w / tw))))[:h, :w]
    elif mode == 5:
        for _ in range(int(rng.integers(2, 6))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            mask[r:min(h, r + int(rng.integers(1, 4))),
                 c:min(w, c + int(rng.integers(1, 4)))] = True
    else:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(2, max(3, h * w // 2 + 1)))):
            mask[r, c] = True
            r = int(np.clip(r + int(rng.integers(-1, 2)), 0, h - 1))
            c = int(np.clip(c + int(rng.integers(-1, 2)), 0, w - 1))
    return mask


def _masks(rng, h, w, regime, panel_count):
    masks = np.stack([_motif(rng, h, w) for _ in range(panel_count)])
    if rng.random() < 0.42:
        source, target = [int(x) for x in rng.choice(panel_count, 2, replace=False)]
        relation = int(rng.integers(4))
        if relation == 0:
            masks[target] = masks[source] ^ (rng.random((h, w)) < 0.12)
        elif relation == 1:
            masks[target] &= masks[source]
        elif relation == 2:
            masks[target] &= ~masks[source]
        else:
            masks[target] |= masks[source] & (rng.random((h, w)) < 0.45)
    if regime == "target_test" and rng.random() < 0.58:
        layer = int(rng.integers(panel_count))
        if rng.random() < 0.55:
            masks[layer, int(rng.integers(h)), :] = True
        else:
            masks[layer, :, int(rng.integers(w))] = True
        masks[layer] |= rng.random((h, w)) < float(rng.uniform(0.15, 0.45))
    if rng.random() < 0.12:
        masks[int(rng.integers(panel_count))] = rng.random() < 0.5
    full = [i for i in range(panel_count) if np.all(masks[i])]
    for layer in full[1:]:
        masks[layer, int(rng.integers(h)), int(rng.integers(w))] = False
    if not np.any(masks):
        masks[int(rng.integers(panel_count)), int(rng.integers(h)), int(rng.integers(w))] = True
    return masks


def _compose(masks, background, foregrounds, order):
    out = np.full(masks.shape[1:], background, dtype=np.int8)
    for layer in reversed(order):
        if foregrounds[layer] is not None:
            out[masks[layer]] = foregrounds[layer]
    return out


def _assemble(masks, palette, orientation, order):
    background, foregrounds = palette[0], palette[1:]
    panels = np.where(masks, np.asarray(foregrounds, np.int8)[:, None, None], background)
    inp = np.concatenate(list(panels), axis=0 if orientation == "vertical" else 1)
    return {"input": inp.astype(np.int8),
            "output": _compose(masks, background, foregrounds, order)}


def _parse_input(inp, panel_count):
    h, w = inp.shape
    vertical = h % panel_count == 0 and w % panel_count != 0
    horizontal = w % panel_count == 0 and h % panel_count != 0
    assert vertical or horizontal
    panels = np.stack(np.split(inp, panel_count, axis=0 if vertical else 1))
    sets = [set(int(x) for x in np.unique(panel)) for panel in panels]
    common = [c for c in range(10) if sum(c in s for s in sets) >= panel_count - 1]
    assert len(common) == 1
    background = common[0]
    foregrounds = []
    for colors in sets:
        others = colors - {background}
        assert len(others) <= 1
        foregrounds.append(next(iter(others)) if others else None)
    visible = [x for x in foregrounds if x is not None]
    assert len(visible) == len(set(visible))
    assert sum(bool(np.all(panel != background)) for panel in panels) <= 1
    return panels != background, background, tuple(foregrounds), ("vertical" if vertical else "horizontal")


def _panel_count(pair):
    inp, out = pair["input"], pair["output"]
    if inp.shape[1] == out.shape[1] and inp.shape[0] % out.shape[0] == 0:
        return inp.shape[0] // out.shape[0]
    if inp.shape[0] == out.shape[0] and inp.shape[1] % out.shape[1] == 0:
        return inp.shape[1] // out.shape[1]
    raise AssertionError("input and output dimensions do not specify equal panels")


def _candidate_orders(train):
    panel_count = _panel_count(train[0])
    analyses = [(_parse_input(pair["input"], panel_count), pair["output"]) for pair in train]
    return [order for order in permutations(range(panel_count))
            if all(np.array_equal(_compose(m, bg, fg, order), out)
                   for (m, bg, fg, _), out in analyses)]


def _predict(train, test_input):
    orders = _candidate_orders(train)
    assert orders
    masks, background, foregrounds, _ = _parse_input(test_input, _panel_count(train[0]))
    predictions = [_compose(masks, background, foregrounds, order) for order in orders]
    assert all(np.array_equal(predictions[0], p) for p in predictions[1:])
    return predictions[0]


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _source_input(inp):
    return tuple("".join(str(int(x)) for x in row) for row in inp) in SOURCE_INPUTS


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    panel_count = (4, 3, 5, 4, 4)[seed % 5]
    order = SOURCE_ORDER if panel_count == 4 and seed % 6 == 0 else tuple(int(x) for x in rng.permutation(panel_count))
    specs = []
    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        local = index if split == "train" else index - train_count
        regime = seed % 4
        orientation = ("vertical" if regime == 0 else "horizontal" if regime == 1 else
                       ("vertical" if local % 2 == 0 else "horizontal") if regime == 2 else
                       ("vertical" if split == "train" else "horizontal"))
        h, w = _shape(rng, orientation, panel_count)
        if specs and (h, w, orientation) == specs[-1]["shape"]:
            h, w = _shape(rng, orientation, panel_count)
        palette = tuple(int(x) for x in rng.choice(10, panel_count + 1, replace=False))
        specs.append({"shape": (h, w, orientation), "palette": palette,
                      "masks": _masks(rng, h, w, "source_train" if split == "train" else "target_test", panel_count)})
    # Adjacent pure overlaps establish the inequalities of the total order.
    train_slots = list(int(x) for x in rng.permutation(train_count))
    reserved = [set() for _ in range(train_count)]
    for position in range(panel_count - 1):
        index = train_slots[position % train_count]
        masks = specs[index]["masks"]
        h, w = masks.shape[1:]
        available = [cell for cell in range(h * w) if cell not in reserved[index]]
        cell = int(rng.choice(available))
        reserved[index].add(cell)
        masks[:, cell // w, cell % w] = False
        masks[list(order[position:position + 2]), cell // w, cell % w] = True
    for index in range(train_count):
        masks = specs[index]["masks"]
        full = [layer for layer in range(panel_count) if np.all(masks[layer])]
        for layer in full[1:]:
            available = [cell for cell in range(masks.shape[1] * masks.shape[2])
                         if cell not in reserved[index]]
            cell = int(rng.choice(available))
            masks[layer, cell // masks.shape[2], cell % masks.shape[2]] = False
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, spec in enumerate(specs):
        h, w, orientation = spec["shape"]
        pair = _assemble(spec["masks"], spec["palette"], orientation, order)
        for _ in range(8):
            if _key(pair["input"]) not in seen and not _source_input(pair["input"]):
                break
            spec["masks"] = _masks(rng, h, w, "target_test", panel_count)
            pair = _assemble(spec["masks"], spec["palette"], orientation, order)
        else:
            raise RuntimeError("could not construct a distinct pair")
        seen.add(_key(pair["input"]))
        puzzle["train" if index < train_count else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    panel_count = _panel_count(puzzle["train"][0])
    assert 3 <= panel_count <= 5
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert not _source_input(inp)
            assert _panel_count(pair) == panel_count
            masks, _, _, orientation = _parse_input(inp, panel_count)
            assert out.shape == masks.shape[1:]
            if orientation == "vertical":
                assert inp.shape == (panel_count * out.shape[0], out.shape[1])
            else:
                assert inp.shape == (out.shape[0], panel_count * out.shape[1])
    assert len(_candidate_orders(puzzle["train"])) == 1
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"])


def self_test() -> None:
    def equal(a, b):
        return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[r], y[r])
                   for x, y in zip(a[s], b[s]) for r in ("input", "output"))
                   for s in ("train", "test"))
    assert equal(generate_puzzle(314159), generate_puzzle(314159))
    assert equal(generate_puzzle(271828, 7, 3), generate_puzzle(271828, 7, 3))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1000 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(99, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    counts, orders, palettes, shapes, layouts, orientations, panel_counts = set(), set(), set(), set(), set(), set(), set()
    within_palettes = within_shapes = full_rows_train = sparse_third_test = empty_or_full = 0
    mixed_orientation = orientation_shift = source_density_test = 0
    high_density_train = full_third_train = hidden_color = background_free = 0
    small_output = thirty_column_output = triple_overlap = 0
    first_all_comparisons = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        orders.add(_candidate_orders(puzzle["train"])[0])
        panel_count = _panel_count(puzzle["train"][0])
        panel_counts.add(panel_count)
        local_palettes, local_shapes = set(), set()
        local_orientations = {"train": set(), "test": set()}
        for split in ("train", "test"):
            for pair_index, pair in enumerate(puzzle[split]):
                masks, bg, fg, orientation = _parse_input(pair["input"], _panel_count(pair))
                local_orientations[split].add(orientation)
                palette = (bg,) + fg
                palettes.add(palette)
                local_palettes.add(palette)
                shapes.add(tuple(pair["output"].shape))
                local_shapes.add(tuple(pair["output"].shape))
                orientations.add(orientation)
                layouts.add((masks.shape, np.packbits(masks).tobytes()))
                empty_or_full += sum(not np.any(m) or np.all(m) for m in masks)
                hidden_color += len(set(int(x) for x in pair["output"].flat)) < panel_count + 1
                background_free += bg not in pair["output"]
                small_output += pair["output"].size <= 12
                thirty_column_output += 30 in pair["output"].shape
                triple_overlap += bool(np.any(np.sum(masks, axis=0) >= 3))
                if split == "train":
                    full_rows_train += int(any(np.any(np.all(m, axis=1)) for m in masks))
                    full_third_train += bool(np.any(np.all(masks[2], axis=1)))
                    high_density_train += bool(np.mean(masks[0]) > 0.72)
                    if pair_index == 0:
                        pure = np.sum(masks, axis=0) == 2
                        comparisons = sum(bool(np.any(masks[a] & masks[b] & pure))
                                          for a in range(panel_count) for b in range(a + 1, panel_count))
                        first_all_comparisons += comparisons == panel_count * (panel_count - 1) // 2
                else:
                    densities = np.sum(masks, axis=(1, 2))
                    sparse_third_test += int(densities[2] < np.median(densities))
                    source_density_test += bool(np.any(np.all(masks[2], axis=1))
                                                and densities[2] > max(np.delete(densities, 2)))
        within_palettes += len(local_palettes) > 1
        within_shapes += len(local_shapes) > 1
        mixed_orientation += len(local_orientations["train"] | local_orientations["test"]) > 1
        orientation_shift += local_orientations["train"].isdisjoint(local_orientations["test"])
    assert len({x[0] for x in counts}) >= 4 and len({x[1] for x in counts}) >= 3
    assert len(orders) >= 10 and SOURCE_ORDER in orders
    assert len(palettes) >= 50 and within_palettes >= 38
    assert len(shapes) >= 30 and within_shapes >= 38
    assert len(layouts) >= 150 and orientations == {"vertical", "horizontal"}
    assert panel_counts == {3, 4, 5} and mixed_orientation >= 10 and orientation_shift >= 5
    assert source_density_test > 0 and full_rows_train > 0 and sparse_third_test > 0
    assert high_density_train > 0 and full_third_train > 0 and empty_or_full > 0
    assert hidden_color > 0 and background_free > 0 and triple_overlap > 0
    assert small_output > 0 and thirty_column_output > 0
    assert first_all_comparisons < 21
    assert not equal(generate_puzzle(40001), generate_puzzle(40002))
