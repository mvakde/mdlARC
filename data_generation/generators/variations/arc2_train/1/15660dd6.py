"""Generator for three-band, keyed substitution tables."""

import hashlib
import numpy as np

_SOURCE = {
    "c4c60d73e6f2128094e282b308b0f574891c311fdfdb19ba14cd30efd1dbd832",
    "912ea07bbd6f8f48ac6858d409236514c06a7c0f426997e0211b81527e734666",
    "623c731796c25500352c736fe98a3fafb3ed867d93b40e84144917c053616ca4",
}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside [{low}, {high}]")
    return value


def _counts(seed, train, test):
    if train is not None:
        train = _count(train, "num_train", 2, 10)
    if test is not None:
        test = _count(test, "num_test", 1, 4)
    if train is None and test is None:
        return ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))[seed % 9]
    if train is None:
        choices = range(max(2, test + 1), 11)
        train = choices[seed % len(choices)]
    if test is None:
        choices = range(1, min(4, train - 1) + 1)
        test = choices[(seed // 3) % len(choices)]
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _roles(rng):
    colors = tuple(int(c) for c in rng.permutation(10))
    return colors[:6], colors[6:]


def _mask(rng, h, w, complex_bias):
    y, x = np.indices((h, w))
    for _ in range(24):
        style = int(rng.integers(9))
        phase = int(rng.integers(5))
        if style == 0:
            mask = rng.random((h, w)) < (0.5 if complex_bias else 0.28)
        elif style == 1:
            mask = (y + x + phase) % 2 == 0
        elif style == 2:
            mask = (y + phase) % 2 == 0
        elif style == 3:
            mask = (x + phase) % 2 == 0
        elif style == 4:
            mask = (2 * y + x + phase) % 3 == 0
        elif style == 5:
            radius = int(rng.integers(1, max(2, h + w - 2)))
            mask = abs(y - (h - 1) / 2) + abs(x - (w - 1) / 2) <= radius
        elif style == 6:
            y0, x0 = int(rng.integers(h)), int(rng.integers(w))
            y1 = int(rng.integers(y0 + 1, h + 1))
            x1 = int(rng.integers(x0 + 1, w + 1))
            mask = (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
        elif style == 7:
            mask = rng.random((h, w)) < 0.7
            mask |= mask[:, ::-1]
        else:
            mask = (y - x + phase) % 3 == 0
        for _ in range(int(rng.integers(3 if complex_bias else 2))):
            mask[int(rng.integers(h)), int(rng.integers(w))] ^= True
        if 0 < np.count_nonzero(mask) < mask.size:
            return mask
    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True
    return mask


def _widths(rng, columns):
    budget = 29 - columns
    widths = []
    for i in range(columns):
        high = min(9, budget - 4 * (columns - i - 1))
        width = int(rng.integers(4, high + 1))
        widths.append(width)
        budget -= width
    rng.shuffle(widths)
    if columns >= 2 and rng.random() < 0.48:
        trial = widths.copy()
        trial[1] = trial[0]
        if sum(trial) + columns + 1 <= 30:
            widths = trial
    return widths


def _make_pair(rng, split, index, roles, available):
    structural, base, placeholder, *keys = roles
    height = int(rng.integers(4, 8 if split == "train" else 9))
    columns = int(rng.integers(2 if split == "train" and index < 2 else 1, 6))
    widths = _widths(rng, columns)
    rows = [int(v) for v in rng.integers(0, 3, size=columns)]
    if split == "train" and index == 0:
        rows[0] = rows[1] = 2
    if split == "train" and index == 1:
        rows[0], rows[1] = 0, 1
    vocab_size = int(rng.integers(2 if split == "train" and index == 0 else 1, 5))
    vocab = [int(v) for v in rng.choice(available, size=vocab_size, replace=False)]
    accents = [int(v) for v in rng.choice(vocab, size=columns, replace=True)]
    if split == "train" and index == 0:
        accents[0], accents[1] = vocab[:2]
    if split == "train" and index == 1:
        accents[1] = accents[0]
    templates = []
    for width in widths:
        reusable = [t for t in templates if t.shape[1] == width]
        if reusable and rng.random() < 0.46:
            templates.append(reusable[int(rng.integers(len(reusable)))].copy())
        else:
            template = np.full((height, width), base, dtype=np.int8)
            mask = _mask(rng, height - 2, width - 2,
                         split == "test" and rng.random() < 0.6)
            template[1:-1, 1:-1] = np.where(mask, structural, placeholder)
            templates.append(template)
    grid = np.full((3 * height + 4, 1 + sum(w + 1 for w in widths)),
                   structural, dtype=np.int8)
    output = np.full((height, sum(widths) + columns - 1), structural, dtype=np.int8)
    starts = []
    x = 1
    for width in widths:
        starts.append(x)
        x += width + 1
    for band in range(3):
        y = 1 + band * (height + 1)
        grid[y:y + height, 0] = keys[band]
        for col, (width, x) in enumerate(zip(widths, starts)):
            if band == rows[col]:
                panel = np.full((height, width), base, dtype=np.int8)
                panel[1:-1, 1:-1] = accents[col]
            else:
                panel = templates[col]
            grid[y:y + height, x:x + width] = panel
    x = 0
    for col, template in enumerate(templates):
        decoded = template.copy()
        decoded[template == base] = keys[rows[col]]
        decoded[template == placeholder] = accents[col]
        output[:, x:x + widths[col]] = decoded
        x += widths[col] + 1
    return {"input": grid, "output": output}


def _border(panel, color):
    return (np.all(panel[0] == color) and np.all(panel[-1] == color)
            and np.all(panel[:, 0] == color) and np.all(panel[:, -1] == color))


def _decode_visible(grid):
    """Decode using visible separators, repeated panels, keys, and swatches."""
    total_h, total_w = grid.shape
    assert (total_h - 4) % 3 == 0
    height = (total_h - 4) // 3
    assert 4 <= height <= 8
    structural = int(grid[0, 0])
    for y in (0, height + 1, 2 * height + 2, 3 * height + 3):
        assert np.all(grid[y] == structural)
    separators = [x for x in range(1, total_w) if np.all(grid[:, x] == structural)]
    assert separators and separators[-1] == total_w - 1
    starts = [1] + [x + 1 for x in separators[:-1]]
    widths = [end - start for start, end in zip(starts, separators)]
    assert 1 <= len(widths) <= 5 and all(4 <= w <= 9 for w in widths)
    base = int(grid[1, 1])
    keys = []
    for band in range(3):
        y = 1 + band * (height + 1)
        key = int(grid[y, 0])
        assert np.all(grid[y:y + height, 0] == key)
        keys.append(key)
    assert len({structural, base, *keys}) == 5
    output = np.full((height, sum(widths) + len(widths) - 1),
                     structural, dtype=np.int8)
    placeholder = None
    rows, accents, templates = [], [], []
    out_x = 0
    for width, start in zip(widths, starts):
        panels = [grid[1 + band * (height + 1):1 + band * (height + 1) + height,
                       start:start + width] for band in range(3)]
        assert all(_border(p, base) for p in panels)
        matches = [(a, b) for a in range(3) for b in range(a + 1, 3)
                   if np.array_equal(panels[a], panels[b])]
        assert len(matches) == 1
        a, b = matches[0]
        row = 3 - a - b
        template, swatch = panels[a], panels[row]
        interior = template[1:-1, 1:-1]
        extra = set(int(c) for c in np.unique(interior)) - {structural}
        assert len(extra) == 1 and np.any(interior == structural)
        role = extra.pop()
        if placeholder is None:
            placeholder = role
        assert role == placeholder and role not in {base, *keys}
        accent = int(swatch[1, 1])
        assert np.all(swatch[1:-1, 1:-1] == accent)
        assert accent not in {structural, base, placeholder, *keys}
        decoded = template.copy()
        decoded[template == base] = keys[row]
        decoded[template == placeholder] = accent
        output[:, out_x:out_x + width] = decoded
        out_x += width + 1
        rows.append(row)
        accents.append(accent)
        templates.append(template.copy())
    return output, {"height": height, "widths": tuple(widths), "rows": tuple(rows),
                    "accents": tuple(accents),
                    "roles": (structural, base, placeholder, *keys),
                    "templates": templates}


def _signature(pair):
    raw = pair["input"].astype(np.int8).tobytes() + pair["output"].astype(np.int8).tobytes()
    return hashlib.sha256(raw).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert 0 <= grid.min() and grid.max() <= 9
            identifier = (pair["input"].shape, pair["input"].astype(np.int8).tobytes())
            assert identifier not in seen
            seen.add(identifier)
            assert _signature(pair) not in _SOURCE
            predicted, features = _decode_visible(pair["input"])
            assert np.array_equal(predicted, pair["output"])
            assert not np.array_equal(pair["input"], pair["output"])
            analyses[split].append(features)
    first, second = analyses["train"][:2]
    assert len(first["rows"]) >= 2 and first["rows"][0] == first["rows"][1]
    assert first["accents"][0] != first["accents"][1]
    assert len(second["rows"]) >= 2 and second["rows"][0] != second["rows"][1]
    assert second["accents"][0] == second["accents"][1]
    assert {r for f in analyses["train"] for r in f["rows"]} == {0, 1, 2}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    stable_palette = bool(rng.random() < 0.25)
    fixed_roles, fixed_accents = _roles(rng)
    puzzle = {"train": [], "test": []}
    prior_roles = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if stable_palette:
                roles, available = fixed_roles, fixed_accents
            else:
                roles, available = _roles(rng)
                if split == "train" and index == 1 and roles == prior_roles:
                    roles = (roles[0], roles[1], roles[2], roles[4], roles[3], roles[5])
            prior_roles = roles
            for _ in range(8):
                pair = _make_pair(rng, split, index, roles, available)
                if not any(np.array_equal(pair["input"], p["input"])
                           for s in ("train", "test") for p in puzzle[s]):
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    layouts, roles, row_patterns, accent_patterns = set(), set(), set(), set()
    heights, columns, train_counts, test_counts = set(), set(), set(), set()
    repeated_templates = shared_accents = repeated_rows = 0
    varying_palettes = stable_palettes = 0
    varied_encoding = varied_keys = varied_train_heights = varied_train_columns = 0
    split_coverage = {"train": set(), "test": set()}
    unequal = rectangular = sparse = complex_patterns = 0
    test_shifts = set()
    color_counts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_roles = set()
        train_heights, train_columns = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                predicted, f = _decode_visible(pair["input"].copy())
                assert np.array_equal(predicted, pair["output"])
                layouts.add((f["height"], f["widths"]))
                heights.add(f["height"])
                columns.add(len(f["widths"]))
                split_coverage[split].add(len(f["widths"]))
                roles.add(f["roles"])
                puzzle_roles.add(f["roles"])
                row_patterns.add(f["rows"])
                accent_patterns.add(f["accents"])
                color_counts.add(len(np.unique(pair["input"])))
                repeated_rows += len(set(f["rows"])) < len(f["rows"])
                shared_accents += len(set(f["accents"])) < len(f["accents"])
                unequal += len(set(f["widths"])) > 1
                rectangular += any(w != f["height"] for w in f["widths"])
                repeated_templates += any(np.array_equal(a, b)
                                          for i, a in enumerate(f["templates"])
                                          for b in f["templates"][i + 1:])
                for template in f["templates"]:
                    mask = template[1:-1, 1:-1] == f["roles"][0]
                    count = int(mask.sum())
                    sparse += min(count, mask.size - count) <= 2
                    transitions = (np.count_nonzero(mask[:, 1:] != mask[:, :-1])
                                   + np.count_nonzero(mask[1:, :] != mask[:-1, :]))
                    complex_patterns += transitions >= 2 * min(mask.shape)
                if split == "train":
                    train_heights.add(f["height"])
                    train_columns.add(len(f["widths"]))
                else:
                    max_train_height = max(p["input"].shape[0] for p in puzzle["train"])
                    relation = ("larger" if pair["input"].shape[0] > max_train_height
                                else "same" if pair["input"].shape[0] == max_train_height
                                else "smaller")
                    test_shifts.add(relation)
        varying_palettes += len(puzzle_roles) > 1
        stable_palettes += len(puzzle_roles) == 1
        varied_encoding += len({r[:3] for r in puzzle_roles}) > 1
        varied_keys += len({r[3:] for r in puzzle_roles}) > 1
        varied_train_heights += len(train_heights) > 1
        varied_train_columns += len(train_columns) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(layouts) >= 35 and len(roles) >= 40
    assert heights == {4, 5, 6, 7, 8} and columns == {1, 2, 3, 4, 5}
    assert len(row_patterns) >= 15 and len(accent_patterns) >= 20
    assert repeated_templates and shared_accents and repeated_rows
    assert varying_palettes and stable_palettes
    assert varied_encoding and varied_keys and varied_train_heights and varied_train_columns
    assert split_coverage["train"] == {1, 2, 3, 4, 5}
    assert split_coverage["test"] == {1, 2, 3, 4, 5}
    assert unequal and rectangular and sparse and complex_patterns
    assert test_shifts == {"larger", "same", "smaller"}
    assert color_counts == {7, 8, 9, 10}
    for counts in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(210 + sum(counts), *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
    for counts in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(91, *counts))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    a, b = generate_puzzle(123, 4, 2), generate_puzzle(123, 4, 2)
    assert all(np.array_equal(p[key], q[key]) for split in ("train", "test")
               for p, q in zip(a[split], b[split]) for key in ("input", "output"))
    assert _signature(a["train"][0]) != _signature(generate_puzzle(124, 4, 2)["train"][0])
