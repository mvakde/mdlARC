"""Generate framed positive-clue template-completion ARC puzzles."""

import hashlib
import itertools

import numpy as np


_SOURCE_PAIR_HASHES = {
    "b158b71c893f7fb5e697a21b7f98f34bd5d707fa3c01623a4c1454eca5dd9549",
    "2607d2d31ab1b53ff117b387dbcfa8f5741b06bd53d5ea21d1e0a3a433fedb6e",
    "939babb18e581bb39b95f8c6ebf422c0d02ff1647820935427ae88f5b178bd9e",
    "b65508d3c4a6e67b7268a31f03b872409b9e174ff878f0ada7ebd85f636a1f65",
}


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = None if num_train is None else _integer(num_train, "num_train")
    test = None if num_test is None else _integer(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is None and test is None:
        test = 1 + seed % 4
        train = list(range(max(2, test + 1), 11))[(seed // 4) % (11 - max(2, test + 1))]
    elif train is None:
        choices = list(range(test + 1, 11))
        train = choices[seed % len(choices)]
    elif test is None:
        choices = list(range(1, min(4, train - 1) + 1))
        test = choices[seed % len(choices)]
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _bounded(mask):
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _sample_mask(rng, height, width, seen, previous):
    area = height * width
    for _ in range(120):
        style = int(rng.integers(6))
        if style == 0:
            mask = rng.random((height, width)) < rng.uniform(.18, .36)
        elif style == 1:
            mask = rng.random((height, width)) < rng.uniform(.66, .88)
        elif style == 2:
            mask = rng.random((height, width)) < rng.uniform(.28, .68)
            mask[int(rng.integers(height)), :] = True
        elif style == 3:
            mask = np.zeros((height, width), dtype=bool)
            mask[int(rng.integers(height)), :] = True
            mask[:, int(rng.integers(width))] = True
            mask |= rng.random((height, width)) < .16
        elif style == 4 and previous:
            mask = previous[int(rng.integers(len(previous)))].copy()
            for _ in range(int(rng.integers(1, 4))):
                r = int(rng.integers(height))
                c = int(rng.integers(width))
                mask[r, c] = ~mask[r, c]
        else:
            mask = rng.random((height, width)) < rng.uniform(.33, .57)
        if style != 4:
            mask[0, int(rng.integers(width))] = True
            mask[-1, int(rng.integers(width))] = True
            mask[int(rng.integers(height)), 0] = True
            mask[int(rng.integers(height)), -1] = True
        total = int(mask.sum())
        if 3 <= total < area and _bounded(mask) and mask.tobytes() not in seen:
            return mask
    raise RuntimeError("could not sample a distinct template")


def _clue_choices(masks):
    choices = {"single": [], "joint": [], "pair": [], "partial": []}
    for target_index, target in enumerate(masks):
        cells = [tuple(map(int, x)) for x in np.argwhere(target)]
        if len(cells) < 2:
            continue
        singles = []
        for cell in cells:
            matches = [i for i, mask in enumerate(masks) if mask[cell]]
            if matches == [target_index]:
                singles.append(cell)
                choices["single"].append((target_index, (cell,)))
        pairs = []
        for first, second in itertools.combinations(cells, 2):
            matches = [i for i, mask in enumerate(masks) if mask[first] and mask[second]]
            if matches == [target_index]:
                pairs.append((first, second))
                choices["pair"].append((target_index, (first, second)))
                if any(i != target_index and mask[first] for i, mask in enumerate(masks)) and any(
                    i != target_index and mask[second] for i, mask in enumerate(masks)
                ):
                    choices["joint"].append((target_index, (first, second)))
        if len(cells) >= 4 and (singles or pairs):
            base = (singles[0],) if singles else pairs[0]
            choices["partial"].append((target_index, base))
    return choices


def _make_catalog(rng, height, width, count, preferred, sparse_target):
    for attempt in range(100):
        masks = []
        seen = set()
        try:
            for _ in range(count):
                mask = _sample_mask(rng, height, width, seen, masks)
                masks.append(mask)
                seen.add(mask.tobytes())
        except RuntimeError:
            continue
        options = _clue_choices(masks)
        categories = (preferred, "joint", "partial", "pair", "single") if attempt < 70 else (
            "pair", "single", "partial", "joint"
        )
        if sparse_target:
            options = {
                kind: [item for item in entries if masks[item[0]].mean() <= .44]
                for kind, entries in options.items()
            }
        selected = next((kind for kind in categories if options[kind]), None)
        if selected is None:
            continue
        targets = sorted({item[0] for item in options[selected]})
        target_index = targets[int(rng.integers(len(targets)))]
        target_options = [item for item in options[selected] if item[0] == target_index]
        _, clue_base = target_options[int(rng.integers(len(target_options)))]
        clues = list(clue_base)
        if selected == "partial":
            cells = [tuple(map(int, x)) for x in np.argwhere(masks[target_index]) if tuple(x) not in clues]
            rng.shuffle(cells)
            desired = int(rng.integers(3, min(7, int(masks[target_index].sum()) - 1) + 1))
            clues.extend(cells[:desired - len(clues)])
        elif selected == "pair" and rng.random() < .2 and int(masks[target_index].sum()) >= 4:
            extras = [tuple(map(int, x)) for x in np.argwhere(masks[target_index]) if tuple(x) not in clues]
            clues.append(extras[int(rng.integers(len(extras)))])
        return masks, target_index, tuple(clues)
    raise RuntimeError("could not make an identifiable catalog")


def _layout(rng, height, width, scale, count):
    frame_h, frame_w = height * scale + 2, width * scale + 2
    for _ in range(70):
        cols = int(rng.integers(1, count + 1))
        rows = (count + cols - 1) // cols
        xbase = [0]
        ybase = [0]
        for _ in range(1, cols):
            xbase.append(xbase[-1] + width + int(rng.integers(1, 5)))
        for _ in range(1, rows):
            ybase.append(ybase[-1] + height + int(rng.integers(1, 5)))
        slots = []
        for i in range(count):
            row, col = divmod(i, cols)
            slots.append((ybase[row] + int(rng.integers(2)), xbase[col] + int(rng.integers(2))))
        bank_h = max(y + height for y, _ in slots)
        bank_w = max(x + width for _, x in slots)
        gap = int(rng.integers(2, 5))
        direction = int(rng.integers(4))
        if direction in (0, 2):
            content_h = bank_h + gap + frame_h
            content_w = max(bank_w, frame_w)
        else:
            content_h = max(bank_h, frame_h)
            content_w = bank_w + gap + frame_w
        if content_h > 28 or content_w > 28:
            continue
        if direction == 0:
            bank_y, frame_y = 0, bank_h + gap
            bank_x = int(rng.integers(content_w - bank_w + 1))
            frame_x = int(rng.integers(content_w - frame_w + 1))
        elif direction == 2:
            frame_y, bank_y = 0, frame_h + gap
            bank_x = int(rng.integers(content_w - bank_w + 1))
            frame_x = int(rng.integers(content_w - frame_w + 1))
        elif direction == 1:
            frame_x, bank_x = 0, frame_w + gap
            bank_y = int(rng.integers(content_h - bank_h + 1))
            frame_y = int(rng.integers(content_h - frame_h + 1))
        else:
            bank_x, frame_x = 0, bank_w + gap
            bank_y = int(rng.integers(content_h - bank_h + 1))
            frame_y = int(rng.integers(content_h - frame_h + 1))
        extra_h = int(rng.integers(2, min(6, 30 - content_h) + 1))
        extra_w = int(rng.integers(2, min(6, 30 - content_w) + 1))
        top = int(rng.integers(1, extra_h))
        left = int(rng.integers(1, extra_w))
        origins = [(top + bank_y + y, left + bank_x + x) for y, x in slots]
        return (content_h + extra_h, content_w + extra_w), origins, (
            top + frame_y, left + frame_x, frame_h, frame_w
        )
    raise RuntimeError("no layout fits this specification")


def _compose_pair(rng, spec, background, fill, palette, preferred, sparse_target):
    height, width, scale, count = spec
    masks, target_index, clues = _make_catalog(rng, height, width, count, preferred, sparse_target)
    shape, origins, frame = _layout(rng, height, width, scale, count)
    available = [color for color in palette if color not in (background, fill)]
    rng.shuffle(available)
    border = available[0]
    inks = available[1:count + 1]
    grid = np.full(shape, background, dtype=np.int8)
    for mask, (y, x), ink in zip(masks, origins, inks):
        patch = grid[y:y + height, x:x + width]
        patch[mask] = ink
    y, x, frame_h, frame_w = frame
    grid[y, x:x + frame_w] = border
    grid[y + frame_h - 1, x:x + frame_w] = border
    grid[y:y + frame_h, x] = border
    grid[y:y + frame_h, x + frame_w - 1] = border
    for row, col in clues:
        cy, cx = y + 1 + row * scale, x + 1 + col * scale
        grid[cy:cy + scale, cx:cx + scale] = fill
    output = np.full((frame_h, frame_w), background, dtype=np.int8)
    output[0, :] = output[-1, :] = border
    output[:, 0] = output[:, -1] = border
    enlarged = np.repeat(np.repeat(masks[target_index], scale, axis=0), scale, axis=1)
    output[1:-1, 1:-1] = np.where(enlarged, fill, background)
    return {"input": grid, "output": output}


def _find_frame(grid, background):
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        coords = np.argwhere(grid == color)
        y0, x0 = map(int, coords.min(axis=0))
        y1, x1 = map(int, coords.max(axis=0))
        if y1 - y0 < 2 or x1 - x0 < 2:
            continue
        actual = grid[y0:y1 + 1, x0:x1 + 1] == color
        expected = np.zeros(actual.shape, dtype=bool)
        expected[0, :] = expected[-1, :] = True
        expected[:, 0] = expected[:, -1] = True
        if np.array_equal(actual, expected):
            candidates.append(((y1 - y0 + 1) * (x1 - x0 + 1), color, (y0, x0, y1, x1)))
    assert candidates, "no complete one-cell frame"
    candidates.sort(reverse=True)
    assert len(candidates) == 1 or candidates[0][0] > candidates[1][0], "ambiguous frame"
    return candidates[0][1], candidates[0][2]


def _separated(a, b):
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _analyze_pair(input_grid, output_grid=None):
    background = int(input_grid[0, 0])
    assert all(int(input_grid[y, x]) == background for y, x in ((0, -1), (-1, 0), (-1, -1)))
    border, frame = _find_frame(input_grid, background)
    y0, x0, y1, x1 = frame
    outside = np.ones(input_grid.shape, dtype=bool)
    outside[y0:y1 + 1, x0:x1 + 1] = False
    colors = [int(color) for color in np.unique(input_grid[outside]) if color != background]
    assert 2 <= len(colors) <= 7
    assert border not in colors
    masks, boxes = [], []
    for color in colors:
        coords = np.argwhere(input_grid == color)
        assert np.all(outside[coords[:, 0], coords[:, 1]])
        a, b = map(int, coords.min(axis=0))
        c, d = map(int, coords.max(axis=0))
        box = (a, b, c, d)
        crop = input_grid[a:c + 1, b:d + 1]
        assert set(map(int, np.unique(crop))) <= {background, color}
        mask = crop == color
        assert _bounded(mask)
        masks.append(mask)
        boxes.append(box)
    assert len({mask.shape for mask in masks}) == 1
    assert len({mask.tobytes() for mask in masks}) == len(masks)
    assert all(_separated(box, frame) for box in boxes)
    assert all(_separated(a, b) for a, b in itertools.combinations(boxes, 2))
    height, width = masks[0].shape
    frame_h, frame_w = y1 - y0 + 1, x1 - x0 + 1
    assert (frame_h - 2) % height == (frame_w - 2) % width == 0
    scale = (frame_h - 2) // height
    assert scale == (frame_w - 2) // width and 2 <= scale <= 5
    interior = input_grid[y0 + 1:y1, x0 + 1:x1]
    fills = [int(color) for color in np.unique(interior) if color != background]
    assert len(fills) == 1 and fills[0] != border and fills[0] not in colors
    fill = fills[0]
    clues = np.zeros((height, width), dtype=bool)
    for row in range(height):
        for col in range(width):
            block = interior[row * scale:(row + 1) * scale, col * scale:(col + 1) * scale]
            assert np.all(block == fill) or np.all(block == background)
            clues[row, col] = np.all(block == fill)
    assert clues.any()
    matches = [i for i, mask in enumerate(masks) if np.all(mask[clues])]
    assert len(matches) == 1, "positive clues do not uniquely identify a candidate"
    selected = masks[matches[0]]
    assert int(selected.sum()) > int(clues.sum())
    expected = np.full((frame_h, frame_w), background, dtype=np.int8)
    expected[0, :] = expected[-1, :] = border
    expected[:, 0] = expected[:, -1] = border
    expected[1:-1, 1:-1] = np.where(
        np.repeat(np.repeat(selected, scale, axis=0), scale, axis=1), fill, background
    )
    if output_grid is not None:
        assert np.array_equal(output_grid, expected), "output differs from completed tight frame crop"
    clue_cells = [tuple(map(int, x)) for x in np.argwhere(clues)]
    joint = len(clue_cells) == 2 and all(
        any(i != matches[0] and mask[cell] for i, mask in enumerate(masks)) for cell in clue_cells
    )
    adjacent = len(clue_cells) >= 2 and any(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in itertools.combinations(clue_cells, 2)
    )
    near = min(int(np.count_nonzero(selected != mask)) for i, mask in enumerate(masks) if i != matches[0])
    bank = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    relation = "above" if bank[2] < y0 else "below" if bank[0] > y1 else "left" if bank[3] < x0 else "right"
    return {
        "background": background, "fill": fill, "border": border, "spec": (height, width, scale, len(masks)),
        "clues": len(clue_cells), "joint": joint, "adjacent": adjacent,
        "density": float(selected.mean()), "components": _components(selected), "near_distance": near,
        "relation": relation, "multirow": any(a[2] < b[0] or b[2] < a[0] for a, b in itertools.combinations(boxes, 2))
        and any(a[3] < b[1] or b[3] < a[1] for a, b in itertools.combinations(boxes, 2)),
        "frame": frame, "boxes": tuple(boxes), "output": expected,
    }


def _source_hash(pair):
    raw = pair["input"].astype(np.uint8).tobytes() + pair["output"].astype(np.uint8).tobytes()
    return hashlib.sha256(raw).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            signature = (pair["input"].shape, pair["input"].tobytes())
            assert signature not in seen_inputs
            seen_inputs.add(signature)
            assert _source_hash(pair) not in _SOURCE_PAIR_HASHES, "exact source pair copied"
            infos[split].append(_analyze_pair(pair["input"], pair["output"]))
    assert len({info["spec"] for info in infos["train"]}) >= 2
    assert len({info["relation"] for info in infos["train"]}) >= 2
    # The task has no hidden palette or direction convention. Every test input
    # exposes its own colors, scale, candidates, and positive clues.
    assert all(info["clues"] < round(info["density"] * info["spec"][0] * info["spec"][1]) for split in infos for info in infos[split])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    stable_background = rng.random() < .28
    stable_fill = rng.random() < .25
    base_background, base_fill = palette[:2]
    puzzle = {"train": [], "test": []}
    train_info = []
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for attempt in range(400):
                height = int(rng.integers(3, 6))
                width = int(rng.integers(3, 6))
                scale = int(rng.integers(2, 6))
                candidates = int(rng.integers(2, 8))
                spec = (height, width, scale, candidates)
                if split == "train" and index == 1 and train_info:
                    first = train_info[0]["spec"]
                    if (height == width) == (first[0] == first[1]) and scale == first[2]:
                        continue
                if stable_background:
                    background = base_background
                elif split == "train" and index == 1 and train_info:
                    background = palette[int(rng.integers(1, 10))]
                    if background == train_info[0]["background"]:
                        continue
                else:
                    background = palette[int(rng.integers(10))]
                if stable_fill:
                    fill = base_fill
                elif split == "test" and train_info:
                    familiar = [info["fill"] for info in train_info if info["fill"] != background]
                    unseen = [color for color in palette if color != background and color not in familiar]
                    pool = unseen if unseen and rng.random() < .5 else familiar or unseen
                    fill = pool[int(rng.integers(len(pool)))]
                else:
                    fill = palette[int(rng.integers(10))]
                if fill == background:
                    continue
                if split == "train" and index == 1 and not stable_fill and fill == train_info[0]["fill"]:
                    continue
                preferred = str(rng.choice(
                    ["single", "joint", "pair", "partial"],
                    p=[.12, .53, .15, .20] if split == "test" else [.24, .20, .28, .28],
                ))
                if preferred == "joint" and candidates < 3:
                    preferred = "pair"
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                try:
                    pair = _compose_pair(pair_rng, spec, background, fill, palette, preferred, rng.random() < .2)
                    info = _analyze_pair(pair["input"], pair["output"])
                except (RuntimeError, AssertionError):
                    continue
                if split == "train" and index == 1 and train_info and info["relation"] == train_info[0]["relation"]:
                    continue
                if any(np.array_equal(pair["input"], old["input"]) for side in puzzle.values() for old in side):
                    continue
                if _source_hash(pair) in _SOURCE_PAIR_HASHES:
                    continue
                puzzle[split].append(pair)
                if split == "train":
                    train_info.append(info)
                break
            else:
                raise RuntimeError(f"could not construct {split} pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(a, b):
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
            for x, y in zip(a[split], b[split])
        ) for split in ("train", "test")
    )


def self_test() -> None:
    assert _equal_puzzles(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    assert not _equal_puzzles(generate_puzzle(12345, 7, 3), generate_puzzle(12346, 7, 3))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(8000 + (train or 0) + (test or 0), train, test))
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(99, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
    train_counts, test_counts = set(), set()
    specs, shapes, layouts = set(), set(), set()
    clue_counts, backgrounds, fills, densities, components = set(), set(), set(), set(), set()
    joint = adjacent = separated = multirow = near = 0
    sparse = dense = disconnected = single = complete = holes = runs = 0
    familiar = new = 0
    stable_backgrounds = varying_backgrounds = 0
    stable_fills = varying_fills = 0
    aspects = set()
    puzzle_signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_signatures.add(tuple((p["input"].shape, p["input"].tobytes()) for side in puzzle.values() for p in side))
        information = {"train": [], "test": []}
        for split, side in puzzle.items():
            for pair in side:
                info = _analyze_pair(pair["input"], pair["output"])
                information[split].append(info)
                specs.add(info["spec"])
                shapes.add(pair["input"].shape)
                layouts.add((info["frame"], info["boxes"], info["relation"]))
                clue_counts.add(info["clues"])
                backgrounds.add(info["background"])
                fills.add(info["fill"])
                densities.add(round(info["density"], 1))
                components.add(info["components"])
                joint += info["joint"]
                adjacent += info["adjacent"]
                separated += info["clues"] == 2 and not info["adjacent"]
                multirow += info["multirow"]
                near += info["near_distance"] <= 2
                sparse += info["density"] <= .4
                dense += info["density"] >= .7
                disconnected += info["components"] > 1
                single += info["clues"] == 1
                complete += info["clues"] >= 4
                height, width, scale, _ = info["spec"]
                aspects.add("square" if height == width else "tall" if height > width else "wide")
                mask = pair["output"][1:-1:scale, 1:-1:scale] == info["fill"]
                holes += any(
                    not mask[row, col] and mask[row - 1, col] and mask[row + 1, col]
                    and mask[row, col - 1] and mask[row, col + 1]
                    for row in range(1, height - 1) for col in range(1, width - 1)
                )
                runs += any(
                    np.convolve(line.astype(int), np.ones(3, dtype=int), "valid").max() == 3
                    for line in list(mask) + list(mask.T)
                )
        all_info = information["train"] + information["test"]
        stable_backgrounds += len({info["background"] for info in all_info}) == 1
        varying_backgrounds += len({info["background"] for info in all_info}) > 1
        stable_fills += len({info["fill"] for info in all_info}) == 1
        varying_fills += len({info["fill"] for info in all_info}) > 1
        train_fills = {info["fill"] for info in information["train"]}
        familiar += sum(info["fill"] in train_fills for info in information["test"])
        new += sum(info["fill"] not in train_fills for info in information["test"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_signatures) == 42
    assert len(specs) >= 45 and len(shapes) >= 35 and len(layouts) >= 100
    assert len(clue_counts) >= 5 and len(backgrounds) >= 8 and len(fills) >= 8
    assert min(densities) <= .4 and max(densities) >= .7
    assert len(components) >= 3
    assert joint >= 8 and adjacent >= 8 and multirow >= 20 and near >= 5
    assert separated >= 8 and sparse >= 20 and dense >= 20
    assert disconnected >= 20 and single >= 10 and complete >= 10
    assert holes >= 5 and runs >= 20 and aspects == {"square", "tall", "wide"}
    assert familiar >= 10 and new >= 10
    assert stable_backgrounds >= 3 and varying_backgrounds >= 3
    assert stable_fills >= 3 and varying_fills >= 3
    assert any(
        pair["input"].shape == (27, 21)
        for side in generate_puzzle(80).values() for pair in side
    )
