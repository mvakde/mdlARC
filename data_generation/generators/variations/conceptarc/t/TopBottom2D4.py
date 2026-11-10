"""Generate vertical stacks whose first and last objects disappear."""

import hashlib
from collections import Counter
import numpy as np

# Shape and input bytes of the six original source inputs. Exact dimensions remain legal.
_SOURCE_INPUTS = {
    ((14, 10), "f699575f9e199d005fe8aa06466550ec189bf3d25a131a35d53b0a6aa26ce2a5"),
    ((16, 12), "c44c042afcbddffbc2bbe396afb9840d4be74d54ff704e8f5bcddc45f09ac996"),
    ((13, 6), "372af4888b2dc6c79c75fa3e35323782ed42837fcd1b14628540744a9b2fcf4a"),
    ((11, 10), "68ce31d41042857ee38538fc417dedb030150b7bc28678abec8302638c66231e"),
    ((12, 8), "4cb7fe22efe8dfadd0cb78fcbf5506a0bc0386591626fe6d8ed1180519f737da"),
    ((12, 6), "a9798a32851f26f4eb4fd8c3ecf7e5a322b91646f4201d2347eb48fcfc65247c"),
}


def _checked_count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return int(value)


def _source_input(grid):
    return (grid.shape, hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest()) in _SOURCE_INPUTS


def _sample_layout(rng, count):
    scale = float(rng.uniform(0.25, 1.25))
    regular_max = max(3, 2 + round(6 * scale))
    tall_max = min(16, regular_max + 2 + round(4 * scale))
    heights = []
    for _ in range(count):
        draw = rng.random()
        if draw < 0.18:
            heights.append(1)
        elif draw < 0.88:
            heights.append(int(rng.integers(2, regular_max + 1)))
        else:
            heights.append(int(rng.integers(regular_max + 1, tall_max + 1)))
    gap_max = max(1, round(4 * scale))
    margin_max = max(1, round(6 * scale))
    gaps = [0 if rng.random() < 0.30 else int(rng.integers(1, gap_max + 1))
            for _ in range(count - 1)]
    margins = [int(rng.integers(0, margin_max + 1)) for _ in range(2)]
    # Trim excess space, keeping each object at least one row tall.
    while sum(heights) + sum(gaps) + sum(margins) > 30:
        choices = ([(0, i) for i, h in enumerate(heights) if h > 1]
                   + [(1, i) for i, g in enumerate(gaps) if g > 0]
                   + [(2, i) for i, m in enumerate(margins) if m > 0])
        kind, index = choices[int(rng.integers(len(choices)))]
        (heights, gaps, margins)[kind][index] -= 1
    total = sum(heights) + sum(gaps) + sum(margins)
    if total < 4:
        margins[1] += 4 - total
    return heights, gaps, margins


def _row_widths(rng, height, maximum):
    if height == 1 or maximum == 1:
        return [maximum] * height
    mode = int(rng.integers(0, 100))
    if mode < 17:
        return [maximum] * height
    if mode < 31:
        values = np.linspace(1, maximum, height).astype(int).tolist()
        return values if rng.random() < 0.5 else values[::-1]
    if mode < 45:
        return [max(1, maximum - abs(2 * i - height + 1) * maximum // max(1, height - 1))
                for i in range(height)]
    if mode < 57:
        widths = [int(rng.integers(1, max(2, maximum // 2 + 1))) for _ in range(height)]
        widths[int(rng.integers(0, 2)) * (height - 1)] = maximum
        return widths
    widths = [int(rng.integers(1, maximum + 1)) for _ in range(height)]
    widths[int(rng.integers(height))] = maximum
    if height >= 3 and rng.random() < 0.55:
        widths[int(rng.integers(1, height - 1))] = int(rng.integers(1, min(3, maximum + 1)))
    return widths


def _draw_object(grid, top, height, color, rng, simpler=False):
    width = grid.shape[1]
    roll = rng.random()
    if roll < 0.24:
        envelope = int(rng.integers(1, min(4, width) + 1))
    elif roll < (0.52 if simpler else 0.57):
        envelope = int(rng.integers(max(2, width // 2), width + 1))
    else:
        envelope = int(rng.integers(1, width + 1))
    box_left = int(rng.integers(0, width - envelope + 1))
    widths = _row_widths(rng, height, envelope)
    previous = None
    for offset, row_width in enumerate(widths):
        low, high = 0, envelope - row_width
        if previous is not None:
            low = max(low, previous[0] - row_width + 1)
            high = min(high, previous[1])
        left = int(rng.integers(low, high + 1))
        grid[top + offset, box_left + left:box_left + left + row_width] = color
        previous = (left, left + row_width - 1)


def _background(rng, anchor):
    return anchor if rng.random() < 0.42 else int(rng.integers(0, 10))


def _make_pair(rng, count, background, regime):
    for _ in range(25):
        heights, gaps, margins = _sample_layout(rng, count)
        width = int(rng.integers(4, 31))
        height = sum(heights) + sum(gaps) + sum(margins)
        grid = np.full((height, width), background, dtype=np.int8)
        colors = [int(v) for v in rng.choice([v for v in range(10) if v != background],
                                              size=count, replace=False)]
        top = margins[0]
        for index, (object_height, color) in enumerate(zip(heights, colors)):
            _draw_object(grid, top, object_height, color, rng, simpler=regime == "source_train")
            top += object_height
            if index < count - 1:
                top += gaps[index]
        unique, frequencies = np.unique(grid, return_counts=True)
        background_frequency = int(frequencies[np.flatnonzero(unique == background)[0]])
        if background_frequency <= max(int(n) for c, n in zip(unique, frequencies)
                                       if int(c) != background) or _source_input(grid):
            continue
        output = grid.copy()
        output[(grid == colors[0]) | (grid == colors[-1])] = background
        return {"input": grid, "output": output}
    # Sparse deterministic fallback makes construction total for every seed.
    width = 10
    grid = np.full((2 * count + 2, width), background, dtype=np.int8)
    colors = [int(v) for v in rng.choice([v for v in range(10) if v != background],
                                          size=count, replace=False)]
    for i, color in enumerate(colors):
        grid[2 * i + 1, int(rng.integers(width))] = color
    output = grid.copy()
    output[(grid == colors[0]) | (grid == colors[-1])] = background
    return {"input": grid, "output": output}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None:
        num_train = _checked_count(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # Vertical order and endpoint removal are the only puzzle-level conventions.
    # This anchor biases local backgrounds, but never determines an output.
    anchor = 0 if rng.random() < 0.5 else int(rng.integers(1, 10))
    alternate = int(rng.choice([c for c in range(10) if c != anchor]))
    first = int(rng.integers(3, 9))
    second = int(rng.choice([c for c in range(2, 10) if c != first]))
    train_counts = [first, second]
    train_counts += [int(rng.integers(2, 10)) for _ in range(num_train - 2)]
    rng.shuffle(train_counts)
    train_backgrounds = [anchor, alternate]
    train_backgrounds += [_background(rng, anchor) for _ in range(num_train - 2)]
    rng.shuffle(train_backgrounds)

    puzzle = {"train": [], "test": []}
    seen = set()
    for split, counts, backgrounds in (
        ("train", train_counts, train_backgrounds),
        ("test", [], []),
    ):
        if split == "test":
            observed = set(train_counts)
            for _ in range(num_test):
                same = rng.random() < 0.48
                options = sorted(observed if same else set(range(2, 10)) - observed)
                if not options:
                    options = list(range(2, 10))
                counts.append(int(rng.choice(options)))
                backgrounds.append(_background(rng, anchor))
        for count, background in zip(counts, backgrounds):
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(pair_rng, count, background,
                                  "source_train" if split == "train" else "target_test")
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise AssertionError("Could not construct a unique pair")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _ordered_objects(grid):
    colors, frequencies = np.unique(grid, return_counts=True)
    assert len(colors) >= 3
    assert frequencies.max() > np.partition(frequencies, -2)[-2]
    background = int(colors[int(np.argmax(frequencies))])
    objects = []
    for value in colors:
        color = int(value)
        if color == background:
            continue
        rows = np.flatnonzero(np.any(grid == color, axis=1))
        assert len(rows) and len(rows) == int(rows[-1] - rows[0] + 1)
        spans = []
        for row in rows:
            columns = np.flatnonzero(grid[row] == color)
            assert len(columns) == int(columns[-1] - columns[0] + 1)
            spans.append((int(columns[0]), int(columns[-1])))
        for earlier, later in zip(spans, spans[1:]):
            assert earlier[0] <= later[1] and later[0] <= earlier[1]
        objects.append((int(rows[0]), int(rows[-1]), color, tuple(spans)))
    objects.sort()
    assert 2 <= len(objects) <= 9
    for earlier, later in zip(objects, objects[1:]):
        assert earlier[1] < later[0]
    return background, objects


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_counts, train_backgrounds = [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert not _source_input(inp)
            background, objects = _ordered_objects(inp)
            expected = inp.copy()
            expected[(inp == objects[0][2]) | (inp == objects[-1][2])] = background
            assert np.array_equal(out, expected)
            assert set(int(v) for v in np.unique(out) if v != background) == {
                obj[2] for obj in objects[1:-1]}
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.shape, out.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if split == "train":
                train_counts.append(len(objects))
                train_backgrounds.add(background)
    assert len(set(train_counts)) >= 2 and max(train_counts) >= 3
    assert len(train_backgrounds) >= 2


def _signature(puzzle):
    return tuple(tuple((p["input"].shape, p["input"].tobytes(),
                        p["output"].tobytes()) for p in puzzle[split])
                 for split in ("train", "test"))


def _predict_from_training(train, test_input):
    # Receives only labeled training pairs and an unlabeled test input.
    for pair in train:
        bg, objects = _ordered_objects(pair["input"])
        predicted = pair["input"].copy()
        predicted[np.isin(predicted, (objects[0][2], objects[-1][2]))] = bg
        assert np.array_equal(predicted, pair["output"])
    bg, objects = _ordered_objects(test_input)
    prediction = test_input.copy()
    prediction[np.isin(prediction, (objects[0][2], objects[-1][2]))] = bg
    return prediction


def self_test() -> None:
    train_lengths, test_lengths, sizes = set(), set(), set()
    counts = {"train": set(), "test": set()}
    signatures, layouts, backgrounds, heights, gaps = set(), set(), set(), set(), set()
    one_row = edge_objects = touching = staggered = irregular = 0
    seen_test_count = new_test_count = blank_train = blank_test = 0
    review = Counter()
    source_dimensions = {(14, 10), (16, 12), (13, 6), (11, 10), (12, 8), (12, 6)}
    used_source_dimensions = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _signature(puzzle)
        assert signature == _signature(generate_puzzle(seed))
        signatures.add(signature)
        train_lengths.add(len(puzzle["train"]))
        test_lengths.add(len(puzzle["test"]))
        train_object_counts = {len(_ordered_objects(p["input"])[1]) for p in puzzle["train"]}
        roles_by_color = {}
        puzzle_has_stagger = False
        for split in ("train", "test"):
            for pair_index, pair in enumerate(puzzle[split]):
                inp, out = pair["input"], pair["output"]
                bg, objects = _ordered_objects(inp)
                count = len(objects)
                counts[split].add(count)
                sizes.add(inp.shape)
                used_source_dimensions.add(inp.shape)
                backgrounds.add(bg)
                if count == 2:
                    assert np.all(out == bg)
                    if split == "train":
                        blank_train += 1
                    else:
                        blank_test += 1
                        review[f"blank_test_position_{pair_index + 1}"] += 1
                if split == "test":
                    if count in train_object_counts:
                        seen_test_count += 1
                    else:
                        new_test_count += 1
                    prediction = _predict_from_training(puzzle["train"], inp.copy())
                    assert np.array_equal(prediction, out)
                pair_centers = []
                pair_touch = pair_irregular = pair_edge = False
                for i, obj in enumerate(objects):
                    top, bottom, _, spans = obj
                    role = "top" if i == 0 else "bottom" if i == count - 1 else "middle"
                    roles_by_color.setdefault(obj[2], set()).add(role)
                    h = bottom - top + 1
                    heights.add(h)
                    widths = [right - left + 1 for left, right in spans]
                    bbox_width = max(right for _, right in spans) - min(left for left, _ in spans) + 1
                    center = (min(left for left, _ in spans) + max(right for _, right in spans)) / 2
                    pair_centers.append(center)
                    one_row += h == 1
                    review[f"{role}_one_row"] += h == 1
                    review[f"{role}_tall"] += h >= 8
                    review[f"{role}_broad"] += bbox_width >= inp.shape[1] / 2
                    review[f"{role}_narrow"] += bbox_width <= 2
                    review[f"{role}_rectangle"] += len(set(spans)) == 1
                    is_edge = any(left == 0 or right == inp.shape[1] - 1 for left, right in spans)
                    edge_objects += is_edge
                    review[f"{role}_side_edge"] += is_edge
                    pair_edge |= is_edge
                    is_irregular = h >= 3 and len(set(widths)) >= 3
                    irregular += is_irregular
                    review[f"{role}_irregular"] += is_irregular
                    pair_irregular |= is_irregular
                    review[f"{role}_neck"] += h >= 3 and min(widths[1:-1]) <= 2 and max(widths) >= 5
                    if i:
                        gap = top - objects[i - 1][1] - 1
                        gaps.add(gap)
                        touching += gap == 0
                        pair_touch |= gap == 0
                layouts.add((inp.shape, tuple((o[0], o[1], o[3]) for o in objects)))
                pair_stagger = max(pair_centers) - min(pair_centers) >= 4
                review["staggered_pairs"] += pair_stagger
                review["combined_stagger_touch_irregular_edge"] += (
                    pair_stagger and pair_touch and pair_irregular and pair_edge)
                puzzle_has_stagger |= pair_stagger
                assert not _source_input(inp)
        staggered += puzzle_has_stagger
        review["color_role_switch_puzzles"] += any(
            "middle" in roles and ("top" in roles or "bottom" in roles)
            for roles in roles_by_color.values())
    assert len(signatures) == 42
    assert len(train_lengths) >= 4 and len(test_lengths) >= 3
    assert counts["train"] >= {2, 3, 4, 5, 6, 7, 8}
    assert counts["test"] >= {2, 3, 4, 5, 6, 7, 8}
    assert min(s[0] for s in sizes) <= 8 and max(s[0] for s in sizes) >= 25
    assert min(s[1] for s in sizes) <= 6 and max(s[1] for s in sizes) >= 25
    assert len(sizes) >= 75 and len(layouts) >= 150
    assert len(backgrounds) >= 8 and 0 in backgrounds
    assert {1, 2, 3, 4, 5, 6} <= heights and {0, 1, 2, 3, 4} <= gaps
    assert one_row >= 20 and edge_objects >= 40 and touching >= 20
    assert staggered >= 30 and irregular >= 30
    assert seen_test_count >= 10 and new_test_count >= 10
    assert blank_train >= 5 and blank_test >= 3
    assert len(used_source_dimensions & source_dimensions) >= 2
    assert all(review[f"blank_test_position_{i}"] >= 1 for i in range(1, 5))
    assert review["staggered_pairs"] >= 150
    assert review["combined_stagger_touch_irregular_edge"] >= 100
    assert review["color_role_switch_puzzles"] >= 30
    for role in ("top", "middle", "bottom"):
        assert review[f"{role}_one_row"] >= 20
        assert review[f"{role}_tall"] >= 15
        assert review[f"{role}_broad"] >= 100
        assert review[f"{role}_narrow"] >= 30
        assert review[f"{role}_rectangle"] >= 60
        assert review[f"{role}_irregular"] >= 40
        assert review[f"{role}_neck"] >= 15
        assert review[f"{role}_side_edge"] >= 50

    for seed, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4)), 1000):
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ne)
        assert _signature(puzzle) == _signature(generate_puzzle(seed, nt, ne))
    for seed, (nt, ne) in enumerate(((None, 1), (None, 4), (2, None), (10, None)), 1100):
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        assert nt is None or len(puzzle["train"]) == nt
        assert ne is None or len(puzzle["test"]) == ne
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": True},
                      {"num_test": 1.5}):
        try:
            generate_puzzle(7, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {arguments}")
