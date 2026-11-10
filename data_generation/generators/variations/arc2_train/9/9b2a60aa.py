"""Compositional generators for the token controlled motif-copy task."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = frozenset({
    "af04a79af04b759b2aacc28c5ad9c450c98a0d226ded6ea1ac63a3e2eb59ea5c",
    "61c10ddaf80b07268dc79db0f768a14e1a453ffbae4b9f404c431653f7205eae",
    "5fe00e19e5c0b0722c05e2d8540ccc3fb14dbf02d9b328aa471d7867a6bdee0c",
    "49cb7ca7f2e213635c5f62c249cda31df5772154a2f2aaf6f9736ab16355b22d",
})
_AUTO_COUNTS = tuple(
    (a, b) for b in range(1, 5) for a in range(max(2, b + 1), 11)
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    a, b = _count(num_train, "num_train"), _count(num_test, "num_test")
    if a is not None and not 2 <= a <= 10:
        raise ValueError("num_train must be in 2..10")
    if b is not None and not 1 <= b <= 4:
        raise ValueError("num_test must be in 1..4")
    if a is None and b is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if a is None:
        options = tuple(range(b + 1, 11))
        a = options[(seed + b) % len(options)]
    if b is None:
        options = tuple(range(1, min(4, a - 1) + 1))
        b = options[(seed + a) % len(options)]
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _pair_digest(inp, out):
    data = inp.astype(np.uint8).tobytes() + out.astype(np.uint8).tobytes()
    data += bytes(inp.shape)
    return hashlib.sha256(data).hexdigest()


def _components(mask):
    remaining = {tuple(p) for p in np.argwhere(mask)}
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
    return count


def _make_mask(rng, shape, style):
    h, w = shape
    if style == "source" and shape == (3, 3):
        return np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=bool)
    if style == "hollow":
        mask = np.ones(shape, dtype=bool)
        mask[1:-1, 1:-1] = False
        if h == 2:
            mask[int(rng.integers(0, h)), int(rng.integers(1, w - 1))] = False
        return mask
    for _ in range(80):
        density = float(rng.uniform(0.29, 0.78)) if style == "sparse" else float(rng.uniform(0.48, 0.83))
        mask = rng.random(shape) < density
        if (
            4 <= int(mask.sum()) < h * w
            and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any()
            and np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))
        ):
            return mask
    mask = np.ones(shape, dtype=bool)
    mask[h // 2, w // 2] = False
    return mask


def _sample_colors(rng, count, anchor, background, shared_palette):
    pool = list(shared_palette) if shared_palette is not None else [
        c for c in range(10) if c != background
    ]
    anchor_color = int(rng.choice(pool))
    other_pool = [c for c in pool if c != anchor_color]
    mode = int(rng.integers(0, 3))
    if shared_palette is not None:
        distinct = len(other_pool)
    elif mode == 0:
        distinct = min(count - 1, len(other_pool))
    elif mode == 1:
        distinct = 1 if count <= 3 else int(rng.integers(1, 3))
    else:
        distinct = int(rng.integers(1, min(count - 1, len(other_pool)) + 1))
    chosen = [int(c) for c in rng.choice(other_pool, size=distinct, replace=False)]
    colors = [int(rng.choice(chosen)) for _ in range(count)]
    if count - 1 >= distinct:
        slots = [i for i in range(count) if i != anchor]
        rng.shuffle(slots)
        for slot, color in zip(slots, chosen):
            colors[slot] = color
    colors[anchor] = anchor_color
    return colors


def _sample_gaps(rng, count, axis_size, require_varied):
    for _ in range(50):
        if not require_varied and rng.random() < 0.28:
            gap = int(rng.integers(1, 7))
            gaps = [gap] * (count - 1)
        else:
            gaps = [int(x) for x in rng.integers(1, 8, size=count - 1)]
        if count * axis_size + sum(gaps) <= 30 and (
            not require_varied or len(set(gaps)) > 1
        ):
            return gaps
    gaps = [1] * (count - 1)
    if require_varied:
        gaps[0] = 2
    return gaps


def _sample_configuration(rng, train_configs, split, index):
    if split == "train":
        if index == 0:
            return ("horizontal" if rng.random() < 0.5 else "vertical",
                    "first" if rng.random() < 0.5 else "last")
        if index == 1:
            first = train_configs[0]
            return ("vertical" if first[0] == "horizontal" else "horizontal",
                    "interior" if rng.random() < 0.38 else
                    ("last" if first[1] == "first" else "first"))
        return ("horizontal" if rng.random() < 0.5 else "vertical",
                ("first", "last", "interior")[int(rng.integers(0, 3))])
    if rng.random() < 0.28:
        return train_configs[int(rng.integers(0, len(train_configs)))]
    orientations = {o for o, _ in train_configs}
    sides = {s for _, s in train_configs}
    choices = [
        (o, s) for o in ("horizontal", "vertical")
        for s in ("first", "last", "interior")
        if o in orientations and s in sides
        and (s != "interior" or any(t == "interior" for _, t in train_configs))
    ]
    fresh = [x for x in choices if x not in train_configs]
    options = fresh if fresh and rng.random() < 0.72 else choices
    return options[int(rng.integers(0, len(options)))]


def _build_pair(rng, config, split, background, shared_palette, old_masks,
                require_varied):
    orientation, side = config
    shape = ((2, 3), (3, 3), (3, 4), (4, 4), (3, 2), (4, 3))[
        int(rng.integers(0, 6))
    ]
    if old_masks and rng.random() < 0.30:
        mask = old_masks[int(rng.integers(0, len(old_masks)))].copy()
        shape = mask.shape
    else:
        style = ("random", "sparse", "hollow", "source")[
            int(rng.integers(0, 4))
        ]
        mask = _make_mask(rng, shape, style)
    h, w = shape
    axis_size = w if orientation == "horizontal" else h
    max_count = min(7 if split == "test" else 6, (31 // (axis_size + 1)))
    low = 3 if require_varied or side == "interior" else 2
    if shared_palette is not None:
        low = max(low, len(shared_palette))
    count = int(rng.integers(low, max_count + 1))
    anchor = 0 if side == "first" else count - 1 if side == "last" else int(
        rng.integers(1, count - 1)
    )
    gaps = _sample_gaps(rng, count, axis_size, require_varied)
    lane_length = count * axis_size + sum(gaps)
    code_length = count + sum(gaps)
    if rng.random() < 0.21:
        extra_axis = 0
    else:
        extra_axis = int(rng.integers(0, min(8, 30 - lane_length) + 1))
    axis_length = lane_length + extra_axis
    near_margin = int(rng.integers(0, extra_axis + 1))
    if rng.random() < 0.18:
        code_start = 0
    else:
        code_start = int(rng.integers(0, axis_length - code_length + 1))
    separation = int(rng.integers(1, 9))
    perp_size = h if orientation == "horizontal" else w
    outer = int(rng.integers(0, 7))
    left_outer = int(rng.integers(0, outer + 1))
    right_outer = outer - left_outer
    code_before = bool(rng.integers(0, 2))
    if code_before:
        code_perp = left_outer
        motif_perp = code_perp + 1 + separation
    else:
        motif_perp = left_outer
        code_perp = motif_perp + perp_size + separation
    perp_length = perp_size + 1 + separation + outer
    height, width = (perp_length, axis_length) if orientation == "horizontal" else (axis_length, perp_length)
    colors = _sample_colors(rng, count, anchor, background, shared_palette)
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    token_positions = [code_start]
    motif_positions = [near_margin]
    for gap in gaps:
        token_positions.append(token_positions[-1] + 1 + gap)
        motif_positions.append(motif_positions[-1] + axis_size + gap)
    for pos, color in zip(token_positions, colors):
        if orientation == "horizontal":
            inp[code_perp, pos] = color
            out[code_perp, pos] = color
        else:
            inp[pos, code_perp] = color
            out[pos, code_perp] = color
    for i, (pos, color) in enumerate(zip(motif_positions, colors)):
        r, c = (motif_perp, pos) if orientation == "horizontal" else (pos, motif_perp)
        out[r:r+h, c:c+w][mask] = color
        if i == anchor:
            inp[r:r+h, c:c+w][mask] = color
    return {"input": inp, "output": out}, mask


def _candidate_solutions(inp, background):
    foreground = inp != background
    height, width = inp.shape
    results = []
    for orientation in ("horizontal", "vertical"):
        line_count = height if orientation == "horizontal" else width
        for line in range(line_count):
            coords = np.flatnonzero(foreground[line, :] if orientation == "horizontal" else foreground[:, line])
            if len(coords) < 2 or np.any(np.diff(coords) < 2):
                continue
            adjacent = (line - 1, line + 1)
            if any(
                0 <= q < line_count and np.any(
                    foreground[q, :] if orientation == "horizontal" else foreground[:, q]
                ) for q in adjacent
            ):
                continue
            remainder = foreground.copy()
            if orientation == "horizontal":
                remainder[line, :] = False
            else:
                remainder[:, line] = False
            points = np.argwhere(remainder)
            if len(points) < 4:
                continue
            r0, c0 = points.min(axis=0)
            r1, c1 = points.max(axis=0) + 1
            h, w = int(r1-r0), int(c1-c0)
            if not (2 <= h <= 4 and 2 <= w <= 4):
                continue
            block = inp[r0:r1, c0:c1]
            motif_colors = set(int(x) for x in block.flat if x != background)
            if len(motif_colors) != 1:
                continue
            motif_color = motif_colors.pop()
            colors = [int(inp[line, p] if orientation == "horizontal" else inp[p, line]) for p in coords]
            anchors = [i for i, color in enumerate(colors) if color == motif_color]
            if len(anchors) != 1:
                continue
            anchor = anchors[0]
            gaps = tuple(int(x) - 1 for x in np.diff(coords))
            mask = block != background
            size = w if orientation == "horizontal" else h
            positions = [0] * len(coords)
            positions[anchor] = int(c0 if orientation == "horizontal" else r0)
            for i in range(anchor, len(coords)-1):
                positions[i+1] = positions[i] + size + gaps[i]
            for i in range(anchor-1, -1, -1):
                positions[i] = positions[i+1] - size - gaps[i]
            predicted = inp.copy()
            valid = True
            for i, (pos, color) in enumerate(zip(positions, colors)):
                rr, cc = (int(r0), pos) if orientation == "horizontal" else (pos, int(c0))
                if rr < 0 or cc < 0 or rr+h > height or cc+w > width:
                    valid = False
                    break
                dest = predicted[rr:rr+h, cc:cc+w]
                if i != anchor and np.any(dest != background):
                    valid = False
                    break
                dest[mask] = color
            if valid:
                results.append((predicted, {
                    "orientation": orientation, "anchor": anchor, "count": len(coords),
                    "colors": tuple(colors), "background": background,
                    "gaps": gaps, "shape": (h, w), "mask": mask,
                    "line": line, "motif_start": (int(r0), int(c0)),
                    "added_edge": any(
                        i != anchor and (
                            pos == 0 or
                            pos + size == (width if orientation == "horizontal" else height)
                        )
                        for i, pos in enumerate(positions)
                    ),
                }))
    return results


def _analyze_pair(inp, out):
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    solutions = _candidate_solutions(inp, background)
    _require(len(solutions) == 1, "input must have one unambiguous token line and template")
    predicted, info = solutions[0]
    _require(np.array_equal(predicted, out), "output violates token controlled copying")
    _require(not np.array_equal(inp, out), "a pair must add copies")
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_infos, test_infos = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= x <= 30 for x in grid.shape)
                         and np.issubdtype(grid.dtype, np.integer)
                         and bool(np.all((0 <= grid) & (grid <= 9))),
                         "illegal grid shape, dtype, or color")
            _require(inp.shape == out.shape, "grid size must be preserved")
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            _require(key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            _require(_pair_digest(inp, out) not in _SOURCE_DIGESTS,
                     "exact source pair was copied")
            info = _analyze_pair(inp, out)
            (train_infos if split == "train" else test_infos).append(info)
    train_configs = {
        (x["orientation"], "first" if x["anchor"] == 0 else
         "last" if x["anchor"] == x["count"]-1 else "interior")
        for x in train_infos
    }
    train_orientations = {x[0] for x in train_configs}
    _require(train_orientations == {"horizontal", "vertical"},
             "training must show both orientations")
    _require(any(len(set(x["gaps"])) > 1 for x in train_infos),
             "training must show that unequal token gaps are preserved")
    for x in test_infos:
        if 0 < x["anchor"] < x["count"]-1:
            _require(any(0 < y["anchor"] < y["count"]-1 for y in train_infos),
                     "an interior test anchor needs a training demonstration")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    consistent_palette = bool(rng.random() < 0.24)
    shared_background = int(rng.integers(0, 10))
    shared_palette = tuple(int(x) for x in rng.choice(
        [c for c in range(10) if c != shared_background],
        size=int(rng.integers(2, 5)), replace=False
    )) if consistent_palette else None
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    train_configs = []
    old_masks = []
    seen_inputs = set()
    for split, amount in (("train", train_count), ("test", test_count)):
        for index in range(amount):
            pair_rng = np.random.default_rng(int(pair_seeds[len(seen_inputs)]))
            config = _sample_configuration(pair_rng, train_configs, split, index)
            if split == "train":
                train_configs.append(config)
            background = shared_background if consistent_palette else (
                0 if pair_rng.random() < 0.52 else int(pair_rng.integers(1, 10))
            )
            for _ in range(120):
                pair, mask = _build_pair(
                    pair_rng, config, split, background, shared_palette, old_masks,
                    split == "train" and index == 0
                )
                key = _grid_key(pair["input"])
                if key in seen_inputs:
                    continue
                try:
                    _analyze_pair(pair["input"], pair["output"])
                except AssertionError:
                    continue
                if _pair_digest(pair["input"], pair["output"]) in _SOURCE_DIGESTS:
                    continue
                puzzle[split].append(pair)
                old_masks.append(mask)
                seen_inputs.add(key)
                break
            else:
                raise ValueError("could not construct an unambiguous unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, keys = set(), set(), set()
    orientations, sides, lengths, shapes, backgrounds = set(), set(), set(), set(), set()
    masks, gap_patterns, palettes = set(), set(), set()
    interior, uniform, disconnected, edges, long_gaps = 0, 0, 0, 0, 0
    repeated_masks, variable_background_puzzles = 0, 0
    equal_length_tests, longer_tests = 0, 0
    consistent_palette_puzzles, variable_palette_puzzles = 0, 0
    source_masks, distant_codes, distinct_colors, heavy_repeats = 0, 0, 0, 0
    repeated_configs, new_configs = 0, 0
    source_mask = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=bool)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        keys.add(_puzzle_key(puzzle))
        infos = [_analyze_pair(p["input"], p["output"])
                 for s in ("train", "test") for p in puzzle[s]]
        train_info = infos[:len(puzzle["train"])]
        test_info = infos[len(puzzle["train"]):]
        palettes_in_puzzle = {
            frozenset((x["background"],) + x["colors"]) for x in infos
        }
        consistent_palette_puzzles += len(palettes_in_puzzle) == 1
        variable_palette_puzzles += len(palettes_in_puzzle) > 1
        train_configs = {
            (x["orientation"], "first" if x["anchor"] == 0 else
             "last" if x["anchor"] == x["count"]-1 else "interior")
            for x in train_info
        }
        if len({x["background"] for x in infos}) > 1:
            variable_background_puzzles += 1
        signatures = [(x["shape"], x["mask"].tobytes()) for x in infos]
        if len(set(signatures)) < len(signatures):
            repeated_masks += 1
        for x in infos:
            orientations.add(x["orientation"])
            sides.add("first" if x["anchor"] == 0 else
                      "last" if x["anchor"] == x["count"]-1 else "interior")
            lengths.add(x["count"])
            shapes.add(x["shape"])
            backgrounds.add(x["background"])
            masks.add((x["shape"], x["mask"].tobytes()))
            gap_patterns.add(x["gaps"])
            palettes.add(tuple(sorted(set(x["colors"]))))
            interior += 0 < x["anchor"] < x["count"]-1
            uniform += len(set(x["gaps"])) == 1
            disconnected += _components(x["mask"]) > 1
            edges += x["added_edge"]
            long_gaps += max(x["gaps"]) >= 5
            source_masks += x["shape"] == (3, 3) and np.array_equal(
                x["mask"], source_mask
            )
            distant_codes += (
                x["motif_start"][0] - x["line"] - 1
                if x["orientation"] == "horizontal" and
                x["line"] < x["motif_start"][0]
                else x["line"] - x["motif_start"][0] - x["shape"][0]
                if x["orientation"] == "horizontal"
                else x["motif_start"][1] - x["line"] - 1
                if x["line"] < x["motif_start"][1]
                else x["line"] - x["motif_start"][1] - x["shape"][1]
            ) >= 6
            distinct_colors += len(set(x["colors"])) == x["count"]
            other_colors = [color for i, color in enumerate(x["colors"])
                            if i != x["anchor"]]
            heavy_repeats += max(other_colors.count(color)
                                 for color in set(other_colors)) >= 3
        train_lengths = {x["count"] for x in train_info}
        equal_length_tests += any(x["count"] in train_lengths for x in test_info)
        longer_tests += any(x["count"] > max(train_lengths) for x in test_info)
        for x in test_info:
            config = (x["orientation"], "first" if x["anchor"] == 0 else
                      "last" if x["anchor"] == x["count"]-1 else "interior")
            repeated_configs += config in train_configs
            new_configs += config not in train_configs
    _require(len(train_counts) >= 4 and len(test_counts) >= 3 and len(keys) >= 40,
             "count or puzzle diversity is too low")
    _require(orientations == {"horizontal", "vertical"} and
             sides == {"first", "last", "interior"} and len(lengths) >= 5,
             "configuration or sequence diversity is too low")
    _require(len(shapes) >= 5 and len(backgrounds) >= 5
             and len(masks) >= 30 and len(gap_patterns) >= 30
             and len(palettes) >= 20, "shape, color, or gap diversity is too low")
    _require(min(interior, uniform, disconnected, edges, long_gaps,
                 distinct_colors, heavy_repeats, distant_codes) >= 10,
             "a pair-level reviewed variation is too rare")
    _require(min(repeated_masks, variable_background_puzzles,
                 variable_palette_puzzles, equal_length_tests,
                 repeated_configs, new_configs) >= 5,
             "a puzzle-level reviewed variation is too rare")
    _require(min(longer_tests, consistent_palette_puzzles, source_masks) >= 2,
             "a less common reviewed variation is absent")
    _require(_puzzle_key(generate_puzzle(12345)) ==
             _puzzle_key(generate_puzzle(12345)), "automatic nondeterminism")
    _require(_puzzle_key(generate_puzzle(54321, 7, 3)) ==
             _puzzle_key(generate_puzzle(54321, 7, 3)), "explicit nondeterminism")
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4),
                                 (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(60000 + i, a, b)
        validate_generated_puzzle(p)
        _require((a is None or len(p["train"]) == a)
                 and (b is None or len(p["test"]) == b),
                 "requested count was not honored")
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(62000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
