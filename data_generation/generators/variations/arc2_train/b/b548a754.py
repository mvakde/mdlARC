"""Generate framed rectangles stretched to an isolated marker."""

import numpy as np


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


_SOURCE_ROWS = (
    (("00000000000", "03333300000", "03222300008", "03222300000",
      "03333300000", "00000000000", "00000000000", "00000000000",
      "00000000000", "00000000000", "00000000000"),
     ("00000000000", "03333333333", "03222222223", "03222222223",
      "03333333333", "00000000000", "00000000000", "00000000000",
      "00000000000", "00000000000", "00000000000")),
    (("00000000000", "02222000000", "02112000000", "02112000000",
      "02222000000", "00000000000", "00000000000", "00000000000",
      "00080000000", "00000000000", "00000000000"),
     ("00000000000", "02222000000", "02112000000", "02112000000",
      "02112000000", "02112000000", "02112000000", "02112000000",
      "02222000000", "00000000000", "00000000000")),
    (("000000000000", "001110000000", "001610000000", "001610000000",
      "001610000000", "001110000080", "000000000000", "000000000000",
      "000000000000", "000000000000", "000000000000", "000000000000",
      "000000000000"),
     ("000000000000", "001111111110", "001666666610", "001666666610",
      "001666666610", "001111111110", "000000000000", "000000000000",
      "000000000000", "000000000000", "000000000000", "000000000000",
      "000000000000")),
    (("0000000000000", "0000800000000", "0000000000000", "0000000000000",
      "0000000000000", "0000000000000", "0000000000000", "0000000000000",
      "0000000000000", "0006666600000", "0006444600000", "0006444600000",
      "0006666600000"),
     ("0000000000000", "0006666600000", "0006444600000", "0006444600000",
      "0006444600000", "0006444600000", "0006444600000", "0006444600000",
      "0006444600000", "0006444600000", "0006444600000", "0006444600000",
      "0006666600000")),
)


def _row_signature(rows: tuple[str, ...]) -> tuple:
    return ((len(rows), len(rows[0])), tuple(int(x) for row in rows for x in row))


_SOURCE_SIGNATURES = frozenset(
    (_row_signature(inp), _row_signature(out)) for inp, out in _SOURCE_ROWS
)


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed_key: int, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed_key % 4
        train = max(test + 1, 2 + (3 * seed_key + seed_key // 4) % 9)
    elif train is None:
        train = test + 1 + seed_key % (10 - test)
    else:
        test = 1 + seed_key % min(4, train - 1)
    return int(train), int(test)


def _draw(grid: np.ndarray, bounds: tuple[int, int, int, int],
          frame: int, fill: int) -> None:
    top, bottom, left, right = bounds
    grid[top:bottom + 1, left:right + 1] = fill
    grid[top, left:right + 1] = frame
    grid[bottom, left:right + 1] = frame
    grid[top:bottom + 1, left] = frame
    grid[top:bottom + 1, right] = frame


def _directions(rng: np.random.Generator, train_count: int,
                test_count: int) -> tuple[list[str], list[str]]:
    horizontal = str(rng.choice(("left", "right")))
    vertical = str(rng.choice(("up", "down")))
    train = [horizontal, vertical]
    rng.shuffle(train)
    held_out = str(rng.choice(("left" if horizontal == "right" else "right",
                               "up" if vertical == "down" else "down")))
    mode = int(rng.integers(0, 3))  # novel side, shared side, or both
    choices = tuple(d for d in ("left", "right", "up", "down")
                    if mode == 1 or d != held_out)
    train.extend(str(rng.choice(choices)) for _ in range(train_count - 2))
    seen = tuple(dict.fromkeys(train))
    unseen = tuple(d for d in ("left", "right", "up", "down") if d not in seen)
    if mode == 0:
        test = [str(rng.choice(unseen)) for _ in range(test_count)]
    elif mode == 1:
        test = [str(rng.choice(seen)) for _ in range(test_count)]
    elif test_count >= 2:
        test = [str(rng.choice(seen)), str(rng.choice(unseen))]
        test.extend(str(rng.choice(("left", "right", "up", "down")))
                    for _ in range(test_count - 2))
        rng.shuffle(test)
    else:
        test = [str(rng.choice((str(rng.choice(seen)), str(rng.choice(unseen)))))]
    return train, test


def _margins(rng: np.random.Generator, span: int) -> tuple[int, int]:
    # Zero gives legal canvas-edge cases; a cap of four leaves varied context.
    capacity = 30 - span
    first = int(rng.integers(0, min(4, capacity) + 1))
    second = int(rng.integers(0, min(4, capacity - first) + 1))
    return first, second


def _geometry(rng: np.random.Generator, direction: str,
              inner_alignment: bool) -> tuple[tuple[int, int], tuple[int, int, int, int],
                                                tuple[int, int, int, int], tuple[int, int]]:
    old_height = int(rng.integers(3, 16))
    old_width = int(rng.integers(3, 16))
    old_axis = old_width if direction in ("left", "right") else old_height
    extension = int(rng.integers(2, min(14, 26 - old_axis) + 1))
    final_height = old_height + (extension if direction in ("up", "down") else 0)
    final_width = old_width + (extension if direction in ("left", "right") else 0)
    top_margin, bottom_margin = _margins(rng, final_height)
    left_margin, right_margin = _margins(rng, final_width)
    if top_margin + bottom_margin + left_margin + right_margin == 0:
        # Keep the exterior color visible in the completed image.
        if final_height < 30:
            bottom_margin = 1
        else:
            right_margin = 1
    top, left = top_margin, left_margin
    bottom, right = top + final_height - 1, left + final_width - 1
    final = (top, bottom, left, right)
    if direction == "left":
        old = (top, bottom, left + extension, right)
    elif direction == "right":
        old = (top, bottom, left, right - extension)
    elif direction == "up":
        old = (top + extension, bottom, left, right)
    else:
        old = (top, bottom - extension, left, right)
    ot, ob, ol, oright = old
    perpendicular = old_height if direction in ("left", "right") else old_width
    offset = int(rng.integers(1 if inner_alignment else 0,
                              perpendicular - (1 if inner_alignment else 0)))
    if direction == "left":
        marker = (ot + offset, left)
    elif direction == "right":
        marker = (ot + offset, right)
    elif direction == "up":
        marker = (top, ol + offset)
    else:
        marker = (bottom, ol + offset)
    return (final_height + top_margin + bottom_margin,
            final_width + left_margin + right_margin), old, final, marker


def _palette(rng: np.random.Generator, background: int,
             fixed_marker: int | None, earlier: list[tuple[int, int]],
             prior_marker: int | None) -> tuple[int, int, int]:
    eligible = [p for p in earlier if background not in p and fixed_marker not in p]
    roll = float(rng.random())
    if eligible and roll < 0.55:
        frame, fill = eligible[int(rng.integers(len(eligible)))]
        if roll >= 0.30:
            frame, fill = fill, frame
    else:
        colors = [c for c in range(10) if c != background and c != fixed_marker]
        selected = rng.choice(colors, size=2, replace=False)
        frame, fill = int(selected[0]), int(selected[1])
    if fixed_marker is not None:
        marker = fixed_marker
    else:
        choices = [c for c in range(10) if c not in (background, frame, fill)]
        if prior_marker is not None:
            alternatives = [c for c in choices if c != prior_marker]
            if alternatives:
                choices = alternatives
        marker = int(rng.choice(choices))
    return frame, fill, marker


def _make_pair(rng: np.random.Generator, direction: str, background: int,
               frame: int, fill: int, marker_color: int,
               inner_alignment: bool) -> dict:
    shape, old, final, marker = _geometry(rng, direction, inner_alignment)
    inp = np.full(shape, background, dtype=np.int8)
    out = np.full(shape, background, dtype=np.int8)
    _draw(inp, old, frame, fill)
    inp[marker] = marker_color
    _draw(out, final, frame, fill)
    return {"input": inp, "output": out}


def _parse_input(grid: np.ndarray) -> dict:
    colors, counts = np.unique(grid, return_counts=True)
    _check(len(colors) == 4, "input needs four distinct semantic colors")
    candidates = []
    for marker_color in (int(c) for c, n in zip(colors, counts) if n == 1):
        marker_position = np.argwhere(grid == marker_color)[0]
        mr, mc = int(marker_position[0]), int(marker_position[1])
        for background in (int(c) for c in colors if c != marker_color):
            mask = (grid != background) & (grid != marker_color)
            locations = np.argwhere(mask)
            top, left = (int(x) for x in locations.min(axis=0))
            bottom, right = (int(x) for x in locations.max(axis=0))
            if bottom - top < 2 or right - left < 2:
                continue
            frame, fill = int(grid[top, left]), int(grid[top + 1, left + 1])
            if len({background, marker_color, frame, fill}) != 4:
                continue
            expected = np.full(grid.shape, background, dtype=grid.dtype)
            _draw(expected, (top, bottom, left, right), frame, fill)
            expected[mr, mc] = marker_color
            if not np.array_equal(expected, grid):
                continue
            if top <= mr <= bottom and mc <= left - 2:
                direction, extension = "left", left - mc
                final = (top, bottom, mc, right)
                alignment = mr - top
            elif top <= mr <= bottom and mc >= right + 2:
                direction, extension = "right", mc - right
                final = (top, bottom, left, mc)
                alignment = mr - top
            elif left <= mc <= right and mr <= top - 2:
                direction, extension = "up", top - mr
                final = (mr, bottom, left, right)
                alignment = mc - left
            elif left <= mc <= right and mr >= bottom + 2:
                direction, extension = "down", mr - bottom
                final = (top, mr, left, right)
                alignment = mc - left
            else:
                continue
            if not (np.any(expected == background)):
                continue
            candidates.append({"background": background, "marker_color": marker_color,
                               "frame": frame, "fill": fill, "old": (top, bottom, left, right),
                               "final": final, "marker": (mr, mc), "direction": direction,
                               "extension": extension, "alignment": alignment,
                               "inner_alignment": 0 < alignment <
                               ((bottom - top) if direction in ("left", "right")
                                else (right - left)), "shape": grid.shape,
                               "old_height": bottom - top + 1, "old_width": right - left + 1,
                               "final_height": final[1] - final[0] + 1,
                               "final_width": final[3] - final[2] + 1})
    _check(len(candidates) == 1, "input must have exactly one marker and rectangle reading")
    return candidates[0]


def _analyze_pair(inp: np.ndarray, out: np.ndarray) -> dict:
    _check(inp.shape == out.shape, "input and output shapes differ")
    info = _parse_input(inp)
    expected = np.full(inp.shape, info["background"], dtype=inp.dtype)
    _draw(expected, info["final"], info["frame"], info["fill"])
    _check(np.any(expected == info["background"]), "output needs exterior background")
    _check(np.array_equal(out, expected), "output violates marker-directed stretching")
    _check(not np.array_equal(inp, out), "transformation must change the grid")
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle must have exactly train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "train and test must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    sections = {}
    for section in ("train", "test"):
        sections[section] = []
        for pair in puzzle[section]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair must contain exactly input and output")
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "grid must be a two-dimensional NumPy array")
                _check(all(1 <= side <= 30 for side in grid.shape), "illegal grid shape")
                _check(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
                _check(bool(np.all((grid >= 0) & (grid <= 9))), "illegal ARC color")
            sig = (_signature(pair["input"]), _signature(pair["output"]))
            _check(sig[0] not in seen_inputs and sig not in seen_pairs,
                   "duplicate input or pair")
            _check(sig not in _SOURCE_SIGNATURES, "exact source pair copy")
            seen_inputs.add(sig[0])
            seen_pairs.add(sig)
            sections[section].append(_analyze_pair(pair["input"], pair["output"]))
    train, test = sections["train"], sections["test"]
    _check(any(x["direction"] in ("left", "right") for x in train) and
           any(x["direction"] in ("up", "down") for x in train),
           "training must demonstrate both extension axes")
    _check(any(x["inner_alignment"] and x["direction"] in ("left", "right")
               for x in train) and
           any(x["inner_alignment"] and x["direction"] in ("up", "down")
               for x in train), "training must show interior marker alignment on both axes")
    # Every test input has one reading without hidden color or direction conventions.
    _check(all(x["extension"] >= 2 for x in train + test),
           "marker needs a visible gap from the old frame")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_key = int(seed) % (1 << 64)
    train_count, test_count = _resolve_counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(seed_key)
    train_dirs, test_dirs = _directions(rng, train_count, test_count)
    stable_background = bool(rng.integers(0, 3) == 0)
    stable_marker = bool(rng.integers(0, 2) == 0)
    base_background = int(rng.integers(0, 10))
    base_marker = int(rng.choice([c for c in range(10) if c != base_background]))
    fixed_marker = base_marker if stable_marker else None
    salts = rng.integers(0, 1 << 32, size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    train_palettes: list[tuple[int, int]] = []
    all_palettes: list[tuple[int, int]] = []
    prior_background: int | None = None
    prior_marker: int | None = None
    seen = set()
    for index, direction in enumerate(train_dirs + test_dirs):
        child = np.random.default_rng(np.random.SeedSequence(
            [seed_key & 0xFFFFFFFF, seed_key >> 32, index, int(salts[index])]))
        if stable_background:
            background = base_background
        else:
            choices = [c for c in range(10) if c != fixed_marker]
            if index == 1 and prior_background in choices:
                choices.remove(prior_background)
            background = int(child.choice(choices))
        pool = train_palettes if index >= train_count else all_palettes
        frame, fill, marker = _palette(child, background, fixed_marker,
                                       pool, prior_marker if index == 1 else None)
        section = "train" if index < train_count else "test"
        for _ in range(64):
            pair = _make_pair(child, direction, background, frame, fill, marker,
                              inner_alignment=index < 2)
            signature = (_signature(pair["input"]), _signature(pair["output"]))
            if signature[0] not in seen and signature not in _SOURCE_SIGNATURES:
                try:
                    _analyze_pair(pair["input"], pair["output"])
                except AssertionError:
                    continue
                break
        else:
            raise RuntimeError("could not construct an unambiguous unique pair")
        puzzle[section].append(pair)
        seen.add(signature[0])
        all_palettes.append((frame, fill))
        if section == "train":
            train_palettes.append((frame, fill))
        prior_background, prior_marker = background, marker
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    _check(_equal_puzzles(generate_puzzle(987654321, 7, 3),
                         generate_puzzle(987654321, 7, 3)), "nondeterministic puzzle")
    _check(not _equal_puzzles(generate_puzzle(111, 6, 2),
                             generate_puzzle(222, 6, 2)), "seeds did not vary")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            _check(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _check(len(puzzle["test"]) == test, "explicit test count ignored")
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.5},
                   {"num_test": True}):
        try:
            generate_puzzle(3001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    train_counts, test_counts, directions = set(), set(), set()
    backgrounds, shapes, geometries, palettes = set(), set(), set(), set()
    old_edge, new_edge, alignments = set(), set(), set()
    shared, novel, mixed = 0, 0, 0
    span_overlap = all_test_larger = 0
    varied_background = varied_marker = reused_palette = swapped_palette = 0
    aspect_change = 0
    old_aspects, new_aspects, direction_aspects = set(), set(), set()
    extension_relations, selected_edge_states = set(), set()
    zero_other_roles = 0
    test_aspects, test_directions = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        all_info = train + test
        td = {x["direction"] for x in train}
        tds = {x["direction"] for x in test}
        shared += bool(tds <= td)
        novel += bool(tds.isdisjoint(td))
        mixed += bool(tds & td and tds - td)
        span_overlap += bool(min(x["final_height"] if x["direction"] in ("up", "down")
                                 else x["final_width"] for x in test) <=
                             max(x["final_height"] if x["direction"] in ("up", "down")
                                 else x["final_width"] for x in train))
        all_test_larger += bool(min(x["final_height"] if x["direction"] in ("up", "down")
                                    else x["final_width"] for x in test) >
                                max(x["final_height"] if x["direction"] in ("up", "down")
                                    else x["final_width"] for x in train))
        varied_background += len({x["background"] for x in all_info}) > 1
        varied_marker += len({x["marker_color"] for x in all_info}) > 1
        train_palettes = {(x["frame"], x["fill"]) for x in train}
        reused_palette += any((x["frame"], x["fill"]) in train_palettes for x in test)
        swapped_palette += any((x["fill"], x["frame"]) in train_palettes for x in test)
        for x in all_info:
            directions.add(x["direction"])
            backgrounds.add(x["background"])
            shapes.add(x["shape"])
            palettes.add((x["background"], x["marker_color"], x["frame"], x["fill"]))
            geometries.add((x["old_height"], x["old_width"], x["extension"],
                            x["direction"], x["old"], x["shape"]))
            ot, ob, ol, oright = x["old"]
            ft, fb, fl, fr = x["final"]
            height, width = x["shape"]
            old_edge.add((ot == 0, ob == height - 1, ol == 0, oright == width - 1))
            new_edge.add((ft == 0, fb == height - 1, fl == 0, fr == width - 1))
            alignments.add(x["inner_alignment"])
            before = np.sign(x["old_height"] - x["old_width"])
            after = np.sign(x["final_height"] - x["final_width"])
            aspect_change += before != after
            old_aspect = "square" if before == 0 else "tall" if before > 0 else "wide"
            new_aspect = "square" if after == 0 else "tall" if after > 0 else "wide"
            old_aspects.add(old_aspect)
            new_aspects.add(new_aspect)
            direction_aspects.add((x["direction"], old_aspect))
            old_axis = x["old_width"] if x["direction"] in ("left", "right") else x["old_height"]
            extension_relations.add(int(np.sign(x["extension"] - old_axis)))
            selected_edge_states.add((x["direction"],
                ft == 0 if x["direction"] == "up" else
                fb == height - 1 if x["direction"] == "down" else
                fl == 0 if x["direction"] == "left" else fr == width - 1))
            zero_other_roles += 0 in (x["marker_color"], x["frame"], x["fill"])
            if x in test:
                test_aspects.add(new_aspect)
                test_directions.add(x["direction"])
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "counts lack diversity")
    _check(directions == {"left", "right", "up", "down"}, "directions lack coverage")
    _check(shared > 0 and novel > 0 and mixed > 0, "direction regimes lack coverage")
    _check(span_overlap > 0 and all_test_larger > 0,
           "training and test spans lack both overlapping and larger-test cases")
    _check(varied_background > 0 and varied_marker > 0, "role colors never vary within puzzles")
    _check(reused_palette > 0 and swapped_palette > 0, "test palettes never recur or swap")
    _check(len(backgrounds) >= 8 and 0 in backgrounds and len(palettes) >= 50,
           "palette diversity is too narrow")
    _check(len(shapes) >= 35 and len(geometries) >= 150, "structural diversity is too narrow")
    _check(any(e[0] for e in old_edge) and any(e[1] for e in old_edge) and
           any(e[2] for e in old_edge) and any(e[3] for e in old_edge) and
           any(not any(e) for e in old_edge), "old rectangle edge placement lacks coverage")
    _check(any(any(e) for e in new_edge) and any(not any(e) for e in new_edge),
           "new rectangle edge placement lacks coverage")
    _check(alignments == {False, True} and aspect_change > 0,
           "marker alignments or aspect transitions lack coverage")
    _check(old_aspects == {"square", "tall", "wide"} and
           new_aspects == {"square", "tall", "wide"} and
           test_aspects == {"square", "tall", "wide"},
           "rectangle aspect ratios lack coverage")
    _check(len(direction_aspects) == 12 and test_directions == directions,
           "direction and starting shape combinations lack coverage")
    _check(extension_relations == {-1, 0, 1},
           "extension lengths do not vary around old side lengths")
    _check(all((direction, reaches) in selected_edge_states
               for direction in directions for reaches in (False, True)),
           "marker-defined edges do not vary between edge and interior")
    _check(zero_other_roles > 0, "zero never appears outside the background role")
