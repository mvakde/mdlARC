"""Linked plaques: color-path assembly with exchanged corner tiles and thread.

Only the standard library and NumPy are used. solve() reads no examples, seed,
metadata, or files. Public generation defaults are exactly four train/one test.
"""
from __future__ import annotations

import itertools
import json
from collections import deque

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and 2 <= num_train <= 10,
             "num_train must be an integer in 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4,
             "num_test must be an integer in 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= d <= 30 for d in array.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(array.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= array) & (array <= 9))), "colors must be 0..9")
    return array.astype(np.int8, copy=True)


def _key(grid):
    return tuple(grid.shape), grid.tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        queue = deque([first])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        groups.append(cells)
    return groups


def _pin_positions(h, w, diagonal):
    if diagonal == 0:  # pins NE/SW
        return ((1, w - 2), (h - 2, 1))
    return ((1, 1), (h - 2, w - 2))


def _solid(card):
    patch = np.full((card["h"], card["w"]), card["body"], dtype=np.int8)
    for r, c in _pin_positions(card["h"], card["w"], card["diagonal"]):
        patch[r, c] = card["pin"]
    return patch


def _analyze_input(value):
    grid = _grid(value)
    height, width = grid.shape
    background = int(grid[0, 0])
    _require(all(int(grid[r, c]) == background for r, c in
                 ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))),
             "all four corners must be background")
    frequencies = np.bincount(grid.ravel(), minlength=10)
    _require(all(frequencies[background] > frequencies[c] for c in range(10)
                 if c != background), "background must be uniquely most frequent")
    groups = _components(grid != background)
    singletons = [cells for cells in groups if len(cells) == 1]
    _require(len(singletons) == 1, "exactly one isolated thread key required")
    spool = singletons[0][0]
    thread = int(grid[spool])
    labels = np.full(grid.shape, -1, dtype=int)
    for index, cells in enumerate(groups):
        for r, c in cells:
            labels[r, c] = index
    for cells in groups:
        for r, c in cells:
            for dr, dc in ((1, 1), (1, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    _require(labels[rr, cc] in (-1, labels[r, c]),
                             "source objects must have an eight-neighbor gap")
    cards = []
    for cells in groups:
        if len(cells) == 1:
            continue
        top = min(r for r, _ in cells)
        left = min(c for _, c in cells)
        bottom = max(r for r, _ in cells)
        right = max(c for _, c in cells)
        h, w = bottom - top + 1, right - left + 1
        _require(h >= 7 and w >= 7 and len(cells) == h * w,
                 "plaques must be filled rectangles at least 7x7")
        _require(top > 0 and left > 0 and bottom < height - 1 and right < width - 1,
                 "plaques require an exterior margin")
        patch = grid[top:bottom + 1, left:right + 1]
        colors, counts = np.unique(patch, return_counts=True)
        _require(len(colors) == 2 and sorted(counts.tolist()) == [2, h * w - 2],
                 "plaque requires a body and exactly two pins")
        body = int(colors[int(np.argmax(counts))])
        pin = int(colors[int(np.argmin(counts))])
        pins = set(map(tuple, np.argwhere(patch == pin).tolist()))
        diagonal = next((d for d in (0, 1) if pins == set(_pin_positions(h, w, d))), None)
        _require(diagonal is not None, "pins must occupy an inset corner diagonal")
        cards.append(dict(h=h, w=w, body=body, pin=pin, diagonal=diagonal,
                          top=top, left=left))
    _require(2 <= len(cards) <= 4, "requires two to four plaques")
    bodies = {card["body"] for card in cards}
    pins = {card["pin"] for card in cards}
    _require(len(bodies) == len(cards) and len(pins) == len(cards),
             "path cannot repeat a body or pin role")
    _require(thread not in bodies | pins and background not in bodies | pins,
             "key, background and nodes must have distinct roles")
    starts = bodies - pins
    _require(len(starts) == 1 and len(pins - bodies) == 1,
             "requires a unique directed path start and end")
    by_body = {card["body"]: i for i, card in enumerate(cards)}
    node = next(iter(starts))
    order = []
    while node in by_body:
        index = by_body[node]
        _require(index not in order, "cycle in color path")
        order.append(index)
        node = cards[index]["pin"]
    _require(len(order) == len(cards) and len(bodies | pins) == len(cards) + 1,
             "all plaques must belong to one simple path")
    chain_h = 1 + sum(card["h"] - 1 for card in cards)
    chain_w = 1 + sum(card["w"] - 1 for card in cards)
    _require(chain_h <= height and chain_w <= width, "assembled chain does not fit")
    return dict(grid=grid, background=background, thread=thread, spool=spool,
                cards=cards, order=order, chain_shape=(chain_h, chain_w))


def _dash(gap, two_ended=True):
    return [((min(k, gap + 1 - k) if two_ended else k) % 2 == 1)
            for k in range(1, gap + 1)]


def _frame(card, background, thread, *, exchange=True, two_ended=True):
    h, w = card["h"], card["w"]
    body, pin = card["body"], card["pin"]
    result = np.full((h, w), background, dtype=np.int8)
    pins = set(_pin_positions(h, w, card["diagonal"]))
    for r in (1, h - 2):
        for c in (1, w - 2):
            center = pin if (r, c) in pins else body
            other = body if center == pin else pin
            result[r - 1:r + 2, c - 1:c + 2] = other if exchange else center
            result[r, c] = center
    for k, painted in enumerate(_dash(w - 6, two_ended), start=3):
        if painted:
            result[1, k] = result[h - 2, k] = thread
    for k, painted in enumerate(_dash(h - 6, two_ended), start=3):
        if painted:
            result[k, 1] = result[k, w - 2] = thread
    return result


def _assemble(info, *, order=None, solid=False, earlier=False,
              exchange=True, two_ended=True, in_place=False):
    result = np.full(info["grid"].shape, info["background"], dtype=np.int8)
    order = info["order"] if order is None else order
    patches = []
    r = c = 0
    for index in order:
        card = info["cards"][index]
        patch = (_solid(card) if solid else
                 _frame(card, info["background"], info["thread"],
                        exchange=exchange, two_ended=two_ended))
        if in_place:
            rr, cc = card["top"], card["left"]
        else:
            rr, cc = r, c
        patches.append((rr, cc, patch))
        r += card["h"] - 1
        c += card["w"] - 1
    for r, c, patch in (reversed(patches) if earlier else patches):
        h, w = patch.shape
        result[r:r + h, c:c + w] = patch
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the unique color path and render it, using only input pixels."""
    return _assemble(_analyze_input(input_grid))


def _sample_dimensions(rng, count, teaching, long_side):
    for _ in range(1000):
        if teaching == 0:
            sizes = [(int(rng.choice((10, 12, 14))), int(rng.choice((10, 12, 14)))),
                     (int(rng.choice((7, 9, 11))), int(rng.choice((7, 9, 11))))]
        else:
            upper = 10 if count == 4 else 12
            if teaching == 1:
                upper = 11
            sizes = [(int(rng.integers(7, upper)), int(rng.integers(7, upper)))
                     for _ in range(count)]
        if long_side:
            # Leave enough cumulative span for three links as well as two.
            sizes = [(int(rng.integers(7, 10)), int(rng.integers(7, 10)))
                     for _ in range(count)]
            at = int(rng.integers(count))
            axis = int(rng.integers(2))
            sizes[at] = tuple(int(rng.integers(15, 19)) if j == axis else sizes[at][j]
                              for j in range(2))
        if all(1 + sum(size[axis] - 1 for size in sizes) <= 30 for axis in (0, 1)):
            return sizes
    raise RuntimeError("dimension sampling exhausted")


def _sample_layout(rng, sizes):
    count = len(sizes)
    chain_h = 1 + sum(h - 1 for h, _ in sizes)
    chain_w = 1 + sum(w - 1 for _, w in sizes)
    for _ in range(300):
        permutation = rng.permutation(count).tolist()
        cuts = [i for i in range(1, count) if rng.random() < 0.5]
        rows = []
        start = 0
        for stop in cuts + [count]:
            rows.append(permutation[start:stop])
            start = stop
        gaps = [[int(rng.integers(1, 4)) for _ in range(len(row) - 1)] for row in rows]
        row_heights = [max(sizes[i][0] for i in row) for row in rows]
        row_widths = [sum(sizes[i][1] for i in row) + sum(gap)
                      for row, gap in zip(rows, gaps)]
        vertical_gaps = [int(rng.integers(1, 4)) for _ in range(len(rows) - 1)]
        layout_h = sum(row_heights) + sum(vertical_gaps)
        layout_w = max(row_widths)
        min_h, min_w = max(chain_h, layout_h + 2), max(chain_w, layout_w + 2)
        if min_h > 30 or min_w > 30:
            continue
        def canvas_size(minimum):
            if rng.random() < 0.2:
                return 30
            return minimum + int(rng.integers(0, min(6, 30 - minimum) + 1))
        height, width = canvas_size(min_h), canvas_size(min_w)
        top = int(rng.integers(1, height - layout_h))
        positions = [None] * count
        for j, (row, gap) in enumerate(zip(rows, gaps)):
            left = int(rng.integers(1, width - row_widths[j]))
            for k, index in enumerate(row):
                h, w = sizes[index]
                jitter = int(rng.integers(0, row_heights[j] - h + 1))
                positions[index] = (top + jitter, left)
                left += w + (gap[k] if k < len(gap) else 0)
            top += row_heights[j] + (vertical_gaps[j] if j < len(vertical_gaps) else 0)
        return (height, width), positions
    return None


def _make_pair(rng, count, background, thread, teaching=-1, long_side=False):
    for _ in range(300):
        sizes = _sample_dimensions(rng, count, teaching, long_side)
        layout = _sample_layout(rng, sizes)
        if layout is None:
            continue
        shape, positions = layout
        reading = sorted(range(count), key=lambda i: positions[i])
        # These are evidence requirements for two demonstrations, not parsing rules.
        if teaching == 0 and reading == list(range(count)):
            continue
        if teaching == 1 and reading in (list(range(count)), list(reversed(range(count)))):
            continue
        available = [c for c in range(10) if c not in (background, thread)]
        nodes = [int(c) for c in rng.choice(available, size=count + 1, replace=False)]
        if teaching == 1 and nodes[:-1] in (sorted(nodes[:-1]), sorted(nodes[:-1], reverse=True)):
            # Color equality, rather than ascending or descending color numbers.
            nodes[0], nodes[1] = nodes[1], nodes[0]
        diagonals = rng.integers(0, 2, size=count).tolist()
        if teaching == 0:
            diagonals = [1, 0]  # explicit role exchange AND visible overwrite
        grid = np.full(shape, background, dtype=np.int8)
        for i, ((h, w), (r, c)) in enumerate(zip(sizes, positions)):
            card = dict(h=h, w=w, body=nodes[i], pin=nodes[i + 1], diagonal=diagonals[i])
            grid[r:r + h, c:c + w] = _solid(card)
        free = grid == background
        occupied = ~free
        # Singleton must be isolated in all eight directions.
        for dr, dc in itertools.product((-1, 0, 1), repeat=2):
            src_r = slice(max(0, -dr), min(shape[0], shape[0] - dr))
            src_c = slice(max(0, -dc), min(shape[1], shape[1] - dc))
            dst_r = slice(max(0, dr), min(shape[0], shape[0] + dr))
            dst_c = slice(max(0, dc), min(shape[1], shape[1] + dc))
            free[dst_r, dst_c] &= ~occupied[src_r, src_c]
        free[0, 0] = free[0, -1] = free[-1, 0] = free[-1, -1] = False
        candidates = np.argwhere(free)
        if not len(candidates):
            continue
        r, c = candidates[int(rng.integers(len(candidates)))]
        grid[r, c] = thread
        try:
            info = _analyze_input(grid)
        except ValueError:
            continue
        return {"input": grid, "output": _assemble(info)}
    raise RuntimeError("pair sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one deterministic, internally coherent puzzle with visible roles."""
    num_train, num_test = _counts(num_train, num_test)
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = "count" if rng.random() < 0.5 else "long_side"
    fixed_palette = bool(rng.integers(0, 2))
    base_bg, base_thread = [int(c) for c in rng.choice(10, size=2, replace=False)]
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_roles = None
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            teaching = i if split == "train" and i < 2 else -1
            long_side = split == "test" and shift == "long_side"
            if teaching == 0:
                count = 2
            elif teaching == 1:
                count = 3
            elif split == "test" and shift == "count":
                count = 4
            elif long_side:
                count = int(rng.choice((2, 3)))
            else:
                count = int(rng.integers(2, 4 if shift == "count" else 5))
            for _ in range(100):
                if fixed_palette:
                    background, thread = base_bg, base_thread
                else:
                    role_choices = [(a, b) for a in range(10) for b in range(10)
                                    if a != b and not (teaching == 1 and previous_roles
                                                     and (a == previous_roles[0]
                                                          or b == previous_roles[1]))]
                    background, thread = role_choices[int(rng.integers(len(role_choices)))]
                pair = _make_pair(rng, count, background, thread, teaching, long_side)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if teaching == 0:
                        previous_roles = background, thread
                    break
            else:
                raise RuntimeError("unique pair sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_geometry(info, output):
    """Local invariants plus area counting, beyond whole-solver agreement."""
    bg, thread = info["background"], info["thread"]
    n = len(info["cards"])
    occupied = sum(36 + 2 * sum(_dash(card["h"] - 6))
                   + 2 * sum(_dash(card["w"] - 6)) for card in info["cards"]) - (n - 1)
    assert np.count_nonzero(output != bg) == occupied, "incorrect frame union area"
    row = column = 0
    for j, index in enumerate(info["order"]):
        card = info["cards"][index]
        h, w = card["h"], card["w"]
        source = _solid(card)
        pins = set(_pin_positions(h, w, card["diagonal"]))
        for r in (1, h - 2):
            for c in (1, w - 2):
                center = card["pin"] if (r, c) in pins else card["body"]
                assert output[row + r, column + c] == source[r, c] == center
                other = card["body"] if center == card["pin"] else card["pin"]
                for dr, dc in itertools.product((-1, 0, 1), repeat=2):
                    rr, cc = r + dr, c + dc
                    if (dr == dc == 0 or (j < n - 1 and (rr, cc) == (h - 1, w - 1))):
                        continue
                    assert output[row + rr, column + cc] == other, "ring role/ownership error"
        for k in range(3, w - 3):
            color = thread if min(k - 2, w - 3 - k) % 2 else bg
            assert output[row + 1, column + k] == color
            assert output[row + h - 2, column + k] == color
        for k in range(3, h - 3):
            color = thread if min(k - 2, h - 3 - k) % 2 else bg
            assert output[row + k, column + 1] == color
            assert output[row + k, column + w - 2] == color
        assert np.all(output[row + 3:row + h - 3, column + 3:column + w - 3] == bg)
        assert output[row, column] == _frame(card, bg, thread)[0, 0]
        row += h - 1
        column += w - 1


def _ablations(info):
    reading = list(range(len(info["cards"])))
    return {
        "omit_A": _assemble(info, in_place=True),
        "omit_B": _assemble(info, solid=True),
        "reading_order": _assemble(info, order=reading),
        "reverse_reading": _assemble(info, order=list(reversed(reading))),
        "numeric_order": _assemble(info, order=sorted(reading, key=lambda i: info["cards"][i]["body"])),
        "reverse_numeric": _assemble(info, order=sorted(reading, key=lambda i: info["cards"][i]["body"], reverse=True)),
        "no_exchange": _assemble(info, exchange=False),
        "one_ended": _assemble(info, two_ended=False),
        "earlier_owns": _assemble(info, earlier=True),
    }


def validate_generated_puzzle(puzzle):
    """Validate the family and the evidence supplied by its demonstrations."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    infos = {"train": [], "test": []}
    witnesses = {name: False for name in ("omit_A", "omit_B", "reading_order",
                                         "reverse_reading", "numeric_order", "reverse_numeric", "no_exchange",
                                         "one_ended", "earlier_owns")}
    restart_h = restart_w = odd_h = odd_w = False
    diagonals = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert input_grid.shape == output.shape
            assert _key(input_grid) not in seen, "duplicate input"
            seen.add(_key(input_grid))
            info = _analyze_input(input_grid)
            infos[split].append(info)
            assert np.array_equal(output, solve(input_grid)), "output disagrees with solve"
            assert not np.array_equal(input_grid, output), "identity pair"
            assert set(np.unique(output)) <= set(np.unique(input_grid)), "unencoded output color"
            _check_geometry(info, output)
            if split == "train":
                for name, alternate in _ablations(info).items():
                    witnesses[name] |= not np.array_equal(output, alternate)
                for card in info["cards"]:
                    gh, gw = card["h"] - 6, card["w"] - 6
                    restart_h |= gh >= 4 and gh % 2 == 0
                    restart_w |= gw >= 4 and gw % 2 == 0
                    odd_h |= gh % 2 == 1
                    odd_w |= gw % 2 == 1
                    diagonals.add(card["diagonal"])
    assert all(witnesses.values()), f"missing teaching witnesses: {witnesses}"
    assert restart_h and restart_w and odd_h and odd_w and diagonals == {0, 1}
    train_counts = {len(info["cards"]) for info in infos["train"]}
    max_side = max(max(card["h"], card["w"]) for info in infos["train"] for card in info["cards"])
    for info in infos["test"]:
        assert (len(info["cards"]) > max(train_counts)
                or max(max(c["h"], c["w"]) for c in info["cards"]) > max_side), "missing structural shift"


def _puzzle_key(puzzle):
    return tuple((_key(np.asarray(pair["input"], dtype=np.int8)),
                  _key(np.asarray(pair["output"], dtype=np.int8)))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded checks over 42 seeds; return descriptive structural metrics."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, structures, node_palettes, backgrounds, threads, diagonals = (set() for _ in range(6))
    counts = {"train": set(), "test": set()}
    shifts = set()
    boundaries = set()
    mixed_palettes = 0
    for seed, puzzle in enumerate(puzzles):
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterminism"
        roles = set()
        training_counts = {len(_analyze_input(p["input"])["cards"]) for p in puzzle["train"]}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_input(pair["input"])
                cards = [info["cards"][i] for i in info["order"]]
                shapes.add(pair["input"].shape)
                structures.add(tuple((c["h"], c["w"], c["diagonal"]) for c in cards))
                node_palettes.add(tuple(c["body"] for c in cards) + (cards[-1]["pin"],))
                backgrounds.add(info["background"])
                threads.add(info["thread"])
                diagonals.update(c["diagonal"] for c in cards)
                roles.add((info["background"], info["thread"]))
                counts[split].add(len(cards))
                if split == "test":
                    shifts.add("count" if len(cards) > max(training_counts) else "long_side")
                if info["chain_shape"][0] == pair["input"].shape[0]:
                    boundaries.add("bottom")
                if info["chain_shape"][1] == pair["input"].shape[1]:
                    boundaries.add("right")
        mixed_palettes += len(roles) > 1
        # A global color permutation must commute with the input-only solver.
        permutation = np.random.default_rng(seed + 9000).permutation(10)
        pair = puzzle["test"][0]
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert counts["train"] == counts["test"] == {2, 3, 4}
    assert shifts == {"count", "long_side"} and boundaries == {"bottom", "right"}
    assert len(shapes) >= 25 and len(structures) >= 150 and len(node_palettes) >= 100
    assert len(backgrounds) >= 8 and len(threads) >= 8 and diagonals == {0, 1}
    assert 0 < mixed_palettes < 42
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(123400 + train + test, train, test)
        assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    # Directly constructed, not sampled: exercise a legal side longer than 18,
    # both checkerboard diagonals, and an exact bottom-boundary chain.
    extended = np.full((30, 30), 0, dtype=np.int8)
    first = dict(h=24, w=7, body=6, pin=1, diagonal=0)
    second = dict(h=7, w=7, body=1, pin=8, diagonal=1)
    extended[2:26, 2:9] = _solid(first)
    extended[3:10, 15:22] = _solid(second)
    extended[28, 27] = 4
    extended_output = solve(extended)
    _check_geometry(_analyze_input(extended), extended_output)
    assert extended_output.shape == (30, 30)
    assert np.any(extended_output[-1] != 0)
    # Moving the source plaques changes neither the path nor the reconstruction.
    relocated = np.full((30, 30), 0, dtype=np.int8)
    relocated[1:25, 20:27] = _solid(first)
    relocated[15:22, 3:10] = _solid(second)
    relocated[28, 2] = 4
    assert np.array_equal(solve(relocated), extended_output)
    for train, test in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                        (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    malformed = puzzles[0]["train"][0]["input"].copy()
    info = _analyze_input(malformed)
    malformed[info["spool"]] = info["background"]
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("missing thread key accepted")
    return {"seeds": 42, "distinct_puzzles": 42, "input_shapes": len(shapes),
            "ordered_frame_structures": len(structures), "node_palettes": len(node_palettes),
            "backgrounds": len(backgrounds), "thread_colors": len(threads),
            "counts": {k: sorted(v) for k, v in counts.items()},
            "shift_types": sorted(shifts), "mixed_palette_puzzles": mixed_palettes,
            "exact_fit_boundaries": sorted(boundaries), "independent_semantic_check": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
