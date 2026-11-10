"""Contour loom: carry a radial vector word clockwise around nested frames.

Only NumPy and the standard library are used.  Generation labels use a coordinate
formula; solve and validation traverse the actual input's rectangular contours.
There is no parent import, stored puzzle, seed lookup, or solver-side RNG.
"""

import itertools
import json

import numpy as np


def _arc_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return grid


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _box(mask):
    rows, cols = np.nonzero(mask)
    _need(len(rows) > 0, "missing object")
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _decode(value):
    """Infer background, hub, source side, and complete vector word from pixels."""
    grid = _arc_grid(value)
    height, width = grid.shape
    background = int(grid[0, 0])
    _need(all(int(grid[p]) == background for p in
              ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))),
          "all four corners must show background")
    hubs = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        mask = grid == color
        top, left, bottom, right = _box(mask)
        if (bottom - top >= 2 and right - left >= 2 and top >= 1
                and top == left == height - bottom == width - right
                and np.count_nonzero(mask) == (bottom - top) * (right - left)):
            hubs.append((color, top, left, bottom, right))
    _need(len(hubs) == 1, "need one solid hub with equal margins on all four sides")
    hub, top, left, bottom, right = hubs[0]
    layers = top
    payload = (grid != background) & (grid != hub)
    st, sl, sb, sr = _box(payload)
    _need(np.count_nonzero(payload) == (sb - st) * (sr - sl),
          "seed must be a filled rectangle")
    sides = []
    if st == 0 and sb == top and left <= sl < sr <= right:
        sides.append(("top", 0))
    if sl == right and sr == width and top <= st < sb <= bottom:
        sides.append(("right", 1))
    if st == bottom and sb == height and left <= sl < sr <= right:
        sides.append(("bottom", 2))
    if sl == 0 and sr == left and top <= st < sb <= bottom:
        sides.append(("left", 3))
    _need(len(sides) == 1, "seed must join one hub face to the corresponding canvas edge")
    side, turns = sides[0]
    canonical = np.ascontiguousarray(np.rot90(grid, turns))
    mask = (canonical != background) & (canonical != hub)
    _, start, _, end = _box(mask)
    word = canonical[:layers, start:end].copy()
    return canonical, dict(background=background, hub=hub, layers=layers,
                           word=word, start=start, turns=turns, side=side,
                           hub_shape=(canonical.shape[0] - 2 * layers,
                                      canonical.shape[1] - 2 * layers),
                           offset=start - layers)


def _ring(shape, depth):
    """Clockwise perimeter from its upper left, with each corner counted once."""
    height, width = shape
    top = left = depth
    bottom, right = height - 1 - depth, width - 1 - depth
    return ([(top, c) for c in range(left, right + 1)]
            + [(r, right) for r in range(top + 1, bottom + 1)]
            + [(bottom, c) for c in range(right - 1, left - 1, -1)]
            + [(r, left) for r in range(bottom - 1, top, -1)])


def solve(input_grid: np.ndarray) -> np.ndarray:
    canonical, info = _decode(input_grid)
    result = canonical.copy()
    for depth, word in enumerate(info["word"]):
        contour = _ring(result.shape, depth)
        start_index = info["start"] - depth
        contour = contour[start_index:] + contour[:start_index]
        for index, point in enumerate(contour):
            result[point] = word[index % len(word)]
    return np.ascontiguousarray(np.rot90(result, -info["turns"]))


def _coordinate_output(grid, layers, start, word):
    """Construction formula, rather than calling solve to create its own labels."""
    height, width = grid.shape
    result = grid.copy()
    for r in range(height):
        for c in range(width):
            depth = min(r, c, height - 1 - r, width - 1 - c)
            if depth >= layers:
                continue
            bottom, right = height - 1 - depth, width - 1 - depth
            horizontal, vertical = right - depth, bottom - depth
            if r == depth:
                along = c - depth
            elif c == right:
                along = horizontal + r - depth
            elif r == bottom:
                along = horizontal + vertical + right - c
            else:
                along = 2 * horizontal + vertical + bottom - r
            circumference = 2 * (horizontal + vertical)
            index = (along - (start - depth)) % circumference
            result[r, c] = word[depth, index % word.shape[1]]
    return result


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _period(word, divisor=True):
    """Period of a vector word; only a divisor is a legal complete-word reduction."""
    width = word.shape[1]
    return next(p for p in range(1, width + 1)
                if (not divisor or width % p == 0)
                and np.array_equal(word, word[:, np.arange(width) % p]))


def _alternative(value, kind):
    """Deliberately wrong, plausible simplifications used as training witnesses."""
    canonical, info = _decode(value)
    result = canonical.copy()
    matrix = info["word"]
    if kind == "straight":
        for depth, word in enumerate(matrix):
            for c in range(info["start"], result.shape[1]):
                result[depth, c] = word[(c - info["start"]) % len(word)]
        return np.ascontiguousarray(np.rot90(result, -info["turns"]))
    if kind == "reverse_depth":
        matrix = matrix[::-1]
    elif kind == "compress_rows":
        matrix = matrix[np.r_[True, np.any(matrix[1:] != matrix[:-1], axis=1)]]
    width = info["word"].shape[1]
    prefix = _period(info["word"], divisor=False)
    for depth in range(info["layers"]):
        word = matrix[depth % len(matrix)]
        contour = _ring(result.shape, depth)
        start = info["start"] - depth
        if kind == "counterclockwise":
            start += width - 1
            contour = [contour[(start - i) % len(contour)] for i in range(len(contour))]
            word = word[::-1]
        else:
            contour = contour[start:] + contour[:start]
        phase, previous_step = 0, None
        for index, point in enumerate(contour):
            if index:
                prior = contour[index - 1]
                step = point[0] - prior[0], point[1] - prior[1]
                if kind == "corner_reset" and previous_step is not None and step != previous_step:
                    phase = 0
                previous_step = step
            if kind == "solid_frames":
                color = word[0]
            elif kind == "prefix_period":
                color = word[index % prefix]
            else:
                color = word[phase % width]
            result[point] = color
            phase += 1
    # Even the wrong hypotheses retain authoritative source pixels.  Their
    # witnesses therefore concern newly inferred cells, not trivial seed damage.
    material = canonical != info["background"]
    result[material] = canonical[material]
    return np.ascontiguousarray(np.rot90(result, -info["turns"]))


def _unique(pair, seen, background):
    if _key(pair["input"]) not in seen:
        return pair
    # Preserve the assigned background, and try injective payload/hub recolorings.
    # There are at least 9*8 options and fewer than 14 exclusions in a puzzle.
    colors = [int(c) for c in np.unique(pair["input"]) if c != background]
    available = [c for c in range(10) if c != background]
    for replacements in itertools.permutations(available, len(colors)):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = replacements
        candidate = {name: mapping[grid] for name, grid in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    raise AssertionError("finite uniqueness space exhausted")


def _make_pair(rng, background, layers_range, widths, h_range, w_range,
               kind="free", rotation=None):
    visible = int(rng.choice(widths))
    hub_h = int(rng.integers(h_range[0], h_range[1] + 1))
    hub_w = int(rng.integers(max(visible, w_range[0]), w_range[1] + 1))
    maximum_layers = min(layers_range[1], (30 - max(hub_h, hub_w)) // 2)
    layers = int(rng.integers(layers_range[0], maximum_layers + 1))
    colors = [int(c) for c in rng.permutation([c for c in range(10) if c != background])]
    hub = colors[0]
    minimum = 3 if kind == "whole_word" else 2 if kind in ("copies", "joint") else 1
    palette = colors[1:1 + int(rng.integers(minimum, 9))]
    word = rng.choice(palette, size=(layers, visible)).astype(np.int8)
    if kind == "whole_word":
        # Every cross-sectional column has the vector-word form X Y X.
        word[:, 2] = word[:, 0]
        word[0] = [palette[0], palette[1], palette[0]]
        word[1] = word[0]
        word[-1] = [palette[2], palette[1], palette[2]]
    elif kind == "copies":
        base = word[:, :2].copy()
        base[0] = palette[:2]
        base[-1] = palette[1::-1]
        word = np.tile(base, (1, 2))
    else:
        style = int(rng.integers(6))
        if style == 1:
            word[:] = word[:, :1]  # Classical solid-frame limit.
        elif style == 2:
            word[:] = word[:1]  # Same tangential word at every radial depth.
        elif style == 3 and layers >= 2:
            row = int(rng.integers(layers - 1))
            word[row + 1] = word[row]
        elif style == 4 and visible >= 3:
            word[:, -1] = word[:, 0]
        elif style == 5 and layers >= 3:
            word[-1] = word[0]
        if kind == "joint":
            # Test the interaction even if a random style selects uniform rows.
            assert layers >= 2 and visible >= 2
            word[0, :2] = palette[:2]
            word[-1, 0] = palette[1]
    offset = int(rng.integers(hub_w - visible + 1))
    start = layers + offset
    grid = np.full((hub_h + 2 * layers, hub_w + 2 * layers), background, dtype=np.int8)
    grid[layers:layers + hub_h, layers:layers + hub_w] = hub
    grid[:layers, start:start + visible] = word
    output = _coordinate_output(grid, layers, start, word)
    rotation = int(rng.integers(4)) if rotation is None else rotation
    return {"input": np.ascontiguousarray(np.rot90(grid, rotation)),
            "output": np.ascontiguousarray(np.rot90(output, rotation))}


def _check_count(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _check_count(num_train, "num_train", 2, 10)
    num_test = _check_count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        seed %= 1 << 64
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # Larger radius, longer word, or new aspect ratio.
    backgrounds = [int(c) for c in rng.permutation(10)]
    first_rotation = int(rng.integers(4))
    second_rotation = (first_rotation + int(rng.integers(1, 4))) % 4
    seen, train, test = set(), [], []
    for index in range(num_train):
        if index == 0:
            kwargs = dict(layers_range=(3, 6), widths=[3], kind="whole_word",
                          rotation=first_rotation)
        elif index == 1:
            kwargs = dict(layers_range=(2, 7), widths=[4], kind="copies",
                          rotation=second_rotation)
        else:
            kwargs = dict(layers_range=(1, 7 if regime == 0 else 10),
                          widths=range(1, 5 if regime == 1 else 8))
        h_range, w_range = ((3, 8), (4, 8)) if regime == 2 else ((2, 8), (4, 10))
        background = backgrounds[index % 10]
        pair = _make_pair(rng, background, h_range=h_range, w_range=w_range, **kwargs)
        pair = _unique(pair, seen, background)
        seen.add(_key(pair["input"]))
        train.append(pair)
    rng.shuffle(train)
    for index in range(num_test):
        if regime == 0:
            args = dict(layers_range=(8, 12), widths=range(2 if index == 0 else 1, 7),
                        h_range=(2, 6), w_range=(3, 8))
        elif regime == 1:
            args = dict(layers_range=(2 if index == 0 else 1, 9), widths=range(5, 9),
                        h_range=(2, 8), w_range=(5, 12))
        else:
            args = dict(layers_range=(2 if index == 0 else 1, 8),
                        widths=range(2 if index == 0 else 1, 8),
                        h_range=(2, 2), w_range=(9, 14))
        background = backgrounds[(num_train + index) % 10]
        pair = _make_pair(rng, background, kind="joint" if index == 0 else "free", **args)
        pair = _unique(pair, seen, background)
        seen.add(_key(pair["input"]))
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(value):
    canonical, info = _decode(value)
    info["shape"] = tuple(np.asarray(value).shape)
    info["visible"] = info["word"].shape[1]
    info["period"] = _period(info["word"])
    info["aspect"] = max(info["hub_shape"]) / min(info["hub_shape"])
    info["radial_repeat"] = bool(np.any(np.all(info["word"][1:] == info["word"][:-1], axis=1)))
    info["radial_asymmetry"] = not np.array_equal(info["word"], info["word"][::-1])
    info["clipped"] = any(len(_ring(canonical.shape, d)) % info["period"] != 0
                          for d in range(info["layers"]))
    info["palette"] = len(np.unique(info["word"]))
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _arc_grid(pair["input"]), _arc_grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            canonical, info = _decode(inp)
            predicted = solve(inp)
            assert np.array_equal(out, predicted), "wrong contour continuation"
            authoritative = inp != info["background"]
            assert np.array_equal(out[authoritative], inp[authoritative]), "changed hub or seed"
            assert info["background"] not in out, "unfinished frame"
            assert set(np.unique(out)) == set(np.unique(inp)) - {info["background"]}
            normalized_out = np.rot90(out, info["turns"])
            visited = set()
            for depth, word in enumerate(info["word"]):
                ring = _ring(canonical.shape, depth)
                assert not visited.intersection(ring)
                visited.update(ring)
                start = info["start"] - depth
                ordered = ring[start:] + ring[:start]
                values = np.array([normalized_out[p] for p in ordered])
                assert np.array_equal(values, word[np.arange(len(ring)) % len(word)])
                assert len(ring) == 2 * (sum(canonical.shape) - 4 * depth) - 4
            assert len(visited) + np.count_nonzero(inp == info["hub"]) == inp.size
            records[split].append(_features(inp))
    training = records["train"]
    assert len({f["background"] for f in training}) >= 2
    assert len({f["side"] for f in training}) >= 2
    assert any(f["radial_repeat"] for f in training)
    assert any(f["radial_asymmetry"] for f in training)
    assert any(_period(f["word"], divisor=False) < f["period"] for f in training)
    assert any(f["visible"] > f["period"] > 1 for f in training)
    for kind in ("solid_frames", "straight", "reverse_depth", "compress_rows",
                 "prefix_period", "corner_reset", "counterclockwise"):
        assert any(not np.array_equal(_alternative(p["input"], kind), p["output"])
                   for p in puzzle["train"]), f"training fails to distinguish {kind}"
    for f in records["test"]:
        assert any(f[name] > max(t[name] for t in training)
                   for name in ("layers", "visible", "aspect")), "missing structural test shift"
    assert records["test"][0]["period"] >= 2
    assert records["test"][0]["radial_asymmetry"]
    for kind in ("solid_frames", "straight"):
        assert not np.array_equal(_alternative(puzzle["test"][0]["input"], kind),
                                  puzzle["test"][0]["output"]), "test omits a parent mechanism"


def _fixture(hub_shape, word, offset=0, rotation=0):
    word = np.asarray(word, dtype=np.int8)
    layers, visible = word.shape
    hh, hw = hub_shape
    assert visible + offset <= hw
    grid = np.zeros((hh + 2 * layers, hw + 2 * layers), dtype=np.int8)
    grid[layers:layers + hh, layers:layers + hw] = 9
    grid[:layers, layers + offset:layers + offset + visible] = word
    expected = _coordinate_output(grid, layers, layers + offset, word)
    return {name: np.ascontiguousarray(np.rot90(value, rotation))
            for name, value in (("input", grid), ("output", expected))}


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    features = [_features(p["input"]) for puzzle in puzzles
                for split in ("train", "test") for p in puzzle[split]]
    normalized = set()
    for f in features:
        roles = {}
        normalized.add((f["word"].shape, tuple(roles.setdefault(int(c), len(roles))
                                              for c in f["word"].flat)))
    assert len({f["shape"] for f in features}) >= 50
    assert len({f["layers"] for f in features}) >= 9
    assert len({f["hub_shape"] for f in features}) >= 30
    assert {f["visible"] for f in features} == set(range(1, 9))
    assert len({f["period"] for f in features}) >= 5
    assert {f["side"] for f in features} == {"top", "right", "bottom", "left"}
    assert {f["background"] for f in features} == set(range(10))
    assert len({f["offset"] for f in features}) >= 5
    assert len({f["palette"] for f in features}) >= 6
    assert len(normalized) >= 80
    assert any(f["visible"] == 1 for f in features)
    assert any(f["palette"] == 1 for f in features)
    assert any(f["clipped"] for f in features)
    assert any(not f["clipped"] and f["period"] > 1 for f in features)
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0, 4, 1))
    for counts in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        first = generate_puzzle(13579, *counts)
        assert _puzzle_key(first) == _puzzle_key(generate_puzzle(13579, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError(f"accepted illegal counts {counts}")
    fixtures = [
        _fixture((2, 2), [[3]]),
        _fixture((2, 2), np.array([[1 + d % 8, 1 + (d + 2) % 8] for d in range(14)])),
        _fixture((2, 28), [[1 + i % 8 for i in range(28)]]),
        _fixture((3, 6), [[1, 2, 1], [1, 2, 1], [3, 4, 3]], offset=1),
    ]
    for pair in fixtures:
        for turns in range(4):
            inp = np.rot90(pair["input"], turns)
            expected = np.rot90(pair["output"], turns)
            assert np.array_equal(solve(inp), expected), "rotation or boundary error"
            # Literal palette roles are inferred afresh, even when black is payload.
            mapping = np.array([8, 0, 6, 2, 7, 1, 3, 4, 9, 5], dtype=np.int8)
            assert np.array_equal(solve(mapping[inp]), mapping[expected])
    aba = fixtures[-1]
    assert not np.array_equal(_alternative(aba["input"], "prefix_period"), aba["output"])
    assert not np.array_equal(_alternative(aba["input"], "counterclockwise"), aba["output"])
    repaired = _unique(aba, {_key(aba["input"])}, 0)
    assert _key(repaired["input"]) != _key(aba["input"])
    assert _decode(repaired["input"])[1]["background"] == 0
    assert np.array_equal(solve(repaired["input"]), repaired["output"])
    assert _key(repaired["input"]) == _key(_unique(aba, {_key(aba["input"])}, 0)["input"])
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an incorrect frame")
    return {"seeds": 42, "pairs": len(features), "distinct_puzzles": 42,
            "input_shapes": len({f["shape"] for f in features}),
            "hub_shapes": len({f["hub_shape"] for f in features}),
            "layer_counts": sorted({f["layers"] for f in features}),
            "visible_word_lengths": sorted({f["visible"] for f in features}),
            "primitive_periods": sorted({f["period"] for f in features}),
            "normalized_seed_structures": len(normalized),
            "note": "Author-owned invariant and diversity checks; no independent solver."}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
