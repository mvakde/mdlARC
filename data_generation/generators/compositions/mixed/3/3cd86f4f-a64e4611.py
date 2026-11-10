"""Bottom-anchored row shear followed by empty-neighborhood center marking.

Family conventions: 0 is background; 3 is reserved for newly marked centers.
Every other input color is foreground, and is copied without recoloring.
No parent module, saved example, seed or generation state is used by solve.
"""
from __future__ import annotations

import numpy as np


FOREGROUND_COLORS = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=np.int8)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _input_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or any(n < 1 or n > 30 for n in grid.shape)):
        raise ValueError("input must be an integer rectangle with dimensions 1..30")
    if np.any((grid < 0) | (grid > 9)) or np.any(grid == 3):
        raise ValueError("input colors must be 0..9, with 3 reserved for output")
    if sum(grid.shape) - 1 > 30:
        raise ValueError("sheared width exceeds 30")
    return grid.astype(np.int8, copy=False)


def _shear(grid):
    h, w = grid.shape
    result = np.zeros((h, h + w - 1), dtype=grid.dtype)
    for r in range(h):
        start = h - 1 - r
        result[r, start:start + w] = grid[r]
    return result


def _near(mask, diagonals=True):
    """Dilation by king adjacency; outside the rectangle contributes nothing."""
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            if diagonals or dr == 1 or dc == 1:
                result |= padded[dr:dr + h, dc:dc + w]
    return result


def _centers(grid):
    return ~_near(grid != 0)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select all clear clipped 3x3 centers on the complete sheared canvas."""
    sheared = _shear(_input_grid(input_grid))
    mask = _centers(sheared)
    output = sheared.copy()
    output[mask] = 3
    return output


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _component_sizes(mask):
    unseen = set(zip(*np.nonzero(mask)))
    sizes = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        sizes.append(size)
    return sizes


def _features(inp):
    sheared = _shear(inp)
    marked = _centers(sheared)
    payload = _shear(np.ones_like(inp, dtype=bool))
    premature = _shear(_centers(inp))
    diagonal_fringe = ((sheared == 0) & _near(sheared != 0)
                       & ~_near(sheared != 0, diagonals=False))
    return {
        "centers": int(marked.sum()),
        "payload_centers": int((marked & payload).sum()),
        "padding_centers": int((marked & ~payload).sum()),
        "order_difference": int(((marked != premature) & payload).sum()),
        "edge_centers": int(marked[0].sum() + marked[-1].sum()
                            + marked[1:-1, 0].sum() + marked[1:-1, -1].sum()),
        "diagonal_fringe": int(diagonal_fringe.sum()),
        "footprint_overreach": int((_near(marked) & ~marked & (sheared == 0)).sum()),
        "singleton": 1 in _component_sizes(marked),
        "components": len(_component_sizes(marked)),
    }


def _paint(rng, foreground, anchor=False):
    h, w = foreground.shape
    n = int(rng.integers(2 if anchor else 1, 9))
    palette = rng.choice(FOREGROUND_COLORS, size=n, replace=False)
    rows, cols = np.indices((h, w))
    style = int(rng.integers(0, 4))
    if style == 0:
        indices = rng.integers(0, n, size=(h, w))
    elif style == 1:
        indices = np.broadcast_to((cols // int(rng.integers(1, 4))) % n, (h, w))
    elif style == 2:
        indices = (2 * rows + cols // int(rng.integers(1, 4))) % n
    else:
        indices = np.broadcast_to(rng.integers(0, n, size=(1, w)), (h, w))
    return np.where(foreground, palette[indices], 0).astype(np.int8)


def _post_rectangle(mask, r0, r1, c0, c1):
    """Carve the inverse image of a rectangular hole in sheared coordinates."""
    h, w = mask.shape
    for r in range(r0, r1):
        start = h - 1 - r
        a, b = max(0, c0 - start), min(w, c1 - start)
        if a < b:
            mask[r, a:b] = False


def _anchor(rng, index):
    h = int(rng.integers(6 if index == 0 else 7, 11 if index == 0 else 12))
    w = int(rng.integers(10, min(16, 27 - h) + 1))
    fg = np.ones((h, w), dtype=bool)
    if index == 0:
        # A 3x3 hole surrounded by foreground AFTER shear. Its sole center
        # must be marked even though its candidate component has size one.
        r = int(rng.integers(2, h - 2))
        c = int(rng.integers(4, w - 4))
        target = h - 1 - r + c
        _post_rectangle(fg, r - 1, r + 2, target - 1, target + 2)
    else:
        # A straight original slit is slanted by shear. A wider chamber at
        # one end contrasts with the thin portion under the same predicate.
        thickness = int(rng.integers(3, 5))
        c = int(rng.integers(2, w - thickness - 1))
        fg[:, c:c + thickness] = False
        end = slice(0, 3) if rng.random() < .5 else slice(h - 3, h)
        fg[end, max(0, c - 2):min(w, c + thickness + 2)] = False
    return _paint(rng, fg, anchor=True)


def _random_geometry(rng, h, w):
    style = int(rng.integers(0, 7))
    if style == 0:  # irregular speckles with an optional broad opening
        fg = rng.random((h, w)) < float(rng.uniform(.12, .55))
    elif style == 1:  # separated foreground islands and short streaks
        fg = np.zeros((h, w), dtype=bool)
        for _ in range(int(rng.integers(1, 7))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            a, b = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            fg[r:min(h, r + a), c:min(w, c + b)] = True
    elif style == 2:  # enclosed or edge-touching clear rooms
        fg = np.ones((h, w), dtype=bool)
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            a, b = int(rng.integers(2, 8)), int(rng.integers(2, 9))
            fg[r:min(h, r + a), c:min(w, c + b)] = False
    elif style == 3:  # clean bands, varied thickness and orientation
        axis = int(rng.integers(2))
        length = (h, w)[axis]
        values = np.zeros(length, dtype=bool)
        i, value = 0, bool(rng.integers(2))
        while i < length:
            run = int(rng.integers(1, 6))
            values[i:i + run] = value
            i += run
            value = not value
        fg = np.broadcast_to(values[:, None] if axis == 0 else values[None, :], (h, w)).copy()
    elif style == 4:  # stepped corridor, sometimes widening or touching an edge
        fg = np.ones((h, w), dtype=bool)
        center = int(rng.integers(w))
        breadth = int(rng.integers(1, min(7, w) + 1))
        for r in range(h):
            if rng.random() < .45:
                center = int(np.clip(center + int(rng.integers(-1, 2)), 0, w - 1))
            if rng.random() < .25:
                breadth = int(rng.integers(1, min(7, w) + 1))
            a = max(0, center - breadth // 2)
            fg[r, a:min(w, a + breadth)] = False
    elif style == 5:  # holes that become aligned in the new coordinate frame
        fg = np.ones((h, w), dtype=bool)
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            target = h - 1 - r + c
            _post_rectangle(fg, r, min(h, r + int(rng.integers(2, 7))),
                            target, target + int(rng.integers(2, 8)))
    else:  # repeated rows with variable run lengths
        templates = rng.random((int(rng.integers(1, 4)), w)) < float(rng.uniform(.18, .65))
        fg = np.empty((h, w), dtype=bool)
        i = 0
        while i < h:
            run = int(rng.integers(1, 5))
            fg[i:i + run] = templates[int(rng.integers(len(templates)))]
            i += run
    # Some rooms have a foreground island; all incidental empty centers count.
    if style in (0, 2, 4, 5) and rng.random() < .28:
        fg[int(rng.integers(h)), int(rng.integers(w))] = True
    if not fg.any():
        fg[int(rng.integers(h)), int(rng.integers(w))] = True
    if fg.all():
        fg[int(rng.integers(h)), int(rng.integers(w))] = False
    return fg


def _test_shape(rng, mode, training):
    if mode == 0:  # taller than every demonstration, usually narrow
        h = int(rng.integers(max(g.shape[0] for g in training) + 2, 31))
        return h, int(rng.integers(1, min(8, 31 - h) + 1))
    if mode == 1:  # wider than every demonstration, short
        w = int(rng.integers(max(g.shape[1] for g in training) + 2, 30))
        return int(rng.integers(2, min(4, 31 - w) + 1)), w
    # Width-30 output; all demonstration outputs have width at most 26.
    h = int(rng.integers(2, 31))
    return h, 31 - h


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))
    result = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(64):
                if split == "train" and index < 2:
                    inp = _anchor(rng, index)
                else:
                    if split == "train":
                        h = int(rng.integers(2, 13))
                        w = int(rng.integers(3, min(18, 27 - h) + 1))
                    else:
                        h, w = _test_shape(rng, mode, [p["input"] for p in result["train"]])
                    inp = _paint(rng, _random_geometry(rng, h, w))
                if _key(inp) in seen:
                    continue
                if split == "train" and index == 1:
                    if inp.shape[0] == result["train"][0]["input"].shape[0]:
                        continue
                out = solve(inp)
                if split == "test" and not np.any(out == 3):
                    continue
                if split == "test" and index == 0 and not _features(inp)["order_difference"]:
                    continue
                seen.add(_key(inp))
                result[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("64 bounded attempts could not create a distinct informative pair")
    validate_generated_puzzle(result)
    return result


def _coordinate_check(inp, out):
    """Check via foreground-coordinate exclusion, separately from solve's windows."""
    h, w = inp.shape
    width = h + w - 1
    assert out.shape == (h, width)
    expected = np.zeros((h, width), dtype=np.int8)
    forbidden = np.zeros((h, width), dtype=bool)
    for r, c in zip(*np.nonzero(inp)):
        x = int(c) + h - 1 - int(r)
        expected[r, x] = inp[r, c]
        forbidden[max(0, r - 1):min(h, r + 2), max(0, x - 1):min(width, x + 2)] = True
    expected[~forbidden] = 3
    assert np.array_equal(out, expected), "foreground placement or clipped-distance predicate is wrong"
    for color in FOREGROUND_COLORS:
        assert np.count_nonzero(out == color) == np.count_nonzero(inp == color)
    assert np.count_nonzero(out == 3) == np.count_nonzero(~forbidden)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen = set()
    train_features = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _input_grid(pair["input"])
            out = np.asarray(pair["output"])
            assert out.ndim == 2 and np.issubdtype(out.dtype, np.integer)
            assert all(1 <= n <= 30 for n in out.shape)
            assert np.all((0 <= out) & (out <= 9))
            assert _key(inp) not in seen
            seen.add(_key(inp))
            _coordinate_check(inp, out)
            if split == "train":
                train_features.append(_features(inp))
    # These are evidence checks for generated puzzles, not restrictions on solve.
    assert len({np.asarray(p["input"]).shape[0] for p in puzzle["train"]}) >= 2
    assert any(f["singleton"] and f["payload_centers"] for f in train_features)
    assert any(f["order_difference"] for f in train_features)
    assert any(f["diagonal_fringe"] for f in train_features)
    assert any(f["footprint_overreach"] for f in train_features)
    assert any(f["edge_centers"] and f["padding_centers"] >= 2 for f in train_features)
    assert sum(f["centers"] for f in train_features) >= 3
    assert _features(_input_grid(puzzle["test"][0]["input"]))["order_difference"] > 0
    training = [np.asarray(p["input"]) for p in puzzle["train"]]
    for pair in puzzle["test"]:
        inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
        assert np.any(out == 3)
        assert (inp.shape[0] > max(g.shape[0] for g in training)
                or inp.shape[1] > max(g.shape[1] for g in training)
                or (out.shape[1] == 30 and max(sum(g.shape) - 1 for g in training) <= 26))


def self_test():
    """42 streamed seeds, all legal counts, hand cases; no files or stored runs."""
    shapes, occupancy, signatures, palette_sizes = set(), set(), set(), set()
    totals = {"seeds": 42, "pairs": 0, "order_sensitive_pairs": 0,
              "multicomponent_pairs": 0, "singleton_pairs": 0,
              "payload_highlight_pairs": 0, "boundary_width_pairs": 0}
    shift_modes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = []
        training = [p["input"] for p in puzzle["train"]]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
                shapes.add(inp.shape)
                occupancy.add(_key((inp != 0).astype(np.int8)))
                signature.append((_key(inp), _key(out)))
                palette_sizes.add(len(np.unique(inp[inp != 0])))
                f = _features(inp)
                totals["pairs"] += 1
                totals["order_sensitive_pairs"] += f["order_difference"] > 0
                totals["multicomponent_pairs"] += f["components"] > 1
                totals["singleton_pairs"] += f["singleton"]
                totals["payload_highlight_pairs"] += f["payload_centers"] > 0
                totals["boundary_width_pairs"] += out.shape[1] == 30
                if split == "test":
                    if inp.shape[0] > max(g.shape[0] for g in training):
                        shift_modes.add("taller")
                    if inp.shape[1] > max(g.shape[1] for g in training):
                        shift_modes.add("wider")
                    if out.shape[1] == 30:
                        shift_modes.add("full_width")
        signatures.add(tuple(signature))
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(occupancy) >= 140 and len(palette_sizes) >= 6
    assert shift_modes == {"taller", "wider", "full_width"}
    assert totals["order_sensitive_pairs"] >= 84
    assert totals["singleton_pairs"] >= 42
    assert totals["payload_highlight_pairs"] >= 42
    # Repeatability includes all generated structure, color and output arrays.
    for seed in (0, 41):
        a, b = generate_puzzle(seed), generate_puzzle(seed)
        for split in ("train", "test"):
            for p, q in zip(a[split], b[split]):
                assert all(np.array_equal(p[k], q[k]) for k in ("input", "output"))
    count_calls = 0
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(1000 + tr * 10 + te, tr, te)
            assert len(p["train"]) == tr and len(p["test"]) == te
            count_calls += 1
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-computed clipped boundaries and simultaneous center-only marking.
    hand = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.int8)
    answer = np.array([[3, 0, 1, 0, 3], [3, 0, 0, 0, 3], [3, 0, 1, 0, 3]])
    assert np.array_equal(solve(hand), answer)
    assert np.array_equal(solve(np.array([[0, 0, 1]])), [[3, 0, 1]])
    assert np.array_equal(solve(np.zeros((3, 2), dtype=int)), np.full((3, 4), 3))
    assert np.array_equal(solve(np.array([[8]])), [[8]])
    assert np.array_equal(solve(np.array([[0]])), [[3]])
    for h in range(1, 31):
        # Cover every legal input dimension pair, including degenerate axes.
        for w in range(1, 32 - h):
            inp = np.zeros((h, w), dtype=np.int8)
            inp[h // 2, w // 2] = 7
            _coordinate_check(inp, solve(inp))
    totals.update({"distinct_puzzles": len(signatures), "input_shapes": len(shapes),
                   "occupancy_structures": len(occupancy), "foreground_palette_sizes": sorted(palette_sizes),
                   "legal_count_calls": count_calls, "shift_kinds": sorted(shift_modes)})
    return totals
