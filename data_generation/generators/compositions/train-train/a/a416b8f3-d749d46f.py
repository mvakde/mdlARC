"""Hinged stamps on a cylindrical ribbon. Standard library and NumPy only."""

from __future__ import annotations

import math
import numpy as np


# Items are (anchor_row, anchor_column, source_height, source_width, fill_color).
# Horizontal coordinates are cyclic within the supplied input width.


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("a grid must have dimensions 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid colors must be integers 0..9")
    return a.astype(int, copy=False)


def _cross_cells(y, x, h, w, period):
    short, long = sorted((h, w))
    return {(y + r, (x + c) % period)
            for r in range(long) for c in range(long)
            if r < short or c < short}


def _halo(cells, period):
    return cells | {(y + dy, (x + dx) % period)
                    for y, x in cells
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))}


def _render_input(height, width, background, hinge, items):
    a = np.full((height, width), background, dtype=int)
    for y, x, h, w, color in items:
        for r in range(h):
            for c in range(w):
                a[y + r, (x + c) % width] = color
        a[y, x] = hinge
    return a


def _render_output(height, width, background, hinge, items):
    """Print complete periodic stamps, including their continuations at x=0."""
    period = np.full((height, width), background, dtype=int)
    for y, x, h, w, color in items:
        short, long = sorted((h, w))
        for r, c in _cross_cells(0, 0, h, w, width):
            period[y + r, (x + c) % width] = hinge if r < short and c < short else color
    return np.concatenate((period, period), axis=1)


def _decode(input_grid):
    a = _grid(input_grid)
    height, width = a.shape
    if not 7 <= width <= 15:
        raise ValueError("ribbon period must be 7..15")
    background = int(a[0, 0])
    if not np.all(a[0] == background) or not np.all(a[-1] == background):
        raise ValueError("the first and last rows must be background")
    colors = [int(c) for c in np.unique(a) if c != background]
    # Hinges are the unique color with no orthogonally adjacent equal cells.
    # All fill colors have a connected component of at least two cells.
    candidates = []
    for color in colors:
        mask = a == color
        if not (np.any(mask & np.roll(mask, 1, axis=1))
                or np.any(mask[1:] & mask[:-1])):
            candidates.append(color)
    if len(candidates) != 1:
        raise ValueError("the isolated hinge color must be unique")
    hinge = candidates[0]
    anchors = [tuple(map(int, p)) for p in np.argwhere(a == hinge)]
    if not 1 <= len(anchors) <= 5 or len(colors) != len(anchors) + 1:
        raise ValueError("each of 1..5 stamps needs its own fill color")
    items, used_colors, occupied = [], set(), set()
    for y, x in anchors:
        neighbors = {int(a[y, (x + 1) % width])}
        if y + 1 < height:
            neighbors.add(int(a[y + 1, x]))
        neighbors -= {background, hinge}
        if len(neighbors) != 1:
            raise ValueError("a hinge must have one adjacent fill color")
        color = neighbors.pop()
        if color in used_colors:
            raise ValueError("fill colors cannot be shared between stamps")
        used_colors.add(color)
        h = w = 1
        while y + h < height and int(a[y + h, x]) == color:
            h += 1
        while w < width and int(a[y, (x + w) % width]) == color:
            w += 1
        short, long = sorted((h, w))
        if not (1 <= short <= 3 and 2 <= long <= 7 and h * w >= 3):
            raise ValueError("unsupported rectangle dimensions")
        if width < long + 2 or y < 1 or y + long >= height:
            raise ValueError("stamp needs vertical margins and a horizontal gap")
        cells = _cross_cells(y, x, h, w, width)
        if cells & _halo(occupied, width):
            raise ValueError("completed stamps must have a background moat")
        occupied |= cells
        items.append((y, x, h, w, color))
    if used_colors != set(colors) - {hinge}:
        raise ValueError("unattached fill color")
    if not np.array_equal(a, _render_input(height, width, background, hinge, items)):
        raise ValueError("every source must be a solid rectangle except its hinge")
    return background, hinge, items


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode roles, reconnect cyclic rectangles, and print two ribbon periods."""
    a = _grid(input_grid)
    background, hinge, items = _decode(a)
    return _render_output(*a.shape, background, hinge, items)


def _random_shape(rng, cap):
    roll = float(rng.random())
    if roll < 0.22:
        side = int(rng.integers(2, min(3, cap) + 1))
        return side, side
    long = int(rng.integers(3, cap + 1))
    short = 1 if roll < 0.53 else int(rng.integers(2, min(3, long - 1) + 1))
    return (short, long) if rng.random() < 0.5 else (long, short)


def _dimensions(rng, split, index, shift, cap):
    if split == "train" and index == 0:
        shapes = [(int(rng.integers(3, cap + 1)), 2),
                  (1, int(rng.integers(3, cap + 1)))]
        if rng.random() < 0.5:
            shapes.append(_random_shape(rng, cap))
        return shapes, "source"
    if split == "train" and index == 1:
        long = int(rng.integers(3, cap + 1))
        short = int(rng.integers(2, min(3, long - 1) + 1))
        other_long = int(rng.integers(3, cap + 1))
        other_short = int(rng.integers(2, min(3, other_long - 1) + 1))
        square = int(rng.integers(2, min(3, cap) + 1))
        return [(long, short), (other_short, other_long), (square, square)], "arm"
    if split == "test" and shift == 1:
        count = int(rng.integers(4, 6))
    else:
        count = int(rng.integers(1, 4 if shift == 1 else 6))
    shapes = [_random_shape(rng, cap) for _ in range(count)]
    if split == "test" and shift == 0:
        # cap is the held-out longer side, not a new operation.
        short = int(rng.integers(1, min(3, cap - 1) + 1))
        shapes[0] = (short, cap) if rng.random() < 0.5 else (cap, short)
    if count >= 2 and rng.random() < 0.30:
        # Same canonical shape in opposite source orientations.
        shapes[1] = shapes[0][::-1]
    return shapes, None


def _place(rng, shapes, height, width, seam):
    for _ in range(70):
        placed, forbidden = [], set()
        for index, (h, w) in enumerate(shapes):
            long = max(h, w)
            for _attempt in range(80):
                y = int(rng.integers(1, height - long))
                if index == 0 and seam == "source":
                    x = int(rng.integers(width - w + 1, width))
                elif index == 0 and seam == "arm":
                    x = int(rng.integers(width - long + 1, width - w + 1))
                else:
                    x = int(rng.integers(width))
                cells = _cross_cells(y, x, h, w, width)
                if not cells & forbidden:
                    placed.append((y, x, h, w))
                    forbidden |= _halo(cells, width)
                    break
            else:
                break
        if len(placed) == len(shapes):
            return placed
    return None


def _make_pair(rng, split, index, shift, cap):
    for attempt in range(50):
        shapes, seam = _dimensions(rng, split, index, shift, cap)
        longest = max(max(shape) for shape in shapes)
        width = int(rng.integers(max(7, longest + 2), 16))
        area = sum(2 * min(h, w) * max(h, w) - min(h, w) ** 2 for h, w in shapes)
        lower = max(longest + 3, min(25, math.ceil(area / (0.48 * width)) + 2))
        upper = min(30, max(lower + 3, sum(max(s) + 1 for s in shapes) + 1))
        height = int(rng.integers(lower, upper + 1))
        if attempt >= 10:
            # Extra space only changes geometry; no fallback transformation.
            height, width = 30, 15
        placed = _place(rng, shapes, height, width, seam)
        if placed is None:
            continue
        palette = [int(c) for c in rng.choice(10, len(shapes) + 2, replace=False)]
        background, hinge = palette[:2]
        items = [(*p, color) for p, color in zip(placed, palette[2:])]
        inp = _render_input(height, width, background, hinge, items)
        return {"input": inp,
                "output": _render_output(height, width, background, hinge, items)}
    raise RuntimeError("unable to place separated cylindrical stamps")


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _structure(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(2))
    train_cap = int(rng.integers(3, 7 if shift == 0 else 8))
    puzzle, structures = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        cap = train_cap + 1 if split == "test" and shift == 0 else train_cap
        for index in range(count):
            for _ in range(100):
                pair = _make_pair(rng, split, index, shift, cap)
                structure = _structure(pair["input"])
                if structure in structures:
                    continue
                if split == "train" and index == 1 and pair["input"].shape == puzzle["train"][0]["input"].shape:
                    continue
                structures.add(structure)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("unable to construct distinct layouts")
    validate_generated_puzzle(puzzle)
    return puzzle


def _assert_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    height, width = inp.shape
    background, hinge, items = _decode(inp)
    assert out.shape == (height, 2 * width), "only width doubles"
    assert np.array_equal(out[:, :width], out[:, width:]), "periods must be exact copies"
    # Check by modular coordinate inequalities, separately from the printer.
    yy, xx = np.indices(out.shape)
    expected = np.full(out.shape, background, dtype=int)
    multiplicity = np.zeros(out.shape, dtype=int)
    for y, x, h, w, color in items:
        short, long = sorted((h, w))
        dy, dx = yy - y, (xx - x) % width
        landscape = (dy >= 0) & (dy < short) & (dx < long)
        portrait = (dy >= 0) & (dy < long) & (dx < short)
        overlap = landscape & portrait
        union = landscape | portrait
        multiplicity += union
        expected[union] = color
        expected[overlap] = hinge
        assert np.count_nonzero(union) == 2 * (2 * short * long - short * short)
        assert np.count_nonzero(overlap) == 2 * short * short
        assert np.count_nonzero(inp == color) == h * w - 1
        assert np.count_nonzero(out == color) == 4 * short * (long - short)
    assert multiplicity.max() == 1, "different stamps cannot collide"
    assert np.array_equal(out, expected), "wrong geometry, seam, hinge, or color"
    assert np.array_equal(out, solve(inp)), "input-only solver disagrees"
    return background, hinge, items


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, structures, training, testing = set(), set(), [], []
    source_wrap = arm_only_wrap = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _, _, items = _assert_pair(pair)
            a = np.asarray(pair["input"])
            key, structure = _key(a), _structure(a)
            assert key not in seen and structure not in structures, "duplicate input structure"
            seen.add(key)
            structures.add(structure)
            (training if split == "train" else testing).append(items)
            if split == "train":
                for y, x, h, w, color in items:
                    source_wrap |= x + w > a.shape[1]
                    arm_only_wrap |= x + w <= a.shape[1] < x + max(h, w)
    all_train = [item for items in training for item in items]
    assert any(h > w for _, _, h, w, _ in all_train)
    assert any(h < w for _, _, h, w, _ in all_train)
    assert any(h == w for _, _, h, w, _ in all_train)
    assert any(min(h, w) == 1 for _, _, h, w, _ in all_train)
    assert any(h != w and min(h, w) > 1 for _, _, h, w, _ in all_train)
    assert source_wrap and arm_only_wrap, "training must teach both kinds of seam continuation"
    assert len({np.asarray(p["input"]).shape for p in puzzle["train"]}) >= 2
    largest = max(max(h, w) for _, _, h, w, _ in all_train)
    length_shift = all(max(max(h, w) for _, _, h, w, _ in items) > largest for items in testing)
    count_shift = all(len(items) > max(map(len, training)) for items in testing)
    assert length_shift or count_shift, "test must recombine the rule at a new scale or count"
    assert any(not np.array_equal(np.tile(p["input"], (1, 2)), p["output"])
               for p in puzzle["train"]), "orientation cannot be omitted"


def _clip_then_copy(inp):
    """Deliberately wrong ablation: forbid all horizontal continuation."""
    background, hinge, items = _decode(inp)
    h_canvas, width = inp.shape
    a = np.full(inp.shape, background, dtype=int)
    for y, x, h, w, color in items:
        short, long = sorted((h, w))
        for r in range(long):
            for c in range(long):
                if (r < short or c < short) and x + c < width:
                    a[y + r, x + c] = hinge if r < short and c < short else color
    return np.tile(a, (1, 2))


def self_test():
    """Bounded checks across 42 seeds, legal counts, mutations, and ablations."""
    shapes, structures, palettes, lengths, shorts, counts = set(), set(), set(), set(), set(), set()
    wraps, arm_wraps, squares, bars, repeated, modes = 0, 0, 0, 0, 0, set()
    fingerprints = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprint = tuple(_key(pair[k]) for s in ("train", "test") for pair in p[s] for k in ("input", "output"))
        fingerprints.add(fingerprint)
        assert fingerprint == tuple(_key(pair[k]) for s in ("train", "test")
                                    for pair in generate_puzzle(seed)[s] for k in ("input", "output"))
        training = [_decode(pair["input"])[2] for pair in p["train"]]
        test_items = _decode(p["test"][0]["input"])[2]
        modes.add("count" if len(test_items) > max(map(len, training)) else "length")
        assert any(not np.array_equal(_clip_then_copy(pair["input"]), pair["output"])
                   for pair in p["train"]), "periodic copying has no observable effect"
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                bg, hinge, items = _decode(a)
                shapes.add(a.shape)
                structures.add(_structure(a))
                palettes.add((bg, hinge))
                counts.add(len(items))
                dimensions = [(h, w) for _, _, h, w, _ in items]
                repeated += any(h != w and (w, h) in dimensions for h, w in dimensions)
                for _, x, h, w, _ in items:
                    lengths.add(max(h, w))
                    shorts.add(min(h, w))
                    wraps += x + w > a.shape[1]
                    arm_wraps += x + w <= a.shape[1] < x + max(h, w)
                    squares += h == w
                    bars += min(h, w) == 1
        # These are metamorphic checks of solving, not extra generated examples.
        a = p["train"][0]["input"]
        out = p["train"][0]["output"]
        for offset in (1, a.shape[1] - 1):
            assert np.array_equal(solve(np.roll(a, offset, axis=1)), np.roll(out, offset, axis=1))
        permutation = np.random.default_rng(42000 + seed).permutation(10)
        assert np.array_equal(solve(permutation[a]), permutation[out])
        bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in p[s]]
               for s in ("train", "test")}
        bad["train"][0]["output"][0, 0] = (int(out[0, 0]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted a corrupted output")
    assert len(fingerprints) == 42
    assert len(shapes) >= 45 and len(structures) >= 180 and len(palettes) >= 40
    assert {2, 3, 4, 5, 6, 7} <= lengths and shorts == {1, 2, 3}
    assert counts == {1, 2, 3, 4, 5} and modes == {"count", "length"}
    assert wraps >= 60 and arm_wraps >= 50 and squares >= 50 and bars >= 60 and repeated >= 10
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt}/{nq}")
    for seed in (-1, True, 1.0, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid seed accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, periodic and orientation ablations)")
