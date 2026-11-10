"""Complete endpoint spans, then stagger intact rows above a bottom anchor.

Only NumPy and the standard library are used. No files or generation state are
consulted by solve(). Separator 5 is a fixed family convention.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


SEPARATOR = 5
_ORDINARY_COLORS = tuple(c for c in range(10) if c != SEPARATOR)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if grid.ndim != 2 or any(not 1 <= d <= 30 for d in grid.shape):
        raise ValueError("A grid must be a nonempty rectangle with dimensions 1..30")
    if not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("Grid cells must be integers")
    if np.any((grid < 0) | (grid > 9)):
        raise ValueError("Grid colors must be 0..9")
    return grid


def _parse(value):
    """Recover all conventions from the sparse input, never from an output."""
    inp = _grid(value)
    h, w = inp.shape
    if h < 2 or w < 3 or h + w - 1 > 30:
        raise ValueError("Family requires h>=2, w>=3 and h+w-1<=30")
    if np.any(inp == SEPARATOR):
        raise ValueError("Separator 5 is reserved for completed midpoints")
    colors, counts = np.unique(inp, return_counts=True)
    maximum = counts.max()
    if np.count_nonzero(counts == maximum) != 1:
        raise ValueError("Background must be the unique most frequent color")
    background = int(colors[np.argmax(counts)])
    spans = []
    for r, row in enumerate(inp):
        positions = np.flatnonzero(row != background)
        if len(positions) == 0:
            continue
        if len(positions) != 2:
            raise ValueError("Each active row needs exactly two endpoints")
        left, right = map(int, positions)
        a, b = int(row[left]), int(row[right])
        if right - left < 2 or (right - left) % 2 or a == b:
            raise ValueError("Endpoints need distinct colors and an odd span >=3")
        spans.append((r, left, right, a, b))
    if not spans or len(spans) == h:
        raise ValueError("At least one active row and one blank row are required")
    return inp, background, spans


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill in source coordinates and translate each entire width-w row vector."""
    inp, background, spans = _parse(input_grid)
    h, w = inp.shape
    completed = inp.astype(np.int8, copy=True)
    for r, left, right, a, b in spans:
        middle = (left + right) // 2
        completed[r, left:middle] = a
        completed[r, middle] = SEPARATOR
        completed[r, middle + 1:right + 1] = b
    out = np.full((h, h + w - 1), background, dtype=np.int8)
    for r in range(h):
        offset = h - 1 - r
        out[r, offset:offset + w] = completed[r]
    return out


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _shape(rng, mode, training, anchor=0, different_height=None):
    if anchor:
        heights = list(range(6, 11)) if anchor == 1 else list(range(3, 11))
        if different_height in heights:
            heights.remove(different_height)
        h = int(_choice(rng, heights))
        upper = min(13 if mode == "width" else 21, 31 - h)
        w = int(rng.integers(9 if anchor == 1 else 7, upper + 1))
        return h, w
    if mode == "height":
        h = int(rng.integers(2, 11) if training else rng.integers(11, 29))
        w = int(rng.integers(3, min(23 if training else 15, 31 - h) + 1))
    elif mode == "width":
        w = int(rng.integers(3, 14) if training else rng.integers(14, 30))
        h = int(rng.integers(2, min(24, 31 - w) + 1))
    else:  # Count extrapolation, with overlapping shape ranges.
        h = int(rng.integers(2 if training else 5, 29))
        w = int(rng.integers(3, 32 - h))
    # Sample the legal output-width boundary deliberately, without cropping.
    if rng.random() < 0.18:
        if mode == "width":
            h = 31 - w
        else:
            w = 31 - h
    return h, w


def _lanes(rng, h, count):
    mode = int(rng.integers(4))
    if mode == 0:
        first = int(rng.integers(h - count + 1))
        rows = list(range(first, first + count))
    elif mode == 1 and count >= 2:
        rows = [0, h - 1] + [int(r) for r in rng.choice(
            np.arange(1, h - 1), count - 2, replace=False)]
    else:
        rows = [int(r) for r in rng.choice(h, count, replace=False)]
    # Keep displacement observable even in a one-line generated example.
    if count == 1 and rows == [h - 1]:
        rows = [int(rng.integers(h - 1))]
    return sorted(rows)


def _span(rng, w, length=None, asymmetric=False):
    if length is None:
        lengths = list(range(3, w + 1, 2))
        draw = rng.random()
        if draw < 0.18:
            length = 3
        elif draw < 0.39:
            length = lengths[-1]
        else:
            length = int(_choice(rng, lengths))
    starts = list(range(w - length + 1))
    if asymmetric:
        starts = [s for s in starts if s != w - length - s]
    left = int(_choice(rng, starts))
    return left, left + length - 1


def _construct(rng, shape, background, mode, training, anchor=0):
    """Construct targets in destination coordinates, separately from solve()."""
    h, w = shape
    if anchor == 1:
        rows = [0, 1, int(rng.integers(3, h - 1))]
        lengths = [3] + [int(n) for n in rng.choice(
            np.arange(5, w - 1, 2), 2, replace=False)]
        rng.shuffle(lengths)
        geometry = [_span(rng, w, n, asymmetric=True) for n in lengths]
    elif anchor == 2:
        rows = [int(rng.integers(h - 1)), h - 1]
        geometry = [_span(rng, w, 3), _span(rng, w, int(_choice(
            rng, list(range(5, w + 1, 2)))))]
        if rng.random() < 0.5:
            geometry.reverse()
    else:
        cap = min(h - 1, 3 if mode == "count" and training else 9)
        count = int(rng.integers(4 if mode == "count" and not training else 1, cap + 1))
        rows = _lanes(rng, h, count)
        layout = int(rng.integers(4))
        if layout == 0:  # repeated row geometry, optionally repeated colors
            geometry = [_span(rng, w)] * count
        elif layout == 1:  # concentric spans with a shared source midpoint
            center = int(rng.integers(1, w - 1))
            maximum = min(center, w - 1 - center)
            geometry = []
            for _ in rows:
                half = int(rng.integers(1, maximum + 1))
                geometry.append((center - half, center + half))
        else:
            geometry = [_span(rng, w) for _ in rows]
    legal = [c for c in _ORDINARY_COLORS if c != background]
    assignments = []
    inp = np.full((h, w), background, dtype=np.int8)
    out = np.full((h, h + w - 1), background, dtype=np.int8)
    for index, (r, (left, right)) in enumerate(zip(rows, geometry)):
        if assignments and (anchor == 2 or rng.random() < 0.50):
            a, b = _choice(rng, assignments)
            if rng.random() < 0.40:
                a, b = b, a
        else:
            a, b = map(int, rng.choice(legal, 2, replace=False))
        assignments.append((a, b))
        inp[r, left], inp[r, right] = a, b
        middle = (left + right) // 2
        offset = h - 1 - r
        for source_c in range(left, right + 1):
            out[r, source_c + offset] = (
                a if source_c < middle else SEPARATOR if source_c == middle else b)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = _choice(rng, ("height", "count", "width"))
    fixed_background = bool(rng.random() < 0.30)
    first_background = int(_choice(rng, _ORDINARY_COLORS))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            training = split == "train"
            anchor = index + 1 if training and index < 2 else 0
            for _ in range(64):
                background = first_background if fixed_background or (training and index == 0) else int(
                    _choice(rng, tuple(c for c in _ORDINARY_COLORS
                                       if not (training and index == 1 and c == first_background))))
                shape = _shape(rng, mode, training, anchor,
                               puzzle["train"][0]["input"].shape[0] if anchor == 2 else None)
                pair = _construct(rng, shape, background, mode, training, anchor)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct-pair construction exhausted its bounded retries")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _check_pair(pair):
    _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Bad pair schema")
    inp, bg, spans = _parse(pair["input"])
    out = _grid(pair["output"])
    h, w = inp.shape
    _check(out.shape == (h, h + w - 1), "Wrong expanded canvas size")
    by_row = {span[0]: span for span in spans}
    for r in range(h):
        offset = h - 1 - r
        expected_row = np.full(w, bg, dtype=np.int8)
        if r in by_row:
            _, left, right, a, b = by_row[r]
            middle = (left + right) // 2
            half = (right - left) // 2
            _check(int(out[r, left + offset]) == a and int(out[r, right + offset]) == b,
                   "Endpoint placement is incorrect")
            _check(int(out[r, middle + offset]) == SEPARATOR, "Wrong translated midpoint")
            _check(np.count_nonzero(out[r] == a) == half and
                   np.count_nonzero(out[r] == b) == half, "Unequal colored halves")
            for c in range(w):
                if left <= c <= right:
                    expected_row[c] = a if c < middle else SEPARATOR if c == middle else b
        _check(np.array_equal(out[r, offset:offset + w], expected_row),
               "A completed row vector or a source margin changed")
        _check(np.all(out[r, :offset] == bg) and np.all(out[r, offset + w:] == bg),
               "New padding is not background")
    _check(np.count_nonzero(out == SEPARATOR) == len(spans), "Not one separator per active row")
    _check(np.count_nonzero(out != bg) == sum(b - a + 1 for _, a, b, _, _ in spans),
           "Filled span mass was not preserved by translation")
    sparse_shear = np.full(out.shape, bg, dtype=np.int8)
    no_shear = np.full(out.shape, bg, dtype=np.int8)
    for r in range(h):
        offset = h - 1 - r
        sparse_shear[r, offset:offset + w] = inp[r]
        no_shear[r, :w] = out[r, offset:offset + w]
    _check(not np.array_equal(sparse_shear, out), "Omitting A has no effect")
    if any(r < h - 1 for r, *_ in spans):
        _check(not np.array_equal(no_shear, out), "Omitting B has no effect")
    _check(np.array_equal(solve(inp), out), "Input-only solver disagrees")
    return inp, bg, spans


def validate_generated_puzzle(puzzle) -> None:
    """Check visible-grid invariants, informative demos, and the intended shift."""
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Bad puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal example counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, bg, spans = _check_pair(pair)
            key = _key(inp)
            _check(key not in seen, "Duplicate input")
            seen.add(key)
            infos[split].append((inp, bg, spans))
    train = infos["train"]
    _check(len({inp.shape[0] for inp, _, _ in train}) >= 2, "Need two demonstrated heights")
    _check(any(spans[-1][0] == inp.shape[0] - 1 for inp, _, spans in train),
           "Training must show a bottom row with zero shift")
    _check(any(spans[0][0] == 0 and spans[-1][0] < inp.shape[0] - 1 and
               len(spans) >= 3 and len({q[0] - p[0] for p, q in zip(spans, spans[1:])}) > 1
               for inp, _, spans in train), "Need unequal gaps and an inactive bottom")
    lengths = {right - left + 1 for _, _, spans in train for _, left, right, _, _ in spans}
    _check(3 in lengths and len(lengths) >= 3, "Training must show three span scales")
    _check(any(right - left >= 4 and left != inp.shape[1] - 1 - right
               for inp, _, spans in train for _, left, right, _, _ in spans),
           "Training needs an extending span away from the canvas midpoint")
    # A shift is measured from grids, not from an undisclosed sampler mode.
    max_height = max(inp.shape[0] for inp, _, _ in train)
    max_width = max(inp.shape[1] for inp, _, _ in train)
    max_count = max(len(spans) for _, _, spans in train)
    for inp, _, spans in infos["test"]:
        _check(inp.shape[0] > max_height or inp.shape[1] > max_width or len(spans) > max_count,
               "Test lacks a height, width or active-count extrapolation")
        _check(any(r < inp.shape[0] - 1 for r, *_ in spans), "Test never exercises displacement")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role])
                digest.update(bytes(grid.shape))
                digest.update(np.asarray(grid, dtype=np.int8).tobytes())
    return digest.hexdigest()


def _fixture(rows):
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


def self_test() -> dict:
    """Bounded: 42 default puzzles, 30 count probes, small direct fixtures; no I/O."""
    fixture = _fixture(("0300020", "0000000", "4000007", "0000000", "0000000", "0010900"))
    expected = _fixture(("000000335220", "000000000000", "000444577700",
                         "000000000000", "000000000000", "001590000000"))
    _check(np.array_equal(solve(fixture), expected), "Hand-calculated fixture failed")
    # A blank bottom row increments all existing offsets, without trimming.
    extended = np.vstack((fixture, np.zeros((1, fixture.shape[1]), dtype=np.int8)))
    shifted = solve(extended)
    _check(np.array_equal(shifted[:-1, 1:], expected) and np.all(shifted[:, 0] == 0)
           and np.all(shifted[-1] == 0), "Blank rows must count in the shear")
    for h, w in ((2, 3), (2, 29), (28, 3)):
        inp = np.full((h, w), 8, dtype=np.int8)
        inp[0, 0], inp[0, w - 1] = 0, 2
        _check_pair({"input": inp, "output": solve(inp)})
    # Recoloring all ordinary roles commutes with the transformation; 5 stays 5.
    mapping = np.array([7, 8, 9, 0, 1, 5, 2, 3, 4, 6], dtype=np.int8)
    _check(np.array_equal(solve(mapping[fixture]), mapping[expected]), "Palette-role check failed")
    backgrounds, endpoint_colors, shapes, layouts, signatures = set(), set(), set(), set(), set()
    heights, widths, counts, lengths = set(), set(), set(), set()
    observations = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        puzzle_shapes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, bg, spans = _parse(pair["input"])
                h, w = inp.shape
                rows = [s[0] for s in spans]
                backgrounds.add(bg)
                endpoint_colors.update(c for s in spans for c in s[3:])
                shapes.add((h, w))
                puzzle_shapes.add((h, w))
                layouts.add(((h, w), tuple(s[:3] for s in spans)))
                heights.add(h)
                widths.add(w)
                counts.add(len(spans))
                lengths.update(b - a + 1 for _, a, b, _, _ in spans)
                observations["pairs"] += 1
                observations["boundary_width_30"] += h + w - 1 == 30
                observations["single_active"] += len(spans) == 1
                observations["four_or_more_active"] += len(spans) >= 4
                observations["adjacent_active"] += any(b - a == 1 for a, b in zip(rows, rows[1:]))
                observations["full_span"] += any(a == 0 and b == w - 1 for _, a, b, _, _ in spans)
                observations["internal_span"] += any(a > 0 or b < w - 1 for _, a, b, _, _ in spans)
                observations["repeated_colors"] += len({s[3:] for s in spans}) < len(spans)
                observations["nonblack_background"] += bg != 0
                observations["black_endpoint"] += any(0 in s[3:] for s in spans)
                if split == "test":
                    observations["height_shift"] += h > max(p["input"].shape[0] for p in puzzle["train"])
                    observations["width_shift"] += w > max(p["input"].shape[1] for p in puzzle["train"])
                    observations["count_shift"] += len(spans) > max(len(_parse(p["input"])[2])
                                                                 for p in puzzle["train"])
        _check(len(puzzle_shapes) >= 2, "Only colors changed within a puzzle")
    _check(len(signatures) == 42, "Puzzle seeds are not diverse")
    _check(backgrounds == set(_ORDINARY_COLORS) and endpoint_colors == set(_ORDINARY_COLORS),
           "Ordinary color roles lack coverage")
    _check(len(shapes) >= 70 and len(layouts) >= 150, "Structural variety is too low")
    _check(len(heights) >= 15 and len(widths) >= 15 and len(counts) >= 6 and len(lengths) >= 8,
           "Geometry/count/scale diversity is too low")
    _check(all(value > 0 for value in observations.values()) and
           all(observations[k] > 0 for k in ("height_shift", "width_shift", "count_shift",
                                            "single_active", "four_or_more_active", "full_span")),
           "A required structural feature was not exercised")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(104729 + 10 * a + b, a, b)
            _check((len(p["train"]), len(p["test"])) == (a, b), "Explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    _check(_digest(generate_puzzle(314159, 10, 4)) == _digest(generate_puzzle(314159, 10, 4)),
           "Seed reproducibility failed")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "layouts": len(layouts), "heights": sorted(heights), "widths": sorted(widths),
            "active_counts": sorted(counts), "span_lengths": sorted(lengths),
            "backgrounds": sorted(backgrounds), "endpoint_colors": sorted(endpoint_colors),
            "observations": dict(observations), "legal_count_combinations": 30}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
