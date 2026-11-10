"""Scan-weighted stencil ballots. All solving conventions are input-visible.

Only standard-library modules and NumPy are used. Sampling metadata is never
consulted by solve(); generation builds labels from its planned ballots.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import json

import numpy as np


SIDES = ("top", "right", "bottom", "left")
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRECTIONS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


@lru_cache(maxsize=8192)
def _rotations(shape):
    variants = set()
    shape = _normalize(shape)
    for _ in range(4):
        variants.add(shape)
        shape = _normalize(tuple((c, -r) for r, c in shape))
    return tuple(sorted(variants))


def _mirror(shape):
    return _normalize(tuple((r, -c) for r, c in shape))


@lru_cache(maxsize=8192)
def _variants(shape):
    return tuple(sorted(set(_rotations(shape)) | set(_rotations(_mirror(shape)))))


def _shape(cells, equivalence="free"):
    raw = _normalize(cells)
    if equivalence == "area":
        return len(raw)
    if equivalence == "box":
        return tuple(sorted(_extent(raw)))
    if equivalence == "translation":
        return raw
    if equivalence == "rotation":
        return min(_rotations(raw))
    return min(_variants(raw))


def _extent(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _view(grid, side):
    """Rows in this view measure inward distance from the boundary bar."""
    if side == "top":
        return grid.copy()
    if side == "bottom":
        return grid[::-1].copy()
    if side == "left":
        return grid.T.copy()
    return grid.T[::-1].copy()


def _unview(grid, side):
    if side in ("top", "bottom"):
        return _view(grid, side)
    if side == "left":
        return grid.T.copy()
    return grid[::-1].T.copy()


def _bar(grid, color):
    points = np.argwhere(grid == color)
    if not len(points):
        return None
    rows, cols = points.T
    candidates = []
    for side, fixed, edge, along in (
        ("top", rows, 0, cols), ("bottom", rows, grid.shape[0] - 1, cols),
        ("left", cols, 0, rows), ("right", cols, grid.shape[1] - 1, rows),
    ):
        start = int(along.min())
        if np.all(fixed == edge) and np.array_equal(
                np.sort(along), np.arange(start, start + len(points))):
            candidates.append((side, start, len(points)))
    return candidates[0] if len(candidates) == 1 else None


def _decode(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= grid.size / 2:
        raise ValueError("Background must occupy more than half the canvas")
    bars = [(int(c), _bar(grid, int(c))) for c in np.unique(grid) if c != bg]
    bars = [(c, bar) for c, bar in bars if bar is not None]
    if len(bars) != 1:
        raise ValueError("Expected one uniquely colored, unambiguous edge bar")
    marker, (side, start, length) = bars[0]
    g = _view(grid, side)
    rectangles = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, marker):
            continue
        points = np.argwhere(g == color)
        y, x = points.min(axis=0)
        bottom, right = points.max(axis=0)
        box = g[y:bottom + 1, x:right + 1]
        if min(box.shape) < 5 or set(map(int, np.unique(box))) != {bg, color}:
            continue
        if (np.all(box[0] == color) and np.all(box[-1] == color)
                and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)
                and len(_components(box == color)) == 1):
            rectangles.append((color, int(y), int(x), int(bottom), int(right)))
    if len(rectangles) != 1:
        raise ValueError("Expected one connected rectangular carrier with intact border")
    carrier, y, x, bottom, right = rectangles[0]
    box = g[y:bottom + 1, x:right + 1].copy()
    cavities = _components(box == bg)
    exterior = g.copy()
    exterior[y:bottom + 1, x:right + 1] = bg
    exterior[0, start:start + length] = bg
    objects = []
    for color in np.unique(exterior):
        if color in (bg, marker, carrier):
            continue
        for cells in _components(exterior == color):
            weight = sum(r % 6 == 4 and not start <= c < start + length for r, c in cells)
            objects.append(dict(color=int(color), cells=cells, shape=_shape(cells), weight=int(weight)))
    if not cavities or not objects:
        raise ValueError("The stencil and electorate must be nonempty")
    if len(_components(exterior != bg)) != len(objects):
        raise ValueError("Exterior objects must be separated even when their colors differ")
    if set(o["shape"] for o in objects) != set(_shape(c) for c in cavities):
        raise ValueError("Each ballot shape must have a cavity, and vice versa")
    return dict(grid=g, background=bg, marker=marker, carrier=carrier,
                side=side, start=start, length=length, bbox=(y, x, bottom, right),
                box=box, cavities=cavities, objects=objects)


def _ballots(data, equivalence="free", sampling="scan"):
    votes = defaultdict(Counter)
    start, end = data["start"], data["start"] + data["length"]
    for obj in data["objects"]:
        cells = obj["cells"]
        if sampling == "components":
            weight = 1
        elif sampling == "inclusive":
            weight = sum(r % 6 == 4 for r, c in cells)
        elif sampling == "even":
            weight = sum(r % 2 == 0 and not start <= c < end for r, c in cells)
        elif sampling == "touched":
            weight = int(obj["weight"] > 0)
        elif sampling == "origin":
            weight = int(min(r for r, c in cells) % 6 == 4)
        else:
            weight = obj["weight"]
        votes[_shape(cells, equivalence)][obj["color"]] += weight
    winners = {}
    for cells in data["cavities"]:
        key = _shape(cells, equivalence)
        by_color = votes[key]
        high = max(by_color.values(), default=0)
        best = [c for c, n in by_color.items() if n == high]
        if high <= 0 or len(best) != 1:
            raise ValueError("Every cavity class needs a unique positive weighted plurality")
        winners[key] = best[0]
    return votes, winners


def _trace(box, data):
    y, x, bottom, right = data["bbox"]
    left = max(x, data["start"])
    end = min(right + 1, data["start"] + data["length"])
    if left >= end:
        return box
    for r in range(y, bottom + 1):
        ink = data["background"] if r % 2 else data["carrier"] if r % 6 == 4 else data["marker"]
        box[r - y, left - x:end - x] = ink
    return box


def _prediction(data, equivalence="free", sampling="scan"):
    votes, winners = _ballots(data, equivalence, sampling)
    result = data["box"].copy()
    for cells in data["cavities"]:
        color = winners[_shape(cells, equivalence)]
        for r, c in cells:
            result[r, c] = color
    _trace(result, data)
    return _unview(result, data["side"]), votes, winners


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all geometry, ballots, colors and phases from this input alone."""
    return _prediction(_decode(input_grid))[0]


def _choice(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _hole_free(shape):
    h, w = _extent(shape)
    mask = np.ones((h + 2, w + 2), dtype=bool)
    for r, c in shape:
        mask[r + 1, c + 1] = False
    return len(_components(mask)) == 1


def _pool(rng):
    shapes = {((0, 0),)}
    for n in range(2, 8):
        shapes.add(_shape(tuple((0, c) for c in range(n))))
    for h in range(2, 6):
        for w in range(2, 6):
            shapes.add(_shape(tuple({(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)})))
            if h * w <= 10:
                shapes.add(_shape(tuple((r, c) for r in range(h) for c in range(w))))
            if h <= 3 and w >= 3:
                shapes.add(_shape(tuple({(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)}
                                        | {(h - 1, c) for c in range(w)})))
    for _ in range(140):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(3, 10))):
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in DIRECTIONS} - cells)
            cells.add(_choice(rng, frontier))
        shape = _shape(tuple(cells))
        if max(_extent(shape)) <= 5 and _hole_free(shape):
            shapes.add(shape)
    return tuple(sorted(shapes))


def _block(available, cells):
    for r, c in cells:
        available[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = False


def _locations(available, variant):
    h, w = _extent(variant)
    rows, cols = available.shape[0] - h + 1, available.shape[1] - w + 1
    if rows <= 0 or cols <= 0:
        return np.zeros((0, 0), dtype=bool)
    valid = np.ones((rows, cols), dtype=bool)
    for r, c in variant:
        valid &= available[r:r + rows, c:c + cols]
    return valid


def _place(available, variants, rng, sample=None, weight=None,
           inclusive=None, inclusive_weight=None, zone=None, start=None, length=None):
    for index in rng.permutation(len(variants)):
        variant = variants[int(index)]
        valid = _locations(available, variant)
        if not valid.size:
            continue
        rows, cols = valid.shape
        if sample is not None:
            weights = np.zeros(valid.shape, dtype=int)
            for r, c in variant:
                weights += sample[r:r + rows, c:c + cols]
            valid &= weights == weight
        if inclusive is not None:
            weights = np.zeros(valid.shape, dtype=int)
            for r, c in variant:
                weights += inclusive[r:r + rows, c:c + cols]
            valid &= weights == inclusive_weight
        if zone:
            lateral = np.arange(cols)[None, :]
            for r, c in variant:
                inside = (lateral + c >= start) & (lateral + c < start + length)
                valid &= inside if zone == "inside" else ~inside
        positions = np.argwhere(valid)
        if len(positions):
            y, x = map(int, positions[int(rng.integers(len(positions)))])
            cells = tuple((y + r, x + c) for r, c in variant)
            _block(available, cells)
            return cells
    return None


def _positive_weights(shape):
    return sorted({n for v in _variants(shape) for phase in range(6)
                   if (n := sum((r + phase) % 6 == 4 for r, c in v)) > 0})


def _plan(rng, shapes, palette, teaching, forbidden=None):
    """Requested specimen weights are checked again on the returned input."""
    specs, winners = [], {}
    for i, shape in enumerate(shapes):
        colors = list(map(int, rng.permutation(palette)))
        if i == 0 and forbidden is not None and colors[0] == forbidden:
            colors[0], colors[1] = colors[1], colors[0]
        if i == 1 and teaching:
            colors = [c for c in colors if c != winners[shapes[0]]] + [winners[shapes[0]]]
        variants = _variants(shape)
        if teaching == 1 and i == 0:
            # Winning objects are mirror-images of the first cavity, while
            # losing objects have its handedness. Both have weights 1 and 2.
            ballots = ((colors[0], [2, 2], _rotations(_mirror(shape))),
                       (colors[1], [1, 1, 1], _rotations(shape)))
        elif teaching == 2 and i == 0:
            ballots = ((colors[0], [2, 1], variants),
                       (colors[1], [1, 1], variants), (colors[2], [1, 1], variants))
        elif teaching and i == 1:
            ballots = ((colors[0], [2], variants), (colors[1], [1], variants))
        else:
            options = _positive_weights(shape)
            # A range of one-color, close, and three-color ballots. Weights
            # differ independently between individual objects of a class.
            for _ in range(30):
                counts = _choice(rng, ((1,), (2,), (2, 1), (3, 2), (2, 1, 1)))
                ballots = tuple((color, [_choice(rng, options) for _ in range(n)], variants)
                                for color, n in zip(colors, counts))
                totals = [sum(ws) for color, ws, variants in ballots]
                if totals.count(max(totals)) == 1:
                    break
            else:
                ballots = ((colors[0], [options[-1]], variants),)
        totals = {color: sum(ws) for color, ws, variants in ballots}
        winners[shape] = max(totals, key=totals.get)
        for color, weights, variants in ballots:
            for weight in weights:
                specs.append(dict(shape=shape, color=color, weight=weight, variants=variants,
                                  zone="outside" if teaching else None, inclusive=None))
        if teaching == 1 and i == 0:
            # These two in-corridor objects would reverse the winner if
            # special rays were sampled across the strip as well.
            for _ in range(2):
                specs.append(dict(shape=shape, color=colors[1], weight=0, variants=_rotations(shape),
                                  zone="inside", inclusive=2))
        elif not teaching and rng.random() < .45:
            specs.append(dict(shape=shape, color=_choice(rng, colors), weight=0,
                              variants=variants, zone=None, inclusive=None))
    return specs, winners


def _make_pair(rng, shapes, side, teaching=0, forbidden=None):
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, marker = palette[:3]
    specs, winners = _plan(rng, shapes, palette[3:], teaching, forbidden)
    repeats = [s for s in shapes]
    repeats += [_choice(rng, shapes) for _ in range(int(rng.integers(1, 5)))]
    if teaching:
        repeats.append(shapes[0])
    for attempt in range(100):
        if teaching:
            depth = int(rng.integers(27, 31)) if attempt < 8 else 30
            span = int(rng.integers(25, 31)) if attempt < 8 else 30
            sh, sw = int(rng.integers(13, 17)), int(rng.integers(11, 18))
            length = int(rng.integers(4, 8))
        else:
            depth = int(rng.integers(20, 31)) if attempt < 12 else 30
            span = int(rng.integers(20, 31)) if attempt < 12 else 30
            mode = int(rng.integers(3))
            if mode == 0:
                sh = sw = int(rng.integers(10, 17))
            elif mode == 1:
                sh, sw = int(rng.integers(7, 12)), int(rng.integers(14, 22))
            else:
                sh, sw = int(rng.integers(14, 22)), int(rng.integers(7, 12))
            length = int(rng.integers(1, min(9, span // 2) + 1))
        if sh >= depth - 1 or sw >= span - 1:
            continue
        start = int(rng.integers(1 if length == 1 else 0,
                                 span - length if length == 1 else span - length + 1))
        y = int(rng.integers(1, depth - sh + 1))
        # The printed strip passes through the stencil, at variable offsets.
        lo, hi = max(0, start - sw + 1), min(span - sw, start + length - 1)
        x = int(rng.integers(lo, hi + 1))
        available = np.ones((sh, sw), dtype=bool)
        available[[0, -1], :] = False
        available[:, [0, -1]] = False
        placements, failed = [], False
        for i, shape in enumerate(repeats):
            variants = (shape,) if teaching and shape == shapes[0] and i < len(shapes) else _variants(shape)
            # Every class has at least one visible fill outside the trace.
            cells = _place(available, variants, rng, zone="outside" if i < len(shapes) else None,
                           start=start - x, length=length)
            if cells is None:
                if i < len(shapes):
                    failed = True
                    break
                continue
            placements.append((shape, cells))
        if failed or len(placements) < len(shapes):
            continue
        box = np.full((sh, sw), carrier, dtype=np.int8)
        output = box.copy()
        for shape, cells in placements:
            for r, c in cells:
                box[r, c], output[r, c] = bg, winners[shape]
        if len(_components(box == carrier)) != 1:
            continue
        available = np.ones((depth, span), dtype=bool)
        available[0] = False
        available[max(0, y - 1):y + sh + 1, max(0, x - 1):x + sw + 1] = False
        sample = np.broadcast_to(np.arange(depth)[:, None] % 6 == 4, (depth, span)).copy()
        inclusive = sample.copy()
        sample[:, start:start + length] = False
        external = []
        ordered = [specs[int(i)] for i in rng.permutation(len(specs))]
        # Place constrained corridor voters first, then larger specimens.
        ordered.sort(key=lambda s: (s["zone"] == "inside", max(_extent(s["shape"])), len(s["shape"])), reverse=True)
        for spec in ordered:
            cells = _place(available, spec["variants"], rng, sample, spec["weight"],
                           inclusive if spec["inclusive"] is not None else None, spec["inclusive"],
                           spec["zone"], start, length)
            if cells is None:
                failed = True
                break
            external.append((spec["color"], cells))
        if failed:
            continue
        g = np.full((depth, span), bg, dtype=np.int8)
        g[y:y + sh, x:x + sw] = box
        g[0, start:start + length] = marker
        for color, cells in external:
            for r, c in cells:
                g[r, c] = color
        if np.count_nonzero(g == bg) <= g.size / 2:
            continue
        data = dict(bbox=(y, x, y + sh - 1, x + sw - 1), start=start, length=length,
                    background=bg, marker=marker, carrier=carrier)
        _trace(output, data)
        inp, out = _unview(g, side), _unview(output, side)
        try:
            decoded = _decode(inp)
            _ballots(decoded)
        except ValueError:
            continue
        return dict(input=inp, output=out)
    return None


def _fails(pair, equivalence="free", sampling="scan"):
    try:
        return not np.array_equal(_prediction(_decode(pair["input"]), equivalence, sampling)[0], pair["output"])
    except ValueError:
        return True


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, n, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
                or not low <= n <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    pool = _pool(rng)
    anchors = [s for s in pool if 4 <= len(s) <= 6 and max(_extent(s)) <= 3
               and _mirror(s) not in _rotations(s) and {1, 2} <= set(_positive_weights(s))
               and any(t != s and len(t) == len(s) and sorted(_extent(t)) == sorted(_extent(s))
                       and {1, 2} <= set(_positive_weights(t)) for t in pool)]
    anchor = _choice(rng, anchors)
    companion = _choice(rng, [t for t in pool if t != anchor and len(t) == len(anchor)
                             and sorted(_extent(t)) == sorted(_extent(anchor))
                             and {1, 2} <= set(_positive_weights(t))])
    base = int(rng.integers(4))
    # The first two demonstrate orientation change. Other examples are free;
    # test novelty comes from geometric classes, not a hidden side convention.
    puzzle, seen, known = {"train": [], "test": []}, set(), set()
    first_winner = None
    for split, n in (("train", nt), ("test", ns)):
        for index in range(n):
            teaching = index + 1 if split == "train" and index < 2 else 0
            side = SIDES[(base + index) % 4] if teaching else _choice(rng, SIDES)
            if teaching:
                shapes = [anchor, companion]
            else:
                k = int(rng.integers(1, 5))
                candidates = [s for s in pool if len(s) <= (6 if k == 4 else 10)]
                if split == "test":
                    novel = [s for s in candidates if s not in known]
                    first = _choice(rng, novel)
                    selected = rng.choice(len(candidates), size=k - 1, replace=False)
                    shapes = [first] + [candidates[int(i)] for i in selected if candidates[int(i)] != first]
                else:
                    shapes = [candidates[int(i)] for i in rng.choice(len(candidates), size=k, replace=False)]
            pair = None
            for _ in range(16):
                proposal = _make_pair(rng, shapes, side, teaching, first_winner if teaching == 2 else None)
                if proposal is None or _key(proposal["input"]) in seen:
                    continue
                if teaching == 1 and not all(_fails(proposal, eq, mode) for eq, mode in (
                        ("free", "components"), ("free", "inclusive"), ("free", "touched"),
                        ("area", "scan"), ("box", "scan"), ("rotation", "scan"))):
                    continue
                pair = proposal
                break
            if pair is None:
                raise RuntimeError("Bounded packing could not retain the requested evidence")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            data = _decode(pair["input"])
            if split == "train":
                known.update(_shape(c) for c in data["cavities"])
                if index == 0:
                    first_winner = _ballots(data)[1][anchor]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, history = set(), [], defaultdict(set)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "Repeated input"
            seen.add(_key(a))
            data = _decode(a)
            predicted, votes, winners = _prediction(data)
            assert np.array_equal(b, predicted), "Wrong weighted fill, trace, or crop"
            # Verify pixels directly, including the trace's priority over holes.
            out = _view(b, data["side"])
            y, x, bottom, right = data["bbox"]
            assert out.shape == data["box"].shape
            hole_colors = {}
            for cells in data["cavities"]:
                for r, c in cells:
                    hole_colors[r, c] = winners[_shape(cells)]
            for r in range(out.shape[0]):
                for c in range(out.shape[1]):
                    inside = data["start"] <= x + c < data["start"] + data["length"]
                    if inside:
                        d = y + r
                        expected = data["background"] if d % 2 else data["carrier"] if d % 6 == 4 else data["marker"]
                    else:
                        expected = hole_colors.get((r, c), data["carrier"])
                    assert out[r, c] == expected
            if split == "train":
                training.append((pair, data, votes))
                for shape, color in winners.items():
                    history[shape].add(color)
    for eq, sampling in (("area", "scan"), ("box", "scan"), ("rotation", "scan"),
                         ("translation", "scan"), ("free", "components"),
                         ("free", "touched"), ("free", "inclusive")):
        assert any(_fails(p, eq, sampling) for p, d, v in training), f"Training admits {eq}/{sampling} shortcut"
    assert any(len(colors) > 1 for colors in history.values()), "No changing local shape ballot"
    assert any(max(v.values()) * 2 <= sum(v.values()) and max(v.values()) > 0
               for p, d, votes in training for v in votes.values()), "No non-majority plurality"
    assert any(len(d["box"]) >= 13 for p, d, v in training), "No repeated sweep in training"


def self_test():
    """Bounded 42-seed checks, counterfactual witnesses and symmetry tests."""
    signatures, shapes, crops, palettes, vocabulary = set(), set(), set(), set(), set()
    class_counts, hole_counts, weights, sides, bar_lengths = set(), set(), set(), set(), set()
    repeated, zero, shared, boundary, crossing, multiray, new_tests = 0, 0, 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        known = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                d = _decode(a)
                votes, winners = _ballots(d)
                classes = set(winners)
                vocabulary.update(classes)
                class_counts.add(len(classes))
                hole_counts.add(len(d["cavities"]))
                shapes.add(a.shape)
                crops.add(b.shape)
                palettes.add((d["background"], d["carrier"], d["marker"]))
                weights.update(o["weight"] for o in d["objects"])
                sides.add(d["side"])
                bar_lengths.add(d["length"])
                repeated += len(d["cavities"]) > len(classes)
                zero += any(o["weight"] == 0 for o in d["objects"])
                shared += len(set(winners.values())) < len(winners)
                y, x, bottom, right = d["bbox"]
                boundary += x == 0 or right == d["grid"].shape[1] - 1 or bottom == len(d["grid"]) - 1
                crossing += any(any(d["start"] <= c + x < d["start"] + d["length"] for r, c in cells)
                                for cells in d["cavities"])
                multiray += any(len({r for r, c in o["cells"] if r % 6 == 4}) > 1 for o in d["objects"])
                if split == "train":
                    known.update(classes)
                else:
                    new_tests += bool(classes - known)
                # Rotation and arbitrary relabeling preserve the complete rule.
                if seed < 4:
                    for turns in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(a, turns)), np.rot90(b, turns))
                    perm = np.random.default_rng(seed + 500).permutation(10)
                    assert np.array_equal(solve(perm[a]), perm[b])
    assert len(signatures) == 42 and len(shapes) >= 25 and len(crops) >= 35
    assert len(palettes) >= 80 and len(vocabulary) >= 50
    assert class_counts == {1, 2, 3, 4} and len(hole_counts) >= 6
    assert sides == set(SIDES) and {1, 2, 4, 7} <= bar_lengths
    assert {0, 1, 2, 3} <= weights and max(weights) >= 4
    assert repeated >= 100 and zero >= 42 and shared > 0 and boundary > 0 and crossing > 40
    assert multiray > 0 and new_tests == 42
    # A family member outside the sampler's intersection convention: the
    # strip misses an edge-touching carrier. One seven-cell bar contributes
    # twice on separate rays, defeating any sampled-fragment classification.
    edge = np.zeros((20, 18), dtype=np.int8)
    edge[0, 5] = 9
    edge[8:17, 9:18] = 8
    edge[9:16, 11] = 0
    edge[11:13, 14:16] = 0
    edge[4:11, 2] = 1
    edge[1:8, 4] = 2
    edge[3:5, 6:8] = 3
    edge[12:14, 2:4] = 4
    expected = np.full((9, 9), 8, dtype=np.int8)
    expected[1:8, 2] = 1
    expected[3:5, 5:7] = 3
    assert np.array_equal(solve(edge), expected)
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(edge, turns)), np.rot90(expected, turns))
    tied = edge.copy()
    tied[1:8, 4] = 0
    tied[4:11, 4] = 2
    corner = edge.copy()
    corner[0, 5], corner[0, 0] = 0, 9
    unsupported = edge.copy()
    unsupported[10, 14] = 0
    for invalid in (tied, corner, unsupported):
        try:
            solve(invalid)
        except ValueError:
            continue
        raise AssertionError("Invalid geometric marker or ballot accepted")
    p = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], generate_puzzle(0, 2, 1)[s])
               for k in ("input", "output"))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            q = generate_puzzle(100 + 10 * nt + ns, nt, ns)
            assert len(q["train"]) == nt and len(q["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt output accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                crop_shapes=len(crops), palettes=len(palettes), shape_classes=len(vocabulary),
                class_counts=sorted(class_counts), cavity_counts=sorted(hole_counts),
                weights=sorted(weights), sides=sorted(sides), bar_lengths=sorted(bar_lengths),
                repeated_pairs=repeated, zero_weight_pairs=zero, shared_winner_pairs=shared,
                boundary_carriers=boundary, crossing_cavities=crossing, multiray_pairs=multiray,
                novel_test_pairs=new_tests)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
