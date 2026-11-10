"""Address a colored payload first, then fit it into a scaled blueprint.

No files, seeds, generation metadata, or training labels are used by solve().
Experiments are bounded and in memory; this module writes no artifacts.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import hashlib

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _N4:
                point = r + dr, c + dc
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(tuple(sorted(cells)))
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _normal(cells):
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _mask(shape):
    h = max(r for r, c in shape) + 1
    w = max(c for r, c in shape) + 1
    result = np.zeros((h, w), dtype=bool)
    for p in shape:
        result[p] = True
    return result


def _rotate(shape, turns):
    for _ in range(turns % 4):
        shape = _normal(tuple((-c, r) for r, c in shape))
    return shape


def _canonical(shape):
    return min(_rotate(shape, q) for q in range(4))


def _key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.uint8).tobytes()


def _sector(cells, dimensions):
    """Centroid of cell centers in thirds of the full tight rectangle.

    Exact boundary centroids are undefined. Integer comparisons avoid rounding.
    """
    if not cells:
        return None
    n = len(cells)
    address = []
    for axis, size in enumerate(dimensions):
        value = 3 * (2 * sum(p[axis] for p in cells) + n)
        low, high = 2 * n * size, 4 * n * size
        if value in (low, high):
            return None
        address.append(0 if value < low else 2 if value > high else 1)
    return 3 * address[0] + address[1]


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=False)


def _blueprint(board, background, frame):
    rows = tuple(map(int, np.flatnonzero(np.all(board == frame, axis=1))))
    cols = tuple(map(int, np.flatnonzero(np.all(board == frame, axis=0))))
    if (len(rows) != 4 or len(cols) != 4 or rows[0] != 0 or cols[0] != 0
            or rows[-1] != board.shape[0] - 1 or cols[-1] != board.shape[1] - 1):
        raise ValueError("Blueprint needs a complete 3 by 3 frame lattice")
    heights = tuple(b - a - 1 for a, b in zip(rows, rows[1:]))
    widths = tuple(b - a - 1 for a, b in zip(cols, cols[1:]))
    if not all(2 <= n <= 4 for n in heights + widths):
        raise ValueError("Panel spans must be 2..4")
    lattice = np.zeros(board.shape, dtype=bool)
    lattice[list(rows), :] = True
    lattice[:, list(cols)] = True
    if np.any(board[~lattice] == frame):
        raise ValueError("Frame ink may occur only on blueprint lattice lines")
    fills = set(map(int, np.unique(board))) - {background, frame}
    if len(fills) != 1:
        raise ValueError("Blueprint interiors need one scaffold color")
    scaffold = fills.pop()
    holes = {}
    for cells in _components(board == background):
        r, c = cells[0]
        rr = next((i for i in range(3) if rows[i] < r < rows[i + 1]), None)
        cc = next((i for i in range(3) if cols[i] < c < cols[i + 1]), None)
        if rr is None or cc is None:
            raise ValueError("Unenclosed cavity")
        sector = 3 * rr + cc
        if sector in holes or any(not (rows[rr] < x < rows[rr + 1]
                                       and cols[cc] < y < cols[cc + 1]) for x, y in cells):
            raise ValueError("Each occupied panel must have exactly one cavity")
        bounds = _bounds(cells)
        holes[sector] = {"cells": cells, "shape": _normal(cells), "bounds": bounds}
    if not 2 <= len(holes) <= 9:
        raise ValueError("Expected 2..9 cavities")
    return scaffold, holes, heights, widths


def _fits(payload, hole_shape, background):
    target = _mask(hole_shape)
    fits = {}
    for turns in range(4):
        rotated = np.rot90(payload, turns).copy()
        if rotated.shape == target.shape and np.array_equal(rotated != background, target):
            fits.setdefault(_key(rotated), (rotated, turns))
    return list(fits.values())


def _recover(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= grid.size // 2:
        raise ValueError("Canvas background must be a strict majority")
    candidates = []
    for frame in range(10):
        if frame == background:
            continue
        for cells in _components(grid == frame):
            r0, r1, c0, c1 = _bounds(cells)
            if min(r1 - r0, c1 - c0) < 10 or max(r1 - r0, c1 - c0) > 15:
                continue
            board = grid[r0:r1, c0:c1]
            try:
                scaffold, holes, heights, widths = _blueprint(board, background, frame)
            except ValueError:
                continue
            outside = np.ones(grid.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            moat = np.zeros(grid.shape, dtype=bool)
            moat[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1] = True
            if np.any(outside & moat & (grid != background)):
                continue
            objects = _components(outside & (grid != background))
            if len(objects) != len(holes):
                continue
            rectangles = [(r0, r1, c0, c1)] + [_bounds(obj) for obj in objects]
            if any(a < f + 1 and e < b + 1 and c < h + 1 and g < d + 1
                   for i, (a, b, c, d) in enumerate(rectangles)
                   for e, f, g, h in rectangles[i + 1:]):
                continue  # One-cell separation of tight object rectangles.
            for scale in (2, 3):
                if max(board.shape) * scale > 30:
                    continue
                logical, pieces, occupied = board.copy(), [], set()
                try:
                    for cells in objects:
                        a, b, c, d = _bounds(cells)
                        patch = grid[a:b, c:d]
                        if patch.shape[0] % scale or patch.shape[1] % scale:
                            raise ValueError("Incomplete macrocell")
                        payload = patch[::scale, ::scale].copy()
                        if not np.array_equal(patch, np.repeat(np.repeat(payload, scale, 0), scale, 1)):
                            raise ValueError("Nonuniform macrocell")
                        inks = set(map(int, np.unique(payload))) - {background, frame}
                        if len(inks) != 1 or scaffold in inks:
                            raise ValueError("One body color per payload, distinct from scaffold")
                        body = inks.pop()
                        markers = tuple(map(tuple, np.argwhere(payload == frame)))
                        address = _sector(markers, payload.shape)
                        if address not in holes or address in occupied:
                            raise ValueError("Locator does not address one unused cavity")
                        fits = _fits(payload, holes[address]["shape"], background)
                        if len(fits) != 1:
                            raise ValueError("Addressed cavity has no unique colored quarter-turn fit")
                        fitted, turns = fits[0]
                        occupied.add(address)
                        hr0, hr1, hc0, hc1 = holes[address]["bounds"]
                        view = logical[hr0:hr1, hc0:hc1]
                        mask = fitted != background
                        view[mask] = fitted[mask]
                        after = _sector(tuple(map(tuple, np.argwhere(fitted == frame))), fitted.shape)
                        shape = _normal(tuple(map(tuple, np.argwhere(payload != background))))
                        pieces.append({"address": address, "body": body, "markers": len(markers),
                                       "payload": payload, "shape": shape, "turns": turns,
                                       "after_address": after, "bounds": (a, b, c, d),
                                       "class": _canonical(shape)})
                    if occupied != set(holes):
                        raise ValueError("Cavities not covered exactly once")
                    output = np.repeat(np.repeat(logical, scale, 0), scale, 1)
                    if np.any(output == background):
                        raise ValueError("Unfilled cavity")
                except ValueError:
                    continue
                info = {"background": background, "frame": frame, "scaffold": scaffold,
                        "scale": scale, "board": board.copy(), "board_bounds": (r0, r1, c0, c1),
                        "heights": heights, "widths": widths, "holes": holes, "pieces": pieces,
                        "logical": logical}
                candidates.append((output, info))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one input-derived blueprint, scale and routed assembly")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode local locators BEFORE rotation; preserve every colored payload cell."""
    return _recover(input_grid)[0]


@lru_cache(maxsize=512)
def _marker_sets(shape):
    dimensions = _mask(shape).shape
    result = [[] for _ in range(9)]
    for count in range(1, min(4, len(shape) - 1) + 1):
        for marks in combinations(shape, count):
            sector = _sector(marks, dimensions)
            if sector is not None:
                result[sector].append(marks)
    return tuple(tuple(group) for group in result)


def _options(hole_shape, address, low, high, witness, centroid_witness=False):
    result = []
    for q in range(4):
        source = _rotate(hole_shape, q)
        if witness and q == 0:
            continue
        for marks in _marker_sets(source)[address]:
            if not low <= len(marks) <= high:
                continue
            if centroid_witness:
                r0, r1, c0, c1 = _bounds(marks)
                box_address = _sector(((r0, c0), (r1 - 1, c1 - 1)), _mask(source).shape)
                if box_address is None or box_address == address:
                    continue
            if witness:
                # Rotation is about the full payload rectangle, not marker bbox.
                marks_grid = np.zeros(_mask(source).shape, dtype=bool)
                for p in marks:
                    marks_grid[p] = True
                turned = np.rot90(marks_grid, -q)
                if _sector(tuple(map(tuple, np.argwhere(turned))), turned.shape) == address:
                    continue
            result.append((source, marks))
    return result


def _grow(rng, height, width, large=False):
    limit = min(11, height * width - 1)
    area = int(rng.integers(min(5 if large else 3, limit), limit + 1))
    cells = {(int(rng.integers(height)), int(rng.integers(width)))}
    compact = bool(rng.integers(2))
    while len(cells) < area:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _N4
                           if 0 <= r + dr < height and 0 <= c + dc < width} - cells)
        weights = np.array([1 + (3 if compact else -0.2) * sum((r + dr, c + dc) in cells
                                                             for dr, dc in _N4)
                            for r, c in frontier], dtype=float)
        point = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
        cells.add(point)
    return _normal(tuple(cells))


def _choose_payload(rng, shape, address, height, width, low, high, witness, centroid_witness=False):
    choices = []
    for hole in sorted({_rotate(shape, q) for q in range(4)}):
        mask = _mask(hole)
        if mask.shape[0] <= height and mask.shape[1] <= width:
            options = _options(hole, address, low, high, witness, centroid_witness)
            if options:
                choices.append((hole, options))
    if not choices:
        return None
    hole, options = choices[int(rng.integers(len(choices)))]
    source, marks = options[int(rng.integers(len(options)))]
    return {"hole": hole, "source": source, "marks": marks}


def _free_origins(blocked, height, width):
    if height > blocked.shape[0] or width > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = (summed[height:, width:] - summed[:-height, width:]
              - summed[height:, :-width] + summed[:-height, :-width])
    return np.argwhere(totals == 0)


def _pack(rng, objects, background):
    order = sorted(range(len(objects)), key=lambda i: -objects[i].size)
    for attempt in range(26):
        size = (int(rng.integers(24, 31)), int(rng.integers(24, 31))) if attempt < 12 else (30, 30)
        canvas = np.full(size, background, dtype=np.int8)
        blocked = np.zeros(size, dtype=bool)
        for index in order:
            obj = objects[index]
            h, w = obj.shape
            origins = _free_origins(blocked, h, w)
            if not len(origins):
                break
            # A final deterministic first-fit pass accommodates the guaranteed
            # small fallback pieces used by late construction attempts.
            position = 0 if attempt == 25 else int(rng.integers(len(origins)))
            r, c = map(int, origins[position])
            canvas[r:r + h, c:c + w] = obj
            blocked[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True
        else:
            if np.count_nonzero(canvas == background) > canvas.size // 2:
                # Trim a random amount of empty margins, preserving majority.
                r0, r1, c0, c1 = _bounds(tuple(map(tuple, np.argwhere(canvas != background))))
                r0 = max(0, r0 - int(rng.integers(4)))
                c0 = max(0, c0 - int(rng.integers(4)))
                r1 = min(size[0], r1 + int(rng.integers(4)))
                c1 = min(size[1], c1 + int(rng.integers(4)))
                crop = canvas[r0:r1, c0:c1]
                return crop.copy() if np.count_nonzero(crop == background) > crop.size // 2 else canvas
    return None


def _sizes(rng, scale, rich):
    if scale == 3:
        return (2, 2, 2)
    for _ in range(32):
        values = tuple(map(int, rng.integers(3 if rich else 2, 5, size=3)))
        if sum(values) <= 11:
            return values
    return (3, 3, 3)


def _make_pair(rng, scale, addresses, is_test, rich, centroid_witness=False):
    # Rejection budgets concern in-memory arrays only, with no retained retries.
    for attempt in range(64):
        heights, widths = _sizes(rng, scale, rich), _sizes(rng, scale, rich)
        if centroid_witness:
            # A span of four allows a three-cell centroid to select an edge
            # while the locator's bounding-box center selects the middle.
            heights = tuple(4 if row in (addresses[0] // 3, addresses[1] // 3) else 3
                            for row in range(3))
        palette = list(map(int, rng.permutation(10)))
        background, frame, scaffold = palette[:3]
        bodies = [palette[3 + i % 7] for i in range(len(addresses))]
        if len(addresses) > 2 and rng.random() < .4:
            bodies[2:] = list(map(int, rng.choice(palette[3:6], size=len(addresses) - 2)))
        descriptors = []
        for index, address in enumerate(addresses):
            rr, cc = divmod(address, 3)
            if index < 2:
                h = min(heights[addresses[0] // 3], heights[addresses[1] // 3])
                w = min(widths[addresses[0] % 3], widths[addresses[1] % 3])
            else:
                h, w = heights[rr], widths[cc]
            if is_test and index == 0:
                low = 4
            elif centroid_witness and index == 0:
                low = 3
            else:
                low = 2 if address == 4 else 1
            high = 4 if is_test else 3
            descriptor = None
            for shape_attempt in range(64):
                if index == 1:
                    shape = descriptors[0]["hole"]
                elif attempt >= 48 or shape_attempt >= 48:
                    if centroid_witness and index == 0:
                        shape = ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (3, 1))
                    elif is_test and index < 2:
                        shape = ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0))
                    else:
                        shape = ((0, 0), (0, 1), (1, 0))
                else:
                    shape = _grow(rng, h, w, is_test and index == 0)
                if len({_rotate(shape, q) for q in range(4)}) != 4:
                    continue  # Generation uses unambiguous geometric orientations.
                descriptor = _choose_payload(rng, shape, address, heights[rr], widths[cc],
                                             low, high, index == 0, centroid_witness and index == 0)
                if descriptor is not None:
                    break
            if descriptor is None:
                break
            descriptors.append(descriptor)
        if len(descriptors) != len(addresses):
            continue
        rows, cols = [0], [0]
        for h in heights:
            rows.append(rows[-1] + h + 1)
        for w in widths:
            cols.append(cols[-1] + w + 1)
        board = np.full((rows[-1] + 1, cols[-1] + 1), scaffold, dtype=np.int8)
        board[rows, :] = frame
        board[:, cols] = frame
        logical = board.copy()
        objects = []
        for address, body, descriptor in zip(addresses, bodies, descriptors):
            rr, cc = divmod(address, 3)
            hole_mask = _mask(descriptor["hole"])
            h, w = hole_mask.shape
            top = rows[rr] + 1 + int(rng.integers(heights[rr] - h + 1))
            left = cols[cc] + 1 + int(rng.integers(widths[cc] - w + 1))
            view = board[top:top + h, left:left + w]
            view[hole_mask] = background
            source_mask = _mask(descriptor["source"])
            payload = np.full(source_mask.shape, background, dtype=np.int8)
            payload[source_mask] = body
            for p in descriptor["marks"]:
                payload[p] = frame
            fits = _fits(payload, descriptor["hole"], background)
            assert len(fits) == 1
            fitted = fits[0][0]
            logical[top:top + h, left:left + w][hole_mask] = fitted[hole_mask]
            objects.append(np.repeat(np.repeat(payload, scale, 0), scale, 1))
        canvas = _pack(rng, [board] + objects, background)
        if canvas is not None:
            output = np.repeat(np.repeat(logical, scale, 0), scale, 1)
            return {"input": canvas, "output": output}
    raise RuntimeError("Bounded construction exhausted")


def _count(value, name, low, high):
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _address_order(rng, addresses):
    # An edge address supports the test's larger compound locator, and witnesses
    # that the address is read in the source frame before its quarter-turn fit.
    edges = [a for a in addresses if a in (1, 3, 5, 7)]
    first = int(rng.choice(edges))
    rest = list(map(int, rng.permutation([a for a in addresses if a != first])))
    return [first] + rest


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    core = {4, *map(int, rng.choice([0, 2, 6, 8], size=2, replace=False)),
            *map(int, rng.choice([1, 3, 5, 7], size=2, replace=False))}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            is_test = split == "test"
            if not is_test and index < 2:
                addresses = core if index == 0 else (set(range(9)) - core) | {4}
                scale = 2 if index == 0 else 3
            else:
                count_pieces = int(rng.integers(3 if is_test else 2, 8))
                # Guarantee at least one edge, without forcing a fixed panel.
                edge = int(rng.choice([1, 3, 5, 7]))
                addresses = {edge, *map(int, rng.choice([a for a in range(9) if a != edge],
                                                       size=count_pieces - 1, replace=False))}
                scale = 2 if is_test else int(rng.choice([2, 3]))
            addresses = _address_order(rng, sorted(addresses))
            for _ in range(8):
                pair = _make_pair(rng, scale, addresses, is_test, is_test or index == 0,
                                  not is_test and index == 0)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct-input construction exhausted")
    return puzzle


def _address_ablation(info):
    """A valid shape-only alternate assembly, with two routed payloads swapped."""
    pieces, holes = info["pieces"], info["holes"]
    for i, a in enumerate(pieces):
        for b in pieces[i + 1:]:
            if a["class"] != b["class"] or a["body"] == b["body"]:
                continue
            alternate = info["logical"].copy()
            for piece, address in ((a, b["address"]), (b, a["address"])):
                fitted = _fits(piece["payload"], holes[address]["shape"], info["background"])
                if len(fitted) != 1:
                    break
                patch = fitted[0][0]
                r0, r1, c0, c1 = holes[address]["bounds"]
                mask = patch != info["background"]
                alternate[r0:r1, c0:c1][mask] = patch[mask]
            else:
                assert not np.array_equal(alternate, info["logical"])
                return alternate
    raise AssertionError("No differently colored congruent routing witness")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training_addresses, scales = set(), set(), set()
    centroid_witnesses = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(out, expected), "Wrong routed assembly"
            assert not np.array_equal(inp, out)
            k, board = info["scale"], info["board"]
            logical = out[::k, ::k]
            structure = board != info["background"]
            assert np.array_equal(logical[structure], board[structure])
            for color in {p["body"] for p in info["pieces"]}:
                assert np.count_nonzero(inp == color) == np.count_nonzero(out == color)
            marker_pixels = sum(p["markers"] * k * k for p in info["pieces"])
            assert np.count_nonzero(out == info["frame"]) == (
                np.count_nonzero(board == info["frame"]) * k * k + marker_pixels)
            assert all(3 <= len(p["shape"]) <= 11 for p in info["pieces"])
            assert all(len({_rotate(p["shape"], q) for q in range(4)}) == 4
                       for p in info["pieces"])
            assert any(p["turns"] and p["after_address"] != p["address"] for p in info["pieces"])
            _address_ablation(info)
            if split == "train":
                training_addresses.update(info["holes"])
                scales.add(k)
                assert max(p["markers"] for p in info["pieces"]) <= 3
                for p in info["pieces"]:
                    marks = tuple(map(tuple, np.argwhere(p["payload"] == info["frame"])))
                    r0, r1, c0, c1 = _bounds(marks)
                    box_address = _sector(((r0, c0), (r1 - 1, c1 - 1)), p["payload"].shape)
                    centroid_witnesses += box_address is not None and box_address != p["address"]
            else:
                assert max(p["markers"] for p in info["pieces"]) == 4
    assert training_addresses == set(range(9))
    assert scales == {2, 3}
    assert centroid_witnesses, "Training must distinguish centroid from marker bounding-box center"


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = pair[role]
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, compact feature sets, no retained grids or disk I/O."""
    stats = {k: set() for k in ("puzzles", "input_shapes", "output_shapes", "boards", "shapes",
                               "areas", "counts", "roles", "markers", "addresses", "panels")}
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        stats["puzzles"].add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                out, info = _recover(pair["input"])
                assert np.array_equal(out, pair["output"])
                stats["input_shapes"].add(pair["input"].shape)
                stats["output_shapes"].add(out.shape)
                stats["boards"].add(hashlib.sha256((info["board"] == info["background"]).tobytes()).hexdigest())
                stats["counts"].add(len(info["pieces"]))
                stats["roles"].add((info["background"], info["frame"], info["scaffold"]))
                stats["panels"].add((info["heights"], info["widths"]))
                for piece in info["pieces"]:
                    stats["shapes"].add(piece["class"])
                    stats["areas"].add(len(piece["shape"]))
                    stats["markers"].add(piece["markers"])
                    stats["addresses"].add(piece["address"])
        assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
    assert len(stats["puzzles"]) == 42 and pairs == 210
    assert len(stats["input_shapes"]) >= 20 and len(stats["output_shapes"]) >= 8
    assert len(stats["boards"]) >= 120 and len(stats["panels"]) >= 40
    assert len(stats["shapes"]) >= 30 and len(stats["roles"]) >= 80
    assert stats["addresses"] == set(range(9))
    assert stats["markers"] == {1, 2, 3, 4}
    assert {3, 4, 5, 6, 7, 8} <= stats["areas"]
    assert {2, 3, 4, 5, 6, 7} <= stats["counts"]
    for counts in ((2, 1), (10, 4), (None, 4), (2, None)):
        p = generate_puzzle(193, *counts)
        validate_generated_puzzle(p)
        for split, count in zip(("train", "test"), counts):
            assert count is None or len(p[split]) == count
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (4, "1")):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    sample = generate_puzzle(19, 2, 1)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt label accepted")
    inp = sample["train"][0]["input"].copy()
    _, info = _recover(inp)
    r0, r1, c0, c1 = info["pieces"][0]["bounds"]
    point = np.argwhere(inp[r0:r1, c0:c1] == info["pieces"][0]["body"])[0]
    inp[r0 + int(point[0]), c0 + int(point[1])] = info["background"]
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("Broken macrocell accepted")
    inp = sample["train"][1]["input"].copy()
    _, info = _recover(inp)
    piece = info["pieces"][0]
    marker_options = _marker_sets(piece["shape"])
    other = next(p["address"] for p in info["pieces"][1:] if marker_options[p["address"]])
    marks = marker_options[other][0]
    payload = np.full(piece["payload"].shape, info["background"], dtype=np.int8)
    payload[piece["payload"] != info["background"]] = piece["body"]
    for p in marks:
        payload[p] = info["frame"]
    r0, r1, c0, c1 = piece["bounds"]
    inp[r0:r1, c0:c1] = np.repeat(np.repeat(payload, info["scale"], 0), info["scale"], 1)
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting locator addresses accepted")
    return {"seeds": 42, "pairs": pairs, **{k: len(v) for k, v in stats.items()}}
