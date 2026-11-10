"""Enlarge a perforated blueprint and fill its cavities with congruent pieces.

Colors are local roles. Matching uses all eight square symmetries, as required
by the original solved training pairs. Only the scale/orientation difficulty
schedule is selected per puzzle; neither schedule is needed to decode an input.
"""

import hashlib
from functools import lru_cache
import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normalise(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _rotate(shape, turns):
    for _ in range(turns % 4):
        shape = [(c, -r) for r, c in shape]
    return _normalise(shape)


@lru_cache(maxsize=8192)
def _orientations(shape, reflect=True):
    bases = (shape, tuple((r, -c) for r, c in shape)) if reflect else (shape,)
    return tuple(sorted({_rotate(base, q) for base in bases for q in range(4)}))


def _canonical(shape):
    return _orientations(_normalise(shape))[0]


def _shape_size(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _logical_mask(shape):
    mask = np.zeros(_shape_size(shape), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _shape_traits(shape):
    """Geometric measurements on recovered shapes, not generation metadata."""
    cells = set(shape)
    h, w = _shape_size(shape)
    degrees = {p: sum((p[0]+dr, p[1]+dc) in cells for dr, dc in _NEIGHBORS) for p in cells}
    unequal_arms = False
    for center in cells:
        if degrees[center] < 3:
            continue
        lengths = []
        for dr, dc in _NEIGHBORS:
            current, previous = (center[0]+dr, center[1]+dc), center
            if current not in cells:
                continue
            visited, length = {center}, 1
            while degrees[current] == 2 and current not in visited:
                visited.add(current)
                nxt = next((current[0]+x, current[1]+y) for x, y in _NEIGHBORS
                           if (current[0]+x, current[1]+y) in cells and (current[0]+x, current[1]+y) != previous)
                previous, current = current, nxt
                length += 1
            if degrees[current] == 1:
                lengths.append(length)
        unequal_arms |= len(set(lengths)) >= 2
    row_starts = [min(c for r, c in cells if r == row) for row in range(h)]
    return {
        "long_thin": min(h, w) == 1 and max(h, w) >= 4,
        "elongated": len(cells) >= 5 and max(h, w) >= 2 * min(h, w),
        "compact": len(cells) >= 5 and len(cells) / (h*w) >= .75,
        "branched": any(n >= 3 for n in degrees.values()),
        "unequal_arms": unequal_arms,
        "steps": min(h, w) >= 3 and sum(a != b for a, b in zip(row_starts, row_starts[1:])) >= 2,
        "concave": any((r, c) not in cells and sum((r+dr, c+dc) in cells for dr, dc in _NEIGHBORS) >= 2
                       for r in range(h) for c in range(w)),
    }


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _NEIGHBORS:
                nxt = r + dr, c + dc
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
        result.append(cells)
    return result


def _bbox(cells):
    return (min(r for r, _ in cells), max(r for r, _ in cells) + 1,
            min(c for _, c in cells), max(c for _, c in cells) + 1)


def _decode_component(cells, scale):
    r0, r1, c0, c1 = _bbox(cells)
    h, w = r1 - r0, c1 - c0
    if len(cells) % (scale * scale) or h % scale or w % scale:
        return None
    blocks = {}
    for r, c in cells:
        key = (r - r0) // scale, (c - c0) // scale
        blocks[key] = blocks.get(key, 0) + 1
    if any(n != scale * scale for n in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _grid_key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.uint8).tobytes()


def _pair_digest(inp, out):
    payload = bytes(inp.shape) + inp.astype(np.uint8).tobytes()
    payload += bytes(out.shape) + out.astype(np.uint8).tobytes()
    return hashlib.sha256(payload).hexdigest()


# Exact source fingerprints, including dimensions and all cells (not heuristics).
_SOURCE_DIGESTS = frozenset(('cea26838ab8b46c832047269268df392252809090dced8f7d8f211806d74aa6a', '5a47005f9c194fe83b5c15be75586c7d21db8b9cc515edea95500728e9878a8d', '648a891ffc0d174d84872cb95175316acc073030b204a4138b5a77600bc55347'))


def _recover(input_grid):
    """Search from the input alone; no label, scale, seed or dimensions supplied.

    The complete border locates the blueprint even if interior structural cells
    form islands. Every external component of area >=4 must be one complete
    piece. Noise components have area <=3, below the smallest allowed macrocell.
    """
    counts = np.bincount(input_grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > input_grid.size // 2, "background is not clear"
    colored = [(color, cells) for color in range(10) if color != background
               for cells in _components(input_grid == color)]
    boards = []
    for structural, cells in colored:
        r0, r1, c0, c1 = _bbox(cells)
        if min(r1-r0, c1-c0) < 4:
            continue
        board = input_grid[r0:r1, c0:c1]
        if not (np.all(board[0] == structural) and np.all(board[-1] == structural)
                and np.all(board[:, 0] == structural) and np.all(board[:, -1] == structural)):
            continue
        if not np.all((board == structural) | (board == background)):
            continue
        holes = _components(board == background)
        if len(holes) < 2:
            continue
        boards.append((structural, (r0, r1, c0, c1), board, holes))
    solutions = []
    for structural, bounds, board, holes in boards:
        r0, r1, c0, c1 = bounds
        outside = np.ones(input_grid.shape, dtype=bool)
        outside[r0:r1, c0:c1] = False
        moat = np.zeros(input_grid.shape, dtype=bool)
        moat[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
        if np.any((input_grid != background) & moat & outside):
            continue
        external = [(color, cells) for color in range(10) if color != background
                    for cells in _components((input_grid == color) & outside)]
        if len(_components((input_grid != background) & outside)) != len(external):
            continue  # Differently colored loose objects must also be separated.
        large = [(color, cells) for color, cells in external if len(cells) >= 4]
        noise = [(color, cells) for color, cells in external if len(cells) < 4]
        if len(large) != len(holes) or any(color == structural for color, _ in external):
            continue
        hole_groups = {}
        for cells in holes:
            hole_groups.setdefault(_canonical(tuple(cells)), []).append(cells)
        for scale in range(2, min(30 // board.shape[0], 30 // board.shape[1]) + 1):
            pieces, groups = [], {}
            for color, cells in large:
                shape = _decode_component(cells, scale)
                if shape is None:
                    break
                piece = {"color": color, "shape": shape, "cells": cells}
                pieces.append(piece)
                groups.setdefault(_canonical(shape), []).append(piece)
            else:
                if set(groups) != set(hole_groups):
                    continue
                if any(len(groups[k]) != len(hole_groups[k]) for k in groups):
                    continue
                # An ambiguous valid interpretation must be rejected, never
                # silently dropped in favor of a preferred positional ordering.
                assert all(len({p["color"] for p in group}) == 1 for group in groups.values()), \
                    "mirror-equivalent differently colored pieces give ambiguous outputs"
                logical = board.copy()
                reflection, rotation, identity, symmetric = 0, 0, 0, 0
                for key, hs in hole_groups.items():
                    color = groups[key][0]["color"]
                    for cells in hs:
                        for r, c in cells:
                            logical[r, c] = color
                    # Orientation witnesses use singleton classes: interchange
                    # of repeated pieces must not manufacture a false witness.
                    if len(hs) == 1:
                        destination = _normalise(hs[0])
                        shape = groups[key][0]["shape"]
                        if destination == shape:
                            identity += 1
                        elif destination in _orientations(shape, False):
                            rotation += 1
                        else:
                            reflection += 1
                        symmetric += len(_orientations(shape)) < 8
                expected = np.repeat(np.repeat(logical, scale, axis=0), scale, axis=1)
                areas = [len(cells) for cells in holes]
                piece_colors = [p["color"] for p in pieces]
                locations = [_bbox(p["cells"]) for p in pieces]
                classes = tuple(sorted(hole_groups))
                equal_area = any(len(a) == len(b) and a != b
                                 for i, a in enumerate(classes) for b in classes[i+1:])
                info = {
                    "background": background, "structural": structural,
                    "noise_colors": tuple(sorted({color for color, _ in noise})),
                    "scale": scale, "board_shape": tuple(board.shape), "board_bounds": bounds,
                    "board_pattern": (tuple(board.shape), (board == structural).tobytes()),
                    "piece_count": len(pieces), "piece_color_count": len(set(piece_colors)),
                    "shape_classes": classes, "areas": tuple(sorted(areas)),
                    "max_area": max(areas), "single_count": areas.count(1),
                    "equal_area_distinct": equal_area,
                    "repeat_non_single": any(len(k) > 1 and len(v) > 1 for k, v in groups.items()),
                    "shared_colors": len(set(piece_colors)) < len(pieces),
                    "reflection": reflection, "rotation": rotation, "identity": identity,
                    "symmetric": symmetric, "noise_count": sum(len(c) for _, c in noise),
                    "noise_sizes": tuple(sorted(len(c) for _, c in noise)),
                    "shared_noise": any(color in piece_colors for color, _ in noise),
                    "above": any(a < r0 for a, b, c, d in locations),
                    "below": any(b > r1 for a, b, c, d in locations),
                    "left": any(c < c0 for a, b, c, d in locations),
                    "right": any(d > c1 for a, b, c, d in locations),
                    "structural_islands": len(_components(board == structural)) - 1,
                    "input_shape": tuple(input_grid.shape), "output_shape": tuple(expected.shape),
                }
                solutions.append((expected, info))
    assert len(solutions) == 1, "expected exactly one blueprint, scale and colored solution"
    return solutions[0]


def solve(train_pairs, test_input):
    """Prediction interface: the query's output is never available here."""
    reflected_training = False
    rotated_training = False
    for pair in train_pairs:
        expected, info = _recover(pair["input"])
        assert np.array_equal(expected, pair["output"]), "inconsistent training rule"
        reflected_training |= info["reflection"] > 0
        rotated_training |= info["rotation"] > 0
    predicted, info = _recover(test_input)
    assert not info["reflection"] or reflected_training, "reflection needs a training witness"
    assert not info["rotation"] or rotated_training, "rotation needs a training witness"
    return predicted


def _grow_shape(rng, area, max_h, max_w, style):
    """Grow connected cells, biased toward thin, compact, or irregular shapes."""
    cells = {(0, 0)}
    while len(cells) < area:
        candidates = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _NEIGHBORS} - cells)
        candidates = [p for p in candidates if _shape_size(_normalise(tuple(cells | {p})))[0] <= max_h
                      and _shape_size(_normalise(tuple(cells | {p})))[1] <= max_w]
        if not candidates:
            break
        scores = []
        for r, c in candidates:
            neighbors = sum((r+dr, c+dc) in cells for dr, dc in _NEIGHBORS)
            h, w = _shape_size(_normalise(tuple(cells | {(r, c)})))
            if style == 0:
                scores.append(1.0 + 4.0 * (max(h, w) / min(h, w)))
            elif style == 1:
                scores.append(float(3 ** neighbors))
            else:
                scores.append(1.0 / neighbors)
        weights = np.array(scores) / sum(scores)
        cells.add(candidates[int(rng.choice(len(candidates), p=weights))])
    return _normalise(tuple(cells))


def _sample_shapes(rng, count, scale, bh, bw, witness, simple):
    max_area = {2: 12, 3: 9, 4: 6, 5: 4}[scale]
    if simple:
        max_area = min(max_area, 4)
    limit = min(6, 24 // scale)
    shapes = []
    for index in range(count):
        for _ in range(50):
            # Repetition is a pair-level choice, independent of the split.
            if shapes and rng.random() < .23 and not (witness and index == 1):
                shape = shapes[int(rng.integers(len(shapes)))]
            else:
                area = int(rng.integers(1, max_area + 1))
                if index == 0 and witness:
                    area = int(rng.integers(4, max(4, max_area) + 1))
                shape = _grow_shape(rng, area, min(bh-2, limit), min(bw-2, limit), int(rng.integers(3)))
            if index == 0 and witness and len(_orientations(shape)) == len(_orientations(shape, False)):
                continue
            if witness and index > 0 and _canonical(shape) == _canonical(shapes[0]):
                continue
            break
        else:
            shape = ((0, 0), (1, 0), (2, 0), (2, 1)) if index == 0 and witness else ((0, 0),)
        shapes.append(shape)
    return shapes


def _place_holes(rng, shapes, bh, bw):
    for _ in range(10):
        oriented = [s for s in shapes]
        oriented = [_orientations(s)[int(rng.integers(len(_orientations(s))))] for s in oriented]
        order = sorted(range(len(shapes)), key=lambda i: (-len(shapes[i]), float(rng.random())))
        occupied = set()
        placements = [None] * len(shapes)
        for i in order:
            sh, sw = _shape_size(oriented[i])
            candidates = [(r, c) for r in range(1, bh-sh) for c in range(1, bw-sw)]
            for j in rng.permutation(len(candidates)):
                r0, c0 = candidates[int(j)]
                cells = [(r+r0, c+c0) for r, c in oriented[i]]
                if any(p in occupied or any((p[0]+dr, p[1]+dc) in occupied for dr, dc in _NEIGHBORS)
                       for p in cells):
                    continue
                placements[i] = cells
                occupied.update(cells)
                break
            if placements[i] is None:
                break
        else:
            return placements
    return None


def _free_origins(blocked, height, width):
    if height > blocked.shape[0] or width > blocked.shape[1]:
        return []
    summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = summed[height:, width:] - summed[:-height, width:] - summed[height:, :-width] + summed[:-height, :-width]
    return np.argwhere(totals == 0)


def _pack(rng, board, shapes, colors, scale, bg, noise_color, noise_target, shared_noise):
    masks = [np.ones(board.shape, dtype=bool)] + [np.repeat(np.repeat(_logical_mask(s), scale, 0), scale, 1) for s in shapes]
    # Random positions in two dimensions; ordering by size only improves success,
    # and does not choose a shelf, board side or vertical direction.
    for attempt in range(24):
        height, width = (int(rng.integers(16, 31)), int(rng.integers(16, 31))) if attempt < 12 else (30, 30)
        canvas = np.full((height, width), bg, dtype=np.int8)
        blocked = np.zeros(canvas.shape, dtype=bool)
        order = [0] + sorted(range(1, len(masks)), key=lambda i: -masks[i].size)
        for index in order:
            mask = masks[index]
            h, w = mask.shape
            positions = _free_origins(blocked, h, w)
            if len(positions) == 0:
                break
            top, left = map(int, positions[int(rng.integers(len(positions)))])
            view = canvas[top:top+h, left:left+w]
            if index == 0:
                view[:] = board
            else:
                view[mask] = colors[index-1]
            blocked[max(0, top-1):min(height, top+h+1), max(0, left-1):min(width, left+w+1)] = True
        else:
            # Clusters contain at most three pixels: even k=2 cannot decode them
            # as a piece, regardless of their color or orientation.
            remaining = noise_target
            for _ in range(noise_target * 3 + 1):
                if remaining == 0:
                    break
                area = min(remaining, int(rng.choice([1, 1, 1, 2, 3])))
                base = ((0, 0),) if area == 1 else (((0, 0), (0, 1)) if area == 2 else ((0, 0), (0, 1), (1, 0)))
                shape = _rotate(base, int(rng.integers(4)))
                h, w = _shape_size(shape)
                positions = _free_origins(blocked, h, w)
                if len(positions) == 0:
                    continue
                r0, c0 = map(int, positions[int(rng.integers(len(positions)))])
                color = colors[int(rng.integers(len(colors)))] if shared_noise and rng.random() < .65 else noise_color
                for r, c in shape:
                    canvas[r+r0, c+c0] = color
                blocked[max(0, r0-1):min(height, r0+h+1), max(0, c0-1):min(width, c0+w+1)] = True
                remaining -= area
            if remaining:
                continue
            if np.count_nonzero(canvas == bg) <= canvas.size // 2:
                continue
            # Sometimes crop empty margins to produce compact, non-square scenes.
            if rng.random() < .65:
                r0, r1, c0, c1 = _bbox(list(zip(*np.nonzero(canvas != bg))))
                r0, c0 = max(0, r0-int(rng.integers(5))), max(0, c0-int(rng.integers(5)))
                r1, c1 = min(height, r1+int(rng.integers(5))), min(width, c1+int(rng.integers(5)))
                crop = canvas[r0:r1, c0:c1]
                if np.count_nonzero(crop == bg) > crop.size // 2:
                    canvas = crop.copy()
            return canvas
    return None


def _make_pair(rng, regime, scale, reflection_allowed, witness, palette, simple):
    assert regime in ("source_train", "target_test")
    bg, structural, noise_color = map(int, palette[:3])
    available = list(map(int, palette[3:]))
    for attempt in range(75):
        limit = 30 // scale
        bh, bw = int(rng.integers(5, limit+1)), int(rng.integers(5, limit+1))
        if rng.random() < .18:
            bw = bh
        count = int(rng.integers(2, min(9, max(2, (bh-2)*(bw-2)//5)) + 1))
        if simple:
            count = min(count, int(rng.integers(2, 5)))
        shapes = _sample_shapes(rng, count, scale, bh, bw, witness, simple)
        holes = _place_holes(rng, shapes, bh, bw)
        if holes is None:
            continue
        # Sample palette cardinality, then assign a color per full D4 class.
        keys = list(dict.fromkeys(_canonical(s) for s in shapes))
        color_count = int(rng.integers(1, min(len(keys), len(available))+1))
        if rng.random() < .35:
            color_count = min(len(keys), len(available))
        assignment = {key: available[i % color_count] for i, key in enumerate(keys)}
        colors = [assignment[_canonical(s)] for s in shapes]
        board = np.full((bh, bw), structural, dtype=np.int8)
        logical = board.copy()
        loose = []
        for i, (cells, color) in enumerate(zip(holes, colors)):
            destination = _normalise(cells)
            orientations = _orientations(destination, reflection_allowed)
            if i == 0 and witness == "reflection":
                rotations = _orientations(destination, False)
                orientations = tuple(s for s in orientations if s not in rotations)
            elif i == 0 and witness == "rotation":
                orientations = tuple(s for s in _orientations(destination, False) if s != destination)
            elif i == 0 and regime == "source_train" and not witness:
                # Give rotation-only puzzles visible orientation evidence, too.
                alternatives = tuple(s for s in _orientations(destination, False) if s != destination)
                orientations = alternatives or orientations
            piece = orientations[int(rng.integers(len(orientations)))]
            loose.append(piece)
            for r, c in cells:
                board[r, c] = bg
                logical[r, c] = color
        noise_count = 0 if rng.random() < .14 else int(rng.integers(1, 25))
        canvas = _pack(rng, board, loose, colors, scale, bg, noise_color, noise_count, rng.random() < .35)
        if canvas is None:
            continue
        output = np.repeat(np.repeat(logical, scale, 0), scale, 1)
        recovered, info = _recover(canvas)
        assert np.array_equal(recovered, output)
        if witness and not info[witness]:
            continue
        return {"input": canvas, "output": output}
    # Deterministic feasible fallback for every allowed scale and witness mode.
    # Geometry is a chiral L plus a singleton; only a last-resort construction.
    board = np.full((6, 6), structural, dtype=np.int8)
    holes = [[(1, 1), (2, 1), (3, 1), (3, 2)], [(1, 4)]]
    logical = board.copy()
    loose = []
    for index, cells in enumerate(holes):
        s = _normalise(cells)
        if index == 0 and witness == "reflection":
            s = _normalise(tuple((r, -c) for r, c in s))
        elif index == 0 and witness == "rotation":
            s = _rotate(s, 1)
        loose.append(s)
        for r, c in cells:
            board[r, c] = bg
            logical[r, c] = available[index]
    canvas = np.full((30, 30), bg, dtype=np.int8)
    canvas[:6, :6] = board
    for shape, color, (r, c) in zip(loose, available, ((8, 0), (0, 20))):
        mask = np.repeat(np.repeat(_logical_mask(shape), scale, 0), scale, 1)
        canvas[r:r+mask.shape[0], c:c+mask.shape[1]][mask] = color
    return {"input": canvas, "output": np.repeat(np.repeat(logical, scale, 0), scale, 1)}


def _check_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return int(value)


def _scale_schedule(rng, ntrain, ntest):
    mode = int(rng.integers(6))
    if mode == 0:
        k = int(rng.choice([2, 3, 4, 5], p=[.4, .3, .25, .05]))
        return [k]*ntrain, [k]*ntest
    if mode in (1, 2):
        a, b = (2, 3) if mode == 1 else (3, 2)
        return [a]*ntrain, [b]*ntest
    if mode in (3, 4):
        train = [2, 3] + [int(rng.choice([2, 3])) for _ in range(ntrain-2)]
        rng.shuffle(train)
        test = [int(rng.choice([2, 3])) for _ in range(ntest)] if mode == 3 else [int(rng.choice([4, 5], p=[.85, .15]))]*ntest
        return train, test
    train = [2, 4] + [int(rng.choice([2, 3, 4])) for _ in range(ntrain-2)]
    rng.shuffle(train)
    return train, [int(rng.choice(sorted(set(train)))) for _ in range(ntest)]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    num_train = _check_count(num_train, "num_train", 2, 10)
    num_test = _check_count(num_test, "num_test", 1, 4)
    rng = np.random.default_rng(seed)
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10)-1)+1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test+1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    train_scales, test_scales = _scale_schedule(rng, num_train, num_test)
    reflection_allowed = bool(rng.random() < .8)
    simple = bool(rng.random() < .2)
    puzzle, seen = {"train": [], "test": []}, set()
    previous_palette = None
    previous_first = None
    for split, scales, regime in (("train", train_scales, "source_train"), ("test", test_scales, "target_test")):
        for index, scale in enumerate(scales):
            palette = rng.permutation(10)
            if previous_palette is not None:
                # Each local role changes, making literal-color rules untenable.
                while any(palette[j] == previous_palette[j] for j in range(3)):
                    palette = np.roll(palette, 1)
            previous_palette = palette.copy()
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            witness = None
            if split == "train":
                if index == 0 and reflection_allowed:
                    witness = "reflection"
                elif index == 1 or not reflection_allowed and index == 0:
                    witness = "rotation"
            for attempt in range(25):
                pair = _make_pair(pair_rng, regime, scale, reflection_allowed, witness, palette, simple)
                info = _recover(pair["input"])[1]
                # Even a two-demonstration puzzle has genuinely different shapes
                # and board geometry, rather than a palette/placement-only change.
                varied = not (split == "train" and index == 1 and previous_first is not None) or (
                    info["shape_classes"] != previous_first["shape_classes"]
                    and info["board_pattern"] != previous_first["board_pattern"])
                if _grid_key(pair["input"]) not in seen and _pair_digest(pair["input"], pair["output"]) not in _SOURCE_DIGESTS and varied:
                    break
            else:
                raise AssertionError("bounded distinct-pair construction exhausted")
            if split == "train" and index == 0:
                previous_first = info
            seen.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape) and np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert _grid_key(inp) not in seen and not np.array_equal(inp, out)
            assert _pair_digest(inp, out) not in _SOURCE_DIGESTS
            seen.add(_grid_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), "input-only reconstruction disagrees"
            assert 2 <= info["piece_count"] <= 9 and 2 <= info["scale"] <= 5
            assert 0 <= info["noise_count"] <= 24 and all(n <= 3 for n in info["noise_sizes"])
            assert 1 <= min(info["areas"]) <= info["max_area"] <= 12
            infos[split].append(info)
    assert not any(x["reflection"] for x in infos["test"]) or any(x["reflection"] for x in infos["train"])
    assert not any(x["rotation"] for x in infos["test"]) or any(x["rotation"] for x in infos["train"])
    assert len({x["shape_classes"] for x in infos["train"]}) >= 2
    assert len({x["board_pattern"] for x in infos["train"]}) >= 2
    for role in ("background", "structural"):
        assert len({x[role] for x in infos["train"]}) >= 2
    # The same geometry-based, color-preserving operation handles every scale
    # schedule. No split-specific dimensions, counts or color dictionary exist.


def _puzzle_key(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test() -> None:
    first = generate_puzzle(918273, 6, 3)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(918273, 6, 3))
    assert _puzzle_key(first) != _puzzle_key(generate_puzzle(918274, 6, 3))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1000+index, *counts)
        assert all(n is None or len(p[s]) == n for s, n in zip(("train", "test"), counts))
    # Rare but supported combinations supplement the broad fixed-seed suite.
    large_scale = generate_puzzle(179)
    assert any(_recover(p["input"])[1]["scale"] == 5 for p in large_scale["train"])
    many_pieces = generate_puzzle(46)
    assert any(_recover(p["input"])[1]["piece_count"] == 8 for p in many_pieces["test"])
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4), (True, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, keys, infos, schedules = set(), set(), set(), [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        keys.add(_puzzle_key(p))
        grouped = {s: [_recover(pair["input"])[1] for pair in p[s]] for s in p}
        schedules.add(tuple(tuple(sorted({i["scale"] for i in grouped[s]})) for s in ("train", "test")))
        assert len({i["shape_classes"] for s in grouped for i in grouped[s]}) >= 2
        for s in ("train", "test"):
            infos.extend((s, i) for i in grouped[s])
        # Predict with only training pairs and the query input, never labels.
        for pair in p["test"]:
            assert np.array_equal(solve(p["train"], pair["input"].copy()), pair["output"])
    assert len(keys) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(schedules) >= 6
    assert {((2,), (3,)), ((3,), (2,))} <= schedules
    assert any(a == b and len(a) == 1 for a, b in schedules)
    assert any(len(a) > 1 and set(b) <= set(a) for a, b in schedules)
    assert any(max(b) >= 4 and set(b).isdisjoint(a) for a, b in schedules)
    all_info = [i for _, i in infos]
    assert {i["scale"] for i in all_info} >= {2, 3, 4, 5}
    assert len({i["board_pattern"] for i in all_info}) >= 250
    assert len({i["input_shape"] for i in all_info}) >= 60
    assert len({i["board_shape"] for i in all_info}) >= 40
    assert len({shape for i in all_info for shape in i["shape_classes"]}) >= 80
    traits = [_shape_traits(shape) for i in all_info for shape in i["shape_classes"]]
    assert all(any(t[name] for t in traits) for name in traits[0])
    assert set(range(1, 10)) <= {area for i in all_info for area in i["areas"]}
    for split in ("train", "test"):
        rows = [i for s, i in infos if s == split]
        assert {i["piece_count"] for i in rows} >= {2, 3, 4, 5, 6}
        for feature in ("reflection", "rotation", "identity", "symmetric", "repeat_non_single", "equal_area_distinct", "shared_noise", "structural_islands"):
            assert any(i[feature] for i in rows), (split, feature)
        assert any(not i["shared_colors"] for i in rows)
        assert any(i["piece_count"] - i["piece_color_count"] >= 2 for i in rows)
        assert len({i["single_count"] for i in rows}) >= 3
        assert any(i["noise_count"] == 0 for i in rows) and max(i["noise_count"] for i in rows) >= 18
        assert {1, 2, 3} <= {n for i in rows for n in i["noise_sizes"]}
        for direction in ("above", "below", "left", "right"):
            assert sum(i[direction] for i in rows) >= 10
        assert any(i["board_shape"][0] < i["board_shape"][1] for i in rows)
        assert any(i["board_shape"][0] > i["board_shape"][1] for i in rows)
        assert any(i["board_shape"] == (6, 6) for i in rows)
    assert sum(i["reflection"] > 0 and i["max_area"] >= 5 and i["shared_colors"] and i["board_shape"][0] != i["board_shape"][1] for i in all_info) >= 10
    # Negative diagnostics for wrong labels, broken macrocells and D4 ambiguity.
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    diagnostic = np.zeros((25, 25), dtype=np.int8)
    diagnostic[:7, :9] = 2
    a = ((1, 1), (2, 1), (3, 1), (3, 2))
    b = ((1, 6), (2, 6), (3, 6), (3, 5))
    for cells in (a, b):
        for r, c in cells:
            diagnostic[r, c] = 0
    for cells, color, left in ((a, 3, 0), (b, 4, 8)):
        mask = np.repeat(np.repeat(_logical_mask(_normalise(cells)), 2, 0), 2, 1)
        diagnostic[10:10+mask.shape[0], left:left+mask.shape[1]][mask] = color
    try:
        _recover(diagnostic)
    except AssertionError:
        pass
    else:
        raise AssertionError("differently colored mirror-equivalent pieces accepted")
    diagnostic[diagnostic == 4] = 3
    recovered, _ = _recover(diagnostic)
    assert recovered.shape == (14, 18)
    diagnostic[10, 0] = 0
    try:
        _recover(diagnostic)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken macrocell accepted")
