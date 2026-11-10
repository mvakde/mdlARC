"""Executable literal implementation of the locked composition.

Four rigid pieces exactly tile a marker-anchored square, after which maximal
background squares are selected. Exact coverage makes that final selection
empty. This semantic limitation is deliberate and documented, not concealed by
inventing gaps or changing a carrier into background. Only NumPy and the Python
standard library are used; generation and solving do no filesystem I/O.
"""
from __future__ import annotations

from hashlib import sha256
from math import isqrt

import numpy as np


_EDGE = ((-1, 0), (0, -1), (0, 1), (1, 0))
_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                   if r or c)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape),
          "Grid dimensions must be 1..30")
    _need(np.issubdtype(g.dtype, np.integer), "Grid cells must be integers")
    _need(bool(np.all((g >= 0) & (g <= 9))), "Grid colors must be 0..9")
    return g.astype(np.int8, copy=False)


def _counts(a, b):
    for value, lo, hi in ((a, 2, 10), (b, 1, 4)):
        _need(isinstance(value, (int, np.integer))
              and not isinstance(value, (bool, np.bool_))
              and lo <= int(value) <= hi, "Illegal example count")
    _need(a > b, "num_train must exceed num_test")
    return int(a), int(b)


def _normal(cells):
    r0 = min(r for r, c, color in cells)
    c0 = min(c for r, c, color in cells)
    return tuple(sorted((r-r0, c-c0, color) for r, c, color in cells))


def _rotate(piece, turns=1):
    result = piece
    for _ in range(turns % 4):
        result = _normal([(-c, r, color) for r, c, color in result])
    return result


def _orientations(piece):
    return tuple(dict.fromkeys(_rotate(piece, k) for k in range(4)))


def _shape(piece):
    return max(r for r, c, color in piece)+1, max(c for r, c, color in piece)+1


def _connected(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in _EDGE:
            p = (r+dr, c+dc)
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    return not remaining


def _read_input(g):
    colors, counts = np.unique(g, return_counts=True)
    _need(len(colors) == 6, "Require background, marker and four carriers")
    winners = np.flatnonzero(counts == counts.max())
    _need(len(winners) == 1, "Background must be uniquely most frequent")
    background = int(colors[winners[0]])
    markers = [int(c) for c, n in zip(colors, counts) if n == 1 and c != background]
    _need(len(markers) == 1, "Require a unique singleton marker")
    marker = markers[0]
    area = int(np.sum(g != background))
    side = isqrt(area)
    _need(side*side == area, "Foreground area must be a square")
    seen = np.zeros(g.shape, dtype=bool)
    pieces = []
    for r, c in np.argwhere(g != background):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc, int(g[rr, cc])))
            for dr, dc in _NEIGHBORS:
                nr, nc = rr+dr, cc+dc
                if (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]
                        and not seen[nr, nc] and g[nr, nc] != background):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        pieces.append(_normal(cells))
    _need(len(pieces) == 4, "Require four separated rigid objects")
    carriers = []
    for p in pieces:
        values = {color for r, c, color in p} - {marker}
        _need(len(p) >= 3 and len(values) == 1,
              "Each object needs one carrier and at least three cells")
        carriers.append(next(iter(values)))
    _need(len(set(carriers)) == 4, "Carrier colors must be distinct")
    return background, marker, side, pieces


def _assemblies(pieces, side, marker, limit=2):
    """Input-derived exhaustive exact cover; stop after two colored answers."""
    if sum(map(len, pieces)) != side*side:
        return []
    anchors = [i for i, p in enumerate(pieces) if any(v == marker for r, c, v in p)]
    if len(anchors) != 1:
        return []
    anchor = anchors[0]
    options = []
    for i, p in enumerate(pieces):
        placements = {}
        for oriented in _orientations(p):
            h, w = _shape(oriented)
            if h > side or w > side:
                continue
            if i == anchor:
                if not any((r, c, v) == (0, 0, marker) for r, c, v in oriented):
                    continue
                offsets = (0,)
            else:
                offsets = tuple(r*side+c for r in range(side-h+1)
                                for c in range(side-w+1))
            cells = tuple((r*side+c, color) for r, c, color in oriented)
            base = sum(1 << pos for pos, color in cells)
            for offset in offsets:
                mask = base << offset
                placements.setdefault(mask, (mask, offset, cells))
        if not placements:
            return []
        options.append(tuple(placements.values()))
    full = (1 << (side*side))-1
    placed = [None]*len(pieces)
    answers = []

    def search(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied == full:
                out = np.empty((side, side), dtype=np.int8)
                for mask, offset, cells in placed:
                    for pos, color in cells:
                        out.flat[pos+offset] = color
                answers.append(out)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in options[i]:
                mask = candidate[0]
                if mask & first and not mask & occupied:
                    placed[i] = candidate
                    search(occupied | mask, rest)
                    if len(answers) >= limit:
                        return
            placed[i] = None

    for candidate in options[anchor]:
        placed[anchor] = candidate
        search(candidate[0], tuple(i for i in range(len(pieces)) if i != anchor))
        if len(answers) >= limit:
            break
    return answers


def _highlight_squares(assembly, background, highlight):
    """A's local full-side maximality, then larger-first overlap suppression.

    Operates on the assembled square only. In the locked family its mask is
    empty. Keeping the actual detector makes this limitation inspectable.
    """
    mask = assembly == background
    out = assembly.copy()
    if not mask.any():
        return out
    h, w = mask.shape
    selected = np.zeros(mask.shape, dtype=bool)
    for side in range(min(h, w), 1, -1):
        for r in range(h-side+1):
            for c in range(w-side+1):
                r1, c1 = r+side, c+side
                if not mask[r:r1, c:c1].all():
                    continue
                if ((r and mask[r-1, c:c1].all())
                        or (r1 < h and mask[r1, c:c1].all())
                        or (c and mask[r:r1, c-1].all())
                        or (c1 < w and mask[r:r1, c1].all())):
                    continue
                if selected[r:r1, c:c1].any():
                    continue
                selected[r:r1, c:c1] = True
    out[selected] = highlight
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    background, marker, side, pieces = _read_input(g)
    answers = _assemblies(pieces, side, marker)
    _need(len(answers) == 1, "Require exactly one colored anchored assembly")
    highlight = min(set(range(10)) - set(map(int, np.unique(g))))
    return _highlight_squares(answers[0], background, highlight)


def _growth_partition(rng, side):
    if rng.random() < .5:
        starts = [0, side-1, side*(side-1), side*side-1]
    else:
        starts = [0] + list(map(int, rng.choice(np.arange(1, side*side), 3, replace=False)))
    labels = np.full((side, side), -1, dtype=np.int8)
    adjacent = []
    for r in range(side):
        for c in range(side):
            adjacent.append([nr*side+nc for dr, dc in _EDGE
                             if 0 <= (nr := r+dr) < side and 0 <= (nc := c+dc) < side])
    for i, pos in enumerate(starts):
        labels.flat[pos] = i
    frontiers = [{p for p in adjacent[pos] if labels.flat[p] < 0} for pos in starts]
    areas = np.ones(4)
    weights = np.exp(rng.uniform(-1.0, 1.0, 4))
    for _ in range(side*side-4):
        eligible = [i for i in range(4) if frontiers[i]]
        probabilities = np.array([weights[i]/areas[i]**.35 for i in eligible])
        probabilities /= probabilities.sum()
        i = int(rng.choice(eligible, p=probabilities))
        choices = sorted(frontiers[i])
        pos = choices[int(rng.integers(len(choices)))]
        labels.flat[pos] = i
        areas[i] += 1
        for frontier in frontiers:
            frontier.discard(pos)
        frontiers[i].update(p for p in adjacent[pos] if labels.flat[p] < 0)
    return labels if min(areas) >= 3 else None


def _frame_partition(rng, side):
    """A connected frame with three distinct-area, separated rigid inserts.

    The frame's complete square boundary anchors it. An insert cannot cross a
    frame corridor, and distinct cavity areas force the remaining assignments.
    The cavities are filled by colored inserts, not left as background.
    """
    left_w = int(rng.integers(2, side-4))
    top_h = int(rng.integers(2, side-4))
    right_c, bottom_r = left_w+2, top_h+2
    regions = [(1, 1, side-2, left_w),
               (1, right_c, top_h, side-right_c-1),
               (bottom_r, right_c, side-bottom_r-1, side-right_c-1)]
    for _ in range(64):
        labels = np.zeros((side, side), dtype=np.int8)
        areas = set()
        good = True
        for i, (r, c, height, width) in enumerate(regions, 1):
            h, w = int(rng.integers(1, height+1)), int(rng.integers(1, width+1))
            lengths = rng.integers(1, w+1, h)
            lengths[0] = w
            mask = np.arange(w)[None, :] < lengths[:, None]
            if rng.random() < .5:
                mask = mask[::-1]
            if rng.random() < .5:
                mask = mask[:, ::-1]
            area = int(mask.sum())
            if area < 3 or area in areas:
                good = False
                break
            areas.add(area)
            rr = r+int(rng.integers(height-h+1))
            cc = c+int(rng.integers(width-w+1))
            labels[rr:rr+h, cc:cc+w][mask] = i
        if good and _connected(labels == 0):
            return labels
    # Guaranteed bounded safety net at every generated side (at least eight).
    labels = np.zeros((side, side), dtype=np.int8)
    labels[1:6, 1] = 1
    labels[1, right_c:right_c+2] = 2
    labels[2, right_c] = 2
    labels[bottom_r:bottom_r+2, right_c:right_c+2] = 3
    return labels


def _materialize(labels, carriers, marker):
    output = np.asarray(carriers, dtype=np.int8)[labels]
    output[0, 0] = marker
    pieces = [_normal([(int(r), int(c), int(output[r, c]))
                       for r, c in np.argwhere(labels == i)]) for i in range(4)]
    return output, pieces


def _scatter(rng, pieces, background, anchor_rotation):
    rotations = [anchor_rotation] + list(map(int, rng.integers(1, 4, 3)))
    oriented = [_rotate(p, k) for p, k in zip(pieces, rotations)]
    order = list(map(int, rng.permutation(4)))
    layouts = []
    for groups in ((2, 2), (1, 3), (3, 1), (4,), (1, 1, 1, 1)):
        for gap in (1, 2, 3):
            rows, start = [], 0
            for count in groups:
                indices = order[start:start+count]
                dimensions = [_shape(oriented[i]) for i in indices]
                rows.append((indices, max(h for h, w in dimensions),
                             sum(w for h, w in dimensions)+gap*(count-1)))
                start += count
            h = sum(row[1] for row in rows)+gap*(len(rows)-1)
            w = max(row[2] for row in rows)
            if h <= 28 and w <= 28:
                layouts.append((rows, gap, h, w))
    rows, gap, h, w = layouts[int(rng.integers(len(layouts)))]
    H = h+int(rng.integers(min(4, 30-h)+1))
    W = w+int(rng.integers(min(4, 30-w)+1))
    area = sum(map(len, pieces))
    colors = [v for p in pieces for r, c, v in p]
    largest = max(colors.count(v) for v in set(colors))
    while H*W-area <= largest:
        if H < 30 and (H <= W or W == 30):
            H += 1
        else:
            W += 1
    canvas = np.full((H, W), background, dtype=np.int8)
    top = int(rng.integers(H-h+1))
    for indices, row_h, row_w in rows:
        left = int(rng.integers(W-row_w+1))
        for i in indices:
            ph, pw = _shape(oriented[i])
            pr = top+int(rng.integers(row_h-ph+1))
            for r, c, color in oriented[i]:
                canvas[pr+r, left+c] = color
            left += pw+gap
        top += row_h+gap
    return canvas


def _make_pair(rng, side, background, marker, carriers, rotation, growth):
    if growth:
        for _ in range(12):
            labels = _growth_partition(rng, side)
            if labels is None:
                continue
            output, pieces = _materialize(labels, carriers, marker)
            answers = _assemblies(pieces, side, marker)
            if len(answers) == 1 and np.array_equal(answers[0], output):
                break
        else:
            growth = False
    if not growth:
        labels = _frame_partition(rng, side)
        output, pieces = _materialize(labels, carriers, marker)
        answers = _assemblies(pieces, side, marker)
        _need(len(answers) == 1 and np.array_equal(answers[0], output),
              "Frame construction must have one answer")
    before = _scatter(rng, pieces, background, rotation)
    highlight = min(set(range(10)) - set(map(int, np.unique(before))))
    return {"input": before, "output": _highlight_squares(output, background, highlight)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _need(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
          and seed >= 0, "Seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    background, marker, carriers = palette[0], palette[1], palette[2:6]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train":
                side = (8, 10)[i] if i < 2 else int(rng.integers(8, 11))
                rotation = (0, 1)[i] if i < 2 else int(rng.choice([0, 1, 3]))
            else:
                side, rotation = int(rng.integers(11, 14)), 2
            for _ in range(8):
                pair = _make_pair(rng, side, background, marker, carriers, rotation,
                                  growth=bool((i % 2 == 0) if split == "train" else rng.random() < .5))
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded distinct-input construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, palettes, train_sides, test_sides = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            before, after = _grid(pair["input"]), _grid(pair["output"])
            key = (before.shape, before.tobytes())
            _need(key not in seen, "Duplicate input")
            seen.add(key)
            background, marker, side, pieces = _read_input(before)
            palettes.add((background, marker, tuple(sorted(set(map(int, np.unique(before)))
                                                          - {background, marker}))))
            _need(after.shape == (side, side) and after[0, 0] == marker,
                  "Incorrect output size or anchor")
            _need(not np.any(after == background), "Gapless assembly cannot retain background")
            for color in np.unique(before):
                if color != background:
                    _need(np.sum(before == color) == np.sum(after == color), "Piece color count changed")
            _need(np.array_equal(solve(before), after), "Output disagrees with input-derived solve")
            (train_sides if split == "train" else test_sides).append(side)
    _need(len(palettes) == 1, "Palette conventions must be fixed within the puzzle")
    _need(len(set(train_sides)) >= 2 and min(test_sides) > max(train_sides),
          "Training must establish size variation before larger tests")


def self_test():
    # Check that the A implementation itself rejects square subwindows inside
    # nonsquare runs; this does not pretend such a mask survives B's exact cover.
    probe = np.full((6, 10), 5, dtype=np.int8)
    probe[1:3, 1:3] = 0
    probe[1:3, 5:8] = 0
    probe[4, 2] = 0
    expected = probe.copy()
    expected[1:3, 1:3] = 1
    assert np.array_equal(_highlight_squares(probe, 0, 1), expected)
    signatures, structures, shapes, sides, backgrounds = set(), set(), set(), set(), set()
    frame_count = nonframe_count = pair_count = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if first is None:
            first = puzzle
        digest = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before, after = pair["input"], pair["output"]
                digest.update(bytes(before.shape)+before.tobytes()+after.tobytes())
                background, marker, side, pieces = _read_input(before)
                backgrounds.add(background)
                sides.add(side)
                shapes.add(before.shape)
                roles = {}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in after.flat)
                structures.add((after.shape, normalized))
                marked_piece = next(p for p in pieces if any(v == marker for r, c, v in p))
                anchor_color = next(color for r, c, color in marked_piece if color != marker)
                boundary = np.concatenate((after[0], after[-1], after[:, 0], after[:, -1]))
                frame = bool(np.all((boundary == anchor_color) | (boundary == marker)))
                frame_count += frame
                nonframe_count += not frame
                assert not np.any(after == background)
                assert np.array_equal(_highlight_squares(after, background, 0), after)
                pair_count += 1
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(structures) >= 100 and len(shapes) >= 20
    assert {8, 9, 10, 11, 12, 13} <= sides and len(backgrounds) >= 8
    assert frame_count >= 20 and nonframe_count >= 20
    repeated = generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in ("train", "test")
               for p, q in zip(first[s], repeated[s]) for k in ("input", "output"))
    example = first["train"][0]
    assert np.array_equal(solve(np.rot90(example["input"])), example["output"])
    color_map = np.arange(9, -1, -1, dtype=np.int8)
    assert np.array_equal(solve(color_map[example["input"]]), color_map[example["output"]])
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]]
           for s in ("train", "test")}
    background = _read_input(bad["test"][0]["input"])[0]
    bad["test"][0]["output"][0, 0] = background
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Validator accepted a damaged output")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "distinct_output_structures": len(structures), "input_shapes": len(shapes),
            "output_sides": sorted(sides), "background_colors": len(backgrounds),
            "frame_pairs": frame_count, "nonframe_pairs": nonframe_count,
            "a_changes_assembled_answer": False,
            "limitation": "Gapless coverage makes the locked A operation vacuous"}
