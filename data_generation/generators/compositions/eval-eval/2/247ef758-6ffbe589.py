"""Rotation loom: Cartesian addresses carry additive local rotation phases.

Standard library + NumPy only. No files, examples, seeds or construction state
are consulted by solve(). Generation uses array rotations; the visible-input
checker uses integer offset arithmetic. No execution traces are stored.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_ALTERNATIVES = ("no_rotation", "row_only", "column_only", "product",
                 "difference", "maximum", "zipped", "single_copy", "global_angle")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not 2 <= a <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(b) or not 1 <= b <= 4 or a <= b:
        raise ValueError("num_test must be in 1..4 and smaller than num_train")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid of colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    todo.append(p)
        yield cells


def _rectangle(cells):
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    if r1-r0 < 4 or c1-c0 < 4:
        return None
    perimeter = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                 if r in (r0, r1) or c in (c0, c1)}
    return (r0, c0, r1, c1) if cells == perimeter else None


def _parse(value):
    """Read the field, cards and bars entirely from their visible geometry."""
    inp = _grid(value)
    values, amounts = np.unique(inp, return_counts=True)
    if np.sum(amounts == amounts.max()) != 1:
        raise ValueError("background must be the unique most frequent color")
    bg = int(values[amounts.argmax()])
    rects = []
    for color in values:
        if color == bg:
            continue
        for cells in _components(inp == color):
            box = _rectangle(cells)
            if box is not None:
                rects.append((int(color), box, cells))
    boards = []
    for color, box, cells in rects:
        r0, c0, r1, c1 = box
        if (r1-r0-1 >= 9 and c1-c0-1 >= 9
                and np.all(inp[r0+1:r1, c0+1:c1] == bg)):
            boards.append((color, box, cells))
    if len(boards) != 1:
        raise ValueError("there must be one large empty rectangular field")
    frame, (r0, c0, r1, c1), board_cells = boards[0]
    covered = set(board_cells)
    templates = {}
    frame_cells = set(map(tuple, np.argwhere(inp == frame).tolist()))
    for color, box, cells in rects:
        if color != frame or cells == board_cells:
            continue
        a, b, e, f = box
        side = e-a-1
        if side not in (3, 5) or f-b-1 != side:
            raise ValueError("stencil cards must have interior side 3 or 5")
        content = inp[a+1:e, b+1:f]
        colors = [int(c) for c in np.unique(content) if c != bg]
        if len(colors) != 1 or colors[0] == frame or colors[0] in templates:
            raise ValueError("one nonempty monochromatic card per stencil color")
        color = colors[0]
        coords = np.argwhere(content == color)
        lo, hi = coords.min(0), coords.max(0)
        shape = tuple(map(int, hi-lo+1))
        if any(n not in (1, 3, 5) for n in shape):
            raise ValueError("stencil bounding boxes must be odd and no larger than 5")
        center = (lo+hi)//2
        offsets = tuple((int(y-center[0]), int(x-center[1])) for y, x in coords)
        mask = content[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == color
        templates[color] = {"offsets": offsets, "mask": mask.copy(), "shape": shape,
                            "card": box}
        covered.update(cells)
        covered.update((a+1+int(y), b+1+int(x)) for y, x in coords)
    if not 1 <= len(templates) <= 7 or not frame_cells <= covered:
        raise ValueError("all structural cells must belong to field or stencil cards")
    rows, cols, bars = {}, {}, []
    used_rows, used_cols = set(), set()
    h, w = inp.shape

    def read_bar(y, x, dy, dx, axis, position, side):
        if not (0 <= y < h and 0 <= x < w):
            return
        color = int(inp[y, x])
        if color in (bg, frame):
            return
        if color not in templates:
            raise ValueError("a bar has no matching stencil")
        cells = []
        while 0 <= y < h and 0 <= x < w and inp[y, x] == color:
            cells.append((y, x))
            y, x = y+dy, x+dx
        if not 1 <= len(cells) <= 7:
            raise ValueError("unary bars have lengths 1..7")
        table, used = (rows, used_rows) if axis == 0 else (cols, used_cols)
        if position in used:
            raise ValueError("an axis position may have only one bar")
        used.add(position)
        table.setdefault(color, []).append((position, len(cells)))
        bars.append({"color": color, "length": len(cells), "axis": axis,
                     "side": side, "cells": tuple(cells)})
        covered.update(cells)

    for y in range(r0+1, r1):
        for gap, x, dx, side in ((c0-1, c0-2, -1, "left"),
                                 (c1+1, c1+2, 1, "right")):
            if 0 <= gap < w and inp[y, gap] != bg:
                raise ValueError("bars and field require a one-cell empty gap")
            read_bar(y, x, 0, dx, 0, y-r0-1, side)
    for x in range(c0+1, c1):
        for gap, y, dy, side in ((r0-1, r0-2, -1, "top"),
                                 (r1+1, r1+2, 1, "bottom")):
            if 0 <= gap < h and inp[gap, x] != bg:
                raise ValueError("bars and field require a one-cell empty gap")
            read_bar(y, x, dy, 0, 1, x-c0-1, side)
    actual = set(map(tuple, np.argwhere(inp != bg).tolist()))
    if covered != actual:
        raise ValueError("every non-background cell must be a frame, stencil or aligned bar")
    for table in (rows, cols):
        for entries in table.values():
            entries.sort()
    return {"bg": bg, "frame": frame, "box": (r0, c0, r1, c1),
            "shape": (r1-r0-1, c1-c0-1), "templates": templates,
            "rows": rows, "cols": cols, "bars": bars}


def _render(info, alternative=None):
    """Coordinate-based predictor, independent of generation's np.rot90 paint."""
    out = np.full(info["shape"], info["bg"], dtype=np.int8)
    for color, template in info["templates"].items():
        rows, cols = info["rows"].get(color, []), info["cols"].get(color, [])
        requests = [(r, kr, c, kc) for r, kr in rows for c, kc in cols]
        if alternative == "zipped":
            requests = [(r, kr, c, kc) for (r, kr), (c, kc) in zip(rows, cols)]
        elif alternative == "single_copy":
            requests = requests[:1]
        for r, kr, c, kc in requests:
            turns = kr+kc
            if alternative == "no_rotation":
                turns = 0
            elif alternative == "row_only":
                turns = kr
            elif alternative == "column_only":
                turns = kc
            elif alternative == "product":
                turns = kr*kc
            elif alternative == "difference":
                turns = kr-kc
            elif alternative == "maximum":
                turns = max(kr, kc)
            elif alternative == "global_angle":
                turns = rows[0][1]+cols[0][1]
            for dr, dc in template["offsets"]:
                for _ in range(turns % 4):
                    dr, dc = dc, -dr
                y, x = r+dr, c+dc
                if not (0 <= y < out.shape[0] and 0 <= x < out.shape[1]):
                    raise ValueError("stencil copy leaves the field")
                if out[y, x] not in (info["bg"], color):
                    raise ValueError("different output colors conflict")
                out[y, x] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the woven interior using only the provided input."""
    return _render(_parse(input_grid))


def _difference(info, expected, alternative):
    try:
        return int(np.count_nonzero(_render(info, alternative) != expected))
    except ValueError:
        # An alternative producing clipping/conflicts is also contradicted.
        return int(expected.size)


def _asymmetric(mask):
    return not np.array_equal(mask, np.rot90(mask, 2))


def _sample_mask(rng, h, w, asymmetric=False):
    for _ in range(32):
        mode = int(rng.integers(6))
        mask = np.zeros((h, w), dtype=bool)
        if mode == 0 or min(h, w) == 1:
            mask[:] = True
        elif mode == 1:
            mask[[0, -1], :] = True
            mask[:, [0, -1]] = True
        elif mode == 2:
            for r in range(h):
                mask[r, round(r*(w-1)/(h-1))] = True
            for c in range(w):
                mask[round(c*(h-1)/(w-1)), c] = True
        elif mode == 3:
            mask[h//2, :] = True
            mask[:, w//2] = True
            mask |= rng.random((h, w)) < float(rng.uniform(0, .35))
        else:
            mask = rng.random((h, w)) < float(rng.uniform(.15, .9))
            # Extent witnesses impose no center or connectivity requirement.
            mask[0, int(rng.integers(w))] = True
            mask[-1, int(rng.integers(w))] = True
            mask[int(rng.integers(h)), 0] = True
            mask[int(rng.integers(h)), -1] = True
        if rng.random() < .5:
            mask = mask[::-1]
        if rng.random() < .5:
            mask = mask[:, ::-1]
        if h == w:
            mask = np.rot90(mask, int(rng.integers(4)))
        if not asymmetric or _asymmetric(mask):
            return mask.copy()
    if min(h, w) == 1:
        raise ValueError("a tight solid line cannot be half-turn asymmetric")
    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = mask[0, 1] = mask[-1, -1] = True
    return mask


def _assemble(rng, height, width, masks, rows, cols):
    """Build from sampled records; output construction does not call solve."""
    palette = list(map(int, rng.permutation(10)))
    bg, frame = palette[:2]
    colors = palette[2:2+len(masks)]
    max_row = max((k for entries in rows for _, k in entries), default=0)
    max_col = max((k for entries in cols for _, k in entries), default=0)
    sides = [max(3, max(m.shape)) for m in masks]
    card_width = sum(s+2 for s in sides)+len(sides)-1
    card_height = max(sides)+2
    base_height = height+max_col+card_height+4
    if base_height > 30:
        raise ValueError("vertical packing would exceed 30")
    top_pad = int(rng.integers(min(2, 30-base_height)+1))
    r0, c0 = max_col+1+top_pad, max_row+1
    r1, c1 = r0+height+1, c0+width+1
    canvas_h = r1+2+card_height
    canvas_w = max(c1+1, card_width)
    if canvas_w > 30:
        raise ValueError("horizontal packing would exceed 30")
    left_pad = int(rng.integers(min(2, 30-canvas_w)+1))
    c0, c1, canvas_w = c0+left_pad, c1+left_pad, canvas_w+left_pad
    inp = np.full((canvas_h, canvas_w), bg, dtype=np.int8)
    inp[r0, c0:c1+1] = inp[r1, c0:c1+1] = frame
    inp[r0:r1+1, c0] = inp[r0:r1+1, c1] = frame
    card_left = int(rng.integers(canvas_w-card_width+1))
    for i in rng.permutation(len(masks)):
        i = int(i)
        inner, color = sides[i], colors[i]
        outer = inner+2
        a, b = r1+2, card_left
        inp[a, b:b+outer] = inp[a+outer-1, b:b+outer] = frame
        inp[a:a+outer, b] = inp[a:a+outer, b+outer-1] = frame
        mask = masks[i]
        y, x = a+1+(inner-mask.shape[0])//2, b+1+(inner-mask.shape[1])//2
        view = inp[y:y+mask.shape[0], x:x+mask.shape[1]]
        view[mask] = color
        card_left += outer+1
        for r, k in rows[i]:
            inp[r0+1+r, c0-1-k:c0-1] = color
        for c, k in cols[i]:
            inp[r0-1-k:r0-1, c0+1+c] = color
    out = np.full((height, width), bg, dtype=np.int8)
    for mask, color, rs, cs in zip(masks, colors, rows, cols):
        for r, kr in rs:
            for c, kc in cs:
                rotated = np.rot90(mask, -(kr+kc) % 4)
                hh, ww = rotated.shape
                if not (hh//2 <= r < height-hh//2 and ww//2 <= c < width-ww//2):
                    raise ValueError("generated stencil would be clipped")
                view = out[r-hh//2:r+hh//2+1, c-ww//2:c+ww//2+1]
                if np.any(rotated & (view != bg) & (view != color)):
                    raise ValueError("generated colors would collide")
                view[rotated] = color
    turn = int(rng.integers(4))
    return {"input": np.rot90(inp, turn).copy(), "output": np.rot90(out, turn).copy()}


def _witness(rng, second=False, test=False):
    height = int(rng.integers(11, 14))
    if not second:
        width = int(rng.integers(11, 15))
        masks = [_sample_mask(rng, 3, 3, True), _sample_mask(rng, 3, 3, True)]
        rr, cc = list(map(int, rng.permutation([1, 2]))), list(map(int, rng.permutation([1, 3])))
        rows = [[(2, rr[0]), (height-3, rr[1])], [(height//2, int(rng.integers(6, 8)))]]
        cols = [[(2, cc[0]), (width-3, cc[1])], [(width//2, int(rng.integers(1, 4)))]]
    else:
        width = int(rng.integers(13, 18))
        masks = [_sample_mask(rng, 5, 5, True), _sample_mask(rng, 3, 3, True),
                 _sample_mask(rng, 3, 3)]
        lengths = [6, 7] if test else [1, 2]
        rng.shuffle(lengths)
        rows = [[(2, lengths[0]), (height-3, lengths[1])], [(height//2, 4)], []]
        cols = [[(2, 2)], [(width-3, 5)], []]
        if rng.random() < .5:
            free = [r for r in range(height) if r not in (2, height-3, height//2)]
            rows[2] = [(int(rng.choice(free)), int(rng.integers(1, 6)))]
        else:
            free = [c for c in range(width) if c not in (2, width-3)]
            cols[2] = [(int(rng.choice(free)), int(rng.integers(1, 6)))]
    return _assemble(rng, height, width, masks, rows, cols)


def _random_pair(rng, test=False):
    shapes = [(3, 3), (5, 5), (1, 3), (3, 1), (3, 5), (5, 3)]
    weights = np.array([5, 3, 1, 1, 2, 2], dtype=float)
    weights /= weights.sum()
    for _ in range(80):
        count = int(rng.integers(2, 4))
        dims = [shapes[int(rng.choice(len(shapes), p=weights))] for _ in range(count)]
        if test:
            dims[0] = (5, 5)
        masks = [_sample_mask(rng, *shape, asymmetric=(test and i == 0))
                 for i, shape in enumerate(dims)]
        height, width = int(rng.integers(9, 14)), int(rng.integers(10, 20))
        card_height = max(max(3, max(m.shape))+2 for m in masks)
        col_cap = min(7, 30-height-card_height-4)
        if col_cap < 3:
            continue
        states = list(rng.choice(["A", "R", "C", "N"], count, p=[.72, .10, .10, .08]))
        primary = 0 if test else int(rng.integers(count))
        states[primary] = "A"
        used_rows, used_cols = set(), set()
        rows, cols = [[] for _ in masks], [[] for _ in masks]
        feasible = True
        for i in rng.permutation(count):
            i = int(i)
            active = states[i] == "A"
            sizes = [int(rng.choice([1, 2, 3], p=[.60, .34, .06])) for _ in range(2)]
            if i == primary:
                sizes[int(rng.integers(2))] = 2
            for axis, (n, used, table) in enumerate(((height, used_rows, rows), (width, used_cols, cols))):
                if not active and states[i] != ("R" if axis == 0 else "C"):
                    continue
                rad = max(masks[i].shape)//2 if active else 0
                choices = [v for v in range(rad, n-rad) if v not in used]
                amount = sizes[axis] if active else 1
                if len(choices) < amount:
                    feasible = False
                    break
                positions = sorted(map(int, rng.choice(choices, amount, replace=False)))
                cap = 7 if axis == 0 else col_cap
                if active and max(masks[i].shape) == 5 and not test:
                    cap = min(cap, 5)
                lengths = list(map(int, rng.integers(1, cap+1, size=amount)))
                if test and i == primary and axis == 0:
                    lengths[int(rng.integers(amount))] = int(rng.integers(6, 8))
                table[i] = list(zip(positions, lengths))
                used.update(positions)
            if not feasible:
                break
        if not feasible:
            continue
        try:
            pair = _assemble(rng, height, width, masks, rows, cols)
            info = _parse(pair["input"])
            # Require an observable role for both parent mechanisms in each pair.
            if not all(_difference(info, pair["output"], v) for v in ("no_rotation", "single_copy")):
                continue
            return pair
        except ValueError:
            continue
    # Finite, directly constructible fallback with the same rule and shift.
    return _witness(rng, second=test, test=test)


def _distinct(pair, seen):
    """Twenty bijections suffice against at most thirteen earlier inputs."""
    original = pair
    for j in range(20):
        mapping = np.arange(10, dtype=np.int8)
        if j >= 10:
            mapping = mapping[::-1]
        mapping = (mapping+j % 10) % 10
        candidate = {role: mapping[a] for role, a in original.items()}
        key = _key(candidate["input"])
        if key not in seen:
            seen.add(key)
            return candidate
    raise AssertionError("distinct recoloring bound exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    seen = set()
    train = [_distinct(_witness(rng), seen), _distinct(_witness(rng, second=True), seen)]
    train.extend(_distinct(_random_pair(rng), seen) for _ in range(num_train-2))
    tests = [_distinct(_random_pair(rng, test=True), seen) for _ in range(num_test)]
    puzzle = {"train": train, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def _active(info):
    return [(color, t, info["rows"].get(color, []), info["cols"].get(color, []))
            for color, t in info["templates"].items()
            if info["rows"].get(color) and info["cols"].get(color)]


def _long_large(info):
    return any(t["shape"] == (5, 5) and max(k for _, k in rs+cs) >= 6
               for _, t, rs, cs in _active(info))


def validate_generated_puzzle(puzzle):
    """Check visible family invariants, semantic answers and training witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    defeated = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(inp)
            assert out.shape == info["shape"], "output must be the field interior"
            assert np.array_equal(_render(info), out), "wrong phase, mask, position or crop"
            assert _difference(info, out, "no_rotation"), "B must affect every answer"
            assert _difference(info, out, "single_copy"), "A replication must affect every answer"
            infos[split].append(info)
            if split == "train":
                defeated.update(v for v in _ALTERNATIVES if _difference(info, out, v))
    assert defeated == set(_ALTERNATIVES), "training leaves a named shortcut unchallenged"
    active_train = [entry for info in infos["train"] for entry in _active(info)]
    assert any(len(rs) >= 2 and len(cs) >= 2 and _asymmetric(t["mask"])
               and {(kr+kc) % 4 for _, kr in rs for _, kc in cs} == set(range(4))
               for _, t, rs, cs in active_train), "teach all local angles and the full product"
    assert any(t["shape"] == (5, 5) and _asymmetric(t["mask"])
               and 2 in [k for _, k in rs] and 2 in [k for _, k in cs]
               for _, t, rs, cs in active_train), "teach large masks and an unchanged full cycle"
    assert any(max(t["shape"]) <= 3 and max(k for _, k in rs+cs) >= 6
               for _, t, rs, cs in active_train), "teach long counters before their recombination"
    assert any(4 in [k for _, k in rs+cs] for _, _, rs, cs in active_train)
    assert any(len(_active(info)) < len(info["templates"]) for info in infos["train"])
    assert not any(_long_large(info) for info in infos["train"])
    assert all(_long_large(info) for info in infos["test"]), "test must recombine known size/count factors"


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _hand_regression():
    """Literal geometry/answer, including mixed sides not sampled by assembly.

    A 3x5 stencil has offsets (-1,-2),(-1,-1),(0,-1),(0,1),(1,2).
    Row counts 2,4 and column counts 5,1 give turns 3,3 / 1,1.
    The listed answer cells were calculated directly for this fixture.
    """
    inp = np.zeros((30, 30), dtype=np.int8)
    inp[8, 8:19] = inp[18, 8:19] = 9
    inp[8:19, 8] = inp[8:19, 18] = 9
    # A seven-cell card below and left of the field.
    inp[20, :7] = inp[26, :7] = 9
    inp[20:27, 0] = inp[20:27, 6] = 9
    for y, x in ((22, 1), (22, 2), (23, 2), (23, 4), (24, 5)):
        inp[y, x] = 2
    # An inactive card below and right; its singleton must not appear.
    inp[23, 23:28] = inp[27, 23:28] = 9
    inp[23:28, 23] = inp[23:28, 27] = 9
    inp[25, 25] = 3
    inp[11, 5:7] = 2                 # Left row 2: two turns.
    inp[15, 20:24] = 2               # Right row 6: four turns.
    inp[2:7, 11] = 2                 # Top column 2: five turns.
    inp[20, 15] = 2                  # Bottom column 6: one turn.
    expected = np.zeros((9, 9), dtype=np.int8)
    cells = ((4,1),(3,1),(3,2),(1,2),(0,3),
             (4,5),(3,5),(3,6),(1,6),(0,7),
             (4,3),(5,3),(5,2),(7,2),(8,1),
             (4,7),(5,7),(5,6),(7,6),(8,5))
    for p in cells:
        expected[p] = 2
    before = inp.copy()
    assert np.array_equal(solve(inp), expected), "literal rectangular/mixed-side regression"
    assert np.array_equal(inp, before), "solve must not mutate the input"
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(expected, turn))


def self_test():
    """Bounded 42-seed sweep; retains hashes/features rather than puzzle payloads."""
    _hand_regression()
    signatures, dimensions, masks, backgrounds, bar_sides, products = set(), set(), set(), set(), set(), set()
    mask_shapes, counts, structures = set(), set(), set()
    cases = merged = hollow = symmetric = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _parse(inp)
                assert np.array_equal(solve(inp.copy()), out)
                cases += 1
                dimensions.add(inp.shape)
                backgrounds.add(info["bg"])
                counts.add(len(info["templates"]))
                bar_sides.update(b["side"] for b in info["bars"])
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape)+normalized).hexdigest())
                for color, t, rs, cs in _active(info):
                    mask_shapes.add(t["shape"])
                    masks.add((t["shape"], t["mask"].tobytes()))
                    products.add((len(rs), len(cs)))
                    hollow += not t["mask"][t["shape"][0]//2, t["shape"][1]//2]
                    symmetric += not _asymmetric(t["mask"])
                    # Area below summed template area proves union/merging.
                    merged += int(np.sum(out == color)) < len(t["offsets"])*len(rs)*len(cs)
        if seed in (0, 17, 41):
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
        if seed == 0:
            pair = puzzle["train"][0]
            for turn in range(4):
                assert np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn))
            for shift in range(10):
                assert np.array_equal(solve((pair["input"]+shift) % 10), (pair["output"]+shift) % 10)
            for variant in _ALTERNATIVES:
                info = _parse(pair["input"])
                wrong = _render(info, variant)
                assert not np.array_equal(wrong, pair["output"]), variant
                bad = {s: [dict(x) for x in puzzle[s]] for s in ("train", "test")}
                bad["train"][0] = {"input": pair["input"], "output": wrong}
                try:
                    validate_generated_puzzle(bad)
                except (AssertionError, ValueError):
                    pass
                else:
                    raise AssertionError("accepted a shortcut output")
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(dimensions) >= 40 and len(masks) >= 60
    assert backgrounds == set(range(10)) and bar_sides == {"top", "bottom", "left", "right"}
    assert counts == {2, 3}
    assert {(3, 3), (5, 5), (1, 3), (3, 1), (3, 5), (5, 3)} <= mask_shapes
    assert {(1, 1), (2, 1), (1, 2), (2, 2)} <= products
    assert hollow >= 5 and symmetric >= 5 and merged >= 1
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": cases, "distinct_puzzles": len(signatures),
            "input_shapes": len(dimensions), "normalized_structures": len(structures),
            "active_masks": len(masks), "active_mask_shapes": sorted(mask_shapes),
            "coordinate_products": sorted(products), "backgrounds": sorted(backgrounds),
            "merged_color_layers": merged, "hollow_centers": hollow,
            "symmetric_layers": symmetric, "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
