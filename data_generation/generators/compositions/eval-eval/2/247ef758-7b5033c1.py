"""Turning stamp trail: transparent stencils carried by an oriented path.

Only standard-library modules and NumPy are used. No files are read or written.
The public solver uses only the supplied input pixels.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SHORTCUTS = ("dots", "north", "last", "reverse", "rows", "columns",
              "colors", "cards", "boxes", "incoming", "runs")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_),
          "integer cells required")
    _need(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _components(cells, diagonal=False):
    remaining = set(cells)
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    todo.append(p)
        result.append(found)
    return result


def _rectangle(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    r1, c1 = max(r for r, c in cells), max(c for r, c in cells)
    if min(r1 - r0, c1 - c0) < 2:
        return None
    edge = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}
    return (r0, c0, r1, c1) if cells == edge else None


def _trace(board, background):
    cells = set(map(tuple, np.argwhere(board != background)))
    _need(2 <= len(cells) <= 48, "path must have 2..48 cells")
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    ends = sorted(p for p in cells if len(neighbors[p]) == 1)
    _need(len(ends) == 2 and all(len(v) in (1, 2) for v in neighbors.values()),
          "one simple orthogonal chain required")
    _need(ends[0][0] != ends[1][0], "endpoints must have different rows")
    path, previous, current = [], None, min(ends)
    while True:
        _need(current not in path, "cycle")
        path.append(current)
        following = [p for p in neighbors[current] if p != previous]
        if not following:
            break
        previous, current = current, following[0]
    _need(len(path) == len(cells), "disconnected path")
    return path


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _need(bool(np.all(a[[0, -1], :] == bg) and np.all(a[:, [0, -1]] == bg)),
          "background margin required")
    candidates = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        comps = _components(map(tuple, np.argwhere(a == color)))
        for comp in comps:
            box = _rectangle(comp)
            if box and box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
                candidates.append((color, box, comps))
    _need(len(candidates) == 1, "unique large canvas frame required")
    frame, canvas, comps = candidates[0]
    boxes = [_rectangle(comp) for comp in comps]
    _need(all(box is not None for box in boxes), "all frame components must be rectangles")
    _need(2 <= len(boxes) - 1 <= 4, "two to four stencil cards required")
    r0, c0, r1, c1 = canvas
    board = a[r0 + 1:r1, c0 + 1:c1]
    covered = np.zeros(a.shape, dtype=bool)
    masks, card_order = {}, []
    for box in sorted(boxes):
        y0, x0, y1, x1 = box
        _need(not covered[y0:y1 + 1, x0:x1 + 1].any(), "overlapping framed boxes")
        covered[y0:y1 + 1, x0:x1 + 1] = True
        if box == canvas:
            continue
        interior = a[y0 + 1:y1, x0 + 1:x1]
        h, w = interior.shape
        _need(h in (1, 3, 5) and w in (1, 3, 5), "odd stencil sizes through five")
        colors = set(map(int, np.unique(interior))) - {bg}
        _need(len(colors) == 1, "monochromatic stencil card")
        color = colors.pop()
        _need(color != frame and color not in masks, "unique stencil color")
        mask = interior == color
        cells = set(map(tuple, np.argwhere(mask)))
        _need(len(_components(cells, diagonal=True)) == 1, "8-connected stencil required")
        _need(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any(),
              "stencil must tightly fill its odd bounding box")
        masks[color] = mask
        card_order.append(color)
    _need(bool(np.all(a[~covered] == bg)), "unframed foreground")
    path = _trace(board, bg)
    colors = [int(board[p]) for p in path]
    _need(set(colors) <= set(masks), "path colors need matching stencil cards")
    for i, p in enumerate(path):
        direction = _heading(path, i)
        offsets = _offsets(masks[colors[i]], direction)
        _need(all(0 <= p[0] + dr < board.shape[0] and 0 <= p[1] + dc < board.shape[1]
                  for dr, dc in offsets), "stencil footprint outside canvas")
    return {"background": bg, "frame": frame, "canvas": canvas, "board": board,
            "path": path, "colors": colors, "masks": masks, "card_order": card_order}


def _heading(path, index, incoming=False):
    if index == len(path) - 1 or (incoming and index > 0):
        a, b = path[index - 1], path[index]
    else:
        a, b = path[index], path[index + 1]
    return _DIRS.index((b[0] - a[0], b[1] - a[1]))


def _offsets(mask, quarter_turns):
    h, w = mask.shape
    offsets = [(int(r) - h // 2, int(c) - w // 2) for r, c in np.argwhere(mask)]
    for _ in range(quarter_turns):
        offsets = [(c, -r) for r, c in offsets]
    return offsets


def _render(info, variant="correct"):
    path, colors, masks = info["path"], info["colors"], info["masks"]
    if variant == "reverse":
        path, colors = path[::-1], colors[::-1]
    order = list(range(len(path)))
    if variant == "last":
        order.reverse()
    elif variant == "rows":
        order.sort(key=lambda i: path[i])
    elif variant == "columns":
        order.sort(key=lambda i: (path[i][1], path[i][0]))
    elif variant == "colors":
        order.sort(key=lambda i: colors[i])
    elif variant == "cards":
        order.sort(key=lambda i: info["card_order"].index(colors[i]))
    elif variant == "runs":
        order = [i for i in order if i == 0 or colors[i] != colors[i - 1]]
    out = np.full(info["board"].shape, info["background"], dtype=np.int8)
    claimed = set()
    for i in order:
        mask = masks[colors[i]]
        if variant == "dots":
            mask = np.ones((1, 1), dtype=bool)
        elif variant == "boxes":
            mask = np.ones(mask.shape, dtype=bool)
        q = 0 if variant == "north" else _heading(path, i, variant == "incoming")
        for dr, dc in _offsets(mask, q):
            p = path[i][0] + dr, path[i][1] + dc
            # Diagnostic alternatives can extend beyond the legal correct footprint.
            if 0 <= p[0] < out.shape[0] and 0 <= p[1] < out.shape[1] and p not in claimed:
                out[p] = colors[i]
                claimed.add(p)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read frames, trace from the upper endpoint, turn and print every stencil."""
    return _render(_parse(input_grid))


def _features(info):
    path, colors = info["path"], info["colors"]
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    correct = _render(info)
    return {"length": len(path), "turns": sum(a != b for a, b in zip(steps, steps[1:])),
            "above": min(r for r, c in path) < path[0][0],
            "up": (-1, 0) in steps,
            "reversal": (0, -1) in steps and (0, 1) in steps,
            "stencils": len(info["masks"]),
            "max_stencil": max(max(m.shape) for m in info["masks"].values()),
            "recurrence": len([c for i, c in enumerate(colors) if not i or c != colors[i - 1]])
                          > len(set(colors)),
            "holes": any(not m[m.shape[0] // 2, m.shape[1] // 2] for m in info["masks"].values()),
            "different": {v: int(np.count_nonzero(correct != _render(info, v))) for v in _SHORTCUTS}}


def _mask(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    kind = int(rng.integers(6))
    if min(h, w) == 1 or kind == 0:
        mask[:] = True
    elif kind == 1:
        mask[[0, -1]] = True
        mask[:, [0, -1]] = True
    elif kind == 2 and h == w:
        mask = np.eye(h, dtype=bool)
    else:
        # Connect four independently chosen extremes, then grow adjacent pixels.
        points = [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                  (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
        rng.shuffle(points)
        r, c = points[0]
        mask[r, c] = True
        for y, x in points[1:]:
            while (r, c) != (y, x):
                r, c = r + int(np.sign(y - r)), c + int(np.sign(x - c))
                mask[r, c] = True
        target = int(rng.integers(int(mask.sum()), h * w + 1))
        for _ in range(target - int(mask.sum())):
            frontier = [(y, x) for y in range(h) for x in range(w) if not mask[y, x]
                        and mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any()]
            y, x = frontier[int(rng.integers(len(frontier)))]
            mask[y, x] = True
    if rng.random() < .5:
        mask = mask[:, ::-1]
    if rng.random() < .5:
        mask = mask[::-1]
    return mask.copy()


def _walk(rng, length, above=False, gentle=False):
    for _ in range(100):
        path, occupied, previous = [(0, 0)], {(0, 0)}, None
        for _ in range(length - 1):
            options, weights = [], []
            for direction in _DIRS:
                q = path[-1][0] + direction[0], path[-1][1] + direction[1]
                if q in occupied or sum((q[0] + dr, q[1] + dc) in occupied for dr, dc in _DIRS) != 1:
                    continue
                options.append((q, direction))
                weights.append((12.0 if gentle else 2.0) if direction == previous else 1.0)
            if not options:
                break
            weights = np.asarray(weights)
            q, previous = options[int(rng.choice(len(options), p=weights / weights.sum()))]
            path.append(q)
            occupied.add(q)
        if len(path) != length or path[0][0] == path[-1][0]:
            continue
        if path[0][0] > path[-1][0]:
            path.reverse()
        if above and min(r for r, c in path) >= path[0][0]:
            continue
        return path
    # A simple hook, including an interior point above both endpoints.
    path = [(0, 0), (-1, 0), (-2, 0), (-2, 1), (-2, 2), (-2, 3)]
    down = max(4, (length - 6) // 2)
    for _ in range(down):
        path.append((path[-1][0] + 1, path[-1][1]))
    while len(path) < length:
        path.append((path[-1][0], path[-1][1] - 1))
    return path


def _layout(rng, path, sequence, masks, bg, frame, side):
    """Construct frames and output directly from sampled objects, without parsing."""
    colors = list(masks)
    rng.shuffle(colors)
    horizontal = side in ("top", "bottom")
    sizes = [(masks[c].shape[0] + 2, masks[c].shape[1] + 2) for c in colors]
    gaps = [int(rng.integers(1, 3)) for _ in range(len(colors) - 1)]
    strip_length = sum(s[1 if horizontal else 0] for s in sizes) + sum(gaps)
    thickness = max(s[0 if horizontal else 1] for s in sizes)
    if strip_length > 28:
        return None
    rad = max(max(m.shape) // 2 for m in masks.values())
    lo = min(r for r, c in path), min(c for r, c in path)
    normalized = [(r - lo[0], c - lo[1]) for r, c in path]
    bh, bw = max(r for r, c in normalized) + 1, max(c for r, c in normalized) + 1
    min_h, min_w = max(7, bh + 2 * rad), max(7, bw + 2 * rad)
    max_h, max_w = (25 - thickness, 26) if horizontal else (26, 25 - thickness)
    if min_h > max_h or min_w > max_w:
        return None
    h = int(rng.integers(min_h, min(max_h, min_h + 5) + 1))
    w = int(rng.integers(min_w, min(max_w, min_w + 5) + 1))
    top = int(rng.integers(rad, h - bh - rad + 1))
    left = int(rng.integers(rad, w - bw - rad + 1))
    path = [(r + top, c + left) for r, c in normalized]
    if horizontal:
        ah, aw = h + 2 + thickness + 3, max(w + 2, strip_length) + 2
        board_y = 1 + (thickness + 1 if side == "top" else 0)
        board_x = int(rng.integers(1, aw - (w + 2)))
        dock_y = 1 if side == "top" else h + 4
        cursor = int(rng.integers(1, aw - strip_length))
    else:
        ah, aw = max(h + 2, strip_length) + 2, w + 2 + thickness + 3
        board_x = 1 + (thickness + 1 if side == "left" else 0)
        board_y = int(rng.integers(1, ah - (h + 2)))
        dock_x = 1 if side == "left" else w + 4
        cursor = int(rng.integers(1, ah - strip_length))
    inp = np.full((ah, aw), bg, dtype=np.int8)

    def draw_frame(y, x, hh, ww):
        inp[y:y + hh, x] = inp[y:y + hh, x + ww - 1] = frame
        inp[y, x:x + ww] = inp[y + hh - 1, x:x + ww] = frame

    draw_frame(board_y, board_x, h + 2, w + 2)
    for i, color in enumerate(colors):
        hh, ww = sizes[i]
        if horizontal:
            y, x = dock_y + int(rng.integers(thickness - hh + 1)), cursor
            cursor += ww + (gaps[i] if i < len(gaps) else 0)
        else:
            y, x = cursor, dock_x + int(rng.integers(thickness - ww + 1))
            cursor += hh + (gaps[i] if i < len(gaps) else 0)
        draw_frame(y, x, hh, ww)
        inp[y + 1:y + hh - 1, x + 1:x + ww - 1][masks[color]] = color
    for p, color in zip(path, sequence):
        inp[board_y + 1 + p[0], board_x + 1 + p[1]] = color
    # Expected output paints late-to-early with array rotations, unlike solve's
    # forward traversal and rotated offset occupancy set.
    out = np.full((h, w), bg, dtype=np.int8)
    for i in range(len(path) - 1, -1, -1):
        mask = np.rot90(masks[sequence[i]], -_heading(path, i))
        hh, ww = mask.shape
        y, x = path[i]
        view = out[y - hh // 2:y + hh // 2 + 1, x - ww // 2:x + ww // 2 + 1]
        view[mask] = sequence[i]
    return {"input": inp, "output": out}


def _candidate(rng, mode, target, side, fixed_background, witness=False):
    count = 4 if target and mode == "count" else int(rng.integers(2, 4))
    palette = list(map(int, rng.permutation(10)))
    bg = palette.pop() if fixed_background is None else fixed_background
    if bg in palette:
        palette.remove(bg)
    frame, colors = palette[0], palette[1:count + 1]
    masks = {}
    for i, color in enumerate(colors):
        dimensions = [(3, 3), (3, 3), (1, 3), (3, 1)]
        if target and mode == "stencil" and i == 0:
            dimensions = [(5, 3), (3, 5), (5, 5)]
        elif mode != "stencil" and rng.random() < .25:
            dimensions += [(5, 3), (3, 5), (5, 5)]
        h, w = dimensions[int(rng.integers(len(dimensions)))]
        masks[color] = _mask(rng, h, w)
    if mode == "length":
        length = int(rng.integers(26, 37) if target else rng.integers(12, 23))
    else:
        length = int(rng.integers(14, 29))
    path = _walk(rng, length, above=witness, gentle=not witness and rng.random() < .35)
    sequence = [colors[int(rng.integers(count))]]
    for _ in range(length - 1):
        sequence.append(sequence[-1] if rng.random() < .40 else colors[int(rng.integers(count))])
    # Each displayed card is exercised; recurrent colors still occur freely.
    positions = rng.choice(length, size=count, replace=False)
    for p, color in zip(positions, colors):
        sequence[int(p)] = color
    return _layout(rng, path, sequence, masks, bg, frame, side)


def _fallback(rng, mode, target, side, fixed_background):
    """Finite constructive witness, also preserving the requested shift."""
    palette = list(map(int, rng.permutation(10)))
    bg = palette.pop() if fixed_background is None else fixed_background
    if bg in palette:
        palette.remove(bg)
    frame, a, b, c, d = palette[:5]
    masks = {a: np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool),
             b: np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)}
    if target and mode == "stencil":
        masks[a] = np.ones((5, 3), dtype=bool)
        masks[a][1:4, 1] = False
    if target and mode == "count":
        masks[c] = np.ones((1, 3), dtype=bool)
        masks[d] = np.ones((3, 1), dtype=bool)
    path = [(2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 3), (2, 3), (3, 3), (4, 3), (4, 2), (4, 1),
            (5, 1), (6, 1), (6, 0), (6, -1)]
    sequence = [a, b, a, a, b, b, a, b, a, a, b, a, b, b, a, b]
    if target and mode == "length":
        # Extend away from the original chain; wide layout fits horizontally.
        for _ in range(12):
            path.append((path[-1][0] + 1, path[-1][1]))
            sequence.append(a if len(sequence) % 3 else b)
    if target and mode == "count":
        sequence[3], sequence[10] = c, d
    pair = _layout(rng, path, sequence, masks, bg, frame, side)
    if pair is None:
        # Fit a long fallback beside either a horizontal or a vertical dock.
        # The card designs keep their displayed north; only the guide changes.
        path = [(c, r) for r, c in path]
        if path[0][0] > path[-1][0]:
            path, sequence = path[::-1], sequence[::-1]
        pair = _layout(rng, path, sequence, masks, bg, frame, side)
    return pair


def _key(a):
    return tuple(a.shape), np.asarray(a, dtype=np.int8).tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    mode = str(rng.choice(("length", "stencil", "count")))
    fixed_background = int(rng.integers(10)) if rng.random() < .45 else None
    sides = list(rng.permutation(("top", "bottom", "left", "right")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            target, witness = split == "test", split == "train" and index == 0
            side = str(sides[int(rng.integers(2))]) if target else str(sides[index % 4])
            pair = None
            for attempt in range(160):
                candidate = _candidate(rng, mode, target, side, fixed_background, witness)
                if candidate is None or _key(candidate["input"]) in seen:
                    continue
                info = _parse(candidate["input"])
                if witness:
                    features = _features(info)
                    if not features["above"] or not features["recurrence"] or not all(features["different"].values()):
                        continue
                pair = candidate
                break
            if pair is None:
                base = _fallback(rng, mode, target, side, fixed_background)
                assert base is not None, "constructive geometry must fit"
                info = _parse(base["input"])
                # Eighteen distinct role permutations, fixing background, exceed
                # the maximum thirteen prior inputs. No random uniqueness retry.
                available = [c for c in range(10) if c != info["background"]]
                swap_a, swap_b = info["frame"], next(iter(info["masks"]))
                for attempt in range(18):
                    mapping = np.arange(10, dtype=np.int8)
                    for j, color in enumerate(available):
                        mapping[color] = available[(j + attempt % 9) % 9]
                    if attempt >= 9:
                        mapping[swap_a], mapping[swap_b] = mapping[swap_b], mapping[swap_a]
                    candidate = {k: mapping[v] for k, v in base.items()}
                    if _key(candidate["input"]) in seen:
                        continue
                    if witness and not all(_features(_parse(candidate["input"]))["different"].values()):
                        continue
                    pair = candidate
                    break
            if pair is None:
                raise RuntimeError("bounded construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check topology, rotated footprints, exact pixels, evidence and a real shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse(inp)
            assert out.shape == info["board"].shape, "output must be canvas interior"
            assert np.array_equal(out, _render(info)), "incorrect printed pixels"
            assert set(map(int, np.unique(out))) <= {info["background"], *info["masks"]}
            features[split].append(_features(info))
    train, test = features["train"], features["test"]
    for shortcut in _SHORTCUTS:
        assert any(f["different"][shortcut] for f in train), f"missing witness against {shortcut}"
    if any(f["above"] for f in test):
        assert any(f["above"] and f["different"]["reverse"] for f in train)
    assert any(f["recurrence"] for f in train), "recurrent colors must be taught"
    assert (min(f["length"] for f in test) > max(f["length"] for f in train)
            or min(f["max_stencil"] for f in test) > max(f["max_stencil"] for f in train)
            or min(f["stencils"] for f in test) > max(f["stencils"] for f in train)), "missing train/test shift"


def self_test():
    """Bounded, in-memory 42-seed coverage, ablation and corruption checks."""
    shapes, paths, motifs, signatures, roles, lengths = set(), set(), set(), set(), set(), set()
    totals = {"above": 0, "up": 0, "reversal": 0, "recurrence": 0, "holes": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                f = _features(info)
                shapes.add(pair["input"].shape)
                origin = info["path"][0]
                paths.add(tuple((r - origin[0], c - origin[1]) for r, c in info["path"]))
                motifs.update((m.shape, m.tobytes()) for m in info["masks"].values())
                roles.add((info["background"], info["frame"]))
                lengths.add(f["length"])
                for name in totals:
                    totals[name] += int(f[name])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                # Every generated example uses geometry beyond a bare path.
                assert f["different"]["dots"]
        if seed == 0:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
            bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
            try:
                validate_generated_puzzle(bad)
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError("corrupt output accepted")
    assert len(signatures) == 42 and len(shapes) >= 70 and len(paths) >= 150
    assert len(motifs) >= 70 and len(roles) >= 25 and len(lengths) >= 20
    assert all(n >= 15 for n in totals.values())
    # Exercise the constructive safety net rather than waiting for rare retries.
    for mode in ("length", "stencil", "count"):
        for side in ("top", "bottom", "left", "right"):
            for target in (False, True):
                pair = _fallback(np.random.default_rng(91), mode, target, side, None)
                assert pair is not None
                assert np.array_equal(solve(pair["input"]), pair["output"])
                if not target:
                    f = _features(_parse(pair["input"]))
                    assert f["above"] and f["recurrence"] and all(f["different"].values())
    # Palette permutations preserve geometry, equality matching and occupancy,
    # including a stencil whose literal color becomes zero.
    p = generate_puzzle(0)["train"][0]
    for shift in range(10):
        assert np.array_equal(solve((p["input"] + shift) % 10), (p["output"] + shift) % 10)
    # Invalid chain topologies are rejected before any printing takes place.
    for cells in ([(1, 1), (1, 2), (1, 3)],
                  [(1, 2), (2, 1), (2, 2), (2, 3)],
                  [(1, 1), (1, 2), (2, 1), (2, 2)],
                  [(1, 1), (2, 1), (4, 4)]):
        board = np.zeros((7, 7), dtype=np.int8)
        for cell in cells:
            board[cell] = 1
        try:
            _trace(board, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chain accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9000 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "puzzles": len(signatures), "input_shapes": len(shapes),
            "oriented_paths": len(paths), "stencil_masks": len(motifs),
            "role_pairs": len(roles), "lengths": len(lengths), "features": totals}


def _fingerprint(puzzle):
    # Compact serialization is used even for temporary, in-memory fingerprints.
    value = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
             for s in ("train", "test")}
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
