"""Ballots in rooms: Cartesian coordinates elect transparent room-center stamps.

Only standard-library modules and NumPy are used. No files, seed metadata,
training examples, or construction records are consulted by solve().
"""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    reached = {next(iter(cells))}
    todo = list(reached)
    while todo:
        y, x = todo.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = y + dy, x + dx
                if q in cells and q not in reached:
                    reached.add(q)
                    todo.append(q)
    return reached == cells


def _walls(shape, boxes):
    """Boxes describe boundaries, inclusive at both ends."""
    wall = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        wall[t, l:r + 1] = wall[b, l:r + 1] = True
        wall[t:b + 1, l] = wall[t:b + 1, r] = True
    return wall


def _rooms(wall):
    """Flood fill elementary free components; return inclusive interior boxes."""
    seen = wall.copy()
    h, w = wall.shape
    rooms = []
    for y, x in np.argwhere(~wall):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        todo, cells = [(y, x)], []
        seen[y, x] = True
        while todo:
            a, b = todo.pop()
            cells.append((a, b))
            if a in (0, h - 1) or b in (0, w - 1):
                raise AssertionError("an enclosure is open to the exterior")
            for c, d in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                if 0 <= c < h and 0 <= d < w and not seen[c, d]:
                    seen[c, d] = True
                    todo.append((c, d))
        a = np.asarray(cells)
        t, l = map(int, a.min(axis=0))
        b, r = map(int, a.max(axis=0))
        assert len(cells) == (b - t + 1) * (r - l + 1), "nonrectangular room"
        assert (b - t + 1) % 2 and (r - l + 1) % 2, "nonintegral room center"
        rooms.append((t, l, b, r))
    rooms.sort()
    assert rooms
    boundaries = [(t - 1, l - 1, b + 1, r + 1) for t, l, b, r in rooms]
    assert np.array_equal(wall, _walls(wall.shape, boundaries)), "stray/dangling wall"
    return rooms


def _winner(values):
    maximum = max(values.values())
    best = [c for c, n in values.items() if n == maximum]
    return best[0] if maximum > 0 and len(best) == 1 else None


def _votes(rooms, rows, cols, colors, mode="product"):
    counts, winners = [], []
    for t, l, b, r in rooms:
        tally = {}
        for color in colors:
            nr = int(np.count_nonzero(rows[t:b + 1] == color))
            nc = int(np.count_nonzero(cols[l:r + 1] == color))
            if mode == "rows":
                value = nr
            elif mode == "cols":
                value = nc
            elif mode == "sum":
                value = nr + nc
            else:
                value = nr * nc
            tally[color] = value
        counts.append(tally)
        winners.append(_winner(tally))
    return counts, winners


def _lattice(wall):
    """Deliberately wrong alternative: extend every supported line to the frame."""
    ys = np.flatnonzero(np.any(wall[:, :-1] & wall[:, 1:], axis=1))
    xs = np.flatnonzero(np.any(wall[:-1, :] & wall[1:, :], axis=0))
    full = np.zeros_like(wall)
    full[ys, :] = True
    full[:, xs] = True
    return full


def _parse(input_grid):
    g = _grid(input_grid)
    values, freq = np.unique(g, return_counts=True)
    assert len(values) >= 5
    order = np.argsort(freq)
    assert freq[order[-1]] > freq[order[-2]], "background must be unique"
    bg = int(values[order[-1]])
    candidates = []
    for wc in map(int, values):
        if wc == bg:
            continue
        pos = np.argwhere(g == wc)
        t, l = map(int, pos.min(axis=0))
        b, r = map(int, pos.max(axis=0))
        if b - t < 8 or r - l < 8:
            continue
        if any(g[y, x] != wc for y, x in ((t, l), (t, r), (b, l), (b, r))):
            continue
        board = g[t:b + 1, l:r + 1]
        if not np.array_equal(board[0], board[-1]) or not np.array_equal(board[:, 0], board[:, -1]):
            continue
        rows, cols = board[:, 0].copy(), board[0].copy()
        rc = set(map(int, rows)) - {wc}
        cc = set(map(int, cols)) - {wc}
        if rc != cc or not 2 <= len(rc) <= 4 or bg in rc:
            continue
        remaining = set(map(int, values)) - {bg, wc} - rc
        if len(remaining) != 1:
            continue
        noise = remaining.pop()
        if not np.all(np.isin(board[1:-1, 1:-1], [bg, wc, noise])):
            continue
        try:
            wall = board == wc
            # Simultaneous isolated-gap repair; no inference across background.
            horizontal = np.zeros_like(wall)
            vertical = np.zeros_like(wall)
            horizontal[:, 1:-1] = wall[:, :-2] & wall[:, 2:]
            vertical[1:-1, :] = wall[:-2, :] & wall[2:, :]
            repairs = (board == noise) & (horizontal | vertical)
            wall |= repairs
            wall[[0, -1], :] = True
            wall[:, [0, -1]] = True
            rooms = _rooms(wall)
            outside = np.ones(g.shape, dtype=bool)
            outside[t:b + 1, l:r + 1] = False
            templates = {}
            for color in sorted(rc):
                cells = np.argwhere((g == color) & outside)
                assert len(cells)
                low, high = cells.min(axis=0), cells.max(axis=0)
                bh, bw = map(int, high - low + 1)
                assert bh % 2 and bw % 2
                assert outside[low[0]:high[0] + 1, low[1]:high[1] + 1].all()
                view = g[low[0]:high[0] + 1, low[1]:high[1] + 1]
                mask = view == color
                assert _connected(mask) and np.all(np.isin(view, [bg, color]))
                center = (low + high) // 2
                templates[color] = {"cells": cells, "offsets": cells - center,
                                    "mask": mask, "box": tuple(map(int, (*low, *high)))}
            boxes = [v["box"] for v in templates.values()]
            assert all(a[2] < z[0] or z[2] < a[0] or a[3] < z[1] or z[3] < a[1]
                       for i, a in enumerate(boxes) for z in boxes[i + 1:])
            tallies, winners = _votes(rooms, rows, cols, sorted(rc))
            for room, winner in zip(rooms, winners):
                if winner is not None:
                    a, c, d, e = room
                    mh, mw = templates[winner]["mask"].shape
                    assert mh <= d - a + 1 and mw <= e - c + 1, "master does not fit"
            candidates.append(dict(grid=g, bg=bg, wc=wc, noise=noise, box=(t, l, b, r),
                                   board=board, rows=rows, cols=cols, wall=wall,
                                   repairs=repairs, rooms=rooms, templates=templates,
                                   colors=sorted(rc), tallies=tallies, winners=winners))
        except (AssertionError, ValueError, IndexError):
            continue
    assert len(candidates) == 1, "board and local roles must be unambiguous"
    return candidates[0]


def _render(p, rooms=None, winners=None, dot=False, direct=False):
    rooms = p["rooms"] if rooms is None else rooms
    if winners is None:
        _, winners = _votes(rooms, p["rows"], p["cols"], p["colors"])
    out = p["grid"].copy()
    out[out == p["noise"]] = p["bg"]
    t, l, b, r = p["box"]
    view = out[t:b + 1, l:r + 1]
    view[p["repairs"]] = p["wc"]
    used = set(winners) - {None}
    if direct:
        used = set(p["colors"])
    for color in used:
        cells = p["templates"][color]["cells"]
        out[cells[:, 0], cells[:, 1]] = p["bg"]
    if direct:
        # Clip only this intentionally wrong diagnostic prediction.
        for color in p["colors"]:
            for y in np.flatnonzero(p["rows"] == color):
                for x in np.flatnonzero(p["cols"] == color):
                    for dy, dx in p["templates"][color]["offsets"]:
                        a, c = int(y + dy), int(x + dx)
                        if 0 < a < view.shape[0] - 1 and 0 < c < view.shape[1] - 1 and not p["wall"][a, c]:
                            view[a, c] = color
        return out
    for (a, c, d, e), color in zip(rooms, winners):
        if color is None:
            continue
        cy, cx = (a + d) // 2, (c + e) // 2
        offsets = [(0, 0)] if dot else p["templates"][color]["offsets"]
        for dy, dx in offsets:
            y, x = cy + int(dy), cx + int(dx)
            if 0 < y < view.shape[0] - 1 and 0 < x < view.shape[1] - 1:
                view[y, x] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full result using only visible input evidence."""
    return _render(_parse(input_grid))


def _features(p):
    winners = p["winners"]
    correct = _render(p)
    result = {"rooms": len(p["rooms"]), "shape": tuple(p["grid"].shape),
              "colors": len(p["colors"]), "winner_colors": len(set(winners) - {None}),
              "retained": len(set(p["colors"]) - set(winners)),
              "ties": sum(max(t.values()) > 0 and list(t.values()).count(max(t.values())) > 1 for t in p["tallies"]),
              "empty": sum(max(t.values()) == 0 for t in p["tallies"]),
              "damage": int(p["repairs"].sum()),
              "partial": not np.array_equal(_lattice(p["wall"]), p["wall"]),
              "hollow": any(not p["templates"][c]["mask"][p["templates"][c]["mask"].shape[0] // 2,
                                                                   p["templates"][c]["mask"].shape[1] // 2]
                            for c in set(winners) - {None}),
              "multiple_votes": any(c is not None and v[c] > 1 for v, c in zip(p["tallies"], winners))}
    result["wall_ballots"] = sum(int(p["wall"][y, x]) for c in p["colors"]
                                 for y in np.flatnonzero(p["rows"] == c)
                                 for x in np.flatnonzero(p["cols"] == c))
    for mode in ("rows", "cols", "sum"):
        _, wrong = _votes(p["rooms"], p["rows"], p["cols"], p["colors"], mode)
        result[mode + "_difference"] = not np.array_equal(correct, _render(p, winners=wrong))
    global_counts = {c: int(np.count_nonzero(p["rows"] == c) * np.count_nonzero(p["cols"] == c)) for c in p["colors"]}
    result["global_difference"] = not np.array_equal(correct, _render(p, winners=[_winner(global_counts)] * len(winners)))
    whole = [(1, 1, p["wall"].shape[0] - 2, p["wall"].shape[1] - 2)]
    result["whole_difference"] = not np.array_equal(correct, _render(p, rooms=whole))
    result["lattice_difference"] = not np.array_equal(correct, _render(p, rooms=_rooms(_lattice(p["wall"]))))
    result["direct_difference"] = not np.array_equal(correct, _render(p, direct=True))
    result["dot_difference"] = not np.array_equal(correct, _render(p, dot=True))
    rows, cols = p["rows"].copy(), p["cols"].copy()
    for color in p["colors"]:
        rows[np.flatnonzero(rows == color)[1:]] = p["wc"]
        cols[np.flatnonzero(cols == color)[1:]] = p["wc"]
    _, wrong = _votes(p["rooms"], rows, cols, p["colors"])
    result["single_difference"] = not np.array_equal(correct, _render(p, winners=wrong))
    return result


def _partition(rng, h, w, number, partial):
    for _ in range(80):
        boxes = [(0, 0, h - 1, w - 1)]
        for _ in range(number - 1):
            choices = []
            for i, (t, l, b, r) in enumerate(boxes):
                for y in range(t + 4, b - 3, 2):
                    halves = [(t, l, y, r), (y, l, b, r)]
                    after = boxes[:i] + halves + boxes[i + 1:]
                    if sum(((d - a) // 4) * ((e - c) // 4) for a, c, d, e in after) >= number:
                        choices.append((i, halves))
                for x in range(l + 4, r - 3, 2):
                    halves = [(t, l, b, x), (t, x, b, r)]
                    after = boxes[:i] + halves + boxes[i + 1:]
                    if sum(((d - a) // 4) * ((e - c) // 4) for a, c, d, e in after) >= number:
                        choices.append((i, halves))
            if not choices:
                break
            i, halves = choices[int(rng.integers(len(choices)))]
            boxes[i:i + 1] = halves
        if len(boxes) != number:
            continue
        wall = _walls((h, w), boxes)
        if partial and np.array_equal(wall, _lattice(wall)):
            continue
        rooms = sorted((t + 1, l + 1, b - 1, r - 1) for t, l, b, r in boxes)
        assert _rooms(wall) == rooms
        return boxes, wall, rooms
    # Finite safety layout: two unequal vertical strips with distinct subdivisions.
    boxes = _safe_boxes(h, w, number)
    wall = _walls((h, w), boxes)
    assert not partial or not np.array_equal(wall, _lattice(wall))
    return boxes, wall, _rooms(wall)


def _safe_boxes(h, w, number):
    if number == 1:
        return [(0, 0, h - 1, w - 1)]
    first = min((h - 1) // 4, number - 1)
    boxes = []
    for left, right, count in ((0, 4, first), (4, w - 1, number - first)):
        ys = [2 * (((h - 1) * k) // (2 * count)) for k in range(count + 1)]
        boxes.extend((ys[k], left, ys[k + 1], right) for k in range(count))
    return boxes


def _label_scene(rng, h, w, rooms, colors, kind):
    # Numeric colors are placeholders until the final bijective palette map.
    for _ in range(2400):
        rows = np.full(h, 1, dtype=np.int8)
        cols = np.full(w, 1, dtype=np.int8)
        for axis in (rows, cols):
            if kind == "silence":
                amount = min(len(axis) - 2, int(rng.integers(len(colors), 2 * len(colors) + 3)))
            else:
                amount = int(rng.integers(max(len(colors), (len(axis) - 2) // 2), len(axis) - 1))
            positions = rng.choice(np.arange(1, len(axis) - 1), size=amount, replace=False)
            inks = list(colors) + list(map(int, rng.choice(colors, size=amount - len(colors))))
            rng.shuffle(inks)
            axis[positions] = inks
        tallies, winners = _votes(rooms, rows, cols, colors)
        used = set(winners) - {None}
        if not used:
            continue
        if kind == "contrast":
            if len(used) < 2:
                continue
            if any(_votes(rooms, rows, cols, colors, mode)[1] == winners for mode in ("rows", "cols", "sum")):
                continue
            sparse_rows, sparse_cols = rows.copy(), cols.copy()
            for color in colors:
                sparse_rows[np.flatnonzero(rows == color)[1:]] = 1
                sparse_cols[np.flatnonzero(cols == color)[1:]] = 1
            if _votes(rooms, sparse_rows, sparse_cols, colors)[1] == winners:
                continue
            if not any(c is not None and tally[c] > 1 for tally, c in zip(tallies, winners)):
                continue
        if kind == "silence":
            if len(used) == len(colors):
                continue
            if not any(max(v.values()) == 0 for v in tallies):
                continue
            if not any(max(v.values()) > 0 and list(v.values()).count(max(v.values())) > 1 for v in tallies):
                continue
        return rows, cols, tallies, winners
    return None


def _mask(rng, h, w):
    if min(h, w) == 1:
        return np.ones((h, w), dtype=bool)
    mode = int(rng.integers(6))
    if mode == 0:
        return np.ones((h, w), dtype=bool)
    if mode == 1:
        m = np.ones((h, w), dtype=bool)
        m[1:-1, 1:-1] = False
        return m
    if mode == 2 and h == w:
        m = np.eye(h, dtype=bool)
        return m[:, ::-1].copy() if rng.random() < .5 else m
    for _ in range(120):
        m = rng.random((h, w)) < float(rng.uniform(.35, .85))
        if m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any() and _connected(m):
            return m
    m = np.zeros((h, w), dtype=bool)
    m[h // 2, :] = m[:, w // 2] = True
    return m


def _place_masters(rng, shape, board_box, masks):
    t, l, b, r = board_box
    occupied = np.zeros(shape, dtype=bool)
    occupied[t:b + 1, l:r + 1] = True
    records = {}
    for color, mask in masks.items():
        h, w = mask.shape
        choices = [(y, x) for y in range(shape[0] - h + 1) for x in range(shape[1] - w + 1)
                   if not occupied[max(0, y - 1):min(shape[0], y + h + 1),
                                   max(0, x - 1):min(shape[1], x + w + 1)].any()]
        if not choices:
            return None
        y, x = choices[int(rng.integers(len(choices)))]
        occupied[y:y + h, x:x + w] = True
        records[color] = (y, x, mask)
    return records


def _construction_output(inp, noise, wall_color, bg, board_box, wall, rooms, winners, masters):
    """Ground truth uses clean geometry and construction masks, never solve()."""
    out = inp.copy()
    out[out == noise] = bg
    t, l, b, r = board_box
    target = out[t:b + 1, l:r + 1]
    target[1:-1, 1:-1][wall[1:-1, 1:-1]] = wall_color
    for color in set(winners) - {None}:
        y, x, mask = masters[color]
        source = out[y:y + mask.shape[0], x:x + mask.shape[1]]
        source[mask] = bg
    for (a, c, d, e), color in zip(rooms, winners):
        if color is None:
            continue
        mask = masters[color][2]
        h, w = mask.shape
        y, x = (a + d) // 2, (c + e) // 2
        target[y - h // 2:y + h // 2 + 1, x - w // 2:x + w // 2 + 1][mask] = color
    return out


def _make_pair(rng, number, kind="random"):
    for attempt in range(70):
        h = int(rng.choice([17, 19, 21, 23]))
        w = int(rng.choice([13, 15, 17, 19, 21, 23]))
        if number > 4:
            w = max(w, 17)
        _, wall, rooms = _partition(rng, h, w, number, kind != "random")
        colors = list(range(3, 3 + (3 if kind != "random" else int(rng.integers(2, 5)))))
        labeled = _label_scene(rng, h, w, rooms, colors, kind)
        if labeled is None:
            continue
        rows, cols, tallies, winners = labeled
        masks = {}
        for color in colors:
            won = [room for room, c in zip(rooms, winners) if c == color]
            maxh = min([d - a + 1 for a, _, d, _ in won] + [5])
            maxw = min([e - c + 1 for _, c, _, e in won] + [5])
            sizes = [(a, b) for a in (1, 3, 5) for b in (1, 3, 5)
                     if a <= maxh and b <= maxw and (a, b) != (1, 1)]
            weights = np.array([8 if a == b == 3 else 2 if max(a, b) <= 3 else 1 for a, b in sizes], dtype=float)
            mh, mw = sizes[int(rng.choice(len(sizes), p=weights / weights.sum()))]
            masks[color] = _mask(rng, mh, mw)
        if kind != "random":
            color = next(c for c in winners if c is not None)
            masks[color] = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        eh, ew = int(rng.integers(6, 8)), int(rng.integers(6, 8))
        shape = h + eh, w + ew
        mh = max(m.shape[0] for m in masks.values())
        possible_y = [i for i in range(eh + 1) if max(i, eh - i) >= mh + 1]
        top = int(rng.choice(possible_y))
        left = int(rng.integers(ew + 1))
        board_box = top, left, top + h - 1, left + w - 1
        masters = _place_masters(rng, shape, board_box, masks)
        if masters is None:
            continue
        inp = np.zeros(shape, dtype=np.int8)
        board = inp[top:top + h, left:left + w]
        board[wall] = 1
        board[:, 0] = board[:, -1] = rows
        board[0, :] = board[-1, :] = cols
        protected = np.zeros(shape, dtype=bool)
        for color, (y, x, mask) in masters.items():
            inp[y:y + mask.shape[0], x:x + mask.shape[1]][mask] = color
            protected[y:y + mask.shape[0], x:x + mask.shape[1]] = True
        # Damage non-junction straight internal walls; each gap is independently supported.
        sites = []
        for y, x in np.argwhere(wall):
            y, x = int(y), int(x)
            if not (1 <= y < h - 1 and 1 <= x < w - 1):
                continue
            neighbors = [wall[y - 1, x], wall[y + 1, x], wall[y, x - 1], wall[y, x + 1]]
            support = ((board[y - 1, x] == 1 and board[y + 1, x] == 1)
                       or (board[y, x - 1] == 1 and board[y, x + 1] == 1))
            if sum(map(int, neighbors)) == 2 and support:
                sites.append((y, x))
        rng.shuffle(sites)
        amount = int(rng.integers(1, 6)) if kind != "random" or rng.random() < .8 else 0
        damaged = []
        for y, x in sites:
            if len(damaged) >= amount:
                break
            if all(max(abs(y - a), abs(x - c)) > 1 for a, c in damaged):
                board[y, x] = 2
                damaged.append((y, x))
        if kind != "random" and not damaged:
            continue
        free = np.argwhere((inp == 0) & ~protected)
        count = min(len(free), int(rng.integers(1, 10)))
        for y, x in free[rng.choice(len(free), size=count, replace=False)]:
            inp[y, x] = 2
        expected = _construction_output(inp, 2, 1, 0, board_box, wall, rooms, winners, masters)
        palette = rng.permutation(10).astype(np.int8)
        pair = {"input": palette[inp], "output": palette[expected]}
        try:
            p = _parse(pair["input"])
            assert np.array_equal(p["wall"], wall)
            assert p["rooms"] == rooms
            assert np.array_equal(_render(p), pair["output"])
            f = _features(p)
            if kind == "contrast" and not all(f[k] for k in ("wall_ballots", "global_difference", "whole_difference", "lattice_difference", "direct_difference", "single_difference")):
                continue
            if kind == "silence" and not (f["ties"] and f["empty"] and f["retained"]):
                continue
            return pair
        except AssertionError:
            continue
    return _fallback_pair(rng, number, kind)


def _fallback_pair(rng, number, kind):
    """Small construction recipes, not saved/replayed puzzle payloads."""
    if kind in ("contrast", "silence"):
        h, w, top, left, shape = 21, 21, 6, 3, (28, 28)
        boxes = [(0, 0, 10, 10), (10, 0, 20, 10), (0, 10, 6, 20), (6, 10, 20, 20)]
        labels = (("133111153513414334511", "113534133454315441511") if kind == "contrast"
                  else ("114111111415514451131", "111111111114111115131"))
        rows, cols = (np.array(list(map(int, line)), dtype=np.int8) for line in labels)
    else:
        h, w, top, left, shape = 25, 21, 4, 3, (30, 27)
        boxes = _safe_boxes(h, w, number)
        rows, cols = np.ones(h, dtype=np.int8), np.ones(w, dtype=np.int8)
        rows[1:-1] = cols[1:-1] = 3
        rows[2], rows[4], cols[2], cols[4] = 4, 5, 4, 5
    wall = _walls((h, w), boxes)
    rooms = _rooms(wall)
    _, winners = _votes(rooms, rows, cols, [3, 4, 5])
    masks = {c: np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool) for c in (3, 4, 5)}
    masters = {c: (0 if top == 4 else 1, 1 + 7 * i, masks[c]) for i, c in enumerate((3, 4, 5))}
    inp = np.zeros(shape, dtype=np.int8)
    board = inp[top:top + h, left:left + w]
    board[wall] = 1
    board[:, 0] = board[:, -1] = rows
    board[0, :] = board[-1, :] = cols
    for c, (y, x, mask) in masters.items():
        inp[y:y + 3, x:x + 3][mask] = c
    if kind in ("contrast", "silence"):
        board[3, 10] = 2
    inp[-1, -1] = 2
    box = top, left, top + h - 1, left + w - 1
    out = _construction_output(inp, 2, 1, 0, box, wall, rooms, winners, masters)
    palette = rng.permutation(10).astype(np.int8)
    pair = {"input": palette[inp], "output": palette[out]}
    assert np.array_equal(solve(pair["input"]), pair["output"])
    return pair


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    seed = _integer(seed, "seed") % (1 << 64)
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            kind = "contrast" if split == "train" and i == 0 else "silence" if split == "train" and i == 1 else "random"
            number = int(rng.integers(3, 5)) if kind != "random" else int(rng.integers(1, 5) if split == "train" else rng.integers(5, 8))
            pair = _make_pair(rng, number, kind)
            # At most 13 earlier inputs; 20 bijections are more than sufficient.
            for k in range(20):
                mapping = (np.arange(10, dtype=np.int8) + k % 10) % 10
                if k >= 10:
                    a, b = map(int, np.unique(pair["input"])[:2])
                    mapping[[a, b]] = mapping[[b, a]]
                new = {role: mapping[grid] for role, grid in pair.items()}
                if _key(new["input"]) not in seen:
                    pair = new
                    break
            else:
                raise AssertionError("distinct recolorings exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            p = _parse(inp)
            assert np.array_equal(_render(p), out), "incorrect ballots, stamping, or preservation"
            # Check factored counts against explicit ballot enumeration.
            for room, tally in zip(p["rooms"], p["tallies"]):
                a, c, d, e = room
                for color in p["colors"]:
                    explicit = sum(a <= y <= d and c <= x <= e
                                   for y in np.flatnonzero(p["rows"] == color)
                                   for x in np.flatnonzero(p["cols"] == color))
                    assert explicit == tally[color]
            # Exact occupied-offset stamping in each room, including transparent cells.
            t, l, _, _ = p["box"]
            for (a, c, d, e), color in zip(p["rooms"], p["winners"]):
                view = out[t + a:t + d + 1, l + c:l + e + 1]
                target = np.full(view.shape, p["bg"], dtype=np.int8)
                if color is not None:
                    h, w = p["templates"][color]["mask"].shape
                    cy, cx = view.shape[0] // 2, view.shape[1] // 2
                    target[cy - h // 2:cy + h // 2 + 1, cx - w // 2:cx + w // 2 + 1][p["templates"][color]["mask"]] = color
                assert np.array_equal(view, target), "room must contain exactly one centered transparent stamp"
            features[split].append(_features(p))
    tr = features["train"]
    assert any(f["winner_colors"] >= 2 and f["rows_difference"] and f["cols_difference"] for f in tr)
    assert any(f["ties"] and f["empty"] and f["retained"] for f in tr)
    for k in ("hollow", "multiple_votes", "wall_ballots", "damage", "partial", "global_difference", "whole_difference",
              "lattice_difference", "direct_difference", "dot_difference", "single_difference", "sum_difference"):
        assert any(f[k] for f in tr), f"training lacks witness: {k}"
    assert max(f["rooms"] for f in tr) <= 4
    assert all(5 <= f["rooms"] <= 7 for f in features["test"])


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; bounded, file-free semantic and variation checks."""
    shapes, topologies, masks, roles, signatures = set(), set(), set(), set(), set()
    totals = Counter()
    master_counts, room_counts, label_products, offsets = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _parse(pair["input"])
                f = _features(p)
                shapes.add(tuple(pair["input"].shape))
                topologies.add((p["wall"].shape, tuple(p["rooms"])))
                roles.add((p["bg"], p["wc"], p["noise"]))
                master_counts.add(len(p["colors"]))
                room_counts.add(len(p["rooms"]))
                offsets.add(p["box"][:2])
                for color in p["colors"]:
                    mask = p["templates"][color]["mask"]
                    masks.add(_key(mask))
                    label_products.add((int(np.count_nonzero(p["rows"] == color)), int(np.count_nonzero(p["cols"] == color))))
                for k, v in f.items():
                    if isinstance(v, (int, bool, np.bool_)):
                        totals[k] += bool(v)
                if seed < 3:
                    palette = np.array([7, 0, 9, 2, 8, 1, 5, 3, 6, 4], dtype=np.int8)
                    assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
        if seed in (0, 7):
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(topologies) >= 150 and len(masks) >= 60
    assert len(roles) >= 70 and len(offsets) >= 20 and len(label_products) >= 25
    assert master_counts == {2, 3, 4} and set(range(1, 8)) <= room_counts
    for k in ("ties", "empty", "retained", "hollow", "damage", "partial", "sum_difference"):
        assert totals[k] >= 30, k
    for kind in ("contrast", "silence"):
        f = _features(_parse(_fallback_pair(np.random.default_rng(300), 4, kind)["input"]))
        assert f["hollow"] and f["damage"] and f["partial"]
        if kind == "contrast":
            assert f["winner_colors"] >= 2
            assert all(f[k + "_difference"] for k in ("rows", "cols", "sum", "single", "global", "whole", "lattice", "direct"))
        else:
            assert f["ties"] and f["empty"] and f["retained"]
    for n in range(1, 8):
        f = _features(_parse(_fallback_pair(np.random.default_rng(301), n, "random")["input"]))
        assert f["rooms"] == n and f["winner_colors"]
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(500 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    wrong = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    wrong["test"][0]["output"][0, 0] = (int(wrong["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a changed answer")
    # Task-specific wrong labels, not merely malformed grid rejection.
    for mode in ("rows", "cols", "global", "lattice", "direct", "dot", "single"):
        bad = {s: list(p[s]) for s in p}
        for i, pair in enumerate(p["train"]):
            parsed = _parse(pair["input"])
            if not _features(parsed)[mode + "_difference"]:
                continue
            if mode in ("rows", "cols"):
                _, ws = _votes(parsed["rooms"], parsed["rows"], parsed["cols"], parsed["colors"], mode)
                wrong = _render(parsed, winners=ws)
            elif mode == "global":
                tally = {c: int(np.count_nonzero(parsed["rows"] == c) * np.count_nonzero(parsed["cols"] == c)) for c in parsed["colors"]}
                wrong = _render(parsed, winners=[_winner(tally)] * len(parsed["rooms"]))
            elif mode == "lattice":
                wrong = _render(parsed, rooms=_rooms(_lattice(parsed["wall"])))
            elif mode == "single":
                rs, cs = parsed["rows"].copy(), parsed["cols"].copy()
                for c in parsed["colors"]:
                    rs[np.flatnonzero(rs == c)[1:]] = parsed["wc"]
                    cs[np.flatnonzero(cs == c)[1:]] = parsed["wc"]
                wrong = _render(parsed, winners=_votes(parsed["rooms"], rs, cs, parsed["colors"])[1])
            else:
                wrong = _render(parsed, direct=mode == "direct", dot=mode == "dot")
            bad["train"][i] = {"input": pair["input"], "output": wrong}
            break
        else:
            raise AssertionError(f"missing mutation witness: {mode}")
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator accepted {mode} shortcut")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42-seed variation cohort and all legal counts)")
