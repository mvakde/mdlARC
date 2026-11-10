"""Stencil-defined islands: subtract by color, then filter connected material.

All semantic roles are recovered from the input. Sampling profiles affect only
the curriculum, never solve(). No parent generator or external data is imported.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import numpy as np


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask, connectivity=4):
    """Four-neighbor components, irrespective of the colors of their cells."""
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        cells = [(y, x)]
        seen[y, x] = True
        for r, c in cells:
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            if connectivity == 8:
                neighbors += [(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            for rr, cc in neighbors:
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    cells.append((rr, cc))
        result.append(np.array(cells, dtype=int))
    return result


def _bbox(cells):
    lo, hi = cells.min(axis=0), cells.max(axis=0)+1
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _slice(box):
    y0, y1, x0, x1 = box
    return np.s_[y0:y1, x0:x1]


def _rim(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _view(a, corner):
    if corner & 2:
        a = a[::-1]
    if corner & 1:
        a = a[:, ::-1]
    return a


def _gap(a, b):
    # Exclusive ends: strict inequality leaves at least one background cell.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _d4(mask):
    return [np.rot90(mask, k) for k in range(4)] + [
        np.rot90(np.fliplr(mask), k) for k in range(4)]


def _decode(input_grid):
    """Decode selector, registered cards and filled plates from visible geometry."""
    grid = _grid(input_grid)
    assert min(grid.shape) >= 5
    selectors = []
    for corner in range(4):
        v = _view(grid, corner)
        frame = int(v[3, 3])
        if not (np.all(v[3, :4] == frame) and np.all(v[:4, 3] == frame)):
            continue
        colors, counts = np.unique(v[:3, :3], return_counts=True)
        padding = colors[counts == 6]
        if len(padding) != 1:
            continue
        bg = int(padding[0])
        query = frozenset(int(c) for c in colors if c != bg)
        if len(query) == 3 and frame not in query | {bg} and np.count_nonzero(grid == frame) == 7:
            selectors.append((corner, bg, frame, query))
    assert len(selectors) == 1, "Exactly one corner L selector is required"
    corner, bg, frame, query = selectors[0]
    occupied = grid != bg
    selector_mask = np.zeros(grid.shape, dtype=bool)
    _view(selector_mask, corner)[:4, :4] = True
    occupied = occupied & ~selector_mask
    selector_box = _bbox(np.argwhere(selector_mask))

    # Each binary card has a closed colored outline. Its outline component
    # exposes the entire register even when interior ink is disconnected.
    candidates = []
    for cells in _components(occupied):
        box = _bbox(cells)
        a = grid[_slice(box)]
        if min(a.shape) < 5:
            continue
        color = int(a[0, 0])
        inner = a[1:-1, 1:-1]
        if (color not in query | {bg, frame} and np.all(_rim(a) == color)
                and np.all((a == color) | (a == bg))
                and np.any(inner == bg) and np.any(inner == color)):
            candidates.append((box, color))
    # A ring-shaped ink fragment inside a card is part of that card, not a
    # second register. Keep only inclusion-maximal candidate boxes.
    maximal = []
    for box, color in candidates:
        contained = any(other != box and other[0] <= box[0] and box[1] <= other[1]
                        and other[2] <= box[2] and box[3] <= other[3]
                        for other, _ in candidates)
        if not contained:
            maximal.append((box, color))
    assert maximal, "No stencil cards"
    cards = {}
    excluded = selector_mask.copy()
    boxes = [selector_box]
    for box, color in maximal:
        assert color not in cards, "One stencil per identity"
        assert all(_gap(box, b) for b in boxes)
        boxes.append(box)
        a = grid[_slice(box)]
        cards[color] = {"box": box, "mask": (a[1:-1, 1:-1] == color)}
        excluded[_slice(box)] = True

    plates = []
    for cells in _components((grid != bg) & ~excluded):
        box = _bbox(cells)
        a = grid[_slice(box)]
        assert min(a.shape) >= 5 and len(cells) == a.size, "Plates must be filled rectangles"
        color = int(a[0, 0])
        assert np.all(_rim(a) == color) and color in cards, "Unmatched plate rim"
        assert cards[color]["mask"].shape == a[1:-1, 1:-1].shape
        assert all(_gap(box, b) for b in boxes)
        boxes.append(box)
        assert frame not in a and bg not in a
        labels = set(int(c) for c in np.unique(a)) - {color}
        assert not (labels & set(cards)), "Stencil identities are distinct from label identities"
        plates.append({"box": box, "color": color, "block": a.copy(),
                       "mask": cards[color]["mask"].copy()})
    assert plates, "No marked material plates"
    return {"grid": grid, "background": bg, "frame": frame, "query": query,
            "corner": corner, "selector_box": selector_box, "cards": cards, "plates": plates}


def _material(block, mask, bg, filler, query, connectivity=4):
    carved = block.copy()
    carved[1:-1, 1:-1][mask] = bg
    islands = []
    for cells in _components(carved != bg, connectivity):
        vals, counts = np.unique(carved[cells[:, 0], cells[:, 1]], return_counts=True)
        frequencies = {int(c): int(n) for c, n in zip(vals, counts)}
        labels = frozenset(frequencies) - {filler}
        islands.append({"cells": cells, "labels": labels, "counts": frequencies,
                        "positive": query <= labels})
    return carved, islands


def _finish(block, mask, bg, filler, query):
    carved, islands = _material(block, mask, bg, filler, query)
    result = carved.copy()
    for obj in islands:
        if obj["positive"]:
            p = obj["cells"]
            result[p[:, 0], p[:, 1]] = bg
    return result


def _apply(info, mode="correct", orientation=0):
    out = info["grid"].copy()
    for plate in info["plates"]:
        block, mask = plate["block"], plate["mask"]
        if mode == "no_stencil":
            mask = np.zeros_like(mask)
        elif mode == "orientation":
            mask = _d4(mask)[orientation]
            assert mask.shape == plate["mask"].shape
        elif mode == "copy_ink":
            mask = ~mask
        carved, islands = _material(block, mask, info["background"], plate["color"], info["query"],
                                    8 if mode == "eight_neighbors" else 4)
        result = carved.copy()
        if mode == "global_after":
            labels = frozenset(int(c) for c in np.unique(carved)) - {info["background"], plate["color"]}
            if info["query"] <= labels:
                result[:] = info["background"]
        elif mode == "global_before":
            if info["query"] <= set(int(c) for c in np.unique(block)):
                result[:] = info["background"]
        elif mode != "no_filter":
            for obj in islands:
                if obj["positive"]:
                    p = obj["cells"]
                    if mode == "bounding_boxes":
                        result[_slice(_bbox(p))] = info["background"]
                    else:
                        result[p[:, 0], p[:, 1]] = info["background"]
        out[_slice(plate["box"])] = result
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Subtract the matched ink, then delete complete-query material islands."""
    return _apply(_decode(input_grid))


def _rooms(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    mask[0] = mask[-1] = True
    mask[:, 0] = mask[:, -1] = True
    y = int(rng.integers(3, h-3))
    x = int(rng.integers(3, w-3))
    mask[y, :] = True
    mask[:, x] = True
    corners = ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))
    mask[corners[int(rng.integers(4))]] = False
    return mask


def _bridge_masks(rng, k):
    for _ in range(64):
        mask = np.zeros((k, k), dtype=bool)
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
        divider = int(rng.integers(3, k-3))
        mask[:, divider] = True
        # A one-cell material eye is enclosed at the right chamber's corner.
        # It survives even when the surrounding, rim-connected island qualifies.
        mask[1, k-3] = mask[2, k-2] = True
        notch = int(rng.choice([r for r in range(1, k-1) if r != k//2]))
        mask[notch, divider-1] = True
        door = int(rng.choice([r for r in range(1, k-1) if r != notch]))
        mask[door, 0] = False  # left chamber joins the plate's outer rim
        gate = int(rng.choice([r for r in range(2, k-1) if r != notch]))
        opened = mask.copy()
        opened[gate, divider] = False
        if len({a.tobytes() for a in _d4(mask)}) == 8:
            return mask, opened, divider
    raise AssertionError("Could not construct an asymmetric bridge stencil")


def _sample_mask(rng, h, w, kind=None):
    if kind is None:
        kind = int(rng.integers(6))
    mask = np.zeros((h, w), dtype=bool)
    if kind in (0, 1):
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
        for r in rng.choice(range(1, h-1), int(rng.integers(min(3, h-2)+1)), replace=False):
            mask[int(r), :] = True
        for c in rng.choice(range(1, w-1), int(rng.integers(min(3, w-2)+1)), replace=False):
            mask[:, int(c)] = True
        if kind == 1:
            for _ in range(int(rng.integers(1, 5))):
                p = int(rng.integers(h*w))
                mask.flat[p] = False
    elif kind == 2:
        mask = rng.random((h, w)) < float(rng.uniform(.15, .70))
    elif kind == 3:  # a turning cut with optional branches
        c = int(rng.integers(w))
        for r in range(h):
            mask[r, c] = True
            nc = min(w-1, max(0, c+int(rng.integers(-1, 2))))
            mask[r, min(c, nc):max(c, nc)+1] = True
            c = nc
        if rng.random() < .65:
            mask[int(rng.integers(h)), :] = True
    elif kind == 4:  # remaining material grows into irregular polyominoes
        mask[:] = True
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(int(rng.integers(3, h*w//2+3))):
                mask[r, c] = False
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                r, c = min(h-1, max(0, r+dr)), min(w-1, max(0, c+dc))
    else:
        choice = int(rng.integers(4))
        if choice == 0:
            mask[int(rng.integers(h)), :] = True
            mask[:, int(rng.integers(w))] = True
        elif choice == 1:
            mask[0, 0] = mask[-1, -1] = True
        elif choice == 2:
            mask[:] = True
            mask.flat[int(rng.integers(h*w))] = False
        else:
            for r in range(h):
                mask[r, min(w-1, r*w//h)] = True
    if not mask.any():
        mask.flat[int(rng.integers(h*w))] = True
    if mask.all():
        mask.flat[int(rng.integers(h*w))] = False
    return mask


def _put_labels(rng, block, coordinates, labels):
    assert len(coordinates) >= len(labels)
    order = rng.choice(len(coordinates), len(labels), replace=False)
    for index, color in zip(order, rng.permutation(labels)):
        r, c = coordinates[int(index)]
        block[int(r), int(c)] = int(color)


def _identity_plate(rng, mask, color, query, extras):
    h, w = mask.shape
    block = np.full((h+2, w+2), color, dtype=np.int8)
    rooms = [room for room in _components(~mask) if len(room) >= 4]
    assert len(rooms) == 4 and all(len(r) >= 4 for r in rooms)
    # The surviving inner-corner pixel touches the exact positive chamber
    # diagonally, and joins the rim orthogonally. It must not join that chamber.
    corner = next((r, c) for r, c in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)) if not mask[r, c])
    target = (1 if corner[0] == 0 else h-2, 1 if corner[1] == 0 else w-2)
    positive = next(i for i, room in enumerate(rooms) if any(tuple(p) == target for p in room))
    _put_labels(rng, block, rooms[positive]+1, list(query))
    negatives = [[q for q in query if q != missing]+[extras[0]] for missing in rng.permutation(query)]
    for room, labels in zip([r for i, r in enumerate(rooms) if i != positive], negatives):
        _put_labels(rng, block, room+1, labels)
    return block


def _bridge_plate(rng, mask, divider, color, query, extras):
    k = mask.shape[0]
    block = np.full((k+2, k+2), color, dtype=np.int8)
    left = np.argwhere(~mask & (np.indices(mask.shape)[1] < divider))
    right_parts = _components(~mask & (np.indices(mask.shape)[1] > divider))
    right = max(right_parts, key=len)  # leave the enclosed material eye unmarked
    q0, q1, q2 = query
    _put_labels(rng, block, left+1, [q0, q1, q0, extras[0], extras[0]])
    _put_labels(rng, block, right+1, [q1, q2, q2, extras[1]])
    # These two query witnesses are cut away before membership is tested.
    block[1, 2] = q2
    block[1, divider+2] = q0
    return block


def _free_plate(rng, mask, color, query, extras):
    h, w = mask.shape
    block = np.full((h+2, w+2), color, dtype=np.int8)
    _, islands = _material(block, mask, -1, color, frozenset(query))
    for obj in islands:
        cells = obj["cells"]
        interior = cells[(cells[:, 0] > 0) & (cells[:, 0] < h+1)
                         & (cells[:, 1] > 0) & (cells[:, 1] < w+1)]
        n = len(interior)
        if not n:
            continue
        if n >= 3 and rng.random() < .48:
            labels = list(query)
        else:
            labels = list(rng.permutation(query)[:int(rng.integers(min(2, n)+1))])
        optional = list(rng.permutation(extras)[:int(rng.integers(len(extras)+1))])
        labels += optional[:max(0, n-len(labels))]
        if labels:
            target = int(rng.integers(len(labels), max(len(labels), min(n, int(n*.75)))+1))
            labels += [int(rng.choice(labels)) for _ in range(target-len(labels))]
            _put_labels(rng, block, interior, labels)
    cut_cells = np.argwhere(mask)+1
    if len(cut_cells) and rng.random() < .85:
        n = min(len(cut_cells), int(rng.integers(1, 6)))
        _put_labels(rng, block, cut_cells, [int(rng.choice(list(query)+list(extras))) for _ in range(n)])
    return block


def _selector(rng, bg, frame, query):
    a = np.full((4, 4), bg, dtype=np.int8)
    a[3] = frame
    a[:, 3] = frame
    for p, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
        a[int(p)//3, int(p)%3] = color
    return a


def _card(mask, color, bg):
    h, w = mask.shape
    a = np.full((h+2, w+2), color, dtype=np.int8)
    a[1:-1, 1:-1] = np.where(mask, color, bg)
    return a


def _corner_layouts(rng, shapes):
    """Enumerate constructive packings with the selector fixed at (0, 0)."""
    options = []
    others = list(range(1, len(shapes)))
    for count in range(min(2, len(others))+1):
        for top in itertools.combinations(others, count):
            x, top_height = 5, 4
            positions = {0: (0, 0)}
            for i in top:
                positions[i] = (0, x)
                top_height = max(top_height, shapes[i][0])
                x += shapes[i][1]+1
            top_width = max(4, x-1)
            remaining = [i for i in others if i not in top]
            orders = [sorted(remaining, key=lambda i: (-shapes[i][0], -shapes[i][1])),
                      sorted(remaining, key=lambda i: (-shapes[i][1], -shapes[i][0])),
                      remaining, [int(i) for i in rng.permutation(remaining)]]
            for order in orders:
                for cols in range(1, min(4, max(1, len(order)))+1):
                    layout = dict(positions)
                    y, width = top_height+1, top_width
                    for start in range(0, len(order), cols):
                        row = order[start:start+cols]
                        row_h = max(shapes[i][0] for i in row)
                        xx = 0
                        for i in row:
                            layout[i] = (y, xx)
                            xx += shapes[i][1]+1
                        width = max(width, xx-1)
                        y += row_h+1
                    height = y-1 if remaining else top_height
                    if height <= 30 and width <= 30:
                        options.append((height, width, layout))
    assert options, "Requested geometry does not fit a corner-selector canvas"
    return options


def _pack(rng, shapes):
    options = _corner_layouts(rng, shapes)
    compact = min(h*w for h, w, _ in options)
    options = [o for o in options if o[0]*o[1] <= compact*1.45]
    h0, w0, fallback = options[int(rng.integers(len(options)))]
    h = h0+int(rng.integers(min(6, 30-h0)+1))
    w = w0+int(rng.integers(min(6, 30-w0)+1))
    if rng.random() < .20:
        h = 30
    if rng.random() < .20:
        w = 30
    order = sorted(range(1, len(shapes)),
                   key=lambda i: (-shapes[i][0]*shapes[i][1], float(rng.random())))
    for _ in range(8):
        forbidden = np.zeros((h, w), dtype=int)
        forbidden[:5, :5] = 1
        positions = {0: (0, 0)}
        for i in order:
            hh, ww = shapes[i]
            integral = np.pad(forbidden, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            totals = integral[hh:, ww:]-integral[:-hh, ww:]-integral[hh:, :-ww]+integral[:-hh, :-ww]
            available = np.argwhere(totals == 0)
            if not len(available):
                break
            r, c = map(int, available[int(rng.integers(len(available)))])
            positions[i] = (r, c)
            forbidden[max(0, r-1):min(h, r+hh+1), max(0, c-1):min(w, c+ww+1)] = 1
        if len(positions) == len(shapes):
            return h, w, positions
    # Finite fallback: corner row and shelves, with independent row offsets.
    positions = dict(fallback)
    for y in sorted({p[0] for i, p in positions.items() if i != 0}):
        row = [i for i, p in positions.items() if i != 0 and p[0] == y]
        row_end = max(positions[i][1]+shapes[i][1] for i in row)
        dx = int(rng.integers(w-row_end+1))
        for i in row:
            yy, xx = positions[i]
            positions[i] = (yy, xx+dx)
    return h, w, positions


def _make_pair(rng, package, profile, base_palette, vary_palette, corner):
    palette = [int(c) for c in (rng.permutation(10) if vary_palette else base_palette)]
    bg, frame = palette[:2]
    query, extras = palette[2:5], palette[5:7]
    identities = palette[7:]
    records, masks = [], {}
    if package == "identity":
        k = int(rng.integers(7, 10))
        color = identities[0]
        mask = _rooms(rng, k, k)
        masks[color] = mask
        records.append((color, _identity_plate(rng, mask, color, query, extras)))
        if profile == "reuse" or rng.random() < .35:
            records.append((color, _free_plate(rng, mask, color, query, extras)))
    elif package == "bridge":
        k = int(rng.integers(7, 10))
        closed, opened, divider = _bridge_masks(rng, k)
        first, second = identities[:2]
        block = _bridge_plate(rng, closed, divider, first, query, extras)
        other = block.copy()
        other[other == first] = second
        masks[first], masks[second] = closed, opened
        records.extend(((first, block), (second, other)))
    else:
        n = 3 if package == "test" and profile == "reuse" else int(rng.integers(1, 3 if profile == "reuse" else 4))
        distinct = 1 if package == "test" and profile == "reuse" else int(rng.integers(1, n+1))
        if package == "test" and profile == "scale":
            n = min(n, 2)
            distinct = min(distinct, n)
        for color in identities[:distinct]:
            if package == "test" and profile == "scale":
                hh, ww = 10, int(rng.integers(4, 9))
                if rng.random() < .5:
                    hh, ww = ww, hh
            elif package == "test" and profile == "reuse":
                hh, ww = int(rng.integers(7, 10)), int(rng.integers(5, 10))
            else:
                hh, ww = int(rng.integers(3, 8)), int(rng.integers(3, 8))
            kind = int(rng.integers(2, 5)) if package == "test" and profile == "topology" else None
            if package == "free" and profile == "topology":
                kind = int(rng.choice((0, 1, 5)))
            masks[color] = _sample_mask(rng, hh, ww, kind)
        colors = list(masks)+[int(rng.choice(list(masks))) for _ in range(n-distinct)]
        for color in colors:
            records.append((color, _free_plate(rng, masks[color], color, query, extras)))
    # Optional unmatched cards use the same representation and motifs.
    unused = [c for c in identities if c not in masks]
    can_distract = not (package == "test" and profile == "reuse")
    if unused and can_distract and rng.random() < .48:
        color = int(rng.choice(unused))
        maximum = max(max(mask.shape)+2 for mask in masks.values())
        limit = 4 if maximum >= 11 else 6
        masks[color] = _sample_mask(rng, int(rng.integers(3, limit)), int(rng.integers(3, limit)))
    blocks = [_selector(rng, bg, frame, query)]
    outputs = [blocks[0].copy()]
    for color, block in records:
        blocks.append(block)
        outputs.append(_finish(block, masks[color], bg, color, frozenset(query)))
    for color, mask in masks.items():
        card = _card(mask, color, bg)
        blocks.append(card)
        outputs.append(card.copy())
    h, w, positions = _pack(rng, [a.shape for a in blocks])
    if package == "bridge":
        # All four panels have the same dimensions. Permute their locations
        # so at least one plate's closest same-size card has the wrong color.
        # Geometry alone therefore cannot replace A's identity lookup.
        locations = [positions[i] for i in (1, 2, 3, 4)]
        alternatives = []
        def distance(a, b):
            return (a[0]-b[0])**2+(a[1]-b[1])**2
        for perm in itertools.permutations(locations):
            if distance(perm[0], perm[2]) > distance(perm[0], perm[3]) or distance(perm[1], perm[3]) > distance(perm[1], perm[2]):
                alternatives.append(perm)
        assert alternatives
        for i, position in zip((1, 2, 3, 4), alternatives[int(rng.integers(len(alternatives)))]):
            positions[i] = position
    inp = np.full((h, w), bg, dtype=np.int8)
    out = inp.copy()
    for i, a in enumerate(blocks):
        y, x = positions[i]
        box = np.s_[y:y+a.shape[0], x:x+a.shape[1]]
        inp[box] = a
        out[box] = outputs[i]
    return {"input": _view(inp, corner).copy(), "output": _view(out, corner).copy()}


def _key(a):
    return a.shape, a.astype(np.uint8, copy=False).tobytes()


def _training_coverage(infos):
    rows = []
    for info in infos:
        for plate in info["plates"]:
            carved, islands = _material(plate["block"], plate["mask"], info["background"], plate["color"], info["query"])
            rows.extend((info, plate, obj) for obj in islands)
    positives = [(i, p, o) for i, p, o in rows if o["positive"]]
    negatives = [(i, p, o) for i, p, o in rows if not o["positive"]]
    assert positives and negatives
    assert any(sum(o["counts"].get(c, 0) for c in o["labels"]) == 3 and o["labels"] == i["query"]
               for i, _, o in positives), "Exact three-mark positive missing"
    assert any(all(any(not o["positive"] and o["labels"] & info["query"] == info["query"]-{q}
                      for i, _, o in rows if i is info) for q in info["query"]) for info in infos)
    assert any(len(o["labels"]) == 3 and sum(o["counts"].get(c, 0) for c in o["labels"]) == 3
               for _, _, o in negatives), "Count-matched negative missing"
    assert any(not o["labels"] for _, _, o in negatives), "Unmarked material must survive"
    assert any(o["labels"]-i["query"] and any(o["counts"].get(q, 0) > 1 for q in i["query"])
               and any(o["counts"][x] > 1 for x in o["labels"]-i["query"])
               for i, _, o in positives), "Extras and repetitions are not taught"
    for mode in ("no_stencil", "no_filter", "global_before", "global_after", "copy_ink", "eight_neighbors", "bounding_boxes"):
        assert any(not np.array_equal(_apply(i, mode), _apply(i)) for i in infos), f"Shortcut {mode} survives"
    # Square witness registers allow all seven competing orientations to be
    # tested, even when later examples use rectangular plates.
    square_infos = [i for i in infos if all(p["mask"].shape[0] == p["mask"].shape[1] for p in i["plates"])]
    for transform in range(1, 8):
        assert any(not np.array_equal(_apply(i, "orientation", transform), _apply(i)) for i in square_infos)
    bridge = False
    for info in infos:
        plates = info["plates"]
        for j, a in enumerate(plates):
            for b in plates[j+1:]:
                if a["mask"].shape != b["mask"].shape or a["color"] == b["color"]:
                    continue
                norm_a, norm_b = a["block"].astype(int), b["block"].astype(int)
                norm_a[norm_a == a["color"]] = -1
                norm_b[norm_b == b["color"]] = -1
                if np.array_equal(norm_a, norm_b) and np.count_nonzero(a["mask"] != b["mask"]) == 1:
                    _, aa = _material(a["block"], a["mask"], info["background"], a["color"], info["query"])
                    _, bb = _material(b["block"], b["mask"], info["background"], b["color"], info["query"])
                    if any(o["positive"] for o in aa) != any(o["positive"] for o in bb):
                        # Swapping equal-sized color matches must change truth.
                        swapped = _finish(a["block"], b["mask"], info["background"], a["color"], info["query"])
                        correct = _finish(a["block"], a["mask"], info["background"], a["color"], info["query"])
                        bridge |= not np.array_equal(swapped, correct)
    assert bridge, "One-cell bridge/color-pairing witness missing"
    proximity_counterexample = False
    for info in infos:
        for plate in info["plates"]:
            matching_shape = [(c, card) for c, card in info["cards"].items()
                              if card["mask"].shape == plate["mask"].shape]
            y0, y1, x0, x1 = plate["box"]
            distances = {}
            for color, card in matching_shape:
                yy0, yy1, xx0, xx1 = card["box"]
                distances[color] = (y0+y1-yy0-yy1)**2+(x0+x1-xx0-xx1)**2
            proximity_counterexample |= distances[plate["color"]] > min(distances.values())
    assert proximity_counterexample, "Nearest-stencil shortcut was not contradicted"


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_infos = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(_apply(info), out), "Incorrect carving, membership, or preserved cells"
            altered = np.zeros(inp.shape, dtype=bool)
            for p in info["plates"]:
                altered[_slice(p["box"])] = True
                block_out = out[_slice(p["box"])]
                assert np.all(block_out[1:-1, 1:-1][p["mask"]] == info["background"])
                carved, islands = _material(p["block"], p["mask"], info["background"], p["color"], info["query"])
                for obj in islands:
                    c = obj["cells"]
                    actual = block_out[c[:, 0], c[:, 1]]
                    expected = info["background"] if obj["positive"] else carved[c[:, 0], c[:, 1]]
                    assert np.all(actual == expected), "An island was only partly deleted or a survivor changed"
            assert np.array_equal(inp[~altered], out[~altered])
            assert not np.array_equal(inp, out)
            if split == "train":
                train_infos.append(info)
    _training_coverage(train_infos)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    profile = str(rng.choice(("scale", "reuse", "topology")))
    palette = rng.permutation(10)
    varied = bool(rng.random() < .80)
    corners = [int(rng.integers(4))]
    if rng.random() < .70:
        corners.append((corners[0]+int(rng.integers(1, 4))) % 4)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            package = ("identity" if index == 0 else "bridge" if index == 1 else "free") if split == "train" else "test"
            corner = corners[index % len(corners)] if split == "train" and index < 2 else int(rng.choice(corners))
            for _ in range(24):
                pair = _make_pair(rng, package, profile, palette, varied, corner)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("Distinct-input retries exhausted")
    order = rng.permutation(nt)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return hashlib.sha256(b"".join(bytes(a.shape)+a.tobytes() for s in ("train", "test")
                         for p in puzzle[s] for a in (p["input"], p["output"]))).hexdigest()


def _canonical_structure(a):
    roles = {}
    return tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)


def self_test() -> dict:
    """Bounded 42-seed checks, structural measurements and semantic mutations."""
    metrics = {k: set() for k in ("puzzles", "shapes", "structures", "plate_shapes", "plate_counts",
               "island_counts", "deletions", "stencils", "corners", "backgrounds", "roles", "distractors")}
    irregular = repeated = rim_deleted = sacrificial = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        metrics["puzzles"].add(_fingerprint(puzzle))
        if seed < 3:
            assert _fingerprint(generate_puzzle(seed)) == _fingerprint(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                metrics["shapes"].add(pair["input"].shape)
                metrics["structures"].add((pair["input"].shape, _canonical_structure(pair["input"])))
                metrics["corners"].add(info["corner"])
                metrics["backgrounds"].add(info["background"])
                metrics["roles"].add((info["background"], info["frame"], tuple(sorted(info["query"]))))
                metrics["plate_counts"].add(len(info["plates"]))
                used = {p["color"] for p in info["plates"]}
                metrics["distractors"].add(len(set(info["cards"])-used))
                repeated += len(info["plates"]) > len(used)
                for p in info["plates"]:
                    mask = p["mask"]
                    metrics["plate_shapes"].add(p["block"].shape)
                    metrics["stencils"].add((mask.shape, mask.tobytes()))
                    carved, islands = _material(p["block"], mask, info["background"], p["color"], info["query"])
                    metrics["island_counts"].add(len(islands))
                    metrics["deletions"].add(sum(o["positive"] for o in islands))
                    sacrificed = set(int(c) for c in p["block"][1:-1, 1:-1][mask]) & info["query"]
                    sacrificial += bool(sacrificed)
                    for obj in islands:
                        c = obj["cells"]
                        y0, y1, x0, x1 = _bbox(c)
                        irregular += len(c) != (y1-y0)*(x1-x0)
                        rim_deleted += obj["positive"] and bool(np.any((c[:, 0] == 0) | (c[:, 1] == 0)
                            | (c[:, 0] == carved.shape[0]-1) | (c[:, 1] == carved.shape[1]-1)))
        # Relocating query cells does not change any material decisions.
        p = puzzle["train"][0]
        info = _decode(p["input"])
        moved, expected = p["input"].copy(), p["output"].copy()
        v = _view(moved, info["corner"])
        v[:3, :3] = np.roll(v[:3, :3].copy(), 1, axis=0)
        _view(expected, info["corner"])[:3, :3] = v[:3, :3]
        assert np.array_equal(solve(moved), expected)
        # An identity-only selector mutation turns a matched hard negative
        # into a positive without changing any stencil or plate cells.
        changed_query = False
        for p in puzzle["train"]:
            info = _decode(p["input"])
            for plate in info["plates"]:
                _, islands = _material(plate["block"], plate["mask"], info["background"], plate["color"], info["query"])
                targets = [o for o in islands if not o["positive"] and len(o["labels"]) == 3
                           and len(o["labels"] & info["query"]) == 2]
                if not targets:
                    continue
                changed = p["input"].copy()
                view = _view(changed, info["corner"])
                points = np.argwhere(view[:3, :3] != info["background"])
                for point, color in zip(points, sorted(targets[0]["labels"])):
                    view[tuple(point)] = color
                predicted = solve(changed)
                obj = targets[0]
                cells = obj["cells"]
                block_out = predicted[_slice(plate["box"])]
                assert np.all(block_out[cells[:, 0], cells[:, 1]] == info["background"])
                assert not np.array_equal(predicted[_slice(plate["box"])], p["output"][_slice(plate["box"])])
                changed_query = True
                break
            if changed_query:
                break
        assert changed_query
    assert len(metrics["puzzles"]) == 42
    assert len(metrics["shapes"]) >= 40 and len(metrics["structures"]) >= 150
    assert len(metrics["plate_shapes"]) >= 15 and len(metrics["stencils"]) >= 100
    assert metrics["corners"] == set(range(4)) and metrics["backgrounds"] == set(range(10))
    assert {1, 2, 3} <= metrics["plate_counts"] and {0, 1} <= metrics["distractors"]
    assert len(metrics["island_counts"]) >= 5 and {0, 1, 2} <= metrics["deletions"]
    assert min(irregular, repeated, rim_deleted, sacrificial) > 10
    for nt in range(2, 11):
        for ne in range(1, min(nt-1, 4)+1):
            p = generate_puzzle(20261007, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    for corruption in ("partial", "survivor", "card"):
        p = generate_puzzle(0, 2, 1)
        found = False
        for pair in p["train"]:
            info = _decode(pair["input"])
            if corruption == "card":
                card = next(iter(info["cards"].values()))
                y0, _, x0, _ = card["box"]
                pair["output"][y0, x0] = info["background"]
                found = True
                break
            for plate in info["plates"]:
                carved, islands = _material(plate["block"], plate["mask"], info["background"], plate["color"], info["query"])
                targets = [o for o in islands if o["positive"] == (corruption == "partial")]
                if targets:
                    r, c = targets[0]["cells"][0]
                    y0, _, x0, _ = plate["box"]
                    pair["output"][y0+r, x0+c] = carved[r, c] if corruption == "partial" else info["background"]
                    found = True
                    break
            if found:
                break
        assert found
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupted output accepted")
    report = {k: len(v) for k, v in metrics.items()}
    report.update({"irregular_islands": int(irregular), "reused_card_pairs": int(repeated),
                   "rim_deletions": int(rim_deleted), "plates_with_cut_query_labels": int(sacrificial)})
    return report


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
