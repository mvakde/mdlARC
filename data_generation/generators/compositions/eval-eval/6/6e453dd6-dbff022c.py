"""Addressed cavity transmission: an input-only ARC composition.

Only NumPy and the Python standard library are used. Internal construction colors
are permuted before returning a pair; solving always reads the visible key.
"""
from __future__ import annotations

import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if low is not None and not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, cells = [(r, c)], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        result.append(cells)
    return result


def _holes(mask):
    h, w = mask.shape
    return [cells for cells in _components(~mask)
            if all(r not in (0, h - 1) and c not in (0, w - 1) for r, c in cells)]


def _object(cells, grid):
    rr, cc = np.array(cells).T
    y, x = int(rr.min()), int(cc.min())
    h, w = int(rr.max() - y + 1), int(cc.max() - x + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[rr - y, cc - x] = True
    colors = {int(grid[q]) for q in cells}
    _require(len(colors) == 1, "object must be monochromatic")
    holes = _holes(mask)
    return dict(cells=cells, row=y, col=x, mask=mask, color=colors.pop(), holes=holes)


def _read_scene(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background must be uniquely dominant")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        h, w = a.shape
        for col in range(4, w - 3):
            rail_color = int(a[0, col])
            if rail_color == bg or not np.all(a[:, col] == rail_color):
                continue
            if np.count_nonzero(a == rail_color) != h:
                continue
            mask = (a != bg) & (a != rail_color)
            keys = []
            for cells in _components(mask, diagonal=True):
                rr, cc = np.array(cells).T
                y0, y1, x0, x1 = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
                n = y1 - y0 + 1
                if x0 != 0 or x1 != 1 or not 3 <= n <= 6 or len(cells) != 2 * n:
                    continue
                outer = list(map(int, a[y0:y1 + 1, 0]))
                inner = list(map(int, a[y0:y1 + 1, 1]))
                if len(set(outer)) != n or len(set(outer + inner)) < 2:
                    continue
                keys.append((cells, dict(zip(outer, inner)), y0, n))
            if len(keys) == 1:
                cells, mapping, start, n = keys[0]
                candidates.append(dict(grid=a.copy(), turn=turn, bg=bg, rail=rail_color,
                                       col=col, mapping=mapping, key=cells,
                                       key_start=start, key_length=n))
    _require(len(candidates) == 1, "rail direction and isolated boundary key must be unique")
    scene = candidates[0]
    a, col = scene['grid'], scene['col']
    source_mask = a[:, :col] != bg
    for r, c in scene['key']:
        source_mask[r, c] = False
    emitters = [_object(cells, a) for cells in _components(source_mask)]
    _require(bool(emitters), "no emitters")
    emitters.sort(key=lambda o: o['row'])
    end = -2
    for obj in emitters:
        _require(obj['row'] > end + 1, "emitter bands must have a blank row between them")
        end = obj['row'] + obj['mask'].shape[0] - 1
        obj['shift'] = col - obj['col'] - obj['mask'].shape[1]
        _require(obj['shift'] >= 0, "emitter crosses rail")
        _require(col - obj['mask'].shape[1] >= 3, "aligned emitter could touch key")
        obj['active'] = [hole for hole in obj['holes']
                         if any(c == obj['mask'].shape[1] - 2 for r, c in hole)]
        obj['decoy'] = [hole for hole in obj['holes'] if hole not in obj['active']]
        obj['open_rows'] = {r for r in range(obj['mask'].shape[0])
                            if obj['mask'].shape[1] >= 3 and not obj['mask'][r, -2]
                            and not any((r, obj['mask'].shape[1] - 2) in h for h in obj['holes'])}
    # Any two receiver objects are eight-disconnected, regardless of color.
    receivers = []
    right_mask = a[:, col + 1:] != bg
    for cells in _components(right_mask, diagonal=True):
        receivers.append(_object([(r, c + col + 1) for r, c in cells], a))
    _require(bool(receivers), "no receiver contours")
    cavities = []
    for i, obj in enumerate(receivers):
        for other in receivers[:i]:
            y, x, h, w = obj['row'], obj['col'], *obj['mask'].shape
            yy, xx, hh, ww = other['row'], other['col'], *other['mask'].shape
            _require(y + h < yy or yy + hh < y or x + w < xx or xx + ww < x,
                     "receiver boxes need a one-cell separation")
        for hole in obj['holes']:
            cells = [(r + obj['row'], c + obj['col']) for r, c in hole]
            fronts = {}
            for r, c in cells:
                fronts[r] = min(c, fronts.get(r, c))
            cavities.append(dict(cells=cells, fronts=fronts, color=obj['color'],
                                 channel=scene['mapping'].get(obj['color'])))
    scene.update(emitters=emitters, receivers=receivers, cavities=cavities)
    return scene


def _events(scene, policy='active'):
    events = set()
    for obj in scene['emitters']:
        if policy == 'holes':
            rows = {r for hole in obj['holes'] for r, c in hole}
        elif policy == 'column':
            rows = {r for r in range(obj['mask'].shape[0])
                    if obj['mask'].shape[1] >= 3 and not obj['mask'][r, -2]}
        else:
            rows = {r for hole in obj['active'] for r, c in hole}
        events.update((obj['row'] + r, obj['color']) for r in rows)
    return events


def _select(scene, events, use_key=True, nearest=True):
    selected, row_hits = {}, set()
    for row, channel in sorted(events):
        choices = [(hole['fronts'][row], i) for i, hole in enumerate(scene['cavities'])
                   if row in hole['fronts'] and (not use_key or hole['channel'] == channel)]
        if not choices:
            continue
        choices.sort()
        if nearest:
            choices = choices[:1]
        for _, i in choices:
            if use_key:
                _require(i not in selected or selected[i] == channel, "conflicting channels")
            # The deliberately wrong key-free ablation uses its first delivery
            # when several differently colored emitters hit the same region.
            selected.setdefault(i, channel)
            row_hits.add((i, row))
    return selected, row_hits


def _render(scene, gate='active', use_key=True, nearest=True,
            fill_all=False, rows_only=False, recursive=False):
    a, bg, col = scene['grid'], scene['bg'], scene['col']
    out = a.copy()
    for obj in scene['emitters']:
        for q in obj['cells']:
            out[q] = bg
    for obj in scene['emitters']:
        for r, c in obj['cells']:
            out[r, c + obj['shift']] = obj['color']
    selected, hits = _select(scene, _events(scene, gate), use_key, nearest)
    if fill_all:
        selected = {i: h['channel'] for i, h in enumerate(scene['cavities']) if h['channel'] is not None}
    for i, channel in selected.items():
        if recursive:
            seen = set()
            while channel in scene['mapping'] and channel not in seen:
                seen.add(channel)
                nxt = scene['mapping'][channel]
                if nxt == channel:
                    break
                channel = nxt
        for r, c in scene['cavities'][i]['cells']:
            if not rows_only or (i, r) in hits:
                out[r, c] = channel
    return np.rot90(out, -scene['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the rail, key, emitter topology and receiver topology from input only."""
    return _render(_read_scene(input_grid))


def _ring(h, w):
    mask = np.ones((h, w), dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


def _emitter_mask(rng, kind, h, w):
    mask = np.ones((h, w), dtype=bool)
    if kind == 'solid':
        if h > 2 and w > 2:
            for r in range(h):
                mask[r, :int(rng.integers(w - 1))] = False
    elif kind == 'open':
        mask = _ring(h, w)
        side = int(rng.integers(3))
        if side == 0:
            mask[1:-1, 0] = False
        else:
            mask[0 if side == 1 else -1, -2] = False
    elif kind == 'decoy':
        extension = int(rng.integers(1, min(3, w - 2)))
        body = w - extension
        mask[:, body:] = False
        mask[1:-1, 1:body - 1] = False
        mask[int(rng.choice([0, h - 1])), body:] = True
    elif kind == 'multi':
        n = int(rng.integers(2, min(3, (h - 1) // 2) + 1))
        # Sample positive cavity heights and separating solid rows.
        bins = rng.multinomial(h - (2 * n + 1), np.full(2 * n + 1, 1 / (2 * n + 1)))
        row = 1 + int(bins[0])
        for i in range(n):
            length = 1 + int(bins[2 * i + 1])
            for r in range(row, row + length):
                mask[r, int(rng.integers(1, w - 1)):w - 1] = False
            row += length + 1 + int(bins[2 * i + 2])
    elif kind == 'mixed':
        separator = int(rng.integers(2, w - 2))
        mask[1:-1, 1:separator] = False
        for r in range(1, h - 1):
            mask[r, int(rng.integers(separator + 1, w - 1)):w - 1] = False
    else:
        for r in range(1, h - 1):
            mask[r, int(rng.integers(1, w - 1)):w - 1] = False
        if rng.random() < .4:
            mask[0, 0] = False
    _require(len(_components(mask)) == 1, "construction split emitter")
    return mask


def _receiver_mask(rng, kind, h, w):
    if kind == 'diamond':
        radius = max(1, (min(h, w) - 1) // 2)
        rr, cc = np.indices((2 * radius + 1, 2 * radius + 1))
        return abs(rr - radius) + abs(cc - radius) == radius
    if kind == 'multi' and h >= 5:
        mask = _ring(h, w)
        mask[int(rng.integers(2, h - 2)), :] = True
    elif kind == 'profile' and h >= 4 and w >= 4:
        mask = np.ones((h, w), dtype=bool)
        # Every cavity row includes one common column, so the cavity stays joined.
        anchor = int(rng.integers(1, w - 1))
        for r in range(1, h - 1):
            left = int(rng.integers(1, anchor + 1))
            right = int(rng.integers(anchor, w - 1))
            mask[r, left:right + 1] = False
        mask[0, 0] = False
    else:
        mask = _ring(h, w)
    if kind == 'open':
        mask[0, -2] = False
    _require(len(_components(mask, True)) == 1, "construction split receiver")
    return mask


def _mapping(rng, lesson):
    if lesson in ('route', 'gates'):
        mapping = {1: 2, 2: 4, 3: 2}
        if lesson == 'gates':
            mapping[5] = 5
        n = int(rng.integers(len(mapping), 7))
        extras = list(map(int, rng.permutation([c for c in range(1, 9)
                                               if c not in mapping and c != 4
                                               and not (lesson == 'gates' and c == 8)])))
        for color in extras[:n - len(mapping)]:
            mapping[color] = int(rng.integers(1, 9))
    else:
        n = int(rng.integers(3, 7))
        sources = list(map(int, rng.choice(np.arange(1, 9), n, replace=False)))
        fills = list(map(int, rng.integers(1, 9, n)))
        if len(set(fills)) == 1:
            fills[1] = fills[0] % 8 + 1
        mapping = dict(zip(sources, fills))
    return mapping


def _new_canvas(rng, h, col, band, mapping):
    a = np.zeros((h, col + 1 + band), dtype=np.int8)
    a[:, col] = 9
    start = int(rng.integers(h - len(mapping) + 1))
    for i, c in enumerate(rng.permutation(list(mapping))):
        a[start + i, 0] = c
        a[start + i, 1] = mapping[int(c)]
    return a


def _put_emitter(a, rng, mask, row, color, col, force=None):
    limit = col - mask.shape[1] - 3
    shift = int(rng.integers(limit + 1))
    if force == 'move':
        shift = int(rng.integers(1, limit + 1))
    if force == 'aligned':
        shift = 0
    start = col - mask.shape[1] - shift
    a[row:row + mask.shape[0], start:start + mask.shape[1]][mask] = color


def _put_receiver(a, mask, row, col, color):
    a[row:row + mask.shape[0], col:col + mask.shape[1]][mask] = color


def _partition(rng, spare, bins):
    return list(map(int, rng.multinomial(spare, np.full(bins, 1 / bins))))


def _route_scene(rng):
    mapping = _mapping(rng, 'route')
    h = int(rng.integers(14, 28))
    widths = [int(rng.integers(3, 8)), int(rng.integers(6, 10))]
    col = int(rng.integers(max(widths) + 4, 19))
    band = int(rng.integers(11, 30 - col))
    a = _new_canvas(rng, h, col, band, mapping)
    spare = _partition(rng, h - 13, 3)
    first = 3 + spare[0]
    second = first + 5 + spare[1]
    source = _emitter_mask(rng, 'active', 3, widths[0])
    _put_emitter(a, rng, source, first, 2, col, 'move')
    companion = _emitter_mask(rng, 'mixed', 5, widths[1])
    _put_emitter(a, rng, companion, second, 4, col)
    # Three successive receivers: transparent wrong channel, delivered, shadowed.
    ww = [3, 3, 3]
    slack = band - 11
    for _ in range(slack):
        ww[int(rng.integers(3))] += 1
    x = col + 1
    for i, color in enumerate((2, 1, 3)):
        rh = int(rng.integers(4 if i == 1 else 3, 7))
        row_low = max(0, first + 3 - rh)
        row_high = min(first, first + 4 - rh)
        row = int(rng.integers(row_low, row_high + 1))
        kind = 'profile' if i == 1 else str(rng.choice(['rectangle', 'diamond']))
        mask = _receiver_mask(rng, kind, rh, ww[i])
        # A diamond may shrink; place its central cavity on the signal row.
        if kind == 'diamond':
            row = first + 1 - mask.shape[0] // 2
        _put_receiver(a, mask, row, x, color)
        x += ww[i] + 1
    # The second source has both active and recessed holes; only the former sends.
    mask = _receiver_mask(rng, str(rng.choice(['multi', 'profile', 'rectangle'])), 5,
                          int(rng.integers(3, min(8, band + 1))))
    _put_receiver(a, mask, second, col + 1 + int(rng.integers(band - mask.shape[1] + 1)), 2)
    return np.flipud(a).copy() if rng.random() < .5 else a


def _gate_scene(rng):
    mapping = _mapping(rng, 'gates')
    body_h, body_w = int(rng.integers(3, 6)), int(rng.integers(3, 8))
    masks = [_ring(body_h, body_w), _emitter_mask(rng, 'open', body_h, body_w),
             _emitter_mask(rng, 'decoy', body_h, body_w + 1),
             _emitter_mask(rng, 'multi', 5, int(rng.integers(3, 8)))]
    minimum = 3 * body_h + 5 + 3
    h = int(rng.integers(max(minimum + 1, len(mapping)), 31))
    col = int(rng.integers(max(m.shape[1] for m in masks) + 4, 18))
    band = int(rng.integers(9, 30 - col))
    a = _new_canvas(rng, h, col, band, mapping)
    extras = _partition(rng, h - minimum, 5)
    row = extras[0]
    for band_index, i in enumerate(map(int, rng.permutation(4))):
        mask, color = masks[i], (2, 2, 2, 5)[i]
        _put_emitter(a, rng, mask, row, color, col,
                     'aligned' if i == 0 else 'move' if i == 1 else None)
        mh = mask.shape[0]
        receiver_color = 5 if i == 3 else 1
        if i in (0, 3):
            small = _receiver_mask(rng, 'open' if i == 0 else 'rectangle', mh, 3)
            _put_receiver(a, small, row, col + 1, receiver_color if i == 0 else 8)
            rw = int(rng.integers(5 if i == 3 else 3, band - 3))
            rec = _receiver_mask(rng, 'diamond' if i == 3 else 'profile', mh, rw)
            x = col + 5 + int(rng.integers(band - 4 - rec.shape[1] + 1))
        else:
            rec = _receiver_mask(rng, str(rng.choice(['rectangle', 'profile'])), mh,
                                 int(rng.integers(3, min(8, band + 1))))
            x = col + 1 + int(rng.integers(band - rec.shape[1] + 1))
        _put_receiver(a, rec, row, x, receiver_color)
        row += mh + 1 + extras[band_index + 1]
    return a


def _random_scene(rng, is_test=False):
    mapping = _mapping(rng, 'random')
    count = int(rng.integers(1, 6))
    kinds = [str(rng.choice(['multi', 'mixed'])) if is_test else 'active']
    kinds += list(map(str, rng.choice(['active', 'multi', 'mixed', 'open', 'decoy', 'solid'], count - 1)))
    rng.shuffle(kinds)
    witness = next(i for i, k in enumerate(kinds) if k in ('active', 'multi', 'mixed'))
    minimums = [5 if k in ('multi', 'mixed') else 3 for k in kinds]
    base = sum(minimums) + count - 1
    h = int(rng.integers(max(base + 1, len(mapping), 8), 31))
    budget = min(h - base - 1, 8)
    additions = _partition(rng, budget, count)
    heights = [v + n for v, n in zip(minimums, additions)]
    widths = [int(rng.integers(6 if k == 'mixed' else 4 if k == 'decoy' else 3, 10)) for k in kinds]
    col = int(rng.integers(max(widths) + 4, 20))
    band = int(rng.integers(7, 30 - col))
    a = _new_canvas(rng, h, col, band, mapping)
    extras = _partition(rng, h - sum(heights) - count + 1, count + 1)
    row, emitted = extras[0], []
    channel_pool = list(mapping.values())
    for i, (kind, hh, ww) in enumerate(zip(kinds, heights, widths)):
        mask = _emitter_mask(rng, kind, hh, ww)
        color = int(rng.choice(channel_pool if i == witness or rng.random() < .8 else np.arange(1, 9)))
        _put_emitter(a, rng, mask, row, color, col)
        for hole in _holes(mask):
            if any(c == ww - 2 for r, c in hole):
                emitted.extend((row + r, color) for r in sorted({r for r, c in hole}))
        row += hh + 1 + extras[i + 1]
    blocked = np.zeros((h, band), dtype=bool)

    def place(mask, color, hit_row=None):
        mh, mw = mask.shape
        allowed_rows = range(h - mh + 1)
        if hit_row is not None:
            local_rows = {r for hole in _holes(mask) for r, c in hole}
            allowed_rows = [r for r in allowed_rows if hit_row - r in local_rows]
        choices = [(r, c) for r in allowed_rows for c in range(band - mw + 1)
                   if not np.any(blocked[r:r + mh, c:c + mw])]
        if not choices:
            return False
        r, c = choices[int(rng.integers(len(choices)))]
        _put_receiver(a, mask, r, c + col + 1, color)
        blocked[max(0, r - 1):min(h, r + mh + 1), max(0, c - 1):min(band, c + mw + 1)] = True
        return True

    deliverable = [(r, e) for r, e in emitted if e in channel_pool]
    hit_row, channel = deliverable[int(rng.integers(len(deliverable)))]
    receiver_color = int(rng.choice([c for c, e in mapping.items() if e == channel]))
    # At least one genuine delivery; the remainder may be silent, open or shadowed.
    kind = str(rng.choice(['rectangle', 'profile', 'diamond', 'multi']))
    mask = _receiver_mask(rng, kind, int(rng.integers(3, min(h, 9) + 1)),
                          int(rng.integers(3, min(band, 8) + 1)))
    if not place(mask, receiver_color, hit_row):
        # A full-height partition can put a divider on the requested row. A
        # three-cell ring always fits around an interior emission row.
        _require(place(_ring(3, 3), receiver_color, hit_row), "guaranteed receiver failed to fit")
    wanted = int(rng.integers(2, 9))
    for _ in range(32):
        if wanted <= 1:
            break
        color = int(rng.choice(list(mapping) if rng.random() < .7 else np.arange(1, 9)))
        kind = str(rng.choice(['rectangle', 'profile', 'profile', 'diamond', 'multi', 'open']))
        mask = _receiver_mask(rng, kind, int(rng.integers(3, min(h, 9) + 1)),
                              int(rng.integers(3, min(band, 8) + 1)))
        if place(mask, color):
            wanted -= 1
    return a


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_signature(puzzle):
    return tuple((_signature(p['input']), _signature(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    a = _integer(num_train, 'num_train', 2, 10)
    b = _integer(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    base = int(rng.integers(4))
    regime = str(rng.choice(['fixed', 'mixed', 'novel']))
    lessons = ['random'] * a
    indices = list(map(int, rng.permutation(a)))
    lessons[indices[0]], lessons[indices[1]] = 'route', 'gates'
    turns = [base] * (a + b)
    if regime != 'fixed':
        turns[indices[1]] = (base + 1) % 4
        for i in range(a):
            if i not in indices[:2]:
                turns[i] = (base + int(rng.integers(2))) % 4
        for i in range(a, a + b):
            turns[i] = (base + int(rng.integers(2, 4) if regime == 'novel' else rng.integers(4))) % 4
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(a + b):
        lesson = lessons[i] if i < a else 'random'
        raw = (_route_scene(rng) if lesson == 'route' else _gate_scene(rng) if lesson == 'gates'
               else _random_scene(rng, i >= a))
        # Cyclic offsets plus reflected offsets give 20 different palettes for
        # a scene with >=3 colors, exceeding the <=13 earlier input exclusions.
        palette = np.asarray(rng.permutation(10), dtype=np.int8)
        for offset in range(10):
            inp = np.rot90((palette[raw] + offset) % 10, turns[i]).copy()
            key = _signature(inp)
            if key not in seen:
                break
        else:
            # With 14 pairs, a second color orbit gives a bounded larger fallback.
            for offset in range(10):
                inp = np.rot90((9 - palette[raw] + offset) % 10, turns[i]).copy()
                key = _signature(inp)
                if key not in seen:
                    break
            else:
                raise AssertionError('uniqueness orbit exhausted')
        seen.add(key)
        puzzle['train' if i < a else 'test'].append({'input': inp, 'output': solve(inp)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit(scene):
    expected = _render(scene)
    box_scene = dict(scene)
    box_scene['cavities'] = list(scene['cavities'])
    for obj in scene['receivers']:
        if obj['holes'] or min(obj['mask'].shape) < 3:
            continue
        cells = [(r + obj['row'], c + obj['col'])
                 for r in range(1, obj['mask'].shape[0] - 1)
                 for c in range(1, obj['mask'].shape[1] - 1) if not obj['mask'][r, c]]
        fronts = {}
        for r, c in cells:
            fronts[r] = min(c, fronts.get(r, c))
        box_scene['cavities'].append(dict(cells=cells, fronts=fronts,
                                         color=obj['color'], channel=scene['mapping'].get(obj['color'])))
    alternatives = {
        'ignore_A': _render(scene, fill_all=True),
        'ignore_B': _render(scene, use_key=False),
        'all_holes': _render(scene, gate='holes'),
        'one_column': _render(scene, gate='column'),
        'no_shadow': _render(scene, nearest=False),
        'row_fill': _render(scene, rows_only=True),
        'recursive': _render(scene, recursive=True),
        'receiver_box': _render(box_scene),
    }
    return {k: not np.array_equal(v, expected) for k, v in alternatives.items()}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, witnesses, train_turns, test_turns = set(), set(), set(), set()
    positive, moving, aligned, multi, diagonal = False, False, False, False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "shape changed")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            scene = _read_scene(inp)
            _require(np.array_equal(_render(scene), out), "output violates transmission rule")
            canon_out = np.rot90(out, scene['turn'])
            col, bg = scene['col'], scene['bg']
            _require(np.array_equal(canon_out[:, col], scene['grid'][:, col]), "rail changed")
            for q in scene['key']:
                _require(canon_out[q] == scene['grid'][q], "key changed")
            for obj in scene['receivers']:
                _require(all(canon_out[q] == scene['grid'][q] for q in obj['cells']), "receiver ink changed")
            # Translation preserves each component, not just its bounding box.
            for obj in scene['emitters']:
                _require(all(canon_out[r, c + obj['shift']] == obj['color'] for r, c in obj['cells']),
                         "emitter lost shape or color")
                for hole in obj['holes']:
                    _require(all(canon_out[r + obj['row'], c + obj['col'] + obj['shift']] == bg
                                 for r, c in hole), "emitter hole filled")
            if split == 'train':
                train_turns.add(scene['turn'])
                audit = _audit(scene)
                witnesses.update(k for k, v in audit.items() if v)
                selected, _ = _select(scene, _events(scene))
                positive |= bool(selected)
                moving |= any(o['shift'] > 0 for o in scene['emitters'])
                aligned |= any(o['shift'] == 0 and o['active'] for o in scene['emitters'])
                multi |= any(len(o['active']) >= 2 for o in scene['emitters'])
                selected_cells = {q for i in selected for q in scene['cavities'][i]['cells']}
                diagonal |= any(len(_components(o['mask'])) > 1 and any(
                    (r + o['row'], c + o['col']) in selected_cells for hole in o['holes'] for r, c in hole)
                    for o in scene['receivers'])
            else:
                test_turns.add(scene['turn'])
    _require({'ignore_A', 'ignore_B', 'all_holes', 'one_column', 'no_shadow', 'row_fill', 'recursive', 'receiver_box'} <= witnesses,
             "training misses a distinguishing rule witness")
    _require(positive and moving and aligned and multi and diagonal,
             "training misses motion/active/aligned/multiple/diagonal evidence")
    _require(test_turns <= train_turns or len(train_turns) >= 2, "novel direction needs two training directions")


def self_test():
    """Bounded geometry, ablation, API, equivariance and diversity checks."""
    puzzles, shapes, masks, maps, directions, records = [], set(), set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene = _read_scene(pair['input'])
                shapes.add(pair['input'].shape)
                maps.add(tuple(sorted(scene['mapping'].items())))
                directions.add(scene['turn'])
                masks.update(_signature(o['mask'].astype(np.int8)) for o in scene['emitters'] + scene['receivers'])
                selected, hits = _select(scene, _events(scene))
                records.append(dict(emitters=len(scene['emitters']), receivers=len(scene['receivers']),
                                    key=scene['key_length'], bg=scene['bg'], selected=len(selected),
                                    multi=max(len(o['holes']) for o in scene['emitters'] + scene['receivers']),
                                    irregular=any(len(hole) != (max(r for r, c in hole) - min(r for r, c in hole) + 1) *
                                                  (max(c for r, c in hole) - min(c for r, c in hole) + 1)
                                                  for o in scene['receivers'] for hole in o['holes']),
                                    diagonal=any(len(_components(o['mask'])) > 1 and o['holes']
                                                 for o in scene['receivers'])))
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, "puzzle diversity collapsed")
    _require(len(shapes) >= 45 and len(masks) >= 100 and len(maps) >= 100, "structural diversity collapsed")
    _require(directions == set(range(4)), "missing orientation")
    _require({r['key'] for r in records} == {3, 4, 5, 6}, "key lengths collapsed")
    _require({r['bg'] for r in records} == set(range(10)), "background palette collapsed")
    _require({1, 2, 3, 4, 5} <= {r['emitters'] for r in records}, "emitter counts collapsed")
    _require(max(r['receivers'] for r in records) >= 6, "receiver counts collapsed")
    _require(any(r['irregular'] for r in records) and any(r['diagonal'] for r in records), "topology collapsed")
    _require(max(r['multi'] for r in records) >= 3, "multiple holes missing")
    _require(_puzzle_signature(generate_puzzle(0)) == _puzzle_signature(puzzles[0]), "nondeterminism")
    # A hand-specified transmission: one single-row message fills all three
    # cells of a taller receiver. Wrong-channel, shadowed, open-source and
    # recessed-source recipients remain empty. Expected edits are explicit.
    inp = np.zeros((11, 27), dtype=np.int8)
    inp[:, 9] = 9
    inp[:3, :2] = [[1, 2], [2, 4], [3, 2]]
    closed = _ring(3, 3)
    opened = closed.copy()
    opened[0, 1] = False
    recessed = np.pad(closed, ((0, 0), (0, 1)))
    recessed[0, 3] = True
    for mask, row in ((opened, 0), (closed, 4), (recessed, 8)):
        inp[row:row + 3, 3:3 + mask.shape[1]][mask] = 2
    rr, cc = np.indices((3, 3))
    _put_receiver(inp, opened, 4, 10, 1)
    _put_receiver(inp, abs(rr - 1) + abs(cc - 1) == 1, 4, 14, 2)
    _put_receiver(inp, _ring(5, 3), 2, 18, 1)
    _put_receiver(inp, _ring(3, 3), 4, 22, 3)
    _put_receiver(inp, _ring(3, 3), 0, 22, 1)
    _put_receiver(inp, _ring(3, 3), 8, 10, 1)
    expected = inp.copy()
    expected[:, 3:7] = 0
    expected[0:3, 6:9][opened] = 2
    expected[4:7, 6:9][closed] = 2
    expected[8:11, 5:9][recessed] = 2
    expected[3:6, 19] = 2
    _require(np.array_equal(solve(inp), expected), "explicit transmission fixture disagrees")
    _require(all(_audit(_read_scene(inp)).values()), "explicit fixture lost an ablation contrast")
    # Complete rotation and a fresh permutation must commute with the rule.
    sample = puzzles[0]['train'][0]
    permutation = np.array([3, 0, 8, 1, 9, 4, 2, 7, 5, 6], dtype=np.int8)
    for turn in range(4):
        inp, expected = [np.rot90(permutation[sample[k]], turn).copy() for k in ('input', 'output')]
        _require(np.array_equal(solve(inp), expected), "rotation/palette equivariance failed")
    # Every legal explicit count combination; no None or float silently accepted.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            _require((len(p['train']), len(p['test'])) == (a, b), "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(records), grid_shapes=len(shapes), object_masks=len(masks),
                mappings=len(maps), directions=sorted(directions),
                emitter_counts=sorted({r['emitters'] for r in records}),
                receiver_counts=sorted({r['receivers'] for r in records}),
                diagonal_pairs=sum(bool(r['diagonal']) for r in records),
                irregular_receiver_pairs=sum(bool(r['irregular']) for r in records))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
