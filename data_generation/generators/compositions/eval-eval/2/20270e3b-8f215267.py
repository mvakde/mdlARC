"""Repair census: register transparent films, count jointly supported objects.

Only NumPy and the standard library are used. Generation retains no global RNG,
examples or answer state. No files are read or written by this module.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(bool(np.all((g >= 0) & (g <= 9))), "invalid color")
    return g


def _components(cells, diagonal=False):
    remaining = set(cells)
    result = []
    directions = _EIGHT if diagonal else _FOUR
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, part = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _box(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _perimeter(box):
    t, b, l, r = box
    return ({(t, c) for c in range(l, r + 1)} |
            {(b, c) for c in range(l, r + 1)} |
            {(y, l) for y in range(t, b + 1)} |
            {(y, r) for y in range(t, b + 1)})


def _translate(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _scene(input_grid):
    """Parse carriers, tabs and receivers exclusively from input pixels."""
    g = _grid(input_grid)
    colors, frequencies = np.unique(g, return_counts=True)
    _require(np.count_nonzero(frequencies == frequencies.max()) == 1,
             "background must be the unique most frequent color")
    bg = int(colors[np.argmax(frequencies)])
    parts = []
    for color in colors:
        color = int(color)
        if color != bg:
            for cells in _components(map(tuple, np.argwhere(g == color))):
                parts.append(dict(color=color, cells=cells, box=_box(cells)))
    holders = []
    for part in parts:
        t, b, l, r = part['box']
        if min(b - t + 1, r - l + 1) < 6:
            continue
        perimeter = _perimeter(part['box'])
        missing = perimeter - part['cells']
        if not part['cells'] <= perimeter or not 2 <= len(missing) <= 5:
            continue
        corners = {(t, l), (t, r), (b, l), (b, r)}
        if missing & corners or len(_components(missing)) != 1:
            continue
        sides = [s for s, ok in enumerate((all(y == t for y, x in missing),
                                          all(x == r for y, x in missing),
                                          all(y == b for y, x in missing),
                                          all(x == l for y, x in missing))) if ok]
        if len(sides) != 1:
            continue
        tab_colors = {int(g[q]) for q in missing}
        if len(tab_colors) != 1 or bg in tab_colors or part['color'] in tab_colors:
            continue
        holders.append(dict(**part, tab=missing, tab_color=tab_colors.pop(), side=sides[0]))
    _require(len(holders) == 2, "exactly two rectangular film holders required")
    _require(len({(h['color'], h['tab_color'], h['side'], len(h['tab'])) for h in holders}) == 1,
             "film holders must share structural colors and congruent same-side tabs")
    structural, guide = holders[0]['color'], holders[0]['tab_color']
    occupied = set()
    layers = []
    for h in holders:
        t, b, l, r = h['box']
        rect = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}
        _require(not occupied & rect, "overlapping source holders")
        occupied |= rect
        interior = {(y, x) for y in range(t + 1, b) for x in range(l + 1, r)}
        _require(all(int(g[q]) not in (structural, guide) for q in interior),
                 "structural colors cannot be cargo")
        origin = min(y for y, x in h['tab']), min(x for y, x in h['tab'])
        layer = {}
        for q in interior:
            color = int(g[q])
            if color != bg:
                layer.setdefault(color, set()).add((q[0] - origin[0], q[1] - origin[1]))
        _require(bool(layer), "empty film")
        layers.append(layer)
    all_structural = {q for p in parts if p['color'] == structural for q in p['cells']}
    all_guides = {q for p in parts if p['color'] == guide for q in p['cells']}
    _require(all_structural == set.union(*(h['cells'] for h in holders)), "extra carrier pixels")
    _require(all_guides == set.union(*(h['tab'] for h in holders)), "extra guide pixels")
    frames = []
    for p in parts:
        if p['cells'] & occupied:
            _require(p['cells'] <= occupied, "cargo crosses a holder border")
            continue
        t, b, l, r = p['box']
        height, width = b - t + 1, r - l + 1
        _require(p['cells'] == _perimeter(p['box']), "loose pixels outside films")
        _require(min(height, width) in (5, 7) and max(height, width) > min(height, width),
                 "invalid receiver dimensions")
        _require(bool(np.all(g[t + 1:b, l + 1:r] == bg)), "receiver must start empty")
        frames.append(p)
    _require(1 <= len(frames) <= 3, "one through three receiver frames required")
    _require(len({f['color'] for f in frames}) == len(frames), "duplicate receiver color")
    for i, frame in enumerate(frames):
        t, b, l, r = frame['box']
        for other in frames[i + 1:]:
            tt, bb, ll, rr = other['box']
            _require(b + 1 < tt or bb + 1 < t or r + 1 < ll or rr + 1 < l,
                     "receiver rectangles must have a background moat")
    verticals = {f['box'][1] - f['box'][0] > f['box'][3] - f['box'][2] for f in frames}
    _require(len(verticals) == 1, "receivers must have parallel long axes")
    vertical = verticals.pop()
    axis = 0 if vertical else 2
    low = min(h['box'][axis] for h in holders)
    high = max(h['box'][axis + 1] for h in holders)
    if all(f['box'][axis + 1] < low for f in frames):
        direction = 'down' if vertical else 'right'
    elif all(f['box'][axis] > high for f in frames):
        direction = 'up' if vertical else 'left'
    else:
        raise AssertionError("films must lie beyond one long-axis end of every receiver")
    # A transparent union is defined only for equal-color overlaps.
    owners = {}
    for layer in layers:
        for color, cells in layer.items():
            for q in cells:
                _require(q not in owners or owners[q] == color, "unlike-color film collision")
                owners[q] = color
    return dict(grid=g, background=bg, structural=structural, guide=guide,
                holders=holders, frames=frames, layers=layers, direction=direction)


def _census(detail, model='joint'):
    a, b = detail['layers']
    answer = {}
    for frame in detail['frames']:
        color = frame['color']
        x, y = a.get(color, set()), b.get(color, set())
        components = _components(x | y, diagonal=(model == 'eight'))
        if model in ('joint', 'eight'):
            count = sum(bool(p & x) and bool(p & y) for p in components)
        elif model == 'all_union':
            count = len(components)
        elif model == 'raw':
            count = len(_components(x)) + len(_components(y))
        elif model == 'area':
            count = len(x | y)
        elif model == 'intersection':
            count = len(_components(x & y))
        elif model == 'contacts':
            count = sum(bool(p & q) or any((r + dr, c + dc) in q
                        for r, c in p for dr, dc in _FOUR)
                        for p in _components(x) for q in _components(y))
        else:
            raise ValueError(model)
        answer[color] = int(count)
    return answer


def _corner_aligned_census(detail):
    """A concrete counterfactual: align holder corners instead of their tabs."""
    layers = []
    for layer, holder in zip(detail['layers'], detail['holders']):
        t, b, l, r = holder['box']
        dy = min(y for y, x in holder['tab']) - t
        dx = min(x for y, x in holder['tab']) - l
        layers.append({c: _translate(cells, dy, dx) for c, cells in layer.items()})
    return _census(dict(detail, layers=layers))


def _render(detail, counts):
    out = np.full(detail['grid'].shape, detail['background'], dtype=np.int8)
    for f in detail['frames']:
        color = f['color']
        for q in f['cells']:
            out[q] = color
        t, b, l, r = f['box']
        _require(counts[color] <= (max(b - t, r - l) - 2) // 2, "tally overflow")
        for i in range(1, counts[color] + 1):
            direction = detail['direction']
            row = b - 2 * i if direction == 'down' else t + 2 * i if direction == 'up' else (t + b) // 2
            col = r - 2 * i if direction == 'right' else l + 2 * i if direction == 'left' else (l + r) // 2
            out[row, col] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution: coincident tabs, color union, mixed support, tally."""
    detail = _scene(input_grid)
    return _render(detail, _census(detail))


def _polyomino(rng, area):
    cells = {(0, 0)}
    while len(cells) < area:
        candidates = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _FOUR} - cells)
        candidates = [q for q in candidates if
                      _box(cells | {q})[1] - _box(cells | {q})[0] < 4 and
                      _box(cells | {q})[3] - _box(cells | {q})[2] < 4]
        cells.add(candidates[int(rng.integers(len(candidates)))])
    return cells


def _tile(rng, color, a, b):
    a, b = set(a), set(b)
    for _ in range(int(rng.integers(4))):
        a, b = ({(c, -r) for r, c in a}, {(c, -r) for r, c in b})
    if rng.random() < .5:
        a, b = ({(r, -c) for r, c in a}, {(r, -c) for r, c in b})
    if rng.random() < .5:
        a, b = b, a
    t, _, l, _ = _box(a | b)
    return dict(color=color, a=_translate(a, -t, -l), b=_translate(b, -t, -l))


def _shared(rng, color, area=None, overlap=None):
    area = int(rng.integers(2, 8)) if area is None else area
    cells = _polyomino(rng, area)
    order = sorted(cells)
    cut = int(rng.integers(1, len(order)))
    order = [order[int(i)] for i in rng.permutation(len(order))]
    a, b = set(order[:cut]), set(order[cut:])
    if overlap is None:
        overlap = rng.random() < .4
    if overlap:
        a.add(order[cut])
    return _tile(rng, color, a, b)


def _lone(rng, color, area=None):
    area = int(rng.integers(1, 8)) if area is None else area
    return _tile(rng, color, _polyomino(rng, area), set())


def _tiles(rng, index, colors, noise, witness, test, count_shift):
    counts = {c: 0 for c in colors}
    tiles = []
    if index == 0 and not test:
        h, width, lone_area = witness
        a = {(r, c) for r in range(h) for c in (0, width - 1)}
        b = {(h - 1, c) for c in range(1, width - 1)}
        tiles.append(_tile(rng, colors[0], a, b))
        tiles.append(_lone(rng, colors[0], lone_area))
        # Two shared structures touch diagonally; the second also overlaps.
        tiles.append(_tile(rng, colors[1], {(0, 0), (1, 2), (2, 2)},
                           {(0, 1), (1, 2), (1, 3)}))
        counts[colors[0]], counts[colors[1]] = 1, 2
    elif index == 1 and not test:
        h, width, lone_area = witness
        area = 2 * h + width - 2
        first = int(rng.integers(2, area - 1))
        tiles.extend([_shared(rng, colors[0], first, False),
                      _shared(rng, colors[0], area - first, False),
                      _lone(rng, colors[0], lone_area)])
        tiles.append(_tile(rng, colors[1], {(0, 0)}, {(1, 1)}))
        counts[colors[0]], counts[colors[1]] = 2, 0
    else:
        maximum = 2 if count_shift and not test else 4
        values = [int(rng.integers(maximum + 1)) for c in colors]
        if test and count_shift:
            values[0] = int(rng.integers(3, 5))
        while sum(values) > 5:
            j = max(range(1 if test and count_shift else 0, len(values)), key=lambda j: values[j], default=0)
            values[j] -= 1
        counts = dict(zip(colors, values))
        for color, number in counts.items():
            tiles.extend(_shared(rng, color) for _ in range(number))
        tiles.append(_lone(rng, int(rng.choice(colors))))
        if rng.random() < .25:
            tiles.append(_lone(rng, int(rng.choice(colors))))
    # An unmatched *shared* structure proves that the current frames select colors.
    tiles.append(_shared(rng, noise, int(rng.integers(2, 5))))
    return tiles, counts


def _pack(rng, tiles):
    """Free pixel placement; no invisible object lattice; at most 36 attempts."""
    for attempt in range(36):
        h = int(rng.integers(8, 12)) if attempt < 12 else 11
        w = int(rng.integers(8, 12)) if attempt < 12 else 11
        used = np.zeros((h, w), dtype=bool)
        layers = [{}, {}]
        order = sorted(range(len(tiles)), key=lambda i: (-len(tiles[i]['a'] | tiles[i]['b']), float(rng.random())))
        for i in order:
            tile = tiles[i]
            cells = tile['a'] | tile['b']
            _, bottom, _, right = _box(cells)
            valid = np.ones((h - bottom, w - right), dtype=bool)
            for dr, dc in cells:
                valid &= ~used[dr:dr + h - bottom, dc:dc + w - right]
            rr, cc = np.nonzero(valid)
            if not len(rr):
                break
            j = int(rng.integers(len(rr)))
            row, col = int(rr[j]), int(cc[j])
            for layer, key in zip(layers, ('a', 'b')):
                layer.setdefault(tile['color'], set()).update(_translate(tile[key], row + 1, col + 1))
            for dr, dc in cells:
                r, c = row + dr, col + dc
                used[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
        else:
            return layers, h, w
    return None


def _films(rng, layers, h, w, length, palette, rotation):
    bg, carrier, guide = palette
    start = int(rng.integers(1, w - length + 2))
    result = []
    for i, layer in enumerate(layers):
        cells = set().union(*layer.values())
        t, b, l, r = _box(cells)
        # Independent window bounds; unequal left padding makes tab alignment
        # visibly different from aligning rectangular holder corners.
        left = -1 if i == 0 else int(rng.integers(0, min(l, start)))
        right = int(rng.integers(max(r, start + length - 1) + 1, w + 3))
        right = min(right, left + 13)
        right = max(right, left + 5)
        if right <= max(r, start + length - 1):
            return None
        bottom = min(13, max(5, b + int(rng.integers(1, 4))))
        grid = np.full((bottom + 1, right - left + 1), bg, dtype=np.int8)
        grid[0, :] = grid[-1, :] = carrier
        grid[:, 0] = grid[:, -1] = carrier
        grid[0, start - left:start - left + length] = guide
        for color, part in layer.items():
            for y, x in part:
                grid[y, x - left] = color
        result.append(np.rot90(grid, rotation).copy())
    return result


def _draw_frame(grid, box, color):
    for q in _perimeter(box):
        grid[q] = color


def _stage(rng, films, colors, counts, bg, teaching=False):
    shorts = [int(rng.choice((5, 5, 7))) for c in colors]
    lengths = [min(13, max(7, s + 2, 2 * counts[c] + (5 if teaching else 3)) + int(rng.integers(3)))
               for c, s in zip(colors, shorts)]
    gaps = [int(rng.integers(1, 4)) for _ in colors[1:]]
    bank_height = sum(shorts) + sum(gaps)
    offsets = [int(rng.integers(3)) for c in colors]
    bank_width = max(a + b for a, b in zip(offsets, lengths))
    if rng.random() < .5:
        films = films[::-1]
    ah, aw = films[0].shape
    bh, bw = films[1].shape
    # Horizontal and staggered vertical source staging are both legal.
    if aw + bw + 1 + bank_width + 1 <= 30 and rng.random() < .5:
        gap = int(rng.integers(1, min(3, 30 - aw - bw - bank_width - 1) + 1))
        positions = [(0, 0), (int(rng.integers(3)), aw + gap)]
    else:
        gap = int(rng.integers(1, min(3, 30 - ah - bh) + 1))
        positions = [(0, int(rng.integers(2))), (ah + gap, int(rng.integers(2)))]
    field_h = max(r + film.shape[0] for (r, c), film in zip(positions, films))
    field_w = max(c + film.shape[1] for (r, c), film in zip(positions, films))
    gap = int(rng.integers(1, 3))
    height, width = max(bank_height, field_h), bank_width + gap + field_w
    if max(height, width) > 30:
        return None
    pads = [int(rng.integers(0, min(2, 30 - d) + 1)) for d in (height, width)]
    height, width = height + pads[0], width + pads[1]
    grid, out = (np.full((height, width), bg, dtype=np.int8) for _ in range(2))
    bank_top = int(rng.integers(height - bank_height + 1))
    field_top = int(rng.integers(height - field_h + 1))
    for (r, c), film in zip(positions, films):
        r, c = r + field_top, c + bank_width + gap
        grid[r:r + film.shape[0], c:c + film.shape[1]] = film
    row = bank_top
    for i, color in enumerate(colors):
        col = offsets[i]
        box = row, row + shorts[i] - 1, col, col + lengths[i] - 1
        _draw_frame(grid, box, color)
        _draw_frame(out, box, color)
        for j in range(1, counts[color] + 1):
            out[row + shorts[i] // 2, col + lengths[i] - 1 - 2 * j] = color
        row += shorts[i] + (gaps[i] if i < len(gaps) else 0)
    return grid, out


def _make_pair(rng, index, colors, noise, witness, test, count_shift,
               length, palette, film_rotation, scene_rotation, seen):
    for _ in range(48):
        tiles, counts = _tiles(rng, index, colors, noise, witness, test, count_shift)
        packed = _pack(rng, tiles)
        if packed is None:
            continue
        layers, h, w = packed
        films = _films(rng, layers, h, w, length, palette, film_rotation)
        if films is None:
            continue
        staged = _stage(rng, films, colors, counts, palette[0], teaching=not test and index < 2)
        if staged is None:
            continue
        inp, out = (np.rot90(g, scene_rotation).copy() for g in staged)
        key = (inp.shape, inp.tobytes())
        if key in seen:
            continue
        detail = _scene(inp)
        _require(_census(detail) == counts, "construction changed intended connectivity")
        _require(np.array_equal(_render(detail, counts), out), "builder/solver disagreement")
        if not test and index < 2 and _corner_aligned_census(detail) == counts:
            continue
        return {'input': inp, 'output': out}
    raise AssertionError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    nt, nq = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require 2..10 training, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    persistent, novel, former, second = [int(x) for x in rng.choice(10, 4, replace=False)]
    witness = int(rng.integers(2, 4)), int(rng.integers(3, 6)), int(rng.integers(2, 5))
    count_shift = bool(rng.integers(2))
    base_rotation, base_film = int(rng.integers(4)), int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(nt + nq):
        test = i >= nt
        j = i - nt if test else i
        if not test and i == 0:
            colors, noise = [persistent, former], novel
        elif not test and i == 1:
            colors, noise = [persistent, second], former
        else:
            n = int(rng.integers(1, 4))
            pool = [c for c in range(10) if test or c != novel]
            colors = [int(x) for x in rng.choice(pool, n, replace=False)]
            if test and j == 0:
                colors[0] = novel
                colors = list(dict.fromkeys(colors))
            noise = int(rng.choice([c for c in range(10) if c not in colors]))
        available = [c for c in range(10) if c not in colors and c != noise]
        palette = tuple(int(x) for x in rng.choice(available, 3, replace=False))
        # 2/4 -> 3 tab-length interpolation in one regime, count extrapolation
        # in the other. Neither is an invisible solving parameter.
        length = (2 if j % 2 == 0 else 4) if not test else 3
        if count_shift:
            length = int(rng.integers(2, 6))
        rotation = ((base_rotation + 2 * i) % 4 if i < 2 else
                    (base_rotation + int(rng.choice((1, 3)))) % 4 if test and j == 0 else
                    int(rng.integers(4)))
        film_rotation = (base_film + i) % 4 if i < 2 else int(rng.integers(4))
        pair = _make_pair(rng, j, colors, noise, witness, test, count_shift,
                          length, palette, film_rotation, rotation, seen)
        seen.add((pair['input'].shape, pair['input'].tobytes()))
        puzzle['test' if test else 'train'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(details):
    observations = []
    for d in details:
        counts = _census(d)
        for f in d['frames']:
            c = f['color']
            a, b = (layer.get(c, set()) for layer in d['layers'])
            observations.append((c, counts[c], len(a | b), len(a) + len(b)))
    _require(any(c == cc and n != nn and area == aa and raw == rr
                 for c, n, area, raw in observations for cc, nn, aa, rr in observations),
             "same-color equal-area different-count witness missing")
    _require(any(n == nn and area != aa for c, n, area, raw in observations
                 for cc, nn, aa, rr in observations), "same-count different-area witness missing")
    _require({0, 1, 2} <= {n for c, n, area, raw in observations}, "unary 0/1/2 evidence missing")
    for model in ('raw', 'all_union', 'area', 'intersection', 'contacts', 'eight'):
        _require(any(_census(d, model) != _census(d) for d in details),
                 f"training does not distinguish {model}")
    _require(any(_corner_aligned_census(d) != _census(d) for d in details),
             "training does not distinguish corner alignment from registration")
    opposite = dict(up='down', down='up', left='right', right='left')
    informative = {d['direction'] for d in details if not np.array_equal(
        _render(d, _census(d)),
        _render(dict(d, direction=opposite[d['direction']]), _census(d)))}
    _require(any(opposite[direction] in informative for direction in informative),
             "opposite field directions need asymmetric teaching tallies")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and
             len(puzzle['train']) > len(puzzle['test']), "pair counts")
    seen, details = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = (inp.shape, inp.astype(np.int8).tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _scene(inp)
            _require(np.array_equal(_render(d, _census(d)), out), "wrong census, erasure or tally")
            _require(not np.array_equal(inp, out), "unchanged example")
            details[split].append(d)
    _training_evidence(details['train'])
    categories = {f['color'] for d in details['train'] for f in d['frames']}
    discarded = {c for d in details['train'] for layer in d['layers'] for c in layer
                 if c not in {f['color'] for f in d['frames']}}
    _require(bool(categories & discarded), "local category/distractor switching missing")
    _require(any(f['color'] not in categories for d in details['test'] for f in d['frames']),
             "new test category missing")
    _require(len({d['direction'] for d in details['train']}) >= 2, "field-relative tally teaching missing")


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for key in ('input', 'output'):
                g = np.asarray(pair[key], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, streaming metrics, interface edges, semantic mutation checks."""
    signatures, shapes, palettes, counts, lengths, directions = (set() for _ in range(6))
    source_shapes, masks, offsets, guide_sides = (set() for _ in range(4))
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pair_count += 1
                inp = pair['input'].copy()
                _require(np.array_equal(solve(inp), pair['output']), "input-only solver")
                _require(np.array_equal(inp, pair['input']), "solver mutated input")
                d = _scene(inp)
                shapes.add(inp.shape)
                counts.update(_census(d).values())
                directions.add(d['direction'])
                palettes.add((d['background'], d['structural'], d['guide'],
                              tuple(f['color'] for f in d['frames'])))
                guide_sides.add(d['holders'][0]['side'])
                for h in d['holders']:
                    t, b, l, r = h['box']
                    source_shapes.add((b - t + 1, r - l + 1))
                    lengths.add(len(h['tab']))
                    offsets.add((min(y for y, x in h['tab']) - t,
                                 min(x for y, x in h['tab']) - l))
                for layer in d['layers']:
                    for color, cells in layer.items():
                        for p in _components(cells):
                            t, b, l, r = _box(p)
                            masks.add(tuple(sorted((y - t, x - l) for y, x in p)))
    _require(len(signatures) == 42 and pair_count == 210, "seed diversity")
    _require(len(shapes) >= 30 and len(palettes) >= 100, "scene/palette diversity")
    _require(len(source_shapes) >= 20 and len(masks) >= 40, "structural variation")
    _require(lengths == {2, 3, 4, 5} and guide_sides == {0, 1, 2, 3}, "tab geometry coverage")
    _require(directions == {'up', 'down', 'left', 'right'} and counts >= {0, 1, 2, 3, 4}, "count/orientation coverage")
    _require(len(offsets) >= 12, "registration offsets")
    _require(_fingerprint(generate_puzzle(517)) == _fingerprint(generate_puzzle(517)), "determinism")
    for nt, nq in ((2, 1), (10, 4), (5, 4)):
        p = generate_puzzle(2026, nt, nq)
        _require((len(p['train']), len(p['test'])) == (nt, nq), "explicit counts")
    # Sparse films must retain the minimum holder width after independent trim.
    validate_generated_puzzle(generate_puzzle(1017, 10, 4))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    d = _scene(p['train'][0]['input'])
    # The validator must reject an actual alternative label, not just bad types.
    wrong = _render(d, {c: 0 for c in _census(d)})
    p['train'][0]['output'] = wrong
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("zeroed tallies accepted")
    return dict(seeds=42, pairs=pair_count, input_shapes=len(shapes),
                palettes=len(palettes), holder_shapes=len(source_shapes),
                source_component_masks=len(masks), registration_offsets=len(offsets))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
