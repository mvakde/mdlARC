"""Component-powered stencil sweeps. Standard library and NumPy only.

No files, generation state, seed, or examples are consulted by solve().
Execution and self-tests retain no grid files or traces.
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # top, right, bottom, left


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "illegal color")
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _components(mask, diagonal=False):
    """Deterministic flood fill, returning coordinates of individual objects."""
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if (dy or dx) and (diagonal or abs(dy) + abs(dx) == 1)]
    for yy, xx in np.argwhere(mask):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        group, stack = [], [(y, x)]
        while stack:
            p, q = stack.pop()
            group.append((p, q))
            for dy, dx in steps:
                r, c = p + dy, q + dx
                if (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]
                        and mask[r, c] and not seen[r, c]):
                    seen[r, c] = True
                    stack.append((r, c))
        groups.append(np.array(group, dtype=int))
    return groups


def _span(mask):
    return bool(mask[0].any() and mask[-1].any()
                and mask[:, 0].any() and mask[:, -1].any())


def _sweep(mask, n, direction):
    """Union of n translated ORIGINAL masks, clipped to the interior."""
    h, w = mask.shape
    dy, dx = _DIRECTIONS[direction]
    holes = np.zeros_like(mask)
    # Once an original mask has entirely left the window, later copies are empty.
    for step in range(min(n, h if dy else w)):
        oy, ox = step * dy, step * dx
        y0, y1 = max(0, oy), min(h, h + oy)
        x0, x1 = max(0, ox), min(w, w + ox)
        holes[y0:y1, x0:x1] |= mask[y0-oy:y1-oy, x0-ox:x1-ox]
    return holes


def _point_sweep(mask, n, direction):
    """Construction/validation arithmetic, separate from solve's slice union."""
    holes = np.zeros_like(mask)
    dy, dx = _DIRECTIONS[direction]
    for r, c in np.argwhere(mask):
        for step in range(n):
            y, x = int(r) + step * dy, int(c) + step * dx
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                holes[y, x] = True
    return holes


def _decode(input_grid):
    """Geometry-only parse: complete background border, gated plates, registers."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "input must have a uniform background border")
    tiles = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        for cells in _components(a == color):
            y, x = map(int, cells.min(axis=0))
            bottom, right = map(int, cells.max(axis=0) + 1)
            h, w = bottom - y, right - x
            if min(h, w) < 5 or len(cells) != h * w - 1:
                continue
            region = a[y:bottom, x:right]
            missing = np.argwhere(region != color)
            if len(missing) != 1:
                continue
            gy, gx = map(int, missing[0])
            marker = int(region[gy, gx])
            if marker in (bg, color):
                continue
            if gy == 0 and 0 < gx < w-1:
                direction = 0
            elif gx == w-1 and 0 < gy < h-1:
                direction = 1
            elif gy == h-1 and 0 < gx < w-1:
                direction = 2
            elif gx == 0 and 0 < gy < h-1:
                direction = 3
            else:
                continue
            tiles.append({"y": y, "x": x, "h": h, "w": w,
                          "color": color, "marker": marker, "direction": direction})
    _require(bool(tiles), "no gated plate")
    tiles.sort(key=lambda t: (t["y"], t["x"]))
    _require(len({(t['h'], t['w']) for t in tiles}) == 1, "unequal plates")
    _require(len({t['marker'] for t in tiles}) == 1, "marker color must agree")
    colors = [t['color'] for t in tiles]
    marker = tiles[0]['marker']
    _require(len(set(colors)) == len(tiles) and marker not in colors, "ambiguous palette")
    th, tw = tiles[0]['h'], tiles[0]['w']
    ys, xs = sorted({t['y'] for t in tiles}), sorted({t['x'] for t in tiles})
    _require(all(b-a == step for seq, step in ((ys, th+1), (xs, tw+1))
                 for a, b in zip(seq, seq[1:])), "irregular plate spacing")
    _require({(t['y'], t['x']) for t in tiles} == {(y, x) for y in ys for x in xs},
             "incomplete plate array")
    y0, x0, y1, x1 = ys[0]-1, xs[0]-1, ys[-1]+th+1, xs[-1]+tw+1
    _require(0 <= y0 < y1 <= a.shape[0] and 0 <= x0 < x1 <= a.shape[1], "crop margin")
    crop_mask = np.zeros(a.shape, dtype=bool)
    crop_mask[y0:y1, x0:x1] = True
    plate_mask = np.zeros(a.shape, dtype=bool)
    for t in tiles:
        plate_mask[t['y']:t['y']+th, t['x']:t['x']+tw] = True
    _require(bool(np.all(a[crop_mask & ~plate_mask] == bg)), "non-background gutter")
    outside = ~crop_mask & (a != bg)
    _require(not np.any(a[outside] == marker), "external marker ink")
    patterns, boxes, counts = {}, {}, {}
    for color in map(int, np.unique(a[outside])):
        points = np.argwhere(outside & (a == color))
        top, left = map(int, points.min(axis=0))
        bottom, right = map(int, points.max(axis=0) + 1)
        _require((bottom-top, right-left) == (th-2, tw-2), "stencil must span full inset")
        _require(not crop_mask[top:bottom, left:right].any(), "stencil enters crop")
        register = a[top:bottom, left:right]
        _require(bool(np.all((register == bg) | (register == color))), "mixed stencil register")
        mask = register == color
        groups = _components(mask)
        _require(len(groups) == len(_components(mask, True)), "diagonal connectivity ambiguity")
        patterns[color], counts[color] = mask.copy(), len(groups)
        boxes[color] = (top, left, bottom, right)
    _require(set(colors) <= set(patterns), "missing same-color stencil")
    for i, first in enumerate(list(boxes.values())):
        for second in list(boxes.values())[i+1:]:
            _require(first[2] < second[0] or second[2] < first[0]
                     or first[3] < second[1] or second[3] < first[1], "registers need a blank gap")
    return {"input": a, "background": bg, "marker": marker, "tiles": tiles,
            "patterns": patterns, "counts": counts, "boxes": boxes,
            "crop": (y0, x0, y1, x1), "array": (len(ys), len(xs)),
            "inset": (th-2, tw-2), "distractors": len(patterns)-len(tiles)}


def _render(d, pointwise=False, alternative=None):
    y0, x0, y1, x1 = d['crop']
    out = d['input'][y0:y1, x0:x1].copy()
    ordered = sorted(d['patterns'], key=lambda c: d['boxes'][c][:2])
    all_mask = np.logical_or.reduce(list(d['patterns'].values()))
    for i, t in enumerate(d['tiles']):
        mask = d['patterns'][t['color']]
        n, direction = d['counts'][t['color']], t['direction']
        if alternative == 'single':
            n = 1
        elif alternative == 'area':
            n = int(mask.sum())
        elif alternative == 'plus_one':
            n += 1
        elif alternative == 'minus_one':
            n = max(1, n-1)
        elif alternative == 'fixed_two':
            n = 2
        elif alternative == 'fixed_three':
            n = 3
        elif alternative == 'reverse':
            direction = (direction + 2) % 4
        elif alternative == 'all_stencils':
            mask = all_mask
        elif alternative == 'position':
            mask = d['patterns'][ordered[i]]
            n = d['counts'][ordered[i]]
        elif alternative == 'holes_count':
            n = max(1, len(_components(~mask)))
        elif alternative == 'wrap':
            holes = np.zeros_like(mask)
            for step in range(n):
                dy, dx = _DIRECTIONS[direction]
                holes |= np.roll(mask, (step*dy, step*dx), axis=(0, 1))
        elif isinstance(alternative, int):
            mask = np.rot90(np.fliplr(mask) if alternative >= 4 else mask, alternative % 4)
            if mask.shape != d['inset']:
                return None
        if alternative != 'wrap':
            holes = (_point_sweep if pointwise else _sweep)(mask, n, direction)
        y, x = t['y']-y0+1, t['x']-x0+1
        interior = out[y:y+holes.shape[0], x:x+holes.shape[1]]
        if alternative == 'positive':
            interior[:] = d['background']
            interior[holes] = t['color']
        else:
            interior[holes] = d['background']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count each matched stencil's islands and sweep its negative toward its gate."""
    return _render(_decode(input_grid))


def _partition(length, pieces, rng):
    sizes = np.ones(pieces, dtype=int)
    for _ in range(length - (2*pieces-1)):
        sizes[int(rng.integers(pieces))] += 1
    starts, offset = [], 0
    for size in sizes:
        starts.append((offset, offset+int(size)))
        offset += int(size)+1
    return starts


def _connected_shape(h, w, rng):
    """Connected full-span bars, blocks, rings, crosses, branches and pruned ink."""
    mode = int(rng.integers(7))
    a = np.zeros((h, w), dtype=bool)
    if mode == 0 or min(h, w) == 1:
        a[:] = True
    elif mode == 1:
        a[0] = a[:, 0] = True
    elif mode == 2:
        a[0] = a[-1] = True
        a[:, 0] = a[:, -1] = True
    elif mode == 3:
        a[int(rng.integers(h))] = True
        a[:, int(rng.integers(w))] = True
    elif mode == 4:
        a[::2] = True
        for r in range(1, h, 2):
            a[r, 0 if (r//2) % 2 else w-1] = True
    elif mode == 5:
        py, px = int(rng.integers(h)), int(rng.integers(w))
        for y, x in ((0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                     (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)):
            a[y, x] = True
            while (y, x) != (py, px):
                choices = []
                if y != py:
                    choices.append((y + (1 if y < py else -1), x))
                if x != px:
                    choices.append((y, x + (1 if x < px else -1)))
                y, x = _pick(rng, choices)
                a[y, x] = True
    else:
        a[:] = True
        target = int(rng.integers(max(h, w), h*w+1))
        for flat in rng.permutation(h*w):
            if int(a.sum()) <= target:
                break
            y, x = divmod(int(flat), w)
            a[y, x] = False
            if not _span(a) or len(_components(a)) != 1:
                a[y, x] = True
    if rng.random() < .5:
        a = a[::-1]
    if rng.random() < .5:
        a = a[:, ::-1]
    return a.copy()


def _mask(rng, h, w, count):
    options = [(nr, nc) for nr in range(1, (h+1)//2+1)
               for nc in range(1, (w+1)//2+1)
               if count <= nr*nc <= count+2 and (count != 1 or nr*nc == 1)]
    nr, nc = _pick(rng, options)
    rows, cols = _partition(h, nr, rng), _partition(w, nc, rng)
    slots = [(i, j) for i in range(nr) for j in range(nc)]
    # Select slots spanning the partition in both dimensions. A diagonal corner
    # pair suffices; for count one the only partition is 1 x 1.
    if count == 1:
        selected = [(0, 0)]
    else:
        corner = bool(rng.integers(2))
        selected = [(0, nc-1 if corner else 0), (nr-1, 0 if corner else nc-1)]
        selected = list(dict.fromkeys(selected))
        rest = [s for s in slots if s not in selected]
        rng.shuffle(rest)
        selected += rest[:count-len(selected)]
    a = np.zeros((h, w), dtype=bool)
    for i, j in selected:
        y0, y1 = rows[i]
        x0, x1 = cols[j]
        a[y0:y1, x0:x1] = _connected_shape(y1-y0, x1-x0, rng)
    _require(_span(a) and len(_components(a)) == count, "mask construction")
    return a


def _bank_options(r, c, h, w, n):
    ah, aw = r*(h+3)+1, c*(w+3)+1
    choices = []
    if max(ah, aw) > 30:
        return choices
    for nc in range(1, n+1):
        nr = (n+nc-1)//nc
        for gy in (1, 2):
            for gx in (1, 2):
                bh, bw = nr*h+(nr-1)*gy, nc*w+(nc-1)*gx
                if max(ah, bh+2) <= 30 and aw+bw+2 <= 30:
                    choices.append(('side', nr, nc, gy, gx, bh, bw))
                if ah+bh+2 <= 30 and max(aw, bw+2) <= 30:
                    choices.append(('below', nr, nc, gy, gx, bh, bw))
    return choices


def _layout(rng, config, n, scatter):
    r, c, h, w = config
    ah, aw = r*(h+3)+1, c*(w+3)+1
    if scatter:
        # Bounded 30x30 packing; a bank below guarantees a fallback.
        for _ in range(4):
            hh, ww = 30, 30
            ay, ax = int(rng.integers(hh-ah+1)), int(rng.integers(ww-aw+1))
            occupied = np.zeros((hh, ww), dtype=bool)
            occupied[ay:ay+ah, ax:ax+aw] = True
            slots = []
            candidates = [(y, x) for y in range(1, hh-h) for x in range(1, ww-w)]
            for i in rng.permutation(len(candidates)):
                y, x = candidates[int(i)]
                if not occupied[y:y+h, x:x+w].any():
                    slots.append((y, x))
                    occupied[y-1:y+h+1, x-1:x+w+1] = True
                    if len(slots) == n:
                        return hh, ww, ay, ax, slots
    mode, nr, nc, gy, gx, bh, bw = _pick(rng, _bank_options(*config, n))
    if mode == 'side':
        hh, ww = max(ah, bh+2), aw+bw+2
        ay, ax = int(rng.integers(hh-ah+1)), 0
        by, bx = int(rng.integers(1, hh-bh)), aw+1
    else:
        hh, ww = ah+bh+2, max(aw, bw+2)
        ay, ax = 0, int(rng.integers(ww-aw+1))
        by, bx = ah+1, int(rng.integers(1, ww-bw))
    positions = [(by+i*(h+gy), bx+j*(w+gx)) for i in range(nr) for j in range(nc)]
    slots = [positions[int(i)] for i in rng.choice(len(positions), n, replace=False)]
    if rng.random() < .5:
        ax = ww-aw-ax
        slots = [(y, ww-w-x) for y, x in slots]
    if rng.random() < .5:
        ay = hh-ah-ay
        slots = [(hh-h-y, x) for y, x in slots]
    return hh, ww, ay, ax, slots


def _make_pair(rng, config, bg, witness=None, witness_axis=0, witness_width=4):
    r, c, h, w = config
    matched = r*c
    ds = [d for d in range(9-matched) if _bank_options(*config, matched+d)]
    if isinstance(witness, int):
        ds = [d for d in ds if d >= 1]
    distractors = int(_pick(rng, ds))
    palette = [int(x) for x in rng.choice([c for c in range(10) if c != bg],
                                         matched+distractors+1, replace=False)]
    marker, colors = palette[0], palette[1:]
    max_count = min(3 if witness != 'test' else 6, ((h+1)//2)*((w+1)//2))
    counts = [int(rng.integers(1, max_count+1)) for _ in colors]
    directions = [int(rng.integers(4)) for _ in range(matched)]
    if isinstance(witness, int):
        counts[0] = witness+2
        directions[0] = (0 if witness == 0 else 2) + witness_axis
        directions[1] = (3 if witness == 0 else 1) if witness_axis == 0 else (0 if witness == 0 else 2)
        counts[1] = 2 if witness == 0 else 3
    elif witness == 'test':
        counts[0] = int(rng.integers(4, max_count+1))
    masks = [_mask(rng, h, w, n) for n in counts]
    if witness == 'test':
        for _ in range(96):
            n = int(rng.integers(4, max_count+1))
            direction = int(rng.integers(4))
            m = _mask(rng, h, w, n)
            if not np.array_equal(_sweep(m, n, direction), _sweep(m, 3, direction)):
                masks[0], counts[0], directions[0] = m, n, direction
                break
        else:
            # Four isolated islands with a private row at the far right. Its
            # fourth leftward impression necessarily adds a new hole. All test
            # configurations have h,w>=4, so this is a finite valid fallback.
            m = np.zeros((h, w), dtype=bool)
            m[0, 0] = m[0, w-2] = m[h-2, w-1] = m[h-1, 0] = True
            masks[0], counts[0], directions[0] = m, 4, 3
    if isinstance(witness, int):
        # Equal-area, equal-bounding-box witnesses, with 2 versus 3 components.
        v = np.zeros((5, witness_width), dtype=bool)
        if witness == 0:
            v[0] = v[4] = True
        else:
            lengths = [1, witness_width-1, witness_width]
            rng.shuffle(lengths)
            for y, length in zip((0, 2, 4), lengths):
                x = int(rng.integers(witness_width-length+1))
                v[y, x:x+length] = True
        masks[0] = v.T.copy() if witness_axis else v
    hh, ww, ay, ax, slots = _layout(rng, config, len(colors), rng.random() < .38)
    inp = np.full((hh, ww), bg, dtype=np.int8)
    ah, aw = r*(h+3)+1, c*(w+3)+1
    out = np.full((ah, aw), bg, dtype=np.int8)
    for i, color in enumerate(colors[:matched]):
        y, x = 1+(i//c)*(h+3), 1+(i % c)*(w+3)
        for target, oy, ox in ((inp, ay, ax), (out, 0, 0)):
            target[oy+y:oy+y+h+2, ox+x:ox+x+w+2] = color
        direction = directions[i]
        if direction in (0, 2):
            gy, gx = (0 if direction == 0 else h+1), int(rng.integers(1, w+1))
        else:
            gy, gx = int(rng.integers(1, h+1)), (w+1 if direction == 1 else 0)
        inp[ay+y+gy, ax+x+gx] = out[y+gy, x+gx] = marker
        holes = _point_sweep(masks[i], counts[i], direction)
        out[y+1:y+h+1, x+1:x+w+1][holes] = bg
    for slot, index in zip(slots, rng.permutation(len(colors))):
        y, x = slot
        inp[y:y+h, x:x+w][masks[int(index)]] = colors[int(index)]
    # Crop only incidental canvas space, then add independently varying margins.
    points = np.argwhere(inp != bg)
    low, high = points.min(axis=0)-1, points.max(axis=0)+2
    inp = inp[max(0, low[0]):min(hh, high[0]), max(0, low[1]):min(ww, high[1])]
    pad = []
    for dim in inp.shape:
        extra = int(rng.integers(min(4, 30-dim)+1))
        first = int(rng.integers(extra+1))
        pad.append((first, extra-first))
    return {'input': np.pad(inp, pad, constant_values=bg), 'output': out}


_ALTERNATIVES = ('single', 'area', 'plus_one', 'minus_one', 'fixed_two', 'fixed_three',
                 'reverse', 'all_stencils', 'position', 'holes_count', 'wrap', 'positive',
                 *range(1, 8))


def _training_evidence(infos):
    for direction in range(4):
        if not any(t['direction'] == direction
                   and not np.array_equal(
                       _sweep(d['patterns'][t['color']], d['counts'][t['color']], direction),
                       d['patterns'][t['color']])
                   and not np.array_equal(
                       _sweep(d['patterns'][t['color']], d['counts'][t['color']], direction),
                       _sweep(d['patterns'][t['color']], d['counts'][t['color']], (direction+2) % 4))
                   for d in infos for t in d['tiles']):
            return False
    for alt in _ALTERNATIVES:
        if not any((candidate := _render(d, alternative=alt)) is None
                   or not np.array_equal(candidate, _render(d)) for d in infos):
            return False
    return True


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    bg = int(rng.integers(10))
    mixed_background = bool(rng.random() < .8)
    axis, width = int(rng.integers(2)), int(rng.integers(3, 6))
    h, w = (width, 5) if axis else (5, width)
    # Two compact demonstrations cover all gate sides. Rejection is bounded and
    # tests only visible demonstrations, never held-out labels or hidden rules.
    for _ in range(64):
        first = _make_pair(rng, (*_pick(rng, ((1, 2), (2, 1))), h, w), bg, 0, axis, width)
        bg2 = int(rng.choice([c for c in range(10) if c != bg])) if mixed_background else bg
        second = _make_pair(rng, (*_pick(rng, ((1, 2), (2, 1))), h, w), bg2, 1, axis, width)
        if _training_evidence([_decode(first['input']), _decode(second['input'])]):
            break
    else:
        raise RuntimeError('could not establish training evidence in 64 attempts')
    puzzle = {'train': [first, second], 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    train_configs = [(r, c, hh, ww) for r in range(1, 5) for c in range(1, 5)
                     for hh in range(3, 6) for ww in range(3, 6)
                     if r*c <= 8 and _bank_options(r, c, hh, ww, r*c)]
    test_configs = [(r, c, hh, ww) for r in range(1, 4) for c in range(1, 4)
                    for hh in range(4, 8) for ww in range(4, 8)
                    if r*c <= 6 and _bank_options(r, c, hh, ww, r*c)]
    for split, count, configs in (('train', nt-2, train_configs), ('test', ne, test_configs)):
        for _ in range(count):
            for attempt in range(32):
                config = _pick(rng, configs)
                pair_bg = int(rng.integers(10)) if mixed_background else bg
                p = _make_pair(rng, config, pair_bg, 'test' if split == 'test' else None)
                if _key(p['input']) not in seen:
                    seen.add(_key(p['input']))
                    puzzle[split].append(p)
                    break
            else:
                raise RuntimeError('distinct input budget exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'split lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'pair counts')
    seen, info = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for p in puzzle[split]:
            _require(isinstance(p, dict) and set(p) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(p['input']), _grid(p['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            d = _decode(inp)
            _require(np.array_equal(_render(d, pointwise=True), out), 'incorrect carving/crop')
            _require(np.array_equal(solve(inp), out), 'slice and coordinate disagreement')
            y0, x0, _, _ = d['crop']
            for t in d['tiles']:
                y, x, h, w = t['y']-y0, t['x']-x0, t['h'], t['w']
                plate = out[y:y+h, x:x+w]
                rim = np.ones(plate.shape, dtype=bool)
                rim[1:-1, 1:-1] = False
                _require(np.array_equal(plate[rim], inp[t['y']:t['y']+h, t['x']:t['x']+w][rim]),
                         'rim/marker changed')
                _require(np.all((plate[1:-1, 1:-1] == t['color'])
                                | (plate[1:-1, 1:-1] == d['background'])), 'foreign ink')
            info[split].append(d)
    training = info['train']
    matched = [(d, t, d['patterns'][t['color']]) for d in training for t in d['tiles']]
    _require({t['direction'] for d, t, m in matched} == set(range(4)), 'missing gate side evidence')
    _require(any(int(a.sum()) == int(b.sum())
                 and da['counts'][ta['color']] != db['counts'][tb['color']]
                 for da, ta, a in matched for db, tb, b in matched), 'missing equal-area witness')
    _require(_training_evidence(training), 'simpler alternative fits all demonstrations')
    train_max = max(d['counts'][t['color']] for d, t, m in matched)
    _require(all(any(d['counts'][t['color']] > train_max
                     and not np.array_equal(
                         _sweep(d['patterns'][t['color']], d['counts'][t['color']], t['direction']),
                         _sweep(d['patterns'][t['color']], train_max, t['direction']))
                     for t in d['tiles']) for d in info['test']),
             'missing visible topology extrapolation')


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds; bounded hashes/feature sets, no saved payloads."""
    fingerprints, input_shapes, output_shapes, arrays, insets = set(), set(), set(), set(), set()
    masks, backgrounds, markers, counts, directions, distractors = set(), set(), set(), set(), set(), set()
    pairs = ring_components = singleton_components = unequal_components = mixed_palettes = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        if seed in (0, 17, 41):
            _require(_fingerprint(p) == _fingerprint(generate_puzzle(seed)), 'nondeterminism')
            # Direction, registration and palette are visible, not seed roles.
            sample = p['train'][0]
            for turns in (1, 2, 3):
                _require(np.array_equal(solve(np.rot90(sample['input'], turns)),
                                        np.rot90(sample['output'], turns)), 'rotation covariance')
            mapping = np.random.default_rng(seed+9000).permutation(10)
            _require(np.array_equal(solve(mapping[sample['input']]), mapping[sample['output']]),
                     'literal color dependence')
            d = _decode(sample['input'])
            t = d['tiles'][0]
            moved_input, moved_output = sample['input'].copy(), sample['output'].copy()
            region = moved_input[t['y']:t['y']+t['h'], t['x']:t['x']+t['w']]
            gy, gx = map(int, np.argwhere(region == d['marker'])[0])
            if t['direction'] in (0, 2):
                ny, nx = gy, 1 + gx % (t['w']-2)
            else:
                ny, nx = 1 + gy % (t['h']-2), gx
            region[gy, gx], region[ny, nx] = t['color'], d['marker']
            y0, x0, _, _ = d['crop']
            plate = moved_output[t['y']-y0:t['y']-y0+t['h'], t['x']-x0:t['x']-x0+t['w']]
            plate[gy, gx], plate[ny, nx] = t['color'], d['marker']
            _require(np.array_equal(solve(moved_input), moved_output), 'beacon coordinate dependence')
        palette = set()
        for split in ('train', 'test'):
            for pair in p[split]:
                pairs += 1
                d = _decode(pair['input'])
                input_shapes.add(pair['input'].shape)
                output_shapes.add(pair['output'].shape)
                arrays.add(d['array']); insets.add(d['inset'])
                backgrounds.add(d['background']); markers.add(d['marker'])
                palette.add((d['background'], d['marker']))
                distractors.add(d['distractors'])
                for t in d['tiles']:
                    directions.add(t['direction']); counts.add(d['counts'][t['color']])
                for mask in d['patterns'].values():
                    masks.add(hashlib.sha256(bytes(mask.shape)+mask.tobytes()).hexdigest())
                    groups = _components(mask)
                    sizes = [len(c) for c in groups]
                    singleton_components += sum(s == 1 for s in sizes)
                    unequal_components += len(set(sizes)) > 1
                    # A background component not touching the register border is a hole.
                    ring_components += sum(not np.any((g[:, 0] == 0) | (g[:, 0] == mask.shape[0]-1)
                                                      | (g[:, 1] == 0) | (g[:, 1] == mask.shape[1]-1))
                                           for g in _components(~mask))
                _require(np.array_equal(_render(d), _render(d, pointwise=True)), 'sweep arithmetic')
        mixed_palettes += len(palette) > 1
    _require(len(fingerprints) == 42 and pairs == 210, 'seed coverage')
    _require(len(input_shapes) >= 30 and len(output_shapes) >= 20, 'geometry collapsed')
    _require(len(arrays) >= 10 and len(insets) >= 15, 'structural range collapsed')
    _require(len(masks) >= 150, 'mask variation collapsed')
    _require(backgrounds == markers == set(range(10)), 'palette role range')
    _require(counts == set(range(1, 7)) and directions == set(range(4)), 'count/direction range')
    _require({0, 1, 2, 3} <= distractors, 'distractor variation')
    _require(min(ring_components, singleton_components, unequal_components) > 0, 'topology range')
    _require(mixed_palettes >= 30, 'palette variation')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            q = generate_puzzle(7319, a, b)
            _require((len(q['train']), len(q['test'])) == (a, b), 'ignored counts')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    return {'seeds': 42, 'pairs': pairs, 'input_shapes': len(input_shapes),
            'output_shapes': len(output_shapes), 'array_shapes': len(arrays),
            'inset_shapes': len(insets), 'binary_masks': len(masks),
            'component_counts': sorted(counts), 'ring_holes': int(ring_components),
            'singleton_components': int(singleton_components),
            'unequal_component_masks': int(unequal_components)}
