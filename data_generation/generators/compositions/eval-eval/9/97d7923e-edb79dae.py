"""Self-indexing stencils: hole count chooses a capped bar's ink by length rank.

Only the standard library and NumPy are used. The decoder reads visible objects;
generation records are never passed to solve(). See task_spec.md for conventions.
"""
from __future__ import annotations

import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _key(value):
    a = np.asarray(value, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, connectivity=4):
    """Four neighbors for the rule; eight only for a competing-rule probe."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            if connectivity == 8:
                neighbors += [(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            for p in neighbors:
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 0].max())+1,
            int(a[:, 1].min()), int(a[:, 1].max())+1)


def _hole_info(mask, connectivity=4):
    h, w = mask.shape
    holes, bays = [], []
    for cells in _components(~mask, connectivity):
        boundary = any(r in (0, h-1) or c in (0, w-1) for r, c in cells)
        (bays if boundary else holes).append(cells)
    return holes, bays


def _parse(input_grid):
    g = _grid(input_grid)
    frames = []
    for color in map(int, np.unique(g)):
        cells = np.argwhere(g == color)
        r, e, c, d = _box(cells)
        h, w = e-r, d-c
        if h < 7 or w < 7 or len(cells) != 2*h+2*w-4:
            continue
        if (np.all(g[r, c:d] == color) and np.all(g[e-1, c:d] == color)
                and np.all(g[r:e, c] == color) and np.all(g[r:e, d-1] == color)):
            frames.append(((r, e, c, d), color))
    _require(len(frames) == 1, "need one frame with an exclusive color")
    frame, frame_color = frames[0]
    r, e, c, d = frame
    corners = [int(g[r+1, c+1]), int(g[r+1, d-2]),
               int(g[e-2, c+1]), int(g[e-2, d-2])]
    _require(len(set(corners)) == 1, "inner corners must reveal background")
    bg = corners[0]
    _require(bg != frame_color, "frame is not background")
    places, sizes = [], set()
    interior = g[r+1:e-1, c+1:d-1]
    for cells in _components(interior != bg):
        a, b, x, y = _box(cells)
        tile = interior[a:b, x:y]
        n = b-a
        _require(n == y-x and 4 <= n <= 8 and len(cells) == n*n,
                 "placeholders must be solid size-matched squares")
        _require(len(np.unique(tile)) == 1, "placeholder must have one source")
        _require(a >= 1 and x >= 1 and b < interior.shape[0] and y < interior.shape[1],
                 "placeholder needs a background gap to frame")
        sizes.add(n)
        places.append((a+r+1, x+c+1, int(tile[0, 0])))
    _require(len(sizes) == 1 and places, "missing squares or inconsistent sizes")
    n = next(iter(sizes))
    sources = {s for _, _, s in places}
    _require(1 <= len(sources) <= 5, "invalid source count")
    exterior = g != bg
    exterior[r:e, c:d] = False
    consumed = np.zeros(g.shape, dtype=bool)
    bars = {s: [] for s in sources}
    axis = set()
    for cells in _components(exterior):
        values = {int(g[a, b]) for a, b in cells}
        if len(values) == 1:
            continue
        a, b, x, y = _box(cells)
        _require(min(b-a, y-x) == 1 and max(b-a, y-x) >= 3,
                 "multicolor clues must be thin capped bars")
        _require(len(cells) == (b-a)*(y-x) and len(values) == 2, "malformed bar")
        v = g[a:b, x:y].ravel()
        source, ink = int(v[0]), int(v[1])
        _require(source in sources and int(v[-1]) == source, "bar caps must match a source")
        _require(ink not in (bg, frame_color, source) and np.all(v[1:-1] == ink),
                 "bar ink must be uniform and different from its caps")
        bars[source].append(dict(box=(a, b, x, y), length=len(v), ink=ink))
        consumed[a:b, x:y] = True
        axis.add(0 if y-x == 1 else 1)
    _require(len(axis) == 1, "bars must share one axis within an example")
    remaining = exterior & ~consumed
    _require(set(map(int, np.unique(g[remaining]))) == sources,
             "remaining exterior clues must be source stencils")
    stencils, stencil_boxes, ranks = {}, {}, {}
    for source in sorted(sources):
        cells = np.argwhere(remaining & (g == source))
        a, b, x, y = _box(cells)
        _require((b-a, y-x) == (n, n), "source must supply one square stencil")
        tile = g[a:b, x:y]
        _require(np.all(np.isin(tile, [bg, source])), "foreign cells inside stencil bounds")
        mask = tile == source
        _require(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any(),
                 "stencil must reveal all four bounds")
        holes, _ = _hole_info(mask)
        rank = len(holes)
        family = bars[source]
        _require(len(family) >= 2 and 1 <= rank <= len(family), "hole ordinal needs competitors")
        _require(len({v['length'] for v in family}) == len(family), "tied bar lengths")
        _require(len({v['ink'] for v in family}) == len(family), "candidate inks must be distinct")
        stencils[source], stencil_boxes[source], ranks[source] = mask.copy(), (a, b, x, y), rank
    boxes = [frame] + list(stencil_boxes.values()) + [v['box'] for f in bars.values() for v in f]
    for i, (a, b, x, y) in enumerate(boxes):
        for aa, bb, xx, yy in boxes[i+1:]:
            _require(b < aa or bb < a or y < xx or yy < x, "clue bounds need a blank gap")
    return dict(grid=g, frame=frame, frame_color=frame_color, background=bg,
                n=n, sources=sources, places=places, bars=bars, axis=next(iter(axis)),
                stencils=stencils, stencil_boxes=stencil_boxes, ranks=ranks)


def _paint(a, mapping=None, solid=False):
    g, bg = a['grid'], a['background']
    r, e, c, d = a['frame']
    result = g[r:e, c:d].copy()
    if mapping is None:
        mapping = {s: sorted(a['bars'][s], key=lambda v: v['length'], reverse=True)
                   [a['ranks'][s]-1]['ink'] for s in a['sources']}
    n = a['n']
    for rr, cc, source in a['places']:
        result[rr-r:rr-r+n, cc-c:cc-c+n] = (mapping[source] if solid else
                                         np.where(a['stencils'][source], mapping[source], bg))
    return np.ascontiguousarray(result)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the input; no training labels or generation state."""
    return _paint(_parse(input_grid))


def _alternatives(a):
    maps = {name: {} for name in ('longest', 'shortest', 'ascending', 'spatial', 'reverse_spatial',
                                  'blank_area', 'all_blank_components', 'eight_neighbor_holes', 'source_ink')}
    for source in a['sources']:
        family = sorted(a['bars'][source], key=lambda v: v['length'], reverse=True)
        rank = a['ranks'][source]
        transverse = lambda v: ((v['box'][2], v['box'][0]) if a['axis'] == 0
                                else (v['box'][0], v['box'][2]))
        spatial = sorted(family, key=transverse)
        mask = a['stencils'][source]
        holes, bays = _hole_info(mask)
        maps['longest'][source] = family[0]['ink']
        maps['shortest'][source] = family[-1]['ink']
        maps['ascending'][source] = family[-rank]['ink']
        maps['spatial'][source] = spatial[rank-1]['ink']
        maps['reverse_spatial'][source] = spatial[-rank]['ink']
        # An out-of-range requested alternative cannot explain the input.
        for name, index in (('blank_area', int((~mask).sum())),
                            ('all_blank_components', len(holes)+len(bays)),
                            ('eight_neighbor_holes', len(_hole_info(mask, 8)[0]))):
            maps[name][source] = family[index-1]['ink'] if 1 <= index <= len(family) else -1
        maps['source_ink'][source] = source
    answers = {name: _paint(a, mapping) for name, mapping in maps.items()}
    answers['solid_recolor'] = _paint(a, solid=True)
    return answers


def _visible_shifts(train, test):
    def overlap(a):
        return any(sorted(a['bars'][s], key=lambda v: v['length'], reverse=True)
                   [a['ranks'][s]-1]['ink'] in a['sources'] for s in a['sources'])

    def multiplicities(a):
        return [sum(source == s for _, _, source in a['places']) for s in a['sources']]

    return dict(rank=max(rank for a in test for rank in a['ranks'].values())
                     > max(rank for a in train for rank in a['ranks'].values()),
                scale=min(a['n'] for a in test) > max(a['n'] for a in train),
                overlap=not any(overlap(a) for a in train) and any(overlap(a) for a in test),
                repeat=all(max(multiplicities(a)) == 1 for a in train)
                       and all(min(multiplicities(a)) >= 2 for a in test))


def _draw_frame(g, r, c, h, w, color):
    g[r, c:c+w] = g[r+h-1, c:c+w] = color
    g[r:r+h, c] = g[r:r+h, c+w-1] = color


def _available(occupied, h, w):
    if h > occupied.shape[0] or w > occupied.shape[1]:
        return []
    sums = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:]-sums[:-h, w:]-sums[h:, :-w]+sums[:-h, :-w]
    return np.argwhere(totals == 0).tolist()


def _reserve(occupied, r, c, h, w):
    occupied[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _mask(rng, n, rank, teaching=False):
    """Bounded growth produces arbitrary holes and bays without hidden offsets."""
    for _ in range(60):
        mask = np.ones((n, n), dtype=bool)
        interior = [(r, c) for r in range(1, n-1) for c in range(1, n-1)]
        rng.shuffle(interior)
        for r, c in interior:
            if len(_hole_info(mask)[0]) == rank:
                break
            candidate = mask.copy()
            candidate[r, c] = False
            if len(_hole_info(candidate)[0]) == len(_hole_info(mask)[0])+1:
                mask = candidate
        if len(_hole_info(mask)[0]) != rank:
            continue
        desired = 3 if teaching else int(rng.integers(rank, min((n-2)**2, rank*5)+1))
        for _ in range(60):
            if int((~mask).sum()) >= desired:
                break
            holes, _ = _hole_info(mask)
            frontier = set()
            for cells in holes:
                for r, c in cells:
                    for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                        if 1 <= rr < n-1 and 1 <= cc < n-1 and mask[rr, cc]:
                            frontier.add((rr, cc))
            choices = sorted(frontier)
            rng.shuffle(choices)
            moved = False
            for r, c in choices:
                candidate = mask.copy()
                candidate[r, c] = False
                if len(_hole_info(candidate)[0]) == rank:
                    mask, moved = candidate, True
                    break
            if not moved:
                break
        if teaching and int((~mask).sum()) != 3:
            continue
        if teaching and rank == 2 and len(_hole_info(mask, 8)[0]) != 1:
            continue
        # Add a one-cell boundary bay in teaching masks: equal blank area four,
        # but one versus two holes. Other masks get irregular open carvings.
        edge = [(r, c) for r in range(n) for c in range(n)
                if r in (0, n-1) or c in (0, n-1)]
        rng.shuffle(edge)
        carved = False
        for r, c in edge:
            candidate = mask.copy()
            candidate[r, c] = False
            holes, bays = _hole_info(candidate)
            if (len(holes) == rank and len(bays) == 1 and len(bays[0]) == 1
                    and (not teaching or len(_hole_info(candidate, 8)[0]) == 1)
                    and all(v.any() for v in (candidate[0], candidate[-1], candidate[:, 0], candidate[:, -1]))):
                mask, carved = candidate, True
                break
        if not carved:
            continue
        if not teaching:
            for _ in range(int(rng.integers(0, n*n))):
                r, c = map(int, rng.integers(0, n, 2))
                candidate = mask.copy()
                candidate[r, c] = False
                if (len(_hole_info(candidate)[0]) == rank
                        and all(v.any() for v in (candidate[0], candidate[-1], candidate[:, 0], candidate[:, -1]))):
                    mask = candidate
        return np.rot90(mask, int(rng.integers(4))).copy()
    # Transparent bounded fallback, still respecting the teaching contrast.
    mask = np.ones((n, n), dtype=bool)
    mask[1, 1] = False
    if rank == 1:
        mask[1, 2:4] = False
    else:
        mask[1, 3] = False
        if rank == 2 and teaching:
            mask[1, 3] = True
            mask[1, 2] = False
            mask[2, 3] = False
        if rank >= 3:
            mask[3, 1] = False
        if rank >= 4:
            mask[3, 3] = False
    mask[-1, -1] = False
    _require(len(_hole_info(mask)[0]) == rank, "mask fallback rank")
    return mask


def _board(rng, n, sources, repeated):
    k = len(sources)
    count = (int(rng.integers(2*k, min(6, 2*k+2)+1)) if repeated else k)
    arrangements = [(rows, cols) for rows in range(1, 4) for cols in range(1, 4)
                    if rows*cols >= count and rows*(n+1)+3 <= 23 and cols*(n+1)+3 <= 27]
    weights = np.array([1/(1+rows*cols-count)**2 for rows, cols in arrangements])
    rows, cols = arrangements[int(rng.choice(len(arrangements), p=weights/weights.sum()))]
    h, w = rows*(n+1)+3, cols*(n+1)+3
    h += int(rng.integers(0, min(2, 24-h)+1))
    w += int(rng.integers(0, min(2, 28-w)+1))
    slots = [(2+i*(n+1), 2+j*(n+1)) for i in range(rows) for j in range(cols)]
    chosen = [slots[int(i)] for i in rng.permutation(len(slots))[:count]]
    assignments = (sources + sources + list(map(int, rng.choice(sources, count-2*k)))
                   if repeated else list(sources))
    rng.shuffle(assignments)
    spots = [(r, c, source) for (r, c), source in zip(chosen, assignments)]
    # Independent movement breaks the shared slot lattice where slack allows.
    for i, (r, c, s) in enumerate(spots):
        candidates = [(rr, cc) for rr in range(max(2, r-2), min(h-n-2, r+2)+1)
                      for cc in range(max(2, c-2), min(w-n-2, c+2)+1)
                      if all(i == j or rr+n < a or a+n < rr or cc+n < b or b+n < cc
                             for j, (a, b, _) in enumerate(spots))]
        rr, cc = candidates[int(rng.integers(len(candidates)))]
        spots[i] = (rr, cc, s)
    return h, w, spots


def _pack(rng, n, sources, stencils, candidates, mapping, bg, frame_color, repeated, anchor):
    h, w, spots = _board(rng, n, sources, repeated)
    entries = [(s, length, ink) for s in sources for length, ink in candidates[s]]
    for attempt in range(24):
        ch = int(rng.integers(max(18, h), 31)) if attempt < 16 else 30
        cw = int(rng.integers(max(18, w), 31)) if attempt < 16 else 30
        occupied = np.zeros((ch, cw), dtype=bool)
        if rng.random() < .5:
            fr, fc = int(rng.choice([0, ch-h])), int(rng.choice([0, cw-w]))
        else:
            fr, fc = int(rng.integers(ch-h+1)), int(rng.integers(cw-w+1))
        _reserve(occupied, fr, fc, h, w)
        locations, failed = {}, False
        objects = [('stencil', s, n, n) for s in rng.permutation(sources)]
        # Long bars first; anchor envelopes permit a later spatial contrast.
        for i in sorted(range(len(entries)), key=lambda i: entries[i][1], reverse=True):
            s, length, _ = entries[i]
            envelope = max(v[0] for v in candidates[s]) if anchor and s == sources[0] else length
            objects.append(('bar', i, 1, envelope))
        anchor_rows = set()
        for kind, identity, oh, ow in objects:
            positions = _available(occupied, oh, ow)
            if anchor and kind == 'bar' and entries[identity][0] == sources[0]:
                positions = [p for p in positions if p[0] not in anchor_rows]
            if not positions:
                failed = True
                break
            rr, cc = positions[int(rng.integers(len(positions)))]
            locations[kind, identity] = (rr, cc)
            _reserve(occupied, rr, cc, oh, ow)
            if anchor and kind == 'bar' and entries[identity][0] == sources[0]:
                anchor_rows.add(rr)
        if not failed:
            break
    else:
        # Guaranteed small shelf fallback, independent of requested pair counts.
        # n <= 7, sum(populations) <= 10 for n=7, <=12 otherwise.
        ch = cw = 30
        h, w = n+4, len(sources)*(n+1)+3
        spots = [(2, 2+i*(n+1), s) for i, s in enumerate(sources)]
        fr, fc = n+1, 0
        locations = {('stencil', s): (0, i*(n+1)) for i, s in enumerate(rng.permutation(sources))}
        order = list(map(int, rng.permutation(len(entries))))
        if repeated:
            # Two copies of each of at most two sources. Candidate bars occupy
            # the side band, so repetition survives a packing fallback.
            h = 2*(n+1)+3
            spots += [(2+n+1, 2+i*(n+1), s) for i, s in enumerate(sources)]
            for j, i in enumerate(order):
                locations['bar', i] = (2*j, w+1)
        # Reserve separated rows for the teaching anchor, then use two shelves.
        elif anchor:
            ids = [i for i in order if entries[i][0] == sources[0]]
            rest = [i for i in order if i not in ids]
            for j, i in enumerate(ids):
                locations['bar', i] = (fr+h+1+2*j, 0)
            for j, i in enumerate(rest):
                locations['bar', i] = (fr+h+1+2*j, 13)
        else:
            for j, i in enumerate(order):
                locations['bar', i] = (fr+h+1+2*(j//2), 13*(j % 2))
    if anchor:
        ids = [i for i, v in enumerate(entries) if v[0] == sources[0]]
        slots = sorted((locations['bar', i] for i in ids))
        longest = max(ids, key=lambda i: entries[i][1])
        middle = slots.pop(len(slots)//2)
        rng.shuffle(slots)
        locations['bar', longest] = middle
        for i, point in zip([i for i in ids if i != longest], slots):
            locations['bar', i] = point
    g = np.full((ch, cw), bg, dtype=np.int8)
    _draw_frame(g, fr, fc, h, w, frame_color)
    for r, c, s in spots:
        g[fr+r:fr+r+n, fc+c:fc+c+n] = s
    for s, mask in stencils.items():
        rr, cc = locations['stencil', s]
        g[rr:rr+n, cc:cc+n][mask] = s
    for i, (s, length, ink) in enumerate(entries):
        rr, cc = locations['bar', i]
        g[rr, cc:cc+length] = ink
        g[rr, cc] = g[rr, cc+length-1] = s
    out = g[fr:fr+h, fc:fc+w].copy()
    for rr, cc, s in spots:
        out[rr:rr+n, cc:cc+n] = np.where(stencils[s], mapping[s], bg)
    a, b, x, y = _box(np.argwhere(g != bg))
    a, x = max(0, a-int(rng.integers(3))), max(0, x-int(rng.integers(3)))
    b, y = min(ch, b+int(rng.integers(3))), min(cw, y+int(rng.integers(3)))
    return g[a:b, x:y].copy(), out


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    shift = str(rng.choice(['rank', 'scale', 'overlap', 'repeat']))
    rotation = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    anchor_n = int(rng.choice([5, 6]))
    anchor_count = int(rng.integers(3, 6))
    anchor_lengths = list(map(int, rng.choice(np.arange(3, 12), anchor_count, replace=False)))
    anchor_inks = list(map(int, rng.choice(palette[4:], anchor_count, replace=False)))
    anchor_candidates = list(zip(anchor_lengths, anchor_inks))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            teaching = split == 'train' and index < 2
            for attempt in range(12):
                local = palette if teaching else list(map(int, rng.permutation(10)))
                bg, frame_color = local[:2]
                n = anchor_n if teaching else int(rng.choice([5, 6, 7]))
                if shift == 'scale':
                    n = (anchor_n if teaching else int(rng.choice([5, 6]))) if split == 'train' else 7
                k = (1 if index == 0 else 2) if teaching else int(rng.integers(1, 4))
                if split == 'test' and shift == 'overlap':
                    k = int(rng.integers(2, 4))
                repeated = split == 'test' if shift == 'repeat' else bool(rng.integers(2))
                if repeated:
                    k = min(k, 2)
                sources = local[2:2+k]
                ranks = [int(rng.integers(1, 3)) for _ in sources]
                if teaching:
                    ranks[0] = index+1
                elif split == 'test' and shift == 'rank':
                    ranks[int(rng.integers(k))] = 3
                stencils = {s: _mask(rng, n, rank, teaching=teaching and i == 0)
                            for i, (s, rank) in enumerate(zip(sources, ranks))}
                ink_capacity = min(6, 8-k) if split == 'train' and shift == 'overlap' else 6
                populations = [int(rng.integers(max(2, rank), ink_capacity+1)) for rank in ranks]
                if teaching:
                    populations[0] = anchor_count
                budget = 10 if n == 7 else 12
                while sum(populations) > budget:
                    eligible = [i for i, v in enumerate(populations)
                                if v > max(2, ranks[i]) and not (teaching and i == 0)]
                    populations[int(rng.choice(eligible))] -= 1
                candidates = {}
                for i, s in enumerate(sources):
                    if teaching and i == 0:
                        candidates[s] = list(anchor_candidates)
                        continue
                    length_limit = 10 if n == 7 else 11
                    lengths = list(map(int, rng.choice(np.arange(3, length_limit+1), populations[i], replace=False)))
                    inks = [color for color in local if color not in (bg, frame_color, s)]
                    if split == 'train' and shift == 'overlap':
                        inks = [color for color in inks if color not in sources]
                    chosen = list(map(int, rng.choice(inks, populations[i], replace=False)))
                    candidates[s] = list(zip(lengths, chosen))
                if teaching and index == 1 and shift != 'overlap':
                    # Teach direct substitution with a finite B -> A -> ink
                    # chain. The first source's dictionary stays unchanged.
                    s, target = sources[1], sources[0]
                    selected_length, old = sorted(candidates[s], reverse=True)[ranks[1]-1]
                    candidates[s] = [(length, old if ink == target else ink)
                                     for length, ink in candidates[s]]
                    candidates[s] = [(length, target if length == selected_length else ink)
                                     for length, ink in candidates[s]]
                if split == 'test' and shift == 'overlap':
                    # A visible source cycle in the selected inks; other candidate
                    # colors remain distinct within each source's family.
                    for i, s in enumerate(sources):
                        ordered = sorted(candidates[s], reverse=True)
                        selected_length, _ = ordered[ranks[i]-1]
                        target = sources[(i+1) % k]
                        old = dict(candidates[s])[selected_length]
                        candidates[s] = [(length, old if ink == target else ink)
                                         for length, ink in candidates[s]]
                        candidates[s] = [(length, target if length == selected_length else ink)
                                         for length, ink in candidates[s]]
                mapping = {s: sorted(candidates[s], reverse=True)[rank-1][1]
                           for s, rank in zip(sources, ranks)}
                g, out = _pack(rng, n, sources, stencils, candidates, mapping, bg, frame_color,
                               repeated, anchor=teaching and index == 0)
                turn = int(rng.integers(4)) if mixed else rotation
                if bool(rng.integers(2)):
                    g = np.fliplr(g)
                    out = np.fliplr(out)
                g = np.ascontiguousarray(np.rot90(g, turn))
                out = np.ascontiguousarray(np.rot90(out, turn))
                if _key(g) not in seen:
                    # Construct labels from the declarative objects, then compare
                    # them to a fresh input-only decode (separate paths).
                    _require(np.array_equal(out, solve(g)), "record/decode disagreement")
                    seen.add(_key(g))
                    puzzle[split].append({'input': g, 'output': out})
                    break
            else:
                raise RuntimeError("distinct-input budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _parse(inp)
            _require(np.array_equal(out, _paint(a)), "incorrect stamping, ink rank or crop")
            r, e, c, d = a['frame']
            _require(out.shape == (e-r, d-c), "wrong inclusive crop")
            _require(np.all(out[0] == a['frame_color']) and np.all(out[-1] == a['frame_color'])
                     and np.all(out[:, 0] == a['frame_color']) and np.all(out[:, -1] == a['frame_color']),
                     "frame changed")
            untouched = np.ones(out.shape, bool)
            for rr, cc, s in a['places']:
                untouched[rr-r:rr-r+a['n'], cc-c:cc-c+a['n']] = False
                printed = out[rr-r:rr-r+a['n'], cc-c:cc-c+a['n']]
                _require(np.array_equal(printed != a['background'], a['stencils'][s]), "lost mask geometry")
            _require(np.array_equal(out[untouched], inp[r:e, c:d][untouched]), "changed unused board cells")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "invented a color")
            analyses[split].append(a)
    train = analyses['train']
    _require({1, 2} <= {rank for a in train for rank in a['ranks'].values()}, "missing ordinal progression")
    _require(any(1 < a['ranks'][s] < len(a['bars'][s]) for a in train for s in a['sources']),
             "need an intermediate rank demonstration")
    _require(any(len(a['sources']) >= 2 for a in train), "need cap-based grouping demonstration")
    for name in _alternatives(train[0]):
        _require(any(not np.array_equal(_alternatives(a)[name], _paint(a)) for a in train),
                 f"training fails to distinguish {name}")
    witnesses = []
    for a in train:
        for s in a['sources']:
            holes, bays = _hole_info(a['stencils'][s])
            dictionary = tuple(sorted((v['length'], v['ink']) for v in a['bars'][s]))
            witnesses.append((s, dictionary, len(holes), sum(map(len, holes)),
                              int((~a['stencils'][s]).sum()), len(bays)))
    _require(any(x[:2] == y[:2] and {x[2], y[2]} == {1, 2} and x[3] == y[3] == 3
                 and x[4] == y[4] == 4 and x[5] == y[5] == 1
                 for i, x in enumerate(witnesses) for y in witnesses[i+1:]),
             "missing equal-area topology contrast")
    _require(any(a['ranks'][s] == 2 and len(_hole_info(a['stencils'][s], 8)[0]) == 1
                 for a in train for s in a['sources']), "need diagonal-contact connectivity evidence")
    demonstrated = {rank for a in train for rank in a['ranks'].values()}
    for a in analyses['test']:
        _require(all(rank in demonstrated or set(range(1, rank)) <= demonstrated
                     for rank in a['ranks'].values()), "unprepared ordinal extension")
    _require(any(_visible_shifts(train, analyses['test']).values()), "missing a visible train/test shift")


def _features(pair):
    a = _parse(pair['input'])
    r, e, c, d = a['frame']
    infos = [_hole_info(mask) for mask in a['stencils'].values()]
    selected = {s: sorted(a['bars'][s], key=lambda v: v['length'], reverse=True)
                [a['ranks'][s]-1]['ink'] for s in a['sources']}
    return dict(n=a['n'], sources=len(a['sources']), squares=len(a['places']), axis=a['axis'],
                populations=sorted(len(f) for f in a['bars'].values()), ranks=sorted(a['ranks'].values()),
                areas=[len(h) for holes, _ in infos for h in holes],
                bays=sum(len(bays) for _, bays in infos),
                disconnected=sum(len(_components(m)) > 1 for m in a['stencils'].values()),
                overlap=sum(v in a['sources'] for v in selected.values()),
                shortest=sum(a['ranks'][s] == len(a['bars'][s]) for s in a['sources']),
                input_shape=list(a['grid'].shape), output_shape=[e-r, d-c],
                background=a['background'], frame_color=a['frame_color'],
                edge=r == 0 or c == 0 or e == a['grid'].shape[0] or d == a['grid'].shape[1])


def _general_rule_probes():
    """Hand-constructed cases beyond the sampler's size and source ranges."""
    # Five sources, side four, including diagonal holes counted separately.
    g = np.zeros((26, 28), dtype=np.int8)
    fr, h, w, n = 12, 13, 18, 4
    _draw_frame(g, fr, 0, h, w, 9)
    expected = np.zeros((h, w), dtype=np.int8)
    _draw_frame(expected, 0, 0, h, w, 9)
    for i, s in enumerate(range(1, 6)):
        rank = 1+i % 2
        mask = np.ones((n, n), dtype=bool)
        mask[1, 1] = False
        if rank == 2:
            mask[2, 2] = False
        r, c = (i//3)*5, (i % 3)*5
        g[r:r+n, c:c+n][mask] = s
        rr, cc = 2+(i//3)*5, 2+(i % 3)*5
        g[fr+rr:fr+rr+n, cc:cc+n] = s
        expected[rr:rr+n, cc:cc+n] = np.where(mask, 7 if rank == 1 else 6, 0)
        for j, (length, ink) in enumerate(((4, 6), (6, 7))):
            row = 2*(2*i+j)
            g[row, 20:20+length] = ink
            g[row, 20] = g[row, 20+length-1] = s
    _require(np.array_equal(solve(g), expected), "side-four/five-source contract")

    # Side eight and rank four; the other stencil has an active island in one
    # large hole, and its ink is the first source. This is a direct 3 -> 2 -> 4
    # chain, with visibly different templates, not a recursive substitution.
    g = np.zeros((30, 27), dtype=np.int8)
    n, fr, h, w = 8, 9, 12, 21
    _draw_frame(g, fr, 0, h, w, 9)
    expected = np.zeros((h, w), dtype=np.int8)
    _draw_frame(expected, 0, 0, h, w, 9)
    masks = {2: np.ones((n, n), bool), 3: np.ones((n, n), bool)}
    for r, c in ((1, 1), (1, 3), (3, 1), (3, 3)):
        masks[2][r, c] = False
    masks[3][1:6, 1:6] = False
    masks[3][3, 3] = True
    dictionaries = {2: [(3, 4), (5, 5), (7, 6), (9, 7)],
                    3: [(3, 1), (5, 4), (7, 8), (9, 2)]}
    for i, s in enumerate((2, 3)):
        masks[s][-1, -1] = False
        g[:n, i*10:i*10+n][masks[s]] = s
        cc = 2+i*9
        g[fr+2:fr+2+n, cc:cc+n] = s
        expected[2:2+n, cc:cc+n] = np.where(masks[s], 4 if s == 2 else 2, 0)
        for j, (length, ink) in enumerate(dictionaries[s]):
            row, col = 22+2*j, 13*i
            g[row, col:col+length] = ink
            g[row, col] = g[row, col+length-1] = s
    _require(np.array_equal(solve(g), expected), "side-eight/rank-four/island/direct-chain contract")
    # Connect two holes: rank drops from four to three and all surviving active
    # pixels for source two change ink from 4 to 5, rather than a local edit.
    merged = g.copy()
    merged[1, 2] = 0
    mask = masks[2].copy()
    mask[1, 2] = False
    changed_expected = expected.copy()
    changed_expected[2:10, 2:10] = np.where(mask, 5, 0)
    _require(np.array_equal(solve(merged), changed_expected), "merging holes must change the ordinal ink")
    # Growing one hole leaves its rank and ink unchanged, while transferring
    # the larger blank geometry exactly.
    grown = g.copy()
    grown[3, 4] = 0
    mask = masks[2].copy()
    mask[3, 4] = False
    changed_expected = expected.copy()
    changed_expected[2:10, 2:10] = np.where(mask, 4, 0)
    _require(np.array_equal(solve(grown), changed_expected), "hole area must not select ink")
    tied = g.copy()
    tied[28, :9] = 0
    tied[28, :7] = 7
    tied[28, 0] = tied[28, 6] = 2
    try:
        solve(tied)
    except AssertionError:
        pass
    else:
        raise AssertionError("tied candidate lengths accepted")


def self_test():
    """Bounded 42-seed coverage, invariants, metamorphisms and error checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for p in puzzles:
        validate_generated_puzzle(p)
    _general_rule_probes()
    shifts = [_visible_shifts([_parse(pair['input']) for pair in p['train']],
                             [_parse(pair['input']) for pair in p['test']]) for p in puzzles]
    _require(all(any(s[name] for s in shifts) for name in ('rank', 'scale', 'overlap', 'repeat')),
             "principal shift diversity")
    records = [_features(pair) for p in puzzles for split in ('train', 'test') for pair in p[split]]
    _require(len({tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s])
                  for p in puzzles}) == 42, "seed diversity")
    _require({5, 6, 7} <= {v['n'] for v in records}, "size variation")
    _require({1, 2, 3} <= {v['sources'] for v in records}, "source variation")
    _require({2, 3, 4, 5, 6} <= {n for v in records for n in v['populations']}, "competitor variation")
    _require({1, 2, 3} <= {n for v in records for n in v['ranks']}, "ordinal variation")
    _require({0, 1} == {v['axis'] for v in records}, "bar axis variation")
    _require(len({tuple(v['input_shape']) for v in records}) >= 15, "input shape variation")
    _require(len({tuple(v['output_shape']) for v in records}) >= 20, "board geometry variation")
    _require(any(v['disconnected'] for v in records), "disconnected stencil coverage")
    _require(any(v['overlap'] for v in records) and any(v['shortest'] for v in records),
             "overlapping maps and last-ranked selections")
    _require(any(v['squares'] > v['sources'] for v in records) and any(v['edge'] for v in records),
             "repeated squares and boundary frames")
    _require(len({v['background'] for v in records}) == 10
             and len({v['frame_color'] for v in records}) == 10, "semantic palette variation")
    _require(len({area for v in records for area in v['areas']}) >= 5, "hole-area variation")
    permutation = np.array([8, 2, 7, 0, 5, 1, 9, 3, 6, 4], dtype=np.int8)
    for p in puzzles[:4]:
        pair = p['test'][0]
        for k in range(4):
            inp, out = permutation[np.rot90(pair['input'], k)], permutation[np.rot90(pair['output'], k)]
            _require(np.array_equal(solve(inp), out), "rotation/recoloring equivariance")
            _require(np.array_equal(solve(np.fliplr(inp)), np.fliplr(out)), "reflection equivariance")
    # Editing an unselected candidate ink cannot affect the output; editing a
    # selected ink must repaint precisely its source's active cells.
    pair = puzzles[0]['train'][0]
    a = _parse(pair['input'])
    s = min(a['sources'])
    ordered = sorted(a['bars'][s], key=lambda v: v['length'], reverse=True)
    selected = ordered[a['ranks'][s]-1]
    unselected = next(v for v in ordered if v is not selected)
    for v, should_change in ((unselected, False), (selected, True)):
        replacement = next(c for c in range(10) if c not in (s, a['background'], a['frame_color'])
                           and c not in {bar['ink'] for bar in ordered})
        changed = pair['input'].copy()
        r, e, c, d = v['box']
        if e-r == 1:
            changed[r, c+1:d-1] = replacement
        else:
            changed[r+1:e-1, c] = replacement
        expected = pair['output'].copy()
        if should_change:
            fr, _, fc, _ = a['frame']
            for rr, cc, source in a['places']:
                if source == s:
                    tile = expected[rr-fr:rr-fr+a['n'], cc-fc:cc-fc+a['n']]
                    tile[a['stencils'][s]] = replacement
        _require(np.array_equal(solve(changed), expected), "selected/unselected ink dependency")
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(1000+10*tr+te, tr, te)
            _require((len(p['train']), len(p['test'])) == (tr, te), "explicit counts")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    _require(_key(generate_puzzle(17)['test'][0]['input']) == _key(puzzles[17]['test'][0]['input']),
             "determinism")
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {'seeds': 42, 'pairs': len(records), 'sizes': sorted({v['n'] for v in records}),
            'input_shapes': len({tuple(v['input_shape']) for v in records}),
            'output_shapes': len({tuple(v['output_shape']) for v in records}),
            'disconnected_stencils': sum(v['disconnected'] for v in records),
            'overlapping_selections': sum(v['overlap'] for v in records),
            'visible_shifts': {name: sum(s[name] for s in shifts) for name in shifts[0]}}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
