"""Topology-keyed inlays: count holes, complete footprints, and fit rigid pieces.

Only NumPy and the standard library are used. The decoder never reads the seed,
generator metadata, examples, or files. Arrays passed to solve are not mutated.
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (
    0, 945883526, 357917713, 788265672, 430400386, 278626569, 599198047,
    83370111, 329204715, 79778152, 402835639, 671063175, 998718864,
    279007778, 213642189, 967262474, 27630725, 119931801, 494507958,
    218801953, 196720532, 377768873, 44034778, 49507636, 990631170,
    214580106, 813939921, 351431273, 153347506, 380554529, 204035677,
    76738069, 152388945, 54927133, 802888797, 253677022, 916933221,
    432277946, 9527467, 581456895, 274426069, 853241427,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(grid):
    a = np.asarray(grid)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    neighbors = EIGHT if diagonal else EDGE
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        result.append(cells)
    return result


def _box(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr) + 1, max(cc) + 1


def _mask(cells, shape):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _enclosed(material):
    """Four-connected holes, explicitly excluding the padded exterior."""
    padded = np.pad(material, 1)
    holes = np.zeros(padded.shape, dtype=bool)
    sizes = []
    for cells in _components(~padded):
        if any(r in (0, padded.shape[0] - 1) or c in (0, padded.shape[1] - 1)
               for r, c in cells):
            continue
        for p in cells:
            holes[p] = True
        sizes.append(len(cells))
    return holes[1:-1, 1:-1], tuple(sorted(sizes))


def _erode(mask, diagonal=False):
    h, w = mask.shape
    p = np.pad(mask, 1)
    safe = mask.copy()
    for dr, dc in EIGHT if diagonal else EDGE:
        safe &= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return safe


def _isolated(mask):
    return not (np.any(mask[:-1] & mask[1:]) or np.any(mask[:, :-1] & mask[:, 1:]))


def _fits(inlay, socket):
    """Return distinct colored results, not merely distinct rotation indices."""
    matches, seen = [], set()
    for k in range(4):
        support = np.rot90(inlay['support'], k)
        if support.shape != socket['mask'].shape or not np.array_equal(support, socket['mask']):
            continue
        material = np.rot90(inlay['material'], k).copy()
        key = _key(material)
        if key not in seen:
            seen.add(key)
            matches.append((k, material, np.rot90(inlay['holes'], k).copy()))
    return matches


def _render(facts, assignment):
    out = facts['scaffold'].copy()
    for socket, i in zip(facts['sockets'], assignment):
        fits = _fits(facts['inlays'][i], socket)
        _require(len(fits) == 1, "Each selected inlay must have one colored fit")
        _, material, _ = fits[0]
        r, c = socket['origin']
        h, w = material.shape
        view = out[r:r + h, c:c + w]
        view[socket['mask']] = facts['carrier']
        view[material] = facts['inlays'][i]['color']
    return out


def _decode(input_grid):
    inp = _grid(input_grid)
    border = _perimeter(inp)
    bg = int(border[0])
    _require(bool(np.all(border == bg)), "The outer ring must be background")
    objects = _components(inp != bg, diagonal=True)
    candidates = []
    for cells in objects:
        colors, counts = np.unique([inp[p] for p in cells], return_counts=True)
        if len(colors) != 2 or sum(counts == 1) != 1:
            continue
        marker = int(colors[np.argmin(counts)])
        carrier = int(colors[np.argmax(counts)])
        r0, c0, r1, c1 = _box(cells)
        if r1 - r0 != c1 - c0 or r1 - r0 < 5:
            continue
        crop = inp[r0:r1, c0:c1]
        marked = np.argwhere(crop == marker)
        if len(marked) != 1:
            continue
        mr, mc = map(int, marked[0])
        s = len(crop)
        if (mr, mc) not in ((0, 0), (0, s - 1), (s - 1, 0), (s - 1, s - 1)):
            continue
        if not np.all(np.isin(_perimeter(crop), (carrier, marker))):
            continue
        if int(np.sum(inp == marker)) != 1:
            continue
        candidates.append((cells, (r0, c0, r1, c1), crop, carrier, marker))
    _require(len(candidates) == 1, "Exactly one marked square scaffold is required")
    frame_cells, box, crop, carrier, marker = candidates[0]
    r0, c0, r1, c1 = box
    _require(len(_components(crop == carrier)) == 1, "Scaffold carrier must be four-connected")
    rotation = next(k for k in range(4) if np.rot90(crop, k)[0, 0] == marker)
    scaffold = np.rot90(crop, rotation).copy()
    frame_mask = (scaffold == carrier) | (scaffold == marker)
    regions = _components(~frame_mask)
    _require(2 <= len(regions) <= 4, "There must be two to four enclosed sockets")
    sockets, clue_colors = [], set()
    for cells in regions:
        full = _mask(cells, scaffold.shape)
        vals = scaffold[full]
        clue_colors.update(map(int, vals[vals != bg]))
        clues = full & (scaffold != bg)
        _require(_isolated(clues), "Socket dots must be four-isolated")
        _require(bool(np.all(_erode(full, diagonal=True)[clues])), "Dots must not touch the scaffold")
        a, b, d, e = _box(cells)
        sockets.append(dict(origin=(a, b), mask=full[a:d, b:e],
                            clues=clues[a:d, b:e], count=int(clues.sum())))
    _require(len(clue_colors) == 1, "All dots share one locally visible clue color")
    clue = next(iter(clue_colors))
    _require(len({bg, carrier, marker, clue}) == 4, "Semantic colors must differ")
    counts = [s['count'] for s in sockets]
    _require(len(set(counts)) == len(sockets), "Socket counts must be unique")
    inlays = []
    for cells in objects:
        inside = [r0 <= r < r1 and c0 <= c < c1 for r, c in cells]
        if all(inside):
            continue
        _require(not any(inside), "Inlays must be outside the scaffold box")
        colors = set(int(inp[p]) for p in cells)
        _require(len(colors) == 1, "Inlays must be monochromatic")
        color = next(iter(colors))
        _require(color not in {bg, carrier, marker, clue}, "Inlay material needs a distinct role")
        a, b, d, e = _box(cells)
        material = _mask([(r - a, c - b) for r, c in cells], (d - a, e - b))
        _require(len(_components(material)) == 1, "Inlay material must be four-connected")
        holes, sizes = _enclosed(material)
        inlays.append(dict(color=color, material=material, holes=holes,
                           support=material | holes, count=len(sizes), sizes=sizes,
                           origin=(a, b)))
    _require(len(inlays) == len(sockets), "Use one inlay for every socket")
    by_count = {x['count']: i for i, x in enumerate(inlays)}
    _require(len(by_count) == len(inlays) and set(by_count) == set(counts), "Counts must form a bijection")
    assignment = [by_count[s['count']] for s in sockets]
    facts = dict(input=inp, background=bg, carrier=carrier, marker=marker, clue=clue,
                 box=box, scaffold=scaffold, frame_mask=frame_mask, rotation=rotation,
                 sockets=sockets, inlays=inlays, assignment=assignment)
    facts['output'] = _render(facts, assignment)
    return facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone, with no generation state or demonstrations."""
    return _decode(input_grid)['output']


def _choose_clues(rng, support, count):
    eligible = list(map(tuple, np.argwhere(_erode(support, diagonal=True)).tolist()))
    if count == 0:
        return np.zeros_like(support)
    for _ in range(32):
        chosen = []
        for index in rng.permutation(len(eligible)):
            p = eligible[int(index)]
            if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) != 1 for q in chosen):
                chosen.append(p)
                if len(chosen) == count:
                    return _mask(chosen, support.shape)
    # A finite search proves capacity when greedy choices fail.
    for chosen in combinations(eligible, count):
        if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) != 1
               for p, q in combinations(chosen, 2)):
            return _mask(chosen, support.shape)
    return None


def _shape(rng, count, witness=False):
    if witness:
        # A non-square rectangle with one missing corner is chiral. Its complete
        # interior survives, admitting both a large ring and diagonal holes.
        h, w = (5, 6) if rng.random() < .5 else (6, 5)
        support = np.ones((h, w), dtype=bool)
        corner = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))[int(rng.integers(4))]
        support[corner] = False
        return support
    h, w = map(int, rng.integers(5, 8, size=2))
    support = np.ones((h, w), dtype=bool)
    if count in (0, 1) and rng.random() < .25:
        return support
    for _ in range(int(rng.integers(1, 5))):
        edge = int(rng.integers(4))
        if edge < 2:
            start = int(rng.integers(w))
            length = int(rng.integers(1, min(3, w - start) + 1))
            support[0 if edge == 0 else -1, start:start + length] = False
        else:
            start = int(rng.integers(h))
            length = int(rng.integers(1, min(3, h - start) + 1))
            support[start:start + length, 0 if edge == 2 else -1] = False
    if len(_components(support)) != 1:
        return None
    # Crop any emptied outer row/column, retaining the true rigid outline.
    a, b, d, e = _box(np.argwhere(support).tolist())
    if min(d - a, e - b) < 5:
        return None
    return support[a:d, b:e].copy()


def _make_holes(rng, support, count, witness=False):
    holes = np.zeros_like(support)
    if count == 0:
        return holes
    if witness:
        if count == 1:
            holes[1:-1, 1:-1] = True
        else:
            holes[1, 1] = holes[2, 2] = holes[2, 3] = True
        return holes
    eligible = _erode(support)
    # Large ordinary rings are allowed, including symmetric inlays whose
    # duplicate rotations yield exactly the same colored result.
    if count == 1 and (np.all(support) or rng.random() < .30):
        if len(_components(eligible)) == 1 and len(_components(support & ~eligible)) == 1:
            return eligible
    cells = np.argwhere(eligible).tolist()
    for _ in range(32):
        holes[:] = False
        for index in rng.permutation(len(cells)):
            r, c = cells[int(index)]
            if any(0 <= r + dr < len(holes) and 0 <= c + dc < holes.shape[1] and holes[r + dr, c + dc]
                   for dr, dc in EDGE):
                continue
            holes[r, c] = True
            if len(_components(support & ~holes)) != 1:
                holes[r, c] = False
            if int(holes.sum()) == count:
                break
        if int(holes.sum()) == count:
            break
    else:
        return None
    target = int(rng.integers(count, int(eligible.sum()) + 1))
    for index in rng.permutation(len(cells)):
        if int(holes.sum()) >= target:
            break
        r, c = cells[int(index)]
        if holes[r, c] or not any(0 <= r + dr < len(holes) and 0 <= c + dc < holes.shape[1]
                                  and holes[r + dr, c + dc] for dr, dc in EDGE):
            continue
        holes[r, c] = True
        if len(_components(holes)) != count or len(_components(support & ~holes)) != 1:
            holes[r, c] = False
    return holes.copy()


def _clear(box, others):
    r, c, h, w = box
    return all(r + h < rr or rr + hh < r or c + w < cc or cc + ww < c
               for rr, cc, hh, ww in others)


def _free_origins(h, w, height, width, others, margin=0):
    return [(r, c) for r in range(margin, height - h - margin + 1)
            for c in range(margin, width - w - margin + 1)
            if _clear((r, c, h, w), others)]


def _socket_positions(rng, shapes, side):
    # Independent origins, no prescribed corner ownership or tile array.
    order = sorted(range(len(shapes)), key=lambda i: -shapes[i].size)
    for _ in range(40):
        placed, boxes = {}, []
        for i in order:
            h, w = shapes[i].shape
            origins = _free_origins(h, w, side, side, boxes, margin=1)
            if not origins:
                break
            r, c = origins[int(rng.integers(len(origins)))]
            placed[i] = (r, c)
            boxes.append((r, c, h, w))
        if len(placed) == len(shapes):
            return [placed[i] for i in range(len(shapes))]
    return None


def _scatter(rng, frame, inlays, bg):
    side = len(frame)
    boxes = None
    order = list(map(int, rng.permutation(len(inlays))))
    for attempt in range(32):
        # Some compact corner scenes coexist with independent interior offsets.
        if attempt < 12:
            fr, fc = map(int, rng.integers(0, 29 - side, size=2))
        else:
            fr = int(rng.choice([0, 28 - side]))
            fc = int(rng.choice([0, 28 - side]))
        trial = [(fr, fc, side, side)]
        places = {}
        for i in order:
            h, w = inlays[i].shape
            origins = _free_origins(h, w, 28, 28, trial)
            if not origins:
                break
            r, c = origins[int(rng.integers(len(origins)))]
            places[i] = (r, c)
            trial.append((r, c, h, w))
        if len(places) == len(inlays):
            boxes = ((fr, fc), [places[i] for i in range(len(inlays))])
            break
    if boxes is None:
        # Proven fit: at most two 7x7 boxes on the right and two below the
        # <=20 square. Two right boxes use <=15 rows; lower boxes use <=15 cols.
        origins, row, col = [], 0, 0
        for i, obj in enumerate(inlays):
            if i < 2:
                origins.append((row, side + 1))
                row += obj.shape[0] + 1
            else:
                origins.append((side + 1, col))
                col += obj.shape[1] + 1
        boxes = ((0, 0), origins)
    canvas = np.full((30, 30), bg, dtype=np.int8)
    (r, c), positions = boxes
    canvas[r + 1:r + side + 1, c + 1:c + side + 1] = frame
    for obj, (r, c) in zip(inlays, positions):
        h, w = obj.shape
        canvas[r + 1:r + h + 1, c + 1:c + w + 1] = obj
    foreground = np.argwhere(canvas != bg)
    a, b, d, e = _box(foreground.tolist())
    # Every returned scene has a uniform outer ring, but margin widths vary.
    low_r = int(rng.integers(0, a))
    low_c = int(rng.integers(0, b))
    high_r = int(rng.integers(d + 1, 31))
    high_c = int(rng.integers(e + 1, 31))
    return canvas[low_r:high_r, low_c:high_c].copy()


def _generate_pair(rng, counts, frame_rotation, witness=False):
    n = len(counts)
    for attempt in range(160):
        palette = list(map(int, rng.permutation(10)))
        bg, carrier, marker, clue = palette[:4]
        if witness:
            colors = palette[4:4 + n]
        else:
            pool = palette[4:4 + int(rng.integers(2, 7))]
            colors = list(map(int, rng.choice(pool, size=n, replace=True)))
        # At least one congruent pair makes topology essential in every scene,
        # including tests. Larger scenes can have a second congruent pair or
        # independently shaped extra sockets.
        common = _shape(rng, 1, witness=True) if witness else _shape(rng, max(counts[:2]))
        if common is None:
            continue
        second_common = _shape(rng, max(counts[2:])) if n == 4 and rng.random() < .35 else None
        supports, holes, clues, socket_rotations = [], [], [], []
        good = True
        for i, k in enumerate(counts):
            support = common.copy() if i < 2 else second_common.copy() if second_common is not None else _shape(rng, k)
            if support is None:
                good = False
                break
            hole = _make_holes(rng, support, k, witness)
            dots = _choose_clues(rng, support, k)
            if hole is None or dots is None:
                good = False
                break
            material = support & ~hole
            recovered, sizes = _enclosed(material)
            if not np.array_equal(recovered, hole) or len(sizes) != k:
                good = False
                break
            socket_rotation = int(rng.integers(4))
            inlay = dict(material=material, holes=hole, support=support)
            sock = dict(mask=np.rot90(support, socket_rotation))
            if len(_fits(inlay, sock)) != 1:
                good = False
                break
            supports.append(support)
            holes.append(hole)
            clues.append(dots)
            socket_rotations.append(socket_rotation)
        if not good:
            continue
        if n >= 3 and len({min(_key(np.rot90(s, k)) for k in range(4)) for s in supports}) < 2:
            continue
        side = int(rng.integers(13 if n == 2 else 16 if n == 3 else 17, 21))
        oriented = [np.rot90(s, k) for s, k in zip(supports, socket_rotations)]
        positions = _socket_positions(rng, oriented, side)
        if positions is None:
            continue
        frame = np.full((side, side), carrier, dtype=np.int8)
        frame[0, 0] = marker
        out = frame.copy()
        scattered = []
        for i, (support, hole, dots, k, (r, c)) in enumerate(zip(
                supports, holes, clues, socket_rotations, positions)):
            footprint = np.rot90(support, k)
            h, w = footprint.shape
            view = frame[r:r + h, c:c + w]
            view[footprint] = bg
            view[np.rot90(dots, k)] = clue
            out[r:r + h, c:c + w][np.rot90(support & ~hole, k)] = colors[i]
            donor = np.full(support.shape, bg, dtype=np.int8)
            donor[support & ~hole] = colors[i]
            donor_rotation = (k + 1) % 4 if witness and i == 1 else int(rng.integers(4))
            scattered.append(np.rot90(donor, donor_rotation).copy())
        inp = _scatter(rng, np.rot90(frame, frame_rotation), scattered, bg)
        # Check input-derived interpretation and uniqueness before returning;
        # the output above was drawn directly from the construction.
        try:
            decoded = _decode(inp)
            if np.array_equal(decoded['output'], out):
                return {'input': inp, 'output': out}
        except AssertionError:
            pass
    return _fallback_pair(rng, counts, frame_rotation, witness)


def _fallback_pair(rng, counts, frame_rotation, witness=False):
    """A finite, family-compatible safety net after bounded random proposals."""
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, marker, clue = palette[:4]
    frame = np.full((20, 20), carrier, dtype=np.int8)
    frame[0, 0] = marker
    out, donors = frame.copy(), []
    for i, count in enumerate(counts):
        h, w = (5, 6) if i < 2 else (6, 7)
        support = np.ones((h, w), dtype=bool)
        support[0, 0] = False
        holes = np.zeros_like(support)
        if count == 1:
            holes[1:-1, 1:-1] = True
        elif count == 2:
            holes[1, 1] = holes[2, 2] = holes[2, 3] = True
        elif count == 3:
            holes[1, 1] = holes[2, 2] = holes[3, 3] = True
        elif count == 4:
            holes[1, 1] = holes[1, 3] = holes[3, 1] = holes[3, 3] = True
        _require(0 <= count <= 4, "Safety construction only serves sampler counts")
        dots = _choose_clues(rng, support, count)
        _require(dots is not None, "Safety construction has sufficient clue capacity")
        k = int(rng.integers(4))
        shape = np.rot90(support, k)
        hh, ww = shape.shape
        r, c = (2 if i < 2 else 12), (2 if i % 2 == 0 else 12)
        view = frame[r:r + hh, c:c + ww]
        view[shape] = bg
        view[np.rot90(dots, k)] = clue
        material = support & ~holes
        out[r:r + hh, c:c + ww][np.rot90(material, k)] = palette[4 + i]
        donor = np.full((h, w), bg, dtype=np.int8)
        donor[material] = palette[4 + i]
        donors.append(np.rot90(donor, (k + 1) % 4).copy())
    inp = _scatter(rng, np.rot90(frame, frame_rotation), donors, bg)
    _require(np.array_equal(solve(inp), out), "Safety construction must satisfy the same decoder")
    return {'input': inp, 'output': out}


def _counts(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _counts(num_train, 'num_train', 2, 10)
    b = _counts(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    growth = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(a + b):
        split = 'train' if i < a else 'test'
        witness = i == 0
        if witness:
            counts = [1, 2]
            rotation = int(rng.choice([1, 3]))
        elif i == 1:
            counts = [0] + list(map(int, rng.choice([1, 2, 3, 4], 2, replace=False)))
            rotation = 0
        else:
            n = 4 if split == 'test' and growth else int(rng.integers(2, 4))
            counts = list(map(int, rng.choice(5, n, replace=False)))
            rotation = int(rng.integers(4)) if growth else (2 if split == 'test' else int(rng.choice([0, 1, 3])))
        for _ in range(8):
            pair = _generate_pair(rng, counts, rotation, witness)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("Could not construct distinct inputs")
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(a)]
    return puzzle


def _alternative(facts, mode):
    assignment = facts['assignment'].copy()
    if mode == 'area':
        area = [int(x['holes'].sum()) for x in facts['inlays']]
        if len(set(area)) != len(area):
            return None
        slots = sorted(range(len(assignment)), key=lambda i: facts['sockets'][i]['count'])
        donors = sorted(range(len(assignment)), key=lambda i: area[i])
        for s, d in zip(slots, donors):
            assignment[s] = d
        try:
            return _render(facts, assignment)
        except AssertionError:
            return None
    for i, j in combinations(range(len(assignment)), 2):
        assignment[i], assignment[j] = assignment[j], assignment[i]
        try:
            alternative = _render(facts, assignment)
            if not np.array_equal(alternative, facts['output']):
                return alternative
        except AssertionError:
            pass
        assignment = facts['assignment'].copy()
    return None


def _features(f):
    fits = [_fits(f['inlays'][i], s)[0] for i, s in zip(f['assignment'], f['sockets'])]
    areas = {x['count']: int(x['holes'].sum()) for x in f['inlays']}
    area_alt = _alternative(f, 'area')
    return dict(
        side=len(f['scaffold']), count=len(f['sockets']), counts=tuple(sorted(areas)),
        rotation=f['rotation'], palette=len(np.unique(f['input'])),
        background=f['background'], shape=f['input'].shape,
        count_necessary=_alternative(f, 'swap') is not None,
        area_rank_wrong=area_alt is not None and not np.array_equal(area_alt, f['output']),
        area_inversion=any(a < b and areas[a] > areas[b] for a in areas for b in areas),
        connectivity=any(len(_components(x['holes'], True)) < x['count'] for x in f['inlays']),
        rotation_needed=any(k != 0 for k, _, _ in fits),
        nonpositional=any(not np.array_equal(s['clues'], holes) for s, (_, _, holes) in zip(f['sockets'], fits)),
        symmetric=sum(len({_key(np.rot90(x['material'], k)) for k in range(4)}) < 4 for x in f['inlays']),
        exterior_notches=sum(bool(np.any(~x['support'])) for x in f['inlays']),
    )


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal example counts")
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "Inputs must be distinct")
            seen.add(_key(inp))
            f = _decode(inp)
            _require(np.array_equal(out, f['output']), "Incorrect inlay assembly")
            _require(out[0, 0] == f['marker'] and np.sum(out == f['marker']) == 1, "Preserve the corner anchor")
            _require(not np.any(np.isin(out, [f['background'], f['clue']])), "Remove background and dots")
            for color in {x['color'] for x in f['inlays']}:
                _require(np.sum(inp == color) == np.sum(out == color), "Preserve every inlay material cell")
            hole_area = sum(int(x['holes'].sum()) for x in f['inlays'])
            _require(np.sum(out == f['carrier']) == np.sum(inp == f['carrier']) + hole_area,
                     "Exactly the enclosed cavities become scaffold material")
            _require(_alternative(f, 'swap') is not None,
                     "Every generated scene must require topology beyond outline matching")
            if split == 'train':
                train.append(_features(f))
    _require(any(x['count_necessary'] and x['area_rank_wrong'] and x['connectivity'] for x in train),
             "Training must distinguish count, geometry, area, and connectivity")
    _require(any(x['rotation_needed'] and x['rotation'] != 0 for x in train),
             "Training must demonstrate independent rigid rotations and anchoring")
    _require(any(x['nonpositional'] for x in train), "Clue positions must not define the copied pattern")
    _require(any(0 in x['counts'] for x in train), "Demonstrate the zero-cavity case")
    _require(len({x['count'] for x in train}) > 1, "Demonstrate variable object count")


def _normalized(grid):
    colors = {}
    return tuple(colors.setdefault(int(c), len(colors)) for c in np.asarray(grid).flat)


def self_test():
    """Bounded semantic checks and measurements on the 42 provided host seeds."""
    structures, support_shapes, hole_shapes, canvas, sides, counts, count_sets = set(), set(), set(), set(), set(), set(), set()
    rotations, backgrounds, cavity_areas, signatures = set(), set(), set(), set()
    pairs = symmetry = notches = repeated_colors = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                f = _decode(pair['input'])
                z = _features(f)
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), "Input-only solve failed")
                pairs += 1
                canvas.add(z['shape'])
                sides.add(z['side'])
                counts.add(z['count'])
                count_sets.add(z['counts'])
                rotations.add(z['rotation'])
                backgrounds.add(z['background'])
                symmetry += z['symmetric']
                notches += z['exterior_notches']
                repeated_colors += len({x['color'] for x in f['inlays']}) < len(f['inlays'])
                structures.add((pair['output'].shape, _normalized(pair['output'])))
                for x in f['inlays']:
                    support_shapes.add(min(_key(np.rot90(x['support'], k)) for k in range(4)))
                    hole_shapes.add(min(_key(np.rot90(x['holes'], k)) for k in range(4)))
                    cavity_areas.update(x['sizes'])
    _require(len(signatures) == 42, "Seeds must generate distinct puzzles")
    _require(counts == {2, 3, 4} and rotations == {0, 1, 2, 3}, "Exercise counts and orientations")
    _require(len(canvas) >= 25 and len(sides) >= 5 and len(count_sets) >= 12, "Vary layout, scale, and count sets")
    _require(len(support_shapes) >= 70 and len(hole_shapes) >= 70 and len(structures) >= 150,
             "Vary actual geometry and topology, beyond palette changes")
    _require(len(backgrounds) >= 8 and max(cavity_areas) >= 12, "Vary roles and substantial holes")
    _require(symmetry >= 5 and notches >= 50 and repeated_colors >= 10, "Mix symmetry, notches, and repeated material")
    # Exhaust all safety-net count subsets and anchor orientations. These are
    # construction checks, not a bank used by the input-only solver.
    rng = np.random.default_rng(12345)
    for n in (2, 3, 4):
        for ks in combinations(range(5), n):
            for rotation in range(4):
                pair = _fallback_pair(rng, list(ks), rotation, ks == (1, 2))
                _require(np.array_equal(solve(pair['input']), pair['output']), "Fallback fit failed")
    # Every legal count combination, illegal types/ranges, and determinism.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(0, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    q = generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ('input', 'output')), "Seed reproducibility failed")
    # Relocating a one-dot clue preserves the answer, including canonical
    # conversion back into the input scaffold's orientation.
    pair = next(pair for pair in p['train'] if any(s['count'] == 1 for s in _decode(pair['input'])['sockets']))
    f = _decode(pair['input'])
    sock = next(s for s in f['sockets'] if s['count'] == 1)
    safe = np.argwhere(_erode(sock['mask'], True) & ~sock['clues'])
    _require(len(safe) > 0, "A clue relocation witness must exist")
    changed = f['scaffold'].copy()
    r, c = sock['origin']
    h, w = sock['mask'].shape
    view = changed[r:r + h, c:c + w]
    view[sock['clues']] = f['background']
    rr, cc = map(int, safe[0])
    view[rr, cc] = f['clue']
    inp = pair['input'].copy()
    a, b, d, e = f['box']
    inp[a:d, b:e] = np.rot90(changed, -f['rotation'])
    _require(np.array_equal(solve(inp), pair['output']), "Clue positions must be irrelevant")
    # A reflected witness footprint cannot fit its chiral socket.
    witness = next(_decode(x['input']) for x in p['train']
                   if _features(_decode(x['input']))['area_rank_wrong']
                   and _features(_decode(x['input']))['connectivity'])
    donor = witness['inlays'][witness['assignment'][0]]
    reflected = {k: np.fliplr(donor[k]) for k in ('material', 'holes', 'support')}
    _require(not _fits(reflected, witness['sockets'][0]), "Witness must exclude reflection")
    broken = {s: [{k: np.asarray(v).copy() for k, v in x.items()} for x in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return dict(seeds=42, pairs=pairs, canvas_shapes=len(canvas), scaffold_sides=sorted(sides),
                socket_counts=sorted(counts), count_sets=len(count_sets),
                rotational_support_shapes=len(support_shapes), rotational_hole_shapes=len(hole_shapes),
                normalized_outputs=len(structures), cavity_areas=sorted(cavity_areas),
                symmetric_inlays=symmetry, notched_inlays=notches,
                repeated_material_pairs=repeated_colors, independent_solver=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
