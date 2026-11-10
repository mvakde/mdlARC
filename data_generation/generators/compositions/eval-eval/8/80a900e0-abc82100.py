"""Coincidence printing: intersect invisible endpoint rays, then register masks.

No parent modules, files, seed, or training examples are used by solve().
Generation and diagnostics use bounded in-memory searches and write no files.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import numbers

import numpy as np


DIAGONALS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
CARDINALS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid grid dimensions")
    _need(np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9)),
          "grid must contain integer colors 0..9")
    return grid.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _add(p, q, scale=1):
    return p[0] + scale * q[0], p[1] + scale * q[1]


def _outline(center, radius):
    return {_add(center, (r, c)) for r in range(-radius, radius + 1)
            for c in range(-radius, radius + 1) if abs(r) + abs(c) == radius}


def _cap(center, radius, direction):
    dr, dc = direction
    return [_add(center, (dr * (k + 1), dc * (radius - k + 1)))
            for k in range(radius + 1)]


def _base(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in remaining:
                        remaining.remove(p)
                        component.add(p)
                        stack.append(p)
        result.append(component)
    return result


def _background(grid):
    parity = np.indices(grid.shape).sum(axis=0) % 2
    colors = []
    for k in (0, 1):
        counts = np.bincount(grid[parity == k].astype(int), minlength=10)
        winner = int(counts.argmax())
        _need(counts[winner] > np.count_nonzero(parity == k) / 2, "checkerboard lacks a strict majority")
        colors.append(winner)
    _need(colors[0] != colors[1], "checkerboard colors must differ")
    return tuple(colors)


def _emitters(grid, backgrounds):
    """Find complete radius-2/3 outlines with at least one complete side cap."""
    h, w = grid.shape
    records = []
    for radius in (2, 3):
        if min(h, w) < 2 * radius + 3:
            continue
        out_shape = h - 2 * radius, w - 2 * radius
        offsets = sorted(_outline((0, 0), radius))
        dr, dc = offsets[0]
        colors = grid[radius + dr:radius + dr + out_shape[0],
                      radius + dc:radius + dc + out_shape[1]]
        match = (colors != backgrounds[0]) & (colors != backgrounds[1])
        for dr, dc in offsets[1:]:
            match &= grid[radius + dr:radius + dr + out_shape[0],
                          radius + dc:radius + dc + out_shape[1]] == colors
        for r, c in np.argwhere(match):
            center = int(r) + radius, int(c) + radius
            structural = int(colors[r, c])
            caps = {}
            for direction in DIAGONALS:
                cells = _cap(center, radius, direction)
                if not all(_inside(p, grid.shape) for p in cells):
                    continue
                values = {int(grid[p]) for p in cells}
                if len(values) == 1 and not values.intersection((*backgrounds, structural)):
                    caps[direction] = values.pop()
            if caps:
                footprint = _outline(center, radius)
                for direction in caps:
                    footprint.update(_cap(center, radius, direction))
                records.append(dict(center=center, radius=radius, structural=structural,
                                    caps=caps, footprint=footprint))
    _need(2 <= len(records) <= 4, "expected two to four capped diamonds")
    occupied = set()
    for rec in records:
        _need(not occupied.intersection(rec['footprint']), "overlapping emitter objects")
        occupied.update(rec['footprint'])
    return sorted(records, key=lambda rec: rec['center']), occupied


def _parse(grid):
    """Recover all semantic roles, labels, and registrations from this input."""
    backgrounds = _background(grid)
    base = _base(grid.shape, backgrounds)
    emitters, apparatus = _emitters(grid, backgrounds)
    foreign = set(map(tuple, np.argwhere(grid != base))) - apparatus
    cap_colors = {color for rec in emitters for color in rec['caps'].values()}
    structural = {rec['structural'] for rec in emitters}
    interpretations = []
    for neutral in sorted({int(grid[p]) for p in foreign} - cap_colors - structural - set(backgrounds)):
        masks = _components(p for p in foreign if int(grid[p]) == neutral)
        if not 1 <= len(masks) <= 2 or any(len(mask) < 3 for mask in masks):
            continue
        records, labels = [], set()
        for mask in masks:
            near = {_add(p, (dr, dc)) for p in mask for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            candidates = []
            for q in sorted(near):
                if not _inside(q, grid.shape) or int(grid[q]) not in (*backgrounds, neutral):
                    continue
                for direction in CARDINALS:
                    target_p, source_p = _add(q, direction, -1), _add(q, direction, -2)
                    if source_p not in foreign or target_p not in foreign:
                        continue
                    source, target = int(grid[source_p]), int(grid[target_p])
                    if source not in cap_colors or target in (*backgrounds, neutral, source, *structural):
                        continue
                    candidates.append(dict(source=source, target=target, q=q, direction=direction,
                                           labels={source_p, target_p}, mask=mask,
                                           offsets=tuple(sorted((r - q[0], c - q[1]) for r, c in mask))))
            if len(candidates) != 1:
                break
            rec = candidates[0]
            if labels.intersection(rec['labels']):
                break
            labels.update(rec['labels'])
            records.append(rec)
        else:
            if (len({rec['source'] for rec in records}) == len(records)
                    and labels | {p for mask in masks for p in mask} == foreign):
                interpretations.append((neutral, records, labels))
    _need(len(interpretations) == 1, "instructions do not have one input-only interpretation")
    neutral, records, labels = interpretations[0]
    instruction_cells = labels | {p for rec in records for p in rec['mask']}
    return dict(base=base, backgrounds=backgrounds, emitters=emitters, apparatus=apparatus,
                neutral=neutral, instructions=records, instruction_cells=instruction_cells)


def _ray_visits(emitters, shape, all_lanes=False):
    visits = defaultdict(list)
    for i, rec in enumerate(emitters):
        for direction, color in sorted(rec['caps'].items()):
            cap = _cap(rec['center'], rec['radius'], direction)
            starts = cap if all_lanes else (cap[0], cap[-1])
            for endpoint in starts:
                p = _add(endpoint, direction)
                while _inside(p, shape):
                    visits[p].append((i, color, direction[0] * direction[1]))
                    p = _add(p, direction)
    return visits


def _crossings(visits, source, color_gate=True):
    anchors = set()
    for p, rays in visits.items():
        for i, a in enumerate(rays):
            for b in rays[i + 1:]:
                if a[0] != b[0] and a[2] != b[2]:
                    if (a[1] == b[1] == source if color_gate else source in (a[1], b[1])):
                        anchors.add(p)
    return anchors


def _paint(grid, parsed, *, all_lanes=False, color_gate=True, point_only=False,
           centered=False, closed_form=False):
    """All anchors come from the original input; printed cells do not emit."""
    emitters = parsed['emitters']
    if closed_form:
        # Algebraic membership rather than walking endpoint paths. Parsing is shared.
        visits = defaultdict(list)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                for i, rec in enumerate(emitters):
                    for (dr, dc), color in rec['caps'].items():
                        u = (r - rec['center'][0]) * dr
                        v = (c - rec['center'][1]) * dc
                        if min(u, v) >= 2 and abs(u - v) == rec['radius']:
                            visits[(r, c)].append((i, color, dr * dc))
    else:
        visits = _ray_visits(emitters, grid.shape, all_lanes)
    output = grid.copy()
    for p in parsed['instruction_cells']:
        output[p] = parsed['base'][p]
    paint, anchors_by_source = {}, {}
    for rec in parsed['instructions']:
        anchors = _crossings(visits, rec['source'], color_gate)
        anchors_by_source[rec['source']] = anchors
        offsets = rec['offsets']
        if centered:
            rs, cs = zip(*rec['mask'])
            center = (min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2
            offsets = tuple((r - center[0], c - center[1]) for r, c in rec['mask'])
        if point_only:
            offsets = ((0, 0),)
        for anchor in sorted(anchors):
            for offset in offsets:
                p = _add(anchor, offset)
                if _inside(p, grid.shape):
                    # Counterfactuals are allowed conflicting paint only for teaching diagnostics.
                    if not (all_lanes or not color_gate or centered or point_only):
                        _need(p not in paint or paint[p] == rec['target'], "conflicting stamp colors")
                    paint[p] = rec['target']
    for p, color in paint.items():
        output[p] = color
    return output, visits, anchors_by_source, paint


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique coincidence-printing output using only input pixels."""
    grid = _grid(input_grid)
    return _paint(grid, _parse(grid))[0]


def _rotate(p, direction):
    r, c = p
    dr, dc = direction
    return r * dc + c * dr, -r * dr + c * dc


def _mask(rng, hole, compact=False):
    if hole:
        top, bottom, right = int(rng.integers(1, 3)), int(rng.integers(1, 3)), int(rng.integers(1, 4))
        if min(top, bottom, right) > 1:
            # The visible label rule permits only a registration within one
            # Chebyshev step of the mask. Enforce this before spatial search.
            axis = int(rng.integers(3))
            if axis == 0:
                top = 1
            elif axis == 1:
                bottom = 1
            else:
                right = 1
        left = int(rng.integers(-1, 1))
        cells = {(r, right) for r in range(-top, bottom + 1)}
        cells |= {(r, c) for r in (-top, bottom) for c in range(left, right + 1)}
        # The label ray enters the C's open middle; the registration is an empty hole.
    else:
        cells = {(0, 0)}
        desired = int(rng.integers(3, 6 if compact else 10))
        for _ in range(100):
            if len(cells) >= desired:
                break
            start = sorted(cells)[int(rng.integers(len(cells)))]
            p = _add(start, CARDINALS[int(rng.integers(4))])
            if (-(1 if compact else 2) <= p[0] <= (1 if compact else 2)
                    and 0 <= p[1] <= (2 if compact else 3)):
                cells.add(p)
        if len(cells) < 3:
            cells |= {(0, 1), (0, 2)}
    direction = CARDINALS[int(rng.integers(4))]
    return tuple(sorted(_rotate(p, direction) for p in cells)), direction


def _scene(rng, n, source_colors, structural, radius=None, core_directions=None, simple=False):
    """Construct a crossing core, then freely add emitters and cap subsets."""
    low = 19 if n == 2 else 24 if n == 3 else 28
    shape = tuple(int(rng.integers(low, 31)) for _ in range(2))
    anchor = tuple(int(rng.integers(6, extent - 6)) for extent in shape)
    if core_directions is None:
        d1 = DIAGONALS[int(rng.integers(4))]
        d2 = [d for d in DIAGONALS if d[0] * d[1] != d1[0] * d1[1]][int(rng.integers(2))]
    else:
        d1, d2 = core_directions
    records = []
    for direction in (d1, d2):
        rad = radius if radius is not None else int(rng.integers(2, 4))
        endpoint = _cap((0, 0), rad, direction)[int(rng.choice((0, -1)))]
        center = _add(_add(anchor, direction, -int(rng.integers(2, 8))), endpoint, -1)
        records.append(dict(center=center, radius=rad, structural=structural,
                            caps={direction: source_colors[0]}))
    for extra in range(n - 2):
        rad = int(rng.integers(2, 4))
        existing = _ray_visits(records, shape)
        # Aim a new bundle through an existing ray. The third emitter supplies
        # a different-color near miss; a fourth can also give that color a mate.
        eligible = [(p, rays[0]) for p, rays in sorted(existing.items()) if len(rays) == 1
                    and (extra == 0 or rays[0][1] == source_colors[1])]
        if eligible:
            crossing, ray = eligible[int(rng.integers(len(eligible)))]
            choices = [d for d in DIAGONALS if d[0] * d[1] != ray[2]]
            direction = choices[int(rng.integers(2))]
            endpoint = _cap((0, 0), rad, direction)[int(rng.choice((0, -1)))]
            center = _add(_add(crossing, direction, -int(rng.integers(2, 8))), endpoint, -1)
            caps = {direction: source_colors[1]}
        else:
            center = tuple(int(rng.integers(rad + 1, extent - rad - 1)) for extent in shape)
            caps = {}
        records.append(dict(center=center, radius=rad, structural=structural, caps=caps))
    for rec in records:
        for direction in DIAGONALS:
            if not simple and direction not in rec['caps'] and rng.random() < (.18 if rec['caps'] else .6):
                rec['caps'][direction] = int(rng.choice(source_colors))
        if not rec['caps']:
            rec['caps'][DIAGONALS[int(rng.integers(4))]] = int(rng.choice(source_colors))
        rec['footprint'] = _outline(rec['center'], rec['radius'])
        for direction in rec['caps']:
            rec['footprint'].update(_cap(rec['center'], rec['radius'], direction))
    occupied = set()
    for rec in records:
        if not all(_inside(p, shape) for p in rec['footprint']) or occupied.intersection(rec['footprint']):
            return None
        occupied.update(rec['footprint'])
    visits = _ray_visits(records, shape)
    # The rays are virtual construction lines: apparatus does not occlude them.
    # All qualifying crossing pairs are considered, with each anchor deduplicated.
    if not _crossings(visits, source_colors[0]):
        return None
    return shape, records, occupied, visits


def _make_pair(rng, n, *, source=None, target=None, teaching=None, require_negative=False,
               core_directions=None):
    # Sample local palettes. Only two teaching labels deliberately share a source.
    chosen = [int(c) for c in rng.permutation(10)]
    source = chosen.pop() if source is None else int(source)
    chosen = [c for c in chosen if c != source]
    target = chosen.pop() if target is None else int(target)
    chosen = [c for c in chosen if c != target]
    backgrounds = tuple(chosen[:2])
    structural, neutral, second_source, second_target = chosen[2:6]
    source_colors = (source, second_source)
    for _ in range(900):
        scene = _scene(rng, n, source_colors, structural,
                       radius=2 if teaching == 'occupied' else 3 if teaching == 'hole' else None,
                       core_directions=core_directions, simple=teaching == 'occupied')
        if scene is None:
            continue
        shape, emitters, apparatus, visits = scene
        negative = {p for p, rays in visits.items() for j, a in enumerate(rays) for b in rays[j + 1:]
                    if a[0] != b[0] and a[2] != b[2] and a[1] != b[1]}
        if require_negative and not negative:
            continue
        specs = [(source, target, teaching == 'hole' if teaching else rng.random() < .5)]
        if _crossings(visits, second_source) and rng.random() < .8:
            specs.append((second_source, target if rng.random() < .3 else second_target, rng.random() < .5))
        grid = _base(shape, backgrounds)
        for rec in emitters:
            for p in _outline(rec['center'], rec['radius']):
                grid[p] = structural
            for direction, color in rec['caps'].items():
                for p in _cap(rec['center'], rec['radius'], direction):
                    grid[p] = color
        instructions, reserved, mask_halo, planned = [], set(apparatus), set(), {}
        for src, ink, hole in specs:
            accepted = False
            for _ in range(20):
                offsets, direction = _mask(rng, hole, compact=teaching == 'occupied')
                anchors = _crossings(visits, src)
                paint = {_add(a, offset) for a in anchors for offset in offsets if _inside(_add(a, offset), shape)}
                if (not paint or paint.intersection(reserved)
                        or any(p in planned and planned[p] != ink for p in paint)):
                    continue
                # Teaching shows at least one complete mask; other copies may clip.
                if teaching and not any(all(_inside(_add(a, offset), shape) for offset in offsets) for a in anchors):
                    continue
                relative = set(offsets) | {(0, 0), _add((0, 0), direction, -1), _add((0, 0), direction, -2)}
                r0, r1 = min(r for r, c in relative), max(r for r, c in relative)
                c0, c1 = min(c for r, c in relative), max(c for r, c in relative)
                positions = [(r, c) for r in range(-r0, shape[0] - r1)
                             for c in range(-c0, shape[1] - c1)]
                for index in rng.permutation(len(positions)):
                    q = positions[int(index)]
                    mask = {_add(q, offset) for offset in offsets}
                    labels = {_add(q, direction, -1), _add(q, direction, -2)}
                    cells = mask | labels | {q}
                    if cells.intersection(reserved | paint | set(planned)) or mask.intersection(mask_halo):
                        continue
                    candidate = grid.copy()
                    for p in mask:
                        candidate[p] = neutral
                    candidate[_add(q, direction, -2)] = src
                    candidate[_add(q, direction, -1)] = ink
                    try:
                        parsed = _parse(candidate)
                    except ValueError:
                        continue
                    if len(parsed['instructions']) != len(instructions) + 1:
                        continue
                    recovered = [rec for rec in parsed['instructions'] if rec['source'] == src]
                    if (len(recovered) != 1 or recovered[0]['q'] != q
                            or recovered[0]['offsets'] != offsets or recovered[0]['direction'] != direction
                            or recovered[0]['target'] != ink):
                        continue
                    grid = candidate
                    instructions.append((src, ink, q, offsets))
                    reserved.update(cells)
                    mask_halo |= {_add(p, (dr, dc)) for p in mask for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                    planned.update({p: ink for p in paint})
                    accepted = True
                    break
                if accepted:
                    break
            if not accepted:
                break
        else:
            try:
                parsed = _parse(grid)
                output, _, anchors_by_source, paint = _paint(grid, parsed)
                if set(paint).intersection(parsed['apparatus'] | parsed['instruction_cells']):
                    continue
                if teaching and np.array_equal(output, _paint(grid, parsed, all_lanes=True)[0]):
                    continue
                if require_negative and np.array_equal(output, _paint(grid, parsed, color_gate=False)[0]):
                    continue
                if teaching and np.array_equal(output, _paint(grid, parsed, centered=True)[0]):
                    continue
                if teaching:
                    primary = next(rec for rec in parsed['instructions'] if rec['source'] == source)
                    copies = [(a, {_add(a, off) for off in primary['offsets']})
                              for a in anchors_by_source[source]]
                    if teaching == 'occupied' and not any(
                            all(_inside(p, shape) for p in cells)
                            and not any(cells.intersection(other) for b, other in copies if b != a)
                            for a, cells in copies):
                        continue
                    if teaching == 'hole' and not any(
                            all(_inside(p, shape) for p in cells) and output[a] == parsed['base'][a]
                            for a, cells in copies):
                        continue
                # Construction witness: exact translated cells, computed before pixel parsing.
                expected = grid.copy()
                for p in parsed['instruction_cells']:
                    expected[p] = parsed['base'][p]
                for p, ink in planned.items():
                    expected[p] = ink
                if not np.array_equal(expected, output):
                    continue
                return {'input': grid, 'output': output}
            except ValueError:
                continue
    raise RuntimeError('bounded scene construction exhausted')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= value <= high:
        raise ValueError(name + ' must be an integer in ' + str((low, high)))
    return int(value)


def _key(grid):
    return tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    common_source, ink_a, ink_b = map(int, rng.choice(10, 3, replace=False))
    puzzle, seen, teaching_directions = {'train': [], 'test': []}, set(), None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            teaching = ('occupied' if i == 0 else 'hole' if i == 1 else None) if split == 'train' else None
            n = 2 if teaching == 'occupied' else int(rng.integers(3, 5)) if split == 'test' else int(rng.integers(2, 5))
            if teaching == 'hole':
                n = int(rng.integers(3, 5))
            for _ in range(8):
                try:
                    pair = _make_pair(rng, n, source=common_source if teaching else None,
                                      target=ink_a if teaching == 'occupied' else ink_b if teaching == 'hole' else None,
                                      teaching=teaching, require_negative=teaching == 'hole' or split == 'test',
                                      core_directions=teaching_directions if teaching == 'hole' else None)
                except RuntimeError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if teaching == 'occupied':
                        teaching_directions = tuple((-d[0], -d[1]) for e in _parse(pair['input'])['emitters']
                                                    for d in e['caps'])
                    break
            else:
                raise RuntimeError('distinct-input search exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair, check_algebra=False):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _need(inp.shape == out.shape, 'shape changed')
    parsed = _parse(inp)
    expected, visits, anchors, paint = _paint(inp, parsed)
    _need(np.array_equal(out, expected), 'wrong coincidence-printing output')
    _need(not np.array_equal(inp, out), 'unchanged pair')
    # Passing through a static object, collinear bundles, and multiple incident
    # rays do not introduce an order: eligibility is a set of pairwise crossings.
    _need(all(anchors.values()), 'an instruction has no eligible crossing')
    _need(not set(paint).intersection(parsed['apparatus'] | parsed['instruction_cells']), 'print crosses apparatus')
    _need(all(out[p] == inp[p] for p in parsed['apparatus']), 'emitter changed')
    _need(all(out[p] == parsed['base'][p] for p in parsed['instruction_cells']), 'instruction not erased')
    changed = set(map(tuple, np.argwhere(inp != out)))
    _need(changed <= parsed['instruction_cells'] | set(paint), 'unrelated cell changed')
    _need(not np.array_equal(out, _paint(inp, parsed, point_only=True)[0]), 'stencil has no observable effect')
    if check_algebra:
        _need(np.array_equal(out, _paint(inp, parsed, closed_form=True)[0]), 'algebraic ray calculation differs')
    return parsed, visits, anchors, paint


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _need(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
          and len(puzzle['train']) > len(puzzle['test']), 'invalid counts')
    seen, evidence, mappings, demonstrated_directions = set(), Counter(), defaultdict(set), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            parsed, visits, anchors, _ = _inspect(pair)
            if split == 'train':
                evidence['endpoints'] += not np.array_equal(out, _paint(inp, parsed, all_lanes=True)[0])
                evidence['color_gate'] += not np.array_equal(out, _paint(inp, parsed, color_gate=False)[0])
                evidence['registration'] += not np.array_equal(out, _paint(inp, parsed, centered=True)[0])
                for rec in parsed['instructions']:
                    mappings[rec['source']].add(rec['target'])
                    copies = [(a, {_add(a, offset) for offset in rec['offsets']})
                              for a in anchors[rec['source']]]
                    complete = any(all(_inside(p, inp.shape) for p in cells) for a, cells in copies)
                    if complete:
                        evidence['hole' if (0, 0) not in rec['offsets'] else 'occupied'] += 1
                    if (0, 0) not in rec['offsets']:
                        evidence['visible_hole'] += any(all(_inside(p, inp.shape) for p in cells)
                                                        and out[a] == parsed['base'][a] for a, cells in copies)
                    else:
                        evidence['isolated_copy'] += any(all(_inside(p, inp.shape) for p in cells)
                            and not any(cells.intersection(other) for b, other in copies if b != a)
                            for a, cells in copies)
                    for emitter in parsed['emitters']:
                        for direction, color in emitter['caps'].items():
                            if color == rec['source'] and any(
                                    min((a[0] - emitter['center'][0]) * direction[0],
                                        (a[1] - emitter['center'][1]) * direction[1]) >= 2
                                    and abs((a[0] - emitter['center'][0]) * direction[0]
                                            - (a[1] - emitter['center'][1]) * direction[1]) == emitter['radius']
                                    for a in anchors[rec['source']]):
                                demonstrated_directions.add(direction)
    _need(all(evidence[k] for k in ('endpoints', 'color_gate', 'registration', 'hole', 'occupied')),
          'training lacks a required counterexample or complete registered stamp')
    _need(evidence['isolated_copy'] and evidence['visible_hole'], 'training obscures registration evidence')
    _need(demonstrated_directions == set(DIAGONALS), 'training lacks an active outgoing direction')
    _need(any(len(inks) > 1 for inks in mappings.values()), 'training fails to show local color mapping')


def self_test():
    """42-seed structural audit, counterfactual witnesses, and bounded regressions."""
    stats = Counter()
    sets = defaultdict(set)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                parsed, visits, anchors, paint = _inspect(pair, check_algebra=True)
                fingerprint.update(bytes(inp.shape) + inp.tobytes() + pair['output'].tobytes())
                stats['pairs'] += 1
                sets['shapes'].add(inp.shape)
                sets['backgrounds'].add(parsed['backgrounds'])
                sets['emitter_counts'].add(len(parsed['emitters']))
                sets['instruction_counts'].add(len(parsed['instructions']))
                roles = {}
                sets['structures'].add(tuple(tuple(roles.setdefault(int(v), len(roles)) for v in row) for row in inp))
                for e in parsed['emitters']:
                    sets['radii'].add(e['radius'])
                    sets['directions'].update(e['caps'])
                    sets['cap_counts'].add(len(e['caps']))
                negative = len({p for p, rays in visits.items() for j, a in enumerate(rays) for b in rays[j + 1:]
                                if a[0] != b[0] and a[2] != b[2] and a[1] != b[1]})
                stats['unequal_color_crossings'] += negative
                stats['positive_crossings'] += sum(map(len, anchors.values()))
                for rec in parsed['instructions']:
                    sets['label_directions'].add(rec['direction'])
                    sets['mask_sizes'].add(len(rec['offsets']))
                    rs, cs = zip(*rec['mask'])
                    sets['mask_geometries'].add(tuple(sorted((r - min(rs), c - min(cs)) for r, c in rec['mask'])))
                    stats['hole_instructions'] += (0, 0) not in rec['offsets']
                    footprints = [{_add(a, off) for off in rec['offsets'] if _inside(_add(a, off), inp.shape)}
                                  for a in anchors[rec['source']]]
                    stats['clipped_stamps'] += sum(len(f) < len(rec['offsets']) for f in footprints)
                    stats['overlapping_stamps'] += sum(map(len, footprints)) - len(set().union(*footprints))
        sets['puzzle_hashes'].add(fingerprint.hexdigest())
    assert len(sets['puzzle_hashes']) == 42
    assert len(sets['shapes']) >= 40 and len(sets['structures']) >= 150
    assert sets['emitter_counts'] == {2, 3, 4} and sets['instruction_counts'] == {1, 2}
    assert sets['radii'] == {2, 3} and sets['directions'] == set(DIAGONALS)
    assert sets['label_directions'] == set(CARDINALS)
    assert len(sets['mask_geometries']) >= 40 and len(sets['mask_sizes']) >= 7
    assert len(sets['backgrounds']) >= 30
    assert stats['clipped_stamps'] > 0 and stats['overlapping_stamps'] > 0
    # Exact coordinate regression: two radius-two endpoint bundles cross at four cells.
    emitters = [dict(center=(4, 4), radius=2, caps={(1, 1): 6}),
                dict(center=(4, 14), radius=2, caps={(1, -1): 6})]
    visits = _ray_visits(emitters, (20, 20))
    assert _crossings(visits, 6) == {(7, 9), (9, 7), (9, 11), (11, 9)}
    emitters[1]['caps'][(1, -1)] = 7
    assert not _crossings(_ray_visits(emitters, (20, 20)), 6)
    assert _crossings(_ray_visits(emitters, (20, 20)), 6, False)
    assert not _crossings({(0, 0): [(0, 6, 1), (1, 6, 1)]}, 6)
    original = generate_puzzle(0, 2, 1)
    repeat = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in original for a, b in zip(original[s], repeat[s])
               for k in ('input', 'output'))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False),
                 (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: g.copy() for k, g in pair.items()} for pair in original[s]] for s in original}
    parsed = _parse(bad['test'][0]['input'])
    p = next(iter(_paint(bad['test'][0]['input'], parsed)[3]))
    bad['test'][0]['output'][p] = (int(bad['test'][0]['output'][p]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    return dict(stats, distinct_shapes=len(sets['shapes']), distinct_masks=len(sets['mask_geometries']),
                distinct_structures=len(sets['structures']), seeds=42)
