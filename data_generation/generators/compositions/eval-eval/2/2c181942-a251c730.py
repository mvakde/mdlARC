"""Radial printing press: reusable, keyed silhouettes printed onto hub bars.

Only the input supplies the dictionary and routing. The prediction path never
reads files, samples randomness, or consults generation state. Standard library
and NumPy only. No files or execution traces are written by this module.
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
import hashlib
import json

import numpy as np


HOST_SEEDS = (
    0, 507613780, 134587260, 338521430, 24618298, 924544674, 448504608,
    256762473, 765696697, 885155046, 479108938, 330067297, 379778271,
    32961071, 396144707, 566209051, 201874024, 999066626, 301995012,
    122293451, 535833842, 78649587, 327421406, 754184977, 62870541,
    208348345, 922594915, 511279523, 858710399, 620978158, 584499354,
    104039787, 172424972, 748121978, 584713713, 413234023, 941937207,
    374266462, 337229217, 910420702, 935045517, 801410518,
)
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_BAR_CELLS = (((0, 1), (0, 2)), ((1, 3), (2, 3)),
              ((3, 2), (3, 1)), ((2, 0), (1, 0)))
_ORIGINS = ((0, 1), (1, 3), (3, 2), (2, 0))


def _check_counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                    (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer), 'integer colors required'
    assert np.all((0 <= grid) & (grid <= 9))
    return grid


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _D8:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(sorted(cells))
    return result


def _rotate(r, c, turns):
    for _ in range(turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _hub_records(labels):
    return [(r, c, int(color)) for side, color in enumerate(labels)
            for r, c in _BAR_CELLS[side]]


def _hub_labels(local, bg):
    if local.shape != (4, 4):
        return None
    labels = tuple(int(local[cells[0]]) for cells in _BAR_CELLS)
    if len(set(labels)) != 4 or bg in labels:
        return None
    expected = np.full((4, 4), bg, dtype=int)
    for r, c, color in _hub_records(labels):
        expected[r, c] = color
    return labels if np.array_equal(local, expected) else None


def _normalize_stamp(local, bg):
    """Return key, colored offsets, and native clockwise orientation."""
    candidates = []
    for turns in range(4):
        image = np.rot90(local, turns)
        h, w = image.shape
        if h < 2 or w < 2 or w % 2:
            continue
        a = w // 2 - 1
        columns = np.flatnonzero(image[-1] != bg).tolist()
        if columns != [a, a + 1]:
            continue
        key = int(image[-1, a])
        if image[-1, a + 1] != key or np.count_nonzero(image == key) != 2:
            continue
        colors, counts = np.unique(image[image != bg], return_counts=True)
        if any(int(n) < 3 for color, n in zip(colors, counts) if int(color) != key):
            continue
        signature = tuple(sorted((int(r - h + 1), int(c - a), int(image[r, c]))
                                 for r, c in np.argwhere(image != bg)))
        if len(signature) <= 2:
            continue
        candidates.append((key, signature, turns))
    assert len(candidates) == 1, 'stamp needs one unambiguous two-cell terminal key'
    return candidates[0]


def _body_at(template, side, r, c):
    ar, ac = _ORIGINS[side]
    result = []
    for dr, dc, color in template:
        if dr == 0:
            continue  # The key already exists as the receiving socket bar.
        rr, cc = _rotate(dr, dc, side)
        result.append((r + ar + rr, c + ac + cc, color))
    return result


def _inspect(input_grid):
    """Input-only interpretation, with checks before any answer is consulted."""
    grid = _grid(input_grid)
    bg = int(np.argmax(np.bincount(grid.ravel().astype(int), minlength=10)))
    assert np.count_nonzero(grid == bg) > grid.size // 2, 'majority background required'
    hubs, references, templates = [], [], {}
    for cells in _components(grid != bg):
        rr, cc = np.array(cells).T
        r, c = int(rr.min()), int(cc.min())
        local = grid[r:int(rr.max()) + 1, c:int(cc.max()) + 1]
        # A bounding box cannot include another separated component. Check
        # this explicitly rather than treating unrelated ink as part of it.
        own = np.full(local.shape, bg, dtype=grid.dtype)
        own[rr - r, cc - c] = grid[rr, cc]
        labels = _hub_labels(own, bg)
        if labels is not None:
            hubs.append((r, c, labels))
            continue
        key, sig, native = _normalize_stamp(own, bg)
        assert key not in templates or templates[key] == sig, 'conflicting dictionary entries'
        templates[key] = sig
        references.append({'key': key, 'template': sig, 'native': native,
                           'box': (r, c, *own.shape), 'cells': cells})
    assert hubs and references, 'at least one hub and reference required'
    inks = {color for sig in templates.values() for r, c, color in sig if r < 0}
    labels = {k for _, _, keys in hubs for k in keys}
    assert not (inks & (set(templates) | labels | {bg})), 'key and ink roles must be distinct'
    expected = np.full(grid.shape, bg, dtype=int)
    for r, c, keys in hubs:
        for dr, dc, color in _hub_records(keys):
            expected[r + dr, c + dc] = color
    requests, orientations, bodies = Counter(), [], []
    for r, c, keys in hubs:
        for side, key in enumerate(keys):
            if key not in templates:
                continue
            body = _body_at(templates[key], side, r, c)
            assert body
            for rr, cc, color in body:
                assert 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1], 'clipped print'
                assert expected[rr, cc] == bg, 'prints overlap a body or hub'
                expected[rr, cc] = color
            requests[key] += 1
            orientations.append(side)
            bodies.append((key, side, r, c, tuple(body)))
    assert requests, 'at least one supported socket required'
    assert np.count_nonzero(expected == bg) > expected.size // 2
    return expected, {'background': bg, 'hubs': hubs, 'references': references,
                      'templates': templates, 'requests': requests,
                      'orientations': orientations, 'bodies': bodies}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the dictionary and print its oriented patterns from input only."""
    return _inspect(input_grid)[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _make_template(rng, key, inks, forced_width=None,
                   force_asymmetric=False, force_multicolor=False):
    width = int(forced_width or _pick(rng, (2, 4, 4, 6, 6)))
    depth = int(rng.integers(3, 6))
    a = width // 2 - 1
    mask = np.zeros((depth, width), dtype=bool)
    mask[:, a:a + 2] = True
    style = int(rng.integers(4))
    if width > 2:
        if style == 0:
            # Wide body and independently variable narrow stem length.
            body_depth = int(rng.integers(1, depth - 1))
            mask[:body_depth] = True
        elif style == 1:
            # Diagonal wings; keep an extra stem row behind the colored key.
            reach = width // 2 - 1
            depth = max(depth, reach + 2)
            mask = np.zeros((depth, width), dtype=bool)
            mask[reach:, a:a + 2] = True
            for row in range(reach):
                d = reach - row
                mask[row, a - d] = mask[row, a + 1 + d] = True
                if rng.random() < .3:
                    mask[row, a - d:a + 2 + d] = True
        else:
            # Unequal shoulders and optional holes, with a connected spine.
            for row in range(depth - 2):
                lo = int(rng.integers(a + 1))
                hi = int(rng.integers(a + 2, width + 1))
                mask[row, lo:hi] = True
            mask[int(rng.integers(depth - 2)), :a + 1] = True
            mask[int(rng.integers(depth - 2)), a + 1:] = True
            if style == 3:
                for row, col in np.argwhere(mask[:-2]):
                    if col not in (a, a + 1) and rng.random() < .22:
                        mask[row, col] = False
                # Restore extent without detaching diagonal wings.
                mask[0] = True
    if force_asymmetric:
        assert width >= 4
        if depth < 4:
            mask = np.pad(mask, ((1, 0), (0, 0)))
            depth += 1
        mask[0] = True
        mask[1, a:a + 2] = True
        mask[1, 0], mask[1, -1] = True, False
    # Notches can isolate a lower shoulder pixel. Keep only the component
    # attached to the key; the full top row and central spine retain extent.
    connected = next(cells for cells in _components(mask) if (depth - 1, a) in cells)
    mask[:] = False
    rr, cc = np.array(connected).T
    mask[rr, cc] = True
    if rng.random() < .5:
        mask = np.fliplr(mask).copy()
    body = [(int(r), int(c)) for r, c in np.argwhere(mask) if r != depth - 1]
    ncolors = 2 if len(body) >= 6 and (force_multicolor or rng.random() < .5) else 1
    assert not force_multicolor or ncolors == 2
    chosen_inks = list(rng.choice(inks, size=ncolors, replace=False))
    assignment = chosen_inks * 3
    assignment += [_pick(rng, chosen_inks) for _ in range(len(body) - len(assignment))]
    if ncolors == 1:
        assignment = chosen_inks * len(body)
    rng.shuffle(assignment)
    records = [(0, 0, int(key)), (0, 1, int(key))]
    records += [(r - depth + 1, c - a, int(color))
                for (r, c), color in zip(body, assignment)]
    return tuple(sorted(records))


def _normalize_records(records):
    minr = min(r for r, _, _ in records)
    minc = min(c for _, c, _ in records)
    normalized = [(r - minr, c - minc, color) for r, c, color in records]
    shape = (max(r for r, _, _ in normalized) + 1,
             max(c for _, c, _ in normalized) + 1)
    return normalized, shape, (minr, minc)


def _source_records(rng, template, side=None):
    side = int(rng.integers(4)) if side is None else side
    records = [(*_rotate(r, c, side), color) for r, c, color in template]
    records, shape, _ = _normalize_records(records)
    return {'shape': shape, 'input': records, 'output': []}


def _hub_item(keys, templates):
    inp = _hub_records(keys)
    out = list(inp)
    for side, key in enumerate(keys):
        if key in templates:
            out += _body_at(templates[key], side, 0, 0)
    output, shape, (r, c) = _normalize_records(out)
    assert len({(r, c) for r, c, _ in out}) == len(out), 'construction overlap'
    return {'shape': shape, 'input': [(rr - r, cc - c, color) for rr, cc, color in inp],
            'output': output}


def _combine(items, placements, shape):
    group = {'shape': shape, 'input': [], 'output': []}
    for item, (r, c) in zip(items, placements):
        for role in ('input', 'output'):
            group[role] += [(r + dr, c + dc, color) for dr, dc, color in item[role]]
    return group


def _reference_group(rng, items):
    # Up to four <=6x6 references fit a <=13x13 group. No boundary is drawn.
    ncols = 1 if len(items) == 1 else 2
    nrows = (len(items) + ncols - 1) // ncols
    heights = [max(items[i]['shape'][0] for i in range(row * ncols,
                    min((row + 1) * ncols, len(items)))) for row in range(nrows)]
    widths = [max(items[i]['shape'][1] for i in range(col, len(items), ncols))
              for col in range(ncols)]
    starts_r = [sum(heights[:r]) + r for r in range(nrows)]
    starts_c = [sum(widths[:c]) + c for c in range(ncols)]
    placements = []
    for i, item in enumerate(items):
        row, col = divmod(i, ncols)
        placements.append((starts_r[row] + int(rng.integers(heights[row] - item['shape'][0] + 1)),
                           starts_c[col] + int(rng.integers(widths[col] - item['shape'][1] + 1))))
    shape = (sum(heights) + nrows - 1, sum(widths) + ncols - 1)
    return _combine(items, placements, shape)


def _shelf_layout(rng, groups):
    # Exhaust at most 4! * 2^3 small arrangements; always includes a 2x2 fit.
    candidates = []
    for order in permutations(range(len(groups))):
        for cuts in range(1 << (len(groups) - 1)):
            shelves, current = [], []
            for j, index in enumerate(order):
                current.append(index)
                if j == len(groups) - 1 or cuts & (1 << j):
                    shelves.append(current)
                    current = []
            widths = [sum(groups[i]['shape'][1] for i in row) + len(row) - 1 for row in shelves]
            heights = [max(groups[i]['shape'][0] for i in row) for row in shelves]
            h, w = sum(heights) + len(shelves) - 1, max(widths)
            if h <= 30 and w <= 30:
                candidates.append((h, w, shelves, heights, widths))
    assert candidates, 'bounded grouping must fit in 30x30'
    candidates.sort(key=lambda item: item[0] * item[1])
    # Some compact and some generous layouts; neither split dictates a shape.
    pool = candidates[:max(1, len(candidates) // 3)] if rng.random() < .55 else candidates
    h, w, shelves, heights, widths = _pick(rng, pool)
    shape = (min(30, h + int(rng.integers(0, 5))),
             min(30, w + int(rng.integers(0, 5))))
    if rng.random() < .16:
        shape = (30, 30)
    # Keep the inferred background strictly dominant in both rendered grids.
    required = 2 * max(sum(len(g[role]) for g in groups)
                       for role in ('input', 'output')) + 1
    while shape[0] * shape[1] < required:
        axis = 0 if shape[0] <= shape[1] else 1
        if shape[axis] == 30:
            axis = 1 - axis
        assert shape[axis] < 30
        shape = tuple(n + (i == axis) for i, n in enumerate(shape))
    y = int(rng.integers(shape[0] - h + 1))
    places = [None] * len(groups)
    for row, height, width in zip(shelves, heights, widths):
        x = int(rng.integers(shape[1] - width + 1))
        for index in row:
            item_h, item_w = groups[index]['shape']
            places[index] = (y + int(rng.integers(height - item_h + 1)), x)
            x += item_w + 1
        y += height + 1
    return shape, places


def _scatter(rng, items, shape):
    """Try free placements; at most six trials, no retained failed grids."""
    order = sorted(range(len(items)), key=lambda i: -np.prod(items[i]['shape']))
    for _ in range(6):
        blocked = np.zeros(shape, dtype=bool)
        places = [None] * len(items)
        for i in order:
            h, w = items[i]['shape']
            candidates = [(r, c) for r in range(shape[0] - h + 1)
                           for c in range(shape[1] - w + 1)
                           if not blocked[r:r + h, c:c + w].any()]
            if not candidates:
                break
            r, c = _pick(rng, candidates)
            places[i] = (r, c)
            blocked[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True
        else:
            return places
    return None


def _make_pair(rng, palette, nhubs, witness=0):
    bg = int(palette[0])
    route_keys = [int(c) for c in rng.permutation(palette[1:8])]
    inks = [int(c) for c in palette[8:10]]
    ntypes = 2 if witness == 1 else 3 if witness == 2 else int(rng.integers(1, 4))
    unused = witness == 2 or (witness == 0 and ntypes >= 2 and rng.random() < .3)
    dictionary_keys = route_keys[:ntypes]
    active = dictionary_keys[:-1] if unused else dictionary_keys
    absent = [k for k in route_keys if k not in dictionary_keys]
    templates = {}
    for i, key in enumerate(dictionary_keys):
        forced_width = (4, 6)[i] if witness == 1 else None
        if witness == 2 and i == 0:
            forced_width = 6  # Enough body cells for two visible ink colors.
        template = _make_template(rng, key, inks, forced_width,
                                  force_asymmetric=witness == 1 and i == 0,
                                  force_multicolor=witness == 2 and i == 0)
        if witness == 1:
            # Shared body ink makes template selection depend on the key.
            template = tuple((r, c, color if r == 0 else inks[0]) for r, c, color in template)
        templates[key] = template
    if witness == 1:
        native = int(rng.integers(4))
        # Same source-orientation parity exposes all four relative turns when
        # these two keys print on the four different receiving sides.
        sources = [_source_records(rng, templates[k], (native + 2 * int(rng.integers(2))) % 4)
                   for k in dictionary_keys]
    else:
        sources = [_source_records(rng, templates[k]) for k in dictionary_keys]
    if witness != 1 and len(sources) < 4 and rng.random() < .45:
        sources.append(_source_records(rng, templates[_pick(rng, dictionary_keys)]))
    rng.shuffle(sources)
    hub_labels = []
    for _ in range(nhubs):
        count = int(rng.integers(len(active) + 1))
        supported = [int(k) for k in rng.choice(active, count, replace=False)]
        labels = supported + [int(k) for k in rng.choice(absent, 4 - count, replace=False)]
        rng.shuffle(labels)
        hub_labels.append(labels)
    # Every active key is requested somewhere, with no repeated key per hub.
    for key in active:
        if not any(key in labels for labels in hub_labels):
            labels = _pick(rng, hub_labels)
            options = [i for i, k in enumerate(labels) if k in absent]
            labels[_pick(rng, options)] = key
    if witness == 1:
        a, b = active
        hub_labels[0] = [a, b, absent[0], absent[1]]
        hub_labels[1] = [absent[1], absent[0], a, b]
    if nhubs == 3:
        # Test generalization is three actual uses of one reusable entry,
        # rather than a larger scene padded with unused hubs.
        common_key = active[0]
        for labels in hub_labels:
            if common_key not in labels:
                slot = _pick(rng, [i for i, key in enumerate(labels) if key in absent])
                labels[slot] = common_key
    hubs = [_hub_item(labels, templates) for labels in hub_labels]
    ref_group = _reference_group(rng, sources)
    groups = hubs + [ref_group]
    shape, grouped_places = _shelf_layout(rng, groups)
    items = hubs + sources
    free_places = _scatter(rng, items, shape)
    scene = (_combine(items, free_places, shape) if free_places is not None
             else _combine(groups, grouped_places, shape))
    grids = {}
    for role in ('input', 'output'):
        grid = np.full(shape, bg, dtype=int)
        for r, c, color in scene[role]:
            assert grid[r, c] == bg, 'scene records overlap'
            grid[r, c] = color
        grids[role] = grid
    turns, mirror = int(rng.integers(4)), rng.random() < .5
    for role in grids:
        grid = np.rot90(grids[role], turns)
        grids[role] = (np.fliplr(grid) if mirror else grid).copy()
    predicted, _ = _inspect(grids['input'])
    assert np.array_equal(predicted, grids['output']), 'input-derived reconstruction disagrees'
    return grids


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one reproducible puzzle; count bounds are strict, not optional."""
    ntrain, ntest = _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_palette = rng.permutation(10)
    stable_palette = rng.random() < .35
    train_configs = [(2, 1), (int(rng.integers(1, 3)), 2)]
    train_configs += [(int(rng.integers(1, 3)), 0) for _ in range(ntrain - 2)]
    rng.shuffle(train_configs)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, configs in (('train', train_configs), ('test', [(3, 0)] * ntest)):
        for nhubs, witness in configs:
            for _ in range(8):
                palette = base_palette if stable_palette else rng.permutation(10)
                pair = _make_pair(rng, palette, nhubs, witness)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError('bounded distinct-input construction failed')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, shape):
    templates = info['templates']
    refs = Counter(p['key'] for p in info['references'])
    requests = info['requests']
    native = {p['key']: p['native'] for p in info['references']}
    geometry = [tuple((r, c) for r, c, _ in sig) for sig in templates.values()]
    return {
        'shape': tuple(shape), 'hubs': len(info['hubs']), 'types': len(templates),
        'references': len(info['references']), 'prints': sum(requests.values()),
        'unused': len(set(templates) - set(requests)),
        'unsupported': 4 * len(info['hubs']) - sum(requests.values()),
        'reuse': any(requests[k] > refs[k] for k in requests),
        'sides': set(info['orientations']),
        'turns': {(side - native[k]) % 4 for k, side, _, _, _ in info['bodies']},
        'native': {p['native'] for p in info['references']},
        'geometry': geometry,
        'widths': {max(c for _, c, _ in sig) - min(c for _, c, _ in sig) + 1
                   for sig in templates.values()},
        'depths': {1 - min(r for r, _, _ in sig) for sig in templates.values()},
        'multicolor': any(len({col for r, c, col in sig if r < 0}) > 1 for sig in templates.values()),
        'diagonal': any(len(_four_components(sig)) > 1 for sig in templates.values()),
        'asymmetric': any(set((r, c) for r, c, _ in sig) !=
                          set((r, 1 - c) for r, c, _ in sig) for sig in templates.values()),
    }


def _four_components(signature):
    cells = {(r, c) for r, c, _ in signature}
    result = []
    while cells:
        stack = [cells.pop()]
        result.append(stack[0])
        while stack:
            r, c = stack.pop()
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                point = r + dr, c + dc
                if point in cells:
                    cells.remove(point)
                    stack.append(point)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            expected, info = _inspect(inp)
            assert np.array_equal(out, expected), 'incorrect print, rotation, filtering, or anchor overlay'
            assert not np.array_equal(inp, out)
            # Per-color accounting is about printed copies, never source supply.
            counts = Counter(color for _, _, keys in info['hubs'] for color in keys for _ in range(2))
            for k, copies in info['requests'].items():
                for r, c, color in info['templates'][k]:
                    if r < 0:
                        counts[color] += copies
            actual = Counter(map(int, out[out != info['background']]))
            assert counts == actual, 'per-color print mass is not conserved'
            if split == 'train':
                training.append(_features(info, inp.shape))
    assert set().union(*(f['sides'] for f in training)) == {0, 1, 2, 3}
    assert set().union(*(f['turns'] for f in training)) == {0, 1, 2, 3}
    assert any(f['reuse'] for f in training), 'need a one-to-many copying witness'
    assert any(f['unused'] > 0 for f in training), 'need an unused-reference witness'
    assert any(f['unsupported'] > 0 for f in training), 'need unsupported sockets'
    assert any(len(set(f['geometry'])) >= 2 for f in training), 'need key-specific shape selection'
    assert any(f['asymmetric'] for f in training), 'need rotation versus reflection evidence'
    assert any(f['multicolor'] for f in training), 'need exact multicolor copying evidence'
    assert any(f['turns'] - {0} for f in training), 'ignoring docking rotation must change the answer'


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def _example_regression():
    """A small hand-calculated printing example, including omission checks."""
    inp = np.zeros((18, 28), dtype=int)
    inp[1, 1:5], inp[2, 2:4], inp[3, 2:4] = 8, 8, 1
    inp[1, 14:20], inp[2, 16:18], inp[3, 16:18] = 9, 9, 3
    gold = np.zeros_like(inp)
    for col, labels in ((3, (1, 3, 5, 6)), (19, (3, 1, 6, 5))):
        for r, c, color in _hub_records(labels):
            inp[8 + r, col + c] = gold[8 + r, col + c] = color
    # Key 1 has four ink cells behind its two-cell stem. Key 3 has six.
    # Top prints: horizontal bodies above the receiving top bars.
    gold[6, 3:7], gold[7, 4:6] = 8, 8
    gold[6, 18:24], gold[7, 20:22] = 9, 9
    # Right prints: the same patterns turned clockwise, with no extra key.
    gold[7:13, 8], gold[9:11, 7] = 9, 9
    gold[8:12, 24], gold[9:11, 23] = 8, 8
    assert np.array_equal(solve(inp), gold), 'hand-calculated anchor/rotation regression'
    assert np.count_nonzero(gold == 8) == 2 * np.count_nonzero(inp == 8)
    assert np.count_nonzero(gold == 9) == 2 * np.count_nonzero(inp == 9)
    missing_reference = inp.copy()
    missing_reference[1:4, 1:5] = 0
    no_dictionary_entry = solve(missing_reference)
    assert np.count_nonzero(no_dictionary_entry == 8) == 0
    assert np.count_nonzero(no_dictionary_entry == 9) == 16
    changed_socket = inp.copy()
    changed_socket[8, 4:6] = 7
    changed_answer = solve(changed_socket)
    assert np.count_nonzero(changed_answer == 8) == 6
    assert not np.array_equal(changed_answer, gold)


def self_test():
    """Bounded streaming checks on 42 seeds; no generated grids are retained."""
    _example_regression()
    shapes, geometries, signatures = set(), set(), set()
    widths, depths, directions, turns, native = set(), set(), set(), set(), set()
    hub_counts, type_counts, ref_counts, print_counts = set(), set(), set(), set()
    totals = Counter()
    translation_checked = False
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        assert _digest(puzzle) == _digest(generate_puzzle(seed)), 'seed reproducibility'
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                expected, info = _inspect(pair['input'])
                assert np.array_equal(pair['output'], expected)
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input']), 'solver mutated input'
                if not translation_checked and max(before.shape) <= 28:
                    bg = info['background']
                    padded = np.pad(before, 1, constant_values=bg)
                    target = np.pad(pair['output'], 1, constant_values=bg)
                    assert np.array_equal(solve(padded.tolist()), target)
                    translation_checked = True
                f = _features(info, before.shape)
                shapes.add(f['shape'])
                geometries.update(f['geometry'])
                widths.update(f['widths'])
                depths.update(f['depths'])
                directions.update(f['sides'])
                turns.update(f['turns'])
                native.update(f['native'])
                hub_counts.add(f['hubs'])
                type_counts.add(f['types'])
                ref_counts.add(f['references'])
                print_counts.add(f['prints'])
                totals['pairs'] += 1
                for name in ('reuse', 'multicolor', 'diagonal', 'asymmetric'):
                    totals[name] += bool(f[name])
                totals['unused_pairs'] += f['unused'] > 0
                # Every tested scene is equivariant under a rigid canvas motion.
                transformed = np.fliplr(np.rot90(before))
                predicted = solve(transformed)
                assert np.array_equal(predicted, np.fliplr(np.rot90(pair['output'])))
        assert max(len(_inspect(p['input'])[1]['hubs']) for p in puzzle['train']) <= 2
        for pair in puzzle['test']:
            info = _inspect(pair['input'])[1]
            assert len(info['hubs']) == 3
            assert any(n == 3 for n in info['requests'].values())
            assert all(any(key in info['templates'] for key in keys)
                       for _, _, keys in info['hubs'])
    assert len(signatures) == 42 and len(shapes) >= 30 and len(geometries) >= 70
    assert translation_checked, 'padding/translation check must execute'
    assert widths == {2, 4, 6} and depths == {3, 4, 5}
    assert directions == turns == native == {0, 1, 2, 3}
    assert hub_counts == type_counts == {1, 2, 3}
    assert {2, 3, 4} <= ref_counts and len(print_counts) >= 6
    assert all(totals[name] >= 10 for name in ('multicolor', 'diagonal', 'asymmetric', 'unused_pairs'))
    count_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(601 + 10 * a + b, a, b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
            count_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for seed in (-1, True, 1.25, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal seed accepted')
    # Corrupt an answer: validation must fail before accepting any metadata.
    puzzle = generate_puzzle(0, 2, 1)
    out = puzzle['test'][0]['output']
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt answer accepted')
    # Contradictory complete references with one key must be rejected.
    bg, key, ink = 0, 1, 2
    contradictory = np.zeros((20, 24), dtype=int)
    for r, c, color in _hub_records((1, 3, 4, 5)):
        contradictory[r + 11, c + 11] = color
    template_a = ((0, 0, key), (0, 1, key), (-1, 0, ink), (-1, 1, ink),
                  (-2, -1, ink), (-2, 0, ink), (-2, 1, ink), (-2, 2, ink))
    template_b = tuple((r - (1 if r < 0 else 0), c, color) for r, c, color in template_a)
    # Use a connected, longer stem for the contradictory second template.
    template_b += ((-1, 0, ink), (-1, 1, ink))
    for sig, r0, c0 in ((template_a, 4, 4), (template_b, 4, 15)):
        for r, c, color in sig:
            contradictory[r + r0, c + c0] = color
    try:
        solve(contradictory)
    except AssertionError as error:
        assert 'conflicting' in str(error)
    else:
        raise AssertionError('contradictory dictionary accepted')
    return {'seeds': len(HOST_SEEDS), 'pairs': totals['pairs'],
            'legal_count_combinations': count_calls, 'distinct_puzzles': len(signatures),
            'canvas_shapes': len(shapes), 'canonical_geometries': len(geometries),
            'widths': sorted(widths), 'depths': sorted(depths),
            'hub_counts': sorted(hub_counts), 'template_counts': sorted(type_counts),
            'reference_counts': sorted(ref_counts), 'print_counts': sorted(print_counts),
            'counts': dict(totals), 'independent_solver': False, 'blind_prediction': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
