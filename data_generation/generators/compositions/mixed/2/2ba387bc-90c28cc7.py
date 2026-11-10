"""Registered mosaics: global run collapse, then hollow/solid rank queues.

Only NumPy and the standard library are used. No files, examples, or random
state are consulted by solve(). self_test() keeps only aggregate diagnostics.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


HOLLOW, SOLID = "hollow", "solid"
_COMPACT_LIMIT = 25


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(train, test):
    if not _integer(train) or not 2 <= train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(test) or not 1 <= test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be in 0..9")
    return a.astype(np.int8, copy=True)


def _tile(size, color, substrate, kind):
    tile = np.full((size, size), color, dtype=np.int8)
    if kind == HOLLOW:
        tile[1:-1, 1:-1] = substrate
    return tile


def _collapse(crop):
    """Global equality of complete vectors, never per-object run collapse."""
    rs = np.r_[0, np.flatnonzero(np.any(crop[1:] != crop[:-1], axis=1)) + 1]
    cs = np.r_[0, np.flatnonzero(np.any(crop[:, 1:] != crop[:, :-1], axis=0)) + 1]
    rows = np.diff(np.r_[rs, crop.shape[0]])
    cols = np.diff(np.r_[cs, crop.shape[1]])
    compact = crop[np.ix_(rs, cs)].copy()
    return compact, rows, cols


def _parse_tiles(scene, substrate):
    """Segment spatial instances, including separate instances of one color."""
    height, width = scene.shape
    _require(height >= 5 and width >= 5, "tile scene too small")
    _require(all(bool(np.all(edge == substrate)) for edge in
                 (scene[0], scene[-1], scene[:, 0], scene[:, -1])), "missing substrate margin")
    remaining = set(map(tuple, np.argwhere(scene != substrate)))
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    component.append(p)
        top = int(min(p[0] for p in component))
        left = int(min(p[1] for p in component))
        bottom = int(max(p[0] for p in component)) + 1
        right = int(max(p[1] for p in component)) + 1
        size = bottom - top
        _require(size >= 3 and size == right - left, "object is not a square tile")
        tile = scene[top:bottom, left:right].copy()
        color = int(tile[0, 0])
        _require(color != substrate, "invalid tile corner")
        if np.all(tile == color):
            kind = SOLID
        else:
            kind = HOLLOW
            _require(np.array_equal(tile, _tile(size, color, substrate, HOLLOW)),
                     "tile is neither solid nor a one-cell hollow frame")
        # Bounding boxes have a complete substrate gap, including diagonals.
        neighbor = scene[max(0, top - 1):min(height, bottom + 1),
                         max(0, left - 1):min(width, right + 1)].copy()
        neighbor[top - max(0, top - 1):bottom - max(0, top - 1),
                 left - max(0, left - 1):right - max(0, left - 1)] = substrate
        _require(bool(np.all(neighbor == substrate)), "tiles are not separated")
        objects.append(dict(top=top, left=left, size=size, color=color, kind=kind, tile=tile))
    _require(bool(objects), "no tiles")
    sizes = {o['size'] for o in objects}
    _require(len(sizes) == 1, "tile sizes disagree")
    size = sizes.pop()
    for kind in (HOLLOW, SOLID):
        queue = [o for o in objects if o['kind'] == kind]
        _require(bool(queue), "both tile classes must occur")
        _require(len({o['top'] for o in queue}) == len(queue), "same-class top-row tie")
        _require(len(queue) * size <= 30, "queue exceeds output height")
    _require(2 * size <= 30, "queue exceeds output width")
    objects.sort(key=lambda o: (o['top'], o['left']))
    return objects


def _decode(input_grid):
    a = _grid(input_grid)
    outer = int(a[0, 0])
    _require(all(bool(np.all(edge == outer)) for edge in
                 (a[0], a[-1], a[:, 0], a[:, -1])), "missing uniform outer margin")
    positions = np.argwhere(a != outer)
    _require(bool(len(positions)), "empty input")
    top, left = map(int, positions.min(axis=0))
    bottom, right = map(int, positions.max(axis=0) + 1)
    crop = a[top:bottom, left:right]
    _require(bool(np.all(crop != outer)), "mosaic must be a filled rectangle")
    _require(int(np.count_nonzero(a != outer)) == crop.size, "nonrectangular mosaic")
    compact, rows, cols = _collapse(crop)
    _require(compact.shape[0] >= 7 and compact.shape[1] >= 7, "compact mosaic too small")
    _require(np.array_equal(np.repeat(np.repeat(compact, rows, axis=0), cols, axis=1), crop),
             "collapse did not reconstruct losslessly")
    rail = (int(compact[0, 0]), int(compact[0, 1]))
    _require(rail[0] != rail[1], "registration rim needs two colors")
    rr, cc = np.indices(compact.shape)
    edge = (rr == 0) | (cc == 0) | (rr == compact.shape[0] - 1) | (cc == compact.shape[1] - 1)
    expected = np.asarray(rail, dtype=np.int8)[(rr + cc) % 2]
    _require(np.array_equal(compact[edge], expected[edge]), "invalid alternating registration rim")
    scene = compact[1:-1, 1:-1].copy()
    _require(not bool(np.any(np.isin(scene, rail))), "rim colors occur inside scene")
    substrate = int(scene[0, 0])
    _require(substrate not in (outer, *rail), "palette roles overlap")
    objects = _parse_tiles(scene, substrate)
    # After removing the *expanded* rim, this is the scene seen by a rule
    # that omits B. It generally has distorted, unequal, nonsquare tiles.
    raw_scene = crop[int(rows[0]):crop.shape[0] - int(rows[-1]),
                     int(cols[0]):crop.shape[1] - int(cols[-1])].copy()
    return dict(compact=compact, scene=scene, objects=objects, substrate=substrate,
                outer=outer, rail=rail, rows=rows, cols=cols, raw_scene=raw_scene,
                margins=(top, a.shape[0] - bottom, left, a.shape[1] - right))


def _pack(objects, substrate):
    queues = [[o for o in objects if o['kind'] == kind] for kind in (HOLLOW, SOLID)]
    size = objects[0]['size']
    out = np.full((size * max(map(len, queues)), 2 * size), substrate, dtype=np.int8)
    for col, queue in enumerate(queues):
        for rank, obj in enumerate(queue):
            out[rank * size:(rank + 1) * size, col * size:(col + 1) * size] = obj['tile']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all conventions from this input and apply the fixed family rule."""
    analysis = _decode(input_grid)
    return _pack(analysis['objects'], analysis['substrate'])


def _greedy(objects, substrate):
    slots = [[None, None]]
    for obj in objects:
        col = int(obj['kind'] == SOLID)
        if slots[-1][col] is not None:
            slots.append([None, None])
        slots[-1][col] = obj
    size = objects[0]['size']
    out = np.full((len(slots) * size, 2 * size), substrate, dtype=np.int8)
    for rank, row in enumerate(slots):
        for col, obj in enumerate(row):
            if obj is not None:
                out[rank * size:(rank + 1) * size, col * size:(col + 1) * size] = obj['tile']
    return out


def _needs_registration(scene):
    return (bool(np.any(np.all(scene[1:] == scene[:-1], axis=1))) or
            bool(np.any(np.all(scene[:, 1:] == scene[:, :-1], axis=0))))


def _omitting_b_changes(analysis, answer):
    try:
        alternative = _pack(_parse_tiles(analysis['raw_scene'], analysis['substrate']),
                            analysis['substrate'])
    except ValueError:
        return True
    return not np.array_equal(alternative, answer)


def _training_witness(scene, objects, substrate):
    answer = _pack(objects, substrate)
    if np.array_equal(answer, _greedy(objects, substrate)) or not _needs_registration(scene):
        return False
    # Refute both horizontal scan directions, using the actual visible colors.
    return all(not np.array_equal(answer, _pack(sorted(objects,
               key=lambda o: (o['left'], o['top']), reverse=reverse), substrate))
               for reverse in (False, True))


def _bands(size, count):
    extent = _COMPACT_LIMIT - 4  # registration rim plus substrate margin
    return [b for b in range(1, count + 1)
            if count + b * size - 1 <= extent and
            ((count + b - 1) // b) * (size + 1) - 1 <= extent]


def _positions(rng, size, count):
    """Feasible shelves followed by free relocation, with globally unique tops."""
    band_count = int(rng.choice(_bands(size, count)))
    amounts = [count // band_count] * band_count
    for i in rng.permutation(band_count)[:count % band_count]:
        amounts[int(i)] += 1
    min_h = count + band_count * size - 1
    min_w = max(amounts) * (size + 1) - 1
    extent = _COMPACT_LIMIT - 4
    tight = bool(rng.integers(2))
    h = int(rng.integers(min_h, min(extent, min_h + 3) + 1 if tight else extent + 1))
    w = int(rng.integers(min_w, min(extent, min_w + 3) + 1 if tight else extent + 1))
    vertical = rng.multinomial(h - min_h, np.full(band_count + 1, 1 / (band_count + 1)))
    positions = []
    top = int(vertical[0])
    for band, amount in enumerate(amounts):
        horizontal = rng.multinomial(w - (amount * (size + 1) - 1),
                                    np.full(amount + 1, 1 / (amount + 1)))
        left = int(horizontal[0])
        for j, offset in enumerate(rng.permutation(amount)):
            positions.append((top + int(offset), left))
            left += size + 1 + int(horizontal[j + 1])
        top += size + amount + int(vertical[band + 1])
    rr, cc = np.indices((h - size + 1, w - size + 1))
    for _ in range(2):
        for idx in rng.permutation(count):
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(positions):
                if j != idx:
                    legal &= ((np.abs(rr - r) > size) | (np.abs(cc - c) > size)) & (rr != r)
            sites = np.argwhere(legal)
            r, c = sites[int(rng.integers(len(sites)))]
            positions[int(idx)] = (int(r), int(c))
    # A visible substrate margin isolates the scene from the registration rim.
    return (h + 2, w + 2), sorted((r + 1, c + 1) for r, c in positions)


def _kinds(rng, hollow, solid, witness):
    base = [HOLLOW] * hollow + [SOLID] * solid
    if witness:
        first = HOLLOW if rng.integers(2) else SOLID
        base.remove(first)
        base.remove(first)
        return [first, first] + [str(k) for k in rng.permutation(base)]
    mode = int(rng.integers(4))
    if mode == 0:
        return base if rng.integers(2) else base[::-1]
    if mode == 1:
        out = []
        counts = {HOLLOW: hollow, SOLID: solid}
        order = (HOLLOW, SOLID) if rng.integers(2) else (SOLID, HOLLOW)
        while any(counts.values()):
            for kind in order:
                if counts[kind]:
                    out.append(kind)
                    counts[kind] -= 1
        return out
    return [str(k) for k in rng.permutation(base)]


def _make_scene(rng, size, counts, palette, anchor, forced_kind, witness, fallback=False):
    outer, substrate, a, b = palette[:4]
    if fallback:
        shape = (3 * size + 6, 2 * size + 3)
        positions = [(1, size + 2), (2, 1), (size + 3, 1),
                     (size + 4, size + 2), (2 * size + 5, 1)]
        kinds = [HOLLOW, HOLLOW, SOLID, SOLID, SOLID]
        if counts[0] > counts[1]:
            kinds = [SOLID if k == HOLLOW else HOLLOW for k in kinds]
    else:
        shape, positions = _positions(rng, size, sum(counts))
        kinds = _kinds(rng, *counts, witness)
    tile_palette = list(palette[4:])
    if witness or rng.random() < .45:
        colors = [int(c) for c in rng.permutation(tile_palette)]
        colors = (colors * ((len(kinds) + 5) // 6))[:len(kinds)]
    else:
        pool = rng.choice(tile_palette, size=int(rng.integers(1, 5)), replace=False)
        colors = [int(c) for c in rng.choice(pool, size=len(kinds))]
    if forced_kind is not None:
        idx = kinds.index(forced_kind)
        if anchor in colors:
            other = colors.index(anchor)
            colors[idx], colors[other] = colors[other], colors[idx]
        else:
            colors[idx] = anchor
    if witness:
        for kind in (HOLLOW, SOLID):
            indices = [i for i, k in enumerate(kinds) if k == kind]
            if colors[indices[0]] == colors[indices[-1]]:
                colors[indices[-1]] = next(c for c in tile_palette if c != colors[indices[0]])
    scene = np.full(shape, substrate, dtype=np.int8)
    objects = []
    for (r, c), color, kind in zip(positions, colors, kinds):
        tile = _tile(size, color, substrate, kind)
        scene[r:r + size, c:c + size] = tile
        objects.append(dict(top=r, left=c, size=size, color=color, kind=kind, tile=tile))
    objects.sort(key=lambda o: (o['top'], o['left']))
    return scene, objects


def _axis(rng, length, forced, strong=False, uneven=False):
    budget = 28 - length
    bands = np.ones(length, dtype=np.int64)
    if not strong and not uneven and length <= 14 and rng.random() < .12:
        return bands * 2
    # At least one tile's interior is actually expanded on each axis.
    first_extra = int(rng.integers(3, min(5, budget) + 1)) if strong else 1
    bands[forced] += first_extra
    budget -= first_extra
    cap = 6 if strong else 3
    extra = int(rng.integers(0 if strong else 1, min(8, budget) + 1))
    for _ in range(extra):
        choices = np.flatnonzero(bands < cap)
        bands[int(rng.choice(choices))] += 1
    return bands


def _enlarge(rng, scene, objects, palette, strong=False, uneven=False):
    outer, substrate, a, b = palette[:4]
    h, w = scene.shape
    rr, cc = np.indices((h + 2, w + 2))
    compact = np.asarray((a, b), dtype=np.int8)[(rr + cc) % 2]
    compact[1:-1, 1:-1] = scene
    first = objects[int(rng.integers(len(objects)))]
    second = objects[int(rng.integers(len(objects)))]
    rows = _axis(rng, h + 2, 1 + first['top'] + first['size'] // 2, strong, uneven)
    cols = _axis(rng, w + 2, 1 + second['left'] + second['size'] // 2, strong, uneven)
    enlarged = np.repeat(np.repeat(compact, rows, axis=0), cols, axis=1)
    extras = [int(rng.integers(2, min(6, 30 - n) + 1)) for n in enlarged.shape]
    top, left = [int(rng.integers(1, total)) for total in extras]
    canvas = np.full((enlarged.shape[0] + extras[0], enlarged.shape[1] + extras[1]),
                     outer, dtype=np.int8)
    canvas[top:top + enlarged.shape[0], left:left + enlarged.shape[1]] = enlarged
    return canvas, compact


def _feasible_counts(size):
    limit = min(6, 30 // size)
    return [(h, s) for h in range(1, limit + 1) for s in range(1, limit + 1)
            if _bands(size, h + s)]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    anchor = int(rng.integers(10))
    first_sign = 1 if rng.integers(2) else -1
    first_sizes = [3, 4] if rng.integers(2) else [4, 3]
    puzzle = {'train': [], 'test': []}
    seen = set()
    for index in range(num_train + num_test):
        training = index < num_train
        first_test = index == num_train
        size = first_sizes[index] if index < 2 else int(rng.choice((3, 4, 5, 6), p=(.37, .31, .22, .10)))
        if index < 2:
            sign = first_sign if index == 0 else -first_sign
            counts = (2, 3) if sign > 0 else (3, 2)
        else:
            choices = _feasible_counts(size)
            if training:
                choices = [p for p in choices if abs(p[0] - p[1]) <= 1]
            elif first_test:
                choices = [p for p in choices if 2 <= abs(p[0] - p[1]) <= 3]
            counts = choices[int(rng.integers(len(choices)))]
        # All role colors are locally visible. The one reserved tile color
        # switches hollow/solid roles in the first two demonstrations.
        others = [int(c) for c in rng.permutation([c for c in range(10) if c != anchor])]
        palette = others[:4] + [anchor] + others[4:]
        forced = HOLLOW if index == 0 else (SOLID if index == 1 else None)
        for attempt in range(65):
            scene, objects = _make_scene(rng, size, counts, palette, anchor, forced,
                                        witness=index == 0, fallback=attempt == 64 and index == 0)
            if index == 0 and not _training_witness(scene, objects, palette[1]):
                continue
            inp, compact = _enlarge(rng, scene, objects, palette, strong=not training,
                                    uneven=index < 2)
            out = _pack(objects, palette[1])
            key = _key(inp)
            if key in seen:
                continue
            decoded = _decode(inp)
            _require(np.array_equal(decoded['compact'], compact), "generator lost a logical boundary")
            if not _omitting_b_changes(decoded, out):
                continue
            seen.add(key)
            puzzle['train' if training else 'test'].append({'input': inp, 'output': out})
            break
        else:
            raise RuntimeError("bounded generation exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, tests = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _decode(inp)
            expected = _pack(a['objects'], a['substrate'])
            _require(np.array_equal(out, expected), "incorrect rank queue")
            _require(_omitting_b_changes(a, out), "B has no effect on tile answer")
            _require(not np.array_equal(a['scene'], out), "A has no effect on answer")
            _require(max(a['rows']) > 1 and max(a['cols']) > 1,
                     "both axes must actually expand")
            # Count and exact copy checks include empty trailing slots and
            # preserve every foreground tile pixel, even when colors repeat.
            _require(np.count_nonzero(out != a['substrate']) ==
                     sum(np.count_nonzero(o['tile'] != a['substrate']) for o in a['objects']),
                     "tile pixel conservation failed")
            (training if split == 'train' else tests).append(a)
    _require(any(_training_witness(a['scene'], a['objects'], a['substrate']) for a in training),
             "missing registration/independent-queue ordering witness")
    _require(any(len(set(map(int, a['rows']))) > 1 and len(set(map(int, a['cols']))) > 1
                 for a in training), "missing independent nonuniform-axis demonstration")
    signs, roles = [], {}
    for a in training:
        h = sum(o['kind'] == HOLLOW for o in a['objects'])
        signs.append(len(a['objects']) - 2 * h)
        for o in a['objects']:
            roles.setdefault(o['color'], set()).add(o['kind'])
        _require(max(a['rows']) <= 3 and max(a['cols']) <= 3, "training expansion outside profile")
    _require(min(signs) < 0 < max(signs), "both padding directions must be taught")
    _require(any(len(kinds) == 2 for kinds in roles.values()), "missing color-role switch")
    for kind in (HOLLOW, SOLID):
        _require(any(len({o['color'] for o in a['objects'] if o['kind'] == kind}) >= 2
                     for a in training), "ordering must be visible in each class")
    _require(any(abs(sum(1 if o['kind'] == SOLID else -1 for o in a['objects'])) >= 2
                 for a in tests), "missing held-out longer tail")
    _require(all(max(a['rows']) >= 4 and max(a['cols']) >= 4 for a in tests),
             "missing held-out stronger expansion")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, all legal counts, and semantic negative controls.

    No grids or traces are written. Each puzzle has at most 14 pairs; the
    retained negative-control puzzle and assignment temporaries keep raw grid
    storage below 80 KB, in addition to small analysis arrays and Python sets.
    Returned measurements describe coverage, not independent semantic approval.
    """
    signatures, layouts, shapes, compact_shapes = set(), set(), set(), set()
    sizes, backgrounds, counts, structures, profiles = set(), set(), set(), set(), set()
    metrics = dict(pairs=0, repeated_tile_colors=0, black_tiles=0,
                   registration_needed=0, greedy_counterexamples=0, nonsquare_raw_tiles=0)
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed == 0:
            first = puzzle
        signatures.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _decode(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                objects = a['objects']
                sizes.add(objects[0]['size'])
                backgrounds.add(a['substrate'])
                shapes.add(pair['input'].shape)
                compact_shapes.add(a['compact'].shape)
                counts.add((sum(o['kind'] == HOLLOW for o in objects),
                            sum(o['kind'] == SOLID for o in objects)))
                layouts.add((a['scene'].shape, tuple((o['top'], o['left'], o['size'], o['kind']) for o in objects)))
                # Recoloring cannot inflate this output-structure measure.
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in pair['output'].flat)
                structures.add((pair['output'].shape, normalized))
                profiles.add((tuple(map(int, a['rows'])), tuple(map(int, a['cols']))))
                metrics['pairs'] += 1
                metrics['repeated_tile_colors'] += len({o['color'] for o in objects}) < len(objects)
                metrics['black_tiles'] += any(o['color'] == 0 for o in objects)
                metrics['registration_needed'] += _needs_registration(a['scene'])
                metrics['greedy_counterexamples'] += not np.array_equal(_greedy(objects, a['substrate']), pair['output'])
                for o in objects:
                    r, c, s = o['top'] + 1, o['left'] + 1, o['size']
                    if sum(a['rows'][r:r + s]) != sum(a['cols'][c:c + s]):
                        metrics['nonsquare_raw_tiles'] += 1
                        break
    assert len(signatures) == 42
    assert sizes == {3, 4, 5, 6} and len(backgrounds) == 10
    assert len(layouts) >= 150 and len(shapes) >= 35 and len(compact_shapes) >= 30
    assert len(structures) >= 40 and len(counts) >= 15 and len(profiles) >= 180
    assert all(value >= 10 for key, value in metrics.items() if key != 'pairs')
    assert _digest(first) == _digest(generate_puzzle(0))
    # Exercise every explicit legal count request, without retaining runs.
    legal = 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * train + test, train, test)
            assert len(puzzle['train']) == train and len(puzzle['test']) == test
            legal += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    damaged = {s: [dict(input=p['input'].copy(), output=p['output'].copy()) for p in first[s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt queue accepted")
    # A valid scene that would be erased by an unregistered local collapse:
    # two solid middle rows are identical, but the rim keeps both boundaries.
    scene = np.ones((7, 12), dtype=np.int8)
    scene[1:5, 1:5] = _tile(4, 4, 1, SOLID)
    scene[2:6, 7:11] = _tile(4, 5, 1, HOLLOW)
    objects = _parse_tiles(scene, 1)
    rng = np.random.default_rng(555)
    canvas, compact = _enlarge(rng, scene, objects, list(range(10)), uneven=True)
    assert np.array_equal(_decode(canvas)['compact'], compact)
    assert np.array_equal(solve(canvas), _pack(objects, 1))
    assert _needs_registration(scene)
    assert np.array_equal(scene[3], scene[4])
    return dict(seeds=42, distinct_puzzles=len(signatures), legal_count_combinations=legal,
                tile_sizes=sorted(sizes), substrate_colors=sorted(backgrounds),
                input_shapes=len(shapes), compact_shapes=len(compact_shapes),
                instance_layouts=len(layouts), count_pairs=len(counts),
                normalized_output_structures=len(structures), expansion_profiles=len(profiles),
                **metrics)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
