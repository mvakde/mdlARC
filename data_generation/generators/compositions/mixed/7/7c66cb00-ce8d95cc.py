"""Whole-slab equality-run compaction followed by color-routed stamping.

Only NumPy and the standard library are used. Generation constructs a compact
staging design and expands its vectors; solve rediscovers everything from pixels.
No generation metadata, files, cached examples, or seed enter solve.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be 1..30 by 1..30 integer arrays with colors 0..9")
    return a


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _keep_vectors(a, axis):
    vectors = a if axis == 0 else a.T
    return np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)]


def _collapse(a, rows=True, columns=True):
    # Both equality tests use the ORIGINAL slab, so the independence is explicit.
    # Removing identical vectors on the other axis cannot change these tests.
    rr = _keep_vectors(a, 0) if rows else np.ones(a.shape[0], dtype=bool)
    cc = _keep_vectors(a, 1) if columns else np.ones(a.shape[1], dtype=bool)
    return a[np.ix_(rr, cc)].copy()


def _runs(a, axis):
    starts = np.flatnonzero(_keep_vectors(a, axis))
    return tuple(map(int, np.diff(np.r_[starts, a.shape[axis]])))


def _components(a, background, diagonal=False):
    unseen = {tuple(map(int, p)) for p in np.argwhere(a != background)}
    result = []
    neighbors = _EIGHT if diagonal else _FOUR
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        color = int(a[start])
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in unseen and a[q] == color:
                    unseen.remove(q)
                    stack.append(q)
        result.append((color, tuple(sorted(cells))))
    return result


def _paint(canvas, stop, background, bands, components):
    """Stamp already segmented components; reject real collisions, never stack."""
    out = canvas.copy()
    out[:stop] = background
    lookup = {b['route']: b for b in bands}
    occupied = set()
    for color, cells in components:
        if color not in lookup:
            raise ValueError("staging color has no unique matching band")
        band = lookup[color]
        shift = band['bottom'] - max(r for r, c in cells)
        targets = {(r + shift, c) for r, c in cells}
        if any(not (band['top'] <= r <= band['bottom']
                    and 0 < c < canvas.shape[1] - 1) for r, c in targets):
            raise ValueError("translated component does not fit its band")
        if targets & occupied:
            raise ValueError("independently aligned components collide")
        occupied.update(targets)
        for p in targets:
            out[p] = band['rail']
    return out


def _parse_view(a):
    """Parse the view with staging above full-width, two-railed bands."""
    h, w = a.shape
    if h < 4 or w < 5:
        raise ValueError("scene too small")
    background = int(a[0, 0])
    info = []
    for row in a:
        if (row[0] == row[-1] and row[0] != background
                and row[1] != background and row[0] != row[1]
                and np.all(row[1:-1] == row[1])):
            info.append((int(row[1]), int(row[0])))
        else:
            info.append(None)
    band_rows = [r for r, value in enumerate(info) if value is not None]
    if not band_rows:
        raise ValueError("no recognizable bands")
    stop = band_rows[0]
    if (stop < 3 or not np.all(a[stop - 1] == background)
            or not np.all(a[0] == background)
            or not np.all(a[:stop, [0, -1]] == background)):
        raise ValueError("staging must have a background border")
    bands, r = [], stop
    while r < h:
        if info[r] is None:
            if not np.all(a[r] == background):
                raise ValueError("non-band data in the destination array")
            r += 1
            continue
        route, rail = info[r]
        top = r
        while r + 1 < h and info[r + 1] == (route, rail):
            r += 1
        bands.append(dict(top=top, bottom=r, route=route, rail=rail))
        r += 1
    if (len({b['route'] for b in bands}) != len(bands)
            or any(y['top'] < x['bottom'] + 2 for x, y in zip(bands, bands[1:]))):
        raise ValueError("band routes must be unique and bands separated")
    stage = a[:stop]
    if not np.any(stage != background):
        raise ValueError("empty staging")
    compact = _collapse(stage)
    components = _components(compact, background)
    if not {color for color, cells in components} <= {b['route'] for b in bands}:
        raise ValueError("unroutable staging color")
    out = _paint(a, stop, background, bands, components)
    return dict(canonical=a, background=background, stop=stop, bands=bands,
                stage=stage, compact=compact, components=components, expected=out)


def _analyze(value):
    a = _grid(value)
    candidates = []
    for turn in range(4):
        try:
            result = _parse_view(np.rot90(a, turn))
        except ValueError:
            continue
        result['turn'] = turn
        result['output'] = np.rot90(result['expected'], -turn).copy()
        candidates.append(result)
    if len(candidates) != 1:
        raise ValueError("input must have one staging-to-band direction")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution for every scene satisfying the family specification."""
    return _analyze(input_grid)['output']


def _walk(rng, height, width):
    cells = {(int(rng.integers(height)), int(rng.integers(width)))}
    goal = int(rng.integers(max(height, width), height * width + 1))
    while len(cells) < goal:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _FOUR
                           if 0 <= r + dr < height and 0 <= c + dc < width
                           and (r + dr, c + dc) not in cells})
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _motif(rng, kind, routes):
    """A grammar of masks/composites, not a whole-puzzle parent call."""
    h, w = int(rng.integers(2, 6)), int(rng.integers(2, 7))
    a = int(rng.choice(routes))
    others = [c for c in routes if c != a]
    b = int(rng.choice(others)) if others else a
    cells = {}
    if kind == 'pins':
        # A rectangle whose locally identical rows/columns are distinguished by
        # another color elsewhere in the slab. Two diagonal B pins also teach 4C.
        cells = {(r, c): a for r in range(h) for c in range(w)}
        side = int(rng.integers(max(1, h - 1)))
        lower = int(rng.integers(max(1, w - 1)))
        cells.update({(r, w): b for r in range(side, h)})
        cells.update({(h, c): b for c in range(lower, w)})
    elif kind in ('frame', 'windows'):
        h, w = max(h, 3), max(w, 3)
        cells = {(r, c): a for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
        if kind == 'frame' and others:
            if rng.random() < .5:
                cells.update({(r, c): b for r in range(1, h - 1) for c in range(1, w - 1)})
            else:
                cells[(int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))] = b
        if kind == 'windows':
            for c in range(2, w - 1, 2):
                cells.update({(r, c): a for r in range(1, h - 1)})
            if others:
                for c in range(1, w - 1, 2):
                    cells.update({(r, c): b for r in range(1, h - 1)})
    elif kind == 'rectangle':
        cells = {(r, c): a for r in range(h) for c in range(w)}
    elif kind == 'elbow':
        cells = {(r, 0): a for r in range(h)}
        cells.update({(h - 1, c): a for c in range(w)})
    elif kind == 'stair':
        r = c = 0
        cells[(r, c)] = a
        while r < h - 1 or c < w - 1:
            if r < h - 1 and (c == w - 1 or rng.random() < .5):
                r += 1
            else:
                c += 1
            cells[(r, c)] = a
    elif kind == 'branch':
        spine = int(rng.integers(w))
        cells = {(r, spine): a for r in range(h)}
        for r in range(h):
            if rng.random() < .6:
                end = int(rng.integers(w))
                cells.update({(r, c): a for c in range(min(end, spine), max(end, spine) + 1)})
    else:
        cells = {p: a for p in _walk(rng, h, w)}
    # Reflections change motif geometry, never a routed mask.
    # Keep the pins' orientation so both same-color parts have disjoint columns.
    if kind != 'pins':
        hh, ww = 1 + max(r for r, c in cells), 1 + max(c for r, c in cells)
        if rng.random() < .5:
            cells = {(hh - 1 - r, c): color for (r, c), color in cells.items()}
        if rng.random() < .5:
            cells = {(r, ww - 1 - c): color for (r, c), color in cells.items()}
    return cells


def _design(rng, count, lesson):
    """Build semantic-color staging with disjoint lateral motif slots."""
    routes = list(range(1, count + 1))
    rng.shuffle(routes)
    if lesson == 'pins':
        motifs = [_motif(rng, 'pins', routes[:2])]
    elif lesson == 'recurrence':
        # A closed colored frame provides exactly recurring separated vectors.
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 7))
        a, b = routes[:2]
        motif = {(r + 1, c): a for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
        motif.update({(r + 1, c): b for r in range(1, h - 1) for c in range(1, w - 1)})
        # Same-color piece at another depth, outside the recurring row/col spans.
        motif.update({(0, w + 1 + c): a for c in range(int(rng.integers(1, 4)))})
        motifs = [motif]
    else:
        kinds = ('pins', 'frame', 'windows', 'rectangle', 'elbow', 'stair', 'branch', 'walk')
        number = int(rng.integers(2, 5))
        motifs = [_motif(rng, str(rng.choice(kinds if count > 1 else kinds[1:])), routes)
                  for _ in range(number)]
    used = set(color for m in motifs for color in m.values())
    for route in routes:
        if route not in used:
            motifs.append(_motif(rng, str(rng.choice(('elbow', 'rectangle', 'walk'))), [route]))
    # For the recurrence lesson, added motifs must not disturb the two equal rows.
    positions, cursor = [], int(rng.integers(1, 4))
    for i, motif in enumerate(motifs):
        top = 1 if lesson == 'recurrence' and i == 0 else int(rng.integers(1, 4))
        positions.append((top, cursor, motif))
        cursor += max(c for r, c in motif) + 1 + int(rng.integers(1, 4))
    height = max(top + 1 + max(r for r, c in m) for top, left, m in positions) + 1
    width = cursor + 1
    stage = np.zeros((height, width), dtype=np.int8)
    for top, left, motif in positions:
        for (r, c), color in motif.items():
            stage[top + r, left + c] = color
    # The template is only an authoring aid. All equality, including any
    # accidental equality across motifs, is resolved globally before expansion.
    return _collapse(stage)


def _repeats(rng, length, limit, cap, active, force):
    if length + int(force) > limit:
        raise ValueError("no room for required expanded vector")
    runs = np.ones(length, dtype=int)
    if force:
        runs[int(rng.choice(active))] = cap
        if int(runs.sum()) > limit:
            raise ValueError("required repetition exceeds scene budget")
    choices = list(map(int, rng.permutation(length)))
    for i in choices:
        room = limit - int(runs.sum())
        if room and runs[i] < cap:
            runs[i] += int(rng.integers(0, min(cap - runs[i], room) + 1))
    if force and np.all(runs == runs[0]):
        # The background border is not an active vector: this leaves the forced
        # foreground expansion intact while exposing unequal run lengths.
        runs[0] = 1
    return runs


def _attempt_pair(rng, lesson, test, turn):
    count = int(rng.integers(2, 4)) if lesson else int(rng.integers(1, 5))
    # Keep the second lesson's full-vector recurrence pristine: only two colors.
    if lesson == 'recurrence':
        count = 2
    compact_roles = _design(rng, count, lesson)
    if max(compact_roles.shape) > 24:
        raise ValueError("design too large for expansion")
    palette = list(map(int, rng.permutation(10)))
    background, routes = palette[0], palette[1:count + 1]
    role_map = np.array([background] + routes, dtype=np.int8)
    compact = role_map[compact_roles]
    components = _components(compact, background)
    min_depth = {color: max(1 + max(r for r, c in cells) - min(r for r, c in cells)
                             for cc, cells in components if cc == color)
                 for color in routes}
    depths = {color: depth + int(rng.integers(0, 3)) for color, depth in min_depth.items()}
    order = list(map(int, rng.permutation(routes)))
    gaps = [int(rng.integers(1, 3)) for _ in range(count - 1)]
    margin = int(rng.integers(0, 3))
    row_budget = 30 - sum(depths.values()) - sum(gaps) - margin
    # Training repeats each vector at most twice; tests include a 3- or 4-run.
    cap = int(rng.integers(3, 5)) if test else 2
    mode = int(rng.integers(3)) if not lesson else 0
    row_force, col_force = mode != 2, mode != 1
    # Both axes are necessarily visible in the first lesson. Test expansion is
    # guaranteed novel even if only one of its axes has a repeated vector.
    row_active = np.flatnonzero(np.any(compact != background, axis=1))
    col_active = np.flatnonzero(np.any(compact != background, axis=0))
    rr = (_repeats(rng, compact.shape[0], row_budget, cap, row_active, row_force)
          if row_force else np.ones(compact.shape[0], dtype=int))
    cc = (_repeats(rng, compact.shape[1], 30, cap, col_active, col_force)
          if col_force else np.ones(compact.shape[1], dtype=int))
    stage = np.repeat(np.repeat(compact, rr, axis=0), cc, axis=1)
    if lesson == 'pins':
        # Make the no-B counterfactual a legal placement, not merely an overflow.
        raw = _components(stage, background)
        for color in routes:
            raw_height = max(1 + max(r for r, c in cells) - min(r for r, c in cells)
                             for cc0, cells in raw if cc0 == color)
            depths[color] = max(depths[color], raw_height)
    height = stage.shape[0] + sum(depths.values()) + sum(gaps) + margin
    if height > 30:
        raise ValueError("expanded slab and bands exceed canvas budget")
    canvas = np.full((height, stage.shape[1]), background, dtype=np.int8)
    canvas[:stage.shape[0]] = stage
    bands, cursor = [], stage.shape[0]
    for i, route in enumerate(order):
        rail_options = [c for c in range(10) if c not in (background, route)]
        rail = int(rng.choice(rail_options))
        band = dict(top=cursor, bottom=cursor + depths[route] - 1, route=route, rail=rail)
        canvas[band['top']:band['bottom'] + 1] = route
        canvas[band['top']:band['bottom'] + 1, [0, -1]] = rail
        bands.append(band)
        cursor += depths[route] + (gaps[i] if i < len(gaps) else 0)
    # Construction ground truth uses the sampled COMPACT design, not solve or
    # a second compression of the expanded pixels. Validation rederives it.
    out = _paint(canvas, stage.shape[0], background, bands, components)
    pair = {'input': np.rot90(canvas, -turn).copy(),
            'output': np.rot90(out, -turn).copy()}
    a = _analyze(pair['input'])
    if not np.array_equal(a['compact'], compact):
        raise ValueError("ambiguous compaction or orientation")
    return pair


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    turn = int(rng.integers(4))  # One visible staging direction per puzzle.
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            lesson = ('pins' if i == 0 else 'recurrence' if i == 1 else None) if split == 'train' else None
            for _ in range(160):
                try:
                    pair = _attempt_pair(rng, lesson, split == 'test', turn)
                except ValueError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _recurrence(a, axis, background):
    vectors = a if axis == 0 else a.T
    seen = {}
    for i, vector in enumerate(vectors):
        if np.all(vector == background):
            continue
        key = vector.tobytes()
        if key in seen and i > seen[key] + 1:
            return True
        seen[key] = i
    return False


def _local_witness(a):
    for color, cells in a['components']:
        rr, cc = zip(*cells)
        mask = np.zeros((max(rr) - min(rr) + 1, max(cc) - min(cc) + 1), dtype=np.int8)
        for r, c in cells:
            mask[r - min(rr), c - min(cc)] = 1
        if _collapse(mask).shape != mask.shape:
            return True
    return False


def _alternative(a, rows=True, columns=True, diagonal=False, group_color=False):
    slab = _collapse(a['stage'], rows, columns)
    components = _components(slab, a['background'], diagonal)
    if group_color:
        grouped = {}
        for color, cells in components:
            grouped.setdefault(color, []).extend(cells)
        components = [(color, tuple(cells)) for color, cells in sorted(grouped.items())]
    try:
        return _paint(a['canonical'], a['stop'], a['background'], a['bands'], components)
    except ValueError:
        return None


def _different(a, alternative):
    return alternative is None or not np.array_equal(a['expected'], alternative)


def _measure(a):
    bg, compact, comps = a['background'], a['compact'], a['components']
    masks = []
    for color, cells in comps:
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        masks.append(tuple((r - r0, c - c0) for r, c in cells))
    contact = any(compact[r + dr, c + dc] not in (bg, color)
                  for color, cells in comps for r, c in cells for dr, dc in _FOUR
                  if 0 <= r + dr < compact.shape[0] and 0 <= c + dc < compact.shape[1])
    groups = {}
    for color, cells in comps:
        groups.setdefault(color, set()).add(max(r for r, c in cells))
    return dict(shape=a['canonical'].shape, compact_shape=compact.shape,
                background=bg, turn=a['turn'], bands=len(a['bands']),
                depths=tuple(b['bottom'] - b['top'] + 1 for b in a['bands']),
                row_runs=_runs(a['stage'], 0), column_runs=_runs(a['stage'], 1),
                components=len(comps), masks=tuple(masks), contact=contact,
                independent=any(len(bottoms) > 1 for bottoms in groups.values()),
                global_pinning=_local_witness(a),
                row_recurrence=_recurrence(compact, 0, bg),
                column_recurrence=_recurrence(compact, 1, bg),
                reuse=any(b['rail'] in groups for b in a['bands']),
                shared_rails=len({b['rail'] for b in a['bands']}) < len(a['bands']))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), "illegal example counts")
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "wrong pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "canvas must be preserved")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyze(inp)
            _require(np.array_equal(out, a['output']), "wrong compacted component routing")
            bg, slab, compact = a['background'], a['stage'], a['compact']
            _require(np.all(a['expected'][:a['stop']] == bg), "staging was not cleared")
            _require(np.array_equal(_collapse(compact), compact), "compaction is not idempotent")
            rr, cc = _runs(slab, 0), _runs(slab, 1)
            _require(np.array_equal(np.repeat(np.repeat(compact, rr, axis=0), cc, axis=1), slab),
                     "lost literal staging values or separated recurrences")
            routes = {b['route'] for b in a['bands']}
            _require(routes == {color for color, cells in a['components']}, "generated bands must all be used")
            # Area is conserved from compacted masks to visible destination ink.
            area = sum(len(cells) for color, cells in a['components'])
            ink = 0
            for b in a['bands']:
                region = a['expected'][b['top']:b['bottom'] + 1]
                original = a['canonical'][b['top']:b['bottom'] + 1]
                _require(np.array_equal(region[:, [0, -1]], original[:, [0, -1]]), "rail changed")
                ink += int(np.count_nonzero(region[:, 1:-1] == b['rail']))
            _require(area == ink, "compacted cell count and stamped area disagree")
            analyses[split].append(a)
    turns = {a['turn'] for split in analyses.values() for a in split}
    _require(len(turns) == 1, "generated puzzle must keep one orientation")
    train = analyses['train']
    metrics = [_measure(a) for a in train]
    _require(any(m['contact'] for m in metrics), "training must show multicolor segmentation")
    _require(any(m['independent'] for m in metrics), "training must show independent component alignment")
    _require(any(m['global_pinning'] for m in metrics), "missing whole-slab versus local-collapse witness")
    _require(any(m['row_recurrence'] and m['column_recurrence'] for m in metrics),
             "missing separated vector recurrence witness")
    for rows, columns, description in ((False, True, 'rows'), (True, False, 'columns'),
                                        (False, False, 'both axes')):
        # At least one omission witness must itself be a legal stamp.
        alternatives = [(a, _alternative(a, rows, columns)) for a in train]
        _require(any(alt is not None and _different(a, alt) for a, alt in alternatives),
                 'training does not disambiguate compression on ' + description)
    _require(any(_different(a, _alternative(a, diagonal=True)) for a in train),
             "training does not distinguish 4- from 8-connectivity")
    _require(any(_different(a, _alternative(a, group_color=True)) for a in train),
             "training does not distinguish independent same-color components")
    max_train = max(max(_runs(a['stage'], axis)) for a in train for axis in (0, 1))
    _require(max_train == 2, "training expansion regime must use runs at most two")
    for a in analyses['test']:
        _require(max(max(_runs(a['stage'], axis)) for axis in (0, 1)) > max_train,
                 "test must generalize to longer equality runs")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _expect_error(callback, error=ValueError):
    try:
        callback()
    except error:
        return
    raise AssertionError('expected ' + error.__name__)


def _regressions():
    # Literal recurrence survives, with no replacement of crossing/meeting values.
    slab = np.array([[0, 0, 0, 0, 0], [0, 1, 2, 1, 0],
                     [0, 1, 2, 1, 0], [0, 3, 2, 3, 0],
                     [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]], dtype=int)
    expected = slab[[0, 1, 3, 4, 5]]
    _require(np.array_equal(_collapse(slab), expected), "separated repeat was removed")
    _require(np.array_equal(_collapse(slab.T), expected.T), "axis independence failed")
    # A hand-worked compact pinned block. Expansion doubles each nonempty vector.
    small = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0],
                      [0, 1, 1, 2, 0], [0, 0, 2, 0, 0],
                      [0, 0, 0, 0, 0]], dtype=np.int8)
    raw = np.repeat(np.repeat(small, [1, 2, 2, 2, 1], axis=0), [1, 2, 2, 2, 1], axis=1)
    canvas = np.zeros((18, 8), dtype=np.int8)
    canvas[:8] = raw
    canvas[8:12] = 1
    canvas[8:12, [0, -1]] = 3
    canvas[13:17] = 2
    canvas[13:17, [0, -1]] = 4
    target = canvas.copy()
    target[:8] = 0
    target[10:12, 1:3] = 3
    target[16, 2:4] = 4
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(canvas, turn)), np.rot90(target, turn)),
                 "hand-worked far edge, origin, pinning or 4C regression failed")
    # Collision semantics: bbox overlap is allowed, common target cells are not.
    bands = [dict(top=4, bottom=6, route=1, rail=2)]
    blank = np.zeros((7, 8), dtype=np.int8)
    first = (1, ((0, 1), (0, 2), (1, 1)))
    second = (1, ((2, 2),))
    result = _paint(blank, 4, 0, bands, [first, second])
    _require(result[5, 2] == 2 and result[6, 2] == 2, "valid overlapping boxes rejected")
    _expect_error(lambda: _paint(blank, 4, 0, bands, [first, (1, ((2, 1),))]))
    # End-to-end interleaved same-color pieces and an unused destination. These
    # are within the solver's rule even though slot-based generation avoids them.
    interleaved = np.zeros((13, 8), dtype=np.int8)
    interleaved[1, 1:3] = 1
    interleaved[2, 1] = interleaved[4, 2] = 1
    interleaved[6:9] = 1
    interleaved[6:9, [0, -1]] = 2
    interleaved[10:12] = 5
    interleaved[10:12, [0, -1]] = 6
    target = interleaved.copy()
    target[:6] = 0
    target[7:9, 1:3] = 2
    for turn in range(4):
        inp = np.rot90(interleaved, turn).copy()
        before = inp.copy()
        _require(np.array_equal(solve(inp.tolist()), np.rot90(target, turn)),
                 "interleaved masks, bbox overlap or unused band failed")
        _require(np.array_equal(inp, before), "solve mutated its input")
    # A real same-column overlap is rejected by the input-only solver too.
    interleaved[4, 2], interleaved[4, 1] = 0, 1
    _expect_error(lambda: solve(interleaved))


def self_test():
    """Bounded: 42 five-pair puzzles, count boundaries, and small regressions.

    Retain aggregate sets/counters only; no files or execution traces are saved.
    The returned compact-serializable report measures actual returned grids.
    """
    _regressions()
    digests, shapes, compact_shapes, masks, layouts, normalized = set(), set(), set(), set(), set(), set()
    backgrounds, directions, band_counts, component_counts = set(), set(), set(), set()
    features = Counter()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = _puzzle_digest(puzzle)
        _require(digest == _puzzle_digest(generate_puzzle(seed)), "nondeterministic seed")
        digests.add(digest)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze(pair['input'])
                m = _measure(a)
                pairs += 1
                shapes.add(pair['input'].shape)
                compact_shapes.add(m['compact_shape'])
                masks.update(m['masks'])
                layouts.add((m['shape'], m['depths'], m['row_runs'], m['column_runs']))
                backgrounds.add(m['background'])
                directions.add(m['turn'])
                band_counts.add(m['bands'])
                component_counts.add(m['components'])
                # Normalize literal colors before calling something structural.
                roles, values = {}, []
                for value in pair['input'].flat:
                    values.append(roles.setdefault(int(value), len(roles)))
                normalized.add((pair['input'].shape, bytes(values)))
                for name in ('contact', 'independent', 'global_pinning', 'row_recurrence',
                             'column_recurrence', 'reuse', 'shared_rails'):
                    features[name] += int(m[name])
                r, c = max(m['row_runs']), max(m['column_runs'])
                features['both_axes'] += int(r > 1 and c > 1)
                features['one_axis'] += int((r == 1) != (c == 1))
                features['long_test_run'] += int(split == 'test' and max(r, c) >= 3)
    _require(len(digests) == 42, "cross-seed duplicate puzzles")
    _require(len(shapes) >= 35 and len(compact_shapes) >= 12, "narrow geometry variation")
    _require(len(masks) >= 35 and len(layouts) >= 100, "narrow mask/layout variation")
    _require(len(normalized) >= 150, "mostly palette-only variation")
    _require(backgrounds == set(range(10)) and directions == set(range(4)), "missing visible roles/directions")
    _require(band_counts >= {1, 2, 3, 4} and len(component_counts) >= 5, "narrow inventories")
    _require(all(features[name] for name in features), "unexercised structural feature")
    for a, b in ((2, 1), (10, 4), (10, 1), (5, 4)):
        p = generate_puzzle(200 + a + b, a, b)
        _require((len(p['train']), len(p['test'])) == (a, b), "count boundary failure")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        _expect_error(lambda a=a, b=b: generate_puzzle(0, a, b))
    _expect_error(lambda: generate_puzzle(-1))
    _expect_error(lambda: generate_puzzle(True))
    corrupted = generate_puzzle(991, 2, 1)
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0]) + 1) % 10
    _expect_error(lambda: validate_generated_puzzle(corrupted), AssertionError)
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(digests), input_shapes=len(shapes),
                compact_shapes=len(compact_shapes), component_masks=len(masks), layouts=len(layouts),
                color_normalized_inputs=len(normalized), backgrounds=sorted(backgrounds),
                directions=sorted(directions), band_counts=sorted(band_counts),
                component_counts=sorted(component_counts), features=dict(features),
                independent_solver=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
