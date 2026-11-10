"""Cyclic staircases carve the foreground used by clipped empty neighborhoods.

No parent module, seed, example bank, or generation state is used by solve().
Only NumPy and the Python standard library are required. Nothing is saved here.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require integer counts 2..10 / 1..4 with num_train > num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if not (grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
            and np.issubdtype(grid.dtype, np.integer)
            and np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("Expected an integer 1..30 by 1..30 grid with colors 0..9")
    return grid.astype(np.int8, copy=True)


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _orient(grid, code):
    return np.rot90(grid[:, ::-1] if code >= 4 else grid, code % 4).copy()


def _unorient(grid, code):
    grid = np.rot90(grid, -(code % 4))
    return (grid[:, ::-1] if code >= 4 else grid).copy()


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, group = [(r, c)], []
        while todo:
            y, x = todo.pop()
            group.append((y, x))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if _inside(mask.shape, p) and mask[p] and not seen[p]:
                    seen[p] = True
                    todo.append(p)
        groups.append(group)
    return groups


def _decode_canonical(grid, terrain_colors):
    swatches, starts = [], []
    for color in set(map(int, np.unique(grid))) - terrain_colors:
        for group in _components(grid == color):
            if len(group) == 1:
                starts.append(group[0])
                continue
            rows, cols = zip(*group)
            r, c = min(rows), min(cols)
            assert len(group) == 4 and max(rows) == r+1 and max(cols) == c+1
            swatches.append((r, c, color))
    assert 1 <= len(starts) <= 3 and 3 <= len(swatches) <= 7
    assert len({int(grid[p]) for p in starts}) == 1
    first = int(grid[starts[0]])
    surrounds = set()
    for r, c, color in swatches:
        for y in range(max(0, r-1), min(grid.shape[0], r+3)):
            for x in range(max(0, c-1), min(grid.shape[1], c+3)):
                if not (r <= y <= r+1 and c <= x <= c+1):
                    surrounds.add(int(grid[y, x]))
    assert len(surrounds) == 1 and surrounds <= terrain_colors
    solid = surrounds.pop()
    bg = (terrain_colors - {solid}).pop()
    restored = grid.copy()
    for r, c, color in swatches:
        restored[r:r+2, c:c+2] = solid
    for p in starts:
        restored[p] = bg
    interpretations = []
    for marker in swatches:
        legend = sorted((s for s in swatches if s != marker), key=lambda s: s[1])
        row = legend[0][0]
        word = tuple(s[2] for s in legend)
        if (len({s[0] for s in legend}) != 1 or marker[2] in word
                or first != word[0] or abs(marker[0]-row) < 3
                or any(b[1]-a[1] < 3 for a, b in zip(legend, legend[1:]))):
            continue
        if not all(r >= row+4 and c+1 < grid.shape[1]
                   and restored[r, c+1] == bg
                   and (c == 0 or restored[r, c-1] == solid) for r, c in starts):
            continue
        interpretations.append(dict(base=restored, background=bg, solid=solid,
                                    word=word, ink=marker[2], starts=sorted(starts),
                                    legend=legend, marker=marker, swatches=swatches))
    assert len(interpretations) == 1
    return interpretations[0]


def _decode(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) >= 5
    terrain_colors = set(map(int, colors[np.argsort(counts)[-2:]]))
    answers = []
    for code in range(8):
        try:
            info = _decode_canonical(_unorient(grid, code), terrain_colors)
        except (AssertionError, IndexError):
            continue
        info['orientation'] = code
        answers.append(info)
    assert len(answers) == 1, "Input must have one visible annotation/travel interpretation"
    return answers[0]


def _trace(mask, start):
    """Maximal legs; bend cells counted once; an exit immediately terminates."""
    current, direction = start, 0
    path, collisions, lengths = [], [], []
    directions = ((0, 1), (-1, 0))
    for _ in range(sum(mask.shape)+1):
        r, c = current
        dr, dc = directions[direction]
        leg = []
        while _inside(mask.shape, (r, c)) and mask[r, c]:
            leg.append((r, c))
            r, c = r+dr, c+dc
        assert leg
        path.extend(leg)
        lengths.append(len(leg))
        if not _inside(mask.shape, (r, c)):
            end = 'edge'
            break
        collisions.append((r, c))
        direction = 1-direction
        dr, dc = directions[direction]
        current = (leg[-1][0]+dr, leg[-1][1]+dc)
        if not _inside(mask.shape, current) or not mask[current]:
            end = 'blocked'
            break
    else:
        raise AssertionError("Nonterminating route")
    return dict(path=tuple(path), collisions=tuple(collisions), legs=tuple(lengths), end=end)


def _records(info):
    mask = info['base'] == info['background']
    records = [_trace(mask, p) for p in info['starts']]
    annotations = {(y, x) for r, c, _ in info['swatches']
                   for y in (r, r+1) for x in (c, c+1)}
    occupied = set()
    for record in records:
        cells = set(record['path']) | set(record['collisions'])
        assert not occupied & cells and not annotations & cells
        occupied |= cells
    return records


def _paint(info, records):
    stage = info['base'].copy()
    for record in records:
        for phase, p in enumerate(record['path']):
            stage[p] = info['word'][phase % len(info['word'])]
        for p in record['collisions']:
            stage[p] = info['ink']
    return stage


def _empty_centers(stage, background):
    """Outside the canvas contributes no foreground; select centers only."""
    h, w = stage.shape
    padded = np.pad(stage != background, 1, constant_values=False)
    near = np.zeros((h, w), bool)
    for dy in range(3):
        for dx in range(3):
            near |= padded[dy:dy+h, dx:dx+w]
    return ~near


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode visible cues, paint A's routes, then select B's empty centers once.

    Background is the traversable terrain, identified by the singleton starts;
    every other color in the restored, painted scene is foreground. Annotation
    removal precedes both operations. New highlights do not alter the selection.
    """
    info = _decode(_grid(input_grid))
    stage = _paint(info, _records(info))
    output = stage.copy()
    output[_empty_centers(stage, info['background'])] = info['ink']
    return _orient(output, info['orientation'])


def _features(info, records, stage):
    bg, word = info['background'], info['word']
    core = _empty_centers(stage, bg)
    evidence = set()
    if np.any(core):
        evidence.add('highlight')
    if np.count_nonzero(core) >= 12:
        evidence.add('broad_core')
    if core[0].any() or core[-1].any() or core[:, 0].any() or core[:, -1].any():
        evidence.add('clipped_edge')
    if any(len(group) <= 4 for group in _components(core)):
        evidence.add('small_core')
    cross_clear = stage == bg
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        padded = np.pad(stage != bg, 1, constant_values=False)
        cross_clear &= ~padded[1+dr:1+dr+stage.shape[0], 1+dc:1+dc+stage.shape[1]]
    if np.any(cross_clear & ~core):
        evidence.add('diagonal')
    unpainted = info['base'].copy()
    for p in info['starts']:
        unpainted[p] = word[0]
    # A genuinely changes B on background cells outside the painted path itself.
    if np.any(_empty_centers(unpainted, bg) & (stage == bg) & ~core):
        evidence.add('route_dependency')
    distinct = tuple(dict.fromkeys(word))
    offset = 0
    for record in records:
        n = len(record['path'])
        correct = [word[j % len(word)] for j in range(n)]
        if n >= 2*len(word):
            evidence.add('cycling')
        if correct != [distinct[j % len(distinct)] for j in range(n)]:
            evidence.add('multiplicity')
        reset, consumed, phase = [], [], 0
        for leg_index, length in enumerate(record['legs']):
            reset += [word[j % len(word)] for j in range(length)]
            consumed += [word[(phase+j+leg_index) % len(word)] for j in range(length)]
            phase += length
        if reset != correct:
            evidence.add('bend_phase')
        if consumed != correct:
            evidence.add('marker_phase')
        if correct != [word[(offset+j) % len(word)] for j in range(n)]:
            evidence.add('independent_restart')
        offset += n
        if record['collisions']:
            evidence.add('collision')
    return evidence


_WITNESS_ONE = {'cycling', 'multiplicity', 'bend_phase', 'marker_phase',
                'route_dependency', 'broad_core', 'diagonal'}
_WITNESS_TWO = {'independent_restart', 'collision', 'small_core',
                'clipped_edge', 'route_dependency', 'diagonal'}
_REQUIRED = _WITNESS_ONE | _WITNESS_TWO


def _word(rng, witness, test, length=None):
    if witness == 1:
        patterns = ((0, 0, 1), (0, 1, 1, 0), (0, 0, 1, 1), (0, 1, 0, 2))
        choices = [p for p in patterns if length is None or len(p) == length]
        return choices[int(rng.integers(len(choices)))]
    n = int(rng.integers(2, 7 if test else 6)) if length is None else length
    d = int(rng.integers(2, min(n, 5)+1))
    word = list(range(d)) + list(map(int, rng.integers(0, d, n-d)))
    rng.shuffle(word)
    return tuple(word)


def _terrain(rng, h, w, header, witness, test):
    """Shifted corridors, broad bays, short necks, islands and incidental pockets."""
    mask = np.zeros((h, w), bool)
    rows = np.arange(header, h)
    n = len(rows)
    segments = int(rng.integers(3, min(8, n)+1))
    knots = np.unique(np.r_[0, rng.choice(np.arange(1, n-1),
                                        min(segments-2, n-2), replace=False), n-1])
    drift = float(rng.uniform(w*.25, w*.65))
    low = float(rng.uniform(-3, w*.13))
    left_knots = low + np.linspace(drift, 0, len(knots)) + rng.uniform(-2, 2, len(knots))
    left = np.floor(np.interp(np.arange(n), knots, left_knots)).astype(int)
    widths = rng.integers(4, min(w-2, 11 if witness == 1 else 17)+1, len(knots))
    breadth = np.rint(np.interp(np.arange(n), knots, widths)).astype(int)
    if test:
        # New combinations include short, two-cell necks amid broad chambers.
        for _ in range(int(rng.integers(1, 4))):
            j = int(rng.integers(n))
            breadth[j] = int(rng.integers(2, 5))
    for j, r in enumerate(rows):
        a, b = max(0, int(left[j])), min(w, int(left[j]+breadth[j]))
        mask[r, a:max(a+1, b)] = True
    # Pair-specific offshoots/holes alter the actual maximal paths, not a hidden route.
    for _ in range(int(rng.integers(1, 6 if test else 4))):
        r, c = int(rng.integers(header, h-1)), int(rng.integers(w-2))
        dh, dw = int(rng.integers(1, min(6, h-r)+1)), int(rng.integers(2, min(9, w-c)+1))
        mask[r:r+dh, c:c+dw] = bool(rng.random() < .72)
    # Sometimes clear to the entry edge or add a broad edge bay.
    if rng.random() < .6:
        r = int(rng.integers(header, max(header+1, h-3)))
        mask[r:min(h, r+int(rng.integers(3, 7))), :int(rng.integers(4, w//2+1))] = True
    # Small corner cavity: four eligible centers, with an inward solid frame.
    # This explicitly rules out B-v2's component-size cutoff of five.
    if witness == 2 or rng.random() < .3:
        if rng.random() < .5:
            mask[-4:, :4] = False
            mask[-3:, :3] = True
        else:
            mask[-4:, -4:] = False
            mask[-3:, -3:] = True
    mask[:header] = False
    return mask


def _select(rng, mask, header, count, witness, word):
    candidates = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if (r < header or c+1 >= mask.shape[1] or not mask[r, c+1]
                or (c and mask[r, c-1])):
            continue
        record = _trace(mask, (r, c))
        if len(record['path']) >= 8:
            candidates.append(((r, c), record))
    rng.shuffle(candidates)
    if witness == 1:
        candidates.sort(key=lambda x: (len(x[1]['legs']), len(x[1]['path'])), reverse=True)
    # Try different first routes; choosing one greedy long path can hide valid pairs.
    for first in range(min(len(candidates), 32)):
        selected, occupied = [], set()
        for start, record in candidates[first:]+candidates[:first]:
            footprint = set(record['path']) | set(record['collisions'])
            if footprint & occupied or any(max(abs(start[0]-s[0]), abs(start[1]-s[1])) <= 1
                                            for s, _ in selected):
                continue
            selected.append((start, record))
            occupied |= footprint
            if len(selected) == count:
                selected.sort(key=lambda x: x[0])
                if witness == 2:
                    seqs, offset = [], 0
                    for _, rec in selected:
                        n = len(rec['path'])
                        seqs.append(any(word[j % len(word)] != word[(offset+j) % len(word)]
                                        for j in range(n)))
                        offset += n
                    if not any(seqs):
                        break
                return selected
    return None


def _make_pair(rng, orientation, witness=0, test=False, word_length=None):
    pattern = _word(rng, witness, test, word_length)
    count = 1 if witness == 1 else 2 if witness == 2 else int(rng.choice([1, 2, 3], p=[.5, .4, .1]))
    palette = list(map(int, rng.permutation(10)))
    bg, solid, ink = palette[:3]
    word = tuple(palette[3+x] for x in pattern)
    required = _WITNESS_ONE if witness == 1 else _WITNESS_TWO if witness == 2 else {'route_dependency', 'highlight'}
    for attempt in range(240):
        h = int(rng.integers(20 if witness else 17, 31))
        w = int(rng.integers(max(18, 3*len(word)+1), 31))
        legend_row = int(rng.integers(0, 3))
        header = legend_row+4
        columns = [int(rng.integers(0, min(5, w-3*len(word))+1))]
        for j in range(len(word)-1):
            remaining = len(word)-2-j
            max_gap = min(3, w-(columns[-1]+2)-3*remaining-2)
            columns.append(columns[-1]+2+int(rng.integers(1, max_gap+1)))
        mask = _terrain(rng, h, w, header, witness, test)
        if min(int(mask.sum()), int((~mask).sum())-4*(len(word)+1)) < 35:
            continue
        chosen = _select(rng, mask, header, count, witness, word)
        if chosen is None:
            continue
        starts = [s for s, _ in chosen]
        records = [record for _, record in chosen]
        base = np.where(mask, bg, solid).astype(np.int8)
        blocked = np.zeros(mask.shape, bool)
        blocked[max(0, legend_row-1):legend_row+3, max(0, columns[0]-1):columns[-1]+3] = True
        for record in records:
            for r, c in record['path']+record['collisions']:
                blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
        positions = []
        for r in range(1, h-2):
            if abs(r-legend_row) < 3:
                continue
            for c in range(1, w-2):
                if not mask[r-1:r+3, c-1:c+3].any() and not blocked[r-1:r+3, c-1:c+3].any():
                    positions.append((r, c))
        if not positions:
            continue
        mr, mc = positions[int(rng.integers(len(positions)))]
        swatches = [(legend_row, c, color) for c, color in zip(columns, word)]+[(mr, mc, ink)]
        info = dict(base=base, background=bg, solid=solid, ink=ink, word=word,
                    starts=starts, swatches=swatches, legend=swatches[:-1], marker=swatches[-1],
                    orientation=orientation)
        stage = _paint(info, records)
        if not required <= _features(info, records, stage):
            continue
        inp = base.copy()
        for r, c, color in swatches:
            inp[r:r+2, c:c+2] = color
        for p in starts:
            inp[p] = word[0]
        # Confirm the two terrain roles dominate every annotation color.
        vals, counts = np.unique(inp, return_counts=True)
        if set(map(int, vals[np.argsort(counts)[-2:]])) != {bg, solid}:
            continue
        output = stage.copy()
        output[_empty_centers(stage, bg)] = ink
        pair = {'input': _orient(inp, orientation), 'output': _orient(output, orientation)}
        try:
            decoded = _decode(pair['input'])
            _records(decoded)
        except AssertionError:
            continue
        return pair
    raise RuntimeError(f"Could not construct a diagnostic pair (witness {witness}) in 240 attempts")


def _key(grid):
    array = np.asarray(grid, dtype=np.uint8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_orientation = int(rng.integers(8))
    mode = int(rng.integers(3))
    witness_slots = list(map(int, rng.choice(nt, 2, replace=False)))
    first_length = int(rng.choice([3, 4]))
    second_length = int(rng.choice([n for n in range(2, 6) if n != first_length]))
    training_lengths = set()
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            witness = (1 if i == witness_slots[0] else 2 if i == witness_slots[1] else 0) if split == 'train' else 0
            orientation = base_orientation
            if mode == 1:
                orientation = int(rng.integers(8))
            elif mode == 2 and split == 'test':
                orientation = (base_orientation//4)*4+(base_orientation+1)%4
            word_length = (first_length if witness == 1 else second_length if witness == 2 else None)
            if split == 'test':
                word_length = int(rng.choice([n for n in range(2, 7) if n not in training_lengths]))
            for _ in range(8):
                pair = _make_pair(rng, orientation, witness, split == 'test', word_length)
                key = _key(pair['input'])
                if key not in used:
                    break
            else:
                raise RuntimeError("Could not create distinct inputs")
            used.add(key)
            puzzle[split].append(pair)
            if split == 'train':
                training_lengths.add(len(_decode(pair['input'])['word']))
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_by_steps(info, records, stage, output):
    """Check maximal-run production using cell steps and direct clipped windows."""
    bg, word = info['background'], info['word']
    rebuilt = info['base'].copy()
    for start, record in zip(info['starts'], records):
        current, vertical, phase = start, False, 0
        path, collisions, lengths = [], [], [0]
        while True:
            assert info['base'][current] == bg
            path.append(current)
            lengths[-1] += 1
            rebuilt[current] = word[phase % len(word)]
            phase += 1
            r, c = current
            ahead = (r-1, c) if vertical else (r, c+1)
            if not _inside(rebuilt.shape, ahead):
                end = 'edge'
                break
            if info['base'][ahead] == bg:
                current = ahead
                continue
            rebuilt[ahead] = info['ink']
            collisions.append(ahead)
            turn = (r, c+1) if vertical else (r-1, c)
            if not _inside(rebuilt.shape, turn) or info['base'][turn] != bg:
                end = 'blocked'
                break
            current, vertical = turn, not vertical
            lengths.append(0)
        assert record == dict(path=tuple(path), collisions=tuple(collisions), legs=tuple(lengths), end=end)
    assert np.array_equal(rebuilt, stage)
    selected = np.zeros(stage.shape, bool)
    for r, c in np.argwhere(stage == bg):
        selected[r, c] = np.all(stage[max(0, r-1):r+2, max(0, c-1):c+2] == bg)
    assert np.array_equal(selected, _empty_centers(stage, bg))
    assert np.all(output[selected] == info['ink'])
    assert np.array_equal(output[~selected], stage[~selected])
    assert np.array_equal(output[stage != bg], stage[stage != bg])


def validate_generated_puzzle(puzzle: dict) -> None:
    """Check exact cell behavior, route conflicts, and diagnostic demonstrations."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    used, evidence = set(), set()
    lengths = {'train': set(), 'test': set()}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in used
            used.add(_key(inp))
            info = _decode(inp)
            lengths[split].add(len(info['word']))
            records = _records(info)
            stage = _paint(info, records)
            canonical_output = _unorient(out, info['orientation'])
            _check_by_steps(info, records, stage, canonical_output)
            features = _features(info, records, stage)
            assert {'route_dependency', 'highlight'} <= features
            assert np.any(canonical_output == info['background'])
            assert np.all(info['base'][stage == info['background']] == info['background'])
            if split == 'train':
                evidence |= features
    assert _REQUIRED <= evidence, _REQUIRED-evidence
    assert len(lengths['train']) >= 2 and lengths['train'].isdisjoint(lengths['test'])


def _regressions():
    # Canvas exit must not turn, even when terrain exists around its corner.
    mask = np.zeros((6, 9), bool)
    mask[4, :] = True
    mask[2:5, -1] = True
    trace = _trace(mask, (4, 0))
    assert trace['path'] == tuple((4, c) for c in range(9))
    assert trace['end'] == 'edge' and not trace['collisions']
    mask = np.zeros((8, 10), bool)
    mask[6, :4] = True
    mask[:7, 3] = True
    mask[0, 3:9] = True
    trace = _trace(mask, (6, 0))
    assert trace['path'][-1] == (0, 3) and trace['collisions'] == ((6, 4),)
    # Clipping, all eight neighbors, no component cutoff, simultaneous selection.
    stage = np.zeros((7, 8), np.int8)
    stage[3, 3] = 7
    core = _empty_centers(stage, 0)
    assert core[0, 0] and not core[2, 2] and core[1, 1]
    assert core.sum() == stage.size-9
    cavity = np.ones((5, 5), np.int8)
    cavity[1:4, 1:4] = 0
    assert np.argwhere(_empty_centers(cavity, 0)).tolist() == [[2, 2]]


def self_test() -> None:
    """42 seeds, streamed; no files, unbounded search, or retained puzzle corpus."""
    _regressions()
    signatures, shapes, geometries, orientations, patterns, roles, route_counts = (set() for _ in range(7))
    leg_counts, core_sizes = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                info = _decode(inp)
                records = _records(info)
                shapes.add(inp.shape)
                orientations.add(info['orientation'])
                geometries.add(hashlib.sha256((info['base'] == info['background']).tobytes()).digest())
                labels = {}
                patterns.add(tuple(labels.setdefault(c, len(labels)) for c in info['word']))
                roles.add((info['background'], info['solid'], info['ink']))
                route_counts.add(len(records))
                leg_counts.update(len(r['legs']) for r in records)
                core_sizes.update(len(g) for g in _components(_empty_centers(_paint(info, records), info['background'])))
                assert np.array_equal(solve(inp), pair['output'])
                signature.update(bytes(inp.shape)+inp.tobytes()+pair['output'].tobytes())
        signatures.add(signature.digest())
    assert len(signatures) == 42 and len(shapes) >= 30 and len(geometries) >= 180
    assert len(orientations) == 8 and len(patterns) >= 12 and len(roles) >= 100
    assert route_counts >= {1, 2, 3} and {1, 2, 3} <= leg_counts
    assert min(core_sizes) <= 4 and max(core_sizes) >= 30
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(20261023, nt, ne)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.0, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    a, b = generate_puzzle(31415, 3, 2), generate_puzzle(31415, 3, 2)
    assert all(np.array_equal(x[k], y[k]) for s in ('train', 'test')
               for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
