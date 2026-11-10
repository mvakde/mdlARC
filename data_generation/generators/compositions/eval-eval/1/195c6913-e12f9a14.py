"""Resonant confluences: cyclic phase controls whether frame-emitted rays join.

Only standard-library modules and NumPy are used. The decoder and solver read
visible symbols only; generation has no hidden conventions or stored solutions.
"""
from __future__ import annotations

import numpy as np

# Offsets are measured from the top-left of a 2x2 core.
RING = tuple((r, c) for r in range(-1, 3) for c in range(-1, 3)
             if r in (-1, 2) or c in (-1, 2))
PORTS = {(-1, 0): (-1, 0), (-1, 1): (-1, 0),
         (2, 0): (1, 0), (2, 1): (1, 0),
         (0, -1): (0, -1), (1, -1): (0, -1),
         (0, 2): (0, 1), (1, 2): (0, 1)}
HOST_SEEDS = (0, 758800712, 1955622, 743596129, 127528093, 572972868,
              914877710, 220772904, 376439378, 739078497, 636598532,
              245220229, 783281471, 97823241, 605973575, 513531313,
              257655119, 689566612, 611023005, 48759138, 96903246,
              285044588, 443438721, 274600487, 8416909, 61179511,
              132838448, 802395277, 253961956, 609661365, 967073294,
              706074654, 497612180, 179802254, 868646373, 942667243,
              652623557, 948417903, 505960304, 878859717, 837758517,
              370188551)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= x <= 30 for x in grid.shape),
             'grids must be two-dimensional with axes in 1..30')
    _require(np.issubdtype(grid.dtype, np.integer), 'grid colors must be integers')
    _require(np.all((0 <= grid) & (grid <= 9)), 'grid colors must be in 0..9')
    return grid.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _line(p, d, shape):
    result = []
    while _inside(p, shape):
        result.append(p)
        p = _add(p, d)
    return result


def _footprint(core):
    r, c = core
    return {(r+i, c+j) for i in range(-1, 3) for j in range(-1, 3)}


def _decode(value):
    grid = _grid(value)
    h, w = grid.shape
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[background]) == 1,
             'background must be the unique most frequent color')
    blocks = ((grid[:-1, :-1] != background)
              & (grid[:-1, :-1] == grid[1:, :-1])
              & (grid[:-1, :-1] == grid[:-1, 1:])
              & (grid[:-1, :-1] == grid[1:, 1:]))
    bare, frames = [], []
    for rr, cc in np.argwhere(blocks):
        r, c = int(rr), int(cc)
        _require(1 <= r <= h-3 and 1 <= c <= w-3,
                 'every 2x2 symbol needs its full surrounding ring')
        ring = [((r+i, c+j), int(grid[r+i, c+j]), (i, j)) for i, j in RING]
        foreground = {v for _, v, _ in ring if v != background}
        color = int(grid[r, c])
        symbol = dict(core=(r, c), color=color, ring=ring)
        if not foreground:
            bare.append(symbol)
        else:
            _require(len(foreground) == 1 and color not in foreground,
                     'frame ring must be monochromatic and differ from its core')
            symbol['structure'] = next(iter(foreground))
            frames.append(symbol)
    _require(2 <= len(bare) <= 6, 'need 2..6 bare legend swatches')
    bare.sort(key=lambda s: s['core'][1])
    _require(len({s['core'][0] for s in bare}) == 1,
             'legend swatches must share a horizontal row')
    _require(all(b['core'][1]-a['core'][1] >= 3 for a, b in zip(bare, bare[1:])),
             'legend swatches need a background gap')
    word = tuple(s['color'] for s in bare)
    _require(len(set(word)) >= 2, 'a cyclic word needs at least two distinct colors')
    _require(len(frames) >= 2 and len({s['structure'] for s in frames}) == 1,
             'need emitters and one closed frame with a common ring color')
    structure = frames[0]['structure']
    _require(structure not in word, 'ring color cannot occur in the word')
    closed = [s for s in frames if all(v != background for _, v, _ in s['ring'])]
    _require(len(closed) == 1, 'exactly one closed frame supplies the marker')
    marker = closed[0]
    _require(marker['color'] not in word, 'marker color is separate from the word')
    emitters = sorted([s for s in frames if s is not marker], key=lambda s: s['core'])
    _require(1 <= len(emitters) <= 5, 'need 1..5 emitting frames')
    all_symbols, covered, frame_cells, streams = set(), set(), set(), []
    for s in bare:
        r, c = s['core']
        cells = {(r+i, c+j) for i in range(2) for j in range(2)}
        _require(not all_symbols & cells, 'swatches overlap')
        all_symbols |= cells
        covered |= cells
    legend_row = bare[0]['core'][0]
    for s in frames:
        footprint = _footprint(s['core'])
        _require(min(r for r, _ in footprint) >= legend_row+3,
                 'all frames must be below the legend')
        _require(not all_symbols & footprint, 'symbol footprints overlap')
        _require(not any(_add(p, d) in frame_cells for p in footprint
                         for d in ((0, 1), (0, -1), (1, 0), (-1, 0))),
                 'frames need a cardinal background margin')
        all_symbols |= footprint
        frame_cells |= footprint
        covered |= {p for p in footprint if grid[p] != background}
        if s is marker:
            continue
        _require(s['color'] == word[0], 'every emitting core uses the first word color')
        ports = [(p, PORTS[offset]) for p, v, offset in s['ring']
                 if v == background and offset in PORTS]
        _require(len(ports) == sum(v == background for _, v, _ in s['ring']),
                 'only side gaps emit; frame corners must remain structural')
        _require(1 <= len(ports) <= 8, 'emitting frame needs a side gap')
        s['ports'] = ports
        owner = len({x['owner'] for x in streams})
        for p, direction in sorted(ports):
            streams.append(dict(owner=owner, origin=p, direction=direction))
    foreground = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    _require(covered == foreground, 'input has unaccounted foreground')
    annotations = {(s['core'][0]+i, s['core'][1]+j)
                   for s in bare for i in range(2) for j in range(2)}
    annotations |= _footprint(marker['core'])
    restored = grid.copy()
    for p in annotations:
        restored[p] = background
    return dict(grid=grid, background=background, structure=structure,
                word=word, marker=marker['color'], annotations=annotations,
                emitters=emitters, streams=streams, restored=restored,
                legend_origins=tuple(s['core'] for s in bare))


def _trace(info):
    """Spatial, mutually first intersections; each stream encounters at most once.

    Outgoing diagonals are terminal continuations in this family's domain.
    Painted path validation excludes subsequent crossings, overlaps and hits.
    """
    streams = info['streams']
    shape = info['grid'].shape
    word = info['word']
    paths = [_line(s['origin'], s['direction'], shape) for s in streams]
    lookup = [{p: k for k, p in enumerate(path)} for path in paths]
    active = set(range(len(streams)))
    events, records = [], {}
    while active:
        candidates = []
        nearest = {i: 1000 for i in active}
        for i in sorted(active):
            for j in sorted(active):
                if j <= i:
                    continue
                shared = set(lookup[i]) & set(lookup[j])
                if not shared:
                    continue
                u, v = streams[i]['direction'], streams[j]['direction']
                _require(u[0]*v[0]+u[1]*v[1] == 0 and len(shared) == 1,
                         'overlap or head-on encounter is outside the family')
                point = next(iter(shared))
                a, b = lookup[i][point], lookup[j][point]
                _require(a > 0 and b > 0, 'meeting cannot occur at a port')
                candidates.append((i, j, a, b, point))
                nearest[i] = min(nearest[i], a)
                nearest[j] = min(nearest[j], b)
        if not candidates:
            for i in active:
                records[i] = dict(cells=tuple(paths[i]), bend=None, event=None)
            break
        ready = [e for e in candidates if e[2] == nearest[e[0]] and e[3] == nearest[e[1]]]
        _require(bool(ready), 'cyclic encounter order is outside the family')
        # A tied meeting on any participating stream would be a multiway event.
        for i in active:
            _require(sum((e[0] == i and e[2] == nearest[i])
                         or (e[1] == i and e[3] == nearest[i]) for e in candidates) <= 1,
                     'tied or triple encounters are outside the family')
        i, j, a, b, point = min(ready, key=lambda e: (max(e[2:4]), sum(e[2:4]), e[4]))
        matched = word[a % len(word)] == word[b % len(word)]
        direction = _add(streams[i]['direction'], streams[j]['direction'])
        event = dict(streams=(i, j), distances=(a, b), point=point,
                     matched=matched, outgoing=direction,
                     owners=(streams[i]['owner'], streams[j]['owner']))
        event_index = len(events)
        events.append(event)
        for stream_id, distance in ((i, a), (j, b)):
            cells = list(paths[stream_id][:distance])
            bend = None
            if matched:
                bend = len(cells)
                cells += _line(_add(cells[-1], direction), direction, shape)
            records[stream_id] = dict(cells=tuple(cells), bend=bend, event=event_index)
        active -= {i, j}
    output = info['restored'].copy()
    occupied, diagonal_segments = {}, {}
    for i in sorted(records):
        record = records[i]
        for phase, p in enumerate(record['cells']):
            _require(info['restored'][p] == info['background'], 'stream hits an emitting frame')
            _require(p not in occupied, 'streams overlap or have a second encounter')
            occupied[p] = i
            output[p] = word[phase % len(word)]
        for p, q in zip(record['cells'], record['cells'][1:]):
            if p[0] != q[0] and p[1] != q[1]:
                midpoint = _add(p, q)
                slope = int(np.sign((q[0]-p[0])*(q[1]-p[1])))
                _require((midpoint, -slope) not in diagonal_segments,
                         'outgoing strands cross between cells')
                diagonal_segments[midpoint, slope] = i
    for e in events:
        _require(e['point'] not in occupied, 'a junction is reused by another stream')
        _require(info['restored'][e['point']] == info['background'], 'junction hits a frame')
        if not e['matched']:
            output[e['point']] = info['marker']
            occupied[e['point']] = -1
    info = dict(info, records=records, events=events, occupied=occupied)
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique output using the input's swatches, frames and gaps only."""
    return _trace(_decode(input_grid))[0]


def _pattern(word):
    seen = {}
    return tuple(seen.setdefault(c, len(seen)) for c in word)


def _coverage(info):
    word, k = info['word'], len(info['word'])
    tags = set()
    unique = tuple(dict.fromkeys(word))
    global_offset = 0
    for stream_id in sorted(info['records']):
        record = info['records'][stream_id]
        length = len(record['cells'])
        expected = [word[t % k] for t in range(length)]
        if length >= 2*k:
            tags.add('cycling')
        if expected != [unique[t % len(unique)] for t in range(length)]:
            tags.add('multiplicity')
        if expected != [word[0]]*length:
            tags.add('cyclic_paint')
        if expected != [word[(global_offset+t) % k] for t in range(length)]:
            tags.add('independent_restart')
        global_offset += length
        bend = record['bend']
        if bend is not None and bend < length:
            tags.add('bend')
            if expected[bend:] != [word[t % k] for t in range(length-bend)]:
                tags.add('phase_continuity')
    for e in info['events']:
        a, b = e['distances']
        tags.add('open' if e['matched'] else 'closed')
        if e['matched'] and a % k != b % k:
            tags.add('color_not_index')
        if e['matched'] and a != b:
            tags.add('spatial_not_clock')
        if e['matched'] != (word[(a-1) % k] == word[(b-1) % k]):
            tags.add('prospective')
    return tags


REQUIRED = {'cycling', 'multiplicity', 'cyclic_paint', 'independent_restart',
            'phase_continuity', 'open', 'closed', 'color_not_index',
            'spatial_not_clock', 'prospective'}


def _word(rng, length, witness):
    """Sample words freely, reserving branchable repetitions for witnesses."""
    for _ in range(200):
        distinct = int(rng.integers(2, min(4, length)+1))
        raw = list(range(distinct)) + list(map(int, rng.integers(distinct, size=length-distinct)))
        rng.shuffle(raw)
        # Internal symbols 0/1/8 are not word colors; 8 is the marker role.
        word = tuple(2+x for x in raw)
        if witness in ('open', 'closed'):
            options = _distances(word, witness)
            if not options:
                continue
        return word
    raise RuntimeError('word construction exhausted')


def _distances(word, witness):
    k = len(word)
    choices = []
    for a in range(3, 12):
        for b in range(3, 12):
            matched = word[a % k] == word[b % k]
            last_matched = word[(a-1) % k] == word[(b-1) % k]
            if witness == 'open':
                keep = matched and a % k != b % k and not last_matched
            elif witness == 'closed':
                keep = not matched and last_matched
            else:
                keep = matched == (witness == 'match')
            if keep:
                choices.append((a, b))
    return choices


def _build(shape, specs, word):
    grid = np.zeros(shape, dtype=np.int8)
    occupied = set()
    for spec in specs:
        core, ports = spec['core'], set(spec['ports'])
        footprint = _footprint(core)
        _require(all(_inside(p, shape) for p in footprint), 'frame does not fit')
        _require(not occupied & footprint, 'frame footprints overlap')
        # A background margin makes close repeated-color objects easy to segment.
        _require(not any(_add(p, d) in occupied for p in footprint
                         for d in ((0, 1), (0, -1), (1, 0), (-1, 0))),
                 'frames touch')
        occupied |= footprint
        r, c = core
        grid[r:r+2, c:c+2] = word[0]
        for offset in RING:
            if offset not in ports:
                grid[r+offset[0], c+offset[1]] = 1
    return grid


def _scene_info(grid, specs, word):
    streams = []
    for i, spec in enumerate(sorted(specs, key=lambda s: s['core'])):
        for offset in sorted(spec['ports']):
            streams.append(dict(owner=i, origin=_add(spec['core'], offset), direction=PORTS[offset]))
    return dict(grid=grid, restored=grid, background=0, structure=1,
                word=word, marker=8, streams=streams, emitters=specs,
                annotations=set(), legend_origins=())


def _trial(shape, specs, word):
    grid = _build(shape, specs, word)
    out, info = _trace(_scene_info(grid, specs, word))
    return grid, out, info


def _primary(rng, word, witness):
    choices = _distances(word, witness)
    _require(bool(choices), 'word cannot expose this witness')
    for _ in range(100):
        a, b = choices[int(rng.integers(len(choices)))]
        # Long outgoing runs make differing cursors and continuation visible.
        k = len(word)
        row = int(rng.integers(k+1, min(10, 21-b)+1))
        h = int(rng.integers(max(18, row+b+4), 26))
        w = int(rng.integers(max(18, a+k+5), 26))
        col = int(rng.integers(a+3, w-k-1))
        s, t = int(rng.integers(2)), int(rng.integers(2))
        specs = [dict(core=(row-s, col-a-2), ports=((s, 2),)),
                 dict(core=(row+b+1, col-t), ports=((-1, t),))]
        try:
            grid, out, info = _trial((h, w), specs, word)
            if len(info['events']) != 1:
                continue
            if witness == 'open' and 'phase_continuity' not in _coverage(info):
                continue
            return grid, out, info, specs
        except ValueError:
            continue
    raise RuntimeError('primary scene construction exhausted')


def _extend(rng, grid, out, info, specs, word, extra_count):
    shape = grid.shape
    # Ports are sampled in any cardinal side, subject to geometric validity.
    for _ in range(int(rng.integers(2, 9))):
        owner = int(rng.integers(len(specs)))
        options = [x for x in PORTS if x not in specs[owner]['ports']]
        if not options:
            continue
        offset = options[int(rng.integers(len(options)))]
        candidate = [dict(s) for s in specs]
        candidate[owner]['ports'] += (offset,)
        try:
            g, o, metrics = _trial(shape, candidate, word)
            # Keep the explicit informative primary event after adding ports.
            if not _coverage(info) <= _coverage(metrics):
                continue
            grid, out, info, specs = g, o, metrics, candidate
        except ValueError:
            pass
    for _ in range(extra_count):
        prefer_connected = rng.random() < .65
        found = None
        for attempt in range(65):
            r = int(rng.integers(1, shape[0]-2))
            c = int(rng.integers(1, shape[1]-2))
            nports = int(rng.integers(1, 4))
            keys = list(PORTS)
            selected = rng.choice(len(keys), nports, replace=False)
            new = dict(core=(r, c), ports=tuple(keys[int(j)] for j in selected))
            candidate = specs + [new]
            try:
                g, o, metrics = _trial(shape, candidate, word)
                if not _coverage(info) <= _coverage(metrics):
                    continue
                if prefer_connected and attempt < 45 and len(metrics['events']) <= len(info['events']):
                    continue
                found = g, o, metrics, candidate
                break
            except ValueError:
                pass
        if found is not None:
            grid, out, info, specs = found
    return grid, out, info, specs


def _annotate(rng, scene, scene_output, word):
    h, w = scene.shape
    header = int(rng.integers(4, 6))
    grid = np.zeros((h+header, w), dtype=np.int8)
    grid[header:] = scene
    # Marker annotation is a closed frame, geometrically distinct from emitters.
    candidates = []
    for r in range(1, h-2):
        for c in range(1, w-2):
            block = scene[r-1:r+3, c-1:c+3]
            border = scene[max(0, r-2):min(h, r+4), max(0, c-2):min(w, c+4)]
            if np.all(block == 0) and np.all(border == 0) and np.all(scene_output[r-1:r+3, c-1:c+3] == 0):
                candidates.append((r+header, c))
    _require(bool(candidates), 'no clear place for the marker annotation')
    mr, mc = candidates[int(rng.integers(len(candidates)))]
    grid[mr-1:mr+3, mc-1:mc+3] = 1
    grid[mr:mr+2, mc:mc+2] = 8
    row = int(rng.integers(1, header-2))
    slack = w-(3*len(word)-1)-2
    _require(slack >= 0, 'legend does not fit')
    left = int(rng.integers(1, min(4, slack+1)+1))
    remaining = w-1-left-(3*len(word)-1)
    for j, color in enumerate(word):
        grid[row:row+2, left:left+2] = color
        if j < len(word)-1:
            extra = int(rng.integers(0, min(2, remaining)+1))
            remaining -= extra
            left += 3+extra
    # Cyclic streams must extend into the erased header, not stop at its seam.
    return grid


def _make_pair(rng, length, witness, rotation, extras):
    for _ in range(80):
        word = _word(rng, length, witness)
        scene, out, metrics, specs = _primary(rng, word, witness)
        scene, out, metrics, specs = _extend(rng, scene, out, metrics, specs, word, extras)
        scene, out = np.rot90(scene, rotation), np.rot90(out, rotation)
        if rng.random() < .5:
            scene, out = scene[:, ::-1], out[:, ::-1]
        try:
            grid = _annotate(rng, scene, out, word)
            palette = rng.permutation(10).astype(np.int8)
            grid = palette[grid]
            expected, info = _trace(_decode(grid))
            # Annotations disappear before tracing; header adds real space.
            # Check final-grid coverage, never just construction metadata.
            tags = _coverage(info)
            if witness == 'open' and not {'open', 'color_not_index', 'prospective',
                                         'phase_continuity', 'multiplicity',
                                         'cycling', 'spatial_not_clock',
                                         'independent_restart'} <= tags:
                continue
            if witness == 'closed' and not {'closed', 'prospective'} <= tags:
                continue
            _require(not set(info['occupied']) & info['annotations'],
                     'annotations must be clear of all final stream paths')
            return dict(input=grid, output=expected)
        except ValueError:
            continue
    raise RuntimeError('annotated pair construction exhausted')


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed), 'seed must be an integer')
    _require(_integer(num_train) and _integer(num_test), 'pair counts must be integers')
    _require(2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test,
             'require 2..10 train, 1..4 test, and num_train > num_test')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    schedule = str(rng.choice(('longer_word', 'rotated_scene')))
    base_rotation = int(rng.integers(4))
    # A two-example request still teaches both resonance outcomes.
    witnesses = ['open', 'closed'] + [str(rng.choice(('match', 'mismatch')))
                                     for _ in range(num_train-2)]
    rng.shuffle(witnesses)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            witness = witnesses[i] if split == 'train' else str(rng.choice(('match', 'mismatch')))
            if schedule == 'longer_word':
                length = int(rng.integers(3, 5)) if split == 'train' else int(rng.integers(5, 7))
                rotation = int(rng.integers(4))
            else:
                length = int(rng.integers(3, 7))
                rotation = base_rotation if split == 'train' else (base_rotation+2) % 4
            extras = int(rng.choice((0, 1, 2), p=(.48, .36, .16)))
            for _ in range(12):
                pair = _make_pair(rng, length, witness, rotation, extras)
                if _key(pair['input']) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    break
            else:
                raise RuntimeError('unique pair construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal pair counts')
    seen, evidence = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'dimensions changed')
            _require(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            expected, info = _trace(_decode(a))
            _require(np.array_equal(expected, b), 'output contradicts the resonance rule')
            _require(not np.array_equal(a, b), 'identity pair')
            _require(not set(info['occupied']) & info['annotations'], 'annotation intersects a stream')
            # Preservation and annotation removal are separate semantic checks.
            emitter_cells = set().union(*(_footprint(s['core']) for s in info['emitters']))
            fixed = {p for p in emitter_cells if a[p] != info['background']}
            _require(all(a[p] == b[p] for p in fixed), 'emitter core or structure changed')
            _require(all(b[p] == info['background'] for p in info['annotations']), 'annotation not erased')
            untouched = np.ones(a.shape, dtype=bool)
            for p in set(info['occupied']) | info['annotations']:
                untouched[p] = False
            _require(np.array_equal(a[untouched], b[untouched]), 'untouched cells changed')
            for e in info['events']:
                if e['matched']:
                    _require(b[e['point']] == info['background'], 'open junction must remain empty')
                else:
                    _require(b[e['point']] == info['marker'], 'closed junction needs the marker')
            if split == 'train':
                evidence |= _coverage(info)
    _require(REQUIRED <= evidence, 'training misses distinctions: '+str(sorted(REQUIRED-evidence)))


def _fixture(word, distance_a=4, distance_b=5):
    """Complete hand-specified input for a shared-cell E/N confluence."""
    row, col = 7, 10
    specs = [dict(core=(row, col-distance_a-2), ports=((0, 2),)),
             dict(core=(row+distance_b+1, col), ports=((-1, 0),))]
    scene = _build((22, 23), specs, word)
    grid = np.zeros((26, 23), dtype=np.int8)
    grid[4:] = scene
    for i, color in enumerate(word):
        grid[1:3, 1+3*i:3+3*i] = color
    grid[21:25, 17:21] = 1
    grid[22:24, 18:20] = 8
    output, info = _trace(_decode(grid))
    return grid, output, info


def self_test():
    # Direct semantic regression: repeated colors, not equal indices, open it.
    word = (2, 2, 3)
    a, b, info = _fixture(word, 3, 4)
    event = info['events'][0]
    assert event['matched'] and event['distances'] == (3, 4)
    assert b[11, 10] == 0
    assert b[10, 10] == 2 and b[11, 11] == 2
    assert b[9, 11] == 2 and b[10, 12] == 3
    assert np.array_equal(solve(a), b)
    # Same geometry, changed word, changes whether the output continues.
    other_input, stopped, stopped_info = _fixture((2, 3, 2), 3, 4)
    assert not stopped_info['events'][0]['matched']
    assert stopped[11, 10] == 8 and stopped[10, 10] == 0
    assert np.array_equal(a[4:], other_input[4:])
    assert not np.array_equal(b, stopped)
    # Same word, change a port's approach distance: the other inherited concept.
    _, shifted, shifted_info = _fixture(word, 4, 5)
    assert not shifted_info['events'][0]['matched']
    assert shifted[11, 10] == 8
    # Valid members beyond the sampler's narrower 2..4-source/3..6-entry range.
    # One emitter can expose all eight side slots, without any confluence.
    isolated_specs = [dict(core=(7, 7), ports=tuple(PORTS))]
    isolated = _build((16, 20), isolated_specs, (2, 3))
    isolated[1:3, 1:3] = 2
    isolated[1:3, 4:6] = 3
    isolated[10:14, 12:16] = 1
    isolated[11:13, 13:15] = 8
    isolated_out = solve(isolated)
    assert isolated_out[6, 7] == 2 and isolated_out[5, 7] == 3
    assert isolated_out[7, 9] == 2 and isolated_out[7, 10] == 3
    assert np.all(isolated_out[10:14, 12:16] == 0)
    # Five separated emitters: no interaction, and literal annotation erasure.
    five_specs = [dict(core=(6, 3), ports=((0, -1),)),
                  dict(core=(10, 13), ports=((-1, 0),)),
                  dict(core=(14, 25), ports=((1, 2),)),
                  dict(core=(20, 3), ports=((1, -1),)),
                  dict(core=(24, 22), ports=((2, 0),))]
    five = _build((28, 30), five_specs, (2, 3))
    five[1:3, 1:3] = 2
    five[1:3, 4:6] = 3
    five[19:23, 11:15] = 1
    five[20:22, 12:14] = 8
    five_out = solve(five)
    assert five_out[0, 13] == 3 and five_out[27, 22] == 3
    assert len(_decode(five)['emitters']) == 5
    puzzles, features = [], []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _, data = _trace(_decode(pair['input']))
                features.append(data)
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(HOST_SEEDS[0]))
    assert len({f['grid'].shape for f in features}) >= 35
    assert {len(f['word']) for f in features} >= {3, 4, 5, 6}
    assert len({_pattern(f['word']) for f in features}) >= 20
    assert {len(f['emitters']) for f in features} >= {2, 3, 4}
    assert len({tuple(s['direction'] for s in f['streams']) for f in features}) >= 30
    assert len({e['outgoing'] for f in features for e in f['events']}) == 4
    assert len({(f['background'], f['structure'], f['marker']) for f in features}) >= 100
    assert any(len(f['events']) > 1 for f in features)
    assert any(any(len(s['ports']) > 1 for s in f['emitters']) for f in features)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(991+nt*10+ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output was accepted')
    return None


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, semantic fixtures, and all legal counts)')
