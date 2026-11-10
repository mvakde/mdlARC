"""Rendezvous relays: two keys unlock a frame; two vents vote on a heading.

No files, examples, random state, or construction metadata are used by solve.
The only dependency beyond the standard library is NumPy.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
# Coordinates within a complete 4-by-4 relay envelope.
PORTS = tuple((r, c) for r in range(4) for c in range(4)
              if r in (0, 3) or c in (0, 3))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _normal(port):
    r, c = port
    return (-1 if r == 0 else 1 if r == 3 else 0,
            -1 if c == 0 else 1 if c == 3 else 0)


def _heading(a, b):
    r, c = a[0] + b[0], a[1] + b[1]
    divisor = int(np.gcd(abs(r), abs(c)))
    _require(divisor > 0, 'opposite vents have no resultant')
    return r // divisor, c // divisor


def _kind(ports):
    a, b = map(_normal, ports)
    if a == b:
        return 'straight'
    if sum(map(abs, a)) == sum(map(abs, b)) == 1:
        return 'diagonal'
    if sum(map(abs, a)) == sum(map(abs, b)) == 2:
        return 'cardinal'
    return 'staircase'


VENT_PAIRS = tuple(pair for pair in combinations(PORTS, 2)
                   if sum(x*y for x, y in zip(*map(_normal, pair))) >= 0)
BY_KIND = {kind: tuple(p for p in VENT_PAIRS if _kind(p) == kind)
           for kind in ('straight', 'diagonal', 'cardinal', 'staircase')}


def _box(position):
    r, c = position
    return {(r+i, c+j) for i in range(4) for j in range(4)}


def _separate(a, b):
    return not any((r+dr, c+dc) in b for r, c in a
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _components(cells):
    left, result = set(cells), []
    while left:
        first = min(left)
        left.remove(first)
        group, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r+dr, c+dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must be integral')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'invalid colors')
    return a.astype(np.int8, copy=False)


def _decode(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    bg = int(a[0, 0])
    _require(all(int(a[r, c]) == bg for r, c in ((0,w-1),(h-1,0),(h-1,w-1))),
             'corners must be background')
    foreground = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    groups = _components(foreground)
    tails = [g for g in groups if any(r in (0,h-1) or c in (0,w-1) for r,c in g)]
    _require(len(tails) == 2, 'exactly two boundary keys required')
    sources, colors = [], []
    for group in tails:
        boundary = [(r,c) for r,c in group if r in (0,h-1) or c in (0,w-1)]
        _require(len(boundary) == 1 and 3 <= len(group) <= 5, 'bad boundary key')
        r, c = boundary[0]
        direction = (1,0) if r == 0 else (-1,0) if r == h-1 else (0,1) if c == 0 else (0,-1)
        line = [(r+k*direction[0],c+k*direction[1]) for k in range(len(group))]
        _require(set(line) == group, 'key is not straight')
        signal, cap = int(a[line[0]]), int(a[line[-1]])
        _require(signal != cap and all(int(a[p]) == signal for p in line[:-1]), 'bad key colors')
        colors.append((signal,cap))
        sources.append({'origin':line[-1], 'direction':direction, 'cells':group})
    _require(colors[0] == colors[1], 'keys must share signal and cap roles')
    signal, cap = colors[0]
    source_cells = set().union(*(s['cells'] for s in sources))
    relays, covered, palettes = [], set(source_cells), set()
    for r in range(1,h-4):
        for c in range(1,w-4):
            core = int(a[r+1,c+1])
            if core in (bg,signal,cap) or not np.all(a[r+1:r+3,c+1:c+3] == core):
                continue
            ring = [int(a[r+i,c+j]) for i,j in PORTS]
            nonbg = set(ring) - {bg}
            if len(nonbg) != 1 or ring.count(bg) != 2:
                continue
            frame = next(iter(nonbg))
            if len({bg,signal,cap,core,frame}) != 5:
                continue
            vents = tuple(p for p in PORTS if int(a[r+p[0],c+p[1]]) == bg)
            if vents not in VENT_PAIRS:
                continue
            cells = _box((r,c))
            _require(not any(cells & o['cells'] for o in relays), 'overlapping relay parse')
            relays.append({'position':(r,c), 'ports':vents, 'cells':cells})
            covered |= {q for q in cells if int(a[q]) != bg}
            palettes.add((core,frame))
    _require(1 <= len(relays) <= 7 and len(palettes) == 1, 'invalid relay vocabulary')
    _require(covered == foreground, 'unparsed foreground')
    occupied = set(source_cells)
    for relay in relays:
        _require(_separate(relay['cells'],occupied), 'objects need a background gap')
        occupied |= relay['cells']
    _require(np.count_nonzero(a == bg) > a.size // 2, 'background must be a majority')
    return a, bg, signal, cap, sources, relays, next(iter(palettes))


def _walk(origin, direction, shape, phase='floor'):
    """Major-axis-first rational raster, reset at each relay's vents."""
    dr, dc = direction
    scale = max(abs(dr),abs(dc))
    offset = scale-1 if phase == 'ceil' else 0
    for tick in range(1, 3*max(shape)+2):
        r = origin[0] + (1 if dr >= 0 else -1) * ((tick*abs(dr)+offset)//scale)
        c = origin[1] + (1 if dc >= 0 else -1) * ((tick*abs(dc)+offset)//scale)
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            return
        yield r,c


def _ray(origin, direction, shape, owner, source_cells, phase='floor'):
    points = []
    for p in _walk(origin,direction,shape,phase):
        if p in owner:
            return points,owner[p]
        _require(p not in source_cells, 'a ray returns into a boundary key')
        points.append(p)
    return points,None


def _execute(input_grid, *, law='coupled', one_key=False, phase='floor'):
    a,bg,signal,cap,sources,relays,palette = _decode(input_grid)
    owner = {p:i for i,o in enumerate(relays) for p in o['cells']}
    source_cells = set().union(*(s['cells'] for s in sources))
    out = np.full_like(a,bg)
    for p in source_cells:
        out[p] = signal
    rays = [(s['origin'],s['direction']) for s in sources]
    reached, segments, events, terminal = [], [], [], []
    for _ in range(len(relays)+1):
        contacts, paths = [], []
        for origin,direction in rays:
            path,hit = _ray(origin,direction,a.shape,owner,source_cells,phase)
            paths.append(path)
            contacts.append(hit)
            for p in path:
                out[p] = signal
        segments.append(paths)
        matches = contacts[0] is not None and contacts[0] == contacts[1]
        if one_key and len(set(contacts)-{None}) == 1:
            matches = True
        if not matches:
            terminal = contacts
            break
        index = next(x for x in contacts if x is not None)
        if index in reached:
            if law != 'coupled' or one_key or phase != 'floor':
                break  # Explicit cycle-termination convention for diagnostic alternatives only.
            raise ValueError('relay cycle is outside the family')
        reached.append(index)
        relay = relays[index]
        for p in relay['cells']:
            out[p] = signal
        normals = tuple(map(_normal,relay['ports']))
        resultant = _heading(*normals)
        if law == 'sign':
            resultant = tuple(int(np.sign(x)) for x in resultant)
        r,c = relay['position']
        rays = [((r+p[0],c+p[1]), normals[i] if law == 'independent' else resultant)
                for i,p in enumerate(relay['ports'])]
        events.append({'index':index, 'kind':_kind(relay['ports']), 'heading':resultant,
                       'distances':tuple(len(p) for p in paths)})
    else:
        raise ValueError('unterminated routing')
    # Inactive envelopes vanish, including any gaps; paths stop outside envelopes.
    for i,relay in enumerate(relays):
        if i not in reached:
            for p in relay['cells']:
                out[p] = bg
    info = {'shape':a.shape, 'active':tuple(reached), 'events':events,
            'terminal':tuple(terminal), 'relay_count':len(relays),
            'inactive':len(relays)-len(reached), 'palette':(bg,signal,cap,*palette),
            'sources':sources, 'relays':relays, 'segments':segments}
    return out,info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete output from this input alone."""
    return _execute(input_grid)[0]


def _tail(shape, position, side, lane, length):
    h,w = shape
    r,c = position
    if side == 0:
        start, direction = (0,c+lane),(1,0)
    elif side == 2:
        start, direction = (h-1,c+lane),(-1,0)
    elif side == 3:
        start, direction = (r+lane,0),(0,1)
    else:
        start, direction = (r+lane,w-1),(0,-1)
    line = [(start[0]+k*direction[0],start[1]+k*direction[1]) for k in range(length+1)]
    return {'origin':line[-1], 'direction':direction, 'line':line, 'cells':set(line)}


def _pair_paths(shape, position, ports, occupied, source_cells):
    r,c = position
    heading = _heading(*map(_normal,ports))
    result = []
    for p in ports:
        origin = (r+p[0],c+p[1])
        path,hit = _ray(origin,heading,shape,occupied,source_cells)
        result.append((path,hit))
    return result


def _placements(path, shape):
    h,w = shape
    return {(r-dr,c-dc) for r,c in path for dr in range(4) for dc in range(4)
            if 1 <= r-dr <= h-5 and 1 <= c-dc <= w-5}


def _render(shape, sources, relays):
    a = np.zeros(shape,dtype=np.int8)
    for o in relays:
        r,c = o['position']
        a[r:r+4,c:c+4] = 4
        a[r+1:r+3,c+1:c+3] = 5
        for i,j in o['ports']:
            a[r+i,c+j] = 0
    for s in sources:
        for p in s['line'][:-1]:
            a[p] = 2
        a[s['origin']] = 3
    return a


def _construct(rng, sequence, extra_decoys):
    """Bounded path-aware backtracking; never lower the requested chain length."""
    n = len(sequence)
    for attempt in range(28):
        low = 19 if n < 3 else 24
        shape = tuple(map(int,rng.integers(low,31,size=2)))
        h,w = shape
        first = (int(rng.integers(5,h-8)),int(rng.integers(5,w-8)))
        sources = []
        for _ in range(2):
            choices = []
            for side in range(4):
                for lane in range(4):
                    s = _tail(shape,first,side,lane,int(rng.integers(2,5)))
                    existing = set().union(*(x['cells'] for x in sources)) if sources else set()
                    if (_separate(s['cells'],existing) and _separate(s['cells'],_box(first))):
                        choices.append(s)
            if not choices:
                break
            sources.append(choices[int(rng.integers(len(choices)))])
        if len(sources) != 2:
            continue
        source_cells = set().union(*(s['cells'] for s in sources))
        first_owner = {p:0 for p in _box(first)}
        reserved = set()
        for s in sources:
            path,hit = _ray(s['origin'],s['direction'],shape,first_owner,source_cells)
            if hit != 0:
                break
            reserved.update(path)
        else:
            budget = [500]

            def grow(positions, relays, reserved_cells):
                budget[0] -= 1
                if budget[0] < 0:
                    return None
                current = positions[-1]
                choices = list(BY_KIND[sequence[len(positions)-1]])
                rng.shuffle(choices)
                occupied_cells = set().union(*(_box(p) for p in positions))
                previous = {p:i for i,pos in enumerate(positions[:-1]) for p in _box(pos)}
                for ports in choices[:20]:
                    try:
                        paths = _pair_paths(shape,current,ports,previous,source_cells)
                    except ValueError:
                        continue
                    if any(hit is not None for _,hit in paths):
                        # These paths may still accept a nearer new relay. Keep the
                        # prefix, which ends at the old object, for placement search.
                        pass
                    current_obj = {'position':current,'ports':ports}
                    next_relays = relays+[current_obj]
                    path_sets = [set(p) for p,_ in paths]
                    if len(positions) == n:
                        if any(hit is not None for _,hit in paths):
                            continue
                        # A terminal gate lies on exactly one outgoing strand. The
                        # other strand reaches the edge: visible evidence for AND.
                        candidates = sorted(_placements(paths[0][0],shape) ^ _placements(paths[1][0],shape))
                        rng.shuffle(candidates)
                        for pos in candidates[:160]:
                            box = _box(pos)
                            if (not _separate(box,occupied_cells|source_cells)
                                    or box & reserved_cells):
                                continue
                            decoy = {'position':pos, 'ports':VENT_PAIRS[int(rng.integers(len(VENT_PAIRS)))]}
                            all_relays = next_relays+[decoy]
                            protected = reserved_cells|path_sets[0]|path_sets[1]
                            all_cells = occupied_cells|box|source_cells
                            for _ in range(extra_decoys):
                                possible = [(rr,cc) for rr in range(1,h-4) for cc in range(1,w-4)]
                                rng.shuffle(possible)
                                for other in possible[:500]:
                                    b = _box(other)
                                    if not b & protected and _separate(b,all_cells):
                                        all_relays.append({'position':other,'ports':VENT_PAIRS[int(rng.integers(len(VENT_PAIRS)))]})
                                        all_cells |= b
                                        break
                            a = _render(shape,sources,all_relays)
                            try:
                                out,info = _execute(a)
                            except ValueError:
                                continue
                            if len(info['active']) == n and sum(x is not None for x in info['terminal']) == 1:
                                return a,out
                        continue
                    possible = sorted(_placements(paths[0][0],shape) & _placements(paths[1][0],shape))
                    rng.shuffle(possible)
                    for pos in possible[:70]:
                        box = _box(pos)
                        if not _separate(box,occupied_cells|source_cells) or box & reserved_cells:
                            continue
                        prefixes = []
                        for path,_ in paths:
                            cut = next((i for i,p in enumerate(path) if p in box),None)
                            if cut is None or cut < 1:
                                break
                            prefixes.extend(path[:cut])
                        else:
                            result = grow(positions+[pos],next_relays,reserved_cells|set(prefixes))
                            if result is not None:
                                return result
                        if budget[0] < 0:
                            return None
                return None

            result = grow([first],[],reserved)
            if result is not None:
                return result
    raise RuntimeError('bounded geometry search exhausted without changing requested complexity')


def _counts(value, low, high, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(grid):
    a = np.asarray(grid,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _counts(num_train,2,10,'num_train'),_counts(num_test,1,4,'num_test')
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    companion = str(rng.choice(('straight','diagonal','cardinal')))
    kinds = ('staircase',companion)
    palette = np.asarray(rng.permutation(10),dtype=np.int8)
    vary_palette = bool(rng.integers(2))
    train_sequences = [['staircase'],[companion,'staircase']]
    for _ in range(nt-2):
        train_sequences.append([str(rng.choice(kinds)) for _ in range(int(rng.integers(1,3)))])
    rng.shuffle(train_sequences)
    sequences = {'train':train_sequences,'test':[]}
    for i in range(ne):
        length = 3 if i == 0 else int(rng.integers(1,4))
        seq = [str(rng.choice(kinds)) for _ in range(length)]
        if length >= 2:
            seq[0],seq[-1] = companion,'staircase'
        sequences['test'].append(seq)
    puzzle,seen = {'train':[],'test':[]},set()
    for split in ('train','test'):
        for sequence in sequences[split]:
            for _ in range(12):
                a,b = _construct(rng,sequence,int(rng.integers(3)))
                # Pair-local transformations change geometry, never conventions.
                rotation = int(rng.integers(4))
                a,b = np.rot90(a,rotation),np.rot90(b,rotation)
                if bool(rng.integers(2)):
                    a,b = np.fliplr(a),np.fliplr(b)
                roles = np.asarray(rng.permutation(10),dtype=np.int8) if vary_palette else palette
                a,b = np.ascontiguousarray(roles[a]),np.ascontiguousarray(roles[b])
                if _key(a) in seen:
                    continue
                # Check final transformed evidence rather than construction tags.
                if not np.array_equal(solve(a),b):
                    raise AssertionError('a geometric transform changed the rule')
                puzzle[split].append({'input':a,'output':b})
                seen.add(_key(a))
                break
            else:
                raise RuntimeError('distinct input search exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_diff(a,b,**kwargs):
    try:
        return not np.array_equal(_execute(a,**kwargs)[0],b)
    except ValueError:
        # A shortcut that routes back into a key is also ruled out by the label.
        return True


def _without_reachability(input_grid):
    """Ablate A's all-or-none reachability filter; retain the routed traces."""
    out,info = _execute(input_grid)
    for relay in info['relays']:
        for p in relay['cells']:
            out[p] = info['palette'][1]
    return out


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'},'bad puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in puzzle),'splits must be lists')
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne,'bad pair counts')
    seen, facts = set(),{'train':[],'test':[]}
    taught = set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'},'bad pair keys')
            a,b = _grid(pair['input']),_grid(pair['output'])
            _require(a.shape == b.shape,'output resized')
            _require(_key(a) not in seen,'duplicate input')
            seen.add(_key(a))
            out,f = _execute(a)
            _require(np.array_equal(out,b),'incorrect output')
            bg,signal = f['palette'][:2]
            _require(set(map(int,np.unique(b))) == {bg,signal},'output must contain only signal/background')
            _require(len(f['active']) >= 1 and f['inactive'] >= 1,'need activation and erasure')
            _require(sum(x is not None for x in f['terminal']) == 1,'need a single-key failed relay')
            for i,o in enumerate(f['relays']):
                expected = signal if i in f['active'] else bg
                _require(all(int(b[p]) == expected for p in o['cells']),'relay activation is not all-or-none')
            _require(_alternative_diff(a,b,one_key=True),'two-key rule is not observable')
            _require(not np.array_equal(_without_reachability(a),b),
                     'parent A reachability filter is not observable')
            local = {e['kind'] for e in f['events']}
            if split == 'train':
                taught |= local
            else:
                _require(local <= taught,'test uses an untaught heading class')
            facts[split].append(f)
    _require('staircase' in taught and len(taught) >= 2,'training needs staircase plus simpler vent sum')
    for split in ('train','test'):
        _require(any(_alternative_diff(p['input'],p['output'],law='independent') for p in puzzle[split]),
                 'vent coupling is not observable in this split')
    _require(any(len(f['active']) >= 2 for f in facts['train']),'training must show relay chaining')
    _require(any(len(f['active']) > max(len(t['active']) for t in facts['train'])
                 for f in facts['test']),'missing chain-length generalization')
    _require(any(_alternative_diff(p['input'],p['output'],law='sign') for p in puzzle['train']),
             'training does not distinguish staircases from diagonals')
    _require(any(_alternative_diff(p['input'],p['output'],phase='ceil') for p in puzzle['train']),
             'training does not expose staircase phase')


def _rule_fixtures():
    """Hand-specified expected cells, independent of generated labels."""
    shape = (20,22)
    sources = [_tail(shape,(8,8),3,2,2),_tail(shape,(8,8),2,1,3)]
    relays = [{'position':(8,8),'ports':((0,1),(0,3))},
              {'position':(3,7),'ports':((0,1),(0,2))},
              # This relay lies behind the first hit on the left key's line.
              {'position':(8,14),'ports':((0,1),(0,2))}]
    a = _render(shape,sources,relays)
    expected = np.zeros(shape,dtype=np.int8)
    cells = (_box((8,8)) | {(10,c) for c in range(8)}
             | {(r,9) for r in range(12,20)} | {(7,9)}
             | {(7,11),(6,12),(5,12),(4,13),(3,13),(2,14),(1,14),(0,15)})
    for p in cells:
        expected[p] = 2
    assert np.array_equal(solve(a),expected), 'rendezvous/staircase/first-hit fixture'
    assert _alternative_diff(a,expected,law='independent')
    assert _alternative_diff(a,expected,one_key=True)
    assert _alternative_diff(a,expected,law='sign')
    assert _alternative_diff(a,expected,phase='ceil')
    # Move the bottom key to an empty column: one arrival never unlocks a gate.
    sources[1] = _tail(shape,(8,13),2,0,3)
    a = _render(shape,sources,relays[:2])
    expected = np.zeros(shape,dtype=np.int8)
    expected[10,:8] = 2
    expected[:,13] = 2
    assert np.array_equal(solve(a),expected), 'single arrival must not activate'
    # Different nearest objects also fail, even though both keys have a hit.
    sources = [_tail(shape,(3,7),3,2,2),_tail(shape,(8,8),2,1,3)]
    a = _render(shape,sources,relays[:2])
    expected = np.zeros(shape,dtype=np.int8)
    expected[5,:7] = 2
    expected[12:,9] = 2
    assert np.array_equal(solve(a),expected), 'different first objects must not activate'
    # Both miss. Their paths cross in air and simply union in signal color.
    sources = [_tail(shape,(13,7),3,2,2),_tail(shape,(8,17),2,1,3)]
    a = _render(shape,sources,relays[:2])
    expected = np.zeros(shape,dtype=np.int8)
    expected[15,:] = 2
    expected[:,18] = 2
    assert np.array_equal(solve(a),expected), 'air crossing is not a compass vote'


def self_test() -> dict:
    """42 seeds, streamed in memory; no files, traces, logs, or worker fanout."""
    _rule_fixtures()
    shapes, layouts, headings, entries, kinds, palettes, counts = set(),set(),set(),set(),set(),set(),set()
    max_gap = 0
    hashes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train','test'):
            for p in puzzle[split]:
                a,b = p['input'],p['output']
                prediction,f = _execute(a)
                assert np.array_equal(prediction,b)
                digest.update(a.tobytes())
                shapes.add(a.shape)
                canonical = []
                for k in range(4):
                    for reflect in (False,True):
                        x = np.rot90(a,k)
                        if reflect:
                            x = np.fliplr(x)
                        mapping = {}
                        normalized = bytes(mapping.setdefault(int(v),len(mapping)) for v in x.ravel())
                        canonical.append((x.shape,normalized))
                layouts.add(min(canonical))
                headings.update(e['heading'] for e in f['events'])
                kinds.update(e['kind'] for e in f['events'])
                max_gap = max(max_gap,max(max(e['distances']) for e in f['events']))
                entries.update(s['direction'] for s in f['sources'])
                palettes.add(f['palette'])
                counts.add((len(f['active']),f['inactive']))
                assert np.array_equal(solve(a.tolist()),b)
        hashes.add(digest.digest())
        if seed in (0,17,41):
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(p[k],q[k]) for s in ('train','test')
                       for p,q in zip(puzzle[s],repeat[s]) for k in ('input','output'))
    assert len(hashes) == 42 and len(shapes) >= 25 and len(layouts) >= 150
    assert kinds == set(BY_KIND) and len(headings) >= 12 and entries == set(DIRS)
    assert len(palettes) >= 30 and len(counts) >= 6 and max_gap >= 10
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            puzzle = generate_puzzle(730+nt*10+ne,nt,ne)
            assert (len(puzzle['train']),len(puzzle['test'])) == (nt,ne)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(2.0,1),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupt = generate_puzzle(0,2,1)
    corrupt['test'][0]['output'] = corrupt['test'][0]['output'].copy()
    corrupt['test'][0]['output'][0,0] = (int(corrupt['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt label accepted')
    return {'seeds':42,'distinct_puzzles':len(hashes),'shapes':len(shapes),
            'canonical_structures':len(layouts),'headings':len(headings),
            'heading_classes':sorted(kinds),'active_inactive_count_pairs':len(counts),
            'max_approach_cells':max_gap,'independent_semantic_verification':False}
