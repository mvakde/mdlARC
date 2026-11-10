"""Reactive scaffolds: light-gated identity removal with gravity feedback.

Only standard library and NumPy. No files, cached examples, or random state are
consulted by solve. Generation and self_test retain no execution traces on disk.
"""
from __future__ import annotations

import hashlib
import numbers
import numpy as np

# A rod is (top row, left column, height, width, identity). Exactly one spatial
# extent is 1. Input rods are disjoint straight monochrome components.


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f'{name} outside its legal range')
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected an integer ARC grid of size 1..30 and colors 0..9')
    return a.astype(np.int8, copy=True)


def _cells(p):
    r, c, h, w, _ = p
    return {(y, x) for y in range(r, r+h) for x in range(c, c+w)}


def _shift(p, distance):
    return (p[0]+distance, *p[1:])


def _occupancy(rods):
    occupied = {}
    for i, p in enumerate(rods):
        for cell in _cells(p):
            _require(cell not in occupied, 'rods overlap')
            occupied[cell] = i
    return occupied


def _settle(rods, height):
    """Synchronous rigid gravity; mutual support can descend together."""
    rods = list(rods)
    for _ in range(height):
        occupied = _occupancy(rods)
        moving = {i for i, p in enumerate(rods) if p[0]+p[2] < height}
        while True:
            blocked = {i for i in moving if any(
                (y+1, x) in occupied and occupied[y+1, x] != i
                and occupied[y+1, x] not in moving for y, x in _cells(rods[i]))}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return rods
        rods = [_shift(p, 1) if i in moving else p for i, p in enumerate(rods)]
    raise AssertionError('gravity did not converge within canvas height')


def _settle_direct(rods, height):
    """QA: separate lowest-first maximal-distance implementation."""
    rods = list(rods)
    for _ in range(height * (len(rods)+1)):
        changed = False
        for i in sorted(range(len(rods)), key=lambda j: rods[j][0]+rods[j][2], reverse=True):
            occupied = set().union(*(_cells(q) for j, q in enumerate(rods) if j != i))
            p = rods[i]
            distance = 0
            while p[0]+p[2]+distance < height and not (_cells(_shift(p, distance+1)) & occupied):
                distance += 1
            if distance:
                rods[i] = _shift(p, distance)
                changed = True
        if not changed:
            return rods
    raise AssertionError('direct gravity did not converge')


def _parse(a):
    """Visible conventions determine background, guide, selectors and rods."""
    h, w = a.shape
    if h < 8 or w < 5:
        raise ValueError('reactive scaffold is too small')
    bg = int(a[1, 0])
    if not np.all(a[1] == bg) or not np.all(a[4] == bg):
        raise ValueError('blank control separator and entry clearance are required')
    guide_colors = set(map(int, np.unique(a[2]))) - {bg}
    if len(guide_colors) != 1 or not np.array_equal(a[2], a[3]):
        raise ValueError('two aligned source rows must name a unique guide color')
    guide = guide_colors.pop()
    entries = tuple(map(int, np.flatnonzero(a[2] == guide)))
    if not 1 <= len(entries) <= 4 or any(b-a < 2 for a, b in zip(entries, entries[1:])):
        raise ValueError('sources must be separated by a blank column')
    selectors = tuple(map(int, a[0, a[0] != bg]))
    if not 1 <= len(selectors) <= 3 or len(set(selectors)) != len(selectors) or guide in selectors:
        raise ValueError('one to three distinct non-guide selectors are required')
    if np.any(a[5:] == guide):
        raise ValueError('input guide may occur only in the source stubs')
    pending = {(int(y), int(x)) for y, x in zip(*np.nonzero(a[5:] != bg))}
    pending = {(y+5, x) for y, x in pending}
    rods = []
    while pending:
        start = min(pending)
        pending.remove(start)
        color = int(a[start])
        component, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for p in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                if p in pending and int(a[p]) == color:
                    pending.remove(p)
                    component.add(p)
                    stack.append(p)
        ys, xs = zip(*component)
        top, left = min(ys), min(xs)
        rh, rw = max(ys)-top+1, max(xs)-left+1
        if min(rh, rw) != 1 or len(component) != rh*rw or len(component) < 2:
            raise ValueError('each monochrome input component must be a straight rod')
        rods.append((top, left, rh, rw, color))
    if not rods:
        raise ValueError('at least one rod is required')
    return bg, guide, entries, selectors, rods


def _trace(rods, shape, entries, *, branching=True):
    """Trace a single complete illumination on one immutable silhouette.

    A collision wraps the maximal OCCUPIED transverse run, even when adjacent
    rods have different identities. Every rod with a cell in that run is lit.
    Only forward-going rays activate runs; transverse drawn light is not a ray.
    """
    h, w = shape
    occupied = _occupancy(rods)
    active = set(entries)
    lit, light = set(), set()
    collisions = clipping = multihits = simultaneous = 0
    for y in range(4, h):
        hits, following = {}, set()
        for x in sorted(active):
            if (y, x) not in occupied:
                light.add((y, x))
                following.add(x)
                continue
            left = right = x
            while left and (y, left-1) in occupied:
                left -= 1
            while right+1 < w and (y, right+1) in occupied:
                right += 1
            hits[(left, right)] = hits.get((left, right), 0)+1
        simultaneous += len(hits) >= 2
        for (left, right), count in sorted(hits.items()):
            collisions += 1
            clipping += left == 0 or right == w-1
            multihits += count >= 2
            lit.update(occupied[y, x] for x in range(left, right+1))
            if branching:
                light.update((y-1, x) for x in range(max(0,left-1), min(w,right+2))
                             if (y-1, x) not in occupied)
                for x in (left-1, right+1):
                    if 0 <= x < w:
                        following.add(x)
                        light.add((y, x))
        active = following
    return lit, light, {'collisions': collisions, 'clipped': clipping,
                        'multihits': multihits, 'simultaneous': simultaneous}


def _react(rods, shape, entries, selectors, mode='full', gravity=_settle):
    """Modes other than full are deliberately wrong rules used only for QA."""
    rods = list(rods)
    rounds = moved = remote = deleted = 0
    displacements = set()
    # In full mode each nonterminal round removes an identity permanently:
    # at most len(selectors) removal rounds and one final illumination.
    for _ in range(len(rods)+1):
        lit, light, ray_stats = _trace(rods, shape, entries, branching=mode != 'straight')
        hot = {rods[i][4] for i in lit} & set(selectors)
        if mode == 'all':
            hot = {p[4] for p in rods} & set(selectors)
        if not hot:
            if mode == 'late':
                rods = gravity(rods, shape[0])
                _, light, ray_stats = _trace(rods, shape, entries)
            stats = dict(ray_stats, rounds=rounds, moved=moved, remote=remote,
                         deleted=deleted, falls=tuple(sorted(displacements)))
            return rods, light, stats
        ids = {i for i, p in enumerate(rods) if p[4] in hot
               and (mode != 'local' or i in lit)}
        remote += len(ids-lit)
        deleted += len(ids)
        surviving = [p for i, p in enumerate(rods) if i not in ids]
        after = surviving if mode in ('nogravity', 'late') else gravity(surviving, shape[0])
        for p, q in zip(surviving, after):
            distance = q[0]-p[0]
            moved += distance > 0
            if distance:
                displacements.add(distance)
        rods = after
        rounds += 1
        if mode == 'one':
            _, light, ray_stats = _trace(rods, shape, entries)
            return rods, light, dict(ray_stats, rounds=rounds, moved=moved,
                                    remote=remote, deleted=deleted,
                                    falls=tuple(sorted(displacements)))
    raise AssertionError('every reaction must remove at least one rod')


def _render(source, rods, light=()):
    bg = int(source[1, 0])
    out = np.full(source.shape, bg, dtype=np.int8)
    out[:4] = source[:4]
    for p in rods:
        for y, x in _cells(p):
            out[y, x] = p[4]
    for y, x in light:
        if out[y, x] == bg:
            out[y, x] = source[2, np.flatnonzero(source[2] != bg)[0]]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    _, _, entries, selectors, rods = _parse(a)
    final, light, _ = _react(rods, a.shape, entries, selectors)
    return _render(a, final, light)


def _state_key(rods):
    return tuple(sorted(rods))


def _analyze(a):
    bg, guide, entries, selectors, rods = _parse(a)
    final, light, stats = _react(rods, a.shape, entries, selectors)
    key = _state_key(final)
    for name in ('all', 'straight', 'local', 'one', 'late', 'nogravity'):
        other, _, _ = _react(rods, a.shape, entries, selectors, name)
        stats[name+'_diff'] = _state_key(other) != key
    stats.update(rod_count=len(rods), surviving=len(final), sources=len(entries),
                 selectors=len(selectors), bg=bg, guide=guide,
                 activated=len({p[4] for p in rods}-{p[4] for p in final}),
                 identities=len({p[4] for p in rods}),
                 h_lengths=tuple(p[3] for p in rods if p[2] == 1),
                 v_lengths=tuple(p[2] for p in rods if p[3] == 1))
    return final, light, stats


def _drop(rods, width, height, rng, identity):
    horizontal = bool(rng.random() < .62)
    length = int(rng.integers(2, min(width, 16)+1)) if horizontal else int(rng.integers(2, min(height-6, 9)+1))
    rh, rw = (1, length) if horizontal else (length, 1)
    p = (5, int(rng.integers(width-rw+1)), rh, rw, identity)
    occupied = set().union(*(_cells(q) for q in rods))
    if p[0]+rh > height or _cells(p) & occupied:
        return False
    while p[0]+rh < height and not (_cells(_shift(p,1)) & occupied):
        p = _shift(p,1)
    neighbors = {(y+dy,x+dx) for y,x in _cells(p)
                 for dy,dx in ((-1,0),(1,0),(0,-1),(0,1))}
    if any(q[4] == identity and neighbors & _cells(q) for q in rods):
        return False
    rods.append(p)
    return True


def _core(rng, kind, large=False):
    """Random geometric inequalities, not a bank of saved input/output pairs."""
    h = int(rng.integers(18 if large else 12, 31 if large else 27))
    w = int(rng.integers(16 if large else 10, 31))
    floor = h-1
    if kind == 'feedback':
        left = int(rng.integers(0, w-7))
        small_left = left+int(rng.integers(0, 3))
        small_right = int(rng.integers(small_left+1, w-4))
        canopy_left = small_right+1
        canopy_right = int(rng.integers(canopy_left+2, w))
        fuse_right = int(rng.integers(canopy_left, canopy_right+1))
        post_length = int(rng.integers(2, min(10, h-8)+1))
        shelf_row = floor-post_length
        post_x = int(rng.integers(small_left, small_right+1))
        rods = [(shelf_row+1,post_x,post_length,1,2),
                (shelf_row,small_left,1,small_right-small_left+1,3),
                (shelf_row-1,left,1,fuse_right-left+1,4),
                (shelf_row-2,canopy_left,1,canopy_right-canopy_left+1,5)]
        entries = [int(rng.integers(canopy_left, canopy_right+1))]
        selectors = (3,4)
        source_interval = (canopy_left,canopy_right)
    elif kind == 'cascade':
        x = int(rng.integers(3,w-4))
        beam_right = int(rng.integers(x+3,w))
        bridge_right = int(rng.integers(beam_right,w))
        low_length = int(rng.integers(2,min(5,(h-7)//2)+1))
        upper_length = int(rng.integers(low_length,min(8,h-7-low_length)+1))
        bridge_row = floor-low_length
        remote_x = int(rng.integers(0,x-1))
        remote_base = int(rng.integers(2,min(7,h-8)+1))
        remote_top = int(rng.integers(2,min(7,h-6-remote_base)+1))
        rods = [(bridge_row+1,x,low_length,1,2),
                (bridge_row,x,1,bridge_right-x+1,3),
                (bridge_row-upper_length,x+1,upper_length,1,4),
                (bridge_row-upper_length-1,x+1,1,beam_right-x,5),
                (h-remote_base,remote_x,remote_base,1,3),
                (h-remote_base-remote_top,remote_x,remote_top,1,2)]
        entries = [int(rng.integers(x+1,beam_right+1))]
        selectors = (3,4)
        source_interval = (x+1,beam_right)
    else:
        raise ValueError('unknown construction kind')
    # A source can have two separated entries into the same visible canopy.
    # This explicitly demonstrates merging/idempotence before tests use it.
    if rng.random() < .65:
        lo, hi = source_interval
        candidates = [x for x in range(lo,hi+1) if all(abs(x-y)>=2 for y in entries)]
        if candidates:
            entries.append(int(rng.choice(candidates)))
    if rng.random() < .5:
        rods = [(r,w-c-rw,rh,rw,k) for r,c,rh,rw,k in rods]
        entries = [w-1-x for x in entries]
    return (h,w), rods, sorted(entries), selectors


def _input(shape, rods, entries, selectors, rng, palette):
    # Canonical role indices: 0 background, 1 guide, 2..9 rod identities.
    a = np.zeros(shape,dtype=np.int8)
    positions = rng.choice(shape[1],len(selectors),replace=False)
    for x,k in zip(positions,selectors):
        a[0,int(x)] = k
    a[2:4,entries] = 1
    for p in rods:
        for cell in _cells(p):
            a[cell] = p[4]
    return np.asarray(palette,dtype=np.int8)[a]


def _candidate(rng, kind, palette, extras=0, large=False, test=False):
    if kind == 'random':
        shape = (int(rng.integers(12 if not large else 18,31)),
                 int(rng.integers(9 if not large else 16,31)))
        rods = []
        target = int(rng.integers(6,19))
        for _ in range(4*target):
            if len(rods) >= target:
                break
            _drop(rods,shape[1],shape[0],rng,int(rng.integers(2,8)))
        if not rods:
            return None
        n = int(rng.integers(1,min(3,(shape[1]+1)//2)+1))
        options = list(map(int,rng.permutation(shape[1])))
        entries = []
        for x in options:
            if all(abs(x-y)>=2 for y in entries):
                entries.append(x)
                if len(entries)==n:
                    break
        selectors = tuple(map(int,rng.choice(np.arange(2,8),size=3 if test else int(rng.integers(1,3)),replace=False)))
    else:
        shape, rods, entries, selectors = _core(rng,kind,large)
        if test:
            selectors = (2,3,4)
        for _ in range(extras*5):
            if len(rods) >= (4 if kind=='feedback' else 6)+extras:
                break
            _drop(rods,shape[1],shape[0],rng,int(rng.integers(2,8)))
    a = _input(shape,rods,sorted(entries),selectors,rng,palette)
    try:
        bg,guide,es,ss,parsed = _parse(a)
        if _state_key(parsed) != _state_key([(*p[:4],int(palette[p[4]])) for p in rods]):
            return None
        if _settle(parsed,shape[0]) != parsed:
            return None
        final,light,stats = _analyze(a)
    except (ValueError,AssertionError):
        return None
    if not (stats['moved'] and stats['straight_diff'] and final):
        return None
    if test and (not set(ss) <= {p[4] for p in parsed} or stats['activated'] < 2):
        return None
    if kind == 'cascade' and not (stats['rounds'] >= 2 and stats['local_diff'] and stats['one_diff']):
        return None
    if kind == 'feedback' and not (stats['all_diff'] and stats['late_diff']):
        return None
    return {'input':a,'output':_render(a,final,light)}


def _key(a):
    a = np.asarray(a,dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed,'seed',0)
    num_train = _integer(num_train,'num_train',2,10)
    num_test = _integer(num_test,'num_test',1,4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    mixed_palette = bool(rng.random() < .5)
    scale_shift = bool(rng.random() < .4)
    puzzle = {'train':[],'test':[]}
    seen = set()
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            witness = ('cascade','feedback')[index] if split=='train' and index<2 else None
            for attempt in range(100):
                if witness:
                    kind = witness
                    extras = int(rng.integers(0,3)) if attempt < 60 else 0
                elif attempt < 60:
                    kind = str(rng.choice(['random','cascade','feedback'],p=[.65,.20,.15]))
                    extras = int(rng.integers(2,8)) if split=='test' else int(rng.integers(0,5))
                else:
                    kind = 'cascade' if split=='test' or index%2==0 else 'feedback'
                    extras = 0
                local_palette = rng.permutation(10) if mixed_palette else palette
                pair = _candidate(rng,kind,local_palette,extras,
                                  large=split=='test' and scale_shift, test=split=='test')
                if pair is not None and _key(pair['input']) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    break
            else:
                raise RuntimeError('bounded sampler exhausted its unique scene budget')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle)=={'train','test'},'puzzle schema')
    _require(all(isinstance(puzzle[s],list) for s in puzzle),'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2<=nt<=10 and 1<=nq<=4 and nt>nq,'pair counts')
    seen, training = set(), []
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair)=={'input','output'},'pair schema')
            a,o = _grid(pair['input']),_grid(pair['output'])
            _require(a.shape==o.shape,'shape preservation')
            _require(_key(a) not in seen,'distinct inputs')
            seen.add(_key(a))
            bg,guide,entries,selectors,rods = _parse(a)
            _require(_settle(rods,a.shape[0])==rods,'initial stability')
            final,light,stats = _analyze(a)
            expected = _render(a,final,light)
            _require(np.array_equal(o,expected),'reactive rule output')
            _require(np.array_equal(a[:4],o[:4]),'preserve controls and stubs')
            _require(_settle(final,a.shape[0])==final,'final mechanical stability')
            lit,_,_ = _trace(final,a.shape,entries)
            _require(not ({final[i][4] for i in lit} & set(selectors)), 'final optical stability')
            _require(stats['rounds']<=len(selectors),'identity-removal termination bound')
            _require(stats['moved']>0 and stats['straight_diff'],'both inherited mechanisms affect rods')
            _require(final and not np.array_equal(a,o),'nontrivial surviving scaffold')
            # All surviving cells have the same identity and horizontal geometry;
            # downward translation is checked by the non-overlapping simulator.
            _require(set(map(int,np.unique(o))) <= set(map(int,np.unique(a))), 'no invented colors')
            if split=='train':
                _require(len(selectors)<=2,'training command cardinality')
                training.append(stats)
            else:
                _require(len(selectors)==3 and set(selectors)<={p[4] for p in rods},
                         'test recombines three present identities')
                _require(stats['activated']>=2,'test must exercise multiple activations')
    for mutation in ('all','straight','local','one','late','nogravity'):
        _require(any(s[mutation+'_diff'] for s in training),f'training admits {mutation} shortcut')
    _require(any(s['rounds']>=2 for s in training),'demonstrate re-illumination')
    _require(any(s['remote']>0 for s in training),'demonstrate global identity activation')


def _mechanics_test():
    # Small manually specified final rod states, independent of sampled layouts.
    shape = (12,11)
    rods = [(9,3,3,1,2),(8,2,1,4,3),(7,1,1,8,4),(6,6,1,4,5)]
    final,_,s = _react(rods,shape,[7],(3,4))
    _require(_state_key(final)==_state_key([rods[0],rods[1],(11,6,1,4,5)]),
             'falling canopy must protect the selected small shelf')
    _require(s['rounds']==1,'protected shelf must not activate')
    shape = (14,13)
    rods = [(12,6,2,1,2),(11,6,1,7,3),(9,7,2,1,4),
            (8,7,1,4,5),(11,3,3,1,3),(6,3,5,1,2)]
    final,_,s = _react(rods,shape,[9],(3,4))
    _require(_state_key(final)==_state_key([rods[0],(13,7,1,4,5),(9,3,5,1,2)]),
             'remote fuse and two-stage release diagnostic')
    _require(s['rounds']==2 and s['remote']>=1,'cascade diagnostic')
    rods = [(6,2,1,3,2),(6,5,4,1,3)]
    lit,light,_ = _trace(rods,(10,9),[3])
    _require(lit=={0,1} and all((5,x) in light for x in range(1,7)),
             'wrap the occupied run across color boundaries')
    _,light,_ = _trace([(6,0,1,9,2)],(10,9),[3])
    _require(not any(y>6 for y,x in light),'full-width run absorbs both clipped branches')


def self_test():
    """42 seeds; streaming summaries/hashes, no retained per-seed payload files."""
    _mechanics_test()
    shapes,layouts,puzzles,colors,rod_counts,rounds,falls,sources,palettes = (set() for _ in range(9))
    flags = {k:0 for k in ('all','straight','local','one','late','nogravity')}
    pair_count = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train','test'):
            for pair in p[split]:
                a = pair['input']
                bg,g,es,ss,rods = _parse(a)
                final,light,s = _analyze(a)
                other,other_light,_ = _react(rods,a.shape,es,ss,gravity=_settle_direct)
                _require(_state_key(final)==_state_key(other) and light==other_light,'two gravity implementations disagree')
                _require(np.array_equal(solve(a.tolist()),pair['output']),'list input solver')
                _require(np.array_equal(solve(a.copy()),pair['output']),'pure input-only solver')
                before = a.copy()
                solve(a)
                _require(np.array_equal(before,a),'solver mutated input')
                shapes.add(a.shape)
                roles={bg:0,g:1}
                canonical=bytes(roles.setdefault(int(c),len(roles)) for c in a.flat)
                layouts.add(hashlib.sha256(str(a.shape).encode()+canonical).digest())
                digest.update(str(a.shape).encode()+a.tobytes()+pair['output'].tobytes())
                colors.add(bg);palettes.add(tuple(map(int,np.unique(a))))
                rod_counts.add(s['rod_count']);rounds.add(s['rounds'])
                falls.update(s['falls']);sources.add(s['sources'])
                for k in flags:flags[k]+=s[k+'_diff']
                pair_count+=1
        puzzles.add(digest.digest())
    _require(len(puzzles)==42 and len(layouts)>=150,'structural diversity')
    _require(len(shapes)>=45 and len(rod_counts)>=8,'geometry/count diversity')
    _require(len(colors)==10 and len(palettes)>=25,'palette role diversity')
    _require({1,2}<=rounds and len(falls)>=5 and {1,2}<=sources,'mechanism diversity')
    _require(all(n>=42 for n in flags.values()),'counterfactual coverage')
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(0,nt,nq)
            validate_generated_puzzle(p)
            q = generate_puzzle(0,nt,nq)
            _require(all(_key(x[k])==_key(y[k]) for s in p for x,y in zip(p[s],q[s])
                         for k in ('input','output')),'determinism/count boundary')
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted illegal counts')
    return {'seeds':42,'pairs':pair_count,'shapes':len(shapes),'structures':len(layouts),
            'rod_counts':sorted(rod_counts),'reaction_rounds':sorted(rounds),
            'fall_distances':sorted(falls),'source_counts':sorted(sources),
            'mutation_witness_pairs':flags,'independent_semantic_review':False}
