"""Light develops only the visible parts of inferred frame-corner stencils.

All semantic information is recovered from the input.  The constructor paints
labels from explicit masks; the solver uses a separate, ordered ray scan.
"""
from __future__ import annotations

import json
from collections import deque
import numpy as np


SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = {(int(y),int(x)) for y,x in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, queue = {start}, deque([start])
        while queue:
            y, x = queue.popleft()
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    queue.append(q)
        result.append(group)
    return result


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r+1)} |
            {(b, x) for x in range(l, r+1)} |
            {(y, l) for y in range(t, b+1)} |
            {(y, r) for y in range(t, b+1)})


def _corners(rect):
    t, l, b, r = rect
    return ((t, l, 0), (t, r, 1), (b, l, 2), (b, r, 3))


def _points(corner, pattern):
    y, x, q = corner
    dy, dx = SIGNS[q]
    return {(y + dy*u, x + dx*v) for u, v in pattern}


def _connected8(points):
    reached = {min(points)}
    while True:
        more = {q for q in points if any(max(abs(q[0]-r[0]),abs(q[1]-r[1])) <= 1
                                        for r in reached)}
        if more == reached:
            return more == set(points)
        reached = more


def _fit(values, sign):
    """Fit a thin periodic rail from every visible column, including phase."""
    values = np.asarray(values, dtype=int)
    fits = []
    for period in (1, 2, 3):
        for phase in range(period):
            expected = values[0] + sign * ((np.arange(len(values)) + phase) // period)
            if (len(values) >= 2*period+1 and np.array_equal(expected, values)):
                fits.append((period, phase, int(values[0])))
    if len(fits) != 1:
        raise ValueError("rail must expose two complete, unambiguous steps")
    return fits[0]


def _emitter_candidates(inp, bg):
    for rotation in range(4):
        a = np.rot90(inp, -rotation)
        for color in map(int, np.unique(a)):
            if color == bg:
                continue
            pos = np.argwhere(a == color)
            if len(pos) < 9:
                continue
            rear, mouth = map(int, (pos[:, 1].min(), pos[:, 1].max()))
            cap = np.flatnonzero(a[:, rear] == color)
            if len(cap) < 3 or np.any(np.diff(cap) != 1) or mouth-rear < 3:
                continue
            rails = [np.flatnonzero(a[:, x] == color) for x in range(rear+1, mouth+1)]
            if any(len(rows) != 2 for rows in rails):
                continue
            top, bottom = [int(v[0]) for v in rails], [int(v[1]) for v in rails]
            if cap[0] != top[0] or cap[-1] != bottom[0]:
                continue
            try:
                fits = (_fit(top, -1), _fit(bottom, 1))
            except ValueError:
                continue
            footprint = np.zeros(a.shape, dtype=bool)
            for x in range(rear+1, a.shape[1]):
                j = x-rear-1
                limits = [v + sign*((j+phase)//period)
                          for (period, phase, v), sign in zip(fits, (-1, 1))]
                footprint[:, x] = ((np.arange(a.shape[0]) > limits[0]) &
                                   (np.arange(a.shape[0]) < limits[1]))
            chamber = footprint.copy()
            chamber[:, mouth+1:] = False
            payloads = set(map(int, a[chamber])) - {bg, color}
            if len(payloads) != 1 or np.any(footprint & (a == color)):
                continue
            ink = payloads.pop()
            yield dict(a=a, rotation=rotation, bg=bg, wall=color, ink=ink,
                       rear=rear, mouth=mouth, fits=fits, footprint=footprint,
                       chamber=chamber)


def _read(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= inp.size//2:
        raise ValueError("input background must occupy more than half the canvas")
    matches = []
    for info in _emitter_candidates(inp, bg):
        a, ink = info['a'], info['ink']
        for frame in map(int, np.unique(a)):
            if frame in (bg, info['wall'], ink):
                continue
            groups = _components(a == frame)
            rects = []
            for group in groups:
                t, l = map(min, zip(*group))
                b, r = map(max, zip(*group))
                rect = (t, l, b, r)
                if b-t < 6 or r-l < 6 or group != _outline(rect):
                    break
                rects.append(rect)
            else:
                if not 1 <= len(rects) <= 4:
                    continue
                corners = [c for rect in rects for c in _corners(rect)]
                assigned = {c: set() for c in corners}
                source = (a == ink) & info['chamber']
                if not 1 <= int(source.sum()) <= 4 or len(_components(source)) != 1:
                    continue
                valid = True
                for y, x in np.argwhere((a == ink) & ~source):
                    options = []
                    for c in corners:
                        dy, dx = SIGNS[c[2]]
                        u, v = (int(y)-c[0])*dy, (int(x)-c[1])*dx
                        if 1 <= u <= 3 and 1 <= v <= 3:
                            options.append((c, (u, v)))
                    if len(options) != 1:
                        valid = False
                        break
                    c, p = options[0]
                    assigned[c].add(p)
                if not valid:
                    continue
                occurrences = [(c, frozenset(s)) for c, s in assigned.items() if s]
                if len(occurrences) < 2:
                    continue
                maximal = {s for _, s in occurrences if all(t <= s for _, t in occurrences)}
                if len(maximal) != 1:
                    continue
                pattern = maximal.pop()
                if (not 3 <= len(pattern) <= 8 or not _connected8(pattern)
                        or len({u for u,v in pattern}) < 2
                        or len({v for u,v in pattern}) < 2):
                    continue
                # An intact reference and all inferred copies must fit uniquely
                # within their own corner zones, without touching guide outlines.
                latent, occupied = set(), set()
                for c, _ in occurrences:
                    points = _points(c, pattern)
                    own = next(rect for rect in rects if c in _corners(rect))
                    if (any(not (own[0] < y < own[2] and own[1] < x < own[3]) for y,x in points)
                            or points & occupied):
                        valid = False
                        break
                    for y, x in points:
                        options = [d for d in corners if
                                   1 <= (y-d[0])*SIGNS[d[2]][0] <= 3 and
                                   1 <= (x-d[1])*SIGNS[d[2]][1] <= 3]
                        if options != [c]:
                            valid = False
                    occupied |= points
                    latent |= points
                if not valid:
                    continue
                blockers = ~np.isin(a, (bg, info['wall'], frame, ink))
                if np.any(blockers & info['chamber']):
                    continue
                matches.append(dict(info, frame=frame, rects=sorted(rects),
                                    occurrences=occurrences, pattern=pattern,
                                    latent=latent, blockers=blockers))
    if len(matches) != 1:
        raise ValueError("input must identify one emitter, frame system and intact stencil")
    return matches[0]


def _visibility(info):
    """Solver path: only opaque cells inside the prospective beam stop a lane."""
    a = info['a']
    lit = np.zeros(a.shape, dtype=bool)
    for y in range(a.shape[0]):
        stopped = False
        for x in range(info['rear']+1, a.shape[1]):
            if info['footprint'][y, x]:
                if info['blockers'][y, x]:
                    stopped = True
                elif not stopped:
                    lit[y, x] = True
    return lit


def _paint(info, lit, *, transpose=False):
    out = info['a'].copy()
    pattern = info['pattern']
    for c, _ in info['occurrences']:
        p = frozenset((v,u) for u,v in pattern) if transpose and c[2] in (1,2) else pattern
        for y,x in _points(c, p):
            if out[y,x] == info['bg'] and lit[y,x]:
                out[y,x] = info['ink']
    return out


def _rotation_rival(info, lit):
    """Quarter-turn normalization, anchored by the actual observed reference."""
    observations = []
    for c, clue in info['occurrences']:
        normalized = (frozenset((v,u) for u,v in clue)
                      if c[2] in (1,2) else clue)
        observations.append((c, normalized))
    maximal = {p for _,p in observations if all(q <= p for _,q in observations)}
    if len(maximal) != 1:
        return None
    alternate = dict(info, pattern=maximal.pop())
    return _paint(alternate,lit,transpose=True)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the template and light field using this input alone."""
    info = _read(input_grid)
    out = _paint(info, _visibility(info))
    return np.rot90(out, info['rotation']).copy()


def _mask_label(a, bg, ink, footprint, blockers, latent):
    """Construction path: union all blocker shadows, then intersect masks."""
    shadow = np.zeros(a.shape, dtype=bool)
    for y,x in np.argwhere(blockers & footprint):
        shadow[y, x:] = True
    out = a.copy()
    for y,x in latent:
        if a[y,x] == bg and footprint[y,x] and not shadow[y,x]:
            out[y,x] = ink
    return out


def _sample_pattern(rng, radius, rich):
    domain = [(u,v) for u in range(1,radius+1) for v in range(1,radius+1)]
    for _ in range(100):
        size = int(rng.integers(5 if rich else 3, min(8, len(domain)-1)+1))
        indices = rng.choice(len(domain), size=size, replace=False)
        p = frozenset(domain[int(i)] for i in indices)
        if len({u for u,v in p}) < (3 if rich else 2) or len({v for u,v in p}) < (3 if rich else 2):
            continue
        # Eight-connectivity, as in a small digital ornament.
        if _connected8(p):
            return p
    return frozenset({(1,1),(1,2),(2,1),(2,3),(3,2),(3,3)})


def _build(rng, mode, colors, rotation):
    bg, wall, frame, ink, opaque = map(int, colors[:5])
    radius = 3 if mode in ('cone','shadow','combined') else int(rng.choice((2,3,3)))
    pattern = _sample_pattern(rng, radius, mode in ('cone','shadow','combined'))
    h, w = int(rng.integers(21,31)), int(rng.integers(24,31))
    periods = [int(rng.integers(2,4)) if mode in ('cone','combined') else int(rng.integers(1,4)) for _ in range(2)]
    phases = [int(rng.integers(p)) for p in periods]
    length = 2*max(periods)+1
    rear = int(rng.integers(0,3))
    mouth = rear+length
    extents = [int(rng.integers(1,3)) for _ in range(2)]
    max_ext = [e+(length-1+phase)//period for e,phase,period in zip(extents,phases,periods)]
    center = int(rng.integers(max_ext[0]+1, h-max_ext[1]-1))
    a = np.full((h,w), bg, dtype=np.int8)
    a[center-extents[0]:center+extents[1]+1, rear] = wall
    footprint = np.zeros(a.shape, dtype=bool)
    for x in range(rear+1,w):
        j = x-rear-1
        top = center-extents[0]-(j+phases[0])//periods[0]
        bottom = center+extents[1]+(j+phases[1])//periods[1]
        footprint[:,x] = ((np.arange(h) > top) & (np.arange(h) < bottom))
        if x <= mouth:
            a[top,x] = a[bottom,x] = wall
    source_candidates = sorted(map(tuple, np.argwhere(footprint[:, :rear+3])))
    source_candidates = [p for p in source_candidates if a[p] == bg]
    source_start = source_candidates[int(rng.integers(len(source_candidates)))]
    patch = {source_start}
    for _ in range(int(rng.integers(0,4))):
        front = sorted({q for y,x in patch for q in ((y-1,x),(y+1,x),(y,x+1),(y,x-1))
                        if 0 <= q[0] < h and rear < q[1] <= min(mouth,rear+3)
                        and footprint[q] and a[q] == bg and q not in patch})
        if front:
            patch.add(front[int(rng.integers(len(front)))])
    for p in patch:
        a[p] = ink

    def classify(c):
        points = _points(c, pattern)
        bright = {p for p in points if footprint[p]}
        dark = points-bright
        return points, bright, dark

    # Free rectangle placement; primary acceptance is based on the intended
    # teaching interaction, not on a fixed slot or quadrant.
    minimum = 2*radius+3
    chosen = None
    for _ in range(180):
        sh = int(rng.integers(minimum, min(16,h)+1))
        sw = int(rng.integers(minimum, min(16,w-mouth-2)+1))
        t = int(rng.integers(0,h-sh+1))
        l = int(rng.integers(mouth+2,w-sw+1))
        rect = (t,l,t+sh-1,l+sw-1)
        options = []
        for c in _corners(rect):
            points, bright, dark = classify(c)
            if mode in ('cone','combined'):
                okay = bool(dark) and len({y for y,x in bright}) >= 2
            elif mode == 'shadow':
                okay = len(bright) == len(points) and len({y for y,x in bright}) >= 2
            else:
                okay = len(bright) >= 2
            if okay:
                options.append(c)
        if options:
            chosen = rect, options[int(rng.integers(len(options)))]
            break
    if chosen is None:
        raise ValueError("no teaching corner fits this geometry")
    first, target = chosen
    rects = [first]
    desired = int(rng.integers(1,5))
    for _ in range(90):
        if len(rects) >= desired:
            break
        sh = int(rng.integers(minimum,min(15,h)+1))
        sw = int(rng.integers(minimum,min(15,w-mouth-2)+1))
        t = int(rng.integers(0,h-sh+1))
        l = int(rng.integers(mouth+2,w-sw+1))
        rect = (t,l,t+sh-1,l+sw-1)
        if all(rect[2]+2 <= q[0] or q[2]+2 <= rect[0] or
               rect[3]+2 <= q[1] or q[3]+2 <= rect[1] for q in rects):
            rects.append(rect)
    for rect in rects:
        for p in _outline(rect):
            a[p] = frame
    # Adjacent reference/target orientations expose reflections of asymmetry.
    refs = [c for c in _corners(first) if c[2] ^ target[2] in (1,2)]
    reference = refs[int(rng.integers(len(refs)))]
    occurrences = []
    for rect in rects:
        for c in _corners(rect):
            if c not in (reference,target) and rng.random() < .35:
                continue
            points = sorted(_points(c,pattern))
            if c == reference or (c != target and rng.random() < .17):
                retained = points
            else:
                rng.shuffle(points)
                n = int(rng.integers(1, max(2,len(points)-1)))
                retained = points[:n]
            occurrences.append(c)
            for p in retained:
                a[p] = ink
    latent = set().union(*(_points(c,pattern) for c in occurrences))
    # All witness modes need genuinely missing cells of the relevant kinds.
    missing = {p for p in _points(target,pattern) if a[p] == bg}
    bright = {p for p in missing if footprint[p]}
    dark = missing-bright
    if mode in ('cone','combined') and not dark:
        raise ValueError("target lacks a missing exterior cell")
    if len({y for y,x in bright}) < 2:
        raise ValueError("target lacks two missing light lanes")
    blockers = np.zeros(a.shape,dtype=bool)
    if mode in ('shadow','combined','random') and (mode != 'random' or rng.random() < .8):
        options = []
        for y,x in sorted(bright):
            for bx in range(mouth+1,x):
                if a[y,bx] == bg and footprint[y,bx] and (y,bx) not in latent:
                    options.append((y,bx))
        if not options:
            raise ValueError("no upstream blocker can cut the stencil")
        y,bx = options[int(rng.integers(len(options)))]
        offsets = [(0,0)]
        if mode == 'random' or (mode == 'combined' and rng.random() < .5):
            offsets = [((0,0), (1,0)), ((0,0),(0,1),(1,1)),
                       ((0,0),(0,1),(1,0),(1,1))][int(rng.integers(3))]
        for dy,dx in offsets:
            q = (y+dy,bx+dx)
            if 0 <= q[0] < h and q[1] < w and a[q] == bg and q not in latent:
                a[q], blockers[q] = opaque, True
        if mode == 'combined' and rng.random() < .6:
            repeated = [(y,x) for x in range(bx+2,w) if a[y,x] == bg and (y,x) not in latent]
            if repeated:
                q = repeated[int(rng.integers(len(repeated)))]
                a[q], blockers[q] = opaque, True
    # Inert obstacles make geometric membership, rather than color alone,
    # determine whether a cell casts a shadow.
    if mode != 'cone':
        exterior = [p for p in map(tuple,np.argwhere(~footprint & (a == bg)))
                    if p not in latent and p[1] > mouth]
        for _ in range(int(rng.integers(0,3))):
            if exterior:
                q = exterior.pop(int(rng.integers(len(exterior))))
                a[q], blockers[q] = opaque, True
    out = _mask_label(a,bg,ink,footprint,blockers,latent)
    target_changes = {p for p in missing if out[p] == ink}
    if not target_changes:
        raise ValueError("witness has no developed cell")
    if mode in ('shadow','combined') and not (bright-target_changes):
        raise ValueError("witness has no shadowed missing cell")
    if np.count_nonzero(a == bg) <= a.size//2:
        raise ValueError("background lost dominance")
    return {'input':np.rot90(a,rotation).copy(), 'output':np.rot90(out,rotation).copy()}


def _key(grid):
    a = np.asarray(grid,dtype=np.int8)
    return a.shape,a.tobytes()


def _features(info):
    a = info['a']
    lit = _visibility(info)
    cone_only, shadow_only, combined = False, False, False
    prints, exterior, shadows, widening = set(),set(),set(),set()
    period_top, phase_top, top_start = info['fits'][0]
    period_bottom, phase_bottom, bottom_start = info['fits'][1]
    j = info['mouth']-info['rear']-1
    mouth_top = top_start-(j+phase_top)//period_top
    mouth_bottom = bottom_start+(j+phase_bottom)//period_bottom
    for c,clue in info['occurrences']:
        missing = {p for p in _points(c,info['pattern']) if a[p] == info['bg']}
        bright = {p for p in missing if lit[p]}
        dark = {p for p in missing if not info['footprint'][p]}
        shadow = {p for p in missing if info['footprint'][p] and not lit[p]}
        prints |= bright
        exterior |= dark
        shadows |= shadow
        widening |= {p for p in bright if p[0] <= mouth_top or p[0] >= mouth_bottom}
        cone_only |= bool(bright and dark and not shadow)
        shadow_only |= bool(bright and shadow and not dark)
        combined |= bool(bright and dark and shadow)
    rival = _rotation_rival(info,lit)
    true = _paint(info,lit)
    whole_zone = info['a'].copy()
    for c,_ in info['occurrences']:
        for p in _points(c,[(u,v) for u in range(1,4) for v in range(1,4)]):
            if whole_zone[p] == info['bg'] and lit[p]:
                whole_zone[p] = info['ink']
    return dict(shape=a.shape, frames=len(info['rects']),
                frame_shapes=tuple((b-t+1,r-l+1) for t,l,b,r in info['rects']),
                stencil=tuple(sorted(info['pattern'])), size=len(info['pattern']),
                palette=(info['bg'],info['wall'],info['frame'],info['ink']),
                rotation=info['rotation'], periods=tuple(f[0] for f in info['fits']),
                phases=tuple(f[1] for f in info['fits']),
                marks=len(info['occurrences']), intact=sum(s==info['pattern'] for _,s in info['occurrences']),
                retained=tuple(sorted(len(s) for _,s in info['occurrences'])),
                developed=len(prints), exterior=len(exterior), shadowed=len(shadows), widening=len(widening),
                cone_only=cone_only,shadow_only=shadow_only,combined=combined,
                effective_blockers=int(np.sum(info['blockers'] & info['footprint'])),
                inert_blockers=int(np.sum(info['blockers'] & ~info['footprint'])),
                repeated_lane=bool(np.any(np.sum(info['blockers'] & info['footprint'],axis=1)>1)),
                reflection_witness=rival is None or not np.array_equal(rival,true),
                no_a_witness=not np.array_equal(whole_zone,true))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name,value,lo,hi in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shared = rng.permutation(10) if rng.random() < .2 else None
    puzzle,seen = {'train':[],'test':[]},set()
    stencils = set()
    for split,count in (('train',int(num_train)),('test',int(num_test))):
        for index in range(count):
            mode = ('cone' if index == 0 else 'shadow') if split == 'train' and index < 2 else ('combined' if split == 'test' else 'random')
            accepted = False
            for _ in range(300):
                colors = shared if shared is not None else rng.permutation(10)
                rotation = int(rng.integers(4))
                try:
                    pair = _build(rng,mode,colors,rotation)
                    info = _read(pair['input'])
                    predicted = np.rot90(_paint(info,_visibility(info)),info['rotation'])
                    assert np.array_equal(predicted,pair['output'])
                    f = _features(info)
                    assert f['developed'] > 0 and _key(pair['input']) not in seen
                    if split == 'train' and index == 0:
                        assert f['cone_only'] and f['widening'] and f['reflection_witness'] and f['no_a_witness']
                    if split == 'train' and index == 1:
                        assert f['shadow_only'] and f['stencil'] not in stencils
                    if split == 'test':
                        assert f['combined'] and f['no_a_witness'] and f['stencil'] not in stencils
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    stencils.add(f['stencil'])
                    accepted = True
                    break
                except (ValueError,AssertionError):
                    continue
            if not accepted:
                raise RuntimeError(f"bounded construction exhausted for {mode}")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    nt,nq = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, features = set(), {'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _read(inp)
            expected = np.rot90(_paint(info,_visibility(info)),info['rotation'])
            assert np.array_equal(expected,out), "input-only solution disagrees"
            a,b = info['a'],np.rot90(out,-info['rotation'])
            changed = a != b
            assert np.any(changed) and np.all(a[changed] == info['bg'])
            assert np.all(b[changed] == info['ink'])
            assert np.array_equal(a[a != info['bg']],b[a != info['bg']])
            assert set(np.unique(inp)) == set(np.unique(out))
            lit = _visibility(info)
            expected_cells = {p for p in info['latent'] if a[p] == info['bg'] and lit[p]}
            assert set(map(tuple,np.argwhere(changed))) == expected_cells
            features[split].append(_features(info))
    t = features['train']
    assert any(f['cone_only'] and f['effective_blockers'] == 0 and f['widening'] for f in t)
    assert any(f['shadow_only'] for f in t)
    assert any(f['reflection_witness'] for f in t)
    assert any(f['no_a_witness'] for f in t)
    assert len({f['stencil'] for f in t}) >= 2
    assert all(f['combined'] and f['no_a_witness'] for f in features['test'])
    assert not ({f['stencil'] for f in t} & {f['stencil'] for f in features['test']})


def _signature(puzzle):
    return tuple(_key(pair[k]) for s in ('train','test') for pair in puzzle[s] for k in ('input','output'))


def _handwritten_regressions():
    # Explicitly authored geometry and answer, independent of _build and its
    # mask-label arithmetic. One copy has cone/phase/shadow cuts simultaneously.
    a = np.zeros((23,29),dtype=np.int8)
    a[8:13,1] = 1
    for x,top in enumerate((8,8,8,7,7,7,6),2):
        a[top,x] = a[20-top,x] = 1
    a[10,2] = 3
    for p in _outline((1,11,10,21)):
        a[p] = 2
    reference = ((2,12),(2,14),(3,12),(3,13),(4,13))
    for p in reference:
        a[p] = 3
    a[2,20] = a[9,12] = 3
    for p in ((3,18),(4,18),(4,22),(2,15),(9,0)):
        a[p] = 4
    expected = a.copy()
    additions = {(3,20),(9,14),(8,12),(8,13),(7,13)}
    for p in additions:
        expected[p] = 3
    permutation = np.array((7,0,9,4,3,1,2,5,6,8),dtype=np.int8)
    for k in range(4):
        for mapping in (np.arange(10,dtype=np.int8),permutation):
            inp = np.rot90(mapping[a],k).copy()
            out = np.rot90(mapping[expected],k).copy()
            assert np.array_equal(solve(inp),out), "handwritten optical stencil regression"
    # Distinguish boundary equality from interior, narrow shadows, transparent
    # existing ink/guide outlines, and an illuminated but unmarked corner.
    assert expected[3,19] == 0  # virtual top rail, not illuminated
    assert expected[4,19] == 0  # straight shadow from (4,18)
    assert expected[3,20] == 3  # adjacent lane remains bright
    assert expected[4,20] == 0  # illuminated but absent from the stencil
    assert expected[9,14] == 3  # through the guide and existing ink at (9,12)
    assert expected[9,20] == 0  # unmarked bottom-right corner
    assert expected[2,20] == 3  # preserve the original clue in darkness
    info = _read(a)
    f = _features(info)
    assert f['combined'] and f['no_a_witness'] and f['reflection_witness']
    assert f['repeated_lane'] and f['inert_blockers'] == 3
    # The transverse group at column 18 straddles the virtual top rail:
    # (3,18) is inert, while (4,18) blocks. Later entry into the cone does
    # not turn the former into an upstream blocker of (3,20).
    assert expected[3,18] == 4 and expected[3,20] == 3
    single_blocker = a.copy()
    single_blocker[4,22] = 0
    assert np.array_equal(solve(single_blocker) != single_blocker,expected != a)
    assert _fit((8,7,7,6,6),-1) == (2,1,8)
    assert _fit((8,8,7,7,6),-1) == (2,0,8)
    assert _fit((8,8,7,7,7,6,6),-1) == (3,1,8)
    # Removing the blocker develops the shadow cell, but cannot develop a
    # cone-excluded cell or a cell omitted by the reference stencil.
    no_block = a.copy()
    no_block[no_block == 4] = 0
    clear = solve(no_block)
    assert clear[4,19] == 3 and clear[3,19] == 0 and clear[4,20] == 0
    # The shortest permitted emitter: a three-cell cap and two three-cell
    # period-one flanks. It has only nine structural cells.
    compact = np.zeros((21,24),dtype=np.int8)
    compact[9:12,0] = 1
    for x,top in enumerate((9,8,7),1):
        compact[top,x] = compact[20-top,x] = 1
    compact[10,1] = 3
    for p in _outline((4,7,12,15)):
        compact[p] = 2
    for p in ((5,8),(5,9),(6,8),(6,10)):
        compact[p] = 3
    compact[5,14] = 3
    compact_answer = compact.copy()
    for p in ((5,13),(6,14),(6,12)):
        compact_answer[p] = 3
    assert np.array_equal(solve(compact),compact_answer)


def self_test():
    """Bounded semantic regressions and a measured 42-seed population."""
    _handwritten_regressions()
    records,signatures = [],set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        for split in ('train','test'):
            for pair in p[split]:
                info = _read(pair['input'])
                f = _features(info)
                records.append(f)
                assert np.array_equal(solve(np.asarray(pair['input']).tolist()),pair['output'])
                # No A: expose every illuminated blank corner-zone cell.
                broad = set().union(*(_points(c,[(u,v) for u in range(1,4) for v in range(1,4)])
                                      for c,_ in info['occurrences']))
                # No B: finish every missing stencil cell, regardless of light.
                no_b = info['a'].copy()
                for q in info['latent']:
                    if no_b[q] == info['bg']:
                        no_b[q] = info['ink']
                if f['exterior'] or f['shadowed']:
                    assert not np.array_equal(no_b,np.rot90(pair['output'],-info['rotation']))
                lit = _visibility(info)
                no_a = info['a'].copy()
                for q in broad:
                    if no_a[q] == info['bg'] and lit[q]:
                        no_a[q] = info['ink']
                assert f['no_a_witness'] == (not np.array_equal(no_a,np.rot90(pair['output'],-info['rotation'])))
    assert len(signatures) == 42
    assert len({f['stencil'] for f in records}) >= 65
    assert len({f['shape'] for f in records}) >= 45
    assert len({f['frame_shapes'] for f in records}) >= 55
    assert len({f['palette'] for f in records}) >= 75
    assert {f['rotation'] for f in records} == {0,1,2,3}
    assert {p for f in records for p in f['periods']} == {1,2,3}
    assert {f['frames'] for f in records} >= {1,2,3}
    assert any(f['repeated_lane'] for f in records)
    assert any(f['inert_blockers'] for f in records)
    assert any(f['intact'] > 1 for f in records)
    assert any(f['size'] == 3 for f in records)
    assert sum(f['no_a_witness'] for f in records) > len(records)//2
    for seed in (0,17,41):
        assert _signature(generate_puzzle(seed)) == _signature(generate_puzzle(seed))
    # The host exercises every legal combination. Keep this internal test
    # bounded to representative count boundaries as well as the 42 seeds.
    for nt,nq in ((2,1),(3,2),(5,4),(10,1),(10,4)):
        p = generate_puzzle(1000+10*nt+nq,nt,nq)
        assert len(p['train']) == nt and len(p['test']) == nq
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(9,2,1)
    pair = bad['test'][0]
    y,x = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][y,x] = pair['input'][y,x]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted developed pixel")
    return {'seeds':42,'pairs':len(records),'stencils':len({f['stencil'] for f in records}),
            'shapes':len({f['shape'] for f in records}),
            'frame_configurations':len({f['frame_shapes'] for f in records})}


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
