"""Relation-controlled replacement of damaged terminal routes.

Only the standard library and NumPy are used. No generation state is consulted
by solve. Diagnostics are streamed by self_test; it writes no files.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np


D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0,379264354,603845676,370260233,543238205,528190881,237051595,632601224,225385108,133303505,168755112,84681737,975004950,235688566,40192075,287491844,967763288,955383796,561894583,650355682,837942942,967372132,591313802,177709007,85773102,145462846,586550027,706708163,922654138,629149422,385372602,19531680,929585748,107258407,47954147,27631931,340647583,935388052,805088990,791789286,467903879,910555690)


def _add(p, d, n=1):
    return p[0] + n*d[0], p[1] + n*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _edge(p, shape):
    return p[0] in (0, shape[0]-1) or p[1] in (0, shape[1]-1)


def _support(p):
    return {(p[0]+r, p[1]+c) for r in (-1,0,1) for c in (-1,0,1)}


def _neighbors(p, cells):
    return [_add(p,d) for d in D4 if _add(p,d) in cells]


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
    return [_add(a,d,i) for i in range(max(abs(a[0]-b[0]),abs(a[1]-b[1]))+1)]


def _poly(points):
    result = [points[0]]
    for a,b in zip(points,points[1:]):
        result.extend(_line(a,b)[1:])
    return result


def _ray(p, d, shape):
    cells = set()
    p = _add(p,d)
    while _inside(p,shape):
        cells.add(p)
        p = _add(p,d)
    return cells


def _components(cells):
    todo, parts = set(cells), []
    while todo:
        p = min(todo)
        todo.remove(p)
        part, stack = {p}, [p]
        while stack:
            for q in _neighbors(stack.pop(),todo):
                todo.remove(q)
                part.add(q)
                stack.append(q)
        parts.append(part)
    return parts


def _simple(walk):
    cells = set(walk)
    return len(cells) == len(walk) and all(
        len(_neighbors(p,cells)) == (1 if i in (0,len(walk)-1) else 2)
        for i,p in enumerate(walk))


def _turns(walk):
    ds = [(b[0]-a[0],b[1]-a[1]) for a,b in zip(walk,walk[1:])]
    return sum(a != b for a,b in zip(ds,ds[1:]))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype,np.integer) and np.all((0 <= a)&(a <= 9))
    return a.astype(np.int8,copy=True)


def _tokens(inp):
    counts = np.bincount(inp.ravel(),minlength=10)
    bg = int(counts.argmax())
    assert counts[bg]*2 > inp.size, 'background must occupy a strict majority'
    tokens, used = [], set()
    for r in range(1,inp.shape[0]-1):
        for c in range(1,inp.shape[1]-1):
            p = r,c
            body = int(inp[r-1,c-1])
            if body == bg or int(inp[p]) == bg:
                continue
            cells = _support(p)
            if all(int(inp[q]) == body for q in cells-{p}):
                assert not cells & used, 'overlapping token interpretations'
                used.update(cells)
                tokens.append({'center':p,'body':body,'value':int(inp[p]),'cells':cells})
    assert 3 <= len(tokens) <= 12, 'expected separated 3 by 3 relation tokens'
    mapping = {}
    for t in tokens:
        b,v = t['body'],t['value']
        assert b not in mapping or mapping[b] == v, 'inconsistent duplicate donors'
        mapping[b] = v
    identities = {v for t in tokens for v in (t['body'],t['value'])}
    extras = set(map(int,np.unique(inp)))-identities-{bg}
    assert len(extras) == 1, 'one geometrically separate repair color is required'
    repair = extras.pop()
    return bg,repair,tokens,used,identities


def _reconstruct(inp, bg, repair, token_cells, identities):
    """Markers are waypoints; unmarked omissions are isolated straight holes."""
    work = inp.copy()
    for p in token_cells:
        work[p] = bg
    pending = {tuple(map(int,p)) for p in np.argwhere(work == repair)}
    marker_cells = set(pending)
    for _ in range(len(pending)+1):
        changed = False
        for p in sorted(pending):
            adjacent = {int(work[q]) for d in D4 if _inside(q := _add(p,d),inp.shape)
                        and int(work[q]) in identities}
            if len(adjacent) == 1:
                work[p] = adjacent.pop()
                pending.remove(p)
                changed = True
        if not changed:
            break
    assert not pending, 'repair marker lacks a unique neighboring trace identity'
    holes = {}
    # One simultaneous pass, not iterative extrapolation or joining distant ends.
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            p = r,c
            if work[p] != bg or p in token_cells:
                continue
            colors = set()
            for d in ((1,0),(0,1)):
                a,b = _add(p,d),_add(p,d,-1)
                if _inside(a,inp.shape) and _inside(b,inp.shape):
                    if int(work[a]) == int(work[b]) and int(work[a]) in identities:
                        colors.add(int(work[a]))
            assert len(colors) <= 1, 'ambiguous omitted crossing'
            if colors:
                holes[p] = colors.pop()
    for p,color in holes.items():
        work[p] = color
    layers = {color:{tuple(map(int,p)) for p in np.argwhere(work == color)} for color in identities}
    layers = {c:s for c,s in layers.items() if s}
    assert layers
    for cells in layers.values():
        for part in _components(cells):
            degrees = [len(_neighbors(p,part)) for p in part]
            assert len(part) >= 3 and degrees.count(1) == 2 and max(degrees) <= 2, 'traces must be open simple paths'
    return layers,marker_cells,set(holes)


def _candidates(token, layers, shape):
    candidates = []
    p = token['center']
    for d in D4:
        pivot,first = _add(p,d,2),_add(p,d,3)
        for color,cells in layers.items():
            if pivot not in cells or first not in cells:
                continue
            ns = _neighbors(pivot,cells)
            side = [q for q in ns if q != first]
            if len(ns) != 2 or len(side) != 1:
                continue
            v = side[0][0]-pivot[0],side[0][1]-pivot[1]
            if v[0]*d[0] + v[1]*d[1] != 0:
                continue
            previous,current,walk = pivot,first,[pivot]
            while current not in walk:
                walk.append(current)
                following = [q for q in _neighbors(current,cells) if q != previous]
                if not following:
                    if len(walk) >= 4 and _edge(current,shape):
                        direction = -d[0],-d[1]
                        candidates.append({'pivot':pivot,'color':color,'old':set(walk[1:]),
                                           'walk':tuple(walk),'direction':direction,
                                           'new':_ray(pivot,direction,shape),'body':token['body'],
                                           'token':p})
                    break
                if len(following) != 1:
                    break
                previous,current = current,following[0]
    return candidates


def _interpret(inp):
    bg,repair,tokens,used,identities = _tokens(inp)
    layers,markers,holes = _reconstruct(inp,bg,repair,used,identities)
    edits = []
    for i,t in enumerate(tokens):
        donors = [j for j,u in enumerate(tokens) if i != j and u['body'] == t['value']]
        t['donors'] = donors
        t['candidates'] = _candidates(t,layers,inp.shape)
        values = {tokens[j]['value'] for j in donors}
        assert len(values) <= 1
        target = next(iter(values)) if values else None
        matched = [e for e in t['candidates'] if e['color'] == target]
        assert len(matched) <= 1, 'multiple addressed pivots'
        t['target'] = target
        t['edit'] = matched[0] if matched else None
        if matched:
            edits.append(matched[0])
    assert 1 <= len(edits) <= 3
    for i,a in enumerate(edits):
        for b in edits[i+1:]:
            assert a['color'] != b['color'] or not a['old'] & b['old']
            assert not a['new'] & b['new'] or a['body'] == b['body'], 'conflicting new rays'
            assert a['pivot'] not in b['old'] and b['pivot'] not in a['old']
    return bg,repair,tokens,used,layers,markers,holes,edits


def _render(shape, bg, layers, edits):
    kept = {c:set(s) for c,s in layers.items()}
    for e in edits:
        kept[e['color']].difference_update(e['old'])
    out = np.full(shape,bg,dtype=np.int8)
    for color,cells in kept.items():
        for p in cells:
            out[p] = color
    for e in edits:
        for p in e['new']:
            out[p] = e['body']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Interpret tokens, reconstruct traces, select routes, then edit simultaneously."""
    inp = _grid(input_grid)
    bg,_,_,_,layers,_,_,edits = _interpret(inp)
    return _render(inp.shape,bg,layers,edits)


def _compatible(walk, occupied, blocked, shape):
    s = set(walk)
    return (_simple(walk) and all(_inside(p,shape) for p in s) and not s & blocked
            and not s & occupied and not any(_neighbors(p,occupied) for p in s))


def _route(rng, center, d, shape, occupied, blocked, require_bend=False):
    """A randomly spaced 0/1/2-bend boundary arm plus a separate retained arm."""
    pivot = _add(center,d,2)
    axis = 0 if d[0] else 1
    for _ in range(60):
        side = (-d[1],d[0])
        if rng.integers(2):
            side = -side[0],-side[1]
        length = int(rng.integers(2,7))
        retained_points = [pivot,_add(pivot,side,length)]
        if rng.random() < .45:
            retained_points.append(_add(retained_points[-1],d,int(rng.integers(3,7))))
        bends = int(rng.integers(1 if require_bend else 0,3))
        edge_value = 0 if d[axis] < 0 else shape[axis]-1
        if bends == 0:
            endpoint = list(pivot)
            endpoint[axis] = edge_value
            old_points = [pivot,tuple(endpoint)]
        else:
            q = _add(pivot,d,int(rng.integers(3,8)))
            if bends == 1:
                end = list(q)
                end[1-axis] = 0 if side[1-axis] < 0 else shape[1-axis]-1
                old_points = [pivot,q,tuple(end)]
            else:
                q2 = _add(q,side,int(rng.integers(3,8)))
                end = list(q2)
                end[axis] = edge_value
                old_points = [pivot,q,q2,tuple(end)]
        old,retained = _poly(old_points),_poly(retained_points)
        walk = list(reversed(old))+retained[1:]
        if len(old) < 4 or not _compatible(walk,occupied,blocked,shape):
            continue
        # The branch first leaves away from the switch and ends on a boundary.
        return set(walk),{'pivot':pivot,'old':set(old[1:]),'walk':tuple(old),
                          'new':_ray(pivot,(-d[0],-d[1]),shape),
                          'direction':(-d[0],-d[1])}
    return None


def _make_pair(rng, roles, mode, dense=False, crossing=False):
    for _ in range(160):
        nactive = 1 if mode == 'single' else 2
        h = int(rng.integers(17 if nactive == 1 else 23,31))
        w = int(rng.integers(17 if nactive == 1 else 23,31))
        shape = h,w
        palette = list(map(int,rng.permutation(10)))
        if roles is not None:
            palette = list(roles)+[int(v) for v in rng.permutation([c for c in range(10) if c not in roles])]
        bg,repair = palette[:2]
        a,b,c,d,e = palette[2:7]
        if mode == 'shared':
            relations = [(a,b),(a,b),(b,c),(c,d),(d,e),(e,b)]
            targets,decoys = [c,c],[d,d]
        else:
            relations = [(a,b),(b,c),(c,d),(d,e),(e,b)]
            targets,decoys = ([c],[d]) if nactive == 1 else ([c,d],[d,e])
        # Mutual relations make the selected incoming color equal the outgoing
        # body. This changes equality structure, not the routing convention.
        if mode in ('single','shared') and rng.random() < .3:
            relations = ([(a,b)]*nactive)+[(b,a),(c,d),(d,e),(e,c)]
            targets,decoys = [a]*nactive,[b]*nactive
        for extra in palette[7:7+int(rng.integers(4))]:
            relations.append((extra,int(rng.choice(palette[2:]))))
        for _ in range(int(rng.integers(3))):
            relations.append(relations[int(rng.integers(len(relations)))])
        centers = []
        for i in range(nactive):
            options = [(r,col) for r in range(5,h-5) for col in range(5,w-5)
                       if all(max(abs(r-u),abs(col-v)) >= 9 for u,v in centers)]
            if not options:
                break
            centers.append(options[int(rng.integers(len(options)))])
        if len(centers) != nactive:
            continue
        blocked = set().union(*(_support(p) for p in centers))
        layers,edits,occupied = {},[],set()
        good = True
        for i,p in enumerate(centers):
            count = 3 if rng.random() < (.3 if nactive == 1 else .12) else 2
            dirs = [D4[int(v)] for v in rng.choice(4,count,replace=False)]
            plans = [(targets[i],dirs[0],True),(decoys[i],dirs[1],False)]
            if count == 3:
                other = int(rng.choice([v for v in (a,b,c,d,e) if v not in (targets[i],decoys[i])]))
                plans.append((other,dirs[2],False))
            rng.shuffle(plans)
            for color,side,chosen in plans:
                routed = _route(rng,p,side,shape,occupied,blocked,
                                require_bend=dense and chosen)
                if routed is None:
                    good = False
                    break
                cells,edit = routed
                layers.setdefault(color,set()).update(cells)
                occupied.update(cells)
                if chosen:
                    edit.update(color=color,body=relations[i][0],token=p)
                    edits.append(edit)
            if not good:
                break
        if not good:
            continue
        if any(ae['new'] & be['new'] and ae['body'] != be['body']
               for i,ae in enumerate(edits) for be in edits[i+1:]):
            continue
        if any(ed['new'] & _support(p) for ed in edits for p in centers if p != ed['token']):
            continue
        if any(ae['pivot'] in be['old'] for ae in edits for be in edits if ae is not be):
            continue
        # A transverse intact path witnesses that new rays cover retained traces.
        if crossing or rng.random() < .55:
            success = False
            for ed in edits:
                points = sorted(p for p in ed['new'] if
                                max(abs(p[0]-ed['token'][0]),abs(p[1]-ed['token'][1])) >= 4)
                rng.shuffle(points)
                perp = -ed['direction'][1],ed['direction'][0]
                for p in points:
                    walk = _line(_add(p,perp,-int(rng.integers(2,5))),
                                 _add(p,perp,int(rng.integers(2,5))))
                    if _compatible(walk,occupied,blocked,shape):
                        colors = [v for v in (b,c,d,e) if v != ed['body']]
                        color = int(rng.choice(colors))
                        layers.setdefault(color,set()).update(walk)
                        occupied.update(walk)
                        success = True
                        break
                if success:
                    break
            if crossing and not success:
                continue
        # Detached tokens are evidence for the same routing problem. They can
        # donate to several switches, but may not accidentally become switches.
        for _relation in relations[nactive:]:
            options = []
            for r in range(1,h-1):
                for col in range(1,w-1):
                    p = r,col
                    if any(max(abs(r-u),abs(col-v)) < 4 for u,v in centers):
                        continue
                    if _support(p) & occupied or any(_add(p,di,2) in occupied for di in D4):
                        continue
                    options.append(p)
            if not options:
                good = False
                break
            centers.append(options[int(rng.integers(len(options)))])
        if not good:
            continue
        inp = np.full(shape,bg,dtype=np.int8)
        for color,cells in layers.items():
            for p in cells:
                inp[p] = color
        for p,(body,value) in zip(centers,relations):
            for q in _support(p):
                inp[q] = body
            inp[p] = value
        all_token_cells = set().union(*(_support(p) for p in centers))
        # Accept corruption only if the input-only reconstruction recovers the
        # independently constructed, uncorrupted trace layers exactly.
        sites = sorted(occupied)
        rng.shuffle(sites)
        removed = set().union(*(ed['old'] for ed in edits))
        if dense:
            sites.sort(key=lambda p:p not in removed)
        successes = 0
        desired = int(rng.integers(4 if dense else 2,9 if dense else 6))
        identities = {v for rel in relations for v in rel}
        for p in sites:
            previous = int(inp[p])
            inp[p] = repair
            try:
                recovered,_,_ = _reconstruct(inp,bg,repair,all_token_cells,identities)
                assert recovered == layers
            except AssertionError:
                inp[p] = previous
            else:
                successes += 1
                if successes == desired:
                    break
        if successes < 2:
            continue
        gap_success = 0
        rng.shuffle(sites)
        desired_gaps = int(rng.integers(2 if dense else 1,6 if dense else 3))
        for p in sites:
            if inp[p] == repair:
                continue
            previous = int(inp[p])
            inp[p] = bg
            try:
                recovered,_,_ = _reconstruct(inp,bg,repair,all_token_cells,identities)
                assert recovered == layers
            except AssertionError:
                inp[p] = previous
            else:
                gap_success += 1
                if gap_success == desired_gaps:
                    break
        if gap_success == 0:
            continue
        out = _render(shape,bg,layers,edits)
        try:
            analysis = _interpret(inp)
            actual_edits = analysis[-1]
            assert {(ed['token'],ed['pivot'],ed['color']) for ed in actual_edits} == {
                (ed['token'],ed['pivot'],ed['color']) for ed in edits}
            assert np.array_equal(solve(inp),out)
            f = _features(inp,analysis)
            assert f['active'] == nactive and f['choice_tokens'] == nactive
            assert not crossing or f['ray_crossings'] > 0
            assert not (crossing and not dense) or f['retained_markers'] > 0
            assert not dense or (f['old_bends'] > 0 and f['deleted_markers'] > 0)
            assert f['nonrecursive_decoys'] == nactive
        except AssertionError:
            continue
        # All geometry, including donor placement, is transformed together.
        k,reflect = int(rng.integers(4)),bool(rng.integers(2))
        return {role:np.rot90(np.fliplr(g) if reflect else g,k).copy()
                for role,g in (('input',inp),('output',out))}
    raise RuntimeError('bounded route construction exhausted')


def _features(inp, analysis=None):
    bg,repair,tokens,used,layers,markers,holes,edits = analysis or _interpret(inp)
    active = [t for t in tokens if t['edit'] is not None]
    active_centers = {t['center'] for t in active}
    mapping = {t['body']:t['value'] for t in tokens}
    removed = set().union(*(ed['old'] for ed in edits))
    kept = {c:s-set().union(*(e['old'] for e in edits if e['color'] == c)) for c,s in layers.items()}
    return {'shape':tuple(map(int,inp.shape)),'background':bg,'repair':repair,'tokens':len(tokens),
            'active':len(edits),'paths':sum(len(_components(s)) for s in layers.values()),
            'candidate_counts':tuple(len(t['candidates']) for t in active),
            'same_carrier':sum(t['body'] == t['target'] for t in active),
            'choice_tokens':sum(len(t['candidates']) >= 2 for t in active),
            'remote_donors':sum(any(tokens[j]['center'] not in active_centers for j in t['donors']) for t in active),
            'active_donors':sum(any(tokens[j]['center'] in active_centers for j in t['donors']) for t in active),
            'shared_donors':sum(sum(j in t['donors'] for t in active) > 1 for j in range(len(tokens))),
            'nonrecursive_decoys':sum(any(e['color'] == mapping.get(t['target']) and e['color'] != t['target']
                                        for e in t['candidates']) for t in active),
            'compatible_inactive':sum(bool(t['donors']) and t['edit'] is None for t in tokens),
            'markers':len(markers),'gaps':len(holes),'deleted_markers':len(markers & removed),
            'retained_markers':len(markers-removed),'old_bends':sum(_turns(e['walk']) for e in edits),
            'old_lengths':tuple(len(e['old']) for e in edits),
            'ray_lengths':tuple(len(e['new']) for e in edits),
            'ray_crossings':sum(sum(p in s and color != e['body'] for color,s in kept.items())
                                for e in edits for p in e['new']),
            'directions':tuple(e['direction'] for e in edits),
            'layout':tuple(sorted(t['center'] for t in tokens))}


def _counts(num_train,num_test):
    for name,value,low,high in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train),int(num_test)


def _key(grid):
    a = np.asarray(grid,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    roles = tuple(map(int,rng.choice(10,2,replace=False))) if rng.random() < .4 else None
    puzzle,seen = {'train':[],'test':[]},set()
    modes = ['single','chain']+[str(rng.choice(['single','chain','shared'])) for _ in range(nt-2)]
    for split,count in (('train',nt),('test',ne)):
        for i in range(count):
            # Test combines the two demonstrated interaction conditions with
            # shared-donor fan-out or an active donor, and denser trace damage.
            mode = modes[i] if split == 'train' else str(rng.choice(['chain','shared']))
            dense = split == 'test' or i == 1
            crossing = split == 'test' or i == 0
            for _ in range(8):
                pair = _make_pair(rng,roles,mode,dense=dense,crossing=crossing)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not construct distinct inputs')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,features = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            analysis = _interpret(inp)
            bg,repair,tokens,used,layers,markers,holes,edits = analysis
            predicted = _render(inp.shape,bg,layers,edits)
            assert np.array_equal(predicted,out), 'incorrect repair, lookup, deletion, ray or preserved pixel'
            assert repair not in out, 'annotations must disappear'
            assert all(int(out[e['pivot']]) == e['color'] or any(e['pivot'] in q['new'] for q in edits) for e in edits)
            # Editing the order cannot alter any output, even when a new ray
            # passes through a terminal branch deleted by another switch.
            assert np.array_equal(_render(inp.shape,bg,layers,list(reversed(edits))),out)
            f = _features(inp,analysis)
            assert f['choice_tokens'] == f['active'] and f['nonrecursive_decoys'] == f['active']
            assert f['compatible_inactive'] and f['gaps'] and f['markers'] >= 2
            # Parent ablations: no relational selection leaves old routes;
            # ignoring branch replacement leaves them beside the new rays.
            assert not np.array_equal(_render(inp.shape,bg,layers,[]),out)
            # The one-extra-lookup alternative chooses the explicitly present
            # decoy arm, while all B repair/edit semantics stay the same.
            mapping = {t['body']:t['value'] for t in tokens}
            wrong_edits = []
            for t in tokens:
                if t['edit'] is not None:
                    options = [ed for ed in t['candidates'] if ed['color'] == mapping[t['target']]]
                    assert len(options) == 1
                    wrong_edits.append(options[0])
            assert not np.array_equal(_render(inp.shape,bg,layers,wrong_edits),out), 'relational lookup depth must matter'
            keep_old = _render(inp.shape,bg,layers,[])
            for ed in edits:
                for p in ed['new']:
                    keep_old[p] = ed['body']
            assert not np.array_equal(keep_old,out), 'deleting the old branch must matter'
            features[split].append(f)
    train = features['train']
    assert any(f['remote_donors'] and f['ray_crossings'] for f in train)
    assert any(f['active_donors'] and f['old_bends'] and f['deleted_markers'] for f in train)
    assert any(f['retained_markers'] for f in train)
    assert any(f['active'] == 1 for f in train) and any(f['active'] == 2 for f in train)
    for f in features['test']:
        assert f['active'] == 2 and f['old_bends'] and f['deleted_markers'] and f['ray_crossings']
    return None


def self_test():
    """Bounded streaming checks over the 42 host seeds; no retained grid corpus."""
    stats = Counter()
    shapes,layouts,roles,signatures,directions,token_counts,path_counts,candidate_counts = (set() for _ in range(8))
    first = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp,out = pair['input'],pair['output']
                digest.update(repr(inp.shape).encode())
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                f = _features(inp)
                shapes.add(f['shape'])
                layouts.add((f['shape'],f['layout']))
                roles.add((f['background'],f['repair']))
                token_counts.add(f['tokens'])
                path_counts.add(f['paths'])
                candidate_counts.update(f['candidate_counts'])
                directions.update(f['directions'])
                stats['pairs'] += 1
                for name in ('active_donors','shared_donors','ray_crossings','old_bends',
                             'deleted_markers','retained_markers','nonrecursive_decoys','same_carrier'):
                    stats[name] += f[name]
                assert np.array_equal(solve(np.rot90(inp)),np.rot90(out))
                # Palette equivariance verifies that literal colors carry no roles.
                permutation = np.array([3,7,1,9,0,6,2,8,5,4],dtype=np.int8)
                assert np.array_equal(solve(permutation[inp]),permutation[out])
        signatures.add(digest.hexdigest())
        if first is None:
            first = puzzle
    assert len(signatures) == 42 and len(shapes) >= 45 and len(layouts) >= 180
    assert len(roles) >= 25 and len(token_counts) >= 5 and len(path_counts) >= 3
    assert {2,3} <= candidate_counts
    assert directions == set(D4)
    assert all(stats[name] > 0 for name in stats if name != 'pairs')
    again = generate_puzzle(HOST_SEEDS[0])
    assert all(np.array_equal(a[k],b[k]) for s in ('train','test')
               for a,b in zip(first[s],again[s]) for k in ('input','output'))
    # Full-pixel negative checks include preservation, deleted bends and new rays.
    for split in ('train','test'):
        pair = first[split][0]
        analysis = _interpret(pair['input'])
        edit = analysis[-1][0]
        sites = {min(edit['old']),min(edit['new']),(0,0),edit['pivot']}
        for p in sites:
            bad = {s:[dict(q) for q in first[s]] for s in ('train','test')}
            bad[split][0]['output'] = pair['output'].copy()
            bad[split][0]['output'][p] = (int(pair['output'][p])+1)%10
            try:
                validate_generated_puzzle(bad)
            except (AssertionError,ValueError):
                pass
            else:
                raise AssertionError('corrupt label accepted')
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3,1.5),(None,1)):
        try:
            generate_puzzle(0,*counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    return {'seeds':42,'pairs':stats['pairs'],'shapes':len(shapes),'layouts':len(layouts),
            'role_pairs':len(roles),'token_counts':sorted(token_counts),'path_counts':sorted(path_counts),
            'candidate_counts':sorted(candidate_counts),
            'features':dict(stats),'independent_semantic_check':False}


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
