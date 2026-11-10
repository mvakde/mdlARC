"""Repair-bearing collapse: solid repair halos participate in rigid gravity.

The header's last cell visibly supplies the highlight color. Other non-background
header cells select shoring-stick identities for removal. No training labels or
generation state are needed by solve().
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(not 1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(a, color, start=2):
    unseen = {tuple(map(int, q)) for q in np.argwhere(a == color) if q[0] >= start}
    result = []
    while unseen:
        q = min(unseen)
        unseen.remove(q)
        todo, cells = [q], {q}
        while todo:
            r, c = todo.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    todo.append(p)
        result.append(cells)
    return result


def _pixels(p, shape, fills, highlight, *, halos=True):
    """Absolute colored footprint of one repaired rigid object."""
    r, c = p['r'], p['c']
    if p['kind'] == 'rod':
        return {(r + (i if p['o'] == 'V' else 0),
                 c + (i if p['o'] == 'H' else 0)): p['color']
                for i in range(p['n'])}
    pixels = {(r+i, c+j): p['color'] for i in range(p['h']) for j in range(p['w'])}
    if halos:
        other = next(x for x in fills if x != p['color'])
        for i, j in p['holes']:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r+i+dr, c+j+dc
                    # Horizontal clipping is permanent because objects never
                    # shift sideways. Vertical clipping is not a physics rule.
                    if 0 <= q[1] < shape[1] and (dr or dc):
                        pixels[q] = other
        for i, j in p['holes']:
            pixels[r+i, c+j] = highlight
    return pixels


def _masks(pieces, shape, fills, highlight, halos=True):
    return [set(_pixels(p, shape, fills, highlight, halos=halos)) for p in pieces]


def _settle(pieces, shape, fills, highlight, *, halos=True):
    """Greatest simultaneously movable set; all pieces translate one row."""
    result = [dict(p) for p in pieces]
    for _ in range(shape[0] * (len(result) + 1)):
        cells = _masks(result, shape, fills, highlight, halos)
        targets = [{(r+1, c) for r, c in s} for s in cells]
        moving = {i for i, s in enumerate(targets) if all(r < shape[0] for r, c in s)}
        while True:
            stationary = set(range(len(result))) - moving
            blocked = {i for i in moving if any(targets[i] & cells[j] for j in stationary)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        for i in moving:
            result[i]['r'] += 1
    raise AssertionError("Gravity failed to terminate")


def _settle_events(pieces, shape, fills, highlight):
    """Cross-check using contact groups and event distances, not unit steps."""
    result = [dict(p) for p in pieces]
    for _ in range(shape[0] * (len(result) + 1)):
        cells = _masks(result, shape, fills, highlight)
        # Build dependencies for immediate downward contact. The same fixed
        # point finds the free group; then jump to its next collision event.
        contacts = [{j for j, b in enumerate(cells) if i != j and
                     {(r+1, c) for r, c in a} & b} for i, a in enumerate(cells)]
        fixed = {i for i, s in enumerate(cells) if any(r == shape[0]-1 for r, c in s)}
        while True:
            more = {i for i in range(len(cells)) if contacts[i] & fixed} - fixed
            if not more:
                break
            fixed |= more
        free = set(range(len(cells))) - fixed
        if not free:
            return result
        distance = min(shape[0]-1-r for i in free for r, c in cells[i])
        obstacles = set().union(*(cells[i] for i in fixed)) if fixed else set()
        for i in free:
            for r, c in cells[i]:
                for rr, cc in obstacles:
                    if cc == c and rr > r:
                        distance = min(distance, rr-r-1)
        assert distance > 0
        for i in free:
            result[i]['r'] += distance
    raise AssertionError("Event gravity failed to terminate")


def _render(pieces, shape, bg, header, fills, highlight):
    a = np.full(shape, bg, dtype=np.int8)
    a[0] = header
    occupied = set()
    for p in pieces:
        pixels = _pixels(p, shape, fills, highlight)
        assert all(2 <= r < shape[0] and 0 <= c < shape[1] for r, c in pixels)
        assert not occupied.intersection(pixels), "Overlapping rigid bodies"
        occupied.update(pixels)
        for q, color in pixels.items():
            a[q] = color
    return a


def _decode(input_grid):
    a = _grid(input_grid)
    assert a.shape[0] >= 8 and a.shape[1] >= 8
    bg = int(a[1, 0])
    assert np.all(a[1] == bg)
    highlight = int(a[0, -1])
    assert highlight != bg and a[0, -2] == bg
    commands = {int(c) for c in a[0, :-2] if c != bg}
    assert 1 <= len(commands) <= 2 and highlight not in commands
    assert not np.any(a[2:] == highlight)
    plates = []
    for color in sorted(set(map(int, np.unique(a[2:]))) - {bg}):
        for cells in _components(a, color):
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            r1, c1 = max(r for r, c in cells)+1, max(c for r, c in cells)+1
            h, w = r1-r0, c1-c0
            if min(h, w) >= 4 and len(cells) >= .65*h*w:
                plates.append(dict(kind='plate', r=r0, c=c0, h=h, w=w, color=color))
    fills = tuple(sorted({p['color'] for p in plates}))
    assert len(fills) == 2 and not commands.intersection(fills)
    assert len(plates) >= 2
    rectangles = set()
    for p in plates:
        r, c, h, w = (p[k] for k in ('r', 'c', 'h', 'w'))
        rect = {(r+i, c+j) for i in range(h) for j in range(w)}
        assert not rect & rectangles
        rectangles |= rect
        assert all(a[q] == p['color'] for q in ((r,c),(r+h-1,c),(r,c+w-1),(r+h-1,c+w-1)))
        assert all(int(a[q]) in (*fills, bg) for q in rect)
        assert len(_components(a[r:r+h,c:c+w],p['color'],start=0)) == 1
        p['holes'] = tuple((i,j) for i in range(h) for j in range(w) if a[r+i,c+j] == bg)
        assert p['holes']
    rods = []
    for color in sorted(set(map(int, np.unique(a[2:]))) - {bg, *fills}):
        for cells in _components(a, color):
            rows, cols = {q[0] for q in cells}, {q[1] for q in cells}
            assert len(cells) >= 2 and (len(rows) == 1 or len(cols) == 1)
            o = 'H' if len(rows) == 1 else 'V'
            p = dict(kind='rod', r=min(rows), c=min(cols), n=len(cells), o=o, color=color)
            assert set(_pixels(p, a.shape, fills, highlight)) == cells
            rods.append(p)
    pieces = sorted(plates, key=lambda p: (p['r'], p['c'])) + rods
    # Repaired geometry is inferred before any support is removed. The input
    # plate colors outside all reconstructed rectangles must be isolated noise.
    for color in fills:
        for cells in _components(a, color):
            if not cells & rectangles:
                assert len(cells) <= 2
    _render(pieces, a.shape, bg, a[0], fills, highlight)
    return a, bg, highlight, commands, fills, pieces


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, bg, highlight, commands, fills, pieces = _decode(input_grid)
    survivors = [p for p in pieces if p['kind'] != 'rod' or p['color'] not in commands]
    settled = _settle(survivors, a.shape, fills, highlight)
    return _render(settled, a.shape, bg, a[0], fills, highlight)


def _plate(rng, h, w, color, cluster=False):
    # Corners remain intact, while erasures can reach all four straight sides.
    candidates = [(i,j) for i in range(h) for j in range(w)
                  if (i,j) not in ((0,0),(0,w-1),(h-1,0),(h-1,w-1))]
    holes = []
    if cluster:
        k = int(rng.integers(2, w-3))
        holes = [(h-1,k), (h-1,k+1)]
    target = int(rng.integers(max(1,len(holes)), min(5,h*w//6)+1))
    rng.shuffle(candidates)
    holes += [q for q in candidates if q not in holes][:target-len(holes)]
    return dict(kind='plate', r=3, c=0, h=h, w=w, color=color, holes=tuple(sorted(holes)))


def _drop(p, pieces, shape, fills, highlight):
    p = dict(p)
    p['r'] = 3
    occupied = set().union(*_masks(pieces, shape, fills, highlight)) if pieces else set()
    mask = set(_pixels(p, shape, fills, highlight))
    if mask & occupied or any(r >= shape[0] for r, c in mask):
        return None
    while True:
        target = {(r+1,c) for r,c in mask}
        if any(r >= shape[0] for r,c in target) or target & occupied:
            return p
        p['r'] += 1
        mask = target


def _rod(r, c, n, color, o='V'):
    return dict(kind='rod', r=r, c=c, n=n, o=o, color=color)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(a):
    a, bg, hi, commands, fills, pieces = _decode(a)
    survivors = [p for p in pieces if p['kind'] == 'plate' or p['color'] not in commands]
    settled = _settle(survivors, a.shape, fills, hi)
    plain = _settle(survivors, a.shape, fills, hi, halos=False)
    distances = [q['r']-p['r'] for p,q in zip(survivors,settled)]
    plates = [p for p in pieces if p['kind'] == 'plate']
    wrong = sum(int(a[p['r']+i,p['c']+j]) not in (bg,p['color'])
                for p in plates for i in range(p['h']) for j in range(p['w']))
    rects = {(p['r']+i,p['c']+j) for p in plates for i in range(p['h']) for j in range(p['w'])}
    speckles = sum(int(a[r,c]) in fills and (r,c) not in rects
                   for r in range(2,a.shape[0]) for c in range(a.shape[1]))
    return dict(shape=a.shape, plates=len(plates), rods=len(pieces)-len(plates),
                sizes=tuple(sorted((p['h'],p['w']) for p in plates)),
                background=bg, highlight=hi, fills=fills, commands=len(commands),
                holes=sum(len(p['holes']) for p in plates), wrong=wrong, speckles=speckles,
                cluster=any(max(abs(i-k),abs(j-l)) == 1 for p in plates
                            for i,j in p['holes'] for k,l in p['holes'] if (i,j)!=(k,l)),
                side_clip=any(p['c']+j in (0,a.shape[1]-1) for p in plates for i,j in p['holes']),
                moved=sum(d>0 for d in distances), distances=tuple(sorted(set(distances))),
                halo_changes=any(p['r'] != q['r'] for p,q in zip(settled,plain)),
                partial=any(d==0 for d,p in zip(distances,survivors) if p['kind']=='plate'),
                layout=tuple((p['kind'],p['r'],p['c'],p.get('h'),p.get('w'),p.get('n'),p.get('o'))
                             for p in pieces))


def _make_pair(rng, count, commands_count, bg, hi, *, rich=False, require_wrong=False):
    available = [c for c in range(10) if c not in (bg,hi)]
    for _ in range(180):
        fills = tuple(map(int,rng.choice(available,2,replace=False)))
        support = list(map(int,rng.choice([c for c in available if c not in fills],3,replace=False)))
        selected = support[:commands_count]
        catch_color = support[-1]
        h = int(rng.integers(18 if count==2 else 25,31))
        w = int(rng.integers(13 if not rich else 18,31))
        shape = h,w
        ph = int(rng.integers(4,7))
        pw = int(rng.integers(7,min(13,w)+1))
        x = int(rng.integers(0,w-pw+1))
        tall = int(rng.integers(4,min(9,h-ph-6)))
        primary = _plate(rng,ph,pw,fills[0],cluster=True)
        # Keep the mandatory two lower-edge holes as the only lower protrusion
        # positions used to locate the shorter catch; other holes may decorate
        # any side and can create further contacts.
        primary.update(r=h-tall-ph,c=x)
        lower = sorted(j for i,j in primary['holes'] if i==ph-1)
        landing_x = x + int(rng.choice(lower))
        protrusion_cols = {c for r,c in _pixels(primary,shape,fills,hi) if r==h-tall}
        leg_cols = [c for c in range(x,x+pw) if c not in protrusion_cols]
        if len(leg_cols) < commands_count:
            continue
        rng.shuffle(leg_cols)
        leg_count = commands_count if commands_count==2 else int(rng.integers(1,min(3,len(leg_cols))+1))
        legs = sorted(leg_cols[:leg_count])
        pieces = [_rod(h-tall,c,tall,selected[j%commands_count]) for j,c in enumerate(legs)]
        short = int(rng.integers(2,tall-1))
        pieces.append(_rod(h-short,landing_x,short,catch_color))
        # Additional shorter catches vary landing surfaces and redundant paths.
        if rng.random() < .55:
            options = [c for c in range(x,x+pw) if min(abs(c-v) for v in legs+[landing_x]) >= 2]
            if options:
                xx = int(rng.choice(options)); n = int(rng.integers(2,tall-1))
                pieces.append(_rod(h-n,xx,n,catch_color))
        pieces.append(primary)
        ok = True
        for index in range(1,count):
            found = None
            for _ in range(24):
                rh = int(rng.integers(4,7 if count<4 else 6))
                rw = int(rng.integers(4,min(12,w)+1))
                p = _plate(rng,rh,rw,fills[index%2])
                # Free offsets with actual overlap in horizontal projection;
                # every plate belongs to the same support network.
                anchor = pieces[-1]
                if anchor['kind'] != 'plate':
                    anchor = primary
                lo = max(0,anchor['c']-rw+2)
                high = min(w-rw,anchor['c']+anchor['w']-2)
                if lo > high:
                    continue
                p['c'] = int(rng.integers(lo,high+1))
                p = _drop(p,pieces,shape,fills,hi)
                if p is None:
                    continue
                # Base rectangles of a reused fill cannot merge in the input.
                rect = {(p['r']+i,p['c']+j) for i in range(p['h']) for j in range(p['w'])}
                near = rect | {(r+dr,c+dc) for r,c in rect for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))}
                if any(near & {(q['r']+i,q['c']+j) for i in range(q['h']) for j in range(q['w'])}
                       for q in pieces if q['kind']=='plate' and q['color']==p['color']):
                    continue
                # Reject separate floor piles: there must be a plate directly
                # beneath the new body, possibly together with a shoring rod.
                target = {(r+1,c) for r,c in _pixels(p,shape,fills,hi)}
                if not any(target & set(_pixels(q,shape,fills,hi)) for q in pieces if q['kind']=='plate'):
                    continue
                found = p
                break
            if found is None:
                ok = False
                break
            pieces.append(found)
        if not ok:
            continue
        # A floor post under an upper overhang creates a redundant, branching
        # support path. It can hold an upper plate while the lower one drops.
        if rng.random() < (.65 if rich else .45):
            masks = _masks(pieces,shape,fills,hi)
            occupied = set().union(*masks)
            options = []
            for c in range(w):
                if any(p['kind']=='rod' and p['c']==c for p in pieces):
                    continue
                hits = [(r,i) for i,p in enumerate(pieces) if p['kind']=='plate'
                        for r,cc in masks[i] if cc==c]
                if not hits:
                    continue
                bottom,i = max(hits)
                available_length = h-bottom-1
                if pieces[i] is primary or available_length<2:
                    continue
                # A zero gap holds the upper plate. A small positive gap lets
                # it descend partway and then stop while the lower plate keeps
                # falling: two positive displacements in one connected pile.
                gap = int(rng.integers(0,min(3,available_length-2)+1))
                n = available_length-gap
                rod = _rod(bottom+1+gap,c,n,catch_color)
                if not set(_pixels(rod,shape,fills,hi)) & occupied:
                    options.append(rod)
            if options:
                pieces.append(options[int(rng.integers(len(options)))])
        # Optional horizontal shoring rests on an exposed ledge and can be
        # deleted with a floor leg, reusing an identity in a different role.
        if rich and rng.random() < .55:
            n = int(rng.integers(2,min(7,w)+1))
            rod = _rod(3,int(rng.integers(0,w-n+1)),n,int(rng.choice(support)),o='H')
            rod = _drop(rod,pieces,shape,fills,hi)
            target = {(r+1,c) for r,c in _pixels(rod,shape,fills,hi)} if rod is not None else set()
            if rod is not None and any(target & set(_pixels(p,shape,fills,hi)) for p in pieces):
                pieces.append(rod)
        header = np.full(w,bg,dtype=np.int8)
        header[-1] = hi
        for c,color in zip(rng.choice(w-2,commands_count,replace=False),selected):
            header[int(c)] = color
        try:
            _render(pieces,shape,bg,header,fills,hi)
            assert [p['r'] for p in _settle(pieces,shape,fills,hi)] == [p['r'] for p in pieces]
            inp = np.full(shape,bg,dtype=np.int8); inp[0] = header
            for p in pieces:
                if p['kind']=='rod':
                    for q,color in _pixels(p,shape,fills,hi).items():
                        inp[q] = color
                else:
                    r,c,hh,ww = (p[k] for k in ('r','c','h','w'))
                    inp[r:r+hh,c:c+ww] = p['color']
                    for i,j in p['holes']:
                        inp[r+i,c+j] = bg
                    candidates = [(i,j) for i in range(hh) for j in range(ww)
                                  if (i,j) not in p['holes'] and
                                  (i,j) not in ((0,0),(0,ww-1),(hh-1,0),(hh-1,ww-1))]
                    rng.shuffle(candidates)
                    noise_count = int(rng.integers(1 if require_wrong else 0,3))
                    for i,j in candidates[:noise_count]:
                        inp[r+i,c+j] = next(z for z in fills if z!=p['color'])
            rect = {(p['r']+i,p['c']+j) for p in pieces if p['kind']=='plate'
                    for i in range(p['h']) for j in range(p['w'])}
            near = rect | {(r+dr,c+dc) for r,c in rect for dr in (-1,0,1) for dc in (-1,0,1)}
            exterior = [(r,c) for r in range(2,h) for c in range(w) if inp[r,c]==bg and (r,c) not in near]
            rng.shuffle(exterior)
            used = set()
            for q in exterior:
                if len(used) >= int(2+rich*2):
                    break
                if all(max(abs(q[0]-r),abs(q[1]-c))>1 for r,c in used):
                    inp[q] = int(rng.choice(fills)); used.add(q)
            parsed = _decode(inp)[-1]
            signature = lambda ps: sorted((p['kind'],p['r'],p['c'],p.get('h',0),p.get('w',0),
                                           p.get('n',0),p.get('o',''),p['color'],p.get('holes',())) for p in ps)
            assert signature(parsed)==signature(pieces), "Damage obscured an object's identity"
            survivors = [p for p in pieces if p['kind']=='plate' or p['color'] not in selected]
            settled = _settle(survivors,shape,fills,hi)
            out = _render(settled,shape,bg,header,fills,hi)
            f = _features(inp)
            assert f['moved'] and f['halo_changes'] and f['cluster']
            assert not require_wrong or f['wrong']
            assert np.array_equal(solve(inp),out)
            return {'input':inp,'output':out}
        except AssertionError:
            continue
    raise RuntimeError("Could not construct a recoverable supported scene")


def _integer(value, low, high, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low<=value<=high:
        raise ValueError(f"Illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _integer(num_train,2,10,'num_train')
    num_test = _integer(num_test,1,4,'num_test')
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    bg,hi = map(int,rng.choice(10,2,replace=False))
    puzzle = {'train':[],'test':[]}
    seen = set()
    command_order = [1,2]
    rng.shuffle(command_order)
    for split,total in (('train',num_train),('test',num_test)):
        for index in range(total):
            if split=='train' and index<2:
                count,commands = 2+index,command_order[index]
            else:
                count = int(rng.integers(2 if split=='train' else 3,5))
                commands = int(rng.integers(1,3))
            for _ in range(8):
                pair = _make_pair(rng,count,commands,bg,hi,rich=split=='test' or index>=2,
                                  require_wrong=split=='train' and index<2)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("Distinct example construction failed")
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2<=len(puzzle['train'])<=10 and 1<=len(puzzle['test'])<=4
    assert len(puzzle['train'])>len(puzzle['test'])
    seen, conventions, training = set(),set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            a,out = _grid(pair['input']),_grid(pair['output'])
            assert a.shape==out.shape and _key(a) not in seen
            seen.add(_key(a))
            a,bg,hi,commands,fills,pieces = _decode(a)
            conventions.add((bg,hi))
            assert 2<=sum(p['kind']=='plate' for p in pieces)<=4
            assert [p['r'] for p in _settle(pieces,a.shape,fills,hi)]==[p['r'] for p in pieces]
            survivors = [p for p in pieces if p['kind']=='plate' or p['color'] not in commands]
            assert len(survivors)<len(pieces)
            settled = _settle(survivors,a.shape,fills,hi)
            assert all(q['r']>=p['r'] for p,q in zip(survivors,settled))
            assert [p['r'] for p in _settle(settled,a.shape,fills,hi)]==[p['r'] for p in settled]
            assert np.array_equal(out,_render(settled,a.shape,bg,a[0],fills,hi))
            assert np.array_equal(out[:2],a[:2])
            assert not np.any(np.isin(out[2:],list(commands)))
            assert np.count_nonzero(out[2:]==hi)==sum(len(p['holes']) for p in pieces if p['kind']=='plate')
            f = _features(a)
            assert f['moved'] and f['halo_changes']
            if split=='train':
                training.append(f)
    assert len(conventions)==1
    assert {f['commands'] for f in training}=={1,2}
    assert any(f['plates']>=3 for f in training)
    assert any(f['cluster'] and f['wrong'] and f['speckles'] for f in training)


def self_test():
    # Hand-calculated diagnostic: lower-edge repairs stop the whole plate one
    # row sooner than its bare rectangle. Adjacent centers stay highlighted.
    plate = dict(kind='plate',r=8,c=2,h=4,w=7,color=1,holes=((3,3),(3,4)))
    catch = _rod(16,5,2,4)
    bodies = [plate,catch]
    assert _settle(bodies,(18,12),(1,2),8)[0]['r']==11
    assert _settle(bodies,(18,12),(1,2),8,halos=False)[0]['r']==12
    pixels = _pixels(plate,(18,12),(1,2),8)
    assert pixels[11,5]==pixels[11,6]==8 and pixels[12,5]==2
    clipped = dict(kind='plate',r=3,c=0,h=4,w=4,color=1,holes=((1,0),(1,1)))
    pixels = _pixels(clipped,(12,8),(1,2),8)
    assert all(c>=0 for r,c in pixels) and pixels[4,0]==pixels[4,1]==8
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = []
    signatures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train','test') for p in puzzle[s]))
        for pair in puzzle['train']+puzzle['test']:
            a,bg,hi,commands,fills,pieces = _decode(pair['input'])
            survivors = [p for p in pieces if p['kind']=='plate' or p['color'] not in commands]
            unit = _settle(survivors,a.shape,fills,hi)
            event = _settle_events(survivors,a.shape,fills,hi)
            assert [p['r'] for p in unit]==[p['r'] for p in event]
            assert np.array_equal(solve(a.tolist()),pair['output'])
            records.append(_features(a))
    assert len(signatures)==42
    assert len({f['shape'] for f in records})>=45
    assert {f['plates'] for f in records}=={2,3,4}
    assert len({f['sizes'] for f in records})>=100
    assert len({f['layout'] for f in records})>=180
    assert len({f['background'] for f in records})>=8
    assert len({f['highlight'] for f in records})>=8
    assert len({f['fills'] for f in records})>=25
    assert len({f['rods'] for f in records})>=4
    assert any(f['side_clip'] for f in records)
    assert any(f['partial'] for f in records)
    assert any(len([d for d in f['distances'] if d>0])>=2 for f in records)
    assert all(f['halo_changes'] for f in records)
    first,second = generate_puzzle(808,2,1),generate_puzzle(808,2,1)
    assert all(np.array_equal(p[k],q[k]) for s in first for p,q in zip(first[s],second[s]) for k in p)
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(17,nt,ne)
            assert len(p['train'])==nt and len(p['test'])==ne
    for nt,ne in ((1,1),(11,1),(2,0),(10,5),(2,2),(3,4),(True,1),(3,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s:[{k:np.array(v).copy() for k,v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][1,0] = (int(bad['test'][0]['output'][1,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
