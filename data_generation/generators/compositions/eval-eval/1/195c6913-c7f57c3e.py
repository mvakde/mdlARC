"""Cyclic signals flip scaffold-registered motifs once per matching contact.

No parent code is imported. Generation renders from a sampled geometric plan;
solve recovers the clock, terrain, templates, scales and contacts from pixels.
"""
from __future__ import annotations

from math import gcd
from functools import reduce
import numpy as np


HOST_SEEDS = (0,290961238,701856087,906177633,725626370,253106265,50811743,
695019695,283501879,366093920,345726956,953635414,862007317,591331991,
459581255,942743228,234342584,326769166,181927006,899551604,568221589,
136155331,97417138,139162901,230069100,767200971,608256151,99621489,
937606262,22930897,448229908,233474225,424739974,380184116,880486160,
767034827,860670245,139338389,245706069,844072166,428103138,578593197)
STEPS = ((-1,0),(1,0),(0,-1),(0,1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(x):
    g = np.asarray(x)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9))
    return g.astype(int, copy=True)


def _orient(g, code):
    return np.rot90(g[:, ::-1] if code >= 4 else g, code % 4).copy()


def _unorient(g, code):
    a = np.rot90(g, -(code % 4))
    return (a[:, ::-1] if code >= 4 else a).copy()


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _trace(mask, start):
    """A forward canvas exit is terminal, even if terrain exists round the corner."""
    p, direction = start, 0
    path, collisions, legs = [], [], []
    for _ in range(sum(mask.shape) + 1):
        dr, dc = ((0,1),(-1,0))[direction]
        leg = []
        while _inside(mask.shape, p) and mask[p]:
            leg.append(p)
            p = (p[0]+dr,p[1]+dc)
        assert leg
        path.extend(leg)
        legs.append(len(leg))
        if not _inside(mask.shape, p):
            end = 'edge'
            break
        collisions.append(p)
        direction = 1-direction
        dr, dc = ((0,1),(-1,0))[direction]
        p = (leg[-1][0]+dr, leg[-1][1]+dc)
        if not _inside(mask.shape, p) or not mask[p]:
            end = 'blocked'
            break
    else:
        raise AssertionError('route failed to terminate')
    assert len(path) == len(set(path))
    return dict(path=path, collisions=collisions, legs=legs, end=end)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    neighbors = [(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            y,x = stack.pop()
            for dy,dx in neighbors:
                q = (y+dy,x+dx)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _bounds(points):
    return min(y for y,x in points), max(y for y,x in points), min(x for y,x in points), max(x for y,x in points)


def _signature(m):
    return tuple(sorted((y,x,int(c)) for (y,x),c in m.items()))


def _primitive(crop):
    lengths = []
    for a in (crop,crop.T):
        for row in a:
            cuts = np.r_[0,np.flatnonzero(row[1:] != row[:-1])+1,len(row)]
            lengths.extend(map(int,np.diff(cuts)))
    scale = reduce(gcd,lengths)
    small = crop[::scale,::scale]
    assert np.array_equal(crop,np.repeat(np.repeat(small,scale,0),scale,1))
    return scale,small


def _separate(a,b):
    y,x,h,w = a
    v,u,k,l = b
    return y+h < v or v+k < y or x+w < u or u+l < x


def _render(g,m,anchor,scale):
    for (r,c),color in m.items():
        y,x = anchor[0]+r*scale, anchor[1]+c*scale
        assert 0 <= y and y+scale <= g.shape[0] and 0 <= x and x+scale <= g.shape[1]
        g[y:y+scale,x:x+scale] = color


def _header(g):
    """In semantic orientation, the clock and isolated marker occupy rows 0,1."""
    assert g.shape[0] >= 8 and g.shape[1] >= 12
    bg = int(g[0,0])
    assert np.array_equal(g[0],g[1]) and np.all(g[2] == bg)
    assert all(int(g[p]) == bg for p in ((0,-1),(-1,0),(-1,-1)))
    swatches = []
    c = 0
    while c < g.shape[1]:
        if g[0,c] == bg:
            c += 1
            continue
        assert c >= 2 and c+2 < g.shape[1]
        color = int(g[0,c])
        assert np.all(g[:2,c:c+2] == color)
        assert g[0,c+2] == bg
        swatches.append((c,color))
        c += 2
    assert 3 <= len(swatches) <= 7
    gaps = [b[0]-a[0]-2 for a,b in zip(swatches,swatches[1:])]
    assert all(1 <= d <= 2 for d in gaps[:-1]) and gaps[-1] >= 4
    word = tuple(color for c,color in swatches[:-1])
    marker = swatches[-1][1]
    assert marker not in word and len(set(word)) >= 2
    starts = [(int(r),0) for r in np.flatnonzero(g[3:,0] == word[0])+3]
    assert len(starts) == 1
    start = starts[0]
    route = int(g[start[0],1])
    assert route not in set(word) | {bg,marker}
    assert all(int(g[r,0]) == bg for r in range(3,g.shape[0]) if r != start[0])
    restored = g.copy()
    restored[:2] = bg
    restored[start] = route
    assert np.count_nonzero(restored == marker) == 0
    record = _trace(restored == route,start)
    assert min(r for r,c in record['path']) >= 4
    return dict(bg=bg,route=route,word=word,marker=marker,start=start,
                restored=restored,record=record,swatches=swatches)


def _motifs(g, info):
    bg,route = info['bg'],info['route']
    objects = []
    for group in _components((g != bg) & (g != route)):
        y0,y1,x0,x1 = _bounds(group)
        crop = g[y0:y1+1,x0:x1+1]
        assert np.count_nonzero((crop != bg) & (crop != route)) == len(group)
        assert not np.any(crop == route)
        scale,small = _primitive(crop)
        mapping = {(int(y),int(x)):int(small[y,x]) for y,x in np.argwhere(small != bg)}
        objects.append(dict(box=(y0,x0),scale=scale,sig=_signature(mapping)))
    assert len(objects) >= 3
    signatures = sorted({o['sig'] for o in objects})
    assert len(signatures) == 2, 'exactly two visible motif states required'
    minimum = min(o['scale'] for o in objects)
    assert {o['sig'] for o in objects if o['scale'] == minimum} == set(signatures)
    maps = [{(y,x):c for y,x,c in sig} for sig in signatures]
    shifts = set()
    for color in set(maps[0].values()) & set(maps[1].values()):
        a = {p for p,c in maps[0].items() if c == color}
        b = {p for p,c in maps[1].items() if c == color}
        ay,_,ax,_ = _bounds(a)
        by,_,bx,_ = _bounds(b)
        shift = (ay-by,ax-bx)
        if a == {(y+shift[0],x+shift[1]) for y,x in b}:
            shifts.add(shift)
    candidates = []
    for dy,dx in shifts:
        aligned = {(y+dy,x+dx):c for (y,x),c in maps[1].items()}
        common = {p:c for p,c in maps[0].items() if aligned.get(p) == c}
        if len(common) >= 4:
            candidates.append((len(common),(dy,dx),aligned,common))
    assert candidates
    maximum = max(c[0] for c in candidates)
    candidates = [c for c in candidates if c[0] == maximum]
    assert len(candidates) == 1, 'scaffold registration must be unique'
    _,shift,aligned,common = candidates[0]
    oy,_,ox,_ = _bounds(common)
    templates = [{(y-oy,x-ox):c for (y,x),c in m.items()} for m in (maps[0],aligned)]
    common = {(y-oy,x-ox):c for (y,x),c in common.items()}
    pads = {p:c for p,c in common.items() if c in info['word']}
    assert pads and len(set(common.values()) - set(info['word'])) == 1
    stable_color = next(iter(set(common.values()) - set(info['word'])))
    assert sum(c == stable_color for c in common.values()) >= 3
    assert all(c not in info['word'] for m in templates for p,c in m.items() if p not in common)
    union = set(templates[0]) | set(templates[1])
    ymin,ymax,xmin,xmax = _bounds(union)
    rects,occurrences = [],[]
    for o in objects:
        state = signatures.index(o['sig'])
        s = o['scale']
        dy,dx = shift if state else (0,0)
        anchor = (o['box'][0]+(oy-dy)*s, o['box'][1]+(ox-dx)*s)
        rect = (anchor[0]+ymin*s,anchor[1]+xmin*s,(ymax-ymin+1)*s,(xmax-xmin+1)*s)
        y,x,h,w = rect
        assert 3 <= y and y+h <= g.shape[0] and 1 <= x and x+w < g.shape[1]
        assert all(_separate(rect,r) for r in rects)
        assert not np.any(g[y:y+h,x:x+w] == route), 'union reservation contains terrain'
        rects.append(rect)
        expected = np.full((h,w),bg,dtype=int)
        _render(expected,templates[state],(-ymin*s,-xmin*s),s)
        assert np.array_equal(expected,g[y:y+h,x:x+w])
        occurrences.append(dict(anchor=anchor,scale=s,state=state,rect=rect))
    return dict(templates=templates,common=common,pads=pads,union=union,
                occurrences=occurrences,stable_color=stable_color)


def _contacts(path, word, pads, anchor, scale):
    pixel_pads = {}
    for (y,x),color in pads.items():
        for dy in range(scale):
            for dx in range(scale):
                pixel_pads[(anchor[0]+y*scale+dy,anchor[1]+x*scale+dx)] = color
    events = []
    for phase,(y,x) in enumerate(path):
        for dy,dx in STEPS:
            q = (y+dy,x+dx)
            if q in pixel_pads:
                events.append((phase,q,word[phase % len(word)] == pixel_pads[q]))
    return events


def _read(input_grid):
    inp = _grid(input_grid)
    candidates = []
    for code in range(8):
        canonical = _unorient(inp,code)
        try:
            info = _header(canonical)
            motifs = _motifs(info['restored'],info)
        except (AssertionError,ValueError,IndexError):
            continue
        info.update(motifs)
        info['orientation'] = code
        info['canonical_input'] = canonical
        candidates.append(info)
    assert len(candidates) == 1, 'input has no unique semantic orientation'
    info = candidates[0]
    path = info['record']['path']
    output = info['restored'].copy()
    for i,p in enumerate(path):
        output[p] = info['word'][i % len(info['word'])]
    for p in info['record']['collisions']:
        assert info['restored'][p] == info['bg']
        output[p] = info['marker']
    for o in info['occurrences']:
        events = _contacts(path,info['word'],info['pads'],o['anchor'],o['scale'])
        o['events'] = events
        o['hits'] = sum(e[2] for e in events)
        if o['hits'] % 2:
            for y,x in info['union']:
                yy,xx = o['anchor'][0]+y*o['scale'],o['anchor'][1]+x*o['scale']
                output[yy:yy+o['scale'],xx:xx+o['scale']] = info['bg']
            _render(output,info['templates'][1-o['state']],o['anchor'],o['scale'])
    info['expected'] = _orient(output,info['orientation'])
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all conventions and both motif states from this input only."""
    return _read(input_grid)['expected'].copy()


def _word(rng,n,colors,repeat):
    d = int(rng.integers(2,min(3,n)+1))
    if repeat:
        d = min(d,n-1)
    labels = list(range(d)) + list(map(int,rng.integers(d,size=n-d)))
    rng.shuffle(labels)
    return tuple(colors[i] for i in labels)


def _design(rng,change,word,carrier,accents):
    width = int(rng.integers(2,4))
    height = int(rng.integers(3,6))
    keys = {(0,c):int(rng.choice(word)) for c in range(width)}
    common = dict(keys)
    # The contact row, irregular spine, and optional lower shoulder are stable.
    common.update({(1,c):carrier for c in range(width)})
    common[(2,0)] = carrier
    if width == 3 and rng.random() < .5:
        common[(2,2)] = carrier
    # Variable lower shoulders and gaps create fresh, asymmetric scaffolds.
    # Successive rows remain eight-connected to the multicolor contact head.
    previous = {0,1}
    for r in range(3,height):
        left = int(rng.integers(width))
        right = int(rng.integers(left,width))
        columns = set(range(left,right+1))
        if min(abs(a-b) for a in columns for b in previous) > 1:
            columns.add(min(previous))
        common.update({(r,c):carrier for c in columns})
        previous = columns
    common[(2,1)] = carrier if change == 'relocate' else accents[0]
    changing_internal = change != 'relocate'
    templates = []
    left_depth,right_depth = int(rng.integers(1,3)),int(rng.integers(1,3))
    left_root = int(rng.integers(2,height))
    right_root = int(rng.integers(2,height))
    if change != 'recolor':
        # Stable side roots make both external designs connected to the carrier.
        common[(left_root,0)] = carrier
        common[(right_root,width-1)] = carrier
        # Internal recoloring retains its designated position even at a side root.
        common[(2,1)] = carrier if change == 'relocate' else accents[0]
    for state in range(2):
        m = dict(common)
        if changing_internal:
            m[(2,1)] = accents[state]
        if change == 'recolor':
            positions = {(height,c) for c in range(int(rng.integers(1,width+1)))} if state == 0 else positions
        elif state == 0:
            positions = {(r,-1) for r in range(left_root,left_root+left_depth+1)}
        else:
            positions = {(r,width) for r in range(right_root,right_root+right_depth+1)}
        for p in positions:
            m[p] = accents[state]
        templates.append(m)
    common = {p:c for p,c in templates[0].items() if templates[1].get(p) == c}
    return templates,common,keys


def _coverage(info):
    word = info['word']
    path,legs = info['record']['path'],info['record']['legs']
    good = [word[i % len(word)] for i in range(len(path))]
    unique = tuple(dict.fromkeys(word))
    reset,consume = [],[]
    phase = 0
    for k,n in enumerate(legs):
        reset.extend(word[i % len(word)] for i in range(n))
        consume.extend(word[(phase+i+k) % len(word)] for i in range(n))
        phase += n
    os = info['occurrences']
    evidence = set()
    if len(path) >= 2*len(word): evidence.add('cycle')
    if good != [unique[i % len(unique)] for i in range(len(path))]: evidence.add('multiplicity')
    if good != reset: evidence.add('bend_phase')
    if good != consume: evidence.add('marker_phase')
    if {o['state'] for o in os if o['hits'] % 2} == {0,1}: evidence.add('both_directions')
    if any(o['hits'] >= 2 and o['hits'] % 2 == 0 for o in os): evidence.add('cancellation')
    if any(o['scale'] > min(q['scale'] for q in os) and o['hits'] % 2 for o in os): evidence.add('scaled_exchange')
    if any(e[2] for o in os for e in o['events']) and any(not e[2] for o in os for e in o['events']): evidence.add('color_match')
    if any(o['scale'] > 1 and o['hits'] % 2 != (sum(any(e[2] for e in o['events'] if (e[1][0]-o['anchor'][0])//o['scale'] == p[0] and (e[1][1]-o['anchor'][1])//o['scale'] == p[1]) for p in info['pads']) % 2) for o in os):
        evidence.add('physical_pixels')
    return evidence


REQUIRED = {'cycle','multiplicity','bend_phase','marker_phase','both_directions',
            'cancellation','scaled_exchange','color_match','physical_pixels'}


def _make_pair(rng,change,large,word_range,witness):
    for attempt in range(1000):
        palette = list(map(int,rng.permutation(10)))
        bg,route,marker,carrier,a,b = palette[:6]
        n = int(rng.integers(word_range[0],word_range[1]+1))
        word = _word(rng,n,palette[6:9],witness)
        templates,common,pads = _design(rng,change,word,carrier,(a,b))
        union = set(templates[0]) | set(templates[1])
        ymin,ymax,xmin,xmax = _bounds(union)
        uh,uw = ymax-ymin+1,xmax-xmin+1
        count = int(rng.integers(3,7))
        scales = [1,1,large]+[1]*(count-3)
        rng.shuffle(scales)
        # Several motifs can share one maximal leg; turns are not object events.
        steps = list(map(int,rng.choice([0,0,2,3,4],size=count-1)))
        bottom_row = int(rng.integers(4,7))+sum(steps)
        rows = [bottom_row]
        for step in steps: rows.append(rows[-1]-step)
        ends,rects,anchors = [],[],[]
        c = 0
        for y,s in zip(rows,scales):
            left = c+int(rng.integers(2,4))
            end = left+uw*s-1+int(rng.integers(0,3))
            rects.append((y+1,left,uh*s,uw*s))
            anchors.append((y+1-ymin*s,left-xmin*s))
            ends.append(end)
            c = end
        columns = [int(rng.integers(2,4))]
        for _ in word[1:]: columns.append(columns[-1]+2+int(rng.integers(1,3)))
        mc = columns[-1]+2+int(rng.integers(4,7))
        edge = bool(rng.random() < .3)
        w = max(mc+3,ends[-1]+1+(0 if edge else int(rng.integers(1,4))))
        h = max(y+rh for y,x,rh,rw in rects)+int(rng.integers(1,4))
        if max(h,w) > 30: continue
        path,collisions,legs = [],[],[]
        previous = 0
        for i,(y,end) in enumerate(zip(rows,ends)):
            horizontal = [(y,x) for x in range(previous+(i > 0),end+1)]
            path.extend(horizontal)
            if i > 0 and steps[i-1] == 0:
                legs[-1] += len(horizontal)
            else:
                legs.append(len(horizontal))
            turn = i < count-1 and steps[i] > 0
            if (turn or i == count-1) and end+1 < w:
                collisions.append((y,end+1))
            if turn:
                vertical = [(r,end) for r in range(y-1,rows[i+1]-1,-1)]
                path.extend(vertical); legs.append(len(vertical))
                collisions.append((rows[i+1]-1,end))
            previous = end
        # Sometimes finish with a vertical leg instead of another object shelf.
        if ends[-1]+1 < w and rows[-1] > 4 and rng.random() < .35:
            top = int(rng.integers(4,rows[-1]))
            vertical = [(r,ends[-1]) for r in range(rows[-1]-1,top-1,-1)]
            path.extend(vertical); legs.append(len(vertical))
            collisions.append((top-1,ends[-1]))
        # Optional uncontacted copies use the same two templates and parity rule.
        if rng.random() < .4:
            candidates = []
            path_set,collision_set = set(path),set(collisions)
            for y in range(4,h-uh+1):
                for x in range(1,w-uw):
                    rect = (y,x,uh,uw)
                    if not all(_separate(rect,r) for r in rects): continue
                    footprint = {(r,c) for r in range(y,y+uh) for c in range(x,x+uw)}
                    if footprint & (path_set | collision_set): continue
                    anchor = (y-ymin,x-xmin)
                    if _contacts(path,word,pads,anchor,1): continue
                    candidates.append((rect,anchor))
            if candidates:
                rect,anchor = candidates[int(rng.integers(len(candidates)))]
                rects.append(rect); anchors.append(anchor); scales.append(1)
                count += 1
        events = [_contacts(path,word,pads,anchor,s) for anchor,s in zip(anchors,scales)]
        hits = [sum(e[2] for e in es) for es in events]
        odd = [i for i,k in enumerate(hits) if k % 2]
        if len(odd) < 2 or not any(k >= 2 and k % 2 == 0 for k in hits): continue
        if witness and not any(s > 1 and k % 2 for s,k in zip(scales,hits)): continue
        units = [i for i,s in enumerate(scales) if s == 1]
        states = list(map(int,rng.integers(2,size=count)))
        states[units[0]],states[units[1]] = 0,1
        # Retain both unit reference states while demonstrating both flip directions.
        if len({states[i] for i in odd}) < 2:
            movable = [i for i in odd if i not in units[:2]]
            if not movable: continue
            states[movable[0]] = 1-states[odd[0]]
        mask = np.zeros((h,w),bool)
        for p in path: mask[p] = True
        forbidden = np.zeros_like(mask)
        forbidden[:4] = True
        forbidden[:,0] = True
        for y,x,rh,rw in rects: forbidden[y:y+rh,x:x+rw] = True
        for p in collisions: forbidden[p] = True
        # Untraversed terrain lobes vary independently of the contact reservation.
        for _ in range(int(rng.integers(4,15))):
            y,x = int(rng.integers(4,h)),int(rng.integers(1,w))
            dh,dw = int(rng.integers(1,7)),int(rng.integers(1,9))
            mask[y:y+dh,x:x+dw] |= ~forbidden[y:y+dh,x:x+dw]
        mask[forbidden] = False
        for p in path: mask[p] = True
        # Available terrain beyond an exiting corner exposes A's terminal-edge rule.
        if ends[-1] == w-1 and rows[-1] > 4:
            mask[rows[-1]-1,w-1] = True
        actual = _trace(mask,path[0])
        if actual['path'] != path or actual['collisions'] != collisions: continue
        inp = np.where(mask,route,bg).astype(int)
        out = inp.copy()
        for col,color in zip(columns,word): inp[:2,col:col+2] = color
        inp[:2,mc:mc+2] = marker
        inp[path[0]] = word[0]
        for j,p in enumerate(path): out[p] = word[j % n]
        for p in collisions: out[p] = marker
        for anchor,s,state,k in zip(anchors,scales,states,hits):
            _render(inp,templates[state],anchor,s)
            _render(out,templates[1-state if k % 2 else state],anchor,s)
        code = int(rng.integers(8))
        pair = dict(input=_orient(inp,code),output=_orient(out,code))
        try:
            info = _read(pair['input'])
        except AssertionError:
            continue
        assert np.array_equal(info['expected'],pair['output'])
        if witness and not REQUIRED <= _coverage(info): continue
        return pair
    raise AssertionError('bounded signal/contact construction exhausted')


def _key(g):
    a = np.asarray(g,dtype=np.uint8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError('counts must be integers')
    nt,ne = int(num_train),int(num_test)
    if not (2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',nt),('test',ne)):
        for index in range(count):
            if regime == 1:
                change = ('recolor' if index % 2 == 0 else 'relocate') if split == 'train' else 'combined'
            else:
                change = 'combined' if index == 0 else str(rng.choice(['recolor','relocate','combined']))
            large = 3 if split == 'test' and regime == 0 else 2
            word_range = (5,6) if split == 'test' and regime == 2 else ((3,4) if regime == 2 else (3,6))
            for _ in range(20):
                pair = _make_pair(rng,change,large,word_range,split == 'train' and index == 0)
                key = _key(pair['input'])
                if key not in seen: break
            else: raise AssertionError('unique input construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen,evidence = set(),set()
    kinds = {'train':set(),'test':set()}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _read(inp)
            assert np.array_equal(info['expected'],out), 'wrong clock, contact parity or template exchange'
            canonical_out = _unorient(out,info['orientation'])
            for o in info['occurrences']:
                y,x,h,w = o['rect']
                if o['hits'] % 2 == 0:
                    assert np.array_equal(canonical_out[y:y+h,x:x+w],info['canonical_input'][y:y+h,x:x+w])
                for (r,c),color in info['common'].items():
                    yy,xx = o['anchor'][0]+r*o['scale'],o['anchor'][1]+c*o['scale']
                    assert np.all(canonical_out[yy:yy+o['scale'],xx:xx+o['scale']] == color)
            a,b = info['templates']
            if set(a) != set(b): kinds[split].add('relocate')
            if any(a[p] != b[p] for p in set(a) & set(b)): kinds[split].add('recolor')
            if split == 'train': evidence |= _coverage(info)
    assert REQUIRED <= evidence, REQUIRED-evidence
    assert kinds['test'] <= kinds['train'], 'test requires an unwitnessed state-change mechanism'


def self_test():
    shapes,orientations,clocks,scaffolds,counts,scales,patterns,backgrounds = (set() for _ in range(8))
    puzzle_keys = set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        again = generate_puzzle(seed)
        key = tuple(_key(q[r]) for split in ('train','test') for q in p[split] for r in ('input','output'))
        assert key == tuple(_key(q[r]) for split in ('train','test') for q in again[split] for r in ('input','output'))
        puzzle_keys.add(key)
        for split in ('train','test'):
            for q in p[split]:
                inp = q['input'].copy()
                info = _read(inp)
                assert np.array_equal(solve(inp),q['output']) and np.array_equal(inp,q['input'])
                shapes.add(inp.shape); orientations.add(info['orientation']); clocks.add(info['word'])
                backgrounds.add(info['bg']); counts.add(len(info['occurrences']))
                scales.update(o['scale'] for o in info['occurrences'])
                scaffolds.add(tuple(sorted(info['common'])))
                labels = {}
                patterns.add(tuple(labels.setdefault(c,len(labels)) for c in info['word']))
    assert len(puzzle_keys) == 42 and len(shapes) >= 25 and len(scaffolds) >= 15
    assert orientations == set(range(8)) and scales >= {1,2,3}
    assert len(clocks) >= 80 and len(patterns) >= 15 and len(counts) >= 3
    assert backgrounds == set(range(10))
    # All explicit count combinations are bounded, including two-training puzzles.
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1): generate_puzzle(0,nt,ne)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try: generate_puzzle(0,nt,ne)
        except ValueError: pass
        else: raise AssertionError('illegal counts accepted')
    # Pin terminal edge behavior independently of sampled geometry.
    m = np.zeros((6,9),bool); m[4,:] = True; m[3,8] = True
    assert _trace(m,(4,0))['path'] == [(4,c) for c in range(9)]
    # Pin physical contact counting and cancellation without the scene renderer.
    es = _contacts([(0,c) for c in range(4)],(1,1,2),{(0,0):1,(0,1):2},(1,0),2)
    assert sum(e[2] for e in es) == 3
    es = _contacts([(0,c) for c in range(2)],(1,1,2),{(0,0):1,(0,1):1},(1,0),1)
    assert sum(e[2] for e in es) == 2
    damaged = generate_puzzle(0,2,1)
    # Omitting either inherited mechanism changes a complete demonstrated answer.
    info = _read(damaged['train'][0]['input'])
    no_exchange = _unorient(info['expected'],info['orientation'])
    unconditional = no_exchange.copy()
    for o in info['occurrences']:
        y,x,h,w = o['rect']
        no_exchange[y:y+h,x:x+w] = info['canonical_input'][y:y+h,x:x+w]
        for r,c in info['union']:
            yy,xx = o['anchor'][0]+r*o['scale'],o['anchor'][1]+c*o['scale']
            unconditional[yy:yy+o['scale'],xx:xx+o['scale']] = info['bg']
        _render(unconditional,info['templates'][1-o['state']],o['anchor'],o['scale'])
    assert not np.array_equal(_orient(no_exchange,info['orientation']),info['expected'])
    assert not np.array_equal(_orient(unconditional,info['orientation']),info['expected'])
    damaged['test'][0]['output'][0,0] = (int(damaged['test'][0]['output'][0,0])+1)%10
    try: validate_generated_puzzle(damaged)
    except AssertionError: pass
    else: raise AssertionError('damaged output accepted')
    return {'seeds':42,'shapes':len(shapes),'scaffolds':len(scaffolds),
            'clock_patterns':len(patterns),'counts':sorted(counts),'scales':sorted(scales)}
