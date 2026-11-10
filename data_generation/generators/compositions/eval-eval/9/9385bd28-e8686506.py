"""Inverse occlusion: assemble detached visible faces of corner-defined sheets.

No palette table supplies the paint or priority. The exact oriented inventory
jointly determines both. See task_spec.md for the complete public convention.
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
import numpy as np


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return (a.shape, a.tobytes())


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, part = [start], set()
        while todo:
            r, c = todo.pop()
            part.add((r, c))
            for p in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(frozenset(part))
    return result


def _markers(bounds, diagonal):
    r0, c0, r1, c1 = bounds
    if diagonal == 0:
        return frozenset(((r0,c0), (r0,c0+1), (r0+1,c0),
                          (r1,c1), (r1,c1-1), (r1-1,c1)))
    return frozenset(((r0,c1), (r0,c1-1), (r0+1,c1),
                      (r1,c0), (r1,c0+1), (r1-1,c0)))


def _recognize(cells):
    if len(cells) != 6:
        return None
    b = (min(r for r,c in cells), min(c for r,c in cells),
         max(r for r,c in cells), max(c for r,c in cells))
    h, w = b[2]-b[0]+1, b[3]-b[1]+1
    if min(h,w) < 2 or (min(h,w) == 2 and max(h,w) < 4):
        return None
    for d in (0,1):
        if cells == _markers(b,d):
            return b, d
    return None


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > a.size // 2, "background must occupy a strict majority")
    rects = []
    for color in range(10):
        if color == bg or not counts[color]:
            continue
        cells = frozenset(tuple(map(int,p)) for p in np.argwhere(a == color))
        found = _recognize(cells)
        if found is not None:
            b, d = found
            rects.append(dict(cue=color, bounds=b, markers=cells, diagonal=d))
    _require(2 <= len(rects) <= 4, "expected two to four recognizable corner pairs")
    crop = (min(o['bounds'][0] for o in rects), min(o['bounds'][1] for o in rects),
            max(o['bounds'][2] for o in rects), max(o['bounds'][3] for o in rects))
    r0,c0,r1,c1 = crop
    cue_colors = {o['cue'] for o in rects}
    inside = a[r0:r1+1,c0:c1+1]
    _require(set(map(int,np.unique(inside))) <= cue_colors | {bg}, "payload inside blueprint crop")
    for o in rects:
        o['bounds'] = (o['bounds'][0]-r0, o['bounds'][1]-c0,
                       o['bounds'][2]-r0, o['bounds'][3]-c0)
        o['markers'] = frozenset((r-r0,c-c0) for r,c in o['markers'])
    exterior = frozenset(tuple(map(int,p)) for p in np.argwhere((a != bg) & ~np.isin(a,list(cue_colors))))
    pieces = []
    for part in _components(exterior):
        colors = {int(a[p]) for p in part}
        _require(len(colors) == 1, "loose pieces must be separated and monochrome")
        _require(all(not (r0-1 <= r <= r1+1 and c0-1 <= c <= c1+1) for r,c in part),
                 "loose pieces need a one-cell moat around the blueprint")
        pieces.append((next(iter(colors)), _normalize(part)))
    _require(pieces and 2 <= len({c for c,s in pieces}) <= 4, "expected two to four paint colors")
    return dict(background=bg, rects=rects, crop=crop,
                shape=(r1-r0+1,c1-c0+1), pieces=pieces, exterior=exterior)


def _faces(rects, shape, order):
    """Visit front to back; keep only each sheet's exposed material and cues."""
    owner = np.full(shape, -1, dtype=np.int8)
    for i in order:
        r0,c0,r1,c1 = rects[i]['bounds']
        view = owner[r0:r1+1,c0:c1+1]
        view[view < 0] = i
    parts = []
    for i,o in enumerate(rects):
        material = {tuple(map(int,p)) for p in np.argwhere(owner == i)} - o['markers']
        parts.append(_components(material))
    return owner, parts


def _render(parsed, owner, paints):
    out = np.full(parsed['shape'], parsed['background'], dtype=np.int8)
    for i,o in enumerate(parsed['rects']):
        out[owner == i] = paints[i]
        for p in o['markers']:
            if owner[p] == i:
                out[p] = o['cue']
    return out


def _answers(parsed, area_only=False, limit=2):
    """Enumerate visible layer orders and exact oriented shape/color inventory.

    Different orders of disjoint sheets and identical-piece exchanges count as
    one answer. This small finite search has at most 24 orders and 4^4 paint
    assignments; it has no seed, example cache, output labels or search timeout.
    """
    signature = (lambda p: len(p)) if area_only else _normalize
    inventory = Counter((color, len(s) if area_only else s) for color,s in parsed['pieces'])
    colors = sorted({color for color,s in inventory})
    answers = {}
    for order in permutations(range(len(parsed['rects']))):
        owner, parts = _faces(parsed['rects'],parsed['shape'],order)
        if any(not p for p in parts):
            continue  # Every sheet must contribute visible paint.
        if sum(len(p) for group in parts for p in group) != sum(len(s) for c,s in parsed['pieces']):
            continue
        required = [Counter(signature(p) for p in group) for group in parts]
        indices = sorted(range(len(parts)), key=lambda i: (-sum(required[i].values()), -sum(map(len,parts[i]))))
        paints = [None] * len(parts)

        def assign(k):
            if k == len(indices):
                if any(inventory.values()):
                    return False
                out = _render(parsed,owner,paints)
                answers.setdefault(out.tobytes(), (out.copy(), tuple(order), tuple(paints), parts))
                return len(answers) >= limit
            i = indices[k]
            for color in colors:
                needed = Counter({(color,s):n for s,n in required[i].items()})
                if any(inventory[key] < n for key,n in needed.items()):
                    continue
                inventory.subtract(needed)
                paints[i] = color
                stop = assign(k+1)
                inventory.update(needed)
                if stop:
                    return True
            return False

        if assign(0):
            break
    return list(answers.values())


def solve(input_grid: np.ndarray) -> np.ndarray:
    parsed = _parse(_grid(input_grid))
    answers = _answers(parsed)
    _require(len(answers) == 1, "inventory must determine exactly one colored reconstruction")
    return answers[0][0].copy()


def _rotate_geometry(rects, shape, rotation):
    for _ in range(rotation):
        h,w = shape
        changed = []
        for o in rects:
            cells = frozenset((w-1-c,r) for r,c in o['markers'])
            b,d = _recognize(cells)
            changed.append(dict(bounds=b, markers=cells, diagonal=d))
        rects,shape = changed,(w,h)
    return rects,shape


def _geometry(rng, profile, count):
    if profile == 'contrast':
        # Congruent shifted sheets: reversing priority preserves piece areas,
        # but reflects the lower silhouette. Exact orientation resolves it.
        h,w = map(int,rng.integers(4,8,size=2))
        dr,dc = int(rng.integers(2,h-1)),int(rng.integers(2,w-1))
        bounds = [(0,0,h-1,w-1),(dr,dc,dr+h-1,dc+w-1)]
        diagonals = [0,0]
        order = tuple(map(int,rng.permutation(2)))
    elif profile == 'bridge':
        h,w = int(rng.integers(4,7)),int(rng.integers(8,12))
        top = int(rng.integers(2,4))
        bottom = top+h-1
        end = bottom+int(rng.integers(2,5))
        mid = int(rng.integers(3,w-3))
        thick = min(int(rng.integers(2,4)), w-2-mid)
        bounds = [(top,0,bottom,w-1),(0,mid,end,mid+thick-1),
                  (bottom-1,w-2,end,w+int(rng.integers(1,4)))]
        diagonals = [0,int(rng.integers(2)),1]
        order = (1,2,0) if rng.random() < .5 else (2,1,0)
    else:
        H,W = map(int,rng.integers(8,15,size=2))
        bounds,diagonals,used = [],[],set()
        style = str(rng.choice(['free','free','nested','bands','chain']))
        for i in range(count):
            for _ in range(100):
                h,w = int(rng.integers(3,H+1)),int(rng.integers(3,W+1))
                if style == 'bands':
                    if i % 2:
                        h = int(rng.integers(2,4))
                    else:
                        w = int(rng.integers(2,4))
                r,c = int(rng.integers(H-h+1)),int(rng.integers(W-w+1))
                b = (r,c,r+h-1,c+w-1)
                if style == 'nested' and i and rng.random() < .7:
                    parent = bounds[0]
                    ph,pw = parent[2]-parent[0]+1,parent[3]-parent[1]+1
                    if min(ph,pw) >= 7:
                        h,w = int(rng.integers(3,ph-2)),int(rng.integers(3,pw-2))
                        r = int(rng.integers(parent[0]+1,parent[2]-h+1))
                        c = int(rng.integers(parent[1]+1,parent[3]-w+1))
                        b = (r,c,r+h-1,c+w-1)
                d = int(rng.integers(2))
                cells = _markers(b,d)
                if cells & used or _recognize(cells) is None or b in bounds:
                    continue
                if style == 'chain' and i:
                    overlaps = [not (b[2]<p[0] or p[2]<b[0] or b[3]<p[1] or p[3]<b[1]) for p in bounds]
                    if not overlaps[-1] or any(overlaps[:-1]):
                        continue
                bounds.append(b)
                diagonals.append(d)
                used.update(cells)
                break
            else:
                return None
        order = tuple(map(int,rng.permutation(count)))
    r0,c0 = min(b[0] for b in bounds),min(b[1] for b in bounds)
    bounds = [(r-r0,c-c0,rr-r0,cc-c0) for r,c,rr,cc in bounds]
    shape = (max(b[2] for b in bounds)+1,max(b[3] for b in bounds)+1)
    rects = [dict(bounds=b,markers=_markers(b,d),diagonal=d) for b,d in zip(bounds,diagonals)]
    all_markers = [p for o in rects for p in o['markers']]
    if len(set(all_markers)) != 6*len(rects):
        return None
    rects,shape = _rotate_geometry(rects,shape,int(rng.integers(4)))
    return rects,shape,order


def _scatter(rng, parsed, pieces):
    h,w = parsed['shape']
    largest = max(max(r for r,c in s)+1 for color,s in pieces),max(max(c for r,c in s)+1 for color,s in pieces)
    for attempt in range(18):
        if attempt < 12:
            H = int(rng.integers(max(18,h+largest[0]+2),31))
            W = int(rng.integers(max(18,w+largest[1]+2),31))
        else:
            H=W=30
        # Positions can touch any canvas edge; there is no mandatory border.
        r0,c0 = int(rng.integers(H-h+1)),int(rng.integers(W-w+1))
        a = np.full((H,W),parsed['background'],dtype=np.int8)
        blocked = np.zeros((H,W),dtype=bool)
        blocked[max(0,r0-1):min(H,r0+h+1),max(0,c0-1):min(W,c0+w+1)] = True
        for o in parsed['rects']:
            for r,c in o['markers']:
                a[r0+r,c0+c] = o['cue']
        ranked = sorted(pieces,key=lambda x: (-len(x[1]),x[0],x[1]))
        for color,s in ranked:
            sh,sw = max(r for r,c in s)+1,max(c for r,c in s)+1
            options = [(r,c) for r in range(H-sh+1) for c in range(W-sw+1)
                       if all(not blocked[r+dr,c+dc] for dr,dc in s)]
            if not options:
                break
            edges = [(r,c) for r,c in options if r==0 or c==0 or r+sh==H or c+sw==W]
            choices = edges if edges and rng.random()<.35 else options
            r,c = choices[int(rng.integers(len(choices)))]
            for dr,dc in s:
                rr,cc = r+dr,c+dc
                a[rr,cc] = color
                for ar,ac in ((rr,cc),(rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)):
                    if 0<=ar<H and 0<=ac<W:
                        blocked[ar,ac] = True
        else:
            if np.count_nonzero(a == parsed['background']) > a.size//2:
                return a
    return None


def _make_pair(rng, profile, count):
    for _ in range(180):
        geometry = _geometry(rng,profile,count)
        if geometry is None:
            continue
        rects,shape,order = geometry
        owner,parts = _faces(rects,shape,order)
        if any(not p for p in parts) or sum(map(len,parts)) > 12:
            continue
        if profile == 'bridge' and max(map(len,parts)) < 2:
            continue
        palette = list(map(int,rng.permutation(10)))
        bg,cues = palette[0],palette[1:1+len(rects)]
        # The bridge teaches both multiple islands of one sheet and paint
        # sharing across sheets, even when only two demonstrations are requested.
        npaint = 2 if profile in ('contrast','bridge') else int(rng.integers(2,len(rects)+1))
        paint_colors = palette[1+len(rects):1+len(rects)+npaint]
        paints = [paint_colors[i%npaint] for i in rng.permutation(len(rects))]
        for o,cue in zip(rects,cues):
            o['cue'] = cue
        pieces = [(paints[i],_normalize(p)) for i,group in enumerate(parts) for p in group]
        parsed = dict(background=bg,rects=rects,shape=shape,pieces=pieces)
        expected = _render(parsed,owner,paints)
        answers = _answers(parsed)
        if len(answers)!=1 or not np.array_equal(answers[0][0],expected):
            continue
        if profile == 'contrast' and len(_answers(parsed,area_only=True)) < 2:
            continue
        inp = _scatter(rng,parsed,pieces)
        if inp is None:
            continue
        try:
            public = _parse(inp)
            recovered = _answers(public)
        except ValueError:
            continue
        if len(recovered)==1 and np.array_equal(recovered[0][0],expected):
            return dict(input=inp,output=expected)
    raise RuntimeError('bounded generation exhausted for '+profile)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value,name,lo,hi in ((num_train,'num_train',2,10),(num_test,'num_test',1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo<=int(value)<=hi:
            raise ValueError(name+' has an illegal count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    profiles = [('contrast',2),('bridge',3)]
    if rng.random()<.5:
        profiles.reverse()
    profiles += [('free',int(rng.integers(2,5))) for _ in range(num_train-2)]
    regime = int(rng.integers(3))
    tests = [('free',4 if regime==0 else int(rng.integers(3,5))) for _ in range(num_test)]
    puzzle,seen = {'train':[],'test':[]},set()
    for split,settings in (('train',profiles),('test',tests)):
        for profile,count in settings:
            for _ in range(8):
                pair = _make_pair(rng,profile,count)
                k = _key(pair['input'])
                if k not in seen:
                    break
            else:
                raise RuntimeError('could not produce distinct inputs')
            seen.add(k)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp,out = _grid(pair['input']),_grid(pair['output'])
    parsed = _parse(inp)
    answers = _answers(parsed)
    _require(len(answers)==1 and np.array_equal(out,answers[0][0]), 'wrong or ambiguous reconstruction')
    _,order,paints,parts = answers[0]
    owner,_ = _faces(parsed['rects'],parsed['shape'],order)
    cues = {o['cue'] for o in parsed['rects']}
    for i,o in enumerate(parsed['rects']):
        for p in o['markers']:
            if owner[p] == i:
                _require(out[p] == o['cue'],'visible corner changed')
            else:
                _require(out[p] != o['cue'],'occluded corner wrongly retained')
    hidden = sum(owner[p]!=i for i,o in enumerate(parsed['rects']) for p in o['markers'])
    paint_counts = Counter(c for c,s in parsed['pieces'] for _ in s)
    _require(all(np.count_nonzero(out==c)==n for c,n in paint_counts.items()), 'paint inventory not conserved')
    _require(np.count_nonzero(owner>=0)==sum(paint_counts.values())+np.count_nonzero(np.isin(out,list(cues))),
             'sheet union area does not conserve material and visible corners')
    _require(np.all(out[owner<0]==parsed['background']),'background outside sheets changed')
    relaxed = _answers(parsed,area_only=True)
    return dict(parsed=parsed, hidden=hidden, islands=max(map(len,parts)),
                n=len(parsed['rects']), pieces=len(parsed['pieces']),
                nonrect=any(len(s)!=(max(r for r,c in s)+1)*(max(c for r,c in s)+1) for c,s in parsed['pieces']),
                area_ambiguous=len(relaxed)>1, parts=parts, paints=paints)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle)=={'train','test'},'invalid puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in ('train','test')),'splits must be lists')
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    _require(2<=nt<=10 and 1<=ne<=4 and nt>ne,'invalid example counts')
    seen,training = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair)=={'input','output'},'invalid pair keys')
            a = _analyze(pair)
            key = _key(_grid(pair['input']))
            _require(key not in seen,'duplicate input')
            seen.add(key)
            if split=='train':
                training.append(a)
    _require(any(a['hidden'] and a['area_ambiguous'] for a in training),
             'training must distinguish occlusion and oriented shapes from area-only matching')
    _require(any(a['islands']>=2 and a['pieces']>a['n'] for a in training),
             'training must demonstrate multiple separate pieces for one sheet')
    _require(any(a['nonrect'] for a in training),'training needs visible silhouette geometry')


def self_test():
    """Bounded semantic and diversity checks over 42 seeds plus count edges."""
    signatures,shapes,canvases,layouts,faces,bgs,origins = set(),set(),set(),set(),set(),set(),set()
    counts,piece_counts,diagonals,paint_counts = set(),set(),set(),set()
    within_geometry = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for s in ('train','test') for p in puzzle[s]))
        local = set()
        for split in ('train','test'):
            for pair in puzzle[split]:
                a = _analyze(pair)
                p = a['parsed']
                shapes.add(p['shape']); local.add(p['shape'])
                canvases.add(pair['input'].shape)
                layouts.add((p['shape'],tuple((o['bounds'],o['diagonal']) for o in p['rects'])))
                faces.update(s for c,s in p['pieces'])
                bgs.add(p['background']); origins.add(p['crop'][:2])
                counts.add(a['n']); piece_counts.add(a['pieces'])
                diagonals.update(o['diagonal'] for o in p['rects'])
                paint_counts.add(len(set(a['paints'])))
        within_geometry += len(local)>1
    _require(len(signatures)==42,'seed diversity collapsed')
    _require(len(shapes)>=25 and len(canvases)>=35 and len(layouts)>=160,'insufficient geometry diversity')
    _require(len(faces)>=100 and len(piece_counts)>=5,'insufficient exposed-piece diversity')
    _require(bgs==set(range(10)) and diagonals=={0,1} and counts=={2,3,4},'missing role/count variation')
    _require(paint_counts=={2,3,4} and len(origins)>=25 and within_geometry>=35,'missing within-puzzle variation')
    first = generate_puzzle(314159,5,2)
    again = generate_puzzle(314159,5,2)
    _require(all(_key(a[k])==_key(b[k]) for s in first for a,b in zip(first[s],again[s])
                 for k in ('input','output')),'generation is not deterministic')
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(500+nt*7+ne,nt,ne)
            _require(len(p['train'])==nt and len(p['test'])==ne,'ignored explicit counts')
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted illegal counts')
    # Label corruption and lost material must be detected by input-derived checks.
    for corrupt in ('output','inventory'):
        bad = {s:[{k:v.copy() for k,v in p.items()} for p in first[s]] for s in first}
        pair = bad['test'][0]
        parsed = _parse(pair['input'])
        if corrupt=='inventory':
            pair['input'][min(parsed['exterior'])] = parsed['background']
        else:
            pair['output'][0,0] = (int(pair['output'][0,0])+1)%10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('validator accepted '+corrupt+' corruption')
    return {'seeds':42,'distinct_puzzles':len(signatures),'output_shapes':len(shapes),
            'input_shapes':len(canvases),'layouts':len(layouts),'oriented_faces':len(faces),
            'piece_counts':sorted(piece_counts),'within_puzzle_shape_variation':within_geometry}


if __name__ == '__main__':
    print(self_test())
