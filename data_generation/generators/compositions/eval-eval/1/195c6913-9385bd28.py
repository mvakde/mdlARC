"""Route-clocked rectangular sampling shutters.

No parent modules, source files, random state, or training examples are used by
solve. See task_spec.md for the complete public input grammar and rule.
"""
from __future__ import annotations

import numpy as np


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(x):
    a = np.asarray(x)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=True)


def _components(a, color):
    pending = set(map(tuple, np.argwhere(a == color)))
    result = []
    while pending:
        p = pending.pop()
        stack, comp = [p], {p}
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in pending:
                    pending.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _corners(bounds, diagonal):
    r, c, s, t = bounds
    if diagonal == 0:
        return {(r,c), (r,c+1), (r+1,c), (s,t), (s,t-1), (s-1,t)}
    return {(r,t), (r,t-1), (r+1,t), (s,c), (s,c+1), (s-1,c)}


def _contains(b, p):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _trace(mask, start):
    """Maximal-run implementation; a canvas exit terminates immediately."""
    p, direction, path, legs = start, 0, [], []
    while True:
        dr, dc = ((0,1), (-1,0))[direction]
        r, c = p
        leg = []
        while _inside(mask.shape, (r,c)) and mask[r,c]:
            leg.append((r,c))
            r, c = r+dr, c+dc
        assert leg
        path.extend(leg)
        legs.append(len(leg))
        if not _inside(mask.shape, (r,c)):
            break
        direction = 1-direction
        dr, dc = ((0,1), (-1,0))[direction]
        p = (leg[-1][0]+dr, leg[-1][1]+dc)
        if not _inside(mask.shape, p) or not mask[p]:
            break
    return path, legs


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0,0])
    counts = np.bincount(a.ravel(), minlength=10)
    # The blank upper-left corner identifies background even on dense terrain.
    terrain = max((c for c in range(10) if c != bg), key=lambda c: counts[c])
    assert counts[terrain] >= 30
    swatches, leftovers = {}, {}
    for color in np.unique(a):
        color = int(color)
        if color in (bg, terrain):
            continue
        swatches[color] = []
        leftovers[color] = set()
        for comp in _components(a, color):
            r0, c0 = min(r for r,c in comp), min(c for r,c in comp)
            if len(comp) == 4 and comp == {(r0,c0),(r0+1,c0),(r0,c0+1),(r0+1,c0+1)}:
                swatches[color].append((r0,c0,comp))
            else:
                leftovers[color] |= comp
    boxes, word_swatches, key_swatches, singletons = [], [], [], []
    for color, extra in leftovers.items():
        ss = swatches[color]
        if len(extra) == 6:
            assert len(ss) == 1
            r0, c0 = min(r for r,c in extra), min(c for r,c in extra)
            r1, c1 = max(r for r,c in extra), max(c for r,c in extra)
            bounds = (r0,c0,r1,c1)
            h, w = r1-r0+1, c1-c0+1
            assert min(h,w) >= 2 and (min(h,w) > 2 or max(h,w) >= 4)
            diagonals = [d for d in (0,1) if extra == _corners(bounds,d)]
            assert len(diagonals) == 1
            boxes.append(dict(cue=color, bounds=bounds, markers=extra,
                              diagonal=diagonals[0], key=ss[0][:2]))
            key_swatches.append(ss[0])
        else:
            assert len(extra) <= 1 and ss
            word_swatches.extend((r,c,color,comp) for r,c,comp in ss)
            if extra:
                singletons.append((color,next(iter(extra))))
    assert 1 <= len(boxes) <= 4 and 2 <= len(word_swatches) <= 6
    assert len(singletons) == 1
    word_swatches.sort(key=lambda x:x[1])
    assert len({s[0] for s in word_swatches}) == 1
    assert all(b[1]-a[1] >= 3 for a,b in zip(word_swatches,word_swatches[1:]))
    word = tuple(s[2] for s in word_swatches)
    assert 2 <= len(set(word)) <= 4
    assert singletons[0][0] == word[0]
    boxes.sort(key=lambda b:b['key'][0])
    assert len({b['key'][1] for b in boxes}) == 1
    assert all(b['key'][0]-a['key'][0] >= 3 for a,b in zip(boxes,boxes[1:]))
    annotation = set().union(*(s[3] for s in word_swatches), *(s[2] for s in key_swatches))
    assert (0,0) not in annotation
    for r,c,cells in [(s[0],s[1],s[3]) for s in word_swatches] + key_swatches:
        for rr in range(max(0,r-1),min(a.shape[0],r+3)):
            for cc in range(max(0,c-1),min(a.shape[1],c+3)):
                if (rr,cc) not in cells:
                    assert a[rr,cc] == bg, 'annotation needs a background moat'
    for box in boxes:
        assert not any(_contains(box['bounds'], p) for p in annotation)
    restored = a.copy()
    for color in swatches:
        restored[a == color] = bg
    start = singletons[0][1]
    restored[start] = terrain
    mask = restored == terrain
    assert start[1] == 0 or not mask[start[0],start[1]-1]
    assert start[1]+1 < a.shape[1] and mask[start[0],start[1]+1]
    # All nonterrain symbols have now been accounted for without area guesses.
    assert set(map(int,np.unique(restored))) == {bg,terrain}
    assert counts[terrain] > max(counts[c] for c in swatches)
    return dict(grid=a, bg=bg, terrain=terrain, mask=mask, start=start,
                word=word, boxes=boxes, annotation=annotation,
                legend_origin=word_swatches[0][:2])


def _evaluate(info, mode='correct'):
    """Clock, sample, and composite. Modes are explicit falsifiable shortcuts."""
    a, bg, word = info['grid'], info['bg'], info['word']
    path, legs = _trace(info['mask'],info['start'])
    boxes = list(info['boxes'])
    if mode == 'reverse_priority':
        boxes.reverse()
    owners = [next((j for j,b in enumerate(boxes) if _contains(b['bounds'],p)), -1)
              for p in path]
    if mode == 'dedup':
        word = tuple(dict.fromkeys(word))
    phases = list(range(len(path)))
    if mode == 'reset_bends':
        phases = [k for length in legs for k in range(length)]
    elif mode == 'hidden_pause':
        phases, phase = [], 0
        for owner in owners:
            phases.append(phase)
            phase += owner == -1
    elif mode == 'no_clock':
        phases = [0]*len(path)
    colors = [word[min(k,len(word)-1)] if mode == 'no_wrap' else word[k % len(word)]
              for k in phases]
    hits = [None]*len(boxes)
    for k,p in enumerate(path):
        for j,b in enumerate(boxes):
            eligible = _contains(b['bounds'],p) if mode == 'first_intersection' else owners[k] == j
            if eligible and (hits[j] is None or mode == 'last_sample'):
                hits[j] = k
    samples = [bg if k is None else colors[k] for k in hits]
    out = a.copy()
    for p,color in zip(path,colors):
        out[p] = color
    if mode != 'no_boxes':
        for j in reversed(range(len(boxes))):
            b = boxes[j]
            if mode == 'no_clear' and hits[j] is None:
                continue
            r,c,s,t = b['bounds']
            out[r:s+1,c:t+1] = samples[j]
            if mode != 'recolor_markers':
                for p in b['markers']:
                    out[p] = b['cue']
    return out, dict(path=path, legs=legs, owners=owners, hits=hits,
                     samples=samples, colors=colors)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from public input alone, with fixed left/right and top/down keys."""
    return _evaluate(_decode(input_grid))[0]


_ALTERNATIVES = ('no_clock','no_boxes','dedup','reset_bends','hidden_pause',
                 'first_intersection','reverse_priority','last_sample','no_wrap',
                 'no_clear','recolor_markers')
_CORE = set(_ALTERNATIVES[:9])


def _evidence(info, output=None):
    out, details = _evaluate(info)
    if output is not None:
        assert np.array_equal(out,output)
    found = {m for m in _ALTERNATIVES if not np.array_equal(out,_evaluate(info,m)[0])}
    boxes, bg = info['boxes'], info['bg']
    visible_areas = [any(not any(_contains(higher['bounds'],p) for higher in boxes[:j])
                         and p not in b['markers']
                         for p in ((r,c) for r in range(b['bounds'][0],b['bounds'][2]+1)
                                   for c in range(b['bounds'][1],b['bounds'][3]+1)))
                     for j,b in enumerate(boxes)]
    pause_samples = _evaluate(info,'hidden_pause')[1]['samples']
    for j,b in enumerate(boxes):
        hit = details['hits'][j]
        if hit is None or not visible_areas[j]:
            continue
        geometric = next(k for k,p in enumerate(details['path']) if _contains(b['bounds'],p))
        if hit > geometric and details['colors'][hit] != details['colors'][geometric]:
            found.add('delayed_sample')
        if details['samples'][j] != pause_samples[j]:
            found.add('hidden_sample')
    if len({color for j,color in enumerate(details['samples'])
            if details['hits'][j] is not None and visible_areas[j]}) >= 2:
        found.add('different_samples')
    if sum(owner == -1 for owner in details['owners']) >= len(info['word'])+2:
        found.add('exposed_clock')
    # Check a truly opaque blank shutter hiding a colored lower shutter.
    for j,b in enumerate(boxes):
        if details['hits'][j] is not None:
            continue
        for r in range(b['bounds'][0],b['bounds'][2]+1):
            for c in range(b['bounds'][1],b['bounds'][3]+1):
                p = (r,c)
                stack = [i for i,x in enumerate(boxes) if _contains(x['bounds'],p)]
                if stack[0] != j or p in b['markers']:
                    continue
                if any(details['hits'][i] is not None and details['samples'][i] != bg
                       and p not in boxes[i]['markers'] for i in stack[1:]):
                    assert out[p] == bg
                    found.add('opaque_clear')
    if any(out[p] != b['cue'] for b in boxes for p in b['markers']):
        found.add('marker_cover')
    return found, details


def _mask(rng,h,w):
    """Independent boundaries plus bays, terraces, islands and holes."""
    mask = np.zeros((h,w),bool)
    knots = np.unique(np.linspace(3,h-1,int(rng.integers(3,9))).astype(int))
    low = rng.integers(4,max(5,w//2),len(knots))
    high = rng.integers(max(6,w//2),w,len(knots))
    left = np.rint(np.interp(np.arange(h),knots,low)).astype(int)
    right = np.rint(np.interp(np.arange(h),knots,high)).astype(int)
    for r in range(3,h):
        mask[r,left[r]:max(left[r]+2,right[r])+1] = True
    for _ in range(int(rng.integers(1,9))):
        r,c = int(rng.integers(3,h-1)),int(rng.integers(4,w-1))
        hh = int(rng.integers(1,min(7,h-r)+1))
        ww = int(rng.integers(1,min(9,w-c)+1))
        mask[r:r+hh,c:c+ww] = bool(rng.integers(2))
    return mask


def _geometry(rng,h,w,n):
    boxes, used = [], set()
    for _ in range(n):
        for attempt in range(70):
            hh = int(rng.integers(3,h-3))
            ww = int(rng.integers(3,w-4))
            if rng.random() < .16:
                if rng.random() < .5:
                    hh,ww = 2,max(4,ww)
                else:
                    hh,ww = max(4,hh),2
            r = int(rng.integers(3,h-hh+1))
            c = int(rng.integers(4,w-ww+1))
            bounds = (r,c,r+hh-1,c+ww-1)
            diagonal = int(rng.integers(2))
            cells = _corners(bounds,diagonal)
            if not cells & used and not any(bounds == b['bounds'] for b in boxes):
                boxes.append(dict(bounds=bounds,markers=cells,diagonal=diagonal))
                used |= cells
                break
        else:
            return None
    return boxes


def _word(rng,length,distinct,repeat):
    if repeat:
        distinct = min(distinct,length-1)
    word = list(range(distinct)) + list(map(int,rng.integers(distinct,size=length-distinct)))
    rng.shuffle(word)
    if repeat:
        unique = tuple(dict.fromkeys(word))
        if all(x == unique[i % len(unique)] for i,x in enumerate(word)):
            # Retain a real nonreducible multiplicity witness.
            word[0],word[1] = word[1],word[0]
    return word


def _annotation_positions(mask,boxes,word_length,n,rng):
    h,w = mask.shape
    blocked = mask.copy()
    blocked[0,0] = True
    for b in boxes:
        r,c,s,t = b['bounds']
        blocked[r:s+1,c:t+1] = True
    # Sample variable gaps and freely translate the two groups in empty terrain.
    gaps = list(map(int,rng.integers(1,3,size=word_length-1)))
    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1]+2+gap)
    span = offsets[-1]+2
    locations = []
    for r in range(h-1):
        for c in range(w-span+1):
            if not blocked[max(0,r-1):min(h,r+3),max(0,c-1):min(w,c+span+1)].any():
                locations.append((r,c))
    if not locations:
        return None
    r,c = locations[int(rng.integers(len(locations)))]
    legend = [(r,c+x) for x in offsets]
    blocked[max(0,r-1):min(h,r+3),max(0,c-1):min(w,c+span+1)] = True
    gap = int(rng.integers(1,3))
    span = n*2+(n-1)*gap
    locations = [(r,c) for r in range(h-span+1) for c in range(w-1)
                 if not blocked[max(0,r-1):min(h,r+span+1),max(0,c-1):min(w,c+3)].any()]
    if not locations:
        return None
    r,c = locations[int(rng.integers(len(locations)))]
    key = [(r+j*(2+gap),c) for j in range(n)]
    return legend,key


def _pair(rng,split,schedule,required):
    informative = bool(required)
    for attempt in range(1600):
        if schedule == 'scale':
            lo,hi = (21,26) if split == 'train' else (27,30)
        elif schedule == 'small_test':
            lo,hi = (25,30) if split == 'train' else (19,24)
        else:
            lo,hi = 20,30
        h,w = int(rng.integers(lo,hi+1)),int(rng.integers(lo,hi+1))
        n = 3 if informative else int(rng.integers(1,4 if split == 'train' else 5))
        if schedule == 'layers' and split == 'test':
            n = 4
        if schedule == 'word':
            length = int(rng.integers(3,5) if split == 'train' else rng.integers(5,7))
        else:
            length = int(rng.integers(3,7) if informative else rng.integers(2,7))
        distinct = int(rng.integers(2,min(4,8-n,length-1 if informative else length)+1))
        abstract_word = _word(rng,length,distinct,'dedup' in required)
        palette = list(map(int,rng.permutation(10)))
        bg,terrain = palette[:2]
        cue_colors = palette[2:2+n]
        word = tuple(palette[2+n+x] for x in abstract_word)
        boxes = _geometry(rng,h,w,n)
        if boxes is None:
            continue
        mask = _mask(rng,h,w)
        for b,cue in zip(boxes,cue_colors):
            b['cue'] = cue
            for p in b['markers']:
                mask[p] = False
        if mask.sum() < 40:
            continue
        pos = _annotation_positions(mask,boxes,length,n,rng)
        if pos is None:
            continue
        legend,key = pos
        a = np.where(mask,terrain,bg).astype(int)
        for (r,c),color in zip(legend,word):
            a[r:r+2,c:c+2] = color
        for b,(r,c) in zip(boxes,key):
            b['key'] = (r,c)
            a[r:r+2,c:c+2] = b['cue']
            for p in b['markers']:
                a[p] = b['cue']
        annotation = {(r+dr,c+dc) for r,c in legend+key for dr in (0,1) for dc in (0,1)}
        candidates = [(r,c) for r in range(3,h) for c in range(4,w-1)
                      if mask[r,c] and mask[r,c+1] and not mask[r,c-1]]
        rng.shuffle(candidates)
        # Cheaply rank actual routes by opportunities, without a fixed start row.
        routes = []
        for start in candidates:
            path,legs = _trace(mask,start)
            if len(path) < (2*length if informative else 3):
                continue
            owners = [next((j for j,b in enumerate(boxes) if _contains(b['bounds'],p)),-1) for p in path]
            hits = {j for j in owners if j >= 0}
            score = 4*len(hits)+min(5,len(legs))
            if 'opaque_clear' in required:
                score += 12*(len(hits)<n and len(hits)>0)
            routes.append((score,start))
        routes.sort(key=lambda x:x[0],reverse=True)
        for score,start in routes[:18 if informative else 5]:
            inp = a.copy()
            inp[start] = word[0]
            info = dict(grid=inp,bg=bg,terrain=terrain,mask=mask,start=start,
                        word=word,boxes=boxes,annotation=annotation)
            out,details = _evaluate(info)
            if np.array_equal(inp,out):
                continue
            if informative and not required <= _evidence(info)[0]:
                continue
            # The public arrays, rather than latent generation state, must decode.
            parsed = _decode(inp)
            assert np.array_equal(out,_evaluate(parsed)[0])
            return dict(input=inp,output=out)
    raise RuntimeError('bounded shutter sampler exhausted')


def _signature(a):
    a = np.asarray(a,dtype=np.uint8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError('example counts must be integers')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    schedule = str(rng.choice(['scale','small_test','layers','word']))
    slots = list(map(int,rng.choice(num_train,2,replace=False)))
    requirements = [set() for _ in range(num_train)]
    requirements[slots[0]] = _CORE | {'marker_cover','delayed_sample','hidden_sample',
                                     'different_samples','exposed_clock'}
    requirements[slots[1]] = {'no_clear','opaque_clear','recolor_markers'}
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',num_train),('test',num_test)):
        for j in range(count):
            for retry in range(20):
                pair = _pair(rng,split,schedule,requirements[j] if split == 'train' else set())
                key = _signature(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not obtain distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _reference_output(info):
    """Generator-owned cross-check: cell-step tracing and per-pixel top layer.

    This deliberately does not call the maximal-run tracer or compositor. It
    is useful redundancy, not an independently authored semantic reference.
    """
    mask = info['mask']
    r,c = info['start']
    vertical, path = False, []
    while True:
        path.append((r,c))
        ahead = (r-1,c) if vertical else (r,c+1)
        if not _inside(mask.shape,ahead):
            break
        if mask[ahead]:
            r,c = ahead
            continue
        turn = (r,c+1) if vertical else (r-1,c)
        if not _inside(mask.shape,turn) or not mask[turn]:
            break
        r,c = turn
        vertical = not vertical
    samples = {}
    for k,p in enumerate(path):
        for j,b in enumerate(info['boxes']):
            if _contains(b['bounds'],p):
                samples.setdefault(j,info['word'][k % len(info['word'])])
                break
    path_colors = {p:info['word'][k % len(info['word'])] for k,p in enumerate(path)}
    out = info['grid'].copy()
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            p = (r,c)
            top = next((j for j,b in enumerate(info['boxes']) if _contains(b['bounds'],p)),None)
            if top is None:
                out[p] = path_colors.get(p,int(out[p]))
            else:
                b = info['boxes'][top]
                out[p] = b['cue'] if p in b['markers'] else samples.get(top,info['bg'])
    return out,path


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen,evidence = set(),set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            a,out = _grid(pair['input']),_grid(pair['output'])
            assert a.shape == out.shape and not np.array_equal(a,out)
            sig = _signature(a)
            assert sig not in seen
            seen.add(sig)
            info = _decode(a)
            expected,path = _reference_output(info)
            assert np.array_equal(out,expected)
            produced,details = _evaluate(info)
            assert path == details['path'] and len(path) == len(set(path))
            assert np.array_equal(produced,out)
            assert all(out[p] == a[p] for p in info['annotation'])
            footprint = set(path)
            for b in info['boxes']:
                footprint |= {(r,c) for r in range(b['bounds'][0],b['bounds'][2]+1)
                              for c in range(b['bounds'][1],b['bounds'][3]+1)}
            changed = set(map(tuple,np.argwhere(a != out)))
            assert changed <= footprint
            if split == 'train':
                evidence |= _evidence(info,out)[0]
    required = set(_ALTERNATIVES) | {'opaque_clear','marker_cover','delayed_sample',
                                   'hidden_sample','different_samples','exposed_clock'}
    assert required <= evidence, required-evidence


def _pattern(word):
    roles = {}
    return tuple(roles.setdefault(c,len(roles)) for c in word)


def self_test():
    """Bounded deterministic, semantic, counterfactual, and 42-seed sweep."""
    shapes,patterns,boxes,words,legs,topologies,palettes = set(),set(),set(),set(),set(),set(),set()
    puzzles = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.add(tuple(_signature(x['input']) for s in p for x in p[s]))
        for split in p:
            for pair in p[split]:
                info = _decode(pair['input'])
                out,d = _evaluate(info)
                assert np.array_equal(out,pair['output'])
                shapes.add(out.shape)
                patterns.add(_pattern(info['word']))
                boxes.add(len(info['boxes']))
                words.add(len(info['word']))
                legs.add(len(d['legs']))
                palettes.add((info['bg'],info['terrain']))
                bb = info['boxes']
                overlap = sum(max(a['bounds'][0],b['bounds'][0]) <= min(a['bounds'][2],b['bounds'][2])
                              and max(a['bounds'][1],b['bounds'][1]) <= min(a['bounds'][3],b['bounds'][3])
                              for j,a in enumerate(bb) for b in bb[j+1:])
                topologies.add((len(bb),overlap))
    assert len(puzzles) == 42 and len(shapes) >= 35 and len(patterns) >= 12
    assert boxes == {1,2,3,4} and words == {2,3,4,5,6}
    assert {1,2,3,4} <= legs and len(topologies) >= 6 and len(palettes) >= 45
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(314159,nt,ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    a,b = generate_puzzle(917,2,1),generate_puzzle(917,2,1)
    assert all(np.array_equal(x[k],y[k]) for s in a for x,y in zip(a[s],b[s]) for k in x)
    # Literal color identities carry no meaning, including the background role.
    for shift in range(10):
        permutation = np.roll(np.arange(10),shift)
        for pair in a['train']:
            assert np.array_equal(solve(permutation[pair['input']]),permutation[pair['output']])
    # Changing only the visible priority key must change a teaching answer.
    witness = next(pair for pair in a['train']
                   if 'reverse_priority' in _evidence(_decode(pair['input']))[0])
    info = _decode(witness['input'])
    altered = witness['input'].copy()
    for box,cue in zip(info['boxes'],[b['cue'] for b in reversed(info['boxes'])]):
        r,c = box['key']
        altered[r:r+2,c:c+2] = cue
    assert not np.array_equal(solve(altered),witness['output'])
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(3.0,1),(4,True),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Exiting right must stop, even with terrain available above the endpoint.
    m = np.zeros((5,8),bool)
    m[3,:] = True
    m[1:4,7] = True
    path,lengths = _trace(m,(3,0))
    assert path == [(3,c) for c in range(8)] and lengths == [8]
    # Vertical canvas exit likewise does not turn along the top row.
    m = np.zeros((7,9),bool)
    m[5,:4] = True
    m[:6,3] = True
    m[0,3:] = True
    path,lengths = _trace(m,(5,0))
    assert path == [(5,c) for c in range(4)] + [(r,3) for r in range(4,-1,-1)]
    # Full input-only boundary fixture, including both keys and a blank shutter.
    fixture = np.zeros((15,20),int)
    fixture[5:14,4:16] = 1
    fixture[:14,15] = 1
    fixture[0,15:] = 1
    fixture[1:3,1:3] = 2
    fixture[1:3,5:7] = 3
    fixture[7:9,1:3] = 4
    for p in _corners((6,7,10,12),0):
        fixture[p] = 4
    fixture[13,4] = 2
    output = solve(fixture)
    assert output[0,15] == 2 and output[0,16] == 1 and output[8,9] == 0
    assert np.array_equal(output,_reference_output(_decode(fixture))[0])


if __name__ == '__main__':
    self_test()
    print('self_test passed')
