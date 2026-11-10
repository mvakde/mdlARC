"""A cyclic weave through a shadowed fan. Standard library and NumPy only.

No disk access, parent imports, fixtures-as-answers, or generation state in solve.
Run self_test() for bounded semantic regressions and 42-seed diversity checks.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0,418357350,708664179,179527429,9367262,100756668,186409009,
             982888443,896925505,422686364,367222370,21408620,95310391,
             184343402,324965022,58778842,388667578,530712850,978339534,
             217722758,462311504,51570831,871711021,979358719,787342999,
             132326606,348863533,150469564,70943561,870637252,574782731,
             71255280,469641921,198166937,537945938,989823556,360368831,
             953427125,652803392,468044851,984872010,474439418)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        group, pending = [first], [first]
        while pending:
            y, x = pending.pop()
            for q in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if q in remaining:
                    remaining.remove(q)
                    pending.append(q)
                    group.append(q)
        result.append(group)
    return result


def _tiles(a, background, wall):
    result = []
    for color in np.unique(a):
        if int(color) in (background, wall):
            continue
        for group in _components(a == color):
            if len(group) != 4:
                continue
            ys, xs = zip(*group)
            y, x = min(ys), min(xs)
            if max(ys) == y+1 and max(xs) == x+1:
                result.append((x, y, int(color)))
    return sorted(result)


def _fit(values, sign):
    """Longest periodic terminal suffix, at least two observed outward steps."""
    end = int(values[-1])
    candidates = []
    offsets = np.arange(1-len(values), 1)
    for period in (1, 2, 3):
        for phase in range(period):
            predicted = end + sign * ((offsets + phase) // period)
            count = 0
            for same in (predicted == values)[::-1]:
                if not same:
                    break
                count += 1
            if count >= 2*period+1 and len(set(map(int, values[-count:]))) >= 3:
                candidates.append((count, -period, phase))
    if not candidates:
        raise ValueError("terminal flank needs two full outward steps")
    length, negative_period, phase = max(candidates)
    return -negative_period, phase, length


def _decode(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= inp.size // 2:
        raise ValueError("background must occupy more than half of the input")
    candidates = []
    for rotation in range(4):
        a = np.rot90(inp, -rotation)
        h, w = a.shape
        for wall_value in np.unique(a):
            wall = int(wall_value)
            if wall == background:
                continue
            positions = np.argwhere(a == wall)
            if len(positions) < 17:
                continue
            rear, mouth = int(positions[:,1].min()), int(positions[:,1].max())
            cap = np.flatnonzero(a[:,rear] == wall)
            if (len(cap) < 7 or np.any(np.diff(cap) != 1)
                    or mouth-rear < 5 or mouth >= w-1):
                continue
            upper, lower = [], []
            for x in range(rear+1, mouth+1):
                rows = np.flatnonzero(a[:,x] == wall)
                if len(rows) != 2 or rows[1]-rows[0] < 6:
                    break
                upper.append(int(rows[0]))
                lower.append(int(rows[1]))
            else:
                if (upper[0] != int(cap[0]) or lower[0] != int(cap[-1])
                        or np.any(np.abs(np.diff(upper)) > 1)
                        or np.any(np.abs(np.diff(lower)) > 1)):
                    continue
                try:
                    fits = (_fit(np.array(upper), -1), _fit(np.array(lower), 1))
                except ValueError:
                    continue
                lo = np.zeros(w, dtype=int)
                hi = np.zeros(w, dtype=int)
                lo[rear+1:mouth+1], hi[rear+1:mouth+1] = upper, lower
                for x in range(mouth+1, w):
                    lo[x] = upper[-1] - ((x-mouth+fits[0][1]) // fits[0][0])
                    hi[x] = lower[-1] + ((x-mouth+fits[1][1]) // fits[1][0])
                yy, xx = np.indices(a.shape)
                footprint = (xx > rear) & (yy > lo) & (yy < hi)
                chamber = footprint & (xx <= mouth)
                tiles = _tiles(a, background, wall)
                if not 2 <= len(tiles) <= 5 or len({t[1] for t in tiles}) != 1:
                    continue
                if any(b[0]-t[0] not in (3,4) for t,b in zip(tiles,tiles[1:])):
                    continue
                if tiles[0][0] != rear+1:
                    continue
                tape = np.zeros_like(footprint)
                for x,y,color in tiles:
                    tape[y:y+2,x:x+2] = True
                if np.any(tape & ~chamber):
                    continue
                word = tuple(t[2] for t in tiles)
                if len(set(word)) < 2 or np.any(np.isin(a, word) & ~tape):
                    continue
                extras = (a != background) & (a != wall) & ~tape
                restored = a.copy()
                restored[tape] = background
                candidates.append(dict(a=a.copy(), rotation=rotation, bg=background,
                                       wall=wall, rear=rear, mouth=mouth, upper=lo,
                                       lower=hi, fits=fits, footprint=footprint,
                                       chamber=chamber, tape=tape, word=word,
                                       extras=extras, restored=restored))
    if len(candidates) != 1:
        raise ValueError("input must encode exactly one chamber and ordered tape")
    return candidates[0]


def _visible(info, *, shadow=True, straight=False):
    a, bg = info['restored'], info['bg']
    footprint = info['footprint'].copy()
    if straight:
        for x in range(info['mouth']+1, a.shape[1]):
            footprint[:,x] = footprint[:,info['mouth']]
    hidden = np.zeros(a.shape, dtype=bool)
    if shadow:
        for y,x in np.argwhere(info['extras'] & footprint):
            hidden[y,x:] = True
    return footprint & ~hidden & (a == bg)


def _render(info, variant="correct"):
    """Batch labeling for generation and explicit shortcut comparisons.

    solve uses a separate lane-state/phase walker, not this ordered list.
    """
    out = info['restored'].copy()
    word = info['word']
    if variant == 'flat':
        word = word[:1]
    elif variant == 'deduplicate':
        word = tuple(dict.fromkeys(word))
    elif variant == 'reverse_word':
        word = word[::-1]
    visible = _visible(info, shadow=variant != 'no_shadow', straight=variant == 'straight')
    phase_cells = (_visible(info, shadow=False) if variant == 'mask_after' else visible)
    visits = []
    for x in range(info['rear']+1, out.shape[1]):
        rows = list(map(int, np.flatnonzero(phase_cells[:,x])))
        if (x-info['rear']-1) % 2 and variant != 'one_way':
            rows.reverse()
        if variant == 'reset_slice':
            for i,y in enumerate(rows):
                out[y,x] = word[i % len(word)]
        else:
            visits.extend((y,x) for y in rows)
    if visits:
        ys, xs = np.array(visits).T
        out[ys,xs] = np.array(word, dtype=np.int8)[np.arange(len(visits)) % len(word)]
    if variant == 'mask_after':
        out[~visible] = info['restored'][~visible]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover geometry, tape, shadows, and weave entirely from this input."""
    info = _decode(input_grid)
    a, bg = info['restored'], info['bg']
    out = a.copy()
    live = np.zeros_like(info['footprint'])
    # A blocker outside the inferred footprint never turns off a lane.
    for y in range(a.shape[0]):
        blocked = False
        for x in range(info['rear']+1, a.shape[1]):
            if not info['footprint'][y,x]:
                continue
            if info['extras'][y,x]:
                blocked = True
            elif a[y,x] == bg and not blocked:
                live[y,x] = True
    phase = 0
    for x in range(info['rear']+1, a.shape[1]):
        rows = range(a.shape[0]) if (x-info['rear']-1) % 2 == 0 else range(a.shape[0]-1,-1,-1)
        for y in rows:
            if live[y,x]:
                out[y,x] = info['word'][phase % len(info['word'])]
                phase += 1
    return np.rot90(out, info['rotation']).copy()


_VARIANTS = ('flat','deduplicate','reverse_word','reset_slice','one_way',
             'no_shadow','mask_after','straight')
_CLEAN_EVIDENCE = {'flat','deduplicate','reverse_word','reset_slice','one_way','straight'}


def _evidence(info):
    expected = _render(info)
    return {v for v in _VARIANTS if not np.array_equal(expected, _render(info, v))}


def _word(rng, palette, repeated=False):
    length = int(rng.integers(3 if repeated else 2, 6))
    distinct = int(rng.integers(2, min(4,length-1 if repeated else length)+1))
    symbols = list(range(distinct)) + list(map(int, rng.integers(0, distinct, length-distinct)))
    rng.shuffle(symbols)
    return tuple(int(palette[2+s]) for s in symbols)


def _profile(rng, n):
    # Piecewise independent walls: lobes, necks, plateaus, and asymmetry.
    knots = np.unique(np.r_[0, rng.integers(0,n,size=3), n-1])
    return np.rint(np.interp(np.arange(n), knots, rng.integers(3,6,size=len(knots)))).astype(int)


def _geometry(rng, word, palette):
    gaps = rng.integers(1,3,size=len(word)-1)
    columns = [1]
    for gap in gaps:
        columns.append(columns[-1]+2+int(gap))
    tape_end = columns[-1]+1
    rear_length = tape_end + int(rng.integers(0,3))
    periods = [int(rng.integers(1,4)), int(rng.integers(1,4))]
    phases = [int(rng.integers(p)) for p in periods]
    flare_length = 2*max(periods)+1
    relative_mouth = rear_length + flare_length
    maximum_projection = min(10, 29-relative_mouth)
    if maximum_projection < 3:
        return None
    projection = int(rng.integers(3,maximum_projection+1))
    rear = int(rng.integers(0, min(3,29-relative_mouth-projection)+1))
    mouth = rear+relative_mouth
    profiles = [_profile(rng,rear_length) for _ in range(2)]
    ext = [np.r_[p, [int(p[-1])+(t+phase)//period for t in range(1,flare_length+1)]]
           for p,period,phase in zip(profiles,periods,phases)]
    top_span, bottom_span = int(max(ext[0])), int(max(ext[1]))
    minimum_height = max(12, top_span+bottom_span+1)
    height = int(rng.integers(minimum_height,31))
    center = int(rng.integers(top_span, height-bottom_span))
    width = mouth+1+projection
    bg, wall = map(int,palette[:2])
    a = np.full((height,width),bg,dtype=np.int8)
    a[center-int(profiles[0][0]):center+int(profiles[1][0])+1,rear] = wall
    lo,hi = np.zeros(width,dtype=int),np.zeros(width,dtype=int)
    for j,(u,d) in enumerate(zip(*ext),rear+1):
        lo[j],hi[j] = center-int(u),center+int(d)
        a[lo[j],j] = a[hi[j],j] = wall
    for x in range(mouth+1,width):
        t = x-(rear+rear_length)
        lo[x] = center-int(profiles[0][-1])-(t+phases[0])//periods[0]
        hi[x] = center+int(profiles[1][-1])+(t+phases[1])//periods[1]
    yy,xx = np.indices(a.shape)
    footprint = (xx>rear) & (yy>lo) & (yy<hi)
    tape = np.zeros_like(footprint)
    possible_rows = [y for y in range(height-1)
                     if all(np.all(footprint[y:y+2,rear+x:rear+x+2]) for x in columns)]
    y = int(rng.choice(possible_rows))
    for x,color in zip(columns,word):
        a[y:y+2,rear+x:rear+x+2] = color
        tape[y:y+2,rear+x:rear+x+2] = True
    return dict(a=a, bg=bg, wall=wall, rear=rear, mouth=mouth, upper=lo, lower=hi,
                footprint=footprint, tape=tape, word=word)


def _place(rng, model, color, offsets, kind, unique_rows=False, repeated_rows=()):
    a, fp, bg = model['a'], model['footprint'], model['bg']
    used = set(map(int, np.argwhere((a != bg) & (a != model['wall'])
                                  & ~model['tape'] & fp)[:,0]))
    options = []
    for y in range(a.shape[0]):
        for x in range(model['rear']+1,a.shape[1]):
            cells = [(y+dy,x+dx) for dy,dx in offsets]
            if any(not (0<=r<a.shape[0] and 0<=c<a.shape[1]) or a[r,c]!=bg for r,c in cells):
                continue
            inside = [bool(fp[p]) for p in cells]
            if kind == 'outside':
                if any(inside):
                    continue
            elif kind == 'straddle':
                if all(inside) or not any(inside) or x <= model['mouth']:
                    continue
            else:
                if not all(inside) or x > a.shape[1]-3:
                    continue
                if kind == 'internal' and x >= model['mouth']:
                    continue
                if kind == 'projected' and x <= model['mouth']:
                    continue
                if kind == 'repeat' and (y not in repeated_rows or x <= model['mouth']):
                    continue
            if unique_rows and any(r in used for (r,c),b in zip(cells,inside) if b):
                continue
            options.append(cells)
    if not options:
        return None
    cells = options[int(rng.integers(len(options)))]
    for p in cells:
        a[p] = color
    return cells


def _make_pair(rng, palette, rotation, style):
    # 96 bounded attempts; no files or retained failed grids.
    for _ in range(96):
        word = _word(rng,palette,repeated=style in ('clean_witness','blocked_witness'))
        model = _geometry(rng,word,palette)
        if model is None:
            continue
        colors = [int(palette[6]),int(palette[7])]
        successful = True
        if style in ('blocked_witness','light','merge'):
            first = _place(rng,model,colors[0],((0,0),),'internal',unique_rows=True)
            successful = first is not None
            if style == 'merge' and successful:
                rows = [p[0] for p in first]
                successful &= _place(rng,model,colors[1],((0,0),),'repeat',repeated_rows=rows) is not None
                successful &= _place(rng,model,colors[0],((0,0),(1,0),(2,0)),'straddle') is not None
                for j in range(int(rng.integers(0,3))):
                    offsets = (((0,0),(1,0),(1,1)) if j%2 == 0 else
                               ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2)))
                    _place(rng,model,colors[j%2],offsets,'any')
            elif successful:
                successful &= _place(rng,model,colors[1],((0,0),(1,0)),'projected',unique_rows=True) is not None
        # Inert objects share obstacle colors: opacity is spatial, not a color key.
        successful &= _place(rng,model,colors[0],((0,0),),'outside') is not None
        if not successful:
            continue
        a = model['a']
        try:
            info = _decode(a)
        except ValueError:
            continue
        # The visible decoder must recover the actual construction geometry.
        if (info['rotation'] != 0 or info['word'] != word
                or not np.array_equal(info['footprint'],model['footprint'])
                or not np.array_equal(info['tape'],model['tape'])):
            continue
        ev = _evidence(info)
        if style == 'clean_witness' and not _CLEAN_EVIDENCE <= ev:
            continue
        if style == 'blocked_witness' and not {'no_shadow','mask_after'} <= ev:
            continue
        out = _render(info)
        return {'input':np.rot90(a,rotation).copy(),'output':np.rot90(out,rotation).copy()}
    raise RuntimeError("bounded shadow-loom construction exhausted")


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 training pairs, 1..4 tests, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    shared_palette = bool(rng.integers(0,4) == 0)
    shared_direction = bool(rng.integers(0,3) == 0)
    base_palette, direction = rng.permutation(10),int(rng.integers(4))
    puzzle, seen = {'train':[],'test':[]},set()
    for split,count in (('train',int(num_train)),('test',int(num_test))):
        for i in range(count):
            style = ('merge' if split == 'test' else 'clean_witness' if i == 0
                     else 'blocked_witness' if i == 1 else str(rng.choice(['clean','light'])))
            palette = base_palette if shared_palette else rng.permutation(10)
            rotation = direction if shared_direction else int(rng.integers(4))
            for _ in range(8):
                pair = _make_pair(rng,palette,rotation,style)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            puzzle[split].append(pair)
            seen.add(key)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence = set(),set()
    clean = internal = outside = colored_blockers = False
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            canonical = np.rot90(out,-info['rotation'])
            assert np.array_equal(canonical,_render(info)), "wrong cyclic weave"
            assert np.array_equal(out,solve(inp)), "input-only walker mismatch"
            original = info['a']
            preserved = (original != info['bg']) & ~info['tape']
            assert np.array_equal(original[preserved],canonical[preserved]), "material changed"
            visible = _visible(info)
            assert np.all(canonical[~visible] == info['restored'][~visible])
            assert np.all(np.isin(canonical[visible],info['word']))
            assert np.any(inp != out)
            blockers = info['extras'] & info['footprint']
            if split == 'train':
                assert np.all(blockers.sum(axis=1) <= 1), "training shadow lanes must be distinct"
            else:
                assert np.any(blockers.sum(axis=1) >= 2), "test must superpose shadow lanes"
                groups = [g for color in np.unique(original[info['extras']])
                          for g in _components(original == color)]
                assert any(any(info['footprint'][p] for p in g)
                           and any(not info['footprint'][p] for p in g) for g in groups), "test needs a straddling object"
            # Thin virtual boundary cells and all shadows are left unpainted.
            for x in range(info['mouth']+1,original.shape[1]):
                for y in (info['upper'][x],info['lower'][x]):
                    if 0 <= y < original.shape[0]:
                        assert canonical[y,x] == original[y,x]
            if split == 'train':
                evidence |= _evidence(info)
                clean |= not np.any(blockers)
                internal |= bool(np.any(blockers & info['chamber']))
                outside |= bool(np.any(info['extras'] & ~info['footprint']))
                colored_blockers |= len(np.unique(original[blockers])) >= 2
    assert set(_VARIANTS) <= evidence, set(_VARIANTS)-evidence
    assert clean and internal and outside and colored_blockers, "missing demonstrations"


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train','test'):
        for pair in puzzle[split]:
            for role in ('input','output'):
                a = np.asarray(pair[role],dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _regressions():
    # Literal small footprint: deleting one visit changes the other lane's color.
    a = np.zeros((3,5),dtype=np.int8)
    fp = np.zeros_like(a,dtype=bool)
    fp[:2,1:] = True
    extra = np.zeros_like(fp)
    a[0,2],extra[0,2] = 9,True
    info = dict(restored=a,bg=0,wall=8,rear=0,mouth=1,
                footprint=fp,extras=extra,word=(2,2,3))
    expected = np.array([[0,2,9,0,0],[0,2,3,2,2],[0,0,0,0,0]],dtype=np.int8)
    assert np.array_equal(_render(info),expected)
    assert _render(info,'mask_after')[1,4] != expected[1,4]
    # An exterior obstacle upstream of a lane's entry must never cast a shadow.
    a = np.zeros((3,5),dtype=np.int8)
    fp = np.zeros_like(a,dtype=bool)
    fp[1,1:] = True
    fp[0,3:] = True
    extra = np.zeros_like(fp)
    a[0,1],extra[0,1] = 9,True
    info.update(restored=a,footprint=fp,extras=extra)
    assert _visible(info)[0,3]
    # Fully empty columns still alternate sweep direction; phase never advances.
    fp[:] = False
    fp[:2,1] = fp[:2,3] = True
    a[:] = 0
    extra[:] = False
    info.update(restored=a,footprint=fp,extras=extra,word=(2,3,4))
    result = _render(info)
    assert (result[0,3],result[1,3]) == (4,2)
    # Exact terminal phases and the first virtual corner, independently specified.
    for ys,period,phase,next_y in (((7,6,5,4),1,0,3),
                                  ((7,7,6,6,5,5,4),2,0,4),
                                  ((7,6,6,5,5,4,4),2,1,3),
                                  ((8,8,8,7,7,7,6,6,6,5),3,0,5)):
        p,q,_ = _fit(np.array(ys),-1)
        assert (p,q) == (period,phase)
        assert ys[-1]-(1+q)//p == next_y


def self_test():
    _regressions()
    signatures, shapes, outlines, patterns, palettes, directions = (set() for _ in range(6))
    periods, lengths, effective_counts, projections = set(),set(),set(),set()
    nonmonotone = asymmetric = clipped = repeated_lane = straddling = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _signature(puzzle) not in signatures
        signatures.add(_signature(puzzle))
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ('train','test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                a = info['a']
                shapes.add(a.shape)
                directions.add(info['rotation'])
                periods.update(f[0] for f in info['fits'])
                lengths.add(len(info['word']))
                roles = {}
                patterns.add(tuple(roles.setdefault(c,len(roles)) for c in info['word']))
                palettes.add((info['bg'],info['wall'],*info['word']))
                wall = a == info['wall']
                yy,xx = np.nonzero(wall)
                crop = wall[yy.min():yy.max()+1,xx.min():xx.max()+1]
                outlines.add(_key(crop))
                lo,hi = info['upper'][info['rear']+1:info['mouth']+1],info['lower'][info['rear']+1:info['mouth']+1]
                nonmonotone += bool(np.any(np.diff(hi-lo)<0))
                asymmetric += bool(np.any(lo+hi != lo[0]+hi[0]))
                clipped += bool(np.any(info['footprint'][0]) or np.any(info['footprint'][-1]))
                effective = info['extras'] & info['footprint']
                effective_counts.add(int(effective.sum()))
                projections.add(a.shape[1]-info['mouth']-1)
                repeated_lane += bool(np.any(effective.sum(axis=1)>1))
                for color in np.unique(a[info['extras']]):
                    straddling += sum(any(info['footprint'][p] for p in g)
                                      and any(not info['footprint'][p] for p in g)
                                      for g in _components(a==color))
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(outlines) >= 100
    assert len(patterns) >= 12 and len(palettes) >= 100
    assert directions == {0,1,2,3} and periods == {1,2,3} and lengths == {2,3,4,5}
    assert min(effective_counts) == 0 and max(effective_counts) >= 8
    assert len(projections) >= 6
    assert min(nonmonotone,asymmetric,clipped,repeated_lane,straddling) >= 20
    # All legal explicit counts, plus type and range errors. No retained grids.
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(1701+nt*11+ne,nt,ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(3.0,1),(3,True),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, 1.5, True):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    p = generate_puzzle(83,2,1)
    info = _decode(p['test'][0]['input'])
    bad = np.rot90(_render(info,'mask_after'),info['rotation']).copy()
    assert not np.array_equal(bad,p['test'][0]['output'])
    p['test'][0]['output'] = bad
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        return
    raise AssertionError("phase-on-shadow corruption accepted")
