"""Register a damaged tracing and transfer its true repairs to a visible atlas.

No parent imports, stored bitmaps, training labels, or random state enter solve.
All construction and diagnostics are bounded; self_test retains one puzzle at a time.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


HOST_SEEDS = (0,398061814,488095092,334925978,295766111,743834923,68757460,550832496,589247193,788991995,69896171,348490145,785445120,787962288,549377699,247224981,273074793,939281615,794016902,660886245,85885898,177078490,616651143,493540653,544850102,958869017,873393270,242445098,501813923,313174942,356313731,54049919,55805121,112380056,510624698,925180623,571656078,402151303,854006599,311978450,532310267,247449240)


def _d4(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _orthogonal_neighbors(mask):
    n = np.zeros(mask.shape, bool)
    n[1:] |= mask[:-1]
    n[:-1] |= mask[1:]
    n[:, 1:] |= mask[:, :-1]
    n[:, :-1] |= mask[:, 1:]
    return n


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1,1),(1,-1),(-1,1),(-1,-1)):
                q = r+dr, c+dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(cells)
    return result


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r,c])


def _thick(mask):
    return bool(np.any(mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:] & mask[1:,1:]))


def _gaps(base):
    """Three blanks with TWO visible collinear cells beyond each end."""
    result = []
    for dr, dc in ((0,1),(1,0)):
        for r, c in np.argwhere(base):
            r, c = int(r), int(c)
            if (all(_at(base,r+k*dr,c+k*dc) for k in (-1,0,4,5))
                    and all(not _at(base,r+k*dr,c+k*dc) for k in (1,2,3))
                    and 0 <= r+5*dr < base.shape[0] and 0 <= c+5*dc < base.shape[1]):
                result.append((r,c,dr,dc))
    return result


def _gap_cells(gap):
    r,c,dr,dc = gap
    return [(r+k*dr,c+k*dc) for k in (1,2,3)]


def _spurs(gap):
    r,c,dr,dc = gap
    for endpoint in (0,4):
        ar,ac = r+endpoint*dr,c+endpoint*dc
        toward = 1 if endpoint == 0 else -1
        for side in (-1,1):
            yield [(ar+k*(toward*dr+side*dc),ac+k*(toward*dc+side*dr)) for k in (1,2)]


def _tracing(mask):
    noise = mask & ~_orthogonal_neighbors(mask)
    parts = _components(noise)
    if not parts or any(len(p) != 2 or abs(p[0][0]-p[1][0]) != 1 or abs(p[0][1]-p[1][1]) != 1 for p in parts):
        raise ValueError('off-axis ink must consist of isolated diagonal pairs')
    base = mask & ~noise
    cues = [g for g in _gaps(base) if any(all(_at(noise,r,c) for r,c in s) for s in _spurs(g))]
    if not cues:
        raise ValueError('no bracketed, spur-marked gap')
    return base, noise, cues


def _panels(input_grid):
    a = np.asarray(input_grid)
    if a.ndim != 2 or not np.issubdtype(a.dtype,np.integer) or not all(1 <= d <= 30 for d in a.shape) or np.any((a < 0)|(a > 9)):
        raise ValueError('invalid ARC grid')
    bg = int(a[0,0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg) and np.all(a[:,0] == bg) and np.all(a[:,-1] == bg)):
        raise ValueError('outer margin must be background')
    colors = [int(c) for c in np.unique(a) if c != bg]
    if len(colors) != 2:
        raise ValueError('exactly two ink roles are required')
    panels = []
    for color in colors:
        rr,cc = np.nonzero(a == color)
        r0,r1,c0,c1 = int(rr.min()),int(rr.max()),int(cc.min()),int(cc.max())
        box = a[r0:r1+1,c0:c1+1]
        if min(box.shape) < 5 or not (np.all(box[0] == color) and np.all(box[-1] == color) and np.all(box[:,0] == color) and np.all(box[:,-1] == color)):
            raise ValueError('each ink role must have a solid rectangular frame')
        inside = box[1:-1,1:-1]
        if not np.all((inside == bg)|(inside == color)):
            raise ValueError('panels overlap or contain foreign ink')
        panels.append((inside.size,color,inside == color))
    panels.sort(key=lambda p:p[0],reverse=True)
    if panels[0][0] == panels[1][0]:
        raise ValueError('the atlas must be strictly larger than the tracing')
    return bg,panels[0][1],panels[1][1],panels[0][2],panels[1][2]


def _registration(atlas, tracing):
    """Joint exact matching under licensed repairs; deduplicate OUTPUTS, not poses."""
    answers = {}
    loose_min, loose_answers = None, set()
    for t in range(8):
        base,noise,cues = _tracing(_d4(tracing,t))
        h,w = base.shape
        H,W = atlas.shape
        if h > H or w > W:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(atlas,(h,w))
        # Retained straight ink is inviolable. Only absent ink may be restored.
        possible = np.all(~base | windows,axis=(-2,-1))
        for r,c in np.argwhere(possible):
            r,c = int(r),int(c)
            missing = windows[r,c] & ~base
            cost = int(missing.sum())
            marks = np.zeros_like(atlas)
            marks[r:r+h,c:c+w] = missing
            mk = _key(marks)
            if loose_min is None or cost < loose_min:
                loose_min,loose_answers = cost,{mk}
            elif cost == loose_min:
                loose_answers.add(mk)
            if not cost:
                continue
            licensed = np.zeros_like(base)
            repairs = []
            for gap in cues:
                cells = _gap_cells(gap)
                if all(missing[rr,cc] for rr,cc in cells):
                    repairs.append(gap)
                    for rr,cc in cells:
                        licensed[rr,cc] = True
            if repairs and np.array_equal(missing,licensed):
                used = {frozenset(s) for g in cues for s in _spurs(g) if all(_at(noise,rr,cc) for rr,cc in s)}
                answers.setdefault(mk,dict(marks=marks,transform=t,offset=(r,c),repairs=repairs,
                                           cues=cues,noise_cells=int(noise.sum()),clutter_pairs=int(noise.sum())//2-len(used),base=base))
    if len(answers) != 1:
        raise ValueError(f'tracing has {len(answers)} distinct licensed registrations')
    info = next(iter(answers.values()))
    info['loose_min'] = loose_min
    info['loose_answers'] = len(loose_answers)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    bg,ink,trace,atlas,sample = _panels(input_grid)
    info = _registration(atlas,sample)
    out = np.where(atlas,ink,bg).astype(np.int8)
    out[info['marks']] = trace
    return out


def _core(rng, count, width):
    if (count == 3 and width >= 9 or count == 4 and width == 11) and rng.random() < .55:
        # A broken right upright and two/three transverse rails. The shortened
        # middle rail leaves space for diagonal evidence on the upright.
        top = int(rng.integers(2))
        out = np.zeros((7+top+int(rng.integers(2)),width),bool)
        rows = [top,top+6] if count == 3 else [top,top+3,top+6]
        out[top:top+7,0] = out[top:top+7,-1] = True
        gaps = []
        for row in rows:
            right = width-5 if row == top+3 else width-1
            start = int(rng.integers(2,right-3))
            out[row,:right+1] = True
            gaps.append((row,start-1,0,1))
        gaps.append((top+1,width-1,1,0))
        for row in range(top+1,top+6):
            if row not in rows and rng.random() < .35 and not (out[row-1,1] or out[row+1,1]):
                out[row,1] = True
        base = out.copy()
        for gap in gaps:
            for r,c in _gap_cells(gap):
                base[r,c] = False
        return base,gaps
    spaces = [3] * (count-1) if count == 4 else list(map(int,rng.integers(3,5,count-1)))
    rows = np.r_[0,np.cumsum(spaces)].astype(int)
    top = int(rng.integers(2)) if rows[-1] < 9 else 0
    rows += top
    h = int(rows[-1])+1
    if h < 11 and rng.random() < .5:
        h += 1
    out = np.zeros((h,width),bool)
    out[int(rows[0]):int(rows[-1])+1,0] = True
    common = int(rng.integers(2,width-4))
    loop = rng.random() < .35
    gaps = []
    for row in rows:
        start = int(np.clip(common+rng.integers(-1,2),2,width-5))
        right = width-1 if loop else int(rng.integers(start+4,width))
        out[int(row),:right+1] = True
        gaps.append((int(row),start-1,0,1))
    if loop:
        out[int(rows[0]):int(rows[-1])+1,-1] = True
    # Unequal short side branches vary the retained context, without pixel noise.
    for row in range(int(rows[0])+1,int(rows[-1])):
        if row not in rows and rng.random() < .25 and not (out[row-1,1] or out[row+1,1]):
            out[row,1] = True
    base = out.copy()
    for gap in gaps:
        for r,c in _gap_cells(gap):
            base[r,c] = False
    return base,gaps


def _add_spurs(rng,base,gaps,indices):
    choices = []
    for i in indices:
        options = []
        for cells in _spurs(gaps[i]):
            if all(0 <= r < base.shape[0] and 0 <= c < base.shape[1] and not base[r,c] for r,c in cells):
                options.append(cells)
        rng.shuffle(options)
        choices.append(options)
    def visit(j,noise):
        if j == len(choices):
            return noise
        for cells in choices[j]:
            trial = noise.copy()
            for r,c in cells:
                trial[r,c] = True
            if int(trial.sum()) != 2*(j+1):
                continue
            together = base | trial
            if np.any(trial & _orthogonal_neighbors(together)):
                continue
            if any(len(p) != 2 for p in _components(trial)):
                continue
            result = visit(j+1,trial)
            if result is not None:
                return result
        return None
    return visit(0,np.zeros_like(base))


def _clutter(rng,sample):
    # Extra two-cell oblique near misses may disappear without licensing a gap.
    _,_,before = _tracing(sample)
    for _ in range(int(rng.integers(3))):
        for _ in range(24):
            r,c = int(rng.integers(sample.shape[0]-1)),int(rng.integers(sample.shape[1]-1))
            cells = [(r,c),(r+1,c+1)] if rng.integers(2) else [(r,c+1),(r+1,c)]
            if any(sample[rr,cc] for rr,cc in cells):
                continue
            trial = sample.copy()
            for rr,cc in cells:
                trial[rr,cc] = True
            try:
                base,noise,cues = _tracing(trial)
            except ValueError:
                continue
            if cues == before and all(noise[rr,cc] for rr,cc in cells):
                sample = trial
                break
    return sample


def _frame(mask,color,bg):
    result = np.full((mask.shape[0]+2,mask.shape[1]+2),color,dtype=np.int8)
    result[1:-1,1:-1] = np.where(mask,color,bg)
    return result


def _make_pair(rng,site_count,repair_count,copies,width,palette,pose):
    bg,ink,trace = palette
    for _ in range(160):
        base,gaps = _core(rng,site_count,width)
        order = list(map(int,rng.permutation(site_count)))
        real = set(order[:repair_count])
        false_cue = order[repair_count]
        unmarked = order[repair_count+1:]
        noise = _add_spurs(rng,base,gaps,sorted(real|{false_cue}))
        if noise is None:
            continue
        sample = _clutter(rng,base | noise)
        variants = [real]
        alternatives = [real-{a}|{b} for a in sorted(real) for b in unmarked]
        rng.shuffle(alternatives)
        variants += alternatives[:copies-1]
        rng.shuffle(variants)
        h,w = base.shape
        gap_width = 1 if copies == 3 else int(rng.integers(1,3))
        total_width = copies*w+(copies-1)*gap_width
        if total_width > 26:
            continue
        left = int(rng.integers(min(2,27-total_width)))
        right = int(rng.integers(min(2,27-total_width-left)))
        row_offsets = list(map(int,rng.integers(2,size=copies)))
        H = h+max(row_offsets)
        atlas = np.zeros((H,total_width+left+right),bool)
        marks = np.zeros_like(atlas)
        for j,filled in enumerate(variants):
            tile = base.copy()
            for k in filled:
                for r,c in _gap_cells(gaps[k]):
                    tile[r,c] = True
            r,c = row_offsets[j],left+j*(w+gap_width)
            atlas[r:r+h,c:c+w] = tile
            if filled == real:
                for k in real:
                    for rr,cc in _gap_cells(gaps[k]):
                        marks[r+rr,c+cc] = True
        sample = _d4(sample,pose)
        a,b = _frame(atlas,ink,bg),_frame(sample,trace,bg)
        H,W = a.shape[0]+b.shape[0]+3,max(a.shape[1],b.shape[1])+2
        if H > 30 or W > 30:
            continue
        inp = np.full((H,W),bg,dtype=np.int8)
        ca,cb = int(rng.integers(1,W-a.shape[1])),int(rng.integers(1,W-b.shape[1]))
        inp[1:1+a.shape[0],ca:ca+a.shape[1]] = a
        inp[2+a.shape[0]:2+a.shape[0]+b.shape[0],cb:cb+b.shape[1]] = b
        out = np.where(atlas,ink,bg).astype(np.int8)
        out[marks] = trace
        global_pose = int(rng.integers(8))
        inp,out = _d4(inp,global_pose),_d4(out,global_pose)
        try:
            _,_,_,visible_atlas,visible_sample = _panels(inp)
            info = _registration(visible_atlas,visible_sample)
        except ValueError:
            continue
        if (np.array_equal(solve(inp),out) and len(info['repairs']) == repair_count
                and len(info['cues']) > repair_count and info['loose_answers'] >= 2
                and info['loose_min'] == 3*repair_count):
            return {'input':inp,'output':out}
    raise RuntimeError('bounded construction could not find an unambiguous tracing')


def _count(name,value,lo,hi):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int,num_train: int = 4,num_test: int = 1) -> dict:
    nt,nq = _count('num_train',num_train,2,10),_count('num_test',num_test,1,4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))  # repairs, reference alternatives, compact context
    palette = tuple(map(int,rng.choice(10,3,replace=False)))
    recolor = bool(rng.integers(2))
    result,seen = {'train':[],'test':[]},set()
    poses = [0,1,4] if nt >= 3 else [0,1]
    pose_pool = list(range(8)) if nt >= 3 else list(range(4))
    for split,count in (('train',nt),('test',nq)):
        for i in range(count):
            sites = 4 if shift != 2 else int(rng.integers(3,5))
            repairs = 2 if split == 'test' and shift == 0 else 1
            if split == 'train' and i >= 2 and sites == 4 and rng.random() < .4:
                repairs = 2
            copies = 3 if split == 'test' and shift == 1 else 2
            width = 8 if copies == 3 or (split == 'test' and shift == 2) else int(rng.integers(9,12))
            colors = tuple(map(int,rng.choice(10,3,replace=False))) if recolor else palette
            pose = poses[i] if split == 'train' and i < len(poses) else int(rng.choice(pose_pool))
            for _ in range(8):
                pair = _make_pair(rng,sites,repairs,copies,width,colors,pose)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded duplicate rejection exhausted')
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    nt,nq = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen = set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = np.asarray(pair['input']),np.asarray(pair['output'])
            for grid in (inp,out):
                assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype,np.integer) and np.all((0 <= grid)&(grid <= 9))
            assert _key(inp) not in seen
            seen.add(_key(inp))
            bg,ink,trace,atlas,sample = _panels(inp)
            assert not _thick(atlas), 'reference strokes must remain one cell wide'
            assert np.all(~atlas | _orthogonal_neighbors(atlas)), 'no isolated reference ink'
            info = _registration(atlas,sample)
            assert np.array_equal(solve(inp),out), 'incomplete or incorrect repair map'
            assert out.shape == atlas.shape
            assert np.all((out == ink)|(out == trace)|(out == bg))
            assert np.array_equal(out != bg,atlas), 'repair highlights must preserve atlas geometry'
            assert np.all(atlas[info['marks']]) and int(info['marks'].sum()) == 3*len(info['repairs'])
            assert len(info['cues']) > len(info['repairs']), 'need a spur at an intentional gap'
            assert info['loose_answers'] >= 2 and info['loose_min'] == int(info['marks'].sum()), 'need an equally close unlicensed reference location'
            assert np.any((out == ink)) and np.any((out == trace))
    return True


def _signature(puzzle):
    digest = hashlib.sha256()
    for s in ('train','test'):
        for pair in puzzle[s]:
            for k in ('input','output'):
                a = np.asarray(pair[k],dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    signatures,shapes,atlases,palettes,repairs,cue_counts = set(),set(),set(),set(),set(),set()
    boundary_sides,rotations,repair_axes,clutter_counts = set(),set(),set(),set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        for split in ('train','test'):
            for pair in p[split]:
                bg,ink,trace,atlas,sample = _panels(pair['input'])
                info = _registration(atlas,sample)
                shapes.add(pair['input'].shape)
                atlases.add(min(_key(_d4(atlas,t)) for t in range(8)))
                palettes.add((bg,ink,trace))
                repairs.add(len(info['repairs']))
                cue_counts.add(len(info['cues']))
                repair_axes.add(''.join(sorted({'h' if g[2] == 0 else 'v' for g in info['repairs']})))
                clutter_counts.add(info['clutter_pairs'])
                rotations.add(info['transform'])
                marks = info['marks']
                for name,edge in (('top',marks[0]),('bottom',marks[-1]),('left',marks[:,0]),('right',marks[:,-1])):
                    if edge.any():
                        boundary_sides.add(name)
    assert len(signatures) == 42 and len(shapes) >= 20 and len(atlases) >= 100
    assert len(palettes) >= 30 and repairs == {1,2} and cue_counts >= {2,3}
    assert rotations == set(range(8)) and boundary_sides == {'top','bottom','left','right'}
    assert repair_axes == {'h','v','hv'} and max(clutter_counts) >= 1
    a = generate_puzzle(0)
    assert _signature(a) == _signature(generate_puzzle(0))
    # Full-content validation rejects an omitted real repair and a spurious one.
    for kind in ('omitted','extra'):
        bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in a[s]] for s in ('train','test')}
        pair = bad['test'][0]
        bg,ink,trace,atlas,_ = _panels(pair['input'])
        coords = np.argwhere(pair['output'] == (trace if kind == 'omitted' else ink))
        r,c = coords[0]
        pair['output'][r,c] = ink if kind == 'omitted' else trace
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('bad repair map accepted')
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(1234,nt,nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Equivariance under whole-input D4 and two bijective color-role changes.
    pair = a['train'][0]
    for t in range(8):
        assert np.array_equal(solve(_d4(pair['input'],t)),_d4(pair['output'],t))
    for mapping in (np.arange(9,-1,-1),np.roll(np.arange(10),3)):
        assert np.array_equal(solve(mapping[pair['input']]),mapping[pair['output']])
    return dict(seeds=42,distinct_puzzles=len(signatures),input_shapes=len(shapes),
                atlas_classes=len(atlases),role_palettes=len(palettes),repair_counts=sorted(repairs),
                cue_counts=sorted(cue_counts),repair_axes=sorted(repair_axes),extra_diagonal_pairs=sorted(clutter_counts),
                registration_transforms=sorted(rotations),boundary_sides=sorted(boundary_sides))


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
