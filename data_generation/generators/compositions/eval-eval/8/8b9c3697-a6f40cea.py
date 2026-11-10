"""Forks stretch matching patterned rectangle edges through an opaque sheet.

Only NumPy and the standard library are used. No generation metadata is read by
solve. All diagnostics in self_test are aggregated in memory; no files are made.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0,848890620,778644519,161005986,749092999,710264740,124804560,834639523,303470897,124602601,995267034,192948716,606691917,792294765,904219675,651587042,15678796,454893376,897459482,863221354,809507033,497102255,115788104,795338094,798554613,991216475,176592551,924770781,117865883,980116377,628881673,624300082,92103818,371442721,318495942,831790455,280847701,954706937,717319654,553499596,402652872,205038976)


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _rim(box):
    t, b, l, r = box
    return {(y, x) for y, x in _area(box) if y in (t, b) or x in (l, r)}


def _bbox(cells):
    ys, xs = zip(*cells)
    return min(ys), max(ys), min(xs), max(xs)


def _components(cells):
    remaining, groups = set(cells), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, group = [p], {p}
        while stack:
            y, x = stack.pop()
            for q in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _check_grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))
    return grid


def _fork(cells):
    """Parse a base with three positive spokes, in any orthogonal direction."""
    parses = []
    for out in ((1,0),(-1,0),(0,1),(0,-1)):
        perp = (-out[1], out[0])
        kmin = min(y*out[0]+x*out[1] for y,x in cells)
        qmin = min(y*perp[0]+x*perp[1] for y,x in cells)
        local = {(y*out[0]+x*out[1]-kmin,y*perp[0]+x*perp[1]-qmin) for y,x in cells}
        width = max(q for k,q in local)+1
        lengths = [sum((k,q) in local for k in range(1,31)) for q in range(width)]
        full = {(0,q) for q in range(width)} | {(k,q) for q,d in enumerate(lengths) for k in range(1,d+1)}
        if full != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((start,q,length))
        if len(runs) != 3 or any(runs[i][1] >= runs[i+1][0] for i in (0,1)):
            continue
        a,b,length = runs[1]
        if a+b != width:
            continue
        parses.append(dict(out=out,perp=perp,kmin=kmin,qmin=qmin,
                           q0=a,q1=b-1,stop=length+1,runs=tuple(runs),cells=cells))
    return parses[0] if len(parses) == 1 else None


def _cell(fork, k, q):
    k, q = k+fork['kmin'], q+fork['qmin']
    return (k*fork['out'][0]+q*fork['perp'][0], k*fork['out'][1]+q*fork['perp'][1])


def _local(fork, p):
    y,x = p
    return (y*fork['out'][0]+x*fork['out'][1]-fork['kmin'],
            y*fork['perp'][0]+x*fork['perp'][1]-fork['qmin'])


def _pattern_fits(grid, visible):
    fits = []
    for sign in (-1,1):
        for period in (1,2,3):
            word, counts = {}, [0]*period
            for y,x in sorted(visible):
                residue = (y+sign*x) % period
                color = int(grid[y,x])
                if residue in word and word[residue] != color:
                    break
                word[residue] = color
                counts[residue] += 1
            else:
                if min(counts) >= 2:
                    fits.append((sign,period,tuple(word[i] for i in range(period))))
    assert fits, 'outline lacks a witnessed period of at most three'
    return fits


def _colors(loop, cells):
    ordered = sorted(cells)
    predictions = {tuple(word[(y+sign*x) % period] for y,x in ordered)
                   for sign,period,word in loop['fits']}
    assert len(predictions) == 1, 'visible patterns disagree on reconstructed or extended pixels'
    return dict(zip(ordered,next(iter(predictions))))


def _read(input_grid):
    grid = _check_grid(input_grid)
    bg = int(grid[0,0])
    assert all(int(grid[p]) == bg for p in ((0,-1),(-1,0),(-1,-1)))
    frames = []
    for color in np.unique(grid):
        if int(color) == bg:
            continue
        for group in _components(map(tuple,np.argwhere(grid == color))):
            box = _bbox(group)
            t,b,l,r = box
            if b-t < 2 or r-l < 2 or group != _rim(box):
                continue
            interior = grid[t+1:b,l+1:r]
            if np.all(interior == interior[0,0]) and int(interior[0,0]) not in (bg,int(color)):
                frames.append((box,int(color),int(interior[0,0])))
    assert len(frames) == 1, 'expected one distinctly filled opaque window'
    box, frame_color, fill = frames[0]
    mask = _area(box)
    points = set(map(tuple,np.argwhere(grid != bg))) - mask
    forks, loops = [], []
    for group in _components(points):
        palette = {int(grid[p]) for p in group}
        parsed = _fork(group) if len(palette) == 1 else None
        if parsed is not None:
            parsed['color'] = palette.pop()
            forks.append(parsed)
        else:
            bounds = _bbox(group)
            t,b,l,r = bounds
            assert b-t >= 2 and r-l >= 2
            assert _rim(bounds)-mask == group, 'all rectangle extrema must be exposed'
            loops.append(dict(box=bounds,visible=group,fits=_pattern_fits(grid,group)))
    assert forks and loops
    structure = set().union(*(f['cells'] for f in forks))
    loop_rims = [_rim(loop['box']) for loop in loops]
    assert not structure.intersection(set().union(*loop_rims))
    matches, used, routes = [], set(), set()
    for fi,fork in enumerate(forks):
        candidates = []
        for li,loop in enumerate(loops):
            coords = [_local(fork,p) for p in loop_rims[li]]
            k0,k1 = min(k for k,q in coords),max(k for k,q in coords)
            q0,q1 = min(q for k,q in coords),max(q for k,q in coords)
            if (q0,q1) != (fork['q0'],fork['q1']) or k0 <= fork['stop']:
                continue
            sweep = {_cell(fork,k,q) for k in range(fork['stop'],k0+1) for q in range(q0,q1+1)}
            assert all(0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] for y,x in sweep)
            obstacles = structure | set().union(*(rim for j,rim in enumerate(loop_rims) if j != li))
            if sweep & obstacles:
                continue
            new_box = _bbox(loop_rims[li] | sweep)
            candidates.append(dict(fi=fi,li=li,sweep=sweep,new_box=new_box,
                                   old_side={_cell(fork,k0,q) for q in range(q0,q1+1)},
                                   anchor={_cell(fork,k1,q) for q in range(q0,q1+1)},
                                   distance=k0-fork['stop']))
        assert len(candidates) <= 1, 'a fork has multiple matching edges'
        if candidates:
            match = candidates[0]
            assert match['li'] not in used and not routes.intersection(match['sweep'])
            used.add(match['li'])
            routes.update(match['sweep'])
            matches.append(match)
    scene = dict(bg=bg,window=box,mask=mask,frame=frame_color,fill=fill,
                 forks=forks,loops=loops,matches=matches)
    # Check all original and all prospective extended colors before rendering.
    by_loop = {m['li']:m for m in matches}
    for li,loop in enumerate(loops):
        cells = _rim(loop['box']) | _rim(by_loop[li]['new_box'] if li in by_loop else loop['box'])
        _colors(loop,cells)
    return grid,scene


def _render(grid, scene, move=True):
    output = np.full(grid.shape,scene['bg'],dtype=np.int8)
    by_loop = {m['li']:m for m in scene['matches']} if move else {}
    for match in by_loop.values():
        _paint(output,match['sweep'],scene['forks'][match['fi']]['color'])
    for li,loop in enumerate(scene['loops']):
        box = by_loop[li]['new_box'] if li in by_loop else loop['box']
        for p,color in _colors(loop,_rim(box)).items():
            output[p] = color
    for fork in scene['forks']:
        _paint(output,fork['cells'],fork['color'])
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Remove the sheet and show stretched loops, their trails, and the forks."""
    grid,scene = _read(input_grid)
    return _render(grid,scene)


def _draw_loop(grid, box, word, phase, sign):
    for y,x in _rim(box):
        grid[y,x] = word[(y+sign*x+phase) % len(word)]


def _make_pair(rng, split, index, turn, bg):
    h,w = int(rng.integers(27,31)),int(rng.integers(27,31))
    t,b = int(rng.integers(9,12)),h-int(rng.integers(8,10))
    l,r = int(rng.integers(6,8)),w-int(rng.integers(6,8))
    window = (t,b,l,r)
    palette = list(map(int,rng.permutation([c for c in range(10) if c != bg])))
    frame,fill = palette[:2]
    fork_colors,ink = palette[2:4],palette[4:7]
    train = split == 'train'
    active = {index % 2} if train else {0,1}
    # The first two examples establish periods 1, 2, and 3 and both kinds of
    # unmatched geometry; tests compose two pulls with longer displacements.
    periods = ([2,1] if index == 0 else [2,3]) if train and index < 2 else list(map(int,rng.integers(1,4,size=2)))
    if train and index == 1:
        periods = [1,3]
    loops, forks, moves = [], [], []
    for j in range(2):
        if j == 0:
            x0,x1 = l-int(rng.integers(1,3)),l+int(rng.integers(1,4))
        else:
            x0,x1 = r-int(rng.integers(1,4)),r+int(rng.integers(1,3))
        span = x1-x0+1
        down = (j == 1) if not train or index < 2 else bool(rng.integers(2))
        inset = int(rng.integers(1,3) if train else rng.integers(4,6))
        if down:
            y0,y1 = t-int(rng.integers(1,4)),b-inset
            base = b+6 if train else h-2
            out = -1
        else:
            y0,y1 = t+inset,b+int(rng.integers(1,4))
            base = t-6 if train else int(rng.integers(1,3))
            out = 1
        assert y1-y0 >= 2
        period = periods[j]
        word = tuple(map(int,rng.permutation(ink)[:period]))
        phase,sign = int(rng.integers(period)),int(rng.choice([-1,1]))
        loops.append(((y0,y1,x0,x1),word,phase,sign))
        # An omitted fork and a wrong-width fork both demonstrate inactivity.
        present = j in active or index != 0 or not train
        if not present:
            continue
        fspan = span if j in active else span+2 if span <= 4 else span-2
        q0 = x0+(span-fspan)//2
        if train and j in active and period > 1:
            # A phase reset must disagree with a demonstrated extension.
            options = [d for d in range(1,4) if (5+inset-d) % period]
            center = int(rng.choice(options))
        else:
            center = int(rng.integers(1,4) if train else rng.integers(1,3))
        left,right = map(int,rng.integers(1,5,size=2))
        if j in active and max(left,right) == center:
            left = center % 4+1
        cells = {(base,x) for x in range(q0-2,q0+fspan+2)}
        cells |= {(base+out*k,x) for x,length in ((q0-2,left),(q0+fspan+1,right)) for k in range(1,length+1)}
        cells |= {(base+out*k,x) for k in range(1,center+1) for x in range(q0,q0+fspan)}
        forks.append((cells,fork_colors[j]))
        if j in active:
            final = base+out*(center+1)
            if down:
                new_box = (y0,final,x0,x1)
                sweep = (y1,final,x0,x1)
            else:
                new_box = (final,y1,x0,x1)
                sweep = (final,y0,x0,x1)
            moves.append((j,new_box,sweep,fork_colors[j]))
    # A small fully exposed loop varies the total count. It lies in the central
    # corridor, outside every fork and every possible swept strip.
    if r-l >= 16 and rng.random() < 0.8:
        ww,hh = int(rng.integers(3,min(5,r-l-13)+1)),int(rng.integers(3,5))
        xx = int(rng.integers(l+7,r-ww-6+1))
        yy = int(rng.integers(1,t-hh))
        period = int(rng.integers(1,4))
        loops.append(((yy,yy+hh-1,xx,xx+ww-1),tuple(ink[:period]),int(rng.integers(period)),int(rng.choice([-1,1]))))
    grid = np.full((h,w),bg,dtype=np.int8)
    output = grid.copy()
    for box,word,phase,sign in loops:
        _draw_loop(grid,box,word,phase,sign)
    changed = {j:new for j,new,sweep,color in moves}
    for j,new,sweep,color in moves:
        _paint(output,_area(sweep),color)
    for j,(box,word,phase,sign) in enumerate(loops):
        _draw_loop(output,changed.get(j,box),word,phase,sign)
    for cells,color in forks:
        assert all(0 <= y < h and 0 <= x < w for y,x in cells)
        _paint(grid,cells,color)
        _paint(output,cells,color)
    _paint(grid,_area(window),fill)
    _paint(grid,_rim(window),frame)
    if rng.integers(2):
        grid,output = np.fliplr(grid),np.fliplr(output)
    return {'input':np.rot90(grid,turn).copy(),'output':np.rot90(output,turn).copy()}


def _integer(value, name):
    if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)):
        raise ValueError(name+' must be an integer')
    return int(value)


def _key(grid):
    grid = np.asarray(grid,dtype=np.int8)
    return grid.shape,grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed,'seed')
    num_train,num_test = _integer(num_train,'num_train'),_integer(num_test,'num_test')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed % (1 << 64))
    backgrounds = list(map(int,rng.permutation(10)))
    orientation = int(rng.integers(4))
    puzzle,seen,first_trail = {'train':[],'test':[]},set(),None
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            bg = backgrounds[len(seen) % 10]
            turn = (orientation+index) % 4 if split == 'train' else int(rng.integers(4))
            for _ in range(24):
                pair = _make_pair(rng,split,index,turn,bg)
                key = _key(pair['input'])
                if key in seen:
                    continue
                # Bounded rejection is for accidental touching/ambiguous scenes;
                # ground truth above is rendered without invoking the solver.
                try:
                    predicted = solve(pair['input'])
                except AssertionError:
                    continue
                if np.array_equal(predicted,pair['output']):
                    if split == 'train' and index < 2:
                        _,scene = _read(pair['input'])
                        trail = scene['forks'][scene['matches'][0]['fi']]['color']
                        if index == 1 and trail == first_trail:
                            continue
                        first_trail = trail
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not construct a separated inferable scene')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen,train_periods,phase_witness,train_trails = set(),set(),False,set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            grid,output = _check_grid(pair['input']),_check_grid(pair['output'])
            assert grid.shape == output.shape
            key = _key(grid)
            assert key not in seen
            seen.add(key)
            _,scene = _read(grid)
            assert 2 <= len(scene['loops']) <= 3 and 1 <= len(scene['forks']) <= 2
            assert len(scene['matches']) == (1 if split == 'train' else 2)
            assert np.array_equal(_render(grid,scene),output)
            assert not np.array_equal(_render(grid,scene,move=False),output), 'omitting A must change the answer'
            assert not np.any(output == scene['frame']) and not np.any(output == scene['fill'])
            # Removing B destroys hidden geometry of the very edge A pulls, and
            # loses patterned connecting rails even when its final side is known.
            hidden_restored = 0
            for li,loop in enumerate(scene['loops']):
                if split == 'train':
                    train_periods.add(min(p for s,p,w in loop['fits']))
                match = next((m for m in scene['matches'] if m['li'] == li),None)
                box = match['new_box'] if match else loop['box']
                hidden_restored += len(_rim(box) & scene['mask'])
                if match:
                    assert match['old_side'] & scene['mask'] and match['old_side']-scene['mask']
                    fork = scene['forks'][match['fi']]
                    if split == 'train':
                        train_trails.add(fork['color'])
                        period = min(p for s,p,w in loop['fits'])
                        phase_witness |= period == 3 and match['distance'] % period != 0
                    visible_span = {_local(fork,p)[1] for p in match['old_side']-scene['mask']}
                    assert len(visible_span) < fork['q1']-fork['q0']+1
                    assert _rim(loop['box']) & scene['mask']
                    assert all(int(output[p]) == c for p,c in _colors(loop,match['anchor']).items())
                    interior_sweep = match['sweep']-_rim(box)
                    assert interior_sweep and all(int(output[p]) == scene['forks'][match['fi']]['color'] for p in interior_sweep)
                    assert 3 <= match['distance'] <= 6 if split == 'train' else match['distance'] >= 7
            assert hidden_restored >= 2, 'omitting B must lose reconstructed boundary cells'
            for fork in scene['forks']:
                assert min(run[2] for run in fork['runs']) >= 1
                assert all(int(output[p]) == fork['color'] for p in fork['cells'])
    assert {1,2,3} <= train_periods
    assert phase_witness and len(train_trails) >= 2


def self_test():
    """42 seeds, all legal counts, invariants, variation, and corruption tests."""
    shapes,geometries,structures,palettes,directions,periods,distances,counts = (set() for _ in range(8))
    digests = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for pairs in puzzle.values():
            for pair in pairs:
                grid = pair['input']
                _,scene = _read(grid)
                shapes.add(grid.shape)
                digest.update(grid.tobytes())
                roles = {}
                structures.add((grid.shape,tuple(roles.setdefault(int(c),len(roles)) for c in grid.flat)))
                palettes.add(tuple(sorted(map(int,np.unique(grid)))))
                counts.add((len(scene['loops']),len(scene['forks']),len(scene['matches'])))
                for loop in scene['loops']:
                    t,b,l,r = loop['box']
                    geometries.add((b-t+1,r-l+1))
                    periods.add(min(p for s,p,w in loop['fits']))
                for match in scene['matches']:
                    directions.add(scene['forks'][match['fi']]['out'])
                    distances.add(match['distance'])
        digests.add(digest.digest())
    assert len(digests) == 42 and len(structures) >= 180
    assert len(shapes) >= 10 and len(geometries) >= 30 and len(palettes) >= 15
    assert len(directions) == 4 and periods == {1,2,3} and len(distances) >= 7 and len(counts) >= 4
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            generate_puzzle(7021+nt*10+ne,nt,ne)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(3.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a,b = generate_puzzle(1234,2,1),generate_puzzle(1234,2,1)
    assert all(_key(pa[k]) == _key(pb[k]) for s in a for pa,pb in zip(a[s],b[s]) for k in ('input','output'))
    corrupted = generate_puzzle(7,2,1)
    corrupted['test'][0]['output'][0,0] = (int(corrupted['test'][0]['output'][0,0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt label accepted')
    return {'seeds':42,'pairs':210,'distinct_puzzles':len(digests),'input_shapes':len(shapes),
            'color_normalized_inputs':len(structures),'loop_dimensions':len(geometries),
            'palettes':len(palettes),'directions':len(directions),'periods':sorted(periods),
            'distances':sorted(map(int,distances)),'object_count_combinations':len(counts)}
