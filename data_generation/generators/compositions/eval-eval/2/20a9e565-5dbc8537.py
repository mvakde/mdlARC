"""Phase-locked patchwork: extrapolate stitches while packing finite cloth patches.

No file I/O, external state, parent imports, or generation metadata is used by solve.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0,903945263,912144564,680453431,304598247,84492249,724375775,437570276,422056986,616189948,971299214,721494386,594906061,804109403,802659233,446930062,463002698,965416470,230841148,266287211,865142119,634240443,790266481,795204089,70632873,629233026,202228120,88584254,852234385,775157220,626346645,136036297,62763147,55088863,733993923,666963489,98608948,313167033,832604928,86379576,22821831,201848904)


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError('Grid dimensions must be 1..30')
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError('Grid cells must be integers in 0..9')
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(a, b):
    for name, v, lo, hi in (('num_train', a, 2, 10), ('num_test', b, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ValueError(name + ' is outside its legal integer range')
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    return int(a), int(b)


def _corners(bounds):
    y0, y1, x0, x1 = bounds
    return {(y0,x0),(y0,x0+1),(y0+1,x0),(y1,x1),(y1,x1-1),(y1-1,x1)}


def _find_marker(a):
    candidates = []
    for color in np.unique(a):
        cells = np.argwhere(a == color)
        if len(cells) != 6:
            continue
        y0, x0 = map(int, cells.min(axis=0))
        y1, x1 = map(int, cells.max(axis=0))
        bounds = y0, y1, x0, x1
        if y1-y0 < 2 or x1-x0 < 2:
            continue
        if set(map(tuple, cells)) == _corners(bounds):
            candidates.append((int(color), bounds))
    if len(candidates) != 1:
        raise ValueError('Expected exactly two canonical corner Ls')
    return candidates[0]


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for yy, xx in np.argwhere(mask):
        y, x = int(yy), int(xx)
        if seen[y,x]:
            continue
        todo, cells = [(y,x)], []
        seen[y,x] = True
        while todo:
            r, c = todo.pop()
            cells.append((r,c))
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr,nc] and not seen[nr,nc]:
                    seen[nr,nc] = True
                    todo.append((nr,nc))
        ys, xs = zip(*cells)
        bounds = min(ys), max(ys), min(xs), max(xs)
        if (bounds[1]-bounds[0]+1)*(bounds[3]-bounds[2]+1) != len(cells):
            raise ValueError('A cloth component is not rectangular')
        result.append(bounds)
    return result


def _guide_models(witness, target_height, gap):
    """Fit observed geometry AND cyclic colors; retain distinct future fields."""
    values, counts = np.unique(witness, return_counts=True)
    base = int(values[np.argmax(counts)])
    if counts.max()*2 <= witness.size:
        return []
    colors = tuple(int(c) for c in values if c != base)
    p = len(colors)
    if p not in (2,3) or witness.shape[0] < 2*p+1:
        return []
    h, w = witness.shape
    rr, cc = np.indices(witness.shape)
    observed = witness != base
    answers = {}
    for pitch in range(3,7):
        for run in range(1,3):
            if 2*run >= pitch:
                continue
            for phase in range(pitch):
                for step in range(-2,3):
                    for accel in (-1,0,1):
                        offset = phase + step*rr + accel*rr*(rr-1)//2
                        lane, within = np.divmod(cc-offset, pitch)
                        mask = within < run
                        if not np.array_equal(mask, observed):
                            continue
                        residue = (lane+rr) % p
                        palette = []
                        for k in range(p):
                            used = np.unique(witness[mask & (residue == k)])
                            if len(used) != 1:
                                break
                            palette.append(int(used[0]))
                        if len(palette) != p or set(palette) != set(colors):
                            continue
                        tr = np.arange(h+gap,h+gap+target_height)[:,None]
                        tc = np.arange(w)[None,:]
                        lane2, within2 = np.divmod(tc-phase-step*tr-accel*tr*(tr-1)//2, pitch)
                        field = np.full((target_height,w), -1, dtype=np.int8)
                        m = within2 < run
                        colorfield = np.asarray(palette, dtype=np.int8)[(lane2+tr) % p]
                        field[m] = colorfield[m]
                        answers.setdefault(field.tobytes(), (field, dict(pitch=pitch,run=run,phase=phase,step=step,accel=accel,cycle=p,base=base,guide_colors=colors)))
    return list(answers.values())


def _pack(field, tiles, guide_colors, *, limit=18000, ignore_guides=False):
    """Enumerate rendered completions; identical physical copies have finite counts."""
    h, w = field.shape
    groups, lookup = [], {}
    for tile in tiles:
        key = _key(tile)
        if key in lookup:
            groups[lookup[key]][1] += 1
        else:
            lookup[key] = len(groups)
            groups.append([tile,1])
    placements, by_cell = [], [[] for _ in range(h*w)]
    group_options = [0]*len(groups)
    for g, (tile, _) in enumerate(groups):
        th, tw = tile.shape
        signature = np.where(np.isin(tile, guide_colors), tile, -1)
        for y in range(h-th+1):
            for x in range(w-tw+1):
                if not ignore_guides and not np.array_equal(signature, field[y:y+th,x:x+tw]):
                    continue
                indices = tuple((y+dy)*w+x+dx for dy in range(th) for dx in range(tw))
                bits = sum(1 << i for i in indices)
                pid = len(placements)
                placements.append((g,bits,indices,tile.reshape(-1)))
                group_options[g] += 1
                for i in indices:
                    by_cell[i].append(pid)
    remaining = [n for _,n in groups]
    full = (1 << (h*w))-1
    rendered = np.full(h*w,-1,dtype=np.int8)
    outcomes, dead, path = {}, set(), []
    nodes, finished = 0, True

    def search(covered):
        nonlocal nodes, finished
        if len(outcomes) >= 2 or not finished:
            return
        nodes += 1
        if nodes > limit:
            finished = False
            return
        if covered == full:
            outcomes.setdefault(rendered.tobytes(), (rendered.reshape(h,w).copy(),tuple(path)))
            return
        state = covered, tuple(remaining)
        if state in dead:
            return
        if sum(n*t.size for n,(t,_) in zip(remaining,groups)) < h*w-covered.bit_count():
            dead.add(state)
            return
        options = None
        bitsleft = full ^ covered
        while bitsleft:
            bit = bitsleft & -bitsleft
            i = bit.bit_length()-1
            candidates = [pid for pid in by_cell[i] if remaining[placements[pid][0]] and not covered & placements[pid][1]]
            if not candidates:
                dead.add(state)
                return
            if options is None or len(candidates) < len(options):
                options = candidates
                if len(options) == 1:
                    break
            bitsleft ^= bit
        for pid in options:
            g, mask, indices, colors = placements[pid]
            remaining[g] -= 1
            rendered[list(indices)] = colors
            path.append((g,indices))
            search(covered | mask)
            path.pop()
            rendered[list(indices)] = -1
            remaining[g] += 1
            if len(outcomes) >= 2 or not finished:
                return
        # No dead memoization after branches: equal renderings can share suffixes.

    search(0)
    return list(outcomes.values()), finished, dict(nodes=nodes,groups=groups,placements=len(placements),group_options=group_options)


def _analyze(input_grid):
    original = _grid(input_grid)
    marker, bounds = _find_marker(original)
    values, counts = np.unique(original, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if bg == marker or counts.max()*2 <= original.size:
        raise ValueError('The empty background must occupy more than half the canvas')
    targetmask = np.zeros(original.shape, dtype=bool)
    y0,y1,x0,x1 = bounds
    targetmask[y0:y1+1,x0:x1+1] = True
    if np.any((original[targetmask] != bg) & (original[targetmask] != marker)):
        raise ValueError('The marked window must be erased')
    clean = original.copy()
    clean[clean == marker] = bg
    candidates = []
    for rotation in range(4):
        a = np.rot90(clean,rotation)
        tm = np.rot90(targetmask,rotation)
        pos = np.argwhere(tm)
        r0,c0 = map(int,pos.min(axis=0))
        r1,c1 = map(int,pos.max(axis=0))
        components = _components(a != bg)
        witnesses = [b for b in components if b[2] == c0 and b[3] == c1 and 0 <= r0-b[1]-1 <= 3 and b[1] < r0 and b[1]-b[0]+1 >= 5]
        for wb in witnesses:
            wy0,wy1,wx0,wx1 = wb
            witness = a[wy0:wy1+1,wx0:wx1+1]
            gap = r0-wy1-1
            tiles = [a[t:b+1,l:r+1].copy() for t,b,l,r in components if (t,b,l,r) != wb]
            if not 4 <= len(tiles) <= 11:
                continue
            for field, model in _guide_models(witness,r1-r0+1,gap):
                # Guide and fabric roles are established by the intact strip.
                colors = model['guide_colors']
                outcomes, done, stats = _pack(field,tiles,colors)
                if not done:
                    raise ValueError('Unresolved packing search budget')
                for image,path in outcomes:
                    out = np.rot90(image,-rotation).copy()
                    info = dict(model,rotation=rotation,gap=gap,witness_shape=witness.shape,bounds=bounds,background=bg,marker=marker,tiles=tiles,field=field,path=path,stats=stats,output_canonical=image)
                    candidates.append((out,info))
                    if len({_key(v[0]) for v in candidates}) > 1:
                        raise ValueError('More than one stitched completion')
    if not candidates:
        raise ValueError('No continuation-compatible patch packing')
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the uniquely packed inclusive marker crop, using this input alone."""
    return _analyze(input_grid)[0]


def _paint_stitches(height, width, start_row, pitch, run, phase, step, accel, palette):
    """Construction renderer: direct cells, independent of candidate fitting."""
    a = np.full((height,width),-1,dtype=np.int8)
    for y in range(height):
        row = y+start_row
        shift = phase+step*row+accel*row*(row-1)//2
        for x in range(width):
            lane, inside = divmod(x-shift,pitch)
            if inside < run:
                a[y,x] = palette[(lane+row) % len(palette)]
    return a


def _partition(rng, h, w, n):
    rectangles = [(0,0,h,w)]
    while len(rectangles) < n:
        eligible = [i for i,(_,_,hh,ww) in enumerate(rectangles) if hh*ww >= 4]
        if not eligible:
            raise ValueError('Cannot further split the cloth')
        weights = np.asarray([rectangles[i][2]*rectangles[i][3] for i in eligible],dtype=float)
        idx = int(rng.choice(eligible,p=weights/weights.sum()))
        y,x,hh,ww = rectangles.pop(idx)
        axes = [axis for axis,length in enumerate((hh,ww)) if length >= 2]
        axis = int(rng.choice(axes))
        cut = int(rng.integers(1,(hh,ww)[axis]))
        rectangles.extend([(y,x,cut,ww),(y+cut,x,hh-cut,ww)] if axis == 0 else [(y,x,hh,cut),(y,x+cut,hh,ww-cut)])
    return rectangles


def _texture(rng, h, w, palette):
    yy,xx = np.indices((h,w))
    a,b = map(int,rng.choice(palette,size=2,replace=False))
    mode = int(rng.integers(6))
    scale = int(rng.integers(1,3))
    if mode == 0:
        mask = np.zeros((h,w),dtype=bool)
    elif mode == 1:
        mask = (yy//scale) % 2 == 0
    elif mode == 2:
        mask = (xx//scale) % 2 == 0
    elif mode == 3:
        mask = ((yy//scale)+(xx//scale)) % 2 == 0
    elif mode == 4:
        mask = (yy == 0) & (xx == 0)
    else:
        mask = (yy+xx) % (scale+2) == 0
    return np.where(mask,b,a).astype(np.int8)


def _scatter(rng, witness, target, tiles, gap, bg, marker, rotation):
    bh,bw = witness.shape
    th,tw = target.shape
    # The cloth sits at a random offset; patches scatter throughout free space.
    height = int(rng.integers(max(18,bh+gap+th),31))
    width = min(30,bw+int(rng.integers(14,18)))
    top = int(rng.integers(0,height-bh-gap-th+1))
    left = int(rng.integers(0,min(4,width-bw)+1))
    a = np.full((height,width),bg,dtype=np.int8)
    reserved = np.zeros(a.shape,dtype=bool)
    reserved[max(0,top-1):min(height,top+bh+gap+th+1),max(0,left-1):min(width,left+bw+1)] = True
    a[top:top+bh,left:left+bw] = witness
    positions = {}
    order = sorted(range(len(tiles)),key=lambda i:-tiles[i].size)
    for i in order:
        h,w = tiles[i].shape
        legal = []
        for y in range(height-h+1):
            for x in range(width-w+1):
                if not reserved[y:y+h,x:x+w].any():
                    legal.append((y,x))
        if not legal:
            return None
        y,x = legal[int(rng.integers(len(legal)))]
        positions[i] = (y,x)
        reserved[max(0,y-1):min(height,y+h+1),max(0,x-1):min(width,x+w+1)] = True
        a[y:y+h,x:x+w] = tiles[i]
    r0 = top+bh+gap
    mask = np.zeros(a.shape,dtype=bool)
    mask[r0:r0+th,left:left+tw] = True
    a,mask = np.rot90(a,rotation).copy(),np.rot90(mask,rotation)
    target = np.rot90(target,rotation).copy()
    coords = np.argwhere(mask)
    y0,x0 = map(int,coords.min(axis=0))
    y1,x1 = map(int,coords.max(axis=0))
    for y,x in _corners((y0,y1,x0,x1)):
        a[y,x] = marker
    return {'input':a,'output':target}


def _sample_pair(rng, rotation, gap, pieces, fallback=False, require_packing=False):
    roles = list(map(int,rng.permutation(10)))
    bg,marker = roles[:2]
    p = 3 if fallback else int(rng.integers(2,4))
    guide = roles[2:2+p]
    fabric = roles[2+p:]
    base = fabric[0]
    pitch = 5 if fallback else int(rng.integers(3,7))
    run = 1 if fallback or pitch < 5 else int(rng.integers(1,3))
    step = 1 if fallback else int(rng.integers(-2,3))
    accel = 0 if fallback else int(rng.integers(-1,2))
    phase = int(rng.integers(pitch))
    bh = 2*p+1
    th = pieces+2 if fallback else int(rng.integers(4,9))
    tw = 11 if fallback else int(rng.integers(9,14))
    if fallback:
        # A plain singleton fits many material positions. The other patches
        # determine its one remaining destination through complete coverage.
        while (-phase-step*(bh+gap)) % pitch < run:
            phase = (phase+1) % pitch
    visible = _paint_stitches(bh,tw,0,pitch,run,phase,step,accel,guide)
    witness = np.where(visible >= 0,visible,base).astype(np.int8)
    field = _paint_stitches(th,tw,bh+gap,pitch,run,phase,step,accel,guide)
    rects = ([(0,0,1,1),(0,1,1,tw-1)]+
             [(i,0,1,tw) for i in range(1,pieces-2)]+[(pieces-2,0,4,tw)]
             if fallback else _partition(rng,th,tw,pieces))
    target = np.full((th,tw),base,dtype=np.int8)
    tiles = []
    for y,x,h,w in rects:
        tile = _texture(rng,h,w,fabric)
        local = field[y:y+h,x:x+w]
        tile[local >= 0] = local[local >= 0]
        target[y:y+h,x:x+w] = tile
        tiles.append(tile)
    # A same-size off-phase competitor makes geometry alone insufficient.
    eligible = [i for i,t in enumerate(tiles) if np.isin(t,guide).any()]
    if not eligible or not np.any((field < 0) & (target != base)):
        return None
    ndecoys = 1 if fallback else int(rng.integers(1,3))
    for _ in range(ndecoys):
        source = tiles[int(rng.choice(eligible))]
        decoy = source.copy()
        original = source.copy()
        for j,color in enumerate(guide):
            decoy[original == color] = guide[(j+1) % p]
        tiles.append(decoy)
    outcomes,done,_ = _pack(field,tiles,guide)
    if not done or len(outcomes) != 1 or not np.array_equal(outcomes[0][0],target):
        return None
    pair = _scatter(rng,witness,target,tiles,gap,bg,marker,rotation)
    if pair is None:
        return None
    try:
        prediction,info = _analyze(pair['input'])
    except ValueError:
        return None
    if require_packing and not any(info['stats']['group_options'][g] > 1 for g,_ in info['path']):
        return None
    return pair if np.array_equal(prediction,pair['output']) else None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    orientation = int(rng.integers(4))
    shift = int(rng.integers(3))  # distance, direction, fragmentation
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            test = split == 'test'
            rotation = ((orientation+int(rng.integers(1,4))) % 4 if test and shift == 1 else orientation)
            gap = int(rng.integers(2,4)) if test and shift == 0 else int(rng.integers(0,2))
            if shift == 2:
                pieces = int(rng.integers(6,9)) if test else int(rng.integers(3,6))
            else:
                pieces = int(rng.integers(3,7))
            for attempt in range(100):
                pair = _sample_pair(rng,rotation,gap,pieces,fallback=attempt >= 80,
                                    require_packing=split == 'train' and index == 0)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Bounded patch construction exhausted')
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen = set()
    taught_packing = False
    for index,pair in enumerate(puzzle['train']+puzzle['test']):
        assert isinstance(pair,dict) and set(pair) == {'input','output'}
        a,b = _grid(pair['input']),_grid(pair['output'])
        assert _key(a) not in seen
        seen.add(_key(a))
        predicted,info = _analyze(a)
        assert np.array_equal(predicted,b), 'Input-derived packed answer differs'
        assert info['marker'] not in b and info['background'] not in b
        y0,y1,x0,x1 = info['bounds']
        assert b.shape == (y1-y0+1,x1-x0+1)
        assert np.array_equal(np.where(np.isin(info['output_canonical'],info['guide_colors']),info['output_canonical'],-1),info['field'])
        # The tiling path uses finite physical inventory and covers every cell once.
        inventory = [n for _,n in info['stats']['groups']]
        covered = set()
        for group,cells in info['path']:
            inventory[group] -= 1
            assert inventory[group] >= 0 and not covered.intersection(cells)
            covered.update(cells)
        assert len(covered) == b.size and sum(inventory) >= 1
        # Witnesses for both parent omissions are derived from the returned input.
        used_groups = {g for g,_ in info['path']}
        if index < len(puzzle['train']):
            taught_packing |= any(info['stats']['group_options'][g] > 1 for g in used_groups)
        unused = [t for n,(t,_) in zip(inventory,info['stats']['groups']) if n]
        assert any(t.shape == u.shape and not np.array_equal(t,u)
                   for g in used_groups for t in [info['stats']['groups'][g][0]]
                   for u in unused), 'A same-footprint unused competitor must change the geometric answer'
        painted = np.where(info['field'] >= 0,info['field'],info['base'])
        assert not np.array_equal(painted,info['output_canonical']), 'Fabric information must matter'
        assert set(np.unique(b)) <= set(np.unique(a))
    assert taught_packing, 'Training must show global coverage resolving a local placement choice'


def self_test() -> dict:
    """42 fixed seeds, bounded working memory, input-derived diversity and ablations."""
    signatures,shapes,orientations,gaps,counts,pitches,accelerations,cycles,textures = set(),set(),set(),set(),set(),set(),set(),set(),set()
    pairs,nodes,peak_nodes,packing_choices,used_stitchless = 0,0,0,0,0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for p in puzzle['train']+puzzle['test']))
        for pair in puzzle['train']+puzzle['test']:
            _,f = _analyze(pair['input'])
            pairs += 1
            shapes.add((pair['input'].shape,pair['output'].shape))
            orientations.add(f['rotation']); gaps.add(f['gap'])
            counts.add(len(f['path'])); pitches.add(f['pitch']); accelerations.add(f['accel']); cycles.add(f['cycle'])
            textures.add(_key(f['output_canonical']))
            nodes += f['stats']['nodes']; peak_nodes = max(peak_nodes,f['stats']['nodes'])
            used = {g for g,_ in f['path']}
            packing_choices += any(f['stats']['group_options'][g] > 1 for g in used)
            used_stitchless += any(not np.isin(f['stats']['groups'][g][0],f['guide_colors']).any() for g in used)
            if pair is puzzle['train'][0]:
                alternatives,done,_ = _pack(f['field'],f['tiles'],f['guide_colors'],ignore_guides=True)
                assert len(alternatives) == 2, 'Omitting continuation should leave different images'
    assert len(signatures) == 42 and len(shapes) >= 80 and len(textures) == pairs
    assert orientations == {0,1,2,3} and gaps == {0,1,2,3}
    assert len(counts) >= 4 and len(pitches) >= 3 and accelerations == {-1,0,1} and cycles == {2,3}
    assert packing_choices >= 42 and used_stitchless > 0
    assert tuple(_key(p['input']) for p in generate_puzzle(73)['train']) == tuple(_key(p['input']) for p in generate_puzzle(73)['train'])
    # Finite inventory and indistinguishable physical copies, without a guide mask.
    field = np.full((1,4),-1,dtype=np.int8)
    tile = np.array([[2,3]],dtype=np.int8)
    assert len(_pack(field,[tile],(),ignore_guides=True)[0]) == 0
    assert len(_pack(field,[tile,tile],(),ignore_guides=True)[0]) == 1
    # Exercise the simpler fallback explicitly; it retains the requested count.
    fallback_cases = 0
    rng = np.random.default_rng(12345)
    for rotation in range(4):
        for pieces in (3,8):
            for attempt in range(20):
                pair = _sample_pair(rng,rotation,3,pieces,fallback=True,require_packing=True)
                if pair is not None:
                    prediction,info = _analyze(pair['input'])
                    assert len(info['path']) == pieces and np.array_equal(prediction,pair['output'])
                    fallback_cases += 1
                    break
            else:
                raise AssertionError('Fallback failed its bounded construction check')
    bad = generate_puzzle(73,2,1)
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (ValueError,AssertionError):
        pass
    else:
        raise AssertionError('Corrupted label accepted')
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    return dict(seeds=42,pairs=pairs,distinct_puzzles=len(signatures),shape_combinations=len(shapes),orientations=sorted(orientations),gaps=sorted(gaps),piece_counts=sorted(counts),pitches=sorted(pitches),accelerations=sorted(accelerations),cycles=sorted(cycles),packing_nodes=nodes,max_packing_nodes=peak_nodes,locally_ambiguous_selected_patch_pairs=packing_choices,stitchless_selected_patch_pairs=used_stitchless,fallback_cases=fallback_cases)
