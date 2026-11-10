"""Motion calibration: transfer silhouette-specific motion, then repeat it.

Only NumPy and the standard library are used. No file access or saved state is
needed by generation, validation, or solving. Running this file prints bounded
aggregate self-test measurements as compact JSON; it saves no puzzles or traces.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np


HOST_SEEDS = (0,939636971,745547033,15657191,384646347,738919347,247061969,
              792495739,475230980,230496379,706694404,122986571,793866275,
              214558310,736296438,220262122,831728849,769646798,626273057,
              457490269,377652823,374462605,963215719,614637049,96262154,
              928470363,708830460,462005472,29700133,705395997,221328200,
              578496274,942539014,573991071,92918329,412862557,215079678,
              494497105,934377548,245874379,463250037,493832457)

# A five-cell shaft, with two sloping head cells immediately behind its tip.
_ARROW = np.array([[0,0,0,1,0],[1,1,1,1,1],[0,0,0,1,0]], dtype=bool)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not 2 <= a <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(b) or not 1 <= b <= 4 or a <= b:
        raise ValueError("num_test must be in 1..4 and less than num_train")
    return int(a), int(b)


def _grid(x):
    g = np.asarray(x)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((g >= 0) & (g <= 9))
    return g


def _key(g):
    g = np.asarray(g)
    return g.shape, g.astype(np.uint8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, cells = deque([start]), []
        while queue:
            r, c = queue.popleft()
            cells.append((r,c))
            for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        result.append(cells)
    return result


def _rotated(mask, turns):
    """Coordinate implementation used for solving, independent of rendering."""
    n = len(mask)
    out = np.zeros(mask.shape, dtype=bool)
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        for _ in range(int(turns) % 4):
            r, c = c, n-1-r
        out[r,c] = True
    return out


def _read_scene(input_grid):
    """Find three frame components and the directed calibration arrow."""
    g = _grid(input_grid)
    bg = int(g[0,0])
    assert np.all(g[0] == bg) and np.all(g[-1] == bg)
    assert np.all(g[:,0] == bg) and np.all(g[:,-1] == bg)
    scenes = []
    for fc in (int(c) for c in np.unique(g) if c != bg):
        frames, arrows, stray = [], [], False
        for cells in _components(g == fc):
            coords = np.array(cells)
            r, c = map(int, coords.min(axis=0))
            h, w = map(int, coords.max(axis=0) - coords.min(axis=0) + 1)
            crop = g[r:r+h,c:c+w] == fc
            border = np.ones((h,w), dtype=bool)
            border[1:-1,1:-1] = False
            if h == w and 7 <= h <= 10 and np.array_equal(crop, border):
                frames.append((r,c,h))
            else:
                matches = [k for k in range(4)
                           if np.array_equal(crop, np.rot90(_ARROW,k))]
                if len(matches) == 1:
                    arrows.append((r,c,h,w,matches[0]))
                else:
                    stray = True
        if stray or len(frames) != 3 or len(arrows) != 1:
            continue
        if len({b[2] for b in frames}) != 1:
            continue
        r,c,h,w,k = arrows[0]
        direction = ((0,1),(-1,0),(0,-1),(1,0))[k]
        axis = 1 if direction[1] else 0
        cross = 1-axis
        arrow_center = (r+(h-1)/2,c+(w-1)/2)
        centers = [(y+(m-1)/2,x+(m-1)/2) for y,x,m in frames]
        aligned = [i for i in range(3) if abs(centers[i][cross]-arrow_center[cross]) <= .5]
        if len(aligned) != 2:
            continue
        aligned.sort(key=lambda i: centers[i][axis] * direction[axis])
        i,j = aligned
        if not (centers[i][axis]*direction[axis] < arrow_center[axis]*direction[axis]
                < centers[j][axis]*direction[axis]):
            continue
        t = next(a for a in range(3) if a not in aligned)
        crops = [g[y+1:y+m-1,x+1:x+m-1].copy() for y,x,m in frames]
        before, after, target = crops[i], crops[j], crops[t]
        rp = set(map(int,np.unique(before))) - {bg}
        tp = set(map(int,np.unique(target))) - {bg}
        assert rp == set(map(int,np.unique(after))) - {bg}
        assert 2 <= len(rp) <= 4 and len(tp) == len(rp) and not rp & tp
        assert fc not in rp | tp
        outside = np.ones(g.shape, bool)
        for y,x,m in frames:
            outside[y:y+m,x:x+m] = False
        outside[g == fc] = False
        assert set(map(int,np.unique(g[outside]))) <= tp | {bg}
        markers = {}
        coords_all = []
        for color in sorted(tp):
            cells = np.argwhere(outside & (g == color))
            if not len(cells):
                continue
            ys,xs = cells.T
            horizontal = np.ptp(ys) == 0 and np.ptp(xs)+1 == len(cells)
            vertical = np.ptp(xs) == 0 and np.ptp(ys)+1 == len(cells)
            assert (horizontal or vertical) and 1 <= len(cells) <= 9
            for y,x in cells:
                # Every counter has a full background gap from every panel.
                assert all(not (a-1 <= y <= a+m and b-1 <= x <= b+m)
                           for a,b,m in frames)
            markers[color] = dict(count=len(cells), cells=cells,
                                  orientation='s' if len(cells)==1 else 'h' if horizontal else 'v')
            coords_all.extend((color,int(y),int(x)) for y,x in cells)
        assert markers
        for index,(color,y,x) in enumerate(coords_all):
            assert all(color == other or max(abs(y-v),abs(x-u)) > 1
                       for other,v,u in coords_all[index+1:])
        scenes.append(dict(background=bg, frame=fc, before=before, after=after,
                           target=target, markers=markers, reference_colors=sorted(rp),
                           target_colors=sorted(tp), frames=frames, arrow=k,
                           before_box=frames[i], after_box=frames[j], target_box=frames[t]))
    assert len(scenes) == 1, "expected exactly one directed three-panel scene"
    return scenes[0]


def _relations(scene):
    """Match silhouettes under rotation, then read each one's observed tick."""
    bg = scene['background']
    result, used = {}, set()
    for tc in scene['target_colors']:
        target_mask = scene['target'] == tc
        matches = [rc for rc in scene['reference_colors']
                   if any(np.array_equal(np.rot90(scene['before']==rc,k),target_mask)
                          for k in range(4))]
        assert len(matches) == 1, "silhouette identity is ambiguous"
        rc = matches[0]
        assert rc not in used
        used.add(rc)
        speeds = tuple(k for k in range(4)
                       if np.array_equal(np.rot90(scene['before']==rc,-k),scene['after']==rc))
        assert speeds, "the demonstrated layer must be a pure centered rotation"
        result[tc] = (rc,speeds)
    assert len(used) == len(scene['reference_colors'])
    assert bg not in result
    return result


def _predict(scene, counts=None, unit_clockwise=False, once=False):
    relations = _relations(scene)
    out = np.full_like(scene['target'], scene['background'])
    occupied = np.zeros(out.shape, dtype=bool)
    for tc,(rc,speeds) in relations.items():
        count = scene['markers'].get(tc,{}).get('count',0)
        if counts is not None:
            count = counts.get(tc,0)
        if once:
            count = 1
        options = (1,) if unit_clockwise else speeds
        rotated = [_rotated(scene['target']==tc,count*v) for v in options]
        assert all(np.array_equal(rotated[0],m) for m in rotated[1:]), "ambiguous motion outcome"
        mask = rotated[0]
        assert not np.any(occupied & mask), "layer collision has no overwrite convention"
        out[mask] = tc
        occupied |= mask
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, rate, identity and count from the input alone."""
    return _predict(_read_scene(input_grid))


def _orbits(n):
    unused = {(r,c) for r in range(n) for c in range(n)}
    orbits = []
    while unused:
        r,c = min(unused)
        orbit = []
        for _ in range(4):
            if (r,c) not in orbit:
                orbit.append((r,c))
            r,c = c,n-1-r
        unused.difference_update(orbit)
        if len(orbit) == 4:
            orbits.append(orbit)
    return orbits


def _shapes(rng,n,layers,witness):
    """Disjoint orbit ownership guarantees separation for every possible tick."""
    orbits = _orbits(n)
    order = rng.permutation(len(orbits)).tolist()
    q = min(4,len(orbits)//layers)
    owners = [order[i*q:(i+1)*q] for i in range(layers)]
    # First two layers have equal areas and different silhouettes. Area or
    # numerical color ordering cannot replace geometric correspondence.
    wanted = int(rng.integers(q,3*q))
    result = []
    for i in range(layers):
        mask = np.zeros((n,n),bool)
        mode = 'a' if i < 2 or witness else str(rng.choice(['a','h','q']))
        ids = owners[i]
        phase = int(rng.integers(4))
        if i < 2:
            occupancy = [1] + [1]*(q-1)
            for _ in range(wanted-q):
                available = [j for j in range(1,q) if occupancy[j]<4]
                occupancy[int(rng.choice(available))] += 1
        else:
            occupancy = [int(rng.integers(1,5)) for _ in ids]
            if mode == 'a':
                occupancy[0] = 1
        for j,idx in enumerate(ids):
            orbit = orbits[idx]
            start = (phase+int(rng.integers(2)))%4
            if mode == 'h':
                offsets = (0,2)
            elif mode == 'q':
                offsets = (0,1,2,3)
            else:
                offsets = tuple(range(occupancy[j]))
            for a in offsets:
                mask[orbit[(start+a)%4]] = True
        result.append(mask)
    return result


def _mix(masks,colors,bg,turns):
    """Construction uses NumPy rotations; solve uses coordinate arithmetic."""
    out = np.full(masks[0].shape,bg,dtype=np.int8)
    for mask,color,k in zip(masks,colors,turns):
        moved = np.rot90(mask,-int(k))
        assert np.all(out[moved] == bg)
        out[moved] = color
    return out


def _put_panel(g,interior,r,c,frame):
    n = len(interior)
    g[r:r+n+2,c:c+n+2] = frame
    g[r+1:r+n+1,c+1:c+n+1] = interior


def _put_bars(rng,g,region,colors,counts,bg,forced):
    """Bounded random packing, then a guaranteed horizontal-row fallback."""
    y,x,h,w = region
    assert h >= 10 and w >= 11
    for attempt in range(16):
        patch = np.full((h,w),bg,dtype=np.int8)
        blocked = np.zeros((h,w),bool)
        success = True
        for i in sorted(range(len(colors)),key=lambda j:-counts[j]):
            length = counts[i]
            if not length:
                continue
            orientations = [forced[i]] if i in forced else [0,1]
            choices = []
            for axis in orientations:
                bh,bw = (1,length) if axis==0 else (length,1)
                for r in range(h-bh+1):
                    for c in range(w-bw+1):
                        if not blocked[r:r+bh,c:c+bw].any():
                            choices.append((r,c,bh,bw))
            if not choices:
                success = False
                break
            r,c,bh,bw = choices[int(rng.integers(len(choices)))]
            patch[r:r+bh,c:c+bw] = colors[i]
            blocked[max(0,r-1):min(h,r+bh+1),max(0,c-1):min(w,c+bw+1)] = True
        if success:
            g[y:y+h,x:x+w] = patch
            return
    # Reserved disjoint lanes honor the teaching orientations too. Three
    # horizontal rows fit at the left; one vertical lane fits at the right.
    patch = np.full((h,w),bg,dtype=np.int8)
    vertical = [i for i in range(len(colors)) if counts[i] and forced.get(i)==1]
    assert len(vertical) <= 1
    for i in vertical:
        patch[:counts[i],w-1] = colors[i]
    row = 0
    for i in range(len(colors)):
        if counts[i] and i not in vertical:
            patch[row,:counts[i]] = colors[i]
            row += 2
    g[y:y+h,x:x+w] = patch


def _make_pair(rng,split,index):
    witness = split == 'train' and index < 2
    layers = 3 if witness else int(rng.integers(2,5))
    n = int(rng.integers(5 if layers<4 else 6,9))
    masks = _shapes(rng,n,layers,witness)
    palette = rng.permutation(10).tolist()
    bg,frame = palette[:2]
    ref_colors = palette[2:2+layers]
    target_colors = palette[2+layers:2+2*layers]
    rng.shuffle(target_colors)
    speeds = rng.integers(0,4,layers).tolist()
    counts = rng.integers(0,5 if split=='train' else 10,layers).tolist()
    if witness and index==0:
        speeds,counts = [2,1,1],[1,2,4]
    elif witness:
        speeds,counts = [1,3,2],[3,3,0]
    elif split=='test':
        # Extrapolate on an asymmetric, moving layer, so the long bar has an
        # observable role (length eight deliberately returns to its start).
        long_layer = int(rng.integers(2))
        counts[long_layer] = int(rng.integers(5,10))
        speeds[long_layer] = int(rng.choice([1,3]))
        # Retain random combinations whenever both concepts affect the answer.
        # Otherwise one period-two witness breaks both simpler alternatives.
        no_a = any(all(np.array_equal(np.rot90(mask,-v*k),np.rot90(mask,-sign*k))
                       for mask,v,k in zip(masks,speeds,counts)) for sign in (1,3))
        no_b = all(np.array_equal(np.rot90(mask,-v*k),np.rot90(mask,-v))
                   for mask,v,k in zip(masks,speeds,counts))
        if no_a or no_b:
            other = 1-long_layer
            speeds[other],counts[other] = 2,2
    if not any(counts):
        counts[0] = 1
    phases = rng.integers(0,4,layers).tolist()
    # Ensure target starts at a genuinely different relative arrangement, so
    # copying/recoloring the reference destination is insufficient.
    phases[:2] = [int(rng.integers(1,4)),int(rng.integers(1,4))]
    before = _mix(masks,ref_colors,bg,[0]*layers)
    after = _mix(masks,ref_colors,bg,speeds)
    target = _mix(masks,target_colors,bg,phases)
    output = _mix(masks,target_colors,bg,[p+v*k for p,v,k in zip(phases,speeds,counts)])
    m = n+2
    h,w = m+14,2*m+9
    g = np.full((h,w),bg,dtype=np.int8)
    _put_panel(g,before,1,1,frame)
    _put_panel(g,after,1,m+8,frame)
    ar,ac = 1+(m-3)//2,m+2
    # Even-width frame has a half-integer center; shift panels and arrow onto
    # a common integer center by using the center row just above it. The parser
    # accepts this half-cell alignment explicitly below.
    g[ar:ar+3,ac:ac+5][_ARROW] = frame
    target_left = bool(rng.integers(2))
    tc = 1 if target_left else w-m-1
    tr = m+3+int(rng.integers(11-m))
    _put_panel(g,target,tr,tc,frame)
    region = (m+3,m+3 if target_left else 1,10,w-m-4)
    forced = {0:0,1:1} if witness and index==1 else {}
    _put_bars(rng,g,region,target_colors,counts,bg,forced)
    # Extra padding varies canvas dimensions without changing content roles.
    pads = ((int(rng.integers(0,31-h)),0),(int(rng.integers(0,31-w)),0))
    g = np.pad(g,pads,constant_values=bg)
    if rng.integers(2):
        g,output = np.fliplr(g).copy(),np.fliplr(output).copy()
    k = int(rng.integers(4))
    return {'input':np.rot90(g,k).copy(),'output':np.rot90(output,k).copy()}


def generate_puzzle(seed: int,num_train: int=4,num_test: int=1) -> dict:
    a,b = _counts(num_train,num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',a),('test',b)):
        for index in range(count):
            for _ in range(8):
                pair = _make_pair(rng,split,index)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    return puzzle


def _features(pair):
    s = _read_scene(pair['input'])
    rel = _relations(s)
    symmetries = []
    for c in s['reference_colors']:
        mask = s['before']==c
        symmetries.append('q' if np.array_equal(mask,np.rot90(mask)) else
                          'h' if np.array_equal(mask,np.rot90(mask,2)) else 'a')
    return dict(size=len(s['target']),layers=len(rel),background=s['background'],arrow=s['arrow'],
                counts=sorted(m['count'] for m in s['markers'].values()),
                rates=sorted(v for _,vs in rel.values() for v in vs),symmetries=symmetries,
                unmarked=len(rel)-len(s['markers']),
                orientations=sorted({m['orientation'] for m in s['markers'].values()}),
                ignore_A=not np.array_equal(_predict(s,unit_clockwise=True),pair['output']),
                ignore_B=not np.array_equal(_predict(s,once=True),pair['output']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,evidence = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            s = _read_scene(inp)
            assert out.shape == s['target'].shape
            assert np.array_equal(_predict(s),out)
            # Stronger than checking the chosen rate: all possible rotations
            # of different layers must occupy disjoint territories.
            occupied = np.zeros(out.shape,bool)
            for color in s['target_colors']:
                mask = s['target']==color
                closure = np.logical_or.reduce([np.rot90(mask,k) for k in range(4)])
                assert not np.any(occupied & closure)
                occupied |= closure
                assert np.count_nonzero(mask)==np.count_nonzero(out==color)
            assert len(set(np.count_nonzero(s['target']==c) for c in s['target_colors'])) < len(s['target_colors'])
            if split=='train':
                evidence.append(_features(pair))
            else:
                assert not np.array_equal(_predict(s,unit_clockwise=True),out)
                assert not np.array_equal(_predict(s,once=True),out)
    assert all(any(f[k] for f in evidence) for k in ('ignore_A','ignore_B'))
    assert {1,2,3,4} <= {v for f in evidence for v in f['counts']}
    assert any(f['unmarked'] for f in evidence)
    assert any({'h','v'} <= set(f['orientations']) for f in evidence)


def diagnostic_twins(pair):
    """Full inputs differing only in calibration, or only in one counter."""
    s = _read_scene(pair['input'])
    relations = _relations(s)
    original = solve(pair['input'])
    a = pair['input'].copy()
    tc = next(c for c in s['target_colors']
              if s['markers'].get(c,{}).get('count',0)%2==1 and len(relations[c][1])==1)
    rc,speeds = relations[tc]
    after = s['after'].copy()
    after[after==rc] = s['background']
    after[_rotated(s['before']==rc,speeds[0]+1)] = rc
    r,c,m = s['after_box']
    a[r+1:r+m-1,c+1:c+m-1] = after
    b = pair['input'].copy()
    tc = next(c for c,(_,speeds) in relations.items()
              if len(speeds)==1 and speeds[0]%2 and s['markers'].get(c,{}).get('count',0)>1)
    r,c = s['markers'][tc]['cells'][0]
    b[r,c] = s['background']
    outputs = [solve(a),solve(b)]
    assert all(not np.array_equal(original,o) for o in outputs)
    return [{'input':a,'output':outputs[0]},{'input':b,'output':outputs[1]}]


def self_test():
    records,shapes,patterns,palettes = [],set(),set(),set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ('train','test'):
            for pair in p[split]:
                inp = pair['input'].copy()
                assert np.array_equal(solve(inp),pair['output'])
                assert np.array_equal(inp,pair['input'])
                f = _features(pair)
                f['split'] = split
                records.append(f)
                shapes.add(inp.shape)
                s = _read_scene(inp)
                patterns.add(_key(s['target']!=s['background']))
                palettes.add(tuple(map(int,np.unique(inp))))
        assert _puzzle_key(p)==_puzzle_key(generate_puzzle(seed))
        diagnostic_twins(p['train'][0])
    assert {f['size'] for f in records} == {5,6,7,8}
    assert {f['layers'] for f in records} == {2,3,4}
    assert {f['background'] for f in records} == set(range(10))
    assert {f['arrow'] for f in records} == set(range(4))
    assert {v for f in records for v in f['counts']} == set(range(1,10))
    assert {v for f in records for v in f['rates']} == set(range(4))
    assert {v for f in records for v in f['symmetries']} == {'a','h','q'}
    assert len(shapes)>=20 and len(patterns)>=150 and len(palettes)>=15
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            validate_generated_puzzle(generate_puzzle(71,a,b))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(17,2,1)
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42,pairs=len(records),input_shapes=len(shapes),target_occupancy_masks=len(patterns),
                input_palettes=len(palettes),ignore_A_failures=sum(f['ignore_A'] for f in records),
                ignore_B_failures=sum(f['ignore_B'] for f in records),diagnostic_twin_pairs=84,
                sizes=sorted({f['size'] for f in records}),layers=sorted({f['layers'] for f in records}),
                counts=sorted({v for f in records for v in f['counts']}),
                symmetry_types=sorted({v for f in records for v in f['symmetries']}))


def _puzzle_key(p):
    return tuple((_key(pair['input']),_key(pair['output']))
                 for split in ('train','test') for pair in p[split])


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(",",":")))
