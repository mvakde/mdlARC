"""Periodic vessels: repair a texture and eject only its surplus accent tiles.

The input reader is independent of the scene constructor: it reads monochromatic
shell components, taper geometry and minimum-distance periodic consensus. No
example, seed, generation record, or filesystem access enters solve().
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid colors must be integers 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in remaining:
                        remaining.remove(p)
                        component.add(p)
                        todo.append(p)
        result.append(component)
    return result


def _recover(texture, background, accent):
    """Unique closest short periodic explanation; no hidden true period.

    Every candidate has at least three observations of each phase-coordinate,
    strict majorities, and <= floor(area/6) changes. Equally good candidates
    must describe the identical repaired array. The shortest redundant period
    is returned for measurements only.
    """
    h, length = texture.shape
    candidates = []
    for period in range(1, length // 3 + 1):
        template = np.empty((h, period), dtype=np.int8)
        valid = True
        for phase in range(period):
            group = texture[:, phase::period]
            n = np.count_nonzero(group == accent, axis=1)
            if np.any(2 * n == group.shape[1]):
                valid = False
                break
            template[:, phase] = np.where(2 * n > group.shape[1], accent, background)
        if not valid:
            continue
        clean = template[:, np.arange(length) % period]
        errors = int(np.count_nonzero(clean != texture))
        if errors <= texture.size // 6 and len(np.unique(clean)) == 2:
            candidates.append((errors, period, clean))
    if not candidates:
        raise ValueError("texture has no sparse periodic explanation")
    best_error = min(c[0] for c in candidates)
    best = [c for c in candidates if c[0] == best_error]
    clean = best[0][2]
    if any(not np.array_equal(clean, c[2]) for c in best):
        raise ValueError("texture has competing closest repairs")
    return clean, min(c[1] for c in best)


def _shell_geometry(component, rotation, shape):
    """Read a closed shell after rotating the entire input to point right.

    A full flat rear, parallel flanks, inward monotone taper, and optional tip
    stem delimit a padded rectangular texture. Slopes may pause independently;
    shell recognition does not enumerate the constructor's radius parameters.
    """
    coords = np.zeros(shape, dtype=np.int8)
    for p in component:
        coords[p] = 1
    points = np.argwhere(np.rot90(coords, rotation))
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
    if h < 5 or w < 10:
        return None
    local = {(int(r-r0), int(c-c0)) for r, c in points}
    columns = [sorted(r for r, x in local if x == c) for c in range(w)]
    if columns[0] != list(range(h)) or len(columns[-1]) != 1:
        return None
    root = 0
    for c in range(1, w):
        if columns[c] == [0, h-1]:
            root = c
        else:
            break
    length = root - 3
    if length < 6 or root >= w - 1:
        return None
    low, high = 0, h-1
    single = False
    for c in range(root+1, w):
        row = columns[c]
        if not row or len(row) > 2:
            return None
        lo, hi = row[0], row[-1]
        if not (0 <= lo-low <= 1 and 0 <= high-hi <= 1):
            return None
        if (lo, hi) == (low, high) and not single:
            return None
        if single and (lo != hi or lo != low):
            return None
        if len(row) == 1:
            single = True
        low, high = lo, hi
    if not single or not 1 < low < h-2:
        return None
    expected = {(r, 0) for r in range(h)}
    expected |= {(r, c) for r in (0, h-1) for c in range(1, root+1)}
    expected |= {(r, c) for c in range(root+1, w) for r in columns[c]}
    if local != expected:
        return None
    return int(r0), int(c0), h, w, root


def _read(input_grid):
    grid = _grid(input_grid)
    # All four corners are guaranteed exterior. This avoids an occupancy-based
    # color guess and allows even a heavily patterned small vessel.
    background = int(grid[0, 0])
    if any(int(grid[p]) != background for p in ((0,-1),(-1,0),(-1,-1))):
        raise ValueError("corners must be exterior background")
    found = []
    ids = np.arange(grid.size).reshape(grid.shape)
    for color in (int(c) for c in np.unique(grid) if c != background):
        for component in _components(grid == color):
            if len(component) < 16:
                continue
            for rotation in range(4):
                geometry = _shell_geometry(component, rotation, grid.shape)
                if geometry is None:
                    continue
                r0, c0, h, w, root = geometry
                turned = np.rot90(grid, rotation)
                turned_ids = np.rot90(ids, rotation)
                texture = turned[r0+2:r0+h-2, c0+2:c0+root-1]
                values = set(int(c) for c in np.unique(texture)) - {background}
                if len(values) != 1 or color in values:
                    continue
                accent = values.pop()
                # Everything inside this shell but outside the texture is
                # background, including the taper and one-cell padding.
                shell_ids = {int(ids[p]) for p in component}
                texture_ids = turned_ids[r0+2:r0+h-2, c0+2:c0+root-1].copy()
                rectangle_ids = set(int(i) for i in turned_ids[r0:r0+h,c0:c0+w].flat)
                empty_ids = rectangle_ids - shell_ids - set(int(i) for i in texture_ids.flat)
                if any(grid.flat[i] != background for i in empty_ids):
                    continue
                try:
                    clean, period = _recover(texture, background, accent)
                except ValueError:
                    continue
                surplus = (texture == accent) & (clean == background)
                missing = (texture == background) & (clean == accent)
                q = int(surplus.sum())
                tip_row = r0 + int(np.flatnonzero(turned[r0:r0+h,c0+w-1] == color)[0])
                ray_ids = [int(i) for i in turned_ids[tip_row, c0+w:]]
                # Clear the entire potential ray corridor, not just q cells.
                if any(grid.flat[i] != background for i in ray_ids):
                    continue
                found.append(dict(shell=shell_ids, box=rectangle_ids, texture_ids=texture_ids,
                                  clean=clean, observed=texture.copy(), accent=accent,
                                  frame_color=color,
                                  shell_profile=tuple((r,c) for r in range(h) for c in range(w)
                                                      if int(turned_ids[r0+r,c0+c]) in shell_ids),
                                  background=background, surplus=surplus, missing=missing,
                                  quantity=q, ray=ray_ids, rotation=rotation,
                                  period=period, length=texture.shape[1], cross=texture.shape[0],
                                  shell_shape=(h,w), stem=_stem_length(turned,r0,c0,h,w,color)))
    if not found:
        raise ValueError("no unambiguous patterned vessel")
    if len({frozenset(o['shell']) for o in found}) != len(found):
        raise ValueError("ambiguous shell direction")
    occupied, paths = set(), set()
    for obj in found:
        if occupied & obj['box'] or paths & set(obj['ray']):
            raise ValueError("overlapping vessels or ray corridors")
        occupied.update(obj['box'])
        paths.update(obj['ray'])
    if occupied & paths:
        raise ValueError("ray corridor enters a vessel")
    nonbackground = set(int(i) for i in np.flatnonzero(grid.ravel() != background))
    # Isolated inert exterior marks are allowed, but never larger shapes.
    exterior = nonbackground - occupied
    for index in exterior:
        r, c = divmod(index, grid.shape[1])
        if any((r+dr)*grid.shape[1]+c+dc in nonbackground-{index}
               for dr in (-1,0,1) for dc in (-1,0,1)
               if 0 <= r+dr < grid.shape[0] and 0 <= c+dc < grid.shape[1]):
            raise ValueError("unowned non-isolated colored cells")
    return grid, found


def _stem_length(turned,r0,c0,h,w,color):
    n = 0
    for c in range(w-1,-1,-1):
        if np.count_nonzero(turned[r0:r0+h,c0+c] == color) != 1:
            break
        n += 1
    return n - 1


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore each texture; serialize surplus tiles immediately past its tip."""
    grid, objects = _read(input_grid)
    result = grid.copy()
    for obj in objects:
        result.flat[obj['texture_ids'].ravel()] = obj['clean'].ravel()
        result.flat[obj['ray'][:obj['quantity']]] = obj['accent']
    return result


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train",num_train,2,10),("num_test",num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train),int(num_test)


def _shape(length, a, b, stem, rng):
    """Construct the shell, separately from the cross-section input reader."""
    h, root = a+b+1, length+3
    shell = {(y,0) for y in range(h)}
    shell |= {(y,x) for y in (0,h-1) for x in range(1,root+1)}
    # Walk two independently delayed slopes, always making progress.
    lo, hi, x = 0, h-1, root
    while lo < a or hi > a:
        choices = []
        if lo < a:
            choices.append((1,0))
        if hi > a:
            choices.append((0,1))
        if lo < a and hi > a:
            choices.extend([(1,1)]*5)
        dl, dh = choices[int(rng.integers(len(choices)))]
        lo,hi,x = lo+dl,hi-dh,x+1
        shell.update({(lo,x),(hi,x)})
    for _ in range(stem):
        x += 1
        shell.add((a,x))
    return h,x+1,shell,(a,x)


def _motif(rng,cross,length,period,q,holes):
    """A finite randomized search then a structured, enumerated fallback."""
    if q+holes > cross*period or q+holes > cross*length//6:
        raise ValueError("defect budget does not fit")
    # A template is a binary stencil. Each phase-coordinate is corrupted at
    # most once, but recovery is not given or restricted to that fact.
    for attempt in range(768):
        if attempt < 512:
            unit = rng.integers(0,2,size=(cross,period),dtype=np.int8)
        else:
            # Enumerate row words with varied cyclic phases.
            word = 1 + (attempt-512) % ((1 << period)-2)
            row = np.array([(word >> k)&1 for k in range(period)],dtype=np.int8)
            unit = np.stack([np.roll(row,i) for i in range(cross)])
        zeros,ones = np.argwhere(unit==0),np.argwhere(unit==1)
        if len(zeros)<q or len(ones)<holes:
            continue
        clean = unit[:,np.arange(length)%period]
        observed = clean.copy()
        for positions,n,value in ((zeros,q,1),(ones,holes,0)):
            for group in rng.choice(len(positions),size=n,replace=False):
                row,phase = (int(v) for v in positions[int(group)])
                copies = np.arange(phase,length,period)
                col = int(rng.choice(copies))
                observed[row,col] = value
        try:
            recovered,p = _recover(observed,0,1)
        except ValueError:
            continue
        if p==period and np.array_equal(recovered,clean):
            return observed,clean
    raise RuntimeError("bounded periodic motif search exhausted")


def _make_vessel(rng,background,frame,accent,*,anchor=None,q=None,holes=None,test=False,arrangement=None):
    if anchor is not None:
        length,a,b,stem,period,clean,shell_shape = anchor
        h,w,shell,tip = shell_shape
        if q is None or holes is None:
            raise ValueError("anchor needs explicit defects")
        for _ in range(256):
            noisy = clean.copy()
            zeros,ones = np.argwhere(clean[:, :period]==0),np.argwhere(clean[:, :period]==1)
            if len(zeros)<q or len(ones)<holes:
                raise ValueError("anchor unit lacks defect sites")
            if arrangement is not None:
                sites=[tuple(map(int,p)) for p in np.argwhere(clean==0)]
                pairs=[(u,v) for i,u in enumerate(sites) for v in sites[i+1:]
                       if (u[0],u[1]%period)!=(v[0],v[1]%period)
                       and ((max(abs(u[0]-v[0]),abs(u[1]-v[1]))==1)
                            if arrangement=='compact' else
                            (max(abs(u[0]-v[0]),abs(u[1]-v[1]))>=3))]
                if q!=2 or not pairs:
                    raise ValueError("requested surplus arrangement unavailable")
                for cell in pairs[int(rng.integers(len(pairs)))]:
                    noisy[cell]=1
            for groups,n,value in ((zeros,0 if arrangement else q,1),(ones,holes,0)):
                for k in rng.choice(len(groups),n,replace=False):
                    row,phase = map(int,groups[int(k)])
                    noisy[row,int(rng.choice(np.arange(phase,length,period)))] = value
            try:
                restored,p = _recover(noisy,0,1)
            except ValueError:
                continue
            if np.array_equal(restored,clean) and p==period:
                break
        else:
            raise RuntimeError("anchor defect placement exhausted")
    else:
        # Shorter and longer textures occur in both splits; ordinary tests may
        # have a larger period. The first test's deficit shift is set below.
        for _ in range(128):
            a,b = int(rng.integers(2,5)),int(rng.integers(2,5))
            cross = a+b-3
            period = int(rng.integers(2,5 if not test else 6))
            length = int(rng.integers(max(8,4*period),min(21,4*period+4)))
            maxq = min(5,cross*period-1,cross*length//6)
            if maxq<1:
                continue
            q = int(rng.integers(1,maxq+1))
            maxholes = min(4,cross*period-q,cross*length//6-q)
            holes = int(rng.integers(0,maxholes+1))
            stem = int(rng.integers(0,3))
            shape = _shape(length,a,b,stem,rng)
            if shape[1]+q+1 > 29:
                continue
            try:
                noisy,clean = _motif(rng,cross,length,period,q,holes)
            except (RuntimeError,ValueError):
                continue
            h,w,shell,tip = shape
            break
        else:
            raise RuntimeError("vessel construction exhausted")
    inp = np.full((h,w),background,dtype=np.int8)
    out = inp.copy()
    for p in shell:
        inp[p]=out[p]=frame
    inp[2:h-2,2:length+2] = np.where(noisy,accent,background)
    out[2:h-2,2:length+2] = np.where(clean,accent,background)
    return dict(input=inp,output=out,tip=tip,q=q,holes=holes,period=period,
                length=length,a=a,b=b,stem=stem,clean=clean,shell_shape=(h,w,shell,tip),accent=accent)


def _orient(vessel,rotation):
    # Construction coordinate transform via an ID image rather than the reader.
    marker = np.zeros_like(vessel['input'])
    marker[vessel['tip']] = 1
    return dict(vessel,input=np.rot90(vessel['input'],rotation).copy(),
                output=np.rot90(vessel['output'],rotation).copy(),
                tip=tuple(int(v) for v in np.argwhere(np.rot90(marker,rotation))[0]),
                direction=((0,1),(-1,0),(0,-1),(1,0))[rotation])


def _scene(rng,vessels,background,clearances,mark=False):
    """Random free placement, trying a roomy canvas after smaller layouts.

    Every specified vessel is mandatory. Packing retries are bounded; multiple
    objects are optional at the caller, never silently omitted here.
    """
    min_h=max(v['input'].shape[0] for v in vessels)
    min_w=max(v['input'].shape[1] for v in vessels)
    for attempt in range(32):
        height=int(rng.integers(min_h+2,31)) if min_h+2<31 else 30
        width=int(rng.integers(min_w+2,31)) if min_w+2<31 else 30
        if attempt>=12:
            height=width=30
        placed=[]
        boxes,guards,paths=set(),set(),set()
        for v,room in zip(vessels,clearances):
            h,w=v['input'].shape
            dr,dc=v['direction']
            possibilities=[]
            for r in range(1,height-h):
                for c in range(1,width-w):
                    tr,tc=r+v['tip'][0],c+v['tip'][1]
                    available=tr if dr==-1 else height-1-tr if dr==1 else tc if dc==-1 else width-1-tc
                    if room is not None and available!=room:
                        continue
                    if room is None and available<=v['q']:
                        continue
                    possibilities.append((r,c,tr,tc,available))
            accepted=None
            for ix in rng.permutation(len(possibilities)):
                r,c,tr,tc,available=possibilities[int(ix)]
                box={(r+y,c+x) for y in range(h) for x in range(w)}
                path={(tr+dr*k,tc+dc*k) for k in range(1,available+1)}
                if box & (guards|paths) or path & (boxes|paths):
                    continue
                accepted=(v,r,c,tr,tc,available)
                boxes.update(box)
                guards.update((y+dy,x+dx) for y,x in box for dy in (-1,0,1) for dx in (-1,0,1))
                paths.update(path)
                break
            if accepted is None:
                break
            placed.append(accepted)
        if len(placed)!=len(vessels):
            continue
        inp=np.full((height,width),background,dtype=np.int8)
        out=inp.copy()
        for v,r,c,tr,tc,available in placed:
            h,w=v['input'].shape
            inp[r:r+h,c:c+w]=v['input']
            out[r:r+h,c:c+w]=v['output']
            dr,dc=v['direction']
            for k in range(1,min(available,v['q'])+1):
                out[tr+dr*k,tc+dc*k]=v['accent']
        if mark:
            possible=[(r,c) for r in range(1,height-1) for c in range(1,width-1)
                      if (r,c) not in guards|paths]
            if possible:
                point=possible[int(rng.integers(len(possible)))]
                inp[point]=out[point]=int(rng.choice([c for c in range(10) if c!=background]))
        # Construction labels use the sampled clean pattern and explicit q.
        # The independently read input must predict those exact labels.
        if np.array_equal(solve(inp),out):
            return dict(input=inp,output=out)
    raise RuntimeError("bounded scene packing exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test=_counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError("seed must be a nonnegative integer")
    rng=np.random.default_rng(int(seed))
    bg,frame,accent=(int(v) for v in rng.choice(10,3,replace=False))
    rotation=int(rng.integers(4))
    # Matched witnesses: equal raw accent totals, different surplus quantities.
    # Both miss tiles; an error count and net-count correction also fail.
    for _ in range(128):
        length=16
        a,b=[(2,4),(3,3),(4,2)][int(rng.integers(3))]
        period=int(rng.choice([3,4]))
        shape=_shape(length,a,b,0,rng)
        if shape[1]+5>29:
            continue
        try:
            _,clean=_motif(rng,3,length,period,4,3)
            anchor=(length,a,b,0,period,clean,shape)
            low=_make_vessel(rng,bg,frame,accent,anchor=anchor,q=2,holes=1,arrangement='spread')
            high=_make_vessel(rng,bg,frame,accent,anchor=anchor,q=4,holes=3)
            repeat=_make_vessel(rng,bg,frame,accent,anchor=anchor,q=2,holes=3,arrangement='compact')
            break
        except (ValueError,RuntimeError):
            continue
    else:
        raise RuntimeError("matched witnesses exhausted")
    # A fourth witness rules out copying the common clean teaching texture.
    # Its different period and cross-section are inferred locally in the same
    # scene; its new accent also makes ownership visibly object-specific.
    other_accent=int(rng.choice([c for c in range(10) if c not in (bg,frame,accent)]))
    other_a,other_b=((2,3) if rng.integers(2) else (3,2))
    other_clean=np.zeros((2,2),dtype=np.int8)
    other_clean.flat[int(rng.integers(4))]=1
    other_clean=np.tile(other_clean,(1,6))
    other_stem=int(rng.integers(2))
    other_shape=_shape(12,other_a,other_b,other_stem,rng)
    other_anchor=(12,other_a,other_b,other_stem,2,other_clean,other_shape)
    other=_make_vessel(rng,bg,frame,other_accent,anchor=other_anchor,q=3,holes=0)
    train=[_scene(rng,[_orient(low,rotation),_orient(other,rotation)],bg,[None,None]),
           _scene(rng,[_orient(high,rotation),_orient(repeat,rotation)],bg,[None,None])]
    rng.shuffle(train)
    puzzle={'train':train,'test':[]}
    seen={_key(p['input']) for p in train}
    # Palette is pair-local after the matched witnesses. Test introduces more
    # severe holes relative to surplus, demonstrated already by the witnesses,
    # plus longer/larger-period or multi-vessel structures when they fit.
    for split,count,start in (('train',num_train,2),('test',num_test,0)):
        for index in range(start,count):
            for retry in range(24):
                background=int(rng.integers(10))
                palette=[c for c in range(10) if c!=background]
                count_objects=int(rng.choice([1,1,2,2,3])) if retry<8 else 1
                vessels=[]
                for j in range(count_objects):
                    f,p=(int(c) for c in rng.choice(palette,2,replace=False))
                    v=_make_vessel(rng,background,f,p,test=split=='test')
                    # Extend the negative-net training witness to quantity one:
                    # missing tiles still do not contribute to its export.
                    if split=='test' and index==0 and j==0:
                        # Use the established period/geometry but reverse the
                        # surplus/deficit relationship, with a fresh palette.
                        v=_make_vessel(rng,background,f,p,anchor=anchor,q=1,holes=3)
                    rot=int(rng.integers(4))
                    vessels.append(_orient(v,rot))
                rooms=[]
                for j,v in enumerate(vessels):
                    # First test always exposes its whole surplus ray.
                    mode=0 if split=='test' and index==0 and j==0 else int(rng.integers(4))
                    rooms.append(None if mode<2 else v['q'] if mode==2 else int(rng.integers(1,v['q']+1)))
                try:
                    pair=_scene(rng,vessels,background,rooms,mark=rng.random()<.3)
                except (RuntimeError,ValueError):
                    continue
                key=_key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct scene construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _describe_pair(pair):
    inp,objects=_read(pair['input'])
    out=_grid(pair['output'])
    assert inp.shape==out.shape
    assert np.array_equal(solve(inp),out),"output violates repair/selective export"
    assert not np.array_equal(inp,out)
    changed=set(int(i) for i in np.flatnonzero(inp.ravel()!=out.ravel()))
    allowed=set()
    for o in objects:
        assert o['quantity']>=1
        assert int(o['surplus'].sum()+o['missing'].sum())<=o['clean'].size//6
        assert o['length']//o['period']>=3
        assert o['period']>=2
        tex=set(int(i) for i in o['texture_ids'].flat)
        actual_ray=set(o['ray'][:o['quantity']])
        allowed.update(tex|actual_ray)
        assert all(inp.flat[i]==out.flat[i] for i in o['shell'])
        assert np.array_equal(out.ravel()[o['texture_ids']],o['clean'])
        assert all(out.flat[i]==o['accent'] for i in actual_ray)
        assert all(out.flat[i]==o['background'] for i in o['ray'][o['quantity']:])
    assert changed<=allowed,"changed padding, shell, or unrelated exterior cell"
    return objects


def _training_evidence(pairs):
    witnesses=[]
    for pair in pairs:
        for o in _describe_pair(pair):
            if len(o['ray'])>o['quantity']:
                witnesses.append(o)
    matched=False
    for a in witnesses:
        for b in witnesses:
            if a is b:
                continue
            same=(a['shell_profile']==b['shell_profile'] and a['rotation']==b['rotation']
                  and a['frame_color']==b['frame_color']
                  and a['accent']==b['accent'] and a['background']==b['background']
                  and np.array_equal(a['clean'],b['clean']))
            if same and a['quantity']!=b['quantity'] and np.count_nonzero(a['observed']==a['accent'])==np.count_nonzero(b['observed']==b['accent']):
                assert a['missing'].any() and b['missing'].any()
                assert a['quantity']!=int(a['surplus'].sum()+a['missing'].sum())
                matched=True
    assert matched,"training lacks matched equal-total, unequal-export witnesses"
    assert len({o['period'] for o in witnesses})>=2,"training permits one shared period"
    assert len({_key(o['clean']) for o in witnesses})>=2,"training permits copying one clean texture"
    assert any(o['missing'].sum()>o['quantity'] for o in witnesses),"training has no negative-net defect example"
    arrangements={}
    for o in witnesses:
        arrangements.setdefault(o['quantity'],set()).add(len(_components(o['surplus'])))
    assert any(1 in counts and len(counts)>1 for counts in arrangements.values()),"training does not distinguish surplus cells from surplus components"
    # Explicit parent ablations, with fresh output arrays: no A means applying
    # B to every observed accent; no B means periodic restoration alone.
    no_a_changed = no_b_changed = False
    for pair in pairs:
        inp,objects = _read(pair['input'])
        no_a,no_b = inp.copy(),inp.copy()
        for o in objects:
            observed_ids=o['texture_ids'][o['observed']==o['accent']]
            no_a.flat[observed_ids]=o['background']
            no_a.flat[o['ray'][:len(observed_ids)]]=o['accent']
            no_b.flat[o['texture_ids'].ravel()]=o['clean'].ravel()
        no_a_changed |= not np.array_equal(no_a,pair['output'])
        no_b_changed |= not np.array_equal(no_b,pair['output'])
    assert no_a_changed and no_b_changed,"a parent ablation leaves all training labels unchanged"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen=set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp=_grid(pair['input'])
            key=_key(inp)
            assert key not in seen,"duplicate input"
            seen.add(key)
            objects=_describe_pair(pair)
            assert 1<=len(objects)<=3
    _training_evidence(puzzle['train'])
    test_objects=[o for p in puzzle['test'] for o in _read(p['input'])[1]]
    assert any(len(o['ray'])>o['quantity'] and o['missing'].sum()>o['quantity'] for o in test_objects),"missing the demonstrated negative-net test shift"
    return True


def self_test():
    """Bounded: 42 streamed seeds, no disk writes or retained grid collection."""
    signatures=set()
    shapes=set()
    rotations=set()
    periods=set()
    crosses=set()
    lengths=set()
    quantities=set()
    object_counts=Counter()
    clipping=Counter()
    palettes=set()
    normalized=set()
    for seed in range(42):
        puzzle=generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest=hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp,objects=_read(pair['input'])
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                digest.update(np.asarray(pair['output'],dtype=np.int8).tobytes())
                shapes.add(inp.shape)
                object_counts[len(objects)]+=1
                palettes.add(tuple(int(c) for c in np.unique(inp)))
                roles={}
                normal=bytes(roles.setdefault(int(v),len(roles)) for v in inp.flat)
                normalized.add((inp.shape,hashlib.sha256(normal).digest()))
                for o in objects:
                    rotations.add(o['rotation'])
                    periods.add(o['period'])
                    crosses.add(o['cross'])
                    lengths.add(o['length'])
                    quantities.add(o['quantity'])
                    clipping['free' if len(o['ray'])>o['quantity'] else 'exact' if len(o['ray'])==o['quantity'] else 'clipped']+=1
                # Rotation and literal-color covariance also exercise the
                # input-only reader beyond generated orientations/palettes.
                if seed<3:
                    for rotation in range(4):
                        permutation=np.roll(np.arange(10),seed+2)
                        x=permutation[np.rot90(inp,rotation)]
                        y=permutation[np.rot90(pair['output'],rotation)]
                        assert np.array_equal(solve(x),y)
        signatures.add(digest.digest())
    assert len(signatures)==42
    assert rotations=={0,1,2,3}
    assert len(shapes)>=30 and len(normalized)>=180
    assert {2,3,4}<=periods and len(crosses)>=4 and len(lengths)>=8
    assert {1,2,3,4}<=quantities
    assert 1 in object_counts and 2 in object_counts
    assert set(clipping)=={'free','exact','clipped'}
    assert len(palettes)>=20
    # An explicitly constructed period-five member exercises this supported
    # but uncommon packing corner without pretending it occurred above.
    rng=np.random.default_rng(77)
    clean=np.tile(np.array([[0,0,1,0,1]],dtype=np.int8),(1,4))
    shape=_shape(20,2,2,0,rng)
    anchor=(20,2,2,0,5,clean,shape)
    vessel=_make_vessel(rng,0,8,4,anchor=anchor,q=1,holes=1)
    fixture=_scene(rng,[_orient(vessel,0)],0,[None])
    assert _read(fixture['input'])[1][0]['period']==5
    assert np.array_equal(solve(fixture['input']),fixture['output'])
    p=generate_puzzle(900,2,1)
    repeat=generate_puzzle(900,2,1)
    assert all(np.array_equal(a[k],b[k]) for s in p for a,b in zip(p[s],repeat[s]) for k in a)
    for a,b in ((2,1),(10,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(21,a,b))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return dict(seeds=42,distinct_puzzles=len(signatures),grid_shapes=len(shapes),
                normalized_structures=len(normalized),periods=sorted(periods),
                motif_crosses=sorted(crosses),motif_lengths=sorted(lengths),
                quantities=sorted(quantities),object_counts=dict(sorted(object_counts.items())),
                clearance=dict(clipping),palettes=len(palettes),dedicated_period_five=True)
