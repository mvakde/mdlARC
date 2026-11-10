"""Global equality-run contraction feeds rotational companionship packing.

Only Python's standard library and NumPy are used. No parent code or data is
loaded at runtime. Generation is deliberately narrower than the input-only
solver: teaching witnesses and collision-free layouts constrain sampling.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json

import numpy as np


HOST_SEEDS = (0,23553647,801867145,643474722,676660304,876188231,682229196,
              865085276,110623915,462656829,329663967,458084481,530545293,
              189101048,958574647,600630061,505896973,366600062,410303469,
              401276858,534969362,739970295,319191090,257888244,300594073,
              71636319,635994655,249656175,698117196,591637749,356774375,
              825603912,277088298,490734201,286233399,588105572,556616038,
              898215100,683013969,293594936,809152889,842153502)
DIR8 = tuple((r,c) for r in (-1,0,1) for c in (-1,0,1) if (r,c) != (0,0))
DIR4 = ((-1,0),(1,0),(0,-1),(0,1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _axis_runs(a, axis):
    vectors = a if axis == 0 else a.T
    starts = [0]
    for i in range(1, len(vectors)):
        if not np.array_equal(vectors[i], vectors[i-1]):
            starts.append(i)
    lengths = tuple(b-a for a,b in zip(starts, starts[1:]+[len(vectors)]))
    return tuple(starts), lengths


def _collapse(a):
    # Each equality is of an ENTIRE vector in the original grid. Row and column
    # equivalence are independent; either sequential order gives this result.
    rows, _ = _axis_runs(a, 0)
    cols, _ = _axis_runs(a, 1)
    return a[np.ix_(rows, cols)].copy()


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    i = int(np.argmax(counts))
    if counts[i] * 2 <= a.size:
        raise ValueError("family requires a strict-majority background")
    return int(colors[i])


def _trim(a):
    y,x = np.nonzero(a)
    if not len(y):
        raise ValueError("empty object")
    return np.asarray(a[y.min():y.max()+1, x.min():x.max()+1], dtype=bool).copy()


def _mask_key(mask):
    return mask.shape, np.asarray(mask, dtype=np.uint8).tobytes()


def _rotation_key(mask):
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _segment(a, background, directions=DIR8):
    seen = np.zeros(a.shape, dtype=bool)
    objects = []
    for y,x in np.ndindex(a.shape):
        color = int(a[y,x])
        if color == background or seen[y,x]:
            continue
        seen[y,x] = True
        stack, cells = [(y,x)], []
        while stack:
            r,c = stack.pop()
            cells.append((r,c))
            for dr,dc in directions:
                rr,cc = r+dr,c+dc
                if (0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]
                        and not seen[rr,cc] and a[rr,cc] == color):
                    seen[rr,cc] = True
                    stack.append((rr,cc))
        top = min(r for r,c in cells)
        left = min(c for r,c in cells)
        mask = np.zeros((max(r for r,c in cells)-top+1,
                         max(c for r,c in cells)-left+1), dtype=bool)
        for r,c in cells:
            mask[r-top,c-left] = True
        objects.append(dict(top=top, left=left, color=color, mask=mask,
                            key=_rotation_key(mask)))
    return objects


def _paint(canvas, obj, top, background):
    h,w = obj['mask'].shape
    if top < 0 or top+h > canvas.shape[0] or obj['left']+w > canvas.shape[1]:
        raise ValueError("object would leave the canvas")
    view = canvas[top:top+h, obj['left']:obj['left']+w]
    if np.any(view[obj['mask']] != background):
        raise ValueError("packing would overlap objects")
    view[obj['mask']] = obj['color']


def _pack(a, background, directions=DIR8, key_function=None, exactly_two=False):
    objects = _segment(a, background, directions)
    keys = [o['key'] if key_function is None else key_function(o) for o in objects]
    frequencies = Counter(keys)
    result = np.full(a.shape, background, dtype=np.int8)
    for obj,key in zip(objects, keys):
        up = frequencies[key] == 2 if exactly_two else frequencies[key] >= 2
        top = 0 if up else a.shape[0]-obj['mask'].shape[0]
        _paint(result, obj, top, background)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Contract full-vector runs once, then classify and translate objects.

    There is no final contraction after packing. Columns refer to the
    contracted coordinate system. Invalid ambiguous/colliding scenes raise.
    """
    reduced = _collapse(_grid(input_grid))
    return _pack(reduced, _background(reduced))


def _halo(cells):
    return {(r+dr,c+dc) for r,c in cells for dr in (-1,0,1) for dc in (-1,0,1)}


def _clear_objects(objects):
    occupied = set()
    for obj in objects:
        cells = {(obj['top']+int(r),obj['left']+int(c))
                 for r,c in np.argwhere(obj['mask'])}
        if cells & _halo(occupied):
            return False
        occupied.update(cells)
    return True


def _four_connected(mask):
    a = np.where(mask, 1, 0).astype(np.int8)
    return len(_segment(a, 0, DIR4)) == 1


def _random_mask(rng, limit, growth_only=False):
    kind = int(rng.integers(5,8) if growth_only else rng.integers(8))
    h,w = (int(rng.integers(2,limit+1)) for _ in range(2))
    if kind == 0:
        mask = np.ones((1,int(rng.integers(1,limit+1))), dtype=bool)
    elif kind == 1:
        mask = np.zeros((h,w), dtype=bool)
        mask[:,0] = mask[-1,:] = True
    elif kind == 2:
        mask = np.ones((h,w), dtype=bool)
        if min(h,w) >= 3:
            mask[1:-1,1:-1] = False
    elif kind == 3:
        mask = np.zeros((h,w), dtype=bool)
        mask[:,int(rng.integers(w))] = True
        mask[int(rng.integers(h)),:] = True
    else:
        # Deterministic ordering before every random choice avoids set-order
        # dependence. Four- and eight-connected growth both occur.
        cells = {(0,0)}
        directions = DIR4 if kind == 4 else DIR8
        target = int(rng.integers(3,min(13,limit*limit)+1))
        for _ in range(target-1):
            choices = set()
            min_r,max_r = min(r for r,c in cells),max(r for r,c in cells)
            min_c,max_c = min(c for r,c in cells),max(c for r,c in cells)
            for r,c in sorted(cells):
                for dr,dc in directions:
                    p = r+dr,c+dc
                    if p in cells:
                        continue
                    if (max(max_r,p[0])-min(min_r,p[0]) < limit
                            and max(max_c,p[1])-min(min_c,p[1]) < limit):
                        choices.add(p)
            choices = sorted(choices)
            if not choices:
                break
            cells.add(choices[int(rng.integers(len(choices)))])
        lo_r = min(r for r,c in cells)
        lo_c = min(c for r,c in cells)
        mask = np.zeros((max(r for r,c in cells)-lo_r+1,
                         max(c for r,c in cells)-lo_c+1), dtype=bool)
        for r,c in cells:
            mask[r-lo_r,c-lo_c] = True
    return _trim(np.rot90(mask,int(rng.integers(4))))


def _sample_mask(rng, forbidden, limit, *, special=False, line=False):
    if line:
        candidates = [np.ones((n,1),dtype=bool) for n in range(2,limit+1)]
        candidates = [m for m in candidates if _rotation_key(m) not in forbidden]
        if not candidates:
            raise ValueError("no unused line shape")
        return candidates[int(rng.integers(len(candidates)))].copy()
    for _ in range(240):
        mask = _random_mask(rng,limit,growth_only=special)
        key = _rotation_key(mask)
        mirror = _rotation_key(np.fliplr(mask))
        if key in forbidden or mirror in forbidden:
            continue
        if special and (key == mirror or _four_connected(mask)
                        or len({_mask_key(np.rot90(mask,k)) for k in range(4)}) != 4):
            continue
        return mask
    raise ValueError("shape sampling budget exhausted")


def _objects(rng, sizes, singles, mode, limit, background):
    objects, forbidden, first_base = [], set(), None
    for i,count in enumerate(sizes):
        base = _sample_mask(rng,forbidden,limit,special=(i == 0 and mode == 'contrast'))
        if i == 0:
            first_base = base
        forbidden.add(_rotation_key(base))
        forbidden.add(_rotation_key(np.fliplr(base)))
        turns = [int(rng.integers(4)) for _ in range(count)]
        if i == 0 and mode == 'contrast':
            turns[1] = (turns[0]+int(rng.choice([1,3]))) % 4
        for turn in turns:
            objects.append(dict(mask=np.rot90(base,turn).copy(),up=True,group=i))
    for i in range(singles):
        if i == 0 and mode == 'contrast':
            base = np.fliplr(first_base).copy()
        else:
            base = _sample_mask(rng,forbidden,limit,line=(i == 0 and mode == 'triplet'))
        forbidden.add(_rotation_key(base))
        forbidden.add(_rotation_key(np.fliplr(base)))
        # Preserve a vertical line in the triplet witness so another object can
        # protect its locally equal rows from GLOBAL run contraction.
        turn = 0 if mode == 'triplet' and i == 0 else int(rng.integers(4))
        objects.append(dict(mask=np.rot90(base,turn).copy(),up=False,group=len(sizes)+i))
    palette = [int(c) for c in rng.permutation(10) if c != background]
    if rng.random() < .3:
        colors = palette[:len(objects)]
    else:
        n = int(rng.integers(1,min(9,len(objects))+1))
        colors = [int(rng.choice(palette[:n])) for _ in objects]
    if mode == 'contrast':
        colors[0],colors[1],colors[sum(sizes)] = palette[0],palette[1],palette[0]
    for obj,color in zip(objects,colors):
        obj['color'] = color
    return objects


def _layout(rng, objects, background, independent_columns=False):
    positions, widths = {}, []
    for up in (True,False):
        indices = [i for i,o in enumerate(objects) if o['up'] == up]
        cursor = 0
        for i in rng.permutation(indices):
            i = int(i)
            positions[i] = cursor
            cursor += objects[i]['mask'].shape[1]+int(rng.integers(1,3))
        widths.append(max((positions[i]+objects[i]['mask'].shape[1] for i in indices),default=0))
    needed = max(widths)
    if needed > 26:
        raise ValueError("scene too wide to expand")
    width = int(rng.integers(needed,min(28,needed+5)+1))
    for up,need in zip((True,False),widths):
        offset = int(rng.integers(width-need+1))
        for i,obj in enumerate(objects):
            if obj['up'] == up:
                obj['left'] = positions[i]+offset
    if independent_columns:
        # A teaching scene gives the competing classification rules enough
        # room to yield actual wrong grids rather than mere collisions.
        cursor = 0
        for i in rng.permutation(len(objects)):
            obj = objects[int(i)]
            obj['left'] = cursor
            cursor += obj['mask'].shape[1]+int(rng.integers(1,3))
        needed = max(o['left']+o['mask'].shape[1] for o in objects)
        if needed > 27:
            raise ValueError("contrast scene too wide")
        width = int(rng.integers(needed,min(28,needed+3)+1))
        offset = int(rng.integers(width-needed+1))
        for obj in objects:
            obj['left'] += offset
    up_h = max((o['mask'].shape[0] for o in objects if o['up']),default=0)
    down_h = max((o['mask'].shape[0] for o in objects if not o['up']),default=0)
    minimum = max(7,up_h+down_h+3)
    height = int(rng.integers(minimum,min(23,minimum+8)+1))
    occupied = set()
    for obj in sorted(objects,key=lambda o:-int(o['mask'].sum())):
        local = [(int(r),obj['left']+int(c)) for r,c in np.argwhere(obj['mask'])]
        for top in rng.permutation(height-obj['mask'].shape[0]+1):
            top = int(top)
            cells = {(r+top,c) for r,c in local}
            if not cells & _halo(occupied):
                obj['top'] = top
                occupied.update(cells)
                break
        else:
            raise ValueError("input placement budget exhausted")
    scene = np.full((height,width),background,dtype=np.int8)
    for obj in objects:
        _paint(scene,obj,obj['top'],background)
    reduced = _collapse(scene)
    # The scene is a construction aid. Only blank bands may disappear here;
    # foreground distortion is introduced explicitly by the later expansion.
    if np.count_nonzero(scene != background) != np.count_nonzero(reduced != background):
        raise ValueError("scene erased an intended mask vector")
    if _background(reduced) != background:
        raise ValueError("background lost its majority during normalization")
    output = _pack(reduced,background)
    if not _clear_objects(_segment(output,background)):
        raise ValueError("final objects would touch")
    if np.array_equal(reduced,output):
        raise ValueError("packing must be visible")
    return reduced,output


def _expansion_runs(rng, scene, axis, active, test):
    n = scene.shape[axis]
    runs = np.ones(n,dtype=int)
    if not active or n == 30:
        return runs
    extra = int(rng.integers(1,min(30-n,16 if test else 10)+1))
    # Local bursts, border bursts, or distributed thickness. Any full vector,
    # including a foreground vector, can be repeated.
    mode = int(rng.integers(3))
    if mode == 0:
        target = int(rng.choice([0,n-1]))
        runs[target] += extra
    elif mode == 1:
        runs[int(rng.integers(n))] += extra
    else:
        weights = rng.random(n)+.15
        runs += rng.multinomial(extra,weights/weights.sum())
    return runs


def _different(a,b):
    return a.shape != b.shape or not np.array_equal(a,b)


def _alternative(reduced, background, **kwargs):
    try:
        return _pack(reduced,background,**kwargs)
    except ValueError:
        return None


def _features(pair):
    source,out = (np.asarray(pair[k]) for k in ('input','output'))
    reduced = _collapse(source)
    background = _background(reduced)
    objects = _segment(reduced,background)
    raw_objects = _segment(source,background)
    counts = Counter(o['key'] for o in objects)
    raw_counts = Counter(o['key'] for o in raw_objects)
    partnered = [counts[o['key']] >= 2 for o in objects]
    raw_partnered = [raw_counts[o['key']] >= 2 for o in raw_objects]
    if len(raw_objects) != len(objects):
        raise AssertionError("equality expansion changed connectivity")
    moves = [abs(o['top']-(0 if up else reduced.shape[0]-o['mask'].shape[0]))
             for o,up in zip(objects,partnered)]
    alternatives = {
        'identity':dict(key_function=lambda o:_mask_key(o['mask'])),
        'reflection':dict(key_function=lambda o:min(o['key'],_rotation_key(np.fliplr(o['mask'])))),
        'area':dict(key_function=lambda o:int(o['mask'].sum())),
        'color':dict(key_function=lambda o:o['color']),
        'four_neighbor':dict(directions=DIR4),
        'exactly_two':dict(exactly_two=True),
    }
    rejected, clean_rejected = [], []
    for name,parameters in alternatives.items():
        alt = _alternative(reduced,background,**parameters)
        if alt is None or _different(alt,out):
            rejected.append(name)
            if alt is not None:
                clean_rejected.append(name)
    return dict(
        background=background,input_shape=tuple(source.shape),output_shape=tuple(out.shape),
        objects=len(objects),class_sizes=tuple(sorted(counts.values())),
        repeated_sizes=tuple(sorted(n for n in counts.values() if n >= 2)),
        singleton_count=sum(n == 1 for n in counts.values()),
        mixed=any(partnered) and not all(partnered),
        changed_partnership=raw_partnered != partnered,
        local_compression_wrong=any(_collapse(o['mask']).shape != o['mask'].shape for o in objects),
        separated_row_repeats=len({row.tobytes() for row in reduced}) < reduced.shape[0],
        separated_column_repeats=len({col.tobytes() for col in reduced.T}) < reduced.shape[1],
        shrink_rows=source.shape[0] > reduced.shape[0],
        shrink_columns=source.shape[1] > reduced.shape[1],
        post_packing_compression_wrong=_different(_collapse(out),out),
        moving_up=any(up and d for up,d in zip(partnered,moves)),
        moving_down=any(not up and d for up,d in zip(partnered,moves)),
        diagonal=any(not _four_connected(o['mask']) for o in objects),
        mask_keys=tuple(sorted(o['key'] for o in objects)),
        rejected=tuple(rejected),clean_rejected=tuple(clean_rejected),
        areas=tuple(sorted(int(o['mask'].sum()) for o in objects)),
        extents=tuple(sorted(o['mask'].shape for o in objects)),
        row_runs=_axis_runs(source,0)[1],column_runs=_axis_runs(source,1)[1],
    )


def _make_pair(rng, sizes, singles, mode, background, test):
    for attempt in range(600):
        # Large inventories get small masks; smaller inventories can use
        # larger rings/branches. This is a fit constraint, not a matching rule.
        if sum(sizes) <= 3:
            limit = int(rng.choice([3,4,5]))
        else:
            limit = int(rng.choice([2,3,4],p=[.45,.45,.10]))
        if attempt > 350:
            limit = 3
        try:
            objects = _objects(rng,sizes,singles,mode,limit,background)
            scene,out = _layout(rng,objects,background,independent_columns=mode == 'contrast')
        except ValueError:
            continue
        groups = Counter(o['key'] for o in _segment(scene,background))
        if sorted(groups.values()) != sorted(sizes+[1]*singles):
            continue
        for _ in range(12):
            active = [True,True]
            if test and rng.random() < .3:
                active[int(rng.integers(2))] = False
            rr = _expansion_runs(rng,scene,0,active[0],test)
            cc = _expansion_runs(rng,scene,1,active[1],test)
            source = np.repeat(np.repeat(scene,rr,axis=0),cc,axis=1)
            try:
                if _background(source) != background:
                    continue
            except ValueError:
                continue
            pair = {'input':source,'output':out.copy()}
            if mode == 'ordinary':
                return pair
            f = _features(pair)
            if not (f['shrink_rows'] or f['shrink_columns']):
                continue
            if mode == 'contrast':
                if (not f['changed_partnership'] or not f['moving_up'] or not f['moving_down']
                        or not {'identity','reflection','area','color','four_neighbor'} <= set(f['rejected'])
                        or not {'identity','reflection','area','color'} <= set(f['clean_rejected'])
                        or not f['separated_row_repeats'] or not f['separated_column_repeats']):
                    continue
            if mode == 'triplet':
                if (not f['local_compression_wrong'] or 'exactly_two' not in f['rejected']
                        or not f['moving_up'] or not f['moving_down']):
                    continue
            return pair
    raise RuntimeError("bounded scene sampling exhausted")


def _integer(value, name, low, high=None):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _fingerprint(a):
    return tuple(a.shape),np.asarray(a,dtype=np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed,'seed',0)
    num_train = _integer(num_train,'num_train',2,10)
    num_test = _integer(num_test,'num_test',1,4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    fixed_background = int(rng.integers(10)) if rng.random() < .25 else None
    train_specs = [([2],int(rng.integers(2,4)),'contrast'),
                   ([3],int(rng.integers(1,3)),'triplet')]
    ordinary_profiles = [([2],1),([2],2),([2],3),([3],0),([3],1),([3],2),
                         ([2,2],0),([],3),([],4),([],5)]
    for _ in range(num_train-2):
        sizes,singles = ordinary_profiles[int(rng.integers(len(ordinary_profiles)))]
        train_specs.append((sizes,singles,'ordinary'))
    train_specs = [train_specs[int(i)] for i in rng.permutation(num_train)]
    test_profiles = [([2,2],2),([3,2],1),([3],3),([2],4),([2,2,2],1),
                     ([3,3],1),([3,2],2),([3],4),([2,2],3),([2,2,2],2),
                     ([3,3],2),([3,2,2],1)]
    chosen = rng.choice(len(test_profiles),size=num_test,replace=False)
    test_specs = [(test_profiles[int(i)][0],test_profiles[int(i)][1],'ordinary') for i in chosen]
    puzzle,seen = {'train':[],'test':[]},set()
    for split,specs in (('train',train_specs),('test',test_specs)):
        for sizes,singles,mode in specs:
            for _ in range(24):
                background = int(rng.integers(10)) if fixed_background is None else fixed_background
                pair = _make_pair(rng,list(sizes),singles,mode,background,split == 'test')
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(obj, top):
    return obj['color'],top,obj['left'],_mask_key(obj['mask'])


def _validate_pair(pair):
    assert isinstance(pair,dict) and set(pair) == {'input','output'}
    source,out = (_grid(pair[k]) for k in ('input','output'))
    reduced = _collapse(source)
    background = _background(reduced)
    assert _background(source) == _background(out) == background
    assert out.shape == reduced.shape
    assert np.array_equal(_collapse(reduced),reduced)
    # Reconstruct the raw input exactly from unchanged representatives.
    _,rr = _axis_runs(source,0)
    _,cc = _axis_runs(source,1)
    assert np.array_equal(np.repeat(np.repeat(reduced,rr,axis=0),cc,axis=1),source)
    objects = _segment(reduced,background)
    frequencies = Counter(o['key'] for o in objects)
    expected = sorted(_signature(o,0 if frequencies[o['key']] >= 2 else
                                 reduced.shape[0]-o['mask'].shape[0]) for o in objects)
    actual = sorted(_signature(o,o['top']) for o in _segment(out,background))
    assert expected == actual, "mask, color, orientation, column, edge, or object count changed"
    assert _clear_objects(objects) and _clear_objects(_segment(source,background))
    assert _clear_objects(_segment(out,background))
    assert _different(reduced,out), "parent A must affect the answer"
    assert source.shape != out.shape, "parent B must affect the answer"
    assert np.array_equal(solve(source),out)
    return _features(pair)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    seen,features = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            f = _validate_pair(pair)
            key = _fingerprint(np.asarray(pair['input']))
            assert key not in seen, "duplicate input"
            seen.add(key)
            features[split].append(f)
    train,test = features['train'],features['test']
    assert sum(f['mixed'] for f in train) >= 2
    assert any(f['changed_partnership'] for f in train)
    assert any(f['local_compression_wrong'] for f in train)
    assert any(f['post_packing_compression_wrong'] for f in train)
    assert any(f['separated_row_repeats'] and f['separated_column_repeats'] for f in train)
    assert all(f['shrink_rows'] and f['shrink_columns'] for f in train)
    for alternative in ('identity','reflection','area','color','four_neighbor','exactly_two'):
        assert any(alternative in f['rejected'] for f in train), alternative
    for alternative in ('identity','reflection','area','color'):
        assert any(alternative in f['clean_rejected'] for f in train), alternative
    assert sum(f['moving_up'] and f['moving_down'] for f in train) >= 2
    assert any(3 in f['repeated_sizes'] for f in train)
    assert max(f['objects'] for f in train) < min(f['objects'] for f in test)
    assert all(3 <= f['objects'] <= 5 for f in train)
    assert all(6 <= f['objects'] <= 8 for f in test)


def _puzzle_digest(puzzle):
    digest = sha256()
    for split in ('train','test'):
        for pair in puzzle[split]:
            for role in ('input','output'):
                a = np.asarray(pair[role],dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded in-memory checks over exactly the 42 advertised seeds.

    Return a compact-serializable measurement summary; write no files. This is
    generator-owned verification, not an independent inference evaluation.
    """
    dimensions,output_dimensions,backgrounds,vocabularies,profiles,digests = (set() for _ in range(6))
    counts = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _puzzle_digest(puzzle)
        assert digest == _puzzle_digest(generate_puzzle(seed))
        digests.add(digest)
        for split in ('train','test'):
            for pair in puzzle[split]:
                f = _features(pair)
                dimensions.add(f['input_shape'])
                output_dimensions.add(f['output_shape'])
                backgrounds.add(f['background'])
                vocabularies.add(f['mask_keys'])
                profiles.add((f['repeated_sizes'],f['singleton_count']))
                for name in ('changed_partnership','local_compression_wrong','diagonal',
                             'separated_row_repeats','separated_column_repeats',
                             'post_packing_compression_wrong'):
                    counts[name] += int(f[name])
                counts['pairs'] += 1
                counts['one_noop_axis'] += int(f['shrink_rows'] != f['shrink_columns'])
                counts['three_copies'] += int(3 in f['repeated_sizes'])
                # Metamorphic checks: arbitrary global repetitions change
                # neither the contracted grid nor the answer; color roles
                # also commute with solving. No generator state is consulted.
                source = pair['input']
                before = _fingerprint(source)
                permutation = np.array([(i+3)%10 for i in range(10)],dtype=np.int8)
                assert np.array_equal(solve(permutation[source]),permutation[pair['output']])
                assert _fingerprint(source) == before
                scene = _collapse(source)
                if scene.shape[0] < 30:
                    stretched = np.insert(scene,0,scene[0],axis=0)
                    assert np.array_equal(solve(stretched),pair['output'])
    assert len(digests) == 42
    assert len(dimensions) >= 60 and len(output_dimensions) >= 30
    assert len(vocabularies) >= 100 and len(profiles) >= 10
    assert len(backgrounds) >= 8 and counts['one_noop_axis'] >= 3
    # All 30 legal count combinations; also reject types that happen to compare
    # numerically equal to valid counts (floats and bools).
    legal_count_calls = 0
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            puzzle = generate_puzzle(19,a,b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
            legal_count_calls += 1
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(-2,1),(4,-1),
                (True,1),(4,False),(4.0,1),(4,1.0),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1,True,1.5,None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    valid = generate_puzzle(0,2,1)
    broken = {s:[{r:g.copy() for r,g in pair.items()} for pair in valid[s]] for s in valid}
    broken['test'][0]['output'][0,0] = (int(broken['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError,ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42,pairs=counts['pairs'],distinct_puzzles=len(digests),
                input_dimensions=len(dimensions),output_dimensions=len(output_dimensions),
                mask_vocabularies=len(vocabularies),class_profiles=len(profiles),
                backgrounds=sorted(backgrounds),features=dict(counts),legal_count_calls=legal_count_calls,
                independent_solver=False)


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
