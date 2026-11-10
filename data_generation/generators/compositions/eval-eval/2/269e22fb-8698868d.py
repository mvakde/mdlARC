"""Cavity passport: local pattern recognition constrained by global topology.

All solving conventions are visible in one input. No parent code, stored
reference images, seed, or training labels are used by solve().
"""
from __future__ import annotations

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else STEPS
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                neighbor = (r + dr, c + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(cells)
    return result


def _transform(grid, index):
    # Four rotations, then a left/right reflection followed by four rotations.
    return np.rot90(grid if index < 4 else np.fliplr(grid), index % 4).copy()


def _compose(first, second):
    probe = np.arange(6).reshape(2,3)
    target = _transform(_transform(probe,second),first)
    return next(t for t in range(8) if np.array_equal(_transform(probe,t),target))


def _key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _rim(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= s <= 30 for s in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=True)


def _parse(inp):
    bg = int(inp[0, 0])
    if not np.all(_rim(inp) == bg):
        raise ValueError("the canvas must have a blank perimeter")
    boxes = [_box(cells) for cells in _components(inp != bg)]
    coupons, plates, dots = [], [], []
    for box in boxes:
        r0, r1, c0, c1 = box
        obj = inp[r0:r1, c0:c1]
        colors = np.unique(obj[obj != bg])
        if len(colors) == 3:
            q = int(obj[0, 0])
            if min(obj.shape) < 4 or not np.all(_rim(obj) == q):
                raise ValueError("malformed passport frame")
            patch = obj[1:-1, 1:-1]
            if len(np.unique(patch)) != 2 or np.any((patch == bg) | (patch == q)):
                raise ValueError("passport interior must use two separate colors")
            coupons.append((patch.copy(), q, box))
        elif len(colors) == 1:
            if obj.size == 1:
                dots.append((int(colors[0]), box))
            else:
                color = int(colors[0])
                if min(obj.shape) < 5 or not np.all(_rim(obj) == color):
                    raise ValueError("a plate needs a complete solid perimeter")
                plates.append((obj == bg, color, box))
        else:
            raise ValueError("unrecognized foreground object")
    if len(coupons) != 1 or not 2 <= len(plates) <= 4:
        raise ValueError("need one passport and two to four plates")
    patch, q, coupon_box = coupons[0]
    if any(color != q for color, _ in dots) or any(color == q for _, color, _ in plates):
        raise ValueError("frame color belongs only to the passport and census dots")
    n = len(dots)
    if not 1 <= n <= 4:
        raise ValueError("the passport census must be one to four")
    # Whole rectangles, including holes, must be spatially separated.
    for i, (a, b, c, d) in enumerate(boxes):
        for e, f, g, h in boxes[i + 1:]:
            if not (b < e or f < a or d < g or h < c):
                raise ValueError("objects need a blank-cell gap between their boxes")
    return dict(background=bg, patch=patch, flood=q, census=n,
                plates=plates, coupon_box=coupon_box, dots=dots)


def _locations(full, patch):
    h, w = patch.shape
    if h > full.shape[0] or w > full.shape[1]:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    return list(zip(*np.nonzero(np.all(windows == patch, axis=(-2, -1)))))


def _candidates(info, enforce_census=True, diagonal=False):
    patch = info['patch']
    colors = tuple(map(int, np.unique(patch)))
    matches = []
    for plate_index, (base, _, _) in enumerate(info['plates']):
        for t in range(8):
            holes = _transform(base, t)
            if patch.size >= holes.size:
                continue  # A passport is a proper crop, never the full plate.
            cavities = _components(holes, diagonal)
            labels = np.zeros(holes.shape, dtype=np.int16)
            for k, cells in enumerate(cavities, 1):
                rr, cc = zip(*cells)
                labels[rr, cc] = k
            for hole_color in colors:
                fragment = patch == hole_color
                for r, c in _locations(holes, fragment):
                    r, c = int(r), int(c)
                    touched = set(map(int, np.unique(labels[r:r+len(patch), c:c+patch.shape[1]]))) - {0}
                    if enforce_census and len(touched) != info['census']:
                        continue
                    ink_color = colors[1] if hole_color == colors[0] else colors[0]
                    out = np.where(holes, hole_color, ink_color).astype(np.int8)
                    flood_mask = np.isin(labels, list(touched))
                    out[flood_mask] = info['flood']
                    matches.append(dict(output=out, plate=plate_index, transform=t,
                                        offset=(r, c), hole_color=hole_color,
                                        holes=holes, touched=touched, labels=labels,
                                        flood_mask=flood_mask))
    return matches


def _decode(inp):
    info = _parse(inp)
    matches = _candidates(info)
    if len(matches) != 1:
        raise ValueError("passport and census must give exactly one alignment")
    return info, matches[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer a complete flooded plate from this input alone."""
    return _decode(_grid(input_grid))[1]['output'].copy()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _grow_together(rng, split, joined, crop):
    r0, r1, c0, c1 = crop
    target = int(rng.integers(0, max(2, (r1-r0)*(c1-c0)//2)))
    positions = [(r, c) for r in range(r0, min(r1, len(split)-1))
                 for c in range(max(1, c0), min(c1, split.shape[1]-1))]
    for _ in range(target):
        options = []
        for r, c in positions:
            if split[r, c] or not any(split[r+dr, c+dc] for dr, dc in STEPS):
                continue
            s, j = split.copy(), joined.copy()
            s[r, c] = j[r, c] = True
            if (len(_components(s)) == 2 and len(_components(j)) == 1
                    and len(_components(~s)) == len(_components(~j)) == 1):
                options.append((r, c))
        if not options:
            break
        cell = _choose(rng, options)
        split[cell] = joined[cell] = True


def _new_cavity(rng, masks, allowed, growth=5):
    """Add the same isolated cavity, preserving every existing component."""
    cells = [(r, c) for r, c in allowed
             if all(not m[max(0,r-1):r+2, max(0,c-1):c+2].any() for m in masks)]
    if not cells:
        return False
    start = _choose(rng, cells)
    counts = [len(_components(m)) for m in masks]
    trials = [m.copy() for m in masks]
    for m in trials:
        m[start] = True
    if any(len(_components(~m)) != 1 for m in trials):
        return False
    for m, trial in zip(masks, trials):
        m[:] = trial
    region = {start}
    for _ in range(int(rng.integers(1, growth + 1)) - 1):
        options = []
        for r, c in allowed:
            if (r, c) in region or not any((r+dr, c+dc) in region for dr, dc in STEPS):
                continue
            if any(m[r, c] for m in masks):
                continue
            trials = [m.copy() for m in masks]
            for m in trials:
                m[r, c] = True
            if all(len(_components(m)) == k+1 and len(_components(~m)) == 1
                   for m, k in zip(trials, counts)):
                options.append((r, c))
        if not options:
            break
        cell = _choose(rng, options)
        region.add(cell)
        for m in masks:
            m[cell] = True
    return True


def _twins(rng, common):
    # A pair of branch mouths: a bridge outside the aperture joins them in
    # one plate. Cutting it leaves diagonal contact but no four-connection.
    H, W = int(rng.integers(8, 12)), int(rng.integers(9, 12))
    top = int(rng.integers(1, 3))
    left = int(rng.integers(1, W-6))
    right = left + int(rng.integers(3, min(5, W-left-3)+1))
    ends = list(map(int, rng.choice(np.arange(top+3, H-1), 2, replace=False)))
    s = np.zeros((H, W), dtype=bool)
    s[top:ends[0]+1, left] = True
    s[top:ends[1]+1, right] = True
    s[top, left+2:right+1] = True
    s[top+1, left+1] = True
    j = s.copy()
    j[top, left+1] = True
    r0, r1 = top+2, max(ends)+2
    c0, c1 = left-1, right+2
    # Occasionally retain only one side margin or trim the bottom margin.
    if rng.random() < .25:
        c0 += 1
    if rng.random() < .25:
        r1 -= 1
    crop = (r0, r1, c0, c1)
    _grow_together(rng, s, j, crop)
    interior = [(r,c) for r in range(1,H-1) for c in range(1,W-1)]
    inside = [(r,c) for r,c in interior if r0 <= r < r1 and c0 <= c < c1]
    outside = [(r,c) for r,c in interior if not (r0 <= r < r1 and c0 <= c < c1)]
    # The joined plate gets an untouched chamber, equalizing TOTAL counts.
    if not _new_cavity(rng, [j], outside, growth=7):
        return None
    for _ in range(common):
        if not _new_cavity(rng, [s, j], inside, growth=4):
            return None
    for _ in range(int(rng.integers(0, 3))):
        _new_cavity(rng, [s, j], outside, growth=6)
    if not np.array_equal(s[r0:r1,c0:c1], j[r0:r1,c0:c1]):
        return None
    assert len(_components(s)) == len(_components(j))
    return j, s, crop


def _distractor(rng):
    H, W = int(rng.integers(5, 8)), int(rng.integers(5, 8))
    holes = np.zeros((H, W), dtype=bool)
    allowed = [(r,c) for r in range(1,H-1) for c in range(1,W-1)]
    if rng.random() < .20:
        holes[1:-1,1:-1] = True  # an ordinary hollow plate
    else:
        for _ in range(int(rng.integers(0, 6))):
            _new_cavity(rng, [holes], allowed, growth=9)
    return holes


def _fallback_twins(common):
    """A constructed, identifying witness for the finite rejection fallback."""
    split = np.zeros((11,11),dtype=bool)
    split[1:9,2] = True
    split[1:7,6] = True
    split[1,4:7] = True
    split[2,3] = True
    joined = split.copy()
    joined[1,3] = True
    joined[3:8,9] = True  # untouched, balancing the whole-plate cavity count
    if common >= 1:
        split[8,5:7] = joined[8,5:7] = True
    if common >= 2:
        split[4:6,4] = joined[4:6,4] = True
    return joined,split,(3,10,1,8)


def _place(rng, objects, flood, census, background):
    # Try genuinely scattered placements; the shelf fallback proves packing
    # feasibility for this sampler's dimensions, without affecting the rule.
    boxes = None
    order = sorted(range(len(objects)), key=lambda i: -objects[i].size)
    for _ in range(24):
        trial = {}
        for i in order:
            h, w = objects[i].shape
            options = []
            for r in range(1, 30-h):
                for c in range(1, 30-w):
                    if all(r+h < a or b < r or c+w < d or e < c
                           for a,b,d,e in trial.values()):
                        options.append((r,c))
            if not options:
                break
            r,c = _choose(rng, options)
            trial[i] = (r,r+h,c,c+w)
        if len(trial) == len(objects):
            boxes = trial
            break
    if boxes is None:
        boxes = {}
        c = 1
        for i in (0,1):
            h,w = objects[i].shape
            boxes[i] = (1,1+h,c,c+w)
            c += w+1
        r = 2+max(objects[0].shape[0], objects[1].shape[0])
        c = 1
        for i in range(2,len(objects)):
            h,w = objects[i].shape
            boxes[i] = (r,r+h,c,c+w)
            c += w+1
        assert max(b for _,b,_,_ in boxes.values()) <= 29
        assert max(d for _,_,_,d in boxes.values()) <= 29
    canvas = np.full((30,30), background, dtype=np.int8)
    for i, (r,s,c,d) in boxes.items():
        canvas[r:s,c:d] = objects[i]
    # Dots may be anywhere; frame-color identity connects them to the passport.
    occupied = np.zeros((30,30), dtype=bool)
    for r,s,c,d in boxes.values():
        occupied[r-1:s+1,c-1:d+1] = True
    for _ in range(census):
        positions = [(r,c) for r in range(1,29) for c in range(1,29) if not occupied[r,c]]
        r,c = _choose(rng, positions)
        canvas[r,c] = flood
        occupied[r-1:r+2,c-1:c+2] = True
    rr,cc = np.nonzero(canvas != background)
    # Keep a visibly blank perimeter, with independently varied extra margins.
    r0 = max(0, int(rr.min())-int(rng.integers(1,4)))
    c0 = max(0, int(cc.min())-int(rng.integers(1,4)))
    r1 = min(30, int(rr.max())+int(rng.integers(2,5)))
    c1 = min(30, int(cc.max())+int(rng.integers(2,5)))
    scene_transform = int(rng.integers(8))
    return _transform(canvas[r0:r1,c0:c1], scene_transform), scene_transform


def _make_pair(rng, joined_target, common, plate_count, relative=None):
    for attempt in range(181):
        twins = _twins(rng, common) if attempt < 180 else _fallback_twins(common)
        if twins is None:
            continue
        joined, split, crop = twins
        target = joined if joined_target else split
        r0,r1,c0,c1 = crop
        patch = target[r0:r1,c0:c1]
        if len(np.unique(patch)) != 2:
            continue
        # The visible library orientations do not dictate the answer orientation.
        library_orientations = list(map(int,rng.integers(8,size=2)))
        orient = (int(rng.integers(8)) if relative is None else
                  _compose(relative,library_orientations[0 if joined_target else 1]))
        masks = [_transform(joined,library_orientations[0]),
                 _transform(split,library_orientations[1])]
        masks += [_transform(_distractor(rng), int(rng.integers(8)))
                  for _ in range(plate_count-2)]
        palette = list(map(int, rng.permutation(10)))
        bg,q,a,b = palette[:4]
        ink,hole = (a,b) if rng.random() < .5 else (b,a)
        coupon = np.full(tuple(s+2 for s in _transform(patch,orient).shape), q, dtype=np.int8)
        coupon[1:-1,1:-1] = np.where(_transform(patch,orient), hole, ink)
        plate_colors = [int(rng.choice([color for color in palette if color not in (bg,q)]))
                        for _ in masks]
        plates = [np.where(mask,bg,color).astype(np.int8) for mask,color in zip(masks,plate_colors)]
        # Coupon is first on the second shelf if a fallback is required.
        objects = plates[:2]+[coupon]+plates[2:]
        n = (1 if joined_target else 2)+common
        inp,scene_transform = _place(rng, objects, q, n, bg)
        # Direct label construction uses the sampled geometry, independently of
        # the decoder's search. In particular, it fixes the exact orientation.
        expected = np.where(target,hole,ink).astype(np.int8)
        for cells in _components(target):
            if any(r0 <= r < r1 and c0 <= c < c1 for r,c in cells):
                rr,cc = zip(*cells)
                expected[rr,cc] = q
        expected = _transform(_transform(expected,orient),scene_transform)
        try:
            info, match = _decode(inp)
        except ValueError:
            continue
        # Reject both conflicting answers and duplicate latent alignments.
        local = _candidates(info, enforce_census=False)
        answers_without_b = {_key(m['output']) for m in local}
        if len(answers_without_b) < 2:
            continue
        if not np.array_equal(match['output'],expected):
            raise AssertionError("topological passport label disagrees with construction")
        return {'input':inp,'output':expected.copy()}
    raise AssertionError("constructed passport fallback must identify its plate")


def _count(name, value, low, high):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
            or not low <= int(value) <= high):
        raise ValueError(name+" has an illegal count")
    return int(value)


def _fresh_recolor(pair, seen):
    """A finite uniqueness fallback: 90 distinct background/frame images."""
    info = _parse(pair['input'])
    bg,q = info['background'],info['flood']
    source = [bg,q]+[c for c in range(10) if c not in (bg,q)]
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            target = [a,b]+[c for c in range(10) if c not in (a,b)]
            lookup = np.zeros(10,dtype=np.int8)
            lookup[source] = target
            changed = {k:lookup[v] for k,v in pair.items()}
            if _key(changed['input']) not in seen:
                return changed
    raise AssertionError('at most thirteen previous inputs cannot exhaust ninety recolorings')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count('num_train',num_train,2,10)
    nq = _count('num_test',num_test,1,4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train':[],'test':[]}, set()
    # Growth and extra touched chambers form a visible, same-rule test shift.
    shift = int(rng.integers(2))
    jobs = [('train',True,0,2), ('train',False,0,2)]
    jobs += [('train',bool(rng.integers(2)),int(rng.integers(0,2)),int(rng.integers(2,4)))
             for _ in range(nt-2)]
    jobs += [('test',bool(rng.integers(2)),int(rng.integers(1,3)) if shift else int(rng.integers(0,2)),
              int(rng.integers(3,5))) for _ in range(nq)]
    for job_index,(split, joined, common, number) in enumerate(jobs):
        relative = (int(rng.choice([1,3])) if job_index == 0 else
                    int(rng.integers(4,8)) if job_index == 1 else None)
        for _ in range(12):
            pair = _make_pair(rng,joined,common,number,relative)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            pair = _fresh_recolor(pair,seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp,out = _grid(pair['input']),_grid(pair['output'])
    info, match = _decode(inp)
    assert np.array_equal(out,match['output']), 'incorrect completion, orientation, or flood'
    holes = match['holes']
    r,c = match['offset']
    h,w = info['patch'].shape
    inside = np.zeros(holes.shape,dtype=bool)
    inside[r:r+h,c:c+w] = True
    local = _candidates(info,enforce_census=False)
    counts = [len(_components(mask)) for mask,_,_ in info['plates']]
    areas = [int(mask.sum()) for mask,_,_ in info['plates']]
    touched_8 = set(map(int,np.unique(_eight_labels(holes)[inside]))) - {0}
    inverse = next(t for t in range(8) if _compose(t,match['transform']) == 0)
    return dict(info=info,match=match,shape=inp.shape,output_shape=out.shape,
                patch_shape=(h,w),census=info['census'],plate_count=len(info['plates']),
                no_b_answers=len({_key(m['output']) for m in local}),
                outside_flood=int(np.sum(match['flood_mask'] & ~inside)),
                untouched=int(np.sum(holes & ~match['flood_mask'])),
                local_count=len(_components(holes[r:r+h,c:c+w])),
                eight_count=len(touched_8),total_count=len(_components(holes)),
                equal_total_competitor=any(m['plate'] != match['plate'] and
                    counts[m['plate']] == counts[match['plate']] for m in local),
                area_inversion=info['census'] == 1 and any(m['plate'] != match['plate'] and
                    areas[m['plate']] < areas[match['plate']] for m in local),
                a_ablation_changed=not np.array_equal(out,_transform(out,inverse)),
                area=int(holes.sum()),plate_counts=counts)


def _eight_labels(mask):
    labels = np.zeros(mask.shape,dtype=np.int16)
    for k,cells in enumerate(_components(mask,True),1):
        rr,cc = zip(*cells)
        labels[rr,cc] = k
    return labels


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    nt,nq = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen,analyses = set(),{'train':[],'test':[]}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            key = _key(_grid(pair['input']))
            assert key not in seen, 'duplicate inputs'
            seen.add(key)
            a = _analyze(pair)
            assert a['no_b_answers'] >= 2, 'crop recognition alone must be insufficient'
            assert a['equal_total_competitor'], 'total-hole census shortcut not refuted'
            assert a['outside_flood'] > 0, 'flood must propagate beyond the sample'
            analyses[split].append(a)
    train = analyses['train']
    assert any(a['census'] == 1 and a['local_count'] > 1 and a['untouched'] > 0 for a in train)
    assert any(a['census'] == 2 and a['eight_count'] < 2 for a in train)
    assert any(a['area_inversion'] for a in train), 'area-rank shortcut not refuted'
    assert any(a['a_ablation_changed'] for a in train), 'omitting geometric alignment has no effect'
    # A quarter turn and a reflection visibly generate the whole D4 family,
    # including when only two training examples were requested.
    assert any(a['match']['transform'] in (1,3) for a in train)
    assert any(a['match']['transform'] >= 4 for a in train)
    return None


def _signature(puzzle):
    return tuple((s,_key(p['input']),_key(p['output']))
                 for s in ('train','test') for p in puzzle[s])


def self_test():
    """Bounded actual-grid checks over 42 seeds, plus targeted corruptions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_analyze(p) for puzzle in puzzles for s in ('train','test') for p in puzzle[s]]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    assert len({a['shape'] for a in records}) >= 15
    assert len({a['output_shape'] for a in records}) >= 8
    assert len({a['patch_shape'] for a in records}) >= 12
    assert {a['census'] for a in records} == {1,2,3,4}
    assert {a['plate_count'] for a in records} == {2,3,4}
    assert len({min(_key(_transform(a['match']['holes'],t)) for t in range(8))
                for a in records}) >= 100, 'measure full geometry modulo D4'
    assert len({a['match']['offset'] for a in records}) >= 12
    assert len({a['info']['background'] for a in records}) == 10
    assert len({a['info']['flood'] for a in records}) == 10
    # Change only a census dot, keeping every plate and the passport intact.
    # This must switch the recovered plate when the other census has one match.
    counterfactuals = 0
    for puzzle in puzzles:
        for pair in puzzle['train']:
            info,match = _decode(pair['input'])
            if info['census'] != 2:
                continue
            changed = pair['input'].copy()
            r,_,c,_ = info['dots'][0][1]
            changed[r,c] = info['background']
            candidates = _candidates(_parse(changed))
            if len(candidates) == 1:
                assert not np.array_equal(solve(changed),pair['output'])
                counterfactuals += 1
    assert counterfactuals >= 20, 'census markers must causally change completion'
    # Nested-list inputs work too; solving must not mutate its argument.
    source = puzzles[0]['test'][0]['input'].copy()
    before = source.copy()
    assert np.array_equal(solve(source.tolist()),puzzles[0]['test'][0]['output'])
    assert np.array_equal(solve(source),puzzles[0]['test'][0]['output'])
    assert np.array_equal(source,before)
    # Prove the rejection fallback is identifying in every census case and
    # both color maps, independently of its placement or palette.
    for common in range(3):
        joined,split,crop = _fallback_twins(common)
        r0,r1,c0,c1 = crop
        assert len(_components(~joined)) == len(_components(~split)) == 1
        assert len(_components(joined)) == len(_components(split)) == 2+common
        for target,n in ((joined,1+common),(split,2+common)):
            for orient in range(8):
                patch = _transform(target[r0:r1,c0:c1],orient)
                info = dict(patch=np.where(patch,2,1),flood=3,census=n,
                            plates=[(joined,4,None),(split,4,None)])
                assert len(_candidates(info)) == 1
                assert len(_candidates(info,enforce_census=False)) == 2
    # Count boundaries here; verify.py separately exercises every legal pair.
    for nt,nq in ((2,1),(3,2),(5,4),(10,1),(10,4)):
        p = generate_puzzle(400+10*nt+nq,nt,nq)
        assert len(p['train']) == nt and len(p['test']) == nq
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    broken = {s:[{k:np.array(v,copy=True) for k,v in p.items()} for p in puzzles[0][s]]
              for s in ('train','test')}
    out = broken['test'][0]['output']
    out[0,0] = (int(out[0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42,pairs=len(records),input_shapes=len({a['shape'] for a in records}),
                output_shapes=len({a['output_shape'] for a in records}),
                patch_shapes=len({a['patch_shape'] for a in records}),
                normalized_plate_shapes=len({min(_key(_transform(a['match']['holes'],t))
                                            for t in range(8)) for a in records}),
                census_counterfactuals=counterfactuals)


if __name__ == '__main__':
    print(self_test())
