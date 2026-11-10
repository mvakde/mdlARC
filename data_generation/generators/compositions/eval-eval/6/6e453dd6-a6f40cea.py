"""Aperture-triggered completion of hidden rectangular circuits.

Only standard library and NumPy are used. No files, parent modules, seed state,
or example labels are consulted by solve(). See task_spec.md for the rule.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, todo = {start}, [start]
        while todo:
            y, x = todo.pop()
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    todo.append(q)
        result.append(group)
    return result


def _bbox(points):
    ys, xs = zip(*points)
    return int(min(ys)), int(max(ys)), int(min(xs)), int(max(xs))


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r+1)} | {(b, x) for x in range(l, r+1)}
            | {(y, l) for y in range(t+1, b)} | {(y, r) for y in range(t+1, b)})


def _inside(point, box, strict=False):
    y, x = point
    t, b, l, r = box
    if strict:
        return t < y < b and l < x < r
    return t <= y <= b and l <= x <= r


def _apertures(work, bg):
    """Inspect actual four-connected cavities and the whole component extreme."""
    objects, rows, all_holes, naive_rows = [], set(), set(), set()
    for cells in _components(work != bg):
        t, b, l, r = _bbox(cells)
        mask = np.zeros((b-t+1, r-l+1), dtype=bool)
        for y, x in cells:
            mask[y-t, x-l] = True
        holes, active, opened = [], [], set()
        for region in _components(~mask):
            boundary = any(y in (0, mask.shape[0]-1) or x in (0, mask.shape[1]-1)
                           for y, x in region)
            contact = {int(y+t) for y, x in region if x == mask.shape[1]-2}
            if boundary:
                opened.update(contact)
            else:
                holes.append(region)
                all_holes.update(int(y+t) for y, x in region)
                if contact:
                    # Family members have full-row contact. Explicit row-local
                    # semantics still make solve well-defined on partial contact.
                    active.append(region)
                    rows.update(contact)
        if mask.shape[1] >= 3:
            naive_rows.update(int(t+y) for y in range(1, mask.shape[0]-1) if not mask[y, -2])
        objects.append(dict(box=(t,b,l,r), cells=cells, mask=mask, holes=holes,
                            active=active, open_rows=opened))
    return objects, rows, all_holes, naive_rows


def _canonical(grid):
    candidates = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        h, w = a.shape
        bg = int(a[0, 0])
        if not all(int(a[y,x]) == bg for y,x in ((0,w-1),(h-1,0),(h-1,w-1))):
            continue
        for c in range(3, w-5):
            rail = int(a[0,c])
            if rail == bg or not np.all(a[:,c] == rail) or np.count_nonzero(a == rail) != h:
                continue
            left_colors = set(map(int,np.unique(a[:,:c]))) - {bg}
            right_colors = set(map(int,np.unique(a[:,c+1:]))) - {bg}
            if len(left_colors) == 1 and len(right_colors) >= 2 and not (left_colors & right_colors):
                candidates.append((a,turn,bg,rail,c,left_colors.pop()))
    _require(len(candidates) == 1, "rail direction is not unique")
    return candidates[0]


def _read(input_grid):
    a, turn, bg, rail, col, fg = _canonical(_grid(input_grid))
    objects, rows, all_holes, naive = _apertures(a[:,:col], bg)
    colors = set(map(int,np.unique(a[:,col+1:]))) - {bg}
    frames = []
    color_points = {}
    for color in sorted(colors):
        points = {tuple(map(int,p)) for p in np.argwhere(a == color)}
        color_points[color] = points
        box = _bbox(points)
        t,b,l,r = box
        if b-t >= 3 and r-l >= 3 and points == _perimeter(box) and np.all(a[t+1:b,l+1:r] == bg):
            frames.append((box,color))
    _require(len(frames) == 1, "expected one complete opaque window")
    frame, frame_color = frames[0]
    groups = {}
    for color, points in color_points.items():
        if color != frame_color:
            groups.setdefault(_bbox(points), []).append(color)
    loops, used = [], set()
    for box, palette in sorted(groups.items()):
        t,b,l,r = box
        _require(b-t >= 3 and r-l >= 3 and col < l, "invalid loop extent")
        full = _perimeter(box)
        visible = set().union(*(color_points[c] for c in palette))
        _require(visible == {p for p in full if not _inside(p,frame)}, "outline not uniquely completed")
        hidden = {p for p in full if _inside(p,frame,True)}
        _require(bool(hidden), "loop has no hidden interior")
        _require(not (used & full), "loop perimeters intersect")
        used.update(full)
        _require(1 <= len(palette) <= 2, "unsupported loop palette")
        if len(palette) == 1:
            phase = (palette[0],palette[0])
        else:
            parity = {}
            for y,x in visible:
                k, value = (y+x)%2, int(a[y,x])
                _require(k not in parity or parity[k] == value, "inconsistent alternating phase")
                parity[k] = value
            _require(len(parity) == 2, "unwitnessed alternating phase")
            phase = (parity[0],parity[1])
        loops.append(dict(box=box, full=full, visible=visible, hidden=hidden, phase=phase))
    _require(bool(loops), "no circuits")
    return dict(grid=a, turn=turn, bg=bg, rail=rail, col=col, fg=fg, frame=frame,
                frame_color=frame_color, objects=objects, rows=rows, all_holes=all_holes,
                naive_rows=naive, loops=loops)


def _hit(scene, rows=None, visible_only=False, every_hit=False):
    selected = set()
    for y in scene['rows'] if rows is None else rows:
        hits = []
        for i, loop in enumerate(scene['loops']):
            if visible_only:
                xs = [x for yy,x in loop['visible'] if yy == y]
                if xs:
                    hits.append((min(xs),i))
            else:
                # A completed rectangle has its left-wall cell on every row
                # in its vertical extent; scanning the perimeter is redundant.
                t,b,l,r = loop['box']
                if t <= y <= b:
                    hits.append((l,i))
        if hits:
            selected.update(i for x,i in hits) if every_hit else selected.add(min(hits)[1])
    return selected


def _render_crop(scene, selected, row_stencil=False):
    t,b,l,r = scene['frame']
    out = np.full((b-t-1,r-l-1),scene['bg'],dtype=np.int8)
    for i in selected:
        loop = scene['loops'][i]
        for y,x in loop['hidden']:
            if not row_stencil or y in scene['rows']:
                out[y-t-1,x-l-1] = loop['phase'][(y+x)%2]
    return np.rot90(out,-scene['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete hidden geometry before finding the first contact of each ray."""
    scene = _read(input_grid)
    return _render_crop(scene,_hit(scene))


def _mask(rng, kind, height, width):
    mask = np.ones((height,width),dtype=bool)
    if kind == 'solid':
        # A solid staircase, still one connected component.
        for y in range(height):
            mask[y,:int(rng.integers(max(1,width-1)))] = False
    elif kind == 'multi':
        for y in range(1,height-1,2):
            left = int(rng.integers(1,width-1))
            mask[y,left:-1] = False
    else:
        # Row-varying back edges keep active cavities fully touching the front.
        for y in range(1,height-1):
            left = int(rng.integers(1,max(2,width-2)))
            mask[y,left:-1] = False
        if kind == 'open':
            if rng.random() < .5:
                mask[1:-1,0] = False
                mask[1:-1,1:-1] = False
            else:
                mask[0,-2] = False
        if kind == 'recessed':
            extension = int(rng.integers(1,3))
            extended = np.zeros((height,width+extension),dtype=bool)
            extended[:,:width] = mask
            extended[0 if rng.random() < .5 else -1,width:] = True
            mask = extended
    return mask


def _both_parities_outside(start,stop,cover_start,cover_stop):
    """Whether a line minus one closed interval witnesses both parities."""
    before = min(stop,cover_start-1)-start+1
    after = stop-max(start,cover_stop+1)+1
    return (before >= 2 or after >= 2
            or (before == 1 and after == 1 and (stop-start)%2 == 1))


def _readable_box(box,frame):
    """Exact scalar equivalent of the hidden-cell/extrema set predicates.

    Each side must retain both parity colors outside the inclusive cover.
    At least one side must cross the strict interior. No RNG is consumed.
    """
    t,b,l,r = box
    ft,fb,fl,fr = frame
    vertical = max(t,ft+1) <= min(b,fb-1)
    horizontal = max(l,fl+1) <= min(r,fr-1)
    if not ((vertical and (fl < l < fr or fl < r < fr))
            or (horizontal and (ft < t < fb or ft < b < fb))):
        return False
    return ((not fl <= l <= fr or _both_parities_outside(t,b,ft,fb))
            and (not fl <= r <= fr or _both_parities_outside(t,b,ft,fb))
            and (not ft <= t <= fb or _both_parities_outside(l,r,fl,fr))
            and (not ft <= b <= fb or _both_parities_outside(l,r,fl,fr)))


def _loop_geometry(rng,h,w,col,frame,count,rows,require_two=False,hidden_contrast=False):
    """Bounded search over randomly sized boxes; no saved trial payloads."""
    candidates = []
    seen = set()
    row_options = np.arange(1,h-1)
    col_options = np.arange(col+2,w-1)
    for _ in range(1000):
        t,b = sorted(map(int,rng.choice(row_options,2,replace=False)))
        l,r = sorted(map(int,rng.choice(col_options,2,replace=False)))
        box = (t,b,l,r)
        if box in seen or b-t < 3 or r-l < 3:
            continue
        seen.add(box)
        if not _readable_box(box,frame):
            continue
        full = _perimeter(box)
        visible = {p for p in full if not _inside(p,frame)}
        hidden = {p for p in full if _inside(p,frame,True)}
        candidates.append(dict(box=box,full=full,visible=visible,hidden=hidden))
    if len(candidates) < count:
        return None
    for _ in range(160):
        chosen, occupied = [], set()
        for j in rng.permutation(len(candidates)):
            candidate = candidates[int(j)]
            if not (occupied & candidate['full']):
                chosen.append(candidate)
                occupied.update(candidate['full'])
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            continue
        dummy = dict(rows=rows,loops=chosen)
        hit = _hit(dummy)
        if (hit and (not require_two or len(hit) >= 2)
                and (not hidden_contrast or hit != _hit(dummy,visible_only=True))):
            return chosen
    return None


def _make_pair(rng,kind_schedule,loop_count,patterned,turn,require_two=False,hidden_contrast=False):
    for attempt in range(60):
        masks = []
        for kind in kind_schedule:
            height = int(rng.integers(5 if kind == 'multi' else 3,7))
            width = int(rng.integers(3,6))
            masks.append(_mask(rng,kind,height,width))
        rng.shuffle(masks)
        base = sum(m.shape[0] for m in masks)+len(masks)+1
        h = int(rng.integers(max(19,base),31))
        col = max(m.shape[1] for m in masks)+int(rng.integers(1,4))
        w = int(rng.integers(max(26,col+15),31))
        palette = list(map(int,rng.permutation(10)))
        bg,rail,fg,frame_color = palette[:4]
        a = np.full((h,w),bg,dtype=np.int8)
        a[:,col] = rail
        extras = rng.multinomial(h-base,np.full(len(masks)+1,1/(len(masks)+1)))
        y = 1+int(extras[0])
        for i,mask in enumerate(masks):
            x = int(rng.integers(0,col-mask.shape[1]+1))
            a[y:y+mask.shape[0],x:x+mask.shape[1]][mask] = fg
            y += mask.shape[0]+1+int(extras[i+1])
        objects,rows,all_holes,naive = _apertures(a[:,:col],bg)
        frame = (int(rng.integers(4,9)),h-int(rng.integers(3,7)),
                 col+int(rng.integers(5,9)),w-int(rng.integers(2,5)))
        if frame[1]-frame[0] < 5 or frame[3]-frame[2] < 5:
            continue
        loops = _loop_geometry(rng,h,w,col,frame,loop_count,rows,require_two,hidden_contrast)
        if loops is None:
            continue
        available = palette[4:]
        for i,loop in enumerate(loops):
            first = available.pop()
            two = (patterned and i == 0) or rng.random() < .45
            second = available.pop() if two else first
            phase = (first,second) if rng.random() < .5 else (second,first)
            loop['phase'] = phase
            for yy,xx in loop['full']:
                a[yy,xx] = phase[(yy+xx)%2]
        t,b,l,r = frame
        a[t:b+1,l:r+1] = bg
        for p in _perimeter(frame):
            a[p] = frame_color
        scene = dict(frame=frame,loops=loops,rows=rows,bg=bg,turn=0)
        out = _render_crop(scene,_hit(scene))
        pair = dict(input=np.rot90(a,turn).copy(),output=np.rot90(out,turn).copy())
        read = _read(pair['input'])
        _require(np.array_equal(solve(pair['input']),pair['output']),"builder/reader disagree")
        return pair
    raise RuntimeError('bounded geometry search exhausted')


def _key(a):
    a = np.asarray(a,dtype=np.int8)
    return a.shape,a.tobytes()


def _different(scene, **kwargs):
    return not np.array_equal(_render_crop(scene,_hit(scene)),
                              _render_crop(scene,_hit(scene,**kwargs)))


def _features(scene):
    selected = _hit(scene)
    open_rows = scene['rows'] | {y for o in scene['objects'] if not o['holes']
                                 for y in o['open_rows']}
    return dict(active=bool(scene['rows']),
                open=any(o['open_rows'] for o in scene['objects']),
                recessed=any(o['holes'] and not o['active'] for o in scene['objects']),
                ignore_a=_different(scene,rows=range(scene['grid'].shape[0])),
                all_holes=_different(scene,rows=scene['all_holes']),
                open_shortcut=_different(scene,rows=open_rows),
                visible_only=_different(scene,visible_only=True),
                all_crossed=_different(scene,every_hit=True),
                all_loops=len(selected)<len(scene['loops']),
                whole_loop=not np.array_equal(_render_crop(scene,selected),
                                              _render_crop(scene,selected,row_stencil=True)),
                patterned=any(scene['loops'][j]['phase'][0]!=scene['loops'][j]['phase'][1] for j in selected),
                selected=len(selected))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed,'seed')
    num_train,num_test = _integer(num_train,'num_train'),_integer(num_test,'num_test')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed % (1<<64))
    puzzle,seen = dict(train=[],test=[]),set()
    base_turn = int(rng.integers(4))
    for split,count in (('train',num_train),('test',num_test)):
        for i in range(count):
            # Two complementary demonstrations work even for the minimum 2/1.
            if split == 'train' and i == 0:
                kinds = ['active','open']
            elif split == 'train' and i == 1:
                kinds = ['active','recessed','open']
            else:
                n = int(rng.integers(1,5))
                kinds = ['active']+[str(rng.choice(['active','open','recessed','solid','multi'])) for _ in range(n-1)]
            loops = 3 if split == 'test' else (2 if i < 2 else int(rng.integers(1,3)))
            for _ in range(48):
                turn = (base_turn+i%2) % 4 if split == 'train' else int(rng.integers(4))
                pair = _make_pair(rng,kinds,loops,i == 1 or split == 'test',turn,
                                  require_two=False,hidden_contrast=split == 'train' and i == 0)
                scene = _read(pair['input'])
                f = _features(scene)
                if split == 'train' and i == 0 and not (f['visible_only'] and f['all_crossed']):
                    continue
                if split == 'train' and i == 1 and not (f['all_holes'] and f['open_shortcut'] and f['patterned']):
                    continue
                if _key(pair['input']) in seen:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded demonstration search exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle)=={'train','test'},'invalid puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in ('train','test')),'invalid split')
    a,b = len(puzzle['train']),len(puzzle['test'])
    _require(2<=a<=10 and 1<=b<=4 and a>b,'invalid counts')
    seen,train = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair)=={'input','output'},'invalid pair keys')
            inp,out = _grid(pair['input']),_grid(pair['output'])
            _require(_key(inp) not in seen,'duplicate input')
            seen.add(_key(inp))
            scene = _read(inp)
            objects = scene['objects']
            _require(all(a['box'][1]+1<b['box'][0] for a,b in zip(objects,objects[1:])), 'projectors touch')
            for o in objects:
                for hole in o['active']:
                    _require({y for y,x in hole}=={y for y,x in hole if x==o['mask'].shape[1]-2},
                             'partial-contact cavity outside generation contract')
            selected = _hit(scene)
            _require(np.array_equal(out,_render_crop(scene,selected)),'incorrect circuit crop')
            _require(np.array_equal(out,solve(inp)),'input-only solve disagrees')
            # Each hidden pixel belongs to exactly one inferred loop, and crop
            # palette contains only background and activated visible palettes.
            allowed = {scene['bg']} | {c for j in selected for c in scene['loops'][j]['phase']}
            _require(set(map(int,np.unique(out)))<=allowed,'inactive palette in output')
            if split == 'train':
                train.append(_features(scene))
    for name in ('active','open','recessed','ignore_a','all_holes','open_shortcut',
                 'visible_only','all_crossed','all_loops','whole_loop','patterned'):
        _require(any(f[name] for f in train),f'training lacks {name} contrast')


def self_test():
    """42 seeds, all explicit counts, invariants, equivariance and bad labels.

    Streaming checks retain hashes and small sets, not entire sample suites.
    No filesystem writes or execution traces are made.
    """
    signatures,shapes,layouts,object_shapes,backgrounds,orientations = (set() for _ in range(6))
    # Check the fast search predicate against the original pixel definition,
    # including covers that touch corners or leave isolated parity witnesses.
    regression_rng = np.random.default_rng(92170)
    geometry_checks = 0
    for _ in range(1200):
        t,b = sorted(map(int,regression_rng.choice(30,2,replace=False)))
        l,r = sorted(map(int,regression_rng.choice(30,2,replace=False)))
        ft,fb = sorted(map(int,regression_rng.choice(30,2,replace=False)))
        fl,fr = sorted(map(int,regression_rng.choice(30,2,replace=False)))
        if b-t < 3 or r-l < 3:
            continue
        box,frame = (t,b,l,r),(ft,fb,fl,fr)
        full = _perimeter(box)
        visible = {p for p in full if not _inside(p,frame)}
        pixel_rule = (any(_inside(p,frame,True) for p in full)
                      and all((pts := {p for p in visible if sum(p)%2 == k})
                              and _bbox(pts) == box for k in (0,1)))
        _require(bool(pixel_rule) == _readable_box(box,frame),
                 'fast rectangle search changes the pixel rule')
        geometry_checks += 1
    records = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                records += 1
                inp = pair['input']
                digest.update(repr(inp.shape).encode()+inp.tobytes()+pair['output'].tobytes())
                s = _read(inp)
                pixel_hits = set()
                for y in s['rows']:
                    contacts = [(x,j) for j,loop in enumerate(s['loops'])
                                for yy,x in loop['full'] if yy == y]
                    if contacts:
                        pixel_hits.add(min(contacts)[1])
                _require(pixel_hits == _hit(s),'fast first contact changes selected circuits')
                shapes.add((inp.shape,pair['output'].shape))
                layouts.add((s['frame'],tuple(l['box'] for l in s['loops'])))
                object_shapes.update((o['mask'].shape,o['mask'].tobytes()) for o in s['objects'])
                backgrounds.add(s['bg'])
                orientations.add(s['turn'])
                for k in range(4):
                    _require(np.array_equal(solve(np.rot90(inp,k)),np.rot90(pair['output'],k)),
                             'rotation equivariance failed')
        signatures.add(digest.hexdigest())
    _require(len(signatures)==42,'seed diversity collapsed')
    _require(len(shapes)>=60 and len(layouts)>=180 and len(object_shapes)>=100,'structural diversity collapsed')
    _require(backgrounds==set(range(10)) and orientations==set(range(4)),'role/orientation diversity collapsed')
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            generate_puzzle(620+a*10+b,a,b)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    p,q = generate_puzzle(99173,2,1),generate_puzzle(99173,2,1)
    _require(all(_key(x[k])==_key(y[k]) for s in ('train','test') for x,y in zip(p[s],q[s])
                 for k in ('input','output')),'nondeterminism')
    p['test'][0]['output'][0,0] = (int(p['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42,pairs=records,distinct_puzzles=len(signatures),shape_pairs=len(shapes),
                layouts=len(layouts),object_masks=len(object_shapes),all_counts=True,
                geometry_regressions=geometry_checks,first_contact_regressions=records)
