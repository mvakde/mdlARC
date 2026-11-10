"""Registered cavity punches: translate contours, flood their air, carve tiles.

No parent code, saved examples, RNG state, or training labels are used by solve.
All generation is bounded and all experiments stream one puzzle at a time.
"""
from __future__ import annotations

import hashlib
from collections import deque
import numpy as np


HOST_SEEDS = (0,906630807,607625561,110530303,985942499,711193968,559332912,
90626890,127057655,372577252,450538721,990051858,346803347,892300744,624410645,
432670909,304175370,72978580,292259823,71607321,609186883,858744236,245912761,
817507700,634861330,184549363,547340981,10367687,864919251,518893882,122876882,
154029953,140720977,237244942,86883115,465554614,16701789,465927831,117813724,
878190561,227039127,732679121)


def _components(cells, diagonal=False):
    remaining = set(cells)
    result = []
    directions = [(y,x) for y in (-1,0,1) for x in (-1,0,1)
                  if (y or x) and (diagonal or not (y and x))]
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        part, stack = {first}, [first]
        while stack:
            y, x = stack.pop()
            for dy, dx in directions:
                q = y+dy, x+dx
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _bbox(cells):
    return (min(y for y,x in cells), min(x for y,x in cells),
            max(y for y,x in cells)+1, max(x for y,x in cells)+1)


def _mask(cells):
    y0, x0, y1, x1 = _bbox(cells)
    result = np.zeros((y1-y0,x1-x0), dtype=bool)
    for y,x in cells:
        result[y-y0,x-x0] = True
    return result, (y0,x0)


def _cavities(ink):
    """Four-neighbor air cannot cross ink, including diagonal ink chains."""
    wall = np.pad(ink, 1)
    exterior = np.zeros(wall.shape, dtype=bool)
    exterior[0,0] = True
    queue = deque([(0,0)])
    h,w = wall.shape
    while queue:
        y,x = queue.popleft()
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            yy,xx = y+dy,x+dx
            if 0 <= yy < h and 0 <= xx < w and not wall[yy,xx] and not exterior[yy,xx]:
                exterior[yy,xx] = True
                queue.append((yy,xx))
    return (~wall & ~exterior)[1:-1,1:-1]


def _regions(mask):
    return len(_components(map(tuple,np.argwhere(mask))))


def _transforms(mask):
    return [np.rot90(mask,k) for k in range(4)] + [np.rot90(np.fliplr(mask),k) for k in range(4)]


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _grid(value):
    result = np.asarray(value)
    assert result.ndim == 2 and all(1 <= d <= 30 for d in result.shape)
    assert np.issubdtype(result.dtype,np.integer)
    assert np.all((0 <= result) & (result <= 9))
    return result


def _decode(value):
    grid = _grid(value)
    bg = int(grid[0,0])
    assert all(np.all(edge == bg) for edge in (grid[0],grid[-1],grid[:,0],grid[:,-1])), 'blank outer border required'
    colors = set(map(int,np.unique(grid))) - {bg}
    tiles, tile_cells = {}, set()
    for color in sorted(colors):
        cells = set(map(tuple,np.argwhere(grid == color)))
        for part in _components(cells):
            y0,x0,y1,x1 = _bbox(part)
            if min(y1-y0,x1-x0) >= 7 and len(part) == (y1-y0)*(x1-x0):
                assert color not in tiles, 'duplicate target color'
                tiles[color] = (y0,x0,y1,x1)
                tile_cells.update(part)
    assert 1 <= len(tiles) <= 3, 'need one to three solid target rectangles'
    marker_colors = colors - set(tiles)
    assert len(marker_colors) == 1, 'one temporary guide color'
    marker = marker_colors.pop()
    external = set(map(tuple,np.argwhere(grid != bg))) - tile_cells
    fragments = {c:[] for c in tiles}
    for part in _components(external,diagonal=True):
        payload = {int(grid[y,x]) for y,x in part} - {marker}
        assert len(payload) == 1
        color = payload.pop()
        assert color in tiles
        guide = {p for p in part if int(grid[p]) == marker}
        fg = part - guide
        assert fg and 2 <= len(guide) <= 9
        y0,x0,y1,x1 = _bbox(guide)
        assert (y1-y0 == 1) != (x1-x0 == 1), 'non-singleton straight guide'
        axis = 0 if y1-y0 == 1 else 1
        origin = (y0,x0) if axis == 0 else (x0,y0)
        bar = {(p[axis]-origin[0],p[1-axis]-origin[1]) for p in guide}
        assert bar == {(0,i) for i in range(len(guide))}
        relative = {(p[axis]-origin[0],p[1-axis]-origin[1]) for p in fg}
        negative = any((-1,i) in relative for i in range(len(guide)))
        positive = any((1,i) in relative for i in range(len(guide)))
        assert negative != positive, 'backing determines mating side'
        fragments[color].append((negative,relative,axis,len(guide),_bbox(part)))
    records = {}
    for color,parts in fragments.items():
        assert len(parts) == 2 and parts[0][0] != parts[1][0]
        assert parts[0][2:4] == parts[1][2:4]
        parts.sort(key=lambda p:not p[0])
        a,b = parts
        shifted = {(y-1,x) for y,x in b[1]}
        ink,origin = _mask(a[1] | shifted)
        hole = _cavities(ink)
        guide = {(y,x) for y in (-1,0) for x in range(a[3])}
        # Keep the same stamp register when testing marker transparency.
        guide_ink = ink.copy()
        for y,x in guide:
            yy,xx = y-origin[0],x-origin[1]
            if 0 <= yy < ink.shape[0] and 0 <= xx < ink.shape[1]:
                guide_ink[yy,xx] = True
        if a[2]:
            ink,hole,guide_ink = ink.T.copy(),hole.T.copy(),guide_ink.T.copy()
        y0,x0,y1,x1 = tiles[color]
        assert ink.shape == (y1-y0-2,x1-x0-2), 'assembled bbox must fit the visible tile interior exactly'
        records[color] = {'ink':ink,'holes':hole,'guide_ink':guide_ink,
            'parts':parts,'axis':a[2],'length':a[3], 'origin':origin,
            'collision':len(a[1] & shifted)}
    y0 = min(b[0] for b in tiles.values())-1
    x0 = min(b[1] for b in tiles.values())-1
    y1 = max(b[2] for b in tiles.values())+1
    x1 = max(b[3] for b in tiles.values())+1
    assert y0 >= 0 and x0 >= 0 and y1 <= grid.shape[0] and x1 <= grid.shape[1]
    return {'background':bg,'marker':marker,'tiles':tiles,'records':records,
            'crop':(y0,x0,y1,x1)}


def _render(info, masks=None):
    y0,x0,y1,x1 = info['crop']
    bg = info['background']
    out = np.full((y1-y0,x1-x0),bg,dtype=np.int8)
    for color,(ty,tx,by,bx) in info['tiles'].items():
        out[ty-y0:by-y0,tx-x0:bx-x0] = color
        mask = info['records'][color]['holes'] if masks is None else masks[color]
        assert mask.shape == (by-ty-2,bx-tx-2)
        out[ty-y0+1:by-y0-1,tx-x0+1:bx-x0-1][mask] = bg
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Register by bars, flood exterior air, punch enclosed air by tile color."""
    return _render(_decode(input_grid))


def _stamp(rng,h,w,mode,witness=False,slant=None):
    """Make contour geometry first; split it into two open registration pieces."""
    for attempt in range(160):
        split = int(rng.integers(2,h-1))
        lean = int(rng.choice((-1,0,1))) if slant is None else slant
        corners = rng.integers(0,2,size=4) if rng.random() < .7 else np.zeros(4,dtype=int)
        region = np.zeros((h,w),dtype=bool)
        for y in range(h):
            offset = (int(y >= split) if lean == 1 else int(y < split) if lean == -1 else 0)
            lo = offset + max(0,int(corners[0])-y,int(corners[1])-(h-1-y))
            hi = w-1-(int(lean != 0)-offset)-max(0,int(corners[2])-y,int(corners[3])-(h-1-y))
            region[y,lo:hi+1] = True
        eroded = region.copy()
        padded = np.pad(region,1)
        for dy,dx in ((0,1),(2,1),(1,0),(1,2)):
            eroded &= padded[dy:dy+h,dx:dx+w]
        ink = region & ~eroded
        possible_walls = [x for x in range(2,w-2) if region[:,x].all()]
        nw = int(rng.integers(0,min(2,len(possible_walls))+1))
        if mode in ('divided','mixed'):
            if not possible_walls:
                continue
            nw = int(rng.integers(1,min(2,len(possible_walls))+1))
        if mode == 'leaky':
            nw = 0
        walls = list(map(int,rng.choice(possible_walls,nw,replace=False))) if nw else []
        for x in walls:
            ink[:,x] |= region[:,x]
        # Small attached teeth vary the negative shape without adding loose ink.
        for _ in range(int(rng.integers(0,4))):
            options = [(y,x) for y in range(1,h-1) for x in range(1,w-1)
                       if region[y,x] and not ink[y,x] and
                       any(ink[y+dy,x+dx] for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)))]
            if options:
                y,x = options[int(rng.integers(len(options)))]
                ink[y,x] = True
        sealed_air = _cavities(ink)
        if mode in ('leaky','mixed'):
            # Open an exterior side at the seam, trimming only an arm's tip.
            left = bool(rng.integers(2))
            order = range(w) if left else range(w-1,-1,-1)
            entered = False
            for x in order:
                if region[split-1,x]:
                    entered = True
                    if not ink[split-1,x]:
                        break
                    ink[split-1,x] = False
                elif entered:
                    break
        holes = _cavities(ink)
        count = _regions(holes)
        if mode in ('leaky','mixed') and not (sealed_air & ~holes).any():
            continue
        if (mode == 'leaky' and count != 0 or mode == 'closed' and count < 1
            or mode == 'divided' and count < 2 or mode == 'mixed' and count < 1):
            continue
        if not all((ink[0].any(),ink[-1].any(),ink[:,0].any(),ink[:,-1].any())):
            continue
        if witness and len({_key(m) for m in _transforms(holes)}) != 8:
            continue
        length = int(rng.integers(2,min(w,4)+1))
        starts = [x for x in range(w-length+1)
                  if ink[split-1,x:x+length].any() and ink[split,x:x+length].any()]
        if not starts:
            continue
        start = starts[int(rng.integers(len(starts)))]
        groups = []
        for top in (True,False):
            fg = {(int(y),int(x)) for y,x in np.argwhere(ink)
                  if (int(y) < split) == top}
            guide = {(split if top else split-1,x) for x in range(start,start+length)}
            if len(_components(fg | guide,diagonal=True)) != 1:
                break
            own,_ = _mask(fg)
            if _cavities(own).any():
                break
            y0,x0,y1,x1 = _bbox(fg | guide)
            groups.append({'fg':{(y-y0,x-x0) for y,x in fg},
                           'guide':{(y-y0,x-x0) for y,x in guide},
                           'shape':(y1-y0,x1-x0)})
        if len(groups) == 2:
            return groups,ink,holes
    raise AssertionError('bounded contour construction exhausted')


def _pack(rng,objects):
    """At most twelve random attempts, then a provably fitting shelf layout."""
    order = sorted(range(len(objects)),key=lambda i:-np.prod(objects[i]['shape']))
    for _ in range(12):
        occupied = np.zeros((30,30),dtype=bool)
        occupied[0] = occupied[-1] = True
        occupied[:,0] = occupied[:,-1] = True
        placements = {}
        for i in order:
            h,w = objects[i]['shape']
            sums = np.pad(occupied.astype(int),((1,0),(1,0))).cumsum(0).cumsum(1)
            totals = sums[h:,w:]-sums[:-h,w:]-sums[h:,:-w]+sums[:-h,:-w]
            options = np.argwhere(totals == 0)
            if not len(options):
                break
            y,x = map(int,options[int(rng.integers(len(options)))])
            placements[i] = y,x
            occupied[max(0,y-1):y+h+1,max(0,x-1):x+w+1] = True
        if len(placements) == len(objects):
            return placements
    order = sorted(range(len(objects)),key=lambda i:-objects[i]['shape'][0])
    placements,y,x,row_height = {},1,1,0
    for i in order:
        h,w = objects[i]['shape']
        if x+w > 29:
            y += row_height+1
            x,row_height = 1,0
        assert y+h <= 29, 'shelf packing bound failed'
        placements[i] = y,x
        x += w+1
        row_height = max(row_height,h)
    return placements


def _make_pair(rng,index,training,params):
    first = training and index < 2
    if first:
        modes = ('closed','leaky') if index == 0 else ('divided','mixed')
        n = 2
    else:
        # Hold out scale or assembly count, never a hidden solving convention.
        n = 3 if not training and params['regime'] == 'recombine' else int(rng.integers(1,3))
        modes = tuple(str(rng.choice(('closed','divided','mixed','leaky'))) for _ in range(n))
        if all(m == 'leaky' for m in modes):
            modes = ('closed',) + modes[1:]
    if params['regime'] == 'scale':
        sizes = list(range(5,8)) if training else [8,9]
    else:
        sizes = list(range(5,9))
    if n == 3:
        sizes = [5,6]
    shared = (int(rng.choice(sizes)),int(rng.choice(sizes)))
    # A wide enough first witness can show asymmetry and shifted bounding boxes.
    if first:
        shared = (int(rng.integers(5,8)),int(rng.integers(6,8)))
    palette = tuple(map(int,rng.choice(10,5,replace=False))) if params['mixed_palette'] else params['palette']
    bg,marker = palette[:2]
    colors = palette[2:2+n]
    objects, expected, stamps = [],[],[]
    for j,(mode,color) in enumerate(zip(modes,colors)):
        h,w = shared if first or rng.random() < .55 else (int(rng.choice(sizes)),int(rng.choice(sizes)))
        groups,ink,holes = _stamp(rng,h,w,mode,witness=first and j == 0,
                                slant=int(rng.choice((-1,1))) if first and j == 0 else None)
        objects.append({'shape':(h+2,w+2),'color':color,'tile':True})
        expected.append((len(objects)-1,color,holes))
        for group in groups:
            objects.append(dict(group,color=color,tile=False))
        stamps.append(holes)
    if first and np.array_equal(stamps[0],stamps[1]):
        return None
    placements = _pack(rng,objects)
    grid = np.full((30,30),bg,dtype=np.int8)
    for i,obj in enumerate(objects):
        y,x = placements[i]
        h,w = obj['shape']
        if obj['tile']:
            grid[y:y+h,x:x+w] = obj['color']
        else:
            for yy,xx in obj['fg']:
                grid[y+yy,x+xx] = obj['color']
            for yy,xx in obj['guide']:
                grid[y+yy,x+xx] = marker
    ty = min(placements[i][0] for i,c,m in expected)-1
    tx = min(placements[i][1] for i,c,m in expected)-1
    by = max(placements[i][0]+objects[i]['shape'][0] for i,c,m in expected)+1
    bx = max(placements[i][1]+objects[i]['shape'][1] for i,c,m in expected)+1
    output = np.full((by-ty,bx-tx),bg,dtype=np.int8)
    for i,color,holes in expected:
        y,x = placements[i]
        h,w = objects[i]['shape']
        output[y-ty:y-ty+h,x-tx:x-tx+w] = color
        output[y-ty+1:y-ty+h-1,x-tx+1:x-tx+w-1][holes] = bg
    yy,xx = np.nonzero(grid != bg)
    grid = grid[yy.min()-1:yy.max()+2,xx.min()-1:xx.max()+2]
    pads = []
    for dim in grid.shape:
        extra = int(rng.integers(min(3,30-dim)+1))
        a = int(rng.integers(extra+1))
        pads.append((a,extra-a))
    grid = np.pad(grid,pads,constant_values=bg)
    rotation = (params['rotation']+index%2) % 4 if first else int(rng.integers(4))
    return {'input':np.rot90(grid,rotation).copy(),'output':np.rot90(output,rotation).copy()}


def _count(name,value,lo,hi):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int,num_train: int = 4,num_test: int = 1) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    # None is an optional convenience for the supplied inspection utility.
    nt = None if num_train is None else _count('num_train',num_train,2,10)
    ne = None if num_test is None else _count('num_test',num_test,1,4)
    if nt is None:
        nt = int(rng.integers((ne or 1)+1,11))
    if ne is None:
        ne = int(rng.integers(1,min(4,nt-1)+1))
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    params = {'palette':tuple(map(int,rng.choice(10,5,replace=False))),
              'mixed_palette':bool(rng.integers(2)), 'rotation':int(rng.integers(4)),
              'regime':str(rng.choice(('scale','recombine')))}
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',nt),('test',ne)):
        for i in range(count):
            for _ in range(32):
                pair = _make_pair(rng,i,split == 'train',params)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded distinct-pair construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_masks(info,kind):
    result = {}
    for c,r in info['records'].items():
        ink,hole = r['ink'],r['holes']
        if kind == 'raw_ink': value = ink
        elif kind == 'filled_ink': value = ink | hole
        elif kind == 'whole_register': value = np.ones_like(ink)
        elif kind == 'no_assembly': value = np.zeros_like(ink)
        elif kind == 'exterior': value = ~ink & ~hole
        elif kind == 'opaque_guides': value = _cavities(r['guide_ink'])
        else: raise ValueError(kind)
        result[c] = value
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen,infos = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(_render(info),out), 'incorrect cavity punch or target crop'
            assert not np.array_equal(inp,out)
            bg = info['background']
            assert all(np.all(e == bg) for e in (out[0],out[-1],out[:,0],out[:,-1]))
            assert info['marker'] not in set(map(int,np.unique(out)))
            for c,r in info['records'].items():
                y0,x0,y1,x1 = info['tiles'][c]
                assert np.sum(out == c) == (y1-y0)*(x1-x0)-int(r['holes'].sum())
                # Generated pieces are open separately; registration creates all cavities.
                assert all(not _cavities(_mask(p[1])[0]).any() for p in r['parts'])
            if split == 'train':
                infos.append(info)
    train_records = [r for info in infos for r in info['records'].values()]
    assert {r['axis'] for r in train_records} == {0,1}
    assert any(not r['holes'].any() for r in train_records), 'teach an exterior leak'
    assert any(_regions(r['holes']) >= 2 for r in train_records), 'teach multiple pockets'
    assert any(len({_key(m) for m in _transforms(r['holes'])}) == 8 for r in train_records)
    assert any(_bbox(r['parts'][0][1])[1] != _bbox(r['parts'][1][1])[1]
               for r in train_records), 'bounding-box alignment must be insufficient'
    # Reject concrete simplifications using training pixels only.
    for kind in ('raw_ink','filled_ink','whole_register','no_assembly','exterior','opaque_guides'):
        assert any(not np.array_equal(_render(i),_render(i,_alternative_masks(i,kind))) for i in infos), kind
    assert any(len(i['records']) >= 2 and len({_key(r['holes']) for r in i['records'].values()}) > 1
               and len({r['holes'].shape for r in i['records'].values()}) == 1 for i in infos), 'teach color equality'


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train','test'):
        for pair in puzzle[split]:
            for role in ('input','output'):
                grid = np.asarray(pair[role],dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming survey, all counts, and genuine semantic mutations."""
    structures,shapes,outputs,palettes,axes,lengths,counts,pockets = (set() for _ in range(8))
    signatures = set()
    blank_stamps = positive_stamps = shifted = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ('train','test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                assert np.array_equal(solve(pair['input'].copy()),pair['output'])
                shapes.add(pair['input'].shape)
                outputs.add(pair['output'].shape)
                palettes.add((info['background'],info['marker'],tuple(sorted(info['tiles']))))
                counts.add(len(info['tiles']))
                for r in info['records'].values():
                    structures.add(_key(r['ink']))
                    axes.add(r['axis']); lengths.add(r['length'])
                    pockets.add(_regions(r['holes']))
                    blank_stamps += not r['holes'].any()
                    positive_stamps += bool(r['holes'].any())
                    # A bbox-only join loses the tangent offset when the halves lean.
                    a,b = r['parts']
                    shifted += _bbox(a[1])[1] != _bbox(b[1])[1]
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
    assert len(signatures) == 42 and len(structures) >= 100
    assert len(shapes) >= 15 and len(outputs) >= 30 and len(palettes) >= 30
    assert axes == {0,1} and lengths >= {2,3,4} and counts == {1,2,3}
    assert {0,1,2} <= pockets and min(blank_stamps,positive_stamps,shifted) > 40
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(12345,nt,ne)
            assert (len(p['train']),len(p['test'])) == (nt,ne)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(4,False),(4,1.5)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            continue
        raise AssertionError('accepted illegal counts')
    p = generate_puzzle(0,2,1)
    info = _decode(p['train'][0]['input'])
    for kind in ('raw_ink','filled_ink','whole_register','no_assembly','exterior','opaque_guides'):
        wrong = _render(info,_alternative_masks(info,kind))
        if np.array_equal(wrong,p['train'][0]['output']):
            continue
        changed = {'train':[dict(q) for q in p['train']],'test':p['test']}
        changed['train'][0]['output'] = wrong
        try:
            validate_generated_puzzle(changed)
        except AssertionError:
            pass
        else:
            raise AssertionError('accepted incorrect semantic mutation')
    # Analytic topology checks: closure, a leak, two chambers, diagonal walls.
    ring = np.ones((5,6),dtype=bool); ring[1:-1,1:-1] = False
    assert _cavities(ring).sum() == 12
    ring[2,0] = False
    assert not _cavities(ring).any()
    ring[2,0] = True; ring[:,3] = True
    assert _regions(_cavities(ring)) == 2
    diamond = np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=bool)
    assert _cavities(diamond).sum() == 1


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, all legal explicit counts)')
