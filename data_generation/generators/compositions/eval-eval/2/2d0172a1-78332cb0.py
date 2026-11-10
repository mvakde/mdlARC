"""Socket-grown enclosures: geometric plugs assemble a canonical containment tree.

No parent code, fixtures, saved examples, or random state is used by solve().
Only NumPy and the Python standard library are required. No files are written.
"""
from __future__ import annotations

from itertools import permutations
import hashlib
import numpy as np

HOST_SEEDS = (0,166790452,620698813,59831158,792609213,762696500,215657867,229342091,705328812,445514147,128790324,266766600,191740872,841674496,823012931,337341332,162224307,120427201,627136855,639632934,417922516,676334436,342920133,118455031,854516522,666789261,961278609,609574321,826025038,634845083,542003474,643799681,866950031,608258326,344881147,977771101,455180411,192352932,92531883,723482206,125341643,539146440)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid dimensions must be 1..30')
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), 'invalid colors or dtype')
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        stack, cells = [(y, x)], []
        seen[y, x] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        coords = np.asarray(cells, dtype=int)
        lo, hi = coords.min(0), coords.max(0)
        local = np.zeros(tuple(hi-lo+1), bool)
        local[coords[:,0]-lo[0], coords[:,1]-lo[1]] = True
        result.append(dict(cells=coords, lo=lo, hi=hi, mask=local))
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, bool)
    outside[0,0] = True
    stack = [(0,0)]
    h, w = padded.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0 <= rr < h and 0 <= cc < w and not padded[rr,cc] and not outside[rr,cc]:
                outside[rr,cc] = True
                stack.append((rr,cc))
    return (~padded & ~outside)[1:-1,1:-1]


def _comb(signature):
    mask = np.zeros((3,5), bool)
    mask[0,:] = True
    mask[1,list(signature)] = True
    mask[2,2] = True
    return mask


def _join(parts, axis, gap):
    """Common top edge for a row; common left edge for a column."""
    shape = [max(p.shape[d] for p in parts) for d in (0,1)]
    shape[axis] = sum(p.shape[axis] for p in parts) + gap*(len(parts)-1)
    result = np.zeros(shape, bool)
    offset = 0
    for part in parts:
        r, c = (offset,0) if axis == 0 else (0,offset)
        result[r:r+part.shape[0], c:c+part.shape[1]] = part
        offset += part.shape[axis]+gap
    return result


def _frame(content):
    result = np.pad(content, 2)
    result[0,:] = result[-1,:] = True
    result[:,0] = result[:,-1] = True
    return result


def _layout(items):
    """Recover actual local order/alignment, never compare center spans."""
    if len(items) == 1:
        return 1, items
    candidates = []
    for axis in (0,1):
        cross = 1-axis
        ordered = sorted(items, key=lambda item: int(item['lo'][axis]))
        if len({int(item['lo'][cross]) for item in items}) != 1:
            continue
        if all(int(b['lo'][axis]-a['hi'][axis]) >= 2 for a,b in zip(ordered,ordered[1:])):
            candidates.append((axis,ordered))
    _require(len(candidates) == 1, 'items need one separated, top/left-aligned row or column')
    return candidates[0]


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0,0])
    _require(np.all(grid[0] == bg) and np.all(grid[-1] == bg) and np.all(grid[:,0] == bg) and np.all(grid[:,-1] == bg), 'one background border is required')
    colors = [int(c) for c in np.unique(grid) if c != bg]
    _require(len(colors) == 2, 'background, enclosure and connector are three distinct colors')
    candidates = []
    for wall in colors:
        records = _components(grid == wall)
        loops, blobs = [], []
        for record in records:
            hole = _holes(record['mask'])
            if hole.any():
                record['hole'] = hole
                loops.append(record)
            else:
                blobs.append(record)
        if 2 <= len(loops) <= 4 and blobs:
            candidates.append((wall,loops,blobs))
    _require(len(candidates) == 1, 'enclosure role must be unambiguous')
    wall, loops, blobs = candidates[0]
    connector = next(c for c in colors if c != wall)
    for loop in loops:
        _require(len(_components(loop['hole'])) == 1, 'exactly one hole per shell')
        loop.update(items=[], incoming=None)
    # All shells are separate top-level fragments. Their interiors define scope.
    def owners(cells):
        answer = []
        for i, loop in enumerate(loops):
            local = cells-loop['lo']
            valid = np.all((local >= 0) & (local < loop['hole'].shape), axis=1)
            inside = np.zeros(len(cells), bool)
            inside[valid] = loop['hole'][local[valid,0],local[valid,1]]
            _require(inside.all() or not inside.any(), 'object crosses an enclosure')
            if inside.all():
                answer.append(i)
        return answer
    for loop in loops:
        _require(not owners(loop['cells']), 'input shells must not nest')
    for blob in blobs:
        found = owners(blob['cells'])
        _require(len(found) == 1, 'each payload blob belongs to one shell')
        _require(blob['mask'].sum() >= .5*blob['mask'].size, 'payloads must be compact')
        blob.update(kind='blob')
        loops[found[0]]['items'].append(blob)
    for port in _components(grid == connector):
        mask = port['mask']
        _require(mask.shape == (3,5), 'connector bounding box must be 3 by 5')
        signature = tuple(map(int,np.flatnonzero(mask[1])))
        _require(2 in signature and np.array_equal(mask, _comb(signature)), 'malformed comb')
        found = owners(port['cells'])
        if found:
            _require(len(found) == 1, 'socket scope ambiguous')
            port.update(kind='socket', signature=signature)
            loops[found[0]]['items'].append(port)
        else:
            # Its bottom stem is immediately above a wall cell of its owner.
            r, c = port['lo'] + (3,2)
            _require(r < grid.shape[0], 'external tab is detached')
            attached = [i for i,loop in enumerate(loops) if np.any(np.all(loop['cells'] == (r,c),axis=1))]
            _require(len(attached) == 1, 'external tab must touch precisely one shell')
            owner = loops[attached[0]]
            _require(owner['incoming'] is None, 'one external tab per nonroot shell')
            owner['incoming'] = signature
    roots = [i for i,loop in enumerate(loops) if loop['incoming'] is None]
    _require(len(roots) == 1, 'exactly one shell lacks an external tab')
    targets = {}
    for i,loop in enumerate(loops):
        sig = loop['incoming']
        if sig is not None:
            _require(sig not in targets, 'duplicate incoming signatures are ambiguous')
            targets[sig] = i
    used = []
    for loop in loops:
        _require(any(item['kind'] == 'blob' for item in loop['items']), 'each shell has a payload')
        loop['axis'], loop['items'] = _layout(loop['items'])
        for item in loop['items']:
            if item['kind'] == 'socket':
                _require(item['signature'] in targets, 'socket has no matching tab')
                item['child'] = targets[item['signature']]
                used.append(item['child'])
    _require(sorted(used) == sorted(targets.values()), 'use every nonroot shell exactly once')
    seen, active = set(), set()
    def visit(i):
        _require(i not in active and i not in seen, 'cyclic or repeated graft')
        active.add(i)
        for item in loops[i]['items']:
            if item['kind'] == 'socket':
                visit(item['child'])
        active.remove(i)
        seen.add(i)
    visit(roots[0])
    _require(len(seen) == len(loops), 'disconnected assembly')
    return dict(background=bg,wall=wall,connector=connector,modules=loops,root=roots[0])


def _draw_decoded(info):
    def draw(index):
        loop = info['modules'][index]
        parts = [np.ones((1,1),bool) if item['kind'] == 'blob' else draw(item['child']) for item in loop['items']]
        return _frame(_join(parts,loop['axis'],1))
    return np.where(draw(info['root']),info['wall'],info['background']).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the matching tree and normalize it using only visible input."""
    return _draw_decoded(_decode(input_grid))


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _plan(rng, n, form=None):
    # Ordered local contents, including the spatial place of each child socket.
    parents = [-1]
    if form == 'chain':
        parents += list(range(n-1))
    elif form == 'fork':
        parents += [0,0]
    else:
        for i in range(1,n):
            available = [j for j in range(i) if parents.count(j) < 2]
            parents.append(_choice(rng,available))
    children = [[j for j in range(1,n) if parents[j] == i] for i in range(n)]
    pool = [tuple(j for j in range(5) if bits & (1 << j)) for bits in range(1,32) if bits & 4]
    # Equal-size signatures with different occupied positions defeat port-count matching.
    size = int(rng.choice([2,3,4]))
    same_size = [s for s in pool if len(s) == size]
    chosen = list(map(int,rng.choice(len(same_size),2,replace=False)))
    links = [same_size[j] for j in chosen]
    remaining = [s for s in pool if s not in links]
    rng.shuffle(remaining)
    links += remaining[:max(0,n-3)]
    rng.shuffle(links)
    incoming = [None] + links[:n-1]
    modules = []
    for i in range(n):
        count = int(rng.integers(1,3))
        # The fork's leaves are visibly different, so exchanging their keys changes the answer.
        if form == 'fork' and i in (1,2):
            count = i
        items = [('blob',None)]*count + [('socket',child) for child in children[i]]
        rng.shuffle(items)
        axis = int(rng.integers(2))
        if form == 'chain' and i < 2:
            axis = i  # witness includes both axes, shuffled later among demonstrations
        modules.append(dict(incoming=incoming[i],items=items,axis=axis))
    if form == 'fork':
        # Put the two different subtrees on opposite sides of the payload.
        modules[0]['items'] = [('socket',1),('blob',None),('socket',2)]
        if rng.random() < .5:
            modules[0]['items'].reverse()
    return modules


def _expected(plan):
    def draw(i):
        parts = [np.ones((1,1),bool) if kind == 'blob' else draw(child) for kind,child in plan[i]['items']]
        return _frame(_join(parts,plan[i]['axis'],1))
    return draw(0)


def _blob(rng):
    h, w = int(rng.integers(1,3)), int(rng.integers(1,3))
    mask = np.ones((h,w),bool)
    if h == w == 2 and rng.random() < .5:
        mask[int(rng.integers(2)),int(rng.integers(2))] = False
    return mask


def _outline(rng,h,w):
    # Independent stepped top and bottom; sides remain closed. The central
    # top cell stays on row zero for an unambiguous external tab attachment.
    top, bottom = np.zeros(w,int), np.full(w,h-1,int)
    for edge in (top,bottom):
        for side in (0,1):
            count = int(rng.integers(1,min(3,w//2)))
            if rng.random() < .8:
                ids = range(count) if side == 0 else range(w-count,w)
                for c in ids:
                    edge[c] += 1 if edge is top else -1
    mask = np.zeros((h,w),bool)
    for profile in (top,bottom):
        for c in range(w):
            mask[profile[c],c] = True
            if c:
                mask[min(profile[c-1],profile[c]):max(profile[c-1],profile[c])+1,c] = True
    mask[top[0]:bottom[0]+1,0] = True
    mask[top[-1]:bottom[-1]+1,-1] = True
    return mask


def _metric_module(rng,plan,i,palette):
    bg, wall, connector = palette
    entry = plan[i]
    parts = []
    for kind,child in entry['items']:
        parts.append(_blob(rng) if kind == 'blob' else _comb(plan[child]['incoming']))
    axis, gap = entry['axis'], int(rng.integers(1,4))
    # Input gap is independently sampled; outputs always use one blank cell.
    shape = [max(p.shape[d] for p in parts) for d in (0,1)]
    shape[axis] = sum(p.shape[axis] for p in parts)+gap*(len(parts)-1)
    # Top/bottom corridors allow actual silhouette changes without touching items.
    h, w = shape[0]+6, max(7,shape[1]+4)
    extra = 3 if entry['incoming'] is not None else 0
    grid = np.full((h+extra,w),bg,dtype=np.int8)
    outline = _outline(rng,h,w)
    grid[extra:][outline] = wall
    offset = 0
    for (kind,child),part in zip(entry['items'],parts):
        r, c = (offset,0) if axis == 0 else (0,offset)
        r, c = r+3+extra,c+2
        view = grid[r:r+part.shape[0],c:c+part.shape[1]]
        view[part] = wall if kind == 'blob' else connector
        offset += part.shape[axis]+gap
    if extra:
        x = w//2-2
        grid[:3,x:x+5][_comb(entry['incoming'])] = connector
    return grid


def _pack(rng,tiles,bg):
    # Exhaustive shelf partitions for <=4 rectangles; no retry payload is saved.
    options = []
    for order in permutations(range(len(tiles))):
        for cuts in range(1 << (len(tiles)-1)):
            rows, current = [], []
            for pos,i in enumerate(order):
                current.append(i)
                if pos == len(tiles)-1 or cuts & (1 << pos):
                    rows.append(current)
                    current = []
            widths = [sum(tiles[i].shape[1] for i in row)+len(row)-1 for row in rows]
            heights = [max(tiles[i].shape[0] for i in row) for row in rows]
            if max(widths) <= 28 and sum(heights)+len(rows)-1 <= 28:
                options.append((rows,widths,heights))
    if not options:
        return None
    rows,widths,heights = _choice(rng,options)
    height = sum(heights)+len(rows)-1+2
    width = max(widths)+2
    # Independent extra exterior margins, still bounded by 30.
    height = int(rng.integers(height,31))
    width = int(rng.integers(width,31))
    grid = np.full((height,width),bg,dtype=np.int8)
    spare_y = height-(sum(heights)+len(rows)-1)-2
    y = 1+int(rng.integers(spare_y+1))
    for row,rw,rh in zip(rows,widths,heights):
        x = 1+int(rng.integers(width-rw-1))
        for i in row:
            tile = tiles[i]
            offset_y = int(rng.integers(rh-tile.shape[0]+1))
            grid[y+offset_y:y+offset_y+tile.shape[0],x:x+tile.shape[1]] = tile
            x += tile.shape[1]+1
        y += rh+1
    return grid


def _make_pair(rng,n,form=None,palette=None):
    # Rejection concerns only fit or actual visible recoverability, never labels.
    for attempt in range(160):
        plan = _plan(rng,n,form)
        colors = tuple(map(int,rng.choice(10,3,replace=False))) if palette is None else palette
        tiles = [_metric_module(rng,plan,i,colors) for i in range(n)]
        inp = _pack(rng,tiles,colors[0])
        expected = _expected(plan)
        if inp is None or max(expected.shape) > 30:
            continue
        try:
            info = _decode(inp)
        except ValueError:
            continue
        out = np.where(expected,colors[1],colors[0]).astype(np.int8)
        _require(np.array_equal(_draw_decoded(info),out),'construction and visible decoding disagree')
        # Every example shows actual blob compression, not merely recoloring.
        if not any(item['kind'] == 'blob' and len(item['cells']) > 1 for loop in info['modules'] for item in loop['items']):
            continue
        return dict(input=inp,output=out)
    raise RuntimeError('bounded fit search exhausted')


def _counts(num_train,num_test):
    for name,value,lo,hi in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= value <= hi:
            raise ValueError(name+' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train),int(num_test)


def generate_puzzle(seed: int,num_train: int=4,num_test: int=1) -> dict:
    num_train,num_test = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    stable_palette = tuple(map(int,rng.choice(10,3,replace=False))) if rng.random() < .25 else None
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',num_train),('test',num_test)):
        for i in range(count):
            form = ('chain','fork')[i] if split == 'train' and i < 2 else None
            n = 4 if split == 'test' else (3 if form else int(rng.integers(2,4)))
            for retry in range(32):
                pair = _make_pair(rng,n,form,stable_palette)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-input search exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def _topology(info):
    def visit(i):
        return tuple(visit(item['child']) for item in info['modules'][i]['items'] if item['kind'] == 'socket')
    return visit(info['root'])


def _semantic_tree(info):
    def visit(i):
        return ('C',tuple(('B',) if item['kind'] == 'blob' else visit(item['child'])
                          for item in info['modules'][i]['items']))
    return visit(info['root'])


def _output_invariants(out,bg,wall):
    _require(set(map(int,np.unique(out))) == {bg,wall}, 'outputs retain background and wall only')
    loops,points = [],[]
    for record in _components(out == wall):
        hole = _holes(record['mask'])
        if hole.any():
            mask = record['mask']
            expected = np.zeros(mask.shape,bool)
            expected[0,:] = expected[-1,:] = True
            expected[:,0] = expected[:,-1] = True
            _require(np.array_equal(mask,expected),'canonical frames must be exact rectangles')
            record['hole'] = hole
            loops.append(record)
        else:
            _require(len(record['cells']) == 1,'each payload becomes precisely one point')
            points.append(record)
    objects = loops+points
    parents = []
    for child in objects:
        containers = [i for i,loop in enumerate(loops)
                      if np.all(child['lo'] > loop['lo']) and np.all(child['hi'] < loop['hi'])]
        parents.append(min(containers,key=lambda i: loops[i]['mask'].size) if containers else None)
    roots = [i for i,p in enumerate(parents) if p is None]
    _require(len(roots) == 1 and roots[0] < len(loops),'one actual enclosing root frame')
    for i,loop in enumerate(loops):
        _require(len(_components(loop['hole'])) == 1,'canonical enclosure has one hole')
        children = [objects[j] for j,p in enumerate(parents) if p == i]
        _require(bool(children),'no empty frame')
        _require(np.array_equal(np.min([c['lo'] for c in children],axis=0),loop['lo']+2)
                 and np.array_equal(np.max([c['hi'] for c in children],axis=0),loop['hi']-2),
                 'one-cell interior moat and minimum frame dimensions')
        axis,ordered = _layout(children)
        _require(all(int(b['lo'][axis]-a['hi'][axis]) == 2 for a,b in zip(ordered,ordered[1:])),
                 'one-cell sibling gaps')
        loop['ordered'] = ordered
    def visit(obj):
        if 'ordered' not in obj:
            return ('B',)
        return ('C',tuple(visit(child) for child in obj['ordered']))
    _require(np.all(out[0] == wall) and np.all(out[-1] == wall) and np.all(out[:,0] == wall) and np.all(out[:,-1] == wall), 'tight crop to root frame')
    return loops,points,visit(objects[roots[0]])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'},'puzzle schema')
    _require(all(isinstance(puzzle[s],list) for s in puzzle),'split lists required')
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen = set()
    training = []
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'},'pair schema')
            inp,out = _grid(pair['input']),_grid(pair['output'])
            _require(_key(inp) not in seen,'duplicate input')
            seen.add(_key(inp))
            info = _decode(inp)
            _require(np.array_equal(out,_draw_decoded(info)),'wrong assembled normalized answer')
            loops,points,tree = _output_invariants(out,info['background'],info['wall'])
            _require(len(loops) == len(info['modules']),'every shell becomes one frame')
            _require(tree == _semantic_tree(info),'actual output containment and payload order must match grafts')
            blobs = [item for loop in info['modules'] for item in loop['items'] if item['kind'] == 'blob']
            _require(len(points) == len(blobs),'payload count conservation')
            _require(any(len(item['cells']) > 1 for item in blobs),'demonstrate metric compression')
            if split == 'train':
                training.append(info)
    # Teaching obligations, including at the 2/1 minimum.
    def depth(tree):
        return 1+max((depth(child) for child in tree),default=0)
    _require(any(depth(_topology(info)) >= 3 for info in training),'train nesting witness')
    _require(any(any(sum(item['kind'] == 'socket' for item in loop['items']) == 2 for loop in info['modules']) for info in training),'train branching witness')
    _require({loop['axis'] for info in training for loop in info['modules'] if len(loop['items']) > 1} == {0,1},'teach both local layout axes')


def _features(info):
    modules = info['modules']
    signatures = [loop['incoming'] for loop in modules if loop['incoming'] is not None]
    def depth(i):
        return 1+max((depth(item['child']) for item in modules[i]['items'] if item['kind'] == 'socket'),default=0)
    return dict(modules=len(modules),depth=depth(info['root']),topology=repr(_topology(info)),
                branching=any(sum(item['kind'] == 'socket' for item in loop['items']) == 2 for loop in modules),
                axes=tuple(sorted({loop['axis'] for loop in modules if len(loop['items']) > 1})),
                palette=(info['background'],info['wall'],info['connector']),
                signatures=tuple(signatures),blob_areas=tuple(len(item['cells']) for loop in modules for item in loop['items'] if item['kind'] == 'blob'),
                nonrect=sum(not(loop['mask'][0].all() and loop['mask'][-1].all() and loop['mask'][:,0].all() and loop['mask'][:,-1].all()) for loop in modules))


def self_test():
    """42 seeds, streaming aggregates; no retained corpus, traces, or disk writes."""
    shapes,topologies,outputs,palettes,depths,signatures = set(),set(),set(),set(),set(),set()
    blobs,counts,axes = set(),set(),set()
    branching = nonrect = pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        repeated = generate_puzzle(seed)
        for split in ('train','test'):
            for pair,again in zip(puzzle[split],repeated[split]):
                assert all(np.array_equal(pair[k],again[k]) for k in pair)
                info = _decode(pair['input'])
                f = _features(info)
                pairs += 1
                shapes.add(pair['input'].shape)
                topologies.add(f['topology'])
                # Hash only reconstructible payloads; bounded aggregate size.
                normalized = (pair['output'] == info['wall']).astype(np.int8)
                outputs.add(hashlib.sha256(normalized.tobytes()+bytes(normalized.shape)).hexdigest())
                palettes.add(f['palette'])
                counts.add(f['modules']); depths.add(f['depth'])
                signatures.update(f['signatures']); blobs.update(f['blob_areas']); axes.update(f['axes'])
                branching += f['branching']; nonrect += f['nonrect']
        # Role permutations preserve every geometric decision.
        pair = puzzle['test'][0]
        mapping = np.random.default_rng(seed+91).permutation(10).astype(np.int8)
        assert np.array_equal(solve(mapping[pair['input']]),mapping[pair['output']])
    assert counts == {2,3,4} and depths >= {2,3,4} and axes == {0,1}
    assert len(topologies) >= 7 and len(outputs) >= 100 and len(shapes) >= 35
    assert len(palettes) >= 70 and len(signatures) >= 12 and blobs >= {1,2,3,4}
    assert branching >= 42 and nonrect >= 300
    # Independent metric realizations of the SAME plan must have the same answer.
    rng = np.random.default_rng(984)
    plan = _plan(rng,3,'fork')
    answer = _expected(plan)
    realized = set()
    for _ in range(16):
        tiles = [_metric_module(rng,plan,i,(0,4,8)) for i in range(3)]
        inp = _pack(rng,tiles,0)
        if inp is not None:
            actual = solve(inp)
            assert np.array_equal(actual == 4,answer)
            realized.add(hashlib.sha256(inp.tobytes()).hexdigest())
    assert len(realized) >= 4
    # Position-sensitive matching: swap the two incoming tabs of unequal fork
    # leaves. Geometry and payloads stay in place; only the grafts exchange.
    pair = generate_puzzle(0)['train']
    info = next(_decode(p['input']) for p in pair if len(_topology(_decode(p['input']))) == 2)
    fork_pair = next(p for p in pair if len(_topology(_decode(p['input']))) == 2)
    inp = fork_pair['input'].copy()
    root = info['root']
    leaves = [item['child'] for item in info['modules'][root]['items'] if item['kind'] == 'socket']
    sigs = [info['modules'][i]['incoming'] for i in leaves]
    assert len(sigs[0]) == len(sigs[1]) and sigs[0] != sigs[1]
    ports = [p for p in _components(inp == info['connector']) if tuple(map(int,np.flatnonzero(p['mask'][1]))) in sigs]
    outside = []
    for p in ports:
        r,c = p['lo']+(3,2)
        if r < inp.shape[0] and inp[r,c] == info['wall']:
            outside.append(p)
    assert len(outside) == 2
    masks = [p['mask'].copy() for p in outside]
    for p,mask in zip(outside,masks[::-1]):
        r,c = p['lo']
        inp[r:r+3,c:c+5] = info['background']
        inp[r:r+3,c:c+5][mask] = info['connector']
    assert not np.array_equal(solve(inp),fork_pair['output'])
    # A well-formed but unmatched internal comb is still an invalid socket.
    dangling = fork_pair['input'].copy()
    parsed = _decode(dangling)
    socket = next(item for loop in parsed['modules'] for item in loop['items'] if item['kind'] == 'socket')
    absent = next(tuple(j for j in range(5) if bits & (1 << j))
                  for bits in range(1,32) if bits & 4
                  and tuple(j for j in range(5) if bits & (1 << j)) not in sigs)
    r,c = socket['lo']
    dangling[r:r+3,c:c+5] = parsed['background']
    dangling[r:r+3,c:c+5][_comb(absent)] = parsed['connector']
    try:
        solve(dangling)
    except ValueError:
        pass
    else:
        raise AssertionError('dangling socket accepted')
    # Invalid counts, dangling sockets, and wrong normalized outputs are rejected.
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.0,1),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            validate_generated_puzzle(generate_puzzle(1729,a,b))
    p = generate_puzzle(43)
    broken = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0,1] = _decode(broken['test'][0]['input'])['background']
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('broken frame accepted')
    return dict(seeds=42,pairs=pairs,input_shapes=len(shapes),output_structures=len(outputs),topologies=len(topologies),palettes=len(palettes),signatures=len(signatures),branching_pairs=branching,nonrectangular_shells=nonrect,counts=sorted(counts),depths=sorted(depths),blob_areas=sorted(blobs))
