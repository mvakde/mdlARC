"""Framed, locally keyed, simultaneous one-step mark translation puzzles."""

import numpy as np

_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SOURCE_INPUTS = {
    ('233333333332', '822222222224', '822222282244', '823221222224', '842212122324', '822221223224', '822222222224', '823202288824', '882222222224', '822242422224', '822424222224', '211111111112'),
    ('15555551', '41114142', '45111112', '41112132', '41131112', '41131112', '45111152', '13333331'),
    ('3555555555553', '9333333338336', '9363333939336', '9333333393336', '9331333333336', '9333133336336', '9333313336336', '9333331336336', '9333433133336', '9339333333336', '9333433333336', '9383333335336', '3777777777773'),
    ('0222222220', '1000000003', '1020400003', '1000000103', '1000030003', '1010000003', '1000000403', '1000002003', '1003000003', '0444444440'),
    ('455555554', '344144449', '344949459', '344484449', '394244449', '334444439', '342444849', '343444479', '422222224'),
}
_SOURCE_TEST = (
    ('455555554', '344144449', '344949459', '344484449', '394244449', '334444439', '342444849', '343444479', '422222224'),
    ('455555554', '344144459', '344494949', '344484449', '349444449', '334244349', '344444849', '332444479', '422222224'),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_integer(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _checked_integer(num_train, 'num_train', 2, 10)
    test = _checked_integer(num_test, 'num_test', 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _rows(grid):
    return tuple(''.join(str(int(x)) for x in row) for row in grid)


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _components(grid, background):
    h, w = grid.shape
    seen = set()
    result = []
    for r in range(1, h-1):
        for c in range(1, w-1):
            if (r, c) in seen or int(grid[r, c]) == background:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            cells = set(stack)
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                for nr, nc in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                    if 1 <= nr < h-1 and 1 <= nc < w-1 and (nr,nc) not in seen and int(grid[nr,nc]) == color:
                        seen.add((nr,nc))
                        cells.add((nr,nc))
                        stack.append((nr,nc))
            result.append((color, frozenset(cells)))
    return result


def _interpret(grid):
    h, w = grid.shape
    _require(h >= 5 and w >= 5, 'frame is too small')
    bg = int(grid[0, 0])
    _require(all(int(grid[r,c]) == bg for r,c in ((0,w-1),(h-1,0),(h-1,w-1))), 'corner mismatch')
    sides = (int(grid[0,1]), int(grid[-1,1]), int(grid[1,0]), int(grid[1,-1]))
    _require(len(set(sides)) == 4 and bg not in sides, 'invalid side key')
    _require(bool(np.all(grid[0,1:-1] == sides[0]) and np.all(grid[-1,1:-1] == sides[1]) and np.all(grid[1:-1,0] == sides[2]) and np.all(grid[1:-1,-1] == sides[3])), 'frame not uniform')
    colors, counts = np.unique(grid[1:-1,1:-1], return_counts=True)
    color_counts = dict(zip((int(v) for v in colors), (int(n) for n in counts)))
    _require(color_counts.get(bg, 0) > max((n for v,n in color_counts.items() if v != bg), default=0), 'background not dominant')
    components = _components(grid, bg)
    original_owner = {cell: i for i, (_, cells) in enumerate(components) for cell in cells}
    destinations = []
    moved = set()
    blocked = set()
    inert = set()
    handoffs = 0
    moved_multi = 0
    blocked_count = 0
    motion_status = []
    for i, (color, cells) in enumerate(components):
        if color in sides:
            direction = sides.index(color)
            dr, dc = _DIRECTIONS[direction]
            shifted = frozenset((r+dr,c+dc) for r,c in cells)
            if any(not (1 <= r < h-1 and 1 <= c < w-1) for r,c in shifted):
                dest = cells
                blocked.add(direction)
                blocked_count += 1
                motion_status.append(False)
            else:
                dest = shifted
                moved.add(direction)
                moved_multi += len(cells) > 1
                handoffs += sum(original_owner.get(cell, i) != i for cell in dest)
                motion_status.append(True)
        else:
            direction = None
            dest = cells
            inert.add(color)
            motion_status.append(None)
        destinations.append((color, dest, direction))
    for i,(color,cells) in enumerate(components):
        if motion_status[i] is None:
            continue
        for j,(other,other_cells) in enumerate(components[i+1:],i+1):
            if color == other and motion_status[i] != motion_status[j]:
                _require(not any(max(abs(r-y),abs(c-x)) == 1 for r,c in cells for y,x in other_cells),
                         'diagonally touching marks disagree on boundary blocking')
    final = set()
    for _, cells, _ in destinations:
        _require(not (final & cells), 'conflicting final destinations')
        final.update(cells)
    output = grid.copy()
    output[1:-1,1:-1] = bg
    for color, cells, _ in destinations:
        for r,c in cells:
            output[r,c] = color
    stats = dict(sides=sides, background=bg, components=components, moved=moved,
                 blocked=blocked, blocked_count=blocked_count, inert=inert,
                 handoffs=handoffs, moved_multi=moved_multi,
                 foreground=sum(len(cells) for _,cells in components),
                 sizes=tuple(sorted(len(cells) for _,cells in components)))
    return output, stats


def _dimensions(rng, first):
    if first:
        return int(rng.integers(14, 20)), int(rng.integers(14, 20))
    band = int(rng.integers(0, 4))
    lo, hi = ((6,9),(10,15),(16,22),(23,30))[band]
    aspect = int(rng.integers(0, 4))
    if aspect == 0:
        return int(rng.integers(lo,hi+1)), int(rng.integers(lo,hi+1))
    if aspect == 1:
        return int(rng.integers(6,hi+1)), int(rng.integers(max(lo,10),31))
    if aspect == 2:
        return int(rng.integers(max(lo,10),31)), int(rng.integers(6,hi+1))
    return int(rng.integers(6,31)), int(rng.integers(6,31))


def _shape(rng, kind):
    if kind == 'single':
        return ((0,0),)
    if kind == 'bar':
        length = int(rng.integers(2,6))
        vertical = bool(rng.integers(2))
        return tuple((i,0) if vertical else (0,i) for i in range(length))
    if kind == 'bend':
        a, b = int(rng.integers(2,5)), int(rng.integers(2,5))
        cells = {(i,0) for i in range(a)} | {(a-1,j) for j in range(b)}
        if rng.integers(2):
            cells = {(c,r) for r,c in cells}
        return tuple(sorted(cells))
    if kind == 'block':
        a,b = (2,2) if rng.integers(3) else (2,3)
        return tuple((r,c) for r in range(a) for c in range(b))
    if kind == 'diagonal':
        length = int(rng.integers(2,6))
        if rng.integers(2):
            return tuple((i,i) for i in range(length))
        return tuple((i,length-1-i) for i in range(length))
    cells = {(0,0)}
    for _ in range(int(rng.integers(3,8))-1):
        for _ in range(30):
            r,c = tuple(cells)[int(rng.integers(len(cells)))]
            dr,dc = _DIRECTIONS[int(rng.integers(4))]
            if (r+dr,c+dc) not in cells:
                cells.add((r+dr,c+dc))
                break
    minr = min(r for r,c in cells)
    minc = min(c for r,c in cells)
    return tuple(sorted((r-minr,c-minc) for r,c in cells))


def _destination(cells, direction, h, w):
    if direction is None:
        return cells, False
    dr,dc = _DIRECTIONS[direction]
    shifted = frozenset((r+dr,c+dc) for r,c in cells)
    blocked = any(not (1 <= r < h-1 and 1 <= c < w-1) for r,c in shifted)
    return (cells if blocked else shifted), blocked


def _compatible(objects, color, orig, dest):
    for old in objects:
        if orig & old['orig'] or dest & old['dest']:
            return False
        if color == old['color']:
            if any(abs(r-y) <= 1 and abs(c-x) <= 1 for r,c in orig for y,x in old['orig']):
                return False
    return True


def _place(rng, objects, h, w, color, direction, blocked, kind):
    for _ in range(4):
        shape = _shape(rng, kind)
        maxr = max(r for r,c in shape)
        maxc = max(c for r,c in shape)
        if maxr > h-3 or maxc > w-3:
            continue
        for _ in range(170):
            ar = int(rng.integers(1,h-1-maxr))
            ac = int(rng.integers(1,w-1-maxc))
            if blocked and direction == 0: ar = 1
            if blocked and direction == 1: ar = h-2-maxr
            if blocked and direction == 2: ac = 1
            if blocked and direction == 3: ac = w-2-maxc
            orig = frozenset((ar+r,ac+c) for r,c in shape)
            dest, actual_blocked = _destination(orig,direction,h,w)
            if actual_blocked != blocked or not _compatible(objects,color,orig,dest):
                continue
            objects.append(dict(color=color, orig=orig, dest=dest, direction=direction))
            return True
    return False


def _handoff(rng, objects, h, w, sides):
    direction = int(rng.integers(4))
    other = (2 + int(rng.integers(2))) if direction < 2 else int(rng.integers(2))
    for _ in range(100):
        r = int(rng.integers(3,h-3))
        c = int(rng.integers(3,w-3))
        dr,dc = _DIRECTIONS[direction]
        ar = frozenset(((r,c),))
        br = frozenset(((r+dr,c+dc),))
        ad,_ = _destination(ar,direction,h,w)
        bd,_ = _destination(br,other,h,w)
        if _compatible(objects,sides[direction],ar,ad) and _compatible(objects,sides[other],br,bd) and not ad & bd:
            objects.extend((dict(color=sides[direction],orig=ar,dest=ad,direction=direction),
                            dict(color=sides[other],orig=br,dest=bd,direction=other)))
            return {direction,other}
    return None


def _sample_sides(rng, palette, anchor, mode):
    if mode == 'repeat':
        return anchor
    if mode == 'permute':
        order = rng.permutation(4)
        result = tuple(anchor[int(i)] for i in order)
        return result if result != anchor else tuple(anchor[(i+1)%4] for i in range(4))
    if mode == 'new':
        complement = [c for c in palette if c not in anchor]
        return tuple(complement[int(i)] for i in rng.permutation(len(complement))[:4])
    count = int(rng.integers(1,4))
    old = [anchor[int(i)] for i in rng.permutation(4)[:count]]
    fresh = [c for c in palette if c not in anchor]
    colors = old + [fresh[int(i)] for i in rng.permutation(len(fresh))[:4-count]]
    return tuple(colors[int(i)] for i in rng.permutation(4))


def _generate_pair(rng, sides, regime='source_train', first=False, forced_direction=None):
    for attempt in range(90):
        h,w = _dimensions(rng, first)
        available = [c for c in range(10) if c not in sides]
        order = rng.permutation(len(available))
        bg = available[int(order[0])]
        inert_count = int(rng.integers(0,6))
        if first:
            inert_count = max(1,inert_count)
        inert_colors = [available[int(order[i+1])] for i in range(inert_count)]
        objects = []
        handoff_dirs = set()
        if first:
            handoff_dirs = _handoff(rng, objects, h,w,sides)
            if handoff_dirs is None:
                continue
        active_count = 4 if first else int(rng.integers(1,5))
        dirs = set(handoff_dirs)
        if forced_direction is not None:
            dirs.add(forced_direction)
        remaining = [d for d in rng.permutation(4) if int(d) not in dirs]
        dirs.update(int(d) for d in remaining[:max(0,active_count-len(dirs))])
        if len(dirs) == 0:
            dirs.add(int(rng.integers(4)))
        area = (h-2)*(w-2)
        rate = float(rng.uniform(.025,.13 if regime == 'source_train' else .16))
        desired = max(len(dirs)+inert_count, int(area*rate))
        desired = min(desired, 80)
        singleton_scene = not first and rng.random() < .23
        specs = []
        needed = [d for d in dirs if d not in handoff_dirs]
        for d in needed:
            specs.append((sides[d],d,False,'single'))
        if first:
            multi_direction = int(rng.choice(tuple(dirs)))
            specs.append((sides[multi_direction],multi_direction,False,'irregular'))
        extra_slots = min(20, max(0,(desired-sum(len(o['orig']) for o in objects))//3-len(specs)-inert_count))
        for _ in range(extra_slots):
            d = int(rng.choice(tuple(dirs)))
            kind = 'single' if singleton_scene or rng.random() < .48 else ('bar','bend','block','irregular','diagonal')[int(rng.integers(5))]
            specs.append((sides[d],d,False,kind))
        blocked_count = int(rng.integers(0,4)) if not first else 1+int(rng.integers(0,2))
        if area < 40 and not first:
            blocked_count = min(blocked_count,1)
        for _ in range(blocked_count):
            d = int(rng.integers(4))
            kind = 'single' if rng.random() < .75 else 'bar'
            specs.append((sides[d],d,True,kind))
        for inert_index,color in enumerate(inert_colors):
            kind = 'diagonal' if first and inert_index == 0 else ('single' if singleton_scene or rng.random() < .4 else ('bar','bend','block','irregular','diagonal')[int(rng.integers(5))])
            specs.append((color,None,False,kind))
        if inert_colors and area > 70 and rng.random() < .6:
            specs.append((inert_colors[int(rng.integers(len(inert_colors)))],None,False,('single','bar','diagonal','irregular')[int(rng.integers(4))]))
        # Large objects go first, except for the prescribed handoff.
        specs.sort(key=lambda s: (s[3] == 'single', s[2]))
        good = True
        for color,d,blocked,kind in specs:
            if not _place(rng,objects,h,w,color,d,blocked,kind):
                good = False
                break
        if not good:
            continue
        grid = np.full((h,w),bg,dtype=np.int8)
        grid[0,1:-1] = sides[0]
        grid[-1,1:-1] = sides[1]
        grid[1:-1,0] = sides[2]
        grid[1:-1,-1] = sides[3]
        for obj in objects:
            for r,c in obj['orig']:
                grid[r,c] = obj['color']
        if _rows(grid) in _SOURCE_INPUTS:
            continue
        try:
            output,stats = _interpret(grid)
        except AssertionError:
            continue
        expected = grid.copy()
        expected[1:-1,1:-1] = bg
        for obj in objects:
            for r,c in obj['dest']:
                expected[r,c] = obj['color']
        if not np.array_equal(output,expected) or np.array_equal(output,grid):
            continue
        if first and (stats['moved'] != {0,1,2,3} or stats['handoffs'] == 0 or stats['moved_multi'] == 0 or not stats['inert'] or not stats['blocked']):
            continue
        if forced_direction is not None and forced_direction not in stats['moved']:
            continue
        return grid,output
    raise RuntimeError('could not place a valid scene')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain,ntest = _resolve_counts(seed,num_train,num_test)
    palette = tuple(int(x) for x in rng.permutation(10))
    anchor = tuple(palette[:4])
    pair_seeds = rng.integers(0,np.iinfo(np.int64).max,size=ntrain+ntest,dtype=np.int64)
    puzzle = {'train':[],'test':[]}
    seen = set()
    for i in range(ntrain+ntest):
        pair_rng = np.random.default_rng(int(pair_seeds[i]))
        train = i < ntrain
        if i == 0:
            sides = anchor
            forced = None
        elif i == 1:
            sides = _sample_sides(pair_rng,palette,anchor,'permute')
            changed = [d for d in range(4) if sides[d] in anchor and anchor.index(sides[d]) != d]
            forced = changed[int(pair_rng.integers(len(changed)))]
        else:
            mode = ('repeat','permute','partial','new')[int(pair_rng.integers(4))]
            sides = _sample_sides(pair_rng,palette,anchor,mode)
            forced = None
        for _ in range(12):
            inp,out = _generate_pair(pair_rng,sides,regime='source_train' if train else 'target_test',first=i==0,forced_direction=forced)
            if _key(inp) not in seen:
                break
        else:
            raise RuntimeError('duplicate scene rejection exhausted')
        seen.add(_key(inp))
        puzzle['train' if train else 'test'].append({'input':inp,'output':out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle,dict) and set(puzzle)=={'train','test'}, 'invalid puzzle schema')
    train,test = puzzle['train'],puzzle['test']
    _require(isinstance(train,list) and isinstance(test,list), 'pair groups must be lists')
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), 'invalid pair counts')
    seen = set()
    stats_all = []
    for pair in train+test:
        _require(isinstance(pair,dict) and set(pair)=={'input','output'}, 'invalid pair schema')
        for name in ('input','output'):
            grid = pair[name]
            _require(isinstance(grid,np.ndarray) and grid.ndim==2, 'grid must be an array')
            _require(5 <= grid.shape[0] <= 30 and 5 <= grid.shape[1] <= 30, 'grid size out of range')
            _require(np.issubdtype(grid.dtype,np.integer) and bool(np.all((grid>=0)&(grid<=9))), 'invalid grid colors')
        inp,out = pair['input'],pair['output']
        _require(inp.shape==out.shape and not np.array_equal(inp,out), 'invalid dimensions or unchanged pair')
        k = _key(inp)
        _require(k not in seen and _rows(inp) not in _SOURCE_INPUTS, 'duplicate or source input')
        seen.add(k)
        expected,stats = _interpret(inp)
        _require(np.array_equal(expected,out), 'incorrect simultaneous translation')
        _require(np.array_equal(inp[0],out[0]) and np.array_equal(inp[-1],out[-1]) and np.array_equal(inp[:,0],out[:,0]) and np.array_equal(inp[:,-1],out[:,-1]), 'frame changed')
        stats_all.append(stats)
    first = stats_all[0]
    _require(first['moved']=={0,1,2,3} and first['moved_multi'] >= 1 and first['handoffs'] >= 1 and first['inert'] and first['blocked'], 'first train lacks core demonstrations')
    second = stats_all[1]
    anchor = first['sides']
    _require(any(second['sides'][d] in anchor and anchor.index(second['sides'][d])!=d and d in second['moved'] for d in range(4)), 'training lacks frame reassignment evidence')
    _require(all(stats['moved'] for stats in stats_all), 'pair has no moving mark')


def _equal(a,b):
    return all(len(a[k])==len(b[k]) and all(np.array_equal(x[n],y[n]) for x,y in zip(a[k],b[k]) for n in ('input','output')) for k in ('train','test'))


def _visible_shapes(stats):
    kinds = set()
    components = stats['components']
    for _,cells in components:
        if len(cells) <= 1:
            continue
        rows = [r for r,c in cells]
        cols = [c for r,c in cells]
        if len(set(rows)) == 1 or len(set(cols)) == 1:
            kinds.add('bar')
        elif len(cells) == (max(rows)-min(rows)+1)*(max(cols)-min(cols)+1):
            kinds.add('block')
        elif len(cells) <= 4:
            kinds.add('bend')
        else:
            kinds.add('irregular')
    for i,(color,cells) in enumerate(components):
        for other,other_cells in components[i+1:]:
            if color == other and any(max(abs(r-y),abs(c-x)) == 1 for r,c in cells for y,x in other_cells):
                kinds.add('diagonal')
    return kinds


def self_test() -> None:
    source_in = np.array([[int(c) for c in row] for row in _SOURCE_TEST[0]],dtype=np.int8)
    source_out = np.array([[int(c) for c in row] for row in _SOURCE_TEST[1]],dtype=np.int8)
    solved,source_stats = _interpret(source_in)
    _require(np.array_equal(solved,source_out) and source_stats['handoffs'] >= 1, 'original test handoff regression failed')
    for seed,ntrain,ntest in ((101,2,1),(102,10,1),(103,10,4),(201,None,1),(202,None,4),(203,2,None),(204,10,None),(271828,7,3)):
        puzzle = generate_puzzle(seed,ntrain,ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None: _require(len(puzzle['train'])==ntrain,'train count ignored')
        if ntest is not None: _require(len(puzzle['test'])==ntest,'test count ignored')
        _require(_equal(puzzle,generate_puzzle(seed,ntrain,ntest)),'not deterministic')
    for args in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4),(True,1)):
        try:
            generate_puzzle(999,*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p['train']) for p in puzzles}) >= 4 and len({len(p['test']) for p in puzzles}) >= 3,'automatic counts collapsed')
    _require(len({_key(p['train'][0]['input']) for p in puzzles}) == 42, 'seed diversity collapsed')
    shapes = set()
    sizes = set()
    blocked_counts = set()
    inert_counts = set()
    handoff_pairs = 0
    single_only = 0
    multi_pairs = 0
    inert_multi_pairs = 0
    repeated_inert_pairs = 0
    missing_directions = 0
    busy_pairs = 0
    sparse_pairs = 0
    same_color_move_block = 0
    shape_kinds = set()
    frame_modes = set()
    combined = 0
    varied_sizes = 0
    varied_blocking = 0
    varied_inert = 0
    for p in puzzles:
        validate_generated_puzzle(p)
        local_sizes = set()
        local_blocking = set()
        local_inert = set()
        anchor = _interpret(p['train'][0]['input'])[1]['sides']
        for pair in p['train']+p['test']:
            _,stats = _interpret(pair['input'])
            sizes.add(pair['input'].shape)
            local_sizes.add(pair['input'].shape)
            local_blocking.add(stats['blocked_count'])
            local_inert.add(len(stats['inert']))
            blocked_counts.add(stats['blocked_count'])
            inert_counts.add(len(stats['inert']))
            handoff_pairs += stats['handoffs'] > 0
            single_only += max(stats['sizes'],default=0)==1
            multi_pairs += sum(n>1 for n in stats['sizes']) >= 2
            shapes.update(n for n in stats['sizes'] if n>1)
            shape_kinds.update(_visible_shapes(stats))
            inert_multi_pairs += any(color in stats['inert'] and len(cells)>1 for color,cells in stats['components'])
            repeated_inert_pairs += any(sum(other==color for other,_ in stats['components'])>1 for color in stats['inert'])
            missing_directions += len(stats['moved'] | stats['blocked']) < 4
            busy_pairs += stats['foreground'] >= 20
            sparse_pairs += stats['foreground'] <= 5
            same_color_move_block += bool(stats['moved'] & stats['blocked'])
            combined += bool(stats['inert']) and stats['blocked_count']>0 and sum(n>1 for n in stats['sizes'])>=2
        varied_sizes += len(local_sizes)>1
        varied_blocking += len(local_blocking)>1
        varied_inert += len(local_inert)>1
        for pair in p['test']:
            sides = _interpret(pair['input'])[1]['sides']
            shared = len(set(anchor)&set(sides))
            frame_modes.add('repeat' if sides==anchor else 'same_set' if shared==4 else 'disjoint' if shared==0 else 'partial')
    _require(len(sizes)>=30 and min(h for h,w in sizes)<=9 and max(h for h,w in sizes)>=23,'size diversity collapsed')
    _require({0,1,2} <= blocked_counts and {0,1,2,3} <= inert_counts,'mark counts collapsed')
    _require(handoff_pairs>=42 and single_only>0 and multi_pairs>0 and len(shapes)>=4,'shape and handoff diversity collapsed')
    _require(shape_kinds=={'bar','bend','block','irregular','diagonal'}, 'visible shape variety collapsed')
    _require(frame_modes=={'repeat','same_set','partial','disjoint'}, 'frame palette modes collapsed')
    _require(inert_multi_pairs>=42 and repeated_inert_pairs>=20 and missing_directions>=42, 'inert or directional variety collapsed')
    _require(busy_pairs>=30 and sparse_pairs>=10 and same_color_move_block>=42 and combined>=42, 'population and mixed cases collapsed')
    _require(varied_sizes>=40 and varied_blocking>=35 and varied_inert>=35, 'within-puzzle variation collapsed')
