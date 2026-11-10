import numpy as np


_UP, _DOWN, _LEFT, _RIGHT = 0, 1, 2, 3
_ALL_DIRECTIONS = frozenset((_UP, _DOWN, _LEFT, _RIGHT))

def _grid_key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return tuple(array.shape), array.tobytes()


def _pair_key(input_grid, output_grid):
    return _grid_key(input_grid), _grid_key(output_grid)


def _rows(rows):
    return np.array(
        [[ord(character) - 48 for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_RAW = (
    (
        (
            "7777777777",
            "7777777777",
            "7777717777",
            "7777711900",
            "7777777977",
            "7777776667",
            "7777777677",
            "7297777777",
            "7787777777",
            "7777777777",
        ),
        (
            "7777777777",
            "7777777777",
            "1777777777",
            "1177777900",
            "7777777977",
            "7777777777",
            "7777777777",
            "2797777777",
            "7777776667",
            "7787777677",
        ),
    ),
    (
        (
            "777777777",
            "777777777",
            "777777777",
            "777577777",
            "777922777",
            "777872777",
            "777877777",
            "777777777",
            "777777777",
        ),
        (
            "777577777",
            "777777777",
            "777777777",
            "777777777",
            "777977722",
            "777777772",
            "777777777",
            "777877777",
            "777877777",
        ),
    ),
    (
        (
            "777777777777",
            "777777777777",
            "777777777777",
            "777747477777",
            "777744477777",
            "777779777777",
            "777679777777",
            "777999337777",
            "777879337777",
            "777777777777",
            "777777777777",
            "777777777777",
        ),
        (
            "777647477777",
            "777744477777",
            "777777777777",
            "777777777777",
            "777777777777",
            "777779777777",
            "777779777777",
            "777999777733",
            "777779777733",
            "777777777777",
            "777777777777",
            "777877777777",
        ),
    ),
    (
        (
            "7777777777777777",
            "7777777777779777",
            "7777777777779777",
            "7777777777779977",
            "7755557557777777",
            "7757755577777777",
            "7757777555777777",
            "7755577575777777",
            "7777777779987777",
            "7777777739987777",
            "7779777776077777",
            "7779777777007777",
            "7999997777707077",
            "7479777777000077",
            "7729777777777777",
            "7777777777777777",
        ),
        (
            "7755557557777777",
            "7757755577779777",
            "7757777555779777",
            "7755577575779977",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
            "7777777779977778",
            "3777777779977778",
            "7779777777777777",
            "7779777777777777",
            "7999997777077777",
            "7779777777007777",
            "2779777777707077",
            "7477777776000077",
        ),
    ),
)

_SOURCE_PAIR_KEYS = frozenset(
    _pair_key(_rows(input_rows), _rows(output_rows))
    for input_rows, output_rows in _SOURCE_RAW
)



def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    height, width = mask.shape
    for start in np.argwhere(mask):
        start = tuple(map(int, start))
        if seen[start]:
            continue
        seen[start] = True
        todo = [start]
        group = []
        while todo:
            row, col = todo.pop()
            group.append((row, col))
            for nr, nc in ((row-1,col),(row+1,col),(row,col-1),(row,col+1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr,nc] and not seen[nr,nc]:
                    seen[nr,nc] = True
                    todo.append((nr,nc))
        groups.append(np.array(group, dtype=np.int16))
    return groups


def _norm(cells):
    a = np.array(list(cells), dtype=np.int16).reshape(-1, 2)
    a -= a.min(axis=0)
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def _shape(rng, kind):
    if kind == 'singleton':
        return np.array([[0, 0]], dtype=np.int16)
    if kind == 'edge':
        return _norm(( (0, c) for c in range(int(rng.integers(1, 4))) ))
    if kind == 'bar':
        n = int(rng.integers(3, 9))
        a = [(0, j) for j in range(n)]
        if rng.random() < .45:
            a.append((1, int(rng.integers(n))))
        result = _norm(a)
    elif kind == 'large':
        n = int(rng.integers(9, 20))
        max_h = int(rng.integers(3, 6))
        max_w = int(rng.integers(4, 9))
        cells = {(0, 0)}
        for _ in range(n * 30):
            r, c = tuple(cells)[int(rng.integers(len(cells)))]
            dr, dc = ((-1,0),(1,0),(0,-1),(0,1))[int(rng.integers(4))]
            trial = cells | {(r+dr, c+dc)}
            rr = [x[0] for x in trial]
            cc = [x[1] for x in trial]
            if max(rr)-min(rr) < max_h and max(cc)-min(cc) < max_w:
                cells = trial
            if len(cells) >= n:
                break
        result = _norm(cells)
    elif kind == 'irregular':
        n = int(rng.integers(4, 9))
        cells = {(0, 0)}
        for _ in range(n * 20):
            r, c = tuple(cells)[int(rng.integers(len(cells)))]
            dr, dc = ((-1,0),(1,0),(0,-1),(0,1))[int(rng.integers(4))]
            trial = cells | {(r+dr, c+dc)}
            rr = [x[0] for x in trial]
            cc = [x[1] for x in trial]
            if max(rr)-min(rr) < 4 and max(cc)-min(cc) < 5:
                cells = trial
            if len(cells) >= n:
                break
        result = _norm(cells)
    else:
        choice = int(rng.integers(4))
        if choice == 0:
            return _shape(rng, 'singleton')
        if choice == 1:
            result = _norm([(0, j) for j in range(int(rng.integers(2, 5)))])
        elif choice == 2:
            result = _norm([(0,0),(1,0),(1,1)] if rng.random() < .5 else [(0,0),(0,1),(1,0),(1,1)])
        else:
            return _shape(rng, 'irregular')
    if rng.random() < .5:
        result = result[:, ::-1].copy()
    if rng.random() < .5:
        result[:, 0] *= -1
    if rng.random() < .5:
        result[:, 1] *= -1
    return _norm(result)


def _hub_shape(rng, style):
    if style == 'single':
        return _norm([(0,0)])
    if style == 'line':
        n = int(rng.integers(2, 7))
        cells = [(0, j) for j in range(n)]
    elif style == 't':
        span = int(rng.integers(3, 6))
        middle = int(rng.integers(1, span-1))
        cells = [(0, j) for j in range(span)] + [(k, middle) for k in range(1, int(rng.integers(2, 5)))]
    elif style == 'step':
        n = int(rng.integers(4, 9))
        cells = [(0,0)]
        while len(cells) < n:
            r,c = cells[-1]
            cells.append((r+1,c) if rng.random() < .5 else (r,c+1))
    elif style == 'block':
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        cells = [(r,c) for r in range(h) for c in range(w)]
        if h*w >= 6 and rng.random() < .5:
            cells.remove((0,0))
    else:
        n = int(rng.integers(4, 12))
        cells = {(0,0)}
        for _ in range(n*25):
            r,c = tuple(cells)[int(rng.integers(len(cells)))]
            dr,dc = ((-1,0),(1,0),(0,-1),(0,1))[int(rng.integers(4))]
            trial = cells | {(r+dr,c+dc)}
            rr = [x[0] for x in trial]
            cc = [x[1] for x in trial]
            if max(rr)-min(rr) < 5 and max(cc)-min(cc) < 6:
                cells = trial
            if len(cells) >= n:
                break
    result = _norm(cells)
    if rng.random() < .5:
        result = result[:, ::-1].copy()
    if rng.random() < .5:
        result[:, 0] *= -1
    if rng.random() < .5:
        result[:, 1] *= -1
    return _norm(result)


def _background(grid):
    counts = np.bincount(np.asarray(grid, dtype=np.int16).ravel(), minlength=10)
    choice = int(np.argmax(counts))
    assert counts[choice] > grid.size // 2
    return choice


def _contact(cells, labels):
    h,w = labels.shape
    contacts = set()
    for r,c in cells:
        r,c = int(r),int(c)
        for nr,nc,d in ((r+1,c,_UP),(r-1,c,_DOWN),(r,c+1,_LEFT),(r,c-1,_RIGHT)):
            if 0 <= nr < h and 0 <= nc < w and labels[nr,nc] >= 0:
                contacts.add((int(labels[nr,nc]),d))
    return contacts


def _destination(cells, direction, height, width):
    if direction == _UP:
        shift = (-int(cells[:,0].min()), 0)
    elif direction == _DOWN:
        shift = (height-1-int(cells[:,0].max()), 0)
    elif direction == _LEFT:
        shift = (0, -int(cells[:,1].min()))
    else:
        shift = (0, width-1-int(cells[:,1].max()))
    return cells + np.array(shift, dtype=np.int16)


def _cellset(cells):
    return set(map(tuple, np.asarray(cells).tolist()))


def _analyze_pair(inp, out, background, hub_color):
    assert inp.shape == out.shape
    h,w = inp.shape
    assert _background(inp) == _background(out) == background
    hub_mask = inp == hub_color
    assert np.any(hub_mask) and np.array_equal(hub_mask, out == hub_color)
    hubs = _components(hub_mask)
    labels = np.full(inp.shape, -1, dtype=np.int16)
    for i, cells in enumerate(hubs):
        labels[cells[:,0], cells[:,1]] = i
    payloads = []
    for color in np.unique(inp):
        color = int(color)
        if color in (background, hub_color):
            continue
        for cells in _components(inp == color):
            contact = _contact(cells, labels)
            assert len(contact) == 1
            hub_id, direction = next(iter(contact))
            dest = _destination(cells, direction, h, w)
            payloads.append(dict(color=color, cells=cells, dest=dest, hub=hub_id, direction=direction))
    assert payloads
    sources = [_cellset(p['cells']) for p in payloads]
    destinations = [_cellset(p['dest']) for p in payloads]
    hub_set = _cellset(np.argwhere(hub_mask))
    for i in range(len(payloads)):
        assert not destinations[i] & hub_set
        for j in range(i):
            assert not sources[i] & sources[j]
            assert not destinations[i] & destinations[j]
            assert not destinations[i] & sources[j]
            assert not destinations[j] & sources[i]
            if payloads[i]['color'] == payloads[j]['color']:
                assert not _adjacent(sources[i], sources[j])
    expected = inp.copy()
    for cells in sources:
        for r,c in cells:
            expected[r,c] = background
    for p in payloads:
        for r,c in p['dest']:
            assert expected[r,c] == background
            expected[r,c] = p['color']
    assert np.array_equal(expected, out)
    assert not np.array_equal(inp, out)
    return dict(hubs=hubs, payloads=payloads, background=background, hub_color=hub_color)


def _adjacent(a,b):
    for r,c in a:
        if (r-1,c) in b or (r+1,c) in b or (r,c-1) in b or (r,c+1) in b:
            return True
    return False


def _pair_attempt(rng, regime, background, hub_color, palette, directions, hub_count, kinds, force_zero, size_mode, idle_hubs):
    n = len(directions)
    if size_mode == 'compact_train':
        h = int(rng.integers(10, 13))
        w = int(rng.integers(10, 13))
    elif size_mode == 'compact_test':
        h = int(rng.integers(12, 16))
        w = int(rng.integers(12, 16))
    elif regime == 'source_train':
        h = int(rng.integers(9, 21))
        w = int(rng.integers(9, 21))
    else:
        h = int(rng.integers(11, 27))
        w = int(rng.integers(11, 27))
    if hub_count >= 2 and size_mode != 'compact_train':
        h = max(h, 12)
        w = max(w, 12)
    if size_mode != 'compact_test' and (any(k == 'large' for k in kinds) or n >= 7):
        h = max(h, 15)
        w = max(w, 15)
    if force_zero is not None:
        h = max(h, 13)
        w = max(w, 13)
    inp = np.full((h,w), background, dtype=np.int8)
    labels = np.full((h,w), -1, dtype=np.int16)
    styles = ['single','line','t','step','block','branch']
    hub_shapes = []
    for i in range(hub_count):
        style = styles[int(rng.integers(len(styles)))]
        if size_mode == 'compact_train':
            style = ('single','line','t')[int(rng.integers(3))]
        if force_zero is not None and i == 0:
            style = 'single' if rng.random() < .4 else 'line'
        shape = _hub_shape(rng, style)
        hh,ww = int(shape[:,0].max())+1, int(shape[:,1].max())+1
        if hh > h-2 or ww > w-2:
            return None
        placed = False
        for _ in range(40):
            top = int(rng.integers(1, h-hh))
            left = int(rng.integers(1, w-ww))
            if force_zero is not None and i == 0:
                zd = force_zero
                if zd == _UP: top = 1
                if zd == _DOWN: top = h-hh-1
                if zd == _LEFT: left = 1
                if zd == _RIGHT: left = w-ww-1
            cells = shape + np.array((top,left),dtype=np.int16)
            proposed = _cellset(cells)
            old = _cellset(np.argwhere(labels >= 0))
            if proposed & old or _adjacent(proposed, old):
                continue
            labels[cells[:,0],cells[:,1]] = i
            inp[cells[:,0],cells[:,1]] = hub_color
            hub_shapes.append(cells)
            placed = True
            break
        if not placed:
            return None
    active_count = hub_count - idle_hubs
    assignments = list(range(active_count))
    while len(assignments) < n:
        assignments.append(int(rng.integers(active_count)))
    rng.shuffle(assignments)
    if force_zero is not None:
        zero_index = directions.index(force_zero)
        assignments[zero_index] = 0
        if 1 not in assignments and active_count > 1:
            assignments[-1 if zero_index != n-1 else 0] = 1
    colors = [int(x) for x in rng.permutation(palette)[:n]]
    while len(colors) < n:
        colors.append(int(palette[int(rng.integers(len(palette)))]))
    if n >= 4 and rng.random() < .35:
        colors[-1] = colors[0]
    objects = []
    for i,(direction,hub,kind,color) in enumerate(zip(directions,assignments,kinds,colors)):
        if force_zero is not None and i == directions.index(force_zero):
            kind = 'singleton'
        objects.append(dict(direction=direction, hub=hub, color=color, shape=_shape(rng,kind), zero=force_zero is not None and i == directions.index(force_zero)))
    objects.sort(key=lambda x: (not x['zero'], -len(x['shape'])))
    hub_set = _cellset(np.argwhere(labels >= 0))
    placed = []
    for obj in objects:
        candidates = []
        direction = obj['direction']
        hub = hub_shapes[obj['hub']]
        dr,dc = { _UP:(-1,0), _DOWN:(1,0), _LEFT:(0,-1), _RIGHT:(0,1) }[direction]
        for hr,hc in hub:
            ar,ac = int(hr)+dr,int(hc)+dc
            for sr,sc in obj['shape']:
                cells = obj['shape'] + np.array((ar-int(sr),ac-int(sc)),dtype=np.int16)
                if np.any(cells[:,0] < 0) or np.any(cells[:,0] >= h) or np.any(cells[:,1] < 0) or np.any(cells[:,1] >= w):
                    continue
                src = _cellset(cells)
                if src & hub_set or _contact(cells,labels) != {(obj['hub'],direction)}:
                    continue
                dest = _destination(cells,direction,h,w)
                dst = _cellset(dest)
                if obj['zero'] != (src == dst):
                    continue
                if dst & hub_set:
                    continue
                if any(src & other['src'] or dst & other['dst'] or src & other['dst'] or dst & other['src'] or (obj['color'] == other['color'] and _adjacent(src,other['src'])) for other in placed):
                    continue
                candidates.append((cells,src,dest,dst))
        if not candidates:
            return None
        cells,src,dest,dst = candidates[int(rng.integers(len(candidates)))]
        obj.update(cells=cells,src=src,dest=dest,dst=dst)
        placed.append(obj)
    for obj in placed:
        inp[obj['cells'][:,0],obj['cells'][:,1]] = obj['color']
    out = inp.copy()
    for obj in placed:
        for r,c in obj['src']:
            out[r,c] = background
    for obj in placed:
        out[obj['dest'][:,0],obj['dest'][:,1]] = obj['color']
    if _background(inp) != background or _background(out) != background:
        return None
    try:
        analyzed = _analyze_pair(inp,out,background,hub_color)
        if len(analyzed['payloads']) != n or len(analyzed['hubs']) != hub_count:
            return None
    except AssertionError:
        return None
    return inp,out


def _make_pair(rng, regime, background, hub_color, palette, directions, hub_count, kinds, force_zero, size_mode, idle_hubs):
    for _ in range(260):
        result = _pair_attempt(rng,regime,background,hub_color,palette,directions,hub_count,kinds,force_zero,size_mode,idle_hubs)
        if result is not None:
            return result
    raise ValueError('could not construct pair')


def _valid_count(value,name,lo,hi):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(name)
    return int(value)


def _resolve_counts(seed,rng,num_train,num_test):
    if num_train is not None: num_train = _valid_count(num_train,'num_train',2,10)
    if num_test is not None: num_test = _valid_count(num_test,'num_test',1,4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10-num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test+1,11))
    elif num_test is None:
        num_test = int(rng.integers(1,min(4,num_train-1)+1))
    if num_train <= num_test: raise ValueError('num_train must exceed num_test')
    return num_train,num_test


def _directions(rng, count, initial, per_side_limit):
    result = list(initial)
    while len(result) < count:
        choices = [d for d in range(4) if result.count(d) < per_side_limit]
        result.append(int(choices[int(rng.integers(len(choices)))]))
    return result


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train,num_test = _resolve_counts(seed,rng,num_train,num_test)
    order = [int(x) for x in rng.permutation(10)]
    background,hub_color = order[:2]
    palette = order[2:]
    # The background and scaffold color are conventions shared by all pairs.
    training_directions = [[] for _ in range(num_train)]
    for i,d in enumerate(rng.permutation(4)):
        training_directions[i % num_train].append(int(d))
    specs = []
    for i in range(num_train):
        hub_count = 1 if i == 0 else (2 if i == 1 else int(rng.choice([1,1,1,2,2,3])))
        n = int(rng.integers(max(3,len(training_directions[i]),hub_count),7))
        size_mode = 'compact_train' if i == 1 and seed % 5 == 1 else None
        if size_mode:
            n = 3
        if i == 0 and seed % 3 == 0:
            n = min(n, 4)
        directions = _directions(rng,n,training_directions[i],2)
        kinds = ['ordinary']*n
        if i == 0:
            kinds[0] = 'singleton'
            if n >= 4: kinds[1] = 'bar'
        elif i == 1:
            kinds[0] = 'irregular'
        else:
            if rng.random() < .45: kinds[0] = 'bar'
            if rng.random() < .2: kinds[-1] = 'large'
        zero = None
        if i == 0 and seed % 3 == 0:
            zero = directions[0]
            directions = [d if j == 0 or d != zero else (zero + 1 + (j-1) % 3) % 4 for j,d in enumerate(directions)]
        rng.shuffle(directions)
        rng.shuffle(kinds)
        idle_hubs = int(i >= 2 and hub_count >= 2 and seed % 7 == 4)
        specs.append((directions,hub_count,kinds,zero,size_mode,idle_hubs))
    rng.shuffle(specs)
    puzzle = {'train':[],'test':[]}
    seen = set()
    for split,count in (('train',num_train),('test',num_test)):
        for i in range(count):
            for retry in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0,np.iinfo(np.int64).max)))
                if split == 'train':
                    directions,hub_count,kinds,zero,size_mode,idle_hubs = specs[i]
                else:
                    size_mode = 'compact_test' if i == 0 and seed % 2 == 0 and seed % 7 != 2 else None
                    hub_count = int(rng.choice([1,1,2,2,3]))
                    n = int(rng.integers(max(3,hub_count),9))
                    if size_mode:
                        hub_count = 1
                        n = 5
                    idle_hubs = int(size_mode is None and i == 0 and seed % 3 == 0)
                    if idle_hubs:
                        hub_count = max(hub_count,2)
                    if seed % 7 == 2 and i == 0:
                        n = min(n, 5)
                    directions = _directions(rng,n,[],3)
                    if size_mode:
                        directions[:4] = list(range(4))
                    if i == 0 and seed % 4 == 1:
                        directions[:4] = list(range(4))
                    kinds = ['ordinary']*n
                    kinds[0] = 'large' if rng.random() < .7 else 'bar'
                    if n >= 5 and rng.random() < .5: kinds[1] = 'irregular'
                    zero = directions[0] if seed % 7 == 2 and i == 0 else None
                    if zero is not None:
                        directions = [d if j == 0 or d != zero else (zero + 1 + (j-1) % 3) % 4 for j,d in enumerate(directions)]
                try:
                    inp,out = _make_pair(pair_rng,'source_train' if split=='train' else 'target_test',background,hub_color,palette,directions,hub_count,kinds,zero,size_mode,idle_hubs)
                except ValueError:
                    continue
                key = _grid_key(inp)
                if key in seen or _pair_key(inp,out) in _SOURCE_PAIR_KEYS:
                    continue
                seen.add(key)
                puzzle[split].append({'input':inp,'output':out})
                break
            else:
                raise ValueError('could not construct unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert isinstance(puzzle['train'],list) and isinstance(puzzle['test'],list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs,pairs = set(),set()
    all_meta = {'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp,out = pair['input'],pair['output']
            for grid in (inp,out):
                assert isinstance(grid,np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype,np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert inp.shape == out.shape
            if split == 'train':
                assert 9 <= inp.shape[0] <= 20 and 9 <= inp.shape[1] <= 20
            else:
                assert 11 <= inp.shape[0] <= 26 and 11 <= inp.shape[1] <= 26
            ik,pk = _grid_key(inp),_pair_key(inp,out)
            assert ik not in inputs and pk not in pairs and pk not in _SOURCE_PAIR_KEYS
            inputs.add(ik); pairs.add(pk)
            bg = _background(inp)
            assert bg == _background(out)
            all_meta[split].append((inp,out,bg))
    backgrounds = {x[2] for values in all_meta.values() for x in values}
    assert len(backgrounds) == 1
    background = next(iter(backgrounds))
    possible = set(range(10)) - {background}
    for inp,out,bg in all_meta['train']+all_meta['test']:
        good = set()
        for hub_color in possible:
            try:
                _analyze_pair(inp,out,bg,hub_color)
            except AssertionError:
                pass
            else:
                good.add(hub_color)
        possible &= good
        assert possible
    assert len(possible) == 1
    hub_color = next(iter(possible))
    train = [_analyze_pair(inp,out,bg,hub_color) for inp,out,bg in all_meta['train']]
    test = [_analyze_pair(inp,out,bg,hub_color) for inp,out,bg in all_meta['test']]
    assert set().union(*(set(p['direction'] for p in m['payloads']) for m in train)) == _ALL_DIRECTIONS
    assert all(len(m['hubs']) <= 3 for m in train+test)
    assert all(3 <= len(m['payloads']) <= 6 for m in train)
    assert all(3 <= len(m['payloads']) <= 8 for m in test)


def _equal(a,b):
    return all(len(a[s])==len(b[s]) and all(np.array_equal(x[k],y[k]) for x,y in zip(a[s],b[s]) for k in ('input','output')) for s in ('train','test'))


def self_test() -> None:
    a = generate_puzzle(73)
    b = generate_puzzle(991,6,3)
    assert _equal(b,generate_puzzle(991,6,3)) and not _equal(b,generate_puzzle(992,6,3))
    assert a['train'] and a['test']
    count_train,count_test = set(),set()
    dims,layouts,puzzles = set(),set(),set()
    zero = overhang = repeated = long_train = large_test = 0
    adjacent_pairs = repeated_colors = compact_tests = dense_tests = idle_hub_pairs = 0
    backgrounds, hub_colors = set(), set()
    styles = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        count_train.add(len(p['train'])); count_test.add(len(p['test']))
        puzzles.add(tuple(_pair_key(pair['input'],pair['output']) for split in ('train','test') for pair in p[split]))
        bg = _background(p['train'][0]['input'])
        hub_color = next(c for c in range(10) if c != bg and all(
            np.any(pair['input'] == c) and
            np.array_equal(pair['input'] == c, pair['output'] == c)
            for split in ('train','test') for pair in p[split]))
        backgrounds.add(bg); hub_colors.add(hub_color)
        train_colors = {int(v) for pair in p['train'] for v in np.unique(pair['input'])} - {bg,hub_color}
        assert len(train_colors) >= 6
        for split in ('train','test'):
            for pair in p[split]:
                inp,out = pair['input'],pair['output']
                dims.add(inp.shape)
                mapping = {}
                normalized = []
                for v in inp.ravel():
                    v = int(v)
                    if v not in mapping: mapping[v] = len(mapping)
                    normalized.append(mapping[v])
                layouts.add((inp.shape,bytes(normalized)))
                m = _analyze_pair(inp,out,bg,hub_color)
                idle_hub_pairs += len({x['hub'] for x in m['payloads']}) < len(m['hubs'])
                dirs = [(x['hub'],x['direction']) for x in m['payloads']]
                repeated += len(dirs) != len(set(dirs))
                repeated_colors += len({x['color'] for x in m['payloads']}) < len(m['payloads'])
                sources = [_cellset(x['cells']) for x in m['payloads']]
                adjacent_pairs += any(_adjacent(sources[j],sources[k]) and m['payloads'][j]['color'] != m['payloads'][k]['color'] for j in range(len(sources)) for k in range(j))
                if split == 'test':
                    compact_tests += inp.shape[0] <= 16 and inp.shape[1] <= 16
                    dense_tests += np.mean(inp != bg) >= .2
                styles.add(tuple(sorted((int(x[:,0].max()-x[:,0].min()+1),int(x[:,1].max()-x[:,1].min()+1),len(x)) for x in m['hubs'])))
                for obj in m['payloads']:
                    cells,dest = obj['cells'],obj['dest']
                    zero += np.array_equal(cells,dest)
                    hub = m['hubs'][obj['hub']]
                    if obj['direction'] in (_UP,_DOWN):
                        overhang += int(cells[:,1].min()<hub[:,1].min() or cells[:,1].max()>hub[:,1].max())
                    else:
                        overhang += int(cells[:,0].min()<hub[:,0].min() or cells[:,0].max()>hub[:,0].max())
                    if split=='train': long_train += len(cells)>=5
                    else: large_test += len(cells)>=9
    assert len(count_train)>=4 and len(count_test)>=3
    assert len(puzzles)==42 and len(dims)>=20 and len(layouts)>=200
    assert zero>0 and overhang>0 and repeated>0 and long_train>0 and large_test>0 and len(styles)>=10
    assert adjacent_pairs>0 and repeated_colors>0 and compact_tests>=10 and dense_tests>0 and idle_hub_pairs>0
    assert len(backgrounds)>=7 and len(hub_colors)>=7
    for nt,ne in ((2,1),(10,1),(10,4)):
        p = generate_puzzle(2000+nt*10+ne,nt,ne)
        assert len(p['train'])==nt and len(p['test'])==ne
    for seed,nt,ne in ((3101,None,1),(3104,None,4),(3202,2,None),(3210,10,None)):
        p = generate_puzzle(seed,nt,ne)
        if nt is not None: assert len(p['train'])==nt
        if ne is not None: assert len(p['test'])==ne
    for kw in ({'num_train':1},{'num_train':11},{'num_test':0},{'num_test':5},{'num_train':2,'num_test':2},{'num_train':3,'num_test':4}):
        try: generate_puzzle(7,**kw)
        except ValueError: pass
        else: raise AssertionError('illegal count')
