"""Beam-built mirrors: conditional complementary assembly and ray tracing."""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0,689312262,591552727,342173042,478725428,686818804,366127430,200877467,330500118,535870367,628973324,238427018,819668787,228591237,263229620,818035439,404182387,676155919,856088514,872839615,664882285,612125496,689314295,763075843,313450175,660747892,94883794,919371143,874559194,223233460,351383349,862647053,651582220,798398699,524563989,257113320,459937602,138143984,924526711,423616738,614690416,238313845)
CARDINALS = ((1,0),(-1,0),(0,1),(0,-1))


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError('grid dimensions must be 1..30')
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError('grid must contain integer colors 0..9')
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(a, bg):
    left = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    objects = []
    while left:
        first = min(left)
        left.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in CARDINALS:
                p = r+dr, c+dc
                if p in left and a[p] == a[first]:
                    left.remove(p)
                    stack.append(p)
        objects.append({'cells': tuple(sorted(cells)), 'color': int(a[first])})
    return objects


def _housing(obj):
    """Recognize an open U with solid rails, backing, and a toothed cavity."""
    cells = np.array(obj['cells'])
    found = []
    for normal in CARDINALS:
        n = np.array(normal)
        t = np.array((0,1) if normal[0] else (1,0))
        uv = np.column_stack((cells @ n, cells @ t))
        low = uv.min(axis=0)
        uv -= low
        depth, outer_width = map(int, uv.max(axis=0))
        width = outer_width-1
        if not (3 <= width <= 7 and width % 2 and 2 <= depth <= 8):
            continue
        present = set(map(tuple, uv))
        rails = {(u,v) for u in range(depth+1) for v in (0,width+1)}
        back = {(depth,v) for v in range(width+2)}
        if not rails | back <= present:
            continue
        profile = []
        for v in range(1,width+1):
            occupied = [u for u in range(depth+1) if (u,v) in present]
            p = min(occupied)
            if p < 2 or occupied != list(range(p,depth+1)):
                break
            profile.append(p)
        if len(profile) != width:
            continue
        expected = rails | {(u,v+1) for v,p in enumerate(profile) for u in range(p,depth+1)}
        if expected != present:
            continue
        mid = n*low[0] + t*(low[1]+1+width//2)
        found.append(dict(obj, normal=normal, tangent=tuple(t), midpoint=tuple(map(int,mid)),
                          width=width, depth=depth, profile=tuple(profile)))
    if len(found) > 1:
        raise ValueError('ambiguous housing orientation')
    return found[0] if found else None


def _scene(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= a.size//2:
        raise ValueError('background must be a strict majority')
    sources, frames, pieces = [], [], []
    for obj in _components(a,bg):
        cells = np.array(obj['cells'])
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        if len(cells) == 3 and tuple(hi-lo) == (1,1):
            absent = next(p for p in ((r,c) for r in range(lo[0],hi[0]+1)
                                     for c in range(lo[1],hi[1]+1)) if p not in obj['cells'])
            elbow = tuple(map(int,lo+hi-np.array(absent)))
            sources.append(dict(obj, elbow=elbow, direction=(elbow[0]-absent[0],elbow[1]-absent[1])))
        else:
            frame = _housing(obj)
            if frame:
                frames.append(frame)
            else:
                if len(cells) < 4:
                    raise ValueError('unrecognized small object')
                pieces.append(obj)
    if len(sources) != 1 or not frames or not pieces:
        raise ValueError('expected one source, housings, and loose tiles')
    shell_colors = {f['color'] for f in frames}
    if len(shell_colors) != 1 or any(o['color'] in shell_colors for o in sources+pieces):
        raise ValueError('housing color must be reserved locally')
    return bg, sources, frames, pieces


def _description(piece, frame):
    cells = np.array(piece['cells'])
    n, t = np.array(frame['normal']), np.array(frame['tangent'])
    uv = np.column_stack((cells @ n, cells @ t))
    uv -= uv.min(axis=0)
    if uv[:,1].max()+1 != frame['width']:
        return None
    lo, hi = [], []
    for v in range(frame['width']):
        rows = sorted(int(u) for u,x in uv if x == v)
        if not rows or rows != list(range(rows[0],rows[-1]+1)):
            return None
        lo.append(rows[0])
        hi.append(rows[-1]+1)
    return tuple(lo), tuple(hi), uv


def _fits(frame, pieces, excluded=frozenset()):
    """Enumerate full-width translation chains exactly covering the cavity.

    Different source components remain distinguishable even if their colors and
    masks agree: erasing a different original location changes the full answer.
    """
    descriptions = [_description(p,frame) for p in pieces]
    target = frame['profile']
    solutions = {}
    n, t = np.array(frame['normal']), np.array(frame['tangent'])
    mid = np.array(frame['midpoint'])

    def visit(profile, used, placements):
        if len(solutions) >= 2:
            return
        if profile == target:
            erased = tuple(sorted(p for i,_ in placements for p in pieces[i]['cells']))
            colored = tuple(sorted((p,pieces[i]['color']) for i,dest in placements for p in dest))
            solutions.setdefault((erased,colored),placements)
            return
        offset = min(profile)
        wanted = tuple(x-offset for x in profile)
        for i,d in enumerate(descriptions):
            if i in excluded or i in used or d is None or d[0] != wanted:
                continue
            next_profile = tuple(offset+x for x in d[1])
            if any(x > y for x,y in zip(next_profile,target)):
                continue
            dest = mid + (d[2][:,0]+offset)[:,None]*n + (d[2][:,1]-frame['width']//2)[:,None]*t
            dest = tuple(sorted(map(tuple,dest.tolist())))
            if dest == pieces[i]['cells']:
                continue
            visit(next_profile, used | {i}, placements+[(i,dest)])
    visit((0,)*frame['width'],set(),[])
    return list(solutions.values())


def _run(a):
    bg, sources, frames, pieces = _scene(a)
    fixed = {p for o in sources+frames for p in o['cells']}
    active, used, moves, paint, routes = {}, set(), {}, {}, []
    contact_map = {}
    for i,f in enumerate(frames):
        contact = tuple(x-y for x,y in zip(f['midpoint'],f['normal']))
        contact_map.setdefault(contact,[]).append(i)
    for s in sources:
        position, direction, shade = s['elbow'], s['direction'], s['color']
        visited, contacts, path = set(), [], []
        while True:
            p = position[0]+direction[0],position[1]+direction[1]
            if not (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]):
                break
            state = p,direction
            if state in visited or p in fixed:
                raise ValueError('loop or ray hits a fixed object')
            visited.add(state)
            hits = [i for i in contact_map.get(p,[]) if sum(x*y for x,y in zip(direction,frames[i]['normal'])) == 1]
            if len(hits) > 1:
                raise ValueError('ambiguous midpoint contact')
            if hits:
                i = hits[0]
                f = frames[i]
                if i not in active:
                    options = _fits(f,pieces,used)
                    if len(options) != 1:
                        raise ValueError('reached cavity does not have one unique assembly')
                    placements = options[0]
                    for j,dest in placements:
                        used.add(j)
                        moves[j] = dest
                    # The first full-width tile occupies the entire mouth.
                    active[i] = (pieces[placements[0][0]]['color'],placements)
                shade = active[i][0]
                axis = 0 if f['normal'][0] else 1
                d = list(direction)
                d[axis] *= -1
                direction = tuple(d)
                contacts.append((i,p,shade,axis))
            if p in paint and paint[p] != shade:
                raise ValueError('different beam colors share a cell')
            paint[p] = shade
            path.append(p)
            position = p
        routes.append({'source':s,'contacts':contacts,'path':tuple(path)})
    assembly = a.copy()
    for j in used:
        for p in pieces[j]['cells']:
            assembly[p] = bg
    for j,dest in moves.items():
        for p in dest:
            if assembly[p] != bg:
                raise ValueError('tile overlaps an occupied cell')
            assembly[p] = pieces[j]['color']
    out = assembly.copy()
    for p,shade in paint.items():
        if assembly[p] != bg:
            raise ValueError('beam intersects an installed or unused tile')
        out[p] = shade
    return out,dict(background=bg,sources=sources,frames=frames,pieces=pieces,
                    active=active,used=used,moves=moves,paint=paint,routes=routes,assembly=assembly)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode components, activate reached cavities, fit tiles, and trace beams."""
    return _run(_grid(input_grid))[0]


def _connected(cells):
    left = set(cells)
    if not left:
        return False
    stack = [left.pop()]
    while stack:
        r,c = stack.pop()
        for dr,dc in CARDINALS:
            p = r+dr,c+dc
            if p in left:
                left.remove(p)
                stack.append(p)
    return not left


def _shape(rng,width,count):
    cuts = [np.zeros(width,dtype=int)]
    for i in range(count):
        amplitude = int(rng.integers(1,3 if count == 3 else 4))
        mode = int(rng.integers(4))
        if mode == 0:
            shape = rng.integers(0,amplitude+1,size=width)
        elif mode == 1:
            shape = np.floor(np.linspace(0,amplitude,width)).astype(int)
            if rng.integers(2):
                shape = shape[::-1]
        elif mode == 2:
            shape = np.clip(np.cumsum(rng.integers(-1,2,size=width)),0,amplitude)
        else:
            shape = np.zeros(width,dtype=int)
            a,b = sorted(map(int,rng.choice(width+1,size=2,replace=False)))
            shape[a:b] = amplitude
        shape -= shape.min()
        if i == count-1 and np.ptp(shape) == 0:
            shape[int(rng.integers(width))] += 1
        previous = cuts[-1]
        offset = max(int(np.max(previous+1-shape)),
                     int(np.max(np.maximum(previous[:-1],previous[1:])+1
                                -np.minimum(shape[:-1],shape[1:]))))
        if i == 0:
            offset = max(2,offset)
        next_cut = shape+offset+int(rng.integers(0,2 if count == 3 else 3))
        cuts.append(next_cut)
    profile = tuple(map(int,cuts[-1]))
    depth = max(profile)+int(rng.integers(0,2))
    if depth > 8:
        raise ValueError('housing too deep')
    layers = []
    for low,high in zip(cuts,cuts[1:]):
        layer = tuple((u,v) for v in range(width) for u in range(int(low[v]),int(high[v])))
        if not _connected(layer) or len(layer) < 4:
            raise ValueError('invalid layer')
        layers.append(layer)
    return profile,depth,layers


def _geometry(mid,normal,width,profile,depth):
    nr,nc = normal
    tr,tc = (0,1) if nr else (1,0)
    mr,mc = mid
    shell = []
    box = []
    for u in range(depth+1):
        for v in range(-1,width+1):
            p = mr+u*nr+(v-width//2)*tr,mc+u*nc+(v-width//2)*tc
            box.append(p)
            if v in (-1,width) or u >= profile[v]:
                shell.append(p)
    return tuple(shell),tuple(box)


def _inside(cells,shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r,c in cells)


def _dilated(cells):
    return {(r+dr,c+dc) for r,c in cells for dr,dc in ((0,0),)+CARDINALS}


def _route_layout(rng,axes,shapes,canvas):
    """Search only small geometric placements; no parent generator is invoked."""
    for attempt in range(140):
        direction = (1,1)
        elbow = (int(rng.integers(1,canvas[0]-3)),int(rng.integers(1,canvas[1]-3)))
        source = (elbow,(elbow[0]-1,elbow[1]),(elbow[0],elbow[1]-1))
        occupied, position, path, placements = set(source),elbow,[],[]
        good = True
        for axis,(width,profile,depth,layers) in zip(axes,shapes):
            choices = []
            forbidden_box = _dilated(occupied)
            for distance in range(3,24):
                contact = tuple(position[k]+distance*direction[k] for k in (0,1))
                normal = (direction[0],0) if axis == 0 else (0,direction[1])
                mid = tuple(contact[k]+normal[k] for k in (0,1))
                shell,box = _geometry(mid,normal,width,profile,depth)
                flight = [tuple(position[k]+j*direction[k] for k in (0,1)) for j in range(1,distance+1)]
                if (_inside(box+tuple(flight),canvas)
                        and not set(box) & forbidden_box
                        and not set(box) & set(path+flight)
                        and not set(flight) & occupied):
                    choices.append((contact,normal,mid,shell,box,flight))
            if not choices:
                good = False
                break
            contact,normal,mid,shell,box,flight = choices[int(rng.integers(len(choices)))]
            placements.append((mid,normal,shell,box,layers))
            occupied.update(box)
            path.extend(flight)
            position = contact
            d = list(direction)
            d[axis] *= -1
            direction = tuple(d)
        if not good:
            continue
        tail = []
        while True:
            p = tuple(position[k]+direction[k] for k in (0,1))
            if not _inside([p],canvas):
                break
            if p in occupied or p in path:
                good = False
                break
            tail.append(p)
            position = p
        if good and len(tail) >= 2:
            return source,placements,tuple(path+tail)
    raise ValueError('route placement failed')


def _make_pair(rng,axes,counts,source_anchor=None,first_contact=None):
    if source_anchor is None:
        colors = list(map(int,rng.permutation(10)))
    else:
        available = list(map(int,rng.permutation([c for c in range(10) if c != source_anchor])))
        colors = available[:2]+[source_anchor]+available[2:]
    bg,shell,source_color = colors[:3]
    payload = colors[2:]
    minimum = 16 if len(axes) == 1 else (20 if len(axes) == 2 else 23)
    canvas = (int(rng.integers(minimum,31)),int(rng.integers(minimum,31)))
    shapes = []
    for count in counts:
        width = int(rng.choice([3,5],p=[.72,.28]))
        profile,depth,layers = _shape(rng,width,count)
        shapes.append((width,profile,depth,layers))
    source,placements,path = _route_layout(rng,axes,shapes,canvas)
    base = np.full(canvas,bg,dtype=int)
    for p in source:
        base[p] = source_color
    boxes = {p for item in placements for p in item[3]}
    pieces = []
    for index,(mid,normal,shell_cells,box,layers) in enumerate(placements):
        for p in shell_cells:
            base[p] = shell
        n = np.array(normal)
        t = np.array((0,1) if normal[0] else (1,0))
        width = shapes[index][0]
        for j,layer in enumerate(layers):
            shade = source_color if index == 0 and j == 0 and rng.random() < .35 else int(rng.choice(payload))
            if index == 0 and j == 0 and first_contact is not None:
                shade = source_color if first_contact == 'same' else int(rng.choice(colors[3:]))
            if index == 1 and j == 0:
                shade = int(rng.choice(colors[3:]))
            dest = tuple(sorted(tuple(map(int,np.array(mid)+u*n+(v-width//2)*t)) for u,v in layer))
            pieces.append((dest,shade))
    # A dormant housing has its own spare tile group. It stays unfilled even
    # though complementary inventory is available; proximity is irrelevant.
    dw = int(rng.choice([3,5,7],p=[.65,.30,.05]))
    dp,dd,dl = _shape(rng,dw,int(rng.integers(1,3)))
    forbidden_decoy = _dilated(boxes | set(source) | set(path))
    for attempt in range(600):
        normal = CARDINALS[int(rng.integers(4))]
        mid = int(rng.integers(canvas[0])),int(rng.integers(canvas[1]))
        cells,box = _geometry(mid,normal,dw,dp,dd)
        contact = mid[0]-normal[0],mid[1]-normal[1]
        if _inside(box,canvas) and not set(box) & forbidden_decoy and contact not in path:
            break
    else:
        raise ValueError('dormant housing clearance')
    boxes.update(box)
    for p in cells:
        base[p] = shell
    n = np.array(normal)
    t = np.array((0,1) if normal[0] else (1,0))
    for layer in dl:
        dest = tuple(sorted(tuple(map(int,np.array(mid)+u*n+(v-dw//2)*t)) for u,v in layer))
        pieces.append((dest,int(rng.choice(payload))))
    # All tiles are translated away from destinations, separated as components.
    # Reserve the beam and all cavity boxes so dormant inventory stays visible.
    for attempt in range(20):
        inp = base.copy()
        scattered = {}
        forbidden = np.zeros(canvas,dtype=bool)
        for p in _dilated(boxes | set(source)) | set(path):
            if _inside([p],canvas):
                forbidden[p] = True
        good = True
        order = sorted(range(len(pieces)),key=lambda i:-len(pieces[i][0])+rng.random())
        for i in order:
            dest,shade = pieces[i]
            local = np.array(dest)-np.min(dest,axis=0)
            h,w = local.max(axis=0)+1
            valid = np.ones((canvas[0]-h+1,canvas[1]-w+1),dtype=bool)
            for r,c in local:
                valid &= ~forbidden[r:r+valid.shape[0],c:c+valid.shape[1]]
            options = np.argwhere(valid)
            if not len(options):
                good = False
                break
            cells = tuple(map(tuple,(local+options[int(rng.integers(len(options)))]).tolist()))
            for p in cells:
                inp[p] = shade
            scattered[i] = tuple(sorted(cells))
            for p in _dilated(cells):
                if _inside([p],canvas):
                    forbidden[p] = True
        if good:
            break
    else:
        raise ValueError('tile scattering failed')
    # Whole-scene symmetries preserve translation and the missing-corner code.
    # Check the constructed active translations before any scene symmetry.
    constructed_out,constructed_info = _run(inp)
    actual_moves = {constructed_info['pieces'][j]['cells']: dest
                    for j,dest in constructed_info['moves'].items()}
    intended_moves = {scattered[i]: pieces[i][0] for i in range(sum(counts))}
    if actual_moves != intended_moves:
        raise ValueError('input-derived assembly differs from constructed translations')
    if rng.integers(2):
        inp = inp.T.copy()
    if rng.integers(2):
        inp = inp[::-1].copy()
    if rng.integers(2):
        inp = inp[:,::-1].copy()
    out,info = _run(inp)
    if len(info['active']) != len(axes) or len(info['used']) != sum(counts):
        raise ValueError('route or assembly differs from construction')
    dormant = [f for i,f in enumerate(info['frames']) if i not in info['active']]
    if not any(len(_fits(f,info['pieces'],info['used'])) == 1 for f in dormant):
        raise ValueError('dormant assembly must be a visible alternative')
    return {'input':inp,'output':out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value,low,high,name in ((num_train,2,10,'num_train'),(num_test,1,4,'num_test')):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= int(value) <= high:
            raise ValueError(f'illegal {name}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1<<64))
    puzzle = {'train':[],'test':[]}
    anchor = int(rng.integers(10))
    shift = int(rng.integers(2))  # Route extension or deeper keyed assembly.
    seen = set()
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            if split == 'train' and index == 0:
                axes,counts = [0,1],[1,2]
            elif split == 'train' and index == 1:
                axes,counts = [1,1],[2,2]
            elif split == 'test':
                n = 3 if shift == 0 else 2
                if rng.random() < .6:
                    axes = [int(rng.integers(2))]*n
                else:
                    axes = list(map(int,rng.integers(0,2,size=n)))
                counts = list(map(int,rng.integers(1,3,size=n)))
                if shift == 1:
                    counts[int(rng.integers(n))] = 3
            else:
                n = int(rng.choice([1,2,3],p=[.25,.6,.15]))
                axes = list(map(int,rng.integers(0,2,size=n)))
                counts = list(map(int,rng.integers(1,3,size=n)))
                if n == 1 and rng.random() < .35:
                    counts = [3]
            for attempt in range(80):
                try:
                    teaching = split == 'train' and index < 2
                    pair = _make_pair(rng,axes,counts,anchor if teaching else None,
                                      ('same' if index == 0 else 'change') if teaching else None)
                except ValueError:
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError(f'could not place {axes}/{counts}')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    if not isinstance(pair,dict) or set(pair) != {'input','output'}:
        raise ValueError('pair schema')
    inp,out = _grid(pair['input']),_grid(pair['output'])
    expected,info = _run(inp)
    assert inp.shape == out.shape and np.array_equal(expected,out)
    bg = info['background']
    assert info['active'] and len(info['active']) < len(info['frames'])
    assert 0 < len(info['used']) < len(info['pieces'])
    for o in info['sources']+info['frames']:
        assert all(out[p] == inp[p] for p in o['cells'])
    for i,piece in enumerate(info['pieces']):
        if i not in info['used']:
            assert all(out[p] == piece['color'] for p in piece['cells'])
            continue
        origin = np.array(piece['cells'])
        dest = np.array(info['moves'][i])
        assert np.array_equal(origin-origin.min(axis=0),dest-dest.min(axis=0))
        assert not np.array_equal(origin,dest)
        assert all(out[p] == piece['color'] for p in info['moves'][i])
        assert all(info['assembly'][p] == bg for p in piece['cells'])
    for i,f in enumerate(info['frames']):
        n,t,mid = np.array(f['normal']),np.array(f['tangent']),np.array(f['midpoint'])
        cavity = [tuple(map(int,mid+u*n+(v-f['width']//2)*t)) for v in range(f['width']) for u in range(f['profile'][v])]
        if i in info['active']:
            assert all(info['assembly'][p] != bg for p in cavity)
        else:
            assert all(out[p] == bg for p in cavity)
    # Assembly conserves every color count; only the new beam changes counts.
    assert np.array_equal(np.bincount(inp.ravel(),minlength=10),np.bincount(info['assembly'].ravel(),minlength=10))
    delta = np.zeros(10,dtype=int)
    for shade in info['paint'].values():
        delta[bg] -= 1
        delta[shade] += 1
    assert np.array_equal(np.bincount(out.ravel(),minlength=10)-np.bincount(inp.ravel(),minlength=10),delta)
    assert not np.array_equal(out,info['assembly']), 'omitting ray tracing must change answer'
    assert any(inp[p] != out[p] for i in info['used'] for p in info['pieces'][i]['cells']), 'omitting relocation must change answer'
    dormant_options = [_fits(f,info['pieces'],info['used']) for i,f in enumerate(info['frames']) if i not in info['active']]
    assert any(len(options) == 1 for options in dormant_options), 'a dormant socket must be fillable from unused inventory'
    # A geometry-only all-socket completion changes an untouched cavity pixel,
    # so A determines the selected objects, not merely decorative added lines.
    dormant_move = next(options[0] for options in dormant_options if len(options) == 1)
    assert any(out[p] == bg for _,dest in dormant_move for p in dest)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen,train_axes,train_counts,transitions = set(),set(),set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            key = _key(_grid(pair['input']))
            assert key not in seen
            seen.add(key)
            if split == 'train':
                train_axes.update(c[3] for route in info['routes'] for c in route['contacts'])
                train_counts.update(len(p) for _,p in info['active'].values())
                for route in info['routes']:
                    shade = route['source']['color']
                    for _,_,new_color,_ in route['contacts']:
                        transitions.append((shade,new_color))
                        shade = new_color
    assert train_axes == {0,1}
    assert train_counts >= {1,2}
    assert any(a == b for a,b in transitions) and any(a != b for a,b in transitions)
    successors = {}
    for a,b in transitions:
        successors.setdefault(a,set()).add(b)
    assert any(len(v) > 1 for v in successors.values()), 'training must contradict a hidden successor-color cycle'


def self_test():
    puzzles = [generate_puzzle(s) for s in HOST_SEEDS]
    shapes,structures,widths,depths,profiles,directions,normal_axes,active_counts,tile_counts,palettes = (set() for _ in range(10))
    reused = long_chains = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp = pair['input']
                info = _validate_pair(pair)
                shapes.add(inp.shape)
                mapping = {}
                structures.add((inp.shape,tuple(mapping.setdefault(int(c),len(mapping)) for c in inp.flat)))
                widths.update(f['width'] for f in info['frames'])
                depths.update(f['depth'] for f in info['frames'])
                profiles.update(f['profile'] for f in info['frames'])
                directions.update(s['direction'] for s in info['sources'])
                normal_axes.update(f['normal'] for f in info['frames'])
                active_counts.add(len(info['active']))
                tile_counts.add(len(info['pieces']))
                palettes.add(tuple(sorted(set(map(int,inp.flat)))))
                shades = [p['color'] for p in info['pieces']]
                reused += len(set(shades)) < len(shades)
                long_chains += any(len(p) == 3 for _,p in info['active'].values())
    assert len(shapes) >= 35 and len(structures) >= 200
    assert widths >= {3,5,7} and len(depths) >= 5 and len(profiles) >= 30
    assert directions == {(-1,-1),(-1,1),(1,-1),(1,1)}
    assert normal_axes == set(CARDINALS) and active_counts >= {1,2,3}
    assert len(tile_counts) >= 5 and len(palettes) >= 25 and reused >= 30 and long_chains >= 1
    a,b = generate_puzzle(0),generate_puzzle(0)
    assert all(np.array_equal(x[k],y[k]) for s in ('train','test') for x,y in zip(a[s],b[s]) for k in ('input','output'))
    for train in range(2,11):
        for test in range(1,min(4,train-1)+1):
            p = generate_puzzle(100+train*4+test,train,test)
            assert len(p['train']) == train and len(p['test']) == test
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(4,None)):
        try:
            generate_puzzle(0,*counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A missing tile pixel or a ray pixel with the wrong visible color must fail.
    pair = a['test'][0]
    _,info = _run(pair['input'])
    for cell in (next(iter(info['paint'])),next(iter(info['moves'].values()))[0]):
        bad = {'input':pair['input'],'output':pair['output'].copy()}
        bad['output'][cell] = info['background']
        try:
            _validate_pair(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('damaged label accepted')
    _rule_regressions()


def _rule_regressions():
    """A manually specified small answer, plus symmetries and bad inputs."""
    frame = {'normal':(1,0),'tangent':(0,1),'midpoint':(4,7),
             'width':5,'profile':(4,)*5}
    identical = [{'cells':tuple((r,c+offset) for r in range(2) for c in range(5)),
                  'color':7} for offset in (0,10)]
    assert len(_fits(frame,identical)) == 1, 'same-colored order permutations must deduplicate'
    distinct_colors = [identical[0],dict(identical[1],color=6)]
    assert len(_fits(frame,distinct_colors)) == 2, 'different cap colors must remain ambiguous'
    shallow = dict(frame,profile=(2,)*5)
    assert len(_fits(shallow,identical)) == 2, 'different erased source positions must remain distinct'
    a = np.zeros((16,18),dtype=int)
    for p in ((2,4),(1,4),(2,3)):
        a[p] = 1
    # A right-facing assembly cavity, struck from its left.
    shell,_ = _geometry((5,8),(0,1),3,(4,3,4),4)
    for p in shell:
        a[p] = 8
    # Dormant downward cavity and its one-piece tile remain untouched.
    shell,_ = _geometry((10,5),(1,0),3,(2,3,2),3)
    for p in shell:
        a[p] = 8
    outer = ((0,13),(0,14),(1,13),(2,13),(2,14))
    inner = ((5,16),(5,17),(6,15),(6,16),(7,16),(7,17))
    spare = ((0,8),(1,8),(0,9),(1,9),(2,9),(0,10),(1,10))
    for cells,color in ((outer,2),(inner,3),(spare,4)):
        for p in cells:
            a[p] = color
    expected = a.copy()
    for p in outer+inner:
        expected[p] = 0
    for p in ((4,8),(4,9),(5,8),(6,8),(6,9)):
        expected[p] = 2
    for p in ((4,10),(4,11),(5,9),(5,10),(6,10),(6,11)):
        expected[p] = 3
    for p in ((3,5),(4,6)):
        expected[p] = 1
    for p in ((5,7),(6,6),(7,5),(8,4),(9,3),(10,2),(11,1),(12,0)):
        expected[p] = 2
    assert np.array_equal(solve(a),expected)
    for transform in (lambda x:x.T,lambda x:x[::-1],lambda x:x[:,::-1]):
        assert np.array_equal(solve(transform(a)),transform(expected))
    # Changing only the loose cap hue changes the contact and outgoing flight.
    recolored = a.copy()
    recolored[recolored == 2] = 6
    answer = expected.copy()
    answer[answer == 2] = 6
    assert np.array_equal(solve(recolored),answer)
    # Break the keyed back interface while leaving a connected loose tile.
    broken = a.copy()
    broken[6,15] = 0
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete complementary packing accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds and every legal count combination')
