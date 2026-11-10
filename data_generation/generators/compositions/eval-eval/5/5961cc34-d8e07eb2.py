"""Line-addressed packet relays. All solving conventions are visible in the input.

No parent generator is imported. Geometry construction and pixel parsing are local.
"""
from __future__ import annotations

import itertools
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, found = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    todo.append(p)
        result.append(found)
    return result


def _box(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _rect(box):
    r0, r1, c0, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _step(p, direction):
    dr, dc = DIRS[direction]
    return p[0] + dr, p[1] + dc


def _separate(cells, occupied):
    return not any((r + dr, c + dc) in occupied for r, c in cells
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _separate_box(box,others):
    r0,r1,c0,c1=box
    return all(r1+1 < a or b+1 < r0 or c1+1 < c or d+1 < c0
               for a,b,c,d in others)


def _runs(values):
    result = []
    for v in sorted(values):
        if not result or v != result[-1][-1] + 1:
            result.append([v])
        else:
            result[-1].append(v)
    return result


def _line(positions, n):
    s = set(positions)
    if len(s) != n:
        return None
    rows, cols = {p // n for p in s}, {p % n for p in s}
    if len(rows) == 1:
        return 'row', next(iter(rows))
    if len(cols) == 1:
        return 'col', next(iter(cols))
    return None


def _token_key(a, carrier):
    mask = a != carrier
    assert mask.any() and len(np.unique(a[mask])) == 1
    rr, cc = np.where(mask)
    crop = a[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    return _key(crop)


def _parse(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    bg = int(a[0, 0])
    assert all(int(a[p]) == bg for p in ((0, w-1), (h-1, 0), (h-1, w-1)))
    components = _components(map(tuple, np.argwhere(a != bg)))
    boundary = [x for x in components if any(r in (0, h-1) or c in (0, w-1) for r, c in x)]
    assert len(boundary) == 1
    launch = boundary[0]
    # Canonicalize only by rotation. Reflected scenes require no extra convention.
    sides = [d for d, pred in enumerate((lambda r,c: r == h-1,
             lambda r,c: c == 0, lambda r,c: r == 0, lambda r,c: c == w-1))
             if any(pred(r,c) for r,c in launch)]
    assert len(sides) == 1
    initial = sides[0]
    rotation = (initial - 1) % 4
    canonical = np.rot90(a, rotation).copy()
    if rotation:
        # Now the packet is on the left boundary and the tail points right.
        info = _parse(canonical)
        info['rotation'] = rotation
        return info
    r0, r1, c0, c1 = _box(launch)
    assert c0 == 0 and r0 > 0 and r1 < h-1
    card_width = 0
    while card_width <= c1 and all((r, card_width) in launch for r in range(r0, r1+1)):
        card_width += 1
    card = (r0, r1, 0, card_width-1)
    extension = launch - _rect(card)
    assert len(extension) in (3, 4) and len({r for r,c in extension}) == 1
    lane = next(iter(extension))[0]
    assert r0 < lane < r1
    ordered = sorted(extension)
    assert [c for r,c in ordered] == list(range(card_width, c1+1))
    signal, port_color = int(canonical[ordered[0]]), int(canonical[ordered[-1]])
    carrier = int(canonical[r0, 0])
    assert len({bg, carrier, signal, port_color}) == 4
    assert all(int(canonical[p]) == signal for p in ordered[:-1])
    boards, n_seen, dims = [], set(), set()
    for cells in components:
        if cells == launch:
            continue
        box = _box(cells)
        x0, x1, y0, y1 = box
        assert cells == _rect(box) and 0 < x0 <= x1 < h-1 and 0 < y0 <= y1 < w-1
        assert all(int(canonical[p]) == carrier for p in ((x0,y0),(x0,y1),(x1,y0),(x1,y1)))
        ports = {}
        edges = (([(x0,c) for c in range(y0+1,y1)], False),
                 ([(r,y1) for r in range(x0+1,x1)], True),
                 ([(x1,c) for c in range(y0+1,y1)], False),
                 ([(r,y0) for r in range(x0+1,x1)], True))
        for side, (edge, horizontal) in enumerate(edges):
            assert all(int(canonical[p]) in (carrier, port_color) for p in edge)
            positions = [p[0] if horizontal else p[1] for p in edge if int(canonical[p]) == port_color]
            if positions:
                groups = _runs(positions)
                assert len(groups) in (2, 3)
                ports[side] = [tuple((v,y1) if side == 1 else (v,y0) if side == 3
                              else (x0,v) if side == 0 else (x1,v) for v in g) for g in groups]
        assert len(ports) == 2 and len({s % 2 for s in ports}) == 2
        counts = {len(v) for v in ports.values()}
        assert len(counts) == 1
        n = counts.pop()
        assert (x1-x0-2) % n == 0 and (y1-y0-2) % n == 0
        gh, gw = (x1-x0-2)//n-1, (y1-y0-2)//n-1
        assert 2 <= gh <= 3 and 2 <= gw <= 3
        expected = np.full((x1-x0+1, y1-y0+1), carrier, np.int8)
        tokens, lookup, trimmed = [], {}, set()
        for i, j in itertools.product(range(n), repeat=2):
            rr, cc = 2+i*(gh+1), 2+j*(gw+1)
            token = canonical[x0+rr:x0+rr+gh, y0+cc:y0+cc+gw].copy()
            assert not set(map(int,np.unique(token))) & {bg, port_color, signal}
            tk = _token_key(token, carrier)
            assert tk not in trimmed
            trimmed.add(tk)
            lookup[_key(token)] = i*n+j
            tokens.append(token)
            expected[rr:rr+gh, cc:cc+gw] = token
        for side, groups in ports.items():
            g = gh if side % 2 else gw
            for index, group in enumerate(groups):
                assert 1 <= len(group) <= min(2, g)
                span = [p[0]-x0 if side % 2 else p[1]-y0 for p in group]
                assert 2+index*(g+1) <= min(span) <= max(span) < 2+index*(g+1)+g
                for r,c in group:
                    expected[r-x0,c-y0] = port_color
        assert np.array_equal(expected, canonical[x0:x1+1,y0:y1+1])
        boards.append({'box': box, 'ports': ports, 'tokens': tokens, 'lookup': lookup})
        n_seen.add(n)
        dims.add((gh,gw))
    assert boards and len(n_seen) == len(dims) == 1
    n = n_seen.pop()
    gh, gw = dims.pop()
    assert card_width == gw+2 and r1-r0+1 == n*(gh+1)+1
    query = []
    card_expected = np.full((r1-r0+1,card_width), carrier,np.int8)
    for i in range(n):
        rr = 1+i*(gh+1)
        token = canonical[r0+rr:r0+rr+gh,1:1+gw].copy()
        _token_key(token,carrier)
        assert not set(map(int,np.unique(token))) & {bg,port_color,signal}
        query.append(token)
        card_expected[rr:rr+gh,1:1+gw] = token
    assert len({_key(t) for t in query}) == n
    assert np.array_equal(card_expected,canonical[r0:r1+1,:card_width])
    for board in boards:
        assert all(_key(t) in board['lookup'] for t in query)
        board['selected'] = [board['lookup'][_key(t)] for t in query]
        board['line'] = _line(board['selected'],n)
    assert all(_separate(c, set().union(*(d for d in components if d is not c))) for c in components)
    return {'grid':canonical,'rotation':0,'bg':bg,'signal':signal,'carrier':carrier,
            'port_color':port_color,'launcher':launch,'cap':ordered[-1],
            'boards':boards,'n':n,'glyph':(gh,gw),'card':card}


def _walk(h, w, launcher, cap, boards, force=False):
    """Advance beam fronts against the ORIGINAL solid plates, including decoys."""
    owner = {p:i for i,b in enumerate(boards) for p in _rect(b['box'])}
    owner.update({p:-2 for p in launcher})
    trace, route, events, widths, segments = set(launcher), [], [], [], []
    direction, front, segment = 1, [_step(cap,1)], set()
    for _ in range(h*w*(len(boards)+2)):
        inside = [0 <= r < h and 0 <= c < w for r,c in front]
        if not any(inside):
            segments.append(segment)
            return trace, {'route':route,'events':events,'widths':widths,'segments':segments,'terminal':'edge'}
        assert all(inside), 'partial boundary exit'
        hits = {owner[p] for p in front if p in owner}
        if not hits:
            trace.update(front)
            segment.update(front)
            front = [_step(p,direction) for p in front]
            continue
        assert len(hits) == 1
        index = hits.pop()
        assert index >= 0 and index not in route, 'launcher contact or cyclic route'
        assert all(owner.get(p) == index for p in front), 'partial plate contact'
        segments.append(segment)
        segment = set()
        board = boards[index]
        route.append(index)
        trace.update(_rect(board['box']))
        line = board['line']
        if line is None and not force:
            events.append('absorb')
            return trace, {'route':route,'events':events,'widths':widths,'segments':segments,'terminal':'absorb'}
        if line is None:
            line = ('row',0)
        kind, number = line
        events.append(kind)
        side = next(s for s in board['ports'] if (s % 2 == 1) == (kind == 'row'))
        origins = board['ports'][side][number]
        direction = side
        widths.append(len(origins))
        front = [_step(p,side) for p in origins]
    raise AssertionError('unterminated ray')


def _predict(info, force=False):
    a = info['grid']
    trace, facts = _walk(*a.shape,info['launcher'],info['cap'],info['boards'],force)
    out = np.full_like(a,info['bg'])
    for p in trace:
        out[p] = info['signal']
    facts.update(n=info['n'],glyph=info['glyph'],shape=tuple(a.shape),
                 decoys=len(info['boards'])-len(facts['route']),
                 palette=(info['bg'],info['carrier'],info['port_color'],info['signal']))
    return np.rot90(out,-info['rotation']).copy(), facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the packet, each local board and addressed ports from pixels only."""
    return _predict(_parse(input_grid))[0]


def _bank(gh,gw):
    result = {}
    for code in range(1,1 << (gh*gw)):
        mask = np.array([(code >> k)&1 for k in range(gh*gw)],bool).reshape(gh,gw)
        rr,cc = np.where(mask)
        crop = mask[rr.min():rr.max()+1,cc.min():cc.max()+1]
        centered = np.zeros((gh,gw),bool)
        r,c = (gh-crop.shape[0])//2,(gw-crop.shape[1])//2
        centered[r:r+crop.shape[0],c:c+crop.shape[1]] = crop
        result[(crop.shape,crop.tobytes())] = centered
    return list(result.values())


def _port_table(rng,box,n,gh,gw,h_side=None,v_side=None):
    r0,r1,c0,c1 = box
    sides = (int(rng.choice([1,3])) if h_side is None else h_side,
             int(rng.choice([0,2])) if v_side is None else v_side)
    ports = {}
    for side in sides:
        g = gh if side % 2 else gw
        groups = []
        for i in range(n):
            length = int(rng.integers(1,min(g,2)+1))
            start = 2+i*(g+1)+int(rng.integers(g-length+1))
            groups.append(tuple((r0+v,c1) if side == 1 else (r0+v,c0) if side == 3
                          else (r0,c0+v) if side == 0 else (r1,c0+v)
                          for v in range(start,start+length)))
        ports[side] = groups
    return ports


def _set_port_width(rng,box,ports,side,index,length,gh,gw):
    r0,r1,c0,c1=box
    g=gh if side % 2 else gw
    start=2+index*(g+1)+int(rng.integers(g-length+1))
    ports[side][index]=tuple((r0+v,c1) if side==1 else (r0+v,c0) if side==3
                            else (r0,c0+v) if side==0 else (r1,c0+v)
                            for v in range(start,start+length))


def _make_launcher(rng,h,n,gh,gw,fixed=False):
    ph,pw = n*(gh+1)+1,gw+2
    r0 = 1 if fixed else int(rng.integers(1,h-ph))
    row = r0+1 if fixed else int(rng.integers(r0+1,r0+ph-1))
    length = 2 if fixed else int(rng.integers(2,4))
    card = (r0,r0+ph-1,0,pw-1)
    tail = [(row,pw+k) for k in range(length)]
    cap = (row,pw+length)
    return card,tail,cap,_rect(card)|set(tail)|{cap}


def _random_layout(rng,n,gh,gw,count,decoys,absorb,lesson):
    bh,bw = n*(gh+1)+3,n*(gw+1)+3
    lower_h = max(18,2*bh+3) if count >= 2 or decoys else bh+4
    lower_w = max(22,2*bw+3) if count >= 2 or decoys else bw+gw+10
    if max(lower_h,lower_w)>30:
        return None
    h,w = int(rng.integers(lower_h,31)),int(rng.integers(lower_w,31))
    card,tail,cap,launcher = _make_launcher(rng,h,n,gh,gw)
    budget = [350]
    lesson_widths=list(map(int,rng.permutation([1,2]))) if lesson=='both' else None

    def grow(boards,front,direction,occupied,occupied_boxes,old_rays):
        i = len(boards)
        options = []
        for gap in range(1,22):
            target = (front[0][0]+gap*DIRS[direction][0],front[0][1]+gap*DIRS[direction][1])
            extent = bh if direction % 2 else bw
            for entry in range(1,extent-len(front)):
                if direction == 1:
                    r,c = target[0]-entry,target[1]
                elif direction == 3:
                    r,c = target[0]-entry,target[1]-bw+1
                elif direction == 2:
                    r,c = target[0],target[1]-entry
                else:
                    r,c = target[0]-bh+1,target[1]-entry
                box = (r,r+bh-1,c,c+bw-1)
                if r<1 or c<1 or r+bh>=h or c+bw>=w:
                    continue
                if not _separate_box(box,occupied_boxes):
                    continue
                cells = _rect(box)
                approach = { (rr+k*DIRS[direction][0],cc+k*DIRS[direction][1])
                             for rr,cc in front for k in range(gap)}
                if cells & old_rays or approach & occupied:
                    continue
                options.append((box,cells,approach))
        rng.shuffle(options)
        for box,cells,approach in options[:45]:
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            kinds = ['row','col']
            if lesson == 'both':
                kinds = ['col' if i == 0 else 'row']
            elif lesson == 'absorb' and i == 0:
                kinds = ['col']
            rng.shuffle(kinds)
            for kind in kinds:
                sides = [s for s in (1,3) if s != (direction+2)%4] if kind == 'row' else [s for s in (0,2) if s != (direction+2)%4]
                rng.shuffle(sides)
                for side in sides:
                    ports = _port_table(rng,box,n,gh,gw,
                                        h_side=side if kind=='row' else None,
                                        v_side=side if kind=='col' else None)
                    numbers = list(map(int,rng.permutation(n)))
                    for number in numbers:
                        if lesson_widths:
                            _set_port_width(rng,box,ports,side,number,lesson_widths[i],gh,gw)
                        line = None if i == count-1 and absorb else (kind,number)
                        board = {'box':box,'ports':ports,'line':line}
                        proposed = boards+[board]
                        if i+1 == count:
                            try:
                                trace,facts = _walk(h,w,launcher,cap,proposed)
                            except AssertionError:
                                continue
                            if len(facts['route']) == count:
                                return proposed,trace
                        else:
                            next_front = sorted(_step(p,side) for p in ports[side][number])
                            result = grow(proposed,next_front,side,occupied|cells,occupied_boxes+[box],old_rays|approach)
                            if result is not None:
                                return result
        return None

    launcher_boxes=[card,_box(set(tail)|{cap})]
    found = grow([],[_step(cap,1)],1,launcher,launcher_boxes,set(launcher))
    if found is None:
        return None
    boards,trace = found
    occupied = launcher|set().union(*(_rect(b['box']) for b in boards))
    occupied_boxes=launcher_boxes+[b['box'] for b in boards]
    for _ in range(decoys):
        candidates = [(r,c) for r in range(1,h-bh) for c in range(1,w-bw)]
        rng.shuffle(candidates)
        for r,c in candidates:
            box = (r,r+bh-1,c,c+bw-1)
            if not _separate_box(box,occupied_boxes):
                continue
            cells = _rect(box)
            if not cells & trace:
                ports = _port_table(rng,box,n,gh,gw)
                boards.append({'box':box,'ports':ports,'line':(str(rng.choice(['row','col'])),int(rng.integers(n)))})
                occupied |= cells
                occupied_boxes.append(box)
                break
        else:
            return None
    return h,w,card,tail,cap,launcher,boards,False


def _fallback_layout(rng,n,gh,gw,count,decoys,absorb,lesson=None):
    """Finite construction preserves the intended hit count and absorbing outcome."""
    h=w=30
    bh,bw = n*(gh+1)+3,n*(gw+1)+3
    card,tail,cap,launcher = _make_launcher(rng,h,n,gh,gw,True)
    right,bottom = w-bw-1,h-bh-2
    left = gw+3 if n==2 else 2
    boxes = [(1,1+bh-1,right,right+bw-1),
             (bottom,bottom+bh-1,right,right+bw-1),
             (bottom,bottom+bh-1,left,left+bw-1),
             (4,4+bh-1,left,left+bw-1)]
    assert count <= (4 if n==2 else 3)
    boards=[]
    lesson_widths=list(map(int,rng.permutation([1,2]))) if lesson=='both' else None
    for i in range(count):
        if i == 0:
            kind,side = 'col',2
        elif i == 1:
            kind,side = 'row',3 if count>2 else 1
        elif i == 2 and count>3:
            kind,side = 'col',0
        else:
            kind,side = ('row',3) if i==2 else ('col',0)
        ports = _port_table(rng,boxes[i],n,gh,gw,
                            h_side=side if kind=='row' else None,
                            v_side=side if kind=='col' else None)
        number = int(rng.integers(n))
        if lesson_widths:
            _set_port_width(rng,boxes[i],ports,side,number,lesson_widths[i],gh,gw)
        boards.append({'box':boxes[i],'ports':ports,'line':None if i==count-1 and absorb else (kind,number)})
    # The fallback is called only with <=1 decoy, and none for four-hit scenes.
    if decoys:
        assert decoys==1 and count<=3 and not (n==3 and count==3)
        index = 2 if count<=2 else 3
        boards.append({'box':boxes[index],'ports':_port_table(rng,boxes[index],n,gh,gw),
                       'line':('row',int(rng.integers(n)))})
    _,facts = _walk(h,w,launcher,cap,boards)
    assert len(facts['route'])==count
    occupied=set(launcher)
    for b in boards:
        cells=_rect(b['box'])
        assert _separate(cells,occupied)
        occupied |= cells
    return h,w,card,tail,cap,launcher,boards,True


def _render(rng,layout,n,gh,gw,palette):
    h,w,card,tail,cap,launcher,boards,_ = layout
    bg,carrier,port,signal = palette[:4]
    inks=palette[4:]
    bank=_bank(gh,gw)
    selected_ids=list(map(int,rng.choice(len(bank),n,replace=False)))
    query=[]
    for i,index in enumerate(selected_ids):
        token=np.full((gh,gw),carrier,np.int8)
        token[bank[index]]=inks[0 if i<2 else int(rng.integers(len(inks)))]
        query.append(token)
    available=[]
    for mask in bank:
        for ink in inks:
            t=np.full((gh,gw),carrier,np.int8)
            t[mask]=ink
            if _key(t) not in {_key(q) for q in query}:
                available.append(t)
    # Same shape, different color: explicitly an unselected local-board token.
    confusable=query[0].copy()
    confusable[confusable!=carrier]=inks[1]
    a=np.full((h,w),bg,np.int8)
    r0,r1,c0,c1=card
    a[r0:r1+1,c0:c1+1]=carrier
    order=list(map(int,rng.permutation(n)))
    for i,q in enumerate(order):
        rr=r0+1+i*(gh+1)
        a[rr:rr+gh,1:1+gw]=query[q]
    for p in tail:
        a[p]=signal
    a[cap]=port
    for board in boards:
        r0,r1,c0,c1=board['box']
        a[r0:r1+1,c0:c1+1]=carrier
        line=board['line']
        if line:
            kind,index=line
            positions=[index*n+j for j in range(n)] if kind=='row' else [j*n+index for j in range(n)]
        elif n==2:
            positions=[0,3] if rng.random()<.5 else [1,2]
        else:
            # Full-count near miss, not a partial packet or a new diagonal convention.
            index=int(rng.integers(n))
            missing=int(rng.integers(n))
            positions=[index*n+j for j in range(n) if j!=missing]
            positions.append(((index+int(rng.integers(1,n)))%n)*n+missing)
        assignment=[None]*(n*n)
        for p,q in zip(positions,rng.permutation(n)):
            assignment[p]=query[int(q)]
        empty=[p for p in range(n*n) if assignment[p] is None]
        rng.shuffle(empty)
        assignment[empty.pop()]=confusable
        extras=[t for t in available if _key(t)!=_key(confusable)]
        for p,index in zip(empty,rng.choice(len(extras),len(empty),replace=False)):
            assignment[p]=extras[int(index)]
        for i,t in enumerate(assignment):
            rr,cc=r0+2+(i//n)*(gh+1),c0+2+(i%n)*(gw+1)
            a[rr:rr+gh,cc:cc+gw]=t
        for groups in board['ports'].values():
            for group in groups:
                for p in group:
                    a[p]=port
    trace,_=_walk(h,w,launcher,cap,boards)
    out=np.full_like(a,bg)
    for p in trace:
        out[p]=signal
    return {'input':a,'output':out}


def _count(value,name,lo,hi):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo<=int(value)<=hi:
        raise ValueError(f'{name} must be an integer from {lo} through {hi}')
    return int(value)


def generate_puzzle(seed: int,num_train: int=4,num_test: int=1) -> dict:
    nt,nq=_count(num_train,'num_train',2,10),_count(num_test,'num_test',1,4)
    if nt<=nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    rng=np.random.default_rng(int(seed) % (1<<64))
    n=int(rng.choice([2,3]))
    gh,gw=(2,2) if n==3 else [(2,2),(2,3),(3,2),(3,3)][int(rng.integers(4))]
    base=list(map(int,rng.permutation(10)))
    ink_count=int(rng.integers(2,6))
    vary_colors=bool(rng.random()<.35)
    orientation=int(rng.integers(8))
    mixed=nt>=3 and bool(rng.random()<.45)
    puzzle={'train':[],'test':[]}
    seen=set()
    for split,count in (('train',nt),('test',nq)):
        for i in range(count):
            lesson=('both' if i==0 else 'absorb' if i==1 else None) if split=='train' else None
            hits=2 if lesson else int(rng.choice([1,2])) if split=='train' else (3 if n==3 else int(rng.choice([3,4])))
            absorb=lesson=='absorb' or (lesson is None and bool(rng.random()<.5))
            decoys=1 if lesson=='both' else int(rng.integers(2))
            if hits==4 or (n==3 and hits==3):
                decoys=0
            palette=(base if not vary_colors or (split=='train' and i==0) else
                     list(np.roll(base,1)) if split=='train' and i==1 else list(map(int,rng.permutation(10))))[:4+ink_count]
            for unique_attempt in range(12):
                layout=None
                # Hold hit count fixed during geometry retries.
                for _ in range(14):
                    layout=_random_layout(rng,n,gh,gw,hits,decoys,absorb,lesson)
                    if layout is not None:
                        break
                if layout is None:
                    # A single-hit fallback is a subset of the two-hit layout.
                    if hits==1:
                        layout=_fallback_layout(rng,n,gh,gw,2,0,absorb)
                        h,w,card,tail,cap,launcher,boards,f=layout
                        boards=boards[:1]
                        if absorb:
                            boards[0]['line']=None
                        if decoys:
                            box=(h-(n*(gh+1)+3)-2,h-3,2,2+n*(gw+1)+2)
                            boards.append({'box':box,'ports':_port_table(rng,box,n,gh,gw),'line':('row',0)})
                        layout=(h,w,card,tail,cap,launcher,boards,f)
                    else:
                        layout=_fallback_layout(rng,n,gh,gw,hits,decoys,absorb,lesson)
                pair=_render(rng,layout,n,gh,gw,palette)
                o=int(rng.integers(8)) if mixed and (i>=2 or split=='test') else orientation
                pair={k:np.rot90(np.fliplr(v) if o>=4 else v,o%4).copy() for k,v in pair.items()}
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('input uniqueness budget exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    puzzle['train']=[puzzle['train'][int(i)] for i in rng.permutation(nt)]
    puzzle['test']=[puzzle['test'][int(i)] for i in rng.permutation(nq)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    nt,nq=len(puzzle['train']),len(puzzle['test'])
    assert 2<=nt<=10 and 1<=nq<=4 and nt>nq
    seen,facts,infos=set(),{'train':[],'test':[]},{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp,out=_grid(pair['input']),_grid(pair['output'])
            assert inp.shape==out.shape and not np.array_equal(inp,out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info=_parse(inp)
            prediction,f=_predict(info)
            assert np.array_equal(prediction,out)
            assert set(map(int,np.unique(out)))=={info['bg'],info['signal']}
            # Local matching always needs both occupancy and ink, not color alone.
            for b in info['boards']:
                selected=set(b['selected'])
                shapes=[_key(t!=info['carrier']) for t in b['tokens']]
                assert any(shapes[p]==shapes[q] for p in selected for q in range(len(shapes)) if q not in selected)
                selected_inks=[int(t[t!=info['carrier']][0]) for t in (b['tokens'][p] for p in selected)]
                assert len(set(selected_inks))<len(selected_inks)
            facts[split].append(f)
            infos[split].append(info)
    train=facts['train']
    assert any({'row','col'}<=set(f['events']) for f in train)
    assert any(f['terminal']=='absorb' and len(f['route'])>=2 for f in train)
    assert any(f['decoys']>0 for f in train)
    assert {v for f in train for v in f['widths']}=={1,2}
    # A removed: all locally passing boards would survive, including an off-route one.
    assert any(any(b['line'] is not None and i not in f['route'] for i,b in enumerate(info['boards']))
               for info,f in zip(infos['train'],train))
    # B removed: broadcasting from every marked exit on the absorbing plate
    # paints at least one new adjacent background cell. This is a concrete
    # pixel difference, without assigning any convention to possible later cycles.
    changed=False
    for info,f in zip(infos['train'],train):
        if f['terminal']=='absorb':
            base=np.rot90(_predict(info)[0],info['rotation'])
            b=info['boards'][f['route'][-1]]
            adjacent=[_step(p,s) for s,groups in b['ports'].items() for group in groups for p in group]
            assert all(info['grid'][p]==info['bg'] for p in adjacent)
            changed |= any(base[p]==info['bg'] for p in adjacent)
    assert changed
    assert all(len(f['route'])>=3 for f in facts['test'])
    assert max(len(f['route']) for f in train)<=2
    assert len({(f['n'],f['glyph']) for f in train+facts['test']})==1


def self_test():
    """Bounded semantic, variation, count, determinism and metamorphic checks."""
    puzzles=[]
    records=[]
    for seed in range(42):
        p=generate_puzzle(seed)
        puzzles.append(p)
        repeat=generate_puzzle(seed)
        assert all(np.array_equal(x[k],y[k]) for s in p for x,y in zip(p[s],repeat[s]) for k in x)
        for s in p:
            for pair in p[s]:
                info=_parse(pair['input'])
                out,f=_predict(info)
                assert np.array_equal(out,pair['output'])
                records.append((s,f))
    assert len({tuple(_key(pair['input']) for s in p for pair in p[s]) for p in puzzles})==42
    fs=[f for s,f in records]
    assert {f['n'] for f in fs}=={2,3}
    assert {f['glyph'] for f in fs}>={(2,2),(2,3),(3,2),(3,3)}
    assert len({f['shape'] for f in fs})>=15
    assert len({f['palette'] for f in fs})>=25
    assert {f['terminal'] for s,f in records if s=='test'}=={'edge','absorb'}
    assert {len(f['route']) for s,f in records if s=='test'}=={3,4}
    assert {v for f in fs for v in f['widths']}=={1,2}
    # Rotation/reflection acts on packet, local boards, ports and traces together.
    for pair in puzzles[0]['train']+puzzles[0]['test']:
        for o in range(8):
            transform=lambda a:np.rot90(np.fliplr(a) if o>=4 else a,o%4).copy()
            assert np.array_equal(solve(transform(pair['input'])),transform(pair['output']))
        info=_parse(pair['input'])
        permuted=info['grid'].copy()
        r0,_,_,_=info['card']
        gh,gw=info['glyph']
        blocks=[permuted[r0+1+i*(gh+1):r0+1+i*(gh+1)+gh,1:1+gw].copy()
                for i in range(info['n'])]
        for i,block in enumerate(blocks[1:]+blocks[:1]):
            rr=r0+1+i*(gh+1)
            permuted[rr:rr+gh,1:1+gw]=block
        assert np.array_equal(solve(permuted),np.rot90(pair['output'],info['rotation']))
    # Exercise every legal count combination, including the minimal demonstrations.
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p=generate_puzzle(173,nt,nq)
            assert len(p['train'])==nt and len(p['test'])==nq
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged={s:[{k:v.copy() for k,v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged['test'][0]['output'][0,0]=(int(damaged['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt trace accepted')


if __name__=='__main__':
    self_test()
    print('self_test passed (42 seeds and every legal count combination)')
