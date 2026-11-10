import hashlib
import numpy as np

_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    def checked(
        name: str,
        value: int | None,
        minimum: int,
        maximum: int,
    ) -> int | None:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer or None")
        result = int(value)
        if not minimum <= result <= maximum:
            raise ValueError(
                f"{name} must be between {minimum} and {maximum}"
            )
        return result

    train_count = checked("num_train", num_train, 2, 10)
    test_count = checked("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = (
            test_count
            + 1
            + (seed // 4) % (10 - test_count)
        )
    elif train_count is None:
        minimum = max(2, test_count + 1)
        train_count = int(rng.integers(minimum, 11))
    elif test_count is None:
        maximum = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count

def _turn(direction: int, side: str) -> int:
    if side == "L":
        return (direction - 1) % 4
    return (direction + 1) % 4

def _trace_field(
    field: np.ndarray,
    background: int,
    source: int,
    source_box: tuple[int, int, int],
    turns: dict[int, str],
) -> tuple[np.ndarray, tuple, tuple]:
    height, width = field.shape
    source_row, source_col, thickness = source_box
    output = field.copy()

    key_values = np.array(sorted(turns), dtype=field.dtype)
    chains = []
    events = []

    for initial_direction in range(4):
        row = source_row
        col = source_col
        direction = initial_direction

        seen_states = set()
        chain = []

        while True:
            state = (row, col, direction)
            if state in seen_states:
                raise AssertionError("a beam entered a cycle")
            seen_states.add(state)

            if direction == 0:
                if row == 0:
                    break
                front_rows = np.full(thickness, row - 1, dtype=int)
                front_cols = np.arange(col, col + thickness)

            elif direction == 1:
                if col + thickness == width:
                    break
                front_rows = np.arange(row, row + thickness)
                front_cols = np.full(
                    thickness, col + thickness, dtype=int
                )

            elif direction == 2:
                if row + thickness == height:
                    break
                front_rows = np.full(
                    thickness, row + thickness, dtype=int
                )
                front_cols = np.arange(col, col + thickness)

            else:
                if col == 0:
                    break
                front_rows = np.arange(row, row + thickness)
                front_cols = np.full(thickness, col - 1, dtype=int)

            values = field[front_rows, front_cols]
            is_key = np.isin(values, key_values)

            if np.any(is_key):
                _require(
                    np.all(is_key),
                    "a beam only partially struck a blocker",
                )

                blocker_color = int(values[0])
                _require(
                    np.all(values == blocker_color),
                    "a beam struck two blockers simultaneously",
                )

                old_direction = direction
                direction = _turn(
                    direction,
                    turns[blocker_color],
                )
                chain.append(blocker_color)
                events.append(
                    (
                        initial_direction,
                        blocker_color,
                        old_direction,
                        direction,
                        row,
                        col,
                    )
                )
                continue

            _require(
                np.all(
                    (values == background)
                    | (values == source)
                ),
                "a beam met an unkeyed object",
            )

            paint_mask = values == background
            output[
                front_rows[paint_mask],
                front_cols[paint_mask],
            ] = source

            delta_row, delta_col = _DIRECTIONS[direction]
            row += delta_row
            col += delta_col

        chains.append(tuple(chain))

    return output, tuple(chains), tuple(events)

def _grid_signature(grid: np.ndarray) -> tuple:
    normalized = grid.astype(np.int8, copy=False)
    return normalized.shape, normalized.tobytes()

def _puzzles_equal(first: dict, second: dict) -> bool:
    if list(first) != list(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for first_pair, second_pair in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                first_pair["input"],
                second_pair["input"],
            ):
                return False
            if not np.array_equal(
                first_pair["output"],
                second_pair["output"],
            ):
                return False

    return True

_SOURCE_PAIRS = (('555555555555555555555555\n510005560005540005520005\n510005560005540005520005\n510005560005540005520005\n510005560005540005520005\n555555555555555555555555\n333333333333333333333333\n333333333333333333333333\n333333333333333333333333\n111133333333333333333333\n111133333333333333444433\n111133333333333333444433\n111133333333333333444433\n111133338833333333444433\n111133338833333333444433\n111133333333333333444433\n111133333333333333444433\n111133333333333333333333\n333333333333333333333333\n333333333333333333333333\n333333333333333333333333\n333333333333333333333333\n333322222222222333333333\n333322222222222333333333\n333322222222222333333333', '333333338833333388333333\n333333338833333388333333\n333333338833333388333333\n111133338833333388333333\n111133338833333388444433\n111133338833333388444433\n111133338833333388444433\n111188888888888888444433\n111188888888888888444433\n111188338833333333444433\n111188338833333333444433\n111188338833333333333333\n333388338833333333333333\n333388338833333333333333\n333388888888888888888888\n333388888888888888888888\n333322222222222333333333\n333322222222222333333333\n333322222222222333333333'), ('555555555555555555555555\n500025540005500065530005\n500025540005500065530005\n500025540005500065530005\n500025540005500065530005\n555555555555555555555555\n111111111111166666111111\n111111111111166666111111\n111111111111111111111111\n111111111111111111111111\n111111111111111111111111\n222211111111111133331111\n222211111111111133331111\n222211111111111133331111\n222211118811111133331111\n222211118811111133331111\n222211111111111133331111\n222211111111111111111111\n111111111111111111111111\n111111111111111111111111\n111111111111111111111111\n111111111111111111111111\n111114444444411111111111\n111114444444411111111111\n111114444444411111111111', '111188118811166666111111\n111188118811166666111111\n111188118811118888888888\n111188118811118888888888\n111188118811118811111111\n222288118811118833331111\n222288118811118833331111\n222288118811118833331111\n222288888888888833331111\n222288888888888833331111\n222211118811111133331111\n222211118811111111111111\n111111118811111111111111\n111111118811111111111111\n111111118888888888888888\n111111118888888888888888\n111114444444411111111111\n111114444444411111111111\n111114444444411111111111'), ('555555555555555555555555\n500045560005530005500025\n500045560005530005500025\n500045560005530005500025\n500045560005530005500025\n555555555555555555555555\n777777773333333333777777\n777777773333333333777777\n777777773333333333777777\n777777773333333333776666\n777777777777777777776666\n777777777777777777776666\n777777777777777777776666\n222777777777777777776666\n222777777777777777776666\n222777777777777777776666\n222777777777777777776666\n222777777777887777776666\n222777777777887777776666\n222777777777777777776666\n222777777777777777777777\n222777777777777777777777\n222777774444444444444477\n222777774444444444444477\n222777774444444444444477', '777887773333333333887777\n777887773333333333887777\n777887773333333333887777\n777887773333333333886666\n888888888888887777886666\n888888888888887777886666\n777887777777887777886666\n222887777777887777886666\n222887777777887777886666\n222887777777887777886666\n222887777777887777886666\n222888888888888888886666\n222888888888888888886666\n222887777777887777776666\n222888888888887777777777\n222888888888887777777777\n222777774444444444444477\n222777774444444444444477\n222777774444444444444477'), ('555555555555555555555555\n510005500035520005540005\n510005500035520005540005\n510005500035520005540005\n510005500035520005540005\n555555555555555555555555\n222222222222222222222333\n222222222222222222222333\n222222222222222222222333\n999999999999999999999333\n999999999999999999999333\n999999999999999999999333\n111999999999999999999333\n111999998889999999999333\n111999998889999999999333\n111999998889999999999333\n111999999999999999999333\n111999999999999999999333\n111999999999999999999333\n111999999999999999999999\n111999999999999999999999\n111999999999999999999999\n999999999999999999999999\n999999944444444499999999\n999999944444444499999999\n999999944444444499999999\n999999999999999999999999\n999999999999999999999999\n999999999999999999999999\n999999999999999999999999', '222222222222222222222333\n222222222222222222222333\n222222222222222222222333\n888888888889999999999333\n888888888889999999999333\n888888888889999999999333\n111999998889999999999333\n111888888888888888888333\n111888888888888888888333\n111888888888888888888333\n111888998889999999888333\n111888998889999999888333\n111888998889999999888333\n111888998889999999888999\n111888998888888888888888\n111888998888888888888888\n999888998888888888888888\n999888944444444499888999\n999888944444444499888999\n999888944444444499888999\n999888999999999999888999\n999888999999999999888999\n999888999999999999888999\n999888999999999999888999'))

def _source_pairs():
    return [{name: np.array([[int(c) for c in row] for row in text.splitlines()], dtype=np.int8)
             for name, text in zip(('input', 'output'), values)} for values in _SOURCE_PAIRS]


def _pair_digest(pair):
    return hashlib.sha256(repr(tuple(_grid_signature(pair[n]) for n in ('input', 'output'))).encode()).digest()


_SOURCE_DIGESTS = {_pair_digest(pair) for pair in _source_pairs()}


def _parse_header(grid):
    """Find the bottom frame and segment panels from their visible side walls."""
    frame = int(grid[0, 0])
    _require(np.all(grid[0] == frame), 'missing top frame')
    bottoms = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, 'missing bottom frame')
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(grid) - 1, 'invalid band height')
    band = grid[1:bottom]
    _require(np.all(band == band[0]), 'inconsistent panel rows')
    row = band[0]
    _require(row[0] == row[-1] == frame, 'missing side walls')
    runs = []
    c = 0
    while c < len(row):
        if row[c] == frame:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != frame:
            c += 1
        runs.append((start, c))
    _require(bool(runs) and all(b-a >= 3 for a,b in runs), 'ambiguous panel interiors')
    interior = int(row[runs[0][0]+1])
    turns = {}
    widths = []
    for a,b in runs:
        _require(np.all(row[a+1:b-1] == interior), 'inconsistent empty interiors')
        left, right = int(row[a]), int(row[b-1])
        _require((left == interior) != (right == interior), 'ambiguous side bar')
        color, side = (right, 'R') if left == interior else (left, 'L')
        _require(color not in turns and color not in (frame, interior), 'conflicting key')
        turns[color] = side
        widths.append(b-a+2)
    return bottom+1, frame, interior, turns, tuple(widths)


def _components(field, colors):
    """Parse each connected component, including repeated colors, separately."""
    rectangles = []
    height,width = field.shape
    seen = set()
    for rr,cc in np.argwhere(np.isin(field, list(colors))):
        r,c = int(rr),int(cc)
        if (r,c) in seen:
            continue
        color = int(field[r,c])
        todo = [(r,c)]
        seen.add((r,c))
        cells = []
        while todo:
            x,y = todo.pop()
            cells.append((x,y))
            for dx,dy in _DIRECTIONS:
                xx,yy=x+dx,y+dy
                if 0<=xx<height and 0<=yy<width and (xx,yy) not in seen and field[xx,yy]==color:
                    seen.add((xx,yy))
                    todo.append((xx,yy))
        a,b=np.min(cells,axis=0)
        e,f=np.max(cells,axis=0)+1
        _require(len(cells)==(e-a)*(f-b), 'nonrectangular blocker component')
        rectangles.append((int(a),int(e),int(b),int(f),color))
    return rectangles


def _decode_input(grid):
    crop,frame,interior,turns,widths = _parse_header(grid)
    field=grid[crop:]
    other=set(map(int,np.unique(field)))-set(turns)
    _require(len(other)==2, 'field must have one background and one unkeyed source')
    candidates=[]
    for color in other:
        cells=np.argwhere(field==color)
        a,b=cells.min(axis=0)
        e,f=cells.max(axis=0)+1
        if e-a==f-b and len(cells)==(e-a)*(f-b):
            candidates.append((color,int(a),int(b),int(e-a)))
    _require(len(candidates)==1, 'source square is not unique')
    source,r,c,k=candidates[0]
    background=next(iter(other-{source}))
    rectangles=_components(field,turns)
    return dict(crop=crop,frame=frame,interior=interior,turns=turns,panel_widths=widths,
                field=field,source=source,background=background,k=k,source_position=(r,c),
                rectangles=rectangles)


def _rectangle_trace(field, source_box, source, turns, rectangles):
    """Independent geometric tracer: jump to nearest intersecting rectangle face.

    Segment records contain initial branch, origin, heading, travel distance and
    component index. These records are reconstructed from input geometry.
    """
    h,w=field.shape
    sr,sc,k=source_box
    output=field.copy()
    segments=[]
    chains=[]
    for branch in range(4):
        r,c,d=sr,sc,branch
        seen=set()
        chain=[]
        for _ in range(4*h*w+1):
            state=(r,c,d)
            _require(state not in seen, 'cyclic route')
            seen.add(state)
            limit=(r,w-c-k,h-r-k,c)[d]
            hits=[]
            for i,(a,b,e,f,color) in enumerate(rectangles):
                if d in (0,2):
                    if e>=c+k or f<=c:
                        continue
                    distance=r-b if d==0 else a-r-k
                    ahead=b<=r if d==0 else a>=r+k
                    full=e<=c and f>=c+k
                else:
                    if a>=r+k or b<=r:
                        continue
                    distance=e-c-k if d==1 else c-f
                    ahead=e>=c+k if d==1 else f<=c
                    full=a<=r and b>=r+k
                if ahead and distance<limit:
                    hits.append((distance,i,full))
            if hits:
                distance=min(x[0] for x in hits)
                nearest=[x for x in hits if x[0]==distance]
                _require(len(nearest)==1 and nearest[0][2], 'partial or simultaneous face collision')
                hit=nearest[0][1]
                _require(not chain or distance>0, 'no clear travel after turning')
            else:
                distance,hit=limit,-1
            dr,dc=_DIRECTIONS[d]
            nr,nc=r+dr*distance,c+dc*distance
            a,b=min(r,nr),max(r,nr)+k
            e,f=min(c,nc),max(c,nc)+k
            _require(not np.any(np.isin(field[a:b,e:f],list(turns))), 'blocked exterior turning square')
            output[a:b,e:f]=source
            segments.append((branch,r,c,d,distance,hit))
            if hit<0:
                break
            color=rectangles[hit][4]
            chain.append(hit)
            r,c,d=nr,nc,_turn(d,turns[color])
        else:
            raise AssertionError('unterminated ray')
        chains.append(tuple(chain))
    return output,tuple(chains),tuple(segments)


def _decode_pair(pair):
    info=_decode_input(pair['input'])
    r,c=info['source_position']
    args=(info['field'],info['background'],info['source'],(r,c,info['k']),info['turns'])
    expected,colors,events=_trace_field(*args)
    geometric,chains,segments=_rectangle_trace(info['field'],(r,c,info['k']),info['source'],info['turns'],info['rectangles'])
    _require(np.array_equal(expected,geometric), 'cell and rectangle tracers disagree')
    _require(np.array_equal(expected,pair['output']), 'incorrect cropped beam output')
    _require(pair['output'].shape==info['field'].shape, 'incorrect crop')
    info.update(chains=chains,events=events,segments=segments)
    return info


def _make_header(rng, width, height, colors, turns):
    count=len(colors)
    widths=np.full(count,5,dtype=int)
    if width>5*count:
        if width%count==0 and rng.random()<0.35:
            widths[:]=width//count
        else:
            widths+=rng.multinomial(width-5*count, np.full(count,1/count))
    header=np.full((height,width),9,dtype=np.int8)
    header[[0,-1],:]=8
    c=0
    for color,size in zip(rng.permutation(colors),widths):
        header[:,c]=8
        header[:,c+size-1]=8
        header[1:-1,c+1 if turns[int(color)]=='L' else c+size-2]=color
        c+=size
    return header


def sample_puzzle_parameters(rng):
    """Regimes are shared; visible semantic colors and key tables are local."""
    mode=str(rng.choice(['source','compact','broad','spacious'],p=[.18,.22,.4,.2]))
    sizes=sorted(map(int,rng.choice([1,2,3,4],size=2,replace=False)))
    if mode=='source':
        sizes=[2,3]
    return dict(mode=mode,sizes=sizes,shift=str(rng.choice(['familiar','larger','smaller','combined'])),
                base_palette=rng.permutation(10),color_varies=bool(rng.random()<.88),
                density=float(rng.uniform(.25,.8)),boundary_bias=float(rng.uniform(.2,.65)))


def _box_clear(field, rect, color):
    a,b,c,d=rect
    if not (0<=a<b<=field.shape[0] and 0<=c<d<=field.shape[1]):
        return False
    if np.any(field[a:b,c:d]!=0):
        return False
    # Repeated colors remain distinct components, including at their side faces.
    return not np.any(field[max(0,a-1):min(field.shape[0],b+1),max(0,c-1):min(field.shape[1],d+1)]==color)


def _grow_scene(rng,h,w,k,count,turns,boundary_bias,route_mode):
    r=int(rng.integers(0,h-k+1))
    c=int(rng.integers(0,w-k+1))
    field=np.zeros((h,w),dtype=np.int8)
    field[r:r+k,c:c+k]=1
    rects=[]
    allowed_unhit = int(rng.integers(1, 3)) if rng.random() < .25 else 0
    current=_rectangle_trace(field,(r,c,k),1,turns,rects)
    for _ in range(count*22+15):
        if len(rects)>=count:
            break
        unused=[x for x in turns if x not in [q[4] for q in rects]]
        color=int(rng.choice(unused if unused and rng.random()<.65 else list(turns)))
        if allowed_unhit and rects and rng.random()<.14:
            # An off-route rectangle is a real preserved object, not an unused key.
            a=int(rng.integers(h)); b=int(rng.integers(a+1,min(h,a+5)+1))
            e=int(rng.integers(w)); f=int(rng.integers(e+1,min(w,e+6)+1))
        else:
            segments=[s for s in current[2] if s[4]>=2]
            if not segments:
                break
            off=[s for s in segments if s[3]!=s[0]]
            if off and route_mode=='chain' and rng.random()<.8:
                segments=off
            if route_mode=='radial':
                direct=[s for s in segments if s[3]==s[0] and s[1:3]==(r,c)]
                if direct:
                    segments=direct
            _,x,y,d,length,_=segments[int(rng.integers(len(segments)))]
            distance=int(rng.integers(1,length+1))
            dr,dc=_DIRECTIONS[d]
            x+=dr*distance; y+=dc*distance
            if d in (0,2):
                e=int(rng.integers(0,y+1)); f=int(rng.integers(y+k,w+1))
                if rng.random()<.4:
                    e=max(0,y-int(rng.integers(0,4))); f=min(w,y+k+int(rng.integers(0,4)))
                if rng.random()<boundary_bias:
                    if rng.random()<.5: e=0
                    else: f=w
                depth=int(rng.integers(1,max(2,h//3+1)))
                a,b=(max(0,x-depth),x) if d==0 else (x+k,min(h,x+k+depth))
                if rng.random()<boundary_bias:
                    if d==0: a=0
                    else: b=h
            else:
                a=int(rng.integers(0,x+1)); b=int(rng.integers(x+k,h+1))
                if rng.random()<.4:
                    a=max(0,x-int(rng.integers(0,4))); b=min(h,x+k+int(rng.integers(0,4)))
                if rng.random()<boundary_bias:
                    if rng.random()<.5: a=0
                    else: b=h
                depth=int(rng.integers(1,max(2,w//3+1)))
                e,f=(y+k,min(w,y+k+depth)) if d==1 else (max(0,y-depth),y)
                if rng.random()<boundary_bias:
                    if d==1: f=w
                    else: e=0
        if not _box_clear(field,(a,b,e,f),color):
            continue
        candidate=field.copy(); candidate[a:b,e:f]=color
        trial=rects+[(a,b,e,f,color)]
        try:
            traced=_rectangle_trace(candidate,(r,c,k),1,turns,trial)
        except AssertionError:
            continue
        hit = {i for chain in traced[1] for i in chain}
        if len(trial) - len(hit) > allowed_unhit:
            continue
        # A visible, extensive background keeps source identification clear.
        if np.count_nonzero(candidate==0)<max(k*k*2,h*w//4):
            continue
        field,rects,current=candidate,trial,traced
    return field,current[0]


def _generate_pair(rng, regime, parameters, k=None, teaching=False):
    if k is None:
        k=int(rng.choice(parameters['sizes']))
    mode=parameters['mode']
    if mode=='source' and rng.random()<.7:
        hh,h,w,n=6,19 if regime=='source_train' else 24,24,4
    else:
        hh=int(rng.integers(3,8))
        n=int(rng.integers(2,7))
        low=max(8,2*k+3)
        h=int(rng.integers(low,31-hh))
        w=int(rng.integers(max(5*n,low),31))
        if mode=='spacious': h=int(rng.integers(max(low,19),31-hh))
        if mode=='compact': h=int(rng.integers(low,min(31-hh,max(low+1,18))))
    colors=list(range(2,2+n))
    turn_mode=str(rng.choice(['mixed','L','R'],p=[.64,.18,.18]))
    turns={color:(str(rng.choice(['L','R'])) if turn_mode=='mixed' else turn_mode) for color in colors}
    target=int(rng.integers(1,10))
    if teaching: target=max(target,4)
    if rng.random()>parameters['density'] and not teaching: target=int(rng.integers(1,4))
    route_mode=str(rng.choice(['radial','chain','free'],p=[.25,.45,.3]))
    field,output=_grow_scene(rng,h,w,k,target,turns,parameters['boundary_bias'],route_mode)
    header=_make_header(rng,w,hh,colors,turns)
    inp=np.vstack((header,field))
    palette=parameters['base_palette'].copy()
    if parameters['color_varies']:
        palette=rng.permutation(10)
    else:
        # Source-like palettes can retain their frame and source while the
        # locally identifiable background still changes between examples.
        swap = int(rng.integers(2, 8))
        palette[0], palette[swap] = palette[swap], palette[0]
    if rng.random()<.38:
        palette[9]=palette[0]
    return dict(input=palette[inp].astype(np.int8),output=palette[output].astype(np.int8))


def _training_evidence(infos):
    headings={side:set() for side in ('L','R')}
    recursive=False
    for info in infos:
        recursive |= max(map(len,info['chains']))>=2
        for event in info['events']:
            headings[info['turns'][event[1]]].add(event[2])
    return all(len(x)>=2 for x in headings.values()),recursive


def _fallback_teaching(rng, parameters, k):
    """Guaranteed compositional two-turn lesson, rotated and recolored locally.

    A horizontal beam turns up at a short right blocker, then encounters an
    off-axis top blocker. The other vertical beam hits a bottom blocker. Random
    reflection swaps both turns, and rotation changes the incoming headings.
    """
    side=5*k+3
    field=np.zeros((side,side),dtype=np.int8)
    r=c=2*k+1
    field[r:r+k,c:c+k]=1
    field[r:r+k,c+2*k:c+2*k+1]=2
    field[0:1,c+k:c+2*k]=3
    field[r+2*k:r+2*k+1,c:c+k]=4
    turns={2:'L',3:'R',4:'L',5:'R'}
    if rng.random()<.5:
        field=field[:,::-1].copy()
        turns={x:('R' if t=='L' else 'L') for x,t in turns.items()}
    field=np.rot90(field,int(rng.integers(4))).copy()
    # Width must fit four legible panels; pad without changing the route.
    if side<20: field=np.pad(field,((0,0),(0,20-side)))
    cells=np.argwhere(field==1); sr,sc=cells.min(axis=0)
    rects=_components(field,turns)
    output,_,_=_rectangle_trace(field,(int(sr),int(sc),k),1,turns,rects)
    header=_make_header(rng,field.shape[1],3,list(turns),turns)
    palette=rng.permutation(10)
    return dict(input=palette[np.vstack((header,field))].astype(np.int8),output=palette[output].astype(np.int8))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    rng=np.random.default_rng(int(seed))
    nt,nq=_resolve_counts(int(seed),rng,num_train,num_test)
    parameters=sample_puzzle_parameters(rng)
    train=[]; infos=[]; seen=set()
    # Select demonstrations by their observable evidence, not example index.
    for i in range(nt):
        k=parameters['sizes'][i%2]
        candidates=[]
        for attempt in range(36):
            pair=_generate_pair(rng,'source_train',parameters,k,teaching=i<2)
            if _grid_signature(pair['input']) in seen or _pair_digest(pair) in _SOURCE_DIGESTS:
                continue
            info=_decode_pair(pair)
            if i == 1 and info['background'] == infos[0]['background']:
                continue
            ready,recursive=_training_evidence(infos+[info])
            score=sum(min(2,len({e[2] for q in infos+[info] for e in q['events'] if q['turns'][e[1]]==side})) for side in ('L','R'))+3*recursive
            candidates.append((score,pair,info))
            if i>=2 or (ready and recursive):
                break
            if i==0 and max(map(len,info['chains']))>=2 and len(info['events'])>=3:
                break
        if i==1 and (not candidates or not all(_training_evidence(infos+[max(candidates,key=lambda x:x[0])[2]]))):
            # Bounded fallback supplies the missing opposite-side demonstrations.
            for _ in range(16):
                pair=_fallback_teaching(rng,parameters,k)
                info=_decode_pair(pair)
                score=sum(min(2,len({e[2] for q in infos+[info] for e in q['events'] if q['turns'][e[1]]==side})) for side in ('L','R'))+3*any(max(map(len,q['chains']))>=2 for q in infos+[info])
                candidates.append((score,pair,info))
                if all(_training_evidence(infos+[info])): break
        _,pair,info=max(candidates,key=lambda x:x[0])
        train.append(pair); infos.append(info); seen.add(_grid_signature(pair['input']))
    try:
        _learn_turns(train)
        identifiable = True
    except AssertionError:
        identifiable = False
    if (not all(_training_evidence(infos)) or not identifiable
            or len({info['background'] for info in infos}) < 2):
        # Deterministic final fallback: two mirror lessons cover both sides from
        # horizontal and vertical arrivals regardless of the stochastic samples.
        train[:2]=[_fixed_lesson(1,False),_fixed_lesson(2,True)]
        seen={_grid_signature(p['input']) for p in train}
    rng.shuffle(train)
    test=[]
    for i in range(nq):
        shift=parameters['shift']
        k=int(rng.choice(parameters['sizes']))
        if shift=='larger': k=min(4,max(parameters['sizes'])+1)
        elif shift=='smaller': k=max(1,min(parameters['sizes'])-1)
        for attempt in range(64):
            pair=_generate_pair(rng,'target_test',parameters,k)
            signature=_grid_signature(pair['input'])
            if signature not in seen and _pair_digest(pair) not in _SOURCE_DIGESTS:
                test.append(pair); seen.add(signature); break
        else:
            raise AssertionError('exhausted unique configurations')
    puzzle=dict(train=train,test=test)
    validate_generated_puzzle(puzzle)
    return puzzle


def _fixed_lesson(k, reflected):
    side=5*k+3
    field=np.zeros((side,max(side,20)),dtype=np.int8)
    r=c=2*k+1
    field[r:r+k,c:c+k]=1
    field[r:r+k,c+2*k:c+2*k+1]=2
    field[0:1,c+k:c+2*k]=3
    field[r+2*k:r+2*k+1,c:c+k]=4
    turns={2:'L',3:'R',4:'L',5:'R'}
    if reflected:
        field=field[:,::-1].copy()
        turns={x:('R' if y=='L' else 'L') for x,y in turns.items()}
    r,c=np.argwhere(field==1).min(axis=0)
    out,_,_=_rectangle_trace(field,(int(r),int(c),k),1,turns,_components(field,turns))
    header=_make_header(np.random.default_rng(1729),field.shape[1],3,list(turns),turns)
    pair = dict(input=np.vstack((header,field)),output=out)
    if reflected:
        pair = {name: (grid + 3) % 10 for name, grid in pair.items()}
    return pair


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle,dict) and set(puzzle)=={'train','test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in puzzle), 'splits must be lists')
    _require(2<=len(puzzle['train'])<=10 and 1<=len(puzzle['test'])<=4 and len(puzzle['train'])>len(puzzle['test']), 'invalid counts')
    seen=set(); decoded={s:[] for s in puzzle}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair)=={'input','output'}, 'invalid pair keys')
            for grid in pair.values():
                _require(isinstance(grid,np.ndarray) and grid.ndim==2, 'invalid grid')
                _require(np.issubdtype(grid.dtype,np.integer) and all(1<=x<=30 for x in grid.shape), 'invalid dtype or size')
                _require(np.all((grid>=0)&(grid<=9)), 'invalid color')
            signature=_grid_signature(pair['input'])
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            _require(_pair_digest(pair) not in _SOURCE_DIGESTS, 'exact source pair copy')
            decoded[split].append(_decode_pair(pair))
    ready,recursive=_training_evidence(decoded['train'])
    _require(ready, 'training must reveal both relative turns from multiple headings')
    if any(max(map(len,q['chains']))>=2 for q in decoded['test']):
        _require(recursive, 'test recursion is undemonstrated')
    _require(len({q['k'] for q in decoded['train']})>=2, 'training must demonstrate visible thickness variation')
    _require(len({q['background'] for rows in decoded.values() for q in rows}) >= 2,
             'background must vary within the puzzle')
    _require(_learn_turns(puzzle['train']) == {'L': 'L', 'R': 'R'},
             'training does not uniquely reveal the relative turn convention')


def _learn_turns(train_pairs):
    """Distinguish both side meanings using only demonstration grids."""
    hypotheses=[]
    for left in ('L','R'):
        for right in ('L','R'):
            mapping={'L':left,'R':right}
            fits=True
            for pair in train_pairs:
                info=_decode_input(pair['input'])
                turns={color:mapping[side] for color,side in info['turns'].items()}
                r,c=info['source_position']
                try:
                    out,_,_=_rectangle_trace(info['field'],(r,c,info['k']),info['source'],turns,info['rectangles'])
                except AssertionError:
                    fits=False; break
                if not np.array_equal(out,pair['output']): fits=False; break
            if fits: hypotheses.append(mapping)
    _require(len(hypotheses)==1, 'turn convention not uniquely demonstrated')
    return hypotheses[0]


def solve_from_training(train_pairs, test_input):
    """Internal prediction check; no test label, seed, or generator state.

    This is not a protected independent reference solver.
    """
    mapping = _learn_turns(train_pairs)
    info=_decode_input(test_input)
    turns={color:mapping[side] for color,side in info['turns'].items()}
    r,c=info['source_position']
    return _rectangle_trace(info['field'],(r,c,info['k']),info['source'],turns,info['rectangles'])[0]
def review_features(pair):
    """Task measurements reconstructed solely from returned grids."""
    q=_decode_pair(pair)
    h,w=q['field'].shape
    k=q['k']; r,c=q['source_position']; rects=q['rectangles']
    masks=np.zeros((4,2,h,w),dtype=bool)
    for branch,a,b,d,length,hit in q['segments']:
        dr,dc=_DIRECTIONS[d]; x,y=a+dr*length,b+dc*length
        masks[branch,d%2,min(a,x):max(a,x)+k,min(b,y):max(b,y)+k]=True
    masks[:,:,r:r+k,c:c+k]=False
    crossing=any(np.any(masks[i,0]&masks[j,1]) or np.any(masks[i,1]&masks[j,0]) for i in range(4) for j in range(i))
    merging=any(np.any(masks[i,d]&masks[j,d]) for i in range(4) for j in range(i) for d in range(2))
    hit={x for chain in q['chains'] for x in chain}
    direct={chain[0] for chain in q['chains'] if chain}
    recursive=[i for i,chain in enumerate(q['chains']) if len(chain)>1]
    off={i for i,(a,b,e,f,_) in enumerate(rects) if (b<=r or a>=r+k) and (f<=c or e>=c+k)}
    off_chain = 0
    for chain in q['chains']:
        consecutive = 0
        for component in chain:
            consecutive = consecutive + 1 if component in off else 0
            off_chain = max(off_chain, consecutive)
    gaps = [segment[4] for segment in q['segments'] if segment[5] >= 0]
    return dict(field_shape=(h,w),k=k,source_position=(r,c),
                source_edge_gap=min(r,c,h-r-k,w-c-k),key_count=len(q['turns']),
                header_height=q['crop'],panel_widths=q['panel_widths'],
                unequal_panels=len(set(q['panel_widths']))>1,
                blocker_count=len(rects),unused_keys=len(set(q['turns'])-{x[4] for x in rects}),
                repeated_colors=len(rects)>len({x[4] for x in rects}),unhit_blockers=len(rects)-len(hit),
                boundary_blockers=sum(a==0 or b==h or e==0 or f==w for a,b,e,f,_ in rects),
                longest_blocker=max([max(b-a,f-e) for a,b,e,f,_ in rects],default=0),
                off_axis_blockers=len(off),hit_off_axis=len(off&hit),indirect_only=len(hit-direct),
                off_axis_chain_max=off_chain,
                min_collision_gap=min(gaps, default=0),max_collision_gap=max(gaps, default=0),
                blocker_shapes=tuple(sorted((b-a,f-e) for a,b,e,f,_ in rects)),
                max_turns=max(map(len,q['chains'])),chain_lengths=tuple(map(len,q['chains'])),
                vertical_only_recursion=bool(recursive) and all(i in (0,2) for i in recursive),
                open_directions=sum(not chain for chain in q['chains']),
                open_with_many=len(rects)>=4 and any(not chain for chain in q['chains']),
                crossing=bool(crossing),merging=bool(merging),
                crossing_without_merging=bool(crossing and not merging),
                independent_paths=not crossing and not merging,
                role_triple=(q['frame'],q['interior'],q['source']),background=q['background'],
                interior_is_background=q['interior']==q['background'],
                left_keys=sum(x=='L' for x in q['turns'].values()),
                layout=(h,w,k,r,c,tuple(sorted(x[:4] for x in rects))))


def self_test() -> None:
    # Original fixtures are evidence for the invariant and exact-copy exclusion.
    original=_source_pairs()
    for pair in original: _decode_pair(pair)
    prediction=solve_from_training(original[:3],original[3]['input'].copy())
    _require(np.array_equal(prediction,original[3]['output']), 'source prediction disagrees')
    # Source-supported extensions that the reviewed decoder incorrectly rejected.
    repeated = {name: grid.copy() for name, grid in original[2].items()}
    repeated['input'][6:][repeated['input'][6:] == 3] = 6
    repeated['output'][repeated['output'] == 3] = 6
    _require(review_features(repeated)['repeated_colors'], 'repeated components rejected')
    untouched = {name: grid.copy() for name, grid in original[0].items()}
    untouched['input'][6, 23] = 6
    untouched['output'][0, 23] = 6
    _require(review_features(untouched)['unhit_blockers'] == 1, 'untouched blocker rejected')
    # Reject partial faces, conflicting faces, and a four-corner cycle.
    malformed = []
    for conflicting in (False, True):
        field = np.zeros((9, 9), dtype=np.int8)
        field[4:6, 4:6] = 1
        field[1:2, 4:5] = 2
        if conflicting:
            field[1:2, 5:6] = 3
        malformed.append((field, (4, 4, 2), {2: 'R', 3: 'L'}))
    field = np.zeros((10, 10), dtype=np.int8)
    field[4, 4] = 1
    for row, col in ((4, 7), (7, 6), (6, 3), (3, 4)):
        field[row, col] = 2
    malformed.append((field, (4, 4, 1), {2: 'R'}))
    for field, box, turns in malformed:
        try:
            _rectangle_trace(field, box, 1, turns, _components(field, turns))
        except AssertionError:
            pass
        else:
            raise AssertionError('ambiguous or cyclic geometry accepted')
    p=generate_puzzle(900)
    _require(_puzzles_equal(p,generate_puzzle(900)), 'automatic determinism')
    p=generate_puzzle(901,6,3)
    _require(_puzzles_equal(p,generate_puzzle(901,6,3)), 'explicit determinism')
    _require(not _puzzles_equal(p,generate_puzzle(902,6,3)), 'seed diversity')
    for i,(nt,nq) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        p=generate_puzzle(1100+i,nt,nq)
        validate_generated_puzzle(p)
        _require((nt is None or len(p['train'])==nt) and (nq is None or len(p['test'])==nq), 'requested counts')
    for nt,nq in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,3),(3,4),(4,4)):
        try: generate_puzzle(1200,nt,nq)
        except ValueError: pass
        else: raise AssertionError('illegal counts accepted')
    counts=[set(),set()]; all_features=[]; within={name:0 for name in ('k','field_shape','role_triple','blocker_count','key_count','header_height','max_turns','background')}
    flips=0; scale_relations=set(); overlap=0; unused_test=0
    for seed in range(42):
        p=generate_puzzle(seed)
        for i,split in enumerate(('train','test')): counts[i].add(len(p[split]))
        features=[review_features(pair) for split in p for pair in p[split]]
        all_features.extend(features)
        for name in within: within[name]+=len({x[name] for x in features})>1
        tables=[_decode_input(pair['input'])['turns'] for split in p for pair in p[split]]
        flips+=any(len({table[color] for table in tables if color in table})>1 for color in range(10))
        training_features=features[:len(p['train'])]; test_features=features[len(p['train']):]
        ks={x['k'] for x in training_features}
        overlap+=bool({x['background'] for x in training_features}&{x['background'] for x in test_features})
        for f in test_features:
            unused_test+=f['unused_keys']>0
            scale_relations.add('familiar' if f['k'] in ks else 'larger' if f['k']>max(ks) else 'smaller' if f['k']<min(ks) else 'between')
        # Freeze only the demonstrated pairs and test input before prediction.
        train_copy=[{n:g.copy() for n,g in pair.items()} for pair in p['train']]
        for pair in p['test']:
            expected=solve_from_training(train_copy,pair['input'].copy())
            _require(np.array_equal(expected,pair['output']), 'blind prediction mismatch')
    _require(len(counts[0])>=4 and len(counts[1])>=3, 'automatic count diversity')
    for name,minimum in (('field_shape',35),('layout',200),('role_triple',60),('blocker_count',7),('key_count',5),('header_height',5),('k',4)):
        _require(len({f[name] for f in all_features})>=minimum, 'insufficient diversity: '+name)
    for name in within: _require(within[name]>=15, 'insufficient within-puzzle variation: '+name)
    for name in ('hit_off_axis','indirect_only','vertical_only_recursion','open_with_many','repeated_colors','unhit_blockers','crossing_without_merging','independent_paths','merging','unequal_panels','interior_is_background'):
        _require(sum(bool(f[name]) for f in all_features)>=3, 'missing route/appearance coverage: '+name)
    _require(any(f['max_turns']>=4 for f in all_features), 'missing longer routes')
    _require(any(f['off_axis_chain_max']>=2 for f in all_features), 'missing successive off-axis blockers')
    _require(len({f['max_collision_gap'] for f in all_features})>=12, 'collision spacing is too narrow')
    _require(any(f['max_turns']==1 for f in all_features), 'missing single-turn scenes')
    _require(any(f['boundary_blockers']>=3 for f in all_features), 'missing multiple boundary blockers')
    _require(any(f['longest_blocker']>=21 for f in all_features), 'missing long rectangles')
    _require(any(f['source_edge_gap']<=1 for f in all_features), 'missing edge sources')
    _require(flips>=20 and overlap>=8 and unused_test>=10, 'missing local key/color regimes')
    _require({'familiar','larger','smaller'}<=scale_relations, 'missing test scale regimes')
    _require(within['background']==42, 'background diversity must hold in every puzzle')
    for role in range(3):
        _require(len({f['role_triple'][role] for f in all_features})==10, 'missing role colors')
    _require(len({f['background'] for f in all_features})==10, 'missing backgrounds')
    _require(any(f['left_keys']==0 for f in all_features)
             and any(f['left_keys']==f['key_count'] for f in all_features), 'missing uniform turn tables')
    _require(sum(f['k']>=3 and f['hit_off_axis']>0 and f['unequal_panels']
                 and f['boundary_blockers']>=2 for f in all_features)>=3,
             'compatible geometry, scale, and key variations did not combine')
