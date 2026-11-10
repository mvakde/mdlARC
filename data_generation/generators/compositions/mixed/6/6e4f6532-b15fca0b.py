"""Dock two compass-labelled ports, then route through the resulting geometry.

Only NumPy and the standard library are used. The solver reads pixels alone.
Background is 0 and the newly drawn route is 4 throughout this family.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NEAR = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0,345807733,84566578,234933366,43934451,414857798,26323113,629817470,558183733,263116806,575894459,685247247,809300761,365479313,205726630,125905200,896489524,21484751,54228556,830052080,554574628,850067185,717924396,268048481,321079738,524279,155580370,968984962,452204287,726389618,106165389,647801886,753605453,866288250,438435038,161168521,484074631,416712939,448277244,404196823,511595569,681486031)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8, copy=False).tobytes()


def _transform(cells, t):
    """D4 about (0,0), retaining a geometric center rather than normalizing."""
    result = {}
    for (r, c), color in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result[(r, c)] = int(color)
    return result


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]): v for (r, c), v in cells.items()}


def _signature(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0, v) for (r, c), v in cells.items()))


def _components(mask):
    remaining = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, found = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in NEAR:
                q = (r+dr, c+dc)
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        groups.append(found)
    return groups


def _center(points):
    r0, r1 = min(r for r, _ in points), max(r for r, _ in points)
    c0, c1 = min(c for _, c in points), max(c for _, c in points)
    if (r0+r1) % 2 or (c0+c1) % 2:
        raise ValueError("stencil needs an occupied integer center")
    p = ((r0+r1)//2, (c0+c1)//2)
    if p not in points or any((2*p[0]-r, 2*p[1]-c) not in points for r, c in points):
        raise ValueError("stencil must be centrally symmetric about a marked cell")
    return p


def _frame(a):
    h, w = a.shape
    if min(h, w) < 7:
        raise ValueError("a compass frame is required")
    colors = (int(a[0,w//2]), int(a[h//2,-1]),
              int(a[-1,w//2]), int(a[h//2,0]))
    if len(set(colors)) != 4 or set(colors) & {0, 4}:
        raise ValueError("four distinct compass bands required")
    # Fingers can meet a side midpoint, so use complete edge strips to find
    # the frame thickness, rather than counting a finger as extra frame.
    def strip_depth(axis, reverse, color):
        d = 0
        for k in range(2):
            index = -1-k if reverse else k
            strip = a[index,2:-2] if axis == 0 else a[2:-2,index]
            if np.all(strip == color):
                d += 1
            else:
                break
        return d
    depths = [strip_depth(0,False,colors[0]), strip_depth(1,True,colors[1]),
              strip_depth(0,True,colors[2]), strip_depth(1,False,colors[3])]
    if not depths[0] or len(set(depths)) != 1:
        raise ValueError("frame thickness must be uniformly 1 or 2")
    t = depths[0]
    frame = np.zeros(a.shape, dtype=bool)
    frame[:t,:] = frame[-t:,:] = True
    frame[:,:t] = frame[:,-t:] = True
    if not set(int(v) for v in a[frame]) <= set(colors):
        raise ValueError("invalid frame corners")
    return frame, colors


def _label_sides(cells, band_colors):
    core = [p for p, v in cells.items() if v not in band_colors]
    if not core:
        return None
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    labels = {}
    for p, value in cells.items():
        if value in band_colors:
            r, c = p
            sides = [i for i, yes in enumerate((r < r0, c > c1, r > r1, c < c0)) if yes]
            if len(sides) != 1:
                return None
            labels[p] = sides[0]
    return labels or None


def _parse(a):
    if np.any(a == 4):
        raise ValueError("route color 4 is absent from inputs")
    frame, bands = _frame(a)
    fixed, targets, pieces = frame.copy(), [], []
    for component in _components((a != 0) & ~frame):
        cells = {p: int(a[p]) for p in component}
        values = set(cells.values())
        if values <= set(bands):
            for p in component:
                fixed[p] = True
        elif len(values) == 1:
            targets.append(component)
        else:
            pieces.append(cells)
    if len(targets) != 2 or len(pieces) < 2:
        raise ValueError("two detached stencils and at least two pieces required")
    reg_colors = {int(a[p]) for target in targets for p in target}
    if len(reg_colors) != 1:
        raise ValueError("stencils need one registration color")
    reg = reg_colors.pop()
    body_colors = set().union(*(set(p.values()) for p in pieces)) - set(bands) - {reg}
    if len(body_colors) != 1 or reg in bands or reg == 0:
        raise ValueError("ambiguous body/registration roles")
    centers = [_center(target) for target in targets]
    return {"fixed": fixed, "frame": frame, "bands": bands, "reg": reg,
            "body": body_colors.pop(), "targets": targets, "pieces": pieces,
            "centers": centers}


def _dockings(a, info, use_labels=True, limit=2):
    """Unique A-stage pixels, before route existence is consulted."""
    choices = [[], []]
    h, w = a.shape
    for i, source in enumerate(info["pieces"]):
        for t in range(8):
            cells = _transform(source, t)
            marks = {p for p,v in cells.items() if v == info["reg"]}
            if not marks:
                continue
            labels = _label_sides(cells, set(info["bands"]))
            if use_labels and (not labels or len(set(labels.values())) < 2
                               or any(cells[p] != info["bands"][s] for p,s in labels.items())):
                continue
            for j, target in enumerate(info["targets"]):
                if len(marks) != len(target):
                    continue
                origin = min(marks)
                goal = min(target)
                delta = (goal[0]-origin[0], goal[1]-origin[1])
                placed = _shift(cells, delta)
                if {p for p,v in placed.items() if v == info["reg"]} != target:
                    continue
                if any(not(0 <= r < h and 0 <= c < w) or info["fixed"][r,c] for r,c in placed):
                    continue
                signature = (i, tuple(sorted(placed.items())))
                if all(signature != item[0] for item in choices[j]):
                    choices[j].append((signature, i, placed))
    answers = {}
    for _, i, first in choices[0]:
        for _, j, second in choices[1]:
            if i == j or first.keys() & second.keys():
                continue
            selected = {i,j}
            stationary = set().union(*(set(p) for k,p in enumerate(info["pieces"]) if k not in selected))
            if stationary & (first.keys() | second.keys()):
                continue
            board = a.copy()
            for k in selected:
                for p in info["pieces"][k]:
                    board[p] = 0
            for cells in (first,second):
                for p,v in cells.items():
                    board[p] = v
            answers[_key(board)] = (board, (i,j), (first,second))
            if len(answers) >= limit:
                return list(answers.values())
    return list(answers.values())


def _shortest(board, terminals, blocked=None):
    """BFS counts shortest routes (saturated at two), with unit edge cost."""
    h, w = board.shape
    start, goal = terminals
    open_cells = board == 0
    open_cells[start] = open_cells[goal] = True
    if blocked is not None:
        open_cells[blocked] = False
    distances = {start: 0}
    ways = {start: 1}
    previous = {}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        if goal in distances and distances[p] >= distances[goal]:
            break
        for dr, dc in DIRS:
            q = (p[0]+dr,p[1]+dc)
            if not (0 <= q[0] < h and 0 <= q[1] < w) or not open_cells[q]:
                continue
            d = distances[p]+1
            if q not in distances:
                distances[q], ways[q], previous[q] = d, ways[p], p
                queue.append(q)
            elif distances[q] == d:
                ways[q] = min(2,ways[q]+ways[p])
    if goal not in distances:
        return [], 0
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return path[::-1], ways[goal]


def _analyze(a):
    info = _parse(a)
    answers = _dockings(a, info)
    if len(answers) != 1:
        raise ValueError("docking must determine exactly one resulting grid")
    board, selected, finals = answers[0]
    path, count = _shortest(board, info["centers"])
    if count != 1 or len(path) < 3:
        raise ValueError("docked scene needs one nontrivial shortest route")
    result = board.copy()
    for p in path[1:-1]:
        if board[p] != 0:
            raise ValueError("route intermediates must be background")
        result[p] = 4
    return result, info, board, selected, finals, path


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, training examples, RNG, file access or generation metadata."""
    return _analyze(_grid(input_grid))[0]


def _halo(points):
    return {(r+dr,c+dc) for r,c in points for dr in (-1,0,1) for dc in (-1,0,1)}


def _port(rng, pattern, palette, t):
    body, reg = palette[:2]
    core = {(r,c) for r in (-1,0,1) for c in (-1,0,1)} - {(-1,0)}
    # The open north neighbor is the socket mouth. Lower tails vary independently
    # of the registration pattern and never close that mouth.
    for r in range(2, int(rng.integers(2,5))):
        columns = [int(v) for v in rng.choice((-1,0,1), size=int(rng.integers(1,4)), replace=False)]
        previous = {c for rr,c in core if rr == r-1}
        columns = sorted(set(columns) | {min(previous)})
        core.update((r,c) for c in range(min(columns),max(columns)+1))
    # A side branch changes width independently of lower-tail height and marks.
    core.update((1,c) for c in range(2,int(rng.integers(2,5))))
    marks = ({(0,0)} if pattern == 0 else {(0,-1),(0,0),(0,1)}
             if pattern == 1 else {(-1,-1),(0,0),(1,1)})
    cells = {p: reg if p in marks else body for p in core}
    bottom = max(r for r,_ in core)
    south = sorted(c for r,c in core if r == bottom)
    west_rows = sorted(r for r,c in core if c == -1)
    labels = {(bottom+1,int(rng.choice(south))): -1,
              (int(rng.choice(west_rows)),-2): -1}
    if rng.random() < .35:
        labels[(bottom+1,south[0])] = -1
    cells.update(labels)
    cells = _transform(cells,t)
    facing = _label_sides(cells, {-1})
    for p, side in facing.items():
        cells[p] = palette[2+side]
    return cells


def _skeleton(rng, count, split, palette, thickness):
    """Variable baffles: some are bands, some are unmatched composite pieces."""
    h = int(rng.integers(19,27))
    widths = [int(rng.integers(1,3)) for _ in range(count)]
    # Exactly one or two detached obstacles, never silently deleted by docking.
    detached = set(int(v) for v in rng.choice(count, size=int(rng.integers(1,min(2,count)+1)), replace=False))
    for i in detached:
        widths[i] = 1
    gaps = [1 for _ in range(count-1)]
    left, right = int(rng.integers(6,10)), int(rng.integers(6,10))
    w = left+right+sum(widths)+sum(gaps)
    if w+2*thickness > 30:
        right -= w+2*thickness-30
        w = left+right+sum(widths)+sum(gaps)
    a = np.zeros((h,w), dtype=np.int8)
    col, gates = left, []
    top_first = bool(rng.integers(2))
    for i,width in enumerate(widths):
        top = top_first == (i%2 == 0)
        gap = int(rng.integers(3,7))
        if i in detached:
            r0,r1 = (2,h-gap-2) if top else (gap+1,h-3)
            for r in range(r0,r1+1):
                a[r,col] = palette[0]
            # Four or six marks cannot match a one/three-cell target. The
            # geometry is still a piece, with two outward compass patches.
            n = int(rng.choice((4,6)))
            reg_start = int(rng.integers(r0,r1-n+2))
            a[reg_start:reg_start+n,col] = palette[1]
            a[r0-1,col],a[r1+1,col] = palette[2],palette[4]
            gate = r1+2 if top else r0-2
        else:
            r0,r1 = (0,h-gap-1) if top else (gap,h-1)
            a[r0:r1+1,col:col+width] = palette[2 if top else 4]
            gate = r1+1 if top else r0-1
            if split and 0 < i < count-1:
                opening = int(rng.integers(max(r0+3,4),min(r1-2,h-4)+1))
                a[opening,col:col+width] = 0
        gates.append(gate)
        col += width+(gaps[i] if i < len(gaps) else 0)
    return a, gates, left, right


def _fit(cells, a, forbidden):
    h,w = a.shape
    return all(0 <= r < h and 0 <= c < w and a[r,c] == 0 and (r,c) not in forbidden for r,c in cells)


def _construct(rng, palette, count, split, demonstration, rotation, mirror):
    thickness = int(rng.integers(1,3))
    static,gates,left,right = _skeleton(rng,count,split,palette,thickness)
    h,w = static.shape
    patterns = [1,2] if demonstration == "reflect" else [int(v) for v in rng.choice(3,2,replace=False)]
    rng.shuffle(patterns)
    finals,centers = [], []
    for index in range(2):
        port = _port(rng,patterns[index],palette,1 if index == 0 else 3)
        minc,maxc = min(c for _,c in port),max(c for _,c in port)
        lo,hi = (max(-minc,0),left-maxc) if index == 0 else (w-right-minc,w-maxc)
        if lo >= hi:
            return None
        center = (gates[0 if index == 0 else -1], int(rng.integers(lo,hi)))
        placed = _shift(port,center)
        if not _fit(placed,static,set()) or (finals and placed.keys() & _halo(finals[0])):
            return None
        finals.append(placed)
        centers.append(center)
    board = static.copy()
    for cells in finals:
        for p,v in cells.items():
            board[p] = v
    path,n = _shortest(board,centers)
    manhattan = sum(abs(a-b) for a,b in zip(*centers))
    if n != 1 or len(path)-1 <= manhattan or len(path) < 10:
        return None
    # Put only registration stencils at their destinations in the INPUT.
    inp = static.copy()
    targets = {p for cells in finals for p,v in cells.items() if v == palette[1]}
    for p in targets:
        inp[p] = palette[1]
    forbidden = _halo(targets) | _halo({tuple(int(v) for v in p) for p in np.argwhere(static != 0)})
    sources = []
    for index,final in enumerate(finals):
        poses = list(int(v) for v in rng.permutation(8))
        if index == 0 and demonstration:
            poses = [v for v in poses if (1 <= v <= 3 if demonstration == "rotate" else v >= 4)]
        accepted = None
        for t in poses:
            piece = _transform(final,t)
            r0,r1 = min(r for r,_ in piece),max(r for r,_ in piece)
            c0,c1 = min(c for _,c in piece),max(c for _,c in piece)
            # Search in shuffled order, preferring to occupy cells that the
            # eventual route needs. Relocation therefore changes B's graph.
            origins = [(r,c) for r in range(-r0,h-r1) for c in range(-c0,w-c1)]
            rng.shuffle(origins)
            for origin in origins:
                placed = _shift(piece,origin)
                if not _fit(placed,inp,forbidden):
                    continue
                if not sources and not (placed.keys() & set(path[1:-1])):
                    continue
                accepted = placed
                break
            if accepted is not None:
                break
        if accepted is None:
            return None
        for p,v in accepted.items():
            inp[p] = v
        forbidden.update(_halo(accepted))
        sources.append(accepted)
    out = board.copy()
    for p in path[1:-1]:
        out[p] = 4
    t = thickness
    def frame(a):
        result = np.zeros((h+2*t,w+2*t),dtype=np.int8)
        result[:t,:],result[-t:,:] = palette[2],palette[4]
        result[t:-t,:t],result[t:-t,-t:] = palette[5],palette[3]
        result[t:-t,t:-t] = a
        result = np.rot90(result,rotation)
        return (result[:,::-1] if mirror else result).copy()
    inp,out = frame(inp),frame(out)
    try:
        prediction,info,_,_,_,_ = _analyze(inp)
        if not np.array_equal(prediction,out):
            return None
        if demonstration == "rotate" and len(_dockings(inp,info,use_labels=False)) < 2:
            return None
    except ValueError:
        return None
    return {"input": inp,"output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name,value,lo,hi in (("num_train",num_train,2,10),("num_test",num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(v) for v in rng.permutation([1,2,3,5,6,7,8,9])[:6]]
    axis = int(rng.integers(2))
    puzzle,used = {"train": [],"test": []},set()
    for split,n in (("train",num_train),("test",num_test)):
        for i in range(n):
            demo = ("rotate" if i == 0 else "reflect" if i == 1 else None) if split == "train" else None
            count = (3 if i < 2 else int(rng.integers(3,5))) if split == "train" else int(rng.integers(3,5))
            split_bar = (i == 1 or rng.random() < .45) if split == "train" else True
            rotation = (axis + (1 if split == "test" else 0) + 2*int(rng.integers(2))) % 4
            mirror = bool(rng.integers(2))
            # Pair-local bands vary; body/registration roles stay shared.
            local = palette[:2] + [int(v) for v in rng.permutation(palette[2:])]
            for attempt in range(160):
                pair = _construct(rng,local,count,split_bar,demo,rotation,mirror)
                if pair is not None and _key(pair["input"]) not in used:
                    used.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction exhausted; no weaker fallback used")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pose(source,final):
    poses = [t for t in range(8) if _signature(_transform(source,t)) == _signature(final)]
    assert poses
    return "translation" if 0 in poses else "rotation" if any(t < 4 for t in poses) else "reflection"


def _longer_alternative(board,centers,path):
    # At most 900 blocked-cell BFS queries; generated paths are much shorter.
    # Existence after deleting a shortest-route cell witnesses an alternative.
    for p in path[1:-1]:
        alternative,_ = _shortest(board,centers,p)
        if alternative:
            assert len(alternative) > len(path)
            return True
    return False


def _inspect_pair(inp,out):
    expected,info,board,selected,finals,path = _analyze(inp)
    assert np.array_equal(expected,out), "wrong docking or route"
    assert np.array_equal(inp[info["fixed"]],out[info["fixed"]])
    stationary = [p for k,p in enumerate(info["pieces"]) if k not in selected]
    assert stationary, "must expose an undocked wall piece"
    for cells in stationary:
        assert all(int(out[p]) == v for p,v in cells.items())
    for cells in finals:
        assert all(int(out[p]) == v for p,v in cells.items())
    old = set().union(*(set(info["pieces"][i]) for i in selected))
    freed_route = set(path[1:-1]) & old
    assert freed_route, "routing before relocation must be distinguishable"
    assert all(board[p] == 0 and out[p] == 4 and inp[p] != 0 for p in freed_route)
    assert int(np.count_nonzero(out == 4)) == len(path)-2
    assert all(board[p] == 0 for p in path[1:-1])
    assert all(sum(abs(a-b) for a,b in zip(p,q)) == 1 for p,q in zip(path,path[1:]))
    assert all(int(out[p]) == info["reg"] for p in info["centers"])
    assert np.count_nonzero(board == 0) > len(path)
    distance = sum(abs(a-b) for a,b in zip(*info["centers"]))
    assert len(path)-1 > distance
    poses = [_pose(info["pieces"][i],final) for i,final in zip(selected,finals)]
    patterns = []
    for target in info["targets"]:
        center = _center(target)
        points = {(r-center[0],c-center[1]):1 for r,c in target}
        patterns.append(min(_signature(_transform(points,t)) for t in range(8)))
    return {"shape":inp.shape,"route_length":len(path)-1,"detour":len(path)-1-distance,
            "freed_route_cells":len(freed_route),"undocked":len(stationary),
            "piece_shapes":[(max(r for r,_ in p)-min(r for r,_ in p)+1,
                             max(c for _,c in p)-min(c for _,c in p)+1) for p in finals],
            "patterns":patterns,"poses":poses,"bands":info["bands"],
            "compass_needed":len(_dockings(inp,info,use_labels=False)) > 1}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used,training,roles = set(),[],set()
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in used
            used.add(_key(inp))
            record = _inspect_pair(inp,out)
            info = _parse(inp)
            roles.add((info["body"],info["reg"]))
            if split == "train":
                training.append(record)
    assert len(roles) == 1
    assert any(r["compass_needed"] for r in training)
    assert any("rotation" in r["poses"] for r in training)
    assert any("reflection" in r["poses"] for r in training)
    assert any(len(r["patterns"][0]) == len(r["patterns"][1]) == 3
               and r["patterns"][0] != r["patterns"][1] for r in training)
    # A genuine choice of route is demonstrated, rather than merely tracing the
    # sole available corridor. Check this on the smallest teaching example.
    first = _grid(puzzle["train"][0]["input"])
    _,info,board,_,_,path = _analyze(first)
    assert _longer_alternative(board,info["centers"],path)


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                a = np.asarray(pair[role],dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    records,shapes,structures,ports,patterns,poses,digests = [],set(),set(),set(),set(),set(),set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = _puzzle_digest(puzzle)
        assert digest not in digests
        digests.add(digest)
        for split in ("train","test"):
            for pair in puzzle[split]:
                a,b = pair["input"],pair["output"]
                record = _inspect_pair(a,b)
                records.append(record)
                shapes.add(a.shape)
                ports.update(record["piece_shapes"])
                patterns.update(record["patterns"])
                poses.update(record["poses"])
                mapping = {}
                normalized = bytes(mapping.setdefault(int(v),len(mapping)) for v in a.ravel())
                structures.add((a.shape,hashlib.sha256(normalized).hexdigest()))
    assert len(shapes) >= 20 and len(structures) >= 180 and len(ports) >= 8
    assert len(patterns) == 3 and {"rotation","reflection"} <= poses
    assert len({r["route_length"] for r in records}) >= 20
    assert {r["undocked"] for r in records} >= {1,2}
    assert _puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0))
    for a,b in ((2,1),(10,1),(10,4)):
        puzzle = generate_puzzle(77,a,b)
        assert (len(puzzle["train"]),len(puzzle["test"])) == (a,b)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(2.5,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Routing must reject ambiguous and disconnected boards rather than use an
    # implementation-order tie-break. These graph cases are outside generation.
    open_board = np.zeros((3,3),dtype=np.int8)
    assert _shortest(open_board,((0,0),(2,2)))[1] == 2
    open_board[1,:] = 1
    assert _shortest(open_board,((0,0),(2,2)))[1] == 0
    return {"seeds":42,"pairs":len(records),"shapes":len(shapes),
            "normalized_inputs":len(structures),"port_shapes":len(ports),
            "stencil_types":len(patterns),"route_lengths":sorted({r["route_length"] for r in records}),
            "minimum_freed_route_cells":min(r["freed_route_cells"] for r in records),
            "independent_solver":False}


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",",":")))
