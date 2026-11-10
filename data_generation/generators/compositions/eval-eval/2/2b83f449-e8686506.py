"""Load-constrained assembly of a switch railway. Standard library + NumPy."""
from collections import Counter
import itertools
import numpy as np


BG, FRAME, SOCKET, DEMAND = 0, 5, 8, 4
ROLE_COLORS = (1, 2, 3, 6, 7, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return (a.shape, a.tobytes())


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        group, stack = [], [first]
        while stack:
            p = stack.pop()
            group.append(p)
            for dr, dc in DIRS:
                q = (p[0] + dr, p[1] + dc)
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(group)))
    return result


def _piece(grid, cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0, int(grid[r, c])) for r, c in cells))


def _paths(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    result = []
    for component in _components(cells):
        group = set(component)
        adjacency = {p: [(p[0]+dr, p[1]+dc) for dr, dc in DIRS
                         if (p[0]+dr, p[1]+dc) in group] for p in component}
        _require(all(len(v) in (1, 2) for v in adjacency.values()), "branch or isolated track")
        ends = sorted(p for p in component if len(adjacency[p]) == 1)
        _require(len(ends) == 2, "closed track loop")
        path, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            nxt = [p for p in adjacency[current] if p != previous][0]
            path.append(nxt)
            previous, current = current, nxt
        _require(len(path) == len(component), "incomplete path")
        changes = {b[0]-a[0] for a, b in zip(path, path[1:]) if a[0] != b[0]}
        _require(len(changes) <= 1, "internal gravity trap")
        result.append(path)
    return result


def _route(core, roles, demands=None):
    """Actuate simultaneously, trace simple paths, and settle conserved beads."""
    substrate, track, bead, gate = roles
    h, w = core.shape
    _require(h >= 3 and h % 2 == 1 and w >= 7, "invalid railway extent")
    _require(set(map(int, np.unique(core))) <= set(roles), "unknown railway cell")
    _require(np.all(np.isin(core[::2], [substrate, track, bead])), "invalid lane cell")
    _require(np.all(np.isin(core[1::2], [substrate, gate])), "invalid separator cell")
    _require(np.all(np.isin(core[::2, [0, w-1]], [track, bead])), "missing outer lane terminal")
    writes, centers = {}, []
    for r in range(1, h, 2):
        xs = set(map(int, np.flatnonzero(core[r] == gate)))
        for c0 in sorted(c for c in xs if c-1 not in xs):
            _require({c0, c0+1, c0+2} <= xs and c0+3 not in xs, "divider is not three cells")
            c = c0+1
            _require(1 <= c < w-1, "divider at edge")
            _require(np.all(core[[r-1, r+1], c-1:c+2] == track), "divider neighborhood is broken")
            centers.append((r, c))
            for rr in range(r-1, r+2):
                for cc in range(c-1, c+2):
                    value = gate if cc == c else track
                    _require((rr, cc) not in writes or writes[rr, cc] == value, "conflicting rotations")
                    writes[rr, cc] = value
    structural = core.copy()
    structural[structural == bead] = track
    for p, color in writes.items():
        structural[p] = color
    paths = _paths(structural == track)
    beads = set(map(tuple, np.argwhere(core == bead)))
    endpoints = {p for path in paths for p in (path[0], path[-1])}
    _require(beads <= endpoints, "bead is not at an exposed endpoint")
    _require(all(r % 2 == 0 and (c in (0, w-1) or
                 core[r, c-1] == substrate or core[r, c+1] == substrate)
                 for r, c in endpoints), "unexposed terminal")
    answer, loads, targets = structural.copy(), {}, set()
    single = collision = level = span = moving_span = single_span = 0
    for path in paths:
        occupied = beads & {path[0], path[-1]}
        count = len(occupied)
        span = max(span, abs(path[-1][0]-path[0][0])//2)
        if path[0][0] == path[-1][0]:
            chosen = occupied
            level += bool(count)
            for p in (path[0], path[-1]):
                loads[p] = int(p in occupied)
        else:
            ordered = path if path[0][0] > path[-1][0] else path[::-1]
            chosen = set(ordered[:count])
            loads[ordered[0]], loads[ordered[-1]] = count, 0
            single += count == 1 and chosen != occupied
            collision += count == 2
            if chosen != occupied:
                moving_span = max(moving_span, abs(path[-1][0]-path[0][0])//2)
                if count == 1:
                    single_span = max(single_span, abs(path[-1][0]-path[0][0])//2)
        targets.update(chosen)
    for p in targets:
        answer[p] = bead
    _require(len(targets) == len(beads), "bead conservation failed")
    if demands is not None:
        # Demand ticks exist only for the outer endpoints. Interior endpoints
        # beside permanent lane breaks must finish empty.
        _require(all(loads.get(p, 0) == n for p, n in demands.items()), "exit capacity mismatch")
        _require(all(not n for p, n in loads.items() if p not in demands), "beads stranded at an unmarked break")
    return answer, dict(paths=paths, centers=centers, writes=writes, beads=beads,
                        targets=targets, loads=loads, single=single,
                        collision=collision, level=level, span=span,
                        moving_span=moving_span, single_span=single_span)


def _parse(inp):
    """Visible frame locates the board; lane geometry locates semantic roles."""
    frame = np.argwhere(inp == FRAME)
    _require(len(frame) > 0, "no frame")
    r0, c0 = frame.min(axis=0)
    r1, c1 = frame.max(axis=0)+1
    board = inp[r0:r1, c0:c1].copy()
    bh, bw = board.shape
    _require(bh >= 5 and bh % 2 == 1 and bw >= 13, "bad frame dimensions")
    _require(np.all(board[[0, -1]] == FRAME), "incomplete frame")
    _require(np.all(board[1:-1, [2, bw-3]] == FRAME), "missing gutter separators")
    for r in range(1, bh-1):
        _require(set(map(int, board[r, [0, 1, bw-2, bw-1]])) <= {FRAME, DEMAND}, "invalid demand gutter")
        if r % 2 == 0:
            _require(np.all(board[r, [0, 1, bw-2, bw-1]] == FRAME), "capacity away from lane")
    core = board[1:-1, 3:-3].copy()
    _require(np.all(core[:, [0, -1]] != SOCKET), "socket removes a terminal")
    lane_values = Counter(map(int, core[::2][core[::2] != SOCKET]))
    separator_values = Counter(map(int, core[1::2][core[1::2] != SOCKET]))
    track = lane_values.most_common(1)[0][0]
    substrate = separator_values.most_common(1)[0][0]
    _require(all(lane_values[track] > count for color,count in lane_values.items() if color != track), "tied lane roles")
    _require(all(separator_values[substrate] > count for color,count in separator_values.items() if color != substrate), "tied separator roles")
    _require(track in ROLE_COLORS and substrate in ROLE_COLORS and track != substrate, "invalid track/substrate")
    bead_values = set(lane_values)-{track, substrate}
    _require(len(bead_values) == 1, "one bead identity is required")
    bead = next(iter(bead_values))
    outside = inp.copy()
    outside[r0:r1, c0:c1] = BG
    cells = set(map(tuple, np.argwhere(outside != BG)))
    components = _components(cells)
    _require(components and all(int(outside[p]) in ROLE_COLORS for comp in components for p in comp), "bad loose pieces")
    pieces = [_piece(outside, comp) for comp in components]
    _require(all(v != bead for p in pieces for r, c, v in p), "a loose control piece contains a bead")
    extra = set(int(outside[p]) for p in cells) | set(map(int, np.unique(core)))
    extra -= {SOCKET, substrate, track, bead}
    _require(len(extra) == 1 and next(iter(extra)) in ROLE_COLORS, "divider role is not identifiable")
    gate = next(iter(extra))
    _require(len({substrate, track, bead, gate}) == 4, "roles must differ")
    demands = {}
    for r in range(0, core.shape[0], 2):
        left = board[r+1, :2]
        right = board[r+1, -2:]
        _require(not (left[0] == DEMAND and left[1] != DEMAND), "noncontiguous capacity")
        _require(not (right[1] == DEMAND and right[0] != DEMAND), "noncontiguous capacity")
        demands[(r, 0)] = int(np.count_nonzero(left == DEMAND))
        demands[(r, core.shape[1]-1)] = int(np.count_nonzero(right == DEMAND))
    _require(sum(demands.values()) == np.count_nonzero(core == bead), "demand/bead total differs")
    _require(sum(len(p) for p in pieces) == np.count_nonzero(core == SOCKET), "inventory area differs")
    return dict(board=board, core=core, box=(int(r0), int(c0), int(r1), int(c1)),
                roles=(substrate, track, bead, gate), pieces=pieces,
                components=components, demands=demands)


def _placements(core, pieces):
    holes = set(map(tuple, np.argwhere(core == SOCKET)))
    types = sorted(set(pieces))
    counts = tuple(pieces.count(t) for t in types)
    moves, by_cell = [], {p: [] for p in holes}
    for i, piece in enumerate(types):
        ph = max(r for r, c, v in piece)+1
        pw = max(c for r, c, v in piece)+1
        for r in range(core.shape[0]-ph+1):
            for c in range(core.shape[1]-pw+1):
                paint = tuple((r+dr, c+dc, color) for dr, dc, color in piece)
                occupied = frozenset((rr, cc) for rr, cc, v in paint)
                if occupied <= holes:
                    index = len(moves)
                    moves.append((i, occupied, paint))
                    for p in occupied:
                        by_cell[p].append(index)
    return holes, types, counts, moves, by_cell


def _assemblies(info, *, reuse=False, enforce_load=True, limit=2):
    """Joint exact cover and routing constraint; enumerate distinct boards."""
    core = info["core"]
    holes, types, counts, moves, by_cell = _placements(core, info["pieces"])
    answers, seen, nodes = [], set(), 0

    def visit(remaining, available, painted):
        nonlocal nodes
        nodes += 1
        _require(nodes <= 60000, "assembly search exceeded bound")
        if len(answers) >= limit:
            return
        if not remaining:
            if not reuse and any(available):
                return
            assembled = core.copy()
            for r, c, color in painted:
                assembled[r, c] = color
            try:
                result, details = _route(assembled, info["roles"],
                                         info["demands"] if enforce_load else None)
            except ValueError:
                return
            key = _key(assembled)
            if key not in seen:
                seen.add(key)
                answers.append((assembled, result, details))
            return
        options = None
        for p in sorted(remaining):
            choices = [j for j in by_cell[p] if (reuse or available[moves[j][0]])
                       and moves[j][1] <= remaining]
            if not choices:
                return
            if options is None or len(choices) < len(options):
                options = choices
                if len(options) == 1:
                    break
        for j in options:
            i, occupied, paint = moves[j]
            reduced = list(available)
            if not reuse:
                reduced[i] -= 1
            visit(remaining-occupied, tuple(reduced), painted+paint)
            if len(answers) >= limit:
                return
    visit(holes, counts, ())
    return answers, nodes


def _analyze(inp):
    info = _parse(_grid(inp))
    answers, nodes = _assemblies(info)
    _require(len(answers) == 1, "railway must have one load-compatible inventory assembly")
    assembled, settled, details = answers[0]
    out = info["board"].copy()
    out[1:-1, 3:-3] = settled
    info.update(assembled=assembled, settled=settled, details=details, nodes=nodes, output=out)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique assembled, actuated, settled board from this input."""
    return _analyze(input_grid)["output"].copy()


def _mask(width, left_notch, right_notch):
    # Rectangles and connected H/notched modules; central 3x3 stays complete.
    return tuple((r, c) for r in range(3) for c in range(width)
                 if r != 1 or left_notch <= c < width-right_notch)


def _scatter(rng, board, pieces):
    bh, bw = board.shape
    for attempt in range(16):
        h = int(rng.integers(max(bh+7, 22), 31)) if attempt < 12 else 30
        w = int(rng.integers(max(bw, 22), 31)) if attempt < 12 else 30
        r0, c0 = int(rng.integers(h-bh+1)), int(rng.integers(w-bw+1))
        canvas = np.zeros((h, w), dtype=np.int8)
        canvas[r0:r0+bh, c0:c0+bw] = board
        blocked = np.zeros((h, w), dtype=bool)
        blocked[max(0,r0-1):min(h,r0+bh+1), max(0,c0-1):min(w,c0+bw+1)] = True
        order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), i))
        for i in order:
            piece = pieces[i]
            ph = max(r for r,c,v in piece)+1
            pw = max(c for r,c,v in piece)+1
            choices = [(r,c) for r in range(h-ph+1) for c in range(w-pw+1)
                       if all(not blocked[r+dr,c+dc] for dr,dc,v in piece)]
            if not choices:
                break
            r,c = choices[int(rng.integers(len(choices)))]
            for dr,dc,v in piece:
                rr,cc = r+dr,c+dc
                canvas[rr,cc] = v
                for ar,ac in ((rr,cc), *( (rr+dy,cc+dx) for dy,dx in DIRS)):
                    if 0 <= ar < h and 0 <= ac < w:
                        blocked[ar,ac] = True
        else:
            return canvas
    return None


def _sample_pair(rng, roles, mode):
    substrate, track, bead, gate = roles
    for attempt in range(600):
        n = int(rng.integers(5 if mode == "query" else 4, 8 if mode == "query" else 7))
        width = int(rng.integers(13, 22))
        m = int(rng.integers(3, min(6, n-1)+1))
        selected_rows = sorted(map(int, rng.choice(n-1, m, replace=False)))
        module_width = int(rng.choice([3,5,5,7]))
        max_notch = (module_width-3)//2
        shared_shape = _mask(module_width, int(rng.integers(max_notch+1)), int(rng.integers(max_notch+1)))
        masks = [shared_shape for _ in range(m)]
        if m >= 4 and rng.random() < .6:
            masks[-1] = _mask(3, 0, 0)
        boxes, sockets = [], []
        for i, sep in enumerate(selected_rows):
            mask = masks[i]
            mw = max(c for r,c in mask)+1
            candidates = [x for x in range(2,width-mw-1)
                          if all(abs(sep-s) > 1 or x+mw <= p or p+pw <= x
                                 for s,p,pw in boxes)]
            if not candidates:
                break
            x = candidates[int(rng.integers(len(candidates)))]
            boxes.append((sep,x,mw))
            sockets.append(tuple((2*sep+dr,x+dc) for dr,dc in mask))
        else:
            core = np.full((2*n-1,width), substrate, dtype=np.int8)
            core[::2] = track
            active = np.zeros(m,dtype=bool)
            active[rng.choice(m, int(rng.integers(1,m)), replace=False)] = True
            for i,(sep,x,mw) in enumerate(boxes):
                if active[i]:
                    # A divider's centered 3x3 must be entirely in the module.
                    viable = [c for c in range(1,mw-1)
                              if all((dr,dc) in masks[i] for dr in range(3) for dc in range(c-1,c+2))]
                    if not viable:
                        break
                    c = x+viable[int(rng.integers(len(viable)))]
                    core[2*sep+1,c-1:c+2] = gate
            else:
                # Installed dividers extend paths without adding more inventory.
                for sep in range(n-1):
                    if sep in selected_rows or rng.random() > .65:
                        continue
                    options = [c for c in range(2,width-2)
                               if all(abs(sep-s)>1 or c+1 < p or c-1 >= p+pw for s,p,pw in boxes)]
                    if options:
                        c = options[int(rng.integers(len(options)))]
                        core[2*sep+1,c-1:c+2] = gate
                try:
                    _, routing = _route(core, roles)
                except ValueError:
                    continue
                paths = routing["paths"]
                unequal = [p for p in paths if p[0][0] != p[-1][0]]
                level = [p for p in paths if p[0][0] == p[-1][0]]
                if len(unequal) < 2 or not level:
                    continue
                span = routing["span"]
                if ((mode == "query" and span < 3) or
                    (mode == "basic" and span > 1) or
                    (mode == "linked" and span != 2) or
                    (mode == "demo" and span > 2)):
                    continue
                # Random occupancy with two explicit educational cases.
                occupied = set()
                probability = float(rng.uniform(.10,.75))
                for path in paths:
                    occupied.update(p for p in (path[0],path[-1]) if rng.random() < probability)
                eligible = [i for i,p in enumerate(unequal)
                            if mode not in ("linked","query") or
                            abs(p[0][0]-p[-1][0])//2 >= (3 if mode == "query" else 2)]
                single_index = int(rng.choice(eligible))
                p = unequal[single_index]
                occupied.difference_update((p[0],p[-1]))
                occupied.add(min(p[0],p[-1]))
                p = unequal[int(rng.choice([i for i in range(len(unequal)) if i != single_index]))]
                occupied.update((p[0],p[-1]))
                p = level[int(rng.integers(len(level)))]
                occupied.add(p[int(rng.choice([0,-1]))])
                hole_cells = set(itertools.chain.from_iterable(sockets))
                if occupied & hole_cells:
                    continue
                for p in occupied:
                    core[p] = bead
                settled,routing = _route(core,roles)
                pieces = [_piece(core,socket) for socket in sockets]
                blank = core.copy()
                for p in hole_cells:
                    blank[p] = SOCKET
                demands = {p: routing["loads"][p] for p in routing["loads"]}
                info = dict(core=blank, roles=roles, pieces=pieces, demands=demands)
                answers,_ = _assemblies(info)
                if len(answers) != 1 or not np.array_equal(answers[0][0],core):
                    continue
                alternatives,_ = _assemblies(info,enforce_load=False)
                if len(alternatives) < 2:
                    continue
                # Require a visible exact-inventory witness: copying the
                # supplied modules would make at least two load-valid boards.
                reusable,_ = _assemblies(info,reuse=True)
                if len(reusable) < 2:
                    continue
                board = np.full((2*n+1,width+6),FRAME,dtype=np.int8)
                board[1:-1,3:-3] = blank
                for (r,c),count in demands.items():
                    if c == 0:
                        if count:
                            board[r+1,2-count:2] = DEMAND
                    else:
                        if count:
                            board[r+1,-2:-2+count if count == 1 else None] = DEMAND
                canvas = _scatter(rng,board,pieces)
                if canvas is None:
                    continue
                expected = board.copy()
                expected[1:-1,3:-3] = settled
                # Reflect only horizontally; gravity stays visibly downward.
                if rng.random() < .5:
                    canvas,expected = np.fliplr(canvas).copy(),np.fliplr(expected).copy()
                try:
                    if not np.array_equal(solve(canvas),expected):
                        continue
                except ValueError:
                    continue
                return dict(input=canvas,output=expected)
    raise RuntimeError("bounded construction did not find a unique load-constrained railway")


def _counts(num_train,num_test):
    for name,v,lo,hi in (("num_train",num_train,2,10),("num_test",num_test,1,4)):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or not lo <= v <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train),int(num_test)


def generate_puzzle(seed: int,num_train: int=4,num_test: int=1) -> dict:
    nt,nq = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_roles = tuple(map(int,rng.choice(ROLE_COLORS,4,replace=False)))
    vary_roles = bool(rng.random() < .7)
    demonstration_positions = list(map(int,rng.permutation(nt)))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",nt),("test",nq)):
        for i in range(count):
            for repeat in range(20):
                roles = tuple(map(int,rng.choice(ROLE_COLORS,4,replace=False))) if vary_roles else base_roles
                mode = "query" if split == "test" else (
                    "basic" if i == demonstration_positions[0] else
                    "linked" if i == demonstration_positions[1] else "demo")
                pair = _sample_pair(rng,roles,mode)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"},"invalid puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in puzzle),"splits must be lists")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen=set()
    linked_motion = False
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"},"invalid pair keys")
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            _require(_key(inp) not in seen,"duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            _require(np.array_equal(out,info["output"]),"wrong answer")
            core,assembled,settled = info["core"],info["assembled"],info["settled"]
            fixed = core != SOCKET
            _require(np.array_equal(core[fixed],assembled[fixed]),"assembly changed fixed cells")
            supply = Counter(v for piece in info["pieces"] for r,c,v in piece)
            _require(supply == Counter(map(int,assembled[~fixed])),"piece colors not conserved in assembly")
            d = info["details"]
            _require(d["single"] and d["collision"] and d["level"],"missing motion/packing/level evidence")
            _require(np.count_nonzero(settled == info["roles"][2]) == len(d["beads"]),"lost bead")
            untouched = np.ones(core.shape,dtype=bool)
            for p in set(d["writes"]) | d["beads"] | d["targets"]:
                untouched[p] = False
            _require(np.array_equal(assembled[untouched],settled[untouched]),"actuation changed an unrelated cell")
            for path in d["paths"]:
                cells = set(path)
                _require(len(cells & d["beads"]) == len(cells & d["targets"]),"beads crossed between paths")
                if path[0][0] == path[-1][0]:
                    _require(cells & d["beads"] == cells & d["targets"],"level-path bead moved")
            _require(not np.any(out == SOCKET),"unfilled socket")
            no_load,_ = _assemblies(info,enforce_load=False)
            reusable,_ = _assemblies(info,reuse=True)
            _require(len(no_load) == 2,"shape alone already determines assembly")
            _require(len(reusable) == 2,"exact inventory has no demonstrated role")
            if split == "test":
                _require(d["single_span"] >= 3,"test lacks singleton movement on the longer route")
            else:
                _require(d["span"] <= 2,"training exceeds the demonstrated route-span regime")
                linked_motion |= d["single_span"] == 2
    _require(linked_motion,"training lacks an occupied two-step route")


def self_test():
    # Small hand-built regression isolates rotation, packing, and level ties.
    fixture=np.full((5,9),1,dtype=np.int8)
    fixture[::2]=2
    fixture[1,3:6]=6
    fixture[0,0]=fixture[2,0]=fixture[4,8]=3
    expected=fixture.copy()
    expected[0:3,3:6]=2
    expected[0:3,4]=6
    expected[0,0]=2
    expected[2,0:2]=3
    _require(np.array_equal(_route(fixture,(1,2,3,6))[0],expected),"rotation/packing/level regression")
    cycle=np.full((3,11),1,dtype=np.int8)
    cycle[::2]=2
    cycle[1,2:5]=cycle[1,6:9]=6
    try:
        _route(cycle,(1,2,3,6))
    except ValueError:
        pass
    else:
        raise AssertionError("closed route accepted")
    # A handed notch cannot be repaired by silently reflecting a loose object.
    notch=np.full((3,5),1,dtype=np.int8)
    for p in _mask(5,1,0):
        notch[p]=SOCKET
    reversed_piece=tuple((r,c,2 if r != 1 else 1) for r,c in _mask(5,0,1))
    _require(not _placements(notch,[reversed_piece])[3],"piece reflection was allowed")
    records, signatures, shapes, layouts, palettes, modules = [],set(),set(),set(),set(),set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p[g]) for s in ("train","test") for p in puzzle[s] for g in ("input","output")))
        local_shapes=set()
        for split in ("train","test"):
            for pair in puzzle[split]:
                info=_analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"]),pair["output"]),"solve disagrees")
                shapes.add(pair["input"].shape)
                local_shapes.add(info["core"].shape)
                palettes.add(info["roles"])
                norm=np.zeros(info["core"].shape,dtype=np.int8)
                for i,color in enumerate((*info["roles"],SOCKET),1):
                    norm[info["core"] == color]=i
                layouts.add(_key(norm))
                modules.update(tuple((r,c) for r,c,v in p) for p in info["pieces"])
                records.append((split,info))
        _require(len(local_shapes)>1,"within-puzzle geometry does not vary")
    _require(len(signatures)==42 and len(shapes)>=20 and len(layouts)>=150,"insufficient variation")
    _require(len(palettes)>=30 and len(modules)>=8,"roles/module geometry does not vary")
    _require(len({len(i["pieces"]) for s,i in records})>=3,"piece counts do not vary")
    _require(any(i["details"]["span"]>=3 for s,i in records),"no longer routes")
    first=generate_puzzle(2026,2,1)
    generate_puzzle(2027,2,1)
    second=generate_puzzle(2026,2,1)
    _require(all(np.array_equal(a[g],b[g]) for s in first for a,b in zip(first[s],second[s])
                 for g in ("input","output")),"nondeterminism")
    sample=first["test"][0]
    untouched=sample["input"].copy()
    _require(np.array_equal(solve(np.fliplr(untouched)),np.fliplr(sample["output"])),"horizontal-reflection equivariance")
    _require(np.array_equal(untouched,sample["input"]),"solve mutated its input")
    exchange=dict(zip(ROLE_COLORS,ROLE_COLORS[::-1]))
    recolored_input=untouched.copy()
    recolored_output=sample["output"].copy()
    for old,new in exchange.items():
        recolored_input[untouched == old]=new
        recolored_output[sample["output"] == old]=new
    _require(np.array_equal(solve(recolored_input),recolored_output),"role-color equivariance")
    visible=_parse(untouched)
    rescattered=_scatter(np.random.default_rng(123),visible["board"],visible["pieces"])
    _require(rescattered is not None and np.array_equal(solve(rescattered),sample["output"]),"loose-piece position influenced solve")
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.2),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Validation must reject corrupt labels and a missing inventory cell.
    for kind in ("answer","inventory"):
        bad={s:[{g:v.copy() for g,v in p.items()} for p in first[s]] for s in first}
        p=bad["test"][0]
        if kind=="answer":
            p["output"][0,0]=BG
        else:
            info=_parse(p["input"])
            p["input"][info["components"][0][0]]=BG
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corruption accepted")
    return {"seeds":42,"pairs":len(records),"canvas_shapes":len(shapes),
            "normalized_layouts":len(layouts),"role_palettes":len(palettes),"module_shapes":len(modules)}
