"""Clocked junctions: interfering pulse colors control a moving ray's turns.

Only NumPy and the standard library are used. No files are read by generation
or solving. All semantics are fixed; the complete program is visible in input.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0,819518718,370738775,109318836,690847441,132777589,277369768,223778062,700568898,134159850,348319471,204046081,43984245,482457183,398343260,266280308,914050135,910429027,179093527,174248387,370347688,797321634,918513986,713029570,10665617,443026723,386917204,698055853,243429692,784388875,997836627,27163986,382422952,605170933,762848398,902819661,877820898,252493360,62636863,325353109,381896004,327961803)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color range")
    return a.astype(np.int8, copy=False)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _pulse(tape, tick, mode="near"):
    fired = [(i, c, length) for i, (c, length) in enumerate(tape)
             if (tick - 1) % length == 0]
    if not fired:
        return None
    if mode == "far":
        return fired[-1][1]
    if mode == "short":
        return min(fired, key=lambda x: (x[2], x[0]))[1]
    if mode == "long":
        return min(fired, key=lambda x: (-x[2], x[0]))[1]
    return fired[0][1]


def _decode(input_grid):
    """Parse the cassette, unique arrow, and four-corner junction symbols."""
    a = _grid(input_grid)
    frame = int(a[0, 0])
    _require(np.all(a[0] == frame), "top frame")
    borders = list(map(int, np.flatnonzero(np.all(a == frame, axis=1))))
    _require(len(borders) >= 3 and borders[0] == 0, "cassette frames")
    middle, bottom = borders[1:3]
    _require(2 <= middle <= 3 and 2 <= bottom - middle <= 4 and bottom < len(a) - 2,
             "cassette heights")
    tape_band, key_band = a[1:middle], a[middle + 1:bottom]
    _require(np.all(tape_band == tape_band[0]) and np.all(key_band == key_band[0]),
             "cassette row consistency")
    row = key_band[0]
    _require(row[0] == row[-1] == frame, "key side walls")
    spans, c = [], 0
    while c < len(row):
        if row[c] == frame:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != frame:
            c += 1
        _require(c - start >= 3, "left/right key needs a clear middle")
        spans.append((start, c))
    _require(2 <= len(spans) <= 4, "key count")
    bg = int(row[spans[0][0] + 1])
    _require(bg != frame, "background/frame distinction")
    turns = {}
    for start, end in spans:
        _require(np.all(row[start + 1:end - 1] == bg), "key interior")
        left, right = int(row[start]), int(row[end - 1])
        _require((left == bg) != (right == bg), "one attached key bar")
        color, side = (right, 1) if left == bg else (left, -1)
        _require(color not in turns and color not in (bg, frame), "unique key colors")
        turns[color] = side
    row = tape_band[0]
    _require(row[0] == row[-1] == frame, "tape side walls")
    tape, gaps, c = [], [], len(row) - 2
    while c >= 1:
        gap = 0
        while c >= 1 and row[c] == bg:
            c -= 1
            gap += 1
        if c < 1:
            break
        color, end = int(row[c]), c
        _require(color in turns, "tape color must be keyed")
        while c >= 1 and row[c] == color:
            c -= 1
        tape.append((color, end - c))
        gaps.append(gap)
    _require(1 <= len(tape) <= 4 and all(1 <= n <= 7 for _, n in tape), "tape bounds")
    _require({c for c, _ in tape} == set(turns), "every key has a tape run")
    field = a[bottom + 1:].copy()
    other = set(map(int, np.unique(field))) - set(turns) - {bg, frame}
    _require(len(other) == 1, "unique carrier color")
    carrier = other.pop()
    cells = np.argwhere(field == carrier)
    _require(len(cells) == 1, "one source pixel")
    source = tuple(map(int, cells[0]))
    tail_cells = np.argwhere(field == frame)
    _require(len(tail_cells) == 1, "one tail pixel")
    tail = tuple(map(int, tail_cells[0]))
    delta = (source[0] - tail[0], source[1] - tail[1])
    _require(delta in DIRS, "tail must be cardinally adjacent to source")
    heading = DIRS.index(delta)
    nodes, covered = {}, set()
    h, w = field.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(field[r - 1, c - 1])
            if color not in turns:
                continue
            corners = ((r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1))
            if all(int(field[p]) == color for p in corners):
                cross = ((r,c),(r-1,c),(r+1,c),(r,c-1),(r,c+1))
                _require(all(int(field[p]) == bg for p in cross), "clear junction ports")
                _require(not covered.intersection(corners), "overlapping junctions")
                nodes[r, c] = color
                covered.update(corners)
    actual = {tuple(map(int, p)) for p in np.argwhere(np.isin(field, list(turns)))}
    _require(actual == covered, "all keyed field pixels are junction corners")
    centers = list(nodes)
    for i, (r, c) in enumerate(centers):
        _require(max(abs(r-source[0]), abs(c-source[1])) >= 3, "source/junction spacing")
        _require(max(abs(r-tail[0]), abs(c-tail[1])) >= 3, "tail/junction spacing")
        for x, y in centers[:i]:
            _require(max(abs(r-x), abs(c-y)) >= 5, "junction box clearance")
    return dict(field=field, bg=bg, frame=frame, carrier=carrier, source=source,
                tail=tail, heading=heading, nodes=nodes, tape=tape, turns=turns,
                crop=bottom+1, gaps=tuple(gaps), header_heights=(middle-1,bottom-middle-1))


def _walk(info, pulse_mode="near", gate_mode="match", reset=False):
    """Cell-step ray: only immutable input markers affect traversal."""
    field, tape, nodes = info["field"], info["tape"], info["nodes"]
    r, c = info["source"]
    d, tick, age = info["heading"], 0, 0
    seen, path, events = {(r, c)}, [], []
    h, w = field.shape
    while True:
        dr, dc = DIRS[d]
        r, c = r + dr, c + dc
        if not (0 <= r < h and 0 <= c < w):
            break
        _require((r, c) not in seen, "self-intersecting route")
        _require(field[r, c] == info["bg"], "ray strikes a marker or tail")
        seen.add((r, c))
        tick, age = tick + 1, age + 1
        color = _pulse(tape, age if reset else tick, pulse_mode)
        path.append((r, c, tick, color, d))
        if (r, c) in nodes:
            node_color = nodes[r, c]
            active = color == node_color
            if gate_mode == "always":
                active = True
            elif gate_mode == "own":
                active = any(k == node_color and (tick-1) % n == 0 for k,n in tape)
            elif gate_mode == "none":
                active = False
            elif gate_mode == "swap":
                active = color == node_color
            old = d
            if active:
                side = info["turns"][node_color] * (-1 if gate_mode == "swap" else 1)
                d = (d + side) % 4
                age = 0
            events.append((r, c, tick, node_color, active, old, d, color))
    _require(bool(path), "nonempty ray")
    return path, events


def _paint(info, path):
    out = info["field"].copy()
    for r, c, _, color, _ in path:
        out[r, c] = info["carrier"] if color is None else color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire output from visible input geometry and conventions."""
    info = _decode(input_grid)
    return _paint(info, _walk(info)[0])


def _audit(info, output):
    """Separate segment-distance audit of route, clock, and preserved markers.

    It jumps to the next aligned junction center, rather than using _walk.
    No execution trace is saved; bounded local lists are discarded per pair.
    """
    h, w = info["field"].shape
    r, c = info["source"]
    d, tick, visited = info["heading"], 0, {(r, c)}
    expected = info["field"].copy()
    while True:
        dr, dc = DIRS[d]
        limit = (r, w-1-c, h-1-r, c)[d]
        candidates = []
        for (x,y), color in info["nodes"].items():
            dist = (x-r)*dr + (y-c)*dc
            if dist > 0 and (x-r)*dc == (y-c)*dr:
                candidates.append((dist, x, y, color))
        encounter = min(candidates) if candidates else None
        distance = min(limit, encounter[0]) if encounter else limit
        for step in range(1, distance + 1):
            p = (r+step*dr, c+step*dc)
            _require(p not in visited, "audit: self intersection")
            _require(info["field"][p] == info["bg"], "audit: marker collision")
            visited.add(p)
            t = tick + step
            fired = [color for color, period in info["tape"] if (t-1) % period == 0]
            expected[p] = fired[0] if fired else info["carrier"]
        r, c, tick = r+distance*dr, c+distance*dc, tick+distance
        if not encounter or encounter[0] > limit:
            break
        firing = [color for color, period in info["tape"] if (tick-1) % period == 0]
        if firing and firing[0] == encounter[3]:
            d = (d + info["turns"][encounter[3]]) % 4
        # The current center is excluded by the strictly-positive distance test.
    _require(np.array_equal(expected, output), "segment audit: pulse, route, crop, or preservation")


def _make_header(rng, width, tape, turns, bg, frame, heights):
    ht, hk = heights
    head = np.full((3+ht+hk, width), bg, dtype=np.int8)
    head[[0,ht+1,-1],:] = frame
    head[1:ht+1,[0,-1]] = frame
    # Random gaps include the separator gap and outer margin. Runs are maximal.
    required = [0] + [int(tape[i][0] == tape[i-1][0]) for i in range(1,len(tape))]
    room = width - 2 - sum(n for _,n in tape) - sum(required)
    _require(room >= 0, "tape does not fit")
    extras = rng.multinomial(room, np.full(len(tape)+1,1/(len(tape)+1)))
    c = width - 1
    for i, (color, length) in enumerate(tape):
        c -= required[i] + int(extras[i])
        head[1:ht+1,c-length:c] = color
        c -= length
    # Shared frame walls, with unequal panel widths when slack is available.
    colors = list(map(int, rng.permutation(list(turns))))
    interiors = np.full(len(colors), 3, dtype=int)
    spare = width - 1 - 4*len(colors)
    _require(spare >= 0, "keys do not fit")
    interiors += rng.multinomial(spare, np.full(len(colors),1/len(colors)))
    c = 0
    for color, size in zip(colors, interiors):
        head[ht+2:-1,c] = frame
        bar = c+1 if turns[color] == -1 else c+int(size)
        head[ht+2:-1,bar] = color
        c += int(size)+1
        head[ht+2:-1,c] = frame
    return head


def _scene_info(h, w, source, heading, tape, turns):
    field = np.zeros((h,w), dtype=np.int8)
    dr,dc = DIRS[heading]
    tail = (source[0]-dr,source[1]-dc)
    _require(0 <= tail[0] < h and 0 <= tail[1] < w, "tail bounds")
    field[source] = 1
    field[tail] = 9
    return dict(field=field,bg=0,frame=9,carrier=1,source=source,tail=tail,
                heading=heading,nodes={},tape=tape,turns=turns)


def _add_node(info, position, color):
    r,c = position
    h,w = info["field"].shape
    if not (1 <= r < h-1 and 1 <= c < w-1):
        return None
    for x,y in [info["source"],info["tail"]]:
        if max(abs(r-x),abs(c-y)) < 3:
            return None
    if any(max(abs(r-x),abs(c-y)) < 5 for x,y in info["nodes"]):
        return None
    trial = dict(info, field=info["field"].copy(), nodes=dict(info["nodes"]))
    for p in ((r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1)):
        if trial["field"][p] != 0:
            return None
        trial["field"][p] = color
    trial["nodes"][r,c] = color
    try:
        _walk(trial)
    except AssertionError:
        return None
    return trial


def _finish(rng, info, palette, heights, rotation):
    field = np.rot90(info["field"], rotation).copy()
    header = _make_header(rng,field.shape[1],info["tape"],info["turns"],0,9,heights)
    inp = palette[np.vstack((header,field))]
    return dict(input=inp.astype(np.int8),output=solve(inp))


def _lesson(rng, tape, turns, palette, heights, rotation):
    """Two active bends plus a deliberately suppressed far-color junction.

    Coordinates are sampled from arithmetic constraints, not saved examples.
    The first uninterrupted segment exposes each period and a later collision.
    """
    max_side = 30 - sum(heights) - 3
    a,b = (n for _,n in tape)
    lcm = math.lcm(a,b)
    options = []
    for first in range(lcm+6, max_side-3):
        winner = _pulse(tape,first)
        if winner is None:
            continue
        other = next(c for c,_ in tape if c != winner)
        for leg in range(5,13):
            if _pulse(tape,first+leg) == other:
                options.append((first,leg,winner,other))
    _require(bool(options), "lesson arithmetic packing")
    first,leg,color1,color2 = options[int(rng.integers(len(options)))]
    w = int(rng.integers(first+4,max_side+1))
    h = int(rng.integers(max(16,leg+5),max_side+1))
    source_col = int(rng.integers(1,w-first-2))
    vertical = turns[color1]
    r = (int(rng.integers(leg+2,h-1)) if vertical == -1 else
         int(rng.integers(1,h-leg-2)))
    info = _scene_info(h,w,(r,source_col),1,tape,turns)
    for pos,color in [((r,source_col+lcm+1),tape[1][0]),
                      ((r,source_col+first),color1),
                      ((r+vertical*leg,source_col+first),color2)]:
        info = _add_node(info,pos,color)
        _require(info is not None, "lesson junction placement")
    return _finish(rng,info,palette,heights,rotation)


def _grow(rng, tape, turns, palette, heights, rotation, target):
    """Grow a route at its final straight segment; keep only valid clear scenes."""
    max_side = 30 - sum(heights) - 3
    min_side = max(15,4*len(turns)+1,sum(n for _,n in tape)+2+
                   sum(tape[i][0] == tape[i-1][0] for i in range(1,len(tape))))
    _require(min_side <= max_side, "scene/header packing")
    for _ in range(100):
        h = int(rng.integers(min_side,max_side+1))
        w = int(rng.integers(min_side,max_side+1))
        heading = int(rng.integers(4))
        source = (int(rng.integers(1,h-1)),int(rng.integers(1,w-1)))
        info = _scene_info(h,w,source,heading,tape,turns)
        for _ in range(target):
            path,events = _walk(info)
            last = max((e[2] for e in events if e[4]), default=0)
            options = [p for p in path if p[2] >= last+5 and p[3] is not None]
            rng.shuffle(options)
            found = False
            for r,c,t,color,_ in options:
                trial = _add_node(info,(r,c),color)
                if trial is None:
                    continue
                newpath,newevents = _walk(trial)
                if sum(e[4] for e in newevents) != sum(e[4] for e in events)+1:
                    continue
                # Leave a visible continuation after the newly activated node.
                if len(newpath) < t+3:
                    continue
                info,found = trial,True
                break
            if not found:
                break
        if sum(e[4] for e in _walk(info)[1]) != target:
            continue
        # Optional pass-through junctions and untouched off-route junctions.
        for _ in range(int(rng.integers(0,3))):
            path,_ = _walk(info)
            candidates = [p for p in path if p[2] >= 3]
            rng.shuffle(candidates)
            for r,c,_,pulse,_ in candidates[:30]:
                choices = [k for k in turns if k != pulse]
                if not choices:
                    continue
                trial = _add_node(info,(r,c),int(rng.choice(choices)))
                if trial is not None:
                    info = trial
                    break
        if rng.random() < .45:
            path,_ = _walk(info)
            visited = {(r,c) for r,c,*_ in path}
            for _ in range(20):
                pos = (int(rng.integers(1,h-1)),int(rng.integers(1,w-1)))
                if pos in visited:
                    continue
                trial = _add_node(info,pos,int(rng.choice(list(turns))))
                if trial is not None:
                    info = trial
                    break
        return _finish(rng,info,palette,heights,rotation)
    raise RuntimeError("bounded route construction exhausted")


def _counts(num_train,num_test):
    for name,value,lo,hi in (("num_train",num_train,2,10),("num_test",num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train),int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,nq = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # These are distribution parameters, never hidden solving conventions.
    periods = ((2,3),(2,5),(3,4))[int(rng.integers(3))]
    heights = (int(rng.integers(1,3)),int(rng.integers(1,3)))
    base = rng.permutation(10).astype(np.int8)
    base_rotation = int(rng.integers(4))
    hand = int(rng.choice([-1,1]))
    puzzle,seen = {"train":[],"test":[]},set()
    tapes_taught = []
    for i in range(nt+nq):
        split = "train" if i < nt else "test"
        if i < 2:
            tape = [(2,periods[0]),(3,periods[1])]
            if i == 1:
                tape.reverse()
            turns = {2:hand*(1 if i == 0 else -1),3:-hand*(1 if i == 0 else -1)}
            palette = base.copy()
            if i == 1:
                # Preserve the two witness colors while changing local role colors.
                palette[[0,1,9]] = palette[[1,9,0]]
            rotation = (base_rotation+2*i) % 4
            pair = _lesson(rng,tape,turns,palette,heights,rotation)
        else:
            if split == "train":
                count = int(rng.integers(2,5))
                # Rich tapes must fit all rotations and the 30-cell total height.
                max_total = 30-sum(heights)-6
                lengths = [int(rng.integers(1,8)) for _ in range(count)]
                while sum(lengths) > max_total:
                    j = int(rng.integers(count))
                    lengths[j] = max(1,lengths[j]-1)
                colors = list(range(2,2+count))
                if count >= 3 and rng.random() < .3:
                    colors[-1] = colors[0]
                tape = list(zip(colors,lengths))
                target = int(rng.integers(0,3))
            else:
                # Transfer the learned local programming to a deeper route.
                tape = list(tapes_taught[int(rng.integers(len(tapes_taught)))])
                rng.shuffle(tape)
                target = 3
            keys = sorted({c for c,_ in tape})
            turns = {c:int(rng.choice([-1,1])) for c in keys}
            turns[keys[0]],turns[keys[1]] = -1,1
            palette = rng.permutation(10).astype(np.int8)
            rotation = int(rng.integers(4))
            pair = _grow(rng,tape,turns,palette,heights,rotation,target)
        for attempt in range(16):
            sig = _signature(pair["input"])
            if sig not in seen:
                break
            # Finite color rotations preserve all semantic roles and witnesses.
            pair = {name:((grid.astype(int)+1)%10).astype(np.int8) for name,grid in pair.items()}
        else:
            raise RuntimeError("duplicate input fallback exhausted")
        seen.add(_signature(pair["input"]))
        puzzle[split].append(pair)
        if split == "train":
            tapes_taught.append(tape)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _rival_diff(info, expected, **kwargs):
    try:
        result = _paint(info,_walk(info,**kwargs)[0])
    except AssertionError:
        return True
    return not np.array_equal(result,expected)


def _evidence(pairs):
    orders,edges,headings = set(),set(),{-1:set(),1:set()}
    suppressed,nonreset,passive,active = False,False,False,False
    for pair in pairs:
        info = _decode(pair["input"])
        path,events = _walk(info)
        tape = info["tape"]
        if len(tape) == 2 and tape[0][0] != tape[1][0]:
            # Visible prefix recurrence and both color-priority directions.
            first_turn = min((e[2] for e in events if e[4]),default=len(path)+1)
            a,b = tape
            if first_turn > math.lcm(a[1],b[1])+1 and any(p[3] == b[0] for p in path[:first_turn-1]):
                orders.add(a[1] < b[1])
                edges.add((a[0],b[0]))
        for r,c,t,color,is_active,old,new,pulse in events:
            active |= is_active
            passive |= not is_active
            if is_active:
                headings[info["turns"][color]].add(old)
            else:
                suppressed |= any(k == color and (t-1)%n == 0 for k,n in tape) and pulse != color
        nonreset |= _rival_diff(info,np.asarray(pair["output"]),reset=True)
    return dict(orders=orders,color_cycle=any((b,a) in edges for a,b in edges),
                suppressed=suppressed,nonreset=nonreset,passive=passive,active=active,headings=headings)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen,active_counts = set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"}, "pair keys")
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            info = _decode(inp)
            _require(out.shape == info["field"].shape, "crop")
            path,events = _walk(info)
            _require(np.array_equal(_paint(info,path),out), "cell-step solution")
            _audit(info,out)
            fixed = info["field"] != info["bg"]
            _require(np.array_equal(out[fixed],info["field"][fixed]), "preserve all input marks")
            _require(len(path) <= info["field"].size-2, "bounded simple route")
            active_counts[split].append(sum(e[4] for e in events))
    evidence = _evidence(puzzle["train"])
    _require(evidence["orders"] == {False,True} and evidence["color_cycle"], "priority witnesses")
    _require(evidence["suppressed"] and evidence["passive"], "suppressed junction witness")
    _require(evidence["nonreset"], "continuous clock witness")
    _require(all(len(h) >= 2 for h in evidence["headings"].values()), "relative turn headings")
    for mode in ("short","long","far"):
        _require(any(_rival_diff(_decode(p["input"]),np.asarray(p["output"]),pulse_mode=mode)
                     for p in puzzle["train"]), "priority alternative fits training")
    for mode in ("own","always","none","swap"):
        _require(any(_rival_diff(_decode(p["input"]),np.asarray(p["output"]),gate_mode=mode)
                     for p in puzzle["train"]), "junction alternative fits training")
    _require(max(active_counts["train"]) <= 2 and all(n == 3 for n in active_counts["test"]),
             "test shift must require a deeper route")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for name in ("input","output"):
                a = np.asarray(pair[name],dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed variation, fixed arithmetic, metamorphic, and mutation checks."""
    # Fixed clock oracle, including near-longer and hidden far-color pulses.
    _require([_pulse([(2,3),(3,2)],t) for t in range(1,14)] ==
             [2,None,3,2,3,None,2,None,3,2,3,None,2], "hand clock regression")
    # Hand-routed fixture: pass the color-3 junction at tick 7, left at
    # color 2/tick 13, then right at color 3/tick 21. No generated labels.
    fixture = _scene_info(15,20,(10,1),1,[(2,3),(3,2)],{2:-1,3:1})
    for pos,color in (((10,8),3),((10,14),2),((2,14),3)):
        fixture = _add_node(fixture,pos,color)
        _require(fixture is not None, "hand fixture geometry")
    expected = fixture["field"].copy()
    coordinates = ([(10,c) for c in range(2,15)] +
                   [(r,14) for r in range(9,1,-1)] + [(2,c) for c in range(15,20)])
    fixed_clock = [2,1,3,2,3,1]*4 + [2,1]
    for pos,color in zip(coordinates,fixed_clock):
        expected[pos] = color
    for rotation in range(4):
        field = np.rot90(fixture["field"],rotation).copy()
        head = _make_header(np.random.default_rng(41),field.shape[1],fixture["tape"],
                            fixture["turns"],0,9,(1,1))
        _require(np.array_equal(solve(np.vstack((head,field))),np.rot90(expected,rotation)),
                 "hand route, suppression, continuous phase, and rotation")
    shapes,layouts,headings,periods,run_counts,backgrounds,signatures = (set() for _ in range(7))
    passive = untouched = repeats = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train","test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                path,events = _walk(info)
                shapes.add(info["field"].shape)
                headings.add(info["heading"])
                run_counts.add(len(info["tape"]))
                periods.update(n for _,n in info["tape"])
                backgrounds.add(info["bg"])
                layouts.add((info["field"].shape,info["source"],info["heading"],
                             tuple(sorted(info["nodes"])),tuple(n for _,n in info["tape"])))
                passive += any(not e[4] for e in events)
                untouched += len(events) < len(info["nodes"])
                repeats += len(set(c for c,_ in info["tape"])) < len(info["tape"])
    _require(len(signatures) == 42 and len(shapes) >= 35 and len(layouts) >= 180, "structural variation")
    _require(headings == set(range(4)) and len(backgrounds) == 10, "role/orientation variation")
    _require({2,3,4} <= run_counts and {1,2,3,4,5} <= periods, "program variation")
    _require(passive >= 84 and untouched >= 8 and repeats >= 3, "junction/tape variation")
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(517,nt,nq)
            _require((len(p["train"]),len(p["test"])) == (nt,nq), "explicit counts")
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(None,1)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    _require(_digest(p) == _digest(generate_puzzle(0)), "determinism")
    pair = p["train"][0]
    perm = np.array([7,2,9,0,6,4,1,8,5,3],dtype=np.int8)
    _require(np.array_equal(solve(perm[pair["input"]]),perm[pair["output"]]), "palette equivariance")
    bad = {s:[{k:v.copy() for k,v in q.items()} for q in p[s]] for s in p}
    info = _decode(bad["test"][0]["input"])
    bad["test"][0]["output"][info["tail"]] = info["bg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("marker mutation accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, clock/route invariants")
