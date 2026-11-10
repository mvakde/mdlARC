"""Jammed clockwork: unary rotation commands interrupted by rigid collision.

Standard library and NumPy only. Generation and validation never write files.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0,850329812,767712329,76088174,861684414,822439886,452657357,
276433557,263652878,491873112,968394114,972606983,900675493,312814993,
664443103,769418838,473590703,254636233,734860699,392034913,985875494,
958188655,253694916,64239550,809026131,529691107,582869159,976660832,
405082452,211489669,789240487,748963258,413840004,601299543,683343033,
652604318,952902770,279141271,780802424,998244948,262241434,719723502)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 / 1..4 with num_train > num_test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid color bounds")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _rot(p, n, k=1):
    r, c = p
    for _ in range(k % 4):
        r, c = c, n - 1 - r
    return int(r), int(c)


def _decode(value):
    """Recover all roles from the input; frame color never has an exterior bar."""
    a = _grid(value)
    candidates = []
    for wall in map(int, np.unique(a)):
        cells = np.argwhere(a == wall)
        lo, hi = cells.min(0), cells.max(0) + 1
        h, w = map(int, hi - lo)
        if h != w or h % 2 != 1 or h < 7:
            continue
        r, c = map(int, lo)
        board = a[r:r+h, c:c+h]
        if not (np.all(board[0] == wall) and np.all(board[-1] == wall)
                and np.all(board[:, 0] == wall) and np.all(board[:, -1] == wall)):
            continue
        outside = np.ones(a.shape, dtype=bool)
        outside[r:r+h, c:c+h] = False
        if not outside.any():
            continue
        vals, counts = np.unique(a[outside], return_counts=True)
        if np.sum(counts == counts.max()) != 1:
            continue
        bg = int(vals[counts.argmax()])
        hub = int(board[h//2, h//2])
        if len({bg, wall, hub}) != 3 or np.sum(a == hub) != 1:
            continue
        colors = sorted(set(map(int, np.unique(board))) - {bg, wall, hub})
        if not 2 <= len(colors) <= 5:
            continue
        if set(map(int, vals)) - {bg} - set(colors):
            continue
        commands, bars = {}, {}
        valid = True
        halo = np.zeros(a.shape, dtype=bool)
        halo[max(0,r-1):min(a.shape[0],r+h+1), max(0,c-1):min(a.shape[1],c+h+1)] = True
        for color in colors:
            pts = np.argwhere(outside & (a == color))
            length = len(pts)
            commands[color] = length
            if length == 0:
                continue
            rr, cc = pts.T
            horizontal = np.ptp(rr) == 0 and np.ptp(cc) + 1 == length
            vertical = np.ptp(cc) == 0 and np.ptp(rr) + 1 == length
            if not (1 <= length <= 9 and (horizontal or vertical)) or np.any(halo[rr, cc]):
                valid = False
                break
            bars[color] = pts
        if not valid or not bars:
            continue
        all_bars = [(color, tuple(p)) for color, pts in bars.items() for p in pts]
        if any(x != y and max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1
               for i, (x, p) in enumerate(all_bars) for y, q in all_bars[i+1:]):
            continue
        swept = np.zeros(board.shape, dtype=bool)
        for color in colors:
            mask = board == color
            territory = np.logical_or.reduce([np.rot90(mask, k) for k in range(4)])
            if np.any(swept & territory) or territory[h//2, h//2] or (
                territory[0].any() or territory[-1].any() or
                territory[:, 0].any() or territory[:, -1].any()
            ):
                valid = False
                break
            swept |= territory
        if valid:
            candidates.append({"board": board.copy(), "wall": wall, "hub": hub,
                               "background": bg, "colors": colors, "commands": commands,
                               "bars": bars, "position": (r, c)})
    _require(len(candidates) == 1, "housing must be uniquely decodable from input")
    return candidates[0]


def _render(info, mode="normal"):
    board = info["board"]
    out = np.full(board.shape, info["background"], dtype=np.int8)
    walls = board == info["wall"]
    out[walls] = info["wall"]
    out[board == info["hub"]] = info["hub"]
    for color in info["colors"]:
        mask = board == color
        count = info["commands"][color]
        if mode == "no_turns":
            count = 0
        if mode == "modulo_first":
            count %= 4
        if mode == "no_walls":
            mask = np.rot90(mask, -count)
        elif mode == "direct_final":
            target = np.rot90(mask, -count)
            if not np.any(target & walls):
                mask = target
        else:
            for _ in range(count):
                nxt = np.rot90(mask, -1)
                if np.any(nxt & walls):
                    break
                mask = nxt
        # Alternative renders may overwrite a wall; they are counterfactuals.
        out[mask] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_decode(input_grid))


def _coordinate_check(info):
    """A second calculation using sets and coordinates, with bounded state."""
    board = info["board"]
    n = len(board)
    walls = set(map(tuple, np.argwhere(board == info["wall"]).tolist()))
    expected = np.full(board.shape, info["background"], dtype=np.int8)
    expected[board == info["wall"]] = info["wall"]
    expected[board == info["hub"]] = info["hub"]
    records = []
    for color in info["colors"]:
        source = set(map(tuple, np.argwhere(board == color).tolist()))
        orientations = [{_rot(p, n, k) for p in source} for k in range(4)]
        first = next((j for j in range(1, 4) if orientations[j] & walls), 0)
        count = info["commands"][color]
        done = min(count, first-1) if first else count
        final = orientations[done % 4]
        for p in final:
            _require(expected[p] == info["background"], "layer/scenery overlap")
            expected[p] = color
        _require(len(source) == int(np.sum(expected == color)), "layer area changed")
        records.append({"color": color, "count": count, "jam": first,
                        "done": done, "area": len(source),
                        "asymmetric": len({frozenset(x) for x in orientations}) == 4,
                        "direct_clear": not bool(orientations[count % 4] & walls)})
    return expected, records


def _orbits(n):
    remaining = {(r, c) for r in range(n) for c in range(n)}
    remaining.remove((n//2, n//2))
    result = []
    while remaining:
        p = min(remaining)
        orbit = [_rot(p, n, j) for j in range(4)]
        remaining.difference_update(orbit)
        result.append(orbit)
    return result


def _shapes(rng, n, colors, jams, witness, required_asym=()):
    orbits = _orbits(n)
    center = (n-1)/2
    style = int(rng.integers(3))
    anchor = rng.uniform(0, center, 2)
    anchors = [(anchor[0], anchor[1]), (anchor[1], n-1-anchor[0]),
               (n-1-anchor[0], n-1-anchor[1]), (n-1-anchor[1], anchor[0])]
    if style == 0:
        scores = [min(min(r,c,n-1-r,n-1-c) for r,c in orbit) + rng.random()*.8
                  for orbit in orbits]
    elif style == 1:
        scores = [min((r-y)**2+(c-x)**2 for y,x in anchors) + rng.random()*2
                  for (r,c), *_ in orbits]
    else:
        scores = rng.random(len(orbits))
    order = [int(i) for i in np.argsort(scores)]
    spare = int(rng.integers(1, min(5, len(orbits)-3*len(colors))+1))
    allocations = np.full(len(colors), 3, dtype=int)
    allocations += rng.multinomial(len(orbits)-spare-int(allocations.sum()),
                                  rng.dirichlet(np.full(len(colors), 2.0)))
    source = np.full((n,n), -1, dtype=np.int8)
    teeth = set()
    cursor = 0
    for i in rng.permutation(len(colors)):
        i = int(i)
        ids = order[cursor:cursor+int(allocations[i])]
        cursor += int(allocations[i])
        sentinel = orbits[ids[0]]
        phase = int(rng.integers(4))
        symmetry = "asymmetric" if witness or jams[i] or i in required_asym else str(rng.choice(
            ["asymmetric", "half", "quarter"], p=[.6,.2,.2]))
        chosen = set()
        motif = int(rng.integers(4))
        ar, ac = rng.uniform(0,n,2)
        density = float(rng.uniform(.25,.85))
        for oid in ids[1:]:
            orbit = orbits[oid]
            if symmetry == "quarter":
                groups = [orbit]
            elif symmetry == "half":
                groups = [orbit[::2],orbit[1::2]]
            else:
                groups = [[p] for p in orbit]
            for group in groups:
                r,c = group[0]
                score = (rng.random() if motif == 0 else
                         ((r+c)%2)*.6 + rng.random()*.4 if motif == 1 else
                         (r%3)/3 + rng.random()*.3 if motif == 2 else
                         min(1.0, ((r-ar)**2+(c-ac)**2)/(n*n*.5)))
                if score < density:
                    chosen.update(group)
        chosen.add(sentinel[phase])
        if symmetry == "half":
            chosen.add(sentinel[(phase+2)%4])
        elif symmetry == "quarter":
            chosen.update(sentinel)
        # At least two orbits give visible rigid movement, even for sparse fills.
        if not any(p in chosen for p in orbits[ids[1]]):
            orbit = orbits[ids[1]]
            chosen.add(orbit[phase])
            if symmetry == "half":
                chosen.add(orbit[(phase+2)%4])
            elif symmetry == "quarter":
                chosen.update(orbit)
        for p in chosen:
            source[p] = colors[i]
        j = jams[i]
        if j:
            previous = set().union(*({ _rot(p,n,t) for p in chosen} for t in range(j)))
            eligible = {_rot(p,n,j) for p in chosen} - previous
            # Keep the commanded final angle clear in the skip-obstruction witness.
            if witness == 1 and i == 2:
                eligible -= {_rot(p,n,3) for p in chosen}
            stop = _rot(sentinel[phase], n, j)
            _require(stop in eligible, "stop sentinel not exclusive")
            teeth.add(stop)
            near = sorted(p for p in eligible if max(abs(p[0]-stop[0]),abs(p[1]-stop[1])) <= 1)
            for p in near[:int(rng.integers(1,4))]:
                teeth.add(p)
    # These small teeth lie outside every rotor's swept territory.
    for oid in order[cursor:cursor+int(rng.integers(spare+1))]:
        orbit = orbits[oid]
        teeth.add(orbit[int(rng.integers(4))])
    return source, teeth


def _bar_options(shape, blocked, length, orientation):
    h,w = shape
    bh,bw = (1,length) if orientation == 0 else (length,1)
    if bh > h or bw > w:
        return np.empty((0,2), dtype=int)
    rr,cc = np.indices((h-bh+1,w-bw+1))
    valid = np.ones(rr.shape,dtype=bool)
    for t in range(length):
        valid &= ~blocked[rr+(t if orientation else 0),cc+(0 if orientation else t)]
    return np.argwhere(valid)


def _pack(rng, board, bg, commands, forced):
    s = len(board)
    for attempt in range(16):
        if attempt == 15:
            h = w = 30
            r = c = 0
        else:
            h,w = map(int,rng.integers(max(22,s+6),31,size=2))
            r,c = int(rng.integers(h-s+1)), int(rng.integers(w-s+1))
        inp = np.full((h,w),bg,dtype=np.int8)
        inp[r:r+s,c:c+s] = board
        blocked = np.zeros(inp.shape,dtype=bool)
        blocked[max(0,r-1):min(h,r+s+1),max(0,c-1):min(w,c+s+1)] = True
        success = True
        for color,length in sorted(commands.items(),key=lambda item:-item[1]):
            if not length:
                continue
            options = []
            for orientation in (0,1):
                if color in forced and forced[color] != orientation:
                    continue
                pts = _bar_options(inp.shape,blocked,length,orientation)
                if len(pts):
                    options.append((orientation,pts))
            if not options:
                success = False
                break
            orientation,pts = options[int(rng.integers(len(options)))]
            y,x = map(int,pts[int(rng.integers(len(pts)))])
            for t in range(length):
                yy,xx = y+(t if orientation else 0),x+(0 if orientation else t)
                inp[yy,xx] = color
                blocked[max(0,yy-1):min(h,yy+2),max(0,xx-1):min(w,xx+2)] = True
        if success:
            return inp
    # Provably sufficient strip: s<=19 leaves >=10 clear columns at right.
    # At most five horizontal bars, each length<=9, use separate even rows.
    inp = np.full((30,30),bg,dtype=np.int8)
    inp[:s,:s] = board
    for row,(color,length) in zip(range(0,10,2),[(x,k) for x,k in commands.items() if k]):
        inp[row,s+1:s+1+length] = color
    return inp


def _make_pair(rng, witness=0, testing=False):
    if witness == 1:
        counts,jams = [1,2,3,5,2],[0,0,2,0,1]
    elif witness == 2:
        counts,jams = [3,4,5,0],[0,0,3,0]
    else:
        layer_count = int(rng.integers(2,6))
        cap = 9 if testing else 5
        counts = [int(x) for x in rng.integers(0,cap+1,size=layer_count)]
        jams = [int(rng.choice([0,0,0,1,2,3])) for _ in counts]
        if testing:
            counts[0],jams[0] = int(rng.integers(6,10)),0
            counts[1],jams[1] = int(rng.integers(3,10)),int(rng.integers(1,4))
        elif not any(counts):
            counts[0] = 1
    palette = [int(x) for x in rng.choice(10,len(counts)+3,replace=False)]
    bg,wall,hub = palette[:3]
    colors = palette[3:]
    n = int(rng.choice([9,11,13,15,17]))
    source,teeth = _shapes(rng,n,colors,jams,witness,{0} if testing else ())
    board = np.full((n+2,n+2),wall,dtype=np.int8)
    board[1:-1,1:-1] = bg
    for color in colors:
        board[1:-1,1:-1][source == color] = color
    for r,c in teeth:
        board[r+1,c+1] = wall
    board[n//2+1,n//2+1] = hub
    commands = dict(zip(colors,counts))
    forced = {colors[1]:0,colors[4]:1} if witness == 1 else {}
    inp = _pack(rng,board,bg,commands,forced)
    info = _decode(inp)
    return {"input":inp,"output":_render(info)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test = _counts(num_train,num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train":[],"test":[]}
    seen = set()
    witnesses = [1,2]+[0]*(num_train-2)
    witnesses = [witnesses[int(i)] for i in rng.permutation(num_train)]
    for split,flags in (("train",witnesses),("test",[0]*num_test)):
        for witness in flags:
            for _ in range(8):
                pair = _make_pair(rng,witness,split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen = set()
    training = []
    alternatives = {mode:False for mode in ("no_walls","no_turns","direct_final","modulo_first")}
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"}, "pair keys")
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            key = _key(inp)
            _require(key not in seen,"duplicate input")
            seen.add(key)
            info = _decode(inp)
            expected,records = _coordinate_check(info)
            _require(np.array_equal(out,expected), "incorrect rigid rotation, stop, scenery, or crop")
            _require(np.array_equal(out,solve(inp)), "input-only solver disagreement")
            if split == "train":
                training.extend(records)
                _require(max(info["commands"].values()) <= 5,"training count cap")
                for mode in alternatives:
                    alternatives[mode] |= not np.array_equal(out,_render(info,mode))
            else:
                _require(any(r["count"] >= 6 and not r["jam"] and r["asymmetric"]
                             for r in records),"test lacks informative long free command")
                _require(any(r["jam"] and r["count"] >= r["jam"] for r in records),
                         "test lacks commanded jam")
    for count in (1,2,3,4,5):
        _require(any(r["count"] == count and not r["jam"] and r["asymmetric"]
                     for r in training), "missing free asymmetric count witness")
    for jam in (1,2,3):
        _require(any(r["jam"] == jam and r["count"] >= jam and r["area"] >= 2
                     for r in training), "missing rigid jam witness")
    _require(any(r["count"] == 0 and r["asymmetric"] for r in training), "unmarked default witness")
    _require(any(r["count"] == 3 and r["jam"] == 2 and r["direct_clear"]
                 for r in training), "no intermediate-obstruction witness")
    _require(all(alternatives.values()),"training permits a named parent-omitting shortcut")
    return True


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                a = np.asarray(pair[role],dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds; bounded memory, no files, no retained trace payloads."""
    # Hand-specified coordinates: one tooth blocks only one cell of the next
    # three-cell mask. The whole layer must stop after exactly one turn.
    fixture = np.zeros((14,20),dtype=np.int8)
    board = np.zeros((11,11),dtype=np.int8)
    board[0] = board[-1] = 8
    board[:,0] = board[:,-1] = 8
    board[5,5],board[8,8] = 9,8
    for p in ((2,2),(2,3),(3,2)):
        board[p] = 2
    for p in ((4,4),(4,5)):
        board[p] = 3
    fixture[:11,:11] = board
    fixture[2,13:16],fixture[7,13:18] = 2,3
    expected = board.copy()
    expected[(board == 2) | (board == 3)] = 0
    for p in ((2,8),(3,8),(2,7)):
        expected[p] = 2
    for p in ((4,6),(5,6)):
        expected[p] = 3
    _require(np.array_equal(solve(fixture),expected),"hand-specified rigid stop/cycle failed")
    # Moving and reorienting the same bars cannot change the answer.
    fixture[:,11:] = 0
    fixture[:3,14],fixture[6:11,18] = 2,3
    _require(np.array_equal(solve(fixture),expected),"bar location/orientation changed command")
    digests,shapes,structures,sizes,palettes,areas,symmetries = (set() for _ in range(7))
    lengths,jams,positions = set(),set(),set()
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = _digest(puzzle)
        _require(digest == _digest(generate_puzzle(seed)),"seed nondeterminism")
        digests.add(digest)
        for split in ("train","test"):
            for pair in puzzle[split]:
                pairs += 1
                info = _decode(pair["input"])
                _,records = _coordinate_check(info)
                board = info["board"]
                shapes.add(pair["input"].shape)
                sizes.add(len(board))
                positions.add(info["position"])
                palettes.add((info["background"],info["wall"],info["hub"]))
                canonical = np.full(board.shape,0,dtype=np.int8)
                canonical[board == info["wall"]] = 1
                canonical[board == info["hub"]] = 2
                for code,color in enumerate(info["colors"],3):
                    mask = board == color
                    canonical[mask] = code
                    symmetries.add("quarter" if np.array_equal(mask,np.rot90(mask)) else
                                   "half" if np.array_equal(mask,np.rot90(mask,2)) else "asymmetric")
                structures.add(hashlib.sha256(canonical.tobytes()).hexdigest())
                lengths.update(r["count"] for r in records)
                jams.update(r["jam"] for r in records)
                areas.update(r["area"] for r in records)
    _require(len(digests) == 42 and len(structures) >= 180,"insufficient structural variation")
    _require(len(shapes) >= 35 and len(positions) >= 30 and len(sizes) == 5,"geometry variation")
    _require(len(palettes) >= 80 and lengths == set(range(10)),"palette/count variation")
    _require(jams == {0,1,2,3} and max(areas) >= 25,"footprint/stop variation")
    _require(symmetries == {"asymmetric","half","quarter"},"symmetry variation")
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            generate_puzzle(319,a,b)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),
                (4.0,1),(4,1.0),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Negative mutations use fresh bounded puzzles rather than copies of all seeds.
    bad = generate_puzzle(71,2,1)
    out = bad["test"][0]["output"]
    out[0,0] = (int(out[0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    bad = generate_puzzle(72,2,1)
    pair = bad["train"][0]
    info = _decode(pair["input"])
    color = next(iter(info["bars"]))
    r,c = info["bars"][color][0]
    pair["input"][r,c] = info["hub"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt counter evidence accepted")
    return {"seeds":42,"pairs":pairs,"distinct_puzzles":len(digests),
            "housing_structures":len(structures),"canvas_shapes":len(shapes),
            "housing_sizes":sorted(sizes),"positions":len(positions),
            "role_palettes":len(palettes),"counter_counts":sorted(lengths),
            "first_blocked_attempts":sorted(jams),"area_range":[min(areas),max(areas)],
            "symmetries":sorted(symmetries)}
