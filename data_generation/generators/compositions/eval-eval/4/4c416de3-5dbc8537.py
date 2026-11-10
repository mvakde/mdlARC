"""Assemble one finite mosaic kit, then complete its reflected corner stamps.

All semantic inference starts in _parse_input; no generation metadata is used by
solve. The three structural colors are local roles, not numeric constants.
"""

import hashlib
import numpy as np


_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_HOST_SEEDS = (0, 209596390, 989891189, 52545910, 590330260, 901748091,
               309208096, 974306479, 455381864, 423606629, 2922882,
               753209667, 334619110, 491844163, 21286930, 190615500,
               306526694, 991661925, 498649146, 660010066, 986050613,
               464284140, 169478674, 206525648, 620025265, 686157668,
               153509924, 202744805, 587889532, 429929698, 979736113,
               927651068, 444784596, 838212239, 989557205, 669870173,
               951690860, 540799859, 840059613, 655352581, 867986006,
               35654276)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, comp = [start], [start]
        while todo:
            y, x = todo.pop()
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
                    comp.append(q)
        result.append(comp)
    return result


def _corners(rect):
    t, l, b, r = rect
    return ((t, l, 0), (t, r, 1), (b, l, 2), (b, r, 3))


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r+1)}
            | {(b, x) for x in range(l, r+1)}
            | {(y, l) for y in range(t, b+1)}
            | {(y, r) for y in range(t, b+1)})


def _parse_input(value):
    """Recover frames, loose inventory, master mask and aligned echo clues."""
    inp = _grid(value)
    bg = int(inp[0, 0])
    if not all(np.all(edge == bg) for edge in
               (inp[0], inp[-1], inp[:, 0], inp[:, -1])):
        raise ValueError("The outer border must be background")
    candidates = []
    for color in map(int, np.unique(inp)):
        if color == bg:
            continue
        rectangles = []
        for comp in _components(inp == color):
            ys, xs = zip(*comp)
            rect = (min(ys), min(xs), max(ys), max(xs))
            t, l, b, r = rect
            if (b-t < 6 or r-l < 6 or t == 0 or l == 0
                    or b == inp.shape[0]-1 or r == inp.shape[1]-1
                    or set(comp) != _outline(rect)):
                break
            rectangles.append(rect)
        else:
            if rectangles:
                candidates.append((color, rectangles))
    if len(candidates) != 1:
        raise ValueError("There must be one color of intact hollow frames")
    frame, rectangles = candidates[0]
    inside = np.zeros(inp.shape, dtype=bool)
    observations = {}
    for rect in rectangles:
        t, l, b, r = rect
        if np.any(inside[t:b+1, l:r+1]):
            raise ValueError("Frames must be disjoint, without nesting")
        inside[t:b+1, l:r+1] = True
        cs = _corners(rect)
        for y, x in np.argwhere(inp[t+1:b, l+1:r] != bg):
            y, x = int(y+t+1), int(x+l+1)
            distances = [(y-a)**2+(x-z)**2 for a, z, _ in cs]
            if distances.count(min(distances)) != 1:
                raise ValueError("A clue is equidistant from frame corners")
            corner = cs[distances.index(min(distances))]
            a, z, q = corner
            dy, dx = _DIRS[q]
            u, v = (y-a)*dy, (x-z)*dx
            if not (1 <= u <= 5 and 1 <= v <= 5):
                raise ValueError("An interior mark is outside its corner neighborhood")
            observations.setdefault(corner, {})[(u-1, v-1)] = int(inp[y, x])
    outside = (inp != bg) & ~inside
    tiles = []
    for comp in _components(outside):
        ys, xs = zip(*comp)
        tile = inp[min(ys):max(ys)+1, min(xs):max(xs)+1]
        if tile.size != len(comp) or np.any(tile == bg) or max(tile.shape) > 5:
            raise ValueError("Loose components must be small filled rectangular tiles")
        tiles.append(tile.copy())
    if not tiles or not observations:
        raise ValueError("Missing source kit or corner observations")
    payload = {int(c) for tile in tiles for c in tile.flat}
    interior_colors = {c for obs in observations.values() for c in obs.values()}
    mold_colors = interior_colors - payload
    if len(mold_colors) != 1 or bg in payload or frame in payload:
        raise ValueError("One mold color must occur only inside the frames")
    mold = mold_colors.pop()
    refs = [corner for corner, obs in observations.items() if mold in obs.values()]
    if len(refs) != 1:
        raise ValueError("Exactly one corner exposes the full mold silhouette")
    reference = refs[0]
    ref = observations[reference]
    h = max(y for y, _ in ref)+1
    w = max(x for _, x in ref)+1
    mask = np.zeros((h, w), dtype=bool)
    for p in ref:
        mask[p] = True
    if len(_components(mask)) != 1:
        raise ValueError("The prototype silhouette must be edge connected")
    clues = {}
    for corner, obs in observations.items():
        for p, color in obs.items():
            if p[0] >= h or p[1] >= w or not mask[p]:
                raise ValueError("An echo has a cell outside the master silhouette")
            if color == mold:
                if corner != reference:
                    raise ValueError("Mold outside its reference corner")
            elif p in clues and clues[p] != color:
                raise ValueError("Aligned corner colors disagree")
            else:
                clues[p] = color
    if sum(tile.size for tile in tiles) != int(mask.sum()):
        raise ValueError("The finite kit must have exactly the prototype area")
    for t, l, b, r in rectangles:
        if b-t+1 < 2*h+3 or r-l+1 < 2*w+3:
            raise ValueError("Opposing corner neighborhoods are too close")
    if len(observations) < 2:
        raise ValueError("A mold and at least one indicated echo are required")
    return dict(input=inp, background=bg, frame=frame, mold=mold,
                rectangles=rectangles, observations=observations,
                reference=reference, tiles=tiles, mask=mask, clues=clues)


def _pack(mask, tiles, clues, unlimited=False, max_nodes=20000):
    """Enumerate at most two *rendered* exact covers, keeping finite counts.

    Equal matrices are grouped. Dead states alone are memoized: successful
    states can have different already-painted pixels and cannot be merged.
    Budget exhaustion is an error, never evidence of uniqueness.
    """
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    index = {p: i for i, p in enumerate(coords)}
    groups, lookup = [], {}
    for tile in tiles:
        key = (tile.shape, tile.tobytes())
        if key in lookup:
            groups[lookup[key]][1] += 1
        else:
            lookup[key] = len(groups)
            groups.append([tile, 1])
    remaining = [len(coords)//t.size if unlimited else n for t, n in groups]
    placements, by_cell = [], [[] for _ in coords]
    h, w = mask.shape
    for gi, (tile, _) in enumerate(groups):
        th, tw = tile.shape
        for y in range(h-th+1):
            for x in range(w-tw+1):
                if not np.all(mask[y:y+th, x:x+tw]):
                    continue
                if any(y <= a < y+th and x <= b < x+tw
                       and int(tile[a-y, b-x]) != c for (a, b), c in clues.items()):
                    continue
                assignments = tuple((index[(y+dy, x+dx)], int(tile[dy, dx]))
                                    for dy in range(th) for dx in range(tw))
                bits = sum(1 << i for i, _ in assignments)
                pi = len(placements)
                placements.append((gi, bits, assignments, y, x))
                for i, _ in assignments:
                    by_cell[i].append(pi)
    full = (1 << len(coords))-1
    painted = [-1]*len(coords)
    outcomes, dead, path = {}, set(), []
    nodes = 0

    def search(covered):
        nonlocal nodes
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise ValueError("Exact-cover search exceeded its bounded node budget")
        if covered == full:
            key = tuple(painted)
            if key not in outcomes:
                array = np.full(mask.shape, -1, dtype=np.int8)
                array[mask] = painted
                outcomes[key] = dict(prototype=array, path=tuple(path))
            return True
        state = (covered, tuple(remaining))
        if state in dead:
            return False
        best = None
        for i in range(len(coords)):
            if covered & (1 << i):
                continue
            options = [pi for pi in by_cell[i]
                       if remaining[placements[pi][0]] and not covered & placements[pi][1]]
            if not options:
                dead.add(state)
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        found = False
        for pi in best:
            gi, bits, assignments, y, x = placements[pi]
            remaining[gi] -= 1
            for i, color in assignments:
                painted[i] = color
            path.append((gi, y, x))
            found = search(covered | bits) or found
            path.pop()
            for i, _ in assignments:
                painted[i] = -1
            remaining[gi] += 1
            if len(outcomes) >= 2:
                return True
        if not found:
            dead.add(state)
        return found

    search(0)
    return list(outcomes.values()), nodes


def _paint(info, prototype, quarter_turn=False):
    out = info["input"].copy()
    # Erase just the observations before applying a competing convention too.
    for (y, x, q), obs in info["observations"].items():
        dy, dx = _DIRS[q]
        for u, v in obs:
            out[y+dy*(u+1), x+dx*(v+1)] = info["background"]
    for y, x, q in info["observations"]:
        dy, dx = _DIRS[q]
        for u, v in np.argwhere(info["mask"]):
            a, b = (int(v)+1, int(u)+1) if quarter_turn and q in (1, 2) else (int(u)+1, int(v)+1)
            yy, xx = y+dy*a, x+dx*b
            if not (0 <= yy < out.shape[0] and 0 <= xx < out.shape[1]):
                raise ValueError("A transformed stamp leaves the canvas")
            out[yy, xx] = prototype[u, v]
    return out


def _infer(value):
    info = _parse_input(value)
    outcomes, nodes = _pack(info["mask"], info["tiles"], info["clues"], max_nodes=None)
    if len(outcomes) != 1:
        raise ValueError("The aligned clues and finite kit must give one rendered mosaic")
    info.update(outcomes[0])
    info["nodes"] = nodes
    return _paint(info, info["prototype"]), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Assemble and stamp using this input alone; no seed or examples are read."""
    return _infer(input_grid)[0]


def _grow_kit(rng, count, span):
    """Grow connected rectangles, including a mandatory equal-size pair."""
    for _ in range(90):
        h, w = ((1, 3), (2, 2), (2, 1), (1, 2), (3, 1))[int(rng.integers(5))]
        rects = [(0, 0, h, w)]
        for i in range(1, count):
            choices = set()
            for _ in range(100):
                th, tw = (h, w) if i == 1 else (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
                ry, rx, rh, rw = rects[int(rng.integers(len(rects)))]
                side = int(rng.integers(4))
                if side < 2:
                    y = ry-th if side == 0 else ry+rh
                    x = rx+int(rng.integers(1-tw, rw))
                else:
                    x = rx-tw if side == 2 else rx+rw
                    y = ry+int(rng.integers(1-th, rh))
                candidate = (y, x, th, tw)
                if any(not (y+th <= a or a+ah <= y or x+tw <= b or b+bw <= x)
                       for a, b, ah, bw in rects):
                    continue
                all_rects = rects+[candidate]
                hh = max(a+ah for a, b, ah, bw in all_rects)-min(a for a, b, ah, bw in all_rects)
                ww = max(b+bw for a, b, ah, bw in all_rects)-min(b for a, b, ah, bw in all_rects)
                if hh <= span and ww <= span:
                    choices.add(candidate)
            if not choices:
                break
            choices = sorted(choices)
            rects.append(choices[int(rng.integers(len(choices)))])
        if len(rects) != count:
            continue
        y0, x0 = min(y for y, x, h, w in rects), min(x for y, x, h, w in rects)
        rects = [(y-y0, x-x0, h, w) for y, x, h, w in rects]
        hh, ww = max(y+h for y, x, h, w in rects), max(x+w for y, x, h, w in rects)
        if min(hh, ww) < 2 or sum(h*w for y, x, h, w in rects) < 6:
            continue
        return rects, (hh, ww)
    raise ValueError("Could not grow a compact connected prototype")


def _tile_pattern(rng, shape, base, ink, force_asym=False):
    tile = np.full(shape, base, dtype=np.int8)
    if ink is None or tile.size == 1:
        return tile
    yy, xx = np.indices(shape)
    mode = int(rng.integers(6))
    if force_asym or mode == 1:
        tile[-1, -1] = ink
    elif mode == 2:
        tile[yy % 2 == 1] = ink
    elif mode == 3:
        tile[xx % 2 == 1] = ink
    elif mode == 4:
        tile[(yy+xx) % 2 == 1] = ink
    elif mode == 5:
        tile[rng.random(shape) < .32] = ink
    tile[0, 0] = base
    return tile


def _prototype(rng, count, span):
    rects, shape = _grow_kit(rng, count, span)
    colors = list(map(int, rng.permutation(10)))
    bg, frame, mold = colors[:3]
    bases = colors[3:3+count]
    ink = colors[-1] if rng.random() < .8 else None
    tiles = [_tile_pattern(rng, (h, w), bases[i], ink, i == 0)
             for i, (y, x, h, w) in enumerate(rects)]
    mask = np.zeros(shape, dtype=bool)
    proto = np.full(shape, -1, dtype=np.int8)
    for (y, x, h, w), tile in zip(rects, tiles):
        mask[y:y+h, x:x+w] = True
        proto[y:y+h, x:x+w] = tile
    if shape[0] == shape[1] and np.array_equal(proto, proto.T):
        raise ValueError("Need an observable reflection-versus-rotation witness")
    # Hide one entire member of the equal-dimension pair. Finite inventory,
    # rather than another echo, must determine all of its pixels.
    hidden = int(rng.integers(2))
    y, x, h, w = rects[hidden]
    hidden_cells = {(a, b) for a in range(y, y+h) for b in range(x, x+w)}
    clues = {tuple(map(int, p)): int(proto[tuple(p)]) for p in np.argwhere(mask)
             if tuple(p) not in hidden_cells}
    outcomes, _ = _pack(mask, tiles, clues)
    expected = tuple(map(int, proto[mask]))
    if len(outcomes) != 1 or tuple(map(int, outcomes[0]["prototype"][mask])) != expected:
        raise ValueError("Construction's dense observations did not pin the rendering")
    # Remove clues only after checking the *whole* finite packing, not a tile
    # lookup. This exposes hidden seams and variable amounts of clue loss.
    order = sorted(clues)
    rng.shuffle(order)
    for p in order:
        color = clues.pop(p)
        outcomes, _ = _pack(mask, tiles, clues)
        if len(outcomes) != 1:
            clues[p] = color
    if not clues:
        raise ValueError("Equal-size alternatives should require a corner clue")
    # Restore a variable number of redundant clues, still never exposing the
    # hidden tile. Some examples are sparse and some nearly complete echoes.
    probability = float(rng.uniform(0, .42))
    for p in order:
        if rng.random() < probability:
            clues[p] = int(proto[p])
    return bg, frame, mold, tiles, mask, proto, clues


def _separate(a, b):
    return a[2]+2 <= b[0] or b[2]+2 <= a[0] or a[3]+2 <= b[1] or b[3]+2 <= a[1]


def _scene(rng, data, frame_count, ref_q, witness):
    bg, frame, mold, tiles, mask, proto, clues = data
    ph, pw = mask.shape
    for _ in range(50):
        sizes = [(2*ph+3+int(rng.integers(3)), 2*pw+3+int(rng.integers(3)))
                 for _ in range(frame_count)]
        if frame_count == 1:
            mh, mw = max(14, sizes[0][0]+4), max(14, sizes[0][1]+4)
        elif frame_count == 2:
            if rng.random() < .5:
                mh, mw = sum(h for h, w in sizes)+3, max(w for h, w in sizes)+5
            else:
                mh, mw = max(h for h, w in sizes)+5, sum(w for h, w in sizes)+3
        else:
            mh, mw = 2*max(h for h, w in sizes)+3, 2*max(w for h, w in sizes)+3
        if max(mh, mw) > 30:
            continue
        ch, cw = int(rng.integers(mh, 31)), int(rng.integers(mw, 31))
        frames = []
        for h, w in sizes:
            legal = [(y, x, y+h-1, x+w-1)
                     for y in range(1, ch-h) for x in range(1, cw-w)
                     if all(_separate((y, x, y+h-1, x+w-1), f) for f in frames)]
            if not legal:
                break
            frames.append(legal[int(rng.integers(len(legal)))])
        if len(frames) != frame_count:
            continue
        occupied = np.zeros((ch, cw), dtype=bool)
        for t, l, b, r in frames:
            occupied[t:b+1, l:r+1] = True
        positions = {}
        order = list(map(int, rng.permutation(len(tiles))))
        order.sort(key=lambda i: -tiles[i].size)
        for i in order:
            h, w = tiles[i].shape
            legal = [(y, x) for y in range(1, ch-h) for x in range(1, cw-w)
                     if not np.any(occupied[y-1:y+h+1, x-1:x+w+1])]
            if not legal:
                break
            y, x = legal[int(rng.integers(len(legal)))]
            positions[i] = (y, x)
            occupied[y:y+h, x:x+w] = True
        if len(positions) == len(tiles):
            break
    else:
        raise ValueError("Scene packing exhausted its bounded attempts")
    inp = np.full((ch, cw), bg, dtype=np.int8)
    for rect in frames:
        for p in _outline(rect):
            inp[p] = frame
    for i, (y, x) in positions.items():
        h, w = tiles[i].shape
        inp[y:y+h, x:x+w] = tiles[i]
    ref_id = int(rng.integers(frame_count))
    reference = _corners(frames[ref_id])[ref_q]
    available = [c for f in frames for c in _corners(f) if c != reference]
    rng.shuffle(available)
    count = int(rng.integers(3, min(7, 4*frame_count)+1))
    marked = [reference]+available[:count-1]
    if witness:
        marked = list(dict.fromkeys([reference,
                      _corners(frames[ref_id])[ref_q ^ 1],
                      _corners(frames[ref_id])[ref_q ^ 2]]+marked))
    if frame_count > 1 and all(c in _corners(frames[ref_id]) for c in marked):
        other = next(c for f in frames for c in _corners(f) if c not in _corners(frames[ref_id]))
        marked.append(other)
    echo_clues = {c: {} for c in marked[1:]}
    for p, color in sorted(clues.items()):
        c = marked[1+int(rng.integers(len(marked)-1))]
        echo_clues[c][p] = color
    # Every intended copy is visibly indicated, including offset singleton clues.
    for c, obs in echo_clues.items():
        if not obs:
            p = sorted(clues)[int(rng.integers(len(clues)))]
            obs[p] = clues[p]
        for p, color in clues.items():
            if rng.random() < .16:
                obs[p] = color
    observations = {reference: {tuple(map(int, p)): mold for p in np.argwhere(mask)}, **echo_clues}
    for (y, x, q), obs in observations.items():
        dy, dx = _DIRS[q]
        for (u, v), color in obs.items():
            inp[y+dy*(u+1), x+dx*(v+1)] = color
    # Labels are painted from the construction, then checked against a fresh
    # parse and exact cover. Parsing never receives this output or its geometry.
    expected_info = dict(input=inp, background=bg, observations=observations, mask=mask)
    out = _paint(expected_info, proto)
    predicted, _ = _infer(inp)
    if not np.array_equal(predicted, out):
        raise ValueError("Input-derived solution differs from construction")
    return {"input": inp, "output": out}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    regime = int(rng.integers(3))  # cross-frame, inventory size, master orientation
    train_ref = int(rng.choice((0, 3)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            for _ in range(45):
                try:
                    test = split == "test"
                    ntiles = int(rng.integers(4, 7) if test and regime == 1
                                 else rng.integers(2, 5))
                    if test and regime == 1:
                        ntiles = max(5, ntiles)
                    span = 5 if test and regime == 1 else int(rng.choice((4, 4, 5)))
                    data = _prototype(rng, ntiles, span)
                    if regime == 0:
                        nf = 2 if test else 1
                    else:
                        nf = int(rng.choice((1, 1, 2)))
                        if max(data[4].shape) <= 3 and rng.random() < .18:
                            nf = 3
                    rq = (int(rng.choice((1, 2))) if test else train_ref) if regime == 2 else int(rng.integers(4))
                    if split == "train" and i == 0:
                        rq = train_ref
                    pair = _scene(rng, data, nf, rq, split == "train" and i == 0)
                    key = (pair["input"].shape, pair["input"].tobytes())
                    if key in seen:
                        continue
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("Bounded family construction failed")
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check input-derived semantics and concrete parent-omission witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), []
    for pair in puzzle["train"]+puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = _grid(pair["input"]), _grid(pair["output"])
        key = (inp.shape, inp.tobytes())
        assert key not in seen
        seen.add(key)
        predicted, info = _infer(inp)
        assert np.array_equal(predicted, out), "Input-only solution mismatch"
        assert inp.shape == out.shape and np.any(inp != out)
        assert info["mold"] not in set(map(int, out.flat))
        assert all(int(inp[y, x]) == int(out[y, x])
                   for rect in info["rectangles"] for y, x in _outline(rect))
        # A omission: footprint-only inventory packing has different renderings.
        alternatives, _ = _pack(info["mask"], info["tiles"], {})
        assert len(alternatives) == 2, "Echo alignment should be essential"
        # B omission: allowing a physical piece to be reused gives different
        # answers even after aligning every available corner clue.
        unlimited, _ = _pack(info["mask"], info["tiles"], info["clues"], unlimited=True)
        assert len(unlimited) == 2, "Finite inventory should be essential"
        assert len(info["clues"]) < int(info["mask"].sum()), "Copying clues alone must be insufficient"
        assert len(info["observations"]) >= 3
        infos.append(info)
    first = infos[0]
    assert first["reference"][2] in (0, 3)
    assert not np.array_equal(_paint(first, first["prototype"], quarter_turn=True),
                              puzzle["train"][0]["output"]), "Untaught reflection convention"
    training, testing = infos[:nt], infos[nt:]
    spatial_shift = (all(len(i["rectangles"]) == 1 for i in training)
                     and all(len(i["rectangles"]) == 2 for i in testing))
    inventory_shift = (max(len(i["tiles"]) for i in training) <= 4
                       and min(len(i["tiles"]) for i in testing) >= 5)
    tr_refs = {i["reference"][2] for i in training}
    te_refs = {i["reference"][2] for i in testing}
    orientation_shift = len(tr_refs) == 1 and tr_refs <= {0, 3} and te_refs <= {1, 2}
    assert spatial_shift or inventory_shift or orientation_shift, "Missing train/test shift"
    return infos


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, in-memory checks on the 42 supplied seeds; no files or traces."""
    shapes, masks, frame_counts, kit_counts = set(), set(), set(), set()
    references, palettes, sizes, signatures = set(), set(), set(), set()
    nonrectangular = patterned = distributed = unmarked = 0
    max_nodes = 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        infos = validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for info in infos:
            shapes.add(info["input"].shape)
            masks.add((info["mask"].shape, info["mask"].tobytes()))
            frame_counts.add(len(info["rectangles"]))
            kit_counts.add(len(info["tiles"]))
            references.add(info["reference"][2])
            palettes.add((info["background"], info["frame"], info["mold"]))
            sizes.add(int(info["mask"].sum()))
            nonrectangular += int(not np.all(info["mask"]))
            patterned += int(any(len(np.unique(t)) > 1 for t in info["tiles"]))
            distributed += int(all(len(obs) < len(info["clues"])
                                   for c, obs in info["observations"].items()
                                   if c != info["reference"]))
            unmarked += int(len(info["observations"]) < 4*len(info["rectangles"]))
            max_nodes = max(max_nodes, info["nodes"])
    assert len(signatures) == 42 and len(shapes) >= 40 and len(masks) >= 80
    assert frame_counts >= {1, 2} and kit_counts >= {2, 3, 4, 5, 6}
    assert references == {0, 1, 2, 3} and len(palettes) >= 70 and len(sizes) >= 10
    assert nonrectangular >= 70 and patterned >= 100 and distributed >= 10 and unmarked >= 100
    assert _signature(generate_puzzle(17)) == _signature(generate_puzzle(17))
    for nt, nq in ((2, 1), (10, 4), (10, 1)):
        p = generate_puzzle(128+nt+nq, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
        validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Regression: finite multiplicity, no rotations, and rendered ambiguity.
    mask = np.ones((1, 4), dtype=bool)
    a, b = np.array([[2, 3]], dtype=np.int8), np.array([[4, 5]], dtype=np.int8)
    assert len(_pack(mask, [a, b], {})[0]) == 2
    assert len(_pack(mask, [a, b], {(0, 0): 2})[0]) == 1
    assert len(_pack(mask, [a, b], {(0, 0): 2}, unlimited=True)[0]) == 2
    assert not _pack(mask, [a], {})[0]
    assert len(_pack(mask, [a, a], {})[0]) == 1
    assert not _pack(np.ones((4, 1), dtype=bool), [a, b], {})[0]
    return dict(seeds=42, pairs=210, canvas_shapes=len(shapes), silhouettes=len(masks),
                frame_counts=sorted(frame_counts), kit_counts=sorted(kit_counts),
                silhouette_areas=sorted(sizes), role_palettes=len(palettes),
                nonrectangular=nonrectangular, patterned=patterned,
                distributed_clues=distributed, unmarked_corners=unmarked,
                maximum_constrained_search_nodes=max_nodes)
