"""Uncrating and docking: reconstructed rooms are the rigid moving bodies.

The input's two edge swatches specify the frame-to-destination map and axis.
Walls and noise vanish; each elementary room's complete interior travels,
including its background-colored margins. No parent module is imported.
"""

from collections import deque
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _array(grid):
    g = np.asarray(grid)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("invalid ARC grid")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.tobytes()


def _canonical(grid):
    """The blank line behind the terminal swatches identifies orientation."""
    g = _array(grid)
    candidates = []
    for transpose in (False, True):
        a = g.T if transpose else g
        if min(a.shape) < 5 or a.shape[1] < 7:
            continue
        bg, low, high = map(int, (a[1, 0], a[0, 0], a[0, -1]))
        if (len({bg, low, high}) == 3 and np.all(a[1] == bg)
                and np.all(a[0, :3] == low) and np.all(a[0, -3:] == high)
                and np.all(a[0, 3:-3] == bg)):
            candidates.append((a, bg, low, high, transpose))
    if len(candidates) != 1:
        raise ValueError("need exactly one visible docking axis")
    return candidates[0]


def _bounded_regions(walls):
    """Four-connected negative space, with an explicit outside padding ring."""
    free = np.pad(~walls, 1, constant_values=True)
    seen = np.zeros(free.shape, dtype=bool)
    height, width = free.shape
    rooms = []
    for y, x in np.argwhere(free):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        queue = deque([(y, x)])
        seen[y, x] = True
        cells, outside = [], False
        while queue:
            r, c = queue.popleft()
            cells.append((r - 1, c - 1))
            outside |= r in (0, height - 1) or c in (0, width - 1)
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= rr < height and 0 <= cc < width and free[rr, cc]
                        and not seen[rr, cc]):
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        if not outside:
            p = np.asarray(cells)
            t, l = map(int, p.min(axis=0))
            b, r = map(int, p.max(axis=0))
            if len(cells) != (b-t+1)*(r-l+1):
                raise ValueError("nonrectangular elementary room")
            rooms.append((t, l, b, r))
    return sorted(rooms)


def _repair(g, low, high, noise):
    repaired = g.copy()
    # Only single-cell straight interruptions with two visible supports.
    for color in (low, high):
        horizontal = ((g[:, :-2] == color) & (g[:, 1:-1] == noise)
                      & (g[:, 2:] == color))
        vertical = ((g[:-2] == color) & (g[1:-1] == noise) & (g[2:] == color))
        repaired[:, 1:-1][horizontal] = color
        repaired[1:-1][vertical] = color
    return repaired


def _decode(grid):
    a, bg, low, high, transpose = _canonical(grid)
    board = a[2:]
    solutions = []
    for noise in sorted(set(map(int, np.unique(board))) - {bg, low, high}):
        repaired = _repair(board, low, high, noise)
        walls = (repaired == low) | (repaired == high)
        try:
            boxes = _bounded_regions(walls)
            if not boxes:
                continue
            expected_walls = np.zeros(board.shape, dtype=bool)
            occupied = np.zeros(board.shape, dtype=bool)
            objects = []
            for t, l, b, r in boxes:
                if t == 0 or l == 0 or b+1 == board.shape[0] or r+1 == board.shape[1]:
                    raise ValueError("unbounded room")
                border = np.concatenate((repaired[t-1, l-1:r+2],
                                         repaired[b+1, l-1:r+2],
                                         repaired[t-1:b+2, l-1],
                                         repaired[t-1:b+2, r+1]))
                frame = int(border[0])
                if frame not in (low, high) or not np.all(border == frame):
                    raise ValueError("room has inconsistent boundary color")
                patch = board[t:b+1, l:r+1].copy()
                if (np.any(np.isin(patch, (low, high, noise)))
                        or not np.any(patch != bg)):
                    raise ValueError("room must contain observable cargo")
                expected_walls[t-1, l-1:r+2] = True
                expected_walls[b+1, l-1:r+2] = True
                expected_walls[t-1:b+2, l-1] = True
                expected_walls[t-1:b+2, r+1] = True
                occupied[t:b+1, l:r+1] = True
                objects.append(dict(r=t+2, c=l, h=b-t+1, w=r-l+1,
                                    d=0 if frame == low else 1, patch=patch))
            if not np.array_equal(walls, expected_walls):
                continue
            # True noise has at least one exterior witness. Payload cannot.
            exterior = ~occupied & ~walls
            if (not np.any(exterior & (board == noise))
                    or np.any(exterior & ~np.isin(board, (bg, noise)))):
                continue
            solutions.append((objects, noise, repaired, walls))
        except ValueError:
            continue
    if len(solutions) != 1:
        raise ValueError("noise and room interpretation must be unique")
    objects, noise, repaired, walls = solutions[0]
    return a, bg, low, high, transpose, objects, noise, repaired, walls


def _band(a, b):
    return max(a["r"], b["r"]) < min(a["r"]+a["h"], b["r"]+b["h"])


def _overlap(a, b):
    return _band(a, b) and max(a["c"], b["c"]) < min(a["c"]+a["w"], b["c"]+b["w"])


def _pack(objects, width, *, jumping=False):
    result = [dict(o) for o in objects]
    for direction in (0, 1):
        indices = [i for i, o in enumerate(objects) if o["d"] == direction]
        indices.sort(key=lambda i: (objects[i]["c"] if direction == 0
                                    else -objects[i]["c"]-objects[i]["w"],
                                    objects[i]["r"], objects[i]["h"]))
        placed = []
        for i in indices:
            o = result[i]
            blockers = [p for p in placed if _band(o, p)]
            if jumping:
                positions = range(width-o["w"]+1)
                if direction:
                    positions = reversed(positions)
                for c in positions:
                    o["c"] = c
                    if not any(_overlap(o, p) for p in placed):
                        break
                else:
                    raise ValueError("no jumping placement")
            elif direction == 0:
                o["c"] = max((p["c"]+p["w"] for p in blockers), default=0)
            else:
                o["c"] = min((p["c"] for p in blockers), default=width)-o["w"]
            if not 0 <= o["c"] <= width-o["w"]:
                raise ValueError("chain exceeds canvas")
            placed.append(o)
    if any(_overlap(a, b) for i, a in enumerate(result) for b in result[i+1:]):
        raise ValueError("packed classes collide")
    return result


def _render(a, bg, objects):
    result = np.full(a.shape, bg, dtype=np.int8)
    result[0] = a[0]
    occupied = np.zeros(a.shape, dtype=bool)
    for o in objects:
        r, c, h, w = (o[k] for k in ("r", "c", "h", "w"))
        if np.any(occupied[r:r+h, c:c+w]):
            raise ValueError("full cargo footprints collide")
        occupied[r:r+h, c:c+w] = True
        result[r:r+h, c:c+w] = o["patch"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, bg, _, _, transpose, objects, _, _, _ = _decode(input_grid)
    output = _render(a, bg, _pack(objects, a.shape[1]))
    return output.T.copy() if transpose else output


def _partition(rng, h, w, count, style):
    rooms = [(0, 0, h-1, w-1)]
    first_axis = None
    for step in range(count-1):
        choices = []
        for i, (t, l, b, r) in enumerate(rooms):
            for axis in (0, 1):
                if style == "h" and axis != 0 or style == "v" and axis != 1:
                    continue
                if style == "partial" and step == 1 and axis == first_axis:
                    continue
                start, end = (t, b) if axis == 0 else (l, r)
                for cut in range(start+2, end-1):
                    halves = ((t,l,cut,r),(cut,l,b,r)) if axis == 0 else ((t,l,b,cut),(t,cut,b,r))
                    capacity = sum(((q[2]-q[0])//2)*((q[3]-q[1])//2)
                                   for j, q in enumerate(rooms) if j != i)
                    capacity += sum(((q[2]-q[0])//2)*((q[3]-q[1])//2) for q in halves)
                    if capacity >= count:
                        choices.append((i, axis, halves))
        if not choices:
            raise ValueError("partition does not fit")
        i, axis, halves = choices[int(rng.integers(len(choices)))]
        if first_axis is None:
            first_axis = axis
        rooms[i:i+1] = halves
    return rooms


def _partial(rooms):
    ys = sorted({y for t,l,b,r in rooms for y in (t,b)})
    xs = sorted({x for t,l,b,r in rooms for x in (l,r)})
    return len(rooms) != (len(ys)-1)*(len(xs)-1)


def _layout(rng, purpose):
    if purpose in ("chain", "combined"):
        number = int(rng.integers(3, 5))
        ih = int(rng.integers(3, 6))
        step = int(rng.integers((ih+1)//2, ih))
        widths = list(map(int, rng.integers(3 if purpose == "combined" else 2, 5, size=number)))
        minimal_w = sum(w+2 for w in widths) + number-1
        width = int(rng.integers(minimal_w, 31))
        minimal_h = 2 + (number-1)*step + ih+2
        height = int(rng.integers(minimal_h, 31))
        gaps = rng.multinomial(width-minimal_w, np.full(number+1, 1/(number+1)))
        row = int(rng.integers(2, height-((number-1)*step+ih+2)+1))
        direction, reverse_rows = int(rng.integers(2)), bool(rng.integers(2))
        c, networks = int(gaps[0]), []
        partial_index = int(rng.integers(number))
        for i, iw in enumerate(widths):
            h, w = ih+2, iw+2
            count = 3 if purpose == "combined" and i == partial_index else (int(rng.integers(1, 4)) if purpose == "combined" else 1)
            rooms = _partition(rng, h, w, count, "partial" if count >= 3 else "mixed")
            r = row + (number-1-i if reverse_rows else i)*step
            col = c if direction == 0 else width-c-w
            networks.append(dict(r=r, c=col, h=h, w=w, d=direction, rooms=rooms))
            c += w+1+int(gaps[i+1])
        return (height, width), networks
    witness = purpose == "topology"
    height, width = int(rng.integers(15 if witness else 10, 31)), int(rng.integers(20 if witness else 10, 31))
    number = 2 if witness else int(rng.integers(1, 6))
    networks = []
    first_direction = int(rng.integers(2))
    for i in range(number):
        accepted = None
        for _ in range(70):
            h, w = map(int, rng.integers(7 if witness and i == 0 else 3, 12, size=2))
            if h > height-2 or w > width:
                continue
            cap = ((h-1)//2)*((w-1)//2)
            count = int(rng.integers(3, 5)) if witness and i == 0 else int(rng.integers(1, min(6,cap)+1))
            style = "partial" if witness and i == 0 else str(rng.choice(["h","v","mixed","mixed","partial"]))
            try:
                rooms = _partition(rng, h, w, count, style)
            except ValueError:
                continue
            if witness and i == 0 and not _partial(rooms):
                continue
            r, c = int(rng.integers(2,height-h+1)), int(rng.integers(width-w+1))
            d = first_direction if witness and i == 0 else 1-first_direction if witness else int(rng.integers(2))
            candidate = dict(r=r, c=c, h=h, w=w, d=d, rooms=rooms)
            # A blank-cell separation avoids accidental shared inter-crate walls.
            if any(max(r-1,n["r"]) < min(r+h+1,n["r"]+n["h"])
                   and max(c-1,n["c"]) < min(c+w+1,n["c"]+n["w"]) for n in networks):
                continue
            accepted = candidate
            break
        if accepted is None:
            if witness:
                raise ValueError("teaching layout does not fit")
            break
        networks.append(accepted)
    if not networks:
        raise ValueError("empty layout")
    return (height, width), networks


def _cargo(rng, shape, bg, palette, mode, shared):
    h, w = shape
    patch = np.full(shape, bg, dtype=np.int8)
    if mode == "solid":
        patch[:] = int(rng.choice(palette))
    elif mode == "pattern":
        colors = rng.choice(palette, size=int(rng.integers(2, min(4,len(palette))+1)), replace=False)
        rr, cc = np.indices(shape)
        patch[:] = colors[(rr+2*cc+int(rng.integers(4))) % len(colors)]
        if rng.random() < .5:
            patch[rng.random(shape) < .4] = bg
    else:
        amount = int(rng.integers(1, max(2,patch.size//3+1)))
        positions = rng.choice(patch.size, size=amount, replace=False)
        patch.flat[positions] = rng.choice(palette, size=amount)
    patch[h//2,w//2] = shared
    return patch


def _jump_witness(objects, width):
    try:
        stable, jumping = _pack(objects,width), _pack(objects,width,jumping=True)
    except ValueError:
        return False
    return any(a["c"] != b["c"] for a,b in zip(stable,jumping))


def _tight_objects(objects, bg):
    result = []
    for o in objects:
        p = np.argwhere(o["patch"] != bg)
        t,l = map(int,p.min(axis=0))
        b,r = map(int,p.max(axis=0))
        result.append(dict(o,r=o["r"]+t,c=o["c"]+l,h=b-t+1,w=r-l+1,
                           patch=o["patch"][t:b+1,l:r+1]))
    return result


def _make_pair(rng, parameters, purpose):
    for attempt in range(180):
        if parameters["local_roles"]:
            bg, low, high, noise = map(int,rng.choice(10,4,replace=False))
        else:
            bg, low, high, noise = parameters["roles"]
        palette = [x for x in range(10) if x not in (bg,low,high,noise)]
        shared = int(rng.choice(palette))
        try:
            shape, networks = _layout(rng,purpose)
            a = np.full(shape,bg,dtype=np.int8)
            a[0,:3], a[0,-3:] = low,high
            walls = np.zeros(shape,dtype=bool)
            objects = []
            for n in networks:
                color = (low,high)[n["d"]]
                for t,l,b,r in n["rooms"]:
                    t,l,b,r = t+n["r"],l+n["c"],b+n["r"],r+n["c"]
                    walls[t,l:r+1] = walls[b,l:r+1] = True
                    walls[t:b+1,l] = walls[t:b+1,r] = True
                    a[t,l:r+1] = a[b,l:r+1] = color
                    a[t:b+1,l] = a[t:b+1,r] = color
                    mode = "sparse" if purpose == "topology" else str(rng.choice(["sparse","sparse","solid","pattern"]))
                    patch = _cargo(rng,(b-t-1,r-l-1),bg,palette,mode,shared)
                    a[t+1:b,l+1:r] = patch
                    objects.append(dict(r=t+1,c=l+1,h=b-t-1,w=r-l-1,d=n["d"],patch=patch))
            objects.sort(key=lambda o:(o["r"],o["c"]))
            settled = _pack(objects,shape[1])
            if purpose in ("chain","combined") and not _jump_witness(objects,shape[1]):
                continue
            expected = _render(a,bg,settled)
            if purpose == "topology":
                # Sparse-pixel packing must actually disagree in the teaching scene.
                tight = _render(a,bg,_pack(_tight_objects(objects,bg),shape[1]))
                if np.array_equal(tight,expected):
                    continue
                if not all(any(o["d"] == d and o["c"] != p["c"] for o,p in zip(objects,settled)) for d in (0,1)):
                    continue
            safe = []
            for r,c in np.argwhere(walls):
                color = int(a[r,c])
                if (0 < r < shape[0]-1 and 0 < c < shape[1]-1):
                    horizontal = a[r,c-1] == color == a[r,c+1]
                    vertical = a[r-1,c] == color == a[r+1,c]
                    neighbors = sum(a[rr,cc] == color for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)))
                    if neighbors == 2 and (horizontal or vertical):
                        safe.append((int(r),int(c),color))
            rng.shuffle(safe)
            damaged = set()
            desired = max(1,int(len(safe)*float(rng.uniform(.03,.16)))) if purpose == "topology" or rng.random() < .8 else 0
            for r,c,color in safe:
                if len(damaged) >= desired:
                    break
                if any(abs(r-y)+abs(c-x) <= 2 for y,x in damaged):
                    continue
                a[r,c] = noise
                damaged.add((r,c))
            if purpose == "topology" and not damaged:
                continue
            exterior = a == bg
            exterior[:2] = False
            for o in objects:
                exterior[o["r"]:o["r"]+o["h"],o["c"]:o["c"]+o["w"]] = False
            sites = np.argwhere(exterior)
            if not len(sites):
                continue
            for j in rng.choice(len(sites),size=min(len(sites),int(rng.integers(1,9))),replace=False):
                r,c = sites[j]
                a[r,c] = noise
            if purpose in ("chain","combined") and np.array_equal(
                    _render(a,bg,_pack(objects,shape[1],jumping=True)),expected):
                continue
            if parameters["transpose"]:
                a,expected = a.T.copy(),expected.T.copy()
            decoded = _decode(a)
            if not np.array_equal(decoded[8],walls[2:]):
                continue
            if not np.array_equal(solve(a),expected):
                continue
            return {"input":a,"output":expected}
        except ValueError:
            continue
    raise RuntimeError(f"bounded construction failed for {purpose}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train,num_test)
    seed = _integer(seed,"seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    parameters = dict(roles=tuple(map(int,rng.choice(10,4,replace=False))),
                      local_roles=bool(rng.integers(2)),transpose=bool(rng.integers(2)))
    puzzle = {"train":[],"test":[]}
    seen = set()
    for split,count in (("train",nt),("test",ne)):
        for i in range(count):
            purpose = ("topology" if split == "train" and i == 0 else
                       "chain" if split == "train" and i == 1 else
                       "combined" if split == "test" and i == 0 else "free")
            for _ in range(12):
                pair = _make_pair(rng,parameters,purpose)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _simulate(objects, width):
    """A separate unit-step check cannot jump same-class blocking rectangles."""
    result = [dict(o) for o in objects]
    for direction,delta in ((0,-1),(1,1)):
        group = [o for o in result if o["d"] == direction]
        for _ in range(width*max(1,len(group))+1):
            moved = False
            for o in group:
                nxt = dict(o,c=o["c"]+delta)
                if (0 <= nxt["c"] <= width-o["w"]
                        and not any(p is not o and _overlap(nxt,p) for p in group)):
                    o["c"] = nxt["c"]
                    moved = True
            if not moved:
                break
        else:
            raise AssertionError("unit-step compaction did not converge")
    assert not any(_overlap(a,b) for i,a in enumerate(result) for b in result[i+1:])
    return result


def _features(grid):
    a,bg,low,high,tr,objects,noise,repaired,walls = _decode(grid)
    # Connected wall components give source networks; room bodies give topology.
    components = []
    remaining = walls.copy()
    for y,x in np.argwhere(walls):
        if not remaining[y,x]:
            continue
        stack,points = [(int(y),int(x))],[]
        remaining[y,x] = False
        while stack:
            r,c = stack.pop()
            points.append((r,c))
            for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0 <= rr < walls.shape[0] and 0 <= cc < walls.shape[1] and remaining[rr,cc]:
                    remaining[rr,cc] = False
                    stack.append((rr,cc))
        p = np.asarray(points)
        t,l = map(int,p.min(axis=0)); b,r = map(int,p.max(axis=0))
        contained = [o for o in objects if t < o["r"]-2 <= b and l < o["c"] <= r]
        boxes = [(o["r"]-3,o["c"]-1,o["r"]+o["h"]-2,o["c"]+o["w"]) for o in contained]
        components.append((t,l,b,r,contained,_partial(boxes)))
    return dict(a=a,bg=bg,low=low,high=high,transpose=tr,objects=objects,noise=noise,
                walls=walls,repaired=repaired,components=components,
                partial=any(c[-1] for c in components),
                damaged=bool(np.any(walls & (a[2:] == noise))),
                witness=_jump_witness(objects,a.shape[1]))


def _without_topology(f):
    """A concrete ablation: treat each complete crate as a single cargo body."""
    objects = []
    for t,l,b,r,rooms,_ in f["components"]:
        patch = f["a"][t+3:b+2,l+1:r].copy()
        patch[np.isin(patch,(f["low"],f["high"],f["noise"]))] = f["bg"]
        objects.append(dict(r=t+3,c=l+1,h=b-t-1,w=r-l-1,
                            d=rooms[0]["d"],patch=patch))
    try:
        return _render(f["a"],f["bg"],_pack(objects,f["a"].shape[1]))
    except ValueError:
        return None


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen,axes = set(),set()
    evidence = dict(partial=False,damage=False,jump=False,margin=False,A=False,B=False)
    moved,shared_colors = set(),[set(),set()]
    test_combination = False
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            g,out = _array(pair["input"]),_array(pair["output"])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            f = _features(g)
            axes.add(f["transpose"])
            objects = f["objects"]
            assert all(not _overlap(a,b) for i,a in enumerate(objects) for b in objects[i+1:])
            settled = _simulate(objects,f["a"].shape[1])
            expected = _render(f["a"],f["bg"],settled)
            actual = out.T if f["transpose"] else out
            assert np.array_equal(actual,expected), "incorrect rigid room compaction"
            assert np.array_equal(actual[:2],f["a"][:2]), "docks/blank separator changed"
            assert not np.any(np.isin(actual[2:],(f["low"],f["high"],f["noise"])))
            if split == "train":
                evidence["partial"] |= f["partial"]
                evidence["damage"] |= f["damaged"]
                if f["witness"]:
                    jumping = _render(f["a"],f["bg"],_pack(objects,f["a"].shape[1],jumping=True))
                    evidence["jump"] |= not np.array_equal(jumping,expected)
                omission = _without_topology(f)
                evidence["A"] |= omission is None or not np.array_equal(omission,expected)
                # Keeping extracted interiors at their source positions omits B.
                evidence["B"] |= not np.array_equal(_render(f["a"],f["bg"],objects),expected)
                try:
                    tight = _render(f["a"],f["bg"],_pack(_tight_objects(objects,f["bg"]),f["a"].shape[1]))
                    evidence["margin"] |= not np.array_equal(tight,expected)
                except ValueError:
                    evidence["margin"] = True
                for o,p in zip(objects,settled):
                    if o["c"] != p["c"]:
                        moved.add(o["d"])
                    shared_colors[o["d"]].update(map(int,np.unique(o["patch"][o["patch"] != f["bg"]])))
            else:
                test_combination |= f["partial"] and f["witness"]
    assert len(axes) == 1, "axis must stay consistent in a puzzle"
    assert all(evidence.values()), f"missing demonstration evidence: {evidence}"
    assert moved == {0,1} and shared_colors[0] & shared_colors[1]
    assert test_combination, "need a held-out partition/overlap-chain recombination"


def self_test():
    corpus = [generate_puzzle(seed) for seed in range(42)]
    fingerprints,shapes,palettes,topologies,footprints,counts = set(),set(),set(),set(),set(),set()
    axes,damages,solids,sparse,patterns = set(),set(),False,False,False
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p[k]) for s in ("train","test") for p in puzzle[s] for k in ("input","output")))
        for split in ("train","test"):
            for pair in puzzle[split]:
                f = _features(pair["input"])
                axes.add(f["transpose"]); damages.add(f["damaged"])
                shapes.add(pair["input"].shape)
                palettes.add((f["bg"],f["low"],f["high"],f["noise"]))
                counts.add(len(f["objects"]))
                for *_,rooms,partial in f["components"]:
                    ys = sorted({v for o in rooms for v in (o["r"]-1,o["r"]+o["h"])})
                    xs = sorted({v for o in rooms for v in (o["c"]-1,o["c"]+o["w"])})
                    topologies.add(tuple(sorted((ys.index(o["r"]-1),xs.index(o["c"]-1),ys.index(o["r"]+o["h"]),xs.index(o["c"]+o["w"])) for o in rooms)))
                for o in f["objects"]:
                    footprints.add((o["h"],o["w"]))
                    nonbg = o["patch"] != f["bg"]
                    solids |= bool(nonbg.all())
                    sparse |= bool(not nonbg.all())
                    patterns |= len(np.unique(o["patch"][nonbg])) > 1
    assert len(fingerprints) == 42 and axes == {False,True}
    assert len(shapes) >= 50 and len(palettes) >= 25
    assert len(topologies) >= 20 and len(footprints) >= 25 and len(counts) >= 8
    assert damages == {False,True} and solids and sparse and patterns
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(314159,nt,ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(2.0,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    p = generate_puzzle(0)
    assert tuple(_key(v) for s in p for pair in p[s] for v in pair.values()) == tuple(_key(v) for s in corpus[0] for pair in corpus[0][s] for v in pair.values())
    # Input-only equivariance under recoloring and axis transpose.
    permutation = np.array([9,7,5,3,1,8,6,4,2,0],dtype=np.int8)
    for pair in p["train"] + p["test"]:
        assert np.array_equal(solve(permutation[pair["input"]]),permutation[pair["output"]])
        assert np.array_equal(solve(pair["input"].T),pair["output"].T)
    # A geometry-preserving wrong label from B's rejected jumping convention.
    chain_pair = next(pair for pair in p["train"] if _features(pair["input"])["witness"])
    f = _features(chain_pair["input"])
    jumping = _render(f["a"],f["bg"],_pack(f["objects"],f["a"].shape[1],jumping=True))
    changed = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    index = next(i for i,pair in enumerate(p["train"]) if pair is chain_pair)
    changed["train"][index]["output"] = jumping.T.copy() if f["transpose"] else jumping
    try:
        validate_generated_puzzle(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError("same-class jumping accepted")
    # Deliberately wrong packing labels must fail the validator.
    broken = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt held-out label accepted")
    return dict(seeds=42,distinct_puzzles=len(fingerprints),input_shapes=len(shapes),
                role_palettes=len(palettes),room_topologies=len(topologies),
                room_footprints=len(footprints),parcel_counts=sorted(counts))
