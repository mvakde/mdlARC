"""Complete an enlarged stencil by exact cover, then recover its latent grid.

Only standard-library modules and NumPy are used. No parent code is imported.
Search state is transient: no execution traces, payload caches, or logs are saved.
"""
from collections import Counter
import hashlib
import numpy as np


def _key(a):
    a = np.asarray(a)
    return (a.shape, a.astype(np.int8).tobytes())


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _starts(a, axis):
    lines = a if axis == 0 else a.T
    return np.r_[0, 1 + np.flatnonzero(np.any(lines[1:] != lines[:-1], axis=1))]


def _collapse(a):
    # Both sets of representatives are computed on the complete physical grid.
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def _runs(a, axis):
    return tuple(int(v) for v in np.diff(np.r_[_starts(a, axis), a.shape[axis]]))


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _tiles(tray, vacancy):
    result = []
    for cells in _components(tray != vacancy):
        yy, xx = zip(*cells)
        tile = tray[min(yy):max(yy)+1, min(xx):max(xx)+1]
        if tile.size != len(cells) or np.any(tile == vacancy):
            raise ValueError("Tray objects must be filled, separate rectangles")
        result.append(tile.copy())
    if not result:
        raise ValueError("Empty tray")
    return result


def _cover(mask, tiles, budget=50000):
    """Enumerate up to two rendered answers; interchangeable copies are grouped.

    No compression criterion filters coverings: A must finish uniquely before B.
    Exhausting the node budget raises an error, never certifies uniqueness.
    """
    coords = np.argwhere(mask)
    n = len(coords)
    if n == 0:
        return []
    index = np.full(mask.shape, -1, dtype=int)
    index[mask] = np.arange(n)
    groups, lookup, counts = [], {}, []
    for t in tiles:
        k = _key(t)
        if k not in lookup:
            lookup[k] = len(groups)
            groups.append(t)
            counts.append(0)
        counts[lookup[k]] += 1
    initial = tuple(counts)
    by_cell = [[] for _ in range(n)]
    placements = []
    for g, tile in enumerate(groups):
        h, w = tile.shape
        for y in range(mask.shape[0] - h + 1):
            for x in range(mask.shape[1] - w + 1):
                if not np.all(mask[y:y+h, x:x+w]):
                    continue
                cells = tuple(int(c) for c in index[y:y+h, x:x+w].flat)
                bits = sum(1 << c for c in cells)
                pi = len(placements)
                placements.append((g, bits, cells, tuple(map(int, tile.flat)), y, x))
                for c in cells:
                    by_cell[c].append(pi)
    full = (1 << n) - 1
    pixels = np.full(n, -1, dtype=np.int8)
    outcomes, path, dead = {}, [], set()
    nodes = 0

    def visit(covered):
        nonlocal nodes
        nodes += 1
        if nodes > budget:
            raise ValueError("Exact-cover search budget exceeded")
        if covered == full:
            image = pixels.tobytes()
            if image not in outcomes:
                outcomes[image] = {"pixels": pixels.copy(), "path": tuple(path),
                                   "used": tuple(a-b for a,b in zip(initial, counts))}
            return
        state = (covered, tuple(counts))
        if state in dead:
            return
        best = None
        for c in range(n):
            if (covered >> c) & 1:
                continue
            options = [p for p in by_cell[c]
                       if counts[placements[p][0]] and not (covered & placements[p][1])]
            if not options:
                dead.add(state)
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        old = len(outcomes)
        for pi in best:
            g, bits, cells, values, y, x = placements[pi]
            counts[g] -= 1
            pixels[list(cells)] = values
            path.append((g, y, x))
            visit(covered | bits)
            path.pop()
            pixels[list(cells)] = -1
            counts[g] += 1
            if len(outcomes) >= 2:
                return
        # Memoize only impossible suffixes, not successful ones whose prefixes
        # can have different colors but the same occupied cells/inventory.
        if len(outcomes) == old == 0:
            dead.add(state)

    visit(0)
    result = list(outcomes.values())
    for r in result:
        r.update(groups=groups, counts=initial, nodes=nodes)
    return result


def _panels(grid):
    """A unique-color full-span single line visibly separates the two panels."""
    for axis in (0, 1):
        length = grid.shape[1-axis]
        for k in range(1, grid.shape[axis]-1):
            line = grid[k, :] if axis == 0 else grid[:, k]
            marker = int(line[0])
            if not np.all(line == marker) or np.count_nonzero(grid == marker) != length:
                continue
            a, b = ((grid[:k], grid[k+1:]) if axis == 0
                    else (grid[:, :k], grid[:, k+1:]))
            for stencil, tray, side in ((a, b, "top" if axis == 0 else "left"),
                                        (b, a, "bottom" if axis == 0 else "right")):
                colors = np.unique(stencil)
                if len(colors) != 2:
                    continue
                for board, vacancy in (colors, colors[::-1]):
                    if np.any(tray == board) or not np.any(tray == vacancy):
                        continue
                    try:
                        tiles = _tiles(tray, vacancy)
                    except ValueError:
                        continue
                    if any(np.any(t == marker) for t in tiles):
                        continue
                    yield {"stencil": stencil.copy(), "tray": tray.copy(),
                           "board": int(board), "vacancy": int(vacancy),
                           "marker": marker, "side": side, "tiles": tiles}


def _analyze(input_grid):
    grid = _grid(input_grid)
    answers = []
    for a in _panels(grid):
        mask = a["stencil"] == a["vacancy"]
        if sum(t.size for t in a["tiles"]) < int(mask.sum()):
            continue
        covers = _cover(mask, a["tiles"])
        if not covers:
            continue
        if len(covers) != 1:
            raise ValueError("Ambiguous completed stencil")
        record = covers[0]
        completed = a["stencil"].copy()
        completed[mask] = record["pixels"]
        a.update(mask=mask, completed=completed, output=_collapse(completed), record=record)
        answers.append(a)
    if len(answers) != 1:
        raise ValueError("Expected one visible panel interpretation and completion")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only visible markers, finite tray inventory, and full-line equality."""
    return _analyze(input_grid)["output"].copy()


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _schedule(rng, n, limit, nonuniform=False):
    # Every latent line has >=2 copies: all redundant runs are removed.
    for _ in range(40):
        mode = int(rng.integers(4))
        if mode == 0 and not nonuniform:
            v = [int(rng.integers(2, min(4, limit//n)+1))] * n
        else:
            v = list(map(int, rng.integers(2, 5, size=n)))
            if mode == 1:
                v.sort()
            elif mode == 2:
                v.sort(reverse=True)
        if sum(v) <= limit and (not nonuniform or len(set(v)) > 1):
            return tuple(v)
    v = [2] * n
    if limit > 2*n:
        v[int(rng.integers(n))] = 3
    return tuple(v)


def _latent(rng, h, w, board, palette, witness):
    for _ in range(70):
        mask = rng.random((h, w)) < rng.uniform(.48, .82)
        latent = rng.choice(palette, (h, w)).astype(np.int8)
        latent[~mask] = board
        if witness == "row":
            # Two occupied rows with the same silhouette but different content;
            # a separated full-row recurrence must survive compression.
            latent[1] = rng.choice(palette, w)
            latent[1, latent[0] == board] = board
            latent[-1] = latent[0]
        elif witness == "column":
            latent[:, 1] = rng.choice(palette, h)
            latent[latent[:, 0] == board, 1] = board
            latent[:, -1] = latent[:, 0]
        if (np.count_nonzero(latent == board) < 2
                or np.count_nonzero(latent != board) < 4
                or len(np.unique(latent[latent != board])) < 2):
            continue
        if _collapse(latent).shape != latent.shape:
            continue
        return latent
    return None


def _partition(mask, rng):
    """Tile a mask geometrically, ignoring its future colors and logical bands."""
    remaining = mask.copy()
    rects = []
    while np.any(remaining):
        y, x = map(int, np.argwhere(remaining)[0])
        choices = []
        width = remaining.shape[1] - x
        for h in range(1, remaining.shape[0]-y+1):
            row = remaining[y+h-1, x:x+width]
            bad = np.flatnonzero(~row)
            if len(bad):
                width = int(bad[0])
            if width == 0:
                break
            choices.append((h, width))
        # Maximal rectangles in two independent directions, rather than a fixed L.
        choices = [(h,w) for h,w in choices
                   if h == choices[-1][0] or choices[h][1] < w]
        weights = np.array([(h*w)**2 for h,w in choices], dtype=float)
        h,w = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
        rects.append((y, x, h, w))
        remaining[y:y+h, x:x+w] = False
        if len(rects) > 7:
            return None
    if len(rects) < 2:
        return None
    if len(rects) < 7 and rng.random() < .65:
        eligible = [i for i,r in enumerate(rects) if max(r[2:]) >= 4]
        if eligible:
            i = int(rng.choice(eligible))
            y,x,h,w = rects.pop(i)
            axis = int(rng.integers(2)) if min(h,w) >= 4 else (0 if h >= 4 else 1)
            if axis == 0:
                k = int(rng.integers(1,h))
                rects.extend([(y,x,k,w), (y+k,x,h-k,w)])
            else:
                k = int(rng.integers(1,w))
                rects.extend([(y,x,h,k), (y,x+k,h,w-k)])
    return rects


def _scatter(tiles, height, width_limit, vacancy, rng):
    minimum = max(t.shape[1] for t in tiles)
    if minimum > width_limit or max(t.shape[0] for t in tiles) > height:
        return None
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    for attempt in range(12):
        width = width_limit if attempt < 4 else int(rng.integers(minimum, width_limit+1))
        tray = np.full((height, width), vacancy, dtype=np.int8)
        occupied = np.zeros_like(tray, dtype=bool)
        for i in order:
            h,w = tiles[i].shape
            legal = [(y,x) for y in range(height-h+1) for x in range(width-w+1)
                     if not np.any(occupied[max(0,y-1):min(height,y+h+1),
                                             max(0,x-1):min(width,x+w+1)])]
            if not legal:
                break
            y,x = legal[int(rng.integers(len(legal)))]
            tray[y:y+h, x:x+w] = tiles[i]
            occupied[y:y+h, x:x+w] = True
        else:
            # Keep a sampled portion of unused edge space, preserving the full
            # shared height; tray coordinates carry no target ordering.
            last = int(np.flatnonzero(np.any(occupied, axis=0))[-1])+1
            width = int(rng.integers(last, width+1))
            return tray[:, :width].copy()
    return None


def _assemble(stencil, tray, marker, side):
    divider = np.full((stencil.shape[0],1), marker, dtype=np.int8)
    panels = (stencil, divider, tray) if side in ("left", "top") else (tray, divider, stencil)
    inp = np.concatenate(panels, axis=1)
    return inp.T.copy() if side in ("top", "bottom") else inp.copy()


def _add_surplus(mask, tiles, expected, rng, palette):
    dimensions = [(h,w) for h in range(1,5) for w in range(1,5)
                  if (h,w) not in {t.shape for t in tiles}]
    rng.shuffle(dimensions)
    for h,w in dimensions:
        tile = rng.choice(palette, (h,w)).astype(np.int8)
        try:
            answers = _cover(mask, tiles+[tile], budget=2500)
        except ValueError:
            continue
        if len(answers) == 1 and np.array_equal(answers[0]["pixels"], expected):
            return tiles+[tile]
    return None


def _build(rng, side, h, w, witness, surplus):
    board,vacancy,marker = map(int, rng.choice(10,3,replace=False))
    choices = [c for c in range(10) if c not in (board,vacancy,marker)]
    palette = rng.choice(choices, int(rng.integers(2, min(7,h*w)+1)), replace=False)
    for _ in range(100):
        latent = _latent(rng,h,w,board,palette,witness)
        if latent is None:
            continue
        rows = _schedule(rng,h,22,nonuniform=True)
        columns = _schedule(rng,w,14,nonuniform=True)
        completed = np.repeat(np.repeat(latent,rows,axis=0),columns,axis=1)
        mask = completed != board
        rects = _partition(mask,rng)
        if rects is None:
            continue
        tiles = [completed[y:y+rh,x:x+rw].copy() for y,x,rh,rw in rects]
        # Same-size distinct patterns are geometrically exchangeable everywhere.
        by_shape = {}
        ambiguous = False
        for t in tiles:
            if t.shape in by_shape and _key(t) != by_shape[t.shape]:
                ambiguous = True
            by_shape[t.shape] = _key(t)
        if ambiguous:
            continue
        try:
            answers = _cover(mask, tiles, budget=3500)
        except ValueError:
            continue
        if len(answers) != 1 or not np.array_equal(answers[0]["pixels"],completed[mask]):
            continue
        if surplus:
            tiles = _add_surplus(mask, tiles, completed[mask], rng, palette)
            if tiles is None:
                continue
        tray = _scatter(tiles,completed.shape[0],29-completed.shape[1],vacancy,rng)
        if tray is None:
            continue
        stencil = completed.copy()
        stencil[mask] = vacancy
        inp = _assemble(stencil,tray,marker,side)
        out = latent.T.copy() if side in ("top","bottom") else latent.copy()
        try:
            a = _analyze(inp)
        except ValueError:
            continue
        if np.array_equal(a["output"],out):
            if witness == "column":
                f = _features(a)
                if f["max_component_tiles"] < 2 or not f["seam_in_run"]:
                    continue
            return {"input":inp,"output":out}
    return _fallback(rng,side,h,board,vacancy,marker,palette,surplus)


def _fallback(rng,side,h,board,vacancy,marker,palette,surplus):
    """Bounded guaranteed geometry, still following the exact same visible rule."""
    p,q = map(int,palette[:2])
    latent = np.full((h,4),board,dtype=np.int8)
    for i in range(h-1):
        latent[i,1:3] = (p,q) if i%2 == 0 else (q,p)
    latent[-1,[0,3]] = p
    rows = [2]*h
    rows[1] = 3
    columns = [2,3,2,2]
    completed = np.repeat(np.repeat(latent,rows,axis=0),columns,axis=1)
    mask = completed != board
    # Split one solid corner inside its two-row run. Either corner can take the
    # two identical strips; exchanging these tilings has the same rendered image.
    tiles = [completed[:-2,2:7].copy(),np.full((2,2),p,dtype=np.int8),
             np.full((1,2),p,dtype=np.int8),np.full((1,2),p,dtype=np.int8)]
    if surplus:
        tiles.append(np.array([[q]],dtype=np.int8))
    # Deterministic placement reserves room for a full-height main tile and the
    # two isolated copies. Within-panel coordinates are not a solution key.
    tray = np.full((completed.shape[0],15),vacancy,dtype=np.int8)
    tray[:tiles[0].shape[0],:5] = tiles[0]
    tray[:2,6:8] = p
    tray[3,6:8] = p
    tray[5,6:8] = p
    if surplus:
        tray[0,10] = q
    stencil = np.where(mask,vacancy,board).astype(np.int8)
    inp = _assemble(stencil,tray,marker,side)
    out = latent.T.copy() if side in ("top","bottom") else latent.copy()
    a = _analyze(inp)
    assert np.array_equal(a["output"],out) and _features(a)["seam_in_run"]
    return {"input":inp,"output":out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train",num_train,2,10)
    num_test = _count("num_test",num_test,1,4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    sides = list(rng.permutation(["left","right","top","bottom"]))
    with_surplus = bool(rng.integers(2))
    puzzle = {"train":[],"test":[]}
    seen = set()
    for i in range(num_train+num_test):
        training = i < num_train
        h = (4 if i == 0 else int(rng.integers(3,5))) if training else int(rng.integers(5,7))
        w = 4 if i == 1 else int(rng.integers(3,5))
        witness = "row" if i == 0 else "column" if i == 1 else "free"
        # Both panel orientations are taught even with two demonstrations; tests
        # recombine familiar orientation with a larger latent extent.
        side = sides[i%2] if training else sides[int(rng.integers(2))]
        surplus = with_surplus and (i == 1 or (not training and bool(rng.integers(2))))
        for _ in range(12):
            pair = _build(rng,side,h,w,witness,surplus)
            k = _key(pair["input"])
            if k not in seen:
                seen.add(k)
                puzzle["train" if training else "test"].append(pair)
                break
        else:
            raise RuntimeError("Could not generate distinct inputs")
    return puzzle


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _features(a):
    completed, out, rec = a["completed"],a["output"],a["record"]
    owner = np.full(completed.shape,-1,dtype=int)
    for i,(g,y,x) in enumerate(rec["path"]):
        h,w = rec["groups"][g].shape
        owner[y:y+h,x:x+w] = i
    component_counts = [len({int(owner[y,x]) for y,x in cc}) for cc in _components(a["mask"])]
    repeat_row = any(np.array_equal(out[i],out[j]) for i in range(out.shape[0])
                     for j in range(i+2,out.shape[0]))
    repeat_column = any(np.array_equal(out[:,i],out[:,j]) for i in range(out.shape[1])
                        for j in range(i+2,out.shape[1]))
    used = [t for t,n in zip(rec["groups"],rec["used"]) if n]
    row_equal = np.all(completed[1:]==completed[:-1],axis=1)
    col_equal = np.all(completed[:,1:]==completed[:,:-1],axis=0)
    row_seams = ((owner[1:]>=0) & (owner[:-1]>=0) & (owner[1:]!=owner[:-1]))
    col_seams = ((owner[:,1:]>=0) & (owner[:,:-1]>=0) & (owner[:,1:]!=owner[:,:-1]))
    seam_in_run = bool(np.any(row_seams & row_equal[:,None])
                       or np.any(col_seams & col_equal[None,:]))
    return {"side":a["side"],"roles":(a["board"],a["vacancy"],a["marker"]),
            "shape":out.shape,"physical_shape":completed.shape,
            "rows":_runs(completed,0),"columns":_runs(completed,1),
            "tiles":sum(rec["used"]),"supplied":len(a["tiles"]),
            "surplus":len(a["tiles"])-sum(rec["used"]),
            "copies":max(rec["used"]),"components":len(component_counts),
            "max_component_tiles":max(component_counts),
            "seam_in_run":seam_in_run,
            "nonrectangular":any(len(cc) != (max(y for y,x in cc)-min(y for y,x in cc)+1)
                                *(max(x for y,x in cc)-min(x for y,x in cc)+1)
                                for cc in _components(a["mask"])),
            "hidden_boundary":any(not np.array_equal(_starts(a["stencil"],axis),_starts(completed,axis))
                                  for axis in (0,1)),
            "repeat_row":repeat_row,"repeat_column":repeat_column,
            "asymmetric":any(not np.array_equal(t,t[::-1,::-1]) for t in used),
            "multicolor":any(len(np.unique(t))>1 for t in used),
            "partial_join":bool(np.any(out[1:]==out[:-1]) or np.any(out[:,1:]==out[:,:-1])),
            "nodes":rec["nodes"]}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle)=={"train","test"},"Bad puzzle schema")
    _require(all(isinstance(puzzle[s],list) for s in ("train","test")),"Pairs must be lists")
    nt,ne = len(puzzle["train"]),len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne,"Invalid counts")
    seen, features = set(), []
    for pair in puzzle["train"]+puzzle["test"]:
        _require(isinstance(pair,dict) and set(pair)=={"input","output"},"Bad pair schema")
        inp,out = _grid(pair["input"]),_grid(pair["output"])
        _require(_key(inp) not in seen,"Duplicate input")
        seen.add(_key(inp))
        a = _analyze(inp)  # No label is provided to panel inference or exact cover.
        _require(np.array_equal(a["output"],out),"Incorrect answer")
        _require(_collapse(out).shape==out.shape,"Unobservable latent boundary")
        rr,cc = _runs(a["completed"],0),_runs(a["completed"],1)
        _require(min(rr+cc)>=2,"Every physical run must be redundant")
        expanded = np.repeat(np.repeat(out,rr,axis=0),cc,axis=1)
        _require(np.array_equal(expanded,a["completed"]),"Inconsistent latent enlargement")
        _require(not np.any(out==a["vacancy"]) and not np.any(out==a["marker"]),"Marker leaked")
        _require(a["board"] in out,"Board background lost")
        _require(np.all(a["completed"][~a["mask"]]==a["board"]),"Board cell overwritten")
        _require(not np.array_equal(_collapse(a["stencil"]),out),"Ablating exact cover did not matter")
        _require(a["completed"].shape!=out.shape,"Ablating compression did not matter")
        f = _features(a)
        _require(2 <= f["tiles"] <= 7,"Unexpected selected tile count")
        features.append(f)
    train,test = features[:nt],features[nt:]
    _require(any(f["hidden_boundary"] for f in train),"No fill-before-collapse witness")
    _require(any(f["max_component_tiles"]>=2 and f["seam_in_run"] for f in train),
             "No packing seam inside an identical full-line run")
    _require(any(f["repeat_row"] or f["repeat_column"] for f in train),"No separated recurrence witness")
    _require(any(len(set(f["rows"]))>1 and len(set(f["columns"]))>1 for f in train),"No nonuniform witness")
    _require(not any(f["surplus"] for f in test) or any(f["surplus"] for f in train),"Untaught surplus")
    _require(min(max(f["shape"]) for f in test)>max(max(f["shape"]) for f in train),"Missing latent-size shift")


def self_test():
    """Stream 42 seeds and small semantic adversaries, retaining only summaries."""
    signatures, features, shapes, roles, schedules = set(),[],set(),set(),set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for pair in puzzle["train"]+puzzle["test"]:
            for name in ("input","output"):
                digest.update(repr(_key(pair[name])).encode())
            a = _analyze(pair["input"])
            f = _features(a)
            features.append(f)
            shapes.add(f["shape"])
            roles.add(f["roles"])
            schedules.add((f["rows"],f["columns"]))
            # A separate vectorized expression checks the compression primitive.
            e = a["completed"]
            rr = np.r_[True,np.any(e[1:]!=e[:-1],axis=1)]
            cc = np.r_[True,np.any(e[:,1:]!=e[:,:-1],axis=0)]
            _require(np.array_equal(e[rr][:,cc],pair["output"]),"Full-line check failed")
        signatures.add(digest.hexdigest())
    _require(len(signatures)==42,"Seed diversity failed")
    _require(len(shapes)>=10 and len(roles)>=50 and len(schedules)>=70,"Limited variation")
    _require({f["side"] for f in features}=={"left","right","top","bottom"},"Missing panel side")
    for name in ("hidden_boundary","repeat_row","repeat_column","asymmetric","multicolor",
                 "partial_join","nonrectangular","surplus","seam_in_run"):
        _require(any(f[name] for f in features),f"Missing {name}")
    _require(any(f["max_component_tiles"]>=3 for f in features),"No joined packing")
    _require(any(f["copies"]>=2 for f in features),"No repeated inventory")
    _require(any(f["surplus"]==0 for f in features),"No all-used inventory")
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(1000+10*nt+ne,nt,ne)
            _require((len(p["train"]),len(p["test"]))==(nt,ne),"Counts ignored")
    p,q = generate_puzzle(91),generate_puzzle(91)
    _require(all(_key(a[k])==_key(b[k]) for s in p for a,b in zip(p[s],q[s])
                 for k in ("input","output")),"Nondeterminism")
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
    tile = np.array([[2,3]],dtype=np.int8)
    mask = np.ones((1,4),dtype=bool)
    _require(not _cover(mask,[tile]),"Inventory reused")
    _require(len(_cover(mask,[tile,tile]))==1,"Identical-copy ambiguity")
    _require(len(_cover(mask,[tile,np.array([[4,5]])]))==2,"Pattern ambiguity missed")
    solid = [np.full((1,4),2),np.full((1,2),2),np.full((1,2),2)]
    _require(len(_cover(mask,solid))==1,"Pixel-equivalent tilings rejected")
    recurrence = np.array([[2,3,2],[4,5,4],[2,3,2]])
    _require(np.array_equal(_collapse(recurrence),recurrence),"Separated recurrence removed")
    joined = np.array([[2,2,3],[2,4,3]])
    _require(np.array_equal(_collapse(joined),joined),"Local join mistaken for full-line equality")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, finite-inventory and compression adversaries)")
