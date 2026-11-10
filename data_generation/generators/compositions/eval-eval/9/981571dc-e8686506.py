"""Mirror mosaics: exact rigid-fragment assembly constrained by two folds.

Only Python's standard library and NumPy are used. Nothing from the parents is
imported. The solver parses its own query and performs a complete joint search.
"""
from collections import Counter
from functools import lru_cache
import numpy as np

DIRS=((-1,0),(1,0),(0,-1),(0,1))

def orbits(h,w,axes="both"):
    result=[]; seen=set()
    for r in range(h):
        for c in range(w):
            if (r,c) not in seen:
                o=tuple(sorted((rr,cc) for rr in ({r,h-1-r} if axes in ("both","tb") else {r}) for cc in ({c,w-1-c} if axes in ("both","lr") else {c})))
                result.append(o); seen.update(o)
    return result

class Limit(Exception): pass

def _assemble(board, bg, pieces, sym="both", limit=None):
    h,w=board.shape
    holes=[tuple(map(int,x)) for x in np.argwhere(board==bg)]
    if len(holes)!=sum(len(p) for p in pieces): return [],0
    pos={p:i for i,p in enumerate(holes)}
    obs=orbits(h,w,sym if isinstance(sym,str) else "both"); oid={p:i for i,o in enumerate(obs) for p in o}
    known=[]
    for o in obs:
        colors={int(board[p]) for p in o if board[p]!=bg}
        if sym and len(colors)>1: return [],0
        known.append(next(iter(colors)) if colors else -1)
    inv=Counter(pieces); types=sorted(inv); bycell=[[] for _ in holes]
    for t,piece in enumerate(types):
        ph=max(r for r,c,v in piece)+1; pw=max(c for r,c,v in piece)+1
        for r0 in range(h-ph+1):
            for c0 in range(w-pw+1):
                cells=[((r0+r,c0+c),v) for r,c,v in piece]
                if any(p not in pos for p,v in cells): continue
                assignments={}; good=True
                for p,v in cells:
                    o=oid[p]
                    if sym and (assignments.get(o,v)!=v or known[o] not in (-1,v)):
                        good=False; break
                    assignments[o]=v
                if not good: continue
                mask=sum(1<<pos[p] for p,v in cells)
                paint=sum((v+1)<<(4*pos[p]) for p,v in cells)
                move=(t,mask,paint,tuple(sorted(assignments.items())))
                for p,v in cells: bycell[pos[p]].append(move)
    nodes=0
    @lru_cache(None)
    def search(remaining,counts,values):
        nonlocal nodes
        nodes+=1
        if limit is not None and nodes>limit: raise Limit()
        if remaining==0: return (0,) if not any(counts) else ()
        best=None; bits=remaining
        while bits:
            b=bits&-bits; i=b.bit_length()-1
            choices=[m for m in bycell[i] if counts[m[0]] and remaining&m[1]==m[1]
                     and (not sym or all(values[o] in (-1,v) for o,v in m[3]))]
            if not choices: return ()
            if best is None or len(choices)<len(best):
                best=choices
                if len(choices)==1: break
            bits-=b
        answers=set()
        for t,mask,paint,assign in best:
            cnt=list(counts); cnt[t]-=1
            val=list(values)
            if sym:
                for o,v in assign: val[o]=v
            for tail in search(remaining^mask,tuple(cnt),tuple(val)):
                answers.add(paint|tail)
                if len(answers)>=2: return tuple(sorted(answers))
        return tuple(sorted(answers))
    codes=search((1<<len(holes))-1,tuple(inv[t] for t in types),tuple(known))
    answers=[]
    for code in codes:
        a=board.copy()
        for p,i in pos.items(): a[p]=((code>>(4*i))&15)-1
        answers.append(a)
    return answers,nodes

def _build_board(rng, h, w, bg, colors, nblind, style):
    k = len(colors); obs=orbits(h,w)
    out=np.empty((h,w),dtype=np.int8)
    for o in obs:
        r, c = o[0]
        if style == 0 or rng.random() < .20:
            v = int(rng.choice(colors))
        elif style == 1:
            v = colors[(r // 2 + c // 2) % k]
        elif style == 2:
            v = colors[max(r, c) % k]
        else:
            v = colors[(2 * r + c // 2) % k]
        for p in o: out[p]=v
    # A connected seed within a quadrant; repeat its cells at all mirror positions.
    qh,qw=h//2,w//2
    core={(int(rng.integers(1,qh-1)),int(rng.integers(1,qw-1)))}
    while len(core)<nblind:
        neighbors=sorted({(r+dr,c+dc) for r,c in core for dr,dc in DIRS
                          if 0<=r+dr<qh and 0<=c+dc<qw}-core)
        core.add(neighbors[int(rng.integers(len(neighbors)))])
    for i,(r,c) in enumerate(sorted(core)):
        for rr,cc in {(r,c),(h-1-r,c),(r,w-1-c),(h-1-r,w-1-c)}:
            out[rr,cc]=colors[i%k]
    parts=[]; used=set()
    for flipr,flipc in ((False,False),(False,True),(True,False),(True,True)):
        cells={(h-1-r if flipr else r,w-1-c if flipc else c) for r,c in core}
        # Each fragment grows independently; this creates witnessed anchors.
        for _ in range(int(rng.integers(1,5))):
            neighbors=sorted({(r+dr,c+dc) for r,c in cells for dr,dc in DIRS
                              if 0<=r+dr<h and 0<=c+dc<w and (r+dr,c+dc) not in used}-cells)
            if not neighbors: break
            # Do not intrude into other quadrants or merge cores.
            neighbors=[p for p in neighbors if (p[0]<qh if not flipr else p[0]>=h-qh)
                       and (p[1]<qw if not flipc else p[1]>=w-qw)]
            if not neighbors: break
            cells.add(neighbors[int(rng.integers(len(neighbors)))])
        parts.append(cells); used|=cells
    # Pairs of rectangular fragments removed at reflected positions. Swapping
    # one pair preserves one fold but violates the other: both folds matter.
    for axis in ("lr", "tb"):
        dims=[(1,2),(2,2),(1,3),(2,3)] if axis=="lr" else [(2,1),(2,2),(3,1),(3,2)]
        candidates=[]
        for ph,pw in dims:
            for r in range(h-ph+1):
                for c in range(w-pw+1):
                    first={(r+dr,c+dc) for dr in range(ph) for dc in range(pw)}
                    second={(rr,w-1-cc) if axis=="lr" else (h-1-rr,cc) for rr,cc in first}
                    if first&second or used&(first|second): continue
                    ar=min(rr for rr,cc in first); ac=min(cc for rr,cc in first)
                    br=min(rr for rr,cc in second); bc=min(cc for rr,cc in second)
                    if np.array_equal(out[ar:ar+ph,ac:ac+pw],out[br:br+ph,bc:bc+pw]): continue
                    candidates.append((first,second))
        if not candidates: return None
        first,second=candidates[int(rng.integers(len(candidates)))]
        parts.extend((first,second)); used|=first|second
    # Optional extra fragments may abut earlier holes.
    for _ in range(int(rng.integers(0,3))):
        free=sorted({(r,c) for r in range(h) for c in range(w)}-used)
        if not free: break
        cells={free[int(rng.integers(len(free)))]}
        for _ in range(int(rng.integers(1,5))):
            neighbors=sorted({(r+dr,c+dc) for r,c in cells for dr,dc in DIRS
                              if 0<=r+dr<h and 0<=c+dc<w}-used-cells)
            if not neighbors: break
            cells.add(neighbors[int(rng.integers(len(neighbors)))])
        parts.append(cells); used|=cells
    if len(used)>h*w*.6: return None
    board=out.copy()
    for p in used: board[p]=bg
    pieces=[]
    for part in parts:
        r0=min(r for r,c in part); c0=min(c for r,c in part)
        pieces.append(tuple(sorted((r-r0,c-c0,int(out[r,c])) for r,c in part)))
    blind=[o for o in obs if all(board[p]==bg for p in o)]
    four_blind = [o for o in blind if len(o) == 4]
    if len(four_blind) != nblind: return None
    if len({int(out[o[0]]) for o in four_blind}) != nblind: return None
    try:
        a,n=_assemble(board,bg,pieces,limit=40000)
        if len(a)!=1 or not np.array_equal(a[0],out): return None
        for ax in ("lr","tb"):
            aa,nn=_assemble(board,bg,pieces,ax,limit=40000)
            if len(aa)<2: return None
        b,m=_assemble(board,bg,pieces,False,limit=40000)
        if len(b)<2: return None
    except Limit: return None
    return board,out,pieces


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _parse(inp):
    """A uniform, uniquely colored rectangular rim visibly identifies the board."""
    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > inp.size // 2, "Background must be a strict majority")
    candidates = []
    for color in map(int, colors):
        if color == bg:
            continue
        rr, cc = np.where(inp == color)
        top, bottom = int(rr.min()), int(rr.max()) + 1
        left, right = int(cc.min()), int(cc.max()) + 1
        h, w = bottom - top, right - left
        if min(h, w) < 7 or len(rr) != 2 * h + 2 * w - 4:
            continue
        box = inp[top:bottom, left:right]
        rim = np.zeros(box.shape, dtype=bool)
        rim[[0, -1], :] = True
        rim[:, [0, -1]] = True
        if np.all(box[rim] == color) and not np.any(box[~rim] == color):
            candidates.append((color, (top, left, h, w)))
    _require(len(candidates) == 1, "The input must identify one rectangular rim")
    frame, (r, c, h, w) = candidates[0]
    board = inp[r + 1:r + h - 1, c + 1:c + w - 1].copy()
    exterior = inp != bg
    exterior[r:r + h, c:c + w] = False
    comps = _components(exterior)
    _require(bool(comps), "Missing exterior fragments")
    pieces = []
    for cells in comps:
        r0 = min(x for x, y in cells)
        c0 = min(y for x, y in cells)
        pieces.append(tuple((x - r0, y - c0, int(inp[x, y])) for x, y in cells))
    _require(sum(map(len, pieces)) == int(np.count_nonzero(board == bg)),
             "Inventory area must equal board vacancy area")
    return bg, frame, (r, c, h, w), board, pieces, comps


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the visible input alone, using no examples or random state."""
    inp = _grid(input_grid)
    bg, _, _, board, pieces, _ = _parse(inp)
    answers, _ = _assemble(board, bg, pieces)
    _require(len(answers) == 1, "Expected one colored completion under both folds")
    return answers[0].copy()


def _scatter(rng, board, pieces, bg, frame):
    h, w = board.shape
    for _ in range(16):
        ch = int(rng.integers(max(20, h + 5), 31))
        cw = int(rng.integers(max(20, w + 5), 31))
        if ch * cw <= 2 * (h + 2) * (w + 2):
            continue
        r0 = int(rng.integers(ch - h - 1))
        c0 = int(rng.integers(cw - w - 1))
        canvas = np.full((ch, cw), bg, dtype=np.int8)
        canvas[r0:r0 + h + 2, c0:c0 + w + 2] = frame
        canvas[r0 + 1:r0 + h + 1, c0 + 1:c0 + w + 1] = board
        blocked = np.zeros(canvas.shape, dtype=bool)
        blocked[max(0, r0 - 1):min(ch, r0 + h + 3),
                max(0, c0 - 1):min(cw, c0 + w + 3)] = True
        order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), float(rng.random())))
        for i in order:
            piece = pieces[i]
            ph = max(r for r, c, v in piece) + 1
            pw = max(c for r, c, v in piece) + 1
            offsets = [(r, c) for r, c, v in piece]
            valid = np.ones((ch - ph + 1, cw - pw + 1), dtype=bool)
            for dr, dc in offsets:
                valid &= ~blocked[dr:dr + ch - ph + 1, dc:dc + cw - pw + 1]
            candidates = np.argwhere(valid)
            if not len(candidates):
                break
            if rng.random() < .25:
                edge = candidates[(candidates[:, 0] == 0) | (candidates[:, 1] == 0)
                                  | (candidates[:, 0] == ch - ph) | (candidates[:, 1] == cw - pw)]
                if len(edge):
                    candidates = edge
            r, c = map(int, candidates[int(rng.integers(len(candidates)))])
            for dr, dc, color in piece:
                rr, cc = r + dr, c + dc
                canvas[rr, cc] = color
                # A blank moat preserves even multicolor fragments as objects.
                blocked[max(0, rr - 1):min(ch, rr + 2),
                        max(0, cc - 1):min(cw, cc + 2)] = True
        else:
            return canvas
    return None


def _key(a):
    return tuple(a.shape), a.astype(np.int8).tobytes()


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    # The palette tendency changes distributions only; all roles are visible.
    base_palette = rng.permutation(10)
    exchange = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for _ in range(count):
            for attempt in range(200):
                h, w = map(int, rng.integers(8, 14 if split == "train" else 16, size=2))
                palette = rng.permutation(10) if exchange else base_palette
                bg, frame = map(int, palette[:2])
                nblind = 2 if split == "train" else int(rng.integers(3, 5))
                k = int(rng.integers(max(3, nblind), 7))
                colors = list(map(int, rng.choice(palette[2:], size=k, replace=False)))
                built = _build_board(rng, h, w, bg, colors, nblind, int(rng.integers(4)))
                if built is None:
                    continue
                board, output, pieces = built
                canvas = _scatter(rng, board, pieces, bg, frame)
                if canvas is None:
                    continue
                # Whole-scene rotations vary oriented shapes, never tile actions.
                turns = int(rng.integers(4))
                canvas = np.rot90(canvas, turns).copy()
                output = np.rot90(output, turns).copy()
                key = _key(canvas)
                if key in seen:
                    continue
                try:
                    predicted = solve(canvas)
                except (AssertionError, ValueError):
                    continue
                if not np.array_equal(predicted, output):
                    continue
                seen.add(key)
                puzzle[split].append({"input": canvas, "output": output})
                break
            else:
                raise RuntimeError("Bounded mirror-mosaic construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(inp, out):
    bg, frame, bbox, board, pieces, comps = _parse(inp)
    _require(board.shape == out.shape, "Incorrect interior crop")
    fixed = board != bg
    _require(np.array_equal(board[fixed], out[fixed]), "A fixed cell changed")
    _require(not np.any((out == bg) | (out == frame)), "Background/rim leaked into the mosaic")
    _require(np.array_equal(out, out[::-1]) and np.array_equal(out, out[:, ::-1]),
             "The completed mosaic must obey both central folds")
    required = Counter(int(x) for x in out[~fixed])
    inventory = Counter(v for p in pieces for r, c, v in p)
    _require(required == inventory, "Exterior cells were copied, lost, or recolored")
    predicted, nodes = _assemble(board, bg, pieces)
    _require(len(predicted) == 1 and np.array_equal(predicted[0], out), "Incorrect or ambiguous joint assembly")
    blind = [o for o in orbits(*board.shape) if all(board[p] == bg for p in o)]
    four = [o for o in blind if len(o) == 4]
    _require(len(four) >= 2 and len({int(out[o[0]]) for o in four}) >= 2,
             "Inventory must resolve multiple differently colored blind orbits")
    # A-only plus color totals is still insufficient: exchange equal-size orbits.
    a, b = next((a, b) for a in four for b in four if out[a[0]] != out[b[0]])
    counterfeit = out.copy()
    av, bv = int(out[a[0]]), int(out[b[0]])
    for p in a:
        counterfeit[p] = bv
    for p in b:
        counterfeit[p] = av
    _require(not np.array_equal(counterfeit, out), "Blind-orbit ablation must change the answer")
    _require(Counter(map(int, counterfeit[~fixed])) == required, "Ablation must retain color counts")
    _require(np.array_equal(counterfeit, counterfeit[::-1]) and
             np.array_equal(counterfeit, counterfeit[:, ::-1]), "Ablation must retain both folds")
    for axis in ("lr", "tb", False):
        alternatives, _ = _assemble(board, bg, pieces, axis)
        _require(len(alternatives) == 2, "Each fold must disambiguate inventory placements")
    type_counts = Counter(pieces)
    orbit_ids = {p: i for i, o in enumerate(orbits(*board.shape)) for p in o}
    witnessed = {i for i, o in enumerate(orbits(*board.shape)) if any(fixed[p] for p in o)}
    # Record real cross-orbit rigidity, not only palette/grid hashes.
    shapes = tuple(sorted({tuple((r, c) for r, c, v in p) for p in pieces}))
    return {
        "board_shape": tuple(board.shape), "canvas_shape": tuple(inp.shape),
        "background": bg, "frame": frame, "piece_count": len(pieces),
        "blind_four": len(four), "blind_total": len(blind), "holes": int((~fixed).sum()),
        "multicolor_pieces": sum(len({v for r, c, v in p}) > 1 for p in pieces),
        "repeated_piece": any(n > 1 for n in type_counts.values()),
        "shape_signatures": shapes, "piece_areas": tuple(sorted({len(p) for p in pieces})),
        "gap_components": len(_components(~fixed)),
        "shared_gap": any(len(set(map(int, (out[p] for p in comp)))) > 1
                          for comp in _components(~fixed)),
        "boundary_holes": bool((~fixed)[[0, -1], :].any() or (~fixed)[:, [0, -1]].any()),
        "edge_piece": any(r in (0, inp.shape[0]-1) or c in (0, inp.shape[1]-1)
                          for comp in comps for r, c in comp),
        "nodes": nodes, "payload_count": len(np.unique(out)),
        "layout_key": (tuple(board.shape), (~fixed).tobytes()),
        "axis_parity": (board.shape[0] % 2, board.shape[1] % 2),
    }


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "Illegal pair counts")
    seen = set()
    metrics = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "Duplicate inputs")
            seen.add(key)
            metrics[split].append(_analyze(inp, out))
    _require(all(m["blind_four"] == 2 for m in metrics["train"]), "Training blind-orbit regime changed")
    _require(all(m["blind_four"] in (3, 4) for m in metrics["test"]), "Test blind-orbit shift missing")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed structural/semantic audit plus targeted negative checks."""
    measurements, puzzle_keys = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                _require(np.array_equal(solve(inp), out), "Solver mismatch")
                _require(np.array_equal(inp, before), "Solver mutated input")
                m = _analyze(inp, out)
                m["split"] = split
                measurements.append(m)
    _require(len(puzzle_keys) == 42, "Seed diversity collapsed")
    _require(len({m["board_shape"] for m in measurements}) >= 35, "Insufficient board geometry")
    _require(len({m["canvas_shape"] for m in measurements}) >= 50, "Insufficient canvas geometry")
    _require(len({m["layout_key"] for m in measurements}) >= 180, "Insufficient mask variation")
    _require(len({s for m in measurements for s in m["shape_signatures"]}) >= 60, "Insufficient oriented fragment shapes")
    _require(len({m["piece_count"] for m in measurements}) >= 3, "Fragment counts collapsed")
    _require({m["background"] for m in measurements} == set(range(10)), "Background roles collapsed")
    _require({m["frame"] for m in measurements} == set(range(10)), "Frame roles collapsed")
    _require({m["axis_parity"] for m in measurements} == {(0,0), (0,1), (1,0), (1,1)}, "Missing through-cell folds")
    _require({m["blind_four"] for m in measurements} == {2,3,4}, "Missing blind-orbit shift")
    _require(all(m["multicolor_pieces"] >= 2 for m in measurements), "Rigid color coupling missing")
    for split in ("train", "test"):
        group = [m for m in measurements if m["split"] == split]
        _require(any(m["shared_gap"] for m in group), "No adjacent multicolor hole")
        _require(any(m["boundary_holes"] for m in group), "No boundary holes")
        _require(any(m["edge_piece"] for m in group), "No edge fragments")
    _require(_puzzle_key(generate_puzzle(73)) == _puzzle_key(generate_puzzle(73)), "Nondeterminism")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(101 + nt * 5 + ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "Explicit counts ignored")
    for nt, ne in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4,1.5), (None,1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted illegal counts")
    # A frame/payload relabeling, translation, and rotation cannot change the rule.
    p = generate_puzzle(17, 2, 1)
    perm = np.array([4,9,0,7,1,6,3,8,5,2], dtype=np.int8)
    for pair in p["train"] + p["test"]:
        for turns in range(4):
            inp = np.rot90(perm[pair["input"]], turns).copy()
            expected = np.rot90(perm[pair["output"]], turns).copy()
            _require(np.array_equal(solve(inp), expected), "Role or orientation convention is hidden")
    # Corrupt a fixed cell, damage the inventory, and counterfeit two blind orbits.
    inp, out = p["test"][0]["input"], p["test"][0]["output"]
    bg, frame, bbox, board, pieces, comps = _parse(inp)
    fixed_point = tuple(np.argwhere(board != bg)[0])
    wrong = out.copy(); wrong[fixed_point] = bg
    try:
        _analyze(inp, wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted changed fixed cells")
    missing = inp.copy(); missing[comps[0][0]] = bg
    try:
        solve(missing)
    except AssertionError:
        pass
    else:
        raise AssertionError("Solver accepted a depleted inventory")
    # Different colors in congruent slots are ambiguous; identical fragments
    # exchanged are one colored grid. A multicolor fragment cannot be reflected.
    board = np.array([[0,8,0]], dtype=np.int8)
    _require(len(_assemble(board,0,[((0,0,1),),((0,0,2),)],False)[0])==2, "Ambiguity not detected")
    _require(len(_assemble(board,0,[((0,0,1),),((0,0,1),)],False)[0])==1, "Identical exchanges counted twice")
    board = np.array([[0,0],[8,0]], dtype=np.int8)
    wrong_shape = [((0,1,1),(1,0,2),(1,1,3))]
    _require(not _assemble(board,0,wrong_shape,False)[0], "A tile was illegally reflected")
    return {"seeds":42,"pairs":len(measurements),"board_shapes":len({m["board_shape"] for m in measurements}),
            "mask_layouts":len({m["layout_key"] for m in measurements}),
            "oriented_shapes":len({s for m in measurements for s in m["shape_signatures"]})}
