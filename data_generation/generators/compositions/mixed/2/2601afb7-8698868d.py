"""Counter-circulating carriers and cavity classes. No runtime evidence reads."""
from functools import lru_cache
import hashlib
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0,153200996,784159593,861683990,504455912,931134468,20548733,
              770024280,934047565,2275061,65123242,907925261,601341037,
              295611135,994932815,548122465,78246656,733471446,77298247,
              357835958,519458717,5334628,326126596,879469939,195431942,
              402961837,344961234,998002942,628837339,263703736,443761722,
              405215268,990766189,131054154,15481493,750041079,942725550,
              407742084,847852794,683745683,204404550,840438886)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _FOUR + ((-1,-1),(-1,1),(1,-1),(1,1)) if diagonal else _FOUR
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _symmetries(a):
    return [np.rot90(a, k) for k in range(4)] + [np.rot90(a.T, k) for k in range(4)]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _capacity(t):
    # Constructible cavity counts with connected motif foreground.
    return {6:2, 7:5, 8:7, 9:8, 10:8}[t]


def _isolated(rng, side, count):
    """Interior clue cells; four-isolated and with connected surrounding carrier."""
    if not count:
        return np.zeros((side, side), dtype=bool)
    for _ in range(16):
        mask = np.zeros((side, side), dtype=bool)
        for index in rng.permutation(side * side):
            r, c = divmod(int(index), side)
            if any(0 <= r+dr < side and 0 <= c+dc < side and mask[r+dr,c+dc]
                   for dr, dc in _FOUR):
                continue
            mask[r,c] = True
            if len(_components(np.pad(~mask, 1, constant_values=True))) != 1:
                mask[r,c] = False
                continue
            if int(mask.sum()) == count:
                return mask
    if side in (2, 3):
        cells = [(r,c) for r in range(side) for c in range(side) if (r+c) % 2 == 0]
    elif side == 4:
        cells = [(0,0),(0,2),(1,3),(2,0),(2,2),(3,1),(3,3)]
    else:
        cells = [(r,c) for r in range(0,side,2) for c in range(0,side,2)]
    assert len(cells) >= count
    mask = np.zeros((side, side), dtype=bool)
    for i in rng.permutation(len(cells))[:count]:
        mask[cells[int(i)]] = True
    assert len(_components(np.pad(~mask, 1, constant_values=True))) == 1
    return mask


def _rectangles(rng, side, count):
    boxes = [(0,side,0,side)]
    while len(boxes) < count:
        choices = []
        for i, (r0,r1,c0,c1) in enumerate(boxes):
            choices.extend((i,0,p) for p in range(r0+1,r1-1))
            choices.extend((i,1,p) for p in range(c0+1,c1-1))
        if not choices:
            return None
        i, axis, p = _pick(rng, choices)
        r0,r1,c0,c1 = boxes.pop(i)
        boxes.extend(((r0,p,c0,c1),(p+1,r1,c0,c1)) if axis == 0
                     else ((r0,r1,c0,p),(r0,r1,p+1,c1)))
    mask = np.zeros((side, side), dtype=bool)
    for r0,r1,c0,c1 in boxes:
        if rng.random() < .45:
            r0 = int(rng.integers(r0,r1))
            r1 = int(rng.integers(r0+1,r1+1))
            c0 = int(rng.integers(c0,c1))
            c1 = int(rng.integers(c0+1,c1+1))
        mask[r0:r1,c0:c1] = True
    return np.pad(_pick(rng, _symmetries(mask)), 1).copy()


def _cavities(rng, side, count):
    if count == 0:
        return np.zeros((side, side), dtype=bool)
    inner = side - 2
    if count == 1 and rng.random() < .35:
        return np.pad(np.ones((inner, inner), dtype=bool), 1)
    if rng.random() < .5:
        mask = _rectangles(rng, inner, count)
        if mask is not None:
            return mask
    mask = np.pad(_isolated(rng, inner, count), 1)
    target = int(rng.integers(count, inner * inner + 1))
    # At most four finite sweeps. Never merge cavities or disconnect the frame.
    for _ in range(4):
        changed = False
        for index in rng.permutation(inner * inner):
            if int(mask.sum()) >= target:
                return mask
            r,c = divmod(int(index), inner)
            r,c = r+1,c+1
            if mask[r,c] or not any(mask[r+dr,c+dc] for dr,dc in _FOUR):
                continue
            trial = mask.copy()
            trial[r,c] = True
            if len(_components(trial)) == count and len(_components(~trial)) == 1:
                mask = trial
                changed = True
        if not changed:
            break
    return mask


def _free_origins(boxes, side):
    rr, cc = np.indices((31-side, 31-side))
    valid = np.ones(rr.shape, dtype=bool)
    for r0,r1,c0,c1 in boxes:
        valid &= (rr+side < r0) | (rr > r1) | (cc+side < c0) | (cc > c1)
    return np.argwhere(valid)


@lru_cache(maxsize=512)
def _fallback_boxes(t, rows, cols, classes):
    boxes = [(0,rows*t,0,cols*t)]
    for _ in range(classes):
        free = _free_origins(boxes, t-2)
        if not len(free):
            return None
        r,c = map(int, free[0])
        boxes.append((r,r+t-2,c,c+t-2))
    return tuple(boxes)


@lru_cache(maxsize=1)
def _configs():
    result = []
    for t in range(6,11):
        for rows in range(1,6):
            for cols in range(1,6):
                n = rows * cols
                if not (3 <= n <= 12 and rows*t <= 30 and cols*t <= 30):
                    continue
                feasible = [m for m in range(2,min(6,n,_capacity(t)+1)+1)
                            if _fallback_boxes(t,rows,cols,m) is not None]
                if feasible:
                    result.append((t,rows,cols,max(feasible)))
    return tuple(result)


def _place(rng, bg, mosaic, motifs, witness):
    h,w = mosaic.shape
    s = next(iter(motifs.values())).shape[0]
    boxes = None
    for _ in range(12):
        r,c = int(rng.integers(31-h)), int(rng.integers(31-w))
        trial = [(r,r+h,c,c+w)]
        for _ in motifs:
            free = _free_origins(trial,s)
            if not len(free):
                break
            rr,cc = map(int, free[int(rng.integers(len(free)))])
            trial.append((rr,rr+s,cc,cc+s))
        if len(trial) == len(motifs)+1:
            boxes = trial
            break
    if boxes is None:
        boxes = list(_fallback_boxes(s+2,h//(s+2),w//(s+2),len(motifs)))
    canvas = np.full((30,30),bg,dtype=np.int8)
    r0,r1,c0,c1 = boxes[0]
    canvas[r0:r1,c0:c1] = mosaic
    locations = sorted(boxes[1:], key=lambda b:(b[0],b[2]))
    # One teaching example explicitly refutes spatial-order assignment.
    order = sorted(motifs, reverse=True) if witness else list(map(int,rng.permutation(list(motifs))))
    for (a,b,c,d), k in zip(locations,order):
        canvas[a:b,c:d] = motifs[k]
    a = min(b[0] for b in boxes)
    b = max(b[1] for b in boxes)
    c = min(b[2] for b in boxes)
    d = max(b[3] for b in boxes)
    a -= int(rng.integers(min(3,a)+1))
    c -= int(rng.integers(min(3,c)+1))
    b += int(rng.integers(min(3,30-b)+1))
    d += int(rng.integers(min(3,30-d)+1))
    return canvas[a:b,c:d].copy()


def _make_pair(rng, config, classes, witness=False):
    t,rows,cols,_ = config
    n,s = rows*cols,t-2
    bg = int(rng.integers(10))
    order = list(classes) + list(map(int,rng.choice(classes,n-len(classes))))
    if witness:
        order = [1,1,2,2]
    for _ in range(32):
        order = [order[int(i)] for i in rng.permutation(n)]
        if not witness or order != order[2:]+order[:2]:
            break
    else:
        order = [1,1,2,2]
    palette = list(map(int,rng.permutation([c for c in range(10) if c != bg])))
    if witness:
        colors = palette[:n]
    else:
        pool = palette[:int(rng.integers(1,min(6,n)+1))]
        colors = list(map(int,rng.choice(pool,n)))
        if rng.random() < .35:
            colors = list(map(int,rng.choice(palette,n,replace=n>9)))
            # Bound the color pool so at least one motif foreground is available.
            if len(set(colors)) > 6:
                colors = [palette[i%6] for i in rng.permutation(n)]
    motifs = {}
    for k in sorted(classes):
        mask = _cavities(rng,s,k)
        if witness and k == 1:
            mask = np.pad(np.ones((s-2,s-2),dtype=bool),1)
        if witness and k == 2:
            mask = np.zeros((s,s),dtype=bool)
            mask[1,1] = mask[2,2] = mask[2,3] = True
            mask = _pick(rng,_symmetries(mask)).copy()
        forbidden = {bg} | {colors[(i-1)%n] for i in range(n) if order[(i+1)%n] == k}
        fg = int(_pick(rng,[c for c in range(10) if c not in forbidden]))
        motifs[k] = np.where(mask,bg,fg).astype(np.int8)
    mosaic = np.empty((rows*t,cols*t),dtype=np.int8)
    output = np.empty_like(mosaic)
    shifted_colors = np.roll(np.asarray(colors),1)
    shifted_classes = np.roll(np.asarray(order),-1)
    for i in range(n):
        r,c = (i//cols)*t,(i%cols)*t
        tile = np.full((t,t),colors[i],dtype=np.int8)
        tile[1:-1,1:-1][_isolated(rng,s,order[i])] = bg
        mosaic[r:r+t,c:c+t] = tile
        color,k = int(shifted_colors[i]),int(shifted_classes[i])
        tile[:] = color
        tile[1:-1,1:-1] = np.where(motifs[k] == bg,color,motifs[k])
        output[r:r+t,c:c+t] = tile
    return {"input":_place(rng,bg,mosaic,motifs,witness),"output":output}


def _interpret_background(a, bg):
    components = sorted(_components(a != bg),key=len,reverse=True)
    assert 3 <= len(components) <= 7
    boxes = [_box(cells) for cells in components]
    r0,r1,c0,c1 = boxes[0]
    sizes = {(b-a,d-c) for a,b,c,d in boxes[1:]}
    assert len(sizes) == 1
    s,sw = next(iter(sizes))
    assert s == sw and 4 <= s <= 8
    t = s+2
    assert (r1-r0)%t == (c1-c0)%t == 0
    rows,cols = (r1-r0)//t,(c1-c0)//t
    n = rows*cols
    assert 3 <= n <= 12
    for i,(a0,a1,b0,b1) in enumerate(boxes):
        for c0_,c1_,d0,d1 in boxes[i+1:]:
            assert a1 < c0_ or c1_ < a0 or b1 < d0 or d1 < b0
    motifs, physical = {}, []
    for a0,a1,b0,b1 in sorted(boxes[1:],key=lambda b:(b[0],b[2])):
        motif = a[a0:a1,b0:b1]
        fg = int(motif[0,0])
        assert fg != bg and np.all(_perimeter(motif) == fg)
        assert set(map(int,np.unique(motif))) <= {bg,fg}
        k = len(_components(motif == bg))
        assert k not in motifs and k <= 8
        motifs[k] = motif.copy()
        physical.append(k)
    mosaic = a[r0:r1,c0:c1]
    colors,counts,clues = [],[],[]
    for i in range(n):
        r,c = (i//cols)*t,(i%cols)*t
        tile = mosaic[r:r+t,c:c+t]
        color = int(tile[0,0])
        assert color != bg and np.all(_perimeter(tile) == color)
        assert set(map(int,np.unique(tile))) <= {bg,color}
        clue = tile[1:-1,1:-1] == bg
        assert not np.any(clue[1:] & clue[:-1])
        assert not np.any(clue[:,1:] & clue[:,:-1])
        assert len(_components(tile != bg)) == 1
        colors.append(color)
        counts.append(int(clue.sum()))
        clues.append(clue.copy())
    assert set(counts) == set(motifs) and 2 <= len(motifs) <= 6
    assert all(int(motifs[counts[(i+1)%n]][0,0]) != colors[(i-1)%n] for i in range(n))
    masks = {k:m == bg for k,m in motifs.items()}
    areas = {k:int(mask.sum()) for k,mask in masks.items()}
    return {"bg":bg,"t":t,"rows":rows,"cols":cols,"n":n,
            "colors":colors,"counts":counts,"motifs":motifs,"clues":clues,
            "masks":masks,"physical":physical,"mosaic_box":boxes[0],
            "motif_boxes":boxes[1:],"shape":a.shape,
            "area_inversion":any(k < j and areas[k] > areas[j] for k in areas for j in areas),
            "diagonal":any(len(_components(mask,True)) < k for k,mask in masks.items()),
            "asymmetric":any(all(not np.array_equal(mask,v) for v in _symmetries(mask)[1:])
                             for mask in masks.values()),
            "nonpositional":any(not np.array_equal(clue,masks[k]) for k,clue in zip(counts,clues))}


def _interpret(value):
    a = _grid(value)
    candidates = []
    for bg in map(int,np.unique(a)):
        try:
            candidates.append(_interpret_background(a,bg))
        except (AssertionError,IndexError,ValueError):
            continue
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous mosaic and cavity dictionary")
    return candidates[0]


def _render(info, color_shift=1, class_shift=-1, assignment=None):
    t,n,cols = info["t"],info["n"],info["cols"]
    output = np.empty((info["rows"]*t,cols*t),dtype=np.int8)
    motifs = info["motifs"] if assignment is None else assignment
    for i in range(n):
        color = info["colors"][(i-color_shift)%n]
        motif = motifs[info["counts"][(i-class_shift)%n]]
        tile = np.full((t,t),color,dtype=np.int8)
        tile[1:-1,1:-1] = np.where(motif == info["bg"],color,motif)
        r,c = (i//cols)*t,(i%cols)*t
        output[r:r+t,c:c+t] = tile
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only visible geometry, carrier colors, clue counts and cavity counts."""
    return _render(_interpret(input_grid))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    relationship = _pick(rng,("growth","new_class","new_scale"))
    puzzle = {"train":[],"test":[]}
    used,train_sizes,train_tiles = set(),[],[]
    for split,number in (("train",int(num_train)),("test",int(num_test))):
        for i in range(number):
            witness = split == "train" and i == 0
            eligible = list(_configs())
            if split == "train":
                if relationship == "growth":
                    eligible = [c for c in eligible if c[1]*c[2] <= 6]
                if relationship == "new_scale":
                    eligible = [c for c in eligible if c[0] in (7,8)]
                if witness:
                    eligible = [c for c in eligible if c[1:3] == (2,2) and c[0] >= 7]
                elif i == 1:
                    eligible = [c for c in eligible if c[1]*c[2] == 3 and c[0] >= 7 and c[3] >= 3]
                    if relationship == "new_scale":
                        eligible = [c for c in eligible if c[0] != train_tiles[0]]
            elif relationship == "growth":
                eligible = [c for c in eligible if c[1]*c[2] > max(train_sizes)]
            elif relationship == "new_scale":
                eligible = [c for c in eligible if c[0] >= 9]
            else:
                eligible = [c for c in eligible if _capacity(c[0]) >= 4]
            config = _pick(rng,eligible)
            cap = min(_capacity(config[0]),3) if split == "train" and relationship == "new_class" else _capacity(config[0])
            if witness:
                classes = [1,2]
            elif split == "train" and i == 1:
                classes = _pick(rng,([0,1,3],[0,2,3],[1,2,3]))
            else:
                m = int(rng.integers(2,min(config[3],cap+1)+1))
                classes = list(map(int,rng.choice(cap+1,m,replace=False)))
                if split == "test" and relationship == "new_class" and max(classes) <= 3:
                    classes[0] = int(rng.integers(4,cap+1))
            for _ in range(16):
                pair = _make_pair(rng,config,classes,witness)
                key = _key(pair["input"])
                if key not in used:
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
            used.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_sizes.append(config[1]*config[2])
                train_tiles.append(config[0])
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(int(num_train))]
    return puzzle


def _validate(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    nt,ne = len(puzzle["train"]),len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen,analyses = set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            a,b = _grid(pair["input"]),_grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _interpret(a)
            assert np.array_equal(_render(info),b), "wrong cycle, motif, substitution or crop"
            assert not np.any(b == info["bg"]), "neutral clues must disappear"
            colors,counts = [],[]
            t,n,cols = info["t"],info["n"],info["cols"]
            for i in range(n):
                r,c = (i//cols)*t,(i%cols)*t
                tile = b[r:r+t,c:c+t]
                color = int(tile[0,0])
                assert np.all(_perimeter(tile) == color)
                colors.append(color)
                counts.append(len(_components(tile[2:-2,2:-2] == color)))
                assert color == info["colors"][(i-1)%n]
                assert counts[-1] == info["counts"][(i+1)%n]
            assert sorted(colors) == sorted(info["colors"])
            assert sorted(counts) == sorted(info["counts"])
            analyses[split].append(info)
    train = analyses["train"]
    for feature in ("area_inversion","diagonal","asymmetric","nonpositional"):
        assert any(a[feature] for a in train), "missing training evidence: " + feature
    assert any(a["rows"] > 1 and a["cols"] > 1 for a in train)
    assert any(min(a["rows"],a["cols"]) == 1 for a in train)
    assert any(len(set(a["counts"])) < a["n"] for a in train)
    for cshift,kshift in ((0,0),(0,-1),(1,0),(1,1),(-1,1),(-1,-1)):
        assert any(not np.array_equal(_render(a),_render(a,cshift,kshift)) for a in train), "training permits a simpler cycle"
    assert any(not np.array_equal(_render(a),_render(a,assignment=dict(zip(sorted(a["motifs"]),
                    (a["motifs"][k] for k in a["physical"]))))) for a in train), "spatial matching not refuted"
    return analyses


def validate_generated_puzzle(puzzle):
    """Check pixels, class/color multiset conservation and training witnesses."""
    try:
        _validate(puzzle)
    except (ValueError,TypeError,KeyError,IndexError) as exc:
        raise AssertionError("malformed generated puzzle") from exc


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        h.update(bytes((len(puzzle[split]),)))
        for pair in puzzle[split]:
            for role in ("input","output"):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 fixed seeds, bounded in-memory metrics, all explicit count contracts."""
    shapes,layouts,counts,patterns,backgrounds,fingerprints = set(),set(),set(),set(),set(),set()
    measured = {k:0 for k in ("pairs","repeated_classes","repeated_colors","zero",
                              "large_cavities","area_inversion","diagonal","asymmetric",
                              "growth_shift","class_shift","scale_shift")}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        analyses = _validate(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        tr,te = analyses["train"],analyses["test"]
        shifts = (max(a["n"] for a in te) > max(a["n"] for a in tr),
                  max(max(a["counts"]) for a in te) > max(max(a["counts"]) for a in tr),
                  min(a["t"] for a in te) > max(a["t"] for a in tr))
        assert any(shifts), "missing visible train/test shift"
        for name,flag in zip(("growth_shift","class_shift","scale_shift"),shifts):
            measured[name] += int(flag)
        for split in ("train","test"):
            for pair,a in zip(puzzle[split],analyses[split]):
                assert np.array_equal(solve(pair["input"].copy()),pair["output"])
                shapes.add(a["shape"])
                layouts.add((a["t"],a["rows"],a["cols"]))
                counts.add(tuple(sorted(set(a["counts"]))))
                backgrounds.add(a["bg"])
                measured["pairs"] += 1
                measured["repeated_classes"] += len(set(a["counts"])) < a["n"]
                measured["repeated_colors"] += len(set(a["colors"])) < a["n"]
                measured["zero"] += 0 in a["counts"]
                for name in ("area_inversion","diagonal","asymmetric"):
                    measured[name] += int(a[name])
                for mask in a["masks"].values():
                    patterns.add((mask.shape,min(m.tobytes() for m in _symmetries(mask))))
                    measured["large_cavities"] += any(len(c) >= 5 for c in _components(mask))
    assert len(fingerprints) == 42 and len(shapes) >= 20 and len(layouts) >= 15
    assert len(counts) >= 15 and len(patterns) >= 50 and len(backgrounds) == 10
    assert all(measured[k] >= 3 for k in measured if k != "pairs")
    assert _fingerprint(generate_puzzle(818,7,3)) == _fingerprint(generate_puzzle(818,7,3))
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            puzzle = generate_puzzle(73,nt,ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
            validate_generated_puzzle(puzzle)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    sample = generate_puzzle(0,2,1)
    sample["test"][0]["output"][0,0] = (int(sample["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(sample)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds":len(HOST_SEEDS),"puzzles":len(fingerprints),"input_shapes":len(shapes),
            "layouts":len(layouts),"count_sets":len(counts),"cavity_shapes_mod_D4":len(patterns),
            "backgrounds":len(backgrounds),"measured":measured}
