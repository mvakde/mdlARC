"""Stencil interference loom: paired axes meet as keyed bitmap XOR.

No files, parent code, seed state, or demonstrations are used by solve().
All sampling and diagnostics are bounded; self_test does not write artifacts.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


HOST_SEEDS = (0,18095931,255159643,101593546,391942768,22513014,604655999,
    356609590,674548635,991975226,952279821,784943602,567971522,589426215,
    492753019,567982884,325961093,87162204,973055825,826415818,651114115,
    319662217,813377723,732272847,127577282,216104548,454184393,708911400,
    128185463,971707131,512747647,913457293,287603602,422036570,626701246,
    183298993,799075419,710381076,874896868,427548649,766393835,736016730)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(a, b):
    a, b = _integer(a, "num_train"), _integer(b, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2<=num_train<=10, 1<=num_test<=4, num_train>num_test")
    return a, b


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid bounds"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((a >= 0) & (a <= 9)), "palette bounds"
    return a


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _symmetry(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4)


def _parse(value):
    """Recover every command from the input's explicit upright registration."""
    a = _array(value)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)
    rr, _ = np.nonzero(a != bg)
    assert len(rr), "missing structures"
    y = int(rr.min())
    top = np.flatnonzero(a[y] != bg)
    x, end = int(top[0]), int(top[-1]) + 1
    frame = int(a[y, x])
    assert frame != bg and np.all(a[y, x:end] == frame)
    dividers = np.flatnonzero(a[y+1, x:end] == frame)
    assert len(dividers) >= 2 and dividers[0] == 0 and dividers[-1] == end-x-1
    gaps = np.diff(dividers)
    assert np.all(gaps == gaps[0])
    n, k = int(gaps[0])-1, len(dividers)-1
    assert 2 <= n <= 5 and 1 <= k <= 4
    assert end-x == k*(n+1)+1 and y+n+4 < a.shape[0]
    header = a[y:y+n+4, x:end]
    assert np.all(header[[0, n+1, n+3]] == frame)
    assert np.all(header[:, dividers] == frame)
    labels, masks, targets = [], [], []
    for j in range(k):
        c = 1+j*(n+1)
        panel = header[1:n+1, c:c+n]
        colors = np.unique(panel[panel != bg])
        assert len(colors) == 1
        label = int(colors[0])
        assert label != frame and label not in labels
        swatch = header[n+2, c:c+n]
        target = int(swatch[0])
        assert target != bg and np.all(swatch == target)
        labels.append(label)
        masks.append(panel == label)
        targets.append(target)
    by_label = {label: i for i, label in enumerate(labels)}
    board_y = y+n+5
    assert np.all(a[y+n+4] == bg), "one blank spacer row"
    bottom = int(rr.max())+1
    assert board_y+3 <= bottom < a.shape[0]
    border_cols = np.flatnonzero(a[board_y] != bg)
    assert len(border_cols) >= 3
    board_x, board_end = int(border_cols[0]), int(border_cols[-1])+1
    board = a[board_y:bottom, board_x:board_end]
    R, C = board.shape[0]-2, board.shape[1]-2
    assert 1 <= R <= 10 and 1 <= C <= 10 and R*n <= 30 and C*n <= 30
    assert np.all(board[1:-1, 1:-1] == bg)
    assert np.all(board[[0, -1]][:, [0, -1]] == frame)
    assert np.all(board[[0, -1]] != bg) and np.all(board[:, [0, -1]] != bg)
    used = np.zeros(a.shape, dtype=bool)
    used[y:y+n+4, x:end] = True
    used[board_y:bottom, board_x:board_end] = True
    assert np.all(a[~used] == bg), "unexpected material outside the two structures"

    def decode(first, second):
        return np.array([by_label.get(int(u), -1) if u == v else -1
                         for u, v in zip(first, second)], dtype=np.int8)

    row_first, row_second = board[1:-1, 0], board[1:-1, -1]
    col_first, col_second = board[0, 1:-1], board[-1, 1:-1]
    return dict(n=n, k=k, R=R, C=C, bg=bg, frame=frame, labels=labels,
        masks=masks, targets=targets, rows=decode(row_first, row_second),
        cols=decode(col_first, col_second),
        single_rows=np.array([by_label.get(int(v), -1) for v in row_first], dtype=np.int8),
        single_cols=np.array([by_label.get(int(v), -1) for v in col_first], dtype=np.int8),
        broken=int(np.count_nonzero(row_first != row_second)+np.count_nonzero(col_first != col_second)),
        orphan=int(sum(u == v and int(u) not in by_label and u != frame
                       for first, second in ((row_first,row_second),(col_first,col_second))
                       for u,v in zip(first,second))),
        library=(y,x,end), board=(board_y,board_x,bottom,board_end))


def _render(p, variant="xor", symmetry=0):
    """Forward tile renderer used by solve and explicit wrong-rule probes."""
    n, bg = p["n"], p["bg"]
    rows, cols = p["rows"].copy(), p["cols"].copy()
    if variant == "one_sided":
        rows, cols = p["single_rows"], p["single_cols"]
    if variant == "first_occurrence":
        for axis in (rows, cols):
            seen = set()
            for i, label in enumerate(axis):
                if label >= 0 and label in seen:
                    axis[i] = -1
                elif label >= 0:
                    seen.add(int(label))
    masks = [np.ones((n,n), dtype=bool) if variant == "solid" else _symmetry(m, symmetry)
             for m in p["masks"]]
    empty = np.zeros((n,n), dtype=bool)
    out = np.full((p["R"]*n, p["C"]*n), bg, dtype=np.int8)
    for r, ri in enumerate(rows):
        for c, ci in enumerate(cols):
            ri, ci = int(ri), int(ci)
            a, b = masks[ri] if ri >= 0 else empty, masks[ci] if ci >= 0 else empty
            tile = out[r*n:(r+1)*n, c*n:(c+1)*n]
            if variant == "intersection":
                if ri >= 0:
                    tile[a & b] = p["targets"][ri]
                continue
            if variant == "same_identity_only" and ri >= 0 and ci >= 0 and ri != ci:
                continue
            aa, bb = a & ~b, b & ~a
            if variant == "union":
                aa = a  # Row priority on overlap is a plausible wrong answer.
            if variant == "pigment_cancel" and ri >= 0 and ci >= 0:
                if p["targets"][ri] != p["targets"][ci]:
                    aa = a
            if ri >= 0:
                tile[aa] = p["labels"][ri] if variant == "no_key" else p["targets"][ri]
            if ci >= 0:
                tile[bb] = p["labels"][ci] if variant == "no_key" else p["targets"][ci]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_parse(input_grid))


def _motif(rng, n):
    m = np.zeros((n,n), dtype=bool)
    style = int(rng.integers(11))
    if style == 0:
        m[:] = True
    elif style == 1:
        m[[0,-1],:] = True
        m[:,[0,-1]] = True
    elif style == 2:
        np.fill_diagonal(m, True)
        if rng.random() < .4:
            m |= np.fliplr(np.eye(n, dtype=bool))
    elif style == 3:
        m[int(rng.integers(n)),:] = True
        m[:,int(rng.integers(n))] = True
    elif style == 4:
        rr,cc = np.indices((n,n))
        m = (rr+cc) % 2 == int(rng.integers(2))
    elif style == 5:
        m[rng.choice(n, int(rng.integers(1,n+1)), replace=False),:] = True
    elif style == 6:
        y0,y1 = sorted(map(int, rng.integers(0,n,size=2)))
        x0,x1 = sorted(map(int, rng.integers(0,n,size=2)))
        m[y0:y1+1,x0:x1+1] = True
    elif style == 7:
        r,c = map(int,rng.integers(n,size=2))
        for _ in range(int(rng.integers(2,2*n*n+1))):
            m[r,c] = True
            dr,dc = ((1,0),(-1,0),(0,1),(0,-1))[int(rng.integers(4))]
            r,c = min(n-1,max(0,r+dr)),min(n-1,max(0,c+dc))
    else:
        m = rng.random((n,n)) < float(rng.uniform(.12,.90))
    if not m.any():
        m.flat[int(rng.integers(n*n))] = True
    return _symmetry(m, int(rng.integers(8))).copy()


def _witness_masks(masks):
    a,b = masks[:2]
    return (bool((a & b).any() and (a & ~b).any() and (b & ~a).any())
            and all(any(not np.array_equal(m, _symmetry(m,t)) for m in masks[:2])
                    for t in range(1,8)))


def _masks(rng, n, k, witness=False, bank=None):
    if bank:
        return [bank[int(rng.integers(len(bank)))].copy() for _ in range(k)]
    masks = [_motif(rng,n) for _ in range(k)]
    if witness:
        for _ in range(64):
            if _witness_masks(masks):
                return masks
            masks[:2] = [_motif(rng,n), _motif(rng,n)]
        # Dense masks with different missing cells have overlap and exclusive
        # parts; their joint stabilizer is the identity, even at n=2.
        masks[:2] = [np.ones((n,n),bool),np.ones((n,n),bool)]
        masks[0][0,0] = False
        masks[1][0,1] = False
        assert _witness_masks(masks)
    elif k >= 2 and rng.random() < .20:
        masks[1] = masks[0].copy()
    return masks


def _axis(rng, length, k, kind, max_repeat=None):
    if kind == 0:
        values = [0,0,1,-1,-1]
        nulls = ["blank","broken"]
    elif kind == 1:
        values = list(range(k))+[-1,-1]
        nulls = ["blank","orphan"]
    elif kind == 2:
        values = [0,0,0,1,-1]
        nulls = ["blank"]
    else:
        values = [int(rng.integers(k)), -1, int(rng.integers(k))]
        nulls = ["blank"]
    while len(values) < length:
        options = [-1]+[i for i in range(k) if max_repeat is None or values.count(i)<max_repeat]
        values.append(int(rng.choice(options)))
        if values[-1] == -1:
            nulls.append(str(rng.choice(["blank","broken","mismatch","orphan"])))
    assert len(values) == length
    descriptors = []
    ni = 0
    for value in values:
        descriptors.append((value, "paired" if value >= 0 else nulls[ni]))
        ni += value < 0
    # Mix contiguous runs and scattered coordinates without changing semantics.
    if rng.random() < .80:
        rng.shuffle(descriptors)
    return descriptors


def _make_pair(rng, n, k, R, C, kind=-1, bank=None, max_repeat=None):
    palette = list(map(int, rng.permutation(10)))
    bg,frame = palette[:2]
    labels = palette[2:2+k]
    orphan = palette[2+k]
    masks = _masks(rng,n,k,witness=kind in (0,1),bank=bank)
    targets = [int(rng.choice([x for x in range(10) if x != bg])) for _ in range(k)]
    if kind == 0:
        targets[0] = int(rng.choice([x for x in range(10) if x not in (bg,labels[0])]))
        targets[1] = int(rng.choice([x for x in range(10) if x not in (bg,labels[1],targets[0])]))
    elif kind == 1:
        shared = int(rng.choice([x for x in range(10) if x not in (bg,labels[0],labels[1])]))
        targets[0] = targets[1] = shared
        targets[2] = int(rng.choice([x for x in range(10) if x not in (bg,shared,labels[2])]))
    rows = _axis(rng,R,k,kind,max_repeat)
    cols = _axis(rng,C,k,kind,max_repeat)
    W = max(k*(n+1)+1, C+2)
    top,bottom,left,right = map(int,rng.integers(1,3,size=4))
    assert W+left+right <= 30
    header_x = left+int(rng.integers(W-(k*(n+1)+1)+1))
    board_x = left+int(rng.integers(W-(C+2)+1))
    board_y = top+n+5
    a = np.full((top+n+R+7+bottom,W+left+right),bg,dtype=np.int8)
    header = a[top:top+n+4,header_x:header_x+k*(n+1)+1]
    header[[0,n+1,n+3],:] = frame
    header[:,::n+1] = frame
    # Permuting card order never permutes the visibly color-addressed commands.
    order = list(map(int,rng.permutation(k)))
    for slot,i in enumerate(order):
        c = 1+slot*(n+1)
        header[1:n+1,c:c+n] = np.where(masks[i],labels[i],bg)
        header[n+2,c:c+n] = targets[i]
    board = a[board_y:board_y+R+2,board_x:board_x+C+2]
    board[[0,-1],:] = frame
    board[:,[0,-1]] = frame

    def marks(descriptors):
        first,second = [],[]
        for i,typ in descriptors:
            if i >= 0:
                u = v = labels[i]
            elif typ == "blank":
                u = v = frame
            elif typ == "orphan":
                u = v = orphan
            elif typ == "mismatch":
                u,v = map(int,rng.choice(labels,2,replace=False))
            else:
                # First edge carries the tempting false command, guaranteeing
                # an observable counterexample to reading only that edge.
                u,v = int(rng.choice(labels)),frame
            first.append(u)
            second.append(v)
        return first,second

    board[1:-1,0],board[1:-1,-1] = marks(rows)
    board[0,1:-1],board[-1,1:-1] = marks(cols)
    # Construct labels from the sampled records by Boolean tensor expansion.
    # This is separate from solve's parsed, forward, per-tile renderer.
    def field(descriptors, axis):
        on = np.zeros((R,C,n,n),dtype=bool)
        ink = np.full((R,C,n,n),bg,dtype=np.int8)
        for pos,(i,_) in enumerate(descriptors):
            if i >= 0:
                if axis == 0:
                    on[pos,:,:,:] = masks[i]
                    ink[pos,:,:,:] = targets[i]
                else:
                    on[:,pos,:,:] = masks[i]
                    ink[:,pos,:,:] = targets[i]
        return on,ink

    row_on,row_ink = field(rows,0)
    col_on,col_ink = field(cols,1)
    pixels = np.where(row_on ^ col_on,np.where(row_on,row_ink,col_ink),bg)
    output = pixels.transpose(0,2,1,3).reshape(R*n,C*n).astype(np.int8)
    return {"input":a,"output":output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _counts(num_train,num_test)
    seed = _integer(seed,"seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    base = int(rng.choice([2,3])) if regime in (0,1) else int(rng.choice([2,3,4]))
    train = [(base,int(rng.integers(2,5)),0),
             (base+1 if regime == 0 else base,int(rng.integers(3,5)),1)]
    train += [(int(rng.choice([base,base+1])) if regime == 0 else base,
               int(rng.integers(2,5)),-1) for _ in range(nt-2)]
    puzzle = {"train":[],"test":[]}
    seen,bank = set(),[]
    for split,count in (("train",nt),("test",ne)):
        for index in range(count):
            if split == "train":
                n,k,kind = train[index]
                minimum = 5 if kind == 0 else k+2 if kind == 1 else 3
                if regime == 1:
                    minimum = max(5,minimum)
                R = int(rng.integers(minimum,min(7,30//n)+1))
                C = int(rng.integers(minimum,min(7,30//n)+1))
            else:
                n = int(rng.integers(base+2,6)) if regime == 0 else base
                k = int(rng.choice([t[1] for t in train])) if regime == 0 else int(rng.integers(2,5))
                kind = 2 if regime == 2 else -1
                if regime == 1:
                    R,C = int(rng.integers(3,5)),int(rng.integers(8,min(10,30//n)+1))
                    if rng.random() < .5:
                        R,C = C,R
                else:
                    low = 5 if kind == 2 else 3
                    R,C = map(int,rng.integers(low,min(10,30//n)+1,size=2))
            for _ in range(32):
                pair = _make_pair(rng,n,k,R,C,kind,
                    bank=bank if split == "test" and regime == 2 else None,
                    max_repeat=2 if split == "train" and regime == 2 else None)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness search exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                bank.extend(m.copy() for m in _parse(pair["input"])["masks"])
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(p):
    rows,cols,masks = p["rows"],p["cols"],p["masks"]
    same = overlap = mixed = 0
    for ri in rows:
        for ci in cols:
            if ri >= 0 and ci >= 0:
                if ri == ci:
                    same += 1
                else:
                    a,b = masks[int(ri)],masks[int(ci)]
                    overlap += int((a & b).sum())
                    mixed += int((a & ~b).any() and (b & ~a).any())
    return dict(n=p["n"],k=p["k"],R=p["R"],C=p["C"],
        roles=(p["bg"],p["frame"]),same=same,overlap=overlap,mixed=mixed,
        row_only=bool(np.any(rows>=0) and np.any(cols<0)),
        col_only=bool(np.any(cols>=0) and np.any(rows<0)),
        shared=len(set(p["targets"]))<p["k"],broken=p["broken"],orphan=p["orphan"],
        repeats=max([0]+list(Counter(int(x) for x in rows if x>=0).values())
                    +list(Counter(int(x) for x in cols if x>=0).values())))


_WRONG_RULES = ("union","intersection","pigment_cancel","one_sided",
    "first_occurrence","solid","no_key","same_identity_only")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen,training = set(),[]
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={"input","output"}
            a,out = _array(pair["input"]),_array(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            p = _parse(a)
            n = p["n"]
            assert out.shape == (p["R"]*n,p["C"]*n)
            assert set(map(int,np.unique(out))) <= {p["bg"],*p["targets"]}
            assert np.any(out != p["bg"]), "generated answers must be visible"
            # Inverse indexing checks every pixel without calling solve or
            # generation's tensor-field construction.
            for y,x in np.ndindex(out.shape):
                ri,ci = int(p["rows"][y//n]),int(p["cols"][x//n])
                a_on = ri>=0 and bool(p["masks"][ri][y%n,x%n])
                b_on = ci>=0 and bool(p["masks"][ci][y%n,x%n])
                expected = p["bg"] if a_on==b_on else p["targets"][ri if a_on else ci]
                assert out[y,x] == expected, "incorrect exclusive/canceled pixel"
            if split == "train":
                training.append((p,out,_features(p)))
    for feature in ("same","overlap","mixed","row_only","col_only","shared","broken","orphan"):
        assert any(f[feature] for _,_,f in training), f"missing training evidence: {feature}"
    assert any(f["repeats"]>=2 for _,_,f in training)
    for variant in _WRONG_RULES:
        assert any(not np.array_equal(out,_render(p,variant)) for p,out,_ in training), variant
    for t in range(1,8):
        assert any(not np.array_equal(out,_render(p,symmetry=t)) for p,out,_ in training), "orientation"


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                a = np.asarray(pair[role],dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded streaming checks; returns small descriptive measurements."""
    shapes,motifs,axis_layouts,roles,signatures = set(),set(),set(),set(),set()
    ns,ks,features = set(),set(),Counter()
    shifts = Counter(larger_test_scale=0,more_rectangular_test=0,
                     familiar_masks_with_more_repetition=0)
    pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        local = {"train":[],"test":[]}
        for split in ("train","test"):
            for pair in p[split]:
                a = pair["input"]
                info = _parse(a)
                local[split].append(info)
                assert np.array_equal(solve(a.copy()),pair["output"])
                f = _features(info)
                pairs += 1
                ns.add(info["n"])
                ks.add(info["k"])
                shapes.add((a.shape,pair["output"].shape))
                roles.add((info["bg"],info["frame"]))
                motifs.update((m.shape,m.tobytes()) for m in info["masks"])
                axis_layouts.add((tuple(info["rows"]),tuple(info["cols"])))
                for name in ("same","overlap","mixed","shared","broken","orphan"):
                    features[name] += bool(f[name])
        training,testing = local["train"],local["test"]
        shifts["larger_test_scale"] += min(q["n"] for q in testing)>max(q["n"] for q in training)
        aspect = lambda q: max(q["R"],q["C"])/min(q["R"],q["C"])
        shifts["more_rectangular_test"] += min(map(aspect,testing))>max(map(aspect,training))
        bank = {_key(m) for q in training for m in q["masks"]}
        familiar = all(_key(m) in bank for q in testing for m in q["masks"])
        more_repeat = min(_features(q)["repeats"] for q in testing)>max(_features(q)["repeats"] for q in training)
        shifts["familiar_masks_with_more_repetition"] += familiar and more_repeat
        if seed == 0:
            assert _signature(p) == _signature(generate_puzzle(seed))
            # Palette equivariance includes zero as a stencil identity.
            sample = p["train"][0]
            for shift in range(10):
                perm = (np.arange(10,dtype=np.int8)+shift)%10
                assert np.array_equal(solve(perm[sample["input"]]),perm[sample["output"]])
            # Reordering entire cards leaves the color-addressed cloth intact.
            a = sample["input"].copy()
            info = _parse(a)
            y,x,_ = info["library"]
            n,k = info["n"],info["k"]
            first = a[y+1:y+n+3,x+1:x+n+1].copy()
            last_c = x+1+(k-1)*(n+1)
            a[y+1:y+n+3,x+1:x+n+1] = a[y+1:y+n+3,last_c:last_c+n]
            a[y+1:y+n+3,last_c:last_c+n] = first
            assert np.array_equal(solve(a),sample["output"])
    assert len(signatures) == 42
    assert ns == {2,3,4,5} and ks == {2,3,4}
    assert len(shapes)>=35 and len(motifs)>=100 and len(axis_layouts)>=150 and len(roles)>=40
    assert all(features[name]>=42 for name in features)
    assert all(value>0 for value in shifts.values())
    # Exercise all legal explicit counts, not just boundaries.
    legal_calls = 0
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(17,nt,ne)
            assert len(p["train"])==nt and len(p["test"])==ne
            legal_calls += 1
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),
                  (4,False),(4.0,1),(4,1.0),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Literal worked crossing, independent of parsing and random generation.
    p = dict(n=3,bg=0,R=1,C=1,rows=np.array([0]),cols=np.array([1]),
        masks=[np.array([[1,1,0],[0,1,0],[0,0,0]],bool),
               np.array([[0,1,0],[0,1,1],[0,0,0]],bool)],targets=[2,7])
    assert np.array_equal(_render(p),np.array([[2,0,0],[0,0,7],[0,0,0]]))
    # A literal one-card, one-macro-cell member exercises supported boundary
    # inputs outside the demonstration sampler, including an entirely blank
    # answer, an unused stencil, an orphan command, and structural-color ink.
    small = np.zeros((12,6),dtype=np.int8)
    small[[1,4,6],1:5] = 5
    small[1:7,[1,4]] = 5
    small[2:4,2:4] = [[1,0],[0,1]]
    small[5,2:4] = 7
    small[8:11,1:4] = [[5,1,5],[5,0,5],[5,1,5]]
    diagonal = np.array([[7,0],[0,7]],dtype=np.int8)
    assert np.array_equal(solve(small),diagonal)
    small[9,[1,3]] = 1
    assert not solve(small).any()  # Equal identities cancel every pixel.
    small[[8,10],2] = 9  # Unknown paired column is an empty stencil.
    assert np.array_equal(solve(small),diagonal)
    small[5,2:4] = 5
    assert np.array_equal(solve(small),np.where(diagonal,5,0))
    small[9,[1,3]] = 5
    assert not solve(small).any()
    good = generate_puzzle(0,2,1)
    bad = {s:[{name:a.copy() for name,a in pair.items()} for pair in good[s]] for s in good}
    bad["test"][0]["output"][0,0] = (int(bad["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    broken = good["test"][0]["input"].copy()
    q = _parse(broken)
    y,x,_,_ = q["board"]
    broken[y+1,x+1] = q["frame"]
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("nonempty board interior accepted")
    return dict(seeds=42,pairs=pairs,legal_count_calls=legal_calls,
        distinct_puzzles=len(signatures),shape_pairs=len(shapes),masks=len(motifs),
        axis_layouts=len(axis_layouts),palette_roles=len(roles),scales=sorted(ns),
        library_counts=sorted(ks),feature_pairs=dict(features),observed_shifts=dict(shifts))


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",", ":")))
