"""Color-frequency lookup puzzle generator."""
import numpy as np

_BASE = (
    np.array([[0,0,1],[1,1,0],[0,1,0]], dtype=bool),
    np.array([[0,0,0],[1,1,1],[0,0,0]], dtype=bool),
    np.array([[0,1,1],[0,1,0],[0,1,0]], dtype=bool),
)
_SOURCE_ROWS = (
    ("1010700","1191010","0011020","0000301","0401001","0010208","0010731"),
    ("0303530","0033000","8000003","3439303","0093133","0333030","0000003"),
    ("0020153","0029020","2224200","0202720","2200226","0220200","5042022"),
    ("2002205","0220002","0100000","0000209","0900002","0021008","2002200"),
    ("0400413","3343037","3000103","0030300","3003303","3030303","3330423"),
    ("0002202","0229220","0502460","2000092","0002200","8029063","0202024"),
    ("0410016","0010000","1100110","0100011","0010020","1010107","1110410"),
    ("3030003","3095005","0303029","8303007","0350033","0033000","0030400"),
    ("0081101","5110110","0101001","1020060","6011500","0030005","0100201"),
)
_SOURCE = {tuple(int(x) for row in rows for x in row) for rows in _SOURCE_ROWS}


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not (2 <= value <= 10 if name == "num_train" else 1 <= value <= 4):
        raise ValueError(f"illegal {name}")
    return value


def _resolve_counts(rng, num_train, num_test):
    nt, ne = _count(num_train, "num_train"), _count(num_test, "num_test")
    if nt is None and ne is None:
        ne = int(rng.integers(1, 5))
    if nt is None:
        nt = int(rng.integers(max(2, ne + 1), 11))
    if ne is None:
        ne = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    return nt, ne


def _conventions(rng):
    bg = 0 if rng.random() < .65 else int(rng.choice([4,5,6,7]))
    ink = 8 if rng.random() < .55 else int(rng.choice([c for c in (4,5,6,7,8,9) if c != bg]))
    masks = [mask.copy() for mask in _BASE]
    rng.shuffle(masks)
    if rng.random() < .5:
        for index in rng.choice(3, size=int(rng.integers(1,4)), replace=False):
            for _ in range(32):
                cells = rng.choice(9, size=int(rng.integers(3,7)), replace=False)
                mask = np.zeros(9, dtype=bool)
                mask[cells] = True
                mask = mask.reshape(3,3)
                if not any(np.array_equal(mask, other) for other in masks):
                    masks[int(index)] = mask
                    break
    glyphs = {label: np.where(masks[label-1], ink, bg).astype(np.int8)
              for label in (1,2,3)}
    return bg, ink, glyphs


def _labels(rng, nt, ne):
    labels = [1,2,3]
    rng.shuffle(labels)
    taught = labels[:2] if nt == 2 else labels
    train = taught.copy()
    while len(train) < nt:
        train.append(int(rng.choice(taught)))
    test_order = taught.copy()
    rng.shuffle(test_order)
    test = test_order[:ne]
    while len(test) < ne:
        test.append(int(rng.choice(taught)))
    rng.shuffle(test)
    return train, test


def _histogram(rng, regime, winner, bg, forced_runner=None, forced_margin=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    for _ in range(96):
        wc = int(rng.integers(9,25))
        if forced_margin is None:
            probabilities = (.22,.35,.43) if regime == "source_train" else (.62,.23,.15)
            band = int(rng.choice(3, p=probabilities))
            lo, hi = ((1,2),(3,7),(8,18))[band]
            margin = int(rng.integers(lo, min(hi, wc-1)+1))
        else:
            margin = forced_margin
        if forced_runner is None:
            other_labels = [c for c in (1,2,3) if c != winner]
            choices = [c for c in range(10) if c not in (winner,bg)]
            runner = int(rng.choice(other_labels if rng.random() < .35 else choices))
        else:
            runner = forced_runner
        rc = wc-margin
        choices = [c for c in range(10) if c not in (winner,bg,runner)]
        rng.shuffle(choices)
        hist = {winner:wc, runner:rc}
        for color in choices[:int(rng.integers(0,6))]:
            hist[color] = int(rng.integers(1, min(5,rc)+1))
        if sum(hist.values()) <= 69:
            return hist
    raise RuntimeError("histogram sampling failed")


def _shape(rng, total, banned=()):
    density = int(rng.choice(3, p=[.31,.49,.20]))
    lo, hi = ((.26,.46),(.45,.68),(.67,.84))[density]
    def choices(low, high):
        return [(h,w) for h in range(5,13) for w in range(5,13)
                if (h,w) not in banned and h*w-total >= 8
                and low <= total/(h*w) <= high]
    options = choices(lo,hi) or choices(.25,.84)
    if not options:
        raise RuntimeError("shape sampling failed")
    weights = np.array([3. if shape == (7,7) else 1. for shape in options])
    weights /= weights.sum()
    return options[int(rng.choice(len(options), p=weights))]


def _arrange(rng, hist, shape, bg):
    h,w = shape
    area = h*w
    flat = np.full(area, bg, dtype=np.int8)
    corners = {0,w-1,(h-1)*w,area-1}
    reserved = set(corners)
    if bg != 0:
        border = [r*w+c for r in range(h) for c in range(w)
                  if r in (0,h-1) or c in (0,w-1)]
        extras = [x for x in border if x not in corners]
        reserved.update(int(x) for x in rng.choice(extras, size=4, replace=False))
    available = np.array([x for x in range(area) if x not in reserved])
    assert sum(hist.values()) <= len(available)
    style = int(rng.integers(3))
    if style == 0:
        values = [c for c,n in hist.items() for _ in range(n)]
        values += [bg]*(len(available)-len(values))
        rng.shuffle(values)
        flat[available] = values
    elif style == 1:
        colors = list(hist)
        rng.shuffle(colors)
        for color in colors:
            cr,cc = int(rng.integers(h)), int(rng.integers(w))
            rows,cols = np.divmod(available,w)
            score = abs(rows-cr)+abs(cols-cc)+rng.random(len(available))*1.5
            selected = np.argsort(score)[:hist[color]]
            flat[available[selected]] = color
            available = np.delete(available,selected)
    else:
        winner = max(hist,key=hist.get)
        rows,cols = np.divmod(available,w)
        parity = int(rng.integers(2))
        score = ((rows+cols)%2 != parity).astype(float)*2+rng.random(len(available))
        selected = np.argsort(score)[:hist[winner]]
        flat[available[selected]] = winner
        remaining = np.delete(available,selected)
        values = [c for c,n in hist.items() if c != winner for _ in range(n)]
        values += [bg]*(len(remaining)-len(values))
        rng.shuffle(values)
        flat[remaining] = values
    return flat.reshape(shape)


def _counts(grid,bg):
    counts = np.bincount(grid.ravel(),minlength=10)
    counts[bg] = 0
    return counts


def _winner(grid,bg):
    counts = _counts(grid,bg)
    maximum = int(counts.max())
    winners = np.flatnonzero(counts == maximum)
    assert len(winners) == 1 and maximum > 0
    winner = int(winners[0])
    counts[winner] = 0
    return winner, maximum-int(counts.max())


def _pair(rng, regime, winner, bg, glyphs, banned=(), runner=None, margin=None):
    hist = _histogram(rng,regime,winner,bg,runner,margin)
    shape = _shape(rng,sum(hist.values()),banned)
    return {"input":_arrange(rng,hist,shape,bg), "output":glyphs[winner].copy()}


def _signature(grid):
    return grid.shape, tuple(int(x) for x in grid.ravel())


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    bg, ink, glyphs = _conventions(rng)
    nt, ne = _resolve_counts(rng,num_train,num_test)
    train_labels,test_labels = _labels(rng,nt,ne)
    pair_seeds = rng.integers(0,np.iinfo(np.uint64).max,size=nt+ne+4,dtype=np.uint64)
    pair_rngs = [np.random.default_rng(x) for x in pair_seeds]
    train = []
    base_count = len(set(train_labels))
    for index in range(base_count):
        banned = [pair["input"].shape for pair in train]
        train.append(_pair(pair_rngs[index],"source_train",train_labels[index],bg,glyphs,banned,
                           train_labels[1] if index == 0 and nt >= 5 else None,
                           1 if index == 0 and nt >= 5 else None))
    next_rng = base_count
    if nt >= 4:
        source = train[2]
        counts = _counts(source["input"],bg)
        hist = {c:int(n) for c,n in enumerate(counts) if n > 0}
        contrasted = _arrange(pair_rngs[next_rng],hist,source["input"].shape,bg)
        train.append({"input":contrasted,"output":source["output"].copy()})
        next_rng += 1
    if nt >= 5:
        source = train[0]
        flipped = source["input"].copy()
        positions = np.argwhere(flipped == train_labels[0])
        position = positions[int(pair_rngs[next_rng].integers(len(positions)))]
        flipped[tuple(position)] = train_labels[1]
        train.append({"input":flipped,"output":glyphs[train_labels[1]].copy()})
        next_rng += 1
    while len(train) < nt:
        label = train_labels[len(train)]
        train.append(_pair(pair_rngs[next_rng],"source_train",label,bg,glyphs))
        next_rng += 1
    test = []
    for label in test_labels:
        test.append(_pair(pair_rngs[next_rng],"target_test",label,bg,glyphs))
        next_rng += 1
    puzzle = {"train":train,"test":test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _background(grid):
    corners = [grid[0,0],grid[0,-1],grid[-1,0],grid[-1,-1]]
    assert len(set(int(x) for x in corners)) == 1
    return int(corners[0])


def _learn(train):
    bg = _background(train[0]["input"])
    ink = None
    codebook = {}
    for pair in train:
        grid,output = pair["input"],pair["output"]
        assert _background(grid) == bg
        label,_ = _winner(grid,bg)
        assert label in (1,2,3)
        colors = set(int(x) for x in np.unique(output))
        assert bg in colors and len(colors) == 2
        pair_ink = next(iter(colors-{bg}))
        assert ink is None or ink == pair_ink
        ink = pair_ink
        if label in codebook:
            assert np.array_equal(codebook[label],output)
        else:
            codebook[label] = output.copy()
    assert len({_signature(out) for out in codebook.values()}) == len(codebook)
    return bg,ink,codebook


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert isinstance(puzzle["train"],list) and isinstance(puzzle["test"],list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            for grid in pair.values():
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype,np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            grid,out = pair["input"],pair["output"]
            assert 5 <= grid.shape[0] <= 12 and 5 <= grid.shape[1] <= 12
            assert out.shape == (3,3)
            sig = _signature(grid)
            assert sig not in seen
            seen.add(sig)
            assert not (grid.shape == (7,7) and sig[1] in _SOURCE)
            bg = _background(grid)
            assert bg in (0,4,5,6,7)
            assert np.count_nonzero(grid == bg) >= 8
            label,margin = _winner(grid,bg)
            assert label in (1,2,3) and margin >= 1
            assert np.count_nonzero(out == bg) >= 3
    bg,ink,codebook = _learn(puzzle["train"])
    assert len(codebook) == (2 if len(puzzle["train"]) == 2 else 3)
    for pair in puzzle["test"]:
        grid,out = pair["input"],pair["output"]
        assert _background(grid) == bg
        label,_ = _winner(grid,bg)
        assert label in codebook and np.array_equal(out,codebook[label])
        assert set(int(x) for x in np.unique(out)) == {bg,ink}
    if len(puzzle["train"]) >= 4:
        a,b = puzzle["train"][2:4]
        assert a["input"].shape == b["input"].shape
        assert np.array_equal(_counts(a["input"],bg),_counts(b["input"],bg))
        assert not np.array_equal(a["input"],b["input"])
        assert np.array_equal(a["output"],b["output"])
    if len(puzzle["train"]) >= 5:
        a,b = puzzle["train"][0],puzzle["train"][4]
        assert a["input"].shape == b["input"].shape
        assert np.count_nonzero(a["input"] != b["input"]) == 1
        assert not np.array_equal(a["output"],b["output"])
    assert len({p["input"].shape for p in puzzle["train"][:3]}) == min(3,len(puzzle["train"]))


def _equal(a,b):
    return all(len(a[s]) == len(b[s]) and
               all(np.array_equal(x[k],y[k]) for x,y in zip(a[s],b[s])
                   for k in ("input","output")) for s in ("train","test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes,palettes,layouts,codebooks = set(),set(),set(),set()
    tm,em,tw = set(),set(),set()
    bgs,inks = set(),set()
    near,clear,no8,label_runner = 0,0,0,0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        bg,ink,codebook = _learn(puzzle["train"])
        codebooks.add(tuple((c,_signature(g)) for c,g in sorted(codebook.items())))
        bgs.add(bg); inks.add(ink)
        for split in ("train","test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                winner,margin = _winner(grid,bg)
                assert np.array_equal(pair["output"],codebook[winner])
                shapes.add(grid.shape)
                palettes.add(tuple(int(c) for c in np.unique(grid)))
                layouts.add((grid.shape,int(np.count_nonzero(grid != bg)),
                             int(np.count_nonzero(grid[:,1:] == grid[:,:-1]))))
                if split == "train":
                    tm.add(margin)
                else:
                    em.add(margin); tw.add(winner)
                    near += margin <= 2; clear += margin >= 8
                    counts = _counts(grid,bg)
                    no8 += counts[8] == 0
                    runner = max(int(counts[c]) for c in range(10) if c != winner)
                    label_runner += any(counts[c] == runner for c in (1,2,3) if c != winner)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len(shapes) >= 25 and any(h != w for h,w in shapes)
    assert len(palettes) >= 60 and len(layouts) >= 100
    assert len(codebooks) >= 20 and len(bgs) >= 3 and len(inks) >= 3
    assert {1,2,3} <= tw
    assert min(tm) == 1 and max(tm) >= 10
    assert min(em) == 1 and max(em) >= 8
    assert near >= 15 and clear >= 5 and no8 >= 10 and label_runner >= 8
    assert _equal(generate_puzzle(812345,8,3),generate_puzzle(812345,8,3))
    assert _equal(generate_puzzle(17),generate_puzzle(17))
    assert not _equal(generate_puzzle(17),generate_puzzle(18))
    for i,(nt,ne) in enumerate(((2,1),(10,1),(10,4),(5,2),(7,3))):
        p = generate_puzzle(9000+i,nt,ne)
        assert (len(p["train"]),len(p["test"])) == (nt,ne)
    for i,args in enumerate(({"num_train":None,"num_test":1},
                             {"num_train":None,"num_test":4},
                             {"num_train":2,"num_test":None},
                             {"num_train":10,"num_test":None})):
        p = generate_puzzle(9100+i,**args)
        for name,split in (("num_train","train"),("num_test","test")):
            if args[name] is not None:
                assert len(p[split]) == args[name]
    for args in ({"num_train":1},{"num_train":11},{"num_test":0},
                 {"num_test":5},{"num_train":2,"num_test":2},
                 {"num_train":3,"num_test":4},
                 {"num_train":True,"num_test":1}):
        try:
            generate_puzzle(123,**args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
