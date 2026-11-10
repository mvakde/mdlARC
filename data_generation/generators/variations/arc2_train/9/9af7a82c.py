"""Generate solved ARC color-frequency histogram puzzles."""
from functools import lru_cache
import numpy as np

_SOURCE_INPUTS = (
    ((8,8,2),(3,8,8),(3,3,4),(3,3,4)),
    ((3,1,1,4),(2,2,2,4),(4,4,4,4)),
    ((1,1,1),(2,2,1),(2,8,1),(2,8,1)),
    ((2,2,1),(2,3,1),(1,1,1)),
    ((8,8,2,2),(1,8,8,2),(1,3,3,4),(1,1,1,1)),
)
_SOURCE_KEYS = frozenset((len(rows), len(rows[0]), tuple(v for row in rows for v in row)) for rows in _SOURCE_INPUTS)
_TRAIN_SHAPES = tuple((h,w) for h in range(3,7) for w in range(3,7) if h*w <= 30)
_TEST_SHAPES = tuple((h,w) for h in range(3,8) for w in range(3,8) if h*w <= 42)
_FAMILIAR = ((3,3),(3,4),(4,3),(4,4))


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _count(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return int(value)


def _pair_counts(seed, num_train, num_test):
    nt = _count("num_train", num_train, 2, 10)
    ns = _count("num_test", num_test, 1, 4)
    if nt is None and ns is None:
        ns = 1 + seed % 4
        nt = ns + 1 + (seed // 4) % (10 - ns)
    elif nt is None:
        nt = ns + 1 + seed % (10 - ns)
    elif ns is None:
        ns = 1 + seed % min(4, nt - 1)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _key(grid):
    return (int(grid.shape[0]), int(grid.shape[1]), tuple(int(v) for v in grid.flat))


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train","test"))


@lru_cache(maxsize=None)
def _profiles(area, k):
    found = []
    def extend(left, slots, ceiling, prefix):
        if slots == 0:
            if left == 0:
                found.append(prefix)
            return
        for value in range(min(ceiling-1,left), slots-1, -1):
            if left-value >= slots*(slots-1)//2:
                extend(left-value, slots-1, value, prefix+(value,))
    extend(area, k, 31, ())
    return tuple(found)


def _output(colors, counts, padding):
    order = sorted(range(len(colors)), key=lambda i: -counts[i])
    result = np.full((counts[order[0]], len(colors)), padding, dtype=np.int8)
    for j, i in enumerate(order):
        result[:counts[i], j] = colors[i]
    return result


def _arrange(rng, colors, counts, shape, mode):
    h,w = shape
    if mode == "scatter":
        flat = np.repeat(np.asarray(colors, dtype=np.int8), counts)
        rng.shuffle(flat)
        return flat.reshape(shape).copy()
    if mode in ("patches","stripes"):
        order = rng.permutation(len(colors))
        flat = np.concatenate([np.full(counts[int(i)], colors[int(i)], dtype=np.int8) for i in order])
        grid = flat.reshape(shape) if mode == "patches" else flat.reshape((w,h)).T
        if bool(rng.integers(2)):
            grid = grid[::-1,:]
        if bool(rng.integers(2)):
            grid = grid[:,::-1]
        return grid.copy()
    # A compact lesser-color patch and a fragmented dominant color.
    grid = np.full(shape, -1, dtype=np.int8)
    dominant = int(np.argmax(counts))
    patch = int(np.argsort(counts)[-2])
    anchor = (int(rng.integers(h)), int(rng.integers(w)))
    cells = [(r,c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    cells.sort(key=lambda rc: abs(rc[0]-anchor[0])+abs(rc[1]-anchor[1]))
    for rc in cells[:counts[patch]]:
        grid[rc] = colors[patch]
    empty = [rc for rc in cells if grid[rc] == -1]
    rng.shuffle(empty)
    parity = int(rng.integers(2))
    empty.sort(key=lambda rc: (rc[0]+rc[1])%2 != parity)
    for rc in empty[:counts[dominant]]:
        grid[rc] = colors[dominant]
    remaining = [rc for rc in cells if grid[rc] == -1]
    rng.shuffle(remaining)
    others = np.concatenate([np.full(counts[i], colors[i], dtype=np.int8) for i in range(len(colors)) if i not in (patch,dominant)])
    rng.shuffle(others)
    for rc,v in zip(remaining, others):
        grid[rc] = v
    assert not np.any(grid == -1)
    return grid


def _make_pair(rng, available, padding, shapes, ks, mode=None, required=None,
               excluded_area=None, excluded_height=None, no_singleton=False, unseen_heights=None):
    choices = []
    for k in ks:
        if k > len(available):
            continue
        for shape in shapes:
            area = shape[0]*shape[1]
            if area == excluded_area:
                continue
            profiles = tuple(p for p in _profiles(area,k)
                             if (excluded_height is None or p[0] != excluded_height)
                             and (not no_singleton or p[-1] >= 2)
                             and (unseen_heights is None or p[0] not in unseen_heights))
            if profiles:
                choices.append((k,shape,profiles))
    if not choices:
        raise ValueError("no feasible profile")
    k,shape,profiles = choices[int(rng.integers(len(choices)))]
    profile = profiles[int(rng.integers(len(profiles)))]
    if required is None:
        chosen = rng.choice(available, size=k, replace=False)
    else:
        rest = tuple(c for c in available if c != required)
        chosen = np.concatenate((np.asarray([required]), rng.choice(rest, size=k-1, replace=False)))
    colors = tuple(int(c) for c in chosen)
    counts = tuple(int(c) for c in np.asarray(profile)[rng.permutation(k)])
    mode = mode or ("scatter","patches","stripes","mixed")[int(rng.integers(4))]
    return {"input": _arrange(rng, colors, counts, shape, mode),
            "output": _output(colors, counts, padding)}


def _profile(pair):
    return tuple(sorted((int(c) for c in np.unique(pair["input"],return_counts=True)[1]), reverse=True))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _seed(seed)
    nt,ns = _pair_counts(seed,num_train,num_test)
    rng = np.random.default_rng(seed)
    padding = int(rng.integers(10))
    categories = tuple(c for c in range(10) if c != padding)
    # Only padding is a learned puzzle-level color role. A held-out category
    # optionally tests whether colors are copied rather than memorized.
    novel = int(rng.choice(tuple(c for c in categories if c != 0))) if bool(rng.integers(2)) else None
    train_pool = tuple(c for c in categories if c != novel)
    seen_inputs, seen_outputs = set(),set()
    train,test = [],[]

    def add(pair, section, repeat_output=False):
        ik,ok = _key(pair["input"]),_key(pair["output"])
        if ik in seen_inputs or ik in _SOURCE_KEYS or (ok in seen_outputs and not repeat_output):
            return False
        seen_inputs.add(ik)
        seen_outputs.add(ok)
        section.append(pair)
        return True

    k1 = int(rng.integers(3,6))
    for _ in range(64):
        first = _make_pair(rng,train_pool,padding,_TRAIN_SHAPES,(k1,),
                           mode="patches" if nt >= 3 else None,
                           required=0 if padding != 0 else None)
        if add(first,train):
            break
    else:
        raise RuntimeError("first train construction failed")
    ks2 = tuple(k for k in (3,4,5) if k != k1)
    for _ in range(64):
        second = _make_pair(rng,train_pool,padding,_TRAIN_SHAPES,ks2,
                            excluded_area=first["input"].size,
                            excluded_height=first["output"].shape[0])
        if add(second,train):
            break
    else:
        raise RuntimeError("second train construction failed")
    if nt >= 3:
        colors,counts = np.unique(first["input"],return_counts=True)
        for _ in range(64):
            matched = {"input": _arrange(rng, tuple(int(c) for c in colors),
                                          tuple(int(c) for c in counts),
                                          first["input"].shape, "scatter"),
                       "output": first["output"].copy()}
            if add(matched,train,repeat_output=True):
                break
        else:
            raise RuntimeError("matched demonstration failed")
    while len(train) < nt:
        for _ in range(96):
            if add(_make_pair(rng,train_pool,padding,_TRAIN_SHAPES,(3,4,5,6)),train):
                break
        else:
            raise RuntimeError("train construction failed")

    train_colors = tuple(sorted({int(v) for p in train for v in np.unique(p["input"])}))
    heights = {p["output"].shape[0] for p in train}
    configs = {(tuple(p["input"].shape),_profile(p)) for p in train}
    test_pool = train_colors + ((novel,) if novel is not None else ())
    max_width = max(p["output"].shape[1] for p in train)
    while len(test) < ns:
        required = novel if novel is not None and len(test) == 0 else None
        familiar = bool(rng.random() < .27)
        no_singleton = bool(rng.random() < .37)
        new_height = bool(rng.random() < .40)
        wider = bool(rng.random() < .43)
        made = False
        for relaxed in (False,True):
            shapes = _FAMILIAR if familiar and not relaxed else _TEST_SHAPES
            ks = tuple(k for k in range(3,min(7,len(test_pool))+1)
                       if relaxed or not wider or k > max_width)
            if not ks:
                continue
            for _ in range(96):
                try:
                    pair = _make_pair(rng,test_pool,padding,shapes,ks,required=required,
                                      no_singleton=no_singleton and not relaxed,
                                      unseen_heights=heights if new_height and not relaxed else None)
                except ValueError:
                    break
                if (tuple(pair["input"].shape),_profile(pair)) in configs or not add(pair,test):
                    continue
                made = True
                break
            if made:
                break
        if not made:
            raise RuntimeError("test construction failed")
    puzzle = {"train":train,"test":test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    train,test = puzzle["train"],puzzle["test"]
    assert isinstance(train,list) and isinstance(test,list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs,outputs,paddings = set(),set(),set()
    for section,pairs in (("train",train),("test",test)):
        for pair in pairs:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            x,y = pair["input"],pair["output"]
            for grid in (x,y):
                assert isinstance(grid,np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype,np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert _key(x) not in inputs and _key(x) not in _SOURCE_KEYS
            inputs.add(_key(x))
            colors,raw_counts = np.unique(x,return_counts=True)
            counts = tuple(int(v) for v in raw_counts)
            assert 3 <= len(colors) <= (6 if section == "train" else 7)
            assert len(set(counts)) == len(counts)
            padset = set(int(v) for v in np.unique(y))-set(int(v) for v in colors)
            assert len(padset) == 1
            pad = padset.pop()
            paddings.add(pad)
            ordered = sorted(zip((int(c) for c in colors),counts),key=lambda item: -item[1])
            assert y.shape == (ordered[0][1],len(ordered))
            for j,(c,n) in enumerate(ordered):
                assert np.all(y[:n,j] == c)
                assert np.all(y[n:,j] == pad)
            assert not np.array_equal(x,y)
            assert tuple(x.shape) in (_TRAIN_SHAPES if section == "train" else _TEST_SHAPES)
            if section == "test":
                assert _key(y) not in outputs
            outputs.add(_key(y))
    assert len(paddings) == 1
    padding = next(iter(paddings))
    if padding != 0:
        assert any(np.any(p["input"] == 0) for p in train)
    assert train[0]["output"].shape[1] != train[1]["output"].shape[1]
    assert train[0]["output"].shape[0] != train[1]["output"].shape[0]
    assert train[0]["input"].size != train[1]["input"].size
    if len(train) >= 3:
        assert any(np.array_equal(a["output"],b["output"]) and not np.array_equal(a["input"],b["input"])
                   for i,a in enumerate(train) for b in train[i+1:])
    configs = {(tuple(p["input"].shape),_profile(p)) for p in train}
    assert all((tuple(p["input"].shape),_profile(p)) not in configs for p in test)


def _components(grid,color):
    cells = {(r,c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r,c] == color}
    count = 0
    while cells:
        count += 1
        frontier = [cells.pop()]
        while frontier:
            r,c = frontier.pop()
            for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if neighbor in cells:
                    cells.remove(neighbor)
                    frontier.append(neighbor)
    return count


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _signature(generate_puzzle(123)) == _signature(generate_puzzle(123))
    assert _signature(generate_puzzle(123)) != _signature(generate_puzzle(124))
    shapes,profiles,layouts,paddings,test_shapes = set(),set(),set(),set(),set()
    novel_tests = no_singleton_tests = new_height_tests = familiar_tests = fragmented = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_colors = {int(v) for p in puzzle["train"] for v in np.unique(p["input"])}
        train_heights = {p["output"].shape[0] for p in puzzle["train"]}
        for section in ("train","test"):
            for pair in puzzle[section]:
                x,y = pair["input"],pair["output"]
                shapes.add(tuple(x.shape))
                profiles.add(_profile(pair))
                labels = {}
                layouts.add((tuple(x.shape),tuple(labels.setdefault(int(v),len(labels)) for v in x.flat)))
                ordered = sorted(((int(c),int(n)) for c,n in zip(*np.unique(x,return_counts=True))),
                                 key=lambda item: -item[1])
                pad = (set(int(v) for v in np.unique(y))-{c for c,_ in ordered}).pop()
                paddings.add(pad)
                assert np.array_equal(y,_output(tuple(c for c,_ in ordered),
                                                tuple(n for _,n in ordered),pad))
                if _components(x,ordered[0][0]) >= 3 and any(
                    _components(x,c) == 1 and n >= 3 for c,n in ordered[1:]):
                    fragmented += 1
                if section == "test":
                    test_shapes.add(tuple(x.shape))
                    novel_tests += bool(set(int(v) for v in np.unique(x))-train_colors)
                    no_singleton_tests += _profile(pair)[-1] >= 2
                    new_height_tests += y.shape[0] not in train_heights
                    familiar_tests += tuple(x.shape) in _FAMILIAR
    assert len(shapes) >= 15 and len(test_shapes) >= 12
    assert len(profiles) >= 35 and len(layouts) >= 100
    assert len(paddings) >= 5
    assert novel_tests >= 8 and no_singleton_tests >= 8
    assert new_height_tests >= 8 and familiar_tests >= 5
    assert fragmented >= 5
    for seed,nt,ns in ((1000,2,1),(1001,10,1),(1002,10,4),
                       (2000,None,1),(2001,None,4),(2002,2,None),(2003,10,None)):
        p = generate_puzzle(seed,nt,ns)
        assert (nt is None or len(p["train"]) == nt) and (ns is None or len(p["test"]) == ns)
        validate_generated_puzzle(p)
    for request in ({"num_train":1},{"num_train":11},{"num_test":0},{"num_test":5},
                    {"num_train":2,"num_test":2},{"num_train":3,"num_test":4},
                    {"num_train":True},{"num_test":1.5}):
        try:
            generate_puzzle(0,**request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {request}")
