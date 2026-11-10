"""Self-addressed foundry: fit large keys, then cast their two-scale addresses.

Only the standard library and NumPy are used. Generation and tests do no I/O.
All search/retry budgets and memoization sizes are bounded.
"""
from functools import lru_cache
import hashlib
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


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
            for dr, dc in _N4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _normal(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _rotate(cells):
    return _normal(tuple((c, -r) for r, c in cells))


@lru_cache(maxsize=2048)
def _poses(cells, mirror=True):
    cells = _normal(cells)
    bases = [cells]
    if mirror:
        bases.append(_normal(tuple((r, -c) for r, c in cells)))
    result = set()
    for base in bases:
        for _ in range(4):
            result.add(base)
            base = _rotate(base)
    return tuple(sorted(result))


def _canonical(cells):
    return _poses(_normal(cells))[0]


def _mask(cells):
    cells = _normal(cells)
    a = np.zeros((1+max(r for r, c in cells), 1+max(c for r, c in cells)), dtype=bool)
    for r, c in cells:
        a[r, c] = True
    return a


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "grid must contain integers"
    assert np.all((a >= 0) & (a <= 9)), "invalid ARC color"
    return a.astype(np.int8, copy=True)


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells)+1,
            min(c for r, c in cells), max(c for r, c in cells)+1)


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and i == result[-1][-1]+1:
            result[-1].append(i)
        else:
            result.append([i])
    return result


def _lattice(field, divider):
    rg = _runs(np.flatnonzero(np.all(field == divider, axis=1)))
    cg = _runs(np.flatnonzero(np.all(field == divider, axis=0)))
    assert len(rg) >= 3 and len(cg) >= 3
    assert rg[0][0] == cg[0][0] == 0
    assert rg[-1][-1] == field.shape[0]-1 and cg[-1][-1] == field.shape[1]-1
    assert len({len(g) for g in rg+cg}) == 1
    t = len(rg[0])
    hs = {b[0]-a[-1]-1 for a, b in zip(rg, rg[1:])}
    ws = {b[0]-a[-1]-1 for a, b in zip(cg, cg[1:])}
    assert len(hs) == len(ws) == 1
    h, w = hs.pop(), ws.pop()
    assert 2 <= h <= 4 and 2 <= w <= 4 and 1 <= t <= 2
    rows, cols = len(rg)-1, len(cg)-1
    assert rows >= h and cols >= w
    return h, w, t, rows, cols


def _tile_slice(geometry, r, c):
    h, w, t, _, _ = geometry
    y, x = t+r*(h+t), t+c*(w+t)
    return np.s_[y:y+h, x:x+w]


def _render(geometry, divider, ordinary, holes, positions, flags=False):
    h, w, t, rows, cols = geometry
    field = np.full((rows*(h+t)+t, cols*(w+t)+t), divider, dtype=np.int8)
    tile_at = {}
    for color, sites in positions.items():
        tile = np.full((h, w), color, dtype=np.int8) if flags else ordinary.copy()
        if not flags:
            for r, c in holes[color]:
                tile[r, c] = color
        for p in sites:
            assert p not in tile_at, "two cast types overlap"
            tile_at[p] = tile
    for r in range(rows):
        for c in range(cols):
            field[_tile_slice(geometry, r, c)] = tile_at.get((r, c), ordinary)
    return field


def _origins(holes, visible, geometry):
    h, w, _, rows, cols = geometry
    sets = {color: set(cells) for color, cells in holes.items()}
    return [(r, c) for r in range(rows-h+1) for c in range(cols-w+1)
            if all((y-r, x-c) in sets[color]
                   for color, sites in visible.items() for y, x in sites)]


def _address_symmetries(holes, shape):
    layout = np.full(shape, -1, dtype=np.int8)
    for color, cells in holes.items():
        for p in cells:
            layout[p] = color
    def cropped_key(a):
        y, x = np.nonzero(a >= 0)
        return _key(a[y.min():y.max()+1, x.min():x.max()+1])
    target = cropped_key(layout)
    turns = [np.rot90(layout, k) for k in range(4)]
    return {i for i, a in enumerate(turns+[np.fliplr(a) for a in turns]) if cropped_key(a) == target}


def _decode(cells, k):
    r0, r1, c0, c1 = _bounds(cells)
    if (r1-r0) % k or (c1-c0) % k or len(cells) % (k*k):
        return None
    counts = {}
    for r, c in cells:
        p = ((r-r0)//k, (c-c0)//k)
        counts[p] = counts.get(p, 0)+1
    if any(n != k*k for n in counts.values()):
        return None
    return tuple(sorted(counts))


def _recover(input_grid):
    """Infer the field, keys, scale, color binding and common origin from pixels."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "blank exterior frame required"
    candidates = []
    for divider in map(int, np.unique(a)):
        if divider == bg:
            continue
        for comp in _components(a == divider):
            r0, r1, c0, c1 = _bounds(comp)
            field = a[r0:r1, c0:c1]
            try:
                geometry = _lattice(field, divider)
            except AssertionError:
                continue
            candidates.append((divider, (r0, r1, c0, c1), field.copy(), geometry))
    assert len(candidates) == 1, "exactly one complete divider lattice required"
    divider, bounds, field, geometry = candidates[0]
    h, w, t, rows, cols = geometry
    records = {}
    for r in range(rows):
        for c in range(cols):
            tile = field[_tile_slice(geometry, r, c)]
            assert divider not in tile
            records.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
    ordered = sorted(records.values(), key=lambda v: len(v[1]), reverse=True)
    ordinary, ordinary_sites = ordered[0]
    assert len(ordinary_sites) > rows*cols/2, "empty mould must have a strict majority"
    values = set(map(int, np.unique(ordinary)))
    assert bg in values and len(values) == 2
    structural = (values-{bg}).pop()
    cavities = _components(ordinary == bg)
    assert 2 <= len(cavities) <= 3
    classes = {_canonical(cells): cells for cells in cavities}
    assert len(classes) == len(cavities), "geometrically interchangeable cavity classes forbidden"
    visible = {}
    for tile, sites in ordered[1:]:
        assert len(np.unique(tile)) == 1, "seed flags must fill the entire tile interior"
        color = int(tile[0, 0])
        assert color not in values | {divider}
        visible[color] = tuple(sites)
    assert 1 <= len(visible) <= len(cavities), "at least one seed color required"
    outside = np.ones(a.shape, dtype=bool)
    r0, r1, c0, c1 = bounds
    outside[r0:r1, c0:c1] = False
    external = _components(outside & (a != bg))
    assert len(external) == len(cavities), "one separated prototype per cavity required"
    pieces = {}
    for comp in external:
        colors = {int(a[p]) for p in comp}
        assert len(colors) == 1
        color = colors.pop()
        assert color not in values | {divider} and color not in pieces
        pieces[color] = comp
    assert set(visible) <= set(pieces), "every flag color needs a prototype"
    solutions = []
    for k in range(2, 5):
        holes, decoded, used = {}, {}, set()
        for color, cells in pieces.items():
            shape = _decode(cells, k)
            if shape is None or _canonical(shape) not in classes:
                break
            key = _canonical(shape)
            if key in used:
                break
            used.add(key)
            holes[color], decoded[color] = classes[key], shape
        else:
            if len(used) == len(classes):
                origins = _origins(holes, visible, geometry)
                if len(origins) == 1:
                    solutions.append((k, holes, decoded, origins[0]))
    assert len(solutions) == 1, "unique scale, fitting assignment and joint registration required"
    k, holes, decoded, origin = solutions[0]
    complete = {color: tuple((r+origin[0], c+origin[1]) for r, c in cells)
                for color, cells in holes.items()}
    supplied = sum(map(len, visible.values()))
    active = sum(map(len, complete.values()))
    assert supplied < active, "at least one unflagged cast must be inferred"
    expected = _render(geometry, divider, ordinary, holes, complete)
    reflected = sum(_normal(holes[c]) not in _poses(decoded[c], False) for c in holes)
    rotated = sum(_normal(holes[c]) != decoded[c] and
                  _normal(holes[c]) in _poses(decoded[c], False) for c in holes)
    separate_origins = {c: len(_origins({c: holes[c]}, {c: visible[c]}, geometry)) for c in visible}
    info = dict(background=bg, divider=divider, structural=structural, bounds=bounds,
                geometry=geometry, scale=k, ordinary=ordinary, holes=holes,
                visible=visible, complete=complete, origin=origin, decoded=decoded,
                pieces=pieces, supplied=supplied, active=active, missing=active-supplied,
                classes=tuple(sorted(classes)), mould=_key(ordinary == bg),
                reflected=reflected, rotated=rotated, separate_origins=separate_origins,
                equal_area=len({len(c) for c in cavities}) < len(cavities))
    return expected, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver; never reads a seed, examples, files, or generated state."""
    return _recover(input_grid)[0]


def _sample_mould(rng, witness=None, count=None):
    shapes = ((2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3), (4, 4))
    if witness:
        shapes = ((3, 4), (4, 3), (4, 4))
    if count == 3:
        shapes = ((3, 4), (4, 3), (4, 4))
    shape = shapes[int(rng.integers(len(shapes)))]
    for _ in range(350):
        void = rng.random(shape) < rng.uniform(.35, .7)
        cells = _components(void)
        if len(cells) not in ((count,) if count else (2, 3)):
            continue
        keys = [_canonical(c) for c in cells]
        if len(set(keys)) != len(cells) or max(map(len, cells)) > 8:
            continue
        if witness == "area" and len({len(c) for c in cells}) == len(cells):
            continue
        if witness == "reflection" and not any(len(_poses(_normal(c))) > len(_poses(_normal(c), False)) for c in cells):
            continue
        return void
    # Finite feasible fallbacks; transform the entire mould together.
    if witness == "area":
        void = np.array([[1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=bool)
    elif count == 3:
        void = np.array([[1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0]], dtype=bool)
    elif witness == "reflection":
        void = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]], dtype=bool)
    else:
        void = np.array([[1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0]], dtype=bool)
    void = np.rot90(void, int(rng.integers(4)))
    return np.fliplr(void).copy() if rng.integers(2) else void.copy()


def _choose_visible(rng, holes, geometry, origin, sparse, witness):
    colors = list(holes)
    if witness == "area":
        # Hide both equal-area keys' coarse evidence. Swapping their bindings
        # preserves every flag and every area; only actual shape fitting rules
        # out that different completion.
        a, b = next((a, b) for i, a in enumerate(colors) for b in colors[i+1:]
                    if len(holes[a]) == len(holes[b]))
        colors = [c for c in colors if c not in (a, b)]
    elif witness != "reflection" and rng.random() < .35:
        colors = list(map(int, rng.choice(colors, int(rng.integers(1, len(colors)+1)), replace=False)))
    active = [(color, (r+origin[0], c+origin[1])) for color in colors for r, c in holes[color]]
    _, _, _, rows, cols = geometry
    n = len(colors)
    cap = min(len(active), sum(map(len, holes.values()))-1, (rows*cols-1)//2)
    if cap < n:
        return None
    sizes = list(range(n, cap+1))
    pivot = max(n, (len(active)+1)//2)
    if sparse or witness == "reflection":
        sizes = sorted(sizes, key=lambda v: (v, float(rng.random())))
    else:
        sizes = [s for s in sizes if s >= min(pivot, cap)]
        rng.shuffle(sizes)
    for size in sizes:
        for _ in range(24):
            # One clue for every color, then a random strict subset of the rest.
            chosen = {int(rng.choice([i for i, (c, p) in enumerate(active) if c == color])) for color in colors}
            unused = [i for i in range(len(active)) if i not in chosen]
            if size > n:
                chosen.update(map(int, rng.choice(unused, size-n, replace=False)))
            visible = {color: tuple(p for i, (c, p) in enumerate(active) if i in chosen and c == color) for color in colors}
            if _origins(holes, visible, geometry) == [origin]:
                return visible
    return None


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    s = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = s[h:, w:]-s[:-h, w:]-s[h:, :-w]+s[:-h, :-w]
    return np.argwhere(sums == 0)


def _pack(rng, field, prototypes, bg):
    objects = [(None, field)] + list(prototypes.items())
    for attempt in range(28):
        a = np.full((30, 30), bg, dtype=np.int8)
        blocked = np.zeros((30, 30), dtype=bool)
        blocked[[0, -1], :] = True
        blocked[:, [0, -1]] = True
        order = [0]+sorted(range(1, len(objects)), key=lambda i: -objects[i][1].size)
        for index in order:
            color, data = objects[index]
            h, w = data.shape
            positions = _free_origins(blocked, h, w)
            if not len(positions):
                break
            if index == 0 and attempt >= 12:
                corners = [(1, 1), (1, 29-w), (29-h, 1), (29-h, 29-w)]
                y, x = corners[int(rng.integers(4))]
            else:
                y, x = map(int, positions[int(rng.integers(len(positions)))])
            view = a[y:y+h, x:x+w]
            if color is None:
                view[:] = data
            else:
                view[data] = color
            blocked[max(0, y-1):y+h+1, max(0, x-1):x+w+1] = True
        else:
            r0, r1, c0, c1 = _bounds(tuple(map(tuple, np.argwhere(a != bg))))
            # A visible blank exterior frame defines the canvas background.
            r0 = max(0, r0-1-int(rng.integers(3)))
            c0 = max(0, c0-1-int(rng.integers(3)))
            r1 = min(30, r1+1+int(rng.integers(3)))
            c1 = min(30, c1+1+int(rng.integers(3)))
            return a[r0:r1, c0:c1].copy()
    return None


def _make_pair(rng, scale, witness, count, sparse, forbidden, thickness=None):
    for attempt in range(100):
        void = _sample_mould(rng, witness, count)
        if _key(void) in forbidden:
            continue
        h, w = void.shape
        # Thickness 2 is feasible only for smaller moulds; both splits use it.
        if thickness == 2 and max(h, w) > 3:
            continue
        t = thickness or (2 if max(h, w) <= 3 and rng.random() < .25 else 1)
        max_rows, max_cols = (27-t)//(h+t), (27-t)//(w+t)
        if max_rows < h or max_cols < w:
            continue
        rows = int(rng.integers(h, min(max_rows, h+2)+1))
        cols = int(rng.integers(w, min(max_cols, w+2)+1))
        geometry = h, w, t, rows, cols
        palette = list(map(int, rng.permutation(10)))
        bg, divider, structural = palette[:3]
        ordinary = np.where(void, bg, structural).astype(np.int8)
        cavities = _components(void)
        colors = palette[3:3+len(cavities)]
        holes = dict(zip(colors, cavities))
        if witness == "reflection" and _address_symmetries(holes, void.shape) != {0}:
            continue
        origin = int(rng.integers(rows-h+1)), int(rng.integers(cols-w+1))
        visible = _choose_visible(rng, holes, geometry, origin, sparse, witness)
        if visible is None:
            continue
        prototypes = {}
        reflection_done = False
        rotation_done = False
        for color, cells in holes.items():
            destination = _normal(cells)
            poses = _poses(destination)
            if witness == "reflection" and not reflection_done:
                mirrored = tuple(p for p in poses if p not in _poses(destination, False))
                if mirrored:
                    poses = mirrored
                    reflection_done = True
            if witness == "area" and not rotation_done:
                rotated = tuple(p for p in _poses(destination, False) if p != destination)
                if rotated:
                    poses = rotated
                    rotation_done = True
            pose = poses[int(rng.integers(len(poses)))]
            prototypes[color] = np.repeat(np.repeat(_mask(pose), scale, 0), scale, 1)
        inp_field = _render(geometry, divider, ordinary, holes, visible, flags=True)
        inp = _pack(rng, inp_field, prototypes, bg)
        if inp is None:
            continue
        complete = {color: tuple((r+origin[0], c+origin[1]) for r, c in cells) for color, cells in holes.items()}
        out = _render(geometry, divider, ordinary, holes, complete)
        recovered, info = _recover(inp)
        assert np.array_equal(recovered, out), "construction/recovery disagreement"
        return {"input": inp, "output": out}, info
    raise RuntimeError("bounded construction budget exhausted")


def _count(value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError("illegal example count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))
    base_scale = int(rng.choice((2, 3)))
    test_scale = int(rng.choice([k for k in (2, 3, 4) if k != base_scale]))
    puzzle, seen, training_moulds = {"train": [], "test": []}, set(), set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for i in range(count):
            witness = ("area" if i == 0 else "reflection" if i == 1 else None) if split == "train" else None
            scale = (base_scale if split == "train" else test_scale) if mode == 0 else int(rng.choice((2, 3)))
            cavity_count = 3 if witness == "area" else None
            sparse = split == "test" and mode == 1
            thickness = (1 if split == "train" else 2) if mode == 2 else None
            for _ in range(12):
                pair, info = _make_pair(rng, scale, witness, cavity_count, sparse,
                                        training_moulds if split == "test" or i == 1 else set(), thickness)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct-pair budget exhausted")
            seen.add(_key(pair["input"]))
            if split == "train":
                training_moulds.add(info["mould"])
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, info_by_split = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, f = _recover(inp)
            assert np.array_equal(out, expected), "incorrect cast/completion/crop"
            info_by_split[split].append(f)
            h, w, t, nr, nc = f["geometry"]
            assert out.shape == (nr*(h+t)+t, nc*(w+t)+t)
            assert np.array_equal(out == f["divider"], expected == f["divider"])
            # Each color's final support is the product of its coarse and fine
            # cavity supports: a cells at each of a destinations gives a^2.
            for color, cells in f["holes"].items():
                assert np.count_nonzero(out == color) == len(cells)**2
                assert len(f["pieces"][color]) == len(cells)*f["scale"]**2
                assert set(f["visible"].get(color, ())) <= set(f["complete"][color])
            only_flags = _render(f["geometry"], f["divider"], f["ordinary"], f["holes"], f["visible"])
            assert not np.array_equal(out, only_flags), "B completion must change the answer"
    train = info_by_split["train"]
    assert any(f["equal_area"] for f in train), "area alone must not identify keys"
    assert any(f["reflected"] for f in train), "D4 fitting needs a chiral witness"
    assert any(f["rotated"] for f in train), "rotation witness required"
    assert len({f["mould"] for f in train}) >= 2, "demonstrate multiple address maps"
    orientation_rules = set(range(8))
    for f in train:
        orientation_rules &= _address_symmetries(f["holes"], f["ordinary"].shape)
    assert orientation_rules == {0}, "demonstrations must distinguish relative D4 address rules"
    # An area-only assignment can swap equal-area noncongruent cavities. Its
    # resulting casts/addresses differ, even though its area counts agree.
    f = next(f for f in train if f["equal_area"] and len(f["visible"]) < len(f["holes"])-1)
    colors = list(f["holes"])
    a, b = next((a, b) for i, a in enumerate(colors) for b in colors[i+1:]
                if len(f["holes"][a]) == len(f["holes"][b]) and a not in f["visible"] and b not in f["visible"])
    swapped = dict(f["holes"])
    swapped[a], swapped[b] = swapped[b], swapped[a]
    assert _origins(swapped, f["visible"], f["geometry"]) == [f["origin"]]
    wrong_sites = {c: tuple((r+f["origin"][0], x+f["origin"][1]) for r, x in cells) for c, cells in swapped.items()}
    wrong = _render(f["geometry"], f["divider"], f["ordinary"], swapped, wrong_sites)
    right = _render(f["geometry"], f["divider"], f["ordinary"], f["holes"], f["complete"])
    assert not np.array_equal(wrong, right), "A shape binding must change the answer"
    return None


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, payload = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(payload)
    return digest.hexdigest()


def self_test():
    """42 seeds, bounded streaming diagnostics, semantic and negative checks."""
    signatures, moulds, geometries, classes, origins, scales = set(), set(), set(), set(), set(), set()
    shapes, count_sets = set(), set()
    reflected = rotated = joint_only = missing = 0
    ambiguous_case = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                predicted, f = _recover(np.asarray(pair["input"]).copy())
                assert np.array_equal(predicted, pair["output"])
                moulds.add(f["mould"])
                geometries.add(f["geometry"])
                classes.update(f["classes"])
                origins.add(f["origin"])
                scales.add(f["scale"])
                shapes.add(tuple(pair["input"].shape))
                count_sets.add(len(f["holes"]))
                reflected += bool(f["reflected"])
                rotated += bool(f["rotated"])
                joint_only += all(n > 1 for n in f["separate_origins"].values())
                if all(n > 1 for n in f["separate_origins"].values()):
                    ambiguous_case = (np.asarray(pair["input"]).copy(), f)
                missing += f["missing"]
    assert len(signatures) == 42 and len(moulds) >= 60
    assert len(geometries) >= 25 and len(classes) >= 10
    assert len(origins) >= 4 and len(shapes) >= 20
    assert scales == {2, 3, 4} and count_sets == {2, 3}
    assert reflected >= 42 and rotated >= 42 and missing >= 210
    assert joint_only > 0, "include registration that needs more than one color"
    p = generate_puzzle(0)
    assert _digest(p) == _digest(generate_puzzle(0))
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            q = generate_puzzle(8128+ntrain*10+ntest, ntrain, ntest)
            assert len(q["train"]) == ntrain and len(q["test"]) == ntest
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad output accepted")
    pair = p["train"][0]
    _, f = _recover(pair["input"])
    broken = np.asarray(pair["input"]).copy()
    c = next(iter(f["pieces"]))
    broken[f["pieces"][c][0]] = f["background"]
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken macro block accepted")
    ambiguous, f = ambiguous_case
    color = next(iter(f["visible"]))
    one_color = {color: f["visible"][color]}
    ambiguous_field = _render(f["geometry"], f["divider"], f["ordinary"], f["holes"], one_color, flags=True)
    r0, r1, c0, c1 = f["bounds"]
    ambiguous[r0:r1, c0:c1] = ambiguous_field
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous shared origin accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "mould_maps": len(moulds), "geometries": len(geometries),
            "shape_classes": len(classes), "input_shapes": len(shapes),
            "origins": len(origins), "scales": sorted(scales),
            "reflected_pairs": reflected, "rotated_pairs": rotated,
            "joint_only_registration_pairs": joint_only, "inferred_casts": missing}
