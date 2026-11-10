"""Generate puzzles that count sparse markers inside rectangular regions."""

import hashlib
import numpy as np


_SOURCE_INPUT_DIGESTS = frozenset({
    "1f3871a3a4291a7a31a4ffe60ba973036b77d71b0b4062888c5b5e4b47758728",
    "96403b0f138c8a897c8aace42a35b717a290a3f0dac3ff81d9e5975cf72f65a7",
    "ca79857dd97eea6bf90d0a38ca62740bb3bd5691babf32932f3aabb9453d870c",
    "5af5fbb807eb3b96caa5e068778d30ccb4f3abee0b29875f65f4a94b7506c0c1",
    "a2d3214619f88a5f73622e6d18e0ea1db37c519abe1de6c15bd372b1471847a6",
})


def _checked_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _lengths(rng, extent, n, minimum=5):
    if extent < n * minimum:
        raise ValueError("axis too short")
    if n == 1:
        return [extent]
    spare = extent - n * minimum
    dividers = sorted(int(x) for x in rng.choice(spare + n - 1, n - 1, replace=False))
    gaps = np.diff([-1] + dividers + [spare + n - 1]) - 1
    return [minimum + int(x) for x in gaps]


def _bands(rng, h, w, n, horizontal):
    edges = np.cumsum([0] + _lengths(rng, h if horizontal else w, n))
    if horizontal:
        return [(int(edges[i]), int(edges[i+1]), 0, w) for i in range(n)]
    return [(0, h, int(edges[i]), int(edges[i+1])) for i in range(n)]


def _grid(rng, h, w, rows, cols):
    re = np.cumsum([0] + _lengths(rng, h, rows))
    ce = np.cumsum([0] + _lengths(rng, w, cols))
    return [(int(re[i]), int(re[i+1]), int(ce[j]), int(ce[j+1]))
            for i in range(rows) for j in range(cols)]


def _rotate_boxes(boxes, h, w, turns):
    if turns == 0:
        return boxes, h, w
    result = []
    for r0, r1, c0, c1 in boxes:
        if turns == 1:
            result.append((w-c1, w-c0, r0, r1))
        elif turns == 2:
            result.append((h-r1, h-r0, w-c1, w-c0))
        else:
            result.append((c0, c1, h-r1, h-r0))
    return result, (w if turns % 2 else h), (h if turns % 2 else w)


def _layout(rng, mode):
    if mode == "bands2":
        n, min_h, min_w = 2, 10, 8
    elif mode == "bands3":
        n, min_h, min_w = 3, 15, 8
    elif mode == "bands4":
        n, min_h, min_w = 4, 20, 8
    elif mode == "bands5":
        n, min_h, min_w = 5, 25, 8
    elif mode == "grid4":
        n, min_h, min_w = 4, 10, 10
    elif mode == "grid6":
        n, min_h, min_w = 6, 15, 10
    elif mode == "t3":
        n, min_h, min_w = 3, 10, 10
    elif mode == "mixed4":
        n, min_h, min_w = 4, 15, 10
    else:
        n, min_h, min_w = 5, 15, 15
    if mode.startswith("bands") and bool(rng.integers(0, 2)):
        min_h, min_w = min_w, min_h
        horizontal = False
    else:
        horizontal = True
    if min_h <= 15 and min_w <= 15 and rng.random() < 0.28:
        h = int(rng.integers(min_h, min(30, min_h + 5) + 1))
        w = int(rng.integers(min_w, min(30, min_w + 5) + 1))
    else:
        h = int(rng.integers(min_h, 31))
        w = int(rng.integers(min_w, 31))
    if mode.startswith("bands"):
        return _bands(rng, h, w, n, horizontal), h, w
    if mode == "grid4":
        return _grid(rng, h, w, 2, 2), h, w
    if mode == "grid6":
        if bool(rng.integers(0, 2)):
            return _grid(rng, h, w, 3, 2), h, w
        if h < 10 or w < 15:
            h, w = w, h
        return _grid(rng, h, w, 2, 3), h, w
    if mode == "t3":
        row = _lengths(rng, h, 2)[0]
        col = _lengths(rng, w, 2)[0]
        boxes = [(0, row, 0, w), (row, h, 0, col), (row, h, col, w)]
    elif mode == "mixed4":
        rows = np.cumsum([0] + _lengths(rng, h, 3))
        col = _lengths(rng, w, 2)[0]
        boxes = [(0, int(rows[1]), 0, w),
                 (int(rows[1]), int(rows[2]), 0, col),
                 (int(rows[2]), h, 0, col),
                 (int(rows[1]), h, col, w)]
    else:
        rows = np.cumsum([0] + _lengths(rng, h, 3))
        cols = np.cumsum([0] + _lengths(rng, w, 3))
        boxes = [(0, int(rows[1]), 0, int(cols[2])),
                 (0, int(rows[1]), int(cols[2]), w),
                 (int(rows[1]), h, 0, int(cols[1])),
                 (int(rows[1]), int(rows[2]), int(cols[1]), w),
                 (int(rows[2]), h, int(cols[1]), w)]
    return _rotate_boxes(boxes, h, w, int(rng.integers(0, 4)))


def _witness_layout(rng):
    h = int(rng.integers(14, 31))
    w = int(rng.integers(14, 31))
    boxes = [(0, 5, 0, 5), (0, 5, 5, w),
             (5, h, 0, 5), (5, h, 5, w)]
    boxes, h, w = _rotate_boxes(boxes, h, w, int(rng.integers(0, 4)))
    winner = 1 if bool(rng.integers(0, 2)) else 2
    return boxes, h, w, winner


def _adjacent(a, b):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return (((ar1 == br0 or br1 == ar0) and max(ac0, bc0) < min(ac1, bc1))
            or ((ac1 == bc0 or bc1 == ac0) and max(ar0, br0) < min(ar1, br1)))


def _counts(rng, boxes, regime):
    capacity = [min(9, max(1, ((r1-r0)*(c1-c0)) // 6)) for r0, r1, c0, c1 in boxes]
    possible = [i for i, cap in enumerate(capacity) if cap >= 2]
    winner = int(rng.choice(possible))
    cap = capacity[winner]
    if regime == "target_test" and cap >= 5 and rng.random() < 0.65:
        winning = int(rng.integers(5, cap + 1))
    else:
        winning = int(rng.integers(1, min(cap, 5) + 1))
    margin = int(rng.integers(1, min(4, winning) + 1))
    rival_count = winning - margin
    counts = [int(rng.integers(0, min(capacity[i], rival_count) + 1))
              for i in range(len(boxes))]
    counts[winner] = winning
    rivals = [i for i in range(len(boxes)) if i != winner and capacity[i] >= rival_count]
    if rivals:
        counts[int(rng.choice(rivals))] = rival_count
    return counts, winner


def _choose_palette(rng, boxes, marker, winner, repeated, force_winner, force_base):
    n = len(boxes)
    pool = [c for c in range(10) if c != marker]
    winner_pool = [c for c in pool if c != force_base]
    winner_color = int(rng.choice(winner_pool)) if force_winner is None else force_winner
    assert winner_color != marker
    others = [c for c in pool if c != winner_color]
    rng.shuffle(others)
    if force_base is not None:
        others.remove(force_base)
        others.insert(0, force_base)
    colors = [0] * n
    colors[winner] = winner_color
    j = 0
    for i in range(n):
        if i != winner:
            colors[i] = others[j]
            j += 1
    if repeated is not None:
        a, b = repeated
        assert a != winner and b != winner and not _adjacent(boxes[a], boxes[b])
        colors[b] = colors[a]
    return colors


def _marker_positions(rng, boxes, counts, h, w):
    for attempt in range(32):
        selected = []
        positions = []
        parity = (int(rng.integers(0, 2)), int(rng.integers(0, 2)))
        for box, count in zip(boxes, counts):
            r0, r1, c0, c1 = box
            options = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
            rng.shuffle(options)
            if attempt >= 24:
                options = [p for p in options if p[0] % 2 == parity[0] and p[1] % 2 == parity[1]]
            if count and attempt < 24:
                category = int(rng.integers(0, 5))
                if category == 0:
                    preferred = lambda p: r0 < p[0] < r1-1 and c0 < p[1] < c1-1
                elif category == 1:
                    preferred = lambda p: p[0] in (r0, r1-1) or p[1] in (c0, c1-1)
                elif category == 2:
                    preferred = lambda p: p[0] in (0, h-1) or p[1] in (0, w-1)
                elif category == 3:
                    preferred = lambda p: p[0] in (r0, r1-1) and p[1] in (c0, c1-1)
                else:
                    preferred = lambda p: True
                options.sort(key=lambda p: not preferred(p))
            local = []
            corners = ((r0,c0),(r0,c1-1),(r1-1,c0),(r1-1,c1-1))
            for p in options:
                if len(local) == count:
                    break
                if p in corners and any(q in corners for q in local):
                    continue
                if all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) > 1 for q in selected):
                    local.append(p)
                    selected.append(p)
            if len(local) != count:
                break
            positions.append(local)
        if len(positions) == len(boxes):
            return positions
    raise ValueError("could not place isolated markers")


def _make_pair(rng, regime, marker, force_winner=None, force_base=None, witness=False):
    for _ in range(80):
        if witness:
            boxes, h, w, winner = _witness_layout(rng)
            counts = [2, 0, 0, 2]
            counts[winner] = 3
            repeated = (0, 3)
        else:
            modes = ("bands2", "bands3", "bands4", "bands5", "grid4",
                     "grid6", "t3", "mixed4", "mixed5")
            weights = (1,1,1,1,1,2,1,2,2) if regime == "target_test" else (2,2,1,1,2,1,3,1,1)
            mode = str(rng.choice(modes, p=np.array(weights) / sum(weights)))
            boxes, h, w = _layout(rng, mode)
            counts, winner = _counts(rng, boxes, regime)
            repeated = None
            if force_base is None and len(boxes) >= 4 and rng.random() < 0.28:
                choices = [(i,j) for i in range(len(boxes)) for j in range(i+1,len(boxes))
                           if i != winner and j != winner and not _adjacent(boxes[i], boxes[j])]
                if choices:
                    repeated = choices[int(rng.integers(0, len(choices)))]
        try:
            colors = _choose_palette(rng, boxes, marker, winner, repeated, force_winner, force_base)
            positions = _marker_positions(rng, boxes, counts, h, w)
        except ValueError:
            continue
        grid = np.empty((h,w), dtype=np.int8)
        for (r0,r1,c0,c1), color in zip(boxes, colors):
            grid[r0:r1,c0:c1] = color
        for group in positions:
            for r,c in group:
                grid[r,c] = marker
        output = np.array([[colors[winner]]], dtype=np.int8)
        if len(_analyze_input(grid)) == 1:
            return {"input": grid, "output": output}
    raise ValueError("could not construct a recoverable pair")


def _components(grid, color):
    h,w = grid.shape
    seen = np.zeros((h,w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if grid[r,c] != color or seen[r,c]:
                continue
            stack = [(r,c)]
            seen[r,c] = True
            cells = []
            while stack:
                y,x = stack.pop()
                cells.append((y,x))
                for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy,xx] and grid[yy,xx] == color:
                        seen[yy,xx] = True
                        stack.append((yy,xx))
            result.append(cells)
    return result


def _analyze_input(grid):
    h,w = grid.shape
    solutions = []
    for marker in (int(x) for x in np.unique(grid)):
        boxes, colors, counts = [], [], []
        coverage = np.zeros((h,w), dtype=np.int8)
        for color in (int(x) for x in np.unique(grid) if int(x) != marker):
            for cells in _components(grid, color):
                r0 = min(y for y,x in cells)
                r1 = max(y for y,x in cells)+1
                c0 = min(x for y,x in cells)
                c1 = max(x for y,x in cells)+1
                if r1-r0 < 4 or c1-c0 < 4:
                    break
                block = grid[r0:r1,c0:c1]
                if not np.all((block == color) | (block == marker)):
                    break
                nmarker = int(np.count_nonzero(block == marker))
                if nmarker > (r1-r0)*(c1-c0)//5:
                    break
                boxes.append((r0,r1,c0,c1))
                colors.append(color)
                counts.append(nmarker)
                coverage[r0:r1,c0:c1] += 1
            else:
                continue
            break
        else:
            if not 2 <= len(boxes) <= 7 or not np.all(coverage == 1):
                continue
            if sum(counts) != int(np.count_nonzero(grid == marker)):
                continue
            if counts.count(max(counts)) != 1:
                continue
            if any(colors[i] == colors[j] and _adjacent(boxes[i], boxes[j])
                   for i in range(len(boxes)) for j in range(i+1,len(boxes))):
                continue
            solutions.append({"marker":marker,"boxes":boxes,"colors":colors,
                              "counts":counts,"winner":int(np.argmax(counts))})
    return solutions


def _input_digest(grid):
    h,w = grid.shape
    return hashlib.sha256(bytes((h,w)) + grid.astype(np.uint8).tobytes()).hexdigest()


def _grid_key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _checked_seed(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(seed, rng, num_train, num_test)
    # There is no puzzle-wide semantic color: the sparse marker is read within each input.
    markers = [int(x) for x in rng.integers(0, 10, size=ntrain+ntest)]
    if markers[1] == markers[0]:
        markers[1] = (markers[0] + int(rng.integers(1,10))) % 10
    pair_seeds = [int(x) for x in rng.integers(0, 2**63-1, size=ntrain+ntest)]
    pairs = [
        _make_pair(np.random.default_rng(pair_seeds[0]), "source_train",
                   markers[0], force_winner=markers[1], witness=True),
        _make_pair(np.random.default_rng(pair_seeds[1]), "source_train",
                   markers[1], force_base=markers[0]),
    ]
    for i in range(2,ntrain):
        pairs.append(_make_pair(np.random.default_rng(pair_seeds[i]), "source_train", markers[i]))
    train = [pairs[int(i)] for i in rng.permutation(ntrain)]
    test = [_make_pair(np.random.default_rng(pair_seeds[ntrain+i]), "target_test", markers[ntrain+i])
            for i in range(ntest)]
    puzzle = {"train":train,"test":test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_region_color_witness(record):
    boxes, colors, counts = record["boxes"], record["colors"], record["counts"]
    winner = record["winner"]
    return any(colors[i] == colors[j] and i != winner and j != winner and
               counts[i]+counts[j] > counts[winner] and not _adjacent(boxes[i],boxes[j])
               for i in range(len(boxes)) for j in range(i+1,len(boxes)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train","test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_records = []
    for section in ("train","test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input","output"}
            for grid in (pair["input"],pair["output"]):
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype,np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            grid, output = pair["input"], pair["output"]
            assert output.shape == (1,1)
            key = _grid_key(grid)
            assert key not in seen
            seen.add(key)
            assert _input_digest(grid) not in _SOURCE_INPUT_DIGESTS
            solutions = _analyze_input(grid)
            assert len(solutions) == 1
            record = solutions[0]
            assert int(output[0,0]) == record["colors"][record["winner"]]
            positions = np.argwhere(grid == record["marker"])
            assert all(max(abs(int(a[0]-b[0])),abs(int(a[1]-b[1]))) > 1
                       for i,a in enumerate(positions) for b in positions[i+1:])
            if section == "train":
                train_records.append(record)
    assert len({int(p["output"][0,0]) for p in puzzle["train"]}) >= 2
    assert len({r["marker"] for r in train_records}) >= 2
    assert any(r["marker"] in s["colors"] for i,r in enumerate(train_records)
               for j,s in enumerate(train_records) if i != j)
    assert any(_has_region_color_witness(r) for r in train_records)
    assert any((lambda a,c,w: a[w] < max(a) and
                any(c[i]/a[i] > c[w]/a[w] for i in range(len(a))))
               ([(b[1]-b[0])*(b[3]-b[2]) for b in r["boxes"]],r["counts"],r["winner"])
               for r in train_records)


def _puzzles_equal(a,b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k],y[k])
               for x,y in zip(a[s],b[s]) for k in ("input","output"))
               for s in ("train","test"))


def _layout_class(boxes, h, w):
    if all(r0 == 0 and r1 == h for r0,r1,c0,c1 in boxes):
        return "vertical_bands"
    if all(c0 == 0 and c1 == w for r0,r1,c0,c1 in boxes):
        return "horizontal_bands"
    if len(boxes) == 4 and len({r0 for r0,r1,c0,c1 in boxes}) == 2 and \
            len({c0 for r0,r1,c0,c1 in boxes}) == 2:
        return "grid4"
    if len(boxes) == 3:
        return "T3"
    return "multi_cut"


def self_test() -> None:
    for seed, train, test in ((101,2,1),(102,10,1),(103,10,4),
                              (201,None,1),(202,None,4),(203,2,None),
                              (204,10,None),(271828,7,3)):
        puzzle = generate_puzzle(seed,train,test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        assert _puzzles_equal(puzzle,generate_puzzle(seed,train,test))
    for args in ((1,None),(11,None),(None,0),(None,5),(2,2),(2,4)):
        try:
            generate_puzzle(999,*args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts = set(),set()
    shapes, colors, regions, signatures = set(),set(),set(),set()
    cross_role = repeated = area_witness = density_witness = 0
    varied_shapes = varied_layouts = familiar_tests = high_tests = 0
    interior = edge = outer = corner = 0
    layout_classes, margins = set(), set()
    low = high = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_grid_key(p["input"]) for s in ("train","test") for p in puzzle[s]))
        records = []
        train_shapes, train_layouts = set(), set()
        for section in ("train","test"):
            for pair in puzzle[section]:
                g = pair["input"]
                rec = _analyze_input(g)[0]
                records.append(rec)
                klass = _layout_class(rec["boxes"], *g.shape)
                layout_classes.add(klass)
                shapes.add(g.shape)
                colors.add(rec["marker"])
                regions.add(len(rec["boxes"]))
                m = max(rec["counts"])
                margins.add(m - sorted(rec["counts"])[-2])
                low += m <= 2
                high += m >= 5
                for r0,r1,c0,c1 in rec["boxes"]:
                    for r in range(r0,r1):
                        for c in range(c0,c1):
                            if g[r,c] != rec["marker"]:
                                continue
                            if r in (r0,r1-1) or c in (c0,c1-1):
                                edge += 1
                            else:
                                interior += 1
                            outer += r in (0,g.shape[0]-1) or c in (0,g.shape[1]-1)
                            corner += r in (r0,r1-1) and c in (c0,c1-1)
                if section == "train":
                    train_shapes.add(g.shape)
                    train_layouts.add(klass)
                    repeated += _has_region_color_witness(rec)
                    areas = [(b[1]-b[0])*(b[3]-b[2]) for b in rec["boxes"]]
                    win = rec["winner"]
                    area_witness += areas[win] < max(areas)
                    density_witness += any(rec["counts"][i]/areas[i] > rec["counts"][win]/areas[win]
                                           for i in range(len(areas)))
                else:
                    familiar_tests += m <= 4
                    high_tests += m >= 5
        varied_shapes += len(train_shapes) >= 2
        varied_layouts += len(train_layouts) >= 2
        cross_role += any(a["marker"] in b["colors"] for i,a in enumerate(records)
                          for j,b in enumerate(records) if i != j)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 60 and len(colors) >= 8
    assert {2,3,4,5,6}.issubset(regions)
    assert cross_role == 42 and repeated >= 42
    assert area_witness >= 42 and density_witness >= 42
    assert low >= 20 and high >= 20
    assert varied_shapes == 42 and varied_layouts >= 35
    assert familiar_tests >= 10 and high_tests >= 20
    assert {"vertical_bands","horizontal_bands","grid4","T3","multi_cut"}.issubset(layout_classes)
    assert {1,2,3,4}.issubset(margins)
    assert interior >= 50 and edge >= 50 and outer >= 20 and corner >= 5
