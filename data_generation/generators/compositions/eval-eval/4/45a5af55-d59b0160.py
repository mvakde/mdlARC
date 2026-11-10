"""Query-guided concentric excavation. No file I/O or retained puzzle state."""

from collections import Counter
from hashlib import sha256

import numpy as np


HOST_SEEDS = (0, 865459631, 205429979, 312049205, 480376377, 801660276,
              277853096, 393335166, 339147607, 327919069, 237084026, 902156844,
              792564755, 256986915, 639672886, 250554648, 603200937, 756949484,
              824602194, 598726713, 144980702, 114806755, 158859319, 747208799,
              678115850, 244162230, 323475700, 597028558, 796474161, 5000905,
              134392982, 964270596, 608895266, 222775905, 232181030, 645688291,
              104310835, 238284429, 763511244, 606846169, 772862117, 270159379)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a


def _depth(shape):
    h, w = shape
    r = np.minimum(np.arange(h), np.arange(h)[::-1])
    c = np.minimum(np.arange(w), np.arange(w)[::-1])
    return np.minimum(r[:, None], c[None, :])


def _key(a):
    return a.shape, a.astype(np.uint8, copy=False).tobytes()


def _read_input(value):
    """Read the fixed legend and segment records using only visible cells."""
    a = _grid(value)
    h, w = a.shape
    assert h >= 11 and w >= 10
    frame = int(a[3, 3])
    assert np.all(a[3, :4] == frame) and np.all(a[:4, 3] == frame)
    colors, counts = np.unique(a[:3, :3], return_counts=True)
    padding = colors[counts == 6]
    assert len(padding) == 1
    bg = int(padding[0])
    query = frozenset(int(c) for c in colors if c != bg)
    assert len(query) == 3 and frame not in query and frame != bg
    assert np.all(a[5, :5] == frame)
    code = []
    bottom = None
    for row in range(6, h):
        assert a[row, 0] == frame and a[row, 4] == frame
        if np.all(a[row, :5] == frame):
            bottom = row
            break
        color = int(a[row, 1])
        assert color not in (bg, frame) and np.all(a[row, 1:4] == color)
        code.append(color)
    assert bottom is not None and 3 <= len(code) <= 16 and code[-1] == code[-2]
    # The duplicated final row follows A's convention; it is not another layer.
    code = np.asarray(code[:-1], dtype=np.int8)
    legend_mask = np.zeros((h, 6), dtype=bool)
    legend_mask[:4, :4] = True
    legend_mask[5:bottom+1, :5] = True
    assert np.all(a[:, :6][~legend_mask] == bg)
    assert not np.any(a[:, 6:] == frame)
    pending = set(zip(*np.nonzero(a[:, 6:] != bg)))
    records = []
    while pending:
        start = min(pending)
        pending.remove(start)
        cells = [start]
        for r, c in cells:
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in pending:
                    pending.remove(point)
                    cells.append(point)
        rows, cols = zip(*cells)
        box = min(rows), max(rows)+1, min(cols)+6, max(cols)+7
        r0, r1, c0, c1 = box
        tile = a[r0:r1, c0:c1]
        assert tile.size == len(cells) and np.all(tile != bg)
        depth = _depth(tile.shape)
        assert int(depth.max()) < len(code)
        values, frequencies = np.unique(tile, return_counts=True)
        modes = values[frequencies == frequencies.max()]
        assert len(modes) == 1 and int(modes[0]) not in query
        filler = int(modes[0])
        present = frozenset(int(c) for c in values if c != filler)
        deepest = {q: int(depth[tile == q].max()) for q in query if np.any(tile == q)}
        cut = min(deepest.values()) if query <= present else None
        records.append({"box": box, "depth": depth, "cut": cut, "deepest": deepest,
                        "filler": filler, "labels": present,
                        "counts": Counter(int(c) for c in tile.flat),
                        "marks": int(np.count_nonzero(tile != filler))})
    assert records and len({r["filler"] for r in records}) == 1
    for i, first in enumerate(records):
        a0, a1, b0, b1 = first["box"]
        for second in records[i+1:]:
            c0, c1, d0, d1 = second["box"]
            assert a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0
    return {"background": bg, "frame": frame, "query": query, "code": code,
            "records": records}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase the deepest centered query-containing core; coat its outer shells."""
    a = _grid(input_grid)
    info = _read_input(a)
    out = a.copy()
    for record in info["records"]:
        cut = record["cut"]
        if cut is None:
            continue
        r0, r1, c0, c1 = record["box"]
        d = record["depth"]
        tile = out[r0:r1, c0:c1]
        tile[:] = info["background"]
        shell = d < cut
        tile[shell] = info["code"][d[shell]]
    return out


def _capacity(box, minimum):
    r0, r1, c0, c1 = box
    return ((r1-r0+1)//(minimum+1))*((c1-c0+1)//(minimum+1))


def _partition(rng, box, count, minimum):
    """Random guillotine cuts with a capacity proof, no unbounded retries."""
    if count == 1:
        return [box]
    r0, r1, c0, c1 = box
    options = []
    for gap in (1, 2, 3):
        for axis in (0, 1):
            lo, hi = (r0, r1) if axis == 0 else (c0, c1)
            for cut in range(lo+minimum, hi-gap-minimum+1):
                if axis == 0:
                    first, second = (r0, cut, c0, c1), (cut+gap, r1, c0, c1)
                else:
                    first, second = (r0, r1, c0, cut), (r0, r1, cut+gap, c1)
                cap_a, cap_b = _capacity(first, minimum), _capacity(second, minimum)
                for left_count in range(max(1, count-cap_b), min(count-1, cap_a)+1):
                    options.append((first, second, left_count))
    assert options
    first, second, n = options[int(rng.integers(len(options)))]
    return (_partition(rng, first, n, minimum)
            + _partition(rng, second, count-n, minimum))


def _crop(rng, box, minimum, layers, shape=None):
    r0, r1, c0, c1 = box
    if shape is None:
        options = [(h, w) for h in range(minimum, r1-r0+1)
                   for w in range(minimum, c1-c0+1) if min(h, w) <= 2*layers]
        # Full leaf, ordinary rectangles and extreme aspect ratios all have mass.
        mode = int(rng.integers(4))
        if mode == 1:
            largest = max(h*w for h, w in options)
            options = [(h, w) for h, w in options if h*w >= .75*largest]
        elif mode == 2:
            ratio = max(max(h/w, w/h) for h, w in options)
            options = [(h, w) for h, w in options if max(h/w, w/h) >= .8*ratio]
        shape = options[int(rng.integers(len(options)))]
    h, w = shape
    row = r0+int(rng.integers(r1-r0-h+1))
    col = c0+int(rng.integers(c1-c0-w+1))
    return row, row+h, col, col+w


def _sample_code(rng, layers, filler, labels, repeat_witness):
    palette = [filler]+list(labels)
    code = [int(rng.choice(palette)) for _ in range(layers)]
    code[0] = int(rng.choice(labels))
    code[1] = int(rng.choice([c for c in palette if c != code[0]]))
    if repeat_witness:
        code[2] = code[1]
    code[-1] = int(rng.choice([c for c in palette if c != code[0]]))
    if repeat_witness:
        # a,b,b,a defeats both numeric sorting and run compression, even
        # though only the first three rows are exposed by the teaching cut.
        code[3] = code[0]
    # Ordinary codes can have long bands, alternation or nonadjacent returns.
    if not repeat_witness and rng.random() < .45:
        for d in range(2, layers-1):
            code[d] = code[d-1] if rng.random() < .5 else code[d-2]
    return np.asarray(code+[code[-1]], dtype=np.int8)


def _paint_record(rng, shape, filler, query, extras, kind, cut):
    tile = np.full(shape, filler, dtype=np.int8)
    depth = _depth(shape)
    q = list(query)
    rng.shuffle(q)
    occupied = np.zeros(shape, dtype=bool)

    def place(color, mask):
        available = np.flatnonzero(mask & ~occupied)
        assert len(available)
        position = int(rng.choice(available))
        tile.flat[position] = color
        occupied.flat[position] = True

    if cut is not None:
        # q[0] is the bottleneck. Two deeper marks must actually fit; a
        # one-cell odd center cannot hold two distinct query identities.
        place(q[0], depth == cut)
        inner_level = cut+1 if np.count_nonzero(depth > cut) >= 2 else cut
        for color in q[1:]:
            place(color, depth >= inner_level)
        allowed = list(query)+list(extras)
        if kind == "rich_positive":
            for color in (q[0], extras[0], extras[0], q[1]):
                place(color, (depth <= cut) if color == q[0] else np.ones(shape, bool))
            if cut > 0:
                # A shallow copy cannot disqualify the deeper occurrence.
                place(q[0], depth == 0)
    elif kind == "empty":
        return tile
    else:
        missing = int(kind.split(":")[1]) if kind.startswith("missing:") else q[0]
        allowed = [color for color in query if color != missing]+list(extras)
        marks = [color for color in query if color != missing]+[extras[0]]
        if kind == "rich_negative" or kind.endswith(":rich"):
            marks += [extras[1], marks[0], extras[0], marks[1]]
        for color in marks:
            place(color, np.ones(shape, bool))

    exact = kind == "exact_positive" or (kind.startswith("missing:") and not kind.endswith(":rich"))
    if not exact:
        target = max(int(occupied.sum()), int(tile.size*rng.uniform(.08, .62)))
        for _ in range(target-int(occupied.sum())):
            color = int(rng.choice(allowed))
            mask = (depth <= cut) if cut is not None and color == q[0] else np.ones(shape, bool)
            candidates = np.flatnonzero(mask & ~occupied)
            if not len(candidates):
                continue
            # Preserve a uniquely readable common filler even at >50% marks.
            frequencies = Counter(int(c) for c in tile.flat if c != filler)
            remaining_filler = int(np.count_nonzero(~occupied))-1
            if max([frequencies.get(color, 0)+1]+list(frequencies.values())) >= remaining_filler:
                break
            position = int(rng.choice(candidates))
            tile.flat[position] = color
            occupied.flat[position] = True
    if cut is not None:
        assert min(int(depth[tile == c].max()) for c in query) == cut
    else:
        assert not set(query) <= set(tile.flat)
    return tile


def _make_pair(rng, package, layers, requested_cut=None):
    colors = [int(c) for c in rng.permutation(10)]
    bg, frame, filler = colors[:3]
    # Two non-query identities permit count-matched and richer hard negatives.
    labels = colors[3:3+int(rng.integers(5, 8))]
    query = [int(c) for c in rng.choice(labels, 3, replace=False)]
    extras = [c for c in labels if c not in query]
    code = _sample_code(rng, layers, filler, labels, package == "thickness")
    if package == "identity":
        kinds = [("exact_positive", 1), (f"missing:{query[0]}", None),
                 (f"missing:{query[1]}:rich", None), (f"missing:{query[2]}:rich", None)]
        minimum, count = 6, 4+int(rng.integers(3))
    elif package == "thickness":
        kinds = [("rich_positive", 3), ("rich_positive", 0), ("empty", None)]
        minimum, count = 8, 3+int(rng.integers(3))
    elif package == "test":
        minimum = 2*(requested_cut+1)
        cap = _capacity((0, 30, 6, 30), minimum)
        count = int(rng.integers(1, min(4, cap)+1))
        kinds = [("rich_positive", requested_cut)]
    else:
        minimum, count, kinds = 4, int(rng.integers(1, 8)), []
    while len(kinds) < count:
        if rng.random() < .5:
            kinds.append(("rich_positive", int(rng.integers(min(3, minimum//2-1)+1))))
        else:
            kinds.append((str(rng.choice(("rich_negative", "empty"))), None))
    h = max(layers+8, int(rng.integers(17, 31)))
    w = int(rng.integers(18, 31))
    # At most 25 increments; ordinary sampling is retained whenever it fits.
    while _capacity((0, h, 6, w), minimum) < count:
        if h < 30:
            h += 1
        elif w < 30:
            w += 1
        else:
            raise AssertionError("infeasible bounded layout")
    leaves = _partition(rng, (0, h, 6, w), count, minimum)
    leaves = [leaves[int(i)] for i in rng.permutation(count)]
    common_shape = None
    if package == "identity":
        a, b = leaves[:2]
        common = (0, min(a[1]-a[0], b[1]-b[0]), 0, min(a[3]-a[2], b[3]-b[2]))
        trial = _crop(rng, common, minimum, layers)
        common_shape = trial[1]-trial[0], trial[3]-trial[2]
    boxes = [_crop(rng, leaf, minimum, layers, common_shape if i < 2 else None)
             for i, leaf in enumerate(leaves)]
    a = np.full((h, w), bg, dtype=np.int8)
    a[3, :4], a[:4, 3] = frame, frame
    for position, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
        a[int(position)//3, int(position)%3] = color
    a[5:len(code)+7, :5] = frame
    a[6:len(code)+6, 1:4] = code[:, None]
    out = a.copy()
    for box, (kind, cut) in zip(boxes, kinds):
        r0, r1, c0, c1 = box
        tile = _paint_record(rng, (r1-r0, c1-c0), filler, query, extras, kind, cut)
        a[r0:r1, c0:c1] = tile
        if cut is None:
            out[r0:r1, c0:c1] = tile
        else:
            d = _depth(tile.shape)
            result = np.full(tile.shape, bg, dtype=np.int8)
            for level in range(cut):
                result[d == level] = code[level]
            out[r0:r1, c0:c1] = result
    return {"input": a, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    training_layers = []
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            package = ("identity" if index == 0 else "thickness" if index == 1 else "free") if split == "train" else "test"
            layers = int(rng.integers(4, 7)) if split == "train" else int(rng.integers(max(training_layers)+1, max(training_layers)+4))
            cut = None if split == "train" else int(rng.integers(4, min(layers-1, 7)+1))
            for _ in range(8):
                pair = _make_pair(rng, package, layers, cut)
                fingerprint = _key(pair["input"])
                if fingerprint not in seen:
                    break
            else:
                raise AssertionError("bounded uniqueness retries exhausted")
            seen.add(fingerprint)
            puzzle[split].append(pair)
            if split == "train":
                training_layers.append(layers)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _erosion_prediction(a, info):
    """Check the intended rule by literally shrinking each rectangle."""
    expected = a.copy()
    q, bg, code = info["query"], info["background"], info["code"]
    for record in info["records"]:
        r0, r1, c0, c1 = record["box"]
        original = a[r0:r1, c0:c1]
        last = None
        for inset in range((min(original.shape)+1)//2):
            inner = original[inset:original.shape[0]-inset, inset:original.shape[1]-inset]
            if not q <= set(int(c) for c in inner.flat):
                break
            last = inset
        assert last == record["cut"]
        if last is None:
            continue
        painted = np.full((original.shape[0]-2*last, original.shape[1]-2*last), bg, dtype=np.int8)
        for inset in range(last-1, -1, -1):
            painted = np.pad(painted, 1, constant_values=int(code[inset]))
        expected[r0:r1, c0:c1] = painted
    return expected


def _training_evidence(infos):
    records = [(i, r) for i in infos for r in i["records"]]
    positive = [(i, r) for i, r in records if r["cut"] is not None]
    negative = [(i, r) for i, r in records if r["cut"] is None]
    assert any(r["cut"] == 0 for _, r in positive)
    assert any(r["cut"] == 1 for _, r in positive)
    assert any(r["cut"] == 3 and i["code"][0] != i["code"][1]
               and i["code"][1] == i["code"][2] and i["code"][3] == i["code"][0]
               for i, r in positive)
    assert any(r["cut"] > 0 and i["code"][0] != i["code"][-1] for i, r in positive)
    assert all(r["cut"] <= 3 for _, r in positive)
    assert any(not r["labels"] for _, r in negative)
    assert any(r["marks"] > 3 and len(r["labels"]) >= 4 for _, r in negative)
    assert any(r["labels"]-i["query"] and any(r["counts"][q] > 1 for q in i["query"])
               for i, r in positive)
    for info in infos:
        q = info["query"]
        if all(any(r["cut"] is None and r["labels"] & q == q-{color}
                   for r in info["records"]) for color in q):
            break
    else:
        raise AssertionError("every query identity needs a missing-identity witness")
    def shape(r):
        a, b, c, d = r["box"]
        return b-a, d-c
    assert any(i is j and p["marks"] == n["marks"] == 3
               and len(p["labels"]) == len(n["labels"]) == 3 and shape(p) == shape(n)
               for i, p in positive for j, n in negative)
    assert any(min(r["deepest"].values()) < max(r["deepest"].values()) for _, r in positive)
    assert any(r["cut"] > 0 and any(np.any((i["source"][r["box"][0]:r["box"][1], r["box"][2]:r["box"][3]] == q)
                                              & (r["depth"] < r["cut"])) for q in i["query"])
               for i, r in positive)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    infos = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _read_input(a)
            assert np.array_equal(out, _erosion_prediction(a, info))
            assert np.array_equal(out, solve(a))
            assert np.array_equal(a[:, :6], out[:, :6])
            assert set(int(c) for c in out.flat) <= set(int(c) for c in a.flat)
            info["source"] = a
            infos[split].append(info)
    _training_evidence(infos["train"])
    largest_key = max(len(i["code"]) for i in infos["train"])
    for info in infos["test"]:
        assert len(info["code"]) > largest_key
        assert any(r["cut"] is not None and r["cut"] >= 4 for r in info["records"])


def _signature(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds, count checks and counterfactuals; write no artifacts."""
    metrics = {name: set() for name in ("signatures", "shapes", "object_shapes", "layers",
               "cuts", "roles", "counts", "codes", "normalized_inputs")}
    pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        metrics["signatures"].add(_signature(p))
        for split in ("train", "test"):
            for pair in p[split]:
                pairs += 1
                a = pair["input"]
                info = _read_input(a)
                metrics["shapes"].add(a.shape)
                metrics["layers"].add(len(info["code"]))
                metrics["roles"].add((info["background"], info["frame"], info["records"][0]["filler"]))
                metrics["counts"].add(len(info["records"]))
                normal = {}
                signature = bytes(normal.setdefault(int(c), len(normal)) for c in a.flat)
                metrics["normalized_inputs"].add(sha256(bytes(a.shape)+signature).digest())
                normal = {}
                metrics["codes"].add(tuple(normal.setdefault(int(c), len(normal)) for c in info["code"]))
                for record in info["records"]:
                    r0, r1, c0, c1 = record["box"]
                    metrics["object_shapes"].add((r1-r0, c1-c0))
                    metrics["cuts"].add(record["cut"])
        # Query-only changes must alter which fixed records qualify.
        for pair in p["train"]:
            info = _read_input(pair["input"])
            options = [r for r in info["records"] if r["cut"] is None and len(r["labels"]) == 3]
            if options:
                changed = pair["input"].copy()
                points = list(zip(*np.nonzero(changed[:3, :3] != info["background"])))
                for point, color in zip(points, sorted(options[0]["labels"])):
                    changed[point] = color
                r0, r1, c0, c1 = options[0]["box"]
                assert not np.array_equal(solve(changed)[r0:r1, c0:c1], pair["output"][r0:r1, c0:c1])
                # Permuting the query is immaterial.
                moved = pair["input"].copy()
                moved[:3, :3] = np.rot90(moved[:3, :3].copy())
                expected = pair["output"].copy()
                expected[:3, :3] = moved[:3, :3]
                assert np.array_equal(solve(moved), expected)
                break
        else:
            raise AssertionError("no query counterfactual")
        # Keep all identities and mark counts, moving just the limiting mark.
        # The deepest complete core must move one shell inward.
        for pair in p["train"]:
            info = _read_input(pair["input"])
            options = [r for r in info["records"] if r["cut"] == 1 and r["marks"] == 3
                       and min(r["deepest"].values()) < max(r["deepest"].values())]
            if options:
                record = options[0]
                limiting = next(q for q, d in record["deepest"].items() if d == 1)
                changed = pair["input"].copy()
                r0, r1, c0, c1 = record["box"]
                tile = changed[r0:r1, c0:c1]
                destination = np.argwhere((record["depth"] == 2) & (tile == record["filler"]))[0]
                tile[tile == limiting] = record["filler"]
                tile[tuple(destination)] = limiting
                after = _read_input(changed)
                shifted = next(r for r in after["records"] if r["box"] == record["box"])
                assert shifted["cut"] == 2 and shifted["counts"] == record["counts"]
                assert not np.array_equal(solve(changed), pair["output"])
                break
        else:
            raise AssertionError("no mark-depth counterfactual")
        # Swapping key rows changes a depth>=3 coating, with unchanged marks.
        for pair in p["train"]:
            info = _read_input(pair["input"])
            if any(r["cut"] == 3 for r in info["records"]) and info["code"][0] != info["code"][1]:
                changed = pair["input"].copy()
                changed[[6, 7], 1:4] = changed[[7, 6], 1:4]
                prediction = solve(changed)
                assert not np.array_equal(prediction[:, 6:], pair["output"][:, 6:])
                # Unused terminal code rows cannot affect shallow excavation.
                unused = pair["input"].copy()
                bottom_rows = [6+len(info["code"])-1, 6+len(info["code"])]
                unused_color = info["code"][1] if info["code"][-1] != info["code"][1] else info["code"][0]
                unused[bottom_rows, 1:4] = unused_color
                assert np.array_equal(solve(unused)[:, 6:], pair["output"][:, 6:])
                # An unrelated mark never changes a qualifying core.
                record = next(r for r in info["records"] if r["cut"] == 3)
                marked = pair["input"].copy()
                r0, r1, c0, c1 = record["box"]
                tile = marked[r0:r1, c0:c1]
                position = tuple(np.argwhere(tile == record["filler"])[0])
                extra = next(c for c in range(10) if c not in info["query"]
                             and c not in (info["background"], info["frame"], record["filler"]))
                tile[position] = extra
                assert np.array_equal(solve(marked), pair["output"])
                break
    assert len(metrics["signatures"]) == 42
    assert len(metrics["normalized_inputs"]) >= 200
    assert len(metrics["shapes"]) >= 25 and len(metrics["object_shapes"]) >= 50
    assert {4, 5, 6, 7, 8, 9} <= metrics["layers"]
    assert {None, 0, 1, 2, 3, 4, 5} <= metrics["cuts"]
    assert len(metrics["roles"]) >= 100 and len(metrics["codes"]) >= 50
    assert {1, 2, 3, 4, 5, 6} <= metrics["counts"]
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(93000+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    assert _signature(generate_puzzle(1234, 10, 4)) == _signature(generate_puzzle(1234, 10, 4))
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # A mutation of a single erased core cell must be rejected.
    bad = generate_puzzle(20261023, 2, 1)
    info = _read_input(bad["test"][0]["input"])
    record = next(r for r in info["records"] if r["cut"] is not None and r["cut"] >= 4)
    r0, _, c0, _ = record["box"]
    bad["test"][0]["output"][r0+record["cut"], c0+record["cut"]] = info["frame"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted excavation accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": pairs,
            **{name: len(values) for name, values in metrics.items()}}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
