"""Screened records: a binary tile both reads labels and writes a new texture.

Only stdlib and NumPy. Generation uses forward block painting; validation uses
cellwise periodic coordinates. No file access, cached examples, or hidden rule
parameters are needed by solve(). self_test() streams 42 seeds without files.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells = [start]
        for r, c in cells:
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    cells.append(p)
        rr, cc = zip(*cells)
        box = (min(rr), max(rr)+1, min(cc), max(cc)+1)
        result.append((box, len(cells)))
    return result


def _apart(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _header(header, frame):
    """Decode visible enclosures; no hard-coded object slots or palette IDs."""
    h, w = header.shape
    answers = []
    for r in range(h-3):
        for c in range(w-3):
            qbox = header[r:r+4, c:c+4]
            if not (np.all(qbox[-1] == frame) and np.all(qbox[:, -1] == frame)
                    and np.count_nonzero(qbox == frame) == 7):
                continue
            vals, counts = np.unique(qbox[:3, :3], return_counts=True)
            grounds = vals[counts == 6]
            if len(grounds) != 1 or len(vals) != 4:
                continue
            ground = int(grounds[0])
            query = frozenset(map(int, vals[vals != ground]))
            if frame == ground or frame in query:
                continue
            selector = (r, r+4, c, c+4)
            for n in range(2, 6):
                for tr in range(h-n-1):
                    for tc in range(w-n-1):
                        box = (tr, tr+n+2, tc, tc+n+2)
                        if not _apart(selector, box):
                            continue
                        v = header[tr:tr+n+2, tc:tc+n+2]
                        if not (np.all(v[0] == frame) and np.all(v[-1] == frame)
                                and np.all(v[:, 0] == frame) and np.all(v[:, -1] == frame)):
                            continue
                        tile = v[1:-1, 1:-1].copy()
                        colors = set(map(int, np.unique(tile)))
                        if len(colors) != 2 or colors & ({ground, frame} | query):
                            continue
                        keep = np.ones(header.shape, bool)
                        keep[r:r+4, c:c+4] = False
                        keep[tr:tr+n+2, tc:tc+n+2] = False
                        if np.all(header[keep] == ground):
                            answers.append(dict(ground=ground, frame=int(frame), query=query,
                                                tile=tile, selector=selector, tile_box=box))
    if len(answers) != 1:
        raise ValueError("Header must have one L selector and one complete framed tile")
    return answers[0]


def _decode(input_grid):
    g = _array(input_grid)
    choices = []
    h, _ = g.shape
    # Only a short end strip can contain the two instruction objects. Frame
    # is exclusive to that strip and its one full-width separating line.
    for d in range(4, h-4):
        if not np.all(g[d] == g[d, 0]):
            continue
        frame = int(g[d, 0])
        for bottom in (False, True):
            header = g[d+1:] if bottom else g[:d]
            body = g[:d] if bottom else g[d+1:]
            if not 5 <= len(header) <= 9 or len(body) < 4:
                continue
            try:
                info = _header(header, frame)
                bg = info["ground"]
                if np.any(body == frame):
                    continue
                body_start = 0 if bottom else d+1
                header_start = d+1 if bottom else 0
                # A blank row visibly separates records from the divider.
                assert np.all(body[-1 if bottom else 0] == bg)
                records = []
                for (r0, r1, c0, c1), area in _components(body != bg):
                    assert area == (r1-r0)*(c1-c0), "Nonrectangular record"
                    box = (r0+body_start, r1+body_start, c0, c1)
                    records.append(box)
                assert records
                assert all(_apart(a, b) for i, a in enumerate(records) for b in records[i+1:])
                tile_colors = set(map(int, np.unique(info["tile"])))
                common = set.intersection(*(set(map(int, np.unique(g[a:b, c:e])))
                                            for a, b, c, e in records))
                fillers = tile_colors & common
                assert len(fillers) == 1
                filler = fillers.pop()
                ink = (tile_colors - {filler}).pop()
                assert not np.any(body == ink), "Closed-screen ink is not a record label"
                n = len(info["tile"])
                assert all(min(b-a, e-c) >= n+1 for a, b, c, e in records)
                for name in ("selector", "tile_box"):
                    a, b, c, e = info[name]
                    info[name] = (a+header_start, b+header_start, c, e)
                info.update(grid=g, filler=filler, ink=ink, records=records,
                            divider=d, bottom=bottom, body_start=body_start)
                choices.append(info)
            except (ValueError, AssertionError):
                continue
    if len(choices) != 1:
        raise ValueError("Expected one visibly bounded instruction strip")
    return choices[0]


def _screen(tile, filler, height, width):
    n = len(tile)
    rr, cc = np.indices((height, width))
    return tile[rr % n, cc % n] == filler


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only this input; reset the periodic screen at every record origin."""
    info = _decode(input_grid)
    out = info["grid"].copy()
    tile, filler, ink = info["tile"], info["filler"], info["ink"]
    for r0, r1, c0, c1 in info["records"]:
        record = info["grid"][r0:r1, c0:c1]
        opened = _screen(tile, filler, *record.shape)
        if info["query"] <= set(map(int, np.unique(record[opened]))):
            out[r0:r1, c0:c1] = np.where(opened, ink, filler)
    return out


def _facts(info):
    """Cellwise audit: unlike solve, does not use array tiling/screen helpers."""
    g, tile, n = info["grid"], info["tile"], len(info["tile"])
    expected = g.copy()
    facts = []
    for r0, r1, c0, c1 in info["records"]:
        visible, opened, blocked, counts = set(), set(), set(), {}
        for r in range(r0, r1):
            for c in range(c0, c1):
                color = int(g[r, c])
                if color != info["filler"]:
                    visible.add(color)
                    counts[color] = counts.get(color, 0)+1
                target = opened if tile[(r-r0) % n, (c-c0) % n] == info["filler"] else blocked
                target.add(color)
        good = info["query"] <= opened
        if good:
            for r in range(r0, r1):
                for c in range(c0, c1):
                    original = tile[(r-r0) % n, (c-c0) % n]
                    expected[r, c] = info["ink"] if original == info["filler"] else info["filler"]
        facts.append(dict(box=(r0, r1, c0, c1), visible=visible, opened=opened,
                          blocked=blocked, counts=counts, good=good,
                          marks=sum(counts.values()), shape=(r1-r0, c1-c0)))
    return expected, facts


def _alternative(info, behavior):
    """Explicit simpler hypotheses, for training-witness and mutation checks."""
    g, tile, filler, ink = info["grid"], info["tile"], info["filler"], info["ink"]
    out = g.copy()
    for r0, r1, c0, c1 in info["records"]:
        record = g[r0:r1, c0:c1]
        opened = _screen(tile, filler, *record.shape)
        if behavior == "global_phase":
            rr, cc = np.indices(record.shape)
            opened = tile[(rr+r0) % len(tile), (cc+c0) % len(tile)] == filler
        labels = set(map(int, np.unique(record[opened])))
        good = info["query"] <= labels
        if behavior == "all_colors":
            good = info["query"] <= set(map(int, np.unique(record)))
        elif behavior == "any_color":
            good = bool(info["query"] & labels)
        elif behavior == "all_records":
            good = True
        if good:
            pattern = np.where(opened, ink, filler)
            if behavior == "no_exchange":
                pattern = np.where(opened, filler, ink)
            elif behavior == "rotate_tile":
                pattern = np.where(_screen(np.rot90(tile, -1), filler, *record.shape), ink, filler)
            elif behavior == "delete":
                pattern = np.full(record.shape, info["ground"], int)
            elif behavior == "only_open":
                pattern = np.where(opened, ink, record)
            elif behavior == "stretch":
                rr, cc = np.indices(record.shape)
                stretched = tile[rr*len(tile)//len(record), cc*len(tile)//record.shape[1]]
                pattern = np.where(stretched == filler, ink, filler)
            out[r0:r1, c0:c1] = pattern
    return out


def _tile(rng, n, filler, ink):
    for _ in range(64):
        bits = rng.random((n, n)) < rng.uniform(.3, .7)
        # Each smallest supported record has space for the three query marks
        # and for a blocked mark. Witness patterns expose tile orientation.
        visible = bits[np.arange(n+1)[:, None] % n, np.arange(n+1)[None, :] % n]
        if (visible.sum() >= 3 and (~visible).sum() >= 1
                and bits.any() and not bits.all()
                and not np.array_equal(bits, np.rot90(bits, 2))):
            return np.where(bits, filler, ink)
    bits = np.zeros((n, n), bool)
    bits[0, :] = True
    bits[1, 0] = True
    return np.where(bits, filler, ink)


def _capacity(box, minimum):
    a, b, c, d = box
    return ((b-a+1)//(minimum+1))*((d-c+1)//(minimum+1))


def _partition(rng, box, count, minimum):
    """A bounded guillotine packing; capacities guarantee recursive feasibility."""
    if count == 1:
        return [box]
    r0, r1, c0, c1 = box
    options = []
    for gap in (1, 2):
        for axis, lo, hi in ((0, r0, r1), (1, c0, c1)):
            for cut in range(lo+minimum, hi-gap-minimum+1):
                a, b = ((r0, cut, c0, c1), (cut+gap, r1, c0, c1)) if axis == 0 else (
                    (r0, r1, c0, cut), (r0, r1, cut+gap, c1))
                ca, cb = _capacity(a, minimum), _capacity(b, minimum)
                for left_count in range(max(1, count-cb), min(count-1, ca)+1):
                    options.append((a, b, left_count))
    if not options:
        raise AssertionError("Feasible partition has no cut")
    a, b, left_count = options[int(rng.integers(len(options)))]
    return (_partition(rng, a, left_count, minimum)
            + _partition(rng, b, count-left_count, minimum))


def _crop(rng, leaf, minimum, clipped=False, large=False, shape=None):
    r0, r1, c0, c1 = leaf
    n = minimum-1
    choices = [(h, w) for h in range(minimum, r1-r0+1)
               for w in range(minimum, c1-c0+1)
               if not clipped or h % n or w % n]
    if large:
        area = max(h*w for h, w in choices)
        choices = [(h, w) for h, w in choices if h*w >= .7*area]
    h, w = choices[int(rng.integers(len(choices)))] if shape is None else shape
    r = r0+int(rng.integers(r1-r0-h+1))
    c = c0+int(rng.integers(c1-c0-w+1))
    return r, r+h, c, c+w


def _record(rng, shape, tile, filler, query, extras, kind, missing=0):
    a = np.full(shape, filler, int)
    opened = _screen(tile, filler, *shape)
    holes = list(map(tuple, np.argwhere(opened)))
    closed = list(map(tuple, np.argwhere(~opened)))
    rng.shuffle(holes)
    rng.shuffle(closed)
    used = set()

    def put(color, choices):
        p = next(p for p in choices if p not in used)
        a[p] = int(color)
        used.add(p)

    if kind == "empty":
        return a
    if kind in ("exact", "repeat", "positive"):
        for q in query:
            put(q, holes)
    elif kind == "gate":
        for j, q in enumerate(query):
            put(q, closed if j == missing else holes)
    elif kind in ("dense", "negative"):
        for j, q in enumerate(query):
            if j != missing:
                put(q, holes)
    else:
        raise AssertionError(kind)
    if kind in ("exact", "gate"):
        return a
    if kind == "repeat":
        # A qualifying record always has a label under a closed cell too.
        # It must be overwritten, even though it did not help qualification.
        put(extras[0], closed)
    remaining = [tuple(p) for p in np.argwhere(np.ones(shape, bool)) if tuple(p) not in used]
    rng.shuffle(remaining)
    if kind in ("repeat", "dense"):
        more = list(extras)+[query[(missing+1) % 3], extras[0]]
    else:
        more = list(extras[:int(rng.integers(len(extras)+1))])
    available = list(query) if kind != "dense" and kind != "negative" else [q for j, q in enumerate(query) if j != missing]
    available += list(extras)
    target = int(rng.integers(len(more), max(len(more), int(a.size*.6))+1))
    more += [int(rng.choice(available)) for _ in range(target-len(more))]
    # Always retain filler. Its literal identity links the tile and records.
    for color, p in zip(more[:max(0, len(remaining)-1)], remaining):
        a[p] = color
    return a


def _forward_paint(record, tile, filler, ink, query):
    """Generator construction: build periodic blocks, exchange, then crop."""
    n = len(tile)
    expanded = np.tile(tile, ((len(record)+n-1)//n, (record.shape[1]+n-1)//n))
    expanded = expanded[:len(record), :record.shape[1]]
    good = set(query) <= set(map(int, np.unique(record[expanded == filler])))
    if good:
        return np.where(expanded == filler, ink, filler)
    return record.copy()


def _make_pair(rng, n, package, bottom, test=False):
    colors = list(map(int, rng.permutation(10)))
    bg, frame, filler, ink = colors[:4]
    query = colors[4:7]
    extra_count = int(rng.integers(2 if package == "extras" else 0, 4))
    extras = colors[7:7+extra_count]
    tile = _tile(rng, n, filler, ink)
    if package == "identity":
        special = [("exact", 0)]+[("gate", j) for j in range(3)]
    elif package == "extras":
        special = [("repeat", 0), ("dense", 2), ("empty", 0), ("positive", 0)]
    elif test:
        special = [("exact", 0), ("gate", int(rng.integers(3))), ("empty", 0)]
    else:
        special = [("exact", 0), ("gate", int(rng.integers(3)))]
    count = int(rng.integers(7, 9)) if test else len(special)+int(rng.integers(0, 3))
    header_h = n+3+int(rng.integers(2))
    body_max = 30-header_h-2
    minimum = n+1
    shapes = [(h, w) for h in range(max(7, minimum), body_max+1)
              for w in range(max(14, n+7), 31)
              if _capacity((0, h, 0, w), minimum) >= count]
    body_h, width = shapes[int(rng.integers(len(shapes)))]
    leaves = _partition(rng, (0, body_h, 0, width), count, minimum)
    leaves = [leaves[int(i)] for i in rng.permutation(count)]
    boxes = [_crop(rng, leaf, minimum, clipped=(i == 0), large=test and i == 0)
             for i, leaf in enumerate(leaves)]
    if package == "identity":
        # The exact positive and one gate negative also match in dimensions,
        # in addition to their identical physical label multiset.
        mh = min(leaf[1]-leaf[0] for leaf in leaves[:2])
        mw = min(leaf[3]-leaf[2] for leaf in leaves[:2])
        choices = [(h, w) for h in range(minimum, mh+1) for w in range(minimum, mw+1)
                   if h % n or w % n]
        shared = choices[int(rng.integers(len(choices)))]
        boxes[:2] = [_crop(rng, leaf, minimum, shape=shared) for leaf in leaves[:2]]
    header = np.full((header_h, width), bg, int)
    # Randomize the left/right order, gaps, and vertical offsets of both keys.
    tw = n+2
    slack = width-tw-4-1
    left = int(rng.integers(slack+1))
    gap = 1+int(rng.integers(slack-left+1))
    if rng.integers(2):
        tc, qc = left, left+tw+gap
    else:
        qc, tc = left, left+4+gap
    tr = int(rng.integers(header_h-tw+1))
    qr = int(rng.integers(header_h-4+1))
    header[tr:tr+tw, tc:tc+tw] = frame
    header[tr+1:tr+n+1, tc+1:tc+n+1] = tile
    header[qr+3, qc:qc+4] = frame
    header[qr:qr+4, qc+3] = frame
    for p, q in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
        header[qr+int(p)//3, qc+int(p)%3] = q
    body = np.full((body_h, width), bg, int)
    output_body = body.copy()
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        kind, missing = special[i] if i < len(special) else (
            str(rng.choice(["positive", "negative", "empty", "gate"])), int(rng.integers(3)))
        record = _record(rng, (r1-r0, c1-c0), tile, filler, query, extras, kind, missing)
        body[r0:r1, c0:c1] = record
        output_body[r0:r1, c0:c1] = _forward_paint(record, tile, filler, ink, query)
    divider = np.full((1, width), frame, int)
    gap_row = np.full((1, width), bg, int)
    if bottom:
        inp = np.concatenate([body, gap_row, divider, header])
        out = np.concatenate([output_body, gap_row, divider, header])
    else:
        inp = np.concatenate([header, divider, gap_row, body])
        out = np.concatenate([header, divider, gap_row, output_body])
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    sizes = [int(rng.choice([2, 3])), int(rng.choice([4, 5]))]
    top_first = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            package = ("identity" if i == 0 else "extras" if i == 1 else "free") if split == "train" else "free"
            n = sizes[i] if split == "train" and i < 2 else int(rng.choice(sizes))
            bottom = ((i == 1) == top_first) if split == "train" and i < 2 else bool(rng.integers(2))
            for _ in range(12):
                pair = _make_pair(rng, n, package, bottom, test=split == "test")
                info = _decode(pair["input"])
                # Resetting at object origins must be taught, including at the
                # two-demonstration minimum. Other alternatives have explicit
                # constructive witnesses rather than acceptance retries.
                if (_key(pair["input"]) not in seen
                        and (package != "identity" or not np.array_equal(
                            _alternative(info, "global_phase"), pair["output"]))):
                    break
            else:
                raise AssertionError("Bounded distinct/witness construction failed")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            expected, facts = _facts(info)
            assert np.array_equal(out, expected), "Incorrect screen selection or full-record texture"
            assert np.array_equal(solve(inp), expected)
            assert any(f["good"] for f in facts) and any(not f["good"] for f in facts)
            assert any(f["good"] and any(d % len(info["tile"]) for d in f["shape"]) for f in facts)
            assert not np.array_equal(inp, out)
            (training if split == "train" else testing).append((info, facts, out))
    assert {i["bottom"] for i, _, _ in training} == {False, True}
    assert len({len(i["tile"]) for i, _, _ in training}) >= 2
    for i, _, _ in testing:
        assert len(i["tile"]) in {len(t["tile"]) for t, _, _ in training}
        assert len(i["records"]) > max(len(t["records"]) for t, _, _ in training)
    assert any(all(any(i["query"] <= f["visible"] and not f["good"]
                           and i["query"]-f["opened"] == {q} for f in fs)
                       for q in i["query"]) for i, fs, _ in training)
    records = [(i, f) for i, fs, _ in training for f in fs]
    assert any(f["good"] and f["marks"] == 3 for _, f in records)
    assert any(not f["visible"] for _, f in records)
    assert any(f["good"] and len(f["visible"]-i["query"]) >= 2
               and max(f["counts"].values()) > 1 for i, f in records)
    assert any(not f["good"] and len(f["visible"]) >= 4 and f["marks"] > 3 for _, f in records)
    assert any(p["good"] and not q["good"] and p["shape"] == q["shape"]
               and p["counts"] == q["counts"] for _, fs, _ in training for p in fs for q in fs)
    for behavior in ("all_colors", "any_color", "all_records", "global_phase", "no_exchange",
                     "rotate_tile", "delete", "only_open", "stretch"):
        assert any(not np.array_equal(_alternative(i, behavior), out) for i, _, out in training), behavior
    return None


def _counterfactuals(puzzle):
    for pair in puzzle["train"]:
        info = _decode(pair["input"])
        _, facts = _facts(info)
        targets = [f for f in facts if not f["good"] and info["query"] <= f["visible"]]
        if not targets:
            continue
        f = targets[0]
        missing = next(iter(info["query"]-f["opened"]))
        r0, r1, c0, c1 = f["box"]
        changed = info["grid"].copy()
        record = changed[r0:r1, c0:c1]
        opened = _screen(info["tile"], info["filler"], *record.shape)
        source = tuple(np.argwhere((record == missing) & ~opened)[0])
        dest = tuple(np.argwhere((record == info["filler"]) & opened)[0])
        record[source], record[dest] = info["filler"], missing
        # Move a mark without changing the physical color multiset: it flips
        # qualification, proving that ordinary unordered containment is weaker.
        prediction = solve(changed)
        assert not np.array_equal(prediction[r0:r1, c0:c1], record)
        assert np.array_equal(prediction[r0:r1, c0:c1], _forward_paint(
            record, info["tile"], info["filler"], info["ink"], info["query"]))
        # Permuting the selector positions does not affect unordered membership.
        moved = info["grid"].copy()
        a, _, c, _ = info["selector"]
        moved[a:a+3, c:c+3] = np.roll(moved[a:a+3, c:c+3], 1, axis=1)
        expected = pair["output"].copy()
        expected[a:a+3, c:c+3] = moved[a:a+3, c:c+3]
        assert np.array_equal(solve(moved), expected)
        return
    raise AssertionError("No gate counterfactual")


def self_test():
    """42 streamed seeds; one puzzle at a time, no writes or retained traces."""
    shapes, tiles, layouts, roles, numbers, structures, digests = (set() for _ in range(7))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _counterfactuals(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                shapes.add(pair["input"].shape)
                tiles.add(len(info["tile"]))
                layouts.add(info["bottom"])
                roles.add((info["ground"], info["frame"], info["filler"], info["ink"]))
                numbers.add(len(info["records"]))
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)
                structures.add(hashlib.sha256(normalized).digest())
                digest.update(pair["input"].astype(np.uint8).tobytes())
                digest.update(pair["output"].astype(np.uint8).tobytes())
        digests.add(digest.digest())
        if seed == 0:
            other = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], other[s]) for k in a)
    assert tiles == {2, 3, 4, 5} and layouts == {False, True}
    assert len(shapes) >= 30 and len(roles) >= 100 and len(structures) >= 180
    assert numbers >= {2, 3, 4, 5, 6, 7, 8} and len(digests) == 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(901, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(digests), "input_shapes": len(shapes),
            "tile_sizes": sorted(tiles), "role_assignments": len(roles),
            "record_counts": sorted(numbers), "color_normalized_inputs": len(structures)}
