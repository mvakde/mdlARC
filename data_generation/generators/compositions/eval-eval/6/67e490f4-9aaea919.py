"""Elect and pool: one stack is one ballot; its height is its transferable load.

Only the input is consulted by solve. Generation draws outputs from explicit
lane records, while validation reconstructs those records from visible pixels.
No files, examples, seeds, or puzzle-level learned conventions enter solving.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product

import numpy as np


def _norm(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-y, c-x) for r, c in cells))


@lru_cache(maxsize=2048)
def _variants(shape, reflect=True):
    result = set()
    for flip in range(2 if reflect else 1):
        current = _norm((r, -c if flip else c) for r, c in shape)
        for _ in range(4):
            result.add(current)
            current = _norm((c, -r) for r, c in current)
    return tuple(sorted(result))


def _key(cells, grouping="shape"):
    raw = _norm(cells)
    if grouping == "area":
        return len(raw)
    if grouping == "translation":
        return raw
    return min(_variants(raw, reflect=grouping != "rotation"))


def _size(shape):
    return max(r for r, c in shape)+1, max(c for r, c in shape)+1


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        point = min(pending)
        pending.remove(point)
        cells, todo = [], [point]
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in pending:
                    pending.remove(q)
                    todo.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _hole_free(shape):
    h, w = _size(shape)
    mask = np.ones((h+2, w+2), dtype=bool)
    for r, c in shape:
        mask[r+1, c+1] = False
    return len(_components(mask)) == 1


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg), "blank top row identifies background"
    sources, targets, pitches, baselines = [], [], set(), set()
    for left, right in _runs(np.any(a != bg, axis=0)):
        strip = a[:, left:right]
        occupied = np.argwhere(strip != bg)
        top, bottom = int(occupied[:, 0].min()), int(occupied[:, 0].max())
        box = strip[top:bottom+1]
        colors = set(map(int, np.unique(box))) - {bg}
        assert len(colors) == 1, "each input lane has one foreground color"
        color = colors.pop()
        framed = (np.any(box == bg) and np.all(box[0] == color)
                  and np.all(box[-1] == color) and np.all(box[:, 0] == color)
                  and np.all(box[:, -1] == color))
        if framed:
            parts = _components(box == bg)
            assert len(_components(box == color)) == 1
            cells = [tuple((r+top, c+int(left)) for r, c in p) for p in parts]
        else:
            cells = [tuple((r, c+int(left)) for r, c in p)
                     for p in _components(strip != bg)]
        assert cells
        cells.sort(key=lambda p: min(r for r, c in p))
        raw = _norm(cells[0])
        h, w = _size(raw)
        assert max(h, w) <= 5 and _hole_free(raw)
        assert all(_norm(p) == raw for p in cells), "exact repeated silhouette"
        starts = [min(r for r, c in p) for p in cells]
        xs = {min(c for r, c in p) for p in cells}
        assert len(xs) == 1
        pitches.update(b-a for a, b in zip(starts, starts[1:]))
        baseline = max(r for r, c in cells[-1])
        baselines.add(baseline)
        lane = dict(raw=raw, color=color, count=len(cells), cells=cells,
                    x=xs.pop(), bottom=baseline-h+1, left=int(left),
                    right=int(right), top=top, last=bottom)
        if framed:
            assert 1 <= len(cells) <= 3
            assert w+2 == right-left and starts[0] == top+1
            assert baseline == bottom-1 and lane["x"] == left+1
            targets.append(lane)
        else:
            sources.append(lane)
    assert sources and 1 <= len(targets) <= 3
    assert len(baselines) == 1 and len(pitches) == 1, "visible common pitch"
    pitch = pitches.pop()
    assert pitch > max(_size(s["raw"])[0] for s in sources+targets)
    carriers = {s["color"] for s in targets}
    assert len(carriers) == 1
    carrier = carriers.pop()
    assert all(s["color"] not in (bg, carrier) for s in sources)
    keys = [_key(t["raw"]) for t in targets]
    assert len(set(keys)) == len(keys), "one receiver per shape class"
    assert {_key(s["raw"]) for s in sources} == set(keys)
    return dict(grid=a, background=bg, carrier=carrier, pitch=pitch,
                baseline=baselines.pop(), sources=sources, targets=targets)


def _plans(data, grouping="shape", election="stacks", transfer="sum"):
    groups = defaultdict(list)
    for source in data["sources"]:
        groups[_key(source["raw"], grouping)].append(source)
    plans = []
    for target in data["targets"]:
        candidates = groups[_key(target["raw"], grouping)]
        assert candidates, "no supporting ballot"
        votes = Counter()
        for source in candidates:
            votes[source["color"]] += source["count"] if election == "tokens" else 1
        high = max(votes.values())
        winners = [color for color, n in votes.items() if n == high]
        assert len(winners) == 1, "elections must have a unique plurality"
        winner = winners[0]
        selected = [s for s in candidates if s["color"] == winner]
        total = sum(s["count"] for s in selected)
        if transfer == "columns":
            total = len(selected)
        elif transfer == "maximum":
            total = max(s["count"] for s in selected)
        elif transfer == "all":
            total = sum(s["count"] for s in candidates)
        elif transfer == "double":
            total = target["count"]
        elif transfer == "none":
            total = 0
        plans.append(dict(target=target, selected=selected, total=total,
                          winner=winner, votes=dict(votes)))
    return plans


def _paint(a, raw, y, x, color):
    h, w = _size(raw)
    assert 0 <= y and y+h <= a.shape[0] and 0 <= x and x+w <= a.shape[1]
    for r, c in raw:
        a[y+r, x+c] = color


def _render(data, **options):
    output = data["grid"].copy()
    for plan in _plans(data, **options):
        target = plan["target"]
        for cells in target["cells"]:
            for r, c in cells:
                output[r, c] = plan["winner"]
        for source in plan["selected"]:
            for cells in source["cells"]:
                for r, c in cells:
                    output[r, c] = data["carrier"]
        for level in range(target["count"], target["count"]+plan["total"]):
            y = target["bottom"]-level*data["pitch"]
            assert y >= 1, "complete extension needs blank top margin"
            h, w = _size(target["raw"])
            block = data["grid"][y:y+h, target["x"]:target["x"]+w]
            assert np.all(block == data["background"]), "unobstructed headroom"
            _paint(output, target["raw"], y, target["x"], plan["winner"])
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover shape elections and stack loads entirely from this input."""
    return _render(_parse(input_grid))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _pool(rng):
    shapes = set()
    for length in range(1, 6):
        shapes.add(_key(tuple((0, c) for c in range(length))))
    for h, w in product(range(2, 5), repeat=2):
        shapes.add(_key(tuple({(r, 0) for r in range(h)} |
                             {(h-1, c) for c in range(w)})))
        if h*w <= 8:
            shapes.add(_key(tuple(product(range(h), range(w)))))
    shapes.update(_key(s) for s in _ANCHORS)
    for _ in range(45):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(3, 8))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}-cells)
            cells.add(_choose(rng, frontier))
        raw = _norm(cells)
        if max(_size(raw)) <= 5 and _hole_free(raw):
            shapes.add(_key(raw))
    return sorted(shapes)


_ANCHORS = (
    ((0, 0), (1, 0), (2, 0), (2, 1)),  # L
    ((0, 0), (1, 0), (1, 1), (2, 1)),  # skew
    ((0, 0), (1, 0), (1, 1), (2, 0)),  # T
)


def _partition(rng, total, count):
    if count == 1:
        return [total]
    cuts = sorted(map(int, rng.choice(np.arange(1, total), count-1, replace=False)))
    values = np.diff([0]+cuts+[total]).tolist()
    rng.shuffle(values)
    return values


def _palette(rng, previous=None):
    colors = list(map(int, rng.permutation(10)))
    bg = _choose(rng, [c for c in colors if previous is None or c != previous[0]])
    carrier = _choose(rng, [c for c in colors if c != bg and
                          (previous is None or c != previous[1])])
    return bg, carrier, [c for c in colors if c not in (bg, carrier)]


def _build(rng, shapes, ballots, teaching=None, previous=None, forbidden=None):
    """A finite constructive layout: allocate lane widths before choosing heights."""
    bg, carrier, palette = _palette(rng, previous)
    classes, lanes = [], []
    for index, (shape, ballot) in enumerate(zip(shapes, ballots)):
        colors = list(map(int, rng.choice(palette, len(ballot), replace=False)))
        if index == 0 and forbidden is not None and colors[0] == forbidden:
            replacement = _choose(rng, [c for c in palette if c not in colors])
            colors[0] = replacement
        variants = _variants(shape)
        narrow = min(_size(v)[1] for v in variants)
        raw = _choose(rng, [v for v in variants if _size(v)[1] == narrow])
        target = dict(raw=raw, variants=variants, receiver=True, class_id=index,
                      color=colors[0], width=narrow+2)
        classes.append(dict(target=target, sources=[], ballot=ballot))
        lanes.append(target)
        for color, count in zip(colors, ballot):
            for _ in range(count):
                source = dict(raw=raw, variants=variants, receiver=False,
                              class_id=index, color=color, width=narrow)
                if teaching and index == 0:
                    source["raw"] = (_norm((r, -c) for r, c in raw)
                                     if color == colors[0] else raw)
                classes[-1]["sources"].append(source)
                lanes.append(source)
    minimum = sum(s["width"] for s in lanes)+len(lanes)-1
    if minimum > 30:
        return None
    spare = 30-minimum
    # First two witnesses retain their narrow orientations, including a
    # reflected winner and same-area/same-box distinct classes.
    if not teaching:
        for j in rng.permutation(len(lanes)):
            lane = lanes[int(j)]
            extra = 2 if lane["receiver"] else 0
            legal = [v for v in lane["variants"] if _size(v)[1]+extra <= lane["width"]+spare]
            raw = _choose(rng, legal)
            width = _size(raw)[1]+extra
            spare -= width-lane["width"]
            lane["raw"], lane["width"] = raw, width
    max_h = max(_size(s["raw"])[0] for s in lanes)
    pitch = max_h+int(rng.integers(1, 3))
    capacity = 1+(27-max_h)//pitch
    if any(ballot[0]+1 > capacity for ballot in ballots):
        return None
    for index, group in enumerate(classes):
        target, sources = group["target"], group["sources"]
        winner = target["color"]
        selected = [s for s in sources if s["color"] == winner]
        losers = [s for s in sources if s["color"] != winner]
        if teaching == "shapes" and index == 0:
            heights = [1, 2]
            rng.shuffle(heights)
            destination = 2
            for s in losers:
                s["count"] = int(rng.integers(4, min(5, capacity)+1))
        elif teaching == "plurality" and index == 0:
            total = int(rng.integers(4, min(5, capacity-1)+1))
            heights, destination = _partition(rng, total, len(selected)), 1
            loser_colors = list(dict.fromkeys(s["color"] for s in losers))
            for s in losers:
                s["count"] = (3 if s["color"] == loser_colors[0]
                              else int(rng.integers(1, 3)))
        else:
            destination = int(rng.integers(1, min(3, capacity-len(selected))+1))
            total = int(rng.integers(len(selected), capacity-destination+1))
            heights = _partition(rng, total, len(selected))
            for s in losers:
                s["count"] = int(rng.integers(1, min(5, capacity)+1))
        for s, count in zip(selected, heights):
            s["count"] = count
        target["count"] = destination
        target["final"] = destination+sum(heights)
        assert target["final"] <= capacity
    if max(s["count"] for s in lanes) == 1:
        # Visible repetition is mandatory, even in singleton-rich scenes.
        target = classes[0]["target"]
        if target["final"] < capacity:
            target["count"] += 1
            target["final"] += 1
        else:
            return None
    needed = max(_size(s["raw"])[0]+((s["final"] if s["receiver"] else s["count"])-1)*pitch
                 for s in lanes)
    if needed+2 > 30:
        return None
    margin = int(rng.integers(0, min(2, 30-needed-2)+1))
    height = int(rng.integers(needed+2+margin, 31))
    baseline = height-margin-2
    width = int(rng.integers(minimum if teaching else 30-spare, 31))
    extra = rng.multinomial(width-(sum(s["width"] for s in lanes)+len(lanes)-1),
                            np.full(len(lanes)+1, 1/(len(lanes)+1)))
    a = np.full((height, width), bg, dtype=np.int8)
    b = a.copy()
    x = int(extra[0])
    for position, j in enumerate(rng.permutation(len(lanes))):
        lane = lanes[int(j)]
        raw = lane["raw"]
        h, w = _size(raw)
        bottom = baseline-h+1
        if lane["receiver"]:
            y = bottom-(lane["count"]-1)*pitch-1
            a[y:baseline+2, x:x+w+2] = carrier
            b[y:baseline+2, x:x+w+2] = carrier
            for k in range(lane["count"]):
                _paint(a, raw, bottom-k*pitch, x+1, bg)
            for k in range(lane["final"]):
                _paint(b, raw, bottom-k*pitch, x+1, lane["color"])
        else:
            winner = classes[lane["class_id"]]["target"]["color"]
            for k in range(lane["count"]):
                _paint(a, raw, bottom-k*pitch, x, lane["color"])
                _paint(b, raw, bottom-k*pitch, x,
                       carrier if lane["color"] == winner else lane["color"])
        x += lane["width"]+1+int(extra[position+1])
    return {"input": a, "output": b}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _input_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    anchor_index = int(rng.integers(2))
    anchor = _key(_ANCHORS[anchor_index])
    companion = _key(_choose(rng, [s for s in _ANCHORS if _key(s) != anchor]))
    pool = _pool(rng)
    puzzle, seen, known = {"train": [], "test": []}, set(), set()
    previous = forbidden = None
    teaching_order = list(rng.permutation(["shapes", "plurality"]))
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = teaching_order[index] if split == "train" and index < 2 else None
            for attempt in range(80):
                if teaching == "shapes":
                    shapes, ballots = [anchor, companion], [(2, 1), (2, 1)]
                elif teaching == "plurality":
                    shapes, ballots = [anchor], [(3, 2, 2)]
                else:
                    k = int(rng.choice([1, 2, 3], p=[.40, .43, .17]))
                    available = [s for s in pool if k == 1 or
                                 min(_size(v)[1] for v in _variants(s)) <= 2]
                    if split == "test":
                        fresh = [s for s in available if s not in known]
                        if not fresh:
                            continue
                        shapes = [_choose(rng, fresh)]
                    else:
                        shapes = [_choose(rng, available)]
                    others = [s for s in available if s not in shapes]
                    shapes += [others[int(j)] for j in rng.choice(len(others), k-1, replace=False)]
                    modes = ([(1,), (2,), (2, 1), (3, 1), (3, 2), (2, 1, 1),
                              (3, 2, 2), (4, 2, 1)] if k == 1 else
                             [(1,), (2,), (2, 1)] if k == 2 else [(1,), (1,), (2,)])
                    ballots = [_choose(rng, modes) for _ in shapes]
                pair = _build(rng, shapes, ballots, teaching,
                              previous if teaching else None, forbidden if teaching else None)
                if pair is not None and _input_key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
            puzzle[split].append(pair)
            seen.add(_input_key(pair["input"]))
            if split == "train":
                known.update(shapes)
                if teaching:
                    parsed = _parse(pair["input"])
                    previous = parsed["background"], parsed["carrier"]
                    forbidden = next(p["winner"] for p in _plans(parsed)
                                     if _key(p["target"]["raw"]) == anchor)
    validate_generated_puzzle(puzzle)
    return puzzle


def _different(pair, **options):
    try:
        return not np.array_equal(_render(_parse(pair["input"]), **options), pair["output"])
    except (AssertionError, ValueError, IndexError):
        return True


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_shapes, history, operands = set(), set(), defaultdict(set), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            key = _input_key(a)
            assert key not in seen
            seen.add(key)
            data = _parse(a)
            assert np.array_equal(_render(data), b), "wrong election, sum, fill, highlight, or preservation"
            assert not np.array_equal(a, b)
            plans = _plans(data)
            shapes = {_key(p["target"]["raw"]) for p in plans}
            if split == "train":
                train_shapes.update(shapes)
                for p in plans:
                    history[_key(p["target"]["raw"])].add(p["winner"])
                    operands.append((p["total"], p["target"]["count"]))
            else:
                assert shapes-train_shapes, "test introduces a locally supported new shape"
    train = puzzle["train"]
    for options in ({"election": "tokens"}, {"grouping": "area"},
                    {"grouping": "translation"}, {"grouping": "rotation"},
                    {"transfer": "columns"}, {"transfer": "maximum"},
                    {"transfer": "all"}, {"transfer": "double"}, {"transfer": "none"}):
        assert any(_different(pair, **options) for pair in train), str(options)
    assert any(len(colors) > 1 for colors in history.values()), "no memorized shape-color map"
    assert len({s for s, d in operands}) > 1 and len({d for s, d in operands}) > 1
    assert len({(s, d) for s, d in operands}) > 1
    assert any(s*d0 != s0*d for s, d in operands for s0, d0 in operands)
    assert any(len(p["votes"]) >= 3 and max(p["votes"].values())*2 <= sum(p["votes"].values())
               for pair in train for p in _plans(_parse(pair["input"])))


def _fingerprint(puzzle):
    return tuple(_input_key(p[r]) for split in ("train", "test")
                 for p in puzzle[split] for r in ("input", "output"))


def self_test():
    """Stream 42 seeds; retain feature sets, never full sample collections."""
    signatures, dimensions, shapes, pitches, layouts, palettes, ballots = (set() for _ in range(7))
    class_counts, loads, destination_counts, source_counts = (set() for _ in range(4))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(hash(_fingerprint(puzzle)))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                data = _parse(pair["input"])
                dimensions.add(pair["input"].shape)
                pitches.add(data["pitch"])
                palettes.add((data["background"], data["carrier"]))
                class_counts.add(len(data["targets"]))
                layouts.add(tuple((s["left"], s["right"], s["count"])
                                  for s in sorted(data["sources"]+data["targets"], key=lambda s: s["left"])))
                for p in _plans(data):
                    shapes.add(_key(p["target"]["raw"]))
                    ballots.add(tuple(sorted(p["votes"].values(), reverse=True)))
                    loads.add(p["total"])
                    destination_counts.add(p["target"]["count"])
                    source_counts.add(len(p["selected"]))
    assert len(signatures) == 42
    assert len(dimensions) >= 30 and len(layouts) >= 100
    assert len(shapes) >= 20 and len(palettes) >= 30 and len(pitches) >= 4
    assert class_counts == {1, 2, 3} and {1, 2, 3} <= source_counts
    assert {1, 2, 3} <= destination_counts and len(loads) >= 5
    assert {(1,), (2, 1), (3, 2, 2)} <= ballots and len(ballots) >= 6
    original = generate_puzzle(0, 2, 1)
    assert _fingerprint(original) == _fingerprint(generate_puzzle(0, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(913, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    data = _parse(original["train"][0]["input"])
    plan = _plans(data)[0]
    source = plan["selected"][0]
    target = plan["target"]
    defects = [(source["cells"][0][0], source["color"]),
               (target["cells"][0][0], data["background"]),
               ((target["bottom"]-target["count"]*data["pitch"]+target["raw"][0][0],
                 target["x"]+target["raw"][0][1]), data["background"]),
               ((target["top"], target["left"]), data["background"])]
    for cell, color in defects:
        broken = {s: [{r: a.copy() for r, a in pair.items()} for pair in original[s]]
                  for s in original}
        broken["train"][0]["output"][cell] = color
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, legal counts, input-only reconstruction, mutations")
