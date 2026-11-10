"""Edge-fed color reactions: stable arrival, contact aggregation, group deletion.

The two L-shaped ports visibly encode the movement axis, frame-to-edge map,
background, and query. No puzzle parameters are needed by solve().
"""
from itertools import combinations, groupby, permutations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _row_overlap(a, b):
    return max(a["r"], b["r"]) < min(a["r"]+a["h"], b["r"]+b["h"])


def _overlap(a, b):
    return (_row_overlap(a, b)
            and max(a["c"], b["c"]) < min(a["c"]+a["w"], b["c"]+b["w"]))


def _touch(a, b):
    horizontal = (_row_overlap(a, b)
                  and (a["c"]+a["w"] == b["c"] or b["c"]+b["w"] == a["c"]))
    vertical = (max(a["c"], b["c"]) < min(a["c"]+a["w"], b["c"]+b["w"])
                and (a["r"]+a["h"] == b["r"] or b["r"]+b["h"] == a["r"]))
    return horizontal or vertical


def _components(objects):
    unseen = set(range(len(objects)))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        group = [start]
        for i in group:
            adjacent = [j for j in sorted(unseen) if _touch(objects[i], objects[j])]
            unseen.difference_update(adjacent)
            group.extend(adjacent)
        groups.append([objects[i] for i in group])
    return groups


def _ports(view):
    """Recognize the canonical five-row header, rather than a majority color."""
    if view.shape[0] < 8 or view.shape[1] < 11:
        return None
    w = view.shape[1]
    left, right = view[:3, :3], view[:3, w-3:]
    values, counts = np.unique(left, return_counts=True)
    padding = values[counts == 6]
    if len(padding) != 1:
        return None
    bg = int(padding[0])
    query = frozenset(int(c) for c in values if c != bg)
    if len(query) != 3 or sum(int(np.count_nonzero(left == c)) for c in query) != 3:
        return None
    rv, rc = np.unique(right, return_counts=True)
    if {int(c): int(n) for c, n in zip(rv, rc)} != {bg: 6, **{c: 1 for c in query}}:
        return None
    frames = (int(view[3, 3]), int(view[3, w-4]))
    if len({bg, *frames, *query}) != 6:
        return None
    expected = np.full((5, w), bg, dtype=np.int8)
    expected[:3, :3] = left
    expected[:3, w-3:] = right
    expected[3, :4] = expected[:4, 3] = frames[0]
    expected[3, w-4:] = expected[:4, w-4] = frames[1]
    if not np.array_equal(expected, view[:5]):
        return None
    return bg, frames, query


def _decode(input_grid):
    grid = _grid(input_grid)
    choices = []
    for rotation in range(4):
        view = np.rot90(grid, rotation)
        port = _ports(view)
        if port is not None:
            choices.append((rotation, view.copy(), port))
    if len(choices) != 1:
        raise ValueError("Exactly one pair of opposed L ports is required")
    rotation, view, (bg, frames, query) = choices[0]
    covered = view == bg
    covered[:5] = True
    objects = []
    h, w = view.shape
    for r in range(5, h):
        for c in range(w):
            if covered[r, c]:
                continue
            frame = int(view[r, c])
            if frame not in frames or r+2 >= h or c+2 >= w:
                raise ValueError("Unframed cargo or incomplete rectangle")
            bottom, right = r+2, c+2
            while bottom < h and view[bottom, c+1] != frame:
                bottom += 1
            while right < w and view[r+1, right] != frame:
                right += 1
            if bottom == h or right == w:
                raise ValueError("Open cargo frame")
            patch = view[r:bottom+1, c:right+1]
            inside = patch[1:-1, 1:-1]
            if (not np.all(patch[0] == frame) or not np.all(patch[-1] == frame)
                    or not np.all(patch[:, 0] == frame) or not np.all(patch[:, -1] == frame)
                    or np.any(np.isin(inside, (bg, *frames)))
                    or np.any(covered[r:bottom+1, c:right+1])):
                raise ValueError("Cargo must have a complete frame and a solid distinguishable interior")
            covered[r:bottom+1, c:right+1] = True
            objects.append(dict(id=len(objects), r=r, c=c, h=bottom-r+1,
                                w=right-c+1, d=frames.index(frame), frame=frame,
                                patch=patch.copy(),
                                labels=frozenset(int(x) for x in np.unique(inside))))
    if not np.all(covered) or not objects:
        raise ValueError("Unaccounted cells or no cargo")
    return dict(rotation=rotation, view=view, bg=bg, frames=frames,
                query=query, objects=objects)


def _run(objects, width, query, *, reaction="group", packing="stable", batching="batch"):
    """Insert equal-distance batches; erase completed contact components now.

    The alternative modes exist only to test concrete shortcut counterexamples.
    Generation, solving and intended validation always use group/stable.
    """
    survivors, events = [], []
    for direction in (0, 1):
        group = [o for o in objects if o["d"] == direction]
        def distance(o):
            return o["c"] if direction == 0 else width-o["c"]-o["w"]
        group.sort(key=lambda o: (distance(o), o["r"]))
        settled = []
        batches = [(lead, list(batch)) for lead, batch in groupby(group, key=distance)]
        if batching == "serial":
            batches = [(distance(o), [o]) for o in group]
        elif batching != "batch":
            raise ValueError("Unknown arrival mode")
        for lead, batch in batches:
            arrivals = []
            before = [dict(o) for o in settled]
            for original in batch:
                o = dict(original)
                blockers = [p for p in settled if _row_overlap(o, p)]
                if packing == "stable":
                    o["c"] = (max((p["c"]+p["w"] for p in blockers), default=0)
                              if direction == 0 else
                              min((p["c"] for p in blockers), default=width)-o["w"])
                elif packing == "edge_first":
                    columns = range(width-o["w"]+1)
                    if direction:
                        columns = reversed(columns)
                    for col in columns:
                        o["c"] = col
                        if not any(_overlap(o, p) for p in settled):
                            break
                    else:
                        raise ValueError("No placement")
                elif packing == "source":
                    o["c"] = original["c"]
                else:
                    raise ValueError("Unknown packing mode")
                if not 0 <= o["c"] <= width-o["w"]:
                    raise ValueError("Intermediate cargo exceeds the field")
                if any(_overlap(o, p) for p in settled+arrivals):
                    raise ValueError("Cargo overlap within an arrival batch")
                arrivals.append(o)
            settled.extend(arrivals)
            components = _components(settled)
            if reaction == "group":
                erased = [component for component in components
                          if query <= frozenset().union(*(o["labels"] for o in component))]
            elif reaction == "individual":
                erased = [[o] for o in settled if query <= o["labels"]]
            elif reaction == "none":
                erased = []
            else:
                raise ValueError("Unknown reaction mode")
            gone = {o["id"] for component in erased for o in component}
            settled = [o for o in settled if o["id"] not in gone]
            events.append(dict(d=direction, distance=lead, before=before,
                               arrivals=arrivals, erased=erased,
                               after=[dict(o) for o in settled]))
        survivors.extend(settled)
    if any(_overlap(a, b) for i, a in enumerate(survivors) for b in survivors[i+1:]):
        raise ValueError("Opposite final groups overlap")
    return survivors, events


def _render(info, survivors):
    output = np.full_like(info["view"], info["bg"])
    output[:5] = info["view"][:5]
    for o in survivors:
        output[o["r"]:o["r"]+o["h"], o["c"]:o["c"]+o["w"]] = o["patch"]
    return np.rot90(output, -info["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all conventions and the complete reaction history from this input."""
    info = _decode(input_grid)
    survivors, _ = _run(info["objects"], info["view"].shape[1], info["query"])
    return _render(info, survivors)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _patch(rng, h, w, frame, filler, marks):
    inside = np.full((h-2, w-2), filler, dtype=np.int8)
    if len(marks) > inside.size:
        raise ValueError("Too many requested interior marks")
    positions = rng.choice(inside.size, len(marks), replace=False)
    for pos, mark in zip(positions, rng.permutation(marks)):
        inside.flat[int(pos)] = int(mark)
    patch = np.full((h, w), frame, dtype=np.int8)
    patch[1:-1, 1:-1] = inside
    return patch


def _object(rng, r, lead, h, w, d, width, frames, filler, marks):
    patch = _patch(rng, h, w, frames[d], filler, marks)
    return dict(r=r, c=lead if d == 0 else width-lead-w, h=h, w=w,
                d=d, frame=frames[d], patch=patch,
                labels=frozenset(int(x) for x in np.unique(patch[1:-1, 1:-1])))


def _assign_ids(objects):
    return [dict(o, id=i) for i, o in enumerate(objects)]


def _header(rng, shape, bg, frames, query):
    grid = np.full(shape, bg, dtype=np.int8)
    w = shape[1]
    grid[3, :4] = grid[:4, 3] = frames[0]
    grid[3, w-4:] = grid[:4, w-4] = frames[1]
    for offset in (0, w-3):
        positions = rng.choice(9, 3, replace=False)
        for pos, color in zip(positions, rng.permutation(query)):
            grid[int(pos)//3, offset+int(pos)%3] = color
    return grid


def _pair(rng, shape, bg, frames, query, objects, rotation):
    objects = _assign_ids(objects)
    if any(_overlap(a, b) for i, a in enumerate(objects) for b in objects[i+1:]):
        raise ValueError("Source overlap")
    grid = _header(rng, shape, bg, frames, query)
    for o in objects:
        if not (5 <= o["r"] <= shape[0]-o["h"] and 0 <= o["c"] <= shape[1]-o["w"]):
            raise ValueError("Source outside cargo field")
        grid[o["r"]:o["r"]+o["h"], o["c"]:o["c"]+o["w"]] = o["patch"]
    settled, _ = _run(objects, shape[1], frozenset(query))
    out = np.full(shape, bg, dtype=np.int8)
    out[:5] = grid[:5]
    for o in settled:
        out[o["r"]:o["r"]+o["h"], o["c"]:o["c"]+o["w"]] = o["patch"]
    return {"input": np.rot90(grid, rotation).copy(),
            "output": np.rot90(out, rotation).copy()}


def _collective_witness(rng, bg, frames, filler, query, labels, rotation):
    """Horizontal chain reaction, vertical contact reaction, then late survivors."""
    q = list(rng.permutation(query))
    x = [c for c in labels if c not in query]
    widths = [int(rng.integers(3, 6)) for _ in range(2)] + [int(rng.integers(4, 6)) for _ in range(2)]
    heights = [int(rng.integers(3, 6)) for _ in range(2)] + [int(rng.integers(4, 6))]
    ew, fw = int(rng.integers(3, 6)), int(rng.integers(4, 6))
    eh = fh = 3
    main_height = sum(heights)-2
    second_height = 2*eh+fh
    field_height = main_height+second_height+1
    height = int(rng.integers(5+field_height, min(30, 5+field_height+3)+1))
    width = int(rng.integers(max(24, sum(widths)+4), 31))
    extra_pad = height-5-field_height
    first = 5+int(rng.integers(extra_pad+1))
    if rng.integers(2):
        main_r, other_r = first, first+main_height+1
    else:
        other_r, main_r = first, first+second_height+1
    d = int(rng.integers(2))
    free = width-sum(widths)
    gaps = rng.multinomial(free-3, np.full(5, .2))
    gaps[1:4] += 1
    lead, row = int(gaps[0]), main_r
    objects = []
    for i in range(3):
        ih, iw = heights[i], widths[i]
        marks = [q[i]] * int(rng.integers(1, min((ih-2)*(iw-2), 3)+1))
        objects.append(_object(rng, row, lead, ih, iw, d, width, frames, filler, marks))
        lead += iw+int(gaps[i+1])
        if i < 2:
            row += ih-1
    # The fourth arrival is incomplete and fills the space freed by the first three.
    ih, iw = heights[2], widths[3]
    marks = [q[0], q[1], x[0]]
    if (ih-2)*(iw-2) >= 5:
        marks += [x[1], q[0]]
    objects.append(_object(rng, row, lead, ih, iw, d, width, frames, filler, marks))
    tied_lead = int(rng.integers(width-ew-max(ew, fw)-1))
    late_lead = int(rng.integers(tied_lead+max(ew, fw)+1, width-ew+1))
    e = _object(rng, other_r, tied_lead, eh, ew, 1-d, width, frames, filler, [q[0]])
    f = _object(rng, other_r+eh, tied_lead, fh, fw, 1-d, width, frames, filler, [q[1], q[2]])
    g = dict(e, r=other_r+eh+fh, patch=e["patch"].copy())
    late = dict(e, c=late_lead if 1-d == 0 else width-late_lead-ew, patch=e["patch"].copy())
    objects.extend((e, f, g, late))
    # Optional neutral cargo varies contact branches without changing the witnesses.
    for _ in range(int(rng.integers(3))):
        for attempt in range(60):
            ih, iw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            o = _object(rng, int(rng.integers(5, height-ih+1)),
                        int(rng.integers(width-iw+1)), ih, iw, int(rng.integers(2)),
                        width, frames, filler, [])
            if any(_overlap(o, p) for p in objects):
                continue
            proposed = _assign_ids(objects+[o])
            try:
                settled, events = _run(proposed, width, frozenset(query))
            except ValueError:
                continue
            survivors = {p["id"]: p for p in settled}
            features = _features({"query": frozenset(query), "objects": proposed}, settled, events)
            tied_reaction = any({4, 5, 6} <= {p["id"] for p in component}
                                and {4, 5, 6} <= {p["id"] for p in event["arrivals"]}
                                for event in events for component in event["erased"])
            chain_reaction = any({0, 1, 2} <= {p["id"] for p in component}
                                 and 2 in {p["id"] for p in event["arrivals"]}
                                 for event in events for component in event["erased"])
            if (features["vertical"] and features["timing"] and tied_reaction and chain_reaction
                    and 3 in survivors and 7 in survivors
                    and all(survivors[i]["c"] == (0 if survivors[i]["d"] == 0 else width-survivors[i]["w"])
                            for i in (3, 7))):
                objects.append(o)
                break
    return _pair(rng, (height, width), bg, frames, query, objects, rotation)


def _identity_witness(rng, bg, frames, filler, query, labels, rotation):
    """Stable surviving overlap chain, matched membership records, richer negatives."""
    q = list(rng.permutation(query))
    x = [c for c in labels if c not in query]
    height, width = int(rng.integers(29, 31)), int(rng.integers(22, 31))
    top = 5+int(rng.integers(height-28))
    if rng.integers(2):
        chain_r, records_r = top, top+11
    else:
        records_r, chain_r = top, top+14
    d = int(rng.integers(2))
    wa = int(rng.integers(4, 6))
    # The third rectangle fits in the false edge-first gap beside the first;
    # this makes stable order visibly necessary, even with unequal widths.
    wc = int(rng.integers(4, wa+1))
    widths = [wa, 5, wc]
    free = width-sum(widths)
    gaps = rng.multinomial(free-2, np.full(4, .25))
    gaps[1:3] += 1
    lead = int(gaps[0])
    objects = []
    marks = [[q[0], q[1], x[0]], [q[0], q[1], *x],
             [q[0], q[1], q[0], x[0]]]
    reverse = bool(rng.integers(2))
    for i, iw in enumerate(widths):
        row = chain_r+3*(2-i if reverse else i)
        objects.append(_object(rng, row, lead, 4, iw, d, width, frames, filler, marks[i]))
        lead += iw+int(gaps[i+1])
    # Opposite class, separated row bands. Complete records arrive before
    # incomplete records in each band, so their deletion cannot consume them.
    for band, (positive_marks, negative_marks, positive_w, negative_w) in enumerate((
            (q, [q[1], q[2], x[0]], wa, wa),
            (q+[x[0], q[0], x[0]], [q[0], q[2], x[1]], 6, int(rng.integers(4, 6))))):
        row = records_r+band*5
        first_lead = int(rng.integers(3))
        second_min = first_lead+positive_w+1
        second_lead = int(rng.integers(second_min, width-negative_w+1))
        objects.append(_object(rng, row, first_lead, 4, positive_w, 1-d, width, frames, filler, positive_marks))
        objects.append(_object(rng, row, second_lead, 4, negative_w, 1-d, width, frames, filler, negative_marks))
    iw = int(rng.integers(3, 7))
    objects.append(_object(rng, records_r+10, int(rng.integers(width-iw+1)),
                           3, iw, 1-d, width, frames, filler, []))
    return _pair(rng, (height, width), bg, frames, query, objects, rotation)


def _random_marks(rng, query, labels, area):
    q = list(query)
    x = [c for c in labels if c not in query]
    positive = area >= 3 and rng.random() < .28
    used = list(rng.permutation(q)[:3 if positive else int(rng.integers(3))])
    used += list(rng.permutation(x)[:int(rng.integers(4))])
    while len(used) > area:
        used.pop()
    if not used:
        return []
    target = int(rng.integers(len(used), max(len(used), int(area*rng.uniform(.2, .85)))+1))
    marks = [int(c) for c in used]
    while len(marks) < target:
        marks.append(int(_pick(rng, used)))
    return marks


def _free_pair(rng, bg, frames, filler, query, labels, rotation, testing):
    for attempt in range(60):
        height = int(rng.integers(13, 31))
        width = int(rng.integers(14, 31))
        target = int(rng.integers(1, 13 if testing else 10))
        objects = []
        bias = float(_pick(rng, (0., .2, .5, .5, .8, 1.)))
        for _ in range(target):
            for placement in range(45):
                ih, iw = int(rng.integers(3, 8)), int(rng.integers(3, 8))
                row = int(rng.integers(5, height-ih+1))
                col = int(rng.integers(width-iw+1))
                if objects and rng.random() < .45:
                    p = _pick(rng, objects)
                    row = int(_pick(rng, (p["r"], p["r"]+p["h"]-1, p["r"]+p["h"], p["r"]-ih)))
                    if rng.random() < .3:
                        col = int(_pick(rng, (p["c"]+p["w"], p["c"]-iw)))
                if not (5 <= row <= height-ih and 0 <= col <= width-iw):
                    continue
                d = int(rng.random() >= bias)
                marks = _random_marks(rng, query, labels, (ih-2)*(iw-2))
                o = _object(rng, row, col if d == 0 else width-col-iw,
                            ih, iw, d, width, frames, filler, marks)
                if not any(_overlap(o, p) for p in objects):
                    objects.append(o)
                    break
        if objects:
            try:
                return _pair(rng, (height, width), bg, frames, query, objects, rotation)
            except ValueError:
                pass
    # A bounded constructive fallback still obeys the same visible rule.
    o = _object(rng, 6, 3, 4, 5, int(rng.integers(2)), 17, frames, filler,
                list(query) if rng.integers(2) else [query[0]])
    return _pair(rng, (12, 17), bg, frames, query, [o], rotation)


def _distinct_pair(pair, seen):
    """Guaranteed uniqueness by changing a semantically irrelevant port pattern."""
    if _key(pair["input"]) not in seen:
        return pair
    info = _decode(pair["input"])
    view = info["view"].copy()
    out = np.rot90(pair["output"], info["rotation"]).copy()
    for positions in combinations(range(9), 3):
        for colors in permutations(sorted(info["query"])):
            view[:3, :3] = info["bg"]
            for p, color in zip(positions, colors):
                view[p//3, p%3] = color
            candidate = np.rot90(view, -info["rotation"]).copy()
            if _key(candidate) not in seen:
                out[:3, :3] = view[:3, :3]
                return dict(input=candidate, output=np.rot90(out, -info["rotation"]).copy())
    raise AssertionError("At most 13 earlier inputs cannot exhaust 504 port patterns")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    colors = [int(c) for c in rng.permutation(10)]
    frames, neutral, labels = tuple(colors[:2]), tuple(colors[2:4]), tuple(colors[4:])
    rotation = int(rng.integers(4))
    vary_background = bool(rng.integers(2))
    triples = list(combinations(labels, 3))
    train_queries = [_pick(rng, triples) for _ in range(num_train)]
    # The identity witness exposes all six label identities, so tests can
    # recombine familiar identities using a query triple absent from training.
    available = [q for q in triples if q not in train_queries]
    order = rng.permutation(len(available))
    test_queries = [available[int(i)] for i in order[:num_test]]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, queries in (("train", train_queries), ("test", test_queries)):
        for i, query in enumerate(queries):
            swap = (i % 2 if split == "train" and i < 2 else int(rng.integers(2))) if vary_background else 0
            bg, filler = neutral[swap], neutral[1-swap]
            if split == "train" and i == 0:
                pair = _collective_witness(rng, bg, frames, filler, query, labels, rotation)
            elif split == "train" and i == 1:
                pair = _identity_witness(rng, bg, frames, filler, query, labels, rotation)
            else:
                pair = _free_pair(rng, bg, frames, filler, query, labels, rotation, split == "test")
            pair = _distinct_pair(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    # The teaching packages do not occupy predictable demonstration indices.
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, survivors, events):
    gone = {o["id"] for e in events for group in e["erased"] for o in group}
    groups = [g for e in events for g in e["erased"]]
    by_id = {o["id"]: o for o in info["objects"]}
    query = info["query"]
    collective = [g for g in groups if len(g) > 1 and all(not query <= o["labels"] for o in g)]
    timing = any(a["id"] in gone and b["id"] not in gone and a["d"] == b["d"]
                 and np.array_equal(a["patch"], b["patch"])
                 for a in info["objects"] for b in info["objects"])
    return dict(gone=gone, groups=groups, collective=collective, timing=timing,
                vertical=any(any(not _row_overlap(a, b) and _touch(a, b)
                                 for i, a in enumerate(g) for b in g[i+1:]) for g in collective),
                survivor_ids={o["id"] for o in survivors}, by_id=by_id)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    conventions, queries = set(), {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "Duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            conventions.add((info["rotation"], info["frames"]))
            queries[split].add(info["query"])
            width = info["view"].shape[1]
            survivors, events = _run(info["objects"], width, info["query"])
            assert np.array_equal(_render(info, survivors), out), "Incorrect reaction output"
            canonical_out = np.rot90(out, info["rotation"])
            assert np.array_equal(canonical_out[:5], info["view"][:5]), "Ports changed"
            original = {o["id"]: o for o in info["objects"]}
            for event in events:
                for o in event["arrivals"]:
                    a = original[o["id"]]
                    assert o["r"] == a["r"] and np.array_equal(o["patch"], a["patch"])
                    assert o["c"] <= a["c"] if o["d"] == 0 else o["c"] >= a["c"]
                    blockers = [p for p in event["before"] if _row_overlap(o, p)]
                    boundary = (max((p["c"]+p["w"] for p in blockers), default=0) if o["d"] == 0
                                else min((p["c"] for p in blockers), default=width)-o["w"])
                    assert o["c"] == boundary, "Unstable packing or unused free space"
                for group in event["erased"]:
                    assert info["query"] <= frozenset().union(*(o["labels"] for o in group))
                    assert len(_components(group)) == 1, "Deletion split a contact component"
                for group in _components(event["after"]):
                    assert not info["query"] <= frozenset().union(*(o["labels"] for o in group))
            if split == "train":
                training.append((info, survivors, events, _features(info, survivors, events)))
    assert len(conventions) == 1, "Axis/frame map changed within a puzzle"
    assert queries["test"].isdisjoint(queries["train"]), "Test query should recombine familiar labels"
    assert any(f["collective"] and f["timing"] and f["vertical"] for _, _, _, f in training)
    assert any(any(len({o["id"] for o in group} & {o["id"] for o in event["arrivals"]}) >= 3
                       and all(not info["query"] <= o["labels"] for o in group)
                       for group in event["erased"])
               for info, _, events, _ in training for event in events), "No simultaneous-batch witness"
    records = [(info, o, o["id"] in f["gone"]) for info, _, _, f in training for o in info["objects"]]
    assert any(erased and info["query"] <= o["labels"] for info, o, erased in records)
    for info, survivors, _, _ in training:
        if all(any(o["labels"] & info["query"] == info["query"]-{q} for o in survivors) for q in info["query"]):
            break
    else:
        raise AssertionError("Each query identity needs a surviving missing-identity witness")
    train_labels = frozenset().union(*(o["labels"] for _, o, _ in records))
    assert all(q <= train_labels for q in queries["test"])
    positives = [(info, o) for info, o, erased in records if erased and info["query"] <= o["labels"]]
    negatives = [(info, o) for info, o, erased in records if not erased]
    assert any(len(o["labels"]-info["query"]) >= 3 for info, o in negatives), "No colorful incomplete survivor"
    assert any(not (o["labels"] & info["query"]) for info, o in negatives), "No neutral survivor"
    def histogram(o):
        return sorted(int(n) for n in np.unique(o["patch"][1:-1, 1:-1], return_counts=True)[1])
    assert any(a["patch"].shape == b["patch"].shape and len(a["labels"]) == len(b["labels"])
               and histogram(a) == histogram(b)
               and all(np.count_nonzero(a["patch"][1:-1, 1:-1] == q) == 1 for q in ai["query"])
               and a["labels"] & ai["query"] == ai["query"]
               and len(b["labels"] & bi["query"]) == 2
               for ai, a in positives for bi, b in negatives), "No matched identity contrast"
    witnesses = {"packing": False, "order": False, "selection": False,
                 "individual": False, "late_filter": False, "serial_ties": False}
    for info, survivors, _, _ in training:
        actual = _render(info, survivors)
        for name, options in (("packing", {"packing": "source"}),
                              ("order", {"packing": "edge_first"}),
                              ("serial_ties", {"batching": "serial"}),
                              ("selection", {"reaction": "none"}),
                              ("individual", {"reaction": "individual"})):
            try:
                alternative, _ = _run(info["objects"], info["view"].shape[1], info["query"], **options)
                witnesses[name] |= not np.array_equal(actual, _render(info, alternative))
            except ValueError:
                witnesses[name] = True
        try:
            all_packed, _ = _run(info["objects"], info["view"].shape[1], info["query"], reaction="none")
            late = []
            for d in (0, 1):
                for component in _components([o for o in all_packed if o["d"] == d]):
                    if not info["query"] <= frozenset().union(*(o["labels"] for o in component)):
                        late.extend(component)
            witnesses["late_filter"] |= not np.array_equal(actual, _render(info, late))
        except ValueError:
            witnesses["late_filter"] = True
    assert all(witnesses.values()), "Training permits a parent-omitting shortcut"


def measure_puzzle(puzzle):
    result = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _decode(pair["input"])
            survivors, events = _run(info["objects"], info["view"].shape[1], info["query"])
            f = _features(info, survivors, events)
            result.append(dict(split=split, shape=list(np.asarray(pair["input"]).shape),
                               canonical_shape=list(info["view"].shape), rotation=info["rotation"],
                               frames=list(info["frames"]), background=info["bg"], query=sorted(info["query"]),
                               objects=len(info["objects"]), survivors=len(survivors), deleted=len(f["gone"]),
                               reactions=len(f["groups"]), collective=len(f["collective"]),
                               vertical_reaction=f["vertical"], timing_witness=f["timing"],
                               footprints=sorted({(o["h"], o["w"]) for o in info["objects"]}),
                               classes=[sum(o["d"] == d for o in info["objects"]) for d in (0, 1)],
                               ties=sum(len(e["arrivals"]) > 1 for e in events),
                               travels=[abs(o["c"]-f["by_id"][o["id"]]["c"]) for e in events for o in e["arrivals"]],
                               layout=sorted((o["r"], o["c"], o["h"], o["w"], o["d"]) for o in info["objects"])))
    return result


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    # Hand-specified output: three staggered arrivals react before a fourth,
    # while three tied vertical arrivals react before an identical late copy.
    fixture_rng = np.random.default_rng(501)
    frames, query, filler, width = (8, 2), (1, 3, 4), 7, 28
    cargo = [_object(fixture_rng, r, lead, h, w, d, width, frames, filler, marks)
             for r, lead, h, w, d, marks in (
                 (5, 2, 4, 4, 0, [1]), (8, 8, 4, 4, 0, [3]),
                 (11, 14, 4, 4, 0, [4]), (11, 20, 4, 4, 0, [1, 3]),
                 (17, 2, 3, 4, 1, [1]), (20, 2, 3, 4, 1, [3, 4]),
                 (23, 2, 3, 4, 1, [1]), (17, 10, 3, 4, 1, [1]))]
    cargo[7]["patch"] = cargo[4]["patch"].copy()
    source = _header(fixture_rng, (30, width), 0, frames, query)
    expected = source.copy()
    for o in cargo:
        source[o["r"]:o["r"]+o["h"], o["c"]:o["c"]+o["w"]] = o["patch"]
    for index, col in ((3, 0), (7, width-4)):
        o = cargo[index]
        expected[o["r"]:o["r"]+o["h"], col:col+o["w"]] = o["patch"]
    for rotation in range(4):
        palette = fixture_rng.permutation(10)
        assert np.array_equal(solve(np.rot90(palette[source], rotation)),
                              np.rot90(palette[expected], rotation))
    a = dict(r=5, c=0, h=3, w=3)
    assert _touch(a, dict(r=8, c=1, h=3, w=3))
    assert not _touch(a, dict(r=8, c=3, h=3, w=3)), "Corners are not side contacts"
    corpus = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in corpus}) == 42
    for seed in (0, 13, 41):
        assert _fingerprint(corpus[seed]) == _fingerprint(generate_puzzle(seed))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(12345+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    rows = [r for p in corpus for r in measure_puzzle(p)]
    assert {r["rotation"] for r in rows} == set(range(4))
    assert len({tuple(r["frames"]) for r in rows}) >= 25
    assert len({tuple(r["shape"]) for r in rows}) >= 50
    assert len({r["objects"] for r in rows}) >= 8
    assert len({s for r in rows for s in r["footprints"]}) >= 15
    assert any(r["objects"] == 1 for r in rows)
    assert any(r["deleted"] == 0 for r in rows)
    assert any(r["survivors"] == 0 for r in rows)
    assert any(r["ties"] for r in rows)
    assert {bool(r["vertical_reaction"]) for r in rows} == {False, True}
    assert max(t for r in rows for t in r["travels"]) >= 20
    for puzzle in corpus:
        measures = measure_puzzle(puzzle)
        assert len({tuple(r["layout"]) for r in measures}) == len(measures)
        assert len({tuple(r["query"]) for r in measures}) > 1
    # Selector-only counterfactual and query-position invariance, for every seed.
    for puzzle in corpus:
        for pair in puzzle["train"]:
            info = _decode(pair["input"])
            survivors, _ = _run(info["objects"], info["view"].shape[1], info["query"])
            candidates = [o for o in survivors if len(o["labels"]) == 4 and len(o["labels"] & info["query"]) == 2]
            if candidates:
                break
        assert candidates
        new_query = sorted(candidates[0]["labels"] & info["query"]) + sorted(candidates[0]["labels"]-info["query"])[:1]
        changed = info["view"].copy()
        for offset in (0, changed.shape[1]-3):
            port = changed[:3, offset:offset+3]
            positions = np.flatnonzero(port.ravel() != info["bg"])
            for pos, color in zip(positions, new_query):
                port[int(pos)//3, int(pos)%3] = color
        changed_input = np.rot90(changed, -info["rotation"]).copy()
        changed_info = _decode(changed_input)
        changed_survivors, _ = _run(changed_info["objects"], changed.shape[1], changed_info["query"])
        assert candidates[0]["id"] not in {o["id"] for o in changed_survivors}
        predicted = np.rot90(solve(changed_input), info["rotation"])
        original_output = np.rot90(pair["output"], info["rotation"])
        assert not np.array_equal(predicted[5:], original_output[5:])
        moved = info["view"].copy()
        expected = np.rot90(pair["output"], info["rotation"]).copy()
        for offset in (0, moved.shape[1]-3):
            moved[:3, offset:offset+3] = np.roll(moved[:3, offset:offset+3].copy(), 1, axis=0)
        expected[:5] = moved[:5]
        assert np.array_equal(solve(np.rot90(moved, -info["rotation"])), np.rot90(expected, -info["rotation"]))
    # Reject partial deletion/payload damage, changed ports, and duplicate inputs.
    for corruption in ("output", "port", "duplicate"):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in corpus[0][s]] for s in corpus[0]}
        if corruption == "duplicate":
            bad["train"][1] = {k: v.copy() for k, v in bad["train"][0].items()}
        else:
            pair = bad["train"][0]
            info = _decode(pair["input"])
            view = np.rot90(pair["output"], info["rotation"])
            if corruption == "port":
                view[3, 0] = info["bg"]
            else:
                view[5, 0] = (int(view[5, 0])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("Corruption accepted")
    return dict(seeds=42, pairs=len(rows), shapes=len({tuple(r["shape"]) for r in rows}),
                counts=sorted({r["objects"] for r in rows}),
                footprints=len({s for r in rows for s in r["footprints"]}),
                reactions=sum(r["reactions"] for r in rows),
                collective_reactions=sum(r["collective"] for r in rows))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
