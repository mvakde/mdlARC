"""Rewired color drains: topology defines the records of a color-set filter.

Only NumPy and the standard library are used. No examples, seeds, or external
files are consulted by solve. Running this file executes the bounded self-test.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _paths(mask):
    """Return ordered, monotone, simple four-neighbor paths; reject other graphs."""
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        component, pending = {start}, [start]
        remaining.remove(start)
        for r, c in pending:
            for dr, dc in _DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    pending.append(q)
        adjacency = {p: [q for dr, dc in _DIRS
                         if (q := (p[0] + dr, p[1] + dc)) in component]
                     for p in component}
        ends = sorted(p for p in component if len(adjacency[p]) == 1)
        assert len(ends) == 2 and all(len(v) in (1, 2) for v in adjacency.values())
        ordered, previous = [ends[0]], None
        while ordered[-1] != ends[1]:
            current = ordered[-1]
            q, = [p for p in adjacency[current] if p != previous]
            ordered.append(q)
            previous = current
        assert len(ordered) == len(component)
        assert len({b[0] - a[0] for a, b in zip(ordered, ordered[1:])
                    if a[0] != b[0]}) <= 1, "Internal gravity reversal"
        result.append(ordered)
    return result


def _rewire(core, bg, rail, bar, frame):
    """Simultaneous local writes; labels are forbidden inside these tiles."""
    h, w = core.shape
    assert h >= 3 and h % 2 == 1 and w >= 9
    assert len({bg, rail, bar, frame}) == 4 and frame not in core
    assert not np.any(core[::2] == bar)
    assert np.all(np.isin(core[1::2], (bg, bar)))
    centers, writes, overlaps = [], {}, 0
    for r in range(1, h, 2):
        columns = list(map(int, np.flatnonzero(core[r] == bar)))
        runs = []
        for c in columns:
            if not runs or c != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(c)
        for run in runs:
            assert len(run) == 3
            c = run[1]
            assert 2 <= c <= w - 3
            assert np.all(core[r-1, c-1:c+2] == rail)
            assert np.all(core[r+1, c-1:c+2] == rail)
            centers.append((r, c))
            for rr in range(r-1, r+2):
                for cc in range(c-1, c+2):
                    value = frame if cc == c else rail
                    if (rr, cc) in writes:
                        assert writes[rr, cc] == value, "Conflicting rotations"
                        overlaps += 1
                    writes[rr, cc] = value
    assert centers
    breaks = []
    for r in range(0, h, 2):
        xs = list(map(int, np.flatnonzero(core[r] == bg)))
        assert all(2 <= c <= w-3 for c in xs)
        assert all(b-a >= 3 for a, b in zip(xs, xs[1:]))
        breaks.extend((r, c) for c in xs)
    assert not set(breaks) & set(writes)
    structural = core.copy()
    structural[~np.isin(structural, (bg, rail, bar))] = rail
    for p, color in writes.items():
        structural[p] = color
    routes = _paths(structural == rail)
    for route in routes:
        for r, c in (route[0], route[-1]):
            assert r % 2 == 0
            assert c in (0, w-1) or core[r, c-1] == bg or core[r, c+1] == bg
    slots = [[p for p in route if p not in writes and core[p] != bg]
             for route in routes]
    return {"structural": structural, "routes": routes, "slots": slots,
            "writes": writes, "centers": centers, "breaks": breaks,
            "overlaps": overlaps, "bg": bg, "rail": rail, "bar": bar,
            "frame": frame}


def _read(input_grid):
    """Recover the arrow orientation, local palette, crop and route graph."""
    grid = _array(input_grid)
    candidates = []
    for rotation in range(4):
        view = np.rot90(grid, rotation)
        h, w = view.shape
        if h < 8 or w < 9:
            continue
        frame = int(view[3, 3])
        if not (np.all(view[3, :4] == frame) and np.all(view[:4, 3] == frame)):
            continue
        colors, counts = np.unique(view[:3, :3], return_counts=True)
        padding = colors[counts == 6]
        if len(padding) != 1:
            continue
        bg = int(padding[0])
        query = frozenset(map(int, colors[colors != bg]))
        if len(query) != 3 or frame in query or bg == frame:
            continue
        if np.count_nonzero(view == frame) != 10:
            continue
        extra = [(int(r), int(c)) for r, c in np.argwhere(view[:4] == frame)
                 if c >= 4]
        if len(extra) != 3:
            continue
        r, c = min(extra)
        if c < 5 or set(extra) != {(r, c), (r, c+2), (r+1, c+1)}:
            continue
        header = np.full((4, w), bg, dtype=np.int8)
        header[:4, :4] = view[:4, :4]
        for p in extra:
            header[p] = frame
        if not np.array_equal(header, view[:4]):
            continue
        points = np.argwhere(view[4:] != bg)
        if not len(points):
            continue
        low = points.min(axis=0) + (4, 0)
        high = points.max(axis=0) + (5, 1)
        r0, c0 = map(int, low)
        r1, c1 = map(int, high)
        if r0 < 5:
            continue
        core = view[r0:r1, c0:c1]
        if core.shape[0] < 3 or core.shape[0] % 2 != 1:
            continue
        dividers = set(map(int, np.unique(core[1::2]))) - {bg}
        if len(dividers) != 1:
            continue
        bar = next(iter(dividers))
        values, amounts = np.unique(core[::2], return_counts=True)
        modes = values[amounts == amounts.max()]
        if len(modes) != 1:
            continue
        rail = int(modes[0])
        if len({bg, rail, bar, frame}) != 4 or query & {bg, rail, bar, frame}:
            continue
        try:
            info = _rewire(core, bg, rail, bar, frame)
        except AssertionError:
            continue
        labels = set(map(int, np.unique(core))) - {bg, rail, bar, frame}
        if not labels <= set(range(10)) - {bg, rail, bar, frame}:
            continue
        info.update(query=query, labels=labels, core=core,
                    box=(r0, r1, c0, c1), rotation=rotation, view=view)
        candidates.append(info)
    assert len(candidates) == 1, "Input must have one unambiguous selector/network"
    return candidates[0]


def _evaluate(core, info, query, *, keep_all=False):
    out = info["structural"].copy()
    bg, rail, bar, frame = (info[k] for k in ("bg", "rail", "bar", "frame"))
    records = []
    for path in info["routes"]:
        # A path is ordered from the lower endpoint inward. On a level path
        # this order is only for inspection: every packet stays at its cell.
        inward = path[::-1] if path[-1][0] > path[0][0] else path
        packets = [(p, int(core[p])) for p in inward
                   if int(core[p]) not in (bg, rail, bar, frame)]
        colors = frozenset(color for _, color in packets)
        positive = query <= colors
        if positive and not keep_all:
            for p in path:
                out[p] = bg
        elif path[0][0] == path[-1][0]:
            for p, color in packets:
                out[p] = color
        else:
            for p, (_, color) in zip(inward, packets):
                out[p] = color
        records.append({"path": path, "inward": inward, "packets": packets,
                        "colors": colors, "positive": positive,
                        "level": path[0][0] == path[-1][0]})
    return out, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    core, _ = _evaluate(info["core"], info, info["query"])
    answer = info["view"].copy()
    r0, r1, c0, c1 = info["box"]
    answer[r0:r1, c0:c1] = core
    return np.rot90(answer, -info["rotation"]).copy()


def _base(lanes, width, centers, breaks, palette):
    bg, rail, bar, frame = palette[:4]
    core = np.full((2*lanes-1, width), bg, dtype=np.int8)
    core[::2] = rail
    for r, c in centers:
        core[r, c-1:c+2] = bar
    for p in breaks:
        core[p] = bg
    return core, _rewire(core, bg, rail, bar, frame)


def _layout(rng, lanes, width, package, extra_cuts):
    palette = (0, 1, 2, 3)
    for attempt in range(80):
        centers, previous = [], []
        for r in range(1, 2*lanes-1, 2):
            target = int(rng.choice((0, 1, 1, 2, 2, 3, 4)))
            selected = []
            for c in rng.permutation(np.arange(2, width-2)):
                c = int(c)
                if len(selected) >= target:
                    break
                if all(abs(c-x) >= 4 for x in selected) and all(abs(c-x) >= 2 for x in previous):
                    selected.append(c)
            centers.extend((r, c) for c in sorted(selected))
            previous = selected
        if not centers:
            continue
        breaks = []
        for r in range(0, 2*lanes-1, 2):
            ports = sorted((c, s) for s, c in centers if abs(s-r) == 1)
            for (a, s), (b, t) in zip(ports, ports[1:]):
                if s == t:
                    breaks.append((r, int(rng.integers(a+2, b-1))))
        for _ in range(extra_cuts):
            candidates = [(r, c) for r in range(0, 2*lanes-1, 2)
                          for c in range(2, width-2)
                          if all(abs(c-x) >= 2 for s, x in centers if abs(s-r) == 1)
                          and all(abs(c-x) >= 3 for s, x in breaks if s == r)]
            if candidates:
                breaks.append(candidates[int(rng.integers(len(candidates)))])
        try:
            _, info = _base(lanes, width, centers, breaks, palette)
        except AssertionError:
            continue
        slots, routes = info["slots"], info["routes"]
        cross = [i for i, s in enumerate(slots) if len(s) >= 8 and len({r for r, c in s}) >= 2]
        large = [i for i, s in enumerate(slots) if len(s) >= 8]
        cross_four = [s for s in slots if len(s) >= 4 and len({r for r, c in s}) >= 2]
        if package == "identity" and (not cross or len(cross_four) < 2
                                      or sum(len(s) >= 4 for s in slots) < 5):
            continue
        if package == "extras" and (len(cross) < 2 or len(large) < 3 or not any(
                len(s) >= 3 and p[0][0] == p[-1][0] for s, p in zip(slots, routes))):
            continue
        if package == "free" and not cross:
            continue
        return lanes, width, centers, breaks
    # Explicitly bounded, valid rescue network; labels and presentation remain
    # freshly sampled. Usually ordinary sampling succeeds in a few attempts.
    return 7, 25, [(1, 4), (3, 8), (5, 12), (7, 16), (9, 20)], []


def _paint(rng, core, available, marks, *, split_rows=False):
    assert len(available) >= len(marks)
    positions = list(map(int, rng.choice(len(available), len(marks), replace=False)))
    if split_rows:
        first = positions[0]
        other = [i for i, p in enumerate(available) if p[0] != available[first][0]]
        positions[1] = int(rng.choice(other))
        used = set(positions[:2])
        positions[2:] = list(map(int, rng.choice([i for i in range(len(available)) if i not in used],
                                                len(marks)-2, replace=False)))
    # The first three distinct query marks in an exact positive cannot all lie
    # on a single source row. Shuffling only their identities keeps that fact.
    for position, color in zip(positions, rng.permutation(marks)):
        core[available[position]] = color


def _raw_row_deleted(core, info, query):
    """Cargo deleted by the plausible shortcut that filters original lane segments."""
    removed = set()
    roles = {info[k] for k in ("bg", "rail", "bar", "frame")}
    for r in range(0, core.shape[0], 2):
        segment = []
        for c in range(core.shape[1]+1):
            if c == core.shape[1] or core[r, c] == info["bg"]:
                cargo = [(p, int(core[p])) for p in segment if int(core[p]) not in roles]
                if query <= {color for _, color in cargo}:
                    removed.update(p for p, _ in cargo)
                segment = []
            else:
                segment.append((r, c))
    return removed


def _make_pair(rng, palette, query, rotation, package, shift, test):
    if shift == 0:
        lanes = int(rng.integers(9, 13) if test else rng.integers(5, 8))
    else:
        lanes = int(rng.integers(5, 12))
    width = int(rng.integers(18, 29))
    cuts = int(rng.integers(5, 10) if test and shift == 2 else rng.integers(0, 4))
    lanes, width, centers, breaks = _layout(rng, lanes, width, package, cuts)
    blank, info = _base(lanes, width, centers, breaks, palette)
    q = list(query)
    extras = [c for c in palette[4:] if c not in query]
    routes, slots = info["routes"], info["slots"]
    for paint_attempt in range(24):
        core = blank.copy()
        used = set()

        def choose(minimum, cross=False, level=False):
            options = [i for i, s in enumerate(slots) if i not in used and len(s) >= minimum
                       and (not cross or len({r for r, c in s}) >= 2)
                       and (not level or routes[i][0][0] == routes[i][-1][0])]
            assert options
            i = int(rng.choice(options))
            used.add(i)
            return i

        if package == "identity":
            i = choose(8, cross=True)
            _paint(rng, core, slots[i], q, split_rows=True)
            # At least one incomplete, moving route. The other omissions can
            # occur on level or sloping routes; counts and identities are tied.
            for missing in range(3):
                i = choose(4, cross=(missing == 0))
                marks = [c for j, c in enumerate(q) if j != missing] + [extras[0]]
                _paint(rng, core, slots[i], marks)
            i = choose(4)
            _paint(rng, core, slots[i], q[:2] + extras[:2])
        elif package == "extras":
            # Reserve the level demonstration first so other assignments
            # cannot consume the only suitable level route.
            i = choose(3, level=True)
            _paint(rng, core, slots[i], [q[0], extras[0]])
            i = choose(8, cross=True)
            _paint(rng, core, slots[i], q + extras[:2] + [q[0], extras[0]])
            i = choose(8, cross=True)
            _paint(rng, core, slots[i], q[:2] + extras[:2] + [q[0], q[1], extras[0], extras[1]])
            inward = routes[i][::-1]
            occupied = [p for p in inward if core[p] in palette[4:]]
            sequence = [int(core[p]) for p in occupied]
            if sequence == sorted(sequence) or sequence == sorted(sequence, reverse=True):
                # Numeric sorting must be observably different from stable
                # route order, even in the two-demonstration minimum.
                core[occupied[0]], core[occupied[-1]] = core[occupied[-1]], core[occupied[0]]
            choose(1)  # Deliberately empty route.
        else:
            # Free scenes can be all-match, no-match, or mixed. For the dense
            # shift, more packets, rather than a new operation, reach the test.
            mode = str(rng.choice(("all", "none", "mixed"), p=(.15, .15, .70)))
            for i, available in enumerate(slots):
                used.add(i)
                if not available or rng.random() < .16:
                    continue
                positive = len(available) >= 3 and (mode == "all" or (mode == "mixed" and rng.random() < .5))
                marks = q.copy() if positive else list(rng.permutation(q)[:int(rng.integers(3))])
                marks += list(rng.permutation(extras)[:int(rng.integers(len(extras)+1))])
                if not marks:
                    continue
                marks = marks[:len(available)]
                fraction = .45 if test and shift == 1 else float(rng.uniform(.12, .32))
                count = min(len(available), max(len(marks), int(len(available)*fraction)))
                marks += [int(rng.choice(marks)) for _ in range(count-len(marks))]
                _paint(rng, core, available, marks)
        # Additional unused routes vary independently of their geometry.
        for i, available in enumerate(slots):
            if i not in used and available and rng.random() < .55:
                marks = list(map(int, rng.choice(palette[4:], min(len(available), int(rng.integers(1, 5))), replace=True)))
                _paint(rng, core, available, marks)
        outcome, records = _evaluate(core, info, frozenset(query))
        actual = {p for rec in records if rec["positive"] for p, _ in rec["packets"]}
        if package != "identity" or actual != _raw_row_deleted(core, info, frozenset(query)):
            break
    else:
        raise AssertionError("Could not distinguish original-row membership")
    # A reflection changes routing positions without changing downward gravity.
    if rng.random() < .5:
        core, outcome = np.fliplr(core), np.fliplr(outcome)
    bg, rail, bar, frame = palette[:4]
    top = int(rng.integers(5, min(8, 31-core.shape[0])))
    left = int(rng.integers(min(3, 30-width)+1))
    total_h = min(30, top+core.shape[0]+int(rng.integers(2)))
    total_w = min(30, width+left+int(rng.integers(3)))
    inp = np.full((total_h, total_w), bg, dtype=np.int8)
    inp[top:top+core.shape[0], left:left+width] = core
    inp[3, :4] = frame
    inp[:4, 3] = frame
    for flat, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
        inp[int(flat)//3, int(flat)%3] = color
    ar, ac = int(rng.integers(3)), int(rng.integers(5, total_w-2))
    inp[ar, ac] = inp[ar, ac+2] = inp[ar+1, ac+1] = frame
    out = inp.copy()
    out[top:top+core.shape[0], left:left+width] = outcome
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))[:int(rng.integers(9, 11))]
    vary_palette = bool(rng.integers(2))
    rotation, shift = int(rng.integers(4)), int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            package = ("identity" if index == 0 else "extras" if index == 1 else "free") if split == "train" else "free"
            for attempt in range(20):
                local = list(map(int, rng.permutation(10)))[:len(palette)] if vary_palette else palette
                query = list(map(int, rng.choice(local[4:], 3, replace=False)))
                pair = _make_pair(rng, local, query, rotation, package, shift, split == "test")
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("Distinct-input retry bound reached")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _coverage(infos):
    records = [(info, rec) for info in infos for rec in info["records"]]
    positives = [(i, r) for i, r in records if r["positive"]]
    negatives = [(i, r) for i, r in records if not r["positive"]]
    assert positives and negatives
    assert any(len(r["packets"]) == 3 for _, r in positives)
    assert any(len(r["packets"]) == 3 and len(r["colors"]) == 3 for _, r in negatives)
    assert any(len(r["colors"]) >= 4 for _, r in negatives)
    assert any(not r["packets"] for _, r in negatives)
    assert any(r["level"] and r["packets"] for _, r in negatives)
    assert any(not r["level"] and len(r["packets"]) >= 3 and
               [p for p, _ in r["packets"]] != r["inward"][:len(r["packets"])] for _, r in negatives)
    assert any(not r["level"] and [c for _, c in r["packets"]] != sorted(c for _, c in r["packets"])
               and [c for _, c in r["packets"]] != sorted((c for _, c in r["packets"]), reverse=True)
               for _, r in negatives)
    assert any(r["colors"] - i["query"] and len(r["packets"]) > len(r["colors"])
               for i, r in positives)
    assert any(r["colors"] - i["query"] and len(r["packets"]) > len(r["colors"])
               for i, r in negatives)
    assert any(all(any(not r["positive"] and r["colors"] & i["query"] == i["query"] - {q}
                      for r in i["records"]) for q in i["query"]) for i in infos)
    # This compares only deleted cargo cells: divider-color changes alone
    # cannot supply a spurious witness for the importance of topology.
    assert any({p for r in i["records"] if r["positive"] for p, _ in r["packets"]}
               != _raw_row_deleted(i["core"], i, i["query"]) for i in infos)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            info = _read(inp)
            expected_core, records = _evaluate(info["core"], info, info["query"])
            info["records"] = records
            expected = info["view"].copy()
            r0, r1, c0, c1 = info["box"]
            expected[r0:r1, c0:c1] = expected_core
            assert np.array_equal(np.rot90(out, info["rotation"]), expected)
            for rec in records:
                path, packets = rec["inward"], rec["packets"]
                if rec["positive"]:
                    assert all(expected_core[p] == info["bg"] for p in path)
                else:
                    observed = [(p, int(expected_core[p])) for p in path
                                if int(expected_core[p]) not in {info["bg"], info["rail"], info["frame"]}]
                    assert [c for _, c in observed] == [c for _, c in packets]
                    targets = [p for p, _ in packets] if rec["level"] else path[:len(packets)]
                    assert [p for p, _ in observed] == targets
            assert all(expected_core[p] == color for p, color in info["writes"].items()
                       if color == info["frame"])
            if split == "train":
                training.append(info)
    _coverage(training)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; no sample files, retained puzzles, or execution traces."""
    # A hand-constructed two-lane boundary case, below the sampler's usual
    # teaching sizes. Its expected core is written directly, not via _evaluate.
    small = np.zeros((8, 11), dtype=np.int8)
    small[3, :4] = small[:4, 3] = 3
    small[0, 0], small[1, 1], small[2, 2] = 4, 5, 6
    small[1, 7] = small[1, 9] = small[2, 8] = 3
    small[5] = small[7] = 1
    small[6, 4:7] = 2
    small[5, 1], small[5, 2], small[7, 1] = 4, 5, 6
    small[5, 8], small[5, 9], small[7, 9] = 7, 7, 4
    expected = small.copy()
    expected[5:] = np.array([[0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 3, 1, 1, 7, 7, 4]], dtype=np.int8)
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(small, turn)), np.rot90(expected, turn))
    shapes, layouts, signatures, lanes, turns, cuts, populations, directions = (set() for _ in range(8))
    roles = [set() for _ in range(4)]
    totals = {"pairs": 0, "deleted_routes": 0, "surviving_routes": 0,
              "overlap_pairs": 0, "level_cargo_routes": 0, "ordered_packing_routes": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _read(pair["input"])
                _, records = _evaluate(i["core"], i, i["query"])
                totals["pairs"] += 1
                totals["deleted_routes"] += sum(r["positive"] for r in records)
                totals["surviving_routes"] += sum(not r["positive"] for r in records)
                totals["overlap_pairs"] += bool(i["overlaps"])
                totals["level_cargo_routes"] += sum(r["level"] and bool(r["packets"]) and not r["positive"] for r in records)
                totals["ordered_packing_routes"] += sum(not r["level"] and not r["positive"] and len(r["colors"]) >= 3 for r in records)
                shapes.add(pair["input"].shape)
                lanes.add((i["core"].shape[0]+1)//2)
                turns.add(len(i["centers"]))
                cuts.add(len(i["breaks"]))
                directions.add(i["rotation"])
                populations.add(sum(len(r["packets"]) for r in records))
                layouts.add((i["core"].shape, tuple(i["centers"]), tuple(i["breaks"])))
                for j, name in enumerate(("bg", "rail", "bar", "frame")):
                    roles[j].add(i[name])
        # Query-only intervention: an exact three-color negative becomes a
        # dissolved route, and an old exact positive survives the new query.
        for pair in puzzle["train"]:
            i = _read(pair["input"])
            _, records = _evaluate(i["core"], i, i["query"])
            negatives = [r for r in records if not r["positive"] and len(r["colors"]) == 3
                         and len(r["colors"] & i["query"]) == 2]
            positives = [r for r in records if r["positive"] and len(r["packets"]) == 3]
            if negatives and positives:
                changed = i["view"].copy()
                positions = np.argwhere(changed[:3, :3] != i["bg"])
                for p, color in zip(positions, sorted(negatives[0]["colors"])):
                    changed[tuple(p)] = color
                solved = np.rot90(solve(np.rot90(changed, -i["rotation"])), i["rotation"])
                r0, r1, c0, c1 = i["box"]
                new_core = solved[r0:r1, c0:c1]
                assert all(new_core[p] == i["bg"] for p in negatives[0]["path"])
                assert all(new_core[p] != i["bg"] for p in positives[0]["path"])
                # Query position is irrelevant; header itself is always copied.
                moved = i["view"].copy()
                moved[:3, :3] = np.roll(moved[:3, :3], 1, axis=0)
                result = np.rot90(solve(np.rot90(moved, -i["rotation"])), i["rotation"])
                assert np.array_equal(result[r0:r1, c0:c1], np.rot90(pair["output"], i["rotation"])[r0:r1, c0:c1])
                break
        else:
            raise AssertionError("Missing selector intervention witness")
        if seed < 2:
            assert _fingerprint(generate_puzzle(seed)) == _fingerprint(puzzle)
            # Arbitrary color permutation preserves the entire rule.
            permutation = np.random.default_rng(seed+1000).permutation(10)
            for pair in puzzle["train"][:1]:
                assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert len(signatures) == 42 and len(shapes) >= 35 and len(layouts) >= 100
    assert len(lanes) >= 7 and len(turns) >= 10 and len(cuts) >= 8 and len(populations) >= 15
    assert len(directions) == 4 and all(s == set(range(10)) for s in roles)
    assert totals["overlap_pairs"] > 0 and totals["level_cargo_routes"] >= 42
    assert totals["ordered_packing_routes"] >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(314159, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Reject a partial deletion, a changed wall, and a lost packet.
    p = generate_puzzle(0, 2, 1)
    for damage in ("partial", "wall", "packet"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in ("train", "test")}
        found = False
        for pair in bad["train"]:
            i = _read(pair["input"])
            _, records = _evaluate(i["core"], i, i["query"])
            target = None
            if damage == "wall":
                target = next(p for p, c in i["writes"].items() if c == i["frame"])
            else:
                options = [r for r in records if r["positive"] == (damage == "partial") and r["packets"]]
                if options:
                    rec = options[0]
                    target = rec["path"][0] if damage == "partial" else (
                        rec["packets"][0][0] if rec["level"] else rec["inward"][0])
            if target is not None:
                v = np.rot90(pair["output"], i["rotation"])
                r0, r1, c0, c1 = i["box"]
                v[r0+target[0], c0+target[1]] = i["rail"]
                found = True
                break
        assert found
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Damaged output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "layouts": len(layouts), "lane_counts": sorted(lanes),
            "divider_counts": sorted(turns), "break_counts": sorted(cuts),
            "packet_populations": sorted(populations), "orientations": len(directions), **totals}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
