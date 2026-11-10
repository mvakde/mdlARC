"""Moving switchboard: predicted bar positions determine ray connectivity.

Only standard-library modules and NumPy are used.  All semantic information is
in the input: the four frame colors, a fixed launch rail, and movable bars.
"""

from collections import Counter
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _key(a):
    a = np.asarray(a)
    return (a.shape, a.tobytes() if a.dtype == np.int8 else a.astype(np.int8).tobytes())


def _frame(a):
    h, w = a.shape
    _require(h >= 7 and w >= 7, "switchboard is too small")
    bg = int(a[0, 0])
    _require(all(int(a[r, c]) == bg for r, c in ((0, w-1), (h-1, 0), (h-1, w-1))),
             "frame corners disagree")
    sides = tuple(int(x) for x in (a[0, 1], a[-1, 1], a[1, 0], a[1, -1]))
    _require(len(set(sides)) == 4 and bg not in sides, "frame labels must be distinct")
    _require(bool(np.all(a[0, 1:-1] == sides[0]) and np.all(a[-1, 1:-1] == sides[1])
                  and np.all(a[1:-1, 0] == sides[2]) and np.all(a[1:-1, -1] == sides[3])),
             "nonuniform frame")
    return bg, sides


def _read_canonical(a):
    """Read a board whose launch rail is immediately inside its top edge."""
    bg, sides = _frame(a)
    h, w = a.shape
    marker = int(a[1, 1])
    _require(marker not in (bg, *sides) and int(a[1, -2]) == marker, "no launch rail")
    _require(set(int(c) for c in a[1, 1:-1]) <= {marker, sides[2], sides[3]}, "bad rail")
    ports = tuple((c, int(a[1, c])) for c in range(2, w-2) if int(a[1, c]) != marker)
    _require(bool(ports), "rail has no ports")
    _require(bool(np.all(a[2, 1:-1] == bg)), "rail needs an empty row in the input")
    _require(set(int(c) for c in a[3:-1, 1:-1].flat) <= {bg, sides[2], sides[3]},
             "unexpected interior color")
    bars = []
    for r in range(3, h-1):
        c = 1
        while c < w-1:
            color = int(a[r, c])
            if color == bg:
                c += 1
                continue
            lo = c
            while c+1 < w-1 and int(a[r, c+1]) == color:
                c += 1
            bars.append((r, lo, c, color))
            c += 1
    _require(bool(bars), "no bars")
    # Bars of the same color on neighboring rows must not touch, even diagonally.
    # Thus their identity agrees under both four- and eight-connectivity.
    for i, (r, lo, hi, color) in enumerate(bars):
        for y, left, right, other in bars[i+1:]:
            _require(not (color == other and abs(r-y) <= 1 and lo <= right+1 and left <= hi+1),
                     "same-colored bars touch")
    destinations = []
    blocked = []
    occupancy = set()
    for i, (r, lo, hi, color) in enumerate(bars):
        delta = -1 if color == sides[2] else 1
        stop = (lo == 1 if delta == -1 else hi == w-2)
        if stop:
            blocked.append(i)
            delta = 0
        dest = (r, lo+delta, hi+delta, color)
        cells = {(r, c) for c in range(dest[1], dest[2]+1)}
        _require(not occupancy & cells, "bars have conflicting final positions")
        occupancy.update(cells)
        destinations.append(dest)
    return dict(canonical=a, background=bg, sides=sides, marker=marker,
                ports=ports, bars=tuple(bars), destinations=tuple(destinations),
                blocked=tuple(blocked))


def _parse(input_grid):
    a = _grid(input_grid)
    _frame(a)
    candidates = []
    for rotation in range(4):
        try:
            info = _read_canonical(np.rot90(a, rotation))
        except AssertionError:
            continue
        info["rotation"] = rotation
        candidates.append(info)
    _require(len(candidates) == 1, "launch rail is missing or ambiguous")
    return candidates[0]


def _targets(ports, bars):
    result = []
    for col, color in ports:
        eligible = [i for i, (row, lo, hi, c) in enumerate(bars) if c == color and lo <= col <= hi]
        result.append(min(eligible, key=lambda i: bars[i][0]) if eligible else None)
    return tuple(result)


def _trace_cells(port, target, bars, sides, h, w):
    col, color = port
    stop = h-2 if target is None else bars[target][0]
    cells = {(r, col) for r in range(2, stop+1)}
    if target is not None:
        edge = 1 if color == sides[2] else w-2
        cells.update((stop, c) for c in range(min(col, edge), max(col, edge)+1))
    return cells


def _render(info, targets=None):
    a = info["canonical"]
    h, w = a.shape
    bg = info["background"]
    bars = info["destinations"]
    result = a.copy()
    result[2:-1, 1:-1] = bg
    layers = {}
    for r, lo, hi, color in bars:
        for c in range(lo, hi+1):
            layers.setdefault((r, c), set()).add(color)
    if targets is None:
        targets = _targets(info["ports"], bars)
    for port, target in zip(info["ports"], targets):
        for r, c in _trace_cells(port, target, bars, info["sides"], h, w):
            layers.setdefault((r, c), set()).add(port[1])
    for (r, c), colors in layers.items():
        result[r, c] = next(iter(colors)) if len(colors) == 1 else info["marker"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer frame, launch edge, bars, motion, and first interception from input."""
    info = _parse(input_grid)
    canonical = _render(info)
    return np.rot90(canonical, -info["rotation"]).copy()


def _stats(info):
    old = _targets(info["ports"], info["bars"])
    new = _targets(info["ports"], info["destinations"])
    counts = Counter(t for t in new if t is not None)
    gains = sum(a is None and b is not None for a, b in zip(old, new))
    losses = sum(a is not None and b is None for a, b in zip(old, new))
    competitions = sum(sum(c == color and lo <= col <= hi for _, lo, hi, c in info["destinations"]) > 1
                       for col, color in info["ports"])
    owners = {(r, c): i for i, (r, lo, hi, _) in enumerate(info["bars"]) for c in range(lo, hi+1)}
    handoffs = sum(owners.get((r, c), i) != i
                   for i, (r, lo, hi, _) in enumerate(info["destinations"]) for c in range(lo, hi+1))
    open_through_branch = 0
    for (col, color), target in zip(info["ports"], new):
        if target is not None:
            continue
        for (other_col, other_color), other_target in zip(info["ports"], new):
            if color != other_color or other_target is None:
                continue
            edge = 1 if color == info["sides"][2] else info["canonical"].shape[1]-2
            if min(edge, other_col) <= col <= max(edge, other_col):
                open_through_branch += 1
                break
    foreign_before_catcher = 0
    trace_layers = {}
    bar_layers = {(r, c): color for r, lo, hi, color in info["destinations"] for c in range(lo, hi+1)}
    h, w = info["canonical"].shape
    for port, target in zip(info["ports"], new):
        col, color = port
        stop = h-2 if target is None else info["destinations"][target][0]
        foreign_before_catcher += any(bar_layers.get((r, col), color) != color for r in range(3, stop))
        for cell in _trace_cells(port, target, info["destinations"], info["sides"], h, w):
            trace_layers.setdefault(cell, set()).add(color)
    wire_crossings = sum(len(colors) > 1 for colors in trace_layers.values())
    bar_crossings = sum(cell in bar_layers and any(c != bar_layers[cell] for c in colors)
                        for cell, colors in trace_layers.items())
    return dict(old=old, new=new, changed=sum(a != b for a, b in zip(old, new)),
                gains=gains, losses=losses, caught=sum(t is not None for t in new),
                open=sum(t is None for t in new), fan_in=max(counts.values(), default=0),
                unused=len(info["bars"])-len(counts), competitions=competitions, handoffs=handoffs,
                open_through_branch=open_through_branch,
                foreign_before_catcher=foreign_before_catcher, wire_crossings=wire_crossings,
                bar_crossings=bar_crossings,
                lengths=tuple(hi-lo+1 for _, lo, hi, _ in info["bars"]))


def _board(h, w, palette, ports, bars):
    bg, marker, top, bottom, left, right = palette
    a = np.full((h, w), bg, dtype=np.int8)
    a[0, 1:-1], a[-1, 1:-1] = top, bottom
    a[1:-1, 0], a[1:-1, -1] = left, right
    a[1, 1:-1] = marker
    for col, color in ports:
        a[1, col] = color
    for r, lo, hi, color in bars:
        _require(bool(np.all(a[r, lo:hi+1] == bg)), "construction overlap")
        a[r, lo:hi+1] = color
    return a


def _fixture(rng, palette, rotation, mirror):
    """A variable-geometry calibration scene, with causal counterexamples."""
    h, w = int(rng.integers(14, 24)), int(rng.integers(17, 27))
    left, right = palette[4:6]
    length = int(rng.integers(2, 5))
    start = int(rng.integers(5, max(6, w//2-2)))
    lost = start+length-1
    other_col = int(rng.integers(lost+3, w-2))
    first = int(rng.integers(5, h-7))
    later = int(rng.integers(first+2, h-3))
    right_row = int(rng.integers(3, h-3))
    ports = [(start-3, right), (start-2, left), (start-1, left), (start, left),
             (lost, left), (other_col, right)]
    bars = [(first, start, lost, left),
            (later, start, start+int(rng.integers(1, length))-1, left),
            (h-2, 1, int(rng.integers(1, start-2)), left),
            (right_row, other_col-1, other_col, right),
            (h-2, w-3, w-3, left),
            (h-2, w-4, w-4, right),
            (3, start-2, start, right),
            (first+2, start-4, start-3, right)]
    a = _board(h, w, palette, ports, bars)
    if mirror:
        a = np.fliplr(a)
    a = np.rot90(a, -rotation).copy()
    info = _parse(a)
    stats = _stats(info)
    _require(stats["gains"] >= 1 and stats["losses"] >= 1 and stats["fan_in"] >= 2,
             "fixture lost its endpoint evidence")
    _require(stats["competitions"] and stats["unused"] and info["blocked"] and stats["handoffs"]
             and stats["open_through_branch"] and stats["foreign_before_catcher"]
             and stats["bar_crossings"] and stats["wire_crossings"],
             "fixture lacks contrast")
    return {"input": a, "output": solve(a)}


def _compatible_bar(bars, candidate, left, w):
    r, lo, hi, color = candidate
    delta = -1 if color == left else 1
    if (delta == -1 and lo == 1) or (delta == 1 and hi == w-2):
        delta = 0
    for y, l, u, c in bars:
        if r == y and lo <= u and l <= hi:
            return False
        if color == c and abs(r-y) <= 1 and lo <= u+1 and l <= hi+1:
            return False
        d = -1 if c == left else 1
        if (d == -1 and l == 1) or (d == 1 and u == w-2):
            d = 0
        if r == y and lo+delta <= u+d and l+d <= hi+delta:
            return False
    return True


def _random_pair(rng, palette, rotation, test):
    """Sample bars first, then source lines that expose changing interception.

    This is not a layout template: port colors, order, intervals, shared targets,
    unused rods and stopping depths emerge from the sampled geometry.
    """
    left, right = palette[4:6]
    for _ in range(180):
        h = int(rng.integers(9, 31 if test else 26))
        w = int(rng.integers(13 if test else 9, 31 if test else 27))
        length = int(rng.integers(6, min(8, w-5)+1)) if test else int(rng.integers(1, min(5, w-5)+1))
        color = (left, right)[int(rng.integers(2))]
        row = int(rng.integers(3, h-1))
        # A deliberately gained endpoint supplies a strong motion/routing witness.
        if color == left:
            port_col = int(rng.integers(2, w-1-length))
            planted = (row, port_col+1, port_col+length, color)
        else:
            port_col = int(rng.integers(length+1, w-2))
            planted = (row, port_col-length, port_col-1, color)
        bars = [planted]
        desired = int(rng.integers(2, min(17, 3+(h-3)*(w-2)//22)))
        for _ in range(desired*15):
            if len(bars) >= desired:
                break
            c = (left, right)[int(rng.integers(2))]
            n = int(rng.integers(1, min(8 if test else 5, w-2)+1))
            lo = int(rng.integers(1, w-n))
            if rng.random() < .16:
                lo = 1 if c == left else w-1-n
            candidate = (int(rng.integers(3, h-1)), lo, lo+n-1, c)
            if _compatible_bar(bars, candidate, left, w):
                bars.append(candidate)
        # One temporary port is sufficient to parse the scene; no generation
        # state is ever passed to the public solver.
        a = _board(h, w, palette, [(port_col, color)], bars)
        try:
            info = _read_canonical(a)
        except AssertionError:
            continue
        destinations = info["destinations"]
        options = []
        for col in range(2, w-2):
            for co in (left, right):
                port = (col, co)
                target = _targets((port,), destinations)[0]
                cells = _trace_cells(port, target, destinations, info["sides"], h, w)
                old = _targets((port,), info["bars"])[0]
                options.append((port, target, old, cells))
        planted_index = info["bars"].index(planted)
        anchors = [x for x in options if x[0] == (port_col, color)
                   and x[1] == planted_index and x[1] != x[2]]
        if not anchors:
            continue
        anchor = anchors[0]
        chosen = [anchor]
        count = int(rng.integers(2 if test else 1, min(8 if test else 6, w-4)+1))
        for j in rng.permutation(len(options)):
            if len(chosen) >= count:
                break
            option = options[int(j)]
            (col, co), _, _, cells = option
            if any(col == x[0][0] for x in chosen):
                continue
            chosen.append(option)
        if test and len(chosen) < 2:
            continue
        a = _board(h, w, palette, [x[0] for x in chosen], bars)
        a = np.rot90(a, -rotation).copy()
        try:
            out = solve(a)
        except AssertionError:
            continue
        final_info = _parse(a)
        if np.array_equal(_render(final_info), _render(final_info, _targets(final_info["ports"], final_info["bars"]))):
            continue
        return {"input": a, "output": out}
    raise RuntimeError("could not construct a compatible switchboard")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= int(num_train) <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= int(num_test) <= 4:
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = int(num_train), int(num_test)
    anchor_palette = tuple(int(x) for x in rng.permutation(10)[:6])
    orientation = int(rng.integers(4))
    slots = [int(x) for x in rng.choice(num_train, 2, replace=False)]
    mirror = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(30):
                if split == "train" and index in slots:
                    which = slots.index(index)
                    pair = _fixture(rng, anchor_palette, (orientation+which) % 4,
                                    mirror if which == 0 else not mirror)
                else:
                    palette = tuple(int(x) for x in rng.permutation(10)[:6])
                    pair = _random_pair(rng, palette, int(rng.integers(4)), split == "test")
                fingerprint = _key(pair["input"])
                if fingerprint not in seen:
                    break
            else:
                raise RuntimeError("duplicate input rejection exhausted")
            seen.add(fingerprint)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal example counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "dimensions changed")
            k = _key(a)
            _require(k not in seen, "repeated input")
            seen.add(k)
            info = _parse(a)
            expected = np.rot90(_render(info), -info["rotation"])
            _require(np.array_equal(expected, b), "wrong switchboard output")
            canonical_out = np.rot90(b, info["rotation"])
            ca = info["canonical"]
            _require(np.array_equal(ca[0], canonical_out[0]) and np.array_equal(ca[-1], canonical_out[-1])
                     and np.array_equal(ca[:, 0], canonical_out[:, 0])
                     and np.array_equal(ca[:, -1], canonical_out[:, -1])
                     and np.array_equal(ca[1], canonical_out[1]), "frame or source rail changed")
            _require(set(np.unique(a)) == set(np.unique(b)), "palette changed")
            for r, lo, hi, c in info["destinations"]:
                _require(bool(np.all(np.isin(canonical_out[r, lo:hi+1], (c, info["marker"])))),
                         "a translated bar was lost")
            # Every source creates a new stem immediately past its fixed socket:
            # omitting A (keeping only B's translated bars) changes this row.
            _require(all(int(canonical_out[2, col]) == color for col, color in info["ports"]),
                     "source did not launch a ray")
            stats = _stats(info)
            _require(stats["caught"] and stats["changed"], "pair does not couple motion to routing")
            _require(not np.array_equal(canonical_out, _render(info, stats["old"])),
                     "ignoring motion when routing does not change the answer")
            if split == "test":
                _require(any(t is not None and info["destinations"][t][2]-info["destinations"][t][1]+1 >= 6
                             and old != t for old, t in zip(stats["old"], stats["new"])),
                         "test lacks its long-bar interception shift")
            records[split].append((info, stats))
    train = records["train"]
    _require(any(s["gains"] and s["losses"] and s["open"] and s["fan_in"] >= 2
                 and s["unused"] and s["competitions"] and s["handoffs"] and s["open_through_branch"]
                 and s["foreign_before_catcher"] and s["bar_crossings"] and s["wire_crossings"]
                 and i["blocked"] for i, s in train),
             "training lacks causal contrasts")
    _require(len({i["rotation"] for i, _ in train}) >= 2, "training lacks rotated launch evidence")
    # A literal active color must visibly shift along different physical axes.
    vectors = {}
    for info, _ in train:
        for idx, ((_, lo, hi, color), (_, new_lo, new_hi, _)) in enumerate(zip(info["bars"], info["destinations"])):
            if idx in info["blocked"]:
                continue
            dr, dc = 0, new_lo-lo
            for _ in range(info["rotation"]):
                dr, dc = dc, -dr  # undo one counterclockwise canonicalization
            vectors.setdefault(color, set()).add((dr, dc))
    _require(any(len(v) >= 2 for v in vectors.values()), "training permits a fixed color-direction map")


def _fingerprint(puzzle):
    return tuple((split, _key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded checks over 42 seeds, including explicit semantic regressions."""
    # Hand-computable example: two ports gain/share the short left-moving bar;
    # a third loses it and remains open despite another matching bar elsewhere.
    # The right-colored ray crosses two left-colored wires.  Foreign bars are
    # transparent to interception; unlike-color overlaps use the neutral marker.
    palette = (0, 5, 1, 2, 3, 4)
    a = _board(10, 12, palette, [(2, 4), (3, 3), (4, 3), (5, 3)],
               [(3, 2, 5, 4), (4, 4, 5, 3), (6, 1, 2, 4),
                (7, 1, 2, 3), (7, 8, 8, 4), (7, 9, 9, 3)])
    expected = a.copy()
    expected[2:-1, 1:-1] = 0
    expected[4, 1:5] = 3
    expected[2:5, 3:5] = 3
    expected[2:9, 5] = 3
    expected[7, 1:3] = 3
    expected[7, 8:10] = (3, 4)
    expected[2:7, 2] = 4
    expected[6, 2:11] = 4
    expected[3, 3:6] = 5
    expected[3, 6] = 4
    expected[4, 2] = 5
    expected[6, 5] = 5
    _require(np.array_equal(solve(a), expected), "endpoint gain/loss regression failed")
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(a, k)), np.rot90(expected, k)), "rotation equivariance failed")
    _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(expected)), "reflection equivariance failed")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "seed diversity collapsed")
    _require(_fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0)), "not deterministic")
    dimensions, rotations, lengths, ports, bars, backgrounds, palettes, structures = (set() for _ in range(8))
    totals = Counter()
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                info = _parse(a)
                stats = _stats(info)
                dimensions.add(a.shape)
                rotations.add(info["rotation"])
                lengths.update(stats["lengths"])
                ports.add(len(info["ports"]))
                bars.add(len(info["bars"]))
                backgrounds.add(info["background"])
                palettes.add(tuple(info["sides"]))
                mapping = {info["background"]: 0, info["marker"]: 1}
                mapping.update({c: j+2 for j, c in enumerate(info["sides"])})
                structures.add((a.shape, tuple(mapping[int(c)] for c in a.flat)))
                for name in ("gains", "losses", "open", "competitions", "unused", "handoffs", "open_through_branch",
                             "foreign_before_catcher", "bar_crossings", "wire_crossings"):
                    totals[name] += stats[name] > 0
                totals["shared"] += stats["fan_in"] >= 2
                totals["blocked"] += bool(info["blocked"])
                totals["unblocked"] += not info["blocked"]
                # Every ray's selected catcher is in its column and no nearer
                # same-colored shifted bar occupies that column.
                for (col, color), target in zip(info["ports"], stats["new"]):
                    if target is not None:
                        r, lo, hi, c = info["destinations"][target]
                        _require(c == color and lo <= col <= hi, "nonintersecting catcher selected")
                        _require(not any(y < r and co == color and l <= col <= u
                                         for y, l, u, co in info["destinations"]), "wrong first catcher")
    _require(len(dimensions) >= 65 and rotations == set(range(4)), "geometry variation collapsed")
    _require(set(range(1, 9)) <= lengths and len(ports) >= 6 and len(bars) >= 10, "population variation collapsed")
    _require(backgrounds == set(range(10)) and len(palettes) >= 80, "palette variation collapsed")
    _require(len(structures) >= 200 and all(totals[n] >= 10 for n in totals), "structural contrasts collapsed")
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(42+n*7+m, n, m)
            _require(len(p["train"]) == n and len(p["test"]) == m, "counts ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=210, distinct_shapes=len(dimensions),
                color_normalized_structures=len(structures), port_counts=sorted(ports),
                bar_counts=sorted(bars), bar_lengths=sorted(lengths), contrasts=dict(totals))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
