"""Beam-linked length pools. Standard library + NumPy; no file I/O in generation.

Beams survey the ORIGINAL pinned rectangles. Their connected causal communities
define the reservoirs in which transverse intervals are sorted. Survey paths are
not painted. All roles and the direction are recovered from the input cells.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np


def _require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _to_canonical(a, direction):
    return (a, np.flipud(a), a.T, np.fliplr(a).T)[direction]


def _from_canonical(a, direction):
    return np.ascontiguousarray((a, np.flipud(a), a.T, np.fliplr(a.T))[direction])


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _read_bars(a, bg, guide):
    components = _components((a != bg) & (a != guide))
    _require(2 <= len(components) <= 12, "expected 2..12 pinned bars")
    shared = None
    boxes = []
    for cells in components:
        rr, cc = zip(*cells)
        top, bottom, left, right = min(rr), max(rr), min(cc), max(cc)
        _require(len(cells) == (bottom-top+1)*(right-left+1), "bar is not a solid rectangle")
        colors = set(map(int, np.unique(a[top:bottom+1, left:right+1])))
        _require(len(colors) == 2 and right > left, "bar must contain its pin and an accent")
        shared = colors if shared is None else shared & colors
        boxes.append((top, bottom, left, right, colors))
    _require(len(shared) == 1, "pin color must be the unique color shared by every bar")
    pin = next(iter(shared))
    records = []
    for top, bottom, left, right, colors in boxes:
        block = a[top:bottom+1, left:right+1]
        columns = np.flatnonzero(np.any(block == pin, axis=0))
        _require(len(columns) == 1, "pin must be one transverse cell wide")
        x = left + int(columns[0])
        _require(np.all(a[top:bottom+1, x] == pin), "pin must span the bar thickness")
        accent = next(iter(colors - {pin}))
        expected = np.full(block.shape, accent, dtype=np.int8)
        expected[:, x-left] = pin
        _require(np.array_equal(block, expected), "bar accent is not solid")
        records.append({"r": int(top), "t": int(bottom-top+1), "x": int(x),
                        "color": accent, "lo": int(left-x), "hi": int(right-x)})
    records.sort(key=lambda b: (b["r"], b["x"]))
    templates = {}
    for b in records:
        length = b["hi"]-b["lo"]+1
        span = (b["lo"], b["hi"])
        _require(templates.setdefault(length, span) == span, "equal lengths need identical intervals")
    return pin, records


def _analyze(value):
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[bg]) == 1, "background must be uniquely dominant")
    choices = []
    for direction in range(4):
        c = _to_canonical(a, direction)
        h, w = c.shape
        for guide in map(int, np.unique(c[0])):
            if guide == bg:
                continue
            entries = list(map(int, np.flatnonzero(c[0] == guide)))
            mask = np.zeros(c.shape, dtype=bool)
            lengths = []
            for x in entries:
                rows = np.flatnonzero(c[:, x] == guide)
                if not 2 <= len(rows) < h or not np.array_equal(rows, np.arange(len(rows))):
                    break
                lengths.append(len(rows))
                mask[:len(rows), x] = True
            else:
                if not np.array_equal(mask, c == guide):
                    continue
                try:
                    pin, bars = _read_bars(c, bg, guide)
                    _require(max(lengths) <= min(b["r"] for b in bars), "stubs must end before all bars")
                except ValueError:
                    continue
                choices.append({"a": c, "bg": bg, "guide": guide, "pin": pin,
                                "bars": bars, "entries": entries, "stub_lengths": lengths,
                                "direction": direction})
    _require(len(choices) == 1, "roles and entry direction must have one geometric interpretation")
    return choices[0]


def _route(info, straight=False):
    """Propagate source identities; union communities at collisions and mergers.

    A bar is a maximal transverse run regardless of its pin/accent color change.
    No drawn survey crosspiece launches additional rays: only the two flanks do.
    The immutably read footprint, rather than the recalibrated output, is used.
    """
    a, bars = info["a"], info["bars"]
    h, w = a.shape
    occupied = np.full(a.shape, -1, dtype=int)
    for i, b in enumerate(bars):
        occupied[b["r"]:b["r"]+b["t"], b["x"]+b["lo"]:b["x"]+b["hi"]+1] = i
    parent = list(range(len(info["entries"])))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(sources):
        if sources:
            base = root(min(sources))
            for s in sources:
                parent[root(s)] = base

    active = {x: {s} for s, x in enumerate(info["entries"])}
    reached = [set() for _ in bars]
    simultaneous = multi_hit = clipped = mergers = 0
    for r in range(h):
        hits, flowing = {}, {}
        hit_columns = defaultdict(list)
        for x, sources in active.items():
            i = int(occupied[r, x])
            if i < 0:
                flowing.setdefault(x, set()).update(sources)
            else:
                hits.setdefault(i, set()).update(sources)
                hit_columns[i].append(x)
        simultaneous += len(hits) >= 2
        for i, sources in hits.items():
            reached[i].update(sources)
            unite(sources)
            multi_hit += len(hit_columns[i]) > 1
            b = bars[i]
            left, right = b["x"]+b["lo"], b["x"]+b["hi"]
            clipped += left == 0 or right == w-1
            if not straight:
                for x in (left-1, right+1):
                    if 0 <= x < w:
                        mergers += x in flowing and bool(flowing[x] - sources)
                        flowing.setdefault(x, set()).update(sources)
        for sources in flowing.values():
            unite(sources)
        active = flowing
    groups = defaultdict(list)
    naive = defaultdict(list)
    for i, sources in enumerate(reached):
        if sources:
            groups[root(min(sources))].append(i)
            naive[min(sources)].append(i)
    return {"groups": list(groups.values()), "naive_groups": list(naive.values()),
            "reached": {i for i, sources in enumerate(reached) if sources},
            "sources": reached, "simultaneous": int(simultaneous),
            "multi_hit": int(multi_hit), "clipped": int(clipped), "mergers": int(mergers)}


def _assign(info, groups, mode="length"):
    bars = info["bars"]
    spans = [(b["lo"], b["hi"]) for b in bars]
    for group in groups:
        order = sorted(group, key=lambda i: (bars[i]["r"], bars[i]["x"]))
        if mode == "reverse":
            donors = list(reversed(order))
        else:
            donors = sorted(order, key=lambda i: (bars[i]["hi"]-bars[i]["lo"]+1) *
                            (bars[i]["t"] if mode == "area" else 1), reverse=mode == "descending")
        for dst, src in zip(order, donors):
            spans[dst] = (bars[src]["lo"], bars[src]["hi"])
        if mode == "recenter":
            for i in order:
                n = spans[i][1]-spans[i][0]+1
                spans[i] = (-(n//2), (n-1)//2)
    return spans


def _paint(info, spans):
    a, bars = info["a"], info["bars"]
    h, w = a.shape
    out = np.full(a.shape, info["bg"], dtype=np.int8)
    out[a == info["guide"]] = info["guide"]
    owners = np.full(a.shape, -1, dtype=int)
    for i, (b, (lo, hi)) in enumerate(zip(bars, spans)):
        left, right, r, t = b["x"]+lo, b["x"]+hi, b["r"], b["t"]
        _require(0 <= left <= right < w and 0 <= r < r+t <= h, "assigned interval leaves canvas")
        region = np.s_[r:r+t, left:right+1]
        _require(np.all(out[region] == info["bg"]), "assigned bars overlap each other or a stub")
        out[region] = b["color"]
        out[r:r+t, b["x"]] = info["pin"]
        owners[region] = i
    # Four-connected components remain distinct, including repeated accents.
    for p, q in ((owners[:-1], owners[1:]), (owners[:, :-1], owners[:, 1:])):
        _require(not np.any((p >= 0) & (q >= 0) & (p != q)), "assigned bars touch")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    groups = _route(info)["groups"]
    return _from_canonical(_paint(info, _assign(info, groups)), info["direction"])


def _wrong(info, mode):
    route = _route(info)
    groups = route["groups"]
    if mode == "global":
        groups = [list(range(len(info["bars"])))]
    elif mode == "reached_global":
        groups = [sorted(route["reached"])]
    elif mode == "straight":
        groups = _route(info, straight=True)["groups"]
    elif mode == "no_merge":
        groups = route["naive_groups"]
    try:
        return _paint(info, _assign(info, groups, mode))
    except ValueError:
        return None  # A shortcut demanding an overlap/off-canvas cell is also contradicted.


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(pair):
    info = _analyze(pair["input"])
    route = _route(info)
    canonical_out = _to_canonical(np.asarray(pair["output"]), info["direction"])
    flags = {}
    for mode in ("global", "reached_global", "straight", "no_merge", "area", "reverse", "recenter", "descending"):
        wrong = _wrong(info, mode)
        flags[mode] = wrong is None or not np.array_equal(wrong, canonical_out)
    return {"count": len(info["bars"]), "pools": sorted(map(len, route["groups"])),
            "unreached": len(info["bars"])-len(route["reached"]),
            "lengths": [b["hi"]-b["lo"]+1 for b in info["bars"]],
            "thicknesses": [b["t"] for b in info["bars"]],
            "spans": [(b["lo"], b["hi"]) for b in info["bars"]],
            "direction": info["direction"], "roles": (info["bg"], info["guide"], info["pin"]),
            "entries": len(info["entries"]), "shape": np.asarray(pair["input"]).shape,
            "simultaneous": route["simultaneous"], "clipped": route["clipped"],
            "flags": flags}


def _record(r, t, x, length, templates, color):
    lo, hi = templates[length]
    return {"r": int(r), "t": int(t), "x": int(x), "lo": int(lo), "hi": int(hi), "color": int(color)}


def _raw(records, entries, lengths, h, w):
    # Canonical construction roles are mapped to a random palette afterwards.
    blank = np.zeros((h, w), dtype=np.int8)
    for c, n in zip(entries, lengths):
        blank[:n, c] = 1
    info = {"a": blank, "bg": 0, "guide": 1, "pin": 2, "bars": records}
    return _paint(info, [(b["lo"], b["hi"]) for b in records])


def _interval(rng, length):
    if rng.random() < .55:
        lo = -(length//2) + int(length % 2 == 0 and rng.random() < .5)
    else:
        lo = -int(rng.integers(length))
    return lo, lo+length-1


def _witness_tree(rng, baseline=False):
    short, middle, long = (3, 4, 6) if baseline else (int(rng.integers(2, 5)), 0, 0)
    if not baseline:
        middle = short + int(rng.integers(1, 3))
        long = middle + int(rng.integers(1, 3))
    templates = {n: _interval(rng, n) for n in (short, middle, long)}
    if baseline:
        templates = {3: (-1, 1), 4: (-1, 2), 6: (-2, 3)}
    first = int(rng.integers(4, 7))
    second = first + int(rng.integers(3, 6))
    left_pin = templates[middle][0]-1-templates[short][1]
    right_pin = templates[middle][1]+1-templates[long][0]
    records = [_record(first, 1, 0, middle, templates, 3),
               _record(second, 3, left_pin, short, templates, 4),
               _record(second, int(rng.integers(1, 3)), right_pin, long, templates, 5)]
    # An equal-length inert bar precedes the first collision and lies off-beam.
    low = min(b["x"]+b["lo"] for b in records)
    inert_pin = low-2-templates[middle][1]
    records.append(_record(2, 1, inert_pin, middle, templates, int(rng.integers(3, 10))))
    all_spans = [(b["x"]+lo, b["x"]+hi) for b in records for lo, hi in templates.values()]
    xmin, xmax = min(x for x, _ in all_spans), max(x for _, x in all_spans)
    _require(xmax-xmin+1 <= 30, "tree envelopes too wide")
    w = int(rng.integers(max(19, xmax-xmin+1), 31))
    shift = -xmin + int(rng.integers(w-(xmax-xmin+1)+1))
    for b in records:
        b["x"] += shift
    entry = shift + int(rng.integers(templates[middle][0], templates[middle][1]+1))
    h = int(rng.integers(second+6, min(30, second+12)+1))
    return _raw(records, [entry], [2], h, w)


def _witness_communities(rng, baseline=False):
    # One clipped two-bar community beside two sources that later merge.
    # These causal reservoirs share an unframed canvas and interleave by depth.
    small, fork, medium, wide, bridge = (2, 3, 4, 6, 7)
    if not baseline and rng.random() < .4:
        small, fork, medium, wide, bridge = (2, 3, 5, 6, 8)
    templates = {small: (0, small-1), fork: (-1, fork-2),
                 medium: (-1, medium-2), wide: (-2, wide-3),
                 bridge: (-(bridge//2), (bridge-1)//2)}
    first = int(rng.integers(4, 7))
    thick = [int(x) for x in rng.integers(1, 3, size=6)]
    middle_row = first + max(thick[:3]) + int(rng.integers(2, 5))
    last = middle_row + max(thick[3:5]) + int(rng.integers(2, 5))
    left_first = -templates[wide][0]
    left_next = wide-templates[medium][0]
    right_a = 17 + (0 if baseline else int(rng.integers(-1, 2)))
    right_b = 25 + (0 if baseline else int(rng.integers(-1, 2)))
    inner_a, inner_b = right_a+templates[fork][1]+1, right_b+templates[medium][0]-1
    lo, hi = templates[bridge]
    low, high = max(inner_a, inner_b)-hi, min(inner_a, inner_b)-lo
    _require(low <= high, "bridge cannot join both sources")
    bridge_pin = int(rng.integers(low, high+1))
    final_pin = bridge_pin+lo-1-templates[small][1]
    records = [_record(first, thick[0], left_first, wide, templates, 3),
               _record(first, thick[1], right_a, fork, templates, 4),
               _record(first, thick[2], right_b, medium, templates, 5),
               _record(middle_row, thick[3], left_next, medium, templates, 6),
               _record(middle_row, thick[4], bridge_pin, bridge, templates, 7),
               _record(last, thick[5], final_pin, small, templates, 8)]
    w = int(rng.integers(max(28, right_b+templates[medium][1]+1), 31))
    h = int(rng.integers(last+max(thick)+2, min(30, last+max(thick)+5)+1))
    return _raw(records, [left_first, right_a, right_b], [2, 2, 2], h, w)


def _scatter(rng, test=False):
    n = int(rng.integers(8, 11) if test else rng.integers(4, 8))
    layers = int(rng.integers(4, min(6, n)+1) if test else rng.integers(2, min(4, n)+1))
    quotas = [1]*layers
    for _ in range(n-layers):
        eligible = [i for i, x in enumerate(quotas) if x < 3]
        _require(bool(eligible), "too many bars for layers")
        quotas[int(rng.choice(eligible))] += 1
    w = int(rng.integers(23, 31))
    templates = {length: _interval(rng, length) for length in range(2, 9)}
    entry_count = int(rng.integers(1, 5))
    entries = []
    for x in map(int, rng.permutation(np.arange(2, w-2))):
        if all(abs(x-y) >= 2 for y in entries):
            entries.append(x)
            if len(entries) == entry_count:
                break
    active, records, r = set(entries), [], int(rng.integers(4, 6))
    for quota in quotas:
        row_records = []
        for _ in range(quota):
            for attempt in range(48):
                length = int(rng.integers(2, 9))
                if active and rng.random() < .85:
                    pivot = int(rng.choice(sorted(active)))
                    left = int(rng.integers(max(0, pivot-length+1), min(pivot, w-length)+1))
                else:
                    left = int(rng.integers(w-length+1))
                if any(not (left+length < b["x"]+b["lo"] or left > b["x"]+b["hi"]+1) for b in row_records):
                    continue
                pin = left-templates[length][0]
                row_records.append(_record(r, int(rng.integers(1, 3)), pin, length,
                                           templates, 3+len(records+row_records) % 7))
                break
            else:
                raise ValueError("crowded layer")
        following = set(active)
        for b in row_records:
            left, right = b["x"]+b["lo"], b["x"]+b["hi"]
            if any(left <= x <= right for x in active):
                following.difference_update(range(left, right+1))
                following.update(x for x in (left-1, right+1) if 0 <= x < w)
        active = following
        records.extend(row_records)
        r += max(b["t"] for b in row_records) + int(rng.integers(1, 3))
    h = r + int(rng.integers(1, 4))
    _require(h <= 30, "too many forward layers")
    return _raw(records, entries, [int(rng.integers(2, 4)) for _ in entries], h, w)


def _pair_from_raw(raw, palette, direction, mirror):
    if mirror:
        raw = np.fliplr(raw).copy()
    inp = _from_canonical(np.asarray(palette, dtype=np.int8)[raw], direction)
    out = solve(inp)
    return {"input": inp, "output": out}


def _clean_fallback(kind):
    """A parametric-family member used only after the random budget is spent.

    Fourteen distinct accent permutations below exceed the maximum thirteen
    already-used inputs. Thus the finite recovery does not depend on luck.
    """
    templates = {2: (0, 1), 3: (-1, 1), 4: (-1, 2), 6: (-2, 3), 7: (-3, 3)}
    if kind == "tree":
        records = [_record(2, 1, 3, 4, templates, 6),
                   _record(5, 1, 10, 4, templates, 3),
                   _record(10, 3, 7, 3, templates, 4),
                   _record(10, 1, 15, 6, templates, 5)]
        return _raw(records, [10], [2], 20, 25)
    records = [_record(4, 1, 2, 6, templates, 3),
               _record(4, 1, 17, 3, templates, 4),
               _record(4, 1, 25, 4, templates, 5),
               _record(10, 1, 7, 4, templates, 6),
               _record(10, 1, 22, 7, templates, 7),
               _record(16, 1, 17, 2, templates, 8)]
    if kind == "test":
        records.extend([_record(20, 1, 16, 3, templates, 9),
                        _record(24, 1, 19, 4, templates, 3)])
    return _raw(records, [2, 17, 25], [2, 2, 2], 28 if kind == "test" else 23, 28)


def _fallback_pair(kind, palette, direction, seen):
    raw = _clean_fallback(kind)
    for sign in (1, -1):
        for offset in range(7):
            mapping = np.arange(10, dtype=np.int8)
            mapping[3:] = [3+(sign*i+offset) % 7 for i in range(7)]
            pair = _pair_from_raw(mapping[raw], palette, direction, False)
            if _key(pair["input"]) not in seen:
                return pair
    raise AssertionError("at most thirteen inputs can exclude fourteen clean candidates")


def _make_pair(rng, kind, palette, direction, seen):
    for attempt in range(256):
        try:
            if kind == "tree":
                raw = _witness_tree(rng, baseline=attempt >= 96)
            elif kind == "communities":
                raw = _witness_communities(rng, baseline=attempt >= 96)
            else:
                raw = _scatter(rng, test=kind == "test")
            pair = _pair_from_raw(raw, palette, direction, bool(rng.integers(2)))
            if _key(pair["input"]) in seen:
                continue
            f = _features(pair)
            if np.array_equal(pair["input"], pair["output"]):
                continue
            if kind == "tree" and not (f["unreached"] and max(f["pools"]) >= 3 and
                    all(f["flags"][x] for x in ("global", "straight", "area", "reverse", "recenter"))):
                continue
            if kind == "communities" and not (len(f["pools"]) >= 2 and max(f["pools"]) >= 4 and
                    f["simultaneous"] and f["clipped"] and
                    all(f["flags"][x] for x in ("reached_global", "no_merge"))):
                continue
            if kind == "test" and not (max(f["pools"], default=0) >= 5 and
                    all(f["flags"][x] for x in ("global", "straight"))):
                continue
            return pair
        except ValueError:
            continue
    return _fallback_pair(kind, palette, direction, seen)


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)), f"{name} must be an integer")
    _require(low <= int(value) <= high, f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.permutation(10)]
    direction = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, kinds in (("train", ["tree", "communities"]+["train"]*(nt-2)), ("test", ["test"]*nq)):
        for kind in kinds:
            pair = _make_pair(rng, kind, palette, direction, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, features = set(), {"train": [], "test": []}
    conventions = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape and _key(a) not in seen, "duplicate input or changed shape")
            seen.add(_key(a))
            info = _analyze(a)
            out_info = _analyze(out)
            conventions.add((info["bg"], info["guide"], info["pin"], info["direction"]))
            for key in ("bg", "guide", "pin", "direction", "entries", "stub_lengths"):
                _require(info[key] == out_info[key], "role, direction or source changed")
            fixed = lambda b: (b["r"], b["t"], b["x"], b["color"])
            _require(list(map(fixed, info["bars"])) == list(map(fixed, out_info["bars"])), "fixed slots changed")
            route = _route(info)
            original = [(b["lo"], b["hi"]) for b in info["bars"]]
            final = [(b["lo"], b["hi"]) for b in out_info["bars"]]
            for i in set(range(len(original))) - route["reached"]:
                _require(original[i] == final[i], "unreached bar changed")
            for group in route["groups"]:
                indices = sorted(group, key=lambda i: fixed(info["bars"][i])[:1] + (info["bars"][i]["x"],))
                _require(sorted(original[i] for i in group) == sorted(final[i] for i in group), "pool lost or gained an interval")
                lengths = [final[i][1]-final[i][0]+1 for i in indices]
                _require(lengths == sorted(lengths), "pool is not sorted by transverse length")
            # Geometry + pin/color invariance + interval conservation/order fully
            # determine the answer; this check does not call solve or _assign.
            _require(not np.array_equal(a, out), "unchanged generated example")
            features[split].append(_features(pair))
    _require(len(conventions) == 1, "generated puzzle must share its visible conventions")
    train = features["train"]
    for flag in ("global", "reached_global", "straight", "no_merge", "area", "reverse", "recenter", "descending"):
        _require(any(f["flags"][flag] for f in train), f"training admits {flag} shortcut")
    _require(any(f["unreached"] for f in train), "missing inert-bar evidence")
    _require(any(len(f["pools"]) >= 2 for f in train), "missing separate pools")
    _require(any(f["simultaneous"] and f["clipped"] for f in train), "missing simultaneous/clipping evidence")
    _require(max(f["count"] for f in train) <= 7, "training count contract")
    _require(all(8 <= f["count"] <= 10 and max(f["pools"]) >= 5 for f in features["test"]), "missing count/composition shift")


def self_test():
    """Bounded, streaming 42-seed sweep; only scalar/hash summaries are retained."""
    shapes, normalized, directions, role_colors, counts, thicknesses, spans = set(), set(), set(), set(), set(), set(), set()
    puzzle_hashes = set()
    examples = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                a = np.asarray(pair["input"])
                frozen = a.copy()
                _require(np.array_equal(solve(a), pair["output"]) and np.array_equal(a, frozen), "solver mismatch/input mutation")
                f = _features(pair)
                shapes.add(f["shape"])
                directions.add(f["direction"])
                role_colors.add(f["roles"])
                counts.add(f["count"])
                thicknesses.update(f["thicknesses"])
                spans.update(f["spans"])
                mapping = {}
                norm = bytes(mapping.setdefault(int(x), len(mapping)) for x in a.ravel())
                normalized.add((a.shape, hashlib.sha256(norm).digest()))
                digest.update(repr(a.shape).encode())
                digest.update(a.tobytes())
                digest.update(np.asarray(pair["output"], dtype=np.int8).tobytes())
                examples += 1
        puzzle_hashes.add(digest.digest())
    _require(len(puzzle_hashes) == 42 and len(normalized) >= 180, "structural diversity collapsed")
    _require(len(shapes) >= 45 and directions == set(range(4)) and len(role_colors) >= 30, "geometry/palette diversity collapsed")
    _require(counts >= {4, 5, 6, 7, 8, 9, 10} and thicknesses == {1, 2, 3}, "count/thickness diversity collapsed")
    _require(any(lo == 0 and hi >= 2 for lo, hi in spans) and any(hi == 0 and lo <= -2 for lo, hi in spans), "one-sided span variation absent")
    # Exercise the finite recovery at maximum counts, without stochastic retries.
    fallback, seen = {"train": [], "test": []}, set()
    for split, kinds in (("train", ["tree"]+["train"]*9), ("test", ["test"]*4)):
        for kind in kinds:
            pair = _fallback_pair(kind, list(range(10)), 0, seen)
            seen.add(_key(pair["input"]))
            fallback[split].append(pair)
    validate_generated_puzzle(fallback)
    excluded = set()
    for _ in range(14):
        pair = _fallback_pair("test", list(range(10)), 0, excluded)
        excluded.add(_key(pair["input"]))
    _require(len(excluded) == 14, "finite fallback coloring candidates are not distinct")
    # Wrong labels must be rejected by the relational validator, not just
    # described as different by the counterfactual feature collector.
    rejected = set()
    for mode in ("global", "reached_global", "straight", "no_merge", "area", "reverse", "recenter", "descending"):
        for i, pair in enumerate(fallback["train"]):
            info = _analyze(pair["input"])
            wrong = _wrong(info, mode)
            if wrong is None or np.array_equal(wrong, pair["output"]):
                continue
            mutant = {s: [dict(p) for p in fallback[s]] for s in fallback}
            mutant["train"][i]["output"] = wrong
            try:
                validate_generated_puzzle(mutant)
            except ValueError:
                rejected.add(mode)
                break
            raise AssertionError(f"validator accepted {mode} mutation")
    _require(len(rejected) == 8, "a shortcut lacks a valid wrong-label regression")
    # Every legal count combination, processed one at a time.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(901, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    a, b = generate_puzzle(17), generate_puzzle(17)
    _require(all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), "nondeterminism")
    return {"seeds": 42, "pairs": examples, "shapes": len(shapes),
            "normalized_inputs": len(normalized), "role_assignments": len(role_colors),
            "directions": sorted(directions), "object_counts": sorted(counts),
            "rejected_shortcut_labels": sorted(rejected), "finite_fallback_inputs": len(excluded),
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
