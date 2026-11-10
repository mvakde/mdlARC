"""Switched combs: sort outward tooth lengths along reconfigured threads.

Only input geometry determines roles, connected groups and reading direction.
The constructor, solver and relational validator do not consult parent files.
"""
from collections import Counter
import numpy as np


_DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _paths(mask):
    """Return simple paths ordered from smaller to larger (row, column) end."""
    cells = set(map(tuple, np.argwhere(mask)))
    adjacent = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRS
                    if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    assert all(len(v) in (1, 2) for v in adjacent.values()), "branch/isolated track"
    unseen, result = set(cells), []
    while unseen:
        todo, component = [min(unseen)], set()
        while todo:
            p = todo.pop()
            if p in component:
                continue
            component.add(p)
            todo.extend(adjacent[p])
        unseen -= component
        ends = sorted(p for p in component if len(adjacent[p]) == 1)
        assert len(ends) == 2, "closed thread"
        path, previous = [ends[0]], None
        while path[-1] != ends[1]:
            choices = [p for p in adjacent[path[-1]] if p != previous]
            assert len(choices) == 1
            previous, following = path[-1], choices[0]
            path.append(following)
        assert len(path) == len(component)
        # A direction toward the lower bank must have no intermediate trap.
        changes = {b[0] - a[0] for a, b in zip(path, path[1:]) if a[0] != b[0]}
        assert len(changes) <= 1, "thread doubles back across lanes"
        result.append(path)
    return sorted(result, key=lambda p: (p[0], p[-1]))


def _rewire(base, lane_rows, centers, bg, track, divider):
    answer, writes = base.copy(), {}
    for r, c in centers:
        assert np.all(base[r - 1, c - 1:c + 2] == track)
        assert np.all(base[r + 1, c - 1:c + 2] == track)
        for y in range(r - 1, r + 2):
            for x in range(c - 1, c + 2):
                color = divider if x == c else track
                assert (y, x) not in writes or writes[y, x] == color, "conflicting gates"
                writes[y, x] = color
    for p, color in writes.items():
        answer[p] = color
    paths = _paths(answer == track)
    for path in paths:
        for r, c in (path[0], path[-1]):
            assert r in lane_rows
            assert c in (0, base.shape[1] - 1) or base[r, c - 1] == bg or base[r, c + 1] == bg
    return answer, paths


def _canonical_parse(g, vertical):
    a = g.T.copy() if vertical else g.copy()
    bg = int(a[0, 0])
    assert a[0, -1] == a[-1, 0] == a[-1, -1] == bg
    rows = np.flatnonzero((a[:, 0] == a[:, -1]) & (a[:, 0] != bg)).tolist()
    assert len(rows) >= 2 and all(y - x == 2 for x, y in zip(rows, rows[1:]))
    top, bottom = rows[0], rows[-1]
    track = int(a[top, 0])
    assert all(a[r, 0] == track for r in rows)
    assert top >= 1 and bottom < a.shape[0] - 1 and a.shape[1] >= 9
    div_colors = set(map(int, np.unique(a[top + 1:bottom:2]))) - {bg}
    assert len(div_colors) == 1
    divider = div_colors.pop()
    assert divider != track
    centers = []
    for r in range(top + 1, bottom, 2):
        x = 0
        while x < a.shape[1]:
            if a[r, x] == bg:
                x += 1
                continue
            end = x + 1
            while end < a.shape[1] and a[r, end] == divider:
                end += 1
            assert a[r, x] == divider and end - x == 3
            assert x >= 1 and end < a.shape[1]
            centers.append((r, x + 1))
            x = end
    assert centers
    for r, c in centers:
        assert np.all(a[r - 1, c - 1:c + 2] == track)
        assert np.all(a[r + 1, c - 1:c + 2] == track), "gate touches a tooth or break"
    rebuilt = np.full_like(a, bg)
    rebuilt[rows] = track
    breaks = []
    for r in rows:
        xs = np.flatnonzero(a[r] == bg).tolist()
        assert all(2 <= x <= a.shape[1] - 3 for x in xs)
        assert all(y - x >= 3 for x, y in zip(xs, xs[1:]))
        for x in xs:
            rebuilt[r, x] = bg
            breaks.append((r, x))
    for r, c in centers:
        rebuilt[r, c - 1:c + 2] = divider
    slots = []
    for r, outward in ((top, -1), (bottom, 1)):
        x = 1
        while x < a.shape[1] - 1:
            color = int(a[r, x])
            if color in (bg, track):
                x += 1
                continue
            assert color != divider
            end = x + 1
            while end < a.shape[1] and a[r, end] == color:
                end += 1
            assert end < a.shape[1]
            length, y = 1, r + outward
            while 0 <= y < a.shape[0] and np.all(a[y, x:end] == color):
                length += 1
                y += outward
            assert 0 <= y < a.shape[0], "teeth require an outer background margin"
            ys = [r + outward * k for k in range(length)]
            rebuilt[np.ix_(ys, range(x, end))] = color
            slots.append({"row": r, "start": x, "end": end, "color": color,
                          "length": length, "outward": outward})
            x = end
    assert len(slots) >= 2 and {s["outward"] for s in slots} == {-1, 1}
    assert np.array_equal(a, rebuilt), "unexplained cells or nonsolid tooth"
    base = rebuilt.copy()
    for s in slots:
        ys = [s["row"] + s["outward"] * k for k in range(s["length"])]
        base[np.ix_(ys, range(s["start"], s["end"]))] = bg
        base[s["row"], s["start"]:s["end"]] = track
    structural, paths = _rewire(base, rows, centers, bg, track, divider)
    lookup = {cell: (i, j) for i, path in enumerate(paths) for j, cell in enumerate(path)}
    groups = [[] for _ in paths]
    for i, s in enumerate(slots):
        locations = [lookup[(s["row"], x)] for x in range(s["start"], s["end"])]
        assert len({p for p, _ in locations}) == 1
        indices = sorted(j for _, j in locations)
        assert indices == list(range(indices[0], indices[-1] + 1))
        groups[locations[0][0]].append((indices[0], i))
    groups = [[i for _, i in sorted(group)] for group in groups]
    # Every template must be able to move to every destination on its thread.
    for group in groups:
        for i in group:
            s = slots[i]
            for j in group:
                tip = s["row"] + s["outward"] * (slots[j]["length"] - 1)
                assert 1 <= tip <= a.shape[0] - 2, "template does not fit"
    return {"array": a, "vertical": vertical, "background": bg, "track": track,
            "divider": divider, "rows": rows, "centers": centers, "breaks": breaks,
            "slots": slots, "structural": structural, "paths": paths, "groups": groups}


def _parse(input_grid):
    g = _grid(input_grid)
    candidates = []
    for vertical in (False, True):
        try:
            candidates.append(_canonical_parse(g, vertical))
        except (AssertionError, KeyError, IndexError):
            pass
    if len(candidates) != 1:
        raise ValueError("input must have one readable switched-comb interpretation")
    return candidates[0]


def _paint(info, lengths):
    out = info["structural"].copy()
    for s, length in zip(info["slots"], lengths):
        for k in range(length):
            out[s["row"] + s["outward"] * k, s["start"]:s["end"]] = s["color"]
    return out.T.copy() if info["vertical"] else out


def _assigned(info, mode="correct"):
    slots = info["slots"]
    lengths = [s["length"] for s in slots]
    if mode == "unchanged":
        return lengths
    if mode == "banks":
        groups = [[i for i, s in enumerate(slots) if s["outward"] == sign] for sign in (-1, 1)]
    else:
        groups = info["groups"]
    for group in groups:
        if mode == "area":
            values = [slots[i]["length"] for i in sorted(group, key=lambda i:
                      slots[i]["length"] * (slots[i]["end"] - slots[i]["start"]))]
        elif mode == "reverse":
            values = [slots[i]["length"] for i in group[::-1]]
        else:
            values = sorted(slots[i]["length"] for i in group)
        for i, length in zip(group, values):
            lengths[i] = length
    return lengths


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _paint(info, _assigned(info))


def _features(info):
    correct = _assigned(info)
    original = [s["length"] for s in info["slots"]]
    widths = [s["end"] - s["start"] for s in info["slots"]]
    return {"lanes": len(info["rows"]), "width": info["array"].shape[1],
            "slots": len(original), "dividers": len(info["centers"]),
            "breaks": len(info["breaks"]), "paths": len(info["paths"]),
            "group_sizes": tuple(len(g) for g in info["groups"]),
            "lengths": tuple(original), "widths": tuple(widths),
            "vertical": info["vertical"], "changed_lengths": correct != original,
            "topology_matters": correct != _assigned(info, "banks"),
            "area_disagrees": correct != _assigned(info, "area"),
            "reversal_disagrees": correct != _assigned(info, "reverse"),
            "cross_bank": any({info["slots"][i]["outward"] for i in g} == {-1, 1}
                              for g in info["groups"]),
            "ties": any(len({original[i] for i in g}) < len(g) for g in info["groups"])}


def _sample_network(rng, lanes, width, busy):
    base = np.zeros((2 * lanes - 1, width), dtype=np.int8)
    base[::2] = 1
    centers, previous = [], []
    for r in range(1, base.shape[0], 2):
        wanted = int(rng.integers(0, busy + 1))
        chosen = []
        for raw in rng.permutation(np.arange(2, width - 2)):
            c = int(raw)
            if len(chosen) >= wanted:
                break
            if all(abs(c - p) >= 4 for p in chosen) and all(abs(c - p) >= 2 for p in previous):
                chosen.append(c)
        centers.extend((r, c) for c in sorted(chosen))
        previous = chosen
    if not centers:
        centers = [(int(rng.integers(lanes - 1)) * 2 + 1, int(rng.integers(2, width - 2)))]
    breaks = []
    for r in range(0, base.shape[0], 2):
        ports = sorted((c, y) for y, c in centers if abs(y - r) == 1)
        for (a, side_a), (b, side_b) in zip(ports, ports[1:]):
            if side_a == side_b:
                breaks.append((r, int(rng.integers(a + 2, b - 1))))
    for _ in range(int(rng.integers(0, 5))):
        choices = [(r, c) for r in range(0, base.shape[0], 2) for c in range(2, width - 2)
                   if all(abs(c - x) >= 2 for y, x in centers if abs(y - r) == 1)
                   and all(abs(c - x) >= 3 for y, x in breaks if y == r)]
        if choices:
            breaks.append(choices[int(rng.integers(len(choices)))])
    for r, c in centers:
        base[r, c - 1:c + 2] = 2
    for p in breaks:
        base[p] = 0
    structural, paths = _rewire(base, list(range(0, base.shape[0], 2)), centers, 0, 1, 2)
    safe = base == 1
    for r, c in centers:
        safe[r - 1:r + 2, c - 1:c + 2] = False
    return base, safe, paths


def _sample_pair(rng, palette, vertical, lane_range, busy, witness,
                 require_busy=False, require_sort=False):
    if witness:
        return _witness_pair(rng, palette, vertical, lane_range)
    for _ in range(180):
        lanes = int(rng.integers(lane_range[0], lane_range[1] + 1))
        width = int(rng.integers(13, 31))
        base, safe, _ = _sample_network(rng, lanes, width, busy)
        slots = []
        for row, sign in ((0, -1), (base.shape[0] - 1, 1)):
            x = 1 + int(rng.integers(0, 3))
            while x < width - 1:
                if not safe[row, x]:
                    x += 1
                    continue
                cap = 0
                while x + cap < width - 1 and safe[row, x + cap] and cap < 3:
                    cap += 1
                w = int(rng.integers(1, cap + 1))
                slots.append((row, sign, x, w))
                x += w + int(rng.integers(1, 4))
        if len(slots) < 4:
            continue
        wanted = int(rng.integers(4, min(12, len(slots)) + 1))
        chosen = set(map(int, rng.choice(len(slots), wanted, replace=False)))
        slots = [s for i, s in enumerate(slots) if i in chosen]
        if {s[1] for s in slots} != {-1, 1}:
            continue
        lengths = list(map(int, rng.integers(1, 8, size=len(slots))))
        # Both banks can receive the longest template, even if currently shorter.
        cap = max(lengths) - 1
        core_h = base.shape[0]
        extra = int(rng.integers(2, min(8, 30 - core_h - 2 * cap) + 1))
        before = int(rng.integers(1, extra))
        offset, height = cap + before, core_h + 2 * cap + extra
        a = np.full((height, width), palette[0], dtype=np.int8)
        a[offset:offset + core_h] = np.asarray(palette[:3], dtype=np.int8)[base]
        for (r, sign, x, w), length in zip(slots, lengths):
            color = int(rng.choice(palette[3:]))
            for k in range(length):
                a[offset + r + sign * k, x:x + w] = color
        if vertical:
            a = a.T.copy()
        info = _parse(a)
        if require_busy and max(Counter(r for r, _ in info["centers"]).values()) < 2:
            continue
        f = _features(info)
        if require_sort and not (f["changed_lengths"] and f["topology_matters"]):
            continue
        return {"input": a, "output": _paint(info, _assigned(info))}
    if not require_busy:
        return _witness_pair(rng, palette, vertical, lane_range)
    return _busy_fallback(rng, palette, vertical, lane_range)


def _busy_fallback(rng, palette, vertical, lane_range):
    """Finite density-shift fallback: a cut opens the first double gate."""
    lanes = max(3, lane_range[0])
    base = np.zeros((2 * lanes - 1, 30), dtype=np.int8)
    base[::2] = 1
    centers = [(1, 10), (1, 22)] + [(r, 16 if r == 3 else 14 if (r // 2) % 2 == 0 else 18)
                                    for r in range(3, base.shape[0], 2)]
    for r, c in centers:
        base[r, c - 1:c + 2] = 2
    base[0, 16] = 0
    offset, height = 7, base.shape[0] + 14
    a = np.full((height, 30), palette[0], dtype=np.int8)
    a[offset:offset + base.shape[0]] = np.asarray(palette[:3], dtype=np.int8)[base]
    slots = [(0, -1, 2, 1, 6), (0, -1, 5, 3, 1), (0, -1, 26, 1, 2),
             (base.shape[0] - 1, 1, 2, 1, 3), (base.shape[0] - 1, 1, 26, 1, 7)]
    for r, sign, x, w, length in slots:
        color = int(rng.choice(palette[3:]))
        for k in range(length):
            a[offset + r + sign * k, x:x + w] = color
    if vertical:
        a = a.T.copy()
    info = _parse(a)
    f = _features(info)
    assert f["changed_lengths"] and f["topology_matters"]
    return {"input": a, "output": _paint(info, _assigned(info))}


def _witness_pair(rng, palette, vertical, lane_range):
    """Construct an area inversion on a four-slot thread; no rejection luck."""
    lanes = int(rng.integers(lane_range[0], lane_range[1] + 1))
    width = int(rng.integers(23, 31))
    left, right = int(rng.integers(9, 12)), width - 10
    right = max(right, left + 2)
    base = np.zeros((2 * lanes - 1, width), dtype=np.int8)
    base[::2] = 1
    centers = [(r, left if (r // 2) % 2 == 0 else right)
               for r in range(1, base.shape[0], 2)]
    for r, c in centers:
        base[r, c - 1:c + 2] = 2
    _, paths = _rewire(base, list(range(0, base.shape[0], 2)), centers, 0, 1, 2)
    # Symmetric width patterns give each spanning thread widths 1,3,3,1.
    spans = [(int(rng.integers(1, 3)), 1), (int(rng.integers(4, 6)), 3),
             (width - 8, 3), (width - 2, 1)]
    slots = [(r, sign, x, w) for r, sign in ((0, -1), (base.shape[0] - 1, 1))
             for x, w in spans]
    lookup = {p: (i, j) for i, path in enumerate(paths) for j, p in enumerate(path)}
    groups = [[] for _ in paths]
    for i, (r, sign, x, w) in enumerate(slots):
        p, rank = lookup[(r, x)]
        groups[p].append((rank, i))
    eligible = [[i for _, i in sorted(g)] for g in groups if len(g) == 4]
    group = eligible[int(rng.integers(len(eligible)))]
    others = [i for i in range(len(slots)) if i not in group]
    removed = set(map(int, rng.choice(others, int(rng.integers(0, 3)), replace=False)))
    lengths = [7] * len(slots)
    for i, length in zip(group, (int(rng.integers(6, 8)), 1, int(rng.integers(3, 6)), 2)):
        lengths[i] = length
    cap = 6
    extra = int(rng.integers(2, min(7, 30 - base.shape[0] - 2 * cap) + 1))
    before = int(rng.integers(1, extra))
    offset, height = cap + before, base.shape[0] + 2 * cap + extra
    a = np.full((height, width), palette[0], dtype=np.int8)
    a[offset:offset + base.shape[0]] = np.asarray(palette[:3], dtype=np.int8)[base]
    for i, ((r, sign, x, w), length) in enumerate(zip(slots, lengths)):
        if i in removed:
            continue
        color = int(rng.choice(palette[3:]))
        for k in range(length):
            a[offset + r + sign * k, x:x + w] = color
    if vertical:
        a = a.T.copy()
    info = _parse(a)
    f = _features(info)
    assert all(f[k] for k in ("changed_lengths", "topology_matters", "area_disagrees",
                             "reversal_disagrees", "cross_bank"))
    return {"input": a, "output": _paint(info, _assigned(info))}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    fixed_roles = bool(rng.integers(2))
    axis_mode = int(rng.integers(3))  # horizontal, vertical, or demonstrated mix
    shift = int(rng.integers(3))  # grow, shrink, or denser rewiring
    ranges = [((2, 4), (5, 7)), ((5, 7), (2, 4)), ((2, 5), (3, 7))][shift]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            vertical = bool(axis_mode) if axis_mode < 2 else bool(i % 2 if split == "train" and i < 2 else rng.integers(2))
            roles = palette if fixed_roles else list(map(int, rng.permutation(10)))
            busy = (1 if split == "train" else 3) if shift == 2 else int(rng.integers(1, 4))
            for _ in range(20):
                pair = _sample_pair(rng, roles, vertical, ranges[split == "test"], busy,
                                    witness=split == "train" and i < 2,
                                    require_busy=shift == 2 and split == "test",
                                    require_sort=split == "test" and i == 0)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _check_pair(pair):
    """Check fixed geometry and per-thread conservation/order, without solve."""
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _parse(inp)
    a = out.T.copy() if info["vertical"] else out.copy()
    clean, final = a.copy(), []
    for s in info["slots"]:
        r, x, end, sign, color = s["row"], s["start"], s["end"], s["outward"], s["color"]
        assert np.all(a[r, x:end] == color), "anchor width/color changed"
        length, y = 1, r + sign
        while 0 <= y < a.shape[0] and np.all(a[y, x:end] == color):
            length += 1
            y += sign
        assert 0 <= y < a.shape[0] and np.all(a[y, x:end] == info["background"])
        for k in range(length):
            clean[r + sign * k, x:end] = info["background"]
        clean[r, x:end] = info["track"]
        final.append(length)
    assert np.array_equal(clean, info["structural"]), "track/break/gate/unexplained cell changed"
    for group in info["groups"]:
        old = [info["slots"][i]["length"] for i in group]
        new = [final[i] for i in group]
        assert Counter(old) == Counter(new), "tooth lengths crossed a thread boundary"
        assert all(a <= b for a, b in zip(new, new[1:])), "thread is not ascending"
    return info, _features(info)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info, f = _check_pair(pair)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen
            seen.add(key)
            features[split].append(f)
    witnesses = [f for f in features["train"] if f["changed_lengths"] and f["topology_matters"]
                 and f["area_disagrees"] and f["reversal_disagrees"] and f["cross_bank"]]
    assert len(witnesses) >= 2, "need two informative demonstrations"
    assert {f["vertical"] for f in features["test"]} <= {f["vertical"] for f in witnesses}
    assert any(f["changed_lengths"] and f["topology_matters"] for f in features["test"])
    for feature in ("topology_matters", "changed_lengths"):
        assert any(f[feature] for f in features["train"])


def self_test():
    """Bounded 42-seed diversity, input-only prediction and mutation checks."""
    signatures, shapes, layouts, features = set(), set(), set(), []
    role_sets = [set() for _ in range(3)]
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        repeated = generate_puzzle(seed)
        signature = tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s]
                          for k in ("input", "output"))
        assert signature == tuple(_key(p[k]) for s in ("train", "test") for p in repeated[s]
                                  for k in ("input", "output"))
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"].copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, pair["input"]), "solve mutated its argument"
                info, f = _check_pair(pair)
                features.append(f)
                shapes.add(inp.shape)
                mapping = {}
                normalized = tuple(mapping.setdefault(int(c), len(mapping)) for c in inp.ravel())
                layouts.add((inp.shape, normalized))
                for values, role in zip(role_sets, ("background", "track", "divider")):
                    values.add(info[role])
        # Incorrect labels using either omitted concept must fail the relation.
        witness = next(p for p in puzzle["train"] if all(_features(_parse(p["input"]))[k]
                       for k in ("changed_lengths", "topology_matters", "area_disagrees", "reversal_disagrees")))
        info = _parse(witness["input"])
        for mode in ("banks", "unchanged", "area", "reverse"):
            wrong = _paint(info, _assigned(info, mode))
            assert not np.array_equal(wrong, witness["output"])
            try:
                _check_pair({"input": witness["input"], "output": wrong})
            except AssertionError:
                pass
            else:
                raise AssertionError(f"accepted {mode} shortcut")
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(layouts) >= 190
    assert {f["vertical"] for f in features} == {False, True}
    assert {w for f in features for w in f["widths"]} == {1, 2, 3}
    assert {x for f in features for x in f["lengths"]} == set(range(1, 8))
    assert {f["lanes"] for f in features} == set(range(2, 8))
    assert all(len(s) >= 8 for s in role_sets)
    assert any(f["ties"] for f in features) and any(not f["ties"] for f in features)
    assert any(f["breaks"] == 0 for f in features) and any(f["breaks"] >= 4 for f in features)
    assert len({f["slots"] for f in features}) >= 6
    # Exercise the finite fallback constructions directly, regardless of how
    # often ordinary sampling needs them in the measured sweep.
    fallback_rng = np.random.default_rng(271828)
    for lanes in range(3, 8):
        for vertical in (False, True):
            pair = _busy_fallback(fallback_rng, list(range(10)), vertical, (lanes, lanes))
            info, _ = _check_pair(pair)
            assert max(Counter(r for r, _ in info["centers"]).values()) >= 2
            assert np.array_equal(solve(pair["input"]), pair["output"])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(314159, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "color_normalized_inputs": len(layouts), "pairs": len(features)}
