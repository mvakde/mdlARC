"""Communicating vessels with individually traceable, nonjoining hoses.

Only stdlib and NumPy. Generation chooses whole-level volumes; solve recovers
roles, vessel interiors, and hose connectivity exclusively from visible cells.
"""
from __future__ import annotations

from collections import defaultdict
import json
from itertools import permutations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=True)


def _runs(mask):
    start = None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            yield start, i - 1
            start = None


def _vessels(g, wall, bg):
    """Recognize closed frames by top rim and uninterrupted outside sides.

    Floor ports replace wall cells, but cannot interrupt a side. Grounded
    interior terrain is allowed; it is not counted as vessel capacity.
    """
    h, w = g.shape
    vessels = []
    covered = np.zeros(g.shape, dtype=bool)
    for top in range(1, h - 4):
        for left, right in _runs(g[top] == wall):
            if right - left < 4:
                continue
            if g[top - 1, left] == wall or g[top - 1, right] == wall:
                continue
            bottom = top
            while (bottom + 1 < h and g[bottom + 1, left] == wall
                   and g[bottom + 1, right] == wall):
                bottom += 1
            if bottom - top < 4:
                continue
            rect = np.zeros(g.shape, dtype=bool)
            rect[top:bottom + 1, left:right + 1] = True
            if np.any(rect & covered):
                raise ValueError("overlapping vessel frames")
            covered |= rect
            cavity = np.zeros(g.shape, dtype=bool)
            cavity[top + 1:bottom, left + 1:right] = (
                g[top + 1:bottom, left + 1:right] != wall)
            if not np.all(cavity[top + 1, left + 1:right]):
                raise ValueError("a vessel needs an open interior top band")
            # Each interior column is open above a grounded solid profile.
            for c in range(left + 1, right):
                column = g[top + 1:bottom, c] == wall
                if np.any(column[:-1] & ~column[1:]):
                    raise ValueError("ungrounded internal terrain")
            ports = [(bottom, c) for c in range(left + 1, right)
                     if g[bottom, c] != wall]
            if any(b[1] - a[1] < 2 for a, b in zip(ports, ports[1:])):
                raise ValueError("adjacent hose ports")
            if len(ports) > 3:
                raise ValueError("too many vessel ports")
            for _, c in ports:
                if not cavity[bottom - 1, c]:
                    raise ValueError("a hose port must open into the cavity")
            vessels.append(dict(top=top, bottom=bottom, left=left, right=right,
                                cavity=cavity, ports=ports))
    if not 2 <= len(vessels) <= 5 or np.any((g == wall) & ~covered):
        raise ValueError("wall cells must belong to 2..5 vessel frames")
    vessels.sort(key=lambda v: v["left"])
    if any(a["right"] >= b["left"] for a, b in zip(vessels, vessels[1:])):
        raise ValueError("vessels must have disjoint column ranges")
    return vessels, covered


def _trace(g, vessels, pipe, cross):
    visible = set(map(tuple, np.argwhere((g == pipe) | (g == cross)))) if cross is not None else set(map(tuple, np.argwhere(g == pipe)))
    owners = {p: i for i, v in enumerate(vessels) for p in v["ports"]}
    graph = {}
    for p in visible:
        r, c = p
        graph[p] = {d: (r + dr, c + dc) for d, (dr, dc) in enumerate(DIRS)
                    if (r + dr, c + dc) in visible}
        degree = len(graph[p])
        if p in owners:
            if degree != 1 or 2 not in graph[p] or g[p] != pipe:
                raise ValueError("floor ports must point down into a hose")
        elif degree == 4:
            if cross is None or g[p] != cross:
                raise ValueError("unmarked hose junction")
        elif degree != 2 or g[p] != pipe:
            raise ValueError("broken hose or misplaced crossing marker")
    paired, seen, edges, traces = set(), set(), [], []
    for start in sorted(owners):
        if start in paired:
            continue
        p, incoming, states, route = start, 0, set(), []
        while True:
            if (p, incoming) in states:
                raise ValueError("a hose cannot form an unattached loop")
            states.add((p, incoming))
            route.append(p)
            seen.add(p)
            if p != start and p in owners:
                if p in paired or owners[p] == owners[start]:
                    raise ValueError("hose endpoints must be distinct vessels")
                paired.update((start, p))
                edges.append((owners[start], owners[p]))
                traces.append(route)
                break
            if p == start:
                outgoing = 2
            elif len(graph[p]) == 4:
                outgoing = (incoming + 2) % 4
            else:
                if incoming not in graph[p]:
                    raise ValueError("invalid hose continuation")
                outgoing = next(d for d in graph[p] if d != incoming)
            if outgoing not in graph[p]:
                raise ValueError("hose has no continuation")
            p = graph[p][outgoing]
            incoming = (outgoing + 2) % 4
    if paired != set(owners) or seen != visible:
        raise ValueError("unpaired ports or unattached hose fragments")
    return edges, traces


def _groups(n, edges):
    adjacency = [set() for _ in range(n)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining, result = set(range(n)), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, todo = [start], [start]
        while todo:
            for b in sorted(adjacency[todo.pop()]):
                if b in remaining:
                    remaining.remove(b)
                    found.append(b)
                    todo.append(b)
        result.append(tuple(sorted(found)))
    return result


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if not np.all(g[0] == bg) or not np.all(g[-1] == bg):
        raise ValueError("top and bottom margins must show the background")
    candidates = []
    for wall in sorted(set(map(int, np.unique(g))) - {bg}):
        try:
            vessels, covered = _vessels(g, wall, bg)
            cavity = np.logical_or.reduce([v["cavity"] for v in vessels])
            liquid_colors = set(map(int, np.unique(g[cavity]))) - {bg}
            if len(liquid_colors) != 1:
                raise ValueError("exactly one liquid color is required")
            water = liquid_colors.pop()
            ports = [p for v in vessels for p in v["ports"]]
            port_colors = {int(g[p]) for p in ports}
            if len(port_colors) != 1:
                raise ValueError("hoses need one common carrier color")
            pipe = port_colors.pop()
            if len({bg, wall, water, pipe}) != 4:
                raise ValueError("roles must use distinct colors")
            outside = g[~covered]
            others = set(map(int, np.unique(outside))) - {bg, pipe}
            if len(others) > 1 or others & {wall, water}:
                raise ValueError("unrecognized exterior marks")
            cross = next(iter(others)) if others else None
            # Only floor ports may carry a routing role inside a frame.
            frame_marks = covered & ~cavity & (g != wall)
            expected = np.zeros(g.shape, dtype=bool)
            for p in ports:
                expected[p] = True
            if not np.array_equal(frame_marks, expected):
                raise ValueError("unexpected frame marks")
            edges, traces = _trace(g, vessels, pipe, cross)
            candidates.append(dict(grid=g, background=bg, wall=wall, water=water,
                                   pipe=pipe, cross=cross, vessels=vessels,
                                   cavity=cavity, edges=edges, traces=traces,
                                   groups=_groups(len(vessels), edges)))
        except (ValueError, StopIteration):
            continue
    if len(candidates) != 1:
        raise ValueError(f"expected one visible interpretation, found {len(candidates)}")
    return candidates[0]


def _fill_count(mask, volume, whole_levels=True):
    target = np.zeros(mask.shape, dtype=bool)
    for r in range(mask.shape[0] - 1, -1, -1):
        columns = np.flatnonzero(mask[r])
        if not len(columns):
            continue
        if volume < len(columns):
            if volume and whole_levels:
                raise ValueError("liquid volume cuts a shared horizontal level")
            target[r, columns[:volume]] = True
            volume = 0
            break
        target[r, columns] = True
        volume -= len(columns)
        if volume == 0:
            break
    if volume:
        raise ValueError("volume exceeds vessel capacity")
    return target


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Conserve each traced network's volume and fill complete absolute levels."""
    info = _parse(input_grid)
    g, bg, water = info["grid"], info["background"], info["water"]
    out = g.copy()
    out[out == water] = bg
    for group in info["groups"]:
        mask = np.logical_or.reduce([info["vessels"][i]["cavity"] for i in group])
        target = _fill_count(mask, int(np.count_nonzero((g == water) & mask)))
        out[target] = water
    return out


def _geometry(rng, n, edges, palette, dry=False):
    """Draw grounded profiles in frames, then separate U-shaped hose routes."""
    bg, wall, pipe, cross, _ = palette
    degrees = [sum(i in edge for edge in edges) for i in range(n)]
    widths = [max(3, 2 * d - 1) for d in degrees]
    # Extra width is budgeted geometrically, not forced to a terminal pitch.
    gap = int(rng.integers(1, 4)) if n <= 3 else 1
    spare = 30 - sum(w + 2 for w in widths) - gap * (n - 1)
    for _ in range(int(rng.integers(0, min(spare, 6) + 1))):
        eligible = [i for i, w in enumerate(widths) if w < 6]
        if eligible:
            widths[int(rng.choice(eligible))] += 1
            spare -= 1
    left_margin = int(rng.integers(0, min(2, spare) + 1))
    spare -= left_margin
    right_margin = int(rng.integers(0, min(2, spare) + 1))
    width = left_margin + sum(w + 2 for w in widths) + gap * (n - 1) + right_margin
    tops = [int(rng.integers(1, 3)) for _ in range(n)]
    bottoms = [int(rng.integers(8, 13)) for _ in range(n)]
    # Unequal bottoms are essential evidence against equal-depth filling.
    if max(bottoms) == min(bottoms):
        bottoms[-1] = 10 if bottoms[0] != 10 else 12
    if dry:
        bottoms[0] = 6
        for i in range(1, n):
            bottoms[i] = int(rng.integers(10, 13))
    gaps = [int(rng.integers(2, 5)) for _ in range(max(0, len(edges) - 1))]
    # Reserve the bottom margin while retaining independently varied run lengths.
    while max(bottoms) + 2 + sum(gaps) + 2 > 30:
        k = int(np.argmax(gaps))
        gaps[k] -= 1
    rows = [max(bottoms) + 2]
    for step in gaps:
        rows.append(rows[-1] + step)
    height = rows[-1] + int(rng.integers(2, min(4, 30 - rows[-1]) + 1))
    g = np.full((height, width), bg, dtype=np.int8)
    vessels, cursor = [], left_margin
    for i, iw in enumerate(widths):
        l, r, t, b = cursor, cursor + iw + 1, tops[i], bottoms[i]
        g[t, l:r + 1] = wall
        g[t:b + 1, l] = wall
        g[t:b + 1, r] = wall
        g[b, l:r + 1] = wall
        if degrees[i] == 0:
            ports = []
        elif degrees[i] == 1:
            ports = [(b, l + int(rng.integers(1, iw + 1)))]
        else:
            columns = np.rint(np.linspace(l + 1, r - 1, degrees[i])).astype(int)
            ports = [(b, int(c)) for c in columns]
        profile = [int(rng.integers(0, min(3, b - t - 3) + 1)) for _ in range(iw)]
        for _, c in ports:
            profile[c - l - 1] = 0
        for j, depth in enumerate(profile):
            if depth:
                g[b - depth:b, l + j + 1] = wall
        cavity = np.zeros(g.shape, dtype=bool)
        cavity[t + 1:b, l + 1:r] = g[t + 1:b, l + 1:r] == bg
        vessels.append(dict(cavity=cavity, ports=ports, top=t, bottom=b,
                            left=l, right=r))
        cursor = r + 1 + gap
    unused = [list(v["ports"]) for v in vessels]
    # Randomize which physical port is used by which graph edge.
    for ports in unused:
        rng.shuffle(ports)
    visits = defaultdict(set)
    paths = []
    for k, ((a, b), y) in enumerate(zip(edges, rows)):
        p, q = unused[a].pop(), unused[b].pop()
        step = 1 if q[1] > p[1] else -1
        path = ([(r, p[1]) for r in range(p[0], y + 1)]
                + [(y, c) for c in range(p[1] + step, q[1] + step, step)]
                + [(r, q[1]) for r in range(y - 1, q[0] - 1, -1)])
        paths.append(path)
        for cell in path:
            if g[cell] not in (bg, pipe) and cell not in (p, q):
                raise AssertionError("hose intersects terrain")
            g[cell] = pipe
            visits[cell].add(k)
    for cell, owners in visits.items():
        if len(owners) > 1:
            assert len(owners) == 2
            g[cell] = cross
    return g, vessels, paths


def _level(rng, vessels, group, style):
    lo = max(vessels[i]["top"] + 2 for i in group)
    hi = min(vessels[i]["bottom"] - 2 for i in group)
    if style == "dry" and 0 in group:
        return int(rng.integers(7, 10))
    if style == "lesson_cross":
        return 6 + int(rng.integers(2)) if len(group) == 3 else 4
    return int(rng.integers(lo, max(lo, hi) + 1))


def _make_pair(rng, palette, style):
    if style == "lesson_pair":
        n, edges = 3, [(0, 2)]
    elif style == "lesson_cross":
        n, edges = 5, [(0, 2), (2, 4), (1, 3)]
    elif style == "scale":
        n = 5
        order = list(map(int, rng.permutation([0, 1, 3, 4])))
        edges = list(zip(order, order[1:]))
        rng.shuffle(edges)
    elif style == "cycle":
        n = int(rng.choice([4, 5]))
        edges = [(0, 2), (2, n - 1), (n - 1, 0)]
        if n == 5:
            edges.append((1, 3))
        rng.shuffle(edges)
    elif style == "dry":
        n = int(rng.choice([4, 5]))
        edges = [(0, 2), (2, n - 1)]
        if n == 5:
            edges.append((1, 3))
        rng.shuffle(edges)
    elif style == "star":
        n, edges = 4, [(0, 1), (0, 2), (0, 3)]
        rng.shuffle(edges)
    else:
        n = int(rng.integers(2, 5))
        if n == 2:
            edges = [(0, 1)]
        elif n == 3:
            order = list(map(int, rng.permutation(n)))
            edges = [(order[0], order[1])]
        elif rng.random() < .5:
            edges = [(0, 2), (1, 3)]
        else:
            edges = [(0, 2), (2, 3)]
        rng.shuffle(edges)
    g, vessels, _ = _geometry(rng, n, edges, palette, style == "dry")
    water = palette[4]
    out, inp = g.copy(), g.copy()
    for group_index, group in enumerate(_groups(n, edges)):
        mask = np.logical_or.reduce([vessels[i]["cavity"] for i in group])
        level = _level(rng, vessels, group, style)
        target = mask & (np.arange(g.shape[0])[:, None] >= level)
        volume = int(target.sum())
        out[target] = water
        points = np.argwhere(mask)
        # Sources are conserved supplies, not shapes that determine the answer.
        mode = int(rng.integers(4))
        if style in ("lesson_pair", "lesson_cross"):
            mode = 0
        if mode == 0:
            anchor = group[int(rng.integers(len(group)))]
            in_anchor = vessels[anchor]["cavity"][points[:, 0], points[:, 1]]
            score = (~in_anchor) * 100 + points[:, 0] * .3 + rng.random(len(points))
        elif mode == 1:
            score = rng.random(len(points))
        elif mode == 2:
            score = points[:, 0] + rng.random(len(points))
        else:
            score = (~target[points[:, 0], points[:, 1]]) * 2 + rng.random(len(points))
        selected = points[np.argsort(score)[:volume]]
        inp[selected[:, 0], selected[:, 1]] = water
    if rng.random() < .5:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    return {"input": inp, "output": out}


def _key(grid):
    return grid.shape, grid.tobytes()


def _signature(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not 2 <= num_train <= 10
            or not _integer(num_test) or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, 5, replace=False)))
    shift = str(rng.choice(["scale", "dry", "cycle"]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            style = ("lesson_pair" if index == 0 else "lesson_cross" if index == 1
                     else "varied") if split == "train" else shift
            # Retries enforce observable lessons, not an unmarked rule choice.
            for attempt in range(100):
                pair = _make_pair(rng, palette, style)
                info = _parse(pair["input"])
                stats = _inspect_pair(pair, info)
                if (_key(pair["input"]) not in seen and stats["changed"]
                        and stats["transfer"]
                        and (split != "train" or index != 1 or stats["volume"] !=
                             int((puzzle["train"][0]["input"] == palette[4]).sum()))
                        and (style != "lesson_pair" or stats["relative_wrong"])
                        and (style != "lesson_cross" or stats["separate_crossing"])
                        and (style != "dry" or stats["drained_connected_vessel"])):
                    break
            else:
                raise RuntimeError("bounded sampling could not expose the requested lesson")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(info, groups, relative=False):
    """Counterfactuals for validation only; never used by solve or generation truth."""
    g, water = info["grid"], info["water"]
    out = g.copy()
    out[out == water] = info["background"]
    for group in groups:
        mask = np.logical_or.reduce([info["vessels"][i]["cavity"] for i in group])
        volume = int(((g == water) & mask).sum())
        if relative:
            ranked = []
            for i in group:
                cells = np.argwhere(info["vessels"][i]["cavity"])
                bottom = int(cells[:, 0].max())
                ranked.extend((bottom - int(r), int(r), int(c)) for r, c in cells)
            for _, r, c in sorted(ranked)[:volume]:
                out[r, c] = water
        else:
            out[_fill_count(mask, volume, whole_levels=False)] = water
    return out


def _inspect_pair(pair, info=None):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _parse(inp) if info is None else info
    assert inp.shape == out.shape
    bg, water, cavity = info["background"], info["water"], info["cavity"]
    assert np.array_equal(inp[~cavity], out[~cavity]), "static apparatus changed"
    assert set(map(int, np.unique(out[cavity]))) <= {bg, water}
    transfer = False
    levels, group_of = [], {}
    for k, group in enumerate(info["groups"]):
        for i in group:
            group_of[i] = k
        mask = np.logical_or.reduce([info["vessels"][i]["cavity"] for i in group])
        before, after = (inp == water) & mask, (out == water) & mask
        assert int(before.sum()) == int(after.sum()), "network volume not conserved"
        if after.any():
            level = int(np.argwhere(after)[:, 0].min())
            assert np.array_equal(after, mask & (np.arange(inp.shape[0])[:, None] >= level)), "not a complete common level"
            levels.append(level)
        assert np.argwhere(after)[:, 0].sum() >= np.argwhere(before)[:, 0].sum()
        for i in group:
            cm = info["vessels"][i]["cavity"]
            transfer |= int(((inp == water) & cm).sum()) != int(((out == water) & cm).sum())
    assert np.array_equal(solve(inp), out), "input-only solve disagrees"
    separate_crossing = False
    crossing_count = 0
    if info["cross"] is not None:
        for p in map(tuple, np.argwhere(inp == info["cross"])):
            route_indices = [i for i, route in enumerate(info["traces"]) if p in route]
            assert len(route_indices) == 2
            crossing_count += 1
            gs = {group_of[info["edges"][i][0]] for i in route_indices}
            separate_crossing |= len(gs) == 2
    local = _alternative(info, [(i,) for i in range(len(info["vessels"]))])
    global_fill = _alternative(info, [tuple(range(len(info["vessels"])))])
    relative = _alternative(info, info["groups"], relative=True)
    return dict(changed=bool(np.any(inp != out)), transfer=bool(transfer),
                local_wrong=not np.array_equal(local, out),
                global_wrong=not np.array_equal(global_fill, out),
                relative_wrong=not np.array_equal(relative, out),
                separate_crossing=bool(separate_crossing), crossings=crossing_count,
                vessels=len(info["vessels"]), edges=len(info["edges"]),
                group_sizes=tuple(sorted(map(len, info["groups"]))),
                levels=tuple(levels), shape=inp.shape,
                volume=int((inp == water).sum()),
                profiles=tuple(tuple(int(v["cavity"][:, c].sum())
                                    for c in range(v["left"] + 1, v["right"]))
                               for v in info["vessels"]),
                empty_connected_vessel=any(not np.any((out == water) & info["vessels"][i]["cavity"])
                                            for group in info["groups"] if len(group) > 1 for i in group),
                drained_connected_vessel=any(np.any((inp == water) & info["vessels"][i]["cavity"])
                                              and not np.any((out == water) & info["vessels"][i]["cavity"])
                                              for group in info["groups"] if len(group) > 1 for i in group),
                degrees=tuple(sum(i in e for e in info["edges"]) for i in range(len(info["vessels"]))),
                roles=(bg, info["wall"], info["pipe"], info["cross"], water))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, train_stats, crossing_roles = set(), None, [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse(inp)
            actual_roles = tuple(info[k] for k in ("background", "wall", "pipe", "water"))
            assert roles is None or actual_roles == roles, "puzzle roles changed"
            roles = actual_roles
            if info["cross"] is not None:
                crossing_roles.add(info["cross"])
                assert len(crossing_roles) == 1, "crossing marker role changed"
            stats = _inspect_pair(pair, info)
            assert stats["changed"] and stats["transfer"] and stats["local_wrong"]
            if split == "train":
                train_stats.append(stats)
    assert any(s["relative_wrong"] for s in train_stats), "missing absolute-height evidence"
    assert any(s["global_wrong"] and s["separate_crossing"] for s in train_stats), "missing nonjoining crossing evidence"
    assert any(max(s["group_sizes"]) >= 3 for s in train_stats), "missing transitive network evidence"
    assert len({s["volume"] for s in train_stats}) >= 2, "missing varying-volume evidence"


def self_test():
    """Bounded structural and semantic audits on 42 seeds and count boundaries."""
    shapes, profiles, roles, signatures, group_sizes = set(), set(), set(), set(), set()
    degrees, volumes, crossings, empty = set(), set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            s = _inspect_pair(pair)
            shapes.add(s["shape"])
            profiles.add(s["profiles"])
            roles.add(s["roles"])
            group_sizes.add(s["group_sizes"])
            degrees.update(s["degrees"])
            volumes.add(s["volume"])
            crossings.add(s["crossings"])
            empty += s["empty_connected_vessel"]
    assert len(signatures) == 42 and len(shapes) >= 25 and len(profiles) >= 150
    assert len(roles) >= 35 and len(volumes) >= 30 and len(group_sizes) >= 5
    assert {0, 1, 2} <= degrees and max(crossings) >= 4 and empty >= 5
    # Counts are small enough to exercise every legal combination once.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Enumerate route-height orders: crossings must not silently change groups.
    rng = np.random.default_rng(90210)
    for edge_order in permutations([(0, 2), (2, 4), (1, 3)]):
        g, _, _ = _geometry(rng, 5, list(edge_order), [0, 5, 8, 4, 1])
        vs, _ = _vessels(g, 5, 0)
        first = tuple(map(int, np.argwhere(vs[0]["cavity"])[0]))
        g[first] = 1
        info = _parse(g)
        assert set(info["groups"]) == {(0, 2, 4), (1, 3)}
    # Solver supports branching at vessels and a cyclic vessel network.
    for style in ("star", "cycle"):
        pair = _make_pair(rng, [0, 5, 8, 4, 1], style)
        _inspect_pair(pair)
    # Nonrandom boundary members: one full network and one empty network.
    g, vessels, _ = _geometry(rng, 5, [(0, 2), (2, 4), (1, 3)], [0, 5, 8, 4, 1])
    mask = np.logical_or.reduce([vessels[i]["cavity"] for i in (0, 2, 4)])
    g[mask] = 1
    assert np.array_equal(solve(g), g)
    # A partial common level is outside the family and cannot be tie-broken.
    base, vessels, _ = _geometry(rng, 2, [(0, 1)], [0, 5, 8, 4, 1])
    mask = vessels[0]["cavity"] | vessels[1]["cavity"]
    capacities = mask.sum(axis=1)
    row = int(np.flatnonzero(capacities > 1)[-1])
    volume = int(capacities[row + 1:].sum()) + 1
    cells = np.argwhere(mask)[:volume]
    base[cells[:, 0], cells[:, 1]] = 1
    try:
        solve(base)
    except ValueError as error:
        assert "cuts a shared horizontal level" in str(error)
    else:
        raise AssertionError("partial common level accepted")
    damaged = generate_puzzle(123)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError("static-cell corruption accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
